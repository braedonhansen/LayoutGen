"""Shared parts of the classical baselines (Euler, naive).

Columns are tuples ("sd", nmos_net, pmos_net) / ("g", input); an sd net of
None is an empty slot ("-"), which breaks the diffusion in that row.

Contacts follow the validator rule (supply, output, net on another island, or
gate input). Routing gives every non-supply net with more than one node one
horizontal route (li1 or met1 on a channel track, or met1 on a row track when
all its endpoints are sd in that row); an input driving a single gate column is
left unrouted and its pin picks the contact track. A branch-and-bound search
over nets takes the assignment with no validator errors and the fewest
near-misses, then the smallest rendered width, then a fixed order.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cellgym.dsl.plan import parse
from cellgym.dsl.template import CHANNEL_NAMES, Layout, layout_for_netlist
from cellgym.dsl.validate import SUPPLY, validate
from cellgym.render.render import RenderError, render

MAX_TIES = 6            # routings with the fewest near-misses kept for the width tie-break
NODE_CAP = 100_000      # routing search nodes per placement


class BaselineError(ValueError):
    def __init__(self, reason: str, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


# -- contacts and routing ----------------------------------------------------------

@dataclass
class Option:
    layer: str | None      # None: unrouted input, track given on the pin
    track: str
    objs: list             # (group, lo, hi)
    pads: list             # (col, met1 grid index)


@dataclass
class Net:
    net: str
    eps: list              # (kind, col), kind in nmos/pmos/gate
    options: list


class Router:
    def __init__(self, cols: list[tuple], netlist: dict, outputs: list[str], lay: Layout):
        self.cols, self.outputs, self.lay = cols, outputs, lay
        self.inputs = list(netlist["inputs"])
        rows = {"nmos": 1, "pmos": 2}
        islands = []                                  # [row, net, [cols]]
        for row, k in rows.items():
            cur = None
            for i, c in enumerate(cols):
                if c[0] != "sd" or c[k] is None:     # gate, or an empty slot (naive separator)
                    cur = None
                    continue
                if cur is None or cur[1] != c[k]:
                    cur = [row, c[k], []]
                    islands.append(cur)
                cur[2].append(i)
        gates: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(cols):
            if c[0] == "g":
                gates[c[1]].append(i)
        per_net = Counter(isl[1] for isl in islands)
        contacted = [isl for isl in islands
                     if isl[1] in SUPPLY or isl[1] in outputs or per_net[isl[1]] > 1 or isl[1] in gates]
        self.stars: set[tuple[str, int]] = set()
        self.nets: list[Net] = []
        by_net: dict[str, list] = defaultdict(list)
        for isl in contacted:
            by_net[isl[1]].append(isl)
        for net in sorted(set(by_net) | set(gates)):
            isls, gcols = by_net.get(net, []), gates.get(net, [])
            if net in SUPPLY:
                self.stars.update((isl[0], isl[2][0]) for isl in isls)
                continue
            eps = []
            for isl in isls:
                others = [o[2][0] for o in isls if o is not isl] + gcols
                if others:
                    center = sum(others) / len(others)
                    col = min(isl[2], key=lambda c: (abs(c - center), c))
                else:
                    col = isl[2][0]
                self.stars.add((isl[0], col))
                eps.append((isl[0], col))
            eps += [("gate", c) for c in gcols]
            eps.sort(key=lambda e: (e[1], e[0]))
            if len(eps) > 1:
                self.nets.append(Net(net, eps, self._route_options(eps)))
            elif eps and eps[0][0] == "gate":
                col = eps[0][1]
                self.nets.append(Net(net, eps, [
                    Option(None, t, [(("li", t), col, col), (("poly", t), col, col)], [])
                    for t in CHANNEL_NAMES]))

    def _route_options(self, eps: list) -> list[Option]:
        kinds = {k for k, _ in eps}
        tracks = []
        if len(kinds) == 1 and "gate" not in kinds:
            row = next(iter(kinds))
            n = self.lay.n_tracks if row == "nmos" else self.lay.p_tracks
            tracks += [("met1", f"{row[0]}{j}") for j in range(n)]
        tracks += [(layer, t) for layer in ("li1", "met1") for t in CHANNEL_NAMES]
        lo, hi = min(c for _, c in eps), max(c for _, c in eps)
        opts = []
        for layer, t in tracks:
            objs = [(("li" if layer == "li1" else "m1", t), lo, hi)]
            pads = []
            if t in CHANNEL_NAMES:
                ti = CHANNEL_NAMES.index(t)
                for kind, col in eps:
                    if kind == "gate":
                        objs += [(("li", t), col, col), (("poly", t), col, col)]
                    else:
                        crossed = CHANNEL_NAMES[: ti + 1] if kind == "nmos" else CHANNEL_NAMES[ti:]
                        objs += [(("li", x), col, col) for x in crossed]
            if layer == "met1":
                k = self.lay.track_k(t)
                pads = [(col, k) for kind, col in eps if kind != "gate"]
            opts.append(Option(layer, t, objs, pads))
        return opts

    def search(self) -> tuple[list[tuple[int, list[Option]]], bool]:
        order = sorted(self.nets, key=lambda n: (len(n.options), -(n.eps[-1][1] - n.eps[0][1]), n.net))
        placed: dict = defaultdict(list)      # group -> [(lo, hi, net)]
        pads: dict = defaultdict(list)        # col -> [(k, net)]
        best = [math.inf]
        sols: list = []
        nodes = [0]
        capped = [False]
        choice: list = []

        def cost_of(net: str, opt: Option):
            c = 0
            for g, lo, hi in opt.objs:
                gap = 2 if g[0] == "poly" else 1
                for lo2, hi2, n2 in placed[g]:
                    if n2 == net:
                        continue
                    if lo <= hi2 and lo2 <= hi:
                        return None
                    if lo - gap <= hi2 and lo2 - gap <= hi:
                        c += 1
            for col, k in opt.pads:
                for k2, n2 in pads[col]:
                    if n2 != net and abs(k - k2) == 1:
                        return None
            return c

        def dfs(i: int, cost: int) -> None:
            nodes[0] += 1
            if nodes[0] > NODE_CAP:
                capped[0] = True
                return
            if i == len(order):
                if cost < best[0]:
                    best[0] = cost
                    sols.clear()
                sols.append((cost, list(choice)))
                return
            nt = order[i]
            for opt in nt.options:
                c = cost_of(nt.net, opt)
                if c is None or cost + c > best[0] or (cost + c == best[0] and len(sols) >= MAX_TIES):
                    continue
                for g, lo, hi in opt.objs:
                    placed[g].append((lo, hi, nt.net))
                for col, k in opt.pads:
                    pads[col].append((k, nt.net))
                choice.append(opt)
                dfs(i + 1, cost + c)
                choice.pop()
                for g, _, _ in opt.objs:
                    placed[g].pop()
                for col, _ in opt.pads:
                    pads[col].pop()
                if capped[0]:
                    return

        dfs(0, 0)
        return [(c, dict(zip((n.net for n in order), ch))) for c, ch in sols], capped[0]

    def plan_text(self, name: str, choice: dict) -> str:
        lines = [f"cell {name}", "in " + " ".join(self.inputs), "out " + " ".join(self.outputs)]
        for i, c in enumerate(self.cols):
            if c[0] == "g":
                lines.append(f"col {i} g n={c[1]} p={c[1]}")
            else:
                n = (c[1] or "-") + ("*" if ("nmos", i) in self.stars else "")
                p = (c[2] or "-") + ("*" if ("pmos", i) in self.stars else "")
                lines.append(f"col {i} sd n={n} p={p}")
        nets = {n.net: n for n in self.nets}
        for net in sorted(choice):
            opt = choice[net]
            if opt.layer:
                eps = " ".join(f"{k}:{c}" for k, c in nets[net].eps)
                lines.append(f"route {net} {opt.layer} {opt.track} {eps}")
        for net in self.inputs:
            gcols = [i for i, c in enumerate(self.cols) if c[0] == "g" and c[1] == net]
            opt = choice.get(net)
            if opt is not None and opt.layer is None:
                lines.append(f"pin {net} gate:{gcols[0]} {opt.track}")
            else:
                lines.append(f"pin {net} gate:{gcols[0]}")
        for net in self.outputs:
            for row in ("nmos", "pmos"):
                cs = sorted(i for r, i in self.stars if r == row and self.cols[i][1 if row == "nmos" else 2] == net)
                if cs:
                    lines.append(f"pin {net} {row}:{cs[0]}")
                    break
        return "\n".join(lines) + "\n"

# -- driver --------------------------------------------------------------------------

def outputs_of(netlist: dict) -> list[str]:
    return [p for p in netlist["ports"] if p not in netlist["inputs"] and p not in SUPPLY]


@dataclass
class Routed:
    text: str
    near_misses: int
    width_um: float
    sites: int
    capped: bool          # some routing search hit NODE_CAP
    index: int            # which placement won


def route_best(placements: list[list[tuple]], netlist: dict) -> Routed:
    """Route every placement; best (near-misses, width) plan that validates and renders."""
    name = netlist["id"]
    outputs = outputs_of(netlist)
    lay = layout_for_netlist(netlist)
    best = None
    capped = False
    last_err = BaselineError("no-routing", "no legal routing for any placement")
    for idx, cols in enumerate(placements):
        router = Router(cols, netlist, outputs, lay)
        sols, cap = router.search()
        capped |= cap
        for nm, choice in sols:
            text = router.plan_text(name, choice)
            rep = validate(parse(text), netlist)
            if not rep.ok:
                last_err = BaselineError("validate", str(rep).splitlines()[0])
                continue
            if rep.near_misses != nm:
                last_err = BaselineError("near-miss-model", f"search counted {nm}, validator {rep.near_misses}")
                continue
            try:
                cell = render(parse(text), netlist)
            except (RenderError, ValueError, KeyError, IndexError) as e:
                last_err = BaselineError("render", str(e).splitlines()[0])
                continue
            key = (nm, cell.width_nm)
            if best is None or key < best[0]:
                best = (key, text, cell, idx)
    if best is None:
        if capped and last_err.reason == "no-routing":
            last_err = BaselineError("no-routing", "no legal routing found before the search cap")
        raise last_err
    (nm, _), text, cell, idx = best
    return Routed(text, nm, cell.width, cell.sites, capped, idx)


# -- CLI -------------------------------------------------------------------------------

def stratified(netlists: dict[str, dict], n: int) -> list[str]:
    groups: dict = defaultdict(list)
    for cid, nl in sorted(netlists.items()):
        depth = max(nl.get("pdn_depth", 0), nl.get("pun_depth", 0))
        groups[(len(nl["inputs"]), depth)].append(cid)
    picked: list[str] = []
    k = 0
    while len(picked) < n and any(len(g) > k for g in groups.values()):
        for key in sorted(groups):
            if len(groups[key]) > k and len(picked) < n:
                picked.append(groups[key][k])
        k += 1
    return picked


def run_cli(doc: str, solve: Callable[[dict], object], argv=None, default_out: str = "out/plans") -> int:
    """Shared CLI: solve(netlist) returns an object with text, sites, breaks,
    near_misses and capped, or raises BaselineError."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="out/cells")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--verify", type=int, default=0, help="run DRC+LVS on a stratified sample of N cells")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--pdk-root", default=str(Path.home() / "pdk"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    netlists = {}
    for f in sorted(Path(a.cells).glob("*.json")):
        nl = json.loads(f.read_text())
        if "devices" in nl:
            netlists[nl["id"]] = nl
    t0 = time.time()
    results: dict[str, object] = {}
    fails: dict[str, list[str]] = defaultdict(list)
    for cid, nl in netlists.items():
        try:
            r = solve(nl)
        except BaselineError as e:
            fails[e.reason].append(f"{cid}: {e}")
            continue
        results[cid] = r
        (out / f"{cid}.plan").write_text(r.text)
    n = len(netlists)
    print(f"cells {n}  validated+rendered {len(results)}  ({time.time() - t0:.1f}s)")
    if results:
        sites = sorted(r.sites for r in results.values())
        print(f"width sites: min {sites[0]}  median {statistics.median(sites)}  max {sites[-1]}  "
              f"mean {statistics.mean(sites):.2f}")
        print(f"breaks: mean {statistics.mean(r.breaks for r in results.values()):.2f}  "
              f"zero-break cells {sum(r.breaks == 0 for r in results.values())}  "
              f"near-misses mean {statistics.mean(r.near_misses for r in results.values()):.2f}")
        capped = [c for c, r in results.items() if r.capped]
        print(f"routing search cap hits: {len(capped)}" + (f" ({', '.join(capped[:10])})" if capped else ""))
    for reason, items in sorted(fails.items()):
        print(f"FAIL {reason}: {len(items)}")
        for it in items[:20]:
            print(f"  {it}")
    if a.verify:
        from concurrent.futures import ThreadPoolExecutor
        from cellgym.groundtruth.verify import verify
        ids = stratified({c: netlists[c] for c in results}, a.verify)
        wd = (out / "verify").resolve()
        nl_dir = Path(a.cells)

        def one(cid):
            try:
                r = verify(out / f"{cid}.plan", nl_dir / f"{cid}.json", wd / cid, Path(a.pdk_root))
                return cid, r.drc_items, r.lvs_ok, r.lvs_lines
            except Exception as e:  # report, don't abort the batch
                return cid, None, False, [f"{type(e).__name__}: {e}"]

        t1 = time.time()
        with ThreadPoolExecutor(a.jobs) as ex:
            vres = list(ex.map(one, ids))
        drc_ok = sum(1 for _, d, _, _ in vres if d == [])
        lvs_ok = sum(1 for _, _, l, _ in vres if l)
        print(f"verify {len(ids)}: DRC pass {drc_ok}  LVS pass {lvs_ok}  "
              f"both {sum(1 for _, d, l, _ in vres if d == [] and l)}  ({time.time() - t1:.0f}s)")
        for cid, d, l, lines in vres:
            if d != [] or not l:
                drc = "no report" if d is None else ", ".join(f"{k} x{v}" for k, v in sorted(Counter(d).items())) or "ok"
                lvs = "ok" if l else " | ".join(x.strip() for x in lines[-3:])
                print(f"  {cid}: DRC {drc}; LVS {lvs}")
        return 0 if not fails and all(d == [] and l for _, d, l, _ in vres) else 1
    return 0 if not fails else 1
