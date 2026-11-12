"""Step-wise routing environment: one route segment (or pin) per step, after placement.

    env = RouteEnv(netlist, cols)        # cols from placement: ("sd", n, p) / ("g", x)
    while not env.done():
        acts = env.legal()               # DSL lines: "route n3 met1 n0 nmos:0 nmos:4", "pin C gate:5 c0"
        env.step(acts[k])                # False (and no change) if it strands another net
    text = env.plan_text()

Mirrors the validator (cellgym/dsl/validate.py): sd islands, the contact-needed rule, gate contacts
(one per gate column; its track comes from the first route or pin that reaches it), li1 only on
channel tracks, row tracks only for routes whose endpoints are all sd in that row, stubs from an sd
endpoint to the channel track it uses, li/met1/poly overlap per track between different nets, and
met1 pads of different nets on adjacent tracks in one column. Near-misses are legal (they cost
width) and counted.

Larger than the baseline router's space: a net may be connected by several routes that share an
endpoint (split routes), and an island's contact column is chosen by the first route that lands
on it. A route's endpoints are a run of consecutive nodes of one net (in column order) that joins
at least two of its still-separate pieces. step() refuses a move after which some other open net
(or untracked gate) has no legal move left (one-step lookahead, no backtracking).
"""

from __future__ import annotations

import copy
import itertools
from collections import Counter, defaultdict

from cellgym.baselines.common import outputs_of
from cellgym.dsl.template import CHANNEL_NAMES, layout_for_netlist
from cellgym.dsl.validate import SUPPLY

ROWS = ("nmos", "pmos")
MAX_COMBOS = 4          # island-column combinations tried per endpoint run


class RouteEnv:
    def __init__(self, nl: dict, cols: list):
        self.nl, self.cols = nl, [tuple(c) for c in cols]
        self.lay = layout_for_netlist(nl)
        self.inputs = list(nl["inputs"])
        self.outputs = outputs_of(nl)
        self.islands = []                        # [row, net, [cols]]
        self.island_at = {}
        for r, row in enumerate(ROWS):
            cur = None
            for i, c in enumerate(self.cols):
                net = c[1 + r] if c[0] == "sd" else None
                if net is None or net == "-":
                    cur = None
                    continue
                if cur is None or cur[1] != net:
                    cur = [row, net, []]
                    self.islands.append(cur)
                cur[2].append(i)
                self.island_at[(row, i)] = len(self.islands) - 1
        self.gate_net = {i: c[1] for i, c in enumerate(self.cols) if c[0] == "g"}
        per_net = Counter(isl[1] for isl in self.islands)
        gate_nets = set(self.gate_net.values())
        self.needed = [k for k, (row, net, cs) in enumerate(self.islands)
                       if net in SUPPLY or net in self.outputs or per_net[net] > 1 or net in gate_nets]
        self.nodes = defaultdict(list)           # net -> [("isl", k) | ("gc", col)]
        for k in self.needed:
            if self.islands[k][1] not in SUPPLY:
                self.nodes[self.islands[k][1]].append(("isl", k))
        for col, net in self.gate_net.items():
            self.nodes[net].append(("gc", col))
        self.uf = {}
        self.objs = defaultdict(list)            # (layer, track) -> [(lo, hi, net)], layer li/m1/poly
        self.pads = defaultdict(list)            # col -> [(k, net)]
        self.contact = {}                        # island -> chosen contact column
        self.gate_track = {}                     # gate col -> track
        self.lines = []                          # emitted route / pin lines
        self.pins = {}                           # input -> pin line (with track)
        self.near_misses = 0

    # -- union-find / bookkeeping ---------------------------------------------------
    def _find(self, x):
        while self.uf.get(x, x) != x:
            x = self.uf[x]
        return x

    def components(self, net):
        comps = defaultdict(list)
        for nd in self.nodes[net]:
            comps[self._find(nd)].append(nd)
        return list(comps.values())

    def open_nets(self):
        return [n for n in sorted(self.nodes) if len(self.nodes[n]) > 1 and len(self.components(n)) > 1]

    def untracked_gates(self):
        """Gate columns no route reached: they need a pin with a track (single-gate inputs)."""
        return [c for c in sorted(self.gate_net) if c not in self.gate_track
                and len(self.nodes[self.gate_net[c]]) == 1]

    def done(self):
        return not self.open_nets() and not self.untracked_gates()

    # -- geometry of one candidate ----------------------------------------------------
    def _pos(self, nd):
        return self.contact.get(nd[1], self.islands[nd[1]][2][0]) if nd[0] == "isl" else nd[1]

    def _ep(self, nd, col):
        return (self.islands[nd[1]][0], col) if nd[0] == "isl" else ("gate", col)

    def _tracks(self, eps):
        kinds = {k for k, _ in eps}
        out = []
        if len(kinds) == 1 and "gate" not in kinds:
            row = next(iter(kinds))
            n = self.lay.n_tracks if row == "nmos" else self.lay.p_tracks
            out += [("met1", f"{row[0]}{j}") for j in range(n)]
        for layer in ("li1", "met1"):
            for t in CHANNEL_NAMES:
                if all(k != "gate" or self.gate_track.get(c, t) == t for k, c in eps):
                    out.append((layer, t))
        return out

    def _geometry(self, net, layer, track, eps):
        lo, hi = min(c for _, c in eps), max(c for _, c in eps)
        objs = [(("li" if layer == "li1" else "m1", track), lo, hi)]
        pads = []
        if track in CHANNEL_NAMES:
            ti = CHANNEL_NAMES.index(track)
            for kind, col in eps:
                if kind == "gate":
                    if col not in self.gate_track:
                        objs += [(("li", track), col, col), (("poly", track), col, col)]
                else:
                    crossed = CHANNEL_NAMES[: ti + 1] if kind == "nmos" else CHANNEL_NAMES[ti:]
                    objs += [(("li", x), col, col) for x in crossed]
        if layer == "met1":
            k = self.lay.track_k(track)
            pads = [(col, k) for kind, col in eps if kind != "gate"]
        return objs, pads

    def _cost(self, net, objs, pads):
        c = 0
        for g, lo, hi in objs:
            gap = 2 if g[0] == "poly" else 1
            for lo2, hi2, n2 in self.objs[g]:
                if n2 == net:
                    continue
                if lo <= hi2 and lo2 <= hi:
                    return None
                if lo - gap <= hi2 and lo2 - gap <= hi:
                    c += 1
        for col, k in pads:
            for k2, n2 in self.pads[col]:
                if n2 != net and abs(k - k2) == 1:
                    return None
        return c

    # -- moves --------------------------------------------------------------------------
    def _net_moves(self, net, first_only=False):
        nds = sorted(self.nodes[net], key=lambda nd: (self._pos(nd), nd[0]))
        out = []
        for i in range(len(nds)):
            for j in range(i + 1, len(nds)):
                run = nds[i:j + 1]
                if len({self._find(nd) for nd in run}) < 2:
                    continue
                choices = []
                for nd in run:
                    if nd[0] == "isl" and nd[1] not in self.contact:
                        choices.append(self.islands[nd[1]][2])
                    else:
                        choices.append([self._pos(nd)])
                for combo in itertools.islice(itertools.product(*choices), MAX_COMBOS):
                    eps = sorted((self._ep(nd, col) for nd, col in zip(run, combo)), key=lambda e: (e[1], e[0]))
                    for layer, track in self._tracks(eps):
                        objs, pads = self._geometry(net, layer, track, eps)
                        c = self._cost(net, objs, pads)
                        if c is None:
                            continue
                        line = f"route {net} {layer} {track} " + " ".join(f"{k}:{col}" for k, col in eps)
                        out.append((line, (net, layer, track, eps, run, combo, objs, pads, c)))
                        if first_only:
                            return out
        return out

    def _pin_moves(self, col, first_only=False):
        net = self.gate_net[col]
        out = []
        for t in CHANNEL_NAMES:
            objs = [(("li", t), col, col), (("poly", t), col, col)]
            c = self._cost(net, objs, [])
            if c is not None:
                out.append((f"pin {net} gate:{col} {t}", (net, t, col, objs, c)))
                if first_only:
                    return out
        return out

    def _moves(self):
        on = self.open_nets()
        if on:
            return [m for n in on for m in self._net_moves(n)]
        return [m for c in self.untracked_gates() for m in self._pin_moves(c)]

    def legal(self) -> list[str]:
        self._cache = dict(self._moves())
        return list(self._cache)

    def _apply(self, line, info):
        if line.startswith("route"):
            net, layer, track, eps, run, combo, objs, pads, c = info
            for nd, col in zip(run, combo):
                if nd[0] == "isl":
                    self.contact.setdefault(nd[1], col)
            for kind, col in eps:
                if kind == "gate":
                    self.gate_track.setdefault(col, track)
            for g, lo, hi in objs:
                self.objs[g].append((lo, hi, net))
            for col, k in pads:
                self.pads[col].append((k, net))
            roots = [self._find(nd) for nd in run]
            for r in roots[1:]:
                if self._find(r) != self._find(roots[0]):
                    self.uf[self._find(r)] = self._find(roots[0])
            self.lines.append(line)
            self.near_misses += c
        else:
            net, t, col, objs, c = info
            self.gate_track[col] = t
            for g, lo, hi in objs:
                self.objs[g].append((lo, hi, net))
            self.pins[net] = line
            self.near_misses += c

    def _viable(self):
        for n in self.open_nets():
            if not self._net_moves(n, first_only=True):
                return False
        if not self.open_nets():
            for c in self.untracked_gates():
                if not self._pin_moves(c, first_only=True):
                    return False
        return True

    def step(self, line: str) -> bool:
        cache = getattr(self, "_cache", None)
        if cache is None or line not in cache:
            cache = dict(self._moves())
        info = cache.get(line)
        if info is None:
            return False
        snap = (copy.deepcopy(self.uf), copy.deepcopy(self.objs), copy.deepcopy(self.pads),
                dict(self.contact), dict(self.gate_track), list(self.lines), dict(self.pins), self.near_misses)
        self._apply(line, info)
        self._cache = None
        if not self._viable():
            (self.uf, self.objs, self.pads, self.contact, self.gate_track, self.lines, self.pins,
             self.near_misses) = snap
            return False
        return True

    # -- text -----------------------------------------------------------------------------
    def state(self) -> str:
        cs = " | ".join(f"{i} g {c[1]}" if c[0] == "g" else f"{i} sd n={c[1]} p={c[2]}" for i, c in enumerate(self.cols))
        parts = []
        for n in self.open_nets():
            comps = []
            for comp in self.components(n):
                names = []
                for nd in sorted(comp, key=self._pos):
                    if nd[0] == "gc":
                        names.append(f"gate:{nd[1]}")
                    else:
                        row, _, cols = self.islands[nd[1]]
                        cc = self.contact.get(nd[1])
                        names.append(f"{row}:{cc}" if cc is not None or len(cols) == 1 else f"{row}:{cols[0]}-{cols[-1]}")
                comps.append("[" + " ".join(names) + "]")
            parts.append(f"{n} " + " ".join(comps))
        used = []
        for (layer, t), segs in sorted(self.objs.items()):
            if layer == "poly" or not segs:
                continue
            used.append(f"{'li1' if layer == 'li' else 'met1'} {t}: " + ", ".join(
                f"{n} {lo}-{hi}" if lo != hi else f"{n} {lo}" for lo, hi, n in sorted(segs)))
        gates = " ".join(f"{c}={t}" for c, t in sorted(self.gate_track.items()))
        pins = " ".join(f"{self.gate_net[c]} gate:{c}" for c in self.untracked_gates()) if not self.open_nets() else ""
        return (f"cols: {cs}\nopen: {' | '.join(parts) or 'none'}\n"
                + (f"pins: {pins}\n" if pins else "")
                + f"used: {' ; '.join(used) or 'none'}\ngate tracks: {gates or 'none'}\nnext:")

    def plan_text(self) -> str:
        stars = set()
        for k in self.needed:
            row, net, cols = self.islands[k]
            stars.add((row, self.contact.get(k, cols[0])))
        lines = [f"cell {self.nl['id']}", "in " + " ".join(self.inputs), "out " + " ".join(self.outputs)]
        for i, c in enumerate(self.cols):
            if c[0] == "g":
                lines.append(f"col {i} g n={c[1]} p={c[1]}")
            else:
                n = c[1] + ("*" if ("nmos", i) in stars else "")
                p = c[2] + ("*" if ("pmos", i) in stars else "")
                lines.append(f"col {i} sd n={n} p={p}")
        lines += self.lines
        for net in self.inputs:
            if net in self.pins:
                lines.append(self.pins[net])
            else:
                g = min(c for c, n in self.gate_net.items() if n == net)
                lines.append(f"pin {net} gate:{g}")
        for net in self.outputs:
            for row in ROWS:
                cs = sorted(c for r, c in stars if r == row and self.cols[c][1 if row == "nmos" else 2] == net)
                if cs:
                    lines.append(f"pin {net} {row}:{cs[0]}")
                    break
        return "\n".join(lines) + "\n"


def euler_actions(nl: dict, plan: str) -> list[str]:
    """The route / pin lines of a plan, as actions (routes in plan order, then pins with a track)."""
    acts = [l for l in plan.splitlines() if l.startswith("route ")]
    acts += [l for l in plan.splitlines() if l.startswith("pin ") and len(l.split()) == 4]
    return acts
