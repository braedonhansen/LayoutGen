"""Is cell area a width problem or a track problem? Width-vs-tracks curves of many placements per cell.

    python -m cellgym.llm.tradeoff [--jobs 12] [--max-cells 60]     # route the pool (resumable cache)
    python -m cellgym.llm.tradeoff --enum [--enum-n 300]            # random legal placements, small cells
    python -m cellgym.llm.tradeoff --report                         # out/tradeoff/report.md (and stdout)

Pool: per (set, cell) the distinct placements of the sample files (cellgym.llm.infer step samples)
plus the stored Euler plan's placement. Every placement is routed at every channel-track count
T = 1..8 (route_best + evaluate, as cellgym.llm.score does for one T); a T that does not route or
does not pass evaluate is "not routable". Cache: out/tradeoff/placements.jsonl, one line per
(set, cell, column hash) with the whole curve {T: [width_nm, height_nm, capped] | null}.

Diffusion breaks of a column list: sd columns beyond gates + 1 (EulerResult.breaks, "extra sd
columns"); row breaks: per adjacent sd-sd pair, the rows (n, p) whose net changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("out/tradeoff")
T_ALL = range(1, 9)
SAMPLES = [("sl", "out/llm/run10/a/out/llm/step05v2/samples.jsonl"),
           ("rl", "out/llm/run10/b/out/llm/rl6/samples.jsonl"),
           ("sl_run7", "out/llm/run7/out/llm/step05s1/samples.jsonl"),
           ("rl_run7", "out/llm/run7/out/llm/rl4/samples.jsonl")]
SUBSAMPLED = ("ood6_dev", "ood7_dev", "ood8_dev", "ood9_all", "ood10_all")
AXIS = {"val": "in distribution", "test_id": "in distribution", "ood6_dev": "5 -> 6 inputs",
        "ood7_dev": "5 -> 7 inputs", "ood8_dev": "5 -> 8 inputs", "ood9_all": "5 -> 9 inputs",
        "ood10_all": "5 -> 10 inputs", "oodstack5_dev": "stack 4 -> 5", "oodstack6_all": "stack 4 -> 6",
        "ooddev10_dev": "9 -> 10 per side", "ooddev11_dev": "9 -> 11 per side",
        "ooddev12_all": "9 -> 12 per side", "adder": "full-adder cells"}
TRACK_NM = 340
# Time guard (--no-guard turns it off): a routing whose search hit the router's node cap is the slow case
# (2-4 s, and every taller channel is slower still). Once a placement already routes at some T, the first
# taller T whose search is capped AND whose area is no smaller than the best so far ends the curve: the
# remaining T are recorded as "skipped:guard" (absent from the curve, NOT counted as unroutable).
GUARD = True


# -- pure helpers ------------------------------------------------------------------------

def col_hash(cols) -> str:
    return hashlib.sha1(json.dumps([list(c) for c in cols], separators=(",", ":")).encode()).hexdigest()[:16]


def count_breaks(cols) -> tuple[int, int]:
    """(diffusion breaks, row breaks) of a column list.
    breaks = sd columns - gate columns - 1; row breaks = rows whose net changes over each adjacent sd-sd pair."""
    n_g = sum(c[0] == "g" for c in cols)
    n_sd = len(cols) - n_g
    rows = 0
    for a, b in zip(cols, cols[1:]):
        if a[0] == "sd" and b[0] == "sd":
            rows += (a[1] != b[1]) + (a[2] != b[2])
    return max(n_sd - n_g - 1, 0), rows


def best_point(curve: dict) -> tuple[int, int, int] | None:
    """(T, width_nm, height_nm) of the smallest-area routable T (ties: fewer tracks); None if none routes."""
    pts = [(v[0] * v[1], int(t), v[0], v[1]) for t, v in curve.items() if v]
    if not pts:
        return None
    _, t, w, h = min(pts)
    return t, w, h


def var_decomp(pairs) -> tuple[float, float, float, float]:
    """(var log w, var log h, 2 cov(log w, log h), var log(w h)) of (w, h) pairs, population moments."""
    n = len(pairs)
    x = [math.log(w) for w, _ in pairs]
    y = [math.log(h) for _, h in pairs]
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((a - mx) ** 2 for a in x) / n
    vy = sum((b - my) ** 2 for b in y) / n
    cv = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    return vx, vy, 2 * cv, vx + vy + 2 * cv


def pearson(x, y) -> float | None:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x)
    sy = sum((b - my) ** 2 for b in y)
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sx * sy)


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else None


def geomean(x):
    x = list(x)
    return math.exp(sum(math.log(v) for v in x) / len(x)) if x else None


# -- routing -----------------------------------------------------------------------------

def route_curve(job):
    """job = (set, cell, cols, netlist, guard) -> cache record with the width-vs-T curve."""
    from cellgym.baselines.common import BaselineError, route_best
    from cellgym.bench.evaluate import evaluate
    s, cid, cols, nl, guard = job
    cols = [tuple(c) for c in cols]
    curve, fail = {}, {}
    t0 = time.time()
    best_area, stop = None, False
    for t in T_ALL:
        if stop:                       # time guard, see GUARD
            fail[str(t)] = "skipped:guard"
            continue
        try:
            rt = route_best([cols], nl, tracks=t)
            ev = evaluate(rt.text, nl)
            if ev.stage == "pass":
                curve[str(t)] = [ev.width_nm, ev.height_nm, int(rt.capped)]
                area = ev.width_nm * ev.height_nm
                if guard and rt.capped and best_area is not None and area >= best_area:
                    stop = True
                best_area = area if best_area is None else min(best_area, area)
                continue
            fail[str(t)] = "eval:" + ev.stage
        except BaselineError as e:
            fail[str(t)] = "route:" + e.reason
        except Exception as e:  # noqa: BLE001
            fail[str(t)] = "error:" + type(e).__name__
        curve[str(t)] = None
    br, rb = count_breaks(cols)
    return {"set": s, "cell": cid, "h": col_hash(cols), "cols": [list(c) for c in cols],
            "ncols": len(cols), "gates": sum(c[0] == "g" for c in cols), "breaks": br, "row_breaks": rb,
            "ndev": sum(d["type"] == "nmos" for d in nl["devices"]),
            "pdev": sum(d["type"] == "pmos" for d in nl["devices"]),
            "curve": curve, "fail": fail, "secs": round(time.time() - t0, 2)}


def read_cache(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except ValueError:          # a line cut off by an interrupted run
                    continue
                out[(r["set"], r["cell"], r["h"])] = r
    return out


def read_samples(files=SAMPLES):
    """{label: [(set, cell, column hash, cols)]} for the finished ("done") samples."""
    out = {}
    for label, f in files:
        rows = []
        if Path(f).exists():
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("status") == "done" and r.get("cols"):
                        rows.append((r["set"], r["cell"], col_hash(r["cols"]), r["cols"]))
        out[label] = rows
    return out


def pick_cells(name: str, cells: list[str], max_cells: int) -> list[str]:
    """All cells, or for the large input sets the max_cells with the smallest sha1(cell id)."""
    if name not in SUBSAMPLED or len(cells) <= max_cells:
        return sorted(cells)
    return sorted(sorted(cells, key=lambda c: hashlib.sha1(c.encode()).hexdigest())[:max_cells])


def run_jobs(jobs, cache_path: Path, n_jobs: int):
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if cache_path.exists() and not cache_path.read_text().endswith("\n"):
        with open(cache_path, "a", encoding="utf-8") as fh:
            fh.write("\n")
    with open(cache_path, "a", encoding="utf-8") as fh, Pool(n_jobs) as pool:
        for i, rec in enumerate(pool.imap_unordered(route_curve, jobs, chunksize=2), 1):
            fh.write(json.dumps(rec) + "\n")
            if i % 100 == 0 or i == len(jobs):
                fh.flush()
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


def build(a) -> int:
    from cellgym.llm import fmt as F
    from cellgym.llm.infer import SETS, load_sets
    samples = read_samples()
    pool = defaultdict(dict)
    for rows in samples.values():
        for s, c, h, cols in rows:
            pool[(s, c)].setdefault(h, cols)
    sets = load_sets()
    spec = dict(SETS)
    euler, jobs, total = {}, [], Counter()
    cache = read_cache(OUT / "placements.jsonl")
    for s, (cells, _) in sets.items():
        if a.sets and s not in a.sets:
            continue
        src = spec[s].split(":")[0]
        pdir = Path("out/plans") if src == "bench" else Path(src) / "plans"
        ids = [c for c in cells if (s, c) in pool]
        total[s] = len(ids)
        euler[s] = {}
        for c in pick_cells(s, ids, a.max_cells):
            p = pdir / f"{c}.plan"
            if p.exists():
                ec = F.plan_columns(p.read_text())
                euler[s][c] = col_hash(ec)
                pool[(s, c)].setdefault(euler[s][c], ec)
            for h, cols in pool[(s, c)].items():
                if (s, c, h) not in cache:
                    jobs.append((s, c, cols, cells[c], GUARD))
    OUT.mkdir(parents=True, exist_ok=True)
    meta_p = OUT / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"euler": {}, "total_cells": {}}
    meta["euler"].update(euler)
    meta["total_cells"].update(total)
    meta_p.write_text(json.dumps(meta) + "\n")
    print(f"{len(jobs)} placements to route ({len(cache)} cached)", file=sys.stderr, flush=True)
    run_jobs(jobs, OUT / "placements.jsonl", a.jobs)
    return 0


# -- random legal placements for small cells ---------------------------------------------------

def _enables_gate(p, c) -> bool:
    gn = {g for (g, t), n in p.left["nmos"].items() if n > 0 and c[1] in t}
    gp = {g for (g, t), n in p.left["pmos"].items() if n > 0 and c[2] in t}
    return bool(gn & gp)


def random_placement(nl: dict, rng: random.Random, p_break: float):
    """One placement by random legal play of fmt.Placer (a break with probability p_break where a gate
    could continue, and wherever none can; break columns that let a gate follow are preferred)."""
    from cellgym.llm import fmt as F
    p = F.Placer(nl)
    cap = 4 * len(nl["devices"]) + 4
    while not p.done():
        leg = p.legal()
        if not leg or len(p.cols) >= cap:
            return None
        if p.cols and p.cols[-1][0] == "g":
            c = rng.choice(leg)
        else:
            gates = [c for c in leg if c[0] == "g"]
            sds = [c for c in leg if c[0] == "sd"]
            if gates and (not sds or rng.random() >= p_break):
                c = rng.choice(gates)
            else:
                c = rng.choice([x for x in sds if _enables_gate(p, x)] or sds)
        if not p.step(c):
            return None
    return p.cols


def enumerate_placements(nl: dict, n: int, seed: int, tries: int = 30000) -> list:
    rng = random.Random(seed)
    seen = {}
    for i in range(tries):
        cols = random_placement(nl, rng, (0.0, 0.0, 0.05, 0.15, 0.3)[i % 5])
        if cols:
            seen.setdefault(col_hash(cols), cols)
            if len(seen) >= n:
                break
    return list(seen.values())


def enum(a) -> int:
    from cellgym.llm.infer import load_sets
    sets = load_sets(["val", "test_id"])
    cache = read_cache(OUT / "enum.jsonl")
    jobs = []
    for s, (cells, _) in sets.items():
        for c, nl in sorted(cells.items()):
            side = Counter(d["type"] for d in nl["devices"])
            if max(side.values()) > a.enum_max_dev:
                continue
            seed = int(hashlib.sha1(c.encode()).hexdigest()[:8], 16)
            for cols in enumerate_placements(nl, a.enum_n, seed):
                if (s, c, col_hash(cols)) not in cache:
                    jobs.append((s, c, cols, nl, GUARD))
    print(f"{len(jobs)} enumerated placements to route ({len(cache)} cached)", file=sys.stderr, flush=True)
    run_jobs(jobs, OUT / "enum.jsonl", a.jobs)
    return 0


# -- analysis ----------------------------------------------------------------------------------

class P:
    """One routed placement at its area-optimal T."""
    __slots__ = ("h", "ncols", "breaks", "t", "w", "hgt", "area", "t0", "wmin", "curve", "capped")

    def __init__(self, rec):
        self.h, self.ncols, self.breaks = rec["h"], rec["ncols"], rec["breaks"]
        self.curve = {int(t): v for t, v in rec["curve"].items() if v}
        self.t, self.w, self.hgt = best_point(rec["curve"])
        self.area = self.w * self.hgt
        self.t0 = min(self.curve)
        self.wmin = min(v[0] for v in self.curve.values())
        self.capped = sum(v[2] for v in self.curve.values())


def to_cells(cache: dict) -> tuple[dict, int]:
    """{(set, cell): {hash: P}} of the placements that route at some T; count of those that never route."""
    cells, dead = defaultdict(dict), 0
    for (s, c, h), rec in cache.items():
        if best_point(rec["curve"]) is None:
            dead += 1
            continue
        cells[(s, c)][h] = P(rec)
    return cells, dead


def trade_stats(cells: dict) -> dict:
    """(a): does the smallest-area placement have more than the fewest columns?"""
    n = yes = wider = 0
    gains, kinds, extra = [], Counter(), []
    for ps in cells.values():
        ps = list(ps.values())
        cmin = min(p.ncols for p in ps)
        bc = min((p for p in ps if p.ncols == cmin), key=lambda p: (p.area, p.t))
        b = min(ps, key=lambda p: (p.area, p.ncols, p.t))
        n += 1
        wider += any(p.ncols > cmin for p in ps)
        g = 1 - b.area / bc.area
        gains.append(g)
        if b.area < bc.area:
            yes += 1
            extra.append(b.ncols - cmin)
            kinds["fewer" if b.t < bc.t else "same" if b.t == bc.t else "more"] += 1
    return {"n": n, "wider": wider, "yes": yes, "gain_mean": mean(gains), "gain_max": max(gains) if gains else None,
            "gain_yes": mean(g for g in gains if g > 0), "kinds": kinds, "extra": mean(extra)}


def variation_stats(cells: dict) -> dict:
    """(b): pooled within-cell variance of log area split into width, height, covariance; correlations."""
    sv = [0.0, 0.0, 0.0, 0.0]
    n = 0
    r = defaultdict(list)
    sd = []
    for ps in cells.values():
        ps = list(ps.values())
        if len(ps) < 3:
            continue
        n += 1
        d = var_decomp([(p.w, p.hgt) for p in ps])
        sv = [x + y for x, y in zip(sv, d)]
        sd.append(math.sqrt(d[3]))
        ar = [p.area for p in ps]
        for k, xs in (("t_a", [p.t for p in ps]), ("c_a", [p.ncols for p in ps]), ("w_a", [p.w for p in ps])):
            v = pearson(xs, ar)
            if v is not None:
                r[k].append(v)
        v = pearson([p.t for p in ps], [p.w for p in ps])
        if v is not None:
            r["t_w"].append(v)
    tot = sv[3] or 1.0
    return {"n": n, "w": sv[0] / tot, "h": sv[1] / tot, "cov": sv[2] / tot, "sd": mean(sd),
            **{k: (mean(v), len(v)) for k, v in r.items()}}


def order_stats(cells: dict, cmin_only: bool) -> dict:
    """(c): spread among placements of one cell with the same column count."""
    groups = []
    for ps in cells.values():
        by = defaultdict(list)
        for p in ps.values():
            by[p.ncols].append(p)
        keys = [min(by)] if cmin_only else list(by)
        groups += [by[k] for k in keys if len(by[k]) >= 2]
    if not groups:
        return {"n": 0}
    tr = [max(p.t for p in g) - min(p.t for p in g) for g in groups]
    t0r = [max(p.t0 for p in g) - min(p.t0 for p in g) for g in groups]
    ar = [max(p.area for p in g) / min(p.area for p in g) - 1 for g in groups]
    wr = [max(p.w for p in g) / min(p.w for p in g) - 1 for g in groups]
    return {"n": len(groups), "size": mean(len(g) for g in groups), "t_diff": mean(x > 0 for x in tr),
            "t_range": mean(tr), "t0_diff": mean(x > 0 for x in t0r), "a_range": mean(ar), "w_range": mean(wr)}


def headroom_stats(cells: dict, euler: dict, refs: dict) -> dict:
    """(d): tracks and area of the pool against the Euler placement; fewest-track vs fewest-width selection."""
    o = defaultdict(list)
    for (s, c), ps in cells.items():
        lst = list(ps.values())
        b = min(lst, key=lambda p: (p.area, p.t))
        e = ps.get(euler.get(s, {}).get(c))
        if e is not None:
            o["tmin_lt"].append(min(p.t for p in lst) < e.t)
            o["tmin"].append(min(p.t for p in lst))
            o["te"].append(e.t)
            o["tb"].append(b.t)
            o["a_e"].append(b.area / e.area)
            o["w_e"].append(b.w / e.w)
            o["h_e"].append(b.hgt / e.hgt)
        ref = refs.get(s, {}).get(c)
        if ref:
            o["a_ref"].append(b.area / ref)
            o["below"].append(b.area < ref)
        tmin = min(p.t0 for p in lst)
        wmin = min(p.wmin for p in lst)
        cmin = min(p.ncols for p in lst)
        for k, sel in (("ft", [p for p in lst if p.t0 == tmin]), ("fw", [p for p in lst if p.wmin == wmin]),
                       ("fc", [p for p in lst if p.ncols == cmin])):
            x = min(p.area for p in sel)
            o[k + "_hit"].append(x == b.area)
            o[k + "_ex"].append(x / b.area - 1)
    return {"n": len(cells), "n_e": len(o["te"]), "n_ref": len(o["a_ref"]), **{k: mean(v) for k, v in o.items()}}


def single_stats(cells: dict) -> dict:
    """(e): within one placement, optimal T against the fewest routable T."""
    ps = [p for d in cells.values() for p in d.values()]
    up = [p for p in ps if p.t > p.t0]
    first = [1 - p.curve[p.t0 + 1][0] / p.curve[p.t0][0] for p in ps if p.t0 + 1 in p.curve]
    best = [min(d.values(), key=lambda p: (p.area, p.t)) for d in cells.values()]
    pts = sum(len(p.curve) for p in ps)
    return {"n": len(ps), "above": len(up) / len(ps), "extra": mean(p.t - p.t0 for p in ps),
            "extra_hist": dict(sorted(Counter(p.t - p.t0 for p in ps).items())),
            "shrink": mean((1 - p.w / p.curve[p.t0][0]) / (p.t - p.t0) for p in up),
            "first": mean(first), "first_pay": mean(1 - p.curve[p.t0 + 1][0] / p.curve[p.t0][0] > TRACK_NM / (p.curve[p.t0][1] + TRACK_NM)
                                                    for p in ps if p.t0 + 1 in p.curve),
            "track_pct": mean(TRACK_NM / p.hgt for p in ps), "col_pct": mean(1 / p.ncols for p in ps),
            "cell_above": mean(p.t > p.t0 for p in best), "t0": mean(p.t0 for p in ps), "t": mean(p.t for p in ps),
            "capped": sum(p.capped for p in ps) / pts}


def policy_stats(cells: dict, rows: list) -> dict:
    """(f): the samples of one file against the pool's best placement of the same cell (per-sample means)."""
    o = defaultdict(list)
    for s, c, h, _ in rows:
        ps = cells.get((s, c))
        if not ps or h not in ps:
            continue
        p = ps[h]
        b = min(ps.values(), key=lambda q: (q.area, q.ncols, q.t))
        o["t"].append(p.t)
        o["tb"].append(b.t)
        o["c"].append(p.ncols)
        o["cb"].append(b.ncols)
        o["a"].append(p.area / b.area)
        o["w"].append(p.w / b.w)
        o["h"].append(p.hgt / b.hgt)
    if not o["a"]:
        return {"n": 0}
    return {"n": len(o["a"]), "t": mean(o["t"]), "tb": mean(o["tb"]), "c": mean(o["c"]), "cb": mean(o["cb"]),
            "a": mean(o["a"]), "ga": geomean(o["a"]), "gw": geomean(o["w"]), "gh": geomean(o["h"])}


# -- report --------------------------------------------------------------------------------

def tbl(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def pc(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def f2(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def report(a) -> int:
    from cellgym.llm import arearef
    from cellgym.llm.infer import SETS
    cache = read_cache(OUT / "placements.jsonl")
    meta = json.loads((OUT / "meta.json").read_text())
    euler = meta["euler"]
    cells, dead = to_cells(cache)
    samples = read_samples()
    order = [s for s, _ in SETS if any(k[0] == s for k in cells)]
    refs = {s: arearef.areas_nm2(arearef.load(s, Path("out/arearef"), "euler")) for s in order}
    groups = [(s, {k: v for k, v in cells.items() if k[0] == s}) for s in order] + [("ALL", cells)]
    src = defaultdict(set)
    for label, rows in samples.items():
        for s, c, h, _ in rows:
            src[(s, c, h)].add(label)
    L = []
    w = L.append
    w("# Area: a width problem or a track problem?\n")
    w("Generated by `python -m cellgym.llm.tradeoff --report` from `out/tradeoff/placements.jsonl`.\n")
    w("**Policies were trained on** <= 5 inputs, <= 9 devices per side, stack <= 4 (val / test_id are the only "
      "in-distribution sets). **Everything else below is tested on cells outside that range**; the axis column says how far.\n")
    w("Pool per cell: the distinct placements among 4 x 8 policy samples (SL step05v2, RL rl6, SL step05s1, RL rl4) "
      "plus the stored Euler placement. Each placement is routed at every T = 1..8; its area is taken at its own "
      "area-optimal T. Height = cell base + 340 nm per track.\n")
    rows = []
    for s, g in groups:
        n = len(g)
        pl = sum(len(v) for v in g.values())
        tot = sum(meta["total_cells"].get(x, 0) for x in ({k[0] for k in g}))
        ne = sum(1 for (ss, c) in g if euler.get(ss, {}).get(c) in g[(ss, c)])
        rows.append((s, AXIS.get(s, ""), f"{n} of {tot}", pl, f2(pl / n, 1), ne,
                     f2(mean(p.ncols for v in g.values() for p in v.values()), 1),
                     f2(mean(p.breaks for v in g.values() for p in v.values()), 2)))
    w("## Pool\n")
    w(tbl(["set", "trained up to -> tested", "cells analysed", "placements", "per cell", "cells with Euler placement",
           "mean columns", "mean breaks"], rows))
    w(f"Placements that route at no T in 1..8 (dropped): {dead} of {len(cache)}.\n")

    w("## a. Column-for-track trades\n")
    w("Cmin = fewest columns among the cell's pooled placements. \"Wider wins\" = the smallest-area placement has more "
      "than Cmin columns and is strictly smaller than the best Cmin-column placement. Gain = 1 - best area / best "
      "Cmin-column area (0 where the Cmin placement is the best). The last three columns split the wins by the "
      "winner's track count against the best Cmin placement's.\n")

    def trade_rows(grps):
        rows = []
        for s, g in grps:
            t = trade_stats(g)
            k = t["kinds"]
            rows.append((s, t["n"], t["wider"], f"{t['yes']} ({pc(t['yes'] / t['n'])})", pc(t["gain_mean"], 2),
                         pc(t["gain_yes"], 2), pc(t["gain_max"], 1), f2(t["extra"], 2), k["fewer"], k["same"], k["more"]))
        return rows
    th = ["set", "cells", "cells with a > Cmin placement in the pool", "wider wins", "mean gain (all cells)",
          "mean gain (wins)", "max gain", "extra columns of winner", "win by fewer tracks", "same tracks", "more tracks"]
    w(tbl(th, trade_rows(groups)))

    w("## b. Where the area variation lives\n")
    w("Within each cell (>= 3 distinct placements), over placements at their own optimal T: var log(area) = "
      "var log(width) + var log(height) + 2 cov. Shares are pooled within-cell sums. Correlations are means of the "
      "per-cell Pearson r (cells where both vary; count in brackets).\n")
    rows = []
    for s, g in groups:
        v = variation_stats(g)
        if not v["n"]:
            continue
        cr = lambda k: f"{f2(v[k][0])} ({v[k][1]})" if k in v else "-"
        rows.append((s, v["n"], pc(v["sd"], 2), pc(v["w"]), pc(v["h"]), pc(v["cov"]), cr("t_a"), cr("c_a"), cr("w_a"), cr("t_w")))
    w(tbl(["set", "cells", "within-cell sd of log area", "width share", "height share", "2 cov share",
           "r(tracks, area)", "r(columns, area)", "r(width, area)", "r(tracks, width)"], rows))

    w("## c. Same columns, different order\n")
    w("Groups of placements of one cell with the same column count (>= 2 placements). First block: every such group; "
      "second block: only each cell's Cmin group. T* = area-optimal tracks, T0 = fewest routable tracks; ranges are "
      "max - min within the group, area and width ranges relative to the group minimum.\n")
    for flag, name in ((False, "all column counts"), (True, "Cmin group only")):
        rows = []
        for s, g in groups:
            v = order_stats(g, flag)
            if v["n"]:
                rows.append((s, v["n"], f2(v["size"], 1), pc(v["t_diff"]), f2(v["t_range"]), pc(v["t0_diff"]),
                             pc(v["a_range"], 2), pc(v["w_range"], 2)))
        w(f"**{name}**\n")
        w(tbl(["set", "groups", "placements per group", "T* differs", "mean T* range", "T0 differs",
               "mean area range", "mean width range"], rows))

    w("## d. Track headroom\n")
    w("(i) fewest optimal-T in the pool against the Euler placement's optimal-T; (ii) the pool's best area against the "
      "Euler placement (full T curve) and against the stored Euler reference (`out/arearef`, route_min_area); "
      "(iii) picking the best area only among the placements that route at the fewest tracks (min T0), only among the "
      "narrowest (min width over any T), or only among the fewest-column ones: how often each finds the pool optimum, "
      "and the mean area lost.\n")
    r1, r2 = [], []
    for s, g in groups:
        v = headroom_stats(g, euler, refs)
        r1.append((s, v["n_e"], f2(v.get("te")), f2(v.get("tmin")), pc(v.get("tmin_lt")), f2(v.get("tb")),
                   f2(v.get("a_e"), 3), f2(v.get("w_e"), 3), f2(v.get("h_e"), 3), f2(v.get("a_ref"), 3), pc(v.get("below"))))
        r2.append((s, v["n"], pc(v["ft_hit"]), pc(v["ft_ex"], 2), pc(v["fw_hit"]), pc(v["fw_ex"], 2),
                   pc(v["fc_hit"]), pc(v["fc_ex"], 2)))
    w(tbl(["set", "cells with Euler", "Euler T*", "pool min T*", "cells with pool min T* < Euler T*", "T* of pool best",
           "best area / Euler placement", "width ratio", "height ratio", "best area / arearef", "cells below arearef"], r1))
    w(tbl(["set", "cells", "fewest-track finds optimum", "area lost", "fewest-width finds optimum", "area lost",
           "fewest-column finds optimum", "area lost"], r2))

    w("## e. Within a single placement\n")
    w("T0 = fewest tracks that route the placement, T* = its area-optimal tracks. \"Width shrink per extra track\" is "
      "over placements with T* > T0, (1 - W(T*) / W(T0)) / (T* - T0). \"First extra track\" is 1 - W(T0+1) / W(T0) over "
      "all placements, and how often it exceeds the height cost of that track. The exchange rate: one track as a share "
      "of height, one column as a share of the column count.\n")
    rows = []
    for s, g in groups:
        v = single_stats(g)
        rows.append((s, v["n"], f2(v["t0"]), f2(v["t"]), pc(v["above"]), " ".join(f"+{k}:{n}" for k, n in v["extra_hist"].items()),
                     pc(v["shrink"], 2), pc(v["first"], 2), pc(v["first_pay"]), pc(v["track_pct"], 2), pc(v["col_pct"], 2),
                     pc(v["cell_above"]), pc(v["capped"])))
    w(tbl(["set", "placements", "mean T0", "mean T*", "T* > T0", "T* - T0 histogram", "width shrink per extra track",
           "first extra track: width shrink", "first extra track pays", "one track / height", "one column / columns",
           "cells whose best placement has T* > T0", "routed points with capped search"], rows))

    w("## f. Policy vs pool\n")
    w("Per sample (8 per cell, duplicates kept), against the pool's best-area placement of the same cell. The area gap "
      "is the geometric mean of sample area / best area, which factors exactly into a width ratio times a height ratio.\n")
    for label, name in (("sl", "SL (step05v2)"), ("rl", "RL (rl6)")):
        rows = []
        for s, g in groups:
            v = policy_stats(g, samples.get(label, []))
            if v["n"]:
                gap = v["ga"] - 1
                share = math.log(v["gh"]) / math.log(v["ga"]) if v["ga"] > 1 else None
                rows.append((s, v["n"], f2(v["t"]), f2(v["tb"]), f2(v["c"], 1), f2(v["cb"], 1), f2(v["a"], 3),
                             f"{v['ga']:.3f} = {v['gw']:.3f} x {v['gh']:.3f}", pc(gap), pc(share, 0)))
        w(f"**{name}**\n")
        w(tbl(["set", "samples", "mean T*", "T* of pool best", "mean columns", "columns of pool best",
               "mean area / best", "geo area = width x height", "gap", "height share of the gap (log)"], rows))

    enum_cache = read_cache(OUT / "enum.jsonl")
    en = None
    if enum_cache:
        w("## Larger pool for small cells (random legal placements)\n")
        ec, edead = to_cells(enum_cache)
        merged = defaultdict(dict)
        for k, v in ec.items():
            merged[k].update(v)
            merged[k].update(cells.get(k, {}))
        eg = [(s, {k: v for k, v in merged.items() if k[0] == s}) for s in ("val", "test_id")]
        eg = [(s, g) for s, g in eg if g] + [("ALL small", dict(merged))]
        nd = {k[:2]: r["ndev"] for k, r in enum_cache.items()}
        w(f"val / test_id cells with <= {max(nd.values())} devices per side: up to {a.enum_n} distinct placements per cell "
          "by random legal play of `fmt.Placer` (breaks allowed), each routed at every T, merged with the cell's "
          f"main pool. {len(merged)} cells, {sum(len(v) for v in merged.values())} routable placements "
          f"(mean {mean(len(v) for v in merged.values()):.0f} per cell, min {min(len(v) for v in merged.values())}); "
          f"{edead} never route.\n")
        w(tbl(th, trade_rows(eg)))
        rows = []
        for s, g in eg:
            small = {k: cells[k] for k in g if k in cells}
            hb = headroom_stats(g, euler, refs)
            best_main = [min(p.area for p in small[k].values()) for k in g if k in small]
            best_big = [min(p.area for p in g[k].values()) for k in g if k in small]
            v = variation_stats(g)
            sg = single_stats(g)
            rows.append((s, len(g), f2(mean(b / m for b, m in zip(best_big, best_main)), 3),
                         pc(mean(b < m for b, m in zip(best_big, best_main))), f2(hb.get("a_ref"), 3),
                         pc(v["w"]), pc(v["h"]), pc(v["cov"]), pc(hb["ft_hit"]), pc(hb["fw_hit"]), pc(hb["fc_hit"]),
                         pc(sg["above"])))
        w(tbl(["set", "cells", "best area: large pool / main pool", "cells improved", "best area / arearef",
               "width share", "height share", "2 cov share", "fewest-track finds optimum", "fewest-width finds optimum",
               "fewest-column finds optimum", "T* > T0 (placements)"], rows))
        en = trade_stats(dict(merged))
        en["improve"] = mean(b / m for b, m in zip(best_big, best_main))

    w("## What this says\n")
    guarded = sum(any(v == "skipped:guard" for v in r["fail"].values()) for r in cache.values())
    w(conclusions(groups, euler, refs, samples, en, guarded / max(len(cache), 1)))
    text = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    return 0


def conclusions(groups, euler, refs, samples, en, guarded=0.0) -> str:
    """Plain-language summary; every number is computed from the cache."""
    rows = []
    for s, g in groups:
        t = trade_stats(g)
        o = order_stats(g, True)
        sl = policy_stats(g, samples.get("sl", []))
        rl = policy_stats(g, samples.get("rl", []))
        k = t["kinds"]
        rows.append((s, AXIS.get(s, ""), pc(t["gain_mean"], 2), f"{k['fewer']} of {t['n']}",
                     pc(o.get("a_range"), 2) if o["n"] else "-",
                     pc(sl["gw"] - 1) if sl["n"] else "-", pc(sl["gh"] - 1) if sl["n"] else "-",
                     pc(rl["gw"] - 1) if rl["n"] else "-", pc(rl["gh"] - 1) if rl["n"] else "-"))
    allc = groups[-1][1]
    t = trade_stats(allc)
    v = variation_stats(allc)
    h = headroom_stats(allc, euler, refs)
    sg = single_stats(allc)
    rl = policy_stats(allc, samples.get("rl", []))
    sl = policy_stats(allc, samples.get("sl", []))
    k = t["kinds"]
    o = order_stats(allc, True)
    big = [g for s, g in groups if s.startswith(("oodstack", "ooddev"))]
    rb = policy_stats({k: x for g in big for k, x in g.items()}, samples.get("rl", []))
    out = ["**Bottom line: in this template area is a width problem, and width is a column-count problem.** "
           f"Within a cell, area follows the column count (r = {v['c_a'][0]:.2f}) and the width (r = {v['w_a'][0]:.2f}), "
           f"not the tracks (r = {v['t_a'][0]:.2f}). There is no column-for-track lever worth training for "
           f"({pc(t['gain_mean'], 2)} of area on average). What remains for the policy is fewer breaks and a better order, "
           "which lower width and tracks together rather than trading one for the other.\n",
           "Size of each prize, in % of cell area (tested sets are outside the training range except val / test_id):\n",
           tbl(["set", "trained up to -> tested", "column-for-track prize (mean gain of best over best Cmin placement)",
                "cells where a wider placement wins with fewer tracks", "ordering prize at Cmin columns (mean area range)",
                "SL gap to pool best: width", "SL gap: height", "RL gap: width", "RL gap: height"], rows)]
    out.append(
        f"- **Column-for-track trades exist but are rare and small.** Over {t['n']} cells, a placement with more than the "
        f"fewest columns is the smallest in {t['yes']} ({pc(t['yes'] / t['n'])}); in {k['fewer']} of those it wins with fewer "
        f"tracks, in {k['same']} with the same tracks (it simply compacts narrower) and in {k['more']} with more. Averaged "
        f"over all cells the prize is {pc(t['gain_mean'], 2)} of area ({pc(t['gain_yes'], 2)} where it exists, "
        f"max {pc(t['gain_max'], 1)}). Selecting only fewest-column placements finds the pool optimum in {pc(h['fc_hit'])} of cells.\n"
        f"- **Why the trade is weak:** one track is {pc(sg['track_pct'], 1)} of the height and one column is about "
        f"{pc(sg['col_pct'], 1)} of the width, so a column buys a track at roughly break-even; a break pays only when it "
        "saves more than one track or also shortens the compacted width.\n"
        f"- **Area variation between placements of a cell is {pc(v['w'], 0)} width, {pc(v['h'], 0)} height, {pc(v['cov'], 0)} "
        f"covariance** (pooled within-cell variance of log area; within-cell sd {pc(v['sd'], 1)}). "
        f"Tracks and width are {'negatively' if v['t_w'][0] < 0 else 'positively'} correlated within a cell "
        f"(r = {v['t_w'][0]:.2f}): "
        + ("placements that need more tracks are also the wider ones, so the two levers point the same way."
           if v["t_w"][0] > 0 else "narrower placements tend to need more tracks, so part of a width gain is paid back in height.")
        + "\n"
        f"- **Track headroom below Euler is small.** The pool's fewest optimal-T is below the Euler placement's in "
        f"{pc(h.get('tmin_lt'))} of cells, but the pool's best-area placement is {f2(h.get('a_e'), 3)} of the Euler placement's "
        f"area (width x{f2(h.get('w_e'), 3)}, height x{f2(h.get('h_e'), 3)}). Choosing by fewest tracks finds the area "
        f"optimum in {pc(h['ft_hit'])} of cells (loses {pc(h['ft_ex'], 2)}); choosing by fewest width finds it in "
        f"{pc(h['fw_hit'])} (loses {pc(h['fw_ex'], 2)}).\n"
        f"- **The policy's gap to the best known placement:** SL {pc(sl['ga'] - 1)} = width {pc(sl['gw'] - 1)} x height "
        f"{pc(sl['gh'] - 1)}; RL {pc(rl['ga'] - 1)} = width {pc(rl['gw'] - 1)} x height {pc(rl['gh'] - 1)} "
        f"(mean T* {rl['t']:.2f} against {rl['tb']:.2f} for the best placement). The height part is the track prize "
        "still on the table for the policy; the width part is what RL has been collecting."
        + (f" On the stack and 10-12-per-side sets the height part is larger: RL {pc(rb['ga'] - 1)} = width "
           f"{pc(rb['gw'] - 1)} x height {pc(rb['gh'] - 1)}, with {rb['t'] - rb['tb']:.2f} more tracks and "
           f"{rb['c'] - rb['cb']:.1f} more columns than the best placement. The extra tracks come with the extra "
           "columns (more breaks), so removing breaks is what would recover them." if rb["n"] else "") + "\n"
        f"- **Ordering at a fixed column count is a real lever, mostly through width.** Among a cell's fewest-column "
        f"placements ({o['n']} cells with >= 2 of them) the area range is {pc(o['a_range'], 1)} and the width range "
        f"{pc(o['w_range'], 1)}; optimal tracks differ in {pc(o['t_diff'], 0)} of those cells (mean range {o['t_range']:.2f} "
        "tracks), but the narrower orderings tend to be the ones that need more tracks, so the area range is smaller "
        "than the width range.\n"
        f"- **Within one placement** the area-optimal T is above the fewest routable T for {pc(sg['above'])} of placements "
        f"(width shrinks {pc(sg['shrink'], 1)} per extra track there, against a height cost of {pc(sg['track_pct'], 1)}): "
        "this is the router's choice and needs no learning.\n")
    if en:
        ek = en["kinds"]
        out.append(
            f"- **Larger pool, small cells only:** with random legal placements added, a wider placement wins in {en['yes']} of "
            f"{en['n']} cells ({ek['fewer']} with fewer tracks, {ek['same']} same, {ek['more']} more), mean prize "
            f"{pc(en['gain_mean'], 2)}, max {pc(en['gain_max'], 1)}; the best area in the large pool is "
            f"{en['improve']:.3f} of the main pool's best.\n")
    out.append(
        "- **Limitations.** The main pool is a sample (about 5-25 distinct placements per cell, drawn from policies that "
        "imitate Euler placements, so it is biased toward few-column placements), not an exhaustive search: \"no trade "
        "found\" is evidence, not proof. The large-pool check covers only small in-distribution cells. Widths at a "
        f"given T come from a router with a node cap ({pc(sg['capped'])} of routed points hit it, mostly high T on "
        "large cells), so a few curves are not monotone and high-T widths are upper bounds. To bound run time, a "
        f"placement's curve ends at the first T whose search is capped and whose area is no better than a lower T "
        f"({pc(guarded)} of placements have such a truncated curve; the missing points are taller channels, so this can "
        "only hide a win by MORE tracks, never one by fewer). The five large input sets "
        "were subsampled deterministically (see the Pool table).\n")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--enum", action="store_true", help="route random legal placements of small val/test_id cells")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--max-cells", type=int, default=60, help="cells kept per large input set (ood6..ood10)")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--enum-n", type=int, default=300)
    ap.add_argument("--enum-max-dev", type=int, default=6, help="devices per side")
    ap.add_argument("--no-guard", action="store_true", help="route every T even after a capped, no-better routing")
    a = ap.parse_args(argv)
    global GUARD
    GUARD = not a.no_guard
    if a.report:
        return report(a)
    if a.enum:
        return enum(a)
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
