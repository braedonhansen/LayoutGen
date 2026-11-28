"""Training cells beyond the reach of the Euler search, labelled by the classical placer.

    python -m cellgym.llm.bigcorpus --out out/synthbig --single 4000 --per-side 10:32 --multi 3:1000 4:1000
        [--tries 1000] [--placements 2] [--exclude out/probe/i8d14 ...] [--seed 9300] [--jobs 8]

Single-stage cells: per side uniform in --per-side, inputs uniform in [per_side/3, min(per_side, 16)], stack
limit max(4, probegen.min_stack(per_side)) (probegen.sample_networks_fast). Multi-stage cells: `--multi
STAGES:COUNT`, per side 4*STAGES..7*STAGES, 1-2 outputs (probegen.sample_multistage). Every netlist passes the
switch-level check. Label: the fewest-break placements among --tries tries of cellgym.llm.classical.Greedy
(rule smart), up to --placements distinct ones per cell, replayed through fmt.Placer before they are kept.
These are heuristic labels, not minimal ones: `breaks` is what the heuristic found and `bound` is
regret.break_lower_bound. The plan text holds only the column lines (what cellgym.llm.build reads); nothing
is routed. Rows are plans.jsonl rows as cellgym.synth.generate writes them (origin "big1" / "big<stages>").
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cellgym.llm import fmt as F
from cellgym.llm import probegen as G


def _single(job):
    n_in, per, seed = job
    from cellgym.llm.probe import single_json
    nets = G.sample_networks_fast(n_in, per, max(4, G.min_stack(per)), 1, seed=seed, tries=4000)
    out = []
    for net in nets:
        nl = single_json(net)
        if not G.check_netlist(nl, G.expected_single(net)):
            out.append(nl)
    return out


def _multi(job):
    stages, count, seed = job
    from cellgym.synth.generate import SIZING
    cells = G.sample_multistage(stages, count, (4 * stages, 7 * stages), (1, 2), seed=seed,
                                max_primary=min(4 + stages, 8), sizing=SIZING, tries=40_000)
    return [nl for nl in cells if not G.check_netlist(nl, {o: int(f, 16) for o, f in nl["function"].items()})]


def plan_text(nl: dict, cols: list[tuple]) -> str:
    outs = [p for p in nl["ports"] if p not in nl["inputs"] and p not in G.SUPPLY]
    lines = [f"cell {nl['id']}", "in " + " ".join(nl["inputs"]), "out " + " ".join(outs)]
    for i, c in enumerate(cols):
        lines.append(f"col {i} g n={c[1]} p={c[1]}" if c[0] == "g" else f"col {i} sd n={c[1]} p={c[2]}")
    return "\n".join(lines) + "\n"


def _label(job):
    nl, tries, keep = job
    from cellgym.llm.classical import Greedy
    from cellgym.llm.regret import break_lower_bound
    g, rng = Greedy(nl), random.Random(zlib.crc32(nl["id"].encode()))
    best, found = 10 ** 6, {}
    for _ in range(tries):
        path = []
        b = g.one(rng, path)
        if b < best:
            best, found = b, {}
        if b == best and len(found) < keep:
            found.setdefault(tuple(path), None)
    plans = []
    for cols in found:
        p = F.Placer(nl)
        if all(p.step(c) for c in cols) and p.done():
            plans.append({"plan": plan_text(nl, list(cols)), "breaks": best, "columns": len(cols)})
    return nl["id"], plans, break_lower_bound(nl)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--single", type=int, default=4000)
    ap.add_argument("--per-side", default="10:32")
    ap.add_argument("--multi", nargs="*", default=[], help="STAGES:COUNT ...")
    ap.add_argument("--tries", type=int, default=1000)
    ap.add_argument("--placements", type=int, default=2)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=9300)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.time()
    banned = {f.stem for d in a.exclude for f in (d / "cells").glob("*.json")}
    lo, hi = map(int, a.per_side.split(":"))
    rng = random.Random(a.seed)
    sj = []
    for i in range(int(a.single * 1.15)):
        per = rng.randint(lo, hi)
        sj.append((rng.randint(max(2, per // 3), min(per, 16)), per, a.seed * 1000 + i))
    mj = []
    for spec in a.multi:
        st, cnt = map(int, spec.split(":"))
        chunk = 10
        mj += [(st, chunk, a.seed * 1000 + st * 100000 + i) for i in range(-(-int(cnt * 1.1) // chunk))]
    want = {int(s.split(":")[0]): int(s.split(":")[1]) for s in a.multi}
    cells, origin, have = {}, {}, Counter()

    def add(nl, o, limit):
        if nl["id"] not in banned and nl["id"] not in cells and have[o] < limit:
            cells[nl["id"]], origin[nl["id"]] = nl, o
            have[o] += 1

    with ProcessPoolExecutor(a.jobs) as ex:
        for out in ex.map(_single, sj, chunksize=8):
            for nl in out:
                add(nl, "big1", a.single)
        print(f"single-stage cells {len(cells)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        for (st, _, _), out in zip(mj, ex.map(_multi, mj)):
            for nl in out:
                add(nl, f"big{st}", want[st])
        print(f"all cells {len(cells)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        order = sorted(cells)
        res = list(ex.map(_label, [(cells[c], a.tries, a.placements) for c in order], chunksize=4))
    records, gaps = [], []
    for cid, plans, lb in res:
        nl = cells[cid]
        for i, p in enumerate(plans):
            records.append({"id": cid, "plan_index": i, "topology_id": cid, "origin": origin[cid], "bound": lb,
                            "n_inputs": len(nl["inputs"]), "n_devices": nl["n_devices"], "netlist": nl, **p})
        if plans:
            gaps.append(plans[0]["breaks"] - lb)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plans.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records), newline="\n")
    st = [G.stats(cells[c]) for c in order]
    manifest = {"args": " ".join(sys.argv[1:] if argv is None else map(str, argv)), "records": len(records),
                "cells": len(cells), "by_origin": dict(have), "excluded_dirs": [str(d) for d in a.exclude],
                "axes_min_max": {k: [min(s[k] for s in st), max(s[k] for s in st)] for k in st[0]},
                "by_per_side": dict(sorted(Counter(s["per_side"] for s in st).items())),
                "by_stack": dict(sorted(Counter(s["stack"] for s in st).items())),
                "label": f"classical.Greedy smart, best of {a.tries}, up to {a.placements} placements",
                "mean_breaks": round(sum(r["breaks"] for r in records) / len(records), 3),
                "mean_breaks_over_bound": round(sum(gaps) / len(gaps), 3), "cells_at_bound": sum(g == 0 for g in gaps),
                "seconds": round(time.time() - t0, 1)}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", newline="\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
