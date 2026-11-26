"""Column count of the fewest-column sample against a search-free lower bound.

    python -m cellgym.llm.bound SAMPLES.jsonl [--k 32] [--extra name=dir:part ...] [--exact SECONDS] [--out FILE]

For sets with no Euler reference (the 14-24-per-side probes, where neither the Euler search nor
regret.Solver finishes) this is the only yardstick: columns = 2 * devices per side + 1 + breaks, and
regret.break_lower_bound gives a floor on breaks without search. The floor is loose (it ignores that both
rows share one gate order), so "gap" overstates the distance to the true minimum. Per set: mean breaks of
the fewest-column sample among the first K, mean bound, cells at the bound (provably column-minimal), and
mean columns / bound columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from cellgym.llm.infer import load_sets
from cellgym.llm.regret import break_lower_bound


def breaks(cols) -> int:
    return sum(1 for i in range(1, len(cols)) if cols[i][0] == "sd" and cols[i - 1][0] == "sd")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in a.samples.read_text().splitlines()]
    sets = load_sets(sorted({r["set"] for r in rows}), [tuple(x.split("=", 1)) for x in a.extra])
    best = defaultdict(dict)
    for r in rows:
        if r["status"] == "done" and (a.k <= 0 or r.get("sample", 0) < a.k):
            d = best[r["set"]]
            d[r["cell"]] = min(d.get(r["cell"], 10 ** 6), breaks(r["cols"]))
    out = {}
    print("| set | cells with a placement | mean breaks | mean bound | at the bound | mean columns / bound columns |")
    print("|---|---|---|---|---|---|")
    for s in sets:
        d, cells = best.get(s, {}), sets[s][0]
        if not d:
            continue
        lb = {c: break_lower_bound(cells[c]) for c in d}
        n = {c: len(cells[c]["devices"]) // 2 for c in d}
        ratio = mean((2 * n[c] + 1 + d[c]) / (2 * n[c] + 1 + lb[c]) for c in d)
        out[s] = {"cells": len(cells), "placed": len(d), "mean_breaks": round(mean(d.values()), 3),
                  "mean_bound": round(mean(lb.values()), 3), "at_bound": sum(d[c] == lb[c] for c in d),
                  "cols_over_bound_cols": round(ratio, 4)}
        o = out[s]
        print(f"| {s} | {o['placed']} / {o['cells']} | {o['mean_breaks']:.2f} | {o['mean_bound']:.2f} | {o['at_bound']} | {ratio:.3f} |")
    if a.out:
        a.out.write_text(json.dumps({"samples": str(a.samples), "k": a.k, "sets": out}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
