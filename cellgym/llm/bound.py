"""Column count of the fewest-column sample against a search-free lower bound.

    python -m cellgym.llm.bound SAMPLES.jsonl [--k 32] [--extra name=dir:part ...] [--exact SECONDS] [--out FILE]

--exact S also runs regret.Solver (exact fewest breaks, memoised search) with S seconds per cell and reports
on how many solved cells the model's fewest-column sample is at the exact minimum.

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


def _exact(job):
    """(set, cell, exact fewest breaks or None on timeout / memory)"""
    import resource
    import signal
    from cellgym.llm.regret import Solver
    s, c, nl, lim = job

    def stop(*_):
        raise TimeoutError
    sys.setrecursionlimit(100000)
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(lim)
    try:
        r = Solver(nl).restart(0, 0)
    except (TimeoutError, MemoryError, RecursionError):
        r = None
    signal.alarm(0)
    return s, c, r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--exact", type=int, default=0, help="also the exact minimum (regret.Solver), SECONDS per cell")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in a.samples.read_text().splitlines()]
    sets = load_sets(sorted({r["set"] for r in rows}), [tuple(x.split("=", 1)) for x in a.extra])
    best = defaultdict(dict)
    for r in rows:
        if r["status"] == "done" and (a.k <= 0 or r.get("sample", 0) < a.k):
            d = best[r["set"]]
            d[r["cell"]] = min(d.get(r["cell"], 10 ** 6), breaks(r["cols"]))
    exact = defaultdict(dict)
    if a.exact:
        from multiprocessing import Pool
        jobs = [(s, c, sets[s][0][c], a.exact) for s in sets for c in best.get(s, {})]
        with Pool(a.jobs, maxtasksperchild=20) as pool:
            for s, c, r in pool.imap_unordered(_exact, jobs, chunksize=4):
                if r is not None:
                    exact[s][c] = r
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
        if a.exact:
            e = exact[s]
            out[s] |= {"exact_solved": len(e), "at_exact": sum(d[c] == e[c] for c in e),
                       "mean_exact": round(mean(e.values()), 3) if e else None,
                       "mean_breaks_on_solved": round(mean(d[c] for c in e), 3) if e else None,
                       "bound_tight_on_solved": sum(lb[c] == e[c] for c in e)}
        o = out[s]
        print(f"| {s} | {o['placed']} / {o['cells']} | {o['mean_breaks']:.2f} | {o['mean_bound']:.2f} | {o['at_bound']} | {ratio:.3f} |"
              + (f" exact: solved {o['exact_solved']}, model at the minimum {o['at_exact']}, mean minimum {o['mean_exact']}" if a.exact else ""))
    if a.out:
        a.out.write_text(json.dumps({"samples": str(a.samples), "k": a.k, "sets": out}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
