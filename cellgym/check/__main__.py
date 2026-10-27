"""Fast check of plans: python -m cellgym.check PLAN... [--cells DIR] [-v]

Each PLAN is a .plan file (netlist: --cells/<name>.json, else the .json next
to the plan) or a directory of .plan files. Prints per-cell failures and a
timing summary (median / p95 ms).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from cellgym.dsl.plan import parse
from cellgym.render.render import render

from . import check


def plan_files(args) -> list[Path]:
    out = []
    for a in args:
        p = Path(a)
        out += sorted(p.glob("*.plan")) if p.is_dir() else [p]
    return out


def netlist_for(plan: Path, cells: str | None) -> dict:
    if cells and (Path(cells) / f"{plan.stem}.json").exists():
        return json.loads((Path(cells) / f"{plan.stem}.json").read_text())
    return json.loads(plan.with_suffix(".json").read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans", nargs="+")
    ap.add_argument("--cells", default="out/cells")
    ap.add_argument("--repeat", type=int, default=1, help="time each check this many times (min taken)")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args(argv)
    times, bad, rules = [], 0, Counter()
    width_mismatch = 0
    for p in plan_files(a.plans):
        nl = netlist_for(p, a.cells)
        cell = render(parse(p.read_text()), nl)
        res = min((check(cell, nl) for _ in range(a.repeat)), key=lambda r: r.ms)
        times.append(res.ms)
        if round(res.width_um * 1000) != cell.width_nm:
            width_mismatch += 1
            print(f"{p.stem}: width {res.width_um:.3f} um, renderer {cell.width:.3f} um")
        if not res.ok or a.v:
            bad += not res.ok
            rules.update(v.rule for v in res.drc_violations)
            print(f"{p.stem}: {res.summary()}")
            for v in res.drc_violations[:8]:
                print(f"    {v}")
    times.sort()
    n = len(times)
    print(f"cells {n}  failing {bad}  width mismatches vs renderer {width_mismatch}")
    if rules:
        print("rules: " + ", ".join(f"{k} x{v}" for k, v in rules.most_common()))
    if n:
        print(f"time ms: median {statistics.median(times):.2f}  p95 {times[min(n - 1, int(0.95 * n))]:.2f}  max {times[-1]:.2f}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
