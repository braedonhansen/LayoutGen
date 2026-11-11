"""Score cellgym.llm.infer samples with the repo pipeline (parse, validate, render, fast check).

    python -m cellgym.llm.score out/llm/step05/samples.jsonl [--jobs 14]

oneshot samples: header + body -> evaluate. step samples: columns -> the Euler baseline's router
(route_best: contacts, routes, pins) -> evaluate; a placement that ended illegal / stuck / at the
column cap, or that the router cannot route, fails with that stage. Writes <samples>.summary.json
(per set: per-sample pass, pass@k, W/W_euler of passes, stage counts) and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cellgym.llm import fmt as F
from cellgym.llm.infer import load_sets

_CELLS = {}


def _init(sets):
    global _CELLS
    _CELLS = sets


def _score(r):
    from cellgym.baselines.common import BaselineError, route_best
    from cellgym.bench.evaluate import evaluate
    cells, w = _CELLS[r["set"]]
    nl = cells[r["cell"]]
    try:
        if "text" in r:
            ev = evaluate(F.plan_header(nl) + r["text"], nl)
        else:
            if r["status"] != "done":
                return r["set"], r["cell"], r["status"], None
            try:
                rt = route_best([[tuple(c) for c in r["cols"]]], nl)
            except BaselineError as e:
                return r["set"], r["cell"], "route:" + e.reason, None
            ev = evaluate(rt.text, nl)
    except Exception as e:  # noqa: BLE001
        return r["set"], r["cell"], "error:" + type(e).__name__, None
    if ev.stage != "pass":
        return r["set"], r["cell"], ev.stage + ":" + ev.detail.split("]")[0].split("[")[-1][:30], None
    we = w.get(r["cell"])
    return r["set"], r["cell"], "pass", (ev.width_nm / we if we else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in a.samples.read_text().splitlines()]
    sets = load_sets(sorted({r["set"] for r in rows}))
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(sets,)) as ex:
        res = list(ex.map(_score, rows, chunksize=8))
    per = defaultdict(lambda: defaultdict(list))
    stages = defaultdict(Counter)
    for s, c, st, wr in res:
        per[s][c].append((st, wr))
        stages[s][st] += 1
    summ = {}
    for s in [x for x, _ in __import__("cellgym.llm.infer", fromlist=["SETS"]).SETS if x in per]:
        cs = per[s]
        n = sum(len(v) for v in cs.values())
        ok = [wr for v in cs.values() for st, wr in v if st == "pass"]
        wrs = [x for x in ok if x]
        summ[s] = {"cells": len(cs), "samples": n, "sample_pass": len(ok) / n,
                   "pass_at_k": sum(any(st == "pass" for st, _ in v) for v in cs.values()) / len(cs),
                   "w_over_euler": sum(wrs) / len(wrs) if wrs else None,
                   "stages": dict(stages[s].most_common(6))}
    out = a.samples.with_suffix(".summary.json")
    out.write_text(json.dumps(summ, indent=1) + "\n")
    print(f"| set | cells | per-sample | pass@k | W/W_euler | top failure |")
    for s, v in summ.items():
        top = next((f"{k} ({c})" for k, c in v["stages"].items() if k != "pass"), "-")
        we = f"{v['w_over_euler']:.3f}" if v["w_over_euler"] else "-"
        print(f"| {s} | {v['cells']} | {100 * v['sample_pass']:.1f} | {100 * v['pass_at_k']:.1f} | {we} | {top} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
