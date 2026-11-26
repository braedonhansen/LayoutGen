"""Write the layouts a sample file deploys as .plan + .json pairs, for the ground-truth audit.

    python -m cellgym.llm.gtexport out/llm/k32/step05v2s/samples.jsonl --out out/gtaudit/step05v2s \
        [--k 32] [--extra name=dir:part ...] [--sets ...] [--all-routed] [--jobs 8]
    python -m cellgym.check.audit out/gtaudit/step05v2s --cells out/gtaudit/step05v2s ...   (needs env/SETUP.md)

Per cell: the fewest-column placements among the first K samples are routed at the smallest-area track
count (as `score --select fewcol --tracks area`), and the smallest-area layout the fast checker passes is
written as <set>__<cell>.plan with its netlist next to it (first in file order on ties). --all-routed writes
every routed fewest-column placement instead (<set>__<cell>__<i>.plan), fast-check failures included, so the
audit also sees layouts the fast checker rejects. manifest.json records, per set, cells / written / cells
with no passing layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cellgym.llm.infer import load_sets
from cellgym.llm.score import fewest_column_rows

_CELLS = {}


def _init(sets):
    global _CELLS
    _CELLS = sets


def _one(r):
    """(set, cell, stage, area nm^2 or None, plan text or None)"""
    from cellgym.baselines import common as C
    from cellgym.bench.evaluate import evaluate
    nl = _CELLS[r["set"]][0][r["cell"]]
    if r.get("status") != "done":
        return r["set"], r["cell"], r.get("status", "?"), None, None
    try:
        text = C.route_min_area([[tuple(c) for c in r["cols"]]], nl).text
    except C.BaselineError as e:
        return r["set"], r["cell"], "route:" + e.reason, None, None
    ev = evaluate(text, nl)
    area = ev.width_nm * ev.height_nm if ev.stage == "pass" else None
    return r["set"], r["cell"], ev.stage, area, text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--all-routed", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in a.samples.read_text().splitlines()]
    if a.sets:
        rows = [r for r in rows if r["set"] in a.sets]
    if a.k > 0:
        rows = [r for r in rows if r.get("sample", 0) < a.k]
    rows = fewest_column_rows(rows)
    sets = load_sets(sorted({r["set"] for r in rows}), [tuple(x.split("=", 1)) for x in a.extra])
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(sets,)) as ex:
        res = list(ex.map(_one, rows, chunksize=8))
    by = defaultdict(list)
    for s, c, st, area, text in res:
        by[(s, c)].append((st, area, text))
    a.out.mkdir(parents=True, exist_ok=True)
    man = defaultdict(Counter)
    for (s, c), v in by.items():
        man[s]["cells"] += 1
        nl = json.dumps(sets[s][0][c])
        if a.all_routed:
            keep = [(f"{s}__{c}__{i}", t) for i, (st, ar, t) in enumerate(v) if t]
        else:
            ok = [(ar, i, t) for i, (st, ar, t) in enumerate(v) if ar is not None]
            keep = [(f"{s}__{c}", min(ok)[2])] if ok else []
        if not keep:
            man[s]["no_layout"] += 1
        for key, text in keep:
            (a.out / f"{key}.plan").write_text(text)
            (a.out / f"{key}.json").write_text(nl)
            man[s]["written"] += 1
    (a.out / "manifest.json").write_text(json.dumps({"samples": str(a.samples), "k": a.k, "all_routed": a.all_routed,
                                                    "sets": {s: dict(m) for s, m in man.items()}}, indent=1) + "\n")
    for s, m in man.items():
        print(s, dict(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
