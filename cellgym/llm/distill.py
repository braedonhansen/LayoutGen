"""Self-distillation corpus: the step policy's own fewest-column placements on training cells.

    python -m cellgym.llm.distill sample --model DIR --plans A.jsonl [B.jsonl] --cells 5000 --k 8 --out out/distill1
    python -m cellgym.llm.distill select --out out/distill1 [--keep 3]

`sample` draws cells (by id) from the given training plans.jsonl files and samples k placements each with
the step policy (same sampler as cellgym.llm.infer). `select` keeps, per cell, up to --keep distinct
complete placements with the fewest columns, and only when that count is no larger than the cell's label
(Euler) column count; it writes plans.jsonl rows that cellgym.llm.build accepts (col lines only, no
routes). No router is involved. Only cells of the given corpora are used, so the trained axes do not widen.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from cellgym.llm import fmt as F


def load_cells(plans: list[Path]) -> dict:
    """cell id -> {netlist, topology_id, origin, label_cols (fewest over the cell's labels)}."""
    cells = {}
    for f in plans:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            n = len(F.plan_columns(r["plan"]))
            c = cells.setdefault(r["id"], {"netlist": r["netlist"], "topology_id": r["topology_id"],
                                           "origin": r.get("origin", ""), "label_cols": n})
            c["label_cols"] = min(c["label_cols"], n)
    return cells


def plan_text(nl: dict, cols: list) -> str:
    lines = [F.plan_header(nl).rstrip("\n")]
    for i, c in enumerate(cols):
        lines.append(f"col {i} g n={c[1]} p={c[1]}" if c[0] == "g" else f"col {i} sd n={c[1]} p={c[2]}")
    return "\n".join(lines) + "\n"


def select(cells: dict, samples: list[dict], keep: int) -> tuple[list[dict], dict]:
    by = defaultdict(list)
    for s in samples:
        if s["status"] == "done":
            by[s["cell"]].append([tuple(c) for c in s["cols"]])
    rows, st = [], Counter()
    excess = []
    for cid in sorted(by):
        c = cells[cid]
        ps = by[cid]
        excess += [len(p) - c["label_cols"] for p in ps]
        best = min(len(p) for p in ps)
        st["cells"] += 1
        st["below_label" if best < c["label_cols"] else "at_label" if best == c["label_cols"] else "above_label"] += 1
        if best > c["label_cols"]:
            continue
        seen = []
        for p in ps:
            if len(p) == best and p not in seen:
                seen.append(p)
        for j, p in enumerate(seen[:keep]):
            rows.append({"id": cid, "topology_id": c["topology_id"], "origin": "distill", "plan_index": j,
                         "netlist": c["netlist"], "plan": plan_text(c["netlist"], p), "n_cols": best,
                         "label_cols": c["label_cols"]})
    st = dict(st)
    st.update(plans=len(rows), samples_done=len(excess),
              mean_excess_cols=round(sum(excess) / max(1, len(excess)), 3),
              frac_samples_at_or_below_label=round(sum(e <= 0 for e in excess) / max(1, len(excess)), 3))
    return rows, st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "select"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plans", type=Path, nargs="+", default=[Path("out/synth5/plans.jsonl")])
    ap.add_argument("--model", default=None)
    ap.add_argument("--cells", type=int, default=5000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--batch", type=int, default=384)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "sample":
        from cellgym.llm.infer import LM, run_step
        cells = load_cells(a.plans)
        ids = sorted(cells)
        ids = sorted(random.Random(a.seed).sample(ids, min(a.cells, len(ids))))
        (a.out / "cells.json").write_text(json.dumps(
            {"plans": [str(p) for p in a.plans], "model": a.model, "k": a.k, "temperature": a.temperature,
             "seed": a.seed, "cells": {i: cells[i] for i in ids}}) + "\n")
        run_step(LM(a.model, a.batch), {"distill": ({i: cells[i]["netlist"] for i in ids}, None)},
                 a.k, a.temperature, a.out / "samples.jsonl")
        return 0
    meta = json.loads((a.out / "cells.json").read_text())
    samples = [json.loads(l) for l in (a.out / "samples.jsonl").read_text().splitlines()]
    rows, st = select(meta["cells"], samples, a.keep)
    with open(a.out / "plans.jsonl", "w", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    st.update(samples=len(samples), status=dict(Counter(s["status"] for s in samples)), keep=a.keep,
              by_origin=dict(Counter(meta["cells"][r["id"]]["origin"] for r in rows)))
    (a.out / "manifest.json").write_text(json.dumps(
        {k: meta[k] for k in ("plans", "model", "k", "temperature", "seed")} | st, indent=1) + "\n")
    print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
