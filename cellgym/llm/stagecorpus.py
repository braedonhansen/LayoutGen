"""A training corpus of multi-stage cells with Euler labels (composition as a trained axis).

    python -m cellgym.llm.stagecorpus --out out/synth2s --stages 2 --count 2500 --per-side 3:9 --seed 9100 \
        --exclude out/probe/stage2 out/probe/stage3 [--placements 4] [--jobs 12]

Cells come from cellgym.llm.probegen.sample_multistage (each stage <= 5 inputs, stacks <= 4, <= 7 devices
per side; <= 5 primary inputs), labeled like the single-stage corpus (cellgym.synth.generate.label: up to
--placements minimal-break Euler placements, routed at the fixed 3-track channel). Output is a plans.jsonl
that cellgym.llm.build reads next to out/synth5/plans.jsonl. Every cell is its own topology_id (the
train/val split is by cell). Cells whose id is in an --exclude set (the stage probes) are dropped, so the
probes stay never-seen cells; the id is a hash of the stage trees and wiring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cellgym.llm import probegen as G
from cellgym.synth.generate import SIZING, label


def _pair(s: str) -> tuple[int, int]:
    a, b = s.split(":")
    return int(a), int(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--count", type=int, default=2500)
    ap.add_argument("--per-side", type=_pair, default=(3, 9))
    ap.add_argument("--outputs", type=_pair, default=(1, 1))
    ap.add_argument("--seed", type=int, default=9100)
    ap.add_argument("--placements", type=int, default=4)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="devset-style dirs whose cell ids are dropped")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args(argv)
    t0 = time.time()
    banned = {f.stem for d in a.exclude for f in (d / "cells").glob("*.json")}
    cells = G.sample_multistage(a.stages, a.count + len(banned), a.per_side, a.outputs, seed=a.seed, sizing=SIZING)
    for nl in cells:
        problems = G.check_netlist(nl, {o: int(f, 16) for o, f in nl["function"].items()})
        if problems:
            raise RuntimeError(f"{nl['id']}: {problems[:2]}")
    dropped = [nl["id"] for nl in cells if nl["id"] in banned]
    cells = [nl for nl in cells if nl["id"] not in banned][:a.count]
    by_id = {nl["id"]: nl for nl in cells}
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(label, [(nl, a.placements) for nl in cells], chunksize=4))
    records, fails, unlabeled = [], Counter(), []
    for cid, plans, reasons in sorted(results):
        fails.update(reasons)
        if not plans:
            unlabeled.append(cid)
            continue
        nl = by_id[cid]
        for i, p in enumerate(plans):
            records.append({"id": cid, "plan_index": i, "topology_id": cid, "origin": f"stage{a.stages}",
                            "n_inputs": len(nl["inputs"]), "n_devices": nl["n_devices"], "netlist": nl, **p})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plans.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records), newline="\n")
    st = [G.stats(nl) for nl in cells if nl["id"] not in unlabeled]
    manifest = {"args": " ".join(sys.argv[1:] if argv is None else map(str, argv)), "records": len(records),
                "cells": len(st), "excluded_probe_cells": dropped, "unlabeled_cells": unlabeled,
                "route_failures": dict(fails),
                "axes_min_max": {k: [min(s[k] for s in st), max(s[k] for s in st)] for k in st[0]},
                "by_per_side": dict(sorted(Counter(s["per_side"] for s in st).items())),
                "mean_breaks": round(sum(r["breaks"] for r in records) / len(records), 3),
                "self_check": "switch-level, all rows", "seconds": round(time.time() - t0, 1)}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", newline="\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
