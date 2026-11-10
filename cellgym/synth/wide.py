"""Wide-cell supplement for the SL corpus: Euler-labeled plans with 10-12 devices per side.

    python -m cellgym.synth.wide --out out/wide --counts 10:600,11:320,12:120 --inputs 4 5 --jobs 15
    python -m cellgym.synth.wide merge out/synth5/plans.jsonl out/wide/plans.jsonl --dup 2 --out out/synth8

The synth5 corpus stops at 9 devices per side, so no plan ever uses a column index above 19
and models stop every 10+-per-side plan at col ~19. These cells add longer rows (cols up to
~25) at <= 5 inputs, stack <= 4, so device count per side is the only new axis.

Topologies come from devset.sample_networks (random; enumeration is too slow at this size)
with a seed that differs from the ooddev sets', and are dropped when the tree or the OUT
function (under input renaming) equals any ooddev10/11/12 cell's (all parts, so their dev
AND pool stay unseen) or any non-train benchmark cell's. Records use the generate.py schema
(origin "wide"), so train.py reads the merged file unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cellgym.bench.splits import load as load_splits, split_of
from cellgym.netlists.enumerate import canonical_fn, parse_tree, tree_str
from cellgym.netlists.transistors import build_cell
from cellgym.netlists.writers import cell_json, out_function_hex
from cellgym.synth.devset import sample_networks
from cellgym.synth.generate import SIZING, core_cells, euler_plans, forbidden_functions

HOLDOUT_SETS = ("out/ooddev10", "out/ooddev11", "out/ooddev12")
SEED = 7919          # devset build --sample uses seed 0


def _fn(nl: dict) -> tuple[int, int]:
    n = len(nl["inputs"])
    return n, canonical_fn(int(nl["function"], 16), n)


def holdout() -> tuple[set, set]:
    trees, fns = set(), set()
    for d in HOLDOUT_SETS:
        for f in sorted((Path(d) / "cells").glob("*.json")):
            nl = json.loads(f.read_text())
            trees.add(tree_str(parse_tree(nl["canonical_tree"])))
            fns.add(_fn(nl))
    if not fns:
        raise FileNotFoundError(f"no holdout cells under {HOLDOUT_SETS}")
    return trees, fns


def _label(job):
    nl, placements = job
    try:
        plans, fails = euler_plans(nl, placements)
    except Exception as e:  # noqa: BLE001 - any failure leaves the cell out
        return nl["id"], [], [type(e).__name__]
    return nl["id"], plans, fails


def build(out: Path, counts: dict[int, int], inputs: list[int], max_stack: int, placements: int,
          jobs: int) -> dict:
    t0 = time.time()
    bad = forbidden_functions(core_cells(), split_of(load_splits()))
    h_trees, h_fns = holdout()
    stats, netlists = Counter(), []
    for per_side, count in sorted(counts.items()):
        for k, n_in in enumerate(inputs):
            nets = sample_networks(n_in, per_side, max_stack, count // len(inputs), seed=SEED + 100 * per_side + k)
            for net in nets:
                cell = build_cell(net, SIZING)
                nl = cell_json(cell, SIZING)
                f = (n_in, canonical_fn(int(out_function_hex(cell), 16), n_in))
                if tree_str(net.tree) in h_trees or f in h_fns:
                    stats["dropped_holdout"] += 1
                    continue
                if f in bad:
                    stats["dropped_benchmark_function"] += 1
                    continue
                stats[f"candidates_{per_side}"] += 1
                netlists.append(nl)
    by_id = {nl["id"]: nl for nl in netlists}
    work = [(nl, placements) for nl in netlists]
    with ProcessPoolExecutor(jobs) as ex:
        results = list(ex.map(_label, work, chunksize=1))
    records, fails = [], Counter()
    for cid, plans, reasons in sorted(results, key=lambda r: r[0]):
        fails.update(reasons)
        if not plans:
            continue
        nl = by_id[cid]
        stats[f"labeled_{nl['n_devices'] // 2}"] += 1
        for i, p in enumerate(plans):
            records.append({"id": cid, "plan_index": i, "topology_id": cid, "origin": "wide",
                            "n_inputs": len(nl["inputs"]), "n_devices": nl["n_devices"],
                            "read_once": nl["read_once"], "function": nl["function"],
                            "tree": nl["canonical_tree"], "netlist": nl, **p})
    if any(_fn(r["netlist"]) in h_fns or _fn(r["netlist"]) in bad for r in records):
        raise RuntimeError("holdout/benchmark leak in wide corpus")
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records)
    (out / "plans.jsonl").write_text(body, newline="\n")
    manifest = {"counts": counts, "inputs": inputs, "max_stack": max_stack, "placements": placements,
                "seed": SEED, "records": len(records), "cells": len({r["id"] for r in records}),
                "stats": dict(stats), "route_failures": dict(fails),
                "holdout": list(HOLDOUT_SETS), "leak_check": "pass",
                "plans_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", newline="\n")
    return manifest


def merge(base: Path, extra: Path, dup: int, out: Path) -> dict:
    """base records + extra records repeated dup times (same topology_id, so the val split
    keeps every copy on one side)."""
    a = base.read_text().splitlines()
    b = extra.read_text().splitlines()
    out.mkdir(parents=True, exist_ok=True)
    (out / "plans.jsonl").write_text("\n".join(a + b * dup) + "\n", newline="\n")
    m = {"base": str(base), "extra": str(extra), "dup": dup, "base_records": len(a),
         "extra_records": len(b), "records": len(a) + dup * len(b)}
    (out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n", newline="\n")
    return m


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "merge":
        ap = argparse.ArgumentParser()
        ap.add_argument("base", type=Path)
        ap.add_argument("extra", type=Path)
        ap.add_argument("--dup", type=int, default=1)
        ap.add_argument("--out", type=Path, required=True)
        a = ap.parse_args(argv[1:])
        print(json.dumps(merge(a.base, a.extra, a.dup, a.out), indent=1))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("out/wide"))
    ap.add_argument("--counts", default="10:600,11:320,12:120", help="per-side:sampled topologies")
    ap.add_argument("--inputs", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--max-stack", type=int, default=4)
    ap.add_argument("--placements", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    counts = {int(k): int(v) for k, v in (x.split(":") for x in a.counts.split(","))}
    print(json.dumps(build(a.out, counts, a.inputs, a.max_stack, a.placements, a.jobs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
