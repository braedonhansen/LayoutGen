"""Synthetic training corpus: Euler-labeled plans for cells outside the benchmark.

    python -m cellgym.synth.generate --out out/synth [--max-inputs 5] [--extra-devices 4]
        [--renames 3] [--placements 4] [--jobs 4] [--cache out/synth/topologies.json]

The core region (<= 6 inputs, stack <= 3, devices <= inputs + 2) is exhaustive,
and its <= 5-input part is only 163 cells, so new variety must come from
outside it. Two sources:

  1. topologies with more repeated-input devices (devices <= inputs + extra,
     default extra 4, max 9), same <= 5 inputs and stack <= 3; the extra
     devices are not a benchmark axis;
  2. input renamings of every topology (the same cell with the inputs wired in
     a different order), which changes the netlist the model reads and the
     plan's column order.

Plus the core cells that bench/splits_v0.json puts in train (origin
"core_train"). Labels are the Euler baseline: every optimal placement (fewest
diffusion breaks) is routed on its own and each distinct plan is one record.

Leakage control (checked on every run, stored in the manifest): a synthetic topology is
dropped when its OUT function, under any input renaming, equals the function
of a val, test_id or test_ood core cell. That covers the anchors (AOI21 ...,
inverted majority) and every 6-input function. core_train cells are kept as the
benchmark defines them, even where one shares a function with a test_id cell
(the benchmark splits by cell); their count is in the manifest. Inputs are capped at 5, so no
6-input netlist appears. All records of one topology carry the same
topology_id; split by topology_id, never by record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cellgym.baselines.common import BaselineError, route_best
from cellgym.baselines.euler import MAX_PLACEMENTS, _columns, _Placer
from cellgym.bench.common import sha256
from cellgym.bench.splits import load as load_splits, split_of
from cellgym.netlists.enumerate import (
    Network, Params, canonical_fn, enumerate_networks, num_inputs, parse_tree, relabel, tree_str,
)
from cellgym.netlists.transistors import Sizing, build_cell
from cellgym.netlists.writers import cell_json, out_function_hex

VERSION = "synth-v0"
SEED = "cellgym-synth-v0"
SIZING = Sizing()


# -- benchmark side ------------------------------------------------------------------

def core_cells() -> dict[str, tuple[Network, str]]:
    """id -> (network, OUT function hex) for the exhaustive core region."""
    out = {}
    for net in enumerate_networks(Params()):
        cell = build_cell(net, SIZING)
        out[cell.id] = (net, out_function_hex(cell))
    return out


def forbidden_functions(core: dict, splits: dict[str, str]) -> set[tuple[int, int]]:
    """(inputs, canonical OUT function) of every core cell that is not train."""
    bad = set()
    for cid, (net, fn) in core.items():
        if splits[cid] != "train":
            bad.add((net.n_inputs, canonical_fn(int(fn, 16), net.n_inputs)))
    return bad


# -- topology set --------------------------------------------------------------------

def topologies(params: Params, cache: Path | None) -> list[Network]:
    if cache and cache.exists():
        doc = json.loads(cache.read_text())
        if doc["params"] == params.__dict__:
            return [Network(t := parse_tree(s), num_inputs(t)) for s in doc["trees"]]
    nets = enumerate_networks(params)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"params": params.__dict__,
                                     "trees": [tree_str(n.tree) for n in nets]}) + "\n")
    return nets


def renamed(net: Network, base_id: str, renames: int) -> list[Network]:
    """The identity plus up to renames-1 distinct random input renamings."""
    out, seen = [net], {tree_str(net.tree)}
    k = net.n_inputs
    for v in range(1, renames):
        perm = list(range(k))
        random.Random(f"{SEED}:{base_id}:{v}").shuffle(perm)
        t = relabel(net.tree, dict(enumerate(perm)))
        if tree_str(t) not in seen:
            seen.add(tree_str(t))
            out.append(Network(t, k))
    return out


# -- labeling ------------------------------------------------------------------------

def euler_plans(netlist: dict, limit: int) -> tuple[list[dict], list[str]]:
    """Every distinct plan from up to `limit` optimal placements; (plans, failure reasons)."""
    placer = _Placer(netlist)
    breaks, row_breaks = placer.cost(placer.start)
    plans, fails, seen = [], [], set()
    for seq in placer.sequences(limit):
        try:
            r = route_best([_columns(seq)], netlist)
        except BaselineError as e:
            fails.append(e.reason)
            continue
        if r.text in seen:
            continue
        seen.add(r.text)
        plans.append({"plan": r.text, "sites": r.sites, "width_um": r.width_um,
                      "breaks": breaks, "row_breaks": row_breaks, "near_misses": r.near_misses})
    return plans, fails


def label(job: tuple[dict, int]) -> tuple[str, list[dict], list[str]]:
    netlist, limit = job
    try:
        plans, fails = euler_plans(netlist, limit)
    except BaselineError as e:
        return netlist["id"], [], [e.reason]
    return netlist["id"], plans, fails


# -- driver --------------------------------------------------------------------------

def build_jobs(params: Params, renames: int, cache: Path | None, with_core_train: bool):
    """(jobs, stats). Each job is one netlist; meta is keyed by cell id."""
    core = core_cells()
    splits = split_of(load_splits())
    bad = forbidden_functions(core, splits)
    core_trees = {net.tree for net, _ in core.values()}
    stats = Counter()
    meta: dict[str, dict] = {}
    netlists: list[dict] = []
    for net in topologies(params, cache):
        cell = build_cell(net, SIZING)
        base_id = cell.id
        if net.tree in core_trees:
            if not with_core_train or splits.get(base_id) != "train":
                stats["core_not_train"] += 1
                continue
            origin = "core_train"
        else:
            origin = "synthetic"
            if (net.n_inputs, canonical_fn(int(out_function_hex(cell), 16), net.n_inputs)) in bad:
                stats["dropped_benchmark_function"] += 1
                continue
        stats[f"topologies_{origin}"] += 1
        for v in renamed(net, base_id, renames):
            c = build_cell(v, SIZING)
            nl = cell_json(c, SIZING)
            meta[c.id] = {"topology_id": base_id, "origin": origin}
            netlists.append(nl)
    return netlists, meta, bad, stats


def generate(out: Path, params: Params, renames: int, placements: int, jobs: int,
             cache: Path | None, with_core_train: bool = True) -> dict:
    t0 = time.time()
    netlists, meta, bad, stats = build_jobs(params, renames, cache, with_core_train)
    work = [(nl, placements) for nl in netlists]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            results = list(ex.map(label, work, chunksize=8))
    else:
        results = [label(w) for w in work]
    by_id = {nl["id"]: nl for nl in netlists}
    fails: Counter = Counter()
    unlabeled = []
    records = []
    for cid, plans, reasons in sorted(results):
        fails.update(reasons)
        if not plans:
            unlabeled.append(cid)
            continue
        nl = by_id[cid]
        for i, p in enumerate(plans):
            records.append({
                "id": cid, "plan_index": i, **meta[cid],
                "n_inputs": len(nl["inputs"]), "n_devices": nl["n_devices"],
                "read_once": nl["read_once"], "function": nl["function"],
                "tree": nl["canonical_tree"], "netlist": nl, **p,
            })
    shares = {r["id"]: (r["n_inputs"], canonical_fn(int(r["function"], 16), r["n_inputs"])) in bad
              for r in records}
    # core_train cells are train in the benchmark itself (it splits by cell, so a
    # train cell may share a function with a test_id cell); only synthetic ones are checked
    leaks = [r["id"] for r in records if r["origin"] == "synthetic" and shares[r["id"]]]
    if leaks or any(r["n_inputs"] > 5 for r in records):
        raise RuntimeError(f"benchmark leak in synthetic corpus: {leaks[:5]}")
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records)
    (out / "plans.jsonl").write_text(body, newline="\n")
    topo = {r["topology_id"] for r in records}
    manifest = {
        "version": VERSION, "seed": SEED, "params": params.__dict__, "renames": renames,
        "placements": placements, "splits_sha256": load_splits()["sha256"],
        "records": len(records), "cells": len({r["id"] for r in records}), "topologies": len(topo),
        "by_origin": dict(Counter(r["origin"] for r in records)),
        "by_inputs": dict(sorted(Counter(r["n_inputs"] for r in records).items())),
        "topology_stats": dict(stats), "forbidden_functions": len(bad),
        "unlabeled_cells": unlabeled, "route_failures": dict(fails),
        "leak_check": "pass",
        "core_train_cells_sharing_benchmark_function": sum(
            1 for c, v in shares.items() if v and meta[c]["origin"] == "core_train"), "plans_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", newline="\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("out/synth"))
    ap.add_argument("--max-inputs", type=int, default=5)
    ap.add_argument("--max-stack", type=int, default=3)
    ap.add_argument("--extra-devices", type=int, default=4)
    ap.add_argument("--max-devices", type=int, default=9)
    ap.add_argument("--renames", type=int, default=3, help="input renamings per topology (incl. identity)")
    ap.add_argument("--placements", type=int, default=MAX_PLACEMENTS, help="optimal placements routed per cell")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cache", type=Path, default=Path("out/synth/topologies.json"))
    ap.add_argument("--no-core-train", action="store_true", help="synthetic topologies only")
    a = ap.parse_args(argv)
    if a.max_inputs > 5:
        ap.error("--max-inputs > 5 would put 6-input benchmark cells in the corpus")
    m = generate(a.out, Params(a.max_inputs, a.max_stack, a.extra_devices, a.max_devices),
                 a.renames, a.placements, a.jobs, a.cache, not a.no_core_train)
    print(f"{m['records']} plans, {m['cells']} cells, {m['topologies']} topologies -> {a.out} "
          f"({m['seconds']}s)")
    print(f"  origin {m['by_origin']}  inputs {m['by_inputs']}")
    print(f"  dropped (benchmark function) {m['topology_stats'].get('dropped_benchmark_function', 0)}  "
          f"unlabeled cells {len(m['unlabeled_cells'])}  route failures {m['route_failures']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
