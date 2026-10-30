"""Benchmark splits, round 0 (core region) -> bench/splits_v0.json (committed).

    python -m cellgym.bench.splits [--cells out/cells] [--out bench/splits_v0.json] [--check]

Test-OOD: every 6-input cell, plus named anchors found by logic function
(truth table under input renaming): AOI21, AOI22, OAI21, OAI211, NAND3, NOR3
and the inverted majority NOT MAJ3 (a full adder's carry). If several core
cells implement an anchor's function, all go to Test-OOD (no leak of the
function into training); the anchor names the read-once one (else the fewest
devices, then id).

The remaining <=5-input cells split by cell into train / val / Test-ID with
targets 123 / 15 / 25 (train reduced by the number of anchor cells pulled
out), stratified by (input count, read_once): per stratum, cells are ordered
by sha256(seed + id); Test-ID and val quotas are largest-remainder shares of
their targets, taken from the front of that order; the rest is train.

Per cell: id, split, inputs, devices, read_once, sha256 of the netlist JSON
(canonical: sorted keys, no whitespace). The file carries its own sha256
(same canonical serialization, without the "sha256" key). --check regenerates
and compares with the committed file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .common import SPLITS, SPLITS_FILE, dump_rows, hash_ok, load_cells, match, sha256, with_hash

VERSION = "v0"
SEED = "cellgym-splits-v0"
TARGETS = {"train": 123, "val": 15, "test_id": 25}
OOD_INPUTS = 6

ANCHORS = {   # name -> (inputs, function of the argument list)
    "AOI21": (3, lambda x: not ((x[0] and x[1]) or x[2])),
    "AOI22": (4, lambda x: not ((x[0] and x[1]) or (x[2] and x[3]))),
    "OAI21": (3, lambda x: not ((x[0] or x[1]) and x[2])),
    "OAI211": (4, lambda x: not ((x[0] or x[1]) and x[2] and x[3])),
    "NAND3": (3, lambda x: not (x[0] and x[1] and x[2])),
    "NOR3": (3, lambda x: not (x[0] or x[1] or x[2])),
    "NOT_MAJ3": (3, lambda x: not ((x[0] and x[1]) or (x[1] and x[2]) or (x[0] and x[2]))),
}


def find_anchors(cells: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for name, (k, f) in ANCHORS.items():
        hits = match(f, k, cells)
        if not hits:
            raise ValueError(f"anchor {name} not found in the core set")
        ids = [cid for cid, _ in hits]
        best = min(ids, key=lambda c: (not cells[c]["read_once"], cells[c]["n_devices"], c))
        perm = dict(hits)[best]
        out[name] = {"id": best, "all_ids": ids,
                     "mapping": {f"x{j}": cells[best]["inputs"][perm[j]] for j in range(k)},
                     "canonical_tree": cells[best]["canonical_tree"]}
    return out


def _largest_remainder(target: int, sizes: dict, caps: dict) -> dict:
    total = sum(sizes.values())
    ideal = {g: target * n / total for g, n in sizes.items()}
    q = {g: min(int(ideal[g]), caps[g]) for g in sizes}
    order = sorted(sizes, key=lambda g: (-(ideal[g] - int(ideal[g])), g))
    while sum(q.values()) < target:
        progressed = False
        for g in order:
            if sum(q.values()) < target and q[g] < caps[g]:
                q[g] += 1
                progressed = True
        if not progressed:
            raise ValueError("split targets exceed the cells available")
    return q


def make_splits(cells: dict[str, dict]) -> dict:
    anchors = find_anchors(cells)
    anchor_ids = {cid for a in anchors.values() for cid in a["all_ids"]}
    split: dict[str, str] = {}
    for cid, nl in cells.items():
        if len(nl["inputs"]) >= OOD_INPUTS or cid in anchor_ids:
            split[cid] = "test_ood"
    rest = sorted(c for c in cells if c not in split)
    targets = dict(TARGETS)
    targets["train"] -= len(anchor_ids)
    if sum(targets.values()) != len(rest):
        raise ValueError(f"{len(rest)} in-distribution cells, targets sum to {sum(targets.values())}")
    strata: dict[tuple, list[str]] = defaultdict(list)
    for cid in rest:
        strata[(len(cells[cid]["inputs"]), bool(cells[cid]["read_once"]))].append(cid)
    for g in strata:
        strata[g].sort(key=lambda c: hashlib.sha256(f"{SEED}:{c}".encode()).hexdigest())
    sizes = {g: len(v) for g, v in strata.items()}
    q_test = _largest_remainder(targets["test_id"], sizes, sizes)
    q_val = _largest_remainder(targets["val"], sizes, {g: sizes[g] - q_test[g] for g in sizes})
    for g, ids in strata.items():
        for i, cid in enumerate(ids):
            split[cid] = "test_id" if i < q_test[g] else "val" if i < q_test[g] + q_val[g] else "train"
    rows = [{"id": cid, "split": split[cid], "inputs": len(cells[cid]["inputs"]),
             "devices": cells[cid]["n_devices"], "read_once": bool(cells[cid]["read_once"]),
             "netlist_sha256": sha256(cells[cid])} for cid in sorted(cells)]
    counts = Counter(r["split"] for r in rows)
    doc = {
        "version": VERSION,
        "region": "core (M1 defaults: <=6 inputs, stack <=3, devices <= inputs+2, max 8)",
        "seed": SEED,
        "targets": targets | {"test_ood": f"all {OOD_INPUTS}-input cells + anchors"},
        "counts": {s: counts[s] for s in SPLITS},
        "anchors": anchors,
        "strata": {f"{k}i_{'ro' if ro else 'rep'}": {s: sum(1 for c in ids if split[c] == s) for s in SPLITS[:3]}
                   for (k, ro), ids in sorted(strata.items())},
        "cells": rows,
    }
    return with_hash(doc)


def load(path: Path = SPLITS_FILE) -> dict:
    doc = json.loads(Path(path).read_text())
    if not hash_ok(doc):
        raise ValueError(f"{path}: sha256 does not match its contents")
    return doc


def split_of(doc: dict) -> dict[str, str]:
    return {r["id"]: r["split"] for r in doc["cells"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="out/cells")
    ap.add_argument("--out", default=str(SPLITS_FILE))
    ap.add_argument("--check", action="store_true", help="compare with the existing file instead of writing")
    a = ap.parse_args(argv)
    doc = make_splits(load_cells(Path(a.cells)))
    text = dump_rows(doc, "cells")
    if a.check:
        same = Path(a.out).read_text() == text
        print("splits reproducible" if same else f"MISMATCH with {a.out}")
        return 0 if same else 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, newline="\n")
    print(f"{a.out}: {doc['counts']}  sha256 {doc['sha256'][:12]}")
    for name, an in doc["anchors"].items():
        extra = f" (+{len(an['all_ids']) - 1} more)" if len(an["all_ids"]) > 1 else ""
        print(f"  {name}: {an['id']} {an['canonical_tree']}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
