"""Probe sets far outside training, with no Euler label (its search is too slow at these sizes).

    python -m cellgym.llm.probe --out out/probe --spec 8:14 10:16 --count 20
    python -m cellgym.llm.probe --out out/probe --spec 12:20:5 --count 20
    python -m cellgym.llm.probe --out out/probe --stages 2 --per-side 6:12 --outputs 1:1 --count 30
    python -m cellgym.llm.probe --out out/probe --stages 3 --per-side 8:16 --outputs 1:2 --count 30

Each spec inputs:per_side[:stack] gives a devset-style dir out/probe/i<inputs>d<per_side> (cells/,
index.json with part "all" and w_euler_nm null, manifest.json): random irredundant single-stage
networks with both stacks <= stack (default --max-stack, 4). A series-parallel network has at most
stack^2 devices per side, so a spec that cannot exist at the limit (20 or 24 per side at 4) is
built at the smallest stack that can hold it, and the manifest says so.

--stages N gives out/probe/stage<N>: N single-stage cells (each <= 5 inputs, stacks <= 4) wired
into one cell; see cellgym.llm.probegen. Every cell is checked by switch-level simulation.

--legacy uses the old rejection sampler (cellgym.synth.devset.sample_networks; slow above 8:14).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cellgym.llm import probegen as G
from cellgym.netlists.transistors import build_cell
from cellgym.netlists.writers import PORTS_TAIL
from cellgym.synth.generate import SIZING


def single_json(net) -> dict:
    """cellgym.netlists.writers.cell_json without its per-row truth-table loop (16 inputs)."""
    from cellgym.netlists.enumerate import depth, tree_str
    cell = build_cell(net, SIZING)
    rows = 1 << net.n_inputs
    return {"id": cell.id, "inputs": cell.inputs, "ports": cell.inputs + PORTS_TAIL,
            "function": format(G.expected_single(net)["OUT"], "x").zfill(max(1, rows // 4)),
            "function_note": "hex truth table of OUT; bit r = row r, input A = LSB of r",
            "canonical_tree": tree_str(net.tree), "pdn_depth": depth(net.tree),
            "pun_depth": depth(net.tree, dual=True), "read_once": net.read_once,
            "n_devices": len(cell.devices), "max_w_n": cell.max_w_n, "max_w_p": cell.max_w_p,
            "fits": True, "devices": [d.as_dict() for d in cell.devices], "nets": cell.nets}


def write_set(d: Path, netlists: list[dict], extra: dict) -> dict:
    (d / "cells").mkdir(parents=True, exist_ok=True)
    for old in (d / "cells").glob("*.json"):
        old.unlink()
    index = {}
    for nl in netlists:
        (d / "cells" / f"{nl['id']}.json").write_text(json.dumps(nl, indent=1) + "\n", newline="\n")
        index[nl["id"]] = {"part": "all", "w_euler_nm": None, "n_devices": nl["n_devices"]}
    (d / "index.json").write_text(json.dumps(index, indent=1) + "\n", newline="\n")
    st = [G.stats(nl) for nl in netlists]
    axes = {k: [min(s[k] for s in st), max(s[k] for s in st)] for k in st[0]} if st else {}
    man = {"cells": len(index), "axes_min_max": axes} | extra
    (d / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", newline="\n")
    return man


def build_single(out: Path, spec: str, count: int, max_stack: int, legacy: bool = False) -> dict:
    t0 = time.time()
    parts = [int(x) for x in spec.split(":")]
    n_in, per = parts[0], parts[1]
    asked = parts[2] if len(parts) > 2 else max_stack
    stack = max(asked, G.min_stack(per))
    if legacy:
        from cellgym.synth.devset import sample_networks
        nets = sample_networks(n_in, per, stack, count, seed=4242 + per)
    else:
        nets = G.sample_networks_fast(n_in, per, stack, count, seed=4242 + per)
    netlists = []
    for net in nets:
        nl = single_json(net)
        bad = G.check_netlist(nl, G.expected_single(net))
        if bad:
            raise RuntimeError(f"{nl['id']}: switch-level check failed: {bad[0]}")
        netlists.append(nl)
    note = (f"stack limit raised from {asked} to {stack}: at most {asked * asked} devices per side fit in {asked}"
            if stack != asked else "")
    return write_set(out / f"i{n_in}d{per}", netlists, {
        "kind": "single-stage", "spec": spec, "max_stack": stack, "note": note,
        "sampler": "legacy" if legacy else "probegen.sample_networks_fast", "seed": 4242 + per,
        "self_check": "switch-level, " + ("all rows" if n_in <= G.EXHAUSTIVE_INPUTS
                                           else f"{G.SAMPLED_ROWS} random rows + all-0 + all-1"),
        "seconds": round(time.time() - t0, 1)})


def build_multi(out: Path, stages: int, count: int, per_side: tuple[int, int], outputs: tuple[int, int],
                max_primary: int = 5, name: str | None = None) -> dict:
    t0 = time.time()
    netlists = G.sample_multistage(stages, count, per_side, outputs, seed=7000 + stages, max_primary=max_primary)
    for nl in netlists:
        bad = G.check_netlist(nl, {o: int(h, 16) for o, h in nl["function"].items()})
        if bad:
            raise RuntimeError(f"{nl['id']}: switch-level check failed: {bad[0]}")
    return write_set(out / (name or f"stage{stages}"), netlists, {
        "kind": "multi-stage", "stages": stages, "per_side": list(per_side), "outputs": list(outputs),
        "max_primary_inputs": max_primary, "seed": 7000 + stages,
        "stage_limits": "each stage <= 5 inputs, stacks <= 4, <= 7 devices per side",
        "self_check": "switch-level, all rows", "seconds": round(time.time() - t0, 1)})


def _pair(s: str) -> tuple[int, int]:
    a, _, b = s.partition(":")
    return int(a), int(b or a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("out/probe"))
    ap.add_argument("--spec", nargs="+", default=None, help="inputs:per_side[:stack] ...")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--max-stack", type=int, default=4)
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--stages", type=int, default=0, help="build a multi-stage set instead")
    ap.add_argument("--per-side", type=_pair, default=(6, 12), help="multi-stage: lo:hi total devices per side")
    ap.add_argument("--outputs", type=_pair, default=(1, 1), help="multi-stage: lo:hi outputs")
    ap.add_argument("--max-primary", type=int, default=5, help="multi-stage: most primary inputs")
    ap.add_argument("--name", default=None, help="multi-stage: set dir name (default stage<N>)")
    a = ap.parse_args(argv)
    if a.stages:
        man = build_multi(a.out, a.stages, a.count, a.per_side, a.outputs, a.max_primary, a.name)
        print(f"stage{a.stages}", json.dumps(man))
        return 0
    for sp in a.spec or ["8:14", "10:16", "12:20"]:
        man = build_single(a.out, sp, a.count, a.max_stack, a.legacy)
        print(sp, json.dumps(man))
    return 0


if __name__ == "__main__":
    sys.exit(main())
