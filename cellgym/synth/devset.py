"""6-input cells outside the benchmark, for measuring and training input-count growth.

    python -m cellgym.synth.devset build --out out/ood6 [--jobs 6]
    python -m cellgym.synth.devset eval --ckpt CKPT --set out/ood6 [--part dev] [--k 8]
        [--temperature 0.8] [--rename] [--tag NAME]

The benchmark's 6-input cells (all in test_ood) are the exhaustive core region:
stack <= 3, devices <= inputs + 2. This set takes 6-input topologies with a stack
of 4 on one side (still <= 8 devices), which the core never has, and drops any
whose OUT function, under any input renaming, equals the function of ANY core
cell. So no cell here is a benchmark cell or computes a benchmark cell's function.

Each kept topology becomes one netlist (inputs A-F), labeled with one Euler plan
(the width reference for the reward). Cells are split by a hash of the cell id:
`dev` (about 200 cells, for measuring and choosing checkpoints) and `pool` (the
rest, for training on later). Writes <out>/cells/*.json, <out>/plans/*.plan,
<out>/index.json (id -> part, w_euler_nm, sites, stack depths) and manifest.json.

`eval` samples a checkpoint on one part and scores it with the benchmark's own
pipeline (cellgym.synth.eval.score_samples). --rename evaluates the 5-input
benchmark val cells with their inputs renamed onto B-F instead (a check that
the model handles any input letters; no 6-input cell involved).
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

from cellgym.netlists.enumerate import Params, canonical_fn, depth, enumerate_networks
from cellgym.netlists.transistors import Sizing, build_cell
from cellgym.netlists.writers import cell_json, out_function_hex

SIZING = Sizing()
PARAMS = Params(max_inputs=6, max_stack=4, extra_devices=2, max_devices=8)
DEV_MOD = 6          # about 1 in 6 cells -> dev


def part_of(cid: str) -> str:
    h = int(hashlib.sha256(f"cellgym-ood6:{cid}".encode()).hexdigest(), 16)
    return "dev" if h % DEV_MOD == 0 else "pool"


def core_functions() -> set[tuple[int, int]]:
    """(inputs, canonical OUT function) of every core cell, whatever its split."""
    out = set()
    for net in enumerate_networks(Params()):
        cell = build_cell(net, SIZING)
        out.add((net.n_inputs, canonical_fn(int(out_function_hex(cell), 16), net.n_inputs)))
    return out


def _label(nl: dict):
    from cellgym.synth.generate import euler_plans
    try:
        plans, fails = euler_plans(nl, 1)
    except Exception as e:  # noqa: BLE001 - any failure just leaves the cell out
        return nl["id"], None, [f"{type(e).__name__}"]
    return nl["id"], (plans[0] if plans else None), fails


def build(out: Path, jobs: int) -> dict:
    t0 = time.time()
    core_trees = {n.tree for n in enumerate_networks(Params())}
    bad = core_functions()
    stats = Counter()
    netlists = []
    for net in enumerate_networks(PARAMS):
        if net.n_inputs != 6:
            continue
        if net.tree in core_trees:
            stats["core"] += 1
            continue
        cell = build_cell(net, SIZING)
        if (6, canonical_fn(int(out_function_hex(cell), 16), 6)) in bad:
            stats["dropped_core_function"] += 1
            continue
        netlists.append(cell_json(cell, SIZING))
    stats["candidates"] = len(netlists)
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            results = list(ex.map(_label, netlists, chunksize=4))
    else:
        results = [_label(nl) for nl in netlists]
    by_id = {nl["id"]: nl for nl in netlists}
    (out / "cells").mkdir(parents=True, exist_ok=True)
    (out / "plans").mkdir(parents=True, exist_ok=True)
    index, fails = {}, Counter()
    for cid, plan, reasons in sorted(results, key=lambda r: r[0]):
        fails.update(reasons)
        if plan is None:
            stats["unlabeled"] += 1
            continue
        nl = by_id[cid]
        if len(nl["inputs"]) != 6:
            raise RuntimeError(f"unexpected cell {cid}")
        (out / "cells" / f"{cid}.json").write_text(json.dumps(nl, indent=1) + "\n", newline="\n")
        (out / "plans" / f"{cid}.plan").write_text(plan["plan"], newline="\n")
        index[cid] = {"part": part_of(cid), "w_euler_nm": round(plan["width_um"] * 1000),
                      "sites": plan["sites"], "breaks": plan["breaks"],
                      "n_devices": nl["n_devices"], "pdn_depth": nl["pdn_depth"],
                      "pun_depth": nl["pun_depth"]}
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", newline="\n")
    manifest = {"params": PARAMS.__dict__, "stats": dict(stats), "route_failures": dict(fails),
                "cells": len(index), "by_part": dict(Counter(v["part"] for v in index.values())),
                "by_devices": dict(sorted(Counter(v["n_devices"] for v in index.values()).items())),
                "leak_check": "no core tree, no core function (any split), all 6 inputs",
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", newline="\n")
    return manifest


def load(set_dir: Path, part: str) -> tuple[dict[str, dict], dict[str, int]]:
    index = json.loads((set_dir / "index.json").read_text())
    ids = sorted(c for c, v in index.items() if part in ("all", v["part"]))
    cells = {c: json.loads((set_dir / "cells" / f"{c}.json").read_text()) for c in ids}
    return cells, {c: index[c]["w_euler_nm"] for c in ids}


def renamed_val() -> tuple[dict[str, dict], dict[str, int]]:
    """Benchmark val cells with inputs renamed (order kept) so the last input is F."""
    from cellgym.bench import reference, splits
    from cellgym.bench.common import EULER_WIDTHS_FILE, SPLITS_FILE, load_cells
    from cellgym.synth import tokens as T
    from cellgym.synth.train import relabel_netlist
    cells = load_cells(Path("out/cells"))
    sp = splits.split_of(splits.load(SPLITS_FILE))
    w = {c: v["width_nm"] for c, v in reference.load(EULER_WIDTHS_FILE)["cells"].items()}
    out = {}
    for c in sorted(x for x in sp if sp[x] == "val"):
        n = len(cells[c]["inputs"])
        out[c] = relabel_netlist(cells[c], {T.INPUTS[i]: T.INPUTS[6 - n + i] for i in range(n)})
    return out, {c: w[c] for c in out}


def evaluate(ckpt: Path, cells: dict, w: dict, k: int, temperature: float, seed: int = 0,
             device: str = "cuda") -> dict:
    import torch
    from cellgym.synth import model as M
    from cellgym.synth import tokens as T
    from cellgym.synth.constrain import sample_constrained
    from cellgym.synth.eval import MAX_NEW, score_samples
    torch.manual_seed(seed)
    model = M.load(ckpt, device)
    samples = {c: sample_constrained(model, T.ids(T.encode_netlist(nl, model.cfg.prompt)), nl, n=k,
                                     temperature=temperature, max_new=MAX_NEW)
               for c, nl in cells.items()}
    return score_samples(cells, samples, w_euler=w, split_of={c: "set" for c in cells})


def summary_line(res: dict) -> str:
    a = res["summary"]["all"]
    stages = Counter(s["stage"] for c in res["cells"] for s in c["samples"])
    f = lambda x: "-" if x is None else f"{x:.3f}"
    return (f"cells {a['cells']} per-sample {f(a['sample_pass_rate'])} pass@1 {f(a['pass_at_1'])} "
            f"pass@k {f(a['pass_at_k'])} W/W_euler {f(a['w_over_euler_passing'])} stages {dict(stages)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=Path("out/ood6"))
    b.add_argument("--jobs", type=int, default=6)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument("--set", type=Path, default=Path("out/ood6"))
    e.add_argument("--part", default="dev", choices=["dev", "pool", "all"])
    e.add_argument("--rename", action="store_true", help="renamed benchmark val cells instead")
    e.add_argument("--k", type=int, default=8)
    e.add_argument("--temperature", type=float, default=0.8)
    e.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        m = build(a.out, a.jobs)
        print(json.dumps(m, indent=1))
        return 0
    cells, w = renamed_val() if a.rename else load(a.set, a.part)
    res = evaluate(a.ckpt, cells, w, a.k, a.temperature)
    tag = a.tag or ("renamed_val" if a.rename else f"ood6_{a.part}")
    res |= {"ckpt": str(a.ckpt), "k": a.k, "temperature": a.temperature}
    path = a.ckpt.parent / f"eval_{tag}_k{a.k}.json"
    path.write_text(json.dumps(res, indent=1) + "\n", newline="\n")
    print(summary_line(res), "->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
