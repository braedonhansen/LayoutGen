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


def part_of(cid: str, n_inputs: int = 6) -> str:
    h = int(hashlib.sha256(f"cellgym-ood{n_inputs}:{cid}".encode()).hexdigest(), 16)
    return "dev" if h % DEV_MOD == 0 else "pool"


def core_functions() -> set[tuple[int, int]]:
    """(inputs, canonical OUT function) of every core cell, whatever its split."""
    out = set()
    for net in enumerate_networks(Params()):
        cell = build_cell(net, SIZING)
        out.add((net.n_inputs, canonical_fn(int(out_function_hex(cell), 16), net.n_inputs)))
    return out


def sample_networks(n_inputs: int, n_devices: int, max_stack: int, count: int, seed: int = 0,
                    tries: int = 2_000_000) -> list:
    """Random distinct irredundant networks with exactly n_inputs inputs and n_devices devices per
    side, both stacks <= max_stack. For sizes where exhaustive enumeration is too slow."""
    import random
    from cellgym.netlists.enumerate import (Network, _irredundant, _label, canonical, shapes,
                                            var_masks)
    rng = random.Random(seed)
    shp = shapes(n_devices, max_stack)[n_devices]
    masks, full = var_masks(n_inputs)
    seen = set()
    for _ in range(tries):
        if len(seen) >= count:
            break
        lab = [rng.randrange(n_inputs) for _ in range(n_devices)]
        if len(set(lab)) < n_inputs:
            continue
        t = _label(rng.choice(shp), iter(lab))
        if depth(t, dual=True) > max_stack or not _irredundant(t, n_devices, masks, full):
            continue
        seen.add(canonical(t, n_inputs))
    return [Network(t, n_inputs) for t in sorted(seen)]


def _label(nl: dict):
    from cellgym.synth.generate import euler_plans
    try:
        plans, fails = euler_plans(nl, 1)
    except Exception as e:  # noqa: BLE001 - any failure just leaves the cell out
        return nl["id"], None, [f"{type(e).__name__}"]
    return nl["id"], (plans[0] if plans else None), fails


def build(out: Path, jobs: int, params: Params = PARAMS, limit: int | None = None,
          min_devices: int = 0, min_stack: int = 0, nets=None) -> dict:
    """Cells with exactly params.max_inputs inputs. For 6 inputs the core (benchmark) trees and
    functions are excluded; the core has no cell with more than 6 inputs. `limit` keeps a
    hash-ordered subset of the candidates (for very large enumerations)."""
    t0 = time.time()
    n_in = params.max_inputs
    core_trees = {n.tree for n in enumerate_networks(Params())} if n_in <= 6 else set()
    bad = core_functions() if n_in <= 6 else set()
    stats = Counter()
    netlists = []
    for net in (nets if nets is not None else enumerate_networks(params)):
        if net.n_inputs != n_in or net.n_devices < min_devices:
            continue
        if max(depth(net.tree), depth(net.tree, dual=True)) < min_stack:
            continue
        if net.tree in core_trees:
            stats["core"] += 1
            continue
        cell = build_cell(net, SIZING)
        if bad and (6, canonical_fn(int(out_function_hex(cell), 16), 6)) in bad:
            stats["dropped_core_function"] += 1
            continue
        netlists.append(cell_json(cell, SIZING))
    stats["candidates"] = len(netlists)
    if limit and len(netlists) > limit:
        key = lambda nl: hashlib.sha256(f"cellgym-pick:{nl['id']}".encode()).hexdigest()
        netlists = sorted(netlists, key=key)[:limit]
        stats["kept_after_limit"] = len(netlists)
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
        if len(nl["inputs"]) != n_in:
            raise RuntimeError(f"unexpected cell {cid}")
        (out / "cells" / f"{cid}.json").write_text(json.dumps(nl, indent=1) + "\n", newline="\n")
        (out / "plans" / f"{cid}.plan").write_text(plan["plan"], newline="\n")
        index[cid] = {"part": part_of(cid, n_in), "w_euler_nm": round(plan["width_um"] * 1000),
                      "sites": plan["sites"], "breaks": plan["breaks"],
                      "n_devices": nl["n_devices"], "pdn_depth": nl["pdn_depth"],
                      "pun_depth": nl["pun_depth"]}
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", newline="\n")
    manifest = {"params": params.__dict__ | {"min_devices": min_devices, "min_stack": min_stack}, "stats": dict(stats), "route_failures": dict(fails),
                "cells": len(index), "by_part": dict(Counter(v["part"] for v in index.values())),
                "by_devices": dict(sorted(Counter(v["n_devices"] for v in index.values()).items())),
                "leak_check": ("no core tree, no core function (any split), all 6 inputs" if n_in == 6
                               else f"all {n_in} inputs; the core has none"),
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
             device: str = "cuda", save_plans: Path | None = None, pos_shift: int = 0) -> dict:
    import torch
    from cellgym.synth import model as M
    from cellgym.synth import tokens as T
    from cellgym.synth.constrain import sample_constrained
    from cellgym.synth.eval import MAX_NEW, score_samples
    torch.manual_seed(seed)
    model = M.load(ckpt, device)
    if pos_shift:   # probe: same tokens, positions moved up by pos_shift (absolute position embeddings)
        fwd = model.forward
        model.forward = lambda idx, caches=None, start=0, pos=None, mask=None: fwd(
            idx, caches, start=start + pos_shift, pos=None if pos is None else pos + pos_shift, mask=mask)
    samples = {c: sample_constrained(model, T.ids(T.encode_netlist(nl, model.cfg.prompt)), nl, n=k,
                                     temperature=temperature, max_new=MAX_NEW)
               for c, nl in cells.items()}
    res = score_samples(cells, samples, w_euler=w, split_of={c: "set" for c in cells})
    if save_plans:
        save_best_plans(res, cells, samples, Path(save_plans))
    return res


def save_best_plans(res: dict, cells: dict, samples: dict, out: Path) -> int:
    """Write the narrowest passing sample of each cell as <out>/plans/<id>.plan (plus its netlist
    in <out>/cells/), for cellgym.viewer.static. Returns the number of cells written."""
    from cellgym.synth.eval import decode_ids
    NL = chr(10)
    (out / "plans").mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    n = 0
    for c in res["cells"]:
        ok = [(s["width_nm"], i) for i, s in enumerate(c["samples"]) if s["pass"]]
        if not ok:
            continue
        i = min(ok)[1]
        (out / "plans" / f"{c['id']}.plan").write_text(decode_ids(cells[c["id"]], samples[c["id"]][i]), newline=NL)
        (out / "cells" / f"{c['id']}.json").write_text(json.dumps(cells[c["id"]], indent=1) + NL, newline=NL)
        n += 1
    return n


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
    b.add_argument("--inputs", type=int, default=6)
    b.add_argument("--max-stack", type=int, default=PARAMS.max_stack)
    b.add_argument("--extra", type=int, default=PARAMS.extra_devices)
    b.add_argument("--max-devices", type=int, default=None, help="default inputs + 2")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--min-devices", type=int, default=0, help="per side")
    b.add_argument("--min-stack", type=int, default=0, help="deepest of PDN/PUN")
    b.add_argument("--sample", type=int, default=0,
                   help="draw this many random networks with exactly --max-devices devices instead of enumerating")
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument("--set", type=Path, default=Path("out/ood6"))
    e.add_argument("--part", default="dev", choices=["dev", "pool", "all"])
    e.add_argument("--rename", action="store_true", help="renamed benchmark val cells instead")
    e.add_argument("--k", type=int, default=8)
    e.add_argument("--temperature", type=float, default=0.8)
    e.add_argument("--shift", type=int, default=0,
                   help="rename input i to letter i+SHIFT (6 = 6-input cells use G-L)")
    e.add_argument("--tag", default=None)
    e.add_argument("--pos-shift", type=int, default=0,
                   help="probe: add this to every position id (tests position vs. size effects)")
    e.add_argument("--save-plans", type=Path, default=None,
                   help="write each solved cell's narrowest passing plan here (for cellgym.viewer.static)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        prm = Params(max_inputs=a.inputs, max_stack=a.max_stack, extra_devices=a.extra,
                     max_devices=a.max_devices or a.inputs + 2)
        nets = (sample_networks(a.inputs, prm.max_devices, a.max_stack, a.sample) if a.sample else None)
        m = build(a.out, a.jobs, prm, a.limit, a.min_devices, a.min_stack, nets)
        print(json.dumps(m, indent=1))
        return 0
    cells, w = renamed_val() if a.rename else load(a.set, a.part)
    if a.shift:
        from cellgym.synth import tokens as T
        from cellgym.synth.train import relabel_netlist
        cells = {c: relabel_netlist(nl, {x: T.INPUTS[a.shift + i] for i, x in enumerate(nl["inputs"])})
                 for c, nl in cells.items()}
    res = evaluate(a.ckpt, cells, w, a.k, a.temperature, save_plans=a.save_plans, pos_shift=a.pos_shift)
    tag = a.tag or (("renamed_val" if a.rename else f"ood6_{a.part}") + (f"_shift{a.shift}" if a.shift else ""))
    res |= {"ckpt": str(a.ckpt), "k": a.k, "temperature": a.temperature}
    path = a.ckpt.parent / f"eval_{tag}_k{a.k}.json"
    path.write_text(json.dumps(res, indent=1) + "\n", newline="\n")
    print(summary_line(res), "->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
