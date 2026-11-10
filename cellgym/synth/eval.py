"""Score sampled plans from the tiny model with the repo's own pipeline.

    python -m cellgym.synth.eval --ckpt out/synth/run0/best.pt --splits val test_id [test_ood]
        [--k 8] [--temperature 0.8] [--cells out/cells] [--tag NAME] [--out FILE] [--device cuda]

For each cell of the chosen splits the model samples k target-token sequences
(prompt = tokens.encode_netlist). Each is decoded (cut at <eos>, header rebuilt
from the netlist) and run through cellgym.bench.evaluate.evaluate: parse,
validate, render, fast check. Reward and W/W_euler use bench/euler_widths_v0.json
and the same formula as bench.score.reward. Anything that fails to decode,
parse, validate, render or check counts as a failure and never raises.

score_samples() needs no torch; only the CLI imports cellgym.synth.model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from cellgym.bench import reference, splits
from cellgym.bench.common import EULER_WIDTHS_FILE, SPLITS, SPLITS_FILE, load_cells
from cellgym.bench.evaluate import evaluate
from cellgym.bench.score import reward
from cellgym.synth import tokens as T

MAX_NEW = 640                   # capped per prompt by the model's max_len


def decode_ids(netlist: dict, ids: list[int]) -> str:
    """Plan text for generated ids: cut at the first <eos>, rebuild the header."""
    toks = []
    for i in ids:
        if not 0 <= i < len(T.VOCAB):
            raise ValueError(f"id {i} outside the vocabulary")
        if i == T.EOS:
            break
        toks.append(T.VOCAB[i])
    return T.decode_plan(toks, netlist)


def score_sample(netlist: dict, ids: list[int], w_euler_nm: int | None) -> dict:
    try:
        stage, detail, ev = "pass", "", None
        try:
            text = decode_ids(netlist, ids)
        except Exception as e:  # garbled ids
            stage, detail = "decode", f"{type(e).__name__}: {e}"
        else:
            ev = evaluate(text, netlist)
            stage, detail = ev.stage, ev.detail
    except Exception as e:  # the pipeline must not take the run down
        stage, detail, ev = "error", f"{type(e).__name__}: {e}", None
    ok = stage == "pass"
    w = ev.width_nm if ok else None
    return {
        "stage": stage, "detail": detail[:200],
        "parsed": stage not in ("decode", "parse", "error"),
        "valid": stage not in ("decode", "parse", "validate", "error"),
        "pass": ok, "width_nm": w, "sites": ev.sites if ok else None,
        "breaks": ev.breaks if ok else None,
        "w_over_euler": w / w_euler_nm if ok and w_euler_nm else None,
        "reward": reward(w, w_euler_nm, True) if ok and w_euler_nm else (0.0 if w_euler_nm else None),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def aggregate(cells: list[dict]) -> dict:
    samples = [s for c in cells for s in c["samples"]]
    n = len(cells)
    return {
        "cells": n,
        "samples": len(samples),
        "parse_rate": _mean([float(s["parsed"]) for s in samples]),
        "valid_rate": _mean([float(s["valid"]) for s in samples]),
        "sample_pass_rate": _mean([float(s["pass"]) for s in samples]),
        "pass_at_1": _mean([float(c["pass_at_1"]) for c in cells]),
        "pass_at_k": _mean([float(c["pass_at_k"]) for c in cells]),
        "best_of_k_reward": _mean([c["best_reward"] for c in cells]),
        "w_over_euler_passing": _mean([s["w_over_euler"] for s in samples if s["pass"]]),
        "w_over_euler_best": _mean([c["best_w_over_euler"] for c in cells]),
    }


def score_samples(netlists: dict[str, dict], samples: dict[str, list[list[int]]],
                  w_euler: dict[str, int] | None = None,
                  split_of: dict[str, str] | None = None) -> dict:
    """Score sampled id lists per cell id.

    w_euler: cell id -> Euler width in nm (default: bench/euler_widths_v0.json);
    cells without one get no ratio and a reward of None.
    split_of: cell id -> split (default: bench/splits_v0.json; unknown -> "other").
    """
    if w_euler is None:
        w_euler = {c: v["width_nm"] for c, v in reference.load(EULER_WIDTHS_FILE)["cells"].items()}
    if split_of is None:
        split_of = splits.split_of(splits.load(SPLITS_FILE))
    cells = []
    for cid in sorted(samples):
        nl = netlists[cid]
        scored = [score_sample(nl, ids, w_euler.get(cid)) for ids in samples[cid]]
        passing = [s for s in scored if s["pass"]]
        rewards = [s["reward"] for s in scored if s["reward"] is not None]
        ratios = [s["w_over_euler"] for s in passing if s["w_over_euler"] is not None]
        cells.append({
            "id": cid, "split": split_of.get(cid, "other"), "k": len(scored),
            "pass_at_1": bool(scored and scored[0]["pass"]), "pass_at_k": bool(passing),
            "best_reward": max(rewards) if rewards else (0.0 if w_euler.get(cid) else None),
            "best_w_over_euler": min(ratios) if ratios else None,
            "samples": scored,
        })
    groups: dict[str, list[dict]] = {}
    for c in cells:
        groups.setdefault(c["split"], []).append(c)
    return {"summary": {**{g: aggregate(cs) for g, cs in sorted(groups.items())}, "all": aggregate(cells)},
            "cells": cells}


# -- CLI -------------------------------------------------------------------------------

def table(result: dict) -> str:
    cols = ["cells", "parse_rate", "valid_rate", "pass_at_1", "pass_at_k", "best_of_k_reward",
            "w_over_euler_best"]
    f = lambda x: "-" if x is None else (str(x) if isinstance(x, int) else f"{x:.3f}")
    lines = ["| split | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for g, m in result["summary"].items():
        lines.append(f"| {g} | " + " | ".join(f(m[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test_id"], choices=SPLITS)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--cells", type=Path, default=Path("out/cells"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--constrained", action="store_true", help="grammar-constrained decoding (synth.constrain)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    import torch
    from cellgym.synth import model as M      # lazy: scoring itself needs no torch

    if not any(a.cells.glob("cg_*.json")):
        subprocess.run([sys.executable, "-m", "cellgym.netlists.generate", "--out", str(a.cells)], check=True)
    cells = load_cells(a.cells)
    split = splits.split_of(splits.load(SPLITS_FILE))
    ids = sorted(c for c in split if split[c] in a.splits and c in cells)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    model = M.load(a.ckpt, device)
    t0 = time.time()
    samples = {}
    for cid in ids:
        prompt = T.ids(T.encode_netlist(cells[cid], model.cfg.prompt))
        if a.constrained:
            from cellgym.synth.constrain import sample_constrained
            samples[cid] = sample_constrained(model, prompt, cells[cid], n=a.k,
                                              temperature=a.temperature, max_new=MAX_NEW)
        else:
            samples[cid] = M.sample(model, prompt, n=a.k, temperature=a.temperature, max_new=MAX_NEW)
    t_sample = time.time() - t0
    result = score_samples({c: cells[c] for c in ids}, samples)
    result |= {"ckpt": str(a.ckpt), "splits": a.splits, "k": a.k, "temperature": a.temperature,
               "seed": a.seed, "seconds": {"sample": round(t_sample, 1),
                                           "total": round(time.time() - t0, 1)}}
    tag = a.tag or "-".join(a.splits)
    out = a.out or a.ckpt.parent / f"eval_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", newline="\n")
    print(table(result))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
