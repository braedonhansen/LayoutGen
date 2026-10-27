"""CLI: python -m cellgym.netlists.generate --out out/cells"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .enumerate import Params, enumerate_networks
from .selfcheck import check_cell
from .transistors import Sizing, build_cell, fits
from .writers import cell_json, dumps, spice


def generate(out: Path, params: Params, sizing: Sizing) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    index, skipped = [], 0
    for net in enumerate_networks(params):
        cell = build_cell(net, sizing)
        problems = check_cell(cell)
        if problems:
            raise RuntimeError(f"self-check failed for {cell.id}: {problems[:3]}")
        if not fits(cell, sizing):
            skipped += 1
            continue
        (out / f"{cell.id}.sp").write_text(spice(cell), newline="\n")
        j = cell_json(cell, sizing)
        (out / f"{cell.id}.json").write_text(dumps(j), newline="\n")
        index.append({k: j[k] for k in (
            "id", "canonical_tree", "function", "pdn_depth", "pun_depth", "read_once",
            "n_devices", "max_w_n", "max_w_p")} | {"n_inputs": len(j["inputs"])})
    summary = {
        "n_cells": len(index),
        "n_skipped_not_fitting": skipped,
        "params": params.__dict__,
        "sizing": {k: v for k, v in sizing.__dict__.items()},
        "max_w_n": max(c["max_w_n"] for c in index),
        "max_w_p": max(c["max_w_p"] for c in index),
        "cells": index,
    }
    (out / "index.json").write_text(dumps(summary), newline="\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/cells"))
    ap.add_argument("--max-inputs", type=int, default=6)
    ap.add_argument("--max-stack", type=int, default=3)
    ap.add_argument("--extra-devices", type=int, default=2)
    ap.add_argument("--max-devices", type=int, default=8)
    ap.add_argument("--wn-unit", type=float, default=0.42)
    ap.add_argument("--l", type=float, default=0.15)
    ap.add_argument("--wp-wn-ratio", type=float, default=1.5)
    ap.add_argument("--max-diffusion-width", type=float, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    s = generate(a.out, Params(a.max_inputs, a.max_stack, a.extra_devices, a.max_devices),
                 Sizing(a.wn_unit, a.l, a.wp_wn_ratio,
                        max_diffusion_width=a.max_diffusion_width))
    print(f"{s['n_cells']} cells ({s['n_skipped_not_fitting']} skipped) -> {a.out}; "
          f"widest NMOS {s['max_w_n']} um, widest PMOS {s['max_w_p']} um; "
          f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
