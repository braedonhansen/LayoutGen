"""Set the embeddings of width tokens the model never saw, by hand (weight-space edit).

    python -m cellgym.synth.widthedit --ckpt CKPT --mode {mean,nearest,extrapolate} --out OUT

Stack-5 cells need nmos width 2.1 and pmos width 3.15 (5 x 0.42, 5 x 0.63), which no
training cell has: tokens.WIDTHS_STACK5. model.load gives them a generic start (mean of
all rows plus noise). This rewrites those two rows of the (tied) embedding:

  mean         leave the generic start (control)
  nearest      copy the deepest width the model knows on that side (w1.68, w2.52)
  extrapolate  continue each side's width sequence one step in embedding space:
               e(2.1) = 2 e(1.68) - e(1.26), e(3.15) = 2 e(2.52) - e(1.89)

Widths appear only in the prompt (plans never name them), so the question is only
whether the model reads a new width as "a deep-stack device" rather than as noise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from cellgym.synth import model as M
from cellgym.synth import tokens as T

N_SEQ = ["w0.42", "w0.84", "w1.26", "w1.68"]
P_SEQ = ["w0.63", "w1.26", "w1.89", "w2.52"]
NEW = {"w2.1": N_SEQ, "w3.15": P_SEQ}


def edit(model: M.GPT, mode: str) -> M.GPT:
    w = model.tok.weight.data
    e = lambda t: w[T.TOKEN_ID[t]]
    for tok, seq in NEW.items():
        if mode == "nearest":
            w[T.TOKEN_ID[tok]] = e(seq[-1]).clone()
        elif mode == "extrapolate":
            w[T.TOKEN_ID[tok]] = 2 * e(seq[-1]) - e(seq[-2])
        elif mode != "mean":
            raise ValueError(f"unknown mode {mode!r}")
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--mode", required=True, choices=["mean", "nearest", "extrapolate"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    model = edit(M.load(a.ckpt), a.mode)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    M.save(model, model.cfg, a.out)
    print(f"{a.mode}: wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
