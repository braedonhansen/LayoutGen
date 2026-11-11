"""Probe sets far outside training, with no Euler label (its search is too slow at these sizes).

    python -m cellgym.llm.probe --out out/probe --spec 8:14 10:16 12:20 --count 30

Each spec inputs:per_side gives a devset-style dir out/probe/i<inputs>d<per_side> (cells/, index.json
with part "all" and w_euler_nm null), random irredundant networks with stack <= 4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cellgym.netlists.transistors import build_cell
from cellgym.netlists.writers import cell_json
from cellgym.synth.devset import sample_networks
from cellgym.synth.generate import SIZING


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/probe"))
    ap.add_argument("--spec", nargs="+", default=["8:14", "10:16", "12:20"])
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--max-stack", type=int, default=4)
    a = ap.parse_args(argv)
    for sp in a.spec:
        n_in, per = (int(x) for x in sp.split(":"))
        d = a.out / f"i{n_in}d{per}"
        (d / "cells").mkdir(parents=True, exist_ok=True)
        nets = sample_networks(n_in, per, a.max_stack, a.count, seed=4242 + per)
        index = {}
        for net in nets:
            nl = cell_json(build_cell(net, SIZING), SIZING)
            (d / "cells" / f"{nl['id']}.json").write_text(json.dumps(nl, indent=1) + "\n", newline="\n")
            index[nl["id"]] = {"part": "all", "w_euler_nm": None, "n_devices": nl["n_devices"]}
        (d / "index.json").write_text(json.dumps(index, indent=1) + "\n", newline="\n")
        print(sp, len(index), "cells ->", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
