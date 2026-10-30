"""Euler reference widths -> bench/euler_widths_v0.json (committed).

    python -m cellgym.bench.reference [--cells out/cells] [--out bench/euler_widths_v0.json] [--check]

W_euler per cell: the exact rendered width (nm, and um) of the Euler
baseline's plan, plus sites and the netlist sha256 it was computed from. The
scorer's reward 1 + (W_euler - W) / W_euler reads this file, so it stays
reproducible when the baseline changes. --check recomputes and compares.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cellgym.baselines.euler import euler_solve
from cellgym.dsl.plan import parse
from cellgym.render.render import render

from .common import EULER_WIDTHS_FILE, dump_rows, hash_ok, load_cells, sha256, with_hash

VERSION = "v0"


def euler_width(netlist: dict) -> dict:
    r = euler_solve(netlist)
    cell = render(parse(r.text), netlist)
    return {"width_nm": cell.width_nm, "width_um": cell.width_nm / 1000, "sites": cell.sites,
            "netlist_sha256": sha256(netlist)}


def make_reference(cells: dict[str, dict]) -> dict:
    return with_hash({"version": VERSION, "generator": "cellgym.baselines.euler",
                      "cells": {cid: euler_width(nl) for cid, nl in sorted(cells.items())}})


def load(path: Path = EULER_WIDTHS_FILE) -> dict:
    doc = json.loads(Path(path).read_text())
    if not hash_ok(doc):
        raise ValueError(f"{path}: sha256 does not match its contents")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="out/cells")
    ap.add_argument("--out", default=str(EULER_WIDTHS_FILE))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    doc = make_reference(load_cells(Path(a.cells)))
    text = dump_rows(doc, "cells")
    if a.check:
        same = Path(a.out).read_text() == text
        print("euler reference reproducible" if same else f"MISMATCH with {a.out}")
        return 0 if same else 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, newline="\n")
    print(f"{a.out}: {len(doc['cells'])} cells  sha256 {doc['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
