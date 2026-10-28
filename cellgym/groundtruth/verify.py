#!/usr/bin/env python3
"""CellGym ground truth: render a plan, then KLayout DRC + Magic DRC + Netgen LVS.

    python -m cellgym.groundtruth.verify PLAN [NETLIST] [--pdk-root DIR] [--workdir DIR]
                                         [--klayout-only]

PLAN is a .plan file or an example name (aoi21, half_adder); NETLIST defaults
to the .json next to the plan. Run inside WSL/Linux with klayout, magic and
netgen on PATH.

Ground-truth clean (exit code 0) = KLayout deck DRC (sky130A_mr.drc) reports 0
AND Magic full-style DRC reports 0 after excluding exactly the tapless-cell
rules in run_m0.MAGIC_EXCLUDE (nwell.4, LU.2, LU.3) AND Netgen LVS (Magic
extraction) matches uniquely with no property errors. --klayout-only drops
the Magic DRC term.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cellgym.dsl.plan import parse
from cellgym.render.render import Rendered, render

from .run_m0 import MAGIC_EXCLUDE, drc, extract, lvs, magic_drc

EXAMPLES = Path(__file__).resolve().parents[1] / "dsl" / "examples"


@dataclass
class Result:
    cell: Rendered
    drc_items: list[str] | None
    lvs_ok: bool
    lvs_lines: list[str]
    magic_items: list[str] | None = None     # None: not run (--klayout-only) or no result
    magic_excluded: list[str] | None = None
    magic_run: bool = True

    @property
    def magic_ok(self) -> bool:
        return not self.magic_run or self.magic_items == []

    @property
    def ok(self) -> bool:
        return self.drc_items == [] and self.magic_ok and self.lvs_ok


def resolve(plan: str, netlist: str | None) -> tuple[Path, Path]:
    p = Path(plan)
    if not p.exists():
        p = EXAMPLES / f"{plan}.plan"
    return p, Path(netlist) if netlist else p.with_suffix(".json")


def verify(plan_path: Path, netlist_path: Path, out: Path, pdk_root: Path, pdk: str = "sky130A",
           magic: bool = True) -> Result:
    """Render, write GDS + reference SPICE into `out`, run DRC and LVS."""
    cell = render(parse(plan_path.read_text()), json.loads(netlist_path.read_text()))
    return verify_rendered(cell, out, pdk_root, pdk, magic)


def verify_rendered(cell: Rendered, out: Path, pdk_root: Path, pdk: str = "sky130A",
                    magic: bool = True) -> Result:
    """Ground truth of an already rendered (possibly modified) cell."""
    os.environ["PDK_ROOT"] = str(pdk_root)   # magicrc reads this
    tech = pdk_root / pdk / "libs.tech"
    out.mkdir(parents=True, exist_ok=True)
    gds = out / f"{cell.name}.gds"
    ref = out / f"{cell.name}.ref.spice"
    gds.write_bytes(cell.gds())
    ref.write_text(cell.spice)
    items, _ = drc(cell.name, gds, tech, out)
    ext = extract(cell.name, gds, tech, out)
    if ext.exists():
        ok, lines, _ = lvs(cell.name, ext, ref, tech, out)
    else:
        ok, lines = False, ["magic produced no netlist"]
    mi, mx = magic_drc(cell.name, gds, tech, out) if magic else (None, None)
    return Result(cell, items, ok, lines, mi, mx, magic)


def report(r: Result, out: Path) -> str:
    c = r.cell
    lines = [f"== {c.name}  width {c.width:.3f} um ({c.sites} sites)  height {c.height:.2f} um  -> {out}"]
    if r.drc_items is None:
        lines.append("DRC : FAIL (no report produced)")
    else:
        lines.append(f"DRC : {'PASS' if not r.drc_items else 'FAIL'}  violations={len(r.drc_items)}")
        lines += [f"      {n:4d} x {rule}" for rule, n in sorted(Counter(r.drc_items).items())]
    if r.magic_run:
        if r.magic_items is None:
            lines.append("MDRC: FAIL (no result produced)")
        else:
            lines.append(f"MDRC: {'PASS' if not r.magic_items else 'FAIL'}  errors={len(r.magic_items)}  "
                         f"excluded (tapless, by design)={len(r.magic_excluded)}")
            lines += [f"      {n:4d} x {why}" for why, n in sorted(Counter(r.magic_items).items())]
    lines.append(f"LVS : {'PASS' if r.lvs_ok else 'FAIL'}")
    lines += [f"      {l.strip()}" for l in r.lvs_lines]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("netlist", nargs="?")
    ap.add_argument("--pdk-root", default=os.environ.get("PDK_ROOT", str(Path.home() / "pdk")))
    ap.add_argument("--pdk", default=os.environ.get("PDK", "sky130A"))
    ap.add_argument("--workdir", default=os.environ.get("CELLGYM_OUT", "out/verify"))
    ap.add_argument("--klayout-only", action="store_true", help="skip Magic DRC (KLayout deck + LVS only)")
    a = ap.parse_args(argv)
    plan, netlist = resolve(a.plan, a.netlist)
    out = (Path(a.workdir) / plan.stem).resolve()
    r = verify(plan, netlist, out, Path(a.pdk_root), a.pdk, magic=not a.klayout_only)
    print(report(r, out))
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
