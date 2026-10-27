#!/usr/bin/env python3
"""CellGym M2: render a plan, then KLayout DRC + Magic extract + Netgen LVS.

    python -m cellgym.groundtruth.verify PLAN [NETLIST] [--pdk-root DIR] [--workdir DIR]

PLAN is a .plan file or an example name (aoi21, half_adder); NETLIST defaults
to the .json next to the plan. Run inside WSL/Linux with klayout, magic and
netgen on PATH. Exit code 0 only if DRC has 0 violations and LVS matches.
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

from .run_m0 import drc, extract, lvs

EXAMPLES = Path(__file__).resolve().parents[1] / "dsl" / "examples"


@dataclass
class Result:
    cell: Rendered
    drc_items: list[str] | None
    lvs_ok: bool
    lvs_lines: list[str]

    @property
    def ok(self) -> bool:
        return self.drc_items == [] and self.lvs_ok


def resolve(plan: str, netlist: str | None) -> tuple[Path, Path]:
    p = Path(plan)
    if not p.exists():
        p = EXAMPLES / f"{plan}.plan"
    return p, Path(netlist) if netlist else p.with_suffix(".json")


def verify(plan_path: Path, netlist_path: Path, out: Path, pdk_root: Path, pdk: str = "sky130A") -> Result:
    """Render, write GDS + reference SPICE into `out`, run DRC and LVS."""
    os.environ["PDK_ROOT"] = str(pdk_root)   # magicrc reads this
    tech = pdk_root / pdk / "libs.tech"
    cell = render(parse(plan_path.read_text()), json.loads(netlist_path.read_text()))
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
    return Result(cell, items, ok, lines)


def report(r: Result, out: Path) -> str:
    c = r.cell
    lines = [f"== {c.name}  width {c.width:.3f} um ({c.sites} sites)  height {c.height:.2f} um  -> {out}"]
    if r.drc_items is None:
        lines.append("DRC : FAIL (no report produced)")
    else:
        lines.append(f"DRC : {'PASS' if not r.drc_items else 'FAIL'}  violations={len(r.drc_items)}")
        lines += [f"      {n:4d} x {rule}" for rule, n in sorted(Counter(r.drc_items).items())]
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
    a = ap.parse_args(argv)
    plan, netlist = resolve(a.plan, a.netlist)
    out = (Path(a.workdir) / plan.stem).resolve()
    r = verify(plan, netlist, out, Path(a.pdk_root), a.pdk)
    print(report(r, out))
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
