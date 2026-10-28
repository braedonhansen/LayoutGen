#!/usr/bin/env python3
"""CellGym M0: DRC (KLayout) + LVS (Magic extract -> Netgen) of one stock
sky130_fd_sc_hd cell, plus Magic full DRC. Run inside WSL/Linux with klayout,
magic, netgen on PATH.

    python run_m0.py [CELL] [--pdk-root DIR] [--workdir DIR] [--klayout-only]

Env: PDK_ROOT (default ~/pdk), PDK (default sky130A), CELLGYM_OUT (workdir).
Ground truth (exit code 0) = KLayout deck DRC 0 violations AND Magic full DRC 0
errors outside MAGIC_EXCLUDE (skipped with --klayout-only) AND Netgen LVS
matches uniquely with no property errors.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

LIB = "sky130_fd_sc_hd"


def run(cmd, log, stdin=None, env=None, cwd=None):
    """Run a tool, write combined output to `log`, return the text."""
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env, cwd=cwd)
    text = p.stdout + p.stderr
    Path(log).write_text(text)
    if p.returncode != 0:
        print(f"[warn] {cmd[0]} exited {p.returncode}; see {log}")
    return text


def drc(cell, gds, tech, out):
    """KLayout batch DRC with the SKY130 runset (FEOL+BEOL+offgrid enabled;
    the deck defaults all groups to off when run with -b)."""
    rep = out / f"{cell}.drc.lyrdb"
    run(["klayout", "-b", "-r", str(tech / "klayout/drc/sky130A_mr.drc"),
         "-rd", f"input={gds}", "-rd", f"top_cell={cell}",
         "-rd", f"report={rep}", "-rd", "feol=true", "-rd", "beol=true",
         "-rd", "offgrid=true", "-rd", "seal=false"], out / f"{cell}.drc.log")
    if not rep.exists():
        return None, rep
    xml = rep.read_text()
    items = re.findall(r"<item>.*?<category>'?([^<']*)'?</category>", xml, re.S)
    return items, rep


# Magic DRC results expected by design and excluded from ground truth: CellGym
# cells (like the stock sky130_fd_sc_hd cells) are tapless; taps come from
# separate tap cells in the row, so the n-well tap and latch-up distance rules
# cannot be met inside one cell. Exact `drc listall why` texts (sky130A.tech).
MAGIC_EXCLUDE = (
    "All nwells must contain metal-connected N+ taps (nwell.4)",
    "N-diff distance to P-tap must be < 15.0um (LU.2)",
    "P-diff distance to N-tap must be < 15.0um (LU.3)",
)


def magic_drc(cell, gds, tech, out):
    """Magic full-style DRC of `cell` read from `gds`. Returns (items, excluded):
    one rule text per error box, split by MAGIC_EXCLUDE; items is None if
    Magic produced no result."""
    tcl = f"""
gds read {gds}
load {cell}
select top cell
drc style drc(full)
drc check
drc catchup
puts "DRCBEGIN"
foreach {{why boxes}} [drc listall why] {{ puts "WHY\t$why\t[llength $boxes]" }}
puts "DRCEND"
quit -noprompt
"""
    text = run(["magic", "-dnull", "-noconsole", "-rcfile", str(tech / "magic/sky130A.magicrc")],
               out / f"{cell}.mdrc.log", stdin=tcl, env=os.environ, cwd=out)
    if "DRCEND" not in text:
        return None, []
    items, excluded = [], []
    for line in text.splitlines():
        if line.startswith("WHY\t"):
            _, why, n = line.rsplit("\t", 2)
            (excluded if why in MAGIC_EXCLUDE else items).extend([why] * int(n))
    return items, excluded


def extract(cell, gds, tech, out):
    """Magic batch: read the library GDS, extract `cell` to SPICE."""
    sp = out / f"{cell}.ext.spice"
    tcl = f"""
gds read {gds}
load {cell}
select top cell
extract all
ext2spice lvs
ext2spice hierarchy off
ext2spice subcircuit top on
ext2spice -o {sp}
quit -noprompt
"""
    run(["magic", "-dnull", "-noconsole", "-rcfile", str(tech / "magic/sky130A.magicrc")],
        out / f"{cell}.magic.log", stdin=tcl, env=os.environ, cwd=out)  # .ext files land in cwd
    return sp


def lvs(cell, ext_sp, ref, tech, out):
    rpt = out / f"{cell}.lvs.txt"
    run(["netgen", "-batch", "lvs", f"{ext_sp} {cell}", f"{ref} {cell}",
         str(tech / "netgen/sky130A_setup.tcl"), str(rpt)], out / f"{cell}.netgen.log")
    text = rpt.read_text() if rpt.exists() else ""
    ok = bool(re.search(r"Circuits match uniquely", text)) and \
        not re.search(r"Pin names? (don't|do not) match|Netlists do not match|[Pp]roperty errors", text)
    final = [l for l in text.splitlines() if "match" in l.lower()][-2:]
    return ok, final, rpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", nargs="?", default=f"{LIB}__nand2_1")
    ap.add_argument("--pdk-root", default=os.environ.get("PDK_ROOT", str(Path.home() / "pdk")))
    ap.add_argument("--pdk", default=os.environ.get("PDK", "sky130A"))
    ap.add_argument("--workdir", default=os.environ.get("CELLGYM_OUT", "out"))
    ap.add_argument("--klayout-only", action="store_true", help="skip Magic DRC")
    a = ap.parse_args()

    root = Path(a.pdk_root)
    env_root = str(root)
    os.environ["PDK_ROOT"] = env_root  # magicrc reads this
    tech = root / a.pdk / "libs.tech"
    ref_dir = root / a.pdk / "libs.ref" / LIB
    gds = ref_dir / "gds" / f"{LIB}.gds"
    ref = ref_dir / "spice" / f"{LIB}.spice"
    out = (Path(a.workdir) / a.cell).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in (gds, ref):
        if not f.exists():
            sys.exit(f"missing {f}")

    items, rep = drc(a.cell, gds, tech, out)
    drc_ok = items is not None and len(items) == 0
    ext = extract(a.cell, gds, tech, out)
    if not ext.exists():
        lvs_ok, final, lrpt = False, ["magic produced no netlist"], out
    else:
        lvs_ok, final, lrpt = lvs(a.cell, ext, ref, tech, out)
    mitems, mexcl = ([], []) if a.klayout_only else magic_drc(a.cell, gds, tech, out)
    magic_ok = mitems == []

    print(f"== {a.cell}")
    if items is None:
        print("DRC : FAIL (no report produced)")
    else:
        by_rule = {}
        for r in items:
            by_rule[r] = by_rule.get(r, 0) + 1
        print(f"DRC : {'PASS' if drc_ok else 'FAIL'}  violations={len(items)}  report={rep}")
        for r, n in sorted(by_rule.items()):
            print(f"      {n:4d} x {r}")
    if not a.klayout_only:
        if mitems is None:
            print("MDRC: FAIL (magic produced no result)")
        else:
            print(f"MDRC: {'PASS' if magic_ok else 'FAIL'}  errors={len(mitems)}  excluded={len(mexcl)}")
            for why in sorted(set(mitems)):
                print(f"      {mitems.count(why):4d} x {why}")
            for why in sorted(set(mexcl)):
                print(f"      {mexcl.count(why):4d} x (excluded) {why}")
    print(f"LVS : {'PASS' if lvs_ok else 'FAIL'}  report={lrpt}")
    for l in final:
        print(f"      {l.strip()}")
    sys.exit(0 if (drc_ok and magic_ok and lvs_ok) else 1)


if __name__ == "__main__":
    main()
