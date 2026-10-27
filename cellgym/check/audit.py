#!/usr/bin/env python3
"""Audit the fast checker against ground truth (KLayout DRC + Magic/Netgen LVS).

    python -m cellgym.check.audit [PLANS...] [--cells DIR] [--mutants N] [--seed S]
                                  [--jobs J] [--magic] [--workdir DIR]

PLANS are .plan files or directories (default: out/plans and the two DSL
examples). Without --mutants, every plan is rendered and audited as is. With
--mutants N, N mutants (cellgym.check.mutate) are drawn from the plans whose
unmutated layout the fast checker passes. --magic also runs Magic's own DRC
(tap / latch-up rules excluded: cells are tapless by design), which is the
reference for the Magic-tech rules the KLayout deck lacks.

Outcome per layout: agree (both pass or both fail), false pass (fast ok,
ground truth fails), false fail (fast fails, ground truth ok). A false fail
whose fast violations are all Magic-tech rules is marked "strict"; with
--magic it is "strict, Magic agrees" when Magic DRC also fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cellgym.dsl.plan import parse
from cellgym.groundtruth.verify import verify_rendered
from cellgym.render.render import render

from . import check
from .drc import KLAYOUT_RULES
from .mutate import mutate

EXAMPLES = Path(__file__).resolve().parents[1] / "dsl" / "examples"
MAGIC_IGNORE = ("(nwell.4)", "(LU.2)", "(LU.3)")


def magic_drc(cell, out: Path, pdk_root: Path, pdk: str = "sky130A") -> list[str]:
    """Magic DRC (full style) on the cell's GDS; returns rule texts (one per box)."""
    os.environ["PDK_ROOT"] = str(pdk_root)
    out.mkdir(parents=True, exist_ok=True)
    gds = out / f"{cell.name}.mdrc.gds"
    gds.write_bytes(cell.gds())
    tcl = f"""
gds read {gds}
load {cell.name}
select top cell
drc style drc(full)
drc check
drc catchup
foreach {{why boxes}} [drc listall why] {{ puts "WHY\\t$why\\t[llength $boxes]" }}
quit -noprompt
"""
    p = subprocess.run(["magic", "-dnull", "-noconsole", "-rcfile",
                        str(pdk_root / pdk / "libs.tech/magic/sky130A.magicrc")],
                       input=tcl, capture_output=True, text=True, cwd=out)
    items = []
    for line in p.stdout.splitlines():
        if line.startswith("WHY\t"):
            _, why, n = line.split("\t")
            if not any(k in why for k in MAGIC_IGNORE):
                items += [why] * int(n)
    return items


def _rule_of(why: str) -> str:
    m = re.search(r"[(]([^()]*)[)]\s*$", why)
    return m.group(1) if m else why


def plan_files(args: list[str]) -> list[Path]:
    out = []
    for a in args:
        p = Path(a)
        out += sorted(p.glob("*.plan")) if p.is_dir() else [p]
    return out


def netlist_for(plan: Path, cells: Path) -> dict:
    p = cells / f"{plan.stem}.json"
    return json.loads((p if p.exists() else plan.with_suffix(".json")).read_text())


def audit_one(key: str, cell, netlist: dict, wd: Path, pdk_root: Path, magic: bool) -> dict:
    fast = check(cell, netlist)
    try:
        gt = verify_rendered(cell, wd / key, pdk_root)
        gt_drc, gt_lvs, gt_lines = gt.drc_items, gt.lvs_ok, [l.strip() for l in gt.lvs_lines]
    except Exception as e:  # report, don't abort the batch
        gt_drc, gt_lvs, gt_lines = None, False, [f"{type(e).__name__}: {e}"]
    rec = dict(key=key, fast_drc=[str(v) for v in fast.drc_violations],
               fast_rules=sorted({v.rule for v in fast.drc_violations}),
               fast_lvs=fast.lvs_ok, fast_lvs_errors=fast.lvs_errors, ms=fast.ms,
               gt_drc=gt_drc, gt_lvs=gt_lvs, gt_lvs_lines=gt_lines)
    if magic:
        rec["magic_drc"] = magic_drc(cell, wd / key, pdk_root)
    return classify(rec)


def classify(r: dict) -> dict:
    fast_ok = not r["fast_rules"] and r["fast_lvs"]
    gt_ok = r["gt_drc"] == [] and r["gt_lvs"]
    if fast_ok == gt_ok:
        r["outcome"] = "agree"
    elif fast_ok:
        r["outcome"] = "false pass"
    else:
        r["outcome"] = "false fail"
        w_only = not r["fast_lvs"] and all(e.startswith("wrong W") for e in r["fast_lvs_errors"])
        strict = (r["fast_lvs"] or (w_only and r["gt_lvs"])) and not (set(r["fast_rules"]) & KLAYOUT_RULES)
        if strict and not r["fast_lvs"]:
            r["outcome"] = "false fail (W tolerance 1 nm vs netgen 1%)"
        elif strict:
            r["outcome"] = "false fail (strict)"
            if "magic_drc" in r:
                r["outcome"] += ", Magic agrees" if r["magic_drc"] else ", Magic passes"
    # per-check agreement
    gk = sorted(set(r["gt_drc"] or []))
    fk = sorted(set(r["fast_rules"]) & KLAYOUT_RULES)
    r["drc_agree"] = (not fk) == (r["gt_drc"] == [])
    r["lvs_agree"] = r["fast_lvs"] == r["gt_lvs"]
    r["gt_rules"] = gk
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans", nargs="*")
    ap.add_argument("--cells", default="out/cells")
    ap.add_argument("--mutants", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--magic", action="store_true")
    ap.add_argument("--pdk-root", default=os.environ.get("PDK_ROOT", str(Path.home() / "pdk")))
    ap.add_argument("--workdir", default="out/audit")
    ap.add_argument("--json", default=None, help="write per-layout records here")
    a = ap.parse_args(argv)

    plans = plan_files(a.plans or ["out/plans", str(EXAMPLES / "aoi21.plan"), str(EXAMPLES / "half_adder.plan")])
    cells = Path(a.cells)
    base = []
    for p in plans:
        nl = netlist_for(p, cells)
        base.append((p.stem if p.parent != EXAMPLES else f"ex_{p.stem}", render(parse(p.read_text()), nl), nl))
    jobs = []
    if a.mutants:
        clean = [b for b in base if check(b[1], b[2]).ok]
        import random
        rng = random.Random(a.seed)
        for k in range(a.mutants):
            key, cell, nl = rng.choice(clean)
            s = a.seed * 100_000 + k
            m, desc = mutate(cell, s)
            jobs.append((f"{key}.m{s}", m, nl, desc))
        print(f"{len(jobs)} mutants from {len(clean)} fast-clean base layouts (seed {a.seed})")
    else:
        jobs = [(k, c, nl, "") for k, c, nl in base]
    wd = Path(a.workdir).resolve()
    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        recs = list(ex.map(lambda j: audit_one(j[0], j[1], j[2], wd, Path(a.pdk_root), a.magic) | {"mutation": j[3]}, jobs))
    dt = time.time() - t0

    outcomes = Counter(r["outcome"] for r in recs)
    ms = sorted(r["ms"] for r in recs)
    print(f"audited {len(recs)} layouts in {dt:.0f}s; fast check median {ms[len(ms) // 2]:.2f} ms, "
          f"p95 {ms[min(len(ms) - 1, int(0.95 * len(ms)))]:.2f} ms")
    for k, n in sorted(outcomes.items()):
        print(f"  {k:32s} {n}")
    print(f"  DRC agree (KLayout-deck rules only) {sum(r['drc_agree'] for r in recs)}/{len(recs)}; "
          f"LVS agree {sum(r['lvs_agree'] for r in recs)}/{len(recs)}")
    gt_fail = sum(1 for r in recs if not (r["gt_drc"] == [] and r["gt_lvs"]))
    print(f"  ground truth fails {gt_fail}; fast fails {sum(1 for r in recs if r['fast_rules'] or not r['fast_lvs'])}")
    for r in recs:
        if r["outcome"] != "agree" or not r["drc_agree"] or not r["lvs_agree"]:
            print(f"- {r['key']}: {r['outcome']}  {r['mutation']}")
            print(f"    fast: DRC {', '.join(r['fast_rules']) or 'ok'}; LVS {'ok' if r['fast_lvs'] else '; '.join(r['fast_lvs_errors'][:3])}")
            gd = "no report" if r["gt_drc"] is None else ", ".join(f"{k} x{v}" for k, v in sorted(Counter(r["gt_drc"]).items())) or "ok"
            print(f"    gt:   DRC {gd}; LVS {'ok' if r['gt_lvs'] else ' | '.join(r['gt_lvs_lines'][-2:])}")
            if "magic_drc" in r:
                md = ", ".join(f"{_rule_of(k)} x{v}" for k, v in sorted(Counter(r["magic_drc"]).items()))
                print(f"    magic DRC: {md or 'ok'}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(recs, indent=1))
    bad = outcomes.get("false pass", 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
