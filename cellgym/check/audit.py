#!/usr/bin/env python3
"""Audit the fast checker against ground truth.

    python -m cellgym.check.audit [PLANS...] [--cells DIR] [--mutants N] [--seed S[,S...]]
                                  [--jobs J] [--klayout-only] [--workdir DIR]
                                  [--json FILE] [--md FILE]

PLANS are .plan files or directories (default: out/plans and the two DSL
examples). Without --mutants, every plan is rendered and audited as is. With
--mutants N, N mutants (cellgym.check.mutate) are drawn per seed from the plans
whose unmutated layout the fast checker passes: bases are taken round-robin
from a per-seed shuffle and mutation ops round-robin over mutate.OPS, so both
are spread evenly.

Ground truth (cellgym.groundtruth.verify): KLayout deck DRC = 0 AND Magic
full DRC = 0 outside run_m0.MAGIC_EXCLUDE AND Netgen LVS match without
property errors. --klayout-only drops the Magic DRC term.

Outcome per layout: agree (both pass or both fail), false pass (fast ok,
ground truth fails), false fail (fast fails, ground truth clean).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cellgym.dsl.plan import parse
from cellgym.groundtruth.run_m0 import MAGIC_EXCLUDE
from cellgym.groundtruth.verify import verify_rendered
from cellgym.render.render import render

from . import check
from .drc import RULES
from .mutate import OPS, mutate

EXAMPLES = Path(__file__).resolve().parents[1] / "dsl" / "examples"


def rule_of(why: str) -> str:
    """Magic DRC text -> its rule tag, e.g. '... < 0.25um (poly.7)' -> 'poly.7'."""
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


def audit_one(key: str, cell, netlist: dict, wd: Path, pdk_root: Path, magic: bool = True) -> dict:
    fast = check(cell, netlist)
    try:
        gt = verify_rendered(cell, wd / key, pdk_root, magic=magic)
        gt_drc, gt_lvs, gt_lines = gt.drc_items, gt.lvs_ok, [l.strip() for l in gt.lvs_lines]
        gt_magic = gt.magic_items if magic else []
    except Exception as e:  # report, don't abort the batch
        gt_drc, gt_lvs, gt_lines, gt_magic = None, False, [f"{type(e).__name__}: {e}"], None
    rec = dict(key=key, fast_drc=[str(v) for v in fast.drc_violations],
               fast_rules=sorted({v.rule for v in fast.drc_violations}),
               fast_lvs=fast.lvs_ok, fast_lvs_errors=fast.lvs_errors, ms=fast.ms,
               gt_drc=gt_drc, gt_magic=gt_magic, gt_lvs=gt_lvs, gt_lvs_lines=gt_lines)
    return classify(rec)


def classify(r: dict) -> dict:
    fast_drc_ok, fast_lvs_ok = not r["fast_rules"], r["fast_lvs"]
    gt_drc_ok = r["gt_drc"] == [] and r["gt_magic"] == []
    gt_ok = gt_drc_ok and r["gt_lvs"]
    fast_ok = fast_drc_ok and fast_lvs_ok
    r["outcome"] = "agree" if fast_ok == gt_ok else "false pass" if fast_ok else "false fail"
    r["drc_agree"] = fast_drc_ok == gt_drc_ok
    r["lvs_agree"] = fast_lvs_ok == r["gt_lvs"]
    r["gt_rules"] = sorted(set(r["gt_drc"] or []) | {rule_of(w) for w in r["gt_magic"] or []})
    r["ff_class"] = ff_class(r) if r["outcome"] == "false fail" else ""
    return r


def ff_class(r: dict) -> str:
    """Why the fast checker failed a ground-truth-clean layout."""
    if not r["fast_lvs"]:
        return "LVS: " + (r["fast_lvs_errors"][0].split(":")[0] if r["fast_lvs_errors"] else "?")
    strict = [x for x in r["fast_rules"] if RULES.get(x, (0, "K"))[1] == "M"]
    if strict == r["fast_rules"]:
        return "strict DRC (Magic-tech-derived rules, not flagged by ground truth here): " + ", ".join(strict)
    return "DRC: " + ", ".join(r["fast_rules"])


def mutant_jobs(clean: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    order = list(clean)
    rng.shuffle(order)
    ops = list(OPS)
    jobs = []
    for k in range(n):
        if k % len(ops) == 0:
            rng.shuffle(ops)
        key, cell, nl = order[k % len(order)]
        s = seed * 100_000 + k
        op = ops[k % len(ops)]
        m, desc = mutate(cell, s, op)
        jobs.append((f"{key}.m{s}", m, nl, desc, op))
    return jobs


def summarize(recs: list[dict]) -> str:
    """Markdown summary: totals, per mutation type, rules hit."""
    n = len(recs)
    oc = Counter(r["outcome"] for r in recs)
    ms = sorted(r["ms"] for r in recs)
    fp, ff = oc.get("false pass", 0), oc.get("false fail", 0)
    gt_fail = sum(1 for r in recs if r["gt_drc"] != [] or r["gt_magic"] != [] or not r["gt_lvs"])
    lines = [
        "# M3 fast-checker audit",
        "",
        "Ground truth: KLayout deck DRC (sky130A_mr.drc) = 0 AND Magic full DRC = 0 "
        "(excluding only the tapless-by-design rules below) AND Netgen LVS match, no property errors.",
        "",
        "Excluded Magic rules: " + "; ".join(f"`{w}`" for w in MAGIC_EXCLUDE),
        "",
        f"- layouts audited: {n} (ground truth fails {gt_fail}, clean {n - gt_fail})",
        f"- agree: {oc.get('agree', 0)}; false pass: {fp}; false fail: {ff} "
        f"({100 * ff / n:.2f}% of all, {100 * ff / max(1, n - gt_fail):.2f}% of ground-truth-clean)",
        f"- DRC agree {sum(r['drc_agree'] for r in recs)}/{n}; LVS agree {sum(r['lvs_agree'] for r in recs)}/{n}",
        f"- fast check time: median {ms[n // 2]:.2f} ms, p95 {ms[min(n - 1, int(0.95 * n))]:.2f} ms",
        "",
        "## Per mutation type",
        "",
        "| mutation | n | gt fails | agree | false pass | false fail |",
        "|---|---|---|---|---|---|",
    ]
    by = defaultdict(list)
    for r in recs:
        by[r.get("op") or "none"].append(r)
    for op in sorted(by):
        rs = by[op]
        c = Counter(r["outcome"] for r in rs)
        g = sum(1 for r in rs if r["gt_drc"] != [] or r["gt_magic"] != [] or not r["gt_lvs"])
        lines.append(f"| {op} | {len(rs)} | {g} | {c.get('agree', 0)} | {c.get('false pass', 0)} | "
                     f"{c.get('false fail', 0)} |")
    fr, gr = Counter(), Counter()
    for r in recs:
        fr.update(r["fast_rules"] + ([] if r["fast_lvs"] else ["LVS"]))
        gr.update(r["gt_rules"] + ([] if r["gt_lvs"] else ["LVS"]))
    lines += ["", "## Rules hit (layouts flagged)", "", "| rule | fast checker | ground truth |", "|---|---|---|"]
    for k in sorted(set(fr) | set(gr)):
        lines.append(f"| {k} | {fr.get(k, 0)} | {gr.get(k, 0)} |")
    ffc = Counter(r.get("ff_class", "") for r in recs if r["outcome"] == "false fail")
    if ffc:
        lines += ["", "## False fails by class", ""]
        lines += [f"- {k}: {n}" for k, n in ffc.most_common()]
    bad = [r for r in recs if r["outcome"] != "agree"]
    if bad:
        lines += ["", "## Disagreements", ""]
        for r in bad:
            lines.append(f"- `{r['key']}` {r['outcome']}: {r.get('mutation', '')}; fast "
                         f"{', '.join(r['fast_rules']) or 'DRC ok'} / "
                         f"{'LVS ok' if r['fast_lvs'] else '; '.join(r['fast_lvs_errors'][:2])}; gt "
                         f"{', '.join(r['gt_rules']) or 'DRC ok'} / {'LVS ok' if r['gt_lvs'] else 'LVS fail'}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans", nargs="*")
    ap.add_argument("--cells", default="out/cells")
    ap.add_argument("--mutants", type=int, default=0, help="mutants per seed")
    ap.add_argument("--seed", default="0", help="seed or comma-separated seeds")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--klayout-only", action="store_true", help="ground truth without Magic DRC")
    ap.add_argument("--pdk-root", default=os.environ.get("PDK_ROOT", str(Path.home() / "pdk")))
    ap.add_argument("--workdir", default="out/audit")
    ap.add_argument("--json", default=None, help="write per-layout records here")
    ap.add_argument("--md", default=None, help="write a markdown summary here")
    ap.add_argument("--keep", action="store_true", help="keep per-layout tool outputs")
    ap.add_argument("--from-json", default=None, help="only rewrite --md from saved records")
    a = ap.parse_args(argv)
    if a.from_json:
        recs = [classify(r) for r in json.loads(Path(a.from_json).read_text())]
        Path(a.md).write_text(summarize(recs))
        return 0

    plans = plan_files(a.plans or ["out/plans", str(EXAMPLES / "aoi21.plan"), str(EXAMPLES / "half_adder.plan")])
    cells = Path(a.cells)
    base = []
    for p in plans:
        nl = netlist_for(p, cells)
        base.append((p.stem if p.parent != EXAMPLES else f"ex_{p.stem}", render(parse(p.read_text()), nl), nl))
    if a.mutants:
        clean = [b for b in base if check(b[1], b[2]).ok]
        jobs = []
        for seed in (int(s) for s in a.seed.split(",")):
            jobs += mutant_jobs(clean, a.mutants, seed)
        print(f"{len(jobs)} mutants from {len(clean)} fast-clean base layouts (seeds {a.seed})")
    else:
        jobs = [(k, c, nl, "", "none") for k, c, nl in base]
    wd = Path(a.workdir).resolve()

    def one(j):
        key, cell, nl, desc, op = j
        r = audit_one(key, cell, nl, wd, Path(a.pdk_root), not a.klayout_only) | {"mutation": desc, "op": op}
        if not a.keep and r["outcome"] == "agree":
            d = wd / key
            for f in d.glob("*"):
                f.unlink()
            d.rmdir()
        return r

    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        recs = list(ex.map(one, jobs))
    dt = time.time() - t0

    outcomes = Counter(r["outcome"] for r in recs)
    ms = sorted(r["ms"] for r in recs)
    print(f"audited {len(recs)} layouts in {dt:.0f}s; fast check median {ms[len(ms) // 2]:.2f} ms, "
          f"p95 {ms[min(len(ms) - 1, int(0.95 * len(ms)))]:.2f} ms")
    for k, n in sorted(outcomes.items()):
        print(f"  {k:12s} {n}")
    print(f"  DRC agree {sum(r['drc_agree'] for r in recs)}/{len(recs)}; "
          f"LVS agree {sum(r['lvs_agree'] for r in recs)}/{len(recs)}")
    for r in recs:
        if r["outcome"] != "agree" or not r["drc_agree"] or not r["lvs_agree"]:
            print(f"- {r['key']}: {r['outcome']}  [{r['op']}] {r['mutation']}")
            print(f"    fast: DRC {', '.join(r['fast_rules']) or 'ok'}; "
                  f"LVS {'ok' if r['fast_lvs'] else '; '.join(r['fast_lvs_errors'][:3])}")
            gd = "no report" if r["gt_drc"] is None else ", ".join(
                f"{k} x{v}" for k, v in sorted(Counter(r["gt_drc"]).items())) or "ok"
            md = "no result" if r["gt_magic"] is None else ", ".join(
                f"{rule_of(k)} x{v}" for k, v in sorted(Counter(r["gt_magic"]).items())) or "ok"
            print(f"    gt:   KLayout {gd}; Magic {md}; LVS {'ok' if r['gt_lvs'] else ' | '.join(r['gt_lvs_lines'][-2:])}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(recs, indent=1))
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(summarize(recs))
    return 1 if outcomes.get("false pass", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
