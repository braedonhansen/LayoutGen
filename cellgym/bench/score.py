"""Score a directory of plans against the benchmark splits.

    python -m cellgym.bench.score <plans_dir> --name <label> [--split all|train|val|test_id|test_ood]
                                  [--confirm] [--jobs J] [--cells out/cells] [--out out/bench/<label>]

Plans are `<plans_dir>/<cell id>.plan`; a missing plan fails. Per split:

- pass@1: the plan parses, validates, renders and the fast checker is clean
- width in sites and um (mean / median over passing plans), mean diffusion
  breaks (passing plans), mean W / W_euler (passing plans)
- mean GRPO reward over all cells: 1 + (W_euler - W) / W_euler for a passing
  plan, 0 for a failing one; W_euler from bench/euler_widths_v0.json
- mean fast-checker time (ms)

--confirm re-runs full ground truth (KLayout DRC + Magic DRC + Netgen LVS,
cellgym.groundtruth.verify) on every fast-passing plan, in parallel (--jobs),
and reports the same metrics with pass = fast pass AND ground truth clean.
Writes score.json and score.md into --out and prints the table.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import reference, splits
from .common import EULER_WIDTHS_FILE, SPLITS, SPLITS_FILE, load_cells, sha256
from .evaluate import evaluate, read_plan


def reward(width_nm: int | None, w_euler_nm: int, ok: bool) -> float:
    if not ok or not width_nm:
        return 0.0
    return 1.0 + (w_euler_nm - width_nm) / w_euler_nm


def summarize(rows: list[dict], key: str) -> dict:
    """Metrics over rows; key = "pass" (fast) or "confirmed" (ground truth)."""
    n = len(rows)
    ok = [r for r in rows if r[key]]
    mean = lambda xs: statistics.mean(xs) if xs else None
    med = lambda xs: statistics.median(xs) if xs else None
    ms = [r["check_ms"] for r in rows if r["check_ms"] is not None]
    return {
        "n": n,
        "passed": len(ok),
        "pass_at_1": len(ok) / n if n else None,
        "sites_mean": mean([r["sites"] for r in ok]),
        "sites_median": med([r["sites"] for r in ok]),
        "width_um_mean": mean([r["width_nm"] / 1000 for r in ok]),
        "width_um_median": med([r["width_nm"] / 1000 for r in ok]),
        "breaks_mean": mean([r["breaks"] for r in ok]),
        "w_over_euler_mean": mean([r["width_nm"] / r["w_euler_nm"] for r in ok]),
        "reward_mean": sum(reward(r["width_nm"], r["w_euler_nm"], r[key]) for r in rows) / n if n else None,
        "check_ms_mean": mean(ms),
    }


def _ground_truth(rows: list[dict], plans_dir: Path, cells: dict, workdir: Path, pdk_root: Path, jobs: int):
    from cellgym.dsl.plan import parse
    from cellgym.groundtruth.verify import verify_rendered
    from cellgym.render.render import render

    def one(r):
        cid = r["id"]
        try:
            cell = render(parse(read_plan(plans_dir, cid)), cells[cid])
            g = verify_rendered(cell, workdir / cid, pdk_root)
            return cid, {"ok": g.ok, "drc": g.drc_items, "magic": g.magic_items, "lvs_ok": g.lvs_ok,
                         "lvs_tail": [] if g.lvs_ok else [x.strip() for x in g.lvs_lines[-3:]]}
        except Exception as e:  # report, don't abort the batch
            return cid, {"ok": False, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(jobs) as ex:
        return dict(ex.map(one, [r for r in rows if r["pass"]]))


def score(plans_dir: Path, name: str, which: str = "all", confirm: bool = False, jobs: int = 4,
          cells_dir: Path = Path("out/cells"), splits_file: Path = SPLITS_FILE,
          euler_file: Path = EULER_WIDTHS_FILE, workdir: Path | None = None,
          pdk_root: Path | None = None) -> dict:
    sdoc = splits.load(splits_file)
    edoc = reference.load(euler_file)
    cells = load_cells(cells_dir)
    split = splits.split_of(sdoc)
    expect = {r["id"]: r["netlist_sha256"] for r in sdoc["cells"]}
    stale = [c for c in expect if c not in cells or sha256(cells[c]) != expect[c]]
    if stale:
        raise ValueError(f"{len(stale)} netlists differ from {splits_file} (e.g. {stale[0]}); "
                         "regenerate out/cells with the pinned M1 generator")
    chosen = SPLITS if which == "all" else (which,)
    ids = sorted(c for c in expect if split[c] in chosen)
    rows = []
    t0 = time.time()
    for cid in ids:
        ev = evaluate(read_plan(plans_dir, cid), cells[cid])
        rows.append({"id": cid, "split": split[cid], "stage": ev.stage, "detail": ev.detail,
                     "pass": ev.ok, "width_nm": ev.width_nm, "sites": ev.sites, "breaks": ev.breaks,
                     "check_ms": ev.check_ms, "w_euler_nm": edoc["cells"][cid]["width_nm"]})
    fast_s = time.time() - t0
    gt_s = None
    if confirm:
        t1 = time.time()
        pdk_root = pdk_root or Path(os.environ.get("PDK_ROOT", str(Path.home() / "pdk")))
        gt = _ground_truth(rows, plans_dir, cells, (workdir or Path("out/bench") / name / "gt").resolve(),
                           pdk_root, jobs)
        for r in rows:
            r["gt"] = gt.get(r["id"])
            r["confirmed"] = bool(r["pass"] and r["gt"] and r["gt"]["ok"])
        gt_s = time.time() - t1
    groups = {s: [r for r in rows if r["split"] == s] for s in chosen}
    if which == "all":
        groups["all"] = rows
    result = {
        "name": name, "plans_dir": str(plans_dir), "split_filter": which,
        "splits_sha256": sdoc["sha256"], "euler_widths_sha256": edoc["sha256"],
        "confirmed": confirm, "seconds": {"fast": round(fast_s, 2), "ground_truth": gt_s and round(gt_s, 1)},
        "fast": {g: summarize(rs, "pass") for g, rs in groups.items()},
        "failures": {r["id"]: f"{r['stage']}: {r['detail']}" for r in rows if not r["pass"]},
        "cells": rows,
    }
    if confirm:
        result["ground_truth"] = {g: summarize(rs, "confirmed") for g, rs in groups.items()}
        result["gt_failures"] = {r["id"]: r["gt"] for r in rows if r["pass"] and not r["confirmed"]}
    return result


def _f(x, fmt):
    return "-" if x is None else format(x, fmt)


def markdown(result: dict) -> str:
    conf = result["confirmed"]
    head = f"### {result['name']}  ({'ground truth confirmed' if conf else 'fast checker only'})\n\n"
    cols = ["split", "n", "pass@1 fast"] + (["pass@1 GT"] if conf else []) + [
        "sites mean", "sites median", "W um mean", "W/W_euler", "reward", "breaks", "check ms"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for g, m in result["fast"].items():
        h = result["ground_truth"][g] if conf else m
        row = [g, str(m["n"]), _f(m["pass_at_1"], ".3f")] + ([_f(h["pass_at_1"], ".3f")] if conf else []) + [
            _f(h["sites_mean"], ".2f"), _f(h["sites_median"], "g"), _f(h["width_um_mean"], ".3f"),
            _f(h["w_over_euler_mean"], ".3f"), _f(h["reward_mean"], ".3f"), _f(h["breaks_mean"], ".2f"),
            _f(m["check_ms_mean"], ".1f")]
        lines.append("| " + " | ".join(row) + " |")
    note = ("\nWidth, ratio, reward and breaks use ground-truth-confirmed passes.\n" if conf else "\n")
    fails = len(result["failures"]) + len(result.get("gt_failures", {}))
    return head + "\n".join(lines) + "\n" + note + (f"\nFailures: {fails} (see score.json).\n" if fails else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans_dir", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="all", choices=("all",) + SPLITS)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cells", type=Path, default=Path("out/cells"))
    ap.add_argument("--splits", type=Path, default=SPLITS_FILE)
    ap.add_argument("--euler-widths", type=Path, default=EULER_WIDTHS_FILE)
    ap.add_argument("--pdk-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or Path("out/bench") / a.name
    res = score(a.plans_dir, a.name, a.split, a.confirm, a.jobs, a.cells, a.splits, a.euler_widths,
                out / "gt", a.pdk_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps(res, indent=1) + "\n", newline="\n")
    md = markdown(res)
    (out / "score.md").write_text(md, newline="\n")
    print(md)
    for cid, why in list(res["failures"].items())[:10]:
        print(f"  FAIL {cid}: {why}")
    for cid, g in list(res.get("gt_failures", {}).items())[:10]:
        print(f"  GT FAIL {cid}: {g}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
