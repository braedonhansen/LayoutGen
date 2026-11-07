"""Combine score.json files into the results table (bench/results_v0.md).

    python -m cellgym.bench.results out/bench/naive/score.json out/bench/euler/score.json
                                    [--out bench/results_v0.md]

Uses ground-truth-confirmed metrics where a run has them (score --confirm).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import SPLITS


def _f(x, fmt):
    return "-" if x is None else format(x, fmt)


def table(results: list[dict]) -> str:
    cols = ["split", "baseline", "n", "pass@1 (fast)", "pass@1 (GT)", "mean sites", "mean W um",
            "mean W/W_euler", "mean reward", "mean breaks"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for g in SPLITS + ("all",):
        for r in results:
            if g not in r["fast"]:
                continue
            f = r["fast"][g]
            h = r["ground_truth"][g] if r["confirmed"] else f
            lines.append("| " + " | ".join([
                g, r["name"], str(f["n"]), _f(f["pass_at_1"], ".3f"),
                _f(h["pass_at_1"], ".3f") if r["confirmed"] else "not run",
                _f(h["sites_mean"], ".2f"), _f(h["width_um_mean"], ".3f"), _f(h["w_over_euler_mean"], ".3f"),
                _f(h["reward_mean"], ".3f"), _f(h["breaks_mean"], ".2f")]) + " |")
    return "\n".join(lines) + "\n"


def document(results: list[dict]) -> str:
    r0 = results[0]
    gt = ", ".join(f"{r['name']} {r['seconds']['ground_truth']:.0f} s" for r in results
                   if r["seconds"].get("ground_truth"))
    return "\n".join([
        "# CellGym baseline results (v0)", "",
        f"Splits `bench/splits_v0.json` (sha256 `{r0['splits_sha256'][:12]}`), Euler reference widths "
        f"`bench/euler_widths_v0.json` (sha256 `{r0['euler_widths_sha256'][:12]}`). pass@1 (fast) = the "
        "plan parses, validates, renders and the fast checker is clean; pass@1 (GT) additionally needs "
        "full ground truth (KLayout DRC + Magic DRC + Netgen LVS). Width, W/W_euler, reward and breaks "
        "are over GT-confirmed passes; reward = 1 + (W_euler - W) / W_euler for a pass, 0 for a fail, "
        "averaged over all cells of the split. Breaks = adjacent gate columns that do not share "
        "diffusion in some row.", "",
        table(results),
        "Baselines: `euler` = exact Euler-path placement (fewest diffusion breaks) plus the shared "
        "routing search; `naive` = no diffusion sharing (every NMOS/PMOS pair on its own islands), same "
        "routing. Regenerate with `python -m cellgym.bench.score <plans> --name <label> --confirm` and "
        "`python -m cellgym.bench.results`."
        + (f" Ground-truth wall time: {gt}." if gt else ""), ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("bench/results_v0.md"))
    a = ap.parse_args(argv)
    results = [json.loads(p.read_text()) for p in a.scores]
    doc = document(results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(doc, newline="\n")
    print(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
