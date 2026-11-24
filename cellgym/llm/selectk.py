"""Router-free selection against k: how good is the best of k samples if only the fewest-column ones are routed?

    python -m cellgym.llm.selectk [--draws 200] [--out out/tradeoff/selectk.md]

Uses the saved samples (8 per cell) and the routed curves cached by cellgym.llm.tradeoff
(out/tradeoff/placements.jsonl); nothing is routed here. Per set and k = 1..8, averaged over random
k-subsets of a cell's 8 samples (all 8 at k = 8) and then over cells, area / Euler area of:

  mean      a random one of the k (no selection)
  fewcol    the fewest-column samples, ties averaged (no routing at all)
  fewroute  the best of the fewest-column samples (route only those)
  best      the best of all k (route everything)

plus `routed` = distinct placements fewroute has to route, against the distinct placements among the k.
Cells are those with an Euler area reference whose 8 samples are all in the cache (the 6-10-input sets
were cut to 60 cells there).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from cellgym.llm import arearef
from cellgym.llm import tradeoff as T

GROUPS = [("val / test_id", ("val", "test_id")),
          ("ood6-10", ("ood6_dev", "ood7_dev", "ood8_dev", "ood9_all", "ood10_all")),
          ("stack 5 / 6", ("oodstack5_dev", "oodstack6_all")),
          ("10-12 per side", ("ooddev10_dev", "ooddev11_dev", "ooddev12_all"))]
AXIS = {"val / test_id": "in distribution", "ood6-10": "5 -> 6-10 inputs", "stack 5 / 6": "stack 4 -> 5 / 6",
        "10-12 per side": "9 -> 10-12 per side"}


def pick_stats(samples, draws: int, rng: random.Random, kmax: int = 8):
    """samples: [(ncols, area ratio, hash)] of one cell. {k: (mean, fewcol, fewroute, best, routed, distinct)}."""
    out = {}
    for k in range(1, min(kmax, len(samples)) + 1):
        subs = [samples] if k == len(samples) else [rng.sample(samples, k) for _ in range(draws)]
        acc = [0.0] * 6
        for sub in subs:
            cmin = min(s[0] for s in sub)
            few = [s for s in sub if s[0] == cmin]
            vals = (sum(s[1] for s in sub) / k, sum(s[1] for s in few) / len(few), min(s[1] for s in few),
                    min(s[1] for s in sub), len({s[2] for s in few}), len({s[2] for s in sub}))
            acc = [a + v for a, v in zip(acc, vals)]
        out[k] = tuple(a / len(subs) for a in acc)
    return out


def collect(rows, cells, refs):
    """{(set, cell): [(ncols, area / ref, hash)]} for cells whose samples are all routed in the cache."""
    by = defaultdict(list)
    for s, c, h, cols in rows:
        by[(s, c)].append((h, len(cols)))
    out = {}
    for (s, c), hs in by.items():
        ref = refs.get(s, {}).get(c)
        ps = cells.get((s, c), {})
        if ref and all(h in ps for h, _ in hs):
            out[(s, c)] = [(n, ps[h].area / ref, h) for h, n in hs]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", type=Path, default=T.OUT / "selectk.md")
    ap.add_argument("--arearef", type=Path, default=Path("out/arearef"))
    a = ap.parse_args(argv)
    cells, _ = T.to_cells(T.read_cache(T.OUT / "placements.jsonl"))
    samples = T.read_samples(T.SAMPLES[:2])
    sets = [s for _, ss in GROUPS for s in ss]
    refs = {s: arearef.areas_nm2(arearef.load(s, a.arearef)) for s in sets}
    lines = ["# Router-free selection against k (area / Euler area; `python -m cellgym.llm.selectk`)", ""]
    for label, rows in samples.items():
        per = collect(rows, cells, refs)
        rng = random.Random(0)
        stats = {key: pick_stats(v, a.draws, rng) for key, v in per.items()}
        lines += [f"## {label} ({len(per)} cells)", "",
                  "| group | trained up to -> tested | cells | k | mean | fewcol (0 routed) | fewroute | best-of-k | routed / distinct |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for g, ss in GROUPS:
            keys = [k for k in stats if k[0] in ss]
            if not keys:
                continue
            for k in range(1, 9):
                v = [stats[key][k] for key in keys if k in stats[key]]
                m = [sum(x[i] for x in v) / len(v) for i in range(6)]
                lines.append(f"| {g} | {AXIS[g]} | {len(v)} | {k} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} | "
                             f"{m[4]:.2f} / {m[5]:.2f} |")
        lines.append("")
    text = "\n".join(lines)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
