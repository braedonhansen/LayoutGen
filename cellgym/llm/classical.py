"""A search-free classical placer, as the scalable reference where the Euler search does not finish.

    python -m cellgym.llm.classical SAMPLES.jsonl|none [--k 32] [--extra name=dir:part ...] [--tries 32 1000]
                                    [--secs 10] [--rule smart|plain] [--jobs 8] [--out FILE]

Randomised greedy trail building on the same state machine as regret.Solver (two rows, one gate order):
after a diffusion column, take a gate that continues both rows if one exists, else break and restart.
rule "plain": uniform choice among continuing moves and among restart columns.
rule "smart": among continuing moves prefer those whose next state can itself continue; on a restart prefer
columns on nets with an odd number of unplaced device ends in their row (the ends of an Euler trail) from
which a gate can follow. No lookahead beyond one step, no memo.

Per set, for the cells of SAMPLES: mean breaks of the model's fewest-column sample among the first K, and of
the best of N tries of this placer for each N in --tries, plus the best found within --secs seconds per cell
(0 = skip); cells where the model has fewer / the same / more breaks. Breaks are columns: columns =
2 * devices per side + 1 + breaks. With SAMPLES "none" there is no model: every cell of the --extra sets is
placed and the "model" column is regret.break_lower_bound (a floor, so "fewer" there means the floor is not
reached).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from cellgym.llm.bound import breaks
from cellgym.llm.infer import load_sets
from cellgym.llm.regret import Solver


class Greedy:
    def __init__(self, nl: dict, rule: str = "smart"):
        self.s = Solver(nl)
        self.rule = rule

    def _odd(self, k: str, mask: int) -> set:
        deg = Counter()
        for i, (g, a, b) in enumerate(self.s.dev[k]):
            if not mask >> i & 1 and a != b:
                deg[a] += 1
                deg[b] += 1
        return {x for x, d in deg.items() if d % 2}

    def _moves(self, st):
        return [x for ss in self.s._cont(*st).values() for x in ss]

    def one(self, rng: random.Random, path: list | None = None) -> int:
        """Breaks of one randomised greedy placement; with `path`, its columns are appended to it."""
        s, smart = self.s, self.rule == "smart"
        mn = mp = 0
        st, br = None, -1

        def go(prev, nxt):
            if path is not None:
                path.append(("g", next(g for g, ss in s._cont(*prev).items() if nxt in ss)))
                path.append(("sd", nxt[2], nxt[3]))
            return nxt
        while (mn, mp) != s.full:
            mv = self._moves(st) if st else []
            if mv:
                if smart and len(mv) > 1:
                    good = [x for x in mv if (x[0], x[1]) == s.full or self._moves(x)]
                    mv = good or mv
                st = go(st, rng.choice(mv))
            else:
                br += 1
                starts = [(mn, mp, a, b) for a in s.terms("n", mn) for b in s.terms("p", mp)]
                if smart:
                    live = [x for x in starts if self._moves(x)]
                    if live:
                        on, op = self._odd("n", mn), self._odd("p", mp)
                        top = max((x[2] in on) + (x[3] in op) for x in live)
                        starts = [x for x in live if (x[2] in on) + (x[3] in op) == top]
                st = rng.choice(starts)
                if path is not None:
                    path.append(("sd", st[2], st[3]))
                nx = self._moves(st)
                if not nx:      # a column no gate can follow: the next loop turn breaks again
                    continue
                if smart and len(nx) > 1:
                    good = [x for x in nx if (x[0], x[1]) == s.full or self._moves(x)]
                    nx = good or nx
                st = go(st, rng.choice(nx))
            mn, mp = st[0], st[1]
        return max(br, 0)


def _cell(job):
    """(set, cell, {tries: best breaks}, best within secs or None, tries done within secs, seconds for max tries)"""
    s, c, nl, tries, secs, rule = job
    g, rng = Greedy(nl, rule), random.Random(zlib.crc32(c.encode()))
    out, best, t0 = {}, 10 ** 6, time.time()
    for i in range(1, max(tries) + 1):
        best = min(best, g.one(rng))
        if i in tries:
            out[i] = best
    dt = time.time() - t0
    tb, n = None, 0
    if secs:
        tb, t0 = best, time.time()
        n = max(tries)
        while time.time() - t0 < secs and tb > 0:
            tb = min(tb, g.one(rng))
            n += 1
    return s, c, out, tb, n, dt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=Path)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--tries", type=int, nargs="+", default=[32, 1000])
    ap.add_argument("--secs", type=float, default=0)
    ap.add_argument("--rule", default="smart", choices=["smart", "plain"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    from cellgym.llm.regret import break_lower_bound
    none = str(a.samples) == "none"         # no model: every cell of the --extra sets, model column = lower bound
    rows = [] if none else [json.loads(l) for l in a.samples.read_text().splitlines()]
    extra = [tuple(x.split("=", 1)) for x in a.extra]
    sets = load_sets([n for n, _ in extra] if none else sorted({r["set"] for r in rows}), extra)
    model = defaultdict(dict)
    if none:
        for s in sets:
            model[s] = {c: break_lower_bound(nl) for c, nl in sets[s][0].items()}
    for r in rows:
        if r["status"] == "done" and r.get("sample", 0) < a.k:
            d = model[r["set"]]
            d[r["cell"]] = min(d.get(r["cell"], 10 ** 6), breaks(r["cols"]))
    jobs = [(s, c, sets[s][0][c], tuple(a.tries), a.secs, a.rule) for s in sets for c in model.get(s, {})]
    res = defaultdict(dict)
    with ProcessPoolExecutor(a.jobs) as ex:
        for s, c, out, tb, n, dt in ex.map(_cell, jobs, chunksize=2):
            res[s][c] = (out, tb, n, dt)
    cols = [f"classical best of {n}" for n in a.tries] + ([f"classical {a.secs:g} s"] if a.secs else [])
    print(f"| set | cells | model best of {a.k} | " + " | ".join(cols) + " | model fewer / same / more breaks than the last column |"
          " classical s per cell for " + str(max(a.tries)) + " |")
    print("|" + "---|" * (5 + len(cols)))
    summary = {}
    for s in sets:
        m, r = model.get(s, {}), res.get(s, {})
        if not m:
            continue
        cl = {n: mean(r[c][0][n] for c in m) for n in a.tries}
        last = {c: (r[c][1] if a.secs else r[c][0][max(a.tries)]) for c in m}
        w = sum(m[c] < last[c] for c in m), sum(m[c] == last[c] for c in m), sum(m[c] > last[c] for c in m)
        summary[s] = {"cells": len(m), "model": round(mean(m.values()), 3), "classical": {str(n): round(v, 3) for n, v in cl.items()},
                      "classical_secs": round(mean(last.values()), 3) if a.secs else None,
                      "tries_in_secs": round(mean(r[c][2] for c in m)) if a.secs else None,
                      "model_fewer_same_more": w, "secs_for_max_tries": round(mean(r[c][3] for c in m), 3)}
        print(f"| {s} | {len(m)} | {mean(m.values()):.2f} | " + " | ".join(f"{cl[n]:.2f}" for n in a.tries)
              + (f" | {mean(last.values()):.2f}" if a.secs else "") + f" | {w[0]} / {w[1]} / {w[2]} | {summary[s]['secs_for_max_tries']:.2f} |")
    if a.out:
        a.out.write_text(json.dumps({"samples": str(a.samples), "k": a.k, "rule": a.rule, "secs": a.secs, "sets": summary}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
