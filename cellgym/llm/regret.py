"""Where does a sampled placement lose columns?  Exact break-regret per step.

    python -m cellgym.llm.regret --cells out/distill1/cells.json --samples out/distill1_greedy/step05v2s/samples.jsonl \
        [--limit 1000] [--jobs 6] [--out regret.json]

For a cell, the fewest diffusion breaks still reachable from any placer state is computed exactly (memoised
search over the same moves as fmt.Placer). Replaying a placement, the regret of a step is how much that
number rises (plus 1 if the step is itself a break). Regrets sum to the placement's breaks minus the cell's
minimum. Steps are typed:
    start   first sd column                      gate    g column continuing the row
    sd      sd column closing a gate column      break   sd column directly after an sd column
A `break` step with regret 0 was forced (no continuation could do better); regret on a `gate` or `sd` step is
a lookahead error (the row continued, but into a state that needs more breaks later).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

INF = 99


class Solver:
    def __init__(self, nl: dict):
        self.dev = {"n": [], "p": []}
        for d in nl["devices"]:
            self.dev["n" if d["type"] == "nmos" else "p"].append((d["gate"], d["drain"], d["source"]))
        self.full = tuple((1 << len(self.dev[k])) - 1 for k in "np")
        self.cont = lru_cache(maxsize=None)(self._cont)
        self.f = lru_cache(maxsize=None)(self._f)
        self.g = lru_cache(maxsize=None)(self._g)

    def _side(self, k, mask, at):
        """{gate: {(device bit, other terminal)}} for unplaced devices of row k touching net `at`."""
        out = defaultdict(set)
        for i, (g, a, b) in enumerate(self.dev[k]):
            if not mask >> i & 1 and at in (a, b):
                out[g].add((1 << i, b if a == at else a))
        return out

    def _cont(self, mn, mp, fn, fp):
        """{gate: {next state}} for continuing moves."""
        sn, sp = self._side("n", mn, fn), self._side("p", mp, fp)
        return {g: frozenset((mn | bn, mp | bp, a, b) for bn, a in sn[g] for bp, b in sp[g])
                for g in sn.keys() & sp.keys()}

    def terms(self, k, mask):
        return sorted({x for i, (g, a, b) in enumerate(self.dev[k]) if not mask >> i & 1 for x in (a, b)})

    def _g(self, mn, mp, fn, fp):
        """Fewest further breaks if the next column must be a gate (INF if none fits)."""
        if (mn, mp) == self.full:
            return 0
        return min((self.f(*s) for ss in self.cont(mn, mp, fn, fp).values() for s in ss), default=INF)

    def _f(self, mn, mp, fn, fp):
        """Fewest further breaks from the state after an sd column (fn, fp)."""
        if (mn, mp) == self.full:
            return 0
        return min(self.g(mn, mp, fn, fp), 1 + self.restart(mn, mp))

    def restart(self, mn, mp):
        return min((self.g(mn, mp, a, b) for a in self.terms("n", mn) for b in self.terms("p", mp)), default=INF)

    def replay(self, cols: list) -> list[tuple[str, int]]:
        """[(step type, regret)] along a complete placement."""
        out = []
        mn = mp = 0
        st = None
        for i, c in enumerate(cols):
            if c[0] == "g":
                before = self.f(*st)
                nxt = self.cont(*st)[c[1]]
                out.append(("gate", min(self.f(*s) for s in nxt) - before))
                pend = nxt
            elif i and cols[i - 1][0] == "g":
                cand = [s for s in pend if s[2:] == (c[1], c[2])]
                before = min(self.f(*s) for s in pend)
                best = min(cand, key=lambda s: self.f(*s))
                out.append(("sd", self.f(*best) - before))
                st = best
                mn, mp = st[0], st[1]
            elif i == 0:
                st = (0, 0, c[1], c[2])
                out.append(("start", self.f(*st) - self.restart(0, 0)))
            else:
                before = self.f(*st)
                st = (mn, mp, c[1], c[2])
                out.append(("break", 1 + self.f(*st) - before))
        return out


def break_lower_bound(nl: dict) -> int:
    """A lower bound on diffusion breaks without search: a placement with b breaks covers each row's diffusion
    graph (nets as nodes, devices as edges) with b + 1 trails, and a connected component with k odd-degree
    nets needs at least max(1, k / 2) trails. Not tight: it ignores that both rows must share the gate order."""
    need = 0
    for kind in ("nmos", "pmos"):
        deg, par = Counter(), {}

        def find(x):
            while par.setdefault(x, x) != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for d in nl["devices"]:
            if d["type"] == kind:
                a, b = d["drain"], d["source"]
                deg[a] += 1
                deg[b] += 1
                par[find(a)] = find(b)
        odd = Counter(find(x) for x, k in deg.items() if k % 2)
        need = max(need, sum(max(1, odd[r] // 2) for r in {find(x) for x in deg}))
    return max(0, need - 1)


def work(job):
    nl, placements = job
    s = Solver(nl)
    opt = s.restart(0, 0)
    return opt, [s.replay([tuple(c) for c in cols]) for cols in placements]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True, help="cells.json written by `distill sample`")
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N cells (sorted by id) only")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    cells = json.loads(a.cells.read_text())["cells"]
    by = defaultdict(list)
    for line in a.samples.read_text().splitlines():
        r = json.loads(line)
        if r["status"] == "done":
            by[r["cell"]].append(r["cols"])
    ids = sorted(by)[: a.limit or None]
    with Pool(a.jobs) as pool:
        res = pool.map(work, [(cells[i]["netlist"], by[i]) for i in ids], chunksize=8)
    n = 0
    reg = Counter()          # total regret by step type
    first = Counter()        # type of the first step with regret
    cnt = Counter()          # steps by type
    wrong = Counter()        # steps with regret > 0 by type
    forced = 0               # break steps with regret 0
    excess = 0
    pos = []
    for cid, (opt, reps) in zip(ids, res):
        for rep in reps:
            n += 1
            tot = sum(r for _, r in rep)
            excess += tot
            for t, r in rep:
                cnt[t] += 1
                reg[t] += r
                wrong[t] += r > 0
                forced += t == "break" and r == 0
            bad = [(j, t) for j, (t, r) in enumerate(rep) if r > 0]
            if bad:
                first[bad[0][1]] += 1
                pos.append(bad[0][0] / len(rep))
    out = {"cells": len(ids), "placements": n, "mean_excess_breaks": round(excess / max(1, n), 3),
           "placements_with_regret": sum(first.values()),
           "regret_share_by_type": {t: round(reg[t] / max(1, excess), 3) for t in ("start", "gate", "sd", "break")},
           "first_regret_type_share": {t: round(first[t] / max(1, sum(first.values())), 3) for t in ("start", "gate", "sd", "break")},
           "error_rate_per_step": {t: round(wrong[t] / max(1, cnt[t]), 4) for t in ("start", "gate", "sd", "break")},
           "steps": dict(cnt), "forced_breaks": forced,
           "mean_position_of_first_regret": round(sum(pos) / max(1, len(pos)), 3)}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
