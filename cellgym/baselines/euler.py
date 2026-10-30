"""Euler-path baseline plan generator (M3 audit corpus, M4 baseline).

    python -m cellgym.baselines.euler --cells out/cells --out out/plans [--verify N] [--jobs J]

Placement (Uehara & vanCleemput style, exact): every gate column pairs one
NMOS and one PMOS device with the same gate input. A memoized search over the
remaining devices and each row's frontier net finds the fewest diffusion
breaks (extra sd columns), then the fewest row breaks; a break column carries
the new left terminals, so the row that does not break gets one island of two
sd columns. Gate columns always carry the shared input on both rows.

Contacts, routing and the CLI are shared with the naive baseline
(cellgym.baselines.common).
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass

from .common import BaselineError, route_best, run_cli

MAX_PLACEMENTS = 4      # optimal placements tried per cell

EulerError = BaselineError     # backwards-compatible name


@dataclass
class EulerResult:
    text: str
    breaks: int            # extra sd columns (diffusion breaks)
    row_breaks: int        # row-level breaks (a break column can break both rows)
    near_misses: int
    width_um: float
    sites: int
    capped: bool           # some routing search hit NODE_CAP
    placements: int        # optimal placements tried


# -- placement -------------------------------------------------------------------

def _classes(devs: list[dict]) -> tuple[list, tuple[int, ...]]:
    sig = Counter((d["gate"], tuple(sorted((d["drain"], d["source"])))) for d in devs)
    keys = sorted(sig)
    return keys, tuple(sig[k] for k in keys)


def _dec(counts: tuple[int, ...], i: int) -> tuple[int, ...]:
    return counts[:i] + (counts[i] - 1,) + counts[i + 1:]


def _orients(terms: tuple[str, str]):
    a, b = terms
    return ((a, b), (b, a)) if a != b else ((a, b),)


class _Placer:
    def __init__(self, netlist: dict):
        n = [d for d in netlist["devices"] if d["type"] == "nmos"]
        p = [d for d in netlist["devices"] if d["type"] == "pmos"]
        if Counter(d["gate"] for d in n) != Counter(d["gate"] for d in p):
            raise EulerError("unpaired", "NMOS and PMOS gate inputs differ; no complementary pairing")
        self.nk, n0 = _classes(n)
        self.pk, p0 = _classes(p)
        self.start = (n0, p0, None, None)
        self.memo: dict = {}

    def moves(self, state):
        nc, pc, fn, fp = state
        for i, (g, nt) in enumerate(self.nk):
            if not nc[i]:
                continue
            for j, (g2, pt) in enumerate(self.pk):
                if g2 != g or not pc[j]:
                    continue
                for on in _orients(nt):
                    for op in _orients(pt):
                        nb = fn is not None and on[0] != fn
                        pb = fp is not None and op[0] != fp
                        yield (g, on, op), (int(nb or pb), nb + pb), (_dec(nc, i), _dec(pc, j), on[1], op[1])

    def cost(self, state) -> tuple[int, int]:
        if state not in self.memo:
            if not any(state[0]):
                self.memo[state] = (0, 0)
            else:
                best = None
                for _, step, nxt in self.moves(state):
                    rest = self.cost(nxt)
                    c = (step[0] + rest[0], step[1] + rest[1])
                    if best is None or c < best:
                        best = c
                self.memo[state] = best
        return self.memo[state]

    def sequences(self, limit: int) -> list[list]:
        out: list[list] = []

        def dfs(state, seq):
            if not any(state[0]):
                out.append(list(seq))
                return
            target = self.cost(state)
            for mv, step, nxt in self.moves(state):
                rest = self.cost(nxt)
                if (step[0] + rest[0], step[1] + rest[1]) == target:
                    seq.append(mv)
                    dfs(nxt, seq)
                    seq.pop()
                    if len(out) >= limit:
                        return

        dfs(self.start, [])
        return out


def _columns(seq: list) -> list[tuple]:
    """("sd", n, p) / ("g", input) columns for a placement sequence."""
    cols: list[tuple] = []
    fn = fp = None
    for g, (a, b), (c, d) in seq:
        if fn is None or a != fn or c != fp:
            cols.append(("sd", a, c))     # leading sd, or the break column
        cols.append(("g", g))
        cols.append(("sd", b, d))
        fn, fp = b, d
    return cols


# -- driver --------------------------------------------------------------------------

def euler_solve(netlist: dict) -> EulerResult:
    """Best Euler plan for a netlist; raises EulerError if none is found."""
    placer = _Placer(netlist)
    seqs = placer.sequences(MAX_PLACEMENTS)
    breaks, row_breaks = placer.cost(placer.start)
    r = route_best([_columns(s) for s in seqs], netlist)
    return EulerResult(r.text, breaks, row_breaks, r.near_misses, r.width_um, r.sites, r.capped, len(seqs))


def euler_plan(netlist: dict) -> str:
    """Plan text for a netlist that passes validate() and renders."""
    return euler_solve(netlist).text


def main(argv=None) -> int:
    return run_cli(__doc__, euler_solve, argv)


if __name__ == "__main__":
    sys.exit(main())
