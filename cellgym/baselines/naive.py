"""Naive baseline plan generator: no diffusion sharing.

    python -m cellgym.baselines.naive --cells out/cells --out out/plans_naive [--verify N] [--jobs J]

Placement: NMOS devices in netlist order, each paired with the first unused
PMOS device on the same gate input; every pair is its own gate column with its
own two sd columns (drain left, source right), so every pair has its own
diffusion islands in both rows. Where the next pair's left terminal equals the
previous pair's right terminal in a row (which would merge the islands), an
empty sd column (`n=- p=-`) separates them. There is exactly one placement and
no search over it.

Contacts, routing (single horizontal segment per net), pins and the best-plan
choice are the Euler baseline's (cellgym.baselines.common).
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass

from .common import BaselineError, route_best, run_cli


@dataclass
class NaiveResult:
    text: str
    breaks: int            # pairs - 1: a diffusion break between every pair
    separators: int        # empty sd columns inserted
    near_misses: int
    width_um: float
    sites: int
    capped: bool


def naive_columns(netlist: dict) -> list[tuple]:
    n = [d for d in netlist["devices"] if d["type"] == "nmos"]
    p = [d for d in netlist["devices"] if d["type"] == "pmos"]
    if Counter(d["gate"] for d in n) != Counter(d["gate"] for d in p):
        raise BaselineError("unpaired", "NMOS and PMOS gate inputs differ; no complementary pairing")
    used: set[str] = set()
    cols: list[tuple] = []
    for dn in n:
        dp = next(d for d in p if d["gate"] == dn["gate"] and d["name"] not in used)
        used.add(dp["name"])
        a, b, c, d = dn["drain"], dn["source"], dp["drain"], dp["source"]
        if cols and (cols[-1][1] == a or cols[-1][2] == c):
            cols.append(("sd", None, None))
        cols += [("sd", a, c), ("g", dn["gate"]), ("sd", b, d)]
    return cols


def naive_solve(netlist: dict) -> NaiveResult:
    """Naive plan for a netlist; raises BaselineError if routing fails."""
    cols = naive_columns(netlist)
    r = route_best([cols], netlist)
    pairs = sum(c[0] == "g" for c in cols)
    seps = sum(c[0] == "sd" and c[1] is None for c in cols)
    return NaiveResult(r.text, pairs - 1, seps, r.near_misses, r.width_um, r.sites, r.capped)


def naive_plan(netlist: dict) -> str:
    return naive_solve(netlist).text


def main(argv=None) -> int:
    return run_cli(__doc__, naive_solve, argv, default_out="out/plans_naive")


if __name__ == "__main__":
    sys.exit(main())
