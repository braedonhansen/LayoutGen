"""Channel congestion of a placement (column list), size-free.

A net needs the channel between the rows when it has terminals in more than one column and either
touches both rows (nmos and pmos diffusion) or drives a gate column (gate pads sit in the channel).
Rails are excluded. Density at a column = number of channel nets whose column span covers it;
`density` returns the max. The cell has 3 channel tracks on 2 layers (li1, met1), and contact
stubs and gate pads use the same tracks, so density near or above 4 is where routing starts failing.
"""

from __future__ import annotations

RAILS = {"VSS", "VDD"}


def spans(cols) -> dict[str, tuple[int, int]]:
    at: dict[str, list] = {}
    for i, c in enumerate(cols):
        if c[0] == "g":
            at.setdefault(c[1], []).append((i, "g"))
        else:
            at.setdefault(c[1], []).append((i, "n"))
            at.setdefault(c[2], []).append((i, "p"))
    out = {}
    for net, t in at.items():
        if net in RAILS:
            continue
        xs = sorted({i for i, _ in t})
        kinds = {k for _, k in t}
        if len(xs) > 1 and ("g" in kinds or {"n", "p"} <= kinds):
            out[net] = (xs[0], xs[-1])
    return out


def profile(cols) -> list[int]:
    d = [0] * len(cols)
    for a, b in spans(cols).values():
        for x in range(a, b + 1):
            d[x] += 1
    return d


def density(cols) -> int:
    p = profile(cols)
    return max(p) if p else 0
