"""Deterministic layout mutations for checker audits.

    mutate(rendered, seed, op=None) -> (mutant Rendered, description)

One perturbation per mutant, chosen by seed (or forced with `op`):

- shrink / grow / shift: one rectangle, one edge or the whole rect, x or y,
  by 5-40 nm
- delete: one rectangle of any drawn layer
- contact: delete one licon or mcon
- label: move one li/met1 text label onto another shape of its layer
- label_net: move one li/met1 label onto a shape of a different net
- label_swap: swap the positions of two same-layer labels of different nets
- neighbor: shift a poly/diff/li/met1/licon/mcon rect toward its nearest
  same-layer neighbour, leaving a gap just under spacing, touching or
  overlapping it
- gap_close: the same, but by moving only the facing edge of a li/met1 rect
- implant: move one in-cell edge of an nsdm/psdm/nwell/hvtp/npc rect by 5-200 nm

Coordinates stay on the 5 nm grid. Ops with no candidate fall back to shift.
"""

from __future__ import annotations

import random
from dataclasses import replace

LAYERS = ("nwell", "diff", "poly", "licon", "li", "mcon", "met1", "hvtp", "nsdm", "psdm", "npc")
IMPLANTS = ("nsdm", "psdm", "nwell", "hvtp", "npc")
OPS = ("shrink", "grow", "shift", "delete", "contact", "label", "label_net", "label_swap",
       "neighbor", "gap_close", "implant")
OP_WEIGHTS = (3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2)


def _where(s) -> str:
    return f"{s.layer} ({s.x0}, {s.y0}, {s.x1}, {s.y1})"


def _set(shapes, i, x0, y0, x1, y1, desc):
    s = shapes[i]
    if x1 <= x0 or y1 <= y0:
        del shapes[i]
        return desc + " (vanished)"
    shapes[i] = replace(s, x0=x0, y0=y0, x1=x1, y1=y1)
    return desc


def _neighbours(shapes, i):
    """(gap, direction) to the nearest same-layer rect in each direction whose
    projection overlaps (gap > 0: not touching)."""
    s = shapes[i]
    best = {}
    for j, o in enumerate(shapes):
        if j == i or o.layer != s.layer:
            continue
        yo = min(s.y1, o.y1) > max(s.y0, o.y0)
        xo = min(s.x1, o.x1) > max(s.x0, o.x0)
        for d, g, ok in (("right", o.x0 - s.x1, yo), ("left", s.x0 - o.x1, yo),
                         ("top", o.y0 - s.y1, xo), ("bottom", s.y0 - o.y1, xo)):
            if ok and g > 0 and (d not in best or g < best[d]):
                best[d] = g
    return sorted((g, d) for d, g in best.items())


def _target_gap(rng, g: int) -> int:
    """New gap: just under the old one, a few steps under, touching, or overlapping."""
    t = rng.choice((g - 5, g - 5 * rng.randint(1, 8), 0, -5 * rng.randint(1, 4)))
    return min(max(t, -20), g - 5)


def _toward(rng, shapes, layers, edge_only: bool):
    idx = [i for i, s in enumerate(shapes) if s.layer in layers]
    rng.shuffle(idx)
    for i in idx[:50]:
        nb = _neighbours(shapes, i)
        if not nb:
            continue
        g, d = rng.choice(nb[:2])
        if g > 1000:
            continue
        t = _target_gap(rng, g)
        m = g - t
        s = shapes[i]
        x0, y0, x1, y1 = s.x0, s.y0, s.x1, s.y1
        dx = {"right": m, "left": -m}.get(d, 0)
        dy = {"top": m, "bottom": -m}.get(d, 0)
        if edge_only:
            x1 += max(dx, 0)
            x0 += min(dx, 0)
            y1 += max(dy, 0)
            y0 += min(dy, 0)
            what = f"gap_close {_where(s)} {d} edge by {m} (gap {g} -> {t})"
        else:
            x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
            what = f"neighbor shift {_where(s)} {d} by {m} (gap {g} -> {t})"
        return _set(shapes, i, x0, y0, x1, y1, what)
    return None


def mutate(cell, seed: int, op: str | None = None):
    rng = random.Random(seed)
    if op is None:
        op = rng.choices(OPS, OP_WEIGHTS)[0]
    shapes = list(cell.shapes)
    labels = list(cell.labels)
    lab_idx = [i for i, lab in enumerate(labels) if lab.kind in ("li", "met1")]

    if op == "label":
        i = rng.choice(lab_idx)
        lab = labels[i]
        targets = [s for s in shapes if s.layer == lab.kind and (s.x0, s.y0, s.x1, s.y1) != lab.rect]
        t = rng.choice(targets)
        labels[i] = replace(lab, rect=(t.x0, t.y0, t.x1, t.y1))
        cx, cy = (t.x0 + t.x1) // 2, (t.y0 + t.y1) // 2
        return replace(cell, labels=labels), f"label {lab.net} ({lab.kind}) moved to ({cx}, {cy})"
    if op == "label_net":
        i = rng.choice(lab_idx)
        lab = labels[i]
        targets = [s for s in shapes if s.layer == lab.kind and len(s.owner) == 2
                   and s.owner[0] == "net" and s.owner[1] != lab.net]
        if targets:
            t = rng.choice(targets)
            labels[i] = replace(lab, rect=(t.x0, t.y0, t.x1, t.y1))
            return replace(cell, labels=labels), \
                f"label {lab.net} ({lab.kind}) moved onto net {t.owner[1]} {_where(t)}"
        op = "shift"
    if op == "label_swap":
        pairs = [(i, j) for i in lab_idx for j in lab_idx if i < j and labels[i].net != labels[j].net
                 and labels[i].kind == labels[j].kind]
        if pairs:
            i, j = rng.choice(pairs)
            a, b = labels[i], labels[j]
            labels[i], labels[j] = replace(a, rect=b.rect), replace(b, rect=a.rect)
            return replace(cell, labels=labels), f"label swap {a.net} <-> {b.net}"
        op = "shift"
    if op in ("neighbor", "gap_close"):
        layers = ("poly", "diff", "li", "met1", "licon", "mcon") if op == "neighbor" else ("li", "met1")
        desc = _toward(rng, shapes, layers, op == "gap_close")
        if desc is not None:
            return replace(cell, shapes=shapes), desc
        op = "shift"
    if op == "implant":
        # edges inside the cell boundary (the others only move margins outside)
        bx = next((t for t in shapes if t.layer == "boundary"), None)
        X0, Y0, X1, Y1 = (bx.x0, bx.y0, bx.x1, bx.y1) if bx else (-10**9, -10**9, 10**9, 10**9)
        edges = [(i, side) for i, t in enumerate(shapes) if t.layer in IMPLANTS
                 for side, c, lo, hi in (("left", t.x0, X0, X1), ("right", t.x1, X0, X1),
                                         ("bottom", t.y0, Y0, Y1), ("top", t.y1, Y0, Y1))
                 if lo < c < hi]
        if not edges:
            edges = [(i, rng.choice(("left", "right", "bottom", "top")))
                     for i, t in enumerate(shapes) if t.layer in IMPLANTS]
        i, side = rng.choice(edges)
        s = shapes[i]
        d = 5 * rng.randint(1, 40) * rng.choice((-1, 1))
        x0, y0, x1, y1 = s.x0, s.y0, s.x1, s.y1
        if side == "left":
            x0 += d
        elif side == "right":
            x1 += d
        elif side == "bottom":
            y0 += d
        else:
            y1 += d
        desc = _set(shapes, i, x0, y0, x1, y1, f"implant {_where(s)} {side} edge by {d}")
        return replace(cell, shapes=shapes), desc
    if op == "contact":
        idx = [i for i, s in enumerate(shapes) if s.layer in ("licon", "mcon")]
        i = rng.choice(idx)
        where = _where(shapes[i])
        del shapes[i]
        return replace(cell, shapes=shapes), f"contact delete {where}"

    idx = [i for i, s in enumerate(shapes) if s.layer in LAYERS]
    i = rng.choice(idx)
    s = shapes[i]
    where = _where(s)
    if op == "delete":
        del shapes[i]
        return replace(cell, shapes=shapes), f"delete {where}"
    d = 5 * rng.randint(1, 8)
    x0, y0, x1, y1 = s.x0, s.y0, s.x1, s.y1
    if op == "shift":
        axis = rng.choice("xy")
        d *= rng.choice((-1, 1))
        if axis == "x":
            x0, x1 = x0 + d, x1 + d
        else:
            y0, y1 = y0 + d, y1 + d
        desc = f"shift {where} by {d} in {axis}"
    else:
        side = rng.choice(("left", "right", "bottom", "top"))
        sign = 1 if op == "grow" else -1
        if side == "left":
            x0 -= sign * d
        elif side == "right":
            x1 += sign * d
        elif side == "bottom":
            y0 -= sign * d
        else:
            y1 += sign * d
        desc = f"{op} {where} {side} by {d}"
    desc = _set(shapes, i, x0, y0, x1, y1, desc)
    return replace(cell, shapes=shapes), desc
