"""Deterministic layout mutations for checker audits.

    mutate(rendered, seed) -> (mutant Rendered, description)

One perturbation per mutant, chosen by seed: shrink / grow / shift one
rectangle by 5-40 nm (one edge or the whole rect, x or y), delete one
rectangle, or move one li/met1 text label onto another shape of its layer.
Coordinates stay on the 5 nm grid.
"""

from __future__ import annotations

import random
from dataclasses import replace

LAYERS = ("nwell", "diff", "poly", "licon", "li", "mcon", "met1", "hvtp", "nsdm", "psdm", "npc")
OPS = ("shrink", "grow", "shift", "delete", "label")
OP_WEIGHTS = (3, 3, 3, 1, 1)


def mutate(cell, seed: int):
    rng = random.Random(seed)
    op = rng.choices(OPS, OP_WEIGHTS)[0]
    shapes = list(cell.shapes)
    labels = list(cell.labels)
    if op == "label":
        cands = [i for i, lab in enumerate(labels) if lab.kind in ("li", "met1")]
        i = rng.choice(cands)
        lab = labels[i]
        targets = [s for s in shapes if s.layer == lab.kind and (s.x0, s.y0, s.x1, s.y1) != lab.rect]
        t = rng.choice(targets)
        labels[i] = replace(lab, rect=(t.x0, t.y0, t.x1, t.y1))
        cx, cy = (t.x0 + t.x1) // 2, (t.y0 + t.y1) // 2
        desc = f"label {lab.net} ({lab.kind}) moved to ({cx}, {cy})"
        return replace(cell, labels=labels), desc
    idx = [i for i, s in enumerate(shapes) if s.layer in LAYERS]
    i = rng.choice(idx)
    s = shapes[i]
    where = f"{s.layer} ({s.x0}, {s.y0}, {s.x1}, {s.y1})"
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
    if x1 <= x0 or y1 <= y0:
        del shapes[i]
        desc += " (vanished)"
    else:
        shapes[i] = replace(s, x0=x0, y0=y0, x1=x1, y1=y1)
    return replace(cell, shapes=shapes), desc
