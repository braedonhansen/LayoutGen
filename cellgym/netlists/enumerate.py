"""Enumeration of static-CMOS cells as series-parallel pull-down networks.

A network is a flattened series-parallel tree of nested tuples:
    ("L", i)                  leaf: NMOS gated by input i
    ("S", (child, ...))       series
    ("P", (child, ...))       parallel
Children are sorted, and a child never has the same kind as its parent
(flattened), so every function-preserving rearrangement of siblings maps to
the same tuple.

Pipeline: unlabeled shapes (pruned by stack depth while building) ->
restricted-growth labelings -> irredundancy filter -> canonical form under
input renaming.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from itertools import permutations, product

INPUT_NAMES = string.ascii_uppercase  # A, B, C, ...


@dataclass(frozen=True)
class Params:
    max_inputs: int = 6
    max_stack: int = 3          # applied to both PDN and PUN depth
    extra_devices: int = 2      # repeated-input networks: devices <= inputs + extra
    max_devices: int = 8


@dataclass(frozen=True)
class Network:
    tree: tuple
    n_inputs: int

    @property
    def n_devices(self) -> int:
        return num_leaves(self.tree)

    @property
    def read_once(self) -> bool:
        return self.n_devices == self.n_inputs


# ---------------------------------------------------------------- tree basics

def make(kind, children):
    """Build a normalized node: flatten same-kind children, sort."""
    flat = []
    for c in children:
        flat.extend(c[1] if c[0] == kind else [c])
    return (kind, tuple(sorted(flat)))


def num_leaves(t) -> int:
    return 1 if t[0] == "L" else sum(num_leaves(c) for c in t[1])


def depth(t, dual: bool = False) -> int:
    """Stack depth of the PDN (dual=False) or of its dual PUN (dual=True)."""
    if t[0] == "L":
        return 1
    ds = [depth(c, dual) for c in t[1]]
    return sum(ds) if (t[0] == "S") != dual else max(ds)


def dual_tree(t):
    if t[0] == "L":
        return t
    return make("P" if t[0] == "S" else "S", [dual_tree(c) for c in t[1]])


def tree_str(t) -> str:
    if t[0] == "L":
        return INPUT_NAMES[t[1]]
    return f"{t[0]}({','.join(tree_str(c) for c in t[1])})"


def parse_tree(s: str):
    """Inverse of tree_str, e.g. 'P(S(A,B),S(A,C))'. Result is normalized."""
    pos = 0

    def node():
        nonlocal pos
        ch = s[pos]
        if ch in "SP" and pos + 1 < len(s) and s[pos + 1] == "(":
            pos += 2
            kids = [node()]
            while s[pos] == ",":
                pos += 1
                kids.append(node())
            assert s[pos] == ")"
            pos += 1
            return make(ch, kids)
        pos += 1
        return ("L", INPUT_NAMES.index(ch))

    t = node()
    assert pos == len(s), f"trailing text in {s!r}"
    return t


def num_inputs(t) -> int:
    def mx(t):
        return t[1] if t[0] == "L" else max(mx(c) for c in t[1])
    return mx(t) + 1


# --------------------------------------------------------------------- shapes

def shapes(max_leaves: int, max_stack: int | None):
    """Unlabeled flattened SP trees by leaf count, PDN depth <= max_stack.

    max_stack=None disables pruning (counts 1, 2, 4, 10, 24, 66, ...).
    """
    by_n = {1: [("L", 0)]}
    for n in range(2, max_leaves + 1):
        found = set()
        pool = [(t, s) for s in range(1, n) for t in by_n[s]]
        for kind in "SP":
            ok = [(t, s) for t, s in pool if t[0] != kind]

            def pick(start, remaining, chosen):
                if remaining == 0 and len(chosen) >= 2:
                    t = make(kind, chosen)
                    if max_stack is None or depth(t) <= max_stack:
                        found.add(t)
                    return
                for i in range(start, len(ok)):
                    t, s = ok[i]
                    if s <= remaining:
                        pick(i, remaining - s, chosen + [t])

            pick(0, n, [])
        by_n[n] = sorted(found)
    return by_n


def _label(t, it):
    if t[0] == "L":
        return ("L", next(it))
    return make(t[0], [_label(c, it) for c in t[1]])


def rgs(n: int, k: int):
    """Restricted-growth strings of length n using exactly k symbols."""
    def go(prefix, top):
        if len(prefix) == n:
            if top == k:
                yield tuple(prefix)
            return
        if k - top > n - len(prefix):
            return
        for v in range(min(top + 1, k)):
            yield from go(prefix + [v], max(top, v + 1))
    yield from go([], 0)


# ------------------------------------------------------------- canonical form

def relabel(t, m):
    if t[0] == "L":
        return ("L", m[t[1]])
    return make(t[0], [relabel(c, m) for c in t[1]])


def _strip(t):
    return ("L",) if t[0] == "L" else (t[0], tuple(sorted(_strip(c) for c in t[1])))


def signatures(t, k):
    """Renaming-invariant signature per input: label-free contexts of its leaves."""
    sig = [[] for _ in range(k)]

    def walk(t, path):
        if t[0] == "L":
            sig[t[1]].append(path)
            return
        for j, c in enumerate(t[1]):
            sibs = tuple(sorted(_strip(s) for i, s in enumerate(t[1]) if i != j))
            walk(c, path + ((t[0], sibs),))

    walk(t, ())
    return [tuple(sorted(s)) for s in sig]


def canonical(t, k):
    """Smallest relabeling of t over permutations within signature-tied groups."""
    sig = signatures(t, k)
    order = sorted(range(k), key=lambda i: sig[i])
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if sig[i] == sig[cur[0]]:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    best = None
    for perms in product(*(permutations(g) for g in groups)):
        flat = [x for p in perms for x in p]
        cand = relabel(t, {old: new for new, old in enumerate(flat)})
        if best is None or cand < best:
            best = cand
    return best


# --------------------------------------------------------- truth tables / logic

def var_masks(k):
    rows = 1 << k
    return [sum(1 << r for r in range(rows) if (r >> i) & 1) for i in range(k)], (1 << rows) - 1


def truth(t, masks, full, force=None):
    """Bitmask truth table of 'PDN conducts'. Bit r = row r (input i = bit i of r).

    force = (leaf index in evaluation order, 0|1) replaces that leaf by an
    open (0) or short (1).
    """
    idx = [0]

    def ev(t):
        if t[0] == "L":
            i = idx[0]
            idx[0] += 1
            if force is not None and force[0] == i:
                return full if force[1] else 0
            return masks[t[1]]
        vals = [ev(c) for c in t[1]]
        acc = vals[0]
        for v in vals[1:]:
            acc = (acc & v) if t[0] == "S" else (acc | v)
        return acc

    return ev(t)


def pdn_function(t, k) -> int:
    masks, full = var_masks(k)
    return truth(t, masks, full)


def irredundant(t, k) -> bool:
    """True if forcing any single device open or shorted changes the function."""
    masks, full = var_masks(k)
    return _irredundant(t, num_leaves(t), masks, full)


def _irredundant(t, n, masks, full):
    base = truth(t, masks, full)
    return all(truth(t, masks, full, (i, v)) != base for i in range(n) for v in (0, 1))


def canonical_fn(f, k):
    """Smallest truth table over all input renamings."""
    rows = 1 << k
    best = None
    for p in permutations(range(k)):
        g = 0
        for r in range(rows):
            if (f >> r) & 1:
                g |= 1 << sum(((r >> i) & 1) << p[i] for i in range(k))
        if best is None or g < best:
            best = g
    return best


# ---------------------------------------------------------------- enumeration

def enumerate_networks(params: Params = Params(), *, read_once_pdn_only: bool = False):
    """All canonical networks satisfying params, sorted by (inputs, devices, tree).

    Both PDN and PUN stack depth are <= max_stack unless read_once_pdn_only
    (which only bounds the PDN; used to pin the PDN-only counts).
    """
    top = min(params.max_inputs + params.extra_devices, params.max_devices)
    by_n = shapes(top, params.max_stack)
    out = []
    for k in range(1, params.max_inputs + 1):
        masks, full = var_masks(k)
        for n in range(k, min(k + params.extra_devices, params.max_devices) + 1):
            seen = set()
            for shp in by_n[n]:
                for s in rgs(n, k):
                    t = _label(shp, iter(s))
                    if n > k and not _irredundant(t, n, masks, full):
                        continue
                    seen.add(canonical(t, k))
            for t in sorted(seen):
                if read_once_pdn_only or depth(t, dual=True) <= params.max_stack:
                    out.append(Network(t, k))
    return out
