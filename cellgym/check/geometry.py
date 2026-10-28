"""Manhattan region geometry for the fast checker (integer nm, axis-aligned).

A Region is the merged union of rectangles, stored as horizontal slabs: a
sorted list of (y0, y1, xs) where xs is a flat, sorted list of disjoint
x-intervals [a0, b0, a1, b1, ...] covering the band y0 <= y < y1. Adjacent
slabs with identical intervals are merged, so every slab is maximal.

Everything here is independent of the renderer: it only sees rectangles.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

Rect = tuple  # (x0, y0, x1, y1)


def _union(ivs: list[tuple[int, int]]) -> list[int]:
    """Union of (a, b) intervals -> flat sorted disjoint list (touching merge)."""
    if not ivs:
        return []
    ivs.sort()
    out = [ivs[0][0], ivs[0][1]]
    for a, b in ivs[1:]:
        if a <= out[-1]:
            if b > out[-1]:
                out[-1] = b
        else:
            out += [a, b]
    return out


def _bool(xa: list[int], xb: list[int], op: str) -> list[int]:
    """Boolean of two flat interval lists. op: 'and' | 'not' | 'or'."""
    if op == "or":
        return _union([(xa[i], xa[i + 1]) for i in range(0, len(xa), 2)]
                      + [(xb[i], xb[i + 1]) for i in range(0, len(xb), 2)])
    if not xa:
        return []
    if not xb:
        return list(xa) if op == "not" else []
    out: list[int] = []
    if op == "and":
        i = j = 0
        while i < len(xa) and j < len(xb):
            lo = max(xa[i], xb[j])
            hi = min(xa[i + 1], xb[j + 1])
            if lo < hi:
                out += [lo, hi]
            if xa[i + 1] < xb[j + 1]:
                i += 2
            else:
                j += 2
        return out
    # not: xa minus xb
    j = 0
    for i in range(0, len(xa), 2):
        a, b = xa[i], xa[i + 1]
        while j < len(xb) and xb[j + 1] <= a:
            j += 2
        k = j
        cur = a
        while k < len(xb) and xb[k] < b:
            if xb[k] > cur:
                out += [cur, xb[k]]
            cur = max(cur, xb[k + 1])
            k += 2
        if cur < b:
            out += [cur, b]
    return out


class Region:
    __slots__ = ("slabs", "_comp", "_edges")

    def __init__(self, slabs=None):
        self.slabs: list[tuple[int, int, list[int]]] = slabs or []
        self._comp = None
        self._edges = None

    # -- construction -----------------------------------------------------------

    @classmethod
    def from_rects(cls, rects) -> "Region":
        rects = [r for r in rects if r[2] > r[0] and r[3] > r[1]]
        if not rects:
            return cls([])
        ys = sorted({r[1] for r in rects} | {r[3] for r in rects})
        starts: dict[int, list] = {}
        for r in rects:
            starts.setdefault(r[1], []).append(r)
        active: list = []
        slabs = []
        for k in range(len(ys) - 1):
            y0, y1 = ys[k], ys[k + 1]
            if y0 in starts:
                active += starts[y0]
            active = [r for r in active if r[3] > y0]
            if not active:
                continue
            xs = _union([(r[0], r[2]) for r in active])
            if slabs and slabs[-1][1] == y0 and slabs[-1][2] == xs:
                slabs[-1] = (slabs[-1][0], y1, xs)
            else:
                slabs.append((y0, y1, xs))
        return cls(slabs)

    def _combine(self, other: "Region", op: str) -> "Region":
        ys = sorted({s[0] for s in self.slabs} | {s[1] for s in self.slabs}
                    | {s[0] for s in other.slabs} | {s[1] for s in other.slabs})
        out = []
        ia = ib = 0
        A, B = self.slabs, other.slabs
        for k in range(len(ys) - 1):
            y0, y1 = ys[k], ys[k + 1]
            while ia < len(A) and A[ia][1] <= y0:
                ia += 1
            while ib < len(B) and B[ib][1] <= y0:
                ib += 1
            xa = A[ia][2] if ia < len(A) and A[ia][0] <= y0 else []
            xb = B[ib][2] if ib < len(B) and B[ib][0] <= y0 else []
            xs = _bool(xa, xb, op)
            if not xs:
                continue
            if out and out[-1][1] == y0 and out[-1][2] == xs:
                out[-1] = (out[-1][0], y1, xs)
            else:
                out.append((y0, y1, xs))
        return Region(out)

    def __and__(self, o): return self._combine(o, "and")
    def __sub__(self, o): return self._combine(o, "not")
    def __or__(self, o): return self._combine(o, "or")

    # -- queries ------------------------------------------------------------------

    def __bool__(self) -> bool:
        return bool(self.slabs)

    def rects(self) -> list[Rect]:
        return [(xs[i], y0, xs[i + 1], y1) for y0, y1, xs in self.slabs for i in range(0, len(xs), 2)]

    def area(self) -> int:
        return sum((y1 - y0) * (xs[i + 1] - xs[i]) for y0, y1, xs in self.slabs for i in range(0, len(xs), 2))

    def bbox(self) -> Rect | None:
        if not self.slabs:
            return None
        return (min(s[2][0] for s in self.slabs), self.slabs[0][0],
                max(s[2][-1] for s in self.slabs), self.slabs[-1][1])

    def covers(self, r: Rect) -> bool:
        """True if the closed-open box r is entirely inside the region."""
        x0, y0, x1, y1 = r
        if x1 <= x0 or y1 <= y0:
            return True
        y = y0
        for sy0, sy1, xs in self.slabs:
            if sy1 <= y:
                continue
            if sy0 > y:
                return False
            i = bisect_right(xs, x0)
            if i % 2 == 0 or xs[i] < x1:   # x0 not inside an interval, or interval too short
                return False
            y = sy1
            if y >= y1:
                return True
        return False

    def contains_point(self, x: int, y: int) -> bool:
        """Closed containment (boundary counts)."""
        for sy0, sy1, xs in self.slabs:
            if sy0 <= y <= sy1:
                i = bisect_left(xs, x)
                if i < len(xs) and (i % 2 == 1 or xs[i] == x):
                    return True
        return False

    def overlaps(self, r: Rect) -> bool:
        x0, y0, x1, y1 = r
        for sy0, sy1, xs in self.slabs:
            if sy1 <= y0 or sy0 >= y1:
                continue
            i = bisect_right(xs, x0)
            if i % 2 == 1 or (i < len(xs) and xs[i] < x1):
                return True
        return False

    # -- components -----------------------------------------------------------------

    def components(self) -> list[list[Rect]]:
        """Connected components (edge-sharing; corner touch does not connect),
        each as a list of slab rectangles."""
        if self._comp is not None:
            return self._comp
        rects = []
        rows: list[list[int]] = []
        for y0, y1, xs in self.slabs:
            row = []
            for i in range(0, len(xs), 2):
                row.append(len(rects))
                rects.append((xs[i], y0, xs[i + 1], y1))
            rows.append(row)
        parent = list(range(len(rects)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for k in range(1, len(rows)):
            if self.slabs[k - 1][1] != self.slabs[k][0]:
                continue
            below, above = rows[k - 1], rows[k]
            i = j = 0
            while i < len(below) and j < len(above):
                a, b = rects[below[i]], rects[above[j]]
                if min(a[2], b[2]) > max(a[0], b[0]):
                    ra, rb = find(below[i]), find(above[j])
                    if ra != rb:
                        parent[ra] = rb
                if a[2] < b[2]:
                    i += 1
                else:
                    j += 1
        groups: dict[int, list] = {}
        for idx, r in enumerate(rects):
            groups.setdefault(find(idx), []).append(r)
        self._comp = list(groups.values())
        return self._comp

    # -- edges ----------------------------------------------------------------------

    def edges(self):
        """Boundary edges, maximal. Returns (left, right, bottom, top):
        left/right: (x, y0, y1) with the interior to the right / left;
        bottom/top: (y, x0, x1) with the interior above / below."""
        if self._edges is not None:
            return self._edges
        left, right, bottom, top = [], [], [], []
        open_l: dict[int, list] = {}
        open_r: dict[int, list] = {}
        prev_y1, prev_xs = None, []
        for y0, y1, xs in self.slabs:
            below = prev_xs if prev_y1 == y0 else []
            d = _bool(xs, below, "not")
            bottom += [(y0, d[i], d[i + 1]) for i in range(0, len(d), 2)]
            if prev_y1 is not None:
                u = _bool(prev_xs, xs if prev_y1 == y0 else [], "not")
                top += [(prev_y1, u[i], u[i + 1]) for i in range(0, len(u), 2)]
            nl, nr = {}, {}
            for i in range(0, len(xs), 2):
                a, b = xs[i], xs[i + 1]
                e = open_l.pop(a, None) if prev_y1 == y0 else None
                if e is not None:
                    e[2] = y1
                else:
                    e = [a, y0, y1]
                    left.append(e)
                nl[a] = e
                e = open_r.pop(b, None) if prev_y1 == y0 else None
                if e is not None:
                    e[2] = y1
                else:
                    e = [b, y0, y1]
                    right.append(e)
                nr[b] = e
            open_l, open_r = nl, nr
            prev_y1, prev_xs = y1, xs
        if prev_y1 is not None:
            top += [(prev_y1, prev_xs[i], prev_xs[i + 1]) for i in range(0, len(prev_xs), 2)]
        self._edges = ([tuple(e) for e in left], [tuple(e) for e in right], bottom, top)
        return self._edges


def gap(a0: int, a1: int, b0: int, b1: int) -> int:
    """Gap between intervals [a0,a1] and [b0,b1] (0 if they overlap or touch)."""
    return max(0, b0 - a1, a0 - b1)


def rect_dist2(a: Rect, b: Rect) -> int:
    """Squared Euclidean distance between closed rectangles (0 if touching/overlapping)."""
    dx = gap(a[0], a[2], b[0], b[2])
    dy = gap(a[1], a[3], b[1], b[3])
    return dx * dx + dy * dy


def bloat(r: Rect, dx: int, dy: int | None = None) -> Rect:
    dy = dx if dy is None else dy
    return (r[0] - dx, r[1] - dy, r[2] + dx, r[3] + dy)
