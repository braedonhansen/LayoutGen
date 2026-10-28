"""Fast DRC on rendered rectangles (SKY130 subset).

Rule values are taken from the pinned PDK, never from the renderer:
- K: KLayout deck libs.tech/klayout/drc/sky130A_mr.drc (the ground truth run
  by cellgym.groundtruth.verify; every width/space/enclosing there is
  `euclidian`, li.5/m1.5 use `projection` + adjacent-corner detection).
- M: Magic tech libs.tech/magic/sky130A.tech (drc section; cifoutput for the
  0.125 implant enclosure). These rules are not in the KLayout deck, so the
  audit cannot confirm them against KLayout; they are reported as "strict".

Width/space follow KLayout's edge-pair semantics on merged polygons:
anti-parallel facing edges, Euclidean distance between the edge segments
(corner-to-corner when their projections do not overlap); width is checked
within one polygon, space between any edges (including notches).
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .geometry import Region, bloat, gap, rect_dist2

GRID = 5

# rule -> (value nm, source). K = KLayout deck, M = Magic tech.
RULES = {
    "nwell.1": (840, "K"), "nwell.2a": (1270, "K"),
    "hvtp.1": (380, "K"), "hvtp.2": (380, "K"),
    "difftap.1": (150, "K"), "difftap.3": (270, "K"),
    "poly.1a": (150, "K"), "poly.2": (210, "K"),
    "npc.1": (270, "K"), "npc.2": (270, "K"),
    "nsdm.1": (380, "K"), "nsdm.2": (380, "K"), "psdm.1": (380, "K"), "psdm.2": (380, "K"),
    "nsdm_psdm_overlap": (0, "K"),
    "licon.1": (170, "K"), "MR_licon.SP.1": (170, "K"),
    "licon.13": (90, "K"), "licon.13_a": (0, "K"), "licon.17": (0, "K"),
    "li.1": (170, "K"), "li.3": (170, "K"), "li.5": (80, "K"), "li.6": (56100, "K"),
    "ct.1": (0, "K"), "ct.1_a": (170, "K"), "ct.1_b": (170, "K"), "ct.2": (190, "K"), "ct.4": (0, "K"),
    "m1.1": (140, "K"), "m1.2": (140, "K"), "791_m1.4": (30, "K"), "m1.4": (0, "K"),
    "m1.5": (60, "K"), "m1.6": (83000, "K"),
    "offgrid": (GRID, "K"),
    "poly.4": (75, "M"), "poly.7": (250, "M"), "poly.8": (130, "M"), "poly.11": (0, "M"),
    "licon.5a": (40, "M"), "licon.5c": (60, "M"), "licon.8": (50, "M"), "licon.8a": (80, "M"),
    "licon.9": (235, "M"), "licon.11": (55, "M"), "licon.14": (190, "M"),
    "difftap.8": (180, "M"), "difftap.9": (340, "M"),
    "licon.1_li": (0, "M"),
    "nsd.5a": (125, "M"), "psd.5a": (125, "M"), "diff.implant": (0, "M"),
}
KLAYOUT_RULES = {r for r, (_, src) in RULES.items() if src == "K"}

LAYER_NAMES = ("nwell", "diff", "poly", "licon", "li", "mcon", "met1", "hvtp", "nsdm", "psdm", "npc")
# SKY130 GDS drawing layers (deck input() / polygons() declarations)
GDS_LAYERS = {
    (64, 20): "nwell", (65, 20): "diff", (66, 20): "poly", (66, 44): "licon", (67, 20): "li",
    (67, 44): "mcon", (68, 20): "met1", (78, 44): "hvtp", (93, 44): "nsdm", (94, 20): "psdm",
    (95, 20): "npc",
}


@dataclass(frozen=True)
class Violation:
    rule: str
    layer: str
    box: tuple  # (x0, y0, x1, y1) nm

    def __str__(self) -> str:
        return f"{self.rule} [{self.layer}] at {self.box}"


class Geometry:
    """Per-layer merged regions plus derived layers, built lazily."""

    def __init__(self, rects: dict[str, list]):
        self.raw = {k: rects.get(k, []) for k in LAYER_NAMES}
        self._r: dict[str, Region] = {}

    def __getitem__(self, name: str) -> Region:
        r = self._r.get(name)
        if r is None:
            r = self._derive(name)
            self._r[name] = r
        return r

    def _derive(self, name: str) -> Region:
        if name in self.raw:
            return Region.from_rects(self.raw[name])
        if name == "gate":
            return self["poly"] & self["diff"]
        if name == "sd":
            return self["diff"] - self["poly"]
        if name == "fieldpoly":
            return self["poly"] - self["diff"]
        if name == "pdiff":
            return self["diff"] & self["psdm"]
        if name == "ndiff":
            return self["diff"] & self["nsdm"]
        raise KeyError(name)

    def licons(self):
        """Licon components classified as ('diff'|'poly'|'none', rects, bbox)."""
        if "_licons" not in self.__dict__:
            out = []
            diff, poly = self["diff"], self["poly"]
            for comp in self["licon"].components():
                bb = _bbox(comp)
                on_diff = any(diff.overlaps(r) for r in comp)
                on_poly = any(poly.overlaps(r) for r in comp)
                kind = "diff" if on_diff else "poly" if on_poly else "none"
                out.append((kind, comp, bb, on_diff and on_poly))
            self.__dict__["_licons"] = out
        return self.__dict__["_licons"]


def _bbox(rects) -> tuple:
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


def is_rect(rects) -> bool:
    """A component (list of disjoint slab rects) that is exactly its bbox."""
    if len(rects) == 1:
        return True
    x0, y0, x1, y1 = _bbox(rects)
    return sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects) == (x1 - x0) * (y1 - y0)


# -- generic checks ---------------------------------------------------------------------

def _pairs_lo_hi(los, his, d, need_hi_gt_lo: bool):
    """Pairs (lo_edge, hi_edge) of parallel edges (c, a0, a1) with
    0 < hi.c - lo.c < d and Euclidean distance < d. Yields (lo, hi)."""
    if not los or not his:
        return
    his = sorted(his)
    keys = [h[0] for h in his]
    d2 = d * d
    for lo in los:
        c = lo[0]
        i = bisect_right(keys, c)
        j = bisect_left(keys, c + d)
        for h in his[i:j]:
            dc = h[0] - c
            g = gap(lo[1], lo[2], h[1], h[2])
            if dc * dc + g * g < d2:
                yield lo, h


def _edge_box(lo, hi, vertical: bool) -> tuple:
    a0 = min(lo[1], hi[1]) if gap(lo[1], lo[2], hi[1], hi[2]) else max(lo[1], hi[1])
    a1 = max(lo[2], hi[2]) if gap(lo[1], lo[2], hi[1], hi[2]) else min(lo[2], hi[2])
    a0, a1 = min(a0, a1), max(a0, a1)
    if vertical:
        return (lo[0], a0, hi[0], a1)
    return (a0, lo[0], a1, hi[0])


def width_check(reg: Region, d: int, rule: str, layer: str, out: list) -> None:
    for comp in reg.components():
        if is_rect(comp):
            x0, y0, x1, y1 = bb = _bbox(comp)
            if x1 - x0 < d or y1 - y0 < d:
                out.append(Violation(rule, layer, bb))
            continue
        left, right, bottom, top = Region.from_rects(comp).edges()
        for lo, hi in _pairs_lo_hi(left, right, d, True):
            out.append(Violation(rule, layer, _edge_box(lo, hi, True)))
        for lo, hi in _pairs_lo_hi(bottom, top, d, True):
            out.append(Violation(rule, layer, _edge_box(lo, hi, False)))


def space_check(reg: Region, d: int, rule: str, layer: str, out: list) -> None:
    left, right, bottom, top = reg.edges()
    for lo, hi in _pairs_lo_hi(right, left, d, True):
        out.append(Violation(rule, layer, _edge_box(lo, hi, True)))
    for lo, hi in _pairs_lo_hi(top, bottom, d, True):
        out.append(Violation(rule, layer, _edge_box(lo, hi, False)))


def area_check(reg: Region, a: int, rule: str, layer: str, out: list) -> None:
    for comp in reg.components():
        if sum((r[2] - r[0]) * (r[3] - r[1]) for r in comp) < a:
            out.append(Violation(rule, layer, _bbox(comp)))


def separation(a_rects, b_rects, d: int, rule: str, layer: str, out: list, corner_ok: bool = False) -> None:
    """Euclidean spacing between two layers; overlap and touching are errors
    (touching only at a corner is allowed when corner_ok)."""
    if not a_rects or not b_rects:
        return
    bs = sorted(b_rects)
    keys = [b[0] for b in bs]
    maxw = max(b[2] - b[0] for b in bs)
    d2 = d * d
    for a in a_rects:
        i = bisect_left(keys, a[0] - d - maxw)
        j = bisect_right(keys, a[2] + d)
        for b in bs[i:j]:
            dist2 = rect_dist2(a, b)
            if dist2 >= d2:
                continue
            if dist2 == 0 and corner_ok:
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox <= 0 and oy <= 0:    # corner touch
                    continue
            out.append(Violation(rule, layer, (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))))


def not_empty(reg: Region, rule: str, layer: str, out: list) -> None:
    for comp in reg.components():
        out.append(Violation(rule, layer, _bbox(comp)))


def surround(outer: Region, rects, d: int, rule: str, layer: str, out: list) -> None:
    """Magic `surround ... absence_illegal`: the rect grown by d is inside outer."""
    for r in rects:
        if not outer.covers(bloat(r, d)):
            out.append(Violation(rule, layer, r))


def directional(outer: Region, rects, v: int, rule: str, layer: str, out: list) -> None:
    """Enclosure >= v on both edges of one direction (Magic `directional`;
    KLayout li.5/m1.5: no two adjacent edges with less enclosure)."""
    for r in rects:
        if not (outer.covers(bloat(r, v, 0)) or outer.covers(bloat(r, 0, v))):
            out.append(Violation(rule, layer, r))


def adjacent_less(outer: Region, rects, v: int, rule: str, layer: str, out: list) -> None:
    """KLayout li.5 / m1.5: `outer.enclosing(inner, v, projection).second_edges`
    then a corner where two such edge parts meet. Per inner side, the parts
    with less enclosure are where a same-facing outer edge lies 0..v in front
    (projection overlap > 0); sides outside the outer are not reported."""
    left, right, bottom, top = outer.edges()
    for r in rects:
        x0, y0, x1, y1 = r
        # per side: does a flagged part reach the low end / high end of the side?
        def side(edges, c0, lo, hi, sign):
            reach_lo = reach_hi = False
            for c, a0, a1 in edges:
                dist = (c - c0) * sign
                if 0 <= dist < v and min(a1, hi) > max(a0, lo):
                    reach_lo |= a0 <= lo
                    reach_hi |= a1 >= hi
            return reach_lo, reach_hi
        rl, rh = side(right, x1, y0, y1, 1)     # right side: lo = bottom corner, hi = top corner
        ll, lh = side(left, x0, y0, y1, -1)
        bl, bh = side(bottom, y0, x0, x1, -1)  # bottom side: lo = left corner, hi = right corner
        tl, th = side(top, y1, x0, x1, 1)
        if (rh and th) or (rl and bh) or (lh and tl) or (ll and bl):
            out.append(Violation(rule, layer, r))


def enclosing(outer: Region, rects, d: int, rule: str, layer: str, out: list) -> None:
    """KLayout `outer.enclosing(inner, d, euclidian)` for inner rectangles that
    lie inside outer: same-facing outer edges closer than d (Euclidean)."""
    left, right, bottom, top = outer.edges()
    d2 = d * d
    for r in rects:
        x0, y0, x1, y1 = r
        bad = False
        for c, a0, a1 in right:
            if x1 <= c < x1 + d:
                g = gap(y0, y1, a0, a1)
                bad |= (c - x1) ** 2 + g * g < d2
        for c, a0, a1 in left:
            if x0 - d < c <= x0:
                g = gap(y0, y1, a0, a1)
                bad |= (x0 - c) ** 2 + g * g < d2
        for c, a0, a1 in top:
            if y1 <= c < y1 + d:
                g = gap(x0, x1, a0, a1)
                bad |= (c - y1) ** 2 + g * g < d2
        for c, a0, a1 in bottom:
            if y0 - d < c <= y0:
                g = gap(x0, x1, a0, a1)
                bad |= (y0 - c) ** 2 + g * g < d2
        if bad:
            out.append(Violation(rule, layer, r))


# -- the deck subset --------------------------------------------------------------------

def run_drc(g: Geometry) -> list[Violation]:
    v: list[Violation] = []
    R = lambda name: RULES[name][0]

    # off-grid (x.1b): raw shapes, every layer
    for layer, rects in g.raw.items():
        for r in rects:
            if any(c % GRID for c in r):
                v.append(Violation("offgrid", layer, tuple(r)))

    # width / spacing (euclidian, merged)
    for layer, wr, sr in (("nwell", "nwell.1", "nwell.2a"), ("hvtp", "hvtp.1", "hvtp.2"),
                          ("diff", "difftap.1", "difftap.3"), ("poly", "poly.1a", "poly.2"),
                          ("npc", "npc.1", "npc.2"), ("nsdm", "nsdm.2", "nsdm.1"),
                          ("psdm", "psdm.2", "psdm.1"), ("li", "li.1", "li.3"),
                          ("met1", "m1.1", "m1.2")):
        reg = g[layer]
        if reg:
            width_check(reg, R(wr), wr, layer, v)
            space_check(reg, R(sr), sr, layer, v)
    area_check(g["li"], R("li.6"), "li.6", "li", v)
    area_check(g["met1"], R("m1.6"), "m1.6", "met1", v)

    # implants
    not_empty(g["psdm"] & g["nsdm"] & g["diff"], "nsdm_psdm_overlap", "diff", v)
    not_empty(g["diff"] - (g["nsdm"] | g["psdm"]), "diff.implant", "diff", v)
    ndiff_only = g["ndiff"] - g["psdm"]
    pdiff_only = g["pdiff"] - g["nsdm"]
    surround(g["nsdm"], ndiff_only.rects(), R("nsd.5a"), "nsd.5a", "nsdm", v)
    surround(g["psdm"], pdiff_only.rects(), R("psd.5a"), "psd.5a", "psdm", v)

    # wells
    surround(g["nwell"], g["pdiff"].rects(), R("difftap.8"), "difftap.8", "nwell", v)
    separation(ndiff_only.rects(), g["nwell"].rects(), R("difftap.9"), "difftap.9", "nwell", v)

    # licon
    diff, poly = g["diff"], g["poly"]
    diff_licons, poly_licons = [], []
    for kind, comp, bb, both in g.licons():
        if not is_rect(comp) or bb[2] - bb[0] != R("licon.1") or bb[3] - bb[1] != R("licon.1"):
            v.append(Violation("licon.1", "licon", bb))
        if both:
            v.append(Violation("licon.17", "licon", bb))
        if kind == "diff":
            diff_licons.append(bb)
        elif kind == "poly":
            poly_licons.append(bb)
    space_check(g["licon"], R("MR_licon.SP.1"), "MR_licon.SP.1", "licon", v)
    if g["npc"]:
        separation(diff_licons, g["npc"].rects(), R("licon.13"), "licon.13", "licon", v)
    surround(diff, diff_licons, R("licon.5a"), "licon.5a", "diff", v)
    directional(diff, diff_licons, R("licon.5c"), "licon.5c", "diff", v)
    surround(poly, poly_licons, R("licon.8"), "licon.8", "poly", v)
    directional(poly, poly_licons, R("licon.8a"), "licon.8a", "poly", v)
    separation(diff_licons, g["gate"].rects(), R("licon.11"), "licon.11", "licon", v)
    separation(poly_licons, diff.rects(), R("licon.14"), "licon.14", "licon", v)
    separation(poly_licons, g["pdiff"].rects(), R("licon.9"), "licon.9", "licon", v)
    lic_all = [bb for _, _, bb, _ in g.licons()]
    adjacent_less(g["li"], lic_all, R("li.5"), "li.5", "li", v)
    # Magic builds its contact types from licon AND li; a licon partly off li
    # becomes an undersized contact ("contact width < 0.17um (licon.1)"). A
    # licon entirely off li is no contact at all for Magic (DRC-clean; the
    # missing connection is an LVS matter).
    li = g["li"]
    for _, comp, bb, _ in g.licons():
        if any(li.overlaps(r) for r in comp) and not all(li.covers(r) for r in comp):
            v.append(Violation("licon.1_li", "licon", bb))

    # gates (poly.7 / poly.8 / poly.11) and poly.4
    for comp in g["gate"].components():
        bb = _bbox(comp)
        if not is_rect(comp):
            v.append(Violation("poly.11", "poly", bb))
            continue
        x0, y0, x1, y1 = bb
        vert = poly.overlaps((x0, y1, x1, y1 + 1)) or poly.overlaps((x0, y0 - 1, x1, y0))
        if vert:
            ends = [(x0, y1, x1, y1 + R("poly.8")), (x0, y0 - R("poly.8"), x1, y0)]
            sides = [(x0 - R("poly.7"), y0, x0, y1), (x1, y0, x1 + R("poly.7"), y1)]
        else:
            ends = [(x0 - R("poly.8"), y0, x0, y1), (x1, y0, x1 + R("poly.8"), y1)]
            sides = [(x0, y1, x1, y1 + R("poly.7")), (x0, y0 - R("poly.7"), x1, y0)]
        if not all(poly.covers(e) for e in ends):
            v.append(Violation("poly.8", "poly", bb))
        if not all(diff.covers(s) for s in sides):
            v.append(Violation("poly.7", "diff", bb))
    separation(g["fieldpoly"].rects(), g["sd"].rects(), R("poly.4"), "poly.4", "poly", v, corner_ok=True)

    # mcon / met1
    mcon = g["mcon"]
    for comp in mcon.components():
        bb = _bbox(comp)
        if not is_rect(comp):
            v.append(Violation("ct.1", "mcon", bb))
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if min(w, h) < R("ct.1_a"):
            v.append(Violation("ct.1_a", "mcon", bb))
        if max(w, h) > R("ct.1_b"):
            v.append(Violation("ct.1_b", "mcon", bb))
    space_check(mcon, R("ct.2"), "ct.2", "mcon", v)
    not_empty(mcon - g["li"], "ct.4", "mcon", v)
    m1 = g["met1"]
    not_empty(mcon - m1, "m1.4", "mcon", v)
    mrects = [_bbox(c) for c in mcon.components()]
    inside = [r for r in mrects if m1.covers(r)]
    enclosing(m1, inside, R("791_m1.4"), "791_m1.4", "met1", v)
    adjacent_less(m1, mrects, R("m1.5"), "m1.5", "met1", v)

    # npc licon.13_a
    if g["npc"] and diff_licons:
        lic_diff = Region.from_rects(diff_licons) & diff
        not_empty(lic_diff & g["npc"], "licon.13_a", "licon", v)

    seen, uniq = set(), []
    for x in v:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq
