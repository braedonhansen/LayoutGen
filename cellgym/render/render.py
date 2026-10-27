"""Plan + netlist -> sky130 layout (SPEC.md sections 1-3, 6, 8).

The plan fixes topology (column order, sharing, contacts, tracks); this module
owns every coordinate. y comes from the cell template (template.layout); x
comes from x-only compaction: columns are placed left to right, each at the
smallest x where its shapes keep the layer spacings to every earlier shape of
a different owner (projection rule: any y-ranges closer than the spacing).

All geometry is integer nm on the 5 nm grid.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from cellgym.dsl.plan import ROWS, Plan
from cellgym.dsl.template import CHANNEL_NAMES, MET1_OFFSET, MET1_PITCH
from cellgym.dsl.validate import RAILS, SUPPLY, validate

from .gds import GdsCell, dumps as gds_dumps

LAYERS = {
    "nwell": (64, 20), "diff": (65, 20), "poly": (66, 20), "licon": (66, 44),
    "li": (67, 20), "mcon": (67, 44), "met1": (68, 20), "hvtp": (78, 44),
    "nsdm": (93, 44), "psdm": (94, 20), "npc": (95, 20), "boundary": (235, 4),
}
PIN_LAYERS = {   # shape layer -> (text layer, pin layer)
    "li": ((67, 5), (67, 16)), "met1": ((68, 5), (68, 16)),
    "nwell": ((64, 5), (64, 16)), "sub": ((64, 59), (122, 16)),
}

SITE = 460
GRID = 5
# same-layer spacing between different owners (nm)
SPACE = {"poly": 210, "diff": 270, "li": 170, "met1": 140, "licon": 170, "mcon": 190}
# half-spacing margin from the outermost shape to the cell edge
MARGIN = {"poly": 105, "diff": 135, "li": 85, "met1": 70, "licon": 85, "mcon": 95}
LICON_GATE = 55          # diff licon to gate poly (licon.11)
POLY_DIFF = 75           # poly.4
POLY_LICON_DIFF = {"nmos": 190, "pmos": 235}   # licon.14 / licon.9 + psdm.5a
MIN_PITCH = 215          # gate edge 75 + licon-gate 55 + half licon 85

LICON = 170
LICON_HALF = 85
DIFF_LICON_Y = 60        # licon.5c
LI_LICON = 80            # li.5
RAIL_CLEAR = 335         # non-rail licon edge from the rail center: li strip 85 + li space 170 + li.5 80
POLY_END = 130           # poly.8
POLY_PAD = (165, 135)    # half sizes: licon 85 + 80 (x), + 50 (y)
LI_PAD = (165, 85)
M1_PAD = (145, 115)      # 0.29 x 0.23: m1.5 0.06 on x edges, m1.4 0.03 on y edges
M1_HALF = 70
SD_DIFF_HALF = 125       # licon 85 + diff enclosure 40
NDIFF_TO_TRACK = 275
TRACK_TO_PDIFF = 320
NWELL_PDIFF = 180
NDIFF_NWELL = 340
PSDM_PDIFF = 125         # psdm.5a; keeps poly licons 0.11 outside psdm (licon.9)
IMPLANT_EXT = 190
NPC_EXT = 185            # licon half 85 + 0.1
RAIL_HALF = 240
LI_RAIL_HALF = 85


class RenderError(ValueError):
    pass


@dataclass
class Shape:
    layer: str
    x0: int
    y0: int
    x1: int
    y1: int
    owner: tuple = ()
    tag: str = ""          # licon: "diff" | "poly"; diff: row

    def moved(self, dx: int) -> "Shape":
        return Shape(self.layer, self.x0 + dx, self.y0, self.x1 + dx, self.y1, self.owner, self.tag)


@dataclass
class Label:
    net: str
    kind: str              # key of PIN_LAYERS
    rect: tuple[int, int, int, int]


@dataclass
class Rendered:
    name: str
    width_nm: int          # compacted width (shape extents + half spacings)
    height_nm: int
    xs: list[int]          # column centers
    shapes: list[Shape] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    spice: str = ""

    @property
    def width(self) -> float:
        return self.width_nm / 1000

    @property
    def height(self) -> float:
        return self.height_nm / 1000

    @property
    def sites(self) -> int:
        return math.ceil(self.width_nm / SITE)

    def gds_cell(self) -> GdsCell:
        cell = GdsCell(self.name)
        for s in self.shapes:
            cell.rect(LAYERS[s.layer], s.x0, s.y0, s.x1, s.y1)
        for lab in self.labels:
            text, pin = PIN_LAYERS[lab.kind]
            x0, y0, x1, y1 = lab.rect
            cell.rect(pin, x0, y0, x1, y1)
            cell.text(text, (x0 + x1) // 2, (y0 + y1) // 2, lab.net)
        return cell

    def gds(self) -> bytes:
        return gds_dumps(self.gds_cell())


def _nm(um: float) -> int:
    return int(round(um * 1000))


def _up(v: float) -> int:
    return int(math.ceil(v / GRID - 1e-9)) * GRID


def _fmt(x: float) -> str:
    s = f"{x:.3f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def reference_spice(netlist: dict, name: str, outputs: list[str]) -> str:
    """LVS reference: ports inputs, outputs, VDD VSS VPB VNB; bulks on VPB/VNB."""
    ports = list(netlist["inputs"]) + list(outputs) + ["VDD", "VSS", "VPB", "VNB"]
    lines = [f"* {name}: reference for LVS (from netlist {netlist.get('id', name)})",
             f".subckt {name} {' '.join(ports)}"]
    for d in netlist["devices"]:
        bulk = "VPB" if d["type"] == "pmos" else "VNB"
        lines.append(f"X{d['name']} {d['drain']} {d['gate']} {d['source']} {bulk} {d['model']} "
                     f"w={_fmt(d['w'])} l={_fmt(d['l'])}")
    lines.append(".ends")
    return "\n".join(lines) + "\n"


class _Builder:
    def __init__(self, plan: Plan, netlist: dict):
        rep = validate(plan, netlist)
        if not rep.ok:
            raise RenderError(f"plan does not validate:\n{rep}")
        self.plan, self.netlist, self.rep = plan, netlist, rep
        lay = rep.layout
        self.H = _nm(lay.height)
        ty = lambda k: _nm(MET1_OFFSET + MET1_PITCH * k)
        k0, k2 = lay.channel_k[0], lay.channel_k[-1]
        self.track_y = {name: ty(k) for name, k in zip(CHANNEL_NAMES, lay.channel_k)}
        self.track_y.update({f"n{j}": ty(k0 - 1 - j) for j in range(lay.n_tracks)})
        self.track_y.update({f"p{j}": ty(k2 + 1 + j) for j in range(lay.p_tracks)})
        self.ntop = self.track_y["c0"] - NDIFF_TO_TRACK
        self.pbot = self.track_y[CHANNEL_NAMES[-1]] + TRACK_TO_PDIFF
        self.cols = plan.columns
        self._device_widths()
        self._islands()
        self._endpoints()

    # -- topology ---------------------------------------------------------------

    def _device_widths(self) -> None:
        pool = defaultdict(list)
        for d in self.netlist["devices"]:
            pool[(d["type"], d["gate"], frozenset((d["drain"], d["source"])))].append(_nm(d["w"]))
        self.gate_w: dict[tuple[str, int], int] = {}
        for c in self.cols:
            if c.kind != "g" or not 0 < c.index < len(self.cols) - 1:
                continue
            for row in ROWS:
                a, b = self.cols[c.index - 1].nets[row], self.cols[c.index + 1].nets[row]
                if c.nets[row] and a and b:
                    self.gate_w[(row, c.index)] = pool[(row, c.nets[row], frozenset((a, b)))].pop(0)

    def _islands(self) -> None:
        """Physical diffusion islands per row: broken by an empty slot, a gate
        with no device, or two adjacent sd columns with different nets."""
        self.island: dict[tuple[str, int], int] = {}
        self.island_gates: dict[int, set[int]] = defaultdict(set)
        n = 0
        for row in ROWS:
            cur, prev_sd = None, None
            for c in self.cols:
                if c.kind == "g":
                    if (row, c.index) in self.gate_w and cur is not None:
                        self.island[(row, c.index)] = cur
                        self.island_gates[cur].add(c.index)
                        prev_sd = None
                    else:
                        cur = None
                    continue
                net = c.nets[row]
                if net is None:
                    cur = None
                    continue
                if cur is None or (prev_sd is not None and prev_sd != net):
                    cur, n = n, n + 1
                self.island[(row, c.index)] = cur
                prev_sd = net

    def _endpoints(self) -> None:
        self.sd_tracks: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
        self.gate_m1: set[tuple[int, str]] = set()
        for r in self.plan.routes:
            for ep in r.endpoints:
                if ep.kind == "gate":
                    if r.layer == "met1":
                        self.gate_m1.add((ep.col, ep.side))
                else:
                    self.sd_tracks[(ep.kind, ep.col)].add((r.layer, r.track))

    def _row_y(self, row: str, h: int) -> tuple[int, int]:
        return (self.ntop - h, self.ntop) if row == "nmos" else (self.pbot, self.pbot + h)

    def _sd_widths(self, row: str, i: int) -> tuple[int, int]:
        """(max, min) width of the devices next to sd column i in this row."""
        ws = [self.gate_w[(row, j)] for j in (i - 1, i + 1) if (row, j) in self.gate_w]
        if not ws:   # an sd between two sds: take its island's devices
            isl = self.island[(row, i)]
            ws = [self.gate_w[(row, g)] for g in self.island_gates[isl]] or [_nm(0.42)]
        return max(ws), min(ws)

    # -- per-column shapes (x relative to the column center) --------------------

    def column_shapes(self, i: int) -> list[Shape]:
        c = self.cols[i]
        return self._gate_shapes(i) if c.kind == "g" else self._sd_shapes(i)

    def _gate_shapes(self, i: int) -> list[Shape]:
        c = self.cols[i]
        out = []
        for row in ROWS:
            if (row, i) in self.gate_w:
                y0, y1 = self._row_y(row, self.gate_w[(row, i)])
                out.append(Shape("diff", -75, y0, 75, y1, ("isl", self.island[(row, i)]), row))
        tracks = self.rep.gate_tracks
        lo = self.ntop - self.gate_w[("nmos", i)] - POLY_END if ("nmos", i) in self.gate_w else None
        hi = self.pbot + self.gate_w[("pmos", i)] + POLY_END if ("pmos", i) in self.gate_w else None
        if (i, "") in tracks:
            y = self.track_y[tracks[(i, "")]]
            pads = [((i, ""), c.nets["nmos"] or c.nets["pmos"], y)]
            segs = [(lo if lo is not None else y - POLY_PAD[1], hi if hi is not None else y + POLY_PAD[1])]
        else:
            yn, yp = self.track_y[tracks[(i, "n")]], self.track_y[tracks[(i, "p")]]
            pads = [((i, "n"), c.nets["nmos"], yn), ((i, "p"), c.nets["pmos"], yp)]
            segs = [(lo, yn + POLY_PAD[1]), (yp - POLY_PAD[1], hi)]
        for y0, y1 in segs:
            out.append(Shape("poly", -75, y0, 75, y1, ("g", i)))
        for key, net, y in pads:
            px, py = POLY_PAD
            out.append(Shape("poly", -px, y - py, px, y + py, ("g", i)))
            out.append(Shape("licon", -LICON_HALF, y - LICON_HALF, LICON_HALF, y + LICON_HALF, (), "poly"))
            out.append(Shape("li", -LI_PAD[0], y - LI_PAD[1], LI_PAD[0], y + LI_PAD[1], ("net", net), "gate"))
            if key in self.gate_m1:
                out += self._via(net, y)
        return out

    def _via(self, net: str, y: int) -> list[Shape]:
        return [Shape("mcon", -LICON_HALF, y - LICON_HALF, LICON_HALF, y + LICON_HALF),
                Shape("met1", -M1_PAD[0], y - M1_PAD[1], M1_PAD[0], y + M1_PAD[1], ("net", net))]

    def licon_ys(self, row: str, i: int) -> list[int]:
        net = self.cols[i].nets[row]
        rail = net == RAILS[row]
        _, h_min = self._sd_widths(row, i)
        d0, d1 = self._row_y(row, h_min)
        lo, hi = d0 + DIFF_LICON_Y, d1 - DIFF_LICON_Y
        if row == "nmos":
            hi = min(hi, self.track_y["c0"] - RAIL_CLEAR)
            if not rail:
                lo = max(lo, RAIL_CLEAR)
        else:
            lo = max(lo, self.track_y[CHANNEL_NAMES[-1]] + RAIL_CLEAR)
            if not rail:
                hi = min(hi, self.H - RAIL_CLEAR)
        if hi - lo < LICON:
            raise RenderError(f"column {i}: no room for a {row} licon")
        n = (hi - lo + LICON) // (2 * LICON)
        span = 2 * LICON * n - LICON
        start = lo + ((hi - lo - span) // 2 // GRID) * GRID
        return [start + LICON_HALF + 2 * LICON * k for k in range(n)]

    def _sd_shapes(self, i: int) -> list[Shape]:
        c = self.cols[i]
        out = []
        for row in ROWS:
            net = c.nets[row]
            if net is None:
                continue
            h, _ = self._sd_widths(row, i)
            y0, y1 = self._row_y(row, h)
            out.append(Shape("diff", -SD_DIFF_HALF, y0, SD_DIFF_HALF, y1, ("isl", self.island[(row, i)]), row))
            if not c.contact[row]:
                continue
            ys = self.licon_ys(row, i)
            for y in ys:
                out.append(Shape("licon", -LICON_HALF, y - LICON_HALF, LICON_HALF, y + LICON_HALF, (), "diff"))
            s0, s1 = ys[0] - LICON_HALF - LI_LICON, ys[-1] + LICON_HALF + LI_LICON
            if net == RAILS[row]:
                s0, s1 = (0, s1) if row == "nmos" else (s0, self.H)
            vias = []
            for layer, track in sorted(self.sd_tracks.get((row, i), ())):
                y = self.track_y[track]
                s0, s1 = min(s0, y - LI_PAD[1]), max(s1, y + LI_PAD[1])
                if layer == "met1":
                    vias += self._via(net, y)
            out.append(Shape("li", -LICON_HALF, s0, LICON_HALF, s1, ("net", net), f"strap {row}"))
            out += vias
        return out

    # -- compaction -------------------------------------------------------------

    def spacing(self, a: Shape, b: Shape) -> int | None:
        if a.layer == b.layer:
            if a.layer in ("licon", "mcon"):
                return SPACE[a.layer]
            return None if a.owner == b.owner else SPACE[a.layer]
        pair = {a.layer, b.layer}
        if pair == {"licon", "poly"}:
            lic = a if a.layer == "licon" else b
            return LICON_GATE if lic.tag == "diff" else None
        if pair == {"diff", "poly"}:
            d, p = (a, b) if a.layer == "diff" else (b, a)
            return None if p.owner[1] in self.island_gates[d.owner[1]] else POLY_DIFF
        if pair == {"licon", "diff"}:
            lic, d = (a, b) if a.layer == "licon" else (b, a)
            return POLY_LICON_DIFF[d.tag] if lic.tag == "poly" else None
        return None

    def compact(self) -> tuple[list[int], list[Shape]]:
        xs: list[int] = []
        placed: list[Shape] = []
        for i in range(len(self.cols)):
            shapes = self.column_shapes(i)
            x = xs[-1] + MIN_PITCH if xs else 0
            for s in shapes:
                for e in placed:
                    sp = self.spacing(s, e)
                    if sp is None or max(s.y0 - e.y1, e.y0 - s.y1) >= sp:
                        continue
                    x = max(x, e.x1 + sp - s.x0)
            x = _up(x)
            xs.append(x)
            placed += [s.moved(x) for s in shapes]
        return xs, placed

    # -- drawing ----------------------------------------------------------------

    def diffusion(self, xs: list[int]) -> list[Shape]:
        out = []
        for row in ROWS:
            for c in self.cols:
                i = c.index
                if c.kind == "g":
                    if (row, i) in self.gate_w:
                        y0, y1 = self._row_y(row, self.gate_w[(row, i)])
                        out.append(Shape("diff", xs[i] - 75, y0, xs[i] + 75, y1, ("isl", self.island[(row, i)]), row))
                    continue
                if c.nets[row] is None:
                    continue
                isl = self.island[(row, i)]
                h, _ = self._sd_widths(row, i)
                edges = []
                for j, sign in ((i - 1, -1), (i + 1, 1)):
                    if (row, j) in self.gate_w:          # gate side: narrow slice, then full height
                        g_edge = xs[j] - sign * 75
                        y0, y1 = self._row_y(row, self.gate_w[(row, j)])
                        a, b = sorted((g_edge, g_edge - sign * 75))
                        out.append(Shape("diff", a, y0, b, y1, ("isl", isl), row))
                        edges.append(g_edge - sign * 75)
                    elif self.island.get((row, j)) == isl:   # same-island sd: run to its center
                        edges.append(xs[j])
                    else:                                    # island end
                        edges.append(xs[i] + sign * SD_DIFF_HALF)
                y0, y1 = self._row_y(row, h)
                out.append(Shape("diff", min(edges), y0, max(edges), y1, ("isl", isl), row))
        return out

    def routes(self, xs: list[int]) -> list[Shape]:
        out = []
        for r in self.plan.routes:
            y = self.track_y[r.track]
            x0 = min(xs[e.col] for e in r.endpoints)
            x1 = max(xs[e.col] for e in r.endpoints)
            if r.layer == "li1":
                out.append(Shape("li", x0 - LICON_HALF, y - LI_PAD[1], x1 + LICON_HALF, y + LI_PAD[1], ("net", r.net)))
            else:
                out.append(Shape("met1", x0, y - M1_HALF, x1, y + M1_HALF, ("net", r.net)))
        return out

    def build(self) -> Rendered:
        xs, placed = self.compact()
        shapes = [s for s in placed if s.layer != "diff"] + self.diffusion(xs) + self.routes(xs)
        left = min(s.x0 - MARGIN[s.layer] for s in shapes)
        right = max(s.x1 + MARGIN[s.layer] for s in shapes)
        width = _up(right - left)
        shapes = [s.moved(-left) for s in shapes]
        xs = [x - left for x in xs]
        H = self.H
        wb = math.ceil(width / SITE) * SITE
        # rails: met1 + li strip + mcons every site
        for y, net in ((0, "VSS"), (H, "VDD")):
            shapes.append(Shape("met1", 0, y - RAIL_HALF, wb, y + RAIL_HALF, ("net", net)))
            shapes.append(Shape("li", 0, y - LI_RAIL_HALF, wb, y + LI_RAIL_HALF, ("net", net)))
            for k in range(wb // SITE):
                cx = SITE // 2 + SITE * k
                shapes.append(Shape("mcon", cx - LICON_HALF, y - LICON_HALF, cx + LICON_HALF, y + LICON_HALF))
        split = _up(((self.ntop + NDIFF_NWELL) + (self.pbot - NWELL_PDIFF)) / 2)
        sdm = self.pbot - PSDM_PDIFF
        xa, xb = -IMPLANT_EXT, wb + IMPLANT_EXT
        shapes += [
            Shape("nwell", xa, split, xb, H + IMPLANT_EXT),
            Shape("hvtp", xa, split, xb, H + IMPLANT_EXT),
            Shape("psdm", xa, sdm, xb, H + IMPLANT_EXT),
            Shape("nsdm", xa, -IMPLANT_EXT, xb, sdm),
            Shape("npc", 0, self.track_y["c0"] - NPC_EXT, wb, self.track_y[CHANNEL_NAMES[-1]] + NPC_EXT),
            Shape("boundary", 0, 0, wb, H),
        ]
        self._xs = xs
        labels = self._labels(shapes, wb, split)
        outputs = list(self.plan.outputs)
        return Rendered(self.plan.name, width, H, xs, shapes, labels,
                        reference_spice(self.netlist, self.plan.name, outputs))

    def _labels(self, shapes: list[Shape], wb: int, split: int) -> list[Label]:
        H = self.H
        labels = []
        straps = {}
        pads = {}
        for s in shapes:
            if s.layer == "li" and s.tag.startswith("strap"):
                straps[(s.tag.split()[1], s.owner[1], (s.x0 + s.x1) // 2)] = s
            if s.layer == "li" and s.tag == "gate":
                pads[(s.owner[1], (s.x0 + s.x1) // 2, (s.y0 + s.y1) // 2)] = s
        for p in self.plan.pins:
            ep = p.landing
            if ep.kind == "gate":
                y = self.track_y[self.rep.gate_tracks[(ep.col, ep.side)]]
                s = pads[(p.net, self._xs[ep.col], y)]
            else:
                s = straps[(ep.kind, p.net, self._xs[ep.col])]
            labels.append(Label(p.net, "li", (s.x0, s.y0, s.x1, s.y1)))
        labels += [
            Label("VSS", "met1", (0, -RAIL_HALF, wb, RAIL_HALF)),
            Label("VDD", "met1", (0, H - RAIL_HALF, wb, H + RAIL_HALF)),
            Label("VPB", "nwell", (0, split, wb, H)),
            Label("VNB", "sub", (0, -LI_RAIL_HALF, wb, LI_RAIL_HALF)),
        ]
        return labels


def render(plan: Plan, netlist: dict) -> Rendered:
    """Validate the plan, then lay it out. Raises RenderError if it can't."""
    b = _Builder(plan, netlist)
    return b.build()
