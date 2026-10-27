"""Transistor-level schematic SVGs for cell JSONs.

    python -m cellgym.netlists.schematic --open            # all cells, gallery in a browser
    python -m cellgym.netlists.schematic nand3 "P(C,S(A,B))" --open
    python -m cellgym.netlists.schematic --cells out/cells/<id>.json --out out/schematics

Cells are generated into --cells first if it does not exist yet. Select cells by
id (or id prefix), canonical PDN tree (S = series, P = parallel), or inv,
nandN, norN.

Placement comes from the device list itself: each network (PMOS between VDD
and OUT, NMOS between OUT and VSS) is decomposed into series/parallel blocks
using only device terminals, so every device is drawn exactly once on its own
nets. Series = stacked vertically, parallel = side by side, recursively.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from xml.sax.saxutils import escape

from .enumerate import INPUT_NAMES, parse_tree
from .writers import _fmt

# geometry (px)
LEAF_W, LEAF_CX, LEAF_H = 108, 62, 64    # one transistor; CX = terminal x within it
GAP = 18                                  # between parallel branches
PAD = 12                                  # bus to branch top/bottom
MARGIN = 20
HEADER_H = 70
RAIL_GAP = 16                             # rail to network
OUT_GAP = 22                              # network to OUT node

STYLE = """
.bg{fill:#fff}
.wire{stroke:#222;stroke-width:1.5;fill:none;stroke-linecap:square}
.rail{stroke:#222;stroke-width:3;fill:none}
.nmos{stroke:#1f5fbf;fill:none;stroke-width:1.8}
.pmos{stroke:#b8322a;fill:none;stroke-width:1.8}
.dot{fill:#222}
text{font-family:Helvetica,Arial,sans-serif;fill:#222}
.hdr{font-size:14px;font-weight:bold}
.sub{font-size:12px;fill:#444}
.gate{font-size:13px;font-weight:bold}
.w{font-size:10px;fill:#555}
.dev{font-size:8px;fill:#999}
.net{font-size:9px;fill:#888;font-style:italic}
.port{font-size:13px;font-weight:bold}
"""


# ------------------------------------------------- series/parallel decomposition

def _components(devs, top, bot):
    """Group devices connected through nets other than top/bot (parallel branches)."""
    parent = list(range(len(devs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, d in enumerate(devs):
        for n in (d["drain"], d["source"]):
            if n in (top, bot):
                continue
            if n in seen:
                parent[find(i)] = find(seen[n])
            else:
                seen[n] = i
    groups = {}
    for i in range(len(devs)):
        groups.setdefault(find(i), []).append(devs[i])
    return sorted(groups.values(), key=lambda g: devs.index(g[0]))


def _reach(devs, start, avoid):
    """Nets reachable from start through device channels, never entering avoid."""
    seen, todo = {start}, [start]
    while todo:
        n = todo.pop()
        for d in devs:
            a, b = d["drain"], d["source"]
            for x, y in ((a, b), (b, a)):
                if x == n and y != avoid and y not in seen:
                    seen.add(y)
                    todo.append(y)
    return seen


def decompose(devs, top, bot):
    """Two-terminal network -> ("L", dev, top, bot) | ("P", kids, top, bot)
    | ("S", kids, nets) with nets[i], nets[i+1] the terminals of kids[i]."""
    if len(devs) == 1:
        d = devs[0]
        if {d["drain"], d["source"]} != {top, bot}:
            raise ValueError(f"{d['name']} is not between {top} and {bot}")
        return ("L", d, top, bot)
    groups = _components(devs, top, bot)
    if len(groups) > 1:
        return ("P", [decompose(g, top, bot) for g in groups], top, bot)
    # series: the cut net nearest top splits the stack in two
    inner = sorted({n for d in devs for n in (d["drain"], d["source"])} - {top, bot})
    best = None
    for v in inner:
        side = _reach(devs, top, v)
        if bot not in side and (best is None or len(side) < len(best[1])):
            best = (v, side)
    if best is None:
        raise ValueError(f"network {top}-{bot} is not series-parallel")
    v, side = best
    upper = [d for d in devs if d["drain"] in side or d["source"] in side]
    lower = [d for d in devs if d not in upper]
    kids, nets = [], [top]
    for part, a, b in ((upper, top, v), (lower, v, bot)):
        k = decompose(part, a, b)
        if k[0] == "S":
            kids += k[1]
            nets += k[2][1:]
        else:
            kids.append(k)
            nets.append(b)
    return ("S", kids, nets)


# --------------------------------------------------------------------- sizing

def size(b):
    """(width, height, terminal x) of a block."""
    if b[0] == "L":
        return LEAF_W, LEAF_H, LEAF_CX
    sz = [size(k) for k in b[1]]
    if b[0] == "S":
        cx = max(s[2] for s in sz)
        return cx + max(s[0] - s[2] for s in sz), sum(s[1] for s in sz), cx
    w = sum(s[0] for s in sz) + GAP * (len(sz) - 1)
    first = sz[0][2]
    last = w - sz[-1][0] + sz[-1][2]
    return w, max(s[1] for s in sz) + 2 * PAD, (first + last) // 2


# -------------------------------------------------------------------- drawing

class Canvas:
    def __init__(self):
        self.wires = []     # (x1, y1, x2, y2), axis aligned
        self.pins = []      # transistor terminal points (one wire end each)
        self.fets = []
        self.texts = []
        self.dots = set()

    def wire(self, x1, y1, x2, y2):
        if (x1, y1) != (x2, y2):
            self.wires.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    def text(self, x, y, s, cls, anchor="start"):
        a = "" if anchor == "start" else f' text-anchor="{anchor}"'
        self.texts.append(f'<text x="{x}" y="{y}" class="{cls}"{a}>{escape(s)}</text>')

    def fet(self, d, x, y):
        """Transistor with terminals at (x, y) and (x, y + LEAF_H)."""
        k = d["type"]
        c = x - 14
        s = [f'<g class="fet {k}" data-device="{escape(d["name"])}" data-type="{k}">',
             f'<path class="{k}" d="M{x} {y}V{y + 18}H{c}M{x} {y + LEAF_H}V{y + LEAF_H - 18}H{c}'
             f'M{c} {y + 13}V{y + LEAF_H - 13}M{c - 6} {y + 16}V{y + LEAF_H - 16}"/>']
        g = y + LEAF_H // 2
        if k == "pmos":
            s.append(f'<circle class="{k}" cx="{c - 10}" cy="{g}" r="4"/>')
            s.append(f'<path class="{k}" d="M{c - 14} {g}H{x - 40}"/>')
        else:
            s.append(f'<path class="{k}" d="M{c - 6} {g}H{x - 40}"/>')
        s.append("</g>")
        self.fets.append("".join(s))
        self.pins += [(x, y), (x, y + LEAF_H)]
        self.text(x - 43, g + 5, d["gate"], "gate", "end")
        self.text(x + 6, g - 2, _fmt(d["w"]), "w")
        self.text(x + 6, g + 9, d["name"], "dev")

    def junctions(self):
        """Dots where three or more wire ends meet (T and cross junctions)."""
        pts = {(w[0], w[1]) for w in self.wires} | {(w[2], w[3]) for w in self.wires}
        out = set(self.dots)
        for p in pts:
            deg = self.pins.count(p)
            for x1, y1, x2, y2 in self.wires:
                if p in ((x1, y1), (x2, y2)):
                    deg += 1
                elif x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                    deg += 2
            if deg >= 3:
                out.add(p)
        return sorted(out)


def draw(cv, b, x, y):
    """Draw block b with its top terminal at (x, y)."""
    if b[0] == "L":
        cv.fet(b[1], x, y)
        return
    if b[0] == "S":
        for i, k in enumerate(b[1]):
            draw(cv, k, x, y)
            y += size(k)[1]
            if i < len(b[1]) - 1:
                cv.dots.add((x, y))
                cv.text(x + 5, y - 4, b[2][i + 1], "net")
        return
    w, h, cx = size(b)
    left = x - cx
    xs = []
    for k in b[1]:
        kw, kh, kcx = size(k)
        tx = left + kcx
        xs.append(tx)
        cv.wire(tx, y, tx, y + PAD)
        draw(cv, k, tx, y + PAD)
        cv.wire(tx, y + PAD + kh, tx, y + h)
        left += kw + GAP
    lo, hi = min(xs + [x]), max(xs + [x])
    cv.wire(lo, y, hi, y)
    cv.wire(lo, y + h, hi, y + h)


def expression(tree: str) -> str:
    """'P(S(A,B),C)' -> 'OUT = ¬(A·B + C)' (PDN conducts -> OUT low)."""
    def ex(t, inside_s):
        if t[0] == "L":
            return INPUT_NAMES[t[1]]
        if t[0] == "S":
            return "·".join(ex(c, True) for c in t[1])
        s = " + ".join(ex(c, False) for c in t[1])
        return f"({s})" if inside_s else s
    return f"OUT = ¬({ex(parse_tree(tree), False)})"


def render(cell: dict) -> str:
    """One cell JSON dict -> standalone SVG text."""
    devs = cell["devices"]
    pun = decompose([d for d in devs if d["type"] == "pmos"], "VDD", "OUT")
    pdn = decompose([d for d in devs if d["type"] == "nmos"], "OUT", "VSS")
    (pw, ph, pcx), (nw, nh, ncx) = size(pun), size(pdn)

    x = MARGIN + 16 + max(pcx, ncx)
    right = x + max(pw - pcx, nw - ncx, 56)
    vdd = HEADER_H + 14
    yp = vdd + RAIL_GAP
    yo = yp + ph + OUT_GAP
    yn = yo + OUT_GAP
    vss = yn + nh + RAIL_GAP

    cv = Canvas()
    cv.wire(x, vdd, x, yp)
    draw(cv, pun, x, yp)
    cv.wire(x, yp + ph, x, yn)
    cv.wire(x, yo, right, yo)
    draw(cv, pdn, x, yn)
    cv.wire(x, yn + nh, x, vss)

    sub = [f"{len(cell['inputs'])} inputs, {len(devs)} devices",
           f"PDN depth {cell.get('pdn_depth', '?')}, PUN depth {cell.get('pun_depth', '?')}"]
    if "function" in cell:
        sub.append(f"f=0x{cell['function']}")
    lines = [cell["id"]]
    if "canonical_tree" in cell:
        lines.append(expression(cell["canonical_tree"]))
    lines.append(" | ".join(sub))
    width = max(right + 44 + MARGIN, 8 * max(len(s) for s in lines[1:]) + 2 * MARGIN, 300)
    height = vss + MARGIN

    cv.text(MARGIN, 22, lines[0], "hdr")
    for i, s in enumerate(lines[1:]):
        cv.text(MARGIN, 40 + 15 * i, s, "sub")
    cv.text(MARGIN, vdd - 5, "VDD", "port")
    cv.text(MARGIN, vss - 5, "VSS", "port")
    cv.text(right + 4, yo + 5, "OUT", "port")
    cv.dots.add((x, yo))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" data-cell="{escape(cell["id"])}">',
           f"<style>{STYLE}</style>",
           f'<rect class="bg" width="{width}" height="{height}"/>',
           f'<path class="rail" d="M{MARGIN} {vdd}H{width - MARGIN}M{MARGIN} {vss}H{width - MARGIN}"/>']
    out.append('<path class="wire" d="' + "".join(
        f"M{x1} {y1}L{x2} {y2}" for x1, y1, x2, y2 in cv.wires) + '"/>')
    out += cv.fets
    # rails are drawn separately; count them as wires only for junction dots
    cv.wires += [(MARGIN, vdd, width - MARGIN, vdd), (MARGIN, vss, width - MARGIN, vss)]
    out += [f'<circle class="dot" cx="{px}" cy="{py}" r="3"/>' for px, py in cv.junctions()]
    out += cv.texts
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------ CLI

GALLERY = """<!doctype html>
<html><head><meta charset="utf-8"><title>CellGym schematics</title>
<style>
body{{font-family:Helvetica,Arial,sans-serif;margin:16px;background:#f4f4f4;color:#222}}
.grid{{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}}
figure{{margin:0;padding:8px;background:#fff;border:1px solid #ddd}}
figure img{{display:block;max-width:100%}}
figcaption{{font-size:12px;font-family:monospace;margin-top:4px}}
</style></head><body>
<h1>CellGym schematics ({n} cells)</h1>
<div class="grid">
{figs}
</div></body></html>
"""


def gallery(ids: list[str]) -> str:
    figs = "\n".join(f'<figure><a href="{escape(i)}.svg"><img src="{escape(i)}.svg" '
                     f'alt="{escape(i)}" loading="lazy"></a>'
                     f"<figcaption>{escape(i)}</figcaption></figure>" for i in ids)
    return GALLERY.format(n=len(ids), figs=figs)


def cell_files(path: Path) -> list[Path]:
    """A single cell JSON, or every cell JSON in a directory (index.json skipped)."""
    if path.is_file():
        return [path]
    return sorted(p for p in path.glob("*.json") if p.name != "index.json")


def _alias(name: str) -> str | None:
    """inv / nandN / norN -> canonical PDN tree."""
    name = name.lower()
    if name == "inv":
        return "A"
    for prefix, op in (("nand", "S"), ("nor", "P")):
        n = name[len(prefix):]
        if name.startswith(prefix) and n.isdigit() and 2 <= int(n) <= len(INPUT_NAMES):
            return f"{op}({','.join(INPUT_NAMES[:int(n)])})"
    return None


def matches(cell: dict, select: list[str]) -> bool:
    if not select:
        return True
    for s in select:
        tree = _alias(s) or s.replace(" ", "")
        if cell["id"].startswith(s) or cell["canonical_tree"] == tree:
            return True
    return False


def render_dir(cells: Path, out: Path, select: list[str] = ()) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    ids = []
    for p in cell_files(cells):
        cell = json.loads(p.read_text())
        if not matches(cell, select):
            continue
        (out / f"{cell['id']}.svg").write_text(render(cell), encoding="utf-8", newline="\n")
        ids.append(cell["id"])
    if cells.is_dir() and len(ids) > 1:
        (out / "index.html").write_text(gallery(ids), encoding="utf-8", newline="\n")
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("select", nargs="*",
                    help="cell ids/prefixes, canonical trees or inv/nandN/norN (default: all)")
    ap.add_argument("--cells", type=Path, default=Path("out/cells"),
                    help="directory of cell JSONs, or one cell JSON (generated if missing)")
    ap.add_argument("--out", type=Path, default=Path("out/schematics"))
    ap.add_argument("--open", action="store_true", help="open the result in a browser")
    a = ap.parse_args(argv)
    t0 = time.time()
    if not a.cells.exists():
        from .enumerate import Params
        from .generate import generate
        from .transistors import Sizing
        print(f"generating cells into {a.cells} ...")
        generate(a.cells, Params(), Sizing())
    ids = render_dir(a.cells, a.out, a.select)
    if not ids:
        raise SystemExit(f"no cells match {a.select}")
    page = a.out / ("index.html" if len(ids) > 1 and a.cells.is_dir() else f"{ids[0]}.svg")
    print(f"{len(ids)} schematics -> {page}; {time.time() - t0:.1f}s")
    if a.open:
        import webbrowser
        webbrowser.open(page.resolve().as_uri())


if __name__ == "__main__":
    main()
