"""Static layout renders: one SVG per cell plus an index.html gallery.

    python -m cellgym.viewer.static <plans_dir> --out out/gallery/<name> [--cells out/cells]
                                    [--splits bench/splits_v0.json] [--only ID ...]

Each plan is parsed, validated, rendered and fast-checked
(cellgym.bench.evaluate). The SVG draws the Rendered shapes (nwell, diff,
poly, licon, li, mcon, met1), pin labels, and every fast-checker DRC violation
as a red outlined box with its rule name. index.html (a local file, no
external assets) lists cell, split, width and pass/fail with a thumbnail.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from cellgym.bench.common import SPLITS_FILE, load_cells
from cellgym.bench.evaluate import Evaluation, evaluate, read_plan
from cellgym.render.render import Rendered

# drawing order and style: (layer, fill, opacity, stroke)
STYLE = [
    ("nwell", "#e9d8a6", 0.35, "#c9a227"),
    ("diff", "#3fa34d", 0.55, "#2b7a35"),
    ("poly", "#d1495b", 0.75, "#a33545"),
    ("licon", "#222222", 0.9, "none"),
    ("li", "#4361ee", 0.40, "#3046b8"),
    ("mcon", "#111111", 0.9, "none"),
    ("met1", "#7b2cbf", 0.30, "#5a1f8f"),
]
FONT = "font-family=\"Helvetica, Arial, sans-serif\""
SCALE = 0.12      # px per nm
PAD = 30


def svg(r: Rendered, violations=(), title: str | None = None) -> str:
    x_min = min(s.x0 for s in r.shapes if s.layer != "nwell")
    x_max = max(s.x1 for s in r.shapes if s.layer != "nwell")
    y_min, y_max = -240, r.height_nm + 240
    W = (x_max - x_min) * SCALE + 2 * PAD
    H = (y_max - y_min) * SCALE + 2 * PAD + 28

    def X(x):
        return PAD + (x - x_min) * SCALE

    def Y(y):
        return PAD + (y_max - y) * SCALE

    label = html.escape(title or r.name)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" '
             f'height="{H:.0f}" role="img" aria-label="{label} layout" {FONT}>',
             f'<title>{label}</title>',
             f'<rect width="{W:.0f}" height="{H:.0f}" fill="#ffffff"/>']
    for layer, fill, op, stroke in STYLE:
        for s in r.shapes:
            if s.layer != layer:
                continue
            x0, x1 = max(s.x0, x_min - 190), min(s.x1, x_max + 190)
            parts.append(f'<rect x="{X(x0):.1f}" y="{Y(s.y1):.1f}" width="{(x1 - x0) * SCALE:.1f}" '
                         f'height="{(s.y1 - s.y0) * SCALE:.1f}" fill="{fill}" fill-opacity="{op}"'
                         + (f' stroke="{stroke}" stroke-width="0.8"' if stroke != "none" else "") + "/>")
    for lb in r.labels:
        if lb.net in ("VPB", "VNB"):
            continue
        x = (lb.rect[0] + lb.rect[2]) / 2
        y = (lb.rect[1] + lb.rect[3]) / 2
        parts.append(f'<text x="{X(x):.1f}" y="{Y(y) + 4:.1f}" font-size="12" font-weight="600" '
                     f'fill="#111" text-anchor="middle">{html.escape(lb.net)}</text>')
    for v in violations:
        x0, y0, x1, y1 = v.box
        w, h = max((x1 - x0) * SCALE, 3), max((y1 - y0) * SCALE, 3)
        parts.append(f'<rect class="drc" x="{X(x0):.1f}" y="{Y(y1):.1f}" width="{w:.1f}" height="{h:.1f}" '
                     f'fill="none" stroke="#e00000" stroke-width="1.6"/>')
        parts.append(f'<text x="{X(x0):.1f}" y="{Y(y1) - 3:.1f}" font-size="10" fill="#e00000">'
                     f'{html.escape(v.rule)}</text>')
    lx = PAD
    for layer, fill, op, _ in STYLE:
        ly = H - 16
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" fill="{fill}" '
                     f'fill-opacity="{max(op, 0.5)}"/>')
        parts.append(f'<text x="{lx + 16}" y="{ly}" font-size="11.5" fill="#424a53">{layer}</text>')
        lx += 30 + 7 * len(layer)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def index_html(name: str, rows: list[tuple[Evaluation, str, bool]]) -> str:
    n_pass = sum(ev.ok for ev, _, _ in rows)
    out = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           f"<title>{html.escape(name)} gallery</title>",
           "<style>body{font-family:Helvetica,Arial,sans-serif;margin:16px;color:#1f2328;background:#fff}"
           "table{border-collapse:collapse}td,th{padding:4px 8px;border-bottom:1px solid #ddd;"
           "text-align:left;vertical-align:middle}.pass{color:#1a7f37}.fail{color:#cf222e;font-weight:600}"
           "img{max-height:90px;max-width:260px}code{font-size:12px}</style></head><body>",
           f"<h1>{html.escape(name)}</h1>",
           f"<p>{len(rows)} cells, {n_pass} pass (validate + render + fast check). "
           "Click a thumbnail for the full SVG; red boxes are fast-checker DRC violations.</p>",
           "<table><thead><tr><th>cell</th><th>split</th><th>width</th><th>result</th>"
           "<th>layout</th></tr></thead><tbody>"]
    for ev, split, has_svg in rows:
        width = f"{ev.sites} sites ({ev.width_nm / 1000:.3f} um)" if ev.width_nm else "-"
        res = ('<span class="pass">pass</span>' if ev.ok else
               f'<span class="fail">fail: {html.escape(ev.stage)}</span><br><code>{html.escape(ev.detail[:160])}</code>')
        img = (f'<a href="{ev.cell}.svg"><img src="{ev.cell}.svg" alt="{ev.cell} layout" loading="lazy"></a>'
               if has_svg else "")
        out.append(f"<tr><td><code>{ev.cell}</code></td><td>{split}</td><td>{width}</td><td>{res}</td>"
                   f"<td>{img}</td></tr>")
    out.append("</tbody></table></body></html>")
    return "\n".join(out) + "\n"


def build(plans_dir: Path, out: Path, cells_dir: Path, splits_file: Path | None, name: str,
          only: list[str] | None = None) -> list[Evaluation]:
    cells = load_cells(cells_dir)
    split = {}
    if splits_file and Path(splits_file).exists():
        split = {r["id"]: r["split"] for r in json.loads(Path(splits_file).read_text())["cells"]}
    ids = [c for c in sorted(cells) if not only or any(c.startswith(o) for o in only)]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in ids:
        ev = evaluate(read_plan(plans_dir, cid), cells[cid], keep=True)
        has_svg = ev.rendered is not None
        if has_svg:
            viol = ev.checked.drc_violations if ev.checked else []
            (out / f"{cid}.svg").write_text(svg(ev.rendered, viol), newline="\n")
        ev.rendered = ev.checked = None          # free the shapes
        rows.append((ev, split.get(cid, "-"), has_svg))
    (out / "index.html").write_text(index_html(name, rows), newline="\n")
    return [ev for ev, _, _ in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", type=Path, default=Path("out/cells"))
    ap.add_argument("--splits", type=Path, default=SPLITS_FILE)
    ap.add_argument("--name", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="cell ids or id prefixes")
    a = ap.parse_args(argv)
    evs = build(a.plans_dir, a.out, a.cells, a.splits, a.name or a.out.name, a.only)
    print(f"{len(evs)} cells, {sum(e.ok for e in evs)} pass -> {a.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
