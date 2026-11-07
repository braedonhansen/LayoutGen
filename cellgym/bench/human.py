"""Human reference: sky130_fd_sc_hd cells matched to core cells by logic function.

    python -m cellgym.bench.human [--pdk-root ~/pdk] [--cells out/cells]
                                  [--plans euler=out/plans --plans naive=out/plans_naive]
                                  [--out bench/human_v0.md] [--json out/bench/human.json]

Reads the hd liberty (tt_025C_1v80) function strings of single-output
combinational `_1` cells (no ff / latch / statetable / three_state), computes
their truth tables and matches them to core cells by function up to input
permutation (cellgym.bench.common.match). Width in sites comes from the hd LEF
`SIZE` (0.46 um site). For each --plans directory the matched cell's plan is
rendered for our width (sites) and height. hd cells are 2.72 um tall; ours
are sized per cell (3.06 to 5.10 um), so widths compare like for like only
within a row height and area is a secondary, caveated number.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

from .common import load_cells, match

SITE_UM = 0.46
HD = "sky130A/libs.ref/sky130_fd_sc_hd"


# -- liberty --------------------------------------------------------------------------

def _block(text: str, start: int) -> tuple[str, int]:
    """Body of the {...} group whose '{' is at or after start; (body, end index)."""
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


_CELL = re.compile(r'\bcell\s*\(\s*"?([\w]+)"?\s*\)\s*\{')
_PIN = re.compile(r'(?<![\w_])pin\s*\(\s*"?([\w\[\]]+)"?\s*\)\s*\{')
_ATTR = lambda name: re.compile(rf'\b{name}\s*:\s*"?([^";]+)"?\s*;')


def _top_level(body: str) -> str:
    """body with nested groups removed (attributes of this group only)."""
    out, depth = [], 0
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def liberty_cells(text: str) -> dict[str, dict]:
    """name -> {"inputs": [...], "outputs": {pin: function or None}, "sequential": bool}."""
    cells = {}
    pos = 0
    while True:
        m = _CELL.search(text, pos)
        if not m:
            break
        body, pos = _block(text, m.end() - 1)
        inputs, outputs = [], {}
        p = 0
        while True:
            pm = _PIN.search(body, p)
            if not pm:
                break
            pbody, p = _block(body, pm.end() - 1)
            attrs = _top_level(pbody)
            d = _ATTR("direction").search(attrs)
            f = _ATTR("function").search(attrs)
            if d and d.group(1).strip() == "input":
                inputs.append(pm.group(1))
            elif d and d.group(1).strip() == "output":
                outputs[pm.group(1)] = f.group(1).strip() if f else None
                if _ATTR("three_state").search(attrs):
                    outputs[pm.group(1)] = None
        seq = bool(re.search(r"\b(ff|latch|ff_bank|latch_bank|statetable)\s*\(", body))
        cells[m.group(1)] = {"inputs": inputs, "outputs": outputs, "sequential": seq}
    return cells


# -- boolean expressions (liberty syntax) ---------------------------------------------

_TOK = re.compile(r"\s*(?:([A-Za-z_][\w\[\]]*)|([01])|(.))")


def _tokens(expr: str) -> list[tuple[str, str]]:
    out = []
    for m in _TOK.finditer(expr):
        if m.group(1):
            out.append(("id", m.group(1)))
        elif m.group(2):
            out.append(("const", m.group(2)))
        elif m.group(3) and not m.group(3).isspace():
            out.append(("op", m.group(3)))
    return out


def compile_function(expr: str):
    """Liberty function -> callable(dict name->bool). Precedence high to low:
    postfix ' and prefix !, ^, AND (& * or juxtaposition), OR (| +)."""
    toks = _tokens(expr)
    i = 0

    def peek():
        return toks[i] if i < len(toks) else ("end", "")

    def take():
        nonlocal i
        i += 1
        return toks[i - 1]

    def primary():
        k, v = take()
        if k == "id":
            node = lambda env, v=v: env[v]
        elif k == "const":
            node = lambda env, v=v: v == "1"
        elif (k, v) == ("op", "("):
            node = orx()
            if take() != ("op", ")"):
                raise ValueError(f"missing ) in {expr!r}")
        elif (k, v) == ("op", "!"):
            inner = primary()
            node = lambda env, a=inner: not a(env)
        else:
            raise ValueError(f"unexpected {v!r} in {expr!r}")
        while peek() == ("op", "'"):
            take()
            node = (lambda a: lambda env: not a(env))(node)
        return node

    def xorx():
        node = primary()
        while peek() == ("op", "^"):
            take()
            node = (lambda a, b: lambda env: a(env) != b(env))(node, primary())
        return node

    def andx():
        node = xorx()
        while True:
            k, v = peek()
            if (k, v) in (("op", "&"), ("op", "*")):
                take()
            elif not (k in ("id", "const") or (k, v) in (("op", "("), ("op", "!"))):
                return node
            node = (lambda a, b: lambda env: a(env) and b(env))(node, xorx())

    def orx():
        node = andx()
        while peek() in (("op", "|"), ("op", "+")):
            take()
            node = (lambda a, b: lambda env: a(env) or b(env))(node, andx())
        return node

    f = orx()
    if i != len(toks):
        raise ValueError(f"trailing tokens in {expr!r}")
    return f


# -- LEF ------------------------------------------------------------------------------

def lef_sizes(text: str) -> dict[str, tuple[float, float]]:
    out = {}
    for m in re.finditer(r"MACRO\s+(\S+)(.*?)END\s+\1", text, re.S):
        s = re.search(r"SIZE\s+([\d.]+)\s+BY\s+([\d.]+)", m.group(2))
        if s:
            out[m.group(1)] = (float(s.group(1)), float(s.group(2)))
    return out


# -- matching -------------------------------------------------------------------------

def match_library(lib: dict[str, dict], sizes: dict, cells: dict[str, dict]) -> list[dict]:
    """One row per matched core cell. A function realized by several core cells
    (repeated-input variants) maps to the read-once one, else the fewest devices;
    several hd cells with one function (inv_1, clkinv_1, ...) share a row and the
    narrowest gives hd_sites."""
    rows: dict[str, dict] = {}
    for name, c in sorted(lib.items()):
        if not name.endswith("_1") or c["sequential"] or len(c["outputs"]) != 1:
            continue
        (out_pin, fn), = c["outputs"].items()
        if fn is None or not c["inputs"] or name not in sizes:
            continue
        f = compile_function(fn)
        ins = c["inputs"]
        hits = match(lambda x: f(dict(zip(ins, x))), len(ins), cells)
        if not hits:
            continue
        cid, perm = min(hits, key=lambda h: (not cells[h[0]]["read_once"], cells[h[0]]["n_devices"], h[0]))
        w, h = sizes[name]
        short = name.replace("sky130_fd_sc_hd__", "")
        row = rows.setdefault(cid, {"hd_cells": [], "core_id": cid, "function": f"{out_pin} = {fn}",
                                    "canonical_tree": cells[cid]["canonical_tree"],
                                    "mapping": {ins[j]: cells[cid]["inputs"][perm[j]] for j in range(len(ins))},
                                    "other_core_ids": [x for x, _ in hits if x != cid],
                                    "hd_width_um": w, "hd_height_um": h, "hd_sites": round(w / SITE_UM)})
        if short not in row["hd_cells"]:
            row["hd_cells"].append(short)
        if w < row["hd_width_um"]:
            row["hd_width_um"], row["hd_sites"] = w, round(w / SITE_UM)
    return sorted(rows.values(), key=lambda r: (len(cells[r["core_id"]]["inputs"]), r["hd_cells"][0]))


def _our_widths(pairs: list[dict], plans: dict[str, Path], cells: dict[str, dict]) -> None:
    from cellgym.dsl.plan import parse
    from cellgym.render.render import render
    for p in pairs:
        for label, d in plans.items():
            f = Path(d) / f"{p['core_id']}.plan"
            if f.exists():
                r = render(parse(f.read_text()), cells[p["core_id"]])
                p[f"{label}_sites"] = r.sites
                p["our_height_um"] = r.height


def markdown(pairs: list[dict], labels: list[str], n_considered: int) -> str:
    lines = ["# Human reference: sky130_fd_sc_hd vs CellGym baselines (v0)", "",
             f"{len(pairs)} core cells match single-output combinational `_1` hd cells "
             f"({n_considered} considered) by logic function up to input permutation "
             "(liberty tt_025C_1v80 `function`, LEF `SIZE`). Where several core cells "
             "realize one function (repeated-input variants), the read-once one is used. Widths in "
             "0.46 um sites. Heights differ: every hd cell is 2.72 um tall; ours are sized "
             "per cell (column `our h`), so the comparison is width at different heights, "
             "not area.", ""]
    cols = ["hd cell(s)", "function", "core id", "hd sites"] + [f"{l} sites" for l in labels] + ["our h um"]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for p in pairs:
        fn = p["function"].replace("|", "\\|")
        row = [", ".join(p["hd_cells"]), f"`{fn}`", p["core_id"],
               str(p["hd_sites"])] + [str(p.get(f"{l}_sites", "-")) for l in labels] + \
              [f"{p['our_height_um']:.2f}" if "our_height_um" in p else "-"]
        lines.append("| " + " | ".join(row) + " |")
    if pairs and labels:
        lines.append("")
        hd = sum(p["hd_sites"] for p in pairs) / len(pairs)
        lines.append(f"Mean width over the {len(pairs)} matches: hd {hd:.2f} sites"
                     + "".join(f", {l} {sum(p[f'{l}_sites'] for p in pairs) / len(pairs):.2f}"
                               for l in labels if all(f"{l}_sites" in p for p in pairs)) + ".")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdk-root", type=Path, default=Path.home() / "pdk")
    ap.add_argument("--lib", type=Path, default=None)
    ap.add_argument("--lef", type=Path, default=None)
    ap.add_argument("--cells", type=Path, default=Path("out/cells"))
    ap.add_argument("--plans", action="append", default=[], help="label=dir (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("bench/human_v0.md"))
    ap.add_argument("--json", type=Path, default=Path("out/bench/human.json"))
    a = ap.parse_args(argv)
    hd = a.pdk_root / HD
    lib = a.lib or Path(sorted(glob.glob(str(hd / "lib" / "*tt_025C_1v80.lib")))[0])
    lef = a.lef or hd / "lef" / "sky130_fd_sc_hd.lef"
    cells = load_cells(a.cells)
    libcells = liberty_cells(lib.read_text())
    considered = sum(1 for n, c in libcells.items() if n.endswith("_1") and not c["sequential"]
                     and len(c["outputs"]) == 1 and next(iter(c["outputs"].values())))
    pairs = match_library(libcells, lef_sizes(lef.read_text()), cells)
    plans = dict(x.split("=", 1) for x in a.plans)
    _our_widths(pairs, {k: Path(v) for k, v in plans.items()}, cells)
    md = markdown(pairs, list(plans), considered)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, newline="\n")
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(pairs, indent=1) + "\n", newline="\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
