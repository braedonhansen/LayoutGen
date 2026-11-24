"""Rename the stage nets of multi-stage probe cells (a zero-training-cost naming test).

probegen names the outputs of non-sink stages x1, x2, ..: tokens the placement policy never saw. In
training a gate is always a letter A-Z and a diffusion net is always OUT, a rail, n<k> or p<k>. A stage
net is both a gate and a diffusion net, so no spelling is fully in distribution; this writes copies of a
probe set under each candidate spelling so the same model can be sampled on all of them:

  letter   x_i -> the next unused letters after the primary inputs (looks like an input)
  net      x_i -> n<31+i> (looks like an internal nmos-side net)

Cell ids are kept, so the area references of the source set apply unchanged (same devices, same
connectivity): the cache file is copied to out/arearef/<name>.json.

  python -m cellgym.llm.renameprobe --src out/probe/stage2 --style letter --out out/probe/stage2_letter
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from cellgym.llm.fmt import LETTERS

STYLES = ("letter", "net")


def stage_nets(nl: dict) -> list[str]:
    return sorted((x for x in nl["nets"] if re.fullmatch(r"x\d+", x)), key=lambda x: int(x[1:]))


def name_map(nl: dict, style: str) -> dict[str, str]:
    xs = stage_nets(nl)
    if style == "letter":
        free = [c for c in LETTERS if c not in nl["inputs"]]
        return dict(zip(xs, free))
    if style == "net":
        used = {x for x in nl["nets"]}
        m = {x: f"n{31 + i}" for i, x in enumerate(xs)}
        assert not used & set(m.values())
        return m
    raise ValueError(style)


def rename_text(text: str, m: dict[str, str]) -> str:
    if not m:
        return text
    return re.sub(r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, m)) + r")(?![A-Za-z0-9_])", lambda g: m[g[1]], text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--style", choices=STYLES, required=True)
    ap.add_argument("--arearef", type=Path, default=Path("out/arearef"))
    a = ap.parse_args(argv)
    (a.out / "cells").mkdir(parents=True, exist_ok=True)
    (a.out / "plans").mkdir(exist_ok=True)
    n = 0
    for f in sorted((a.src / "cells").glob("*.json")):
        t = f.read_text()
        m = name_map(json.loads(t), a.style)
        n += bool(m)
        (a.out / "cells" / f.name).write_text(rename_text(t, m))
        p = a.src / "plans" / (f.stem + ".plan")
        if p.exists():
            (a.out / "plans" / p.name).write_text(rename_text(p.read_text(), m))
    for x in ("index.json", "manifest.json"):
        if (a.src / x).exists():
            shutil.copy(a.src / x, a.out / x)
    ref = a.arearef / f"{a.src.name}.json"
    if ref.exists():
        shutil.copy(ref, a.arearef / f"{a.out.name}.json")
    (a.out / "rename.txt").write_text(f"renameprobe --src {a.src} --style {a.style}: {n} cells renamed\n")
    print(f"{a.out}: {n} cells with stage nets renamed ({a.style})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
