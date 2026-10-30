"""Shared helpers: canonical hashes, truth tables, function matching, cell loading."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "bench"
SPLITS_FILE = BENCH_DIR / "splits_v0.json"
EULER_WIDTHS_FILE = BENCH_DIR / "euler_widths_v0.json"
SPLITS = ("train", "val", "test_id", "test_ood")


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def with_hash(doc: dict) -> dict:
    """doc plus "sha256" over its canonical serialization (without that key)."""
    body = {k: v for k, v in doc.items() if k != "sha256"}
    return body | {"sha256": sha256(body)}


def hash_ok(doc: dict) -> bool:
    return doc.get("sha256") == sha256({k: v for k, v in doc.items() if k != "sha256"})


def load_cells(cells_dir: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(cells_dir).glob("*.json")):
        nl = json.loads(f.read_text())
        if "devices" in nl:
            out[nl["id"]] = nl
    return out


def dump_rows(doc: dict, list_key: str) -> str:
    """JSON with one line per element of doc[list_key] (or per value if a dict)."""
    head = {k: v for k, v in doc.items() if k != list_key}
    items = doc[list_key]
    lines = ["{"]
    for k, v in head.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v, sort_keys=True)},")
    if isinstance(items, dict):
        body = [f"    {json.dumps(k)}: {json.dumps(v, sort_keys=True)}" for k, v in items.items()]
        lines += [f"  {json.dumps(list_key)}: {{", ",\n".join(body), "  }"]
    else:
        body = [f"    {json.dumps(v, sort_keys=True)}" for v in items]
        lines += [f"  {json.dumps(list_key)}: [", ",\n".join(body), "  ]"]
    return "\n".join(lines) + "\n}\n"


# -- truth tables ---------------------------------------------------------------------

def truth_table(f: Callable[[list[bool]], bool], k: int, perm: tuple[int, ...] | None = None) -> int:
    """Bit r = f at row r. Argument j of f reads core input perm[j] (input i = bit i of r)."""
    perm = perm or tuple(range(k))
    tt = 0
    for r in range(1 << k):
        if f([bool((r >> perm[j]) & 1) for j in range(k)]):
            tt |= 1 << r
    return tt


def match(f: Callable[[list[bool]], bool], k: int, cells: dict[str, dict]) -> list[tuple[str, tuple[int, ...]]]:
    """Core cells with k inputs whose OUT function equals f under some input
    renaming; [(id, perm)] with the first matching perm per cell."""
    by_tt: dict[int, list[str]] = {}
    for cid, nl in cells.items():
        if len(nl["inputs"]) == k:
            by_tt.setdefault(int(nl["function"], 16), []).append(cid)
    found: dict[str, tuple[int, ...]] = {}
    for perm in itertools.permutations(range(k)):
        for cid in by_tt.get(truth_table(f, k, perm), []):
            found.setdefault(cid, perm)
    return sorted(found.items())
