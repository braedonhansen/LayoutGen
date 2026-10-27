"""SPICE and JSON writers."""

from __future__ import annotations

import json

from .enumerate import depth, pdn_function, tree_str
from .transistors import Cell, Sizing


def _fmt(x: float) -> str:
    s = f"{x:.3f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def out_function_hex(cell: Cell) -> str:
    """Truth table of OUT (= NOT PDN conducts); bit r = row r, input A = LSB of r."""
    k = len(cell.inputs)
    rows = 1 << k
    f = pdn_function(cell.network.tree, k)
    out = ~f & ((1 << rows) - 1)
    return format(out, "x").zfill(max(1, rows // 4))


def spice(cell: Cell) -> str:
    """Subcircuit. Port order: inputs (A..), OUT, VDD, VSS."""
    ports = " ".join(cell.inputs + ["OUT", "VDD", "VSS"])
    lines = [f"* {cell.id}: OUT = NOT({tree_str(cell.network.tree)}) [PDN as S/P tree]",
             f".subckt {cell.id} {ports}"]
    for d in cell.devices:
        lines.append(f"X{d.name} {d.drain} {d.gate} {d.source} {d.bulk} {d.model} "
                     f"w={_fmt(d.w)} l={_fmt(d.l)}")
    lines.append(".ends")
    return "\n".join(lines) + "\n"


def cell_json(cell: Cell, sizing: Sizing, fit: bool = True) -> dict:
    t = cell.network.tree
    return {
        "id": cell.id,
        "inputs": cell.inputs,
        "ports": cell.inputs + ["OUT", "VDD", "VSS"],
        "function": out_function_hex(cell),
        "function_note": "hex truth table of OUT; bit r = row r, input A = LSB of r",
        "canonical_tree": tree_str(t),
        "pdn_depth": depth(t),
        "pun_depth": depth(t, dual=True),
        "read_once": cell.network.read_once,
        "n_devices": len(cell.devices),
        "max_w_n": cell.max_w_n,
        "max_w_p": cell.max_w_p,
        "fits": fit,
        "devices": [d.as_dict() for d in cell.devices],
        "nets": cell.nets,
    }


def dumps(obj) -> str:
    return json.dumps(obj, indent=2) + "\n"
