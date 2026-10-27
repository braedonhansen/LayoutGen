"""Switch-level check of the transistor netlist (independent of the tree)."""

from __future__ import annotations

from .enumerate import pdn_function
from .transistors import Cell


def _components(devices, vec):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for d in devices:
        on = vec[d.gate] if d.type == "nmos" else not vec[d.gate]
        find(d.drain), find(d.source)
        if on:
            parent[find(d.drain)] = find(d.source)
    return find


def check_cell(cell: Cell) -> list[str]:
    """Return a list of problems (empty = pass)."""
    k = len(cell.inputs)
    f = pdn_function(cell.network.tree, k)
    problems = []
    for r in range(1 << k):
        vec = {name: bool((r >> i) & 1) for i, name in enumerate(cell.inputs)}
        find = _components(cell.devices, vec)
        to_vdd = find("OUT") == find("VDD")
        to_vss = find("OUT") == find("VSS")
        expect_out = not ((f >> r) & 1)
        if to_vdd == to_vss:
            problems.append(f"row {r}: OUT {'shorted' if to_vdd else 'floating'}")
        elif to_vdd != expect_out:
            problems.append(f"row {r}: OUT={int(to_vdd)}, expected {int(expect_out)}")
    return problems
