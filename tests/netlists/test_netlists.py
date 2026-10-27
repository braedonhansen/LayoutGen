import dataclasses
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from cellgym.netlists.enumerate import (
    Params, canonical, depth, enumerate_networks, irredundant, num_inputs,
    parse_tree, relabel, shapes, tree_str, Network,
)
from cellgym.netlists.generate import generate
from cellgym.netlists.selfcheck import check_cell
from cellgym.netlists.transistors import Sizing, build_cell
from cellgym.netlists.writers import spice


@pytest.fixture(scope="module")
def core():
    return enumerate_networks(Params())


def test_unpruned_series_parallel_counts():
    assert [len(v) for v in shapes(5, None).values()] == [1, 2, 4, 10, 24]


def test_read_once_pdn_depth_only():
    ro = enumerate_networks(Params(extra_devices=0), read_once_pdn_only=True)
    c = Counter(n.n_inputs for n in ro)
    assert [c[k] for k in range(1, 7)] == [1, 2, 4, 9, 19, 43]


def test_core_counts(core):
    ro = Counter(n.n_inputs for n in core if n.read_once)
    assert [ro[k] for k in range(1, 7)] == [1, 2, 4, 8, 14, 20]
    assert sum(ro.values()) == 49
    assert sum(1 for n in core if not n.read_once) == 274
    assert len(core) == 323
    for n in core:
        assert depth(n.tree) <= 3 and depth(n.tree, dual=True) <= 3
        assert n.n_devices <= min(n.n_inputs + 2, 8)


def test_canonical_invariant_under_renaming(core):
    rng = random.Random(1)
    for n in rng.sample(core, 60):
        k = n.n_inputs
        for _ in range(3):
            perm = list(range(k))
            rng.shuffle(perm)
            assert canonical(relabel(n.tree, dict(enumerate(perm))), k) == n.tree


def test_irredundancy_examples():
    assert irredundant(parse_tree("P(S(A,B),S(A,C))"), 3)
    assert not irredundant(parse_tree("S(A,P(A,B))"), 2)
    assert not irredundant(parse_tree("P(A,S(A,B))"), 2)
    assert irredundant(parse_tree("S(A,B)"), 2)


def test_parse_roundtrip(core):
    for n in core[::10]:
        assert parse_tree(tree_str(n.tree)) == n.tree
        assert num_inputs(n.tree) == n.n_inputs


def test_selfcheck_passes_all_core(core):
    for n in core:
        assert check_cell(build_cell(n)) == [], tree_str(n.tree)


def test_selfcheck_catches_broken_netlist():
    cell = build_cell(Network(parse_tree("S(A,B)"), 2))
    assert check_cell(cell) == []
    # drop a PMOS: OUT floats for some inputs
    broken = dataclasses.replace(
        cell, devices=[d for d in cell.devices if d.name != "MP1"])
    assert check_cell(broken)
    # miswire an NMOS gate: wrong function
    devs = [dataclasses.replace(d, gate="A") if d.name == "MN2" else d
            for d in cell.devices]
    assert check_cell(dataclasses.replace(cell, devices=devs))


def test_sizing_nand3_nor3():
    s = Sizing()
    nand3 = build_cell(Network(parse_tree("S(A,B,C)"), 3), s)
    for d in nand3.devices:
        if d.type == "nmos":
            assert d.w == pytest.approx(3 * 0.42)
        else:
            assert d.w == pytest.approx(0.63)
    nor3 = build_cell(Network(parse_tree("P(A,B,C)"), 3), s)
    for d in nor3.devices:
        if d.type == "pmos":
            assert d.w == pytest.approx(3 * 0.63)
        else:
            assert d.w == pytest.approx(0.42)
    # mixed: A*(B+C): PDN longest path through every NMOS is 2
    ao = build_cell(Network(parse_tree("S(A,P(B,C))"), 3), s)
    assert [d.w for d in ao.devices if d.type == "nmos"] == pytest.approx([0.84] * 3)


def test_deterministic_naming():
    c = build_cell(Network(parse_tree("S(A,B,C)"), 3))
    assert [(d.name, d.drain, d.source) for d in c.devices if d.type == "nmos"] == [
        ("MN1", "OUT", "n1"), ("MN2", "n1", "n2"), ("MN3", "n2", "VSS")]
    assert c.nets == ["A", "B", "C", "OUT", "VDD", "VSS", "n1", "n2"]
    assert c.id == build_cell(Network(parse_tree("S(A,B,C)"), 3)).id


def test_spice_format():
    txt = spice(build_cell(Network(parse_tree("S(A,B)"), 2)))
    assert ".subckt " in txt and txt.split("\n")[1].endswith("A B OUT VDD VSS")
    assert re.search(r"^XMN1 OUT A n1 VSS sky130_fd_pr__nfet_01v8 w=0.84 l=0.15$", txt, re.M)
    assert re.search(r"^XMP1 OUT A VDD VDD sky130_fd_pr__pfet_01v8_hvt w=0.63 l=0.15$", txt, re.M)


def test_generate_small_is_deterministic(tmp_path):
    p = Params(max_inputs=3)
    a = generate(tmp_path / "a", p, Sizing())
    b = generate(tmp_path / "b", p, Sizing())
    assert a["n_cells"] == b["n_cells"] > 7
    for f in (tmp_path / "a").iterdir():
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes()


def test_fit_filter(tmp_path):
    s = generate(tmp_path, Params(max_inputs=3), Sizing(max_diffusion_width=1.0))
    assert s["n_skipped_not_fitting"] > 0 and s["max_w_p"] <= 1.0


NETGEN = shutil.which("netgen")
SETUP = Path.home() / "pdk/sky130A/libs.tech/netgen/sky130A_setup.tcl"


@pytest.mark.skipif(not (NETGEN and SETUP.exists()),
                    reason="netgen or sky130A_setup.tcl not found (run inside the WSL cellgym env)")
def test_netgen_parses_and_matches_self(tmp_path):
    cell = build_cell(Network(parse_tree("P(S(A,B),S(A,C))"), 3))
    sp = tmp_path / f"{cell.id}.sp"
    sp.write_text(spice(cell))
    sp2 = tmp_path / "copy.sp"  # netgen refuses to compare within one file
    sp2.write_text(spice(cell))
    res = subprocess.run(
        [NETGEN, "-batch", "lvs", f"{sp} {cell.id}", f"{sp2} {cell.id}", str(SETUP),
         str(tmp_path / "comp.out")],
        capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert "Circuits match uniquely" in res.stdout, res.stdout[-800:]
