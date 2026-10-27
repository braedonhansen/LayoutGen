import json
import xml.etree.ElementTree as ET
from collections import Counter

import pytest

from cellgym.netlists.enumerate import Network, Params, parse_tree
from cellgym.netlists.generate import generate
from cellgym.netlists.schematic import decompose, expression, main, render, render_dir
from cellgym.netlists.transistors import Sizing, build_cell
from cellgym.netlists.writers import cell_json

SVG = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def cells_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cells")
    generate(d, Params(), Sizing())
    return d


@pytest.fixture(scope="module")
def rendered(cells_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("schematics")
    return out, render_dir(cells_dir, out)


def fets(svg_text):
    root = ET.fromstring(svg_text)
    return [g for g in root.iter(f"{SVG}g") if "fet" in g.get("class", "").split()]


def test_renders_all_cells_and_gallery(cells_dir, rendered):
    out, ids = rendered
    assert len(ids) == 323
    html = (out / "index.html").read_text(encoding="utf-8")
    assert html.count("<figure>") == 323
    for i in ids:
        assert f'src="{i}.svg"' in html


def test_symbol_counts_match_devices(cells_dir, rendered):
    out, ids = rendered
    for i in ids:
        cell = json.loads((cells_dir / f"{i}.json").read_text())
        gs = fets((out / f"{i}.svg").read_text(encoding="utf-8"))
        assert len(gs) == cell["n_devices"] == len(cell["devices"]), i
        # every device drawn exactly once, with its own type
        assert sorted(g.get("data-device") for g in gs) == sorted(
            d["name"] for d in cell["devices"]), i
        assert Counter(g.get("data-type") for g in gs) == Counter(
            d["type"] for d in cell["devices"]), i


def test_deterministic(cells_dir, rendered, tmp_path):
    out, ids = rendered
    again = render_dir(cells_dir, tmp_path)
    assert again == ids
    for f in out.iterdir():
        assert f.read_bytes() == (tmp_path / f.name).read_bytes(), f.name


def test_single_file(cells_dir, tmp_path):
    p = next(p for p in sorted(cells_dir.glob("*.json")) if p.name != "index.json")
    ids = render_dir(p, tmp_path)
    assert [f.name for f in tmp_path.iterdir()] == [f"{ids[0]}.svg"]


def _leaves(b):
    return [b] if b[0] == "L" else [x for k in b[1] for x in _leaves(k)]


def test_decompose_uses_netlist_terminals():
    cell = cell_json(build_cell(Network(parse_tree("P(S(A,B),S(A,C))"), 3)), Sizing())
    nmos = [d for d in cell["devices"] if d["type"] == "nmos"]
    b = decompose(nmos, "OUT", "VSS")
    assert b[0] == "P" and [k[0] for k in b[1]] == ["S", "S"]
    for leaf in _leaves(b):
        d = leaf[1]
        assert {leaf[2], leaf[3]} == {d["drain"], d["source"]}
    # a non-series-parallel bridge is rejected rather than drawn wrongly
    bridge = [dict(name=n, drain=a, source=s) for n, a, s in (
        ("M1", "T", "x"), ("M2", "T", "y"), ("M3", "x", "y"), ("M4", "x", "B"),
        ("M5", "y", "B"))]
    with pytest.raises(ValueError):
        decompose(bridge, "T", "B")


def test_header_and_parse():
    cell = cell_json(build_cell(Network(parse_tree("P(C,S(A,B))"), 3)), Sizing())
    assert expression(cell["canonical_tree"]) == "OUT = ¬(C + A·B)"
    assert expression("S(A,P(B,C))") == "OUT = ¬(A·(B + C))"
    root = ET.fromstring(render(cell))
    texts = [t.text for t in root.iter(f"{SVG}text")]
    assert cell["id"] in texts and "OUT" in texts and "VDD" in texts and "VSS" in texts
    assert root.get("data-cell") == cell["id"]


def test_select_by_alias_tree_and_id(cells_dir, tmp_path):
    ids = render_dir(cells_dir, tmp_path / "a", ["nand2", "nor3", "P(C, S(A,B))"])
    trees = {json.loads((cells_dir / f"{i}.json").read_text())["canonical_tree"] for i in ids}
    assert trees == {"S(A,B)", "P(A,B,C)", "P(C,S(A,B))"}
    assert (tmp_path / "a" / "index.html").exists()
    assert render_dir(cells_dir, tmp_path / "b", [ids[0][:12]])
    with pytest.raises(SystemExit):
        main(["no-such-cell", "--cells", str(cells_dir), "--out", str(tmp_path / "c")])
