import json
from pathlib import Path

import pytest

from cellgym.baselines.common import stratified
from cellgym.baselines.euler import euler_solve
from cellgym.baselines.naive import naive_columns, naive_plan, naive_solve
from cellgym.bench.evaluate import diffusion_breaks, evaluate
from cellgym.dsl.plan import parse
from conftest import HAVE_EDA, PDK_ROOT

EXAMPLES = Path(__file__).resolve().parents[2] / "cellgym" / "dsl" / "examples"


def example(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text())


@pytest.mark.parametrize("name", ["aoi21", "half_adder"])
def test_examples_no_sharing(name):
    nl = example(name)
    r = naive_solve(nl)
    plan = parse(r.text)
    gates = [c for c in plan.columns if c.kind == "g"]
    assert len(gates) == len(nl["devices"]) // 2
    assert r.breaks == diffusion_breaks(plan) == len(gates) - 1
    assert evaluate(r.text, nl).ok
    assert r.sites >= euler_solve(nl).sites


def test_separator_only_where_nets_would_merge():
    cols = naive_columns(example("aoi21"))
    for i, c in enumerate(cols):
        if c[0] == "sd" and c[1] is None:
            prev, nxt = cols[i - 1], cols[i + 1]
            assert prev[1] == nxt[1] or prev[2] == nxt[2]


def test_deterministic():
    assert naive_plan(example("half_adder")) == naive_plan(example("half_adder"))


def test_sample_core_cells_pass(core_cells):
    for cid in stratified(core_cells, 24):
        ev = evaluate(naive_plan(core_cells[cid]), core_cells[cid])
        assert ev.ok, (cid, ev.stage, ev.detail)


@pytest.mark.slow
def test_all_core_cells_pass(core_cells):
    bad = []
    for cid, nl in core_cells.items():
        ev = evaluate(naive_plan(nl), nl)
        if not ev.ok:
            bad.append((cid, ev.stage, ev.detail))
    assert not bad, bad[:5]


@pytest.mark.skipif(not HAVE_EDA, reason="klayout/magic/netgen or the sky130A PDK not available")
@pytest.mark.parametrize("name", ["aoi21", "cg_3i5d_1eddaea2"])
def test_drc_lvs(name, tmp_path, core_cells):
    from cellgym.groundtruth.verify import verify
    nl = example(name) if (EXAMPLES / f"{name}.json").exists() else core_cells[name]
    plan, netlist = tmp_path / f"{name}.plan", tmp_path / f"{name}.json"
    plan.write_text(naive_plan(nl))
    netlist.write_text(json.dumps(nl))
    r = verify(plan, netlist, tmp_path / "run", PDK_ROOT)
    assert r.ok, (r.drc_items, r.magic_items, r.lvs_lines)
