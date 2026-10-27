import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cellgym.baselines.euler import euler_plan, euler_solve
from cellgym.dsl.plan import parse
from cellgym.dsl.validate import validate
from cellgym.render.render import render

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "cellgym" / "dsl" / "examples"


@pytest.fixture(scope="module")
def cells(tmp_path_factory):
    d = ROOT / "out" / "cells"
    if not any(d.glob("cg_*.json")):
        d = tmp_path_factory.mktemp("cells")
        subprocess.run([sys.executable, "-m", "cellgym.netlists.generate", "--out", str(d)],
                       cwd=ROOT, check=True, capture_output=True)
    nls = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    return [nl for nl in nls if "devices" in nl]


def example(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text())


def test_all_core_cells_validate_and_render(cells):
    assert len(cells) == 323
    bad = []
    for nl in cells:
        plan = parse(euler_plan(nl))
        rep = validate(plan, nl)
        if not rep.ok:
            bad.append((nl["id"], str(rep)))
            continue
        assert plan.name == nl["id"]
        assert render(plan, nl).width > 0
    assert not bad


def test_deterministic():
    for name in ("aoi21", "half_adder"):
        assert euler_plan(example(name)) == euler_plan(example(name))


def test_aoi21_no_breaks():
    r = euler_solve(example("aoi21"))
    assert r.breaks == 0 and r.row_breaks == 0
    assert r.sites <= 5


def test_half_adder():
    r = euler_solve(example("half_adder"))
    assert r.breaks == 1        # the PMOS row needs one break (SPEC.md)
    assert validate(parse(r.text), example("half_adder")).ok
    assert r.sites <= 12


PDK_ROOT = Path(os.environ.get("PDK_ROOT", str(Path.home() / "pdk")))
HAVE_EDA = all(shutil.which(t) for t in ("klayout", "magic", "netgen")) and (PDK_ROOT / "sky130A").exists()


@pytest.mark.skipif(not HAVE_EDA, reason="klayout/magic/netgen or the sky130A PDK not available")
@pytest.mark.parametrize("name", ["aoi21", "half_adder", "cg_2i2d_23e88f1d"])
def test_drc_lvs(name, tmp_path, cells):
    from cellgym.groundtruth.verify import verify
    nl = example(name) if (EXAMPLES / f"{name}.json").exists() else next(c for c in cells if c["id"] == name)
    plan = tmp_path / f"{name}.plan"
    netlist = tmp_path / f"{name}.json"
    plan.write_text(euler_plan(nl))
    netlist.write_text(json.dumps(nl))
    r = verify(plan, netlist, tmp_path / "run", PDK_ROOT)
    assert r.drc_items == [], r.drc_items
    assert r.lvs_ok, r.lvs_lines
