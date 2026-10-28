import itertools
import json
import math
import os
import shutil
from pathlib import Path

import pytest

from cellgym.dsl.plan import parse
from cellgym.render.gds import dumps as gds_dumps
from cellgym.render.render import SPACE, render

EXAMPLES = Path(__file__).resolve().parents[2] / "cellgym" / "dsl" / "examples"
NAMES = ["aoi21", "half_adder"]


def load(name):
    return parse((EXAMPLES / f"{name}.plan").read_text()), json.loads((EXAMPLES / f"{name}.json").read_text())


@pytest.fixture(scope="module", params=NAMES)
def cell(request):
    return render(*load(request.param))


def test_deterministic():
    for name in NAMES:
        assert render(*load(name)).gds() == render(*load(name)).gds()


@pytest.mark.parametrize("name, width, sites, height", [
    ("aoi21", 2.22, 5, 4.08), ("half_adder", 5.52, 12, 4.08),
])
def test_dimensions(name, width, sites, height):
    c = render(*load(name))
    assert c.width == pytest.approx(width)
    assert c.sites == sites == math.ceil(c.width / 0.46 - 1e-9)
    assert c.height == pytest.approx(height)
    assert all(b > a for a, b in zip(c.xs, c.xs[1:]))


def test_on_grid(cell):
    g = cell.gds_cell()
    for _, *xy in g.rects:
        assert all(v % 5 == 0 for v in xy)
    raw = [(s.x0, s.y0, s.x1, s.y1) for s in cell.shapes]
    assert all(v % 5 == 0 for r in raw for v in r)


def _gap(a, b):
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0)
    return math.hypot(dx, dy), (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1)


def test_no_shorts_or_close_shapes(cell):
    """Same-layer shapes of different owners never overlap and keep the spacing."""
    by_layer = {}
    for s in cell.shapes:
        if s.layer in SPACE:
            by_layer.setdefault(s.layer, []).append(s)
    for layer, shapes in by_layer.items():
        for a, b in itertools.combinations(shapes, 2):
            if layer not in ("licon", "mcon") and a.owner == b.owner:
                continue
            dist, overlap = _gap(a, b)
            assert not overlap, (layer, a, b)
            assert dist >= SPACE[layer] - 1e-6, (layer, a, b)


def test_reference_spice(cell):
    sub = next(l for l in cell.spice.splitlines() if l.startswith(".subckt"))
    assert sub.split()[1] == cell.name and sub.split()[-4:] == ["VDD", "VSS", "VPB", "VNB"]
    for l in cell.spice.splitlines():
        if l.startswith("X"):
            assert l.split()[4] == ("VPB" if "pfet" in l else "VNB")


def test_gds_header():
    data = gds_dumps(render(*load("aoi21")).gds_cell())
    assert data[:6] == bytes([0, 6, 0, 2, 2, 88]) and data[-4:] == bytes([0, 4, 4, 0])


PDK_ROOT = Path(os.environ.get("PDK_ROOT", str(Path.home() / "pdk")))
HAVE_EDA = all(shutil.which(t) for t in ("klayout", "magic", "netgen")) and (PDK_ROOT / "sky130A").exists()


@pytest.mark.skipif(not HAVE_EDA, reason="klayout/magic/netgen or the sky130A PDK not available")
@pytest.mark.parametrize("name", NAMES)
def test_drc_lvs(name, tmp_path):
    from cellgym.groundtruth.verify import resolve, verify
    r = verify(*resolve(name, None), tmp_path, PDK_ROOT)
    assert r.drc_items == [], r.drc_items
    assert r.magic_items == [], r.magic_items     # Magic full DRC, tapless rules excluded
    assert r.lvs_ok, r.lvs_lines
    assert r.ok
