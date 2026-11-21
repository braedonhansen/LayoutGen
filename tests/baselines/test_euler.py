import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cellgym.baselines.euler import euler_plan, euler_solve
from cellgym.check import check
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


def test_all_core_cells_fast_check_clean(cells):
    bad = []
    for nl in cells:
        res = check(render(parse(euler_plan(nl)), nl), nl)
        if not res.ok:
            bad.append((nl["id"], res.summary()))
    assert not bad, bad[:5]


def test_poly7_unequal_widths_shared_sd(cells):
    """NMOS row A(0.84) | n1 | B(0.84) | VSS | C(0.42): the s/d between B and C
    is drawn at W=0.84 up to 0.075 before C's poly, so B's diffusion must still
    extend 0.25 past B (poly.7): the B-C poly gap grows to >= 0.325."""
    nl = next(c for c in cells if c["id"] == "cg_3i3d_29e68bd3")
    c = render(parse(euler_plan(nl)), nl)
    res = check(c, nl)
    assert not [v for v in res.drc_violations if v.rule == "poly.7"], res.summary()
    assert res.ok, res.summary()
    nmos_gates = sorted({(s.x0, s.x1) for s in c.shapes if s.layer == "poly" and s.x1 - s.x0 == 150
                         and s.y0 < 1000})
    gaps = [b[0] - a[1] for a, b in zip(nmos_gates, nmos_gates[1:])]
    assert max(gaps) >= 325, gaps
    # widest diffusion strip right of the 0.84 gate reaches 0.25 past it
    wide = [s for s in c.shapes if s.layer == "diff" and s.y0 < 1000 and s.y1 - s.y0 == 840]
    for g0, g1 in nmos_gates[:2]:
        right = max(s.x1 for s in wide if s.x0 <= g1 < s.x1 or s.x0 == g1)
        assert right - g1 >= 250


PDK_ROOT = Path(os.environ.get("PDK_ROOT", str(Path.home() / "pdk")))
HAVE_EDA = all(shutil.which(t) for t in ("klayout", "magic", "netgen")) and (PDK_ROOT / "sky130A").exists()


@pytest.mark.skipif(not HAVE_EDA, reason="klayout/magic/netgen or the sky130A PDK not available")
@pytest.mark.parametrize("name", ["aoi21", "half_adder", "cg_2i2d_23e88f1d", "cg_3i3d_29e68bd3"])
def test_drc_lvs(name, tmp_path, cells):
    from cellgym.groundtruth.verify import verify
    nl = example(name) if (EXAMPLES / f"{name}.json").exists() else next(c for c in cells if c["id"] == name)
    plan = tmp_path / f"{name}.plan"
    netlist = tmp_path / f"{name}.json"
    plan.write_text(euler_plan(nl))
    netlist.write_text(json.dumps(nl))
    r = verify(plan, netlist, tmp_path / "run", PDK_ROOT)
    assert r.drc_items == [], r.drc_items
    assert r.magic_items == [], r.magic_items
    assert r.lvs_ok, r.lvs_lines
    assert r.ok


# -- channel tracks per plan ---------------------------------------------------

def _placements(nl):
    from cellgym.baselines.euler import MAX_PLACEMENTS, _columns, _Placer
    return [_columns(s) for s in _Placer(nl).sequences(MAX_PLACEMENTS)]


def test_route_best_default_tracks_unchanged():
    from cellgym.baselines.common import route_best
    nl = example("half_adder")
    r = route_best(_placements(nl), nl)
    assert r.text == euler_plan(nl) and "tracks" not in r.text
    assert r.tracks == 3 and r.height_um == pytest.approx(render(parse(r.text), nl).height)


@pytest.mark.parametrize("name", ["aoi21", "half_adder"])
@pytest.mark.parametrize("tracks", [4, 5])
def test_route_best_more_tracks(name, tracks):
    from cellgym.baselines.common import route_best
    from cellgym.bench.evaluate import evaluate
    nl = example(name)
    base = route_best(_placements(nl), nl)
    r = route_best(_placements(nl), nl, tracks=tracks)
    assert r.tracks == tracks and f"\ntracks {tracks}\n" in r.text
    assert r.height_um == pytest.approx(base.height_um + 0.34 * (tracks - 3))
    ev = evaluate(r.text, nl)
    assert ev.ok, ev.detail
    assert ev.tracks == tracks and ev.height_nm == round(r.height_um * 1000)


def test_route_min_tracks_small_cell():
    from cellgym.baselines.common import BaselineError, route_best, route_min_tracks
    from cellgym.bench.evaluate import evaluate
    from cellgym.dsl.template import MIN_CHANNEL_TRACKS
    nl = example("aoi21")
    cols = _placements(nl)
    r = route_min_tracks(cols, nl)
    assert MIN_CHANNEL_TRACKS <= r.tracks <= 3
    assert parse(r.text).tracks == r.tracks
    ev = evaluate(r.text, nl)                      # fast checker: DRC + LVS
    assert ev.ok, ev.detail
    assert r.height_um == pytest.approx(route_best(cols, nl).height_um - 0.34 * (3 - r.tracks))
    for t in range(MIN_CHANNEL_TRACKS, r.tracks):  # minimal: nothing below it routes
        with pytest.raises(BaselineError):
            route_best(cols, nl, t)
    assert route_min_tracks(cols, nl, t_min=4).tracks == 4


def test_route_min_tracks_raises_when_nothing_routes():
    from cellgym.baselines.common import BaselineError, route_min_tracks
    nl = example("half_adder")                     # SPEC section 1: no legal routing with 2 tracks
    with pytest.raises(BaselineError):
        route_min_tracks(_placements(nl), nl, t_max=2)


@pytest.mark.parametrize("name", ["aoi21", "half_adder"])
def test_route_min_area_is_the_smallest_area(name):
    from cellgym.baselines.common import BaselineError, route_best, route_min_area, route_min_tracks
    from cellgym.bench.evaluate import evaluate
    nl = example(name)
    cols = _placements(nl)
    r = route_min_area(cols, nl)
    first = route_min_tracks(cols, nl)
    assert r.area_um2 == pytest.approx(r.width_um * r.height_um) and r.area_um2 > 0
    assert r.tracks >= first.tracks and r.area_um2 <= first.area_um2 + 1e-9
    assert parse(r.text).tracks == r.tracks
    assert evaluate(r.text, nl).ok
    for t in range(first.tracks, first.tracks + 3):   # nothing in the searched window is smaller
        try:
            x = route_best(cols, nl, t)
        except BaselineError:
            continue
        assert r.area_um2 <= x.area_um2 + 1e-9
        if x.area_um2 == pytest.approx(r.area_um2):
            assert r.tracks <= t                      # ties: fewer tracks


def test_route_min_area_search_window(monkeypatch):
    """A taller channel wins only when it routes narrow enough; the search stops when width stalls."""
    from cellgym.baselines import common as C
    widths = {2: 4.0, 3: 3.0, 4: 2.9, 5: 2.8, 6: 0.1}
    calls = []

    def fake(placements, netlist, tracks=3):
        calls.append(tracks)
        if tracks not in widths:
            raise C.BaselineError("no-routing", "x")
        h = 2.0 + 0.34 * tracks
        return C.Routed("", 0, widths[tracks], 0, False, 0, tracks, h, widths[tracks] * h)

    monkeypatch.setattr(C, "route_best", fake)
    r = C.route_min_area([[]], {})
    assert r.tracks == 3 and calls == [1, 2, 3, 4, 5]   # T0 = 2, then T0+1..T0+3 only; 3.0*3.02 is smallest
    widths[3] = 4.0
    calls.clear()
    assert C.route_min_area([[]], {}).tracks == 2 and calls == [1, 2, 3]   # width did not shrink: stop
    with pytest.raises(C.BaselineError):
        C.route_min_area([[]], {}, t_min=7)


def test_route_best_fills_area():
    from cellgym.baselines.common import route_best
    nl = example("aoi21")
    r = route_best(_placements(nl), nl)
    assert r.area_um2 == pytest.approx(r.width_um * r.height_um)
    assert r.text == euler_plan(nl)
