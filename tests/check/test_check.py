import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from cellgym.check import check
from cellgym.check.drc import Geometry, run_drc
from cellgym.check.geometry import Region
from cellgym.check.mutate import mutate
from cellgym.dsl.plan import parse
from cellgym.render.render import Shape, render

EX = Path(__file__).resolve().parents[2] / "cellgym" / "dsl" / "examples"


def _cell(name):
    nl = json.loads((EX / f"{name}.json").read_text())
    return render(parse((EX / f"{name}.plan").read_text()), nl), nl


@pytest.fixture(scope="module")
def aoi21():
    return _cell("aoi21")


# -- examples ----------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["aoi21", "half_adder"])
def test_examples_clean(name):
    cell, nl = _cell(name)
    res = check(cell, nl)
    assert res.drc_violations == [], res.summary()
    assert res.lvs_ok, res.lvs_errors
    assert res.width_um * 1000 == cell.width_nm


def test_timing_bound(aoi21):
    cell, nl = _cell("half_adder")
    best = min(check(cell, nl).ms for _ in range(5))
    assert best < 100


# -- geometry ----------------------------------------------------------------------------

def test_region_merge_and_booleans():
    a = Region.from_rects([(0, 0, 100, 100), (100, 0, 200, 100), (50, 50, 150, 300)])
    assert len(a.components()) == 1
    assert a.area() == 200 * 100 + 100 * 200
    b = Region.from_rects([(0, 0, 50, 50)])
    assert (a & b).area() == 2500
    assert (a - b).area() == a.area() - 2500
    assert a.covers((60, 60, 140, 290)) and not a.covers((0, 0, 210, 10))
    corner = Region.from_rects([(0, 0, 10, 10), (10, 10, 20, 20)])
    assert len(corner.components()) == 2     # corner touch does not connect


# -- one minimal violating shape set per rule --------------------------------------------

def R(layer, x0, y0, x1, y1):
    return (layer, (x0, y0, x1, y1))


BIG_N = R("nsdm", -2000, -2000, 4000, 4000)
BIG_LI = R("li", -2000, -2000, 4000, 4000)


def _gate(x=0, y=0, w=420, ext=300, end=200):
    """A legal nmos: vertical poly over diff (current along x)."""
    return [R("diff", x - ext, y, x + 150 + ext, y + w), R("poly", x, y - end, x + 150, y + w + end), BIG_N]


CASES = {
    "poly.1a": [R("poly", 0, 0, 140, 1000)],
    "poly.2": [R("poly", 0, 0, 150, 1000), R("poly", 350, 0, 500, 1000)],
    "difftap.1": [R("diff", 0, 0, 140, 1000), BIG_N],
    "difftap.3": [R("diff", 0, 0, 300, 300), R("diff", 560, 0, 860, 300), BIG_N],
    "nwell.1": [R("nwell", 0, 0, 800, 2000)],
    "nwell.2a": [R("nwell", 0, 0, 1000, 2000), R("nwell", 2200, 0, 3200, 2000)],
    "hvtp.1": [R("hvtp", 0, 0, 370, 1000)],
    "hvtp.2": [R("hvtp", 0, 0, 400, 1000), R("hvtp", 770, 0, 1170, 1000)],
    "npc.1": [R("npc", 0, 0, 260, 1000)],
    "npc.2": [R("npc", 0, 0, 300, 1000), R("npc", 560, 0, 860, 1000)],
    "nsdm.1": [R("nsdm", 0, 0, 400, 1000), R("nsdm", 770, 0, 1170, 1000)],
    "nsdm.2": [R("nsdm", 0, 0, 370, 1000)],
    "psdm.1": [R("psdm", 0, 0, 400, 1000), R("psdm", 770, 0, 1170, 1000)],
    "psdm.2": [R("psdm", 0, 0, 370, 1000)],
    "nsdm_psdm_overlap": [R("diff", 0, 0, 300, 300), R("nsdm", -500, -500, 800, 800), R("psdm", -500, -500, 800, 800)],
    "diff.implant": [R("diff", 0, 0, 300, 300)],
    "nsd.5a": [R("diff", 0, 0, 300, 300), R("nsdm", -100, -200, 500, 500)],
    "psd.5a": [R("diff", 0, 0, 300, 300), R("psdm", -100, -200, 500, 500), R("nwell", -1000, -1000, 1500, 1500)],
    "difftap.8": [R("diff", 0, 0, 300, 300), R("psdm", -200, -200, 500, 500), R("nwell", -150, -500, 800, 800)],
    "difftap.9": [R("diff", 0, 0, 300, 300), BIG_N, R("nwell", 0, 600, 1000, 1600)],
    "licon.1": [R("licon", 0, 0, 170, 200)],
    "MR_licon.SP.1": [R("licon", 0, 0, 170, 170), R("licon", 330, 0, 500, 170)],
    "licon.13": [R("diff", -100, -100, 270, 270), BIG_N, R("licon", 0, 0, 170, 170), R("npc", 250, -500, 600, 600)],
    "licon.13_a": [R("diff", -100, -100, 270, 270), BIG_N, R("licon", 0, 0, 170, 170), R("npc", 100, -500, 600, 600)],
    "licon.17": [R("diff", -100, -100, 270, 270), BIG_N, R("poly", -100, 100, 270, 300), R("licon", 0, 0, 170, 170)],
    "licon.5a": [R("diff", -30, -100, 270, 270), BIG_N, R("licon", 0, 0, 170, 170)],
    "licon.5c": [R("diff", -40, -40, 210, 210), BIG_N, R("licon", 0, 0, 170, 170)],
    "licon.8": [R("poly", -40, -100, 270, 270), R("licon", 0, 0, 170, 170)],
    "licon.8a": [R("poly", -60, -60, 230, 230), R("licon", 0, 0, 170, 170)],
    "licon.9": [R("poly", -80, -50, 250, 220), R("licon", 0, 0, 170, 170),
                R("diff", 370, 0, 800, 400), R("psdm", 245, -500, 1200, 1000), R("nwell", 0, -1000, 2000, 2000)],
    "licon.14": [R("poly", -80, -50, 250, 220), R("licon", 0, 0, 170, 170), R("diff", 330, 0, 800, 400), BIG_N],
    "licon.11": _gate() + [R("licon", -210, 125, -40, 295)],
    "licon.1_li": [R("li", 20, -100, 400, 400), R("licon", 0, 0, 170, 170)],
    "li.1": [R("li", 0, 0, 160, 1000)],
    "li.3": [R("li", 0, 0, 200, 1000), R("li", 360, 0, 560, 1000)],
    "li.5": [R("li", -40, -40, 210, 210), R("licon", 0, 0, 170, 170)],
    "li.6": [R("li", 0, 0, 200, 200)],
    "ct.1": [R("mcon", 0, 0, 170, 170), R("mcon", 100, 100, 270, 270), BIG_LI],
    "ct.1_a": [R("mcon", 0, 0, 160, 170), BIG_LI],
    "ct.1_b": [R("mcon", 0, 0, 180, 170), BIG_LI],
    "ct.2": [R("mcon", 0, 0, 170, 170), R("mcon", 350, 0, 520, 170), BIG_LI],
    "ct.4": [R("mcon", 0, 0, 170, 170), R("met1", -100, -100, 270, 270)],
    "m1.1": [R("met1", 0, 0, 130, 1000)],
    "m1.2": [R("met1", 0, 0, 200, 1000), R("met1", 330, 0, 530, 1000)],
    "m1.4": [R("mcon", 0, 0, 170, 170), BIG_LI],
    "791_m1.4": [R("mcon", 0, 0, 170, 170), BIG_LI, R("met1", -20, -100, 270, 270)],
    "m1.5": [R("mcon", 0, 0, 170, 170), BIG_LI, R("met1", -30, -30, 200, 200)],
    "m1.6": [R("met1", 0, 0, 200, 400)],
    "poly.4": [R("poly", 0, 0, 150, 1000), R("diff", 200, 0, 600, 500), BIG_N],
    "poly.7": _gate(ext=200),
    "poly.8": _gate(end=100),
    "poly.11": [R("diff", -300, 0, 450, 420), BIG_N, R("poly", 0, -200, 150, 620), R("poly", 150, -200, 300, 200)],
    "offgrid": [R("li", 3, 0, 503, 500)],
}


@dataclass
class _S:
    layer: str
    x0: int
    y0: int
    x1: int
    y1: int


def _drc_rules(items):
    rects = {}
    for layer, r in items:
        rects.setdefault(layer, []).append(r)
    return {v.rule for v in run_drc(Geometry(rects))}


@pytest.mark.parametrize("rule", sorted(CASES))
def test_rule_caught(rule):
    assert rule in _drc_rules(CASES[rule])


def test_legal_gate_and_contacts_clean():
    items = _gate() + [R("licon", -245, 125, -75, 295), R("li", -245, 45, -75, 375)]
    assert _drc_rules(items) == set()


def test_euclidean_corner_spacing():
    # 120 x 120 diagonal offset: 170 apart in projection terms, 169.7 corner to corner
    a = [R("li", 0, 0, 500, 500), R("li", 620, 620, 1120, 1120)]
    assert "li.3" in _drc_rules(a)
    b = [R("li", 0, 0, 500, 500), R("li", 620, 625, 1120, 1125)]
    assert "li.3" not in _drc_rules(b)


def test_check_accepts_shape_list(aoi21):
    cell, nl = aoi21
    shapes = [_S(s.layer, s.x0, s.y0, s.x1, s.y1) for s in cell.shapes]
    res = check(shapes, nl, labels=cell.labels)
    assert res.ok, res.summary()


# -- LVS ---------------------------------------------------------------------------------

def _labelled(cell, net):
    return next(lab for lab in cell.labels if lab.net == net)


def test_lvs_short(aoi21):
    cell, nl = aoi21
    a, b = _labelled(cell, "A").rect, _labelled(cell, "B").rect
    bridge = Shape("li", min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
    res = check(replace(cell, shapes=cell.shapes + [bridge]), nl)
    assert not res.lvs_ok
    assert any("short" in e and "A" in e and "B" in e for e in res.lvs_errors), res.lvs_errors


def test_lvs_open(aoi21):
    cell, nl = aoi21
    shapes = [s for s in cell.shapes if not (s.layer == "li" and s.owner == ("net", "OUT") and not s.tag)]
    assert len(shapes) == len(cell.shapes) - 1
    res = check(replace(cell, shapes=shapes), nl)
    assert not res.lvs_ok
    assert any("open" in e or "OUT" in e for e in res.lvs_errors), res.lvs_errors


def test_lvs_missing_device(aoi21):
    cell, nl = aoi21
    x = cell.xs[3]           # gate column 3 (input A): cut its poly out of the NMOS row
    shapes = []
    for s in cell.shapes:
        if s.layer == "poly" and s.x0 < x < s.x1 and s.y0 < 1000:
            shapes.append(replace(s, y0=1300))       # poly now starts above the ndiff
        else:
            shapes.append(s)
    res = check(replace(cell, shapes=shapes), nl)
    assert not res.lvs_ok
    assert any("missing device: nmos" in e for e in res.lvs_errors), res.lvs_errors


def test_lvs_wrong_w(aoi21):
    cell, nl = aoi21
    nl2 = copy.deepcopy(nl)
    nl2["devices"][0]["w"] = round(nl2["devices"][0]["w"] + 0.1, 3)
    res = check(cell, nl2)
    assert not res.lvs_ok
    assert any("wrong W" in e for e in res.lvs_errors), res.lvs_errors


def test_lvs_label_moved(aoi21):
    cell, nl = aoi21
    b = _labelled(cell, "B")
    labels = [replace(lab, rect=b.rect) if lab.net == "A" else lab for lab in cell.labels]
    res = check(replace(cell, labels=labels), nl)
    assert not res.lvs_ok


def test_mutate_deterministic(aoi21):
    cell, _ = aoi21
    m1, d1 = mutate(cell, 7)
    m2, d2 = mutate(cell, 7)
    assert d1 == d2 and m1.shapes == m2.shapes and m1.labels == m2.labels
    assert any(mutate(cell, s)[1] != d1 for s in range(5))
