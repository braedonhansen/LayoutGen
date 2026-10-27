import json
from pathlib import Path

import pytest

from cellgym.dsl.plan import PlanSyntaxError, dumps, parse
from cellgym.dsl.template import layout
from cellgym.dsl.validate import validate

EXAMPLES = Path(__file__).resolve().parents[2] / "cellgym" / "dsl" / "examples"


def load(name):
    return (EXAMPLES / f"{name}.plan").read_text(), json.loads((EXAMPLES / f"{name}.json").read_text())


def check(text, netlist):
    return validate(parse(text), netlist)


def mutate(text, old, new):
    assert old in text, old
    return text.replace(old, new, 1)


# -- template ------------------------------------------------------------------

@pytest.mark.parametrize("stack, height, n_tracks, p_tracks", [
    (1, 2.72, 1, 2), (2, 4.08, 3, 4), (3, 5.10, 4, 6), (4, 6.12, 5, 8), (5, 7.14, 6, 10),
])
def test_layout_by_stack(stack, height, n_tracks, p_tracks):
    lay = layout(0.42 * stack, 0.63 * stack)
    assert lay.height == pytest.approx(height)
    assert (lay.n_tracks, lay.p_tracks) == (n_tracks, p_tracks)
    assert lay.has_track("c2") and not lay.has_track("c3")
    assert lay.has_track(f"p{p_tracks - 1}") and not lay.has_track(f"p{p_tracks}")


# -- parser --------------------------------------------------------------------

@pytest.mark.parametrize("name", ["aoi21", "half_adder"])
def test_round_trip(name):
    text, _ = load(name)
    plan = parse(text)
    assert parse(dumps(plan)).as_dict() == plan.as_dict()


def test_parse_half_adder_structure():
    plan = parse(load("half_adder")[0])
    assert plan.inputs == ["A", "B"] and plan.outputs == ["S", "C"]
    assert len(plan.columns) == 16
    assert sum(c.kind == "g" for c in plan.columns) == 7
    c9 = plan.columns[9]
    assert c9.nets == {"nmos": "VSS", "pmos": "S"} and c9.contact == {"nmos": False, "pmos": True}
    assert c9.stage == "S"
    assert [str(e) for e in plan.routes[0].endpoints] == ["nmos:2", "pmos:0", "gate:5", "gate:10"]
    assert plan.pins[0].track is None


@pytest.mark.parametrize("old, new, msg", [
    ("col 3 g ", "col 4 g ", "column index"),
    ("route OUT", "rout OUT", "unknown statement"),
    ("nmos:2 pmos:4", "nmos:2 pmos4", "bad endpoint"),
    ("col 1 g  n=C ", "col 1 g  n=C* ", "'*' on a gate column"),
    ("route OUT li1 ", "route OUT met2 ", "bad layer"),
    ("pin A gate:3 c0", "pin A gate:3 x0", "bad track"),
    ("col 0 sd n=VSS*  p=VDD*", "col 0 sd n=VSS* st=x", "both n= and p="),
])
def test_syntax_errors(old, new, msg):
    text, _ = load("aoi21")
    with pytest.raises(PlanSyntaxError, match=msg):
        parse(mutate(text, old, new))


# -- validator: the examples -----------------------------------------------------

def test_aoi21_valid():
    rep = check(*load("aoi21"))
    assert rep.ok, str(rep)
    assert rep.layout.height == pytest.approx(4.08)
    assert rep.gate_tracks == {(1, ""): "c2", (3, ""): "c0", (5, ""): "c2"}
    assert rep.near_misses == 2


def test_half_adder_valid():
    rep = check(*load("half_adder"))
    assert rep.ok, str(rep)
    assert rep.gate_tracks == {(1, ""): "c2", (3, ""): "c1", (5, ""): "c0", (7, ""): "c1",
                               (10, ""): "c0", (12, ""): "c2", (14, ""): "c1"}
    assert rep.near_misses == 7


# -- validator: broken plans -----------------------------------------------------

HA_BROKEN = [
    ("col 2  sd n=x* ", "col 2  sd n=x ", {"contact-missing", "endpoint"}),
    # B on the NMOS side makes column 1 a split column: the A route needs a side, and the new B contact is unconnected
    ("col 1  g  n=A     p=A", "col 1  g  n=B     p=A", {"device-extra", "device-missing", "gate-side", "gate-untracked", "open"}),
    ("route C  li1  c2 nmos:6 pmos:8\n", "", {"open"}),
    ("route C  li1  c2", "route C  li1  n0", {"li-off-channel"}),
    ("route S  li1  c1", "route S  li1  c0", {"overlap"}),
    ("col 0  sd n=VSS* ", "col 0  sd n=VDD* ", {"rail-row", "device-extra", "device-missing"}),
    ("pin A gate:1\n", "", {"pin-missing"}),
    ("pin A gate:1", "pin A gate:1 c0", {"gate-track-conflict"}),
    ("route p1 met1 p0", "route p1 met1 p9", {"track-missing"}),
    ("route p1 met1 p0 pmos:11 pmos:15", "route p1 met1 n0 pmos:11 pmos:15", {"endpoint-track"}),
    ("route x  met1 c0 nmos:2 pmos:0 gate:5 gate:10", "route x  met1 c0 nmos:2 pmos:0 gate:5 gate:10 nmos:4",
     {"endpoint"}),
    ("pin S nmos:11", "pin S gate:7", {"pin-landing"}),
    ("pin C nmos:6", "pin C nmos:6\npin VDD pmos:4", {"pin-net"}),
    ("route p1 met1 p0", "route VDD met1 p1 pmos:4 pmos:13\nroute p1 met1 p0", {"rail-routed"}),
]


@pytest.mark.parametrize("old, new, codes", HA_BROKEN)
def test_half_adder_broken(old, new, codes):
    text, netlist = load("half_adder")
    rep = check(mutate(text, old, new), netlist)
    assert rep.codes() == codes, str(rep)


def test_unrouted_gate_needs_pin_track():
    text, netlist = load("aoi21")
    rep = check(mutate(text, "pin A gate:3 c0", "pin A gate:3"), netlist)
    assert rep.codes() == {"gate-untracked"}


def test_issue_points_at_line():
    text, netlist = load("half_adder")
    rep = check(mutate(text, "route C  li1  c2", "route C  li1  n0"), netlist)
    (issue,) = rep.issues
    assert text.splitlines()[issue.line - 1].startswith("route C")
    assert str(issue).startswith(f"line {issue.line} [li-off-channel]")


def test_split_gate_column():
    # INV whose NMOS and PMOS gates are driven by different nets is not a real
    # cell; this only exercises split-column parsing and contact ordering.
    netlist = {
        "inputs": ["A", "B"], "ports": ["A", "B", "Y", "VDD", "VSS"],
        "nets": ["A", "B", "Y", "VDD", "VSS"],
        "devices": [
            {"type": "nmos", "gate": "A", "drain": "Y", "source": "VSS", "w": 0.42},
            {"type": "pmos", "gate": "B", "drain": "Y", "source": "VDD", "w": 0.63},
        ],
    }
    base = ("cell split\nin A B\nout Y\n"
            "col 0 sd n=VSS* p=VDD*\ncol 1 g n=A p=B\ncol 2 sd n=Y* p=Y*\n"
            "route Y li1 c1 nmos:2 pmos:2\npin Y nmos:2\n")
    ok = base + "pin A gate:1n c0\npin B gate:1p c2\n"
    assert check(ok, netlist).ok, str(check(ok, netlist))
    flipped = base + "pin A gate:1n c2\npin B gate:1p c0\n"
    assert check(flipped, netlist).codes() == {"split-order"}
    no_side = base + "pin A gate:1 c0\npin B gate:1p c2\n"
    assert "gate-side" in check(no_side, netlist).codes()
