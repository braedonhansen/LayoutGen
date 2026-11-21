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
    (1, 3.06, 2, 2), (2, 4.08, 3, 4), (3, 5.10, 4, 6), (4, 6.12, 5, 8), (5, 7.48, 7, 10),
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


def test_met1_pads_adjacent_tracks_one_column():
    # OUT on met1 c2 puts a pad at column 2 right below p1's pad on p0
    text, netlist = load("aoi21")
    rep = check(mutate(text, "route OUT li1  c1", "route OUT met1 c2"), netlist)
    assert rep.codes() == {"met1-pad"}, str(rep)
    rep = check(mutate(text, "route OUT li1  c1", "route OUT met1 c1"), netlist)
    assert rep.ok, str(rep)


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
    adjacent = base + "pin A gate:1n c0\npin B gate:1p c1\n"
    assert check(adjacent, netlist).codes() == {"split-order"}
    no_side = base + "pin A gate:1 c0\npin B gate:1p c2\n"
    assert "gate-side" in check(no_side, netlist).codes()


# -- channel tracks per plan ---------------------------------------------------

from cellgym.dsl.template import (CHANNEL_NAMES, CHANNEL_TRACKS, MAX_CHANNEL_TRACKS,
                                  MIN_CHANNEL_TRACKS, layout_for_netlist)
from cellgym.render.render import render


def with_tracks(text, n):
    return mutate(text, "\ncol 0", f"\ntracks {n}\ncol 0")


def test_tracks_default_is_three_and_unwritten():
    text, nl = load("aoi21")
    plan = parse(text)
    assert plan.tracks == CHANNEL_TRACKS == 3 and CHANNEL_NAMES == ("c0", "c1", "c2")
    assert "tracks" not in dumps(plan) and "tracks" not in plan.as_dict()
    lay = layout_for_netlist(nl)
    assert lay == layout_for_netlist(nl, 3)
    assert lay.channel_tracks == 3 and lay.channel_names == CHANNEL_NAMES
    explicit = parse(with_tracks(text, 3))
    assert dumps(explicit) == dumps(plan)
    assert render(explicit, nl).gds() == render(plan, nl).gds()


@pytest.mark.parametrize("n", [2, 4, 5])
def test_tracks_parse_validate_render(n):
    text, nl = load("aoi21")       # aoi21 routes on c0/c1 only, so it fits 2 tracks
    if n == 2:
        text = text.replace(" c2", " c1")
    base = render(parse(text), nl)
    plan = parse(with_tracks(text, n))
    assert plan.tracks == n and plan.as_dict()["tracks"] == n
    assert f"\ntracks {n}\ncol 0 " in dumps(plan) and parse(dumps(plan)).tracks == n
    rep = validate(plan, nl)
    assert rep.ok, str(rep)
    lay = rep.layout
    assert lay.channel_tracks == n == len(lay.channel_k)
    assert lay.channel_names == tuple(f"c{i}" for i in range(n))
    assert lay.has_track(f"c{n - 1}") and not lay.has_track(f"c{n}")
    ref = layout_for_netlist(nl)
    assert (lay.n_tracks, lay.p_tracks) == (ref.n_tracks, ref.p_tracks)
    assert lay.height == pytest.approx(ref.height + 0.34 * (n - 3))
    cell = render(plan, nl)
    assert cell.height_nm - base.height_nm == 340 * (n - 3)
    assert cell.width_nm == base.width_nm


def test_tracks_uses_the_extra_track():
    text, nl = load("aoi21")
    assert "track-missing" in check(mutate(text, "li1  c1", "li1  c4"), nl).codes()
    assert check(with_tracks(mutate(text, "li1  c1", "li1  c4"), 5), nl).ok
    assert "track-missing" in check(with_tracks(text, 2).replace("li1  c1", "li1  c2"), nl).codes()


@pytest.mark.parametrize("line, msg", [
    (f"tracks {MIN_CHANNEL_TRACKS - 1}", "outside"), (f"tracks {MAX_CHANNEL_TRACKS + 1}", "outside"),
    ("tracks", "one integer"), ("tracks x", "one integer"), ("tracks 4 5", "one integer"),
    ("tracks 4\ntracks 4", "duplicate"),
])
def test_tracks_line_errors(line, msg):
    text, _ = load("aoi21")
    with pytest.raises(PlanSyntaxError, match=msg):
        parse(mutate(text, "\ncol 0", f"\n{line}\ncol 0"))


def test_tracks_line_must_precede_columns():
    text, _ = load("aoi21")
    with pytest.raises(PlanSyntaxError, match="before the first col"):
        parse(mutate(text, "\ncol 1", "\ntracks 4\ncol 1"))


def test_layout_rejects_out_of_range_tracks():
    with pytest.raises(ValueError):
        layout(0.42, 0.63, MAX_CHANNEL_TRACKS + 1)
    with pytest.raises(ValueError):
        layout(0.42, 0.63, MIN_CHANNEL_TRACKS - 1)


def _dev(name, kind, gate, drain, source, w):
    model = "sky130_fd_pr__nfet_01v8" if kind == "nmos" else "sky130_fd_pr__pfet_01v8_hvt"
    return {"name": name, "type": kind, "model": model, "gate": gate, "drain": drain, "source": source,
            "bulk": "VNB" if kind == "nmos" else "VPB", "w": w, "l": 0.15}


NAND2 = {"id": "nand2", "inputs": ["A", "B"], "ports": ["A", "B", "OUT", "VDD", "VSS", "VPB", "VNB"],
         "nets": ["A", "B", "OUT", "VDD", "VSS", "VPB", "VNB", "n1"],
         "devices": [_dev("MN1", "nmos", "A", "OUT", "n1", 0.84), _dev("MN2", "nmos", "B", "n1", "VSS", 0.84),
                     _dev("MP1", "pmos", "A", "OUT", "VDD", 0.63), _dev("MP2", "pmos", "B", "OUT", "VDD", 0.63)]}


def nand2_split(tracks, tb, ta, out="li1 c0"):
    """NAND2 with column 1 split: NMOS B contact on c{tb}, PMOS A contact on c{ta}."""
    hdr = "" if tracks == 3 else f"tracks {tracks}\n"
    return (f"cell nand2\nin A B\nout OUT\n{hdr}col 0 sd n=VSS* p=VDD*\ncol 1 g n=B p=A\ncol 2 sd n=n1 p=OUT*\n"
            "col 3 g n=A p=-\ncol 4 sd n=OUT* p=OUT*\ncol 5 g n=- p=B\ncol 6 sd n=- p=VDD*\n"
            f"route A met1 c{ta} gate:1p gate:3\nroute B met1 c{tb} gate:1n gate:5\n"
            f"route OUT {out} nmos:4 pmos:2 pmos:4\npin A gate:3\npin B gate:5\npin OUT nmos:4\n")


@pytest.mark.parametrize("tracks", [2, 3, 4, 5, 6])
def test_split_contacts_need_two_tracks_between(tracks):
    from cellgym.check import check as fast_check
    for tb in range(tracks):
        for ta in range(tracks):
            rep = check(nand2_split(tracks, tb, ta), NAND2)
            assert ("split-order" in rep.codes()) == (ta - tb < 2), (tb, ta, str(rep))
            if ta - tb >= 2:
                assert rep.ok, str(rep)
                res = fast_check(render(parse(nand2_split(tracks, tb, ta)), NAND2), NAND2)
                assert res.ok, (tb, ta, res.summary())


def test_split_order_message_unchanged_at_three_tracks():
    rep = check(nand2_split(3, 0, 1), NAND2)
    assert "column 1: split contacts must be NMOS on c0, PMOS on c2" in str(rep)
