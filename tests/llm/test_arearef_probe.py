"""arearef: reference choice, loader kinds, gate-preferring placements, Euler placements."""

import json
import random

import pytest

from cellgym.llm import arearef as A
from cellgym.llm import fmt as F


def _nl(devs, inputs, cid="t"):
    nets = {x for d in devs for x in d[1:]}
    return {"id": cid, "inputs": inputs, "ports": sorted(nets & {"VDD", "VSS", "Y"} | set(inputs)),
            "devices": [{"name": f"M{i}", "type": t, "gate": g, "drain": d, "source": s}
                        for i, (t, g, d, s) in enumerate(devs)]}


NAND2 = _nl([("nmos", "A", "Y", "n1"), ("nmos", "B", "n1", "VSS"),
             ("pmos", "A", "Y", "VDD"), ("pmos", "B", "Y", "VDD")], ["A", "B"])
AOI21 = _nl([("nmos", "A", "Y", "n1"), ("nmos", "B", "n1", "VSS"), ("nmos", "C", "Y", "VSS"),
             ("pmos", "A", "p1", "VDD"), ("pmos", "B", "p1", "VDD"), ("pmos", "C", "Y", "p1")], ["A", "B", "C"])


def _rec(w, h, t):
    return {"area_um2": w * h / 1e6, "tracks": t, "width_nm": w, "height_nm": h}


def test_pick_best():
    assert A.pick_best([]) is None
    assert A.pick_best([None, None]) is None
    assert A.pick_best([None, _rec(10, 10, 3), _rec(9, 10, 4), _rec(9, 10, 4)]) == 2
    assert A.pick_best([_rec(10, 12, 4), _rec(12, 10, 3)]) == 1      # equal area: fewer tracks
    assert A.pick_best([_rec(10, 10, 3), _rec(10, 10, 3)]) == 0      # full tie: the earlier one


def test_load_kinds(tmp_path, monkeypatch):
    monkeypatch.delenv(A.ENV_KIND, raising=False)
    e = {"a": _rec(10, 10, 3), "b": None, "d": None}
    h = {"a": _rec(20, 10, 3), "b": _rec(30, 10, 3), "c": _rec(40, 10, 3)}
    assert A.load("s", tmp_path) is None and A.load("s", tmp_path, "euler+heur") is None
    (tmp_path / "s.json").write_text(json.dumps(e))
    (tmp_path / "s.heur.json").write_text(json.dumps(h))
    assert A.load("s", tmp_path) == e                                  # default: Euler only
    assert A.load("s", tmp_path, "heur") == h
    m = A.load("s", tmp_path, "euler+heur")
    assert m == {"a": e["a"], "b": h["b"], "c": h["c"], "d": None}
    assert set(A.areas_nm2(m)) == {"a", "b", "c"}
    monkeypatch.setenv(A.ENV_KIND, "heur")
    assert A.load("s", tmp_path) == h and A.path_of("s", tmp_path).name == "s.heur.json"
    assert A.load("s", tmp_path, "euler") == e                         # the argument wins
    monkeypatch.setenv(A.ENV_KIND, "nope")
    with pytest.raises(ValueError):
        A.load("s", tmp_path)


@pytest.mark.parametrize("nl", [NAND2, AOI21])
def test_gate_pref_placement_is_legal_and_complete(nl):
    rng = random.Random(0)
    for _ in range(30):
        cols = A.gate_pref_placement(nl, rng)
        p = F.Placer(nl)
        assert cols and all(p.step(c) for c in cols) and p.done()
        assert sum(c[0] == "g" for c in cols) == len(nl["devices"]) // 2


def test_gate_pref_few_breaks_on_nand2():
    rng = random.Random(1)
    # sd g sd g sd, or one break column when the random start strands a device; never more
    assert {len(A.gate_pref_placement(NAND2, rng)) for _ in range(40)} == {5, 6}


def test_heur_candidates_distinct_and_sorted():
    c = A.heur_candidates(AOI21, 60, random.Random(0))
    assert len({tuple(x) for x in c}) == len(c) > 1
    assert [len(x) for x in c] == sorted(len(x) for x in c)


def test_euler_placements_replay_on_placer():
    places = A.euler_placements(AOI21)
    assert 1 <= len(places) <= 4 and len({len(c) for c in places}) == 1
    for cols in places:
        p = F.Placer(AOI21)
        assert all(p.step(c) for c in cols) and p.done()
    assert min(len(c) for c in A.heur_candidates(AOI21, 200, random.Random(0))) >= len(places[0])
