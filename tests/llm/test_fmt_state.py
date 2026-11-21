import json
import os
import random
import subprocess
import sys
from itertools import islice
from pathlib import Path

import pytest

from cellgym.llm import fmt as F

PLANS = Path("out/synth5/plans.jsonl")


def _dev(kind, gate, drain, source):
    return {"type": kind, "gate": gate, "drain": drain, "source": source, "w": 0.42}


NAND2 = {"id": "nand2", "inputs": ["A", "B"], "devices": [
    _dev("nmos", "A", "OUT", "n1"), _dev("nmos", "B", "n1", "VSS"),
    _dev("pmos", "A", "OUT", "VDD"), _dev("pmos", "B", "OUT", "VDD")]}
NAND2_COLS = [("sd", "VSS", "VDD"), ("g", "B"), ("sd", "n1", "OUT"), ("g", "A"), ("sd", "OUT", "VDD")]
# the state before each column, in the v1 format (written out by hand from the old code)
NAND2_V1 = [
    "left nmos: A OUT-n1 | B VSS-n1\nleft pmos: A OUT-VDD | B OUT-VDD\nlast: start\nnext:",
    "left nmos: A OUT-n1 | B VSS-n1\nleft pmos: A OUT-VDD | B OUT-VDD\nlast: sd n=VSS p=VDD\nnext:",
    "left nmos: A OUT-n1 | B VSS-n1\nleft pmos: A OUT-VDD | B OUT-VDD\nlast: sd n=VSS p=VDD | g B\nnext:",
    "left nmos: A OUT-n1\nleft pmos: A OUT-VDD\nlast: sd n=VSS p=VDD | g B | sd n=n1 p=OUT\nnext:",
    "left nmos: A OUT-n1\nleft pmos: A OUT-VDD\nlast: g B | sd n=n1 p=OUT | g A\nnext:",
]
# nets crossing the frontier before each column, and after the last one
NAND2_OPEN = [[], [], ["B"], ["OUT", "n1"], ["A", "OUT", "n1"], []]


def _corpus(n):
    with open(PLANS) as f:
        return [json.loads(l) for l in islice(f, n)]


def test_v1_state_unchanged(monkeypatch):
    monkeypatch.setattr(F, "STATE_VERSION", "v1")
    p = F.Placer(NAND2)
    for c, want in zip(NAND2_COLS, NAND2_V1):
        assert p.state() == want
        assert p.state("v2") != want
        assert p.step(c)
    assert p.done()


def test_v2_open_line(monkeypatch):
    monkeypatch.setattr(F, "STATE_VERSION", "v2")
    p = F.Placer(NAND2)
    for c, v1, op in zip(NAND2_COLS, NAND2_V1, NAND2_OPEN):
        assert p.open_nets() == op
        assert p.state() == v1[:-len("next:")] + f"open: {' '.join(op) or 'none'}\nnext:"
        assert p.step(c)
    assert p.open_nets() == NAND2_OPEN[-1]
    assert p.state().splitlines()[-2:] == ["open: none", "next:"]


def test_state_version_from_env():
    for v in ("v1", "v2"):
        out = subprocess.run([sys.executable, "-c", "from cellgym.llm import fmt; print(fmt.STATE_VERSION)"],
                             env={**os.environ, "CELLGYM_STATE": v}, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == v
    bad = subprocess.run([sys.executable, "-c", "import cellgym.llm.fmt"],
                         env={**os.environ, "CELLGYM_STATE": "v3"}, capture_output=True, text=True)
    assert bad.returncode != 0 and "CELLGYM_STATE" in bad.stderr


def test_step_examples_follow_the_version(monkeypatch):
    plan = "".join(f"col {i} {'g' if c[0] == 'g' else 'sd'} n={c[1]} p={c[-1]}\n" for i, c in enumerate(NAND2_COLS))
    monkeypatch.setattr(F, "STATE_VERSION", "v1")
    v1 = F.step_examples(NAND2, plan, random.Random(0), 0)
    assert [s for s, _ in v1] == NAND2_V1
    assert [t for _, t in v1] == [F.col_text(c) for c in NAND2_COLS]
    monkeypatch.setattr(F, "STATE_VERSION", "v2")
    v2 = F.step_examples(NAND2, plan, random.Random(0), 0)
    assert [t for _, t in v2] == [t for _, t in v1]
    for (s2, _), (s1, _), op in zip(v2, v1, NAND2_OPEN):
        lines = s2.split("\n")
        assert lines[-2] == f"open: {' '.join(op) or 'none'}"
        assert "\n".join(lines[:-2] + lines[-1:]) == s1


@pytest.mark.skipif(not PLANS.exists(), reason="no synth5 corpus")
def test_corpus_replay_and_renaming(monkeypatch):
    rows = random.Random(0).sample(_corpus(4000), 80)
    rng = random.Random(1)
    renamed = 0
    for r in rows:
        nl, plan = r["netlist"], r["plan"]
        monkeypatch.setattr(F, "STATE_VERSION", "v1")
        v1 = F.step_examples(nl, plan, rng, 0)
        monkeypatch.setattr(F, "STATE_VERSION", "v2")
        v2 = F.step_examples(nl, plan, rng, 0)
        assert v1 and len(v1) == len(v2) == len(F.plan_columns(plan))
        for (s1, t1), (s2, t2) in zip(v1, v2):          # v2 = v1 + the open: line
            lines = s2.split("\n")
            assert t1 == t2 and lines[-2].startswith("open: ")
            assert "\n".join(lines[:-2] + lines[-1:]) == s1
        # renamed netlist + renamed plan: the open set is the renamed open set at every step
        L, N = F.rename_map(nl, rng)
        f = lambda x: L.get(x, N.get(x, x))
        nl2, plan2 = F.renamed_netlist(nl, L, N), F.apply_names(plan, L, N, plan_side=True)
        a, b = F.Placer(nl), F.Placer(nl2)
        for c, c2 in zip(F.plan_columns(plan), F.plan_columns(plan2)):
            assert b.open_nets() == sorted(f(x) for x in a.open_nets())
            assert b.state().split("\n")[-2] == "open: " + (" ".join(b.open_nets()) or "none")
            assert a.step(c) and b.step(c2)
            renamed += any(f(x) != x for x in a.open_nets())
        assert a.done() and b.done() and a.open_nets() == b.open_nets() == []
        # the definition, recomputed from the columns and the devices still to place
        p = F.Placer(nl)
        for c in F.plan_columns(plan):
            placed = {x for col in p.cols for x in col[1:]}
            todo = {x for k in ("nmos", "pmos") for (g, t), n in p.left[k].items() if n > 0 for x in (g, *t)}
            assert set(p.open_nets()) == {x for x in placed & todo if x not in ("VSS", "VDD")}
            p.step(c)
    assert renamed > 50
