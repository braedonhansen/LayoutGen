import json
from pathlib import Path

import pytest

from cellgym.llm import fmt as F
from cellgym.llm.regret import Solver

PLANS = Path("out/synth5/plans.jsonl")


def _rows(n=300):
    if not PLANS.exists():
        pytest.skip("needs out/synth5")
    rows = []
    with open(PLANS) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def _breaks(cols):
    return sum(1 for i in range(1, len(cols)) if cols[i][0] == "sd" and cols[i - 1][0] == "sd")


def test_regrets_sum_to_excess_breaks_and_labels_are_never_beaten():
    for r in _rows():
        cols = F.plan_columns(r["plan"])
        s = Solver(r["netlist"])
        opt = s.restart(0, 0)
        rep = s.replay(cols)
        assert len(rep) == len(cols)
        assert all(x >= 0 for _, x in rep)
        assert opt + sum(x for _, x in rep) == _breaks(cols)
        assert opt <= r["breaks"]


def test_inverter_has_no_break():
    nl = {"devices": [{"type": "nmos", "gate": "A", "drain": "OUT", "source": "VSS"},
                      {"type": "pmos", "gate": "A", "drain": "OUT", "source": "VDD"}]}
    s = Solver(nl)
    assert s.restart(0, 0) == 0
    assert s.replay([("sd", "OUT", "OUT"), ("g", "A"), ("sd", "VSS", "VDD")]) == [("start", 0), ("gate", 0), ("sd", 0)]
