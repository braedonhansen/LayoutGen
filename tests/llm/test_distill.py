import json
import random
from pathlib import Path

import pytest

from cellgym.llm import distill as D
from cellgym.llm import fmt as F

PLANS = Path("out/synth5/plans.jsonl")


def _rows(n=40):
    if not PLANS.exists():
        pytest.skip("needs out/synth5")
    rows = []
    with open(PLANS) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def test_plan_text_replays_and_builds_step_examples():
    for r in _rows():
        cols = F.plan_columns(r["plan"])
        text = D.plan_text(r["netlist"], cols)
        assert F.plan_columns(text) == cols
        assert len(F.step_examples(r["netlist"], text, random.Random(0), 0)) == len(cols)


def test_select_keeps_fewest_distinct_and_drops_worse_than_label():
    rows = _rows()
    r = next(x for x in rows if len(F.plan_columns(x["plan"])) >= 5)
    cols = [list(c) for c in F.plan_columns(r["plan"])]
    n = len(cols)
    cells = {"a": {"netlist": r["netlist"], "topology_id": "t", "origin": "o", "label_cols": n},
             "b": {"netlist": r["netlist"], "topology_id": "t", "origin": "o", "label_cols": n - 1}}
    longer = cols + [["sd", "VSS", "VDD"]]
    samples = [{"cell": "a", "status": "done", "cols": cols}, {"cell": "a", "status": "done", "cols": cols},
               {"cell": "a", "status": "done", "cols": longer}, {"cell": "a", "status": "cap", "cols": cols[:2]},
               {"cell": "b", "status": "done", "cols": cols}]
    out, st = D.select(cells, samples, keep=3)
    assert [x["id"] for x in out] == ["a"] and out[0]["n_cols"] == n
    assert st["at_label"] == 1 and st["above_label"] == 1 and st["plans"] == 1
