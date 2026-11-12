import json
import random
from pathlib import Path

import pytest

from cellgym.bench.evaluate import evaluate
from cellgym.llm import fmt as F
from cellgym.llm.route_env import RouteEnv, euler_actions

PLANS = Path("out/synth5/plans.jsonl")


@pytest.mark.skipif(not PLANS.exists(), reason="no synth5 corpus")
def test_euler_plans_replay_exactly():
    rows = [json.loads(l) for l in PLANS.read_text().splitlines()]
    for r in random.Random(0).sample(rows, 60):
        env = RouteEnv(r["netlist"], F.plan_columns(r["plan"]))
        for a in euler_actions(r["netlist"], r["plan"]):
            assert a in env.legal(), (r["id"], a)
            assert env.step(a)
        assert env.done()
        assert env.plan_text() == r["plan"]
        assert evaluate(env.plan_text(), r["netlist"]).stage == "pass"
