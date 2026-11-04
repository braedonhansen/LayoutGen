import random

import pytest

from cellgym.baselines.euler import euler_plan
from cellgym.bench import splits
from cellgym.bench.common import SPLITS_FILE
from cellgym.synth import tokens as T
from cellgym.synth.eval import score_samples


def _ids(plan):
    return T.ids(T.encode_plan(plan))


@pytest.fixture(scope="module")
def picks(core_cells):
    split = splits.split_of(splits.load(SPLITS_FILE))
    return {c: core_cells[c] for c in sorted(core_cells)[::40] if split[c] != "test_ood"}


def test_euler_plans_score_perfect(picks):
    samples = {c: [_ids(euler_plan(nl))] for c, nl in picks.items()}
    res = score_samples(picks, samples)
    a = res["summary"]["all"]
    assert a["pass_at_1"] == 1.0 and a["pass_at_k"] == 1.0
    assert a["parse_rate"] == a["valid_rate"] == 1.0
    assert a["w_over_euler_best"] == pytest.approx(1.0)
    assert a["best_of_k_reward"] == pytest.approx(1.0)
    for c in res["cells"]:
        assert c["samples"][0]["stage"] == "pass"


def test_garbage_fails_without_crashing(picks):
    rng = random.Random(0)
    cid, nl = next(iter(picks.items()))
    plan_ids = _ids(euler_plan(nl))
    samples = {cid: [
        plan_ids[: len(plan_ids) // 2],                                # truncated, no <eos>
        [rng.randrange(len(T.VOCAB)) for _ in range(120)],             # random ids
        [],                                                            # empty
        [T.EOS],                                                       # immediate <eos>
        [len(T.VOCAB) + 5, -1],                                        # out-of-range ids
        plan_ids + plan_ids,                                           # text after the first <eos> is ignored
    ]}
    res = score_samples({cid: nl}, samples)
    sm = res["cells"][0]["samples"]
    assert [s["pass"] for s in sm[:5]] == [False] * 5
    assert sm[5]["pass"] is True
    assert sm[4]["stage"] == "decode"
    assert res["cells"][0]["pass_at_1"] is False and res["cells"][0]["pass_at_k"] is True


def test_empty_sample_list_is_a_failure(picks):
    cid, nl = next(iter(picks.items()))
    res = score_samples({cid: nl}, {cid: []})
    c = res["cells"][0]
    assert c["pass_at_1"] is False and c["pass_at_k"] is False and c["best_reward"] == 0.0
    assert res["summary"]["all"]["pass_at_1"] == 0.0


def test_unknown_cell_has_no_ratio(picks):
    cid, nl = next(iter(picks.items()))
    nl2 = dict(nl, id="cg_not_in_benchmark")
    res = score_samples({nl2["id"]: nl2}, {nl2["id"]: [_ids(euler_plan(nl))]}, w_euler={}, split_of={})
    s = res["cells"][0]
    assert s["split"] == "other" and s["samples"][0]["w_over_euler"] is None and s["best_reward"] is None
