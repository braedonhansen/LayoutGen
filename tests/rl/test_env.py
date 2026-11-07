import pytest

from cellgym.baselines.euler import euler_solve
from cellgym.rl.env import CellSampler, make_env
from cellgym.synth import tokens as T


@pytest.fixture(scope="module")
def env():
    return make_env("train", cache=None)


def plan_ids(netlist):
    return T.ids(T.encode_plan(euler_solve(netlist).text)) + [T.EOS]


def test_euler_plan_reward_one_and_garbage_zero(env):
    for cid in env.ids()[:4]:
        info = env.reward(cid, plan_ids(env.cells[cid]))
        assert info.passed and info.reward == pytest.approx(1.0, abs=0.01), (cid, info)
        assert env.reward(cid, [T.NL, T.NL, T.EOS]).reward == 0
        assert env.reward(cid, [9999]).reward == 0 and not env.reward(cid, []).passed


def test_batch_matches_single(env):
    cids = env.ids()[:4]
    samples = [plan_ids(env.cells[c]) for c in cids] + [[T.NL, T.EOS]] * 4
    all_c = cids + cids
    single = [env.reward(c, s) for c, s in zip(all_c, samples)]
    for jobs in (1, 2):
        assert env.reward_batch(all_c, samples, jobs=jobs) == single
    env.close()


def test_refuses_heldout():
    for sp in ("val", "test_id", "test_ood"):
        with pytest.raises(ValueError):
            make_env(sp)
    from cellgym.bench import splits
    from cellgym.bench.common import SPLITS_FILE, load_cells
    from cellgym.rl.env import PlanEnv
    so = splits.split_of(splits.load(SPLITS_FILE))
    cid = next(c for c in so if so[c] == "val")
    nl = load_cells("out/cells")[cid]
    with pytest.raises(ValueError):
        PlanEnv({cid: nl}, {cid: 1000})


def test_sampler_deterministic(env):
    a, b = CellSampler(env, seed=3), CellSampler(env, seed=3)
    assert [a.next_groups(5) for _ in range(3)] == [b.next_groups(5) for _ in range(3)]
    assert len(set(CellSampler(env, 1).next_groups(8))) == 8
    w = CellSampler(env, 0, weights={c: 0.0 for c in env.ids()[1:]})
    assert w.next_groups(1) == env.ids()[:1]
    assert CellSampler.expand(["a", "b"], 2) == ["a", "a", "b", "b"]


def test_prompt(env):
    p = env.prompt_ids(env.ids()[0])
    assert p[0] == T.BOS and p[-1] == T.SEP


def test_bucket_weights_match_reference_split():
    from cellgym.rl.env import CellSampler, PlanEnv
    def nl(k, d):
        return {"inputs": ["A"] * k, "devices": [{}] * d}
    cells = {"t1": nl(2, 4), "t2": nl(5, 14), "s1": nl(5, 14), "s2": nl(5, 14), "s3": nl(5, 18), "s4": nl(5, 18)}
    env = PlanEnv.__new__(PlanEnv)
    env.cells, env.split_of = cells, {"t1": "train", "t2": "train"}
    w = CellSampler.bucket_weights(env, "train", uniform_mix=0.0)
    assert abs(w["t1"] - 0.5) < 1e-9                          # bucket (2,4): half the train cells, 1 cell
    assert abs(w["t2"] + w["s1"] + w["s2"] - 0.5) < 1e-9      # bucket (5,14) shared by 3 cells
    assert w["s3"] == 0 and abs(sum(w.values()) - 1) < 1e-9
    w = CellSampler.bucket_weights(env, "train", uniform_mix=0.1)
    assert w["s3"] > 0 and abs(sum(w.values()) - 1) < 1e-9
