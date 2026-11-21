import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cellgym.llm import rl as R  # noqa: E402


def test_legal_logp_entropy_matches_hand_computation():
    logits = torch.tensor([[0.0, math.log(2.0), math.log(3.0), 5.0, -1.0],
                           [1.0, 1.0, 1.0, 1.0, 9.0],
                           [0.3, -0.2, 4.0, 0.1, 0.0]])
    idx = torch.tensor([[0, 1, 2], [3, -1, -1], [4, 0, -1]])     # legal tokens, -1 = padding
    tgt = torch.tensor([2, 3, 0])
    lp, ent = R.legal_logp_entropy(logits, idx, tgt)
    # row 0: p = 1/6, 2/6, 3/6 over tokens 0, 1, 2 (token 3 with the largest logit is not legal)
    p0 = [1 / 6, 2 / 6, 3 / 6]
    assert lp[0].item() == pytest.approx(math.log(3 / 6), abs=1e-6)
    assert ent[0].item() == pytest.approx(-sum(p * math.log(p) for p in p0), abs=1e-6)
    # row 1: one legal token: certain
    assert lp[1].item() == pytest.approx(0.0, abs=1e-6) and ent[1].item() == pytest.approx(0.0, abs=1e-6)
    # row 2: tokens 4 (logit 0.0) and 0 (logit 0.3)
    q = math.exp(0.3) / (1 + math.exp(0.3))
    assert lp[2].item() == pytest.approx(math.log(q), abs=1e-6)
    assert ent[2].item() == pytest.approx(-(q * math.log(q) + (1 - q) * math.log(1 - q)), abs=1e-6)


def test_logp_is_the_old_formula_and_gradients_are_finite():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(40, 50, generator=g, requires_grad=True)
    idx = torch.full((40, 6), -1, dtype=torch.long)
    for i in range(40):
        k = 1 + i % 6
        idx[i, :k] = torch.randperm(50, generator=g)[:k]
    tgt = idx[:, 0].clone()
    lp, ent = R.legal_logp_entropy(logits, idx, tgt)
    sub = logits.gather(-1, idx.clamp_min(0)).masked_fill(idx < 0, float("-inf"))        # code before the change
    old = logits.gather(-1, tgt[:, None])[:, 0] - torch.logsumexp(sub, -1)
    assert torch.equal(lp, old)
    assert (ent >= -1e-6).all() and (ent <= torch.log((idx >= 0).sum(-1).float()) + 1e-5).all()
    (ent.sum() + lp.sum()).backward()
    assert torch.isfinite(logits.grad).all()
    # uniform logits: entropy = log(number of legal tokens)
    _, e = R.legal_logp_entropy(torch.zeros(40, 50), idx, tgt)
    assert torch.allclose(e, torch.log((idx >= 0).sum(-1).float()), atol=1e-6)


def test_ent_coef_update():
    up = R.ent_coef_update
    assert up(0.02, 1.0, 2.0, 0.0, 0.1) == 0.02                      # no target: fixed
    assert up(0.0, 0.1, 2.0, 0.7, 0.1) == 0.0                        # multiplicative: 0 stays 0
    assert up(0.02, 1.4, 2.0, 0.7, 0.1) == pytest.approx(0.02)       # on target
    assert up(0.02, 0.7, 2.0, 0.7, 0.1) == pytest.approx(0.02 * math.exp(0.1 * 0.5))    # half the target: up
    assert up(0.02, 2.0, 2.0, 0.7, 0.1) == pytest.approx(0.02 * math.exp(0.1 * (1.4 - 2.0) / 1.4))   # first batch: down
    assert up(0.9, 0.0, 2.0, 0.7, 5.0) == 1.0                        # clipped
    assert up(0.02, 0.5, 0.0, 0.7, 0.1) == 0.0                       # h0 = 0: exponent floors, clipped at 0
    c = 0.02
    for _ in range(200):                                             # entropy stuck low: grows to the cap
        c = up(c, 0.2, 2.0, 0.7, 0.1)
    assert c == 1.0


def test_select_score():
    a = {"w": 1.0, "reward": 1.02, "area": 0.99, "best_area": 0.93}
    b = {"w": 1.0, "reward": 1.05, "area": 0.96, "best_area": 0.95}
    c = {"w": 1.0, "reward": 1.00, "area": 0.90, "best_area": 0.945}
    best = lambda mode: max((a, b, c), key=lambda v: R.select_score(v, mode))
    assert best("reward") is b and best("best_area") is a and best("both") is c
    assert R.select_score(a, "both") == pytest.approx(-0.96)
    assert R.select_score({"w": None, "reward": 0.0, "area": None, "best_area": None}, "reward") is None
    assert R.select_score({"w": 1.0, "reward": 1.0, "area": None, "best_area": None}, "best_area") is None
    assert R.select_score({"w": 1.0, "reward": 1.0, "area": None, "best_area": None}, "reward") == 1.0


def test_distinct_per_group():
    x, y = [("sd", "VSS", "VDD"), ("g", "A")], [("sd", "OUT", "VDD"), ("g", "A")]
    assert R.distinct_per_group([x, x, y, x, y, y, y, y], 4) == 1.5
    assert R.distinct_per_group([x, [list(c) for c in x]], 2) == 1.0     # lists or tuples
    assert R.distinct_per_group([], 4) == 0.0


def test_ent_state_round_trip(tmp_path):
    p = tmp_path / "ent_state.json"
    R.save_ent_state(p, 1.2345678901234567, 0.0123456789, 60)
    assert R.load_ent_state(p) == {"h0": 1.2345678901234567, "coef": 0.0123456789, "steps": 60}
    R.save_ent_state(p, None, 0.02, 1)             # no batch with an update yet
    assert R.load_ent_state(p)["h0"] is None


def test_new_flags_parse():
    with pytest.raises(SystemExit) as e:
        R.main(["--init", "x", "--out", "y", "--select", "both"])        # needs --reward area
    assert e.value.code == 2
