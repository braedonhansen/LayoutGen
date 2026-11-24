import random

import pytest

from cellgym.llm import selectk as S


def test_pick_stats_at_full_k_is_exact():
    # (ncols, area ratio, hash): two fewest-column samples (one duplicated placement), one wider but smaller
    samples = [(9, 1.10, "a"), (9, 1.10, "a"), (9, 1.02, "b"), (10, 1.00, "c")]
    st = S.pick_stats(samples, 50, random.Random(0), kmax=4)
    mean, fewcol, fewroute, best, routed, distinct = st[4]
    assert mean == pytest.approx((1.10 + 1.10 + 1.02 + 1.00) / 4)
    assert fewcol == pytest.approx((1.10 + 1.10 + 1.02) / 3)
    assert fewroute == pytest.approx(1.02) and best == pytest.approx(1.00)
    assert routed == 2 and distinct == 3


def test_pick_stats_k1_is_the_plain_mean_and_ordering_holds():
    rng = random.Random(1)
    samples = [(rng.choice((9, 10, 11)), 1 + rng.random() / 5, str(i)) for i in range(8)]
    st = S.pick_stats(samples, 400, random.Random(0))
    m = sum(s[1] for s in samples) / 8
    assert st[1][0] == pytest.approx(m, abs=0.02) and st[1][0] == st[1][1] == st[1][2] == st[1][3]
    for k in range(1, 9):
        mean, fewcol, fewroute, best, routed, distinct = st[k]
        assert best <= fewroute + 1e-12 <= fewcol + 1e-9 and routed <= distinct <= k
    assert st[8][3] == pytest.approx(min(s[1] for s in samples))
