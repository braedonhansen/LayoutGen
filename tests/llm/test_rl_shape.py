import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from cellgym.llm import rl as R  # noqa: E402

SD, G = ("sd", "a", "b"), ("g", "A")
SD2 = ("sd", "c", "d")


def test_breaks_before_counts_sd_after_sd_only():
    cols = [SD, G, SD, SD2, G, SD, SD2, SD, G, SD]      # breaks at index 3, 6 and 7
    assert R.breaks_before(cols) == [0, 0, 0, 0, 1, 1, 1, 2, 3, 3]
    assert R.breaks_before([SD, G, SD]) == [0, 0, 0] and R.breaks_before([]) == []


def test_step_advantages_forgive_only_the_steps_after_a_break():
    cols = [SD, G, SD, SD2, G, SD]                      # 2 devices per side, one break at index 3
    a = R.step_advantages(-1.0, cols, len(cols), 2, std=0.1, shape=1.0)
    assert a[:4] == [-1.0] * 4                          # the break step and everything before it keep the blame
    assert a[4] == a[5] == pytest.approx(-1.0 + (1 / 3) / (0.1 + 1e-4))
    assert R.step_advantages(-1.0, cols, len(cols), 2, std=0.1, shape=0.0) == [-1.0] * 6
    assert R.step_advantages(0.5, [SD, G, SD], 3, 1, std=0.1, shape=1.0) == [0.5] * 3     # no break: unchanged


def test_step_advantages_clip_and_truncate_to_the_recorded_steps():
    cols = [SD, SD2, SD, SD2, G, SD]
    a = R.step_advantages(0.0, cols, 5, 1, std=1e-3, shape=1.0)
    assert len(a) == 5 and a[0] == 0.0 and max(a) == 3.0
    assert min(R.step_advantages(-9.0, cols, 6, 1, std=1.0, shape=1.0)) == -3.0
