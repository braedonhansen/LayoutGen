import math

from cellgym.llm.tradeoff import best_point, count_breaks, pearson, var_decomp


def test_count_breaks():
    cols = [("sd", "VSS", "VDD"), ("g", "A"), ("sd", "n1", "OUT"), ("g", "B"), ("sd", "OUT", "VDD")]
    assert count_breaks(cols) == (0, 0)
    # one break column changing both rows, one changing only the p row
    cols = [("sd", "VSS", "VDD"), ("g", "A"), ("sd", "n1", "OUT"), ("sd", "VSS", "VDD"), ("g", "B"),
            ("sd", "OUT", "OUT"), ("sd", "OUT", "VDD"), ("g", "C"), ("sd", "n1", "OUT")]
    assert count_breaks(cols) == (2, 3)


def test_var_decomp():
    # width varies, height constant: all variance is width
    vw, vh, cov, va = var_decomp([(100, 50), (200, 50), (400, 50)])
    assert vh == 0 and cov == 0 and math.isclose(vw, va)
    # width x height constant: the terms cancel
    vw, vh, cov, va = var_decomp([(100, 400), (200, 200), (400, 100)])
    assert math.isclose(vw, vh) and math.isclose(cov, -2 * vw) and abs(va) < 1e-12
    # two points, both factors double: var log w = (ln2 / 2)^2, total four times that
    vw, vh, cov, va = var_decomp([(1, 1), (2, 2)])
    assert math.isclose(vw, (math.log(2) / 2) ** 2) and math.isclose(va, 4 * vw)


def test_best_point_and_pearson():
    curve = {"1": None, "2": [1000, 500, 0], "3": [900, 550, 0], "4": [900, 600, 0]}
    assert best_point(curve) == (3, 900, 550)
    assert best_point({"1": [10, 10, 0], "2": [5, 20, 0]}) == (1, 10, 10)      # tie: fewer tracks
    assert best_point({"1": None}) is None
    assert pearson([1, 2, 3], [2, 4, 6]) == 1.0 and pearson([1, 1, 1], [1, 2, 3]) is None
