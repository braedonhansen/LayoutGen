import json
import random

import pytest

from cellgym.llm import probe as P
from cellgym.llm import probegen as G
from cellgym.netlists.enumerate import depth, irredundant, num_leaves, var_masks
from cellgym.synth.devset import load


def test_fast_masks_match_var_masks():
    for k in range(1, 7):
        assert G.fast_masks(k) == var_masks(k)


def test_device_bound():
    assert [G.min_stack(n) for n in (1, 4, 5, 16, 17, 20, 24, 25, 26)] == [1, 2, 3, 4, 5, 5, 5, 5, 6]
    with pytest.raises(ValueError):
        G.random_shape(17, 4, 4, random.Random(0))


@pytest.mark.parametrize("n,a,b", [(2, 2, 1), (5, 2, 3), (9, 3, 3), (16, 4, 4), (20, 5, 4), (24, 5, 5)])
def test_random_shape_within_budgets(n, a, b):
    rng = random.Random(n)
    for _ in range(50):
        t = G.random_shape(n, a, b, rng)
        assert num_leaves(t) == n and depth(t) <= a and depth(t, dual=True) <= b


def test_single_stage_sampler():
    nets = G.sample_networks_fast(6, 9, 4, 8, seed=1)
    assert len(nets) == 8 and len({n.tree for n in nets}) == 8
    assert nets == G.sample_networks_fast(6, 9, 4, 8, seed=1)
    for net in nets:
        assert (net.n_inputs, net.n_devices) == (6, 9)
        assert max(depth(net.tree), depth(net.tree, dual=True)) <= 4
        assert irredundant(net.tree, 6)                  # the enumerator's own test
        nl = P.single_json(net)
        assert G.check_netlist(nl, G.expected_single(net)) == []
        assert G.stats(nl) == {"inputs": 6, "per_side": 9, "stack": nl["pdn_depth"] if nl["pdn_depth"] > nl["pun_depth"]
                               else nl["pun_depth"], "stages": 1, "outputs": 1}


def test_check_netlist_catches_a_wrong_device():
    net = G.sample_networks_fast(3, 4, 3, 1, seed=2)[0]
    nl = P.single_json(net)
    nl["devices"][0]["gate"] = "B" if nl["devices"][0]["gate"] != "B" else "A"
    assert G.check_netlist(nl, G.expected_single(net))


def test_half_adder_simulates():
    from pathlib import Path
    nl = json.loads(Path("cellgym/dsl/examples/half_adder.json").read_text())
    for r in range(4):
        a, b = bool(r & 1), bool(r >> 1)
        v = G.simulate(nl, {"A": a, "B": b})
        assert (v["S"], v["C"]) == (a != b, a and b)


@pytest.mark.parametrize("stages,outputs,per_side", [(2, (1, 1), (6, 12)), (3, (1, 2), (8, 16))])
def test_multistage(stages, outputs, per_side):
    cells = G.sample_multistage(stages, 6, per_side, outputs, seed=3)
    assert len(cells) == 6 and len({c["id"] for c in cells}) == 6
    for nl in cells:
        st = G.stats(nl)
        assert st["stages"] == stages and outputs[0] <= st["outputs"] <= outputs[1]
        assert per_side[0] <= st["per_side"] <= per_side[1] and st["inputs"] <= 5 and st["stack"] <= 4
        gates = {d["gate"] for d in nl["devices"]}
        terms = {x for d in nl["devices"] for x in (d["drain"], d["source"])}
        assert set(nl["inputs"]) <= gates                               # every input used
        assert gates - set(nl["inputs"]) <= terms                       # other gates are driven nets
        outs = [p for p in nl["ports"] if p not in nl["inputs"] and p not in G.SUPPLY]
        assert not set(outs) & gates and set(outs) <= terms             # outputs are exactly the sinks
        assert terms | gates | set(G.SUPPLY) == set(nl["nets"])
        assert [s for s in nl["stages"] if len(s["pins"]) == 1 and s["pins"][0][0] == "x"
                and next(t for t in nl["stages"] if t["out"] == s["pins"][0])["tree"] == "A"
                and s["tree"] == "A"] == []                              # no inverter after an inverter
        assert G.check_netlist(nl, {o: int(h, 16) for o, h in nl["function"].items()}) == []


def test_write_set_loads(tmp_path):
    man = P.build_multi(tmp_path, 2, 3, (6, 12), (1, 1))
    assert man["cells"] == 3
    cells, w = load(tmp_path / "stage2", "all")
    assert len(cells) == 3 and set(w.values()) == {None}
    man = P.build_single(tmp_path, "11:17", 2, 4)                        # 17 per side cannot fit in stack 4
    assert man["max_stack"] == 5 and "raised" in man["note"]
    cells, _ = load(tmp_path / "i11d17", "all")
    assert all(len(c["devices"]) == 34 for c in cells.values())
