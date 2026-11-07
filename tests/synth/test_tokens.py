import pytest

from cellgym.baselines.euler import euler_plan
from cellgym.synth import tokens as T


def test_vocab_unique_and_ids_fixed():
    assert len(set(T.VOCAB)) == len(T.VOCAB)
    assert T.VOCAB[T.PAD] == "<pad>" and T.VOCAB[T.EOS] == "<eos>"


def test_round_trip_core_cells(core_cells):
    for cid in sorted(core_cells)[::12]:
        nl = core_cells[cid]
        plan = euler_plan(nl)
        prompt, target = T.encode_netlist(nl), T.encode_plan(plan)
        T.ids(prompt), T.ids(target)
        assert prompt[0] == "<bos>" and prompt[-1] == "<sep>" and target[-1] == "<eos>"
        assert T.decode_plan(target, nl) == plan, cid


def test_unknown_token_is_an_error():
    with pytest.raises(KeyError):
        T.ids(["not-a-token"])


def test_noin_prompt_drops_only_the_in_line():
    nl = {"inputs": ["A", "B"], "devices": [
        {"type": "nmos", "gate": "A", "drain": "OUT", "source": "n1", "w": 0.42},
        {"type": "pmos", "gate": "B", "drain": "OUT", "source": "VDD", "w": 0.63}]}
    v1, noin = T.encode_netlist(nl), T.encode_netlist(nl, "noin")
    assert v1[:5] == ["<bos>", "in", "A", "B", "\n"]
    assert noin == ["<bos>"] + v1[5:]
    import pytest
    with pytest.raises(ValueError):
        T.encode_netlist(nl, "bogus")
