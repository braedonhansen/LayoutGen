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
