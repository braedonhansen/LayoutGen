import random

import torch

from cellgym.synth import tokens as T
from cellgym.synth import train as TR


def test_letter_map_is_order_preserving_and_injective():
    rng = random.Random(0)
    for n in range(1, 7):
        for _ in range(50):
            m = TR.letter_map(n, rng)
            src = [T.TOKEN_ID[c] for c in T.INPUTS[:n]]
            dst = [m[s] for s in src]
            assert dst == sorted(dst) and len(set(dst)) == n
            assert set(dst) <= set(TR.INPUT_IDS)


def test_relabel_rows_renames_prompt_and_target_consistently():
    prompt = T.ids(["<bos>", "in", "A", "B", "\n", "n", "A", "OUT", "n1", "w0.42", "\n",
                    "n", "B", "n1", "VSS", "w0.42", "\n", "<sep>"])
    target = T.ids(["col", "1", "g", "n=", "A", "p=", "B", "\n", "<eos>"])
    x, y = TR.make_example(prompt + target, len(prompt))
    X, Y = TR.relabel_rows(x[None].clone(), y[None].clone(), random.Random(3), 1.0)
    lut = {a: b for a, b in zip(x.tolist(), X[0].tolist())}
    assert all(a == b or a in TR.INPUT_IDS for a, b in lut.items())     # only letters change
    keep = y != TR.IGNORE
    assert torch.equal(Y[0] == TR.IGNORE, ~keep)
    assert [lut.get(t, t) for t in y[keep].tolist()] == Y[0][keep].tolist()


def test_relabel_off_is_identity():
    X = torch.tensor([[T.TOKEN_ID["A"], T.TOKEN_ID["B"]]])
    Y = X.clone()
    X2, Y2 = TR.relabel_rows(X.clone(), Y.clone(), random.Random(0), 0.0)
    assert torch.equal(X, X2) and torch.equal(Y, Y2)


def test_relabel_netlist_changes_only_names():
    nl = {"id": "x", "inputs": ["A", "B"], "nets": ["A", "B", "OUT", "n1"], "ports": ["A", "B", "OUT"],
          "canonical_tree": "S(A,B)", "function": "7",
          "devices": [{"gate": "A", "drain": "OUT", "source": "n1", "w": 0.42},
                      {"gate": "B", "drain": "n1", "source": "VSS", "w": 0.42}]}
    out = TR.relabel_netlist(nl, {"A": "C", "B": "F"})
    assert out["inputs"] == ["C", "F"] and out["ports"] == ["C", "F", "OUT"]
    assert out["nets"] == ["C", "F", "OUT", "n1"] and out["canonical_tree"] == "S(C,F)"
    assert [d["gate"] for d in out["devices"]] == ["C", "F"]
    assert [(d["drain"], d["source"], d["w"]) for d in out["devices"]] == \
        [(d["drain"], d["source"], d["w"]) for d in nl["devices"]]
    assert out["function"] == nl["function"] and nl["inputs"] == ["A", "B"]


def test_rename_nets_changes_only_internal_net_names():
    nl = {"inputs": ["A", "B"], "nets": ["A", "B", "OUT", "n1", "p1"], "devices": [
        {"type": "nmos", "gate": "A", "drain": "OUT", "source": "n1", "w": 0.42},
        {"type": "nmos", "gate": "B", "drain": "n1", "source": "VSS", "w": 0.42},
        {"type": "pmos", "gate": "A", "drain": "OUT", "source": "p1", "w": 0.63},
        {"type": "pmos", "gate": "B", "drain": "p1", "source": "VDD", "w": 0.63}]}
    plan = ("cell x\nin A B\nout OUT\ncol 0 sd n=OUT* p=p1*\ncol 1 g n=A p=A\ncol 2 sd n=n1 p=OUT*\n"
            "route p1 met1 p1 pmos:0 pmos:4\npin A gate:1 c0\n")
    out_nl, out_plan = TR.rename_nets(nl, plan, random.Random(0))
    n_new = out_nl["devices"][0]["source"]
    p_new = out_nl["devices"][2]["source"]
    assert n_new.startswith("n") and p_new.startswith("p")
    assert out_nl["devices"][1]["drain"] == n_new and out_nl["devices"][3]["drain"] == p_new
    assert f"n={n_new} " in out_plan and f"p={p_new}*" in out_plan
    assert f"route {p_new} met1 p1 pmos:0" in out_plan          # track p1 untouched
    assert [d["gate"] for d in out_nl["devices"]] == ["A", "B", "A", "B"]
    assert nl["devices"][0]["source"] == "n1"                   # input not mutated


def test_extended_letters_are_appended_and_reachable():
    n_ext = len(T.INPUTS_EXT)
    tail = T.INPUTS_EXT + T.WIDTHS_STACK5 + T.WIDTHS_STACK6                        # appended after every older token
    assert T.VOCAB[-len(tail):] == tail
    assert T.INPUTS == list("ABCDEFGHIJKL")
    from cellgym.synth.train import letter_map
    rng = random.Random(0)
    hit = set()
    for _ in range(200):
        hit |= set(letter_map(5, rng).values())
    assert {T.TOKEN_ID[c] for c in T.INPUTS_EXT} <= hit


def test_old_checkpoint_grows_vocab(tmp_path):
    import torch
    from dataclasses import asdict
    from cellgym.synth import model as M
    old = len(T.VOCAB) - len(T.INPUTS_EXT) - len(T.WIDTHS_STACK5) - len(T.WIDTHS_STACK6)
    cfg = M.Config(vocab=old, d_model=32, layers=1, heads=2, max_len=64)
    torch.manual_seed(0)
    small = M.GPT(cfg)
    torch.save({"state": small.state_dict(), "config": asdict(cfg)}, tmp_path / "ck.pt")
    m = M.load(tmp_path / "ck.pt")
    assert m.cfg.vocab == len(T.VOCAB) and m.tok.weight.shape[0] == len(T.VOCAB)
    assert torch.equal(m.tok.weight[:old], small.tok.weight)
    assert m.head.weight is m.tok.weight
    new = m.tok.weight[old:]
    assert not torch.isnan(new).any() and len({tuple(r.tolist()) for r in new}) == len(new)
