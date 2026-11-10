"""Rotary positions and value embeddings (model.Config pos="rope", num_embed=True)."""
import torch

from cellgym.synth import model as M
from cellgym.synth import tokens as T


def small(**kw):
    torch.manual_seed(0)
    return M.GPT(M.Config(d_model=32, layers=2, heads=2, max_len=128, **kw)).eval()


def test_rope_is_shift_invariant():
    """Relative positions: the same tokens at positions moved by a constant give the same logits."""
    m = small(pos="rope", num_embed=True)
    idx = torch.randint(0, len(T.VOCAB), (2, 20))
    a = m(idx)
    b = m(idx, pos=torch.arange(37, 57)[None].expand(2, -1))
    assert torch.allclose(a, b, atol=1e-4)


def test_learned_positions_are_not():
    m = small()
    idx = torch.randint(0, len(T.VOCAB), (1, 20))
    assert not torch.allclose(m(idx), m(idx, start=30), atol=1e-4)


def test_cached_decoding_matches_full_pass():
    m = small(pos="rope", num_embed=True)
    idx = torch.randint(0, len(T.VOCAB), (1, 12))
    full = m(idx)[:, -1]
    caches = [[] for _ in m.blocks]
    m(idx[:, :8], caches, start=0)
    for t in range(8, 12):
        last = m(idx[:, t:t + 1], caches, start=t)[:, -1]
    assert torch.allclose(full, last, atol=1e-4)


def test_value_rows_follow_value_and_tie_to_head():
    m = small(pos="rope", num_embed=True)
    tab = m.table()
    n = [T.TOKEN_ID[str(i)] for i in range(40)]
    # value rows come from ValueEmbed, not the free table (the output head reuses this table)
    assert torch.allclose(tab[n[25]], m.num_embed(torch.tensor([25.0]))[0], atol=1e-6)
    w = [k for k, v in T.WIDTH_VALUES.items() if v == 15][0]          # w3.15, never in the corpus
    assert torch.allclose(tab[w], m.width_embed(torch.tensor([15.0]))[0], atol=1e-6)
    # neighbouring numbers are closer than distant ones (smooth in value)
    d = lambda a, b: (tab[n[a]] - tab[n[b]]).norm()
    assert d(20, 21) < d(20, 35)


def test_value_embeddings_get_gradients_and_checkpoint_round_trip(tmp_path):
    m = small(pos="rope", num_embed=True).train()
    idx = torch.randint(0, len(T.VOCAB), (2, 10))
    m(idx).logsumexp(-1).mean().backward()
    assert m.num_embed.proj.weight.grad.abs().sum() > 0
    M.save(m, m.cfg, tmp_path / "x.pt")
    m2 = M.load(tmp_path / "x.pt")
    assert m2.cfg.pos == "rope" and m2.cfg.num_embed
    with torch.no_grad():
        assert torch.allclose(m.eval()(idx), m2(idx), atol=1e-5)
