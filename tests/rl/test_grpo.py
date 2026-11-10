import json
from pathlib import Path

import torch

from cellgym.rl import grpo
from cellgym.rl.env import make_env
from cellgym.synth import model as M


def tiny():
    torch.manual_seed(0)
    return M.GPT(M.Config(d_model=32, layers=2, heads=2, max_len=512))


def test_advantages():
    r = torch.tensor([0., 1., 1., 0.,   0., 0., 0., 0.,   2., 2., 2., 2.,   1., 2., 3., 4.])
    a = grpo.advantages(r, 4)
    assert torch.allclose(a[:4], torch.tensor([-1., 1., 1., -1.]))
    assert torch.all(a[4:12] == 0)
    assert abs(float(a[12:].mean())) < 1e-6 and torch.isfinite(a).all()


def test_loss_grads_and_steps():
    env = make_env("train", cache=None)
    model, ref = tiny(), tiny()
    cids = env.ids()[:2]
    ro = grpo.rollout(model, env, cids, 3, 1.0, 40)
    assert len(ro.completions) == 6 and all(len(o) == len(c) for o, c in zip(ro.old_logp, ro.completions))
    adv = torch.tensor([1., -1., 0.5, -0.5, 1., -1.])
    loss, st = grpo.grpo_loss(model, ro, adv, list(range(6)), ref, beta=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert st["clipfrac"] == 0 and abs(st["kl"]) < 1e-6      # on-policy, ref == model

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        m = grpo.train_step(model, opt, env, cids, 3, ref, max_new=30, beta=0.05, micro_bs=2)
        assert 0 <= m["pass_rate"] <= 1 and torch.isfinite(torch.tensor(m["mean_reward"]))


def test_pluggable_sampler_and_cli(tmp_path):
    env = make_env("train", cache=None)
    calls = []

    def fixed(model, prompt, n, temperature, max_new):
        calls.append(n)
        return [[5, 6, 3]] * n
    model = tiny()
    ro = grpo.rollout(model, env, env.ids()[:1], 2, sampler=fixed)
    assert calls == [2] and ro.completions == [[5, 6, 3]] * 2
    ro = grpo.rollout(model, env, env.ids()[:2], 2, max_new=200, sampler="constrained")
    assert len(ro.completions) == 4 and all(ro.completions)

    ck = tmp_path / "t.pt"
    M.save(model, model.cfg, ck)
    rc = grpo.main(["--ckpt", str(ck), "--out", str(tmp_path / "o"), "--steps", "2", "--groups", "2",
                    "--group-size", "2", "--max-new", "20", "--jobs", "1", "--device", "cpu",
                    "--save-every", "2"])
    assert rc == 0
    rows = [json.loads(x) for x in (tmp_path / "o" / "log.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and {"mean_reward", "pass_rate", "frac_zero_groups", "mean_w_ratio"} <= rows[0].keys()
    assert (tmp_path / "o" / "last.pt").exists()


def test_sample_many_matches_per_prompt_greedy():
    model = tiny()
    prompts = [[1, 7, 9, 4], [1, 5], [1, 8, 8, 8, 8, 8, 2], [1, 6, 4]]
    batched = M.sample_many(model, prompts, temperature=0, max_new=12, chunk=3)
    single = [M.sample(model, p, n=1, temperature=0, max_new=12)[0] for p in prompts]
    assert batched == single


def test_sample_many_respects_max_len():
    torch.manual_seed(0)
    model = M.GPT(M.Config(d_model=32, layers=2, heads=2, max_len=16))
    out = M.sample_many(model, [[1] * 10, [1] * 4], temperature=1.0, max_new=50)
    assert len(out[0]) <= 6 and len(out[1]) <= 12


def test_batched_rollout_groups_adjacent():
    env = make_env("train", cache=None)
    cids = env.ids()[:3]
    ro = grpo.rollout(tiny(), env, cids, 2, 1.0, 20, sample_chunk=4)
    assert ro.cell_ids == [c for c in cids for _ in range(2)]
    assert all(p == env.prompt_ids(c) for p, c in zip(ro.prompts, ro.cell_ids))


def test_kl_clamp_bounds_penalty():
    env = make_env("train", cache=None)
    model, ref = tiny(), tiny()
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 2.0)          # far from model: big log-ratios
    ro = grpo.rollout(model, env, env.ids()[:1], 2, 1.0, 30)
    adv = torch.zeros(2)
    _, free = grpo.grpo_loss(model, ro, adv, [0, 1], ref, beta=1.0)
    _, capped = grpo.grpo_loss(model, ro, adv, [0, 1], ref, beta=1.0, kl_clamp=1.0)
    assert capped["kl"] <= free["kl"] and capped["kl"] <= (2.718281828 - 2) + 1e-4


def test_cli_val_eval_writes_best(tmp_path):
    model = tiny()
    ck = tmp_path / "t.pt"
    M.save(model, model.cfg, ck)
    rc = grpo.main(["--ckpt", str(ck), "--out", str(tmp_path / "o"), "--steps", "2", "--groups", "2",
                    "--group-size", "2", "--max-new", "20", "--jobs", "1", "--device", "cpu",
                    "--eval-every", "1", "--eval-k", "2", "--kl-clamp", "3"])
    assert rc == 0
    rows = [json.loads(x) for x in (tmp_path / "o" / "log.jsonl").read_text().splitlines()]
    evals = [r for r in rows if r.get("eval") == "val"]
    assert [r["step"] for r in evals] == [0, 1, 2] and evals[0]["best"]
    assert (tmp_path / "o" / "best.pt").exists()


def test_cli_ood6_pool_and_dev_selection(tmp_path):
    import shutil
    import pytest
    src = Path("out/ood6")
    if not (src / "index.json").exists():
        pytest.skip("needs out/ood6 (python -m cellgym.synth.devset build)")
    index = json.loads((src / "index.json").read_text())
    pick = {p: next(c for c, v in sorted(index.items()) if v["part"] == p) for p in ("pool", "dev")}
    d = tmp_path / "ood6"
    (d / "cells").mkdir(parents=True)
    for c in pick.values():
        shutil.copy(src / "cells" / f"{c}.json", d / "cells" / f"{c}.json")
    (d / "index.json").write_text(json.dumps({c: index[c] for c in pick.values()}))
    model = tiny()
    ck = tmp_path / "t.pt"
    M.save(model, model.cfg, ck)
    rc = grpo.main(["--ckpt", str(ck), "--out", str(tmp_path / "o"), "--steps", "1", "--groups", "2",
                    "--group-size", "2", "--max-new", "20", "--jobs", "1", "--device", "cpu",
                    "--eval-every", "1", "--eval-k", "2", "--ood6", str(d), "--ood6-frac", "0.9"])
    assert rc == 0
    rows = [json.loads(x) for x in (tmp_path / "o" / "log.jsonl").read_text().splitlines()]
    dev = [r for r in rows if r.get("eval") == "ood6_dev"]
    val = [r for r in rows if r.get("eval") == "val"]
    assert [r["step"] for r in dev] == [0, 1] and [r["step"] for r in val] == [0, 1]
    assert dev[0]["best"] and not any(r["best"] for r in val)
