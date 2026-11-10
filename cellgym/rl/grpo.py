"""GRPO for the tiny plan model.

    python -m cellgym.rl.grpo --ckpt out/synth/run0/best.pt --out out/rl/run0 --steps 200
        --groups 16 --group-size 8 --lr 1e-5 [--synth out/synth/plans.jsonl] [--beta 0.02] [--bf16]

Per step: draw `groups` train cells, sample `group_size` plans each (default: all rows batched across
cells with cellgym.synth.model.sample_many; a pluggable per-cell callable, e.g. a constrained decoder, can replace it), score with PlanEnv, turn the
rewards into group-relative advantages, and take a clipped-surrogate step (optional KL to a frozen
reference copy of the start checkpoint). Logs one JSON line per step to <out>/log.jsonl.

Note: with a constrained sampler the log-probs are still the unconstrained policy's (the mask only
removes probability mass at sampling time).

sampler signature: sampler(model, prompt_ids, n, temperature, max_new) -> list of n id lists,
each cut at and including EOS.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from cellgym.rl.env import CellSampler, PlanEnv, RewardInfo, make_env
from cellgym.synth import model as M
from cellgym.synth import tokens as T


def default_sampler(model, prompt, n, temperature, max_new):
    return M.sample(model, prompt, n=n, temperature=temperature, max_new=max_new)


def advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """Per-group (consecutive blocks of group_size) (r - mean) / std; all-equal groups get 0."""
    r = rewards.float().view(-1, group_size)
    mean = r.mean(1, keepdim=True)
    std = r.std(1, unbiased=False, keepdim=True)
    adv = (r - mean) / std.clamp_min(eps)
    adv = torch.where(std > eps, adv, torch.zeros_like(adv))
    return adv.view(-1)


@dataclass
class Rollout:
    cell_ids: list[str]              # one per sample, groups adjacent
    prompts: list[list[int]]
    completions: list[list[int]]
    old_logp: list[torch.Tensor]     # per sample, (len(completion),) from the sampling model in eval mode
    rewards: torch.Tensor            # (N,)
    infos: list[RewardInfo]
    group_size: int
    seconds: dict


def _pad_batch(prompts, comps, device):
    seqs = [p + c for p, c in zip(prompts, comps)]
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), T.PAD, dtype=torch.long)
    mask = torch.zeros(len(seqs), L - 1)          # mask[b, j]: logit j predicts a completion token
    for b, (p, c) in enumerate(zip(prompts, comps)):
        x[b, :len(p) + len(c)] = torch.tensor(p + c)
        mask[b, len(p) - 1:len(p) + len(c) - 1] = 1
    return x.to(device), mask.to(device)


def token_logps(model, prompts, comps, amp=False):
    """(B, L-1) log-probs of every next token, and the completion mask of the same shape."""
    dev = next(model.parameters()).device
    x, mask = _pad_batch(prompts, comps, dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
        logits = model(x[:, :-1])
    lp = F.log_softmax(logits.float(), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
    return lp, mask


@torch.no_grad()
def _logps_per_sample(model, prompts, comps, micro_bs, amp):
    was = model.training
    model.eval()
    out = []
    for i in range(0, len(prompts), micro_bs):
        lp, mask = token_logps(model, prompts[i:i + micro_bs], comps[i:i + micro_bs], amp)
        for b in range(lp.shape[0]):
            out.append(lp[b][mask[b] > 0].cpu())
    model.train(was)
    return out


def rollout(model, env: PlanEnv, cell_ids: list[str], group_size: int, temperature: float = 1.0,
            max_new: int = 640, sampler=None, jobs: int = 1, micro_bs: int = 16, amp: bool = False,
            sample_chunk: int = 32) -> Rollout:
    """sampler=None samples every (cell, sample) row in batches of sample_chunk across cells
    (model.sample_many); a sampler callable is called once per cell instead; sampler="constrained"
    uses the grammar-constrained decoder (constrain.sample_constrained, device cover on) per cell.
    Old log-probs are always the unmasked model's, so the ratio is unaffected by the mask."""
    t0 = time.time()
    cids, prompts, comps = [], [], []
    if sampler is None:
        cids = CellSampler.expand(cell_ids, group_size)
        prompts = [env.prompt_ids(c) for c in cids]
        comps = [list(s) for s in M.sample_many(model, prompts, temperature, max_new, chunk=sample_chunk)]
    else:
        if sampler == "constrained":
            from cellgym.synth.constrain import sample_constrained

            def sampler(model, p, n, temperature, max_new, _cid=None):
                return sample_constrained(model, p, env.cells[_cid], n=n, temperature=temperature,
                                          max_new=max_new)
        for cid in cell_ids:
            p = env.prompt_ids(cid)
            kw = {"_cid": cid} if "_cid" in sampler.__code__.co_varnames else {}
            for s in sampler(model, p, group_size, temperature, max_new, **kw):
                cids.append(cid), prompts.append(p), comps.append(list(s))
    t1 = time.time()
    infos = env.reward_batch(cids, comps, jobs)
    t2 = time.time()
    old = _logps_per_sample(model, prompts, comps, micro_bs, amp)
    rewards = torch.tensor([i.reward for i in infos])
    return Rollout(cids, prompts, comps, old, rewards, infos, group_size,
                   {"sample": t1 - t0, "reward": t2 - t1, "logp": time.time() - t2,
                    "reward_samples_per_s": len(cids) / max(t2 - t1, 1e-9)})


def grpo_loss(model, ro: Rollout, adv: torch.Tensor, idx: list[int], ref_model=None, beta: float = 0.0,
              clip: float = 0.2, amp: bool = False, kl_clamp: float | None = None) -> tuple[torch.Tensor, dict]:
    """Loss for the samples `idx`: per-sequence token-mean of the clipped surrogate (+ beta * k3 KL),
    summed over idx (the caller divides by the number of sequences). kl_clamp caps the per-token
    log-ratio ref - logp before the k3 estimator (exp of it is otherwise unbounded)."""
    prompts = [ro.prompts[i] for i in idx]
    comps = [ro.completions[i] for i in idx]
    lp, mask = token_logps(model, prompts, comps, amp)
    old = torch.zeros_like(lp)
    for b, i in enumerate(idx):
        n = len(comps[b])
        old[b, len(prompts[b]) - 1:len(prompts[b]) - 1 + n] = ro.old_logp[i].to(lp.device)
    a = adv[idx].to(lp.device)[:, None]
    ratio = torch.exp(lp - old)
    surr = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    tok = -surr
    kl = torch.zeros(())
    if ref_model is not None and beta > 0:
        with torch.no_grad():
            ref_lp, _ = token_logps(ref_model, prompts, comps, amp)
        d = ref_lp - lp
        if kl_clamp is not None:
            d = d.clamp(-kl_clamp, kl_clamp)
        k3 = torch.exp(d) - d - 1
        tok = tok + beta * k3
        kl = ((k3 * mask).sum(1) / mask.sum(1).clamp_min(1)).mean().detach()
    per_seq = (tok * mask).sum(1) / mask.sum(1).clamp_min(1)
    clipfrac = float((((ratio - 1).abs() > clip) * mask).sum() / mask.sum().clamp_min(1))
    return per_seq.sum(), {"kl": float(kl), "clipfrac": clipfrac}


def train_step(model, optimizer, env: PlanEnv, cell_ids: list[str], group_size: int, ref_model=None,
               temperature: float = 1.0, max_new: int = 640, beta: float = 0.0, clip: float = 0.2,
               micro_bs: int = 8, inner_epochs: int = 1, grad_clip: float = 1.0, sampler=None,
               jobs: int = 1, amp: bool = False, sample_chunk: int = 32, kl_clamp: float | None = None) -> dict:
    ro = rollout(model, env, cell_ids, group_size, temperature, max_new, sampler, jobs, micro_bs, amp,
                 sample_chunk)
    adv = advantages(ro.rewards, group_size)
    # flat-group samples carry no policy gradient; without a KL term they are skipped entirely
    use_kl = ref_model is not None and beta > 0
    idx = [i for i in range(len(adv)) if adv[i] != 0 or use_kl]
    loss_v, kl_v, clip_v, gn = float("nan"), 0.0, 0.0, 0.0
    t0 = time.time()
    if idx:
        model.train()
        for _ in range(inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            tot, kls, cfs = 0.0, [], []
            for j in range(0, len(idx), micro_bs):
                sub = idx[j:j + micro_bs]
                loss, st = grpo_loss(model, ro, adv, sub, ref_model, beta, clip, amp, kl_clamp)
                (loss / len(idx)).backward()
                tot += float(loss.detach())
                kls.append(st["kl"]), cfs.append(st["clipfrac"])
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            optimizer.step()
            loss_v, kl_v, clip_v = tot / len(idx), sum(kls) / len(kls), sum(cfs) / len(cfs)
        model.eval()
    r = ro.rewards.view(-1, group_size)
    passed = [i for i in ro.infos if i.passed]
    return {
        "mean_reward": float(ro.rewards.mean()),
        "pass_rate": len(passed) / len(ro.infos),
        "frac_zero_groups": float((r.max(1).values == 0).float().mean()),
        "frac_flat_groups": float((r.std(1, unbiased=False) < 1e-6).float().mean()),
        "mean_w_ratio": sum(i.w_ratio for i in passed) / len(passed) if passed else None,
        "loss": loss_v, "kl": kl_v, "clipfrac": clip_v, "grad_norm": gn,
        "n_train_samples": len(idx),
        "t_sample": ro.seconds["sample"], "t_reward": ro.seconds["reward"], "t_logp": ro.seconds["logp"],
        "t_update": time.time() - t0, "reward_samples_per_s": ro.seconds["reward_samples_per_s"],
    }


@torch.no_grad()
def evaluate_env(model, env: PlanEnv, k: int = 16, temperature: float = 0.8, max_new: int = 640,
                 jobs: int = 1, sample_chunk: int = 32) -> dict:
    """Held-out check during training: k sampled plans per cell plus one greedy plan, unconstrained,
    scored like training. Returns per-sample pass rate, pass@k and greedy pass rate."""
    ids = env.ids()
    rows = CellSampler.expand(ids, k)
    comps = M.sample_many(model, [env.prompt_ids(c) for c in rows], temperature, max_new, chunk=sample_chunk)
    infos = env.reward_batch(rows, comps, jobs)
    greedy = M.sample_many(model, [env.prompt_ids(c) for c in ids], 0, max_new, chunk=sample_chunk)
    ginfos = env.reward_batch(ids, greedy, jobs)
    per_cell = [infos[i * k:(i + 1) * k] for i in range(len(ids))]
    return {"val_sample_pass": sum(x.passed for x in infos) / len(infos),
            "val_pass_at_k": sum(any(x.passed for x in g) for g in per_cell) / len(ids),
            "val_greedy_pass": sum(x.passed for x in ginfos) / len(ids),
            "val_mean_reward": sum(x.reward for x in infos) / len(infos)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.02, help="KL coefficient to the start checkpoint (0 = off)")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--inner-epochs", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sample-chunk", type=int, default=32, help="rows per batched sampling call")
    ap.add_argument("--synth", type=Path, default=None, help="plans.jsonl of synthetic cells to add")
    ap.add_argument("--cells", type=Path, default=Path("out/cells"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--constrained", action="store_true",
                    help="sample rollouts with the grammar-constrained decoder (per cell; slower)")
    ap.add_argument("--kl-clamp", type=float, default=None, help="cap |ref - logp| per token in the KL term")
    ap.add_argument("--eval-every", type=int, default=0, help="val check every N steps (0 = off); best.pt by val")
    ap.add_argument("--eval-k", type=int, default=16)
    ap.add_argument("--weighting", choices=["uniform", "bucket"], default="uniform",
                    help="bucket: draw cells by (inputs, devices) in benchmark-train proportions")
    ap.add_argument("--uniform-mix", type=float, default=0.1)
    ap.add_argument("--ood6", type=Path, default=None,
                    help="cellgym.synth.devset dir: train also on its `pool` cells, select best.pt on its `dev` cells")
    ap.add_argument("--ood6-frac", type=float, default=0.5, help="share of drawn groups from the ood6 pool")
    a = ap.parse_args(argv)
    if a.ood6 and a.weighting != "uniform":
        ap.error("--ood6 sets its own draw weights; use --weighting uniform")

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    env = make_env("train", a.synth, cells_dir=a.cells)
    if a.ood6:
        from cellgym.synth.devset import load as load_set
        pool, pool_w = load_set(a.ood6, "pool")
        clash = sorted(set(pool) & set(env.cells))
        if clash:
            raise ValueError(f"ood6 pool cells already in the env: {clash[:3]}")
        env.cells.update(pool)
        env.w_euler.update(pool_w)
    sampler = CellSampler(env, seed=a.seed)
    if a.weighting == "bucket":
        sampler.set_weights(CellSampler.bucket_weights(env, "train", a.uniform_mix))
    if a.ood6:
        n_o, n_b = len(pool), len(env.cells) - len(pool)
        sampler.set_weights({c: (a.ood6_frac / n_o if c in pool else (1 - a.ood6_frac) / n_b)
                             for c in env.ids()})
    model = M.load(a.ckpt, device)
    env.prompt_fmt = model.cfg.prompt
    ref = None
    if a.beta > 0:
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(env.cells)} training cells, device {device}, params {model.n_params():,}", flush=True)
    val_env, best, checks = None, None, []
    if a.eval_every:
        val_env = make_env("val", cells_dir=a.cells, allow_heldout=True)
        checks.append(("val", val_env))
        if a.ood6:
            dev, dev_w = load_set(a.ood6, "dev")
            checks.insert(0, ("ood6_dev", PlanEnv(dev, dev_w, split_of={}, allow_heldout=True)))
        for _, e in checks:
            e.prompt_fmt = model.cfg.prompt

    def val_check(step, log):
        """Every check env is scored; the first (ood6 dev if given, else val) picks best.pt."""
        nonlocal best
        for i, (name, e) in enumerate(checks):
            torch.manual_seed(10_000 + step)
            v = evaluate_env(model, e, a.eval_k, 0.8, a.max_new, a.jobs, a.sample_chunk)
            is_best = False
            if i == 0:
                key = (v["val_sample_pass"], v["val_greedy_pass"])
                is_best = best is None or key > best
                if is_best:
                    best = key
                    M.save(model, model.cfg, a.out / "best.pt")
            log.write(json.dumps({"step": step, "eval": name, **v, "best": is_best}) + "\n")
            log.flush()
            print(f"{name} step {step} sample {v['val_sample_pass']:.3f} pass@{a.eval_k} "
                  f"{v['val_pass_at_k']:.3f} greedy {v['val_greedy_pass']:.3f}{' *best' if is_best else ''}",
                  flush=True)
        torch.manual_seed(a.seed + step)

    with open(a.out / "log.jsonl", "a") as log:
        if val_env is not None:
            val_check(0, log)
        for step in range(1, a.steps + 1):
            cids = sampler.next_groups(a.groups)
            m = train_step(model, opt, env, cids, a.group_size, ref, a.temperature, a.max_new, a.beta, a.clip,
                           a.micro_bs, a.inner_epochs, sampler="constrained" if a.constrained else None,
                           jobs=a.jobs, amp=a.bf16, sample_chunk=a.sample_chunk,
                           kl_clamp=a.kl_clamp)
            m["step"] = step
            log.write(json.dumps(m) + "\n")
            log.flush()
            print(f"step {step} reward {m['mean_reward']:.3f} pass {m['pass_rate']:.3f} "
                  f"zero {m['frac_zero_groups']:.2f} w/we {m['mean_w_ratio']} kl {m['kl']:.4f}", flush=True)
            if step % a.save_every == 0 or step == a.steps:
                M.save(model, model.cfg, a.out / "last.pt")
            if val_env is not None and (step % a.eval_every == 0 or step == a.steps):
                val_check(step, log)
    env.close()
    for _, e in checks:
        e.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
