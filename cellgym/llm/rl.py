"""Group-relative policy gradient for the step-wise placement policy (width or area reward).

    python -m cellgym.llm.rl --init out/llm/step05s1/model --out out/llm/rl1 [--steps 150] [--cells 32] [--group 8]

Episode = one full placement of a cell by the policy (Placer env, legal moves only, sampled at T=1),
routed by the Euler baseline's router and scored by the repo pipeline. Reward: 0 if it fails, else
1 + (W_euler - W) / W_euler (bench reward). Advantage = (r - group mean) / (group std + 1e-4) over the
G placements of one cell; with --lam-dens L the reward is lowered by
L * max(0, D - 2), D = channel density of the placement (cellgym.llm.congest), a size-free
routability signal (density <= 3 routes every OOD sample, 5-6 mostly fails); every step's action in an episode gets its episode's advantage. Loss =
-mean over episodes of (A * mean over steps of log p(action | state)), one on-policy update per batch,
plus beta * (logp - logp_ref) on the same actions as a cheap KL anchor.

--reward area: the placement is routed at its smallest-area channel-track count
(route_min_area) and the reward is max(0.05, 1 + (A_ref - A) / A_ref), A = width x height, A_ref = the
cell's Euler placement routed the same way (out/arearef/corpus.json, cellgym.llm.arearef; computed
at start if missing; cells without a reference are dropped). The log then also has mean tracks and
mean A / A_ref ("area").

Diversity. The entropy of the legal-move distribution (per action token, the softmax renormalized over
the tokens the legal-move trie allows; summed over an action's tokens = one step's entropy) is logged
every step ("ent" = mean over episodes of the mean per-step entropy) with "distinct" = mean number of
distinct placements per group. An entropy bonus is on when the coefficient is > 0: loss -= coef * ent,
normalized like the policy term. --ent-target F makes the coefficient adaptive (as SAC's alpha): H0 =
"ent" of the first batch (before any update); after each update
coef *= exp(ent_lr * (F*H0 - ent) / max(F*H0, 1e-6)), clipped to [0, 1]. With a bonus, groups whose
rewards are all equal (no policy-gradient signal: the collapsed ones) still take part in the update with
advantage 0, so the bonus and the KL anchor act on them. <out>/ent_state.json (h0, coef, steps) is
rewritten every step; --ent-state continues from one. --select chooses what "best" means on val.

Training cells: synth5 corpus cells of the TRAIN topology split only (<= 5 inputs, <= 9 per side,
stack <= 4), with their Euler width. Selection: in-distribution val (benchmark val + synth val
topologies), never OOD sets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from cellgym.llm import fmt as F
from cellgym.llm.congest import density
from cellgym.llm.infer import LM


def _area_score(job):
    """Area reward at the smallest-area channel (route_min_area): max(0.05, 1 + (A_ref - A) / A_ref),
    A = width x height, A_ref = the cell's Euler placement routed the same way (cellgym.llm.arearef,
    job[6], nm^2). Returns (reward, W/W_euler, tracks, A/A_ref)."""
    from cellgym.baselines.common import BaselineError, route_min_area
    from cellgym.bench.evaluate import evaluate
    nl, cols, we = job[:3]
    aref = job[6]
    try:
        rt = route_min_area([[tuple(c) for c in cols]], nl, t_max=job[4])
        ev = evaluate(rt.text, nl)
    except BaselineError:
        return 0.0, None, None, None
    except Exception:  # noqa: BLE001
        return 0.0, None, None, None
    if ev.stage != "pass":
        return 0.0, None, None, None
    a = ev.width_nm * ev.height_nm
    return max(0.05, 1.0 + (aref - a) / aref), ev.width_nm / we, rt.tracks, a / aref


def _score_job(job):
    """(reward, W/W_euler or None, tracks or None, A/A_ref or None) under the job's reward kind (job[5])."""
    if len(job) > 5 and job[5] == "area":
        return _area_score(job)
    r, wr = _route_score(job)
    return r, wr, (3 if wr else None), None


def _route_score(job):
    r, wr = _route_score1(job[:3])
    if len(job) > 3 and job[3]:
        from cellgym.llm.congest import density
        r -= job[3] * max(0, density(job[1]) - 2) if job[1] else 0.0
    return r, wr


def _route_score1(job):
    from cellgym.baselines.common import BaselineError, route_best
    from cellgym.bench.evaluate import evaluate
    nl, cols, we = job
    try:
        rt = route_best([[tuple(c) for c in cols]], nl)
        ev = evaluate(rt.text, nl)
    except BaselineError:
        return 0.0, None
    except Exception:  # noqa: BLE001
        return 0.0, None
    if ev.stage != "pass":
        return 0.0, None
    return 1.0 + (we - ev.width_nm) / we, ev.width_nm / we


def legal_logp_entropy(logits, idx, tgt):
    """logits [N, V] at N action-token positions; idx [N, K] the token ids legal at each position,
    padded with -1; tgt [N] the token taken. Returns (logp [N], entropy [N]) of the distribution
    renormalized over the legal tokens (the sampling distribution at T=1)."""
    pad = idx < 0
    sub = logits.gather(-1, idx.clamp_min(0)).masked_fill(pad, float("-inf"))
    g = logits.gather(-1, tgt[:, None])[:, 0] - torch.logsumexp(sub, -1)
    lp = torch.log_softmax(sub, -1).masked_fill(pad, 0.0)
    return g, -(lp.exp().masked_fill(pad, 0.0) * lp).sum(-1)


def ent_coef_update(coef: float, h: float, h0: float, target: float, lr: float) -> float:
    """Adaptive entropy coefficient: raise it while the batch entropy h is under target * h0, lower it
    while above; multiplicative, clipped to [0, 1]. target <= 0: unchanged."""
    if target <= 0:
        return coef
    t = target * h0
    return min(1.0, max(0.0, coef * math.exp(lr * (t - h) / max(t, 1e-6))))


def select_score(v: dict, mode: str):
    """Checkpoint-selection score of an eval record (higher is better; None = not selectable).
    reward: mean reward; best_area: -best-of-k area ratio; both: -(mean area + best area) / 2."""
    if v.get("w") is None:
        return None
    if mode == "reward":
        return v["reward"]
    if v.get("best_area") is None or v.get("area") is None:
        return None
    return -v["best_area"] if mode == "best_area" else -(v["area"] + v["best_area"]) / 2


def distinct_per_group(cols_list, group: int) -> float:
    """Mean number of distinct placements per consecutive group of `group` rollouts."""
    n = [len({tuple(map(tuple, c)) for c in cols_list[i:i + group]}) for i in range(0, len(cols_list), group)]
    return sum(n) / len(n) if n else 0.0


def save_ent_state(path: Path, h0, coef: float, steps: int) -> None:
    Path(path).write_text(json.dumps({"h0": h0, "coef": coef, "steps": steps}) + "\n")


def load_ent_state(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    return {"h0": d["h0"], "coef": float(d["coef"]), "steps": int(d.get("steps", 0))}


def corpus_cells(plans: Path):
    from cellgym.synth.train import split_topologies
    rows = [json.loads(l) for l in plans.read_text().splitlines()]
    tr, va = split_topologies([r["topology_id"] for r in rows], 0.05)
    tr, va = set(tr), set(va)
    cells = {}
    for r in rows:
        w = round(r["width_um"] * 1000)
        c = cells.get(r["id"])
        if c is None or w < c[1]:          # narrowest Euler plan of the cell
            cells[r["id"]] = (r["netlist"], w, "train" if r["topology_id"] in tr else "val")
    return cells


class Policy:
    def __init__(self, path, lr, beta):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.lm = LM.__new__(LM)
        self.lm.torch = torch
        self.lm.tok = AutoTokenizer.from_pretrained(path)
        self.lm.tok.padding_side = "left"
        self.lm.pad = self.lm.tok.pad_token_id if self.lm.tok.pad_token_id is not None else self.lm.tok.eos_token_id
        self.lm.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).cuda()
        self.lm.batch = 512
        self.lm._enc = {}
        self.ref = None
        if beta > 0:
            self.ref = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)
        self.beta = beta
        self.opt = torch.optim.AdamW(self.lm.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    def rollout(self, nls, temperature):
        """One placement per netlist; returns (status, cols, [(state, action_text, legal_trie)]) each."""
        m = self.lm.model
        m.eval()
        jobs = [[F.Placer(nl), 4 * len(nl["devices"]) + 4, "run", []] for nl in nls]
        while True:
            act = [j for j in jobs if j[2] == "run"]
            if not act:
                break
            live, prompts, tries = [], [], []
            for j in act:
                legal = j[0].legal()
                if not legal:
                    j[2] = "stuck"
                    continue
                live.append(j)
                prompts.append(j[0].state())
                tries.append(self.lm.trie([" " + F.col_text(c) for c in legal]))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res = self.lm.generate(prompts, temperature, 16, 1, tries)
            for j, p, r, tr in zip(live, prompts, res, tries):
                c = F.parse_col(r[0])
                if c is None or not j[0].step(c):
                    j[2] = "illegal"
                    continue
                j[3].append((p, " " + F.col_text(c), tr))
                if j[0].done():
                    j[2] = "done"
                elif len(j[0].cols) >= j[1]:
                    j[2] = "cap"
        m.train()
        return [(j[2], j[0].cols, j[3]) for j in jobs]

    def _logps(self, model, pairs, ent=False):
        """log p(action | state) under the sampling distribution: at every action token the
        softmax is renormalized over the tokens the legal-move trie allows there. ent=True:
        also the entropy of those distributions summed over the action's tokens, per pair."""
        eos, pad = self.lm.tok.eos_token_id, self.lm.pad
        seqs, allowed = [], []
        for s, a, trie in pairs:
            p, t = self.lm.enc(s), self.lm.enc(a) + [eos]
            seqs.append((p, t))
            node = trie
            for tk in t:
                allowed.append(list(node.keys()) or [eos])
                node = node.get(tk, {})
        L = max(len(p) + len(t) for p, t in seqs)
        x = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros_like(x)
        tgt = torch.full((len(seqs), L), -100, dtype=torch.long)
        for b, (p, t) in enumerate(seqs):
            s = p + t
            x[b, :len(s)] = torch.tensor(s)
            att[b, :len(s)] = 1
            tgt[b, len(p) - 1:len(s) - 1] = torch.tensor(t)
        x, att, tgt = x.cuda(), att.cuda(), tgt.cuda()
        m = tgt != -100
        K = max(map(len, allowed))
        idx = torch.full((len(allowed), K), -1, dtype=torch.long)
        for i, al in enumerate(allowed):
            idx[i, :len(al)] = torch.tensor(al)
        idx = idx.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=x, attention_mask=att).last_hidden_state
            logits = model.lm_head(h[m]).float()          # target positions only, row-major = allowed order
        g, e = legal_logp_entropy(logits, idx, tgt[m])
        row = torch.arange(len(seqs), device=x.device)[:, None].expand_as(m)[m]
        lp = torch.zeros(len(seqs), device=x.device).index_add(0, row, g)   # per pair
        if not ent:
            return lp
        return lp, torch.zeros(len(seqs), device=x.device).index_add(0, row, e)

    def update(self, episodes, micro=48, ent_coef=0.0):
        """episodes: [(advantage, [(state, action)])]; one gradient step. Returns (loss, mean
        logp - logp_ref, mean over episodes of the mean per-step entropy). ent_coef > 0 adds
        -ent_coef * that entropy to the loss."""
        pairs, w, we = [], [], []
        for adv, steps in episodes:
            for s, a, tr in steps:
                pairs.append((s, a, tr))
                w.append(adv / len(steps))
                we.append(1.0 / len(steps))
        n_ep = max(1, len(episodes))
        tot, kl_sum, ent_sum = 0.0, 0.0, 0.0
        for c in range(0, len(pairs), micro):
            pp, ww = pairs[c:c + micro], torch.tensor(w[c:c + micro], device="cuda")
            lp, en = self._logps(self.lm.model, pp, ent=True)
            en = (torch.tensor(we[c:c + micro], device="cuda") * en).sum() / n_ep
            ent_sum += float(en.detach())
            loss = -(ww * lp).sum() / n_ep
            if ent_coef > 0:
                loss = loss - ent_coef * en
            if self.ref is not None:
                with torch.no_grad():
                    lr_ = self._logps(self.ref, pp)
                kl = (lp - lr_)
                loss = loss + self.beta * (kl ** 2).sum() / max(1, len(pairs)) * 0.5
                kl_sum += float(kl.detach().sum())
            loss.backward()
            tot += float(loss.detach())
            del lp, loss, en
        torch.nn.utils.clip_grad_norm_(self.lm.model.parameters(), 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return tot, kl_sum / max(1, len(pairs)), ent_sum


def area_refs(plans: Path, cache: Path, jobs: int) -> dict[str, int]:
    """{corpus cell: Euler reference area, nm^2}, from the arearef cache (computed if missing)."""
    from cellgym.llm import arearef
    refs = arearef.load(arearef.CORPUS, cache)
    if refs is None:
        t0 = time.time()
        with ProcessPoolExecutor(jobs) as pool:
            refs = arearef.build(arearef.CORPUS, arearef.corpus_placements(plans), pool, cache)
        print(f"area references computed: {arearef.describe(refs)} ({time.time() - t0:.0f}s)", flush=True)
    return arearef.areas_nm2(refs)


def _mean(x):
    return sum(x) / len(x) if x else None


def evaluate(pol, pool, cells, k, temperature, lam=0.0, kind="width", t_max=8, aref=None):
    from cellgym.llm.congest import density
    ids = sorted(cells)
    nls = [cells[c][0] for c in ids for _ in range(k)]
    ro = pol.rollout(nls, temperature)
    rep_ids = [c for c in ids for _ in range(k)]
    jobs = [(cells[c][0], cols, cells[c][1], lam, t_max, kind, (aref or {}).get(c)) for c, (st, cols, _) in zip(rep_ids, ro)]
    res3 = list(pool.map(_score_job, jobs, chunksize=4))
    res = [(r, wr) for r, wr, _, _ in res3]
    tr = [t for _, _, t, _ in res3 if t]
    by_a = {}
    for c, x in zip(rep_ids, res3):
        if x[3]:
            by_a.setdefault(c, []).append(x[3])
    ok = [wr for r, wr in res if wr]
    by = {}
    for c, (r, wr) in zip([c for c in ids for _ in range(k)], res):
        by.setdefault(c, []).append(wr)
    best = [min(x for x in v if x) for v in by.values() if any(v)]
    return {"pass": len(ok) / len(res), "w": sum(ok) / len(ok) if ok else None,
            "best_w": sum(best) / len(best) if best else None,
            "reward": sum(r for r, _ in res) / len(res),
            "tracks": sum(tr) / len(tr) if tr else None,
            "area": _mean([y for v in by_a.values() for y in v]),
            "best_area": _mean([min(v) for v in by_a.values()]),
            "distinct": distinct_per_group([cols for _, cols, _ in ro], k),
            "dens": sum(density(cols) for _, cols, _ in ro if cols) / max(1, sum(1 for _, cols, _ in ro if cols))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plans", type=Path, default=Path("out/synth5/plans.jsonl"))
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--cells", type=int, default=32, help="cells per step")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-cells", type=int, default=96)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam-dens", type=float, default=0.0, help="reward penalty per density unit above 2")
    ap.add_argument("--reward", choices=("width", "area"), default="width",
                    help="area: width x height at the smallest-area channel-track count, against the "
                         "Euler placement routed the same way")
    ap.add_argument("--arearef", type=Path, default=Path("out/arearef"), help="Euler area reference cache (area reward)")
    ap.add_argument("--t-max", type=int, default=8, help="most channel tracks tried (area reward)")
    ap.add_argument("--ent-target", type=float, default=0.0,
                    help="hold the mean per-step entropy at this fraction of the first batch's (0 = fixed coefficient)")
    ap.add_argument("--ent-coef", type=float, default=None,
                    help="entropy bonus coefficient (initial one with --ent-target); default 0.02 with "
                         "--ent-target, else 0 = no bonus")
    ap.add_argument("--ent-lr", type=float, default=0.1, help="step of the adaptive coefficient")
    ap.add_argument("--ent-state", type=Path, default=None, help="ent_state.json of an earlier run to continue (h0, coef)")
    ap.add_argument("--step-offset", type=int, default=0, help="added to the step numbers in logs and checkpoint names")
    ap.add_argument("--select", choices=("reward", "best_area", "both"), default="reward",
                    help="val criterion for <out>/best: mean reward, best-of-k area, or the mean of mean and best area")
    ap.add_argument("--keep-ckpts", action="store_true", help="also save every eval checkpoint as <out>/ckpt_<step>")
    a = ap.parse_args(argv)
    if a.select != "reward" and a.reward != "area":
        ap.error("--select best_area/both needs --reward area")
    ent_coef = a.ent_coef if a.ent_coef is not None else (0.02 if a.ent_target > 0 else 0.0)
    h0, off = None, a.step_offset
    if a.ent_state is not None:
        es = load_ent_state(a.ent_state)
        h0, ent_coef = es["h0"], es["coef"]
        print(f"entropy state from {a.ent_state}: h0={h0} coef={ent_coef} after {es['steps']} steps", flush=True)
    ent_on = ent_coef > 0 or a.ent_target > 0
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    allc = corpus_cells(a.plans)
    aref = None
    if a.reward == "area":
        aref = area_refs(a.plans, a.arearef, a.jobs)
        n_all = len(allc)
        allc = {c: v for c, v in allc.items() if c in aref}
        print(f"area reward: {n_all - len(allc)} of {n_all} cells dropped (no area reference)", flush=True)
    train = [c for c, v in allc.items() if v[2] == "train"]
    val_ids = random.Random(1).sample(sorted(c for c, v in allc.items() if v[2] == "val"), a.eval_cells)
    val = {c: allc[c] for c in val_ids}
    pol = Policy(a.init, a.lr, a.beta)
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "log.jsonl", "a")

    def save(path):
        sd = {k: v.detach().to(torch.bfloat16) for k, v in pol.lm.model.state_dict().items()}
        pol.lm.model.save_pretrained(path, state_dict=sd)
        pol.lm.tok.save_pretrained(path)
    best = None
    print(f"{len(train)} train cells, {len(val)} val cells", flush=True)
    with ProcessPoolExecutor(a.jobs) as pool:
        def check(step):
            nonlocal best
            v = evaluate(pol, pool, val, 4, 0.7, a.lam_dens, a.reward, a.t_max, aref)
            rec = {"step": step + off, "eval": "val", **v}
            sc = select_score(v, a.select)
            if sc is not None and (best is None or sc > best):
                best = sc
                save(a.out / "best")
                rec["best"] = True
            if a.keep_ckpts and step > 0:      # step 0 is the --init model
                save(a.out / f"ckpt_{step + off}")
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        check(0)
        for step in range(1, a.steps + 1):
            t0 = time.time()
            ids = rng.sample(train, a.cells)
            nls = [allc[c][0] for c in ids for _ in range(a.group)]
            ro = pol.rollout(nls, a.temperature)
            torch.cuda.empty_cache()
            t1 = time.time()
            jobs = [(allc[c][0], cols, allc[c][1], a.lam_dens, a.t_max, a.reward, (aref or {}).get(c))
                    for c, (_, cols, _) in zip([c for c in ids for _ in range(a.group)], ro)]
            res3 = list(pool.map(_score_job, jobs, chunksize=2))
            res = [(r, wr) for r, wr, _, _ in res3]
            trs = [t for _, _, t, _ in res3 if t]
            t2 = time.time()
            eps, flat = [], 0
            for g in range(a.cells):
                rs = torch.tensor([res[g * a.group + j][0] for j in range(a.group)])
                if float(rs.std()) < 1e-6:
                    flat += 1
                    if not ent_on:
                        continue
                    adv = torch.zeros(a.group)      # no reward signal; entropy bonus and KL anchor only
                else:
                    adv = (rs - rs.mean()) / (rs.std() + 1e-4)
                for j in range(a.group):
                    steps = ro[g * a.group + j][2]
                    if steps:
                        eps.append((float(adv[j]), steps))
            loss, kl, ent = pol.update(eps, ent_coef=ent_coef) if eps else (0.0, 0.0, None)
            coef_used = ent_coef
            if ent is not None:
                if h0 is None:
                    h0 = ent                       # first batch, measured before its update
                ent_coef = ent_coef_update(ent_coef, ent, h0, a.ent_target, a.ent_lr)
            ok = [wr for _, wr in res if wr]
            rec = {"step": step + off, "reward": sum(r for r, _ in res) / len(res), "pass": len(ok) / len(res),
                   "w": sum(ok) / len(ok) if ok else None, "flat_groups": flat,
                   "tracks": sum(trs) / len(trs) if trs else None,
                   "area": _mean([x[3] for x in res3 if x[3]]),
                   "dens": sum(density(cols) for _, cols, _ in ro if cols) / max(1, sum(1 for _, cols, _ in ro if cols)), "loss": loss, "kl": kl,
                   "ent": ent, "ent_coef": coef_used, "distinct": distinct_per_group([cols for _, cols, _ in ro], a.group),
                   "t_roll": round(t1 - t0, 1), "t_score": round(t2 - t1, 1), "t_upd": round(time.time() - t2, 1)}
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            save_ent_state(a.out / "ent_state.json", h0, ent_coef, step + off)
            if step % a.eval_every == 0 or step == a.steps:
                check(step)
    save(a.out / "last")
    return 0


if __name__ == "__main__":
    sys.exit(main())
