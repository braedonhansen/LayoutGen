"""Group-relative policy gradient for the step-wise placement policy (width reward).

    python -m cellgym.llm.rl --init out/llm/step05s1/model --out out/llm/rl1 [--steps 150] [--cells 32] [--group 8]

Episode = one full placement of a cell by the policy (Placer env, legal moves only, sampled at T=1),
routed by the Euler baseline's router and scored by the repo pipeline. Reward: 0 if it fails, else
1 + (W_euler - W) / W_euler (bench reward). Advantage = (r - group mean) / (group std + 1e-4) over the
G placements of one cell; every step's action in an episode gets its episode's advantage. Loss =
-mean over episodes of (A * mean over steps of log p(action | state)), one on-policy update per batch,
plus beta * (logp - logp_ref) on the same actions as a cheap KL anchor.

Training cells: synth5 corpus cells of the TRAIN topology split only (<= 5 inputs, <= 9 per side,
stack <= 4), with their Euler width. Selection: in-distribution val (benchmark val + synth val
topologies), never OOD sets.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from cellgym.llm import fmt as F
from cellgym.llm.infer import LM


def _route_score(job):
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
        """One placement per netlist; returns (status, cols, [(state, action_text)]) each."""
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
            for j, p, r in zip(live, prompts, res):
                c = F.parse_col(r[0])
                if c is None or not j[0].step(c):
                    j[2] = "illegal"
                    continue
                j[3].append((p, " " + F.col_text(c)))
                if j[0].done():
                    j[2] = "done"
                elif len(j[0].cols) >= j[1]:
                    j[2] = "cap"
        m.train()
        return [(j[2], j[0].cols, j[3]) for j in jobs]

    def _logps(self, model, pairs):
        tok, eos, pad = self.lm.tok, self.lm.tok.eos_token_id, self.lm.pad
        seqs = []
        for s, a in pairs:
            p, t = self.lm.enc(s), self.lm.enc(a) + [eos]
            seqs.append((p, t))
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=x, attention_mask=att).last_hidden_state
            logits = model.lm_head(h[m]).float()          # target positions only
        g = torch.log_softmax(logits, -1).gather(-1, tgt[m][:, None])[:, 0]
        row = torch.arange(len(seqs), device=x.device)[:, None].expand_as(m)[m]
        return torch.zeros(len(seqs), device=x.device).index_add(0, row, g)   # per pair

    def update(self, episodes, micro=256):
        """episodes: [(advantage, [(state, action)])]; one gradient step."""
        pairs, w = [], []
        for adv, steps in episodes:
            for s, a in steps:
                pairs.append((s, a))
                w.append(adv / len(steps))
        n_ep = max(1, len(episodes))
        tot, kl_sum = 0.0, 0.0
        for c in range(0, len(pairs), micro):
            pp, ww = pairs[c:c + micro], torch.tensor(w[c:c + micro], device="cuda")
            lp = self._logps(self.lm.model, pp)
            loss = -(ww * lp).sum() / n_ep
            if self.ref is not None:
                with torch.no_grad():
                    lr_ = self._logps(self.ref, pp)
                kl = (lp - lr_)
                loss = loss + self.beta * (kl ** 2).sum() / max(1, len(pairs)) * 0.5
                kl_sum += float(kl.detach().sum())
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(self.lm.model.parameters(), 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return tot, kl_sum / max(1, len(pairs))


def evaluate(pol, pool, cells, k, temperature):
    ids = sorted(cells)
    nls = [cells[c][0] for c in ids for _ in range(k)]
    ro = pol.rollout(nls, temperature)
    jobs = [(cells[c][0], cols, cells[c][1]) for c, (st, cols, _) in zip([c for c in ids for _ in range(k)], ro)]
    res = list(pool.map(_route_score, jobs, chunksize=4))
    ok = [wr for r, wr in res if wr]
    by = {}
    for c, (r, wr) in zip([c for c in ids for _ in range(k)], res):
        by.setdefault(c, []).append(wr)
    best = [min(x for x in v if x) for v in by.values() if any(v)]
    return {"pass": len(ok) / len(res), "w": sum(ok) / len(ok) if ok else None,
            "best_w": sum(best) / len(best) if best else None,
            "reward": sum(r for r, _ in res) / len(res)}


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
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    allc = corpus_cells(a.plans)
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
            v = evaluate(pol, pool, val, 4, 0.7)
            rec = {"step": step, "eval": "val", **v}
            if v["w"] is not None and (best is None or (v["reward"] > best)):
                best = v["reward"]
                save(a.out / "best")
                rec["best"] = True
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        check(0)
        for step in range(1, a.steps + 1):
            t0 = time.time()
            ids = rng.sample(train, a.cells)
            nls = [allc[c][0] for c in ids for _ in range(a.group)]
            ro = pol.rollout(nls, a.temperature)
            t1 = time.time()
            jobs = [(allc[c][0], cols, allc[c][1]) for c, (_, cols, _) in zip([c for c in ids for _ in range(a.group)], ro)]
            res = list(pool.map(_route_score, jobs, chunksize=2))
            t2 = time.time()
            eps, flat = [], 0
            for g in range(a.cells):
                rs = torch.tensor([res[g * a.group + j][0] for j in range(a.group)])
                if float(rs.std()) < 1e-6:
                    flat += 1
                    continue
                adv = (rs - rs.mean()) / (rs.std() + 1e-4)
                for j in range(a.group):
                    steps = ro[g * a.group + j][2]
                    if steps:
                        eps.append((float(adv[j]), steps))
            loss, kl = pol.update(eps) if eps else (0.0, 0.0)
            ok = [wr for _, wr in res if wr]
            rec = {"step": step, "reward": sum(r for r, _ in res) / len(res), "pass": len(ok) / len(res),
                   "w": sum(ok) / len(ok) if ok else None, "flat_groups": flat, "loss": loss, "kl": kl,
                   "t_roll": round(t1 - t0, 1), "t_score": round(t2 - t1, 1), "t_upd": round(time.time() - t2, 1)}
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % a.eval_every == 0 or step == a.steps:
                check(step)
    save(a.out / "last")
    return 0


if __name__ == "__main__":
    sys.exit(main())
