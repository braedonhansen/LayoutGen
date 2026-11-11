"""Full fine-tune of a HF causal LM on prompt/target jsonl (loss on target tokens + EOS only).

    python -m cellgym.llm.train --model Qwen/Qwen2.5-0.5B --data out/llm/data/step_train.jsonl \
        --val out/llm/data/step_val.jsonl --out out/llm/step05 [--epochs 1] [--lr 5e-5] [--tokens-per-step 65536]

fp32 master weights, bf16 autocast, AdamW, linear warmup + cosine to 10%. Micro-batches are
length-bucketed and capped at --micro-tokens padded tokens; the LM head runs only on
target positions (the 151k-vocab logits are the memory hog). Saves bf16 for vLLM.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch


def load(path, tok, max_len):
    ex = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(r["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(p) + len(t) <= max_len:
            ex.append((p, t))
    return ex


def batches(ex, micro_tokens, rng, shuffle=True):
    idx = list(range(len(ex)))
    if shuffle:
        rng.shuffle(idx)
    out = []
    for c in range(0, len(idx), 256):          # sort within chunks: little padding, still mixed
        chunk = sorted(idx[c:c + 256], key=lambda i: len(ex[i][0]) + len(ex[i][1]))
        cur, mx = [], 0
        for i in chunk:
            L = len(ex[i][0]) + len(ex[i][1])
            if cur and max(mx, L) * (len(cur) + 1) > micro_tokens:
                out.append(cur)
                cur, mx = [], 0
            cur.append(i)
            mx = max(mx, L)
        if cur:
            out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


def collate(ex, ids, pad, dev):
    L = max(len(ex[i][0]) + len(ex[i][1]) for i in ids)
    x = torch.full((len(ids), L), pad, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    tgt = torch.full((len(ids), L), -100, dtype=torch.long)
    for b, i in enumerate(ids):
        p, t = ex[i]
        s = p + t
        x[b, :len(s)] = torch.tensor(s)
        att[b, :len(s)] = 1
        tgt[b, len(p) - 1:len(s) - 1] = torch.tensor(t)      # position j predicts token j+1
    return x.to(dev), att.to(dev), tgt.to(dev)


def loss_fn(model, x, att, tgt):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(input_ids=x, attention_mask=att).last_hidden_state
        m = tgt != -100
        logits = model.lm_head(h[m]).float()
    return torch.nn.functional.cross_entropy(logits, tgt[m], reduction="sum"), int(m.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--tokens-per-step", type=int, default=65536, help="padded tokens per optimizer step")
    ap.add_argument("--micro-tokens", type=int, default=16384)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--scratch", default=None, help="LAYERSxDIM: random init with --model's config/tokenizer, resized (control)")
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if a.scratch:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(a.model)
        L, d = (int(x) for x in a.scratch.split("x"))
        cfg.update(dict(num_hidden_layers=L, hidden_size=d, intermediate_size=3 * d, num_attention_heads=d // 64,
                        num_key_value_heads=max(1, d // 256), max_window_layers=L, layer_types=None))
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev)
        print(f"scratch model {L}x{d}: {sum(p.numel() for p in model.parameters()):,} params", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()
    tr = load(a.data, tok, a.max_len)
    va = load(a.val, tok, a.max_len)[:2000] if a.val else []
    accum = max(1, a.tokens_per_step // a.micro_tokens)
    n_micro = len(batches(tr, a.micro_tokens, random.Random(0)))
    total = max(1, int(n_micro * a.epochs) // accum)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    tgt_tokens = sum(len(t) for _, t in tr)
    print(f"{len(tr)} train ex ({tgt_tokens} target tokens), {len(va)} val, {total} steps x {accum} micro", flush=True)

    def val_loss():
        if not va:
            return None
        model.eval()
        s = n = 0
        with torch.no_grad():
            for ids in batches(va, a.micro_tokens, rng, shuffle=False):
                l, k = loss_fn(model, *collate(va, ids, pad, dev))
                s += float(l)
                n += k
        model.train()
        return s / n

    t0, step, run, rn = time.time(), 0, 0.0, 0
    stream = []
    ep = 0
    while step < total:
        if not stream:
            stream = batches(tr, a.micro_tokens, rng)
            ep += 1
        group, stream = stream[:accum], stream[accum:]
        ntok = sum(len(tr[i][1]) for ids in group for i in ids)      # step's target tokens
        loss = 0.0
        for ids in group:
            l, _ = loss_fn(model, *collate(tr, ids, pad, dev))
            (l / ntok).backward()
            loss += float(l) / ntok
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        run += float(loss)
        rn += 1
        if step % 20 == 0 or step == total:
            rec = {"step": step, "loss": run / rn, "lr": sched.get_last_lr()[0], "epoch": ep,
                   "elapsed": round(time.time() - t0, 1)}
            if step % 100 == 0 or step == total:
                rec["val_loss"] = val_loss()
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run, rn = 0.0, 0
    model = model.to(torch.bfloat16)
    model.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    (out / "done.json").write_text(json.dumps({"steps": total, "seconds": round(time.time() - t0, 1),
                                               "args": vars(a)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
