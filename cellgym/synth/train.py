"""Train the small plan model from scratch (imitation of the Euler plans).

    python -m cellgym.synth.train --data out/synth/plans.jsonl --out out/synth/run0 \\
        [--d-model 512 --layers 10 --heads 8 --steps 2000 --batch 16 --accum 4 --lr 6e-4]

Sequence = prompt tokens (encode_netlist, ends with <sep>) + target tokens
(encode_plan, ends with <eos>); the loss covers target tokens only. Validation
is split BY topology_id (about 10% of topologies, chosen by sha256 of the id),
so renamings and alternative plans of one topology never straddle the split.

Writes <out>/split.json, log.jsonl (train/val loss every --eval-every steps),
best.pt (lowest val loss) and last.pt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cellgym.synth import tokens as T
from cellgym.synth.model import GPT, Config, save

IGNORE = -100


# -- data ----------------------------------------------------------------------------------

def load_records(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def split_topologies(topology_ids, frac: float = 0.1) -> tuple[list[str], list[str]]:
    """(train, val) topology ids; val = sha256 bucket, at least one in each side."""
    tops = sorted(set(topology_ids))
    key = {t: int(hashlib.sha256(f"cellgym-train-split:{t}".encode()).hexdigest(), 16) for t in tops}
    m = max(1, round(1 / frac))
    val = [t for t in tops if key[t] % m == 0]
    if not val:
        val = [min(tops, key=key.get)]
    if len(val) == len(tops):
        val = val[:-1]
    vs = set(val)
    return [t for t in tops if t not in vs], val


def encode_record(r: dict) -> tuple[list[int], int]:
    """(full id sequence, prompt length)."""
    p = T.ids(T.encode_netlist(r["netlist"]))
    return p + T.ids(T.encode_plan(r["plan"])), len(p)


def make_example(seq: list[int], n_prompt: int):
    x = torch.tensor(seq[:-1])
    y = torch.tensor(seq[1:])
    y[: n_prompt - 1] = IGNORE            # labels at positions whose next token is a prompt token
    return x, y


def collate(examples):
    L = max(len(x) for x, _ in examples)
    X = torch.full((len(examples), L), T.PAD)
    Y = torch.full((len(examples), L), IGNORE)
    for i, (x, y) in enumerate(examples):
        X[i, : len(x)], Y[i, : len(y)] = x, y
    return X, Y


def batches(examples, batch: int, rng: random.Random):
    """Endless batches, bucketed by length within shuffled chunks."""
    while True:
        order = list(range(len(examples)))
        rng.shuffle(order)
        chunk = batch * 32
        out = []
        for i in range(0, len(order), chunk):
            part = sorted(order[i:i + chunk], key=lambda j: len(examples[j][0]))
            out += [part[k:k + batch] for k in range(0, len(part), batch)]
        rng.shuffle(out)
        for b in out:
            yield collate([examples[j] for j in b])


def loss_fn(model, X, Y):
    logits = model(X)
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), Y.view(-1), ignore_index=IGNORE)


@torch.no_grad()
def evaluate(model, examples, batch: int, device, amp_dtype) -> float:
    """Mean loss per target token over all examples."""
    model.eval()
    tot, cnt = 0.0, 0
    order = sorted(range(len(examples)), key=lambda j: len(examples[j][0]))
    for i in range(0, len(order), batch):
        X, Y = collate([examples[j] for j in order[i:i + batch]])
        X, Y = X.to(device), Y.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(X)
        n = (Y != IGNORE).sum().item()
        tot += F.cross_entropy(logits.float().view(-1, logits.shape[-1]), Y.view(-1),
                               ignore_index=IGNORE, reduction="sum").item()
        cnt += n
    model.train()
    return tot / max(cnt, 1)


# -- training ------------------------------------------------------------------------------

def train(a) -> dict:
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_dtype = None
    scaler = None
    if a.amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler() if amp_dtype is torch.float16 else None

    records = load_records(a.data)
    tr_tops, va_tops = split_topologies([r["topology_id"] for r in records], a.val_frac)
    vs = set(va_tops)
    (out / "split.json").write_text(json.dumps({"train": tr_tops, "val": va_tops}, indent=1) + "\n")
    train_ex = [make_example(*encode_record(r)) for r in records if r["topology_id"] not in vs]
    val_ex = [make_example(*encode_record(r)) for r in records if r["topology_id"] in vs]

    cfg = Config(d_model=a.d_model, layers=a.layers, heads=a.heads, max_len=a.max_len, dropout=a.dropout)
    model = GPT(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": nodecay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    print(f"{model.n_params() / 1e6:.1f}M params | train {len(train_ex)} seqs / {len(tr_tops)} topologies, "
          f"val {len(val_ex)} seqs / {len(va_tops)} topologies | device {device} amp {amp_dtype}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    it = batches(train_ex, a.batch, random.Random(a.seed))
    log = (out / "log.jsonl").open("w")
    best, first_train, last_train, window, ntok = math.inf, None, None, [], 0
    result = {"params": model.n_params()}
    model.train()

    def checkpoint_and_log(step):
        nonlocal best, last_train, ntok
        v = evaluate(model, val_ex, a.batch, device, amp_dtype) if val_ex else float("nan")
        tr = sum(window) / max(len(window), 1)
        last_train = tr
        row = {"step": step, "train_loss": round(tr, 4), "val_loss": round(v, 4),
               "lr": lr_at(step), "elapsed": round(time.time() - t0, 1), "tokens": ntok}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)
        window.clear()
        save(model, cfg, out / "last.pt")
        if v < best:
            best = v
            save(model, cfg, out / "best.pt")

    t_train = time.time()
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)
        for _ in range(a.accum):
            X, Y = next(it)
            X, Y = X.to(device), Y.to(device)
            ntok += int((Y != IGNORE).sum())
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = loss_fn(model, X, Y) / a.accum
            (scaler.scale(loss) if scaler else loss).backward()
            window.append(loss.item() * a.accum)
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        if first_train is None:
            first_train = window[-1]
        if step % a.eval_every == 0 or step == a.steps:
            checkpoint_and_log(step)
    log.close()
    dt = time.time() - t_train
    result |= {"steps": a.steps, "first_train_loss": first_train, "last_train_loss": last_train,
               "best_val_loss": best, "steps_per_s": a.steps / dt, "seconds": dt,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None}
    print(result, flush=True)
    return result


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="out/synth/plans.jsonl")
    ap.add_argument("--out", default="out/synth/run0")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=2000, help="optimizer steps (batch x accum sequences each)")
    ap.add_argument("--batch", type=int, default=16, help="micro-batch sequences")
    ap.add_argument("--accum", type=int, default=4, help="gradient-accumulation micro-batches per step")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    train(parse(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
