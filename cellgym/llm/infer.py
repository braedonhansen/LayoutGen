"""Sample plans from a fine-tuned LM (HF generate), for every evaluation set (scored by cellgym.llm.score).

    python -m cellgym.llm.infer --model out/llm/step05/model --mode step --out out/llm/step05/samples.jsonl [--k 8]

mode oneshot: k samples of the whole plan body per cell.
mode step:    k independent placements per cell; each step the model sees Placer.state() and its
              continuation is restricted to Placer.legal() (generate with a token trie of the
              legal columns, so sampling is renormalized over legal moves only). The environment stops a placement when every device is placed and the last
              column is sd, or after 4 x devices + 4 columns.
Output: one json line per (set, cell, sample): oneshot -> "text" (plan body); step -> "cols".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cellgym.llm import fmt as F

SETS = [("val", "bench:val"), ("test_id", "bench:test_id"), ("ood6_dev", "out/ood6:dev"),
        ("ood7_dev", "out/ood7:dev"), ("ood8_dev", "out/ood8:dev"), ("ood9_all", "out/ood9:all"),
        ("ood10_all", "out/ood10:all"), ("oodstack5_dev", "out/oodstack5:dev"),
        ("oodstack6_all", "out/oodstack6:all"), ("ooddev10_dev", "out/ooddev10:dev"),
        ("ooddev11_dev", "out/ooddev11:dev"), ("ooddev12_all", "out/ooddev12:all"), ("adder", "out/adder:dev")]


def load_sets(names=None, extra=()):
    from cellgym.bench.common import EULER_WIDTHS_FILE, load_cells
    from cellgym.bench.splits import load as load_splits, split_of
    from cellgym.synth.devset import load as load_set
    out = {}
    bench = None
    for name, spec in list(SETS) + list(extra):
        if names and name not in names:
            continue
        src, part = spec.split(":")
        if src == "bench":
            if bench is None:
                cells = load_cells(Path("out/cells"))
                sp = split_of(load_splits())
                w = json.loads(Path(EULER_WIDTHS_FILE).read_text())
                w = w.get("cells", w)
                bench = (cells, sp, w)
            cells, sp, w = bench
            ids = sorted(c for c in cells if sp.get(c) == part)
            out[name] = ({c: cells[c] for c in ids}, {c: _w_nm(w.get(c)) for c in ids})
        else:
            out[name] = load_set(Path(src), part)
    return out


def _w_nm(v):
    if v is None:
        return None
    return v["width_nm"] if isinstance(v, dict) else v


class LM:
    """HF model wrapper: batched sampling, optionally restricted to a set of continuations."""

    def __init__(self, path, batch=384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
        self.batch = batch
        self._enc = {}

    def enc(self, s):
        r = self._enc.get(s)
        if r is None:
            r = self._enc[s] = self.tok(s, add_special_tokens=False)["input_ids"]
        return r

    def _gen(self, prompts, temperature, max_new, allowed=None, n=1):
        torch = self.torch
        ids = [self.enc(p) for p in prompts]
        L = max(map(len, ids))
        x = torch.full((len(ids), L), self.pad, dtype=torch.long)
        att = torch.zeros_like(x)
        for i, s in enumerate(ids):
            x[i, L - len(s):] = torch.tensor(s)
            att[i, L - len(s):] = 1
        kw = dict(max_new_tokens=max_new, pad_token_id=self.pad, eos_token_id=self.tok.eos_token_id,
                  num_return_sequences=n)
        kw.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if temperature > 0 else kw.update(do_sample=False)
        if allowed is not None:
            eos = self.tok.eos_token_id

            def fn(b, sent):
                trie = allowed[b // n]
                node = trie
                for t in sent[L:].tolist():
                    if t == eos or t == self.pad:
                        return [eos]
                    node = node.get(t)
                    if node is None:
                        return [eos]
                return list(node.keys()) or [eos]
            kw["prefix_allowed_tokens_fn"] = fn
        with torch.no_grad():
            out = self.model.generate(input_ids=x.cuda(), attention_mask=att.cuda(), **kw)
        return [self.tok.decode([t for t in o[L:].tolist() if t not in (self.pad, self.tok.eos_token_id)])
                for o in out]

    def generate(self, prompts, temperature, max_new, n=1, allowed=None):
        res = []
        bs = max(1, self.batch // n)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        for c in range(0, len(order), bs):
            idx = order[c:c + bs]
            g = self._gen([prompts[i] for i in idx], temperature, max_new,
                          None if allowed is None else [allowed[i] for i in idx], n)
            for j, i in enumerate(idx):
                out[i] = g[j * n:(j + 1) * n]
        return out

    def trie(self, options):
        root = {}
        eos = self.tok.eos_token_id
        for o in options:
            node = root
            for t in self.enc(o) + [eos]:
                node = node.setdefault(t, {})
        return root


def run_step(lm, sets, k, temperature, out):
    jobs = []      # [set, cid, sample, placer, cap, status]
    for s, (cells, _) in sets.items():
        for cid, nl in cells.items():
            for j in range(k):
                jobs.append([s, cid, j, F.Placer(nl), 4 * len(nl["devices"]) + 4, "run"])
    rounds = 0
    while True:
        act = [jb for jb in jobs if jb[5] == "run"]
        if not act:
            break
        rounds += 1
        live, prompts, tries = [], [], []
        for jb in act:
            legal = jb[3].legal()
            if not legal:
                jb[5] = "stuck"
                continue
            live.append(jb)
            prompts.append(jb[3].state())
            tries.append(None if lm is None else lm.trie([" " + F.col_text(c) for c in legal]))
        if lm is None:          # baseline: uniform over legal moves
            res = [[" " + F.col_text(__import__("random").Random(hash((jb[1], jb[2], len(jb[3].cols)))).choice(jb[3].legal()))]
                   for jb in live]
        else:
            res = lm.generate(prompts, temperature, 16, 1, tries) if prompts else []
        for jb, r in zip(live, res):
            c = F.parse_col(r[0])
            if c is None or not jb[3].step(c):
                jb[5] = "illegal"
            elif jb[3].done():
                jb[5] = "done"
            elif len(jb[3].cols) >= jb[4]:
                jb[5] = "cap"
        print(f"round {rounds}: {sum(jb[5] == 'run' for jb in jobs)} running", flush=True)
    with open(out, "w") as f:
        for s, cid, j, p, _, st in jobs:
            f.write(json.dumps({"set": s, "cell": cid, "sample": j, "status": st, "cols": p.cols}) + "\n")


def run_oneshot(lm, sets, k, temperature, max_tokens, out):
    keys, prompts = [], []
    for s, (cells, _) in sets.items():
        for cid, nl in cells.items():
            keys.append((s, cid))
            prompts.append(F.oneshot_prompt(nl))
    res = lm.generate(prompts, temperature, max_tokens, n=k)
    with open(out, "w") as f:
        for (s, cid), r in zip(keys, res):
            for j, o in enumerate(r):
                f.write(json.dumps({"set": s, "cell": cid, "sample": j, "status": "gen", "text": o}) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["oneshot", "step", "random"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=384, help="sequences per generate call")
    ap.add_argument("--sets", nargs="*", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    sets = load_sets(a.sets)
    lm = None if a.mode == "random" else LM(a.model, a.batch)
    if a.mode == "oneshot":
        run_oneshot(lm, sets, a.k, a.temperature, a.max_tokens, a.out)
    else:
        run_step(lm, sets, a.k, a.temperature, a.out)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
