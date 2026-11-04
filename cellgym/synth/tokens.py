"""Netlist -> prompt tokens, plan <-> target tokens, and a fixed vocabulary.

    python -m cellgym.synth.tokens out/synth/plans.jsonl     # round-trip + length stats

Prompt (what the model reads):

    <bos> in A B C \\n
    n C OUT n1 w1.26 \\n        one line per device: type gate drain source width
    p A VDD p1 w1.89 \\n        (netlist order; net names are the netlist's own)
    <sep>

Target: the plan text without its `cell`, `in` and `out` lines (all derivable
from the netlist), then <eos>. Plan words split into: `n=`/`p=` + net (+ `*`),
`gate:`/`nmos:`/`pmos:` + column number, everything else one token per word.
Newlines are tokens. The vocabulary is fixed (below), so an unknown word is an
error, not a silent <unk>.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

from cellgym.baselines.common import outputs_of

SPECIAL = ["<pad>", "<bos>", "<sep>", "<eos>", "\n"]
INPUTS = list("ABCDEF")
MAX_NUM = 127                                   # column indexes
NETS = ["OUT", "VDD", "VSS", "-"] + [f"{k}{i}" for k in "np" for i in range(1, 17)]
TRACKS = ["c0", "c1", "c2"] + [f"{k}{i}" for k in "np" for i in range(0, 17)]
WIDTHS = ["w0.42", "w0.63", "w0.84", "w1.26", "w1.89"]
WORDS = ["in", "n", "p", "col", "g", "sd", "n=", "p=", "*", "gate:", "nmos:", "pmos:",
         "route", "pin", "li1", "met1"]
VOCAB: list[str] = list(dict.fromkeys(
    SPECIAL + WORDS + INPUTS + NETS + TRACKS + WIDTHS + [str(i) for i in range(MAX_NUM + 1)]))
TOKEN_ID = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, SEP, EOS, NL = (TOKEN_ID[t] for t in SPECIAL)


def ids(tokens: list[str]) -> list[int]:
    try:
        return [TOKEN_ID[t] for t in tokens]
    except KeyError as e:
        raise KeyError(f"token {e.args[0]!r} is not in the vocabulary") from None


# -- encode ----------------------------------------------------------------------------

def encode_netlist(netlist: dict) -> list[str]:
    toks = ["<bos>", "in", *netlist["inputs"], "\n"]
    for d in netlist["devices"]:
        toks += [d["type"][0], d["gate"], d["drain"], d["source"], f"w{d['w']:g}", "\n"]
    return toks + ["<sep>"]


def _word(w: str) -> list[str]:
    m = re.fullmatch(r"([np])=(.+?)(\*?)", w)
    if m:
        return [m[1] + "=", m[2]] + (["*"] if m[3] else [])
    m = re.fullmatch(r"(gate|nmos|pmos):(\d+)", w)
    if m:
        return [m[1] + ":", m[2]]
    return [w]


def encode_plan(plan: str) -> list[str]:
    toks: list[str] = []
    for line in plan.splitlines():
        if line.split()[0] in ("cell", "in", "out"):
            continue
        for w in line.split():
            toks += _word(w)
        toks.append("\n")
    return toks + ["<eos>"]


# -- decode ----------------------------------------------------------------------------

def decode_plan(tokens: list[str], netlist: dict) -> str:
    """Plan text for target tokens (up to <eos>), with the header rebuilt from the netlist."""
    lines = ["cell " + netlist["id"], "in " + " ".join(netlist["inputs"]),
             "out " + " ".join(outputs_of(netlist))]
    words: list[str] = []
    glue = False
    for t in tokens:
        if t == "<eos>":
            break
        if t == "\n":
            if words:
                lines.append(" ".join(words))
            words, glue = [], False
        elif t == "*" and words:
            words[-1] += "*"
        elif glue:
            words[-1] += t
            glue = False
        else:
            words.append(t)
            glue = t.endswith(("=", ":"))
    if words:
        lines.append(" ".join(words))
    return "\n".join(lines) + "\n"


# -- CLI -------------------------------------------------------------------------------

def _pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main(argv=None) -> int:
    path = Path((argv or sys.argv[1:] or ["out/synth/plans.jsonl"])[0])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    plen, tlen, bad = [], [], 0
    for r in rows:
        p, t = encode_netlist(r["netlist"]), encode_plan(r["plan"])
        ids(p), ids(t)
        if decode_plan(t, r["netlist"]) != r["plan"]:
            bad += 1
        plen.append(len(p))
        tlen.append(len(t))
    tot = [a + b for a, b in zip(plen, tlen)]
    print(f"{len(rows)} records, vocab {len(VOCAB)}, round-trip failures {bad}")
    for name, xs in (("prompt", plen), ("target", tlen), ("total", tot)):
        print(f"  {name:6} mean {statistics.mean(xs):6.1f}  p50 {_pct(xs, .5):4}  "
              f"p99 {_pct(xs, .99):4}  max {max(xs):4}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
