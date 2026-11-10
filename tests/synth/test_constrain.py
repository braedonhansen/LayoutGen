import json
import random
from pathlib import Path

import pytest

from cellgym.bench.common import load_cells
from cellgym.dsl.plan import parse
from cellgym.synth import tokens as T
from cellgym.synth.constrain import PlanGrammar, sample_constrained
from cellgym.synth.eval import decode_ids

CELLS = Path("out/cells")
PLANS = Path("out/synth4/plans.jsonl")
EULER = Path("out/plans")


def euler_plan(nl):
    return (EULER / f"{nl['id']}.plan").read_text()

pytestmark = pytest.mark.skipif(not CELLS.exists(), reason="out/cells missing")


def _first_violation(g, plan):
    toks = T.ids(T.encode_plan(plan))
    s = g.initial()
    for i, t in enumerate(toks):
        if t not in g.allowed_ids(s):
            return i, T.VOCAB[t]
        s = g.step(s, t)
    return None if s.sub == "DONE" else (len(toks), "<incomplete>")


@pytest.fixture(scope="module")
def cells():
    return load_cells(CELLS)


def test_accepts_euler_plans_of_all_cells(cells):
    bad, n = [], 0
    for cid, nl in cells.items():
        if not (EULER / f"{cid}.plan").exists():
            continue
        n += 1
        plan = euler_plan(nl)
        v = _first_violation(PlanGrammar(nl), plan)
        if v:
            bad.append((cid, v))
    assert n >= 323 and not bad, (n, bad[:5])


@pytest.mark.skipif(not PLANS.exists(), reason="no synth4 plans")
def test_accepts_synth_plans():
    rows = [json.loads(l) for l in PLANS.read_text().splitlines()]
    random.Random(0).shuffle(rows)
    bad = [(r["id"], v) for r in rows[:500] if (v := _first_violation(PlanGrammar(r["netlist"]), r["plan"]))]
    assert not bad, bad[:5]


def _example(cells):
    for nl in cells.values():
        if len(nl["inputs"]) >= 2 and (EULER / f"{nl['id']}.plan").exists() and "route" in euler_plan(nl):
            return nl
    raise AssertionError


def _toks(plan):
    return T.encode_plan(plan)


def _rejects(g, toks):
    s = g.initial()
    for t in T.ids(toks):
        if t not in g.allowed_ids(s):
            return True
        s = g.step(s, t)
    return not (s and s.sub == "DONE")


def test_rejects_corruptions(cells):
    nl = _example(cells)
    g = PlanGrammar(nl)
    base = _toks(euler_plan(nl))
    assert not _rejects(g, base)
    # skipped column index
    i = base.index("col", base.index("col") + 1)
    bad = list(base)
    bad[i + 1] = str(int(bad[i + 1]) + 1)
    assert _rejects(g, bad)
    # dropped net after n=
    j = base.index("n=")
    assert _rejects(g, base[:j + 1] + base[j + 2:])
    # '*' directly after '-' / missing p= net
    j = base.index("p=")
    assert _rejects(g, base[:j + 1] + ["\n"] + base[j + 2:])
    # track where an endpoint belongs
    r = base.index("route")
    k = base.index("gate:", r) if "gate:" in base[r:base.index("pin")] else base.index("nmos:", r)
    bad = list(base)
    bad[k] = "c1"
    assert _rejects(g, bad)
    # endpoint column out of range
    bad = list(base)
    bad[k + 1] = "99"
    assert _rejects(g, bad)
    # missing input pin -> eos not allowed
    p = len(base) - 1 - base[::-1].index("pin")
    assert _rejects(g, base[:p] + ["<eos>"])
    # route before any column
    assert _rejects(g, ["route"] + base[base.index("route") + 1:])


def test_allowed_prefix_api(cells):
    nl = _example(cells)
    g = PlanGrammar(nl)
    assert g.allowed([]) == {T.TOKEN_ID["col"]}
    assert g.allowed(["col"]) == {T.TOKEN_ID["0"]}
    assert g.allowed(["col", "0", "sd", "n=", "-", "p="]) == {T.TOKEN_ID[x] for x in g.sd_nets + ["-"]}
    assert g.allowed(["row"]) == set() if "row" in T.TOKEN_ID else True


def test_constrained_random_model_parses(cells):
    torch = pytest.importorskip("torch")
    from cellgym.synth.model import GPT, Config
    torch.manual_seed(0)
    model = GPT(Config(d_model=64, layers=2, heads=4, max_len=512)).eval()
    nls = list(cells.values())
    rng = random.Random(0)
    ok = total = 0
    for _ in range(25):
        nl = rng.choice(nls)
        prompt = T.ids(T.encode_netlist(nl))
        outs = sample_constrained(model, prompt, nl, n=2, temperature=1.0, max_new=320)
        for o in outs:
            total += 1
            try:
                parse(decode_ids(nl, o))
                ok += 1
            except Exception:
                pass
    assert total == 50 and ok / total >= 0.99, (ok, total)
    # greedy is deterministic
    nl = nls[0]
    p = T.ids(T.encode_netlist(nl))
    a = sample_constrained(model, p, nl, temperature=0)
    assert a == sample_constrained(model, p, nl, temperature=0)


def test_cover_blocks_early_close(cells):
    """Cutting the col lines before the last gate column leaves a device unplaced: with
    cover the plan cannot leave the col phase there; without it, it can."""
    n = 0
    for cid, nl in list(cells.items())[:80]:
        if not (EULER / f"{cid}.plan").exists():
            continue
        lines = euler_plan(nl).splitlines()
        cols = [l for l in lines if l.startswith("col ")]
        last_g = max(i for i, l in enumerate(cols) if " g " in l)
        prefix = T.ids(T.encode_plan("\n".join(cols[:last_g]) + "\n"))[:-1]   # drop <eos>
        g_on, g_off = PlanGrammar(nl), PlanGrammar(nl, cover=False)
        s_on, s_off = g_on.walk(prefix), g_off.walk(prefix)
        assert s_on is not None and s_on.sub == "L"
        assert T.TOKEN_ID["pin"] not in g_on.allowed_ids(s_on)
        if g_off._can_leave(s_off.cols):
            n += 1
        full = T.ids(T.encode_plan("\n".join(cols) + "\n"))[:-1]
        assert T.TOKEN_ID["pin"] in g_on.allowed_ids(g_on.walk(full))
    assert n > 0
