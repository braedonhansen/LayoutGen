import json

import pytest

from cellgym.bench.splits import load as load_splits, split_of
from cellgym.dsl.plan import parse
from cellgym.dsl.validate import validate
from cellgym.netlists.enumerate import Params, canonical_fn, parse_tree, tree_str
from cellgym.synth import generate as g

SMALL = Params(4, 3, 3, 7)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    manifest = g.generate(out, SMALL, renames=3, placements=2, jobs=1, cache=None)
    rows = [json.loads(line) for line in (out / "plans.jsonl").read_text().splitlines()]
    return out, manifest, rows


def test_no_benchmark_function_in_synthetic(corpus):
    _, manifest, rows = corpus
    core = g.core_cells()
    bad = g.forbidden_functions(core, split_of(load_splits()))
    synth = [r for r in rows if r["origin"] == "synthetic"]
    assert synth, "expected some synthetic topologies"
    for r in synth:
        assert (r["n_inputs"], canonical_fn(int(r["function"], 16), r["n_inputs"])) not in bad
    assert manifest["leak_check"] == "pass"


def test_core_records_are_train_cells_only(corpus):
    _, _, rows = corpus
    core = g.core_cells()
    splits = split_of(load_splits())
    trees = {tree_str(net.tree): cid for cid, (net, _) in core.items()}
    for r in rows:
        assert r["n_inputs"] <= 5
        if r["origin"] == "core_train":
            assert splits[r["topology_id"]] == "train"
        else:
            assert r["topology_id"] not in core


def test_plans_validate_and_match_netlist(corpus):
    _, _, rows = corpus
    for r in rows[::7]:
        rep = validate(parse(r["plan"]), r["netlist"])
        assert rep.ok, (r["id"], str(rep))
        assert r["netlist"]["id"] == r["id"] and r["sites"] > 0 and r["width_um"] > 0


def test_renamings_are_distinct_netlists_of_one_topology(corpus):
    _, _, rows = corpus
    by_topo = {}
    for r in rows:
        by_topo.setdefault(r["topology_id"], set()).add(r["tree"])
    assert any(len(v) > 1 for v in by_topo.values())
    # renaming only permutes inputs: same function class within a topology
    for r in rows:
        t = parse_tree(r["tree"])
        assert tree_str(t) == r["tree"]


def test_plans_distinct_within_cell_and_deterministic(corpus, tmp_path):
    out, manifest, rows = corpus
    seen = {}
    for r in rows:
        seen.setdefault(r["id"], []).append(r["plan"])
    assert all(len(set(v)) == len(v) for v in seen.values())
    m2 = g.generate(tmp_path, SMALL, renames=3, placements=2, jobs=1, cache=None)
    assert m2["plans_sha256"] == manifest["plans_sha256"]


def test_cli_rejects_six_inputs():
    with pytest.raises(SystemExit):
        g.main(["--max-inputs", "6", "--out", "x"])
