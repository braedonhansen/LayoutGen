import json
from pathlib import Path

import pytest

from cellgym.baselines.euler import euler_plan
from cellgym.bench import reference, splits
from cellgym.bench.common import (EULER_WIDTHS_FILE, SPLITS_FILE, dump_rows, hash_ok, sha256, truth_table)
from cellgym.bench.evaluate import diffusion_breaks, evaluate
from cellgym.bench.score import markdown, reward, score
from cellgym.dsl.plan import parse

EXAMPLES = Path(__file__).resolve().parents[2] / "cellgym" / "dsl" / "examples"


# -- splits ---------------------------------------------------------------------------

def test_split_file_reproducible(core_cells):
    doc = splits.make_splits(core_cells)
    assert dump_rows(doc, "cells") == SPLITS_FILE.read_text()
    assert hash_ok(splits.load())


def test_split_hashes_match_netlists(core_cells):
    doc = splits.load()
    assert {r["id"] for r in doc["cells"]} == set(core_cells)
    for r in doc["cells"]:
        assert r["netlist_sha256"] == sha256(core_cells[r["id"]]), r["id"]


def test_split_contents(core_cells):
    doc = splits.load()
    s = splits.split_of(doc)
    anchors = {cid for a in doc["anchors"].values() for cid in a["all_ids"]}
    assert doc["counts"] == {"train": 123 - len(anchors), "val": 15, "test_id": 25,
                             "test_ood": 160 + len(anchors)}
    for cid, nl in core_cells.items():
        if len(nl["inputs"]) == 6 or cid in anchors:
            assert s[cid] == "test_ood"
        else:
            assert s[cid] != "test_ood"
    assert set(doc["anchors"]) == set(splits.ANCHORS)
    trees = {k: v["canonical_tree"] for k, v in doc["anchors"].items()}
    assert trees["NAND3"] == "S(A,B,C)" and trees["NOR3"] == "P(A,B,C)"
    assert trees["AOI21"] == "P(C,S(A,B))" and trees["AOI22"] == "P(S(A,B),S(C,D))"
    # the named NOT MAJ3 cell really is the inverted majority
    maj = core_cells[doc["anchors"]["NOT_MAJ3"]["id"]]
    assert int(maj["function"], 16) == truth_table(lambda x: sum(x) < 2, 3)


def test_split_tamper_detected(tmp_path):
    doc = json.loads(SPLITS_FILE.read_text())
    doc["cells"][0]["split"] = "test_id" if doc["cells"][0]["split"] != "test_id" else "train"
    f = tmp_path / "s.json"
    f.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        splits.load(f)


# -- euler reference ------------------------------------------------------------------

def test_euler_reference_matches(core_cells):
    doc = reference.load()
    assert set(doc["cells"]) == set(core_cells)
    for cid in sorted(core_cells)[:12] + sorted(core_cells)[-3:]:
        assert doc["cells"][cid] == reference.euler_width(core_cells[cid]), cid


# -- scorer ---------------------------------------------------------------------------

def test_breaks_examples():
    assert diffusion_breaks(parse((EXAMPLES / "aoi21.plan").read_text())) == 0
    assert diffusion_breaks(parse((EXAMPLES / "half_adder.plan").read_text())) == 1


def test_reward():
    assert reward(1000, 1000, True) == 1.0
    assert reward(800, 1000, True) == pytest.approx(1.2)
    assert reward(1500, 1000, True) == pytest.approx(0.5)
    assert reward(800, 1000, False) == 0.0


def test_evaluate_stages(core_cells):
    nl = core_cells["cg_2i2d_94893424"]
    text = euler_plan(nl)
    assert evaluate(text, nl).ok
    assert evaluate(None, nl).stage == "missing"
    assert evaluate("cell x\nbogus line\n", nl).stage == "parse"
    assert evaluate(text.replace("*", "", 1), nl).stage == "validate"


def test_score_val_split(tmp_path, core_cells_dir, core_cells):
    s = splits.split_of(splits.load())
    val = sorted(c for c in s if s[c] == "val")
    for cid in val[1:]:
        (tmp_path / f"{cid}.plan").write_text(euler_plan(core_cells[cid]))
    res = score(tmp_path, "t", "val", cells_dir=core_cells_dir)
    m = res["fast"]["val"]
    assert m["n"] == 15 and m["passed"] == 14
    assert m["w_over_euler_mean"] == 1.0
    assert m["reward_mean"] == pytest.approx(14 / 15)
    assert res["failures"] == {val[0]: "missing: no plan file"}
    md = markdown(res)
    assert "| val | 15 | 0.933 |" in md
    from cellgym.bench.results import document
    assert "| val | t | 15 | 0.933 | not run | " in document([res])
