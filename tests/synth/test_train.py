import json

import pytest
import torch

from cellgym.baselines.euler import euler_plan
from cellgym.synth import model as M, tokens as T, train as TR


@pytest.fixture(scope="module")
def data(core_cells, tmp_path_factory):
    path = tmp_path_factory.mktemp("trainset") / "plans.jsonl"
    rows = []
    for cid in sorted(core_cells)[::7][:40]:
        nl = core_cells[cid]
        rows.append({"id": cid, "topology_id": cid, "plan": euler_plan(nl), "netlist": nl})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, rows


def tiny_args(data_path, out, steps=60):
    return TR.parse(["--data", str(data_path), "--out", str(out), "--d-model", "64", "--layers", "2",
                     "--heads", "4", "--steps", str(steps), "--batch", "8", "--accum", "1",
                     "--lr", "3e-3", "--warmup", "5", "--eval-every", "20", "--device", "cpu",
                     "--no-amp"])


@pytest.fixture(scope="module")
def run(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    return out, TR.train(tiny_args(data[0], out))


def test_loss_decreases_and_files_written(run):
    out, res = run
    assert res["last_train_loss"] < res["first_train_loss"] - 1.0
    assert res["best_val_loss"] < 5.0
    for f in ("best.pt", "last.pt", "log.jsonl", "split.json"):
        assert (out / f).exists()
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [20, 40, 60]


def test_split_by_topology_has_no_overlap(run, data):
    out, _ = run
    s = json.loads((out / "split.json").read_text())
    assert s["val"] and s["train"] and not set(s["val"]) & set(s["train"])
    assert set(s["val"]) | set(s["train"]) == {r["topology_id"] for r in data[1]}
    assert TR.split_topologies(["a", "b", "c"] * 5) == TR.split_topologies(["c", "b", "a"] * 5)


def test_loss_covers_target_tokens_only(data):
    seq, n_prompt = TR.encode_record(data[1][0])
    x, y = TR.make_example(seq, n_prompt)
    kept = (y != TR.IGNORE).nonzero().flatten().tolist()
    assert kept[0] == n_prompt - 1 and len(kept) == len(seq) - n_prompt
    assert y[kept[-1]].item() == T.EOS


def test_checkpoint_round_trip_and_sampling(run, data):
    out, _ = run
    model = M.load(out / "best.pt")
    again = M.load(out / "best.pt")
    x = torch.randint(0, len(T.VOCAB), (2, 30))
    assert torch.equal(model(x), again(x))
    prompt = T.ids(T.encode_netlist(data[1][0]["netlist"]))
    outs = M.sample(model, prompt, n=3, temperature=1.0, max_new=80)
    assert len(outs) == 3
    for o in outs:
        assert 0 < len(o) <= 80 and all(0 <= t < len(T.VOCAB) for t in o)
        assert T.EOS not in o[:-1] and (o[-1] == T.EOS or len(o) == 80)


def test_kv_cache_matches_full_forward(run, data):
    out, _ = run
    model = M.load(out / "best.pt")
    prompt = T.ids(T.encode_netlist(data[1][0]["netlist"]))
    cached = M.sample(model, prompt, n=1, temperature=0, max_new=25)[0]
    seq = list(prompt)
    naive = []
    for _ in range(len(cached)):
        with torch.no_grad():
            nxt = model(torch.tensor([seq]))[0, -1].argmax().item()
        seq.append(nxt)
        naive.append(nxt)
        if nxt == T.EOS:
            break
    assert cached == naive
