from cellgym.llm.score import fewest_column_rows

A = [["sd", "a", "b"], ["g", "A"], ["sd", "c", "d"]]
B = [["sd", "c", "d"], ["g", "A"], ["sd", "a", "b"]]
W = A + [["sd", "e", "f"]]


def row(cell, i, cols, status="done", s="val"):
    return {"set": s, "cell": cell, "sample": i, "status": status, "cols": cols}


def test_keeps_distinct_fewest_column_placements_per_cell():
    rows = [row("x", 0, W), row("x", 1, A), row("x", 2, A), row("x", 3, B), row("x", 4, A[:1], "cap"),
            row("y", 0, W), row("y", 1, W)]
    out = fewest_column_rows(rows)
    assert [(r["cell"], r["sample"]) for r in out] == [("x", 1), ("x", 3), ("y", 0)]


def test_cell_without_a_finished_sample_stays_as_one_failure():
    out = fewest_column_rows([row("z", 0, A, "cap"), row("z", 1, [], "stuck")])
    assert len(out) == 1 and out[0]["status"] == "cap"
