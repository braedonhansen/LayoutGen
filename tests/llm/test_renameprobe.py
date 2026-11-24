from cellgym.llm import renameprobe as R

NL = {"inputs": ["A", "B", "C"], "nets": ["A", "B", "C", "OUT", "VDD", "VSS", "x1", "x2", "n1", "p1"]}


def test_name_maps():
    assert R.name_map(NL, "letter") == {"x1": "D", "x2": "E"}
    assert R.name_map(NL, "net") == {"x1": "n31", "x2": "n32"}
    assert R.name_map({"inputs": ["A"], "nets": ["A", "OUT"]}, "letter") == {}


def test_rename_text_is_whole_word():
    m = {"x1": "D", "x2": "E"}
    t = 'cell cg_s3o1_x1_ax1\ncol 3 g n=x1 p=x1\nroute x2 met1 c3 gate:7\n"gate": "x1", "x10"'
    assert R.rename_text(t, m) == 'cell cg_s3o1_x1_ax1\ncol 3 g n=D p=D\nroute E met1 c3 gate:7\n"gate": "D", "x10"'
