import re

from cellgym.baselines.euler import euler_plan
from cellgym.check.drc import Violation
from cellgym.dsl.plan import parse
from cellgym.render.render import render
from cellgym.viewer.static import build, svg

IDS = ["cg_2i2d_94893424", "cg_3i3d_29e68bd3"]


def test_gallery(tmp_path, core_cells_dir, core_cells):
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / f"{IDS[0]}.plan").write_text(euler_plan(core_cells[IDS[0]]))
    (plans / f"{IDS[1]}.plan").write_text("cell broken\n")
    out = tmp_path / "gallery"
    evs = build(plans, out, core_cells_dir, None, "t", only=IDS)
    assert [e.ok for e in evs] == [True, False]
    page = (out / "index.html").read_text()
    assert IDS[0] in page and "fail: parse" in page and f'src="{IDS[0]}.svg"' in page
    assert not re.search(r'(src|href)="(https?:)?//', page)       # no external assets
    s = (out / f"{IDS[0]}.svg").read_text()
    assert s.startswith("<svg") and "sans-serif" in s
    for layer in ("nwell", "diff", "poly", "licon", "li", "mcon", "met1"):
        assert f">{layer}</text>" in s
    assert 'class="drc"' not in s
    assert not (out / f"{IDS[1]}.svg").exists()


def test_violation_boxes(core_cells):
    nl = core_cells[IDS[1]]
    r = render(parse(euler_plan(nl)), nl)
    s = svg(r, [Violation("poly.2", "poly", (100, 200, 300, 400))])
    assert s.count('class="drc"') == 1 and ">poly.2</text>" in s and 'stroke="#e00000"' in s
