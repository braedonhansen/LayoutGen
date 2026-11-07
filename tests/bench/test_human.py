import itertools

import pytest

from cellgym.bench import human
from conftest import PDK_ROOT

LIB = '''
library (x) {
  cell ("lib__nand2_1") {
    pg_pin ("VPWR") { pg_type : "primary_power"; }
    pin ("A") { direction : "input"; capacitance : 0.002; }
    pin ("B") { direction : "input"; }
    pin ("Y") { direction : "output"; function : "(!A) | (!B)";
      timing () { related_pin : "A"; } }
  }
  cell ("lib__a21oi_1") {
    pin ("A1") { direction : "input"; }
    pin ("A2") { direction : "input"; }
    pin ("B1") { direction : "input"; }
    pin ("Y") { direction : "output"; function : "!((A1&A2) | B1)"; }
  }
  cell ("lib__and2_1") {
    pin ("A") { direction : "input"; }
    pin ("B") { direction : "input"; }
    pin ("X") { direction : "output"; function : "(A&B)"; }
  }
  cell ("lib__dfxtp_1") {
    ff ("IQ","IQ_N") { next_state : "D"; clocked_on : "CLK"; }
    pin ("D") { direction : "input"; }
    pin ("CLK") { direction : "input"; }
    pin ("Q") { direction : "output"; function : "IQ"; }
  }
  cell ("lib__nand2_2") {
    pin ("A") { direction : "input"; }
    pin ("B") { direction : "input"; }
    pin ("Y") { direction : "output"; function : "(!A) | (!B)"; }
  }
}
'''
LEF = """MACRO lib__nand2_1
  SIZE 1.380 BY 2.720 ;
END lib__nand2_1
MACRO lib__a21oi_1
  SIZE 1.840 BY 2.720 ;
END lib__a21oi_1
MACRO lib__and2_1
  SIZE 1.840 BY 2.720 ;
END lib__and2_1
"""


@pytest.mark.parametrize("expr,f", [
    ("!A", lambda a, b: not a),
    ("A'", lambda a, b: not a),
    ("A B", lambda a, b: a and b),
    ("A*B + !A", lambda a, b: (a and b) or not a),
    ("A ^ B", lambda a, b: a != b),
    ("!(A | B)", lambda a, b: not (a or b)),
    ("A & B ^ A", lambda a, b: a and (b != a)),      # ^ binds tighter than &
    ("(A)' | 0", lambda a, b: not a),
])
def test_compile_function(expr, f):
    g = human.compile_function(expr)
    for a, b in itertools.product([False, True], repeat=2):
        assert bool(g({"A": a, "B": b})) == f(a, b), (expr, a, b)


def test_liberty_and_lef_parse():
    lib = human.liberty_cells(LIB)
    assert lib["lib__nand2_1"] == {"inputs": ["A", "B"], "outputs": {"Y": "(!A) | (!B)"}, "sequential": False}
    assert lib["lib__dfxtp_1"]["sequential"]
    assert human.lef_sizes(LEF)["lib__a21oi_1"] == (1.84, 2.72)


def test_match_library(core_cells):
    rows = human.match_library(human.liberty_cells(LIB), human.lef_sizes(LEF), core_cells)
    got = {r["core_id"]: (r["hd_cells"], r["hd_sites"]) for r in rows}
    # and2 is not single-stage inverting; nand2_2 is not a _1 cell; the ff is sequential
    assert got == {"cg_2i2d_94893424": (["lib__nand2_1"], 3), "cg_3i3d_29e68bd3": (["lib__a21oi_1"], 4)}


LIBFILE = PDK_ROOT / "sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"


@pytest.mark.skipif(not LIBFILE.exists(), reason="sky130_fd_sc_hd liberty not available")
def test_hd_library(core_cells):
    lef = PDK_ROOT / "sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef"
    rows = human.match_library(human.liberty_cells(LIBFILE.read_text()), human.lef_sizes(lef.read_text()),
                               core_cells)
    by = {c: r for r in rows for c in r["hd_cells"]}
    assert len(rows) >= 20
    assert by["inv_1"]["core_id"] == "cg_1i1d_6dcd4ce2" and by["inv_1"]["hd_sites"] == 3
    assert by["a21oi_1"]["core_id"] == "cg_3i3d_29e68bd3" and by["a21oi_1"]["hd_sites"] == 4
