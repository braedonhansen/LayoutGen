import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PDK_ROOT = Path(os.environ.get("PDK_ROOT", str(Path.home() / "pdk")))
HAVE_EDA = all(shutil.which(t) for t in ("klayout", "magic", "netgen")) and (PDK_ROOT / "sky130A").exists()


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run all-cell sweeps marked slow")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: all-cell sweeps; run with --runslow or CELLGYM_SLOW=1")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") or os.environ.get("CELLGYM_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow all-cell sweep (use --runslow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def core_cells_dir(tmp_path_factory) -> Path:
    """out/cells, generated into a temp dir if missing."""
    d = ROOT / "out" / "cells"
    if not any(d.glob("cg_*.json")):
        d = tmp_path_factory.mktemp("cells")
        subprocess.run([sys.executable, "-m", "cellgym.netlists.generate", "--out", str(d)],
                       cwd=ROOT, check=True, capture_output=True)
    return d


@pytest.fixture(scope="session")
def core_cells(core_cells_dir) -> dict:
    out = {}
    for f in sorted(core_cells_dir.glob("*.json")):
        nl = json.loads(f.read_text())
        if "devices" in nl:
            out[nl["id"]] = nl
    return out
