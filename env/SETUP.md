# Environment setup (WSL2 Ubuntu 22.04)

Everything lives in the Linux filesystem (`$HOME`), not `/mnt/c`. No sudo needed.

Pinned versions:

| Piece | Version | Source |
|---|---|---|
| sky130A PDK | open_pdks `792ef67037d74c8c51811eab3ce23cf793491296` | `ciel` 3.2.0 |
| Magic | 8.3.465 | conda, litex-hub |
| Netgen | 1.5.272 | conda, litex-hub |
| KLayout | 0.30.12 (official Ubuntu-22 .deb, sha256 in step 3) | klayout.de |
| Python | 3.11.18 | conda-forge |

Why the KLayout `.deb`: the conda-forge KLayout is built without Ruby, and the
SKY130 DRC deck (`sky130A_mr.drc`) is Ruby ("no interpreter" error). The litex-hub
KLayout needs Qt 5.6, which no longer exists. So we unpack the official `.deb`
into `$HOME` (no root) and supply its missing Qt5/libgit2/Ruby libs from Ubuntu's own
archive the same way.

## 1. Conda env with Magic, Netgen, ciel

```bash
source ~/miniforge3/etc/profile.d/conda.sh      # miniforge3 must exist
conda create -y -n cellgym --override-channels -c litex-hub -c conda-forge \
    python=3.11 magic netgen pip
conda activate cellgym
pip install ciel==3.2.0
```

(`env/cellgym_env.yml` is the full resolved export, for reference.)

## 2. sky130A PDK, pinned

```bash
export PDK_ROOT=$HOME/pdk
ciel enable --pdk-family sky130 --pdk-root "$PDK_ROOT" 792ef67037d74c8c51811eab3ce23cf793491296
cat $PDK_ROOT/sky130A/SOURCES      # -> open_pdks 792ef67037d74c8c51811eab3ce23cf793491296
```

## 3. KLayout with Ruby (user-space)

```bash
mkdir -p ~/klayout-deb ~/klayout-deps/debs && cd ~/klayout-deb
curl -LO https://www.klayout.org/downloads/Ubuntu-22/klayout_0.30.12-1_amd64.deb
sha256sum klayout_0.30.12-1_amd64.deb
# expect c09637de6d82bfa2a102c2468e285d3d060fa59b86680f10393a0e9f69ab5763
dpkg -x klayout_0.30.12-1_amd64.deb ~/klayout-0.30.12

# runtime deps not present on stock Ubuntu: fetch .debs (no root) and unpack locally
cd ~/klayout-deps/debs
PK="libqt5core5a libqt5gui5 libqt5widgets5 libqt5network5 libqt5sql5 libqt5svg5 libqt5xml5 \
    libqt5xmlpatterns5 libqt5multimedia5 libqt5multimediawidgets5 libqt5printsupport5 \
    libqt5designer5 libgit2-1.1 libruby3.0"
ALL=$(apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts --no-breaks \
      --no-replaces --no-enhances $PK | grep '^\w' | sort -u)
NEED=""; for p in $ALL; do dpkg -s $p >/dev/null 2>&1 || NEED="$NEED $p"; done
apt-get download $NEED
for d in *.deb; do dpkg -x $d ~/klayout-deps/root; done

# wrapper placed in the conda env so `klayout` resolves to it
cat > ~/miniforge3/envs/cellgym/bin/klayout <<'W'
#!/bin/bash
K=$HOME/klayout-0.30.12/usr
D=$HOME/klayout-deps/root/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$K/lib/klayout:$D:$D/pulseaudio${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export QT_PLUGIN_PATH=$D/qt5/plugins
export RUBYLIB=$HOME/klayout-deps/root/usr/lib/ruby/3.0.0:$HOME/klayout-deps/root/usr/lib/x86_64-linux-gnu/ruby/3.0.0
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
exec $K/bin/klayout "$@"
W
chmod +x ~/miniforge3/envs/cellgym/bin/klayout
```

Check: `klayout -v` prints `KLayout 0.30.12`; `magic --version` prints `8.3.465`.
A harmless `_d25_engine.rb ... pathname` message may appear in logs of non-DRC runs.

## 4. Run M0

```bash
conda activate cellgym
export PDK_ROOT=$HOME/pdk
cd <repo>                       # e.g. /mnt/c/Users/braed/LayoutGen
python cellgym/groundtruth/run_m0.py                         # nand2_1
python cellgym/groundtruth/run_m0.py sky130_fd_sc_hd__a21oi_1
```

Flags: `--pdk-root`, `--pdk` (default sky130A), `--workdir` (default `./out`, or `CELLGYM_OUT`).
Exit code is 0 only if DRC has 0 violations and Netgen reports the circuits match.

What it does: KLayout batch DRC (`sky130A_mr.drc`, `feol/beol/offgrid=true`; the deck
defaults every group to OFF in batch mode, so they must be passed) on the cell inside
the library GDS -> Magic batch extract of the same cell (PDK `sky130A.magicrc`) ->
Netgen LVS against `sky130_fd_sc_hd.spice` with `sky130A_setup.tcl`. Per-cell logs and
reports (`.lyrdb`, `.lvs.txt`, extracted `.spice`) land in `<workdir>/<cell>/`.

The reference is the PDK `.spice` (device names `sky130_fd_pr__*`), not the `.cdl`:
the CDL uses short model names that the Netgen setup does not recognise as devices.
