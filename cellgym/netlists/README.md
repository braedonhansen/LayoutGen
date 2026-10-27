# cellgym.netlists (M1)

Generates the "core region" of static-CMOS cells: 1..6 inputs, PDN and PUN stack
depth <= 3, read-once networks plus irredundant repeated-input networks with
devices <= inputs + 2 (max 8). One canonical cell per input-renaming class
(323 cells: 49 read-once, 274 repeated-input).

    python -m cellgym.netlists.generate --out out/cells

Writes `<id>.sp`, `<id>.json` per cell and `index.json`. Every cell is
switch-level self-checked (`selfcheck.py`) before it is written.

- SPICE port order: inputs (A, B, ...), OUT, VDD, VSS. Device lines:
  `X<name> d g s b <model> w=<um> l=0.15` (NMOS bulk VSS, PMOS bulk VDD).
- Sizing defaults: W_n unit 0.42 um, L 0.15 um, W_p/W_n 1.5, models
  `sky130_fd_pr__nfet_01v8` / `sky130_fd_pr__pfet_01v8_hvt`. Each device width =
  unit x longest series path through it in its own network, snapped to 0.005 um.
  `--max-diffusion-width` (default none) drops cells with a wider device.
- Function hex: truth table of OUT, bit r = row r, input A = LSB of r.
- Ids: `cg_<inputs>i<devices>d_<sha1(canonical tree)[:8]>`.

Tests: `python -m pytest tests` (the netgen self-LVS test needs the WSL cellgym env).
