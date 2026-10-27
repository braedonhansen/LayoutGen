# LayoutGen

Netlist-to-layout tooling for SKY130 standard cells.

## Quick start

Python 3.10+, no dependencies. Draw transistor schematics for the generated cell set:

    python -m cellgym.netlists.schematic --open           # all cells, gallery in a browser
    python -m cellgym.netlists.schematic nand3 --open     # one cell

See `cellgym/netlists/README.md` for the netlist generator and `env/SETUP.md`
for the DRC/LVS toolchain.
