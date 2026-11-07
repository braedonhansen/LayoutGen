# CellGym baseline results (v0)

Splits `bench/splits_v0.json` (sha256 `be8c6a6af51e`), Euler reference widths `bench/euler_widths_v0.json` (sha256 `43093a7145d2`). pass@1 (fast) = the plan parses, validates, renders and the fast checker is clean; pass@1 (GT) additionally needs full ground truth (KLayout DRC + Magic DRC + Netgen LVS). Width, W/W_euler, reward and breaks are over GT-confirmed passes; reward = 1 + (W_euler - W) / W_euler for a pass, 0 for a fail, averaged over all cells of the split. Breaks = adjacent gate columns that do not share diffusion in some row.

| split | baseline | n | pass@1 (fast) | pass@1 (GT) | mean sites | mean W um | mean W/W_euler | mean reward | mean breaks |
|---|---|---|---|---|---|---|---|---|---|
| train | naive | 111 | 1.000 | 1.000 | 14.77 | 6.572 | 1.600 | 0.400 | 5.14 |
| train | euler | 111 | 1.000 | 1.000 | 9.39 | 4.090 | 1.000 | 1.000 | 0.19 |
| val | naive | 15 | 1.000 | 1.000 | 15.47 | 6.911 | 1.602 | 0.398 | 5.40 |
| val | euler | 15 | 1.000 | 1.000 | 9.93 | 4.307 | 1.000 | 1.000 | 0.27 |
| test_id | naive | 25 | 1.000 | 1.000 | 14.40 | 6.427 | 1.578 | 0.422 | 5.00 |
| test_id | euler | 25 | 1.000 | 1.000 | 9.32 | 4.052 | 1.000 | 1.000 | 0.32 |
| test_ood | naive | 172 | 1.000 | 1.000 | 16.91 | 7.582 | 1.581 | 0.419 | 6.22 |
| test_ood | euler | 172 | 1.000 | 1.000 | 10.90 | 4.789 | 1.000 | 1.000 | 0.44 |
| all | naive | 323 | 1.000 | 1.000 | 15.91 | 7.114 | 1.588 | 0.412 | 5.72 |
| all | euler | 323 | 1.000 | 1.000 | 10.21 | 4.470 | 1.000 | 1.000 | 0.33 |

Baselines: `euler` = exact Euler-path placement (fewest diffusion breaks) plus the shared routing search; `naive` = no diffusion sharing (every NMOS/PMOS pair on its own islands), same routing. Regenerate with `python -m cellgym.bench.score <plans> --name <label> --confirm` and `python -m cellgym.bench.results`. Ground-truth wall time: naive 220 s, euler 218 s.
