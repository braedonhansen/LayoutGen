# M3 fast-checker audit

Ground truth: KLayout deck DRC (sky130A_mr.drc) = 0 AND Magic full DRC = 0 (excluding only the tapless-by-design rules below) AND Netgen LVS match, no property errors.

Excluded Magic rules: `All nwells must contain metal-connected N+ taps (nwell.4)`; `N-diff distance to P-tap must be < 15.0um (LU.2)`; `P-diff distance to N-tap must be < 15.0um (LU.3)`

- layouts audited: 1249 (ground truth fails 0, clean 1249)
- agree: 1249; false pass: 0; false fail: 0 (0.00% of all, 0.00% of ground-truth-clean)
- DRC agree 1249/1249; LVS agree 1249/1249
- fast check time: median 13.67 ms, p95 28.53 ms

## Per mutation type

| mutation | n | gt fails | agree | false pass | false fail |
|---|---|---|---|---|---|
| none | 1249 | 0 | 1249 | 0 | 0 |

## Rules hit (layouts flagged)

| rule | fast checker | ground truth |
|---|---|---|
