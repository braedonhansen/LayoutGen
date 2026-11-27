# M3 fast-checker audit

Ground truth: KLayout deck DRC (sky130A_mr.drc) = 0 AND Magic full DRC = 0 (excluding only the tapless-by-design rules below) AND Netgen LVS match, no property errors.

Excluded Magic rules: `All nwells must contain metal-connected N+ taps (nwell.4)`; `N-diff distance to P-tap must be < 15.0um (LU.2)`; `P-diff distance to N-tap must be < 15.0um (LU.3)`

- layouts audited: 60 (ground truth fails 0, clean 60)
- agree: 60; false pass: 0; false fail: 0 (0.00% of all, 0.00% of ground-truth-clean)
- DRC agree 60/60; LVS agree 60/60
- fast check time: median 8.17 ms, p95 17.50 ms

## Per mutation type

| mutation | n | gt fails | agree | false pass | false fail |
|---|---|---|---|---|---|
| none | 60 | 0 | 60 | 0 | 0 |

## Rules hit (layouts flagged)

| rule | fast checker | ground truth |
|---|---|---|
