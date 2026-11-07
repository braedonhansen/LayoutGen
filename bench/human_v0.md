# Human reference: sky130_fd_sc_hd vs CellGym baselines (v0)

20 core cells match single-output combinational `_1` hd cells (104 considered) by logic function up to input permutation (liberty tt_025C_1v80 `function`, LEF `SIZE`). Where several core cells realize one function (repeated-input variants), the read-once one is used. Widths in 0.46 um sites. Heights differ: every hd cell is 2.72 um tall; ours are sized per cell (column `our h`), so the comparison is width at different heights, not area.

| hd cell(s) | function | core id | hd sites | euler sites | naive sites | our h um |
|---|---|---|---|---|---|---|
| clkinv_1, inv_1, lpflow_clkinvkapwr_1 | `Y = (!A)` | cg_1i1d_6dcd4ce2 | 3 | 3 | 3 | 3.06 |
| nand2_1 | `Y = (!A) \| (!B)` | cg_2i2d_94893424 | 3 | 4 | 5 | 3.40 |
| nor2_1 | `Y = (!A&!B)` | cg_2i2d_23e88f1d | 3 | 4 | 5 | 3.74 |
| a21oi_1 | `Y = (!A1&!B1) \| (!A2&!B1)` | cg_3i3d_29e68bd3 | 4 | 5 | 7 | 4.08 |
| nand3_1 | `Y = (!A) \| (!B) \| (!C)` | cg_3i3d_4f46a103 | 4 | 5 | 7 | 3.74 |
| nor3_1 | `Y = (!A&!B&!C)` | cg_3i3d_aa88aad0 | 4 | 5 | 7 | 4.42 |
| o21ai_1 | `Y = (!A1&!A2) \| (!B1)` | cg_3i3d_e286d00e | 4 | 5 | 7 | 4.08 |
| a211oi_1 | `Y = (!A1&!B1&!C1) \| (!A2&!B1&!C1)` | cg_4i4d_bae792c8 | 6 | 7 | 9 | 4.76 |
| a22oi_1 | `Y = (!A1&!B1) \| (!A1&!B2) \| (!A2&!B1) \| (!A2&!B2)` | cg_4i4d_57276752 | 6 | 6 | 9 | 4.08 |
| a31oi_1 | `Y = (!A1&!B1) \| (!A2&!B1) \| (!A3&!B1)` | cg_4i4d_d2e26f6e | 5 | 7 | 9 | 4.42 |
| o211ai_1 | `Y = (!A1&!A2) \| (!B1) \| (!C1)` | cg_4i4d_08687c21 | 6 | 7 | 9 | 4.42 |
| o22ai_1 | `Y = (!B1&!B2) \| (!A1&!A2)` | cg_4i4d_f4d71aaa | 5 | 6 | 9 | 4.08 |
| o31ai_1 | `Y = (!A1&!A2&!A3) \| (!B1)` | cg_4i4d_27a949f9 | 6 | 7 | 9 | 4.76 |
| a221oi_1 | `Y = (!A1&!B1&!C1) \| (!A1&!B2&!C1) \| (!A2&!B1&!C1) \| (!A2&!B2&!C1)` | cg_5i5d_e74e4f09 | 7 | 8 | 11 | 4.76 |
| a311oi_1 | `Y = (!A1&!B1&!C1) \| (!A2&!B1&!C1) \| (!A3&!B1&!C1)` | cg_5i5d_f07a9bfc | 7 | 8 | 11 | 5.10 |
| a32oi_1 | `Y = (!A1&!B1) \| (!A1&!B2) \| (!A2&!B1) \| (!A3&!B1) \| (!A2&!B2) \| (!A3&!B2)` | cg_5i5d_589cea8d | 7 | 7 | 11 | 4.42 |
| o221ai_1 | `Y = (!B1&!B2) \| (!A1&!A2) \| (!C1)` | cg_5i5d_7ced2fe7 | 7 | 8 | 11 | 4.42 |
| o311ai_1 | `Y = (!A1&!A2&!A3) \| (!B1) \| (!C1)` | cg_5i5d_68525f8a | 7 | 8 | 11 | 5.10 |
| o32ai_1 | `Y = (!A1&!A2&!A3) \| (!B1&!B2)` | cg_5i5d_5f096d0f | 7 | 8 | 11 | 4.76 |
| a222oi_1 | `Y = (!A1&!B1&!C1) \| (!A1&!B1&!C2) \| (!A1&!B2&!C1) \| (!A2&!B1&!C1) \| (!A1&!B2&!C2) \| (!A2&!B1&!C2) \| (!A2&!B2&!C1) \| (!A2&!B2&!C2)` | cg_6i6d_49c2319a | 8 | 10 | 13 | 4.76 |

Mean width over the 20 matches: hd 5.45 sites, euler 6.40, naive 8.70.
