# Plan DSL (M2)

Status: **all decisions made (2026-10-27); not frozen yet.** The format freezes
when M2 closes (parser, validator and renderer pass on the examples); after
that, changes are additive only.

The model decides topology: column order, diffusion sharing, contacts, and
route layers and tracks (which also place the gate contacts). The renderer owns every
coordinate. Rule values below come from the pinned PDK (open_pdks 792ef67):
the KLayout deck `sky130A_mr.drc`, the Magic tech file `sky130A.tech`, and the
hd tech LEF `sky130_fd_sc_hd__nom.tlef`. Heights and spacings derived from
those rules are rule arithmetic until the renderer passes KLayout DRC.

## Summary of decisions

| # | Question | Decision |
|---|---|---|
| 1 | Cell height | Decided: a renderer parameter, the smallest height for the cell's widest devices plus 3 channel tracks |
| 2 | Rail contacts | Renderer straps contacted VDD/VSS columns to the rail; never routed |
| 3 | Gate contacts | One per gate column; its track comes from the route that reaches it, or from the pin line if unrouted (no `gc=`) |
| 4 | Pins | `pin <net> <landing> [track]`, row-qualified landing; track only for an unrouted gate |
| 5 | Well taps | None in the cell (hd convention); renderer labels VPB/VNB; netlists gain VPB/VNB ports |
| 6 | Mixed-width shared diffusion | Align to the channel-side edge; the shared column takes the wider width |
| 7 | One-shot vs. step-wise | Decided: one-shot baseline, step-wise for RL; step interface accepted (gate+sd action, place then route, fixed state sections, reject and retry) |
| 8 | Width metric | Decided: x-only compaction; reward on exact µm width, reports in sites; checker independent of the compactor |
| 9 | Syntax | Line format, one column per line; parser also emits JSON |

## Rule values used

| Rule | Value | Source |
|---|---|---|
| poly width / spacing | 0.15 / 0.21 | poly.1a, poly.2 |
| poly endcap past diff | 0.13 | poly.8 |
| diff past poly | 0.25 | poly.7 |
| poly to diff (non-gate) | 0.075 | poly.4 |
| diff/tap spacing | 0.27 | difftap.3 |
| licon size | 0.17 | licon.1 |
| licon to gate | 0.055 | Magic tech (licon.11) |
| diff enclosure of licon | 0.04, 0.06 one direction | licon.5a/5c |
| poly enclosure of licon | 0.05, 0.08 one direction | licon.8/8a |
| poly licon to ndiff / pdiff | 0.19 / 0.235 | Magic tech (licon.14; licon.9 + psdm.5a) |
| li width / spacing | 0.17 / 0.17 | li.1, li.3 |
| li enclosure of licon | 0.08 on two adjacent edges | li.5 |
| met1 width / spacing | 0.14 / 0.14 | m1.1, m1.2 |
| mcon size | 0.17 | ct.1 |
| nwell enclosure of pdiff | 0.18 | Magic tech (diff/tap.8) |
| ndiff to nwell | 0.34 | Magic tech (diff/tap.9) |
| hd site | 0.46 × 2.72 | tech LEF |
| hd met1 tracks | horizontal, pitch 0.34, offset 0.17 | tech LEF |
| hd rails | met1, 0.48 wide, centered on y = 0 and the cell height | hd LEF (nand2_1) |

Contacted poly pitch (minimum): 0.15 + 2 × 0.055 + 0.17 = 0.43. The renderer
uses 0.46, the hd site width, as its base pitch.

## 1. Cell height — *Decided (2026-10-27)*

Height is a renderer parameter, not a fixed template. Each cell gets the
smallest height that fits its own widest NMOS and widest PMOS plus 3 channel
tracks, so small cells stay small and RL can push stack depth past 4 with no
format change. This supersedes the decision-log entry "one fixed custom
template". hd's x-conventions stay (0.46 site, met1 rails, li1 pins).

Building a cell up from VSS (y in µm), for widest widths W_n and W_p:

| Piece | y | Why |
|---|---|---|
| VSS rail | −0.24 … 0.24 (met1), li strap −0.085 … 0.085 | hd convention |
| NMOS diffusion | from ≥ 0.135, height W_n | half the 0.27 diff spacing to the abutting row |
| Channel tracks c0, c1, c2 | next 3 met1-grid positions ≥ 0.275 above ndiff | licon-to-ndiff 0.19 + 0.085 |
| PMOS diffusion | from c2 + 0.32, height W_p | licon-to-pdiff 0.235 + 0.085 |
| N-well bottom edge | between ndiff top + 0.34 and pdiff bottom − 0.18 | diff/tap.9, diff/tap.8 |
| VDD rail | at the first multiple of 0.34 ≥ pdiff top + 0.135 | met1 grid |

Heights therefore come in 0.34 steps. With today's sizing (unit × series
depth), uniform stack depth d gives:

| Stack | NMOS / PMOS | Height |
|---|---|---|
| 1 | 0.42 / 0.63 | 2.72 (same as hd) |
| 2 | 0.84 / 1.26 | 4.08 |
| 3 | 1.26 / 1.89 | 5.10 |
| 4 | 1.68 / 2.52 | 6.12 |
| 5 | 2.10 / 3.15 | 7.14 |

Each stack level adds 1.02 µm. Rows are sized separately, so a NAND3 (NMOS
stack 3, PMOS stack 1) is shorter than a uniform stack-3 cell. The M1 core
set spreads from 2.72 to 5.10 µm (262 of 323 cells at 5.10).

Met1 tracks are named by region and **numbered from the channel outward**:
`n0` and `p0` are the row tracks next to the channel; `c0…c2` are the channel
tracks, bottom to top. A taller cell only adds row tracks at the rail end, so
a plan stays valid at any height that fits it, and track counts stay
additive under the format freeze. The renderer reports how many row tracks a
height has; a plan naming a track the cell doesn't have is a validator error.

Why 3 channel tracks: the half adder has no legal routing with 2, 116 with 3,
and 2,064 with 4. Going from 3 to 4 tracks costs 0.34 µm on every cell to save
two compaction spacings on that example.

Consequences:

- No device-width cap from the template; the sizing algorithm is free of one.
  The only cap left is v1's no-fingers rule, which is a scope choice.
- Cells of different heights can't share a row. That is fine for the gym: the
  benchmark compares widths within a region.
- Area against hd is still a caveated, secondary number.

## 2. Rail contacts — *Decided (2026-10-27)*

A contacted sd column whose net is VSS (NMOS row) or VDD (PMOS row) gets a
straight li strap to the rail, plus mcons into the met1 rail. The renderer
draws it; plans never route VDD or VSS. There is nothing to decide: rows align
to the channel side, so the space between a rail contact and its rail is always
free. Validator errors: VDD in the NMOS row, VSS in the PMOS row, a rail
diffusion island with no contact. Rail columns keep the `*`, so every contact is
marked the same way.

## 3. Gate contacts — *Decided (2026-10-27)*

- Every gate column has exactly one poly contact (two for a split column). A
  gate can only be reached through li, never through poly routing, so a contact
  is always needed. The only choice is its channel track.
- **The track is not written on the column line.** It is the track of the route
  that reaches the gate (`gate:<col>` endpoint). A gate with no route (an input
  that drives a single gate column) takes its track from its pin line. A gate
  that is neither routed nor pinned with a track is a validator error. This
  removes a field that only repeated the routes, and every track choice happens
  in the routing phase of a step-wise episode.
- A split column (`n=A p=B`) cuts the poly in the channel. Its contacts are the
  endpoints `gate:<col>n` and `gate:<col>p`, and the NMOS one must be on a lower
  track than the PMOS one.
- A contact is an li obstacle on its track. A met1 route on the same track
  reaches it through an mcon; li reaches it directly.
- Adjacent gate contacts on the same track (0.46 apart) break the poly spacing
  rule: the poly pads are 0.27 wide, leaving 0.19 < 0.21. Compaction
  (section 8) adds the space, so staggering tracks saves width but is not
  required.

## 4. Pins — *Decided (2026-10-27)*

`pin <net> <landing> [track]`, with landing `gate:<col>` for inputs and
`nmos:<col>` / `pmos:<col>` for outputs. The pin shape is the li1 at that
landing, as in hd. The row-qualified landing removes the old ambiguity of a
column-only pin (OUT at nmos:4 or pmos:6?). The track is required only for an
unrouted gate (section 3); on a routed gate it may be given but must match the
route. A cell has 1–3 output pins. An input with several gate columns names one
of them, and a route joins the rest.

## 5. Well taps — *Decided (2026-10-27)*

No taps inside the cell. The row provides them, as with hd's
`tapvpwrvgnd_1`. The renderer labels the n-well VPB and the p-well VNB. M1
change: netlist ports become `inputs…, outputs…, VDD, VSS, VPB, VNB`, with PMOS
bulk on VPB and NMOS bulk on VNB. M0 already passes DRC and LVS on tapless hd
cells, so the flow is proven. An in-cell tap would cost about a site per row in
every cell, which would only dilute the width reward. Before M2 closes, check
the DRC deck for a tap-distance check that our runs don't enable.

## 6. Mixed-width shared diffusion — *Decided (2026-10-27)*

- Devices align to their row's **channel-side edge**; the row is as tall as its
  widest device. Contacts and stubs facing the channel line up, and width steps
  happen on the rail side, away from the routing.
- A shared sd column takes the larger of its two neighbors' widths. The step
  happens on the narrower device's side. Compaction keeps the step at least
  0.25 from the narrower gate (poly.7), which KLayout must confirm in M2.
- The fast checker gets one rule for this: the distance from a diffusion step
  to a gate.
- A wide diffusion gets as many licons as fit along its height (licon 0.17,
  spacing 0.17).

## 7. One-shot vs. step-wise — *Decided (2026-10-27)*

Decided: one-shot is the **baseline**; step-wise is the **intended path for
RL**, which has to place cells larger than anything in the SFT set.

Why step-wise: a non-reasoning model writing a whole plan in one pass has to
track all of its state implicitly (devices still to place, the net at each
row's frontier, open nets, occupied tracks), and that hidden state grows with
the cell. Step-wise shows the state at every step, so each decision is the same
local problem at any cell size. The base model may not be a reasoning model,
so this cannot be left to the model to do internally.

Both modes write the same plan lines. A step-wise episode is a sequence of
(state, one plan line) pairs; joined, its lines are a valid one-shot plan, and
any one-shot plan can be replayed into step-wise training pairs. So the Euler
solver's plans feed both modes.

Because the format freezes at M2, the step interface freezes with it. All four
recommendations below were accepted on 2026-10-27; the retry limit k is a
training parameter, set at M6.

- **Action unit.** One placement step = one gate column plus
  the sd column after it (`col g …` + `col sd …`, two lines), since that is
  the unit where sharing or a break is decided. The first step also emits the
  leading sd column.
- **Phases.** Place all columns first, then route one net per
  step, then pins in a single step. Routing during placement would make
  routes depend on columns that don't exist yet.
- **State shown to the model.** A fixed set of sections,
  each a few lines:
  - the netlist devices not yet placed
  - the frontier: the last sd net in each row and whether it has a contact
  - open nets, with the endpoints placed so far
  - the last few placed columns
  - in the routing phase, per-track occupancy as column ranges

  The unplaced-device list still grows with the cell. The local decision does
  not: which device next, share or break, and (in routing) which layer and
  track.
- **Illegal step.** The environment rejects it with a validator
  message and the model retries, at most k times. This also gives a natural
  per-step penalty if potential-based shaping is adopted (open item
  "Per-step reward").

Cost: a step-wise episode makes about (gates + nets) model calls, each
re-reading the state. Keep the state compact and let vLLM's prefix caching
reuse the fixed prompt.

## 8. Width metric and x-compaction — *Decided (2026-10-27)*

A uniform 0.46 grid cannot work. A li stub crossing the channel at an sd column
sits only 0.06 from a gate-contact pad in the next column, and li needs 0.17.
The rule check found no legal routing for the half adder at uniform pitch, even
with 5 channel tracks. An inverter fails the same way. Real hd cells use uneven
spacing for this reason.

- **Compaction, x only.** The renderer places columns left to right at the
  minimum x that every spacing rule allows (longest-path compaction). It never
  reorders columns, reassigns tracks or adds jogs; those are the plan's
  decisions, and their cost has to stay visible as width.
- The validator rejects only true overlaps: two nets on the same layer, the
  same track and an overlapping column range. Near-misses become width, not
  errors.
- **Width, not area.** Height comes from the netlist's widest devices, which
  no plan can change, so area / area_euler = W / W_euler for a given cell.
- **Reward: exact compacted width in µm** (0.005 grid):
  1 + (W_euler − W) / W_euler. Whole sites are too coarse for GRPO's
  in-group comparisons; two plans a few spacings apart would often tie.
- **Reporting: width in hd sites** (0.46, rounded up), comparable with hd and
  with placement. The gap between µm gains and site gains is reported too.
- The Euler baseline goes through the same router and compactor, so
  W / W_euler compares within one renderer.

Guards against reward hacking through the compactor:

- The fast checker (M3) checks the rendered geometry with its own code path,
  never the compactor's spacing table, so the two cannot share a bug.
- The M3 audit against KLayout includes compacted cells with many near-misses,
  where a too-tight spacing would hide.

## 9. Syntax — *Decided (2026-10-27)*

A line format, one statement per line, `#` comments.

```
cell <name>
in <inputs...>
out <outputs...>
col <i> sd n=<net>[*] p=<net>[*] [st=<stage>]
col <i> g  n=<in> p=<in> [st=<stage>]
route <net> <li1|met1> <track> <endpoint> <endpoint>...
pin <net> <landing> [track]
```

- `*` marks a contacted sd. `-` means nothing in that row (no diffusion, or
  no device).
- Column indices must run 0, 1, 2, … with no gaps. They are redundant, but
  they let routes and error messages point at a column, and they let a
  step-wise model keep its place.
- Consecutive sd columns in a row with the same net form one diffusion island
  (one contact is enough). With different nets they are a **diffusion
  break**. There is no break token.
- Endpoints: `nmos:<col>`, `pmos:<col>`, `gate:<col>` (`gate:<col>n` /
  `gate:<col>p` for a split column).
- `st=` labels are optional. The renderer ignores them; they are for reading
  and for step-wise generation. They name the stage by its output net.

Why not JSON: on the half adder the line format is 710 characters against
1,009 for compact JSON and 1,952 pretty-printed; token counts wait for the
base model's tokenizer (M6). Size is the smaller reason. JSON in the original
sketch splits each column across parallel arrays (`columns`, `nmos`, `pmos`,
`contacts`), so a model must keep them the same length by counting, which is
the kind of error the DSL exists to remove. A line holds one whole column, and
a step-wise action is two whole lines. The parser also writes JSON for
tooling.

### Routing model

- li1 runs horizontally only in the channel (`c*`). met1 runs on any track.
- An sd endpoint reaches a channel track through a vertical li stub in its own
  column. The stub occupies every channel track it crosses.
- A met1 route reaches an sd endpoint through an mcon. This works on any
  track over that endpoint's row (the sd's li strap covers the row height);
  on a channel track it needs a stub.
- A route that reaches a gate puts that gate's contact on the route's track.
  Two routes reaching the same gate must use the same track.
- A route is one horizontal segment. A net may have several routes; they
  connect wherever they share an endpoint.

## Validator (runs before rendering)

1. Grammar: indices are sequential, nets are declared, tracks exist at this
   cell's height.
2. Devices: every netlist device appears exactly once. Each gate column plus
   its two neighboring sd nodes must match a device's gate and terminals, with
   its width.
3. Contacts: an sd node needs a contact if its net appears anywhere else (rail
   nets always). A missing contact is an LVS open.
4. Connectivity: per net, routes plus shared diffusion join every contacted
   node, gate and pin (union-find). Each output is checked on its own.
   Every gate gets a contact track from a route or its pin, with no conflict.
5. Overlaps: per layer and track, no two nets overlap, counting gate contacts
   and stubs.
6. Rails: section 2.

Each error names the line and the rule. These messages are what the agent
reads during repair, so they stay short and stable.

## Example: AOI21

OUT = NOT(A·B + C), matching the M1 netlist `cg_3i3d_29e68bd3`
(`P(C,S(A,B))`, C on the VDD side of the pull-up). The plan doc's sketch had
the pull-up the other way round; same function, different devices, so the
validator rejects it against this netlist. Gate order C, A, B is an Euler path
in both rows: no break, and n1 is internal shared diffusion with no contact.
Each input drives one gate column, so no input is routed and every input pin
carries its gate's contact track.

```
cell aoi21
in A B C
out OUT
col 0 sd n=VSS*  p=VDD*
col 1 g  n=C     p=C
col 2 sd n=OUT*  p=p1*
col 3 g  n=A     p=A
col 4 sd n=n1    p=OUT*
col 5 g  n=B     p=B
col 6 sd n=VSS*  p=p1*
route OUT li1  c1 nmos:2 pmos:4
route p1  met1 p0 pmos:2 pmos:6
pin A gate:3 c0
pin B gate:5 c2
pin C gate:1 c2
pin OUT nmos:2
```

## Example: half adder (M2 exit)

Three stages, two outputs, 14 devices, all stages from the enumerated core
set:

- x = NOR2(A, B)
- S = AOI21(A, B, x) = NOT(A·B + x) = A ⊕ B
- C = NOR2(x, S) = A·B

The C stage is a trick: C is 1 only when neither x (both inputs 0) nor S
(exactly one input 1) holds. It avoids a NAND plus an inverter, which would
make 4 stages, over the current stage cap.

Order: NOR(x) | NOR(C) | AOI21(S). The NMOS row shares diffusion all the way.
The PMOS row needs one break between the C and S stages, because NOR(C)'s
pull-up always ends on C or VDD and AOI21's always ends on p1 or S. Columns 8
and 9 carry that break; in the NMOS row they are one VSS island.

Files: `examples/half_adder.plan`, netlist `examples/half_adder.json` (hand-written, M1 format).

```
cell half_adder
in A B
out S C
col 0  sd n=VSS*  p=x*    st=x
col 1  g  n=A     p=A     st=x
col 2  sd n=x*    p=px    st=x
col 3  g  n=B     p=B     st=x
col 4  sd n=VSS*  p=VDD*
col 5  g  n=x     p=x     st=C
col 6  sd n=C*    p=p2    st=C
col 7  g  n=S     p=S     st=C
col 8  sd n=VSS*  p=C*    st=C
col 9  sd n=VSS   p=S*    st=S
col 10 g  n=x     p=x     st=S
col 11 sd n=S*    p=p1*   st=S
col 12 g  n=A     p=A     st=S
col 13 sd n=n1    p=VDD*  st=S
col 14 g  n=B     p=B     st=S
col 15 sd n=VSS*  p=p1*   st=S
route x  met1 c0 nmos:2 pmos:0 gate:5 gate:10
route S  li1  c1 nmos:11 pmos:9 gate:7
route C  li1  c2 nmos:6 pmos:8
route p1 met1 p0 pmos:11 pmos:15
route A  met1 c2 gate:1 gate:12
route B  met1 c1 gate:3 gate:14
pin A gate:1
pin B gate:3
pin S nmos:11
pin C nmos:6
```

Validated by `cellgym/dsl/validate.py` (tests in `tests/dsl`):

- Function: a switch-level simulation of the 14 transistors read off the
  columns gives S = A ⊕ B and C = A·B (AOI21: NOT(A·B + C)).
- Gate tracks: every gate gets one from its route or pin, with no conflict.
- Overlaps: none. The route set is one of 116 legal ones for this column
  order with 3 channel tracks, and the cheapest.
- Near-misses: 7, each costing compaction space.
  - Stubs next to gate contacts: x's PMOS stub at col 0 next to A's contact
    at col 1 on c2, S's NMOS stub at col 11 next to x's contact at col 10
    on c0.
  - Same-track contacts: x's contacts on c0, A's on c2, B's on c1.

Not rendered or run through DRC/LVS yet.
