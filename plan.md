# Plan: an OSM agent on LARCO washing machines

Started 2026-09-28. The dataset was chosen in `~/Downloads/demo/osm-candidates/`
(its README has the search and every rejection); this plan starts where that
search ended.

## The scenario

Same shape as the Volve example:

- a customer shares labelled cycles from **one washing machine**, the healthy
  Becken BWM5381IX (`becken`);
- we report how the agent does **on settings of that machine it has never
  seen** (test);
- then we run it over a **second unit of the same model, marked faulty**
  (`becken-flt`), with its labels hidden, and score the predictions
  afterwards against the held-back labels (delivery).

The agent labels each 5 s window of vibration as **`fill`**, **`wash`**,
**`spin`**, **`drain`** or **`heating`**, the phases anyone who has used a
washing machine knows. It needs
nothing but a clip-on accelerometer: no access to the machine's controller.

**Decision (2026-09-28): becken for library, validation and test;
becken-flt for delivery only.** becken-flt never informs any choice, so its
delivery score is the cleanest "machine the model has never seen" number.
The test number is weaker than Volve's: new settings of the same machine,
not a new machine. The README reports both, side by side.

## Why this dataset

From the candidate search, LARCO is the first dataset where all of these hold:

| requirement | LARCO |
|---|---|
| licence allows commercial use | **CC BY 4.0** for the data (`general.zip`, `vibrations.zip`, `audio.zip`), stated on Zenodo and in `LICENCE.txt`; scripts MIT |
| intuitive states | fill / wash / spin / drain / heating |
| states defined by dynamics | wash = drum tumbling with ~10 s pauses; spin = steady high-speed rotation; fill = drum mostly still while water enters |
| many independent recordings, one machine | 93 cycles of becken over a controlled grid of programs, wash temperatures and loads |
| trustworthy labels | 1 Hz labels from the machine's own power and water sensors, not a human annotator |
| Omega adds something a threshold cannot | on one cycle per unit, Omega 0.92 F1 on fill vs 0.69 for a threshold (see "Evidence so far") |

## Evidence so far (exploration, not the pipeline)

One cycle per unit (`cold_cotton_40_2`), 1,565 pure windows (159 fill /
1,042 wash, thinned to every 3rd / 364 spin), train on one unit, test on the
other, pooled macro-F1. Code: `osm-candidates/larco_baseline.py` and
`larco_omega.py`; Omega 1.5 from the local encoder-only agent, with global
normalisation fitted on the training unit.

| method | macro-F1 | fill | wash | spin |
|---|---|---|---|---|
| RMS level (threshold) | 0.86 | 0.69 | 0.95 | **0.95** |
| FFT band-power kNN | 0.70 | 0.57 | 0.88 | 0.65 |
| Omega 1.5 kNN | 0.83 | **0.92** | 0.90 | 0.66 |
| Omega + RMS (weight chosen once, after the fact) | ~0.93 | | | |

- **Omega's gain is fill.** The threshold calls 64 of becken's 78 fill
  windows wash.
- **Omega's loss is spin, trained on becken and scored on becken-flt:** 159
  of 183 spin windows go to wash. becken-flt spins roughly twice as hard.
  That is exactly the delivery direction, so expect it again at Stage 7.
- **Not evidence for the example yet.** It is one cycle per unit, the
  combination weight was chosen after seeing the results, and the platform's
  classifier is a kNN over embeddings only, so it cannot add RMS features.

## The two units (from the dataset's `aggregated_data.csv`)

| | becken | becken-flt |
|---|---|---|
| model | Becken BWM5381IX | Becken BWM5381IX |
| `state` in the metadata | **new** | **faulty** ("flt" = faulty; the only faulty machine in LARCO) |
| cycles with vibration | 93 | 106 |
| example recording date (`cold_cotton_40_2`) | 2023-06-26 | 2023-10-16 |
| lab position (`appliance_id`) | m1 | m1 |
| energy, cold cotton 40 °C at 0 kg | 413 Wh | 535 Wh (about 20–40% more across loads) |

- **The fault is not described** in the Zenodo record, the per-cycle table or
  the dataset's scripts. The paper (Scientific Data) may say more; its page
  needs a login.
- **We can't tell whether they are one machine or two.** Both sat in lab
  position `m1` four months apart, and `appliance_id` is a position slot
  shared by unrelated machines.
- **Filenames:** `wm_<unit>_<room>_<program>_<wash °C>_<load kg>`. `<room>` is
  the lab's air-conditioning setting: `cold` 16 °C, `warm` 25 °C, `hot` 32 °C.

## becken's cycles, and the split

**Settings and rooms are tangled.** becken's 93 cycles fall into 58
**setting groups** (program × wash temperature × load):

| program family | groups | cycles | rooms | spin |
|---|---|---|---|---|
| cotton (0/30/40/60 °C × loads 0, 2, 4, 6, 8, 10, 11 kg) | 28 | 56 | each group a **cold + hot pair** | 1400 rpm |
| eco (40–60 °C × the same 7 loads) | 7 | 14 | each group a cold + hot pair | 1400 rpm |
| 11 other programs (15-min, 20-deg, delicate, fast-45, intensive, mix, sport, sterilization, synthetic, wool) | 23 | 23 | **warm room only**, one cycle each | 600–1200 rpm |

The two cycles of a cotton or eco group are near-twins: the same program,
temperature and load, differing only in room temperature. A split that puts
one twin in training and the other in test would test on a near-copy.

**Decision (2026-09-28): random by setting group, stratified by family.**
- **Whole groups** are assigned to library, validation or test at random,
  about 60/20/20. No twin can straddle two roles.

  | role | cotton groups | eco groups | other-program groups | ≈ cycles |
  |---|---|---|---|---|
  | library | 16 | 4 | 14 | 54 |
  | validation | 6 | 2 | 5 | 21 |
  | test | 6 | 1 | 4 | 18 |
- **Seed `20260928`,** fixed before any data is downloaded. The split is
  computed from the archive listing alone (`prep/split.py` →
  `data/split.json`) and is **never re-rolled.**
- **What the test number means:** "works on settings of this machine it has
  never seen". For cotton and eco, that's a load and wash-temperature
  combination absent from training. For the other programs, some may land
  entirely in test: a small taste of "a program it has never seen". Scores
  are reported per family.
- **Check at Stage 4b:** also score validation under a plain random split by
  cycle. If plain random scores much higher, the twins were leaking and the
  group split was needed. If they match, nothing was lost.
- **Delivery:** all 106 of becken-flt's cycles, unlabelled. Exploration
  already scored becken-flt `cold_cotton_40_2`. It stays in delivery because
  delivery informs no choice, but it is reported on its own line as "the
  cycle where the spin failure was found".
- **The draw (2026-09-28):** library 34 groups / 54 cycles, validation 13 /
  21, test 11 / 18, exactly the allocation above. The test groups are:
  - cotton: 0_8, 0_10, 30_10, 40_2, 60_0, 60_10;
  - eco: 40-60_6;
  - other: 20-deg_20_2, delicate_30_2, fast-45_40_0, wool_40_2.

  **The draw put `cotton_40_2` in test,** and becken's cold-room
  `cotton_40_2` cycle is the one exploration used. It stays (no re-roll) and
  is reported on its own line; the test is scored with and without it.

## Scope

- **Model input: the 9 vibration channels only (decided 2026-09-29).**
  - **The 1 Hz measurements** (power, current, voltage, water flows,
    pressure, temperatures, humidity) are kept for analysis only.
  - **Leakage:** the labels are derived from those measurements (heating ≈
    power at 1.7 kW, fill and drain ≈ water flowing in and out), so as
    inputs they would hand the model the answer.
  - **The story:** the example's pitch is a clip-on accelerometer on a
    machine whose controller you can't read.
  - **Audio** (`audio.zip`, 11 kHz, 24.5 GB) is not used.
- **Channels:** all 9 (3 triaxial sensors: `back`, `side`, `top`, each
  x/y/z). No channel is flat on becken or becken-flt. becken clips at ±2.19 g
  during spin (0.1% of samples on becken-flt `side.z`); keep, and note it.
- **Cycles:** all 93 of becken; becken-flt's cycles at matching settings.
  Vibration is ~2.4 GB per unit, so Stage 0 fetches the labels first and the
  vibration second.
- **Out of scope:** the other lab machines (KUBO sensors barely register
  spin; becken 5379's top sensor is flat; the KUNFT pair are different
  models), the unlabelled household machines, dryers, audio.

## Labels

**Decision (2026-09-29, revised the same day): four states,** fill / wash /
spin / drain. Every labelled second maps to exactly one, by priority (first
match wins):

| priority | state | rule | becken, share of time |
|---|---|---|---|
| 1 | `spin` | `centrifuge_label == 1` (also while draining or filling) | 9% |
| 2 | `fill` | `water_label == 1` | 5% |
| 3 | `drain` | `water_label == -1`, or `2` (inlet and outlet at once, 2–8 s) | 3% |
| 4 | `wash` | no label (heater on or off) | 83% |

- **Four states (the revision).** The first decision was five states, with
  `heating` (`heating_label == 1`, 8%) at priority 2. The five-state
  search showed vibration can't tell heating from wash (Stage 4b,
  "Why wash and heating are confused"), so the user chose option A: the
  heater label is no longer a state, and its seconds take their drum/water
  state (almost always wash; fill or drain for the few heated seconds of
  eco cycles). Stages 1b–4b are rerun on four states; the five-state
  outputs are kept in `fit/out/five_states/`. The tables and counts
  below are from the five-state build unless marked.

- **Wash has no label of its own.** It covers the drum's tumbling, its ~10 s
  pauses and the pre-spin phase. That is why no "idle" state was defined:
  almost no low-power run inside a cycle exceeds 60 s.
- **A window is labelled only if every second it covers has the same
  state.**

**Pure windows available** (step = window, 1024 samples at 200 Hz = 5.1 s):

| role | fill | wash | spin | drain | heating |
|---|---|---|---|---|---|
| library | 4,649 | 79,442 | 9,480 | 2,308 | 8,272 |
| validation | 1,662 | 29,372 | 3,203 | 827 | 2,559 |
| test | 1,575 | 23,947 | 3,421 | 740 | 2,925 |

The 512-sample window roughly doubles each count.

**Risks to report, not reasons to drop:**
- **Heating** looks like wash's pauses to the accelerometer (Stage 1a): the
  heater draws 1.7 kW, but the drum mostly rests. Expect heating ↔ wash
  confusion.
- **Drain** runs last a median of ~5 s, about one window; the pure windows
  come from the longer drains.
- **Spin edges are soft** (Stage 1a). Labels are kept as given (decision 3), so
  expect pre-spin windows labelled wash or drain to be predicted as spin.

### Stage 1a findings (2026-09-29, `data/preflight_raw.json`)

- **`water_label = 2`** (undocumented) is inlet and outlet flowing at once,
  2–8 s at a fill → drain switch, in 36 cycles. Mapped to drain.
- **Heating is not tumbling:** its vibration matches the quiet half of wash
  (the drum's pauses), and 0% of heating seconds reach the motion level that
  2–10% of wash seconds reach. (It was then going to be dropped; the
  five-state decision keeps it as its own state instead.)
- **Fill is louder than wash,** about 1.5–2× wash's median level (becken
  0.013 vs 0.0065 g), **yet the motor is off:** fill draws 7 W (p90 13 W).
  The vibration likely comes from water rushing in through the valve and
  pipes, not from the drum. (An interim note said the drum turns during
  fill; the power data says otherwise.)
- **Spin labels have soft edges:**
  - **Spin ends line up** with the vibration (median −2 to +7 s), but edges
    scatter by about ±20 s.
  - **Before each spin label** comes a 2–3 minute pre-spin phase: the drum
    turns at intermediate speed to spread the load while the pump drains.
    It's labelled wash or drain, at 2–10× wash's vibration level and
    130–235 W.
  - **The spin label itself starts quietly** (1× wash level, 15 W) and ramps.
- **Clock:** vibration starts 1–2 s before the labels (its logger starts
  first). Harmless. The spin-end edge test puts vibration within a few
  seconds of the labels overall (median +3 s). 7 cycles are flagged beyond
  ±30 s, 5 of them on a single edge. The gentle programs (delicate, wool)
  quiet down ~75 s before their spin label ends.
- **Recording dates:** becken 2023-05-23 .. 08-20 (54 days), becken-flt
  2023-08-21 .. 12-18 (63 days), back to back. That is consistent with one
  machine that developed a fault, but doesn't prove it.
- **Data quality:**
  - **becken-flt `cold_cotton_30_4`:** vibration covers only 42% of the
    cycle (77 of 180 min). The one FAIL.
  - **Two cycles drop from 200 to 152 Hz mid-cycle** (becken
    `warm_sterilization_2` in the library, becken-flt `warm_synthetic_6`).
    Resampling handles it.
  - **Clipping at ±2.19 g** above 0.1% of samples in 3 becken and 31
    becken-flt cycles (up to 1% on becken-flt `side.z`). becken-flt spins
    harder: its median spin level is 0.084 vs 0.053 g.
  - **becken `warm_wool_40_6` (validation) has no spin at all:** the wool
    program skipped it. Kept.

**Result:** no blocking FAIL. The one coverage FAIL is acknowledged (decision 1). The WARNs are listed in
the report and none blocks Stage 1b.

**Decisions made to close Stage 1a:**
1. ~~becken-flt `cold_cotton_30_4`: exclude from delivery, or keep its
   covered 77 minutes?~~ **Decided 2026-09-29: keep it for now, and revisit**
   whether to drop the uncovered part. Its coverage FAIL is acknowledged in
   `prep/preflight_raw.py` (`ACKNOWLEDGED`), so it is still printed but no
   longer blocks Stage 1b.
2. **State set: decided 2026-09-29,** five states (see "Labels").
3. **Spin edges: decided 2026-09-29, keep the labels as given.** No guard
   band. The 2–3 min pre-spin phase stays labelled wash or drain as the
   dataset has it, and the quiet first seconds of each spin label stay
   spin. This touches roughly 5% of each cycle. The scores therefore include
   windows whose label disagrees with what the sensor feels, and the README
   says so. A guard band is a possible follow-up, reported separately.

**Stage 1a is closed (2026-09-29).**

## Stages

**Numbered to match the Paderborn example (renumbered 2026-09-29).** From
Stage 2 on, the numbers and meanings are Paderborn's, and from Stage 4 on
Volve's too:
- 2: role files;
- 3: preflight them;
- 4: the bar and Omega on the Optimize API;
- 5: test once;
- 6–7: deliver and score.

Data checking and preparation is Stage 1, in three steps. **Each preflight
is still its own step:** read-only, it prints one PASS / WARN / FAIL line
per check and writes a report (`data/preflight_*.json`), and the next step
refuses to run while any check FAILs.

| stage | step | writes | checked by | was |
|---|---|---|---|---|
| 0 | split and download | `data/split.json`, `data/raw/` | 1a | 0 |
| 1a | preflight the raw cycles | `data/preflight_raw.json` | – | 1 |
| 1b | prepare (200 Hz grid, one state per row) | `data/prepared/` | 1c | 2 |
| 1c | preflight the prepared files | `data/preflight_prepared.json` | – | 3 |
| 2 | build the role files | `data/roles/` | 3 | 4 |
| 3 | preflight the role files | `data/preflight_roles.json` | – | 5 |
| 4a / 4b / 4c | the bar / Omega search / confirm on all 21 validation cycles | `fit/out/` | the stage itself | 6 / 7 / (new) |
| 5 | test once | `fit/out/` | the stage itself | 8 |
| 6 / 7 | deliver / score the delivery | `fit/out/` | the stage itself | 9 / 10 |

### Stage 0 — Split and download

- **`prep/split.py`:** reads the archive listing and writes the group split
  (`data/split.json`) before any cycle is downloaded.
- **`prep/download.py`:** fetches files by HTTP range request, avoiding the
  13.5 GB archive. First the 1 Hz label CSVs for becken and becken-flt
  (~1 MB a cycle), then the vibration parquets (`--vibration`, ~25 MB a
  cycle). Also `LICENCE.txt`, `aggregated_data.csv`, `metadata.xlsx`, and a
  manifest of each file's archive path and size. Retries Zenodo's rate
  limit (429), resumes where it stopped.

### Stage 1a — Preflight: raw cycles (`prep/preflight_raw.py`)

Per cycle, against what later stages assume. The thresholds were revised
after the first run; the reasons are in "Stage 1a findings" above.

| check | FAIL if | WARN if |
|---|---|---|
| files | a cycle in `split.json` lacks its label CSV or vibration parquet, or a size differs from the manifest | |
| schema | label columns (`timestamp`, the three labels, `power`) or the 9 vibration channels missing | |
| label values | anything outside `centrifuge` {0,1}, `water` {−1,0,1,2}, `heating` {0,1}, or NaN | |
| label timeline | | gaps > 1.5 s, or non-increasing timestamps |
| vibration rate | median outside 140–220 Hz, or non-increasing timestamps | gaps > 1 s; the hourly rate changing by > 10% |
| channels | a channel flat (std < 1e-4) | clipping at ±2.19 g above 0.1% of samples |
| states | neither fill nor spin | fill or spin under 60 s |
| coverage | vibration for < 50% of label seconds | < 98% |
| alignment | | the median offset between the label's spin end and the vibration's drop beyond ±30 s |

**Alignment, as run:** correlating vibration with power was tried first and
dropped. Tumbling repeats every ~26 s, so the correlation peaks one period
away, and spin is too long to correlate sharply. Spin *starts* can't be used
either, since a loud pre-spin phase precedes the label. The spin-end test:
- **370 measurable edges,** median +3 s;
- **95% of cycles' medians within ±19 s;**
- **~20 s scatter within one cycle** (the labels' precision).

It also writes each cycle's recording date, seconds per phase and vibration
level per phase. Runs in ~30 s over 199 cycles.

### Stage 1b — Prepare (`prep/prepare.py`)

Per cycle:
- **Cut into segments** wherever the vibration has a gap over 1 s, or the
  label timeline a gap over 1.5 s. Nothing is interpolated across a gap.
- **Remove glitches:** raw samples beyond ±2.2 g are physically impossible
  (the sensor saturates at ±2.19 g). One cycle has one: `side.x = 71.5 g`.
- **Resample** each segment onto an exact 5 ms grid (200 Hz), by **cubic
  spline** per channel, with time measured from the segment start. This
  absorbs the 200–212 Hz jitter and the cycles that slow down.
- **Attach the state:** each grid row gets the five-state label of the label
  second it falls in (priority rule in "Labels").
- **Drop segments shorter than one window** (1,024 rows, 5.12 s).

**Output: one Parquet file per cycle,** `data/prepared/<cycle>.parquet`,
with columns `timestamp` (UTC), the 9 channels (float32), `state` and
`segment`. Plus `data/prepare_report.json`: segments, rows, and seconds per
state, for every cycle.

**Decisions:**
- **Resample in Stage 1b** rather than loosen the platform's interval
  tolerance. Volve showed the 5% default silently drops irregular states.
- **Parquet, not CSV,** for this intermediate stage. A 3-hour cycle is ~2.1M
  rows at 200 Hz, ~200 MB as CSV, ~40 GB for all 199. CSV is written only in
  Stage 2, for the files that are uploaded.

**Resampler (2026-09-29): cubic spline, replacing linear.** Stage 1c caught
linear interpolation dulling the higher frequencies (~20% less spread than
the raw). A tone test settled it: pure tones sampled at the cycles' real,
jittered timestamps, then resampled, measuring the share of amplitude kept.

| method | 10 Hz | 23 Hz (spin, 1400 rpm) | 46 Hz (2nd harmonic) | 70 Hz |
|---|---|---|---|---|
| linear | 98–99% | 94–96% | 82–84% | 64–68% |
| **cubic spline** | ~100% | **98–100%** | **97–99%** | **89–92%** |

- **The spline needs times relative to the segment start.** At epoch
  seconds (~1.7e9) it is numerically unstable: NaN or inflated power.
- **Output is clipped to ±2.2 g,** so spline overshoot near saturation stays
  in range.
- **No resampler can recover what isn't there:** in stretches recorded at
  ~120 Hz, content above ~60 Hz is lost.

**Rounding (2026-09-29):** the interpolated values are rounded to 1e-4 g. The
sensor's step is 0.0043 g, so the largest change, 5e-5 g, is ~1% of one
step. Unrounded interpolation makes every value unique: 78 MB per cycle
instead of 31 MB.

**Result (2026-09-29): done** (re-run with the spline), 199 cycles in 77 s, 372M rows, 4.9 GB.

| role | cycles | segments | fragments dropped (< 1 window) | kept / labelled seconds | fill h | wash h | spin h | drain h | heating h |
|---|---|---|---|---|---|---|---|---|---|
| library | 54 | 57 | 1 | 100.0% | 7.1 | 115.7 | 13.6 | 5.3 | 12.0 |
| validation | 21 | 25 | 0 | 100.0% | 2.5 | 42.7 | 4.6 | 1.8 | 3.7 |
| test | 18 | 23 | 0 | 100.0% | 2.4 | 34.9 | 4.9 | 1.7 | 4.2 |
| delivery | 106 | 122 | 45 | 99.3% | 14.3 | 184.0 | 24.0 | 11.2 | 26.4 |

- **17 cycles were cut at gaps** into 2–8 segments. The most is becken-flt
  `cold_cotton_30_11`, with 8.
- **The only real loss is the acknowledged becken-flt `cold_cotton_30_4`**
  (42% kept). Every other cycle kept ≥ 97.7%.

**Open, for Stage 2:**
- ~~**Timestamps:** does the platform accept sub-second timestamps at
  200 Hz?~~ **Answered 2026-09-29 by `fit/probe_timestamps.py`: yes, in all
  three formats tried.**
  - **The test:** one 1-trial optimization per format, pinned blueprint
    `blp_05h8jmsdcy8fra7f0rm5cerwsv`. Library: 3 files × 2 min (fill, wash,
    spin). Validation: 48,000 rows, a continuous wash → fill stretch.
  - **The result:** every format completed, feasible, and scored **all 46
    windows** with identical confusion matrices. Whole-second parsing would
    have given ~200 rows per timestamp, and the monotonic check would have
    rejected the windows.

    | format | example | optimization |
    |---|---|---|
    | fractional epoch seconds | `1687942396.240` | `opt_4sgba59cjm9hmbebnrmdwy6zf2` |
    | integer epoch ms | `1687942396240` | `opt_3bf2bgptv89ratajp7gp9bc7f9` |
    | ISO 8601 | `2023-06-28T08:53:16.240Z` | `opt_32pytyxrdb9h1tp5sp9xmmtw3w` |
  - **Proposed for Stage 2:** fractional epoch seconds with 3 decimals, the
    closest to Volve's numeric `DATE_TIME`. Pending the user's choice.
  - **Speed:** each trial took 40–80 s on these tiny files.
- **Macro-F1 counts absent classes as zero** (same probe). The model knew
  fill / wash / spin and the validation file had no spin:
  - **fill:** F1 0.25;
  - **wash:** F1 0.84;
  - **spin:** F1 0.

  So macro-F1 = (0.25 + 0.84 + 0) / 3 = 0.364. **Every validation and test
  role must contain all five states** (pooled over its files), or the score
  is dragged down by a state that wasn't there. Stage 3 must FAIL on a role
  missing a state. Ours pool all five (Stage 1c's hours table).
- **Upload volume (under discussion, 2026-09-29):** the real limit is
  platform time, not GB. At ~150 windows/min (Volve), whole files cost:

  | role | windows | ~time | CSV |
  |---|---|---|---|
  | validation | 38,907 | ~4.3 h per trial | 3.4 GB |
  | test | 33,787 | ~3.7 h, once | 2.9 GB |
  | delivery | 182,672 | ~20 h, once | 15.9 GB |

  - **Decided:** delivery is full (above); values are written with 4
    decimals (lossless for a 0.0043 g sensor step); 200 Hz stays the default
    (100 Hz only as a Stage 4b search option).
  - **Test: decided 2026-09-29, whole cycles.** All 18, ~33,800 windows,
    ~3.7 h once. It's the published number.
  - **Validation: decided 2026-09-29, six whole cycles** (see Stage 2).
    That is closer to "the customer gave us a few cycles" than excerpts.

### Stage 1c — Preflight: prepared files (`prep/preflight_prepared.py`)

| check | FAIL if | WARN if |
|---|---|---|
| file, columns | no prepared file; columns not `timestamp`, the 9 channels (float32), `state`, `segment` in that order | |
| grid | any step inside a segment not exactly 5 ms; segments out of order or touching | |
| values | any NaN or inf, or beyond ±2.2 g | |
| states | a state outside the five | |
| coverage | a segment shorter than 1,024 rows; a cycle that kept < 80% of its labelled seconds | |
| state match | any row whose state differs from the five-state rule for its label second (recomputed from the raw labels) | |
| resampling | a 23 Hz tone keeps < 80% | 23 Hz keeps < 95% or 46 Hz < 90%, worst of 3 one-minute stretches, using the cycle's own raw timestamps and Stage 1b's `resample()` |

**Result (2026-09-29): PASS,** no blocking failures, ~45 s.
- **Grid, values, states and state match:** all 199 cycles pass. That is 0
  mismatched states in 372M rows.
- **Coverage:** only the acknowledged becken-flt `cold_cotton_30_4` FAILs.
- **Resampling:** 191 pass, 8 warn.
  - **Why:** these cycles have a stretch recorded slowly (down to ~120 Hz)
    or with gaps, where 23 Hz keeps 84–86% and 46 Hz 65–85%.
  - **Which:** library `warm_intensive_40_0` and `warm_sterilization_2`;
    test `cold_cotton_0_8` and `hot_cotton_30_10`; 4 delivery cycles.
  - **Decision:** kept. The loss is in the raw data, and it's local to
    those stretches.

### Stage 2 — Role files (`prep/build_roles.py`)

- **Library: decided 2026-09-29, 400 windows per state** (2,000 windows,
  ~2.8 h of signal).
  - **Why balance:** wash is ~75% of the time, and a kNN vote over raw
    proportions would drift to "wash".
  - **Why 400:** k is searched up to ~15, and 400 gives ~7 examples per
    state from each of the 54 library cycles. It costs ~13 min of platform
    time per trial (Volve's rate, ~150 windows/min); the whole library would
    be 108,000 windows, ~12 h a trial.
  - **Available per state** (whole single-state windows):

    | state | windows | cycles with none |
    |---|---|---|
    | fill | 4,725 | 0 |
    | wash | 79,894 | 0 |
    | spin | 9,501 | 0 |
    | drain | 2,633 (the scarcest) | 1 |
    | heating | 8,318 | 16 (unheated programs) |
  - **How they're drawn:** each state's 400 are spread evenly across the
    library cycles that have it, evenly spaced within each cycle, not the
    first found. So heating comes from 38 cycles and drain from 53.
  - **Files (revised 2026-09-29): one file per state**, `<state>__library.csv`.
    - **Layout:** the state's chosen windows as continuous pieces (adjacent
      chosen windows form one piece; 1,990 pieces in all), in time order
      with real timestamps. Forward jumps occur only between pieces, at
      1,024-row boundaries.
    - **Why:** originally each piece was its own file (1,990 files). That
      config exceeded the platform's 1 MiB ConfigMap limit (Stage 4b), and
      the gap probe showed training files tolerate jumps between pieces.
    - **Provenance:** the manifest lists every piece's cycle, start and
      length.
    - **The draw** uses a fixed seed and informs no score.
    - **Kept after review (the user, 2026-09-29): 5 files, one per state.**
      - **Why it's safe:** in training files, any window containing a time
        jump fails the sampling-rate check and is silently skipped, while
        every jump-free window is used. That's per the platform team's code
        reading, and consistent with the gap probe (O2 = O3).
      - **So the platform trains on exactly the windows inside the pieces,**
        as the bar does.
      - **Considered and not taken:** one contiguous block per state per
        cycle (~250 files, closer to the skills' "one contiguous n-shot
        example" guidance). It would give overlapping steps (1024/512,
        512/256, 1024/256) more training windows (e.g. ~15 instead of 8 per
        8-window block), since with scattered pieces the extra overlapping
        windows cross seams and are dropped. The cost is one moment per
        state per cycle instead of 7–8 spread across it, and drain limits
        blocks to ~8 windows (its median longest stretch is 9).
      - **The trade-off to remember:** at overlapping steps, the library is
        effectively the same 2,000 windows as at step = window.
- **Validation and test:** continuous recordings labelled by a column, one
  file per continuous segment, because a window crossing a gap fails the
  whole file.
- **Validation for the search: decided 2026-09-29, six whole cycles** of the
  21. The rule was set before any scoring: within each family, spread across
  plausible settings, take each room twice, and use only cycles containing
  all five states.

  | # | cycle | why | windows |
  |---|---|---|---|
  | 1 | `cold_cotton_0_2` | cotton, unheated (0 °C), light load | 2,075 |
  | 2 | `hot_cotton_40_11` | cotton, everyday temperature, heaviest load | 2,415 |
  | 3 | `hot_cotton_60_2` | cotton, hottest wash (most heating) | 2,271 |
  | 4 | `cold_eco_40-60_11` | eco, heavy load (the eco cycle with every state) | 2,329 |
  | 5 | `warm_fast-45_40_2` | a short program | 504 |
  | 6 | `warm_mix_40_0` | an ordinary non-cotton program | 746 |

  - **Size:** 10,340 windows, ~15 h, ~70 min per trial at Volve's rate
    (all 21 would be ~4.3 h).
  - **Coverage:** two cycles per room, wash 0/40/60 °C, loads 0–11 kg,
    all five states in each.
  - **Avoided, for missing a state:** `warm_wool_40_6` (no spin),
    `hot_eco_40-60_4` (no drain), `cold_eco_40-60_4`, `warm_15-min_*` and
    `warm_sport_40_0` (no heating).
  - **Rooms:** the other programs exist only in the warm room, and cotton
    and eco only in the cold and hot rooms.
- **Confirmation (decided 2026-09-29):** after the search, score its top 1–3
  settings once on **all 21 validation cycles** before the test (as Volve's
  confirm stage did). That checks the six didn't favour a lucky setting, and
  the test stays untouched.
- **Delivery: decided 2026-09-29, as full as possible.** All 106 becken-flt
  cycles, whole: 122 continuous files (one per segment), ~260 h, ~183,000
  windows at 200 Hz. Unlabelled, with the labels held back in a sidecar.
  - **Why:** the scenario's delivery is "monitor the machine as it runs", so
    every cycle it ran is a real delivery case. Nothing is dropped to save
    platform time.
  - **Cost:** ~20 h of platform time at Volve's ~150 windows/min. It runs
    once and unattended, in batches (e.g. by program family), resumable, as
    Volve's Stage 6 did per well.
  - **Before starting it:** measure the real rate for 9 channels at 200 Hz on
    the test run (Stage 5), then size the batches.
  - **Includes** the acknowledged `cold_cotton_30_4` (42% coverage) and the
    exploration-scored `cold_cotton_40_2`, which is reported on its own line.

**Decision (2026-09-28): global normalisation.**
- **How:** z-score every file (library, validation, test, delivery) with
  the per-channel mean/std of becken's library cycles only, as the Volve
  example does. Stats go to `data/roles/zscore_stats.json`.
- **Why it's a risk:** the spin failure seen in exploration came from
  becken-flt's louder spin under becken's statistics. That is the delivery
  set, so it cannot show up before Stage 7, and it is reported as it comes
  out.
- **Per-unit scaling is a follow-up, not a fallback:** each unit z-scored
  with its own whole-data mean/std, label-free. Reported separately, and
  labelled as chosen after the delivery score.

**Result (2026-09-29): done** (the user's run), 257 s, 25.56 GB:

| role | files | cycles | windows |
|---|---|---|---|
| library | 1,990 | 54 | 2,000 (400 per state) |
| validation | 25 | 21 | 38,907 |
| test | 23 | 18 | 33,787 |
| delivery | 122 | 106 | 182,672 |

The z-score statistics come from 110,644,180 library rows. The library has
1,990 files rather than 2,000 because adjacent chosen windows share one.

### Stage 3 — Preflight: role files (`prep/preflight_roles.py`)

The platform's rules, checked before any upload. Per file:

| check | FAIL if |
|---|---|
| header | columns not `timestamp`, the 9 channels, and `label` for validation and test only |
| timeline | timestamps not epoch seconds with 3 decimals (checked on the raw text). Validation, test and delivery: any step not exactly 5 ms (a jump makes the platform reject the whole file). Library: any jump except a forward one where a manifest piece starts, on a 1,024-row boundary |
| values | any NaN or inf |
| windows | a library file not a whole number of 1,024-row windows; any file shorter than one window |
| labels (validation, test) | a blank label, or one outside the five states |
| one state (library) | any row of any piece not the file's state in that piece's cycle (checked piece by piece, via the manifest) |
| held-back labels (delivery) | no sidecar in `delivery_labels/`, or one whose timestamps don't match row for row |
| scaling | a spot check of 5 rows per file against the prepared data, z-scored with `zscore_stats.json`, off by more than 6e-5 |

Per role:

| check | FAIL if |
|---|---|
| manifest | files on disk and in `manifest.json` differ |
| separation | a cycle or setting group in two roles; a becken-flt cycle outside delivery |
| normalisation | `zscore_stats.json` not computed from exactly the 54 library cycles |
| balance | the library not exactly 400 windows per state |
| all five states | validation (all 21), validation (the 6 search cycles), test or delivery, pooled, lacks a state (the platform scores a missing state as F1 = 0) |

**Result (2026-09-29): PASS,** 2,160 files, ~110 s (reads all 25.6 GB).
**Re-run after the 5-file library (2026-09-29): PASS,** 175 files, ~110 s.
The library checks were negative-tested on a damaged copy, and each damage
was caught:
- a jump inside a piece → timeline and one state;
- wrongly scaled values → scaling;
- a spin file checked as wash → one state, on all 400 pieces.
- **Every check passes** on every file and role.
- **Hours per state, pooled:**

  | set | fill | wash | spin | drain | heating |
  |---|---|---|---|---|---|
  | validation (all 21) | 2.49 | 42.69 | 4.60 | 1.84 | 3.72 |
  | validation (6 search) | 0.70 | 11.00 | 1.28 | 0.52 | 1.21 |
  | test | 2.39 | 34.87 | 4.91 | 1.67 | 4.23 |
  | delivery | 14.30 | 184.02 | 23.96 | 11.18 | 26.44 |

**Bugs found while writing it (fixed):**
- **Format check:** it first read timestamps after the CSV reader had parsed
  them as numbers, dropping trailing zeros, so it now reads the raw text.
- **Missing checks:** the shared report counted every missing check as "not
  reached". That flagged role-specific checks (held-back labels, one state)
  on files they don't apply to. A missing check is now a FAIL only for a
  result that already failed something, which is what "stopped early" means.
  Stages 1a and 1c print the same counts as before.

### Stage 4a — The bar (`fit/baseline_bar.py`)

**What it does:** simple baselines on exactly the role files and split, scored
the way the platform scores.
- **Library:** the 2,000 library windows.
- **Validation windows:** 1,024 rows at step 1,024 from each file's start,
  labelled by the last row (`last_record`).
- **Score:** macro-F1 over all five states, with an absent state counting 0.
- **Features:**
  - **level:** per-channel log RMS, what a threshold uses;
  - **fft:** 16 log bands per channel, each window standardised;
  - **level+fft:** both.
- **Classifier:** kNN, l1 or l2, k in 1, 5, 15, 51.
- **Selection:** the best setting is chosen on the 6 search cycles only, then
  reported on all 21. Test is untouched until Stage 5.

**Result (2026-09-29), ~90 s:**

| features | best setting | macro-F1, 6 search cycles | all 21 | fill | wash | spin | drain | heating (all 21) |
|---|---|---|---|---|---|---|---|---|
| **level** (threshold-like) | k = 5, l1 | **0.5455** | **0.5473** | 0.94 | 0.56 | 0.60 | 0.44 | 0.19 |
| level+fft | k = 5, l1 | 0.5287 | 0.5250 | 0.75 | 0.60 | 0.60 | 0.48 | 0.20 |
| fft | k = 5, l1 | 0.4697 | 0.4539 | 0.49 | 0.55 | 0.56 | 0.47 | 0.20 |

**The bar is level + kNN, k = 5, l1: 0.5455 on the search cycles.** It
agrees on all 21 (0.5473), so the six are not unusually easy.

**Where five states are hard** (the bar's confusions on the 6 search cycles):
- **wash → heating: 3,738 of 7,726 wash windows.** Wash's quiet pauses look
  like heating (Stage 1a found heating vibration matches the pauses), and a
  balanced library gives heating an equal vote. Heating F1 is 0.19–0.20 for
  every baseline.
- **wash → spin (479) and → drain (394):** most likely the loud pre-spin
  phase, labelled wash as given (Stage 1a, decision 3).
- **spin → drain (195):** the spin-while-draining edges.
- **fill is easy for a threshold** (F1 0.94), unlike the 3-state exploration
  (0.69), where fill was only compared with wash.

**The success criterion was changed after seeing the bar (the user's
decision, 2026-09-29).**
- **Originally:** fixed before the pipeline, the rule was "beat the bar's
  macro-F1 by ≥ 0.05 **and** spin F1 ≥ 0.90".
- **Why the spin rule was changed:** it dates from the 3-state exploration,
  where a threshold reached spin 0.95. With five states the bar itself
  scores spin 0.58 on the 6 search cycles (0.60 on all 21). Here "≥ 0.90"
  would demand far more than "not worse on spin", which is what it meant.
- **Now:** Omega's spin F1 must be **at least the bar's on the same cycles**
  (≥ 0.58 on the search cycles).
- **Unchanged:** the macro-F1 rule, ≥ 0.05 above the bar (≥ 0.5955).
- **The README states this change,** and every reported result is judged
  against the new rule.

**Four-state bar (2026-09-29, the user's run), window 1024 / step 1024:**
best is level k=15 l2, macro-F1 **0.7285** on the 6 search cycles (fill
0.95, wash 0.94, spin 0.61, drain 0.42), 0.7405 on all 21. That's in line
with the 0.72 estimated after the fact. It ties k=51 l2 (0.7285); the first
in grid order is kept. Level beats FFT (0.6177) and level+FFT (0.6989), as
on five states.
- **So Omega needs macro-F1 ≥ 0.7785 and spin F1 ≥ 0.61** at 1024 / 1024.
- **All 9 four-state bars** (the user's loop, done by 21:56). Level features
  win at every pair, mostly with k 51:

  | window / step | best | macro-F1, 6 search cycles | fill | wash | spin | drain | all 21 | Omega needs |
  |---|---|---|---|---|---|---|---|---|
  | 256 / 256 | level k 51 l1 | 0.7302 | 0.93 | 0.94 | 0.63 | 0.43 | 0.7374 | 0.7802 |
  | 256 / 512 | level k 51 l1 | 0.7292 | 0.92 | 0.94 | 0.64 | 0.42 | 0.7335 | 0.7792 |
  | 256 / 1024 | level k 15 l2 | 0.7236 | 0.92 | 0.94 | 0.62 | 0.42 | 0.7295 | 0.7736 |
  | 512 / 256 | level k 51 l1 | 0.7379 | 0.95 | 0.94 | 0.64 | 0.42 | 0.7485 | 0.7879 |
  | 512 / 512 | level k 51 l2 | 0.7406 | 0.95 | 0.94 | 0.64 | 0.43 | 0.7511 | 0.7906 |
  | 512 / 1024 | level k 51 l2 | 0.7360 | 0.96 | 0.94 | 0.65 | 0.39 | 0.7481 | 0.7860 |
  | 1024 / 256 | level k 51 l2 | 0.7291 | 0.94 | 0.94 | 0.65 | 0.39 | 0.7415 | 0.7791 |
  | 1024 / 512 | level k 51 l2 | 0.7289 | 0.94 | 0.94 | 0.65 | 0.39 | 0.7419 | 0.7789 |
  | 1024 / 1024 | level k 15 l2 | 0.7285 | 0.95 | 0.94 | 0.61 | 0.42 | 0.7405 | 0.7785 |
- **Where four states are hard** (confusion, 6 search cycles):
  - wash → spin 434 and wash → drain 439, of 8,572 wash windows: the loud
    pre-spin phase and the soft spin edges (labels kept as given);
  - spin → drain 185 and spin → wash 92, of 900;
  - drain → spin 80, of 369.
  So spin and drain are what Omega has to improve; fill and wash are
  already ≥ 0.94.

### Stage 4b — Omega on the Optimize API (`fit/optimize.py`)

- **Search:** window (1024 at 200 Hz = 5.1 s; 512; 1024 at 100 Hz = 10.2 s),
  step, k and metric, on the globally normalised role files. Score on the
  six search-validation cycles; then confirm the top 1–3 on all 21 (Stage 2).
- **Rate check:** measure the real platform rate (windows/min) on the first
  trial, and revise every time estimate here from it.
- **Design (the user, 2026-09-29): the Optimize step is platform-only.**
  - **Why:** a user bringing their own data may have no sensible baseline.
  - **`fit/optimize.py`** runs optimizations and saves every trial's scores.
    It never reads a baseline.
  - **Comparing with the bar is a separate, optional step:**
    `fit/compare_bar.py` pairs each trial with the bar computed on the same
    window/step (`fit/baseline_bar.py --window W --step S` →
    `fit/out/bar_validation_w<W>_s<S>.json`) and applies the criterion.
- **Search: an exhaustive pool, sampled by random search** (the user,
  2026-09-29). The pool is the product of every value below. Setting
  `max_trials` below its size makes the platform sample configurations
  instead of running them all. Volve ran a 40-trial random search this way
  (`opt_64nm5jtt918p6tkvwa5m8vsd0g`). The platform's sampler can't be seeded,
  so `fit/optimize.py` records what was drawn and flags duplicates.

  | parameter | values (`--pool full`) | count |
  |---|---|---|
  | `window_size` | 256, 512, 1024 (1.3 / 2.6 / 5.1 s) | 3 |
  | `step_size` | 256, 512, 1024 | 3 |
  | `k_neighbors` | 1, 3, 5, 7, 9, 15, 21, 31 | 8 |
  | `metric` | `l1`, `cosine` (both used on the platform by Volve; `l2`'s name unconfirmed) | 2 |
  | `weights` | `uniform`, `distance` | 2 |

  - **Size:** 288 configurations. `max_trials` is set once the measured
    rate is in; 24 is the working proposal.
  - **Cost per trial** follows the step: validation windows are ~10,340 at
    step 1024, ~20,680 at 512 and ~41,360 at 256 (1× / 2× / 4×). Step 256
    is open: keep it, or drop it to cut the budget.
  - **Step > window** skips the records between windows. It needs
    `--allow-gaps`, and the user accepts it (e.g. 512 / 1024):
    - validation is scored on a regular sample;
    - each 1,024-row library file gives its first window only;
    - a deployed agent would monitor part of the running time.

    An earlier draft wrongly said step "doesn't change what's learned". It
    does, because it decides how many library windows are cut.
- **The bar per window/step** (all computed 2026-09-29; each is the best
  level/fft/level+fft kNN setting on the 6 search cycles):

  | window / step | bar | macro-F1, 6 search | spin F1, 6 search | macro-F1, all 21 |
  |---|---|---|---|---|
  | 256 / 256 | level, k 15, l2 | 0.5472 | 0.60 | 0.5501 |
  | 256 / 512 | level, k 15, l2 | 0.5511 | 0.61 | 0.5542 |
  | 256 / 1024 | level, k 15, l2 | 0.5526 | 0.62 | 0.5527 |
  | 512 / 256 | level, k 15, l2 | 0.5525 | 0.59 | 0.5541 |
  | 512 / 512 | level, k 15, l2 | 0.5550 | 0.62 | 0.5544 |
  | 512 / 1024 | level, k 5, l2 | 0.5557 | 0.59 | 0.5544 |
  | 1024 / 512 | level, k 5, l1 | 0.5439 | 0.58 | 0.5458 |
  | 1024 / 1024 | level, k 5, l1 | 0.5455 | 0.58 | 0.5473 |

  - **The threshold-like baseline sits at 0.544–0.556 for every
    window/step.** So Omega needs about 0.60 to pass on any of them.
  - **(1024 / 256 isn't computed:** no search-1 trial drew it.)
  - **Fixed while computing these:** the bar's kNN compared blocks of 1,000
    windows with the whole library at once, ~10 GB at step 256. The blocks
    are now sized to ~250 MB, and l2 uses the dot-product form. All 24
    settings at 1024 / 1024 were re-checked: identical scores, 57 s instead
    of 86 s.
- **Both runs stuck: the ConfigMap limit** (found 2026-09-29 by the user, via
  Grafana and a Claude triage of the platform code).
  - **The cause:** the platform puts each optimization's whole config into
    one Kubernetes ConfigMap (max 1 MiB). It stores the config pretty-printed
    and repeats every file list. Ours was 1,362,142 bytes: 1,990 training
    examples at 636 bytes each.
  - **The effect:** JOS retries forever while the API shows `running` /
    `pending`, with no error. Roughly 1,500 training examples fit; 695 went
    through earlier that day.
  - **Short term:** cancel both runs and re-run with fewer, larger library
    files (option A).
  - **Platform fixes** (compact JSON, a size check, failing on 4xx, not
    duplicating file lists) are with the owning teams.
- **Gap probe (`fit/probe_gaps.py`, 2026-09-29): how the APIs treat time gaps.**
  The setup was tiny files, forward jumps of 61 s to 2 h between 1,024-row
  pieces, and window 1024:

  | run | library | validation | step | tolerance | result |
  |---|---|---|---|---|---|
  | O0 | continuous | continuous | 1024 | default | 46/46 scored |
  | O1 | seams | continuous | 1024 | default | 46/46 |
  | O2 | seams | continuous | 512 | default | 92/92 |
  | O3 | seams | continuous | 512 | 10.0 / 10.0 | 92/92, same score as O2 |
  | O4 | continuous | seams | 1024 | default | 12/12 |
  | O5 | continuous | seams | 512 | default | **trial failed** |
  | O6 | continuous | seams | 512 | 10.0 / 10.0 | **trial failed** |
  | E4 | Evals API, O4's promoted blueprint, seamed file | | 1024 | | 12/12 |

  - **Training files tolerate seams,** even where training windows cross one
    (O2, O3). **Option A is safe for the library.**
  - **The platform team's code reading (Claude triage, 2026-09-29, at
    8910c38):**
    - **What training does:** `SamplingRateValidationNode` rejects
      seam-crossing windows in both legs. Training then silently drops
      them: the optimizer turns each file's label into a per-file constant,
      so the encoder's ground-truth check never fires, and the fitter skips
      invalid windows without counting them. So O2 and O3 trained on
      within-piece windows only, which is why they score identically.
      `fit/baseline_bar.py` cuts library windows within pieces too, so the
      bar and the platform see the same library at every step.
    - **What eval does:** `OmegaEncoderNode` errors on any invalid window
      that has a ground-truth column. This was added deliberately (#6441),
      since mislabelled invalid windows would skew scores. The allowed
      outlier fraction defaults to 0, so one gap is enough.
    - **Why 10.0 didn't help:** `sample_rate_interval_tolerance` is a
      fraction of the window's *mean* interval. One big jump in a 1,024-row
      window deviates by ~1,022×, so passing needs roughly the window size
      (~5,000); that's their calculation, untested. `max_temporal_gap` was
      never the check that fired. The search space can't pass `null` or
      "warn".
    - **Their suggested fix:** keep real labels on invalid windows, skip
      them in the scoring sink with a `skipped_invalid_windows` count, and
      document that the tolerance is relative. Owners: Les Sosnowski (eval
      strictness, scoring sink) and Sylvain Lebresne (validation checks,
      the new scoring layer, PLDEV-2113). Offered for PLDEV-2161.
    - **Their verdict:** option A (jumps on window boundaries) is the right
      workaround until then.
  - **Scored files (validation, test; Optimize and Evals alike) tolerate
    seams only on window boundaries.** A window across a seam fails the
    whole trial: *"eval-mode test data must not contain windows a validation
    node rejected"* (O5). So our validation and test files stay one
    continuous file per segment.
  - **Loosening `sample_rate_interval_tolerance` and `max_temporal_gap` to
    10.0 didn't rescue O6.** The jumps are 61 s to 2 h inside a 5.12 s
    window, far beyond ×10. Whether a huge value (e.g. 1e9) would work is
    untested, and not needed here.
  - **Left on dev:** a throwaway blueprint,
    `larco-probe-gaps-o4-20260929t222033z`. Results are in
    `fit/out/probe_gaps_<blueprint>.json`.
- **Stuck runs cancelled (2026-09-29, 15:14–15:15):**
  `opt_6k62qemyja97qt4hh4xnp0s1be` and `opt_5cvewxg15j9p589a694znpzgab`.
  - **A poller bug:** the platform reports `cancelled` (two l's), which the
    pollers didn't treat as finished (they only knew `canceled`). The two
    old background pollers kept looping until stopped by hand; both
    spellings count now (`fit/atai.py`).
  - **The measurement trial restarted at 17:25** with the 5-file library:
    `opt_03x6ye0jcz9kt98nvspajmz04p` (job `job_3s6yktt09a8khavyvdvm32jkpe`).
    Grafana showed it created → Running on an NVIDIA L40S within 21 s, with
    no ConfigMap or dispatch errors; dev's stuck-jobs count was back to 0.
    **The runner fit its k-NN on exactly 2,000 examples:** our library at
    1024 / 1024, 400 per state. That's independent confirmation that no
    training window was lost or added at the piece seams. It detected the
    input rate as exactly 200.000 Hz.
  - **Result (17:39, 12.8 min): completed, all 10,340 validation windows
    scored.**

    | | macro-F1 | fill | wash | spin | drain | heating |
    |---|---|---|---|---|---|---|
    | Omega, w 1024, step 1024, k 5, l1, uniform | 0.5361 | 0.91 | 0.48 | 0.57 | 0.48 | 0.24 |
    | bar, level kNN, k 5, l1 | 0.5455 | 0.95 | 0.55 | 0.58 | 0.42 | 0.22 |

    - **It fails the criterion:** −0.009 macro-F1 vs the bar (it needs
      +0.05), and spin 0.57 < 0.58.
    - **The difference is wash → heating:** 4,333 of 7,726 wash windows
      (bar: 3,738). Omega catches more heating (701 vs 582 of 847) but
      mislabels more wash. Fill, spin and drain are nearly identical.
    - **It's one setting of 288,** and it's the bar's best setting (k 5, l1),
      not Omega's. The platform's default for embeddings is `cosine` /
      `distance`; the search covers that.
  - **The measured platform rate: 964 windows/min** (library + validation
    windows per trial minute). *Superseded by search 1: trial time is
    ≈ 11 min fixed + 0.09 min per 1,000 windows, so this rate doesn't
    scale.* Per trial: step 1024 ≈ 12,340 windows ≈
    13 min; step 512 ≈ 24,700 ≈ 26 min; step 256 ≈ 49,000 ≈ 51 min. So a
    16-trial random search over the full pool is ≈ 7 h if trials run one
    after another; whether the platform runs them in parallel is unknown.
- **Search 1 re-run: `opt_59jf7w1yje8krt5wqr5z0zzex6`** (created 17:40 by the
  user, `--pool full --allow-gaps --max-trials 16`).
  - **16 distinct configurations:** 1024/1024 ×1, 512/1024 ×1, 256/1024
    ×2, 1024/512 ×2, 256/512 ×3, 512/256 ×3, 256/256 ×4. `cosine` in 9,
    k from 1 to 31.
  - **Cost:** 7 trials at step 256, ≈ 9 h in all if run one after another.
    Every pair drawn already has its bar.
  - **A log-count fix:** `fit/optimize.py` now counts library windows within
    pieces, as the platform keeps them. Before, overlapping steps were
    overstated (e.g. 1024/512: 3,995 → 2,010 kept). This run started with
    the old count, so its per-pair lines and its rate figure overstate the
    library slightly at overlapping steps; the platform's work is
    unaffected.
  - **Uploads:** it re-uploaded all 11 files (1,090 MB, 42 s), since Stage
    2's rebuild gave the validation files new timestamps and the cache is
    keyed by path, size and mtime.
  - **Trials run one after another** in one job: one completes every
    12–16 min, with `running` never above 0 at a poll. The step-256 trials
    (~49,000 windows at window 256 or 512) also took ~13–16 min, not the
    51 min the 1,024-row rate predicted: shorter windows cost less per
    window. So search 1 should take ≈ 3.5–4 h, not 9 h.
  - **Interim, 9 of 16 done (19:42): none passes.** Margin = macro-F1
    minus the bar at the same window/step.

    | # | w / step | k | metric | weights | macro-F1 | fill | wash | spin | drain | heating | margin |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | 5 | 512/256 | 15 | cosine | uniform | **0.5534** | 0.92 | 0.47 | 0.63 | 0.52 | 0.23 | +0.0009 |
    | 6 | 1024/512 | 7 | cosine | uniform | 0.5439 | 0.92 | 0.48 | 0.59 | 0.49 | 0.24 | −0.0000 |
    | 9 | 1024/512 | 21 | cosine | uniform | 0.5433 | 0.90 | 0.45 | 0.63 | 0.50 | 0.24 | −0.0006 |
    | 3 | 256/256 | 31 | l1 | uniform | 0.5430 | 0.91 | 0.44 | 0.65 | 0.51 | 0.21 | −0.0042 |
    | 8 | 256/1024 | 31 | cosine | uniform | 0.5456 | 0.92 | 0.46 | 0.66 | 0.49 | 0.20 | −0.0070 |
    | 2 | 256/512 | 21 | cosine | distance | 0.5395 | 0.90 | 0.45 | 0.65 | 0.50 | 0.20 | −0.0116 |
    | 7 | 256/512 | 5 | cosine | uniform | 0.5368 | 0.88 | 0.51 | 0.60 | 0.50 | 0.20 | −0.0143 |
    | 1 | 1024/1024 | 1 | l1 | distance | 0.5185 | 0.89 | 0.51 | 0.49 | 0.47 | 0.23 | −0.0270 |
    | 4 | 256/1024 | 1 | l1 | uniform | 0.5105 | 0.83 | 0.57 | 0.49 | 0.47 | 0.19 | −0.0421 |

    - **At best Omega ties the bar**, far from +0.05. Spin now beats the
      bar's in most trials (0.59–0.66 vs 0.58–0.62).
    - **Heating stays at 0.19–0.24 and wash at 0.44–0.51 in every trial.**
      So the limit looks like the wash/heating confusion, not the kNN
      settings.
    - **Larger k helps:** k 1 is the worst in both trials that drew it.
  - **Final, 16 of 16 (completed 21:25): no trial passes.** The 7 later
    trials changed nothing. The best is still #5 (512/256, k 15, cosine,
    uniform) at 0.5534, +0.0009 against its bar. Next: #16 (512/256, k 15,
    l1) 0.5496; #14 (256/256, k 9, l1, distance) 0.5494; #11 (256/256,
    k 15, l1) 0.5482. Heating stayed at 0.19–0.24 in all 16. Results are in
    `fit/out/five_states/optimize_opt_59jf7w1yje8krt5wqr5z0zzex6.json`.
  - **Trial time is mostly a fixed cost.** The per-trial minutes the run
    printed (up to 224) and its "278 windows/min" were wrong: every trial is
    created with the optimization, and it measured from creation. Measured
    instead from the previous trial's completion, trials took 12.1–16.0 min,
    3.7 h in all. That fits **≈ 11.1 min + 0.09 min per 1,000 windows**
    (library + validation). So step 256 costs only ~3 min more than step
    1024, and the "964 windows/min" from the measurement trial (one
    trial, all fixed cost) doesn't scale. `fit/optimize.py` now reports trial
    time this way.
- **Why wash and heating are confused (`fit/diagnose_wash_heating.py`,
  validation cycles only, w 1024 / step 1024).**
  - **Heating isn't a separate phase; it's the heater switching on during
    wash.** In 17 of 21 validation cycles it comes as bursts inside wash
    (wash → heating → wash). Some cycles have up to 19 bursts of 1–90 s,
    like a thermostat; others have one long run of 5–35 min.
  - **Vibration barely shows it.** Classifying wash vs heating only, with
    the bar's features and training classes balanced, a held-out cycle
    scores balanced accuracy **0.62** pooled (0.5 = chance; 0.54–0.77 per
    cycle). No feature moves by more than 0.5 wash SD; the most consistent
    one (a high band on the top sensor) has the same sign in 94% of cycles.
  - **So heating caps macro-F1 for every model.** It is one-fifth of the
    score, and vibration can't recover it. Merging heating into wash in
    the bars' predictions after the fact gives 4-state macro-F1 **0.72–0.74**
    (fill 0.92–0.96, wash 0.94–0.95, spin 0.58–0.62, drain 0.41–0.44).
    That is indicative only, not a retrained bar.
  - **Options (the user decides):**
    - A: 4 states (heating folded into wash): a drum-motion task that
      vibration can do.
    - B: keep 5 states and add a power channel: the heater is visible
      there, but that goes against the vibration-only decision.
    - C: keep things as they are and accept the cap.
  - **Decision (user, 2026-09-29): A, four states.** Implemented in
    `prep/states.py`. Stages 1b, 1c and 2 were rerun to check the scripts:
    1c PASS with the same WARNs; the library is 4 files (1,600 pieces,
    400 windows per state), and validation / test / delivery have the same
    windows as before. Five-state bars, results and the diagnostic moved to
    `fit/out/five_states/`; `fit/compare_bar.py` skips runs scored on
    other states. **The user's rerun:** Stages 1a, 1b and 1c match the
    docs. 1a: the same WARN counts, the one acknowledged FAIL, and becken
    wash at 0.0065 g (83% of the time) with heating folded in. 1b: 372,231,085
    rows, 4.86 GB. 1c: PASS, with library hours fill 7.1 / wash 127.7 /
    spin 13.6 / drain 5.3. 2: 4 library files (1,600 pieces, 400 windows
    per state), validation / test / delivery unchanged, 25.45 GB. 3: PASS on
    174 files and all four states in every scored set.
  - **Four-state search: `opt_4qggkyx4z487m86da5emyt7txw`** (created 20:58 by
    the user, `--pool full --allow-gaps --max-trials 16`). The first attempt
    died on a failed DNS lookup during uploads, so `upload_file` now retries
    network and gateway errors, and the upload cache is saved per file. The
    rerun hit the same lookup failure on 2 of 8 parallel uploads (probably
    the uplink saturated, so DNS to the router timed out), retried, and
    finished: 10 files, 1,054 MB, 211 s. If it recurs, use a lower
    `--upload-jobs`.
  - **It ran right after search 1:** its trial #1 finished 14 min after
    search 1 ended. So trials queue one at a time across optimizations too.
  - **First 2 trials (21:51): neither passes.**
    - #2, 512/1024, k 15, l1, distance: **0.7466** (fill 0.93, wash 0.95,
      spin 0.62, drain 0.49). +0.0106 against its bar; it needs 0.7860.
    - #1, 512/256, k 5, cosine, uniform: 0.7241 (0.87 / 0.94 / 0.58 /
      0.51). −0.0138.
    - **Drain is over-predicted:** recall 0.71–0.75 but precision
      0.37–0.39. The library gives drain a quarter of the votes, but it
      is 3% of the scored windows. That's the same mechanism that made
      heating swallow wash. Omega's drain F1 (0.49–0.51) already beats the
      bars' (0.39–0.43); spin (0.58–0.62) is at or below them.
  - **Final, 16 of 16 (completed 00:58 on 2026-09-30): no trial passes.**
    From `fit/compare_bar.py`:

    | # | w / step | k | metric | weights | macro-F1 | fill | wash | spin | drain | bar (spin) | margin | spin rule |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|
    | 11 | 256/1024 | 21 | cosine | distance | 0.7537 | 0.93 | 0.95 | 0.65 | 0.49 | 0.7236 (0.62) | **+0.0301** | pass |
    | 9 | 256/1024 | 15 | l1 | uniform | 0.7438 | 0.91 | 0.95 | 0.63 | 0.48 | 0.7236 (0.62) | +0.0202 | pass |
    | 13 | 256/512 | 21 | cosine | distance | 0.7445 | 0.89 | 0.95 | 0.65 | 0.49 | 0.7292 (0.64) | +0.0153 | pass |
    | 16 | 1024/256 | 21 | cosine | uniform | 0.7436 | 0.90 | 0.94 | 0.63 | 0.50 | 0.7291 (0.65) | +0.0145 | fail |
    | 7 | 512/512 | 31 | cosine | uniform | **0.7547** | 0.91 | 0.95 | 0.65 | 0.51 | 0.7406 (0.64) | +0.0141 | pass |
    | 12 | 256/256 | 9 | l1 | uniform | 0.7408 | 0.89 | 0.95 | 0.62 | 0.51 | 0.7302 (0.63) | +0.0106 | fail |
    | 2 | 512/1024 | 15 | l1 | distance | 0.7466 | 0.93 | 0.95 | 0.62 | 0.49 | 0.7360 (0.65) | +0.0106 | fail |
    | 3 | 256/512 | 9 | cosine | distance | 0.7370 | 0.88 | 0.94 | 0.63 | 0.50 | 0.7292 (0.64) | +0.0078 | fail |
    | 10 | 512/256 | 9 | cosine | distance | 0.7359 | 0.89 | 0.94 | 0.61 | 0.51 | 0.7379 (0.64) | −0.0020 | fail |
    | 6 | 512/256 | 9 | cosine | uniform | 0.7357 | 0.89 | 0.94 | 0.61 | 0.51 | 0.7379 (0.64) | −0.0022 | fail |
    | 14 | 512/512 | 5 | cosine | uniform | 0.7280 | 0.88 | 0.94 | 0.59 | 0.51 | 0.7406 (0.64) | −0.0126 | fail |
    | 1 | 512/256 | 5 | cosine | uniform | 0.7241 | 0.87 | 0.94 | 0.58 | 0.51 | 0.7379 (0.64) | −0.0138 | fail |
    | 15 | 256/256 | 3 | l1 | distance | 0.7064 | 0.83 | 0.93 | 0.57 | 0.50 | 0.7302 (0.63) | −0.0238 | fail |
    | 5 | 512/1024 | 3 | l1 | uniform | 0.7081 | 0.88 | 0.94 | 0.54 | 0.48 | 0.7360 (0.65) | −0.0279 | fail |
    | 4 | 256/256 | 3 | cosine | distance | 0.7023 | 0.82 | 0.93 | 0.57 | 0.49 | 0.7302 (0.63) | −0.0279 | fail |
    | 8 | 1024/1024 | 3 | l1 | uniform | 0.6996 | 0.86 | 0.93 | 0.54 | 0.47 | 0.7285 (0.61) | −0.0289 | fail |

    - **Omega beats its bar in 8 of 16 trials, by at most +0.030.** It
      needs +0.05. Four trials also meet the spin rule (#11, #9, #13, #7).
    - **Where Omega gains:** drain, 0.48–0.51 against the bars' 0.39–0.43.
      Where it loses: fill, 0.82–0.93 against 0.92–0.96. Wash is equal.
      Spin is about equal at k ≥ 15 and worse below.
    - **k is the strongest knob:** every k ≥ 15 trial beats its bar, and
      every k ≤ 5 trial is below it. The best trials use the largest k in
      the pool (21, 31). The bars were best at k 51, which the pool
      doesn't reach.
    - **The run's printed minutes and "217 windows/min" are wrong.** Its
      poller started before the timing fix. Measured from completion to
      completion, trials took 12–15 min.
    - Results: `fit/out/optimize_opt_4qggkyx4z487m86da5emyt7txw.json`,
      `fit/out/compare_bar.json`.
  - **Decision (user, 2026-09-30): carry trial #7 forward** (window 512,
    step 512, k 31, cosine, uniform: the highest score, 0.7547, +0.0141
    against its bar, meeting the spin rule) into Stages 4c–7. A search at
    k 51–101 was proposed and declined as too large.
    - **It does not meet the success criterion** (+0.05 over the bar), and
      the docs say so. The criterion is unchanged; every later stage is
      still judged against it, so the example reports what Omega achieved,
      not a pass.
    - Why #7 rather than #11 (the widest margin, +0.0301): #11's margin comes
      from the lowest bar (256 / 1024), while #7 has the higher score.
      Both meet the spin rule.
- **Library rebuilt as 5 files (2026-09-29).** Stage 2 wrote 5 files, Stage 3
  passed, and `fit/baseline_bar.py` cuts library windows within pieces
  (the same windows as the old files, so the 8 bars stand). The old 1,990
  files were removed by the rebuild; the stale Stage 0–1c logs were deleted
  from `data/`. **Next:** cancel the two stuck runs; re-run the measurement
  trial, then search 1.
- **Measurement trial:** `opt_6k62qemyja97qt4hh4xnp0s1be`, created 12:03 on
  2026-09-29 by the user's `--background` run. Uploads: 1,996 files,
  1,090 MB, 4.4 min. **Still `pending` at 121 min** (14:04), with no error
  and no progress fields in the API.
- **The time estimate was wrong.** "~150 windows/min" came from Volve, whose
  windows were 16–128 samples; ours are 1,024, and the encoder's cost grows
  with window length. For scale, the local encoder-only agent on the Mac did
  ~190 nine-channel 1,024-sample windows/min, which would make this trial
  ~65 min. The real platform rate is still unknown.
- **Search 1:** `opt_5cvewxg15j9p589a694znpzgab`, created 13:30 by the user:
  `--pool full --allow-gaps --max-trials 16`.
  - **16 distinct configurations, no duplicates.** The window/step pairs
    drawn: 1024/1024 ×3, 512/512 ×3, 1024/512 ×3, 256/1024 ×2, 512/256 ×2,
    512/1024, 256/512, 256/256. None drew 1024/256.
  - **All 16 were still pending at 14:04,** most likely queued behind the
    measurement trial or other work on dev.

### Stage 4c — Confirm on all 21 validation cycles (`fit/optimize.py --validation all`)

Built into `fit/optimize.py` (`--validation all`) rather than a new script. It
scores one setting, #7, on all 25 validation files (21 cycles, 77,823 windows at
512 / 512), trained on the same 4 library files. `fit/compare_bar.py` compares it
with the bar's all-21 score at the same window/step: level k 51 l2, **0.7511**
(fill 0.93, wash 0.95, spin 0.66, drain 0.46). So the criterion there is
macro-F1 ≥ 0.8011 and spin ≥ 0.66. The 19 non-search files (2.56 GB) are
uploaded first. Expected time: ≈ 11 + 0.09 × 81 ≈ 18 min for the trial.

**Result (2026-09-30): `opt_1dztapszen8n1a89jj08fcfwrr`, 44 min** (the timing
model predicted 18; with 25 validation files the fixed cost is evidently
larger). All 77,823 windows scored.

| | windows | Omega macro-F1 | fill | wash | spin | drain | bar | margin |
|---|---|---|---|---|---|---|---|---|
| 6 search cycles (Stage 4b, #7) | 20,682 | 0.7547 | 0.91 | 0.95 | 0.65 | 0.51 | 0.7406 | +0.0141 |
| **all 21** | 77,823 | **0.7562** | 0.90 | 0.95 | 0.64 | 0.53 | 0.7511 | **+0.0051** |
| the other 15 (all 21 − the 6) | 57,141 | 0.7566 | 0.89 | 0.96 | 0.64 | 0.54 | 0.7553 | +0.0013 |

- **Omega's score held:** 0.7547 on the 6 cycles it was chosen on, 0.7566 on
  the 15 it wasn't. So the choice wasn't selection luck.
- **But the bar rose more** (0.7406 → 0.7553): the 6 search cycles are
  harder for the level baseline than the rest. On the 15 unseen cycles
  Omega and the bar **tie** (+0.0013). On all 21: +0.0051, **fail** against
  ≥ 0.8011, and spin 0.64 < the bar's 0.66.
- **The 15-cycle numbers** come from subtracting the search trial's 6-cycle
  confusion matrix from this run's 21-cycle one (same setting, same
  library, same windows; no cell went negative), and likewise for the bar.
- **Where they differ (15 cycles):**
  - drain: Omega 0.54 against 0.47 (1,188 wash windows called drain,
    against 1,888);
  - fill: Omega 0.89 against 0.93 (257 fill windows called wash,
    against 17);
  - spin: Omega 0.64 against 0.67.
  Fill and spin are loudness differences, which the level baseline reads
  directly. A hypothesis, not checked: the embeddings keep less of the
  absolute level than the bar's RMS feature.

Score the search's top 1–3 settings once on all 21 validation cycles (the 6
search cycles plus the other 15), before the test. That checks the six didn't
favour a lucky setting, and the test stays untouched (see Stage 2,
"Confirmation").

### Stage 5 — Test once (`fit/test.py`)

Promote the chosen trial, then evaluate it once on the test groups with the
Evals API. Report per cycle, per family and per state, not only pooled.

**Built (2026-09-30), not run.**
- **The model:** the Stage 4c trial, promoted as
  `osm-larco-w512-s512-cosine-k31-uniform`. It's the same setting and the same
  library as the search trial #7, so what's tested is exactly what was
  confirmed.
- **Two evals:** all 23 test files, and `cold_cotton_40_2`'s files alone.
  "Without it" is the difference, because the Evals report is pooled.
- **Per cycle and per family** are computed for the bar only
  (`fit/baseline_bar.py --test`, local). For Omega, that would need one eval
  per cycle or the predictions artifact, whose format hasn't been checked.
  That's a follow-up if wanted.
- **The bar on test** uses its validation-chosen setting (level k 51 l2 at
  512 / 512), never re-chosen.
- **Judged against the same criterion,** reported, not a pass/fail gate:
  macro-F1 ≥ the bar's + 0.05, and spin ≥ the bar's.

**Result (2026-09-30), the one-shot test.** Evals `evl_56qkrrh93j8xmsasanhpzfks74`
(all 23 files, 39 min) and `evl_66537rjytd81av44t98yfnb6w7` (cold_cotton_40_2,
41 min: fixed cost again). Both scored exactly the bar's 67,585 windows.

| test set | windows | Omega | fill | wash | spin | drain | bar | fill | wash | spin | drain | margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **all 18 cycles** | 67,585 | **0.7537** | 0.91 | 0.95 | 0.64 | 0.51 | 0.7365 | 0.89 | 0.95 | 0.66 | 0.45 | **+0.0172** |
| without cold_cotton_40_2 | 63,426 | 0.7481 | 0.90 | 0.95 | 0.63 | 0.50 | 0.7305 | 0.88 | 0.95 | 0.65 | 0.44 | +0.0176 |
| cold_cotton_40_2 alone | 4,159 | 0.8584 | 0.97 | 0.98 | 0.83 | 0.65 | 0.8444 | 0.99 | 0.97 | 0.85 | 0.56 | +0.0140 |

- **Against the criterion: fail.** The test needed ≥ 0.7865 (bar + 0.05) and
  spin ≥ 0.66. Omega is +0.017 above the bar, and 0.02 below it on spin.
- **The margin is consistent:** +0.017 with or without the exploration-seen
  cycle. On validation it was +0.005 over all 21 and +0.001 on the unseen 15.
- **Per state:** Omega wins drain (0.51 against 0.45) and, unlike on
  validation, fill (0.91 against 0.89). It ties wash (0.95) and loses spin
  (0.64 against 0.66).
- **Omega's errors, all 18:** spin → wash 1,183 and spin → drain 1,013, of
  6,903 spin windows; wash → spin 2,041 and wash → drain 1,232, of 54,968
  wash windows. These are the soft spin edges and the pre-spin phase.
- **The bar on test** (`fit/baseline_bar.py --test`): cotton 0.7064, eco
  0.8376, other 0.8053. Per cycle 0.52 (`hot_cotton_60_0`) to 0.87. Cotton
  is 80% of the test windows and has the same state mix as eco, so the
  spread is between cycles, not state mixes.
- **The setting and setup are now frozen.**

### Stage 6 — Deliver (`fit/deliver.py`)

Bundle, then run over becken-flt's cycles with their labels hidden.

### Stage 7 — Score the delivery (`fit/score_delivery.py`)

Score the predictions against the held-back labels. Report next to the test
number: same machine, new settings vs a second, faulty unit.

**Built (2026-09-30), not run.** Modelled on the Volve example's
`deliver_witsml.py` / `score_delivery.py`.
- **Stage 6:** one bundle from the Stage 5 blueprint, and runs of 25 files
  (5 runs for the 122 files). Run ids are saved as each run starts, and a
  rerun collects instead of starting again. Outputs are matched to files by
  finish timestamp, since outputs don't name their inputs. `--only` checks go
  to `fit/out/delivery_check/`.
- **Stage 7:** the held-back sidecars are row-aligned with the delivery
  files, so each prediction's finish timestamp gives its label directly
  (checked on a synthetic prediction file: 4,169 of 4,169 windows paired).
  Invalid and unpaired windows are counted, not scored.
- **The bar on delivery:** `baseline_bar.py --delivery`, the validation-chosen
  setting.
- **Checked on one cycle (2026-09-30, `--only warm_fast-15_0`):** bundle
  `bnd_4yhyypvk5w8vmvyz3k9ym9pr28`, run `agt_5hewjy0zk892hs7bvq8n9c0c2n`, **1
  minute** (far below an eval's ~40). The output columns are
  `finish_timestamp, start_timestamp, predicted_state, invalid, p_drain,
  p_fill, p_spin, p_wash`. Timestamps are epoch seconds with trailing zeros
  dropped (`1694181225.45`), one window every 2.56 s. That gives 406 windows
  for 208,220 rows, 0 invalid, and all 406 paired with a held-back label.
  - That one cycle scored macro-F1 0.62, with spin at 0.11. It's a single short
    cycle, so not a result, but it's the delivery-direction spin failure seen
    in exploration (becken-flt spins harder). Stage 7 will show whether it
    holds.

**Stage 6 result (2026-09-30):** bundle `bnd_1vcrydgb9c96s9dgr11tfvhzf0`, 5 runs,
all completed: 122 of 122 files have predictions, 365,414 windows, 0 invalid,
0 unmatched.
- **The runs didn't go side by side.** All showed `running` from 10:03, but they
  finished at 10:35, 11:23, 12:08, 12:32 and 12:53, about 25–50 min apart.
  So they were processed one at a time; the delivery took 2 h 50 min.
- One worker per run takes each file in turn: ~2 min for a full cotton
  cycle.

**Stage 7 result (Omega; the bar on delivery still to run):**

| set | windows | macro-F1 | fill | wash | spin | drain |
|---|---|---|---|---|---|---|
| **all 106 becken-flt cycles** | 365,414 | **0.7008** | 0.80 | 0.94 | 0.55 | 0.51 |
| without cold_cotton_40_2 and cold_cotton_30_4 | 359,359 | 0.6999 | 0.80 | 0.94 | 0.55 | 0.51 |
| cold_cotton_40_2 (seen in exploration) | 4,276 | 0.7710 | 0.81 | 0.96 | 0.64 | 0.67 |
| cold_cotton_30_4 (42% coverage) | 1,779 | 0.4786 | 0.92 | 1.00 | 0.00 | 0.00 |
| family cotton | 244,094 | 0.6657 | 0.76 | 0.94 | 0.50 | 0.46 |
| family eco | 60,493 | 0.7633 | 0.79 | 0.97 | 0.69 | 0.61 |
| family other | 60,827 | 0.7419 | 0.88 | 0.90 | 0.62 | 0.57 |

- **Delivery is 0.05 below test** (0.7008 against 0.7537), from spin
  (0.55 against 0.64) and fill (0.80 against 0.91). That's the
  second-unit, global-normalisation risk the README names.
- **The lowest cycles:** `warm_synthetic_6` 0.32, `hot_cotton_40_2` 0.40,
  `hot_cotton_60_0` 0.42, `hot_cotton_0_6` 0.45. Four of the eight lowest
  (`warm_synthetic_6`, `hot_cotton_40_2`, `hot_cotton_0_6`, `hot_cotton_30_4`)
  are the becken-flt cycles whose raw rate changes within the cycle
  (200 → 152–164 Hz, Stage 1a WARN; Stage 1c resampling WARN for three of
  them). A data-quality link, worth checking.
- **The empty-load pattern seen in run 1 is weaker overall:** the 19 `_0`
  cycles average 0.668, the other 87 0.711.

**The bar on the same delivery windows** (`fit/baseline_bar.py --delivery`):

| set | Omega | fill | wash | spin | drain | bar | fill | wash | spin | drain | margin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **all 106 cycles** | **0.7008** | 0.80 | 0.94 | 0.55 | 0.51 | 0.6927 | 0.79 | 0.93 | 0.66 | 0.38 | **+0.0081** |
| without the 2 | 0.6999 | 0.80 | 0.94 | 0.55 | 0.51 | 0.6916 | 0.79 | 0.93 | 0.66 | 0.38 | +0.0083 |
| cold_cotton_40_2 | 0.7710 | | | 0.64 | | 0.7932 | | | 0.84 | | −0.0222 |
| cold_cotton_30_4 | 0.4786 | | | | | 0.4894 | | | | | −0.0108 |
| cotton | 0.6657 | | | 0.50 | 0.46 | 0.6532 | | | 0.63 | 0.30 | +0.0125 |
| eco | 0.7633 | | | 0.69 | 0.61 | 0.7513 | | | 0.82 | 0.41 | +0.0120 |
| other | 0.7419 | | | 0.62 | 0.57 | 0.7503 | | | 0.69 | 0.55 | −0.0084 |

- **Against the criterion: fail,** +0.008 against +0.05, and spin 0.55 against
  the bar's 0.66.
- **Both drop from test to delivery by about the same:** Omega −0.053, the bar
  −0.044. But in different places:
  - **the bar's spin holds** (0.66 on test and delivery) while Omega's falls
    (0.64 → 0.55);
  - **Omega's drain holds** (0.51 both) while the bar's falls (0.45 → 0.38);
  - fill falls for both (Omega 0.91 → 0.80, bar 0.89 → 0.79).
- **The spin gap fits the exploration hypothesis.** becken-flt spins about
  twice as hard. The bar's level feature is monotonic, so a louder spin is
  still the loudest window. Omega's embedding of an out-of-range amplitude
  lands nearer wash. Not checked; per-unit normalisation (label-free) is the
  follow-up that would test it.

**Success criterion dropped (user, 2026-09-30, after all stages were scored).**
The rule "macro-F1 ≥ the bar's + 0.05, and spin ≥ the bar's" is removed, and
so is the pass/fail verdict.
- **Why:** the 0.05 was a round number set before the pipeline, with no basis
  in how much scores vary. The spin rule was already a replacement (Stage 4a).
- **What it means:** the decision came after seeing every result, and no
  stage had met the rule. The results are reported as margins over the bar,
  unchanged; the tables below and in the README show them all.
- **Code:** `fit/compare_bar.py` prints the margin and the spin difference,
  with no verdict. Earlier entries in this plan that say "fail" refer to the
  dropped rule.

**Summary across stages (macro-F1, Omega against the bar on the same windows):**
validation (6 search cycles) 0.7547 / 0.7406 (+0.014); validation (all 21)
0.7562 / 0.7511 (+0.005); test 0.7537 / 0.7365 (+0.017); delivery 0.7008 /
0.6927 (+0.008). Omega leads at every stage, by 0.005–0.017; it is
consistently better on drain and consistently not better on spin.


## Reproduced on production (2026-10-01)

Stages 4b–7 rerun on production (`api.u1`), from an empty `fit/out/` (dev's
outputs set aside in `fit/out-dev/`; the bar's files copied back, since the bar
is local). First the scripts were made deployment-aware (`7fc5fae`): `osm`
resolved by key, the upload cache keyed by endpoint, results and state tagged
with their endpoint, Stage 5 taking this deployment's latest 4c result.

| stage | dev | production |
|---|---|---|
| 4b, 16-trial search | `opt_4qggkyx4z487m86da5emyt7txw`, best 0.7547 (512 / 512, k 31), 8 trials beat the bar | `opt_4cc7qeq4jb9jzacp1k1y1rmjx5`, 3.6 h, a different draw: best 0.7557 (512 / 1024, k 15), 7 trials beat the bar |
| 4c, all 21 validation cycles | 0.7562 | `opt_5nsfm6b2kv842ts1w4djf5ehn2`: 0.7562, identical to 16 digits, every state too |
| 5, test | 0.7537 / 0.7481 without the seen cycle | `evl_69t9nc16jz9ekv24d5ee8s40mz`: 0.7537 / 0.7481; 1 of 67,585 windows differs |
| 6, delivery | 365,414 windows, 2 h 50 min | `bnd_1jdn4s2xte982v8kxjrj7z6vn3`: 365,414 windows, 0 invalid, 1 h 33 min |
| 7, delivery score | 0.7008 (bar 0.6927) | 0.7008 (bar 0.6927); 4 of 365,414 predictions differ |

- **The search can't be compared trial for trial:** the platform's sampler
  can't be seeded, so production drew its own 16 of the 288 settings. Only one
  setting was in both draws (512 / 1024, k 3, l1, uniform): 0.70815 on both, to
  16 digits. The pattern held: larger k better, every k 1 trial below the bar.
- **5 of 432,999 test and delivery windows were predicted differently.** In
  Stage 5 a wash window went to spin on dev, drain on production. Likely votes
  on a knife-edge (k 31, uniform), tipped by floating-point differences between
  the deployments' hardware: an inference, since the votes aren't visible. Every
  reported number is unchanged at four decimals.
- **Every delivery run reported `completed` before its last output was
  written** (a known platform issue). `deliver.py` now waits until every file's
  predictions reach the file's end (`d64b377`); each run's last file completed
  about a minute after `completed`. Without that, 5 of 122 files would have been
  saved short.

## Housekeeping

- **Reproducing Stages 0–1c:** README, "Run it yourself", has one subsection
  per stage: what it does, the command and run time, the exact expected
  results (the `RESULT:` line, WARN counts, key numbers) and what it writes.
  Confirmed by the user's own runs on 2026-09-29: Stages 1a–1c, 2 and 3
  all matched every expected count (Stage 2: 25.56 GB; Stage 3: all PASS).
  **Re-confirmed from Stage 0 after the 5-file library rebuild (2026-09-29):**
  Stage 2 wrote `library: 5 files (one per state, 1990 pieces)` (25.56 GB),
  and Stage 3 passed on 175 files, every check PASS. Keep
  those subsections in step with this plan whenever a stage's checks or
  outputs change.

- **Blueprint: the canonical `osm` blueprint, resolved by its key** on each
  deployment (dev `blp_6kwmqaqvww8bj95jc1zxcqzbq8`, production
  `blp_1ke4exx9w18w6s64wks23zdr04`; by key since 2026-10-01, `7fc5fae`).
  - **History:** it was pinned to `blp_05h8jmsdcy8fra7f0rm5cerwsv` while the
    latest had a bug colleagues were fixing. The only YAML difference is how
    the encoder is found.
  - **Why switching is safe:** after the fix, the user's repro completed, and
    the timestamp probe gave **identical** results on both blueprints: same
    scores, same confusion matrices, all three formats. The files are
    `fit/out/probe_timestamps_<blueprint>.json`.
  - **Pinning:** every platform script takes `--blueprint` to pin a version.
- **Long runs:** every script that can run for many minutes (download,
  timestamp probe, Optimize, and later test and deliver) takes
  `--background [LOG]`. It relaunches under `nohup` + `caffeinate -i` (macOS)
  with output to LOG (`prep/background.py`). Asked for by the user,
  2026-09-29.
- **API:** a key and endpoint in `.env` (`ATAI_API_KEY`, `ATAI_API_ENDPOINT`), one
  deployment at a time: dev until 2026-09-30, production from 2026-10-01.
  Platform routes are `<endpoint>/agents/...`; uploads are
  `<endpoint>/v0.5/files`.
- **Omega for local checks:** the encoder-only agent
  (`~/Downloads/demo/omega-encoder-only-agent`, image
  `agent-delivery:2026.09.24-da9de4f-arm64`), run as a separate container,
  `omega-encoder-larco`, on ports 19091/19093 with `AGENT_ASSET_KEY` from that
  repo's `.env`. Stopped now.
- **Cached exploration embeddings:** `osm-candidates/larco/emb/`.

## Known issues to keep in mind

- **One machine for everything but delivery.** Any single-cycle anomaly moves
  the test number; report per cycle.
- **The other programs are the hard part:** 23 single, short, warm-room
  cycles at lower spin speeds. A threshold is most likely to break there, and
  the test's share of them is only ~4 cycles.
- **Room temperature is confounded with program:** cotton and eco were run
  in cold and hot rooms, the other programs only in the warm room. No split
  can separate the two.
- **The spin failure is unexplained** until Stages 1a–4a check the amplitude
  hypothesis.
- **KUBO's weak spin signal is systematic across its cycles.** If this
  example later adds more units, check sensor coupling first.
