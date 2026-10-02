# Operational State Monitoring on washing machines (LARCO)

## TL;DR

- **What:** an OSM agent that labels each 2.56 s window of a washing machine's
  vibration as **`fill`**, **`wash`**, **`spin`** or **`drain`**, using frozen
  Omega 1.5 embeddings and the platform's kNN.
- **Data:** LARCO (CC BY 4.0). It's trained and tested on one healthy Becken
  BWM5381IX (new settings in test), then delivered to a second, faulty unit of
  the same model.
- **The model:** window 512, step 512, k 31, cosine, uniform, from a 16-trial
  search on the Optimize API.
- **Compared with:** a vibration-level baseline ("the bar": per-channel RMS
  with kNN), on the same windows at every stage.

| stage | Omega | bar | margin | Omega spin | bar spin |
|---|---|---|---|---|---|
| validation, all 21 cycles | 0.7562 | 0.7511 | +0.005 | 0.64 | 0.66 |
| **test, 18 new-setting cycles (once)** | **0.7537** | 0.7365 | **+0.017** | 0.64 | 0.66 |
| **delivery, 106 cycles of the faulty unit** | **0.7008** | 0.6927 | **+0.008** | 0.55 | 0.66 |

(macro-F1 over the four states)

- **Omega leads the bar at every stage,** by a small margin.
- **Omega is clearly better on drain** (0.51 against 0.38 on delivery) and
  **weaker on spin,** most of all on the second unit. becken-flt spins harder,
  and Omega's spin falls to 0.55 while the loudness-based bar holds at 0.66.
- **Heating was dropped as a state:** vibration can't tell it from wash.
- **All stages (0–7) are done,** on dev, and **reproduced on production**
  (Stages 4b–7, 2026-10-01): every number in the table above came out the same.
  Of the 432,999 test and delivery windows, 5 were predicted differently (see
  [Status](#status)). The plan, every decision and the measurements are in
  [`plan.md`](plan.md).

> **Licence: CC BY 4.0** (the data), so commercial use and redistribution are
> allowed with attribution. See [Data attribution](#data-attribution).

## Status

| stage | status |
|---|---|
| dataset search and exploration | **done**, in `~/Downloads/demo/osm-candidates/` |
| 0: split by setting group, then download | **done:** `data/split.json` (seed 20260928); 199 cycles × (label CSV + vibration parquet), 4.7 GB, every file matching the manifest |
| 1a: preflight the raw cycles | **done, rerun for four states** (`prep/preflight_raw.py`, ~30 s): no blocking FAIL. One coverage FAIL (a delivery cycle with 42% vibration) is kept and acknowledged, to revisit. Findings and decisions (four states, labels kept as given) in `plan.md` |
| 1b: prepare (200 Hz grid, one state per row) | **done, rerun for four states** (`prep/prepare.py`, ~80 s): cubic-spline resampling, glitches removed; 199 cycles → `data/prepared/`, 372M rows, 4.9 GB Parquet; 100% of becken's labelled seconds kept, 99.3% of becken-flt's |
| 1c: preflight the prepared files | **done, rerun for four states** (`prep/preflight_prepared.py`, ~10–45 s): PASS. Exact 5 ms grid, and 0 state mismatches in 372M rows. 8 cycles warn on resampling (slow raw stretches); the one acknowledged coverage FAIL is carried over |
| platform check: timestamps at 200 Hz | **done** (`fit/probe_timestamps.py`, on dev; identical on the pinned and the latest blueprint, which is now the default): fractional epoch seconds, epoch milliseconds and ISO 8601 all accepted, every window scored. Also found that macro-F1 counts a class missing from the validation set as 0 |
| 2: build the role files | **done, rerun for four states** (`prep/build_roles.py`, ~4.5 min): 25.45 GB in `data/roles/`. Library as 4 files (one per state, 1,600 pieces), 400 windows per state; validation 25 files / 38,907 windows; test 23 / 33,787; delivery 122 / 182,672, labels held back |
| 3: preflight the role files (platform contract) | **done, rerun for four states** (`prep/preflight_roles.py`, ~2 min): PASS on all 174 files and every role check, including all four states in validation (all 21 and the 6 search cycles), test and delivery |
| 4a: the bar (threshold and FFT baselines) | **done on four states, all 9 window/step pairs** (`fit/baseline_bar.py`): the level baseline wins everywhere, macro-F1 **0.72–0.74** on the 6 search cycles (0.7285 at 1024 / 1024, spin 0.61). Drain (~0.4) and spin (~0.6) are the hard states. The five-state bars are in `fit/out/five_states/` |
| 4b: Omega on the Optimize API | **done** (`opt_4qggkyx4z487m86da5emyt7txw`, 16 trials, ~3.5 h). Omega beat its bar in 8 trials. The best margin is +0.030 (0.7537 at 256 / 1024, k 21, cosine, distance). The highest score, 0.7547 (512 / 512, k 31, cosine, uniform), was carried forward by decision. Larger k is better throughout |
| 4b, earlier: five states | **superseded.** Search 1 (`opt_59jf7w1yje8krt5wqr5z0zzex6`, 16 trials): the best, 0.5534, only tied its bar, and every trial mislabelled most wash as heating. `fit/diagnose_wash_heating.py` showed vibration can't tell heating from wash, so heating was folded into wash (`plan.md`, "Labels"). Its outputs are in `fit/out/five_states/` |
| 4c: confirm on all 21 validation cycles | **done** (`opt_1dztapszen8n1a89jj08fcfwrr`, 44 min): trial #7's setting (512 / 512, k 31, cosine, uniform) scored **0.7562** on all 21, against the bar's 0.7511 (+0.0051; spin 0.64 against 0.66). On the 15 cycles the search never saw, Omega scored 0.7566 and the bar 0.7553: a tie. Omega is better on drain, worse on fill and spin |
| 5: test once (Evals API) | **done** (`evl_56qkrrh93j8xmsasanhpzfks74`, 39 min): on the 18 test cycles Omega scores **0.7537**, against the bar's 0.7365 on the same 67,585 windows: **+0.017**, and spin 0.64 against 0.66. Without the exploration-seen cycle: 0.7481 against 0.7305. Omega wins drain (0.51 against 0.45) and fill (0.91 against 0.89), ties wash and loses spin |
| 6: deliver to becken-flt (bundle) | **done** (`bnd_1vcrydgb9c96s9dgr11tfvhzf0`, 5 runs, 2 h 50 min): 122 of 122 files, 365,414 windows, 0 invalid |
| 7: score the delivery against held-back labels | **done** (`fit/score_delivery.py`, `fit/baseline_bar.py --delivery`): on all 106 becken-flt cycles Omega scores **0.7008** against the bar's 0.6927 (**+0.008**). Omega wins drain (0.51 against 0.38) but loses spin badly (0.55 against 0.66). The bar's spin holds on the second unit and Omega's doesn't |
| reproduced on production (`api.u1`) | **done** (2026-10-01, Stages 4b–7 from an empty `fit/out/`; dev's outputs set aside). **4b** (`opt_4cc7qeq4jb9jzacp1k1y1rmjx5`, 3.6 h): the platform drew its own 16 settings (its sampler can't be seeded); the one setting both draws share scored the same to 16 digits (0.70815), the best was 0.7557 (512 / 1024, k 15), and Omega beat the bar in 7 trials. **4c** (`opt_5nsfm6b2kv842ts1w4djf5ehn2`, 45 min): **0.7562**, identical to dev. **5** (`evl_69t9nc16jz9ekv24d5ee8s40mz`, 34 min): **0.7537** and 0.7481 without the seen cycle; 1 of 67,585 windows predicted differently (a wash window: spin on dev, drain here). **6** (`bnd_1jdn4s2xte982v8kxjrj7z6vn3`, 1 h 33 min): 122 of 122 files, 365,414 windows, 0 invalid; every run reported `completed` before its last output was written, and `deliver.py` waited for it. **7:** **0.7008** against the bar's 0.6927; 4 of 365,414 predictions differ from dev's |

Numbered to match the Paderborn example: from Stage 2 on the numbers and
meanings are the same there, and from Stage 4 on in Volve too. Stage 1's
three steps check and prepare the data.

Each preflight is a step of its own. It is read-only, prints PASS / WARN /
FAIL per check, and the next stage will not run while any check FAILs.

## The scenario

- A customer shares labelled cycles from **one washing machine**, the healthy
  **Becken BWM5381IX** (`becken`, 93 cycles).
- We report how the agent does on **settings of that machine it has never
  seen** (the test).
- We then run it over a **second unit of the same model** (`becken-flt`, 106
  cycles) with its labels hidden, and score the predictions afterwards
  (the delivery).

Two things about the second unit to keep in mind:
- **The dataset marks becken-flt as faulty,** without saying what the fault
  is. It never informs any choice here, so its delivery score is the
  cleanest "machine the model has never seen" number.
- **The test number is weaker than Volve's:** new settings of the same
  machine, not a new machine. Both numbers are reported side by side.

**The model sees only the 9 vibration channels.** The dataset also records
power, water flow, temperatures and more at 1 Hz. But the labels are derived
from exactly those readings (heating ≈ 1.7 kW of power; fill and drain ≈
water flowing in and out), so as inputs they would give the answer away. They
are kept for analysis only. Nothing but a clip-on accelerometer is needed: no
access to the machine's controller.

### How becken's cycles are split

becken's 93 cycles form **58 setting groups** (program × wash temperature ×
load):
- **cotton and eco groups** are a pair of near-twins, the same setting run in
  a 16 °C room and a 32 °C room;
- **the other 11 programs** were run once each, in a 25 °C room.

**Whole groups** are assigned at random to library, validation or test,
stratified by family, so no twin can land on both sides:

| role | groups (cotton / eco / other) | cycles |
|---|---|---|
| library | 34 (16 / 4 / 14) | 54 |
| validation | 13 (6 / 2 / 5) | 21 |
| test | 11 (6 / 1 / 4) | 18 |
| delivery | all of becken-flt | 106 |

The seed was fixed before any download and is never re-rolled.
[`plan.md`](plan.md) has the draw, and the one test cycle already seen in
exploration (reported on its own line).

### What each role's files are

| role | cycles | files | windows at 1024 / 1024 | labels | used in |
|---|---|---|---|---|---|
| library (training) | 54 becken | **4, one per state** | 1,600 (400 per state) | the state, by filename | 4b: every trial |
| validation | 21 becken | 25 | 38,907 | `label` column | 4b: the **6 search cycles** only; 4c: **all 21** |
| test | 18 becken | 23 | 33,787 | `label` column | 5: scored once |
| delivery | all 106 becken-flt | 122 | 182,672 | none (held back in `delivery_labels/`) | 6: run; 7: scored against the held-back labels |

- **The library.** Each state's file holds its 400 windows as continuous
  pieces from all its cycles, in time order with real timestamps. Jumps fall
  only between pieces, at whole-window boundaries. That keeps the
  optimization config under the platform's 1 MiB limit; in training, any
  window across a jump is skipped.
- **Validation, test and delivery** stay **one continuous file per recording
  segment**. A single scored window across a time jump would fail the whole
  trial or eval.
- **The 6 search-validation cycles** were fixed by a rule set before any
  scoring: spread across settings, each room twice, all four states in each.
  They are `cold_cotton_0_2`, `hot_cotton_40_11`, `hot_cotton_60_2`,
  `cold_eco_40-60_11`, `warm_fast-45_40_2` and `warm_mix_40_0`. The other 15
  are used once, to confirm the search's finalists.
- **Two cycles are reported on their own lines:**
  - **Test, becken `cold_cotton_40_2`:** seen during exploration.
  - **Delivery, becken-flt `cold_cotton_30_4`:** vibration for only 42% of
    the cycle, kept for now.

## The states

| state | share of time | what the machine is doing | what the accelerometer sees |
|---|---|---|---|
| `fill` | 5% | water entering; the motor is off (~7 W) | about 1.5–2× wash's median level, likely water rushing through the valve and pipes |
| `wash` | 83% | drum tumbling back and forth, with ~10 s pauses | a slow, reversing pattern, quiet during the pauses |
| `spin` | 9% | steady high-speed rotation to extract water | a strong, steady tone, ~8× wash's level |
| `drain` | 3% | the pump emptying the drum, without spin | about 2.5× wash's level, in short bursts (median ~5 s) |

(Shares are of becken's 257 hours.)

- **Source of the labels:** the machine's own power and water sensors, at
  1 Hz, not a human annotator. Each second maps to one state, first match
  wins:
  1. `spin`: `centrifuge_label`, also while draining;
  2. `fill`: `water_label == 1`;
  3. `drain`: `water_label == -1`, or `2` (water in and out at once);
  4. `wash`: everything else, including the pauses.
- **Heating is not a state.** The heater (`heating_label`, ~1.7 kW, 8% of the time) runs while the
  drum washes, often in short thermostat bursts, and vibration can't tell it apart from
  wash: held-out cycles scored balanced accuracy 0.62, where 0.5 is chance. Its seconds take their drum/water state, almost always
  `wash`. It was a fifth state until the first search showed this; see `plan.md`,
  "Four states".
- **Expected hard pairs:**
  - **drain vs its neighbours:** its runs are short;
  - **the soft edges around spin:** a loud 2–3 minute pre-spin phase is
    labelled wash or drain, and the spin label starts quietly. **The labels
    are kept as the dataset gives them** (no guard band), so every score here
    includes these windows, roughly 5% of each cycle. See `plan.md`, Stage 1a
    findings.
- **No "idle" state:** inside a recorded cycle the drum never rests for more
  than about a minute. The pauses between tumbles are part of `wash`.

## Why this dataset: the evidence so far

This is exploration, not the pipeline, and it used three states (fill / wash / spin, dropping heating and drain): one cycle per unit (cotton, 40 °C,
2 kg), train on one unit, test on the other, 1,565 windows of 5.1 s, pooled
macro-F1. Omega 1.5 came from the local encoder-only agent, with global
normalisation fitted on the training unit.

| method | macro-F1 | fill | wash | spin |
|---|---|---|---|---|
| RMS level (a threshold) | **0.86** | 0.69 | 0.95 | **0.95** |
| FFT band-power kNN | 0.70 | 0.57 | 0.88 | 0.65 |
| Omega 1.5 kNN | 0.83 | **0.92** | 0.90 | 0.66 |

- **Omega does what a threshold cannot on `fill`.** Fill vs wash is a
  difference in motion pattern, not loudness. The threshold calls 64 of
  becken's 78 fill windows wash; Omega gets 67 right.
- **Omega loses `spin`, trained on becken and scored on becken-flt** (the
  delivery direction): 159 of 183 spin windows go to wash. becken-flt spins roughly
  twice as hard, so its spin windows likely fall outside the training range
  after becken's normalisation. This is a hypothesis, not yet checked.
- **The pipeline didn't bear out the `fill` result.** With four states, the
  library's 54 cycles and the platform's kNN, the level baseline scores fill
  0.93 and Omega 0.89 (15 unseen validation cycles, Stage 4c). The exploration
  was one cycle per unit and three states.
- **Adding the RMS features to Omega's reached ~0.93,** but the weight was
  chosen after seeing the results. The platform's classifier is a kNN over
  embeddings only, so the pipeline cannot use that combination.

**Normalisation: global.** Every file is z-scored with the library unit's
per-channel statistics, as in the Volve example. This is the setup that lost
`spin` in exploration, in the becken → becken-flt direction. That failure can
only show up in the delivery score (Stage 7), and it will be reported as it
comes out. Per-unit scaling (label-free) is a follow-up, reported separately if
tried.

**How Omega is judged:** against the bar, a vibration-level baseline, on the
same windows at every stage: macro-F1 over the four states, and spin F1 on its
own. Each window/step pair has its own bar (at 1024 / 1024, 0.7285 macro-F1 and
spin 0.61 on the 6 search cycles). The margin is reported as it comes out,
with no pass/fail threshold.

Omega leads the bar at every stage, by 0.005–0.017. Its spin F1 is at or below
the bar's, and well below it on the second unit:

| stage | Omega | bar | margin | Omega spin | bar spin |
|---|---|---|---|---|---|
| 4b, 6 search cycles | 0.7547 | 0.7406 | +0.014 | 0.65 | 0.64 |
| 4c, all 21 validation | 0.7562 | 0.7511 | +0.005 | 0.64 | 0.66 |
| 5, test (18 cycles) | 0.7537 | 0.7365 | +0.017 | 0.64 | 0.66 |
| 7, delivery (106 becken-flt cycles) | 0.7008 | 0.6927 | +0.008 | 0.55 | 0.66 |

> **There used to be a pass/fail criterion, changed twice and then dropped.**
> Before the pipeline it was "beat the bar's macro-F1 by ≥ 0.05 **and** spin
> F1 ≥ 0.90". Both numbers were set without evidence: the 0.90 came from a
> 3-state exploration, and the 0.05 was a round number.
> - **After seeing the bar** (Stage 4a), the spin rule became "no worse than
>   the bar on spin", because the bar itself scored spin 0.58.
> - **After all stages were scored** (2026-09-30), the +0.05 rule was dropped,
>   and so was the pass/fail verdict. No stage met it, and it never had a basis.
>
> The tables above show every margin, so the result reads the same either
> way. See `plan.md`, "Success criterion dropped".

## What your data needs

These are the same platform rules as the Volve example, applied here:

| requirement | here |
|---|---|
| A CSV with a timestamp column and numeric channels | 9 accelerometer channels (3 triaxial sensors: back, side, top) |
| Windows of 16–1024 samples | 1024 at 200 Hz = 5.1 s (512 and 100 Hz variants searched in Stage 4b) |
| Training data as one state per file | Stage 2 cuts the library cycles into single-state files |
| Continuous validation and test files, labelled by a column | one file per continuous stretch, cut at vibration gaps |
| Regular sampling | the raw vibration jitters at 200–212 Hz, so Stage 1b resamples to 200 Hz |

## The pipeline

```
Zenodo ─> 0 split + download ─> 1a preflight ─> 1b prepare ─> 1c preflight ─> 2 role files ─> 3 preflight
                                                                                                    │
 7 score delivery <─ 6 deliver to becken-flt <─ 5 test once <─ 4c confirm <─ 4b Omega search <─ 4a the bar
   (held-back labels)    (bundle, unlabelled)    (Evals API)   (all 21 val.)   (Optimize API)   (threshold, FFT)
```

Every decision, and what is still open, is in [`plan.md`](plan.md).

## Platform behaviour worth knowing

Found while building this example on dev, and reproducing it on production
(details in `plan.md`):

- **More than ~1,500 training files in one optimization never runs.** The
  platform puts the whole config in a Kubernetes ConfigMap (1 MiB max). Over
  that, the job is retried forever while the API shows `running` / `pending`,
  with no error. Hence this example's library is a few files, one per state.
- **Training files may contain time gaps, but windows across a gap are
  silently dropped.** A library file made of separate stretches, with forward
  jumps between them, trains fine. Any window that crosses a jump fails the
  sampling-rate check and is skipped without being counted, so the model
  trains only on windows within a stretch. This example puts every jump on a
  whole-window boundary, and its bar cuts windows the same way.
- **Scored files may not contain gaps, unless every jump falls on a window
  boundary.** In a validation or test file, a single window across a jump
  fails the whole trial or eval: "eval-mode test data must not contain
  windows a validation node rejected". This is deliberate (ground truth is
  only scored on data expected to be valid), though the platform team
  regards failing the whole run as a bug. So validation, test and delivery
  files are one continuous file per recording segment.
- **`sample_rate_interval_tolerance` is relative to the window's mean
  interval.** The default is 0.05. One jump inside a 1,024-row window
  deviates by ~1,000×, so loosening to 10 (as tried) doesn't pass it; it
  would take roughly the window size. The search space can't set it to
  `null` or "warn", so the check can't be switched off per run.
- **Timestamps may be fractional epoch seconds, epoch milliseconds or ISO
  8601,** all read to the millisecond at 200 Hz.
- **Macro-F1 counts a state the model knows but the scored data lacks as
  F1 = 0.** Every scored set must contain every state.
- **A run can report `completed` before its last output file is fully
  written.** On production, all 5 delivery runs did: each one's last file was
  still short at `completed`, and complete about a minute later.
  `deliver.py` counts a run as completed only once every file's predictions
  reach that file's end.
- **Results reproduce across deployments, to a few windows in 100,000.** The
  same setting on dev and production gave the same scores to the fourth
  decimal at every stage, and the same window counts; 5 of 432,999 test and
  delivery predictions differed. A random search (`--max-trials` below the
  space) draws different settings on each run, though.

## Run it yourself

Everything runs from the repo root on macOS or Linux. Every stage is safe to
re-run: it overwrites its own outputs and nothing else.

### Setup (once)

```sh
git clone https://github.com/archetypeai/osm-agent-example-larco.git   # needs Git LFS (brew install git-lfs)
cd osm-agent-example-larco
python3 -m venv .venv
source .venv/bin/activate          # every later command assumes this
pip install -r requirements.txt    # numpy, pandas, pyarrow (Parquet), scipy (the resampling spline)
cp .env.example .env               # only needed from Stage 4b on (platform API key)
```

Stage 0 is stdlib-only; Stages 1a–1c need the packages above.

**Switching deployment** (dev, staging, prod, Tokyo: set `ATAI_API_KEY` and
`ATAI_API_ENDPOINT` in `.env`) needs nothing else. The `osm` blueprint is resolved by
its key on each deployment, the upload cache is kept per deployment, and every result
and state file in `fit/out/` records its endpoint: `test.py` only picks this
deployment's Stage 4c result, and `test.py` and `deliver.py` refuse to resume another
deployment's run. Move `fit/out/` aside first to start a deployment from scratch.

### Shortcut: skip Stages 0–3 with the packed role files

The role files Stage 2 builds (`data/roles/`, 25.45 GB) are in the repo, packed as
tar.xz parts in Git LFS under `data/archives/`: 4.45 GB, about 6× smaller. So Stages 4–7
can run without downloading or preparing anything:

```sh
git lfs pull                                   # fetches data/archives/ (skip if the clone already did)
python prep/archive_roles.py --unpack          # checks SHA256SUMS, rebuilds data/roles/ (both archives)
python prep/preflight_roles.py                 # Stage 3, to confirm: RESULT: PASS
```

Stage 3 then warns on two checks, `one state` and `scaling`: they compare the role
files with `data/prepared/`, which only Stage 1 makes. The other checks run as
usual, and the result is still `PASS`.

There are two archives, so you can fetch only what you need:

| archive | holds | size | for |
|---|---|---|---|
| `roles_core` | library, validation, test, `manifest.json`, `zscore_stats.json` | 1.27 GB, 2 parts | Stages 4–5 |
| `roles_delivery` | delivery, delivery_labels | 3.18 GB, 4 parts | Stages 6–7 |

For Stages 4–5 only: `git lfs pull --include "data/archives/roles_core*"`, then
`python prep/archive_roles.py --unpack --only core`. Stage 3 checks every
role, so it will flag the missing delivery files; skip it in that case.

Unpacking needs about 26 GB free. The archives were packed with
`python prep/archive_roles.py --pack --background` after Stage 2 (19 min), and are
repacked whenever Stage 2 changes.

### Stage 0: split and download (skip if `data/raw/` is already complete)

```sh
python prep/split.py                    # a few seconds; writes data/split.json from the archive listing
python prep/download.py                 # label CSVs for both units, ~230 MB, ~2 min
python prep/download.py --vibration --background   # vibration, 4.5 GB, ~35 min, detached
tail -f data/download.log               # resumable, one line per file downloaded
```

- **What you should see:** `split.py` prints library 34 groups / 54 cycles,
  validation 13 / 21, test 11 / 18, delivery 106 cycles, seed 20260928.
  The seed is fixed, so every run gives the same split.
- **Downloads resume:** they skip files already present at the right size,
  and retry Zenodo's rate limit. `--background` relaunches the command under
  `nohup` (it keeps going if the terminal closes) and, on macOS,
  `caffeinate -i` (the Mac doesn't sleep until it ends); the last log
  line is `done: … downloaded, … already present`.
- **Check it's complete:** `ls data/raw/vibration | wc -l` shows 199.

Stages 1a–1c run in order. 1b won't start while 1a has a blocking FAIL,
and 1c won't start until 1a and 1b have run (and 1a has no blocking FAIL).
The three preflights (1a, 1c and 3) are read-only. They print one line per
check with its PASS / WARN / FAIL counts, then every cycle behind a WARN or
FAIL, then a final `RESULT:` line. The exit code is 0 on PASS and 1 on a
blocking FAIL; an acknowledged FAIL (`prep/acknowledged.py`) doesn't block.

### Stage 1a: preflight the raw cycles

```sh
python prep/preflight_raw.py         # ~30 s
```

**What it does:** checks every one of the 199 cycles against what later
stages assume:
- files present, and matching `data/manifest.json`;
- schema and label values;
- label timeline;
- vibration rate and gaps;
- glitches, flat or clipped channels;
- fill and spin present;
- coverage;
- label/vibration alignment at spin ends.

It also measures, per cycle, the recording date, seconds per state and the
vibration level in each state.

**What you should see:**
- **The last line:** `RESULT: PASS, no blocking failures (1 acknowledged,
  listed above). Stage 1b may run.`
- **WARN counts:** label timeline 3, vibration rate 21, glitches 1, channels
  34, states 1, coverage 1, alignment 7.
- **The one FAIL:** becken-flt `cold_cotton_30_4` coverage (42.2%), tagged
  `[acknowledged: …]`.
- **The per-state summary (median level):** becken fill 0.0131 g (5% of the
  time), wash 0.0065 (83%), spin 0.0534 (9%), drain 0.0160 (3%); becken-flt
  fill 0.0182, wash 0.0062, spin 0.0841, drain 0.0118. Recorded 2023-05-23 .. 08-20 (becken) and
  2023-08-21 .. 12-18 (becken-flt).

**Writes:** `data/preflight_raw.json`: every check for every cycle, plus the
dates, seconds and vibration level per state, clock offsets and spin-end
offsets.

### Stage 1b: prepare (200 Hz grid, one state per row)

```sh
python prep/prepare.py               # ~80 s, writes 4.9 GB
```

**What it does, per cycle:**
1. Cuts into segments at vibration gaps over 1 s and label gaps over 1.5 s.
2. Removes impossible raw samples (beyond ±2.2 g).
3. Resamples each segment onto an exact 5 ms grid by cubic spline, rounded
   to 1e-4 g.
4. Gives every row the state of its label second.
5. Drops segments shorter than one window (1,024 rows).

Only the 9 vibration channels are kept.

**What you should see:** one line per cycle as it finishes, in a varying
order (the cycles run in parallel), then `199 cycles, 372,231,085 rows,
4.86 GB; report …/data/prepare_report.json`. becken-flt
`cold_cotton_30_11` is cut into 8 segments; 16 other cycles into 2–5.

**Writes:**
- `data/prepared/<cycle>.parquet`: `timestamp` (UTC, exact 5 ms), the 9
  channels in g (float32), `state`, `segment`;
- `data/prepare_report.json`: segments, rows, glitches removed and seconds
  per state, per cycle.

### Stage 1c: preflight the prepared files

```sh
python prep/preflight_prepared.py    # ~45 s
```

**What it does:** checks every prepared file:
- columns and types;
- the exact 5 ms grid;
- values finite and within ±2.2 g;
- only the four states;
- no segment shorter than one window;
- ≥ 80% of the labelled seconds kept.

It then re-derives Stage 1b's claims from the raw files:
- every row's state against the four-state rule;
- a tone test (10 / 23 / 46 Hz) through Stage 1b's resampler, on the cycle's
  own raw timestamps.

**What you should see:**
- **The last line:** `RESULT: PASS, no blocking failures (1 acknowledged,
  listed above). Stage 2 may run.`
- **Passing on all 199 cycles:** file, columns, grid, values, states and
  state match.
- **Resampling:** 191 pass, 8 warn. These are cycles with a slowly recorded
  stretch, where 23 Hz keeps 84–86%.
- **Coverage:** the same acknowledged FAIL as Stage 1a.
- **Prepared hours per state:** library fill 7.1, wash 127.7, spin 13.6,
  drain 5.3; validation 2.5 / 46.4 / 4.6 / 1.8; test 2.4 / 39.1 / 4.9 / 1.7;
  delivery 14.3 / 210.4 / 24.0 / 11.2.

**Writes:** `data/preflight_prepared.json`: every check for every cycle, plus
the tone-test results.

### Stage 2: build the role files

```sh
python prep/build_roles.py           # ~4.5 min, writes 25.45 GB to data/roles/
```

**What it does:**
1. Computes each channel's mean and std over the 54 library cycles only
   (global normalisation).
2. Draws the library: 400 windows per state, spread evenly across the
   library cycles that have the state and evenly spaced within each, one
   continuous single-state file per stretch.
3. Writes validation, test and delivery as whole cycles, one continuous file
   per segment. Delivery files carry no label; their labels go to
   `delivery_labels/` instead.

Every file starts with `timestamp` (epoch seconds, 3 decimals), then the 9
channels z-scored with the library statistics (4 decimals). Validation and
test files end with a `label` column.

**What you should see** (the order of lines is fixed):
```
z-score stats from 54 library cycles, 110,644,180 rows
library: 4 files (one per state, 1600 pieces), windows per state {'fill': 400, 'wash': 400, 'spin': 400, 'drain': 400}
validation: 25 files from 21 cycles, 38,907 windows
test: 23 files from 18 cycles, 33,787 windows
delivery: 122 files from 106 cycles, 182,672 windows
```
The last line reports the total size, 25.45 GB (about 3.5 GB of it is the delivery labels).
The library is 4 files, one per state. Each holds its state's 400 windows as
1,600 continuous pieces in all (adjacent chosen windows form one piece), in
time order with real timestamps: forward jumps only between pieces, always
at a 1,024-row boundary.

**Why 4 files and not one per piece:** an optimization with more than ~1,500
training files never runs (see "Platform behaviour worth knowing"), and
training files tolerate jumps between pieces (the gap probe). The manifest
lists every piece's cycle, start and length.

**Writes** (`data/roles/`):

| path | what it holds |
|---|---|
| `library/<state>__library.csv` | one file per state: its pieces in time order; the manifest lists each piece |
| `validation/<cycle>__seg<N>.csv` | continuous, with `label`; the manifest's `search_validation` lists the 6 search cycles |
| `test/<cycle>__seg<N>.csv` | continuous, with `label` |
| `delivery/<cycle>__seg<N>.csv` | continuous, no label |
| `delivery_labels/<cycle>__seg<N>.csv` | the held-back `timestamp,label` for delivery |
| `zscore_stats.json` | the library statistics used for every file |
| `manifest.json` | every file with its cycle, rows, windows and seconds per state; the library draw |

Re-running rebuilds each role's folder from scratch. `--roles library test`
limits it to some roles; `--only <cycle>.csv` to some cycles, for a quick
check.

### Stage 3: preflight the role files

```sh
python prep/preflight_roles.py       # ~2 min, reads all 25.45 GB
```

**What it does:** checks every file against the platform's rules:
- header and timestamp format;
- an exact 5 ms timeline;
- finite values and whole windows;
- validation and test labels only from the four states;
- each library file one state in every piece, matching the prepared data, with
  time jumps only between pieces at 1,024-row boundaries;
- each delivery file unlabelled, with a matching held-back sidecar;
- values scaled with the library statistics (spot-checked).

Then it checks the roles:
- the manifest matches the disk;
- no cycle or setting group in two roles;
- the statistics come from the library cycles only;
- the library holds exactly 400 windows per state;
- every scored set contains all four states: validation (all 21 cycles and
  the 6 search cycles), test and delivery. The platform scores a missing
  state as F1 = 0.

**What you should see:** every check `PASS`. That's header, timeline, values,
windows and scaling on 174 files each; held-back labels 122; one state 4,
checked piece by piece; labels 48; and the whole-role checks. Then

```
Windows per role:
  library        4 files     1,600 windows
  validation    25 files    38,907 windows
  test          23 files    33,787 windows
  delivery     122 files   182,672 windows
...
RESULT: PASS, no blocking failures. Stage 4 may run.
```

**Writes:** `data/preflight_roles.json`: every check for every file and
role, including the hours per state of each scored set.

### Stage 4a: the bar

```sh
python fit/baseline_bar.py           # ~90 s, local, no platform time
```

**Why a bar:** the example tests whether frozen Omega embeddings recognise machine
states better than simple signal features. So the bar keeps everything else the
same as Omega's run: the same library windows (cut within the same pieces), the
same validation windows labelled by their last row, the same classifier (kNN), the
same platform-style macro-F1, and the same choice of setting on the 6 search
cycles. What's left to differ is the representation: Omega's embedding against a
hand-made feature.

| features | what it is | why |
|---|---|---|
| **level** | per-channel log RMS of the window, mean removed: 9 numbers | what a **threshold** uses: "how loud is each sensor". Exploration found an RMS threshold already scored 0.86 on 3 states, so this is the baseline most likely to be hard to beat |
| **fft** | per-channel log power in 16 log-spaced frequency bands, each window standardised first: 144 numbers | the classic hand-crafted vibration feature. Standardising removes loudness, so it measures **shape** (which frequencies), not level. It tests whether spectral shape alone does as well as Omega |
| **level+fft** | both: 153 numbers | the strongest simple combination: loudness plus spectrum |

The features are z-scored on the library before kNN, so no one feature dominates
the distances.

**How to read it:** level wins at every window/step pair. On this machine
loudness separates the states well (spin is loud, fill and drain are medium, wash
is quiet), and adding FFT's 144 dimensions dilutes level's 9 in the kNN distance.
So the bar is effectively a loudness classifier. That's what makes the comparison
informative: Omega beats it on drain, which loudness alone can't place, and loses
on spin, where loudness is exactly the right cue.

**What it does:** scores simple baselines on the role files the way the
platform scores Omega:
- windows of 1,024 rows at step 1,024, labelled by their last row;
- macro-F1 over all four states;
- level (per-channel RMS, what a threshold uses), FFT band power, and both,
  each with kNN over a grid of k and distance;
- the best setting picked on the 6 search cycles, then reported on all 21.

**What you should see:** 24 grid lines, then the best setting on the 6 search
cycles and each feature set's best, scored on all 21:

```
best on the 6 search cycles: level k=15 l2, macro-F1 0.7285
  level     k=15  l2  search 0.7285  all 21: 0.7405  fill 0.93  wash 0.95  spin 0.63  drain 0.45
  fft       k=15  l1  search 0.6177  all 21: 0.6041  fill 0.47  wash 0.90  spin 0.57  drain 0.47
  level+fft k=15  l1  search 0.6989  all 21: 0.6973  fill 0.74  wash 0.94  spin 0.63  drain 0.47
```

(The five-state best was 0.5455, with heating at 0.19.)

**Every window/step pair of the search** needs its bar for `fit/compare_bar.py`.
All 9, detached (an hour or more; the step-256 pairs are the slow ones):

```sh
nohup caffeinate -i bash -c 'for w in 256 512 1024; do for s in 256 512 1024; do
  python -u fit/baseline_bar.py --window $w --step $s; done; done' > fit/out/bars.log 2>&1 &
tail -f fit/out/bars.log
```

**Writes:** `fit/out/bar_validation_w1024_s1024.json`: every grid setting's scores
and confusion matrix, and the best settings on all 21 cycles per cycle.
`--window W --step S` computes the bar for another window/step pair,
written to `bar_validation_w<W>_s<S>.json` (512 / 512 takes ~8 min).

### Stage 4b: Omega on the Optimize API

```sh
python fit/optimize.py --pool full --allow-gaps --max-trials 16 --dry-run      # the plan; no uploads, no jobs
python fit/optimize.py --pool full --allow-gaps --max-trials 16 \
    --name "LARCO Stage 4b four states" --background fit/out/optimize_4states.log
tail -f fit/out/optimize_4states.log
```

This is the run behind the results (`opt_4qggkyx4z487m86da5emyt7txw`): a
16-trial random search over the full pool, about 3.5 hours. With no flags,
`python fit/optimize.py --background` runs a single measurement trial
(window 1,024, step 1,024, k 5, l1), logging to `fit/out/optimize.log`.

**What it does:** only the platform. It needs no baseline.
1. Uploads the 4 library files and the 6 search-validation files once.
   The ids are cached in `fit/out/uploads.json`.
2. Creates an optimization on the latest `osm` blueprint. Each library file
   is a training example labelled by its state; each search file is scored
   on its `label` column.
3. Polls until the trials finish.
4. Prints each trial's macro-F1, F1 per state, windows scored and minutes,
   flags configurations sampled twice, and reports trial time as a fixed
   cost plus a cost per 1,000 windows (trials run one after another; search 1
   measured ≈ 11 min + 0.09 min per 1,000 windows).

**The search space** is the product of the values given, by flag or with
`--pool full`:

| parameter | `--pool full` values |
|---|---|
| window | 256, 512, 1024 |
| step | 256, 512, 1024 |
| k | 1, 3, 5, 7, 9, 15, 21, 31 |
| metric | `l1`, `cosine` |
| weights | `uniform`, `distance` |

That's 288 configurations. With `--max-trials` below that, the platform
samples: a random search. The draw can't be seeded, so a rerun scores a
different 16; any setting two draws share scores the same.
- **Step > window** skips records, so it needs `--allow-gaps`.
- **The default is one trial:** window 1,024, step 1,024, k 5, l1.
- **If the poller dies,** the run carries on at the platform: collect it with
  `--resume opt_…`.

**What you should see:** the per-pair window counts (e.g. `window 512, step 512:
library 4 files / 3,200 windows kept; validation 6 files / 20,682 windows`),
`16 trial(s) of a 288-point space (random search)`, the upload (10 files,
1,054 MB), `optimization opt_… created`, then a status line as each trial
finishes, about every 12–15 minutes. At the end it prints every trial ranked
by macro-F1. In our run the best was #7, `w=512 step=512 k=31 cosine uniform
macro-F1 0.7547  fill 0.91  wash 0.95  spin 0.65  drain 0.51`. The platform
samples the 16 configurations, so a new run draws different ones.

**Writes:** `fit/out/optimize_<optimization id>.json`: the optimization, every
trial and the summary.

### Stage 4b, optional: compare the trials with the bar

```sh
python fit/compare_bar.py            # every optimization saved in fit/out/
```

For each trial, it loads the bar computed on the same window/step and
cycles. It shows the macro-F1 margin over the bar and the spin difference,
ranked by margin; there's no pass/fail threshold. For a window/step with no bar yet, it
prints the `fit/baseline_bar.py --window … --step …` command to compute
one. This step is separate so that the Optimize step works for someone with
no baseline at all.

**What you should see** after Stages 4b and 4c: 17 rows, the 16 search trials
(`6 search`) and the Stage 4c confirmation (`all 21`, margin +0.0051). The top row is
`6 search w=256 step=1024 k=21 cosine distance macro-F1 0.7537 … | bar 0.7236 (spin 0.62)  margin +0.0301  spin +0.03`.
Runs scored on other states (the five-state results in `fit/out/five_states/`) are
not in `fit/out/`, so they aren't read.

**Writes:** `fit/out/compare_bar.json`.

**Long runs:** every script that can run for many minutes (`prep/download.py`,
`fit/probe_timestamps.py`, `fit/optimize.py`) takes `--background [LOG]`. It
relaunches the same command under `nohup` and, on macOS, `caffeinate -i`,
prints the pid and the `tail -f` / `kill` commands, and writes all output to
LOG.

### Stage 4c: confirm on all 21 validation cycles

```sh
python fit/optimize.py --windows 512 --steps 512 --k 31 --metrics cosine --weights uniform \
    --validation all --upload-jobs 3 --name "LARCO Stage 4c confirm" --background fit/out/confirm.log
python fit/compare_bar.py            # compares it with the bar on all 21 cycles
```

**What it does:** scores the chosen setting once on **all 21 validation
cycles** (25 files): the 6 search cycles plus the other 15. That checks the
search didn't pick a setting that only suits the six. It runs before the test,
so the test stays untouched.
- **The chosen setting:** four-state search trial #7 (512 / 512, k 31, cosine,
  uniform), the highest score on the 6 search cycles, 0.7547. It was carried
  forward by decision. See `plan.md`.
- **Uploads:** the 19 files not yet on the platform (2.56 GB).
  `--upload-jobs 3` keeps the uplink from saturating (see Stage 4b).
- **Time:** one trial; ours took 44 minutes.
- **`fit/compare_bar.py`** reads the run's `validation_set` and compares it with
  the bar's all-21 score at the same window/step (0.7511 at 512 / 512).

**What you should see:** `validation 25 files / 77,823 windows`, the upload,
then one trial: `w=512 step=512 k=31 cosine uniform completed macro-F1 0.7562
fill 0.90 wash 0.95 spin 0.64 drain 0.53 windows 77,823`. `compare_bar.py`
adds an `all 21` row: `bar 0.7511 (spin 0.66) +0.0051 fail`.

**Writes:** `fit/out/optimize_<optimization id>.json`, with `validation_set: all`.

### Stage 5: test once

```sh
python fit/test.py --background                          # Omega on the Evals API, detached; log fit/out/test.log
python fit/baseline_bar.py --window 512 --step 512 --test   # the bar on the same test windows, local
```

**What `fit/test.py` does** (platform only):
1. Promotes the Stage 4c trial (this deployment's latest Stage 4c result in
   `fit/out/`, or `--optimization opt_...`; on dev `opt_1dztapszen8n1a89jj08fcfwrr`:
   512 / 512, k 31, cosine, uniform, trained on the 4 library files) to the
   blueprint `osm-larco-w512-s512-cosine-k31-uniform` (named after its setting),
   or reuses it if it was already promoted.
2. Uploads the 23 test files (18 cycles, whole), 3 at a time.
3. Submits two evals together: all 23 files (the test number), and the files
   of `cold_cotton_40_2` alone, the one test cycle seen in exploration. The
   platform reports pooled confusion matrices only, so "without it" is the
   first minus the second.
4. Waits, then prints macro-F1 and F1 per state for all 18, without
   `cold_cotton_40_2`, and for it alone, plus the confusion matrix.

Every step's id is recorded in `fit/out/test_state.json`, so a rerun resumes
where it stopped instead of promoting or testing again. **This is the one-shot
test:** once its number is seen, the setting and setup are frozen.

**What `baseline_bar.py --test` does:** takes the bar's setting chosen on
validation for 512 / 512 (level, k 51, l2), never re-chosen. It scores it once
on the same test windows: pooled, without `cold_cotton_40_2`, per program
family (cotton / eco / other) and per cycle. Per-cycle and per-family numbers
exist for the bar only; the platform's report is pooled.

**What you should see:** the promotion, the upload (23 files, 2,994 MB), two
evals (about 40 minutes each, run side by side), then:

```
  all 18 test cycles                           macro-F1 0.7537  fill 0.91  wash 0.95  spin 0.64  drain 0.51  windows 67,585
  without becken_BWM5381IX_cold_cotton_40_2    macro-F1 0.7481  fill 0.90  wash 0.95  spin 0.63  drain 0.50  windows 63,426
  becken_BWM5381IX_cold_cotton_40_2 alone      macro-F1 0.8584  fill 0.97  wash 0.98  spin 0.83  drain 0.65  windows 4,159
```

and the bar on the same windows: `all 18 test cycles macro-F1 0.7365 fill 0.89
wash 0.95 spin 0.66 drain 0.45`. Rerunning `fit/test.py` reuses the saved ids
and doesn't test again.

**Writes:** `fit/out/test.json` (the summary), `fit/out/test_<eval id>.json`
(each eval), `fit/out/bar_test_w512_s512.json`.

### Stage 6: deliver to becken-flt

```sh
python fit/deliver.py --only wm_becken-flt_BWM5381IX_warm_fast-15_0.csv   # optional check, one short cycle
python fit/deliver.py --background                       # all 106 cycles (122 files) in 5 runs of ≤ 25 files, detached; log fit/out/deliver.log
python fit/deliver.py --resume                           # if the poller died: collect without starting anything
```

**What it does:**
1. Uploads the 122 delivery files (all 106 becken-flt cycles, whole, no
   labels; about 15 GB), 3 at a time, cached.
2. Creates one bundle from the Stage 5 blueprint (in `fit/out/test_state.json`;
   `osm-larco-w512-s512-cosine-k31-uniform`), unchanged.
3. Runs it over the files in batches (`--files-per-run`, default 25): the 122
   files, which are the 106 cycles (some split into segments at recording
   gaps, Stage 1b), go out as 5 runs of 25, 25, 25, 25 and 22 files. All five
   start at once and show `running`, but the platform processed them one at a
   time: they finished 25–50 min apart, 2 h 50 min in all. Every run id is saved in `fit/out/delivery/runs.json` as soon as it
   starts. A rerun collects those runs instead of starting new ones.
4. Downloads every run's output and writes one predictions CSV per delivery
   file. A run can report `completed` while its last output is still being
   written, so a run counts as completed only once every file's predictions
   reach that file's end: the log shows `platform: completed; outputs 24 of 25
   complete (… still being written)`, then `completed: all 25 outputs complete`.

A run's outputs don't name their inputs, so rows are matched to files by
their finish timestamp: becken-flt's cycles never overlap in time. The log's
last line counts files with predictions, windows, invalid windows, rows that
matched no file, and failed runs.

- **Why 25 files per run:** it's a batching choice, not something the delivery
  requires. `--files-per-run N` changes it.
  - **1 run of 122 files** has the least overhead, but one failure loses
    everything.
  - **122 runs of 1 file** limits a failure to one file, but every run pays
    the startup cost, about a minute in the one-cycle check.
  - **25 per run** sits between them: a failed run costs about 20% of the
    delivery.
  - A failed run is recorded in `runs.json` and the others are still
    collected. Restarting only the failed batch isn't automated yet.
- **Check first (optional):** `--only <cycle>` delivers just those cycles into
  `fit/out/delivery_check/`, never mixed with the real delivery. It's a cheap
  way to see the output format before the full run.
- **Time:** upload about 7–9 min (15 GB); runs about 2 min per full cotton cycle,
  2 h 50 min for all 122 files on dev and 1 h 33 min on production. It varies
  with the platform's load.

**Writes:** `fit/out/delivery/<file>.csv` (the platform's output rows:
`finish_timestamp`, `predicted_state`, `invalid`, …) and
`fit/out/delivery/runs.json`.

### Stage 7: score the delivery

```sh
python fit/baseline_bar.py --window 512 --step 512 --delivery   # the bar on the same windows, local
python fit/score_delivery.py                                    # Omega's delivery, local
```

**What `score_delivery.py` does:** pairs each prediction with the held-back
label (`data/roles/delivery_labels/`, row for row) at its window's last row,
the `last_record` rule used throughout. It leaves out and counts windows the
platform marked invalid, and any whose finish time matches no labelled row.
Scores are platform-style (macro-F1 over the four states), reported:
- pooled over all 106 cycles;
- without the two cycles listed on their own lines: `cold_cotton_40_2`, seen
  in exploration, and `cold_cotton_30_4`, with only 42% vibration coverage;
- for each of those two alone, per program family and per cycle.

It then prints the delivery number next to the bar on the same delivery
windows and Stage 5's test numbers: the same machine on new settings, against
a second, faulty unit of the same model.

**What `baseline_bar.py --delivery` does:** the same as `--test`, on
delivery. It scores the bar's validation-chosen setting (level k 51 l2),
never re-chosen, against the held-back labels.

**What you should see:** `all 106 cycles macro-F1 0.7008 fill 0.80 wash 0.94
spin 0.55 drain 0.51 windows 365,414`, 0 left out. For the bar: `all 106 delivery
cycles macro-F1 0.6927 fill 0.79 wash 0.93 spin 0.66 drain 0.38`.

**Writes:** `fit/out/delivery/scores.json`, `fit/out/bar_delivery_w512_s512.json`.

### If a result differs

- **A FAIL you have looked at and want to keep for now** goes in
  `prep/acknowledged.py`, with the reason. It is then still printed, but it
  no longer blocks the next stage.
- **On another deployment,** expect the same numbers to the fourth decimal,
  with a handful of windows predicted differently (5 of 432,999 on production),
  and a different Stage 4b draw.
- **Anything else that differs** from the expected results above means the
  inputs changed. Check `data/raw/` against `data/manifest.json`; Stage 1a's
  `files` check does exactly that.

## Repo layout

Following the Volve example:

| path | what it is |
|---|---|
| `prep/larco.py` | where LARCO's files are on Zenodo, and reading single members of its remote zips |
| `prep/split.py` | Stage 0a: the setting-group split → `data/split.json` |
| `prep/download.py` | Stage 0b: label and vibration files → `data/raw/`, plus `data/manifest.json` |
| `prep/preflight_raw.py` | Stage 1a: raw-cycle preflight → `data/preflight_raw.json` |
| `prep/states.py` | the four states and the per-second priority rule, shared by every stage |
| `prep/prepare.py` | Stage 1b: 200 Hz grid + state per row → `data/prepared/<cycle>.parquet`, `data/prepare_report.json` |
| `prep/preflight_prepared.py` | Stage 1c: prepared-file preflight, including a resampling tone test → `data/preflight_prepared.json` |
| `prep/preflight_common.py`, `prep/acknowledged.py` | the shared PASS / WARN / FAIL report, and the FAILs the user chose to keep |
| `prep/build_roles.py` | Stage 2: role files → `data/roles/` (library, validation, test, delivery, delivery_labels, `zscore_stats.json`, `manifest.json`) |
| `prep/preflight_roles.py` | Stage 3: role-file preflight against the platform's rules → `data/preflight_roles.json` |
| `fit/atai.py` | stdlib platform helpers: requests, file uploads, polling optimizations |
| `fit/probe_timestamps.py` | the 200 Hz timestamp-format probe on the Optimize API → `fit/out/probe_timestamps_<blueprint>.json` |
| `fit/baseline_bar.py` | Stage 4a: level / FFT kNN baselines, per window/step → `fit/out/bar_validation_w<W>_s<S>.json`; with `--test` / `--delivery`, the chosen bar on Stage 5's / Stage 7's windows |
| `fit/optimize.py` | Stages 4b and 4c: uploads (cached), optimizations on the Optimize API: random search over `--pool full` on the 6 search cycles, or one setting on all 21 (`--validation all`); platform only → `fit/out/optimize_<id>.json` |
| `fit/compare_bar.py` | optional: each trial against the bar on the same window/step and cycles, with its margin → `fit/out/compare_bar.json` |
| `fit/diagnose_wash_heating.py` | the five-state diagnostic behind folding heating into wash → `fit/out/five_states/diagnose_wash_heating.json` |
| `fit/probe_gaps.py` | the time-jump probe behind the one-file-per-state library → `fit/out/probe_gaps_<blueprint>.json` |
| `prep/background.py` | `--background` for long runs: nohup + caffeinate, output to a log |
| `prep/archive_roles.py` | pack `data/roles/` into Git LFS tar.xz parts under `data/archives/`, or unpack them (skip Stages 0–3) |
| `data/archives/` | the packed role files (Git LFS) and their `SHA256SUMS` |
| `fit/test.py` | Stage 5: promote the Stage 4c trial, test once with the Evals API → `fit/out/test.json` |
| `fit/deliver.py` | Stage 6: bundle from the Stage 5 blueprint, runs over becken-flt in batches → `fit/out/delivery/<file>.csv` |
| `fit/score_delivery.py` | Stage 7: the delivered predictions against the held-back labels → `fit/out/delivery/scores.json` |
| `fit/out/five_states/` | the superseded five-state bars, search results and logs |
| `data/` | the listing, split, licence, metadata and downloaded cycles (`data/raw/` gitignored) |
| `plan.md` | the plan and its decisions |

The exploration code and data are in `~/Downloads/demo/osm-candidates/`:
- `larco_baseline.py` and `larco_omega.py`: the exploration scripts;
- `larco/raw/`: one cycle per lab machine;
- `larco/emb/`: cached Omega embeddings;
- `larco/zip_range_get.py`: pulls single files out of the 13.5 GB
  vibration archive.

## Data attribution

All washing-machine data used here comes from the **LARCO** dataset, "Household
**La**undry Appliance **R**esource **C**onsumption and **O**peration Dataset":

- **Authors:** Žygimantas Jasiūnas, João Alexandre Braz Ferreira, Tiago
  Julião, José Cecílio, Guilherme Carrilho da Graça, Pedro M. Ferreira.
- **Dataset:** Zenodo, 2026, [doi:10.5281/zenodo.19666168](https://doi.org/10.5281/zenodo.19666168).
- **Paper:** *Scientific Data*, 2026, [doi:10.1038/s41597-026-07361-6](https://doi.org/10.1038/s41597-026-07361-6).
- **Licence:** `general.zip`, `vibrations.zip` and `audio.zip` are under
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/);
  the dataset's scripts are MIT, and its weather data ODbL 1.0. The dataset's
  `LICENCE.txt` is in `data/`, next to the downloaded files.
- **No endorsement.** The authors are not involved in this example and do not
  endorse it.

**What was changed** (as CC BY 4.0 requires):
- **selection:** the 199 cycles of the two Becken BWM5381IX units that have
  vibration; the 9 accelerometer channels only;
- **resampling:** every cycle resampled to an exact 200 Hz grid (cubic spline),
  with samples beyond ±2.2 g removed and cycles split at recording gaps;
- **labels:** derived per second from the dataset's measurements: fill / wash
  / spin / drain, with heating folded into its drum or water state;
- **scaling:** every channel z-scored with the library cycles' statistics;
- **layout:** assigned to roles (library, validation, test, delivery), the
  library cut into one file per state, and the delivery labels held back in
  separate files.

The packed role files in `data/archives/` are these derived files.

Code in this repository is Archetype AI's and carries no LARCO licence
obligation; the attribution above applies to the data.
