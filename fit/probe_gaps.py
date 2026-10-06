#!/usr/bin/env python3
"""Probe: how the Optimizations and Evals APIs treat time gaps inside CSV files, and whether loosening helps.

Option A for the 1 MiB config limit (Stage 4b) puts each state's library windows into
one file, with real timestamps, so the file has forward jumps between 1,024-row pieces. This
tests that before anything is rebuilt. Tiny files from one library cycle; each library state
is its own file, so a rejected file shows up as a state that is never predicted.

  run  library     validation  step  tolerance  question
  O0   continuous  continuous  1024  default    control
  O1   seams       continuous  1024  default    aligned seams in training files
  O2   seams       continuous  512   default    training windows that cross a seam
  O3   seams       continuous  512   loosened   does loosening rescue them?
  O4   continuous  seams       1024  default    aligned seams in validation
  O5   continuous  seams       512   default    validation windows that cross a seam
  O6   continuous  seams       512   loosened   ... loosened
  E4-E6  promote O4-O6, then POST /agents/evals on the seamed validation file (Evals inherit the
         promoted trial's values)

"Loosened" = sample_rate_interval_tolerance 10.0 and max_temporal_gap 10.0 as single-point
float_range values; validate_monotonic_timestamps is a boolean,
which the search space can't set, and forward jumps are still increasing anyway.
Window 1024, k 5, l1. Writes the files to fit/out/probe_gaps/ and results to
fit/out/probe_gaps_<blueprint>.json.

    python fit/probe_gaps.py --dry-run        # build the files, print the expected window counts
    python fit/probe_gaps.py --background     # ~5-10 min on dev
"""
import argparse
import datetime
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prep"))
from atai import agents, list_trials, load_dotenv, request, trial_f1, upload_file, wait_optimization  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402
from states import CHANNELS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CYCLE = "wm_becken_BWM5381IX_cold_cotton_40_4"
OUT = os.path.join(ROOT, "fit", "out", "probe_gaps")
BLUEPRINT_ID = "osm"     # the canonical blueprint's key: each deployment resolves it to its own blp_ id
W = 1024
PIECES = 4                     # per library file
LIB_STATES = ["fill", "wash", "spin"]
VAL_PIECES = 12                # seamed validation: 4 pieces of each library state, time-ordered
LOOSE = {"sample_rate_interval_tolerance": 10.0, "max_temporal_gap": 10.0}
RUNS = {  # name: (library, validation, step, loosened)
    "O0": ("continuous", "continuous", 1024, False),
    "O1": ("seams", "continuous", 1024, False),
    "O2": ("seams", "continuous", 512, False),
    "O3": ("seams", "continuous", 512, True),
    "O4": ("continuous", "seams", 1024, False),
    "O5": ("continuous", "seams", 512, False),
    "O6": ("continuous", "seams", 512, True),
}
EVALS = ["O4", "O5", "O6"]


def log(msg):
    print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def runs_of(d, state):
    """(start, length) of maximal single-state, single-segment row runs."""
    st, seg = d.state.astype(str).to_numpy(), d.segment.to_numpy()
    brk = np.r_[True, (st[1:] != st[:-1]) | (seg[1:] != seg[:-1])]
    starts = np.flatnonzero(brk)
    lengths = np.diff(np.r_[starts, len(st)])
    return [(int(s), int(n)) for s, n in zip(starts, lengths) if st[s] == state]


def write(df, path, label):
    ms = df.timestamp.astype("int64").to_numpy()
    out = pd.DataFrame({"timestamp": [f"{m // 1000}.{m % 1000:03d}" for m in ms]})
    for c in CHANNELS:
        out[c] = df[c].to_numpy()
    if label:
        out["label"] = df.state.astype(str).to_numpy()
    out.to_csv(path, index=False, float_format="%.4f")


def windows(n_rows, step, seams):
    """(total windows, windows that contain a seam) for a file of `n_rows` with seams every W rows."""
    starts = range(0, n_rows - W + 1, step)
    crossing = sum(1 for s in starts if seams and s % W != 0)
    return len(starts), crossing


def build():
    d = pd.read_parquet(os.path.join(ROOT, "data", "prepared", CYCLE + ".parquet"))
    os.makedirs(OUT, exist_ok=True)
    files, pieces_of = {}, {}
    for state in LIB_STATES:
        runs = runs_of(d, state)
        longest = max(runs, key=lambda r: r[1])
        if longest[1] < PIECES * W:
            sys.exit(f"{state}: no run of {PIECES * W} rows")
        s0 = longest[0] + (longest[1] - PIECES * W) // 2
        cont = d.iloc[s0:s0 + PIECES * W]
        # seams: one whole window from each of PIECES different runs, spread through the cycle
        cands = [(s, n) for s, n in runs if n >= W]
        pick = [cands[i] for i in np.unique(np.linspace(0, len(cands) - 1, PIECES).round().astype(int))]
        if len(pick) < PIECES:     # too few runs: take windows spread along the longest run instead
            pick = [(longest[0] + i * (longest[1] // PIECES), W) for i in range(PIECES)]
        parts = [d.iloc[s + (n - W) // 2: s + (n - W) // 2 + W] for s, n in pick]
        seam = pd.concat(parts)
        for kind, part in (("continuous", cont), ("seams", seam)):
            files[f"lib_{kind}_{state}"] = os.path.join(OUT, f"lib_{kind}__{state}.csv")
            write(part, files[f"lib_{kind}_{state}"], label=False)
        pieces_of[state] = parts
    # continuous validation: 4 min around the first fill (wash -> fill), as in the timestamp probe
    fill0 = d.index[d.state == "fill"][0]
    val_cont = d.iloc[max(0, fill0 - 12000):max(0, fill0 - 12000) + 48000]
    files["val_continuous"] = os.path.join(OUT, "val_continuous.csv")
    write(val_cont, files["val_continuous"], label=True)
    # seamed validation: every library piece, in time order (pure pieces, so labels are clean)
    val_seam = pd.concat(sorted((p for ps in pieces_of.values() for p in ps), key=lambda p: p.timestamp.iloc[0]))
    files["val_seams"] = os.path.join(OUT, "val_seams.csv")
    write(val_seam, files["val_seams"], label=True)
    rows = {"val_continuous": len(val_cont), "val_seams": len(val_seam)}
    ts = val_seam.timestamp.astype("int64").to_numpy()
    jumps = np.diff(ts)[np.diff(ts) != 5]
    return files, rows, jumps


def space(step, loose):
    one = lambda v: {"type": "categorical", "values": [v]}
    p = {"window_size": {"kind": "value", "spec": one(W)}, "step_size": {"kind": "value", "spec": one(step)},
         "k_neighbors": {"kind": "fitting", "spec": one(5)}, "metric": {"kind": "fitting", "spec": one("l1")},
         "weights": {"kind": "fitting", "spec": one("uniform")}}
    if loose:
        p.update({k: {"kind": "value", "spec": {"type": "float_range", "min": v, "max": v}} for k, v in LOOSE.items()})
    return {"parameters": p}


def run_opt(name, ids, blueprint_id, stamp):
    lib, val, step, loose = RUNS[name]
    training = [{"name": f"{name}-{s}", "inputs": [{"type": "file", "id": ids[f"lib_{lib}_{s}"], "format": "csv"}],
                 "ground_truth": {"state": {"from": {"constant": s}}}} for s in LIB_STATES]
    validation = [{"name": f"{name}-val", "inputs": [{"type": "file", "id": ids[f"val_{val}"], "format": "csv"}],
                   "ground_truth": {"state": {"from": {"column": "label"}, "downsampling": "last_record"}}}]
    try:
        opt = request("POST", f"{agents()}/optimizations", body={
            "name": f"LARCO gap probe {name} {stamp}", "blueprint_id": blueprint_id, "objective": "macro_f1",
            "search_space": space(step, loose), "budget": {"max_trials": 1},
            "training_examples": training, "validation_examples": validation})
    except RuntimeError as e:
        log(f"[{name}] create rejected: {e}")
        return {"run": name, "create_error": str(e)}
    log(f"[{name}] optimization {opt['id']} created")
    opt, trials = wait_optimization(opt["id"], label=f"[{name}]", every_s=20, log=log)
    return {"run": name, "optimization": opt, "trials": trials}


def run_eval(name, opt_result, ids, stamp):
    t = (opt_result.get("trials") or [{}])[0]
    if t.get("status") != "completed":
        return {"run": f"E{name[1:]}", "skipped": f"{name} trial not completed ({t.get('status')})"}
    key = f"larco-probe-gaps-{name.lower()}-{stamp.lower()}"
    try:
        bp = request("POST", f"{agents()}/optimizations/{t['optimization_id']}/trials/{t['id']}/promote", body={
            "blueprint_key": key, "name": f"LARCO gap probe {name}",
            "description": "Throwaway probe blueprint (fit/probe_gaps.py): time gaps in eval files."})
    except RuntimeError as e:
        return {"run": f"E{name[1:]}", "promote_error": str(e)}
    log(f"[E{name[1:]}] promoted {name} -> {bp.get('blueprint_key')} ({bp.get('id')})")
    try:
        ev = request("POST", f"{agents()}/evals", body={
            "blueprint_id": bp["id"], "name": f"LARCO gap probe E{name[1:]} {stamp}",
            "examples": [{"name": f"E{name[1:]}-val_seams",
                          "inputs": [{"type": "file", "id": ids["val_seams"], "format": "csv"}],
                          "ground_truth": {"state": {"from": {"column": "label"}, "downsampling": "last_record"}}}]})
    except RuntimeError as e:
        return {"run": f"E{name[1:]}", "blueprint": bp, "eval_create_error": str(e)}
    eid = ev["id"]
    while ev["status"] not in ("completed", "failed", "canceled", "cancelled"):
        time.sleep(20)
        ev = request("GET", f"{agents()}/evals/{eid}")
    log(f"[E{name[1:]}] eval {eid}: {ev['status']} {ev.get('error') or ''}")
    return {"run": f"E{name[1:]}", "blueprint": bp, "eval": ev}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--blueprint", default=BLUEPRINT_ID)
    ap.add_argument("--no-evals", action="store_true", help="skip the promote + eval part")
    add_background_flag(ap, default_log="fit/out/probe_gaps.log")
    args = ap.parse_args()
    maybe_detach(args)

    files, rows, jumps = build()
    log(f"built {len(files)} files in {OUT}; seamed validation has {len(jumps)} jumps "
        f"({(jumps.min() / 1000):.0f} s to {(jumps.max() / 1000):.0f} s)")
    expected = {}
    for name, (lib, val, step, loose) in RUNS.items():
        n, cross = windows(rows[f"val_{val}"], step, seams=(val == "seams"))
        lib_n, lib_cross = windows(PIECES * W, step, seams=(lib == "seams"))
        expected[name] = {"validation_windows": n, "validation_windows_crossing_a_seam": cross,
                          "library_windows_per_state": lib_n, "library_windows_crossing_a_seam": lib_cross}
        log(f"  {name}: library {lib:<10} validation {val:<10} step {step:<4} {'loosened' if loose else 'default ':<8} "
            f"| validation windows {n} ({cross} cross a seam) | library windows per state {lib_n} ({lib_cross} cross a seam)")
    if args.dry_run:
        return

    load_dotenv()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ids = {k: upload_file(p, rename=f"larco-probe-gaps-{os.path.basename(p)[:-4]}-{stamp}.csv")["file_id"]
           for k, p in files.items()}
    blueprint_id = request("GET", f"{agents()}/blueprints/{args.blueprint}")["id"]
    log(f"uploaded {len(ids)} files; blueprint {blueprint_id}")
    with ThreadPoolExecutor(len(RUNS)) as pool:
        opts = dict(zip(RUNS, pool.map(lambda n: run_opt(n, ids, blueprint_id, stamp), RUNS)))
    evals = {}
    if not args.no_evals:
        with ThreadPoolExecutor(len(EVALS)) as pool:
            evals = dict(zip(EVALS, pool.map(lambda n: run_eval(n, opts[n], ids, stamp), EVALS)))

    print("\nrun  status     trial      windows scored / expected  never predicted  macro-F1  error")
    for name, r in opts.items():
        t = (r.get("trials") or [{}])[0]
        f1, n = trial_f1(t)
        s = (t.get("metrics_report") or {}).get("targets", {}).get("state", {})
        cm, names = s.get("confusion_matrix"), s.get("class_names") or []
        never = [c for j, c in enumerate(names) if cm and sum(row[j] for row in cm) == 0] if cm else []
        print(f"{name:<4} {(r.get('optimization') or {}).get('status', 'rejected'):<10} {t.get('status', '-'):<10} "
              f"{n:>5} / {expected[name]['validation_windows']:<16} {','.join(never) or '-':<16} "
              f"{t.get('objective_value') if t.get('objective_value') is None else round(t['objective_value'], 4)!s:<9} "
              f"{r.get('create_error') or t.get('error') or ''}")
    for name, r in evals.items():
        ev = r.get("eval") or {}
        s = (ev.get("metrics_report") or {}).get("targets", {}).get("state", {})
        cm = s.get("confusion_matrix")
        n = sum(map(sum, cm)) if cm else 0
        print(f"{r['run']:<4} eval {ev.get('status', '-'):<10} windows scored {n} / {expected[name]['validation_windows']}  "
              f"{r.get('skipped') or r.get('promote_error') or r.get('eval_create_error') or ev.get('error') or ''}")
    path = os.path.join(ROOT, "fit", "out", f"probe_gaps_{blueprint_id}.json")
    with open(path, "w") as f:
        json.dump({"cycle": CYCLE, "expected": expected, "runs": opts, "evals": evals}, f, indent=1, default=str)
    log(f"results: {path}")


if __name__ == "__main__":
    main()
