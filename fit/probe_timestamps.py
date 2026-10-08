#!/usr/bin/env python3
"""Probe: which CSV timestamp format does the platform accept at 200 Hz?

Earlier OSM data used whole epoch seconds (5 s rows). LARCO needs 5 ms steps,
so this sends the same tiny data three ways and runs one 1-trial optimization
per format on the Optimize API (they run in parallel):

  epoch_s    1687943467.315             fractional epoch seconds
  epoch_ms   1687943467315              integer epoch milliseconds
  iso        2023-06-28T09:11:07.315Z   ISO 8601, milliseconds

Data, from one prepared becken cycle (library, so it informs no test score):
three library files of ~2 min of pure fill, wash and spin (one state per
file), and one validation file of a continuous ~4 min stretch across a state
change, labelled by a `label` column. Window = step = 1024 (5.12 s), k = 5, l1.
The timestamp is the first column, which the osm blueprint uses by default.
Writes the files to fit/out/probe_timestamps/ and results to
fit/out/probe_timestamps_<blueprint id>.json.

    .venv/bin/python fit/probe_timestamps.py            # build, upload, run, report (~15-20 min)
    .venv/bin/python fit/probe_timestamps.py --dry-run  # build the files only
    .venv/bin/python fit/probe_timestamps.py --background   # detached (nohup + caffeinate), log fit/out/probe_timestamps.log
"""
import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prep"))
from atai import agents, load_dotenv, request, states_override, upload_file, wait_optimization  # noqa: E402
from states import CHANNELS  # noqa: E402
from background import add_background_flag, maybe_detach  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CYCLE = "wm_becken_BWM5381IX_cold_cotton_40_4"
OUT = os.path.join(ROOT, "fit", "out", "probe_timestamps")
LIB_ROWS = 200 * 120          # 2 min
VAL_ROWS = 200 * 240          # 4 min
# Pass --blueprint to pin a version.
BLUEPRINT_ID = "osm"     # the canonical blueprint's key: each deployment resolves it to its own blp_ id
FORMATS = {
    "epoch_s": lambda ms: [f"{m // 1000}.{m % 1000:03d}" for m in ms],
    "epoch_ms": lambda ms: [str(m) for m in ms],
    "iso": lambda ms: [datetime.datetime.fromtimestamp(m / 1000, datetime.timezone.utc)
                       .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" for m in ms],
}


def pick(d):
    """Pure single-state stretches for the library, one continuous mixed stretch for validation."""
    runs = ((d.state != d.state.shift()) | (d.segment != d.segment.shift())).cumsum()
    library = {}
    for st in ("fill", "wash", "spin"):
        best = max((g for _, g in d[d.state == st].groupby(runs[d.state == st])), key=len)
        if len(best) < LIB_ROWS:
            sys.exit(f"no {st} stretch of {LIB_ROWS} rows in {CYCLE}")
        mid = len(best) // 2
        library[st] = best.iloc[mid - LIB_ROWS // 2: mid + LIB_ROWS // 2]
    fill = d.index[d.state == "fill"]
    start = max(0, fill[0] - VAL_ROWS // 4)            # a stretch starting before the first fill
    val = d.iloc[start:start + VAL_ROWS]
    if val.segment.nunique() != 1:
        sys.exit("validation stretch crosses a segment gap")
    return library, val


def write(df, path, fmt, label):
    ms = df.timestamp.astype("int64").to_numpy()       # milliseconds (datetime64[ms])
    out = pd.DataFrame({"timestamp": FORMATS[fmt](ms)})
    for c in CHANNELS:
        out[c] = df[c].to_numpy()
    if label:
        out["label"] = df.state.astype(str).to_numpy()
    out.to_csv(path, index=False, float_format="%.4f")


def run_format(fmt, files, stamp, bp):
    ids = {k: upload_file(p, rename=f"larco-probe-{fmt}-{os.path.basename(p)[:-4]}-{stamp}.csv")["file_id"]
           for k, p in files.items()}
    training = [{"name": f"{fmt}-{st}", "inputs": [{"type": "file", "id": ids[st], "format": "csv"}],
                 "ground_truth": {"state": {"from": {"constant": st}}}} for st in ("fill", "wash", "spin")]
    validation = [{"name": f"{fmt}-validation", "inputs": [{"type": "file", "id": ids["validation"], "format": "csv"}],
                   "ground_truth": {"state": {"from": {"column": "label"}, "downsampling": "last_record"}}}]
    one = lambda v: {"type": "categorical", "values": [v]}
    space = {"parameters": {
        "window_size": {"kind": "value", "spec": one(1024)},
        "step_size": {"kind": "value", "spec": one(1024)},
        "k_neighbors": {"kind": "fitting", "spec": one(5)},
        "metric": {"kind": "fitting", "spec": one("l1")},
        "weights": {"kind": "fitting", "spec": one("uniform")},
    }}
    opt = request("POST", f"{agents()}/optimizations", body={
        "name": f"LARCO timestamp probe {fmt} {stamp}", "blueprint_id": bp["id"],
        "objective": "macro_f1", "search_space": space, "budget": {"max_trials": 1},
        "training_examples": training, "validation_examples": validation, **states_override(bp, training)})
    print(f"[{fmt}] optimization {opt['id']} created", flush=True)
    opt, trials = wait_optimization(opt["id"], label=f"[{fmt}]", log=lambda m: print(m, flush=True))
    return {"format": fmt, "file_ids": ids, "optimization": opt, "trials": trials}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--blueprint", default=BLUEPRINT_ID, help=f"osm blueprint id (default: {BLUEPRINT_ID})")
    add_background_flag(ap, default_log="fit/out/probe_timestamps.log")
    args = ap.parse_args()
    maybe_detach(args)

    d = pd.read_parquet(os.path.join(ROOT, "data", "prepared", CYCLE + ".parquet"))
    library, val = pick(d)
    os.makedirs(OUT, exist_ok=True)
    files = {}
    for fmt in FORMATS:
        files[fmt] = {}
        for st, part in library.items():
            files[fmt][st] = os.path.join(OUT, f"{fmt}__{st}.csv")
            write(part, files[fmt][st], fmt, label=False)
        files[fmt]["validation"] = os.path.join(OUT, f"{fmt}__validation.csv")
        write(val, files[fmt]["validation"], fmt, label=True)
    counts = val.state.value_counts().to_dict()
    print(f"built {sum(len(v) for v in files.values())} files in {OUT}: library 3 x {LIB_ROWS} rows; "
          f"validation {len(val)} rows {counts}")
    for fmt in FORMATS:
        with open(files[fmt]["validation"]) as f:
            print(f"  {fmt:<9} {f.readline().strip()[:60]} | {f.readline().strip()[:60]}")
    if args.dry_run:
        return

    load_dotenv()
    bp = request("GET", f"{agents()}/blueprints/{args.blueprint}")
    blueprint_id = bp["id"]
    print(f"blueprint {blueprint_id}")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with ThreadPoolExecutor(len(FORMATS)) as pool:
        results = list(pool.map(lambda f: run_format(f, files[f], stamp, bp), FORMATS))

    path = os.path.join(ROOT, "fit", "out", f"probe_timestamps_{blueprint_id}.json")
    with open(path, "w") as f:
        json.dump({"cycle": CYCLE, "blueprint": blueprint_id, "results": results}, f, indent=1, default=str)
    print(f"results: {path}")
    print("\nformat     optimization status   trial status  macro-F1  windows  error")
    for r in results:
        t = r["trials"][0] if r["trials"] else {}
        cm = (t.get("metrics_report") or {}).get("targets", {}).get("state", {}).get("confusion_matrix")
        r["windows_scored"] = sum(map(sum, cm)) if cm else None
        score = t.get("objective_value")
        print(f"{r['format']:<10} {r['optimization']['status']:<21} {t.get('status', '-'):<13} "
              f"{score if score is None else round(score, 4)!s:<9} {r['windows_scored']!s:<8} {(t.get('error') or '')[:200]}")


if __name__ == "__main__":
    main()
