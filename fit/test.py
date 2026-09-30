#!/usr/bin/env python3
"""Stage 5: test the chosen setting once, on the 18 test cycles, with the Evals API.

Platform only (the bar on the same windows is fit/baseline_bar.py --test). The model is
the Stage 4c trial: window 512, step 512, k 31, cosine, uniform, trained on the 4 library
files (plan.md, Stage 4b decision). Steps, each recorded in fit/out/test_state.json so a
rerun resumes where it stopped:

  1. promote that trial to a blueprint (its settings and fitted classifier attached),
     or reuse the blueprint if it was already promoted;
  2. upload the 23 test files (cached like the other role files);
  3. two evals, submitted together: all 23 files (the test number), and the files of
     cold_cotton_40_2 alone, the one test cycle seen in exploration. The platform
     reports pooled confusion matrices only, so "without cold_cotton_40_2" is the first
     minus the second.

This is the one-shot test: once its number is seen, the setting and setup are frozen.
Writes fit/out/test_<eval id>.json per eval and fit/out/test.json (the summary).

    python fit/test.py --background              # detached, log fit/out/test.log
"""
import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai import TERMINAL, agents, load_dotenv, request, trial_setting  # noqa: E402
from optimize import OUT, ROLES, log, upload_all  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prep"))
from background import add_background_flag, maybe_detach  # noqa: E402
from states import STATES  # noqa: E402

OPT = "opt_1dztapszen8n1a89jj08fcfwrr"     # Stage 4c: one trial, the chosen setting
KEY = "osm-larco-w512-s512-cosine-k31-uniform"
SEEN = "wm_becken_BWM5381IX_cold_cotton_40_2.csv"   # seen in exploration (plan.md, split)
STATE = os.path.join(OUT, "test_state.json")


def blueprint(trial):
    try:
        return request("GET", f"{agents()}/blueprints/{KEY}")
    except RuntimeError:
        pass  # not promoted yet
    w, st, k, metric, weights = trial_setting(trial)
    bp = request("POST", f"{agents()}/optimizations/{OPT}/trials/{trial['id']}/promote", body={
        "blueprint_key": KEY, "name": f"OSM LARCO washing machine (w{w}, s{st}, {metric}, k{k}, {weights})",
        "description": f"Stage 5 model: window {w}, step {st}, {metric}, k {k}, {weights}; "
                       f"library = 4 files, 400 windows per state, 54 becken cycles."})
    log(f"promoted {trial['id']} -> {bp['blueprint_key']} ({bp['id']})")
    return bp


def matrix(ev):
    """Confusion matrix in STATES order (rows = true), from an eval's report."""
    s = ev["metrics_report"]["targets"]["state"]
    idx = [s["class_names"].index(x) for x in STATES]
    cm = s["confusion_matrix"]
    return [[cm[i][j] for j in idx] for i in idx]


def scores(cm):
    f1 = {}
    for i, s in enumerate(STATES):
        tp, row, col = cm[i][i], sum(cm[i]), sum(r[i] for r in cm)
        f1[s] = round(2 * tp / (row + col), 4) if row + col else 0.0
    return {"macro_f1": round(sum(f1.values()) / len(f1), 4), "f1": f1, "windows": sum(map(sum, cm)), "confusion": cm}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upload-jobs", type=int, default=3)
    add_background_flag(ap, default_log="fit/out/test.log")
    args = ap.parse_args()
    maybe_detach(args)
    load_dotenv()
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    save = lambda: json.dump(state, open(STATE, "w"), indent=1)  # noqa: E731

    trials = request("GET", f"{agents()}/optimizations/{OPT}/trials")["data"]
    if len(trials) != 1 or trials[0]["status"] != "completed":
        sys.exit(f"{OPT} should hold one completed trial")
    trial = trials[0]
    w, st, k, metric, weights = trial_setting(trial)
    log(f"model: {OPT} trial {trial['id']} (w {w}, step {st}, k {k}, {metric}, {weights}; "
        f"all-21 validation macro-F1 {trial['objective_value']:.4f})")
    if "blueprint" not in state:
        bp = blueprint(trial)
        state["blueprint"] = {"id": bp["id"], "key": bp["blueprint_key"]}
        save()

    manifest = json.load(open(os.path.join(ROLES, "manifest.json")))
    files = manifest["test"]["files"]
    paths = {f["file"]: os.path.join(ROLES, f["file"]) for f in files}
    seen = [f["file"] for f in files if f["cycle"] == SEEN]
    ids = upload_all(list(paths.values()), args.upload_jobs)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def example(name):
        return {"name": os.path.basename(name)[:-4], "inputs": [{"type": "file", "id": ids[paths[name]], "format": "csv"}],
                "ground_truth": {"state": {"from": {"column": "label"}, "downsampling": "last_record"}}}

    for part, names in (("all", list(paths)), ("seen", seen)):
        if part not in state.get("evals", {}):
            ev = request("POST", f"{agents()}/evals", body={
                "blueprint_id": state["blueprint"]["id"], "name": f"LARCO Stage 5 test ({part}) {stamp}",
                "emit_predictions": False, "examples": [example(n) for n in names]})
            state.setdefault("evals", {})[part] = ev["id"]
            save()
            log(f"eval {ev['id']} ({part}): {len(names)} test file(s)")

    done = {}
    last = None
    while len(done) < len(state["evals"]):
        for part, eid in state["evals"].items():
            if part not in done:
                ev = request("GET", f"{agents()}/evals/{eid}")
                if ev["status"] in TERMINAL:
                    done[part] = ev
                    json.dump(ev, open(os.path.join(OUT, f"test_{eid}.json"), "w"), indent=1)
        now = {p: ("done" if p in done else "running") for p in state["evals"]}
        if now != last:
            log(f"  evals: {now}")
            last = now
        if len(done) < len(state["evals"]):
            time.sleep(60)
    bad = {p: ev["status"] for p, ev in done.items() if ev["status"] != "completed" or not ev.get("metrics_report")}
    if bad:
        sys.exit(f"evals did not complete: {bad} (see fit/out/test_<eval id>.json)")

    cm_all, cm_seen = matrix(done["all"]), matrix(done["seen"])
    cm_rest = [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(cm_all, cm_seen)]
    res = {"model": {"optimization": OPT, "trial": trial["id"], "blueprint": state["blueprint"],
                     "window": w, "step": st, "k": k, "metric": metric, "weights": weights},
           "evals": state["evals"], "all": scores(cm_all), "without_separate": scores(cm_rest),
           "separate": {SEEN: scores(cm_seen)}}
    json.dump(res, open(os.path.join(OUT, "test.json"), "w"), indent=1)
    print(f"\nStage 5 test, w {w} / step {st}, k {k}, {metric}, {weights}:")
    for name, r in (("all 18 test cycles", res["all"]), (f"without {SEEN[3:-4]}", res["without_separate"]),
                    (f"{SEEN[3:-4]} alone", res["separate"][SEEN])):
        print(f"  {name:<44} macro-F1 {r['macro_f1']:.4f}  " + "  ".join(f"{s} {r['f1'][s]:.2f}" for s in STATES)
              + f"  windows {r['windows']:,}")
    print("confusion, all 18 (rows = true, cols = predicted):")
    print("  " + "".join(f"{s:>9}" for s in STATES))
    for s, row in zip(STATES, cm_all):
        print(f"  {s:<7}" + "".join(f"{v:>9,}" for v in row))
    log(f"wrote {os.path.join(OUT, 'test.json')}; the bar on the same windows: python fit/baseline_bar.py --test")


if __name__ == "__main__":
    main()
