#!/usr/bin/env python3
"""Compare Omega trials with the bar, optionally, on the same windows.

Reads saved Optimize results (fit/out/optimize_<id>.json, from fit/optimize.py) and, for
each trial, the bar computed on that trial's window/step and cycles
(fit/out/bar_validation_w<window>_s<step>.json, from fit/baseline_bar.py). Each trial is
reported with its macro-F1 margin over the bar and its spin F1 next to the bar's, ranked
by margin. There is no pass/fail threshold: the +0.05 criterion was dropped after all
stages were scored (plan.md, "Success criterion dropped").

The Optimize step never needs this: a user without a sensible baseline just reads the
trial scores. Writes fit/out/compare_bar.json.

    python fit/compare_bar.py                         # every optimization in fit/out/
    python fit/compare_bar.py fit/out/optimize_opt_....json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai import trial_f1, trial_setting  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prep"))
from states import STATES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "fit", "out")


def bar_for(window, step, validation_set="search"):
    """The bar's best setting (chosen on the 6 search cycles), scored on the same cycles as the run."""
    path = os.path.join(OUT, f"bar_validation_w{window}_s{step}.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    best = d["best"]
    if validation_set == "all":      # Stage 4c: the same setting's score on all 21 validation cycles
        a = d["all_21"][best["features"]]
        best = {**best, "macro_f1": a["macro_f1"], "f1": a["f1"]}
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="*", help="optimize_<id>.json files (default: all in fit/out/)")
    args = ap.parse_args()
    paths = args.results or sorted(glob.glob(os.path.join(OUT, "optimize_opt_*.json")))
    if not paths:
        sys.exit("no optimization results in fit/out/ yet: run fit/optimize.py first")

    rows, missing, other_states = [], set(), set()
    for p in paths:
        run = json.load(open(p))
        for t in run["trials"]:
            w, st, k, metric, weights = trial_setting(t)
            f1, n = trial_f1(t)
            if f1 and set(f1) != set(STATES):     # e.g. a five-state run (fit/out/five_states/)
                other_states.add(run["optimization"]["id"])
                continue
            macro = t.get("objective_value")
            bar = bar_for(w, st, run.get("validation_set", "search"))
            if bar is None:
                missing.add((w, st))
            row = {"optimization": run["optimization"]["id"], "validation_set": run.get("validation_set", "search"),
                   "trial": t["trial_number"], "window": w, "step": st,
                   "k": k, "metric": metric, "weights": weights, "status": t["status"], "macro_f1": macro,
                   "f1": f1, "windows_scored": n}
            if bar is not None and macro is not None:
                row.update({"bar": {"features": bar["features"], "k": bar["k"], "metric": bar["metric"],
                                    "macro_f1": bar["macro_f1"], "spin_f1": bar["f1"]["spin"]},
                            "margin": round(macro - bar["macro_f1"], 4),
                            "spin_margin": round(f1.get("spin", 0) - bar["f1"]["spin"], 4)})
            rows.append(row)

    print(f"{len(rows)} trial(s) from {len(paths)} optimization(s); margin = Omega's macro-F1 minus the bar's, same windows\n")
    for r in sorted(rows, key=lambda r: -(r.get("margin") if r.get("margin") is not None else -9)):
        m = "   -  " if r["macro_f1"] is None else f"{r['macro_f1']:.4f}"
        if "bar" in r:
            vs = (f"bar {r['bar']['macro_f1']:.4f} (spin {r['bar']['spin_f1']:.2f})  margin {r['margin']:+.4f}  "
                  f"spin {r['spin_margin']:+.2f}")
        else:
            vs = "no bar for this window/step" if r["macro_f1"] is not None else r["status"]
        print(f"  {'all 21' if r['validation_set'] == 'all' else '6 search':<8} w={r['window']:<4} step={r['step']:<4} k={r['k']:<3} {r['metric']:<6} {r['weights']:<8} macro-F1 {m}  "
              + "  ".join(f"{s} {r['f1'].get(s, 0):.2f}" for s in STATES) + f"  | {vs}")
    if other_states:
        print(f"\nskipped {sorted(other_states)}: scored on other states than {STATES}")
    if missing:
        print("\nno bar yet for these window/step pairs; compute them with:")
        for w, st in sorted(missing):
            print(f"  python fit/baseline_bar.py --window {w} --step {st}")
    path = os.path.join(OUT, "compare_bar.json")
    with open(path, "w") as f:
        json.dump({"comparison": "Omega minus the bar, same windows; no threshold", "trials": rows}, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
