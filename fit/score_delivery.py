#!/usr/bin/env python3
"""Stage 7: score the delivered predictions against the labels held back.

The delivery files went out without labels (Stage 6); their labels were kept in
data/roles/delivery_labels/, row for row. Each prediction is paired with the label of its
window's last row (its finish_timestamp: the `last_record` rule used throughout). Windows
the platform marked invalid, or whose finish time matches no labelled row, are left out
and counted. Scored as the platform scores: macro-F1 over all four states, a state with
no windows and no predictions counting 0.

Reported pooled over all 106 becken-flt cycles, without the two listed on their own lines
(cold_cotton_40_2, seen in exploration; cold_cotton_30_4, 42% vibration coverage), per
program family and per cycle, next to Stage 5's test number and, if computed, the bar on
the same delivery windows (fit/baseline_bar.py --delivery). Local only.

Reads fit/out/delivery/<file>.csv; writes fit/out/delivery/scores.json.

    python fit/score_delivery.py
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_bar import SEPARATE, family, scores  # noqa: E402
from deliver import DELIVERY, to_ms  # noqa: E402
from optimize import OUT, ROLES  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prep"))
from states import STATES  # noqa: E402


def pairs_of(pred_path):
    """(true, predicted) per scored window of one delivery file, and what was left out."""
    name = os.path.basename(pred_path)
    pred = pd.read_csv(pred_path, dtype=str)
    side = pd.read_csv(os.path.join(ROLES, "delivery_labels", name), dtype=str, engine="pyarrow")
    label = dict(zip((side.timestamp.astype(float) * 1000).round().astype(np.int64), side.label))
    skipped = Counter()
    if "invalid" in pred:
        bad = pred.invalid.str.lower() == "true"
        skipped["invalid"] = int(bad.sum())
        pred = pred[~bad]
    t = [label.get(to_ms(v)) for v in pred.finish_timestamp]
    keep = np.array([x is not None for x in t])
    skipped["no label at finish time"] = int((~keep).sum())
    return np.array(t, dtype=object)[keep], pred.predicted_state.to_numpy(dtype=object)[keep], skipped


def cycle_of(name):
    return "wm_" + name.split("__seg")[0] + ".csv"


def main():
    paths = sorted(p for p in glob.glob(os.path.join(DELIVERY, "*.csv")))
    if not paths:
        sys.exit(f"no predictions in {DELIVERY}/: run fit/deliver.py first")
    manifest = json.load(open(os.path.join(ROLES, "manifest.json")))
    expected = {os.path.basename(f["file"]) for f in manifest["delivery"]["files"]}
    missing = sorted(expected - {os.path.basename(p) for p in paths})

    ys, ps, cyc, skipped = [], [], [], Counter()
    for p in paths:
        y, pr, sk = pairs_of(p)
        ys.append(y)
        ps.append(pr)
        cyc.append(np.array([cycle_of(os.path.basename(p))] * len(y)))
        skipped.update(sk)
    y, p, cyc = np.concatenate(ys), np.concatenate(ps), np.concatenate(cyc)
    unknown = sorted(set(p) - set(STATES))
    sep = SEPARATE["delivery"]
    rest = ~np.isin(cyc, sep)
    fam = np.array([family(c) for c in cyc])
    out = {"files": len(paths), "files_missing": missing, "cycles": len(set(cyc)), "skipped": dict(skipped),
           "unknown_predictions": unknown, "all": scores(y, p), "without_separate": scores(y[rest], p[rest]),
           "separate": {c: scores(y[cyc == c], p[cyc == c]) for c in sep if (cyc == c).any()},
           "per_family": {f: scores(y[fam == f], p[fam == f]) for f in ("cotton", "eco", "other") if (fam == f).any()},
           "per_cycle": {c: scores(y[cyc == c], p[cyc == c])["macro_f1"] for c in sorted(set(cyc))}}

    print(f"Stage 7: {out['cycles']} becken-flt cycles, {len(paths)} files, {len(y):,} windows scored; "
          f"left out {dict(skipped)}")
    if missing:
        print(f"  WARNING: {len(missing)} delivery file(s) have no predictions: {missing[:3]}{' ...' if len(missing) > 3 else ''}")
    if unknown:
        print(f"  WARNING: predictions outside the four states: {unknown}")
    rows = ([(f"all {out['cycles']} cycles", out["all"]), (f"without the {len(sep)} below", out["without_separate"])]
            + [(f"{c[3:-4]} alone", r) for c, r in out["separate"].items()]
            + [(f"family {f}", r) for f, r in out["per_family"].items()])
    for name, r in rows:
        print(f"  {name:<44} macro-F1 {r['macro_f1']:.4f}  " + "  ".join(f"{s} {r['f1'][s]:.2f}" for s in STATES)
              + f"  windows {r['windows']:,}")

    def pooled(path):
        if not os.path.exists(path):
            return None
        d = json.load(open(path))
        return (d.get("all") or d.get("all_18"))["macro_f1"]
    test_omega, test_bar = pooled(os.path.join(OUT, "test.json")), pooled(os.path.join(OUT, "bar_test_w512_s512.json"))
    bar_path = os.path.join(OUT, "bar_delivery_w512_s512.json")
    bar = json.load(open(bar_path)) if os.path.exists(bar_path) else None
    print("\nnext to the other numbers (macro-F1, all cycles pooled):")
    print(f"  delivery, becken-flt   Omega {out['all']['macro_f1']:.4f}"
          + (f"   bar {bar['all']['macro_f1']:.4f}   margin {out['all']['macro_f1'] - bar['all']['macro_f1']:+.4f}"
             if bar else "   bar: run fit/baseline_bar.py --window 512 --step 512 --delivery"))
    if test_omega is not None or test_bar is not None:
        print(f"  test, becken (Stage 5) Omega {test_omega if test_omega is not None else '-'}   bar {test_bar if test_bar is not None else '-'}")
    out["compare"] = {"test_omega": test_omega, "test_bar": test_bar,
                      "delivery_bar": bar["all"]["macro_f1"] if bar else None}
    path = os.path.join(DELIVERY, "scores.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
