#!/usr/bin/env python3
"""Why wash and heating are confused: is heating visible in the vibration at all?

On the 21 validation cycles only (test untouched), windows cut as fit/baseline_bar.py
cuts them (last_record labels):

  1. Timeline: where heating sits in each cycle (runs, lengths, neighbours).
  2. Separability: wash vs heating only, kNN on the bar's features, leave one cycle
     out. Balanced accuracy near 0.5 means vibration alone can't tell them apart
     in a cycle the model hasn't seen.
  3. Features: per-feature effect size (heating minus wash, in wash SDs) per cycle,
     and whether its sign agrees across cycles.

It ran on the five-state role files (2026-09-29), the reason for folding heating into
wash (prep/states.py). Today's role files carry no heating label, so it only reruns on a
five-state build. Writes fit/out/five_states/diagnose_wash_heating.json.

    python fit/diagnose_wash_heating.py                         # window 1024, step 1024
    python fit/diagnose_wash_heating.py --window 512 --step 256
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_bar as bb  # noqa: E402

ROLES = bb.ROLES
OUT = bb.OUT
HZ = 200


def runs(labels):
    """(state, start_row, length) for each run of equal labels."""
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(labels)]
    return [(labels[a], int(a), int(b - a)) for a, b in zip(starts, ends)]


def timeline(path):
    lab = pd.read_csv(path, usecols=["label"], engine="pyarrow").label.to_numpy()
    rs = runs(lab)
    heat = [(i, r) for i, r in enumerate(rs) if r[0] == "heating"]
    return {
        "rows": int(len(lab)),
        "share": {s: round(float((lab == s).mean()), 4) for s in bb.STATES},
        "heating_runs": [{"start_min": round(r[1] / HZ / 60, 1), "minutes": round(r[2] / HZ / 60, 2),
                          "before": rs[i - 1][0] if i else None,
                          "after": rs[i + 1][0] if i + 1 < len(rs) else None} for i, r in heat],
    }


def balanced_accuracy(y, p):
    return float(np.mean([(p[y == s] == s).mean() for s in ("wash", "heating") if (y == s).any()]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--step", type=int, default=1024)
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()
    bb.WINDOW, bb.STEP = args.window, args.step

    val_paths = sorted(glob.glob(os.path.join(ROLES, "validation", "*.csv")))
    cycle_of = lambda s: s.split("__seg")[0]  # noqa: E731

    print("1. where heating sits (validation cycles)")
    tl = {}
    for p in val_paths:
        c = cycle_of(os.path.basename(p))
        t = timeline(p)
        tl.setdefault(c, []).append(t)
        hr = t["heating_runs"]
        desc = ", ".join(f"{h['minutes']} min at {h['start_min']} ({h['before']}→·→{h['after']})" for h in hr) or "none"
        print(f"  {c[len('becken_BWM5381IX_'):]:<24} heating {t['share']['heating']:.1%}  wash {t['share']['wash']:.1%}  runs: {desc}")

    F, y, src = bb.load(val_paths, labelled=True)
    keep = np.isin(y, ["wash", "heating"])
    X = F["level+fft"][keep]
    y, cyc = y[keep], np.array([cycle_of(s) for s in src[keep]])
    print(f"\n2. wash vs heating, leave one cycle out: {len(y):,} windows "
          f"(wash {(y == 'wash').sum():,}, heating {(y == 'heating').sum():,}), kNN k={args.k}")
    per_cycle, all_y, all_p = {}, [], []
    for c in sorted(set(cyc[y == "heating"])):
        te = cyc == c
        tr = ~te
        # balance training classes so the vote isn't won by wash's larger count
        rng = np.random.default_rng(0)
        idx = {s: np.flatnonzero(tr & (y == s)) for s in ("wash", "heating")}
        n = min(len(v) for v in idx.values())
        tr_idx = np.concatenate([rng.choice(v, n, replace=False) for v in idx.values()])
        for metric in ("l1",):
            p = bb.knn(X[tr_idx], y[tr_idx], X[te], args.k, metric)
        per_cycle[c] = {"balanced_accuracy": round(balanced_accuracy(y[te], p), 3),
                        "heating_windows": int((y[te] == "heating").sum()), "wash_windows": int((y[te] == "wash").sum())}
        all_y.append(y[te]); all_p.append(p)
        print(f"  {c[len('becken_BWM5381IX_'):]:<24} balanced acc {per_cycle[c]['balanced_accuracy']:.3f}  "
              f"(heating {per_cycle[c]['heating_windows']}, wash {per_cycle[c]['wash_windows']})")
    pooled = balanced_accuracy(np.concatenate(all_y), np.concatenate(all_p))
    print(f"  pooled balanced accuracy {pooled:.3f}  (0.5 = chance)")

    print("\n3. features: effect size heating − wash (in wash SDs), per cycle")
    names = [f"rms {c}" for c in bb.CHANNELS] + [f"fft b{b} {c}" for b in range(16) for c in bb.CHANNELS]
    names = names[:X.shape[1]]
    cycles = sorted(set(cyc[y == "heating"]))
    d = np.full((len(cycles), X.shape[1]), np.nan)
    for i, c in enumerate(cycles):
        m = cyc == c
        w, h = X[m & (y == "wash")], X[m & (y == "heating")]
        if len(w) > 2 and len(h) > 2:
            d[i] = (h.mean(0) - w.mean(0)) / np.maximum(w.std(0), 1e-9)
    med = np.nanmedian(d, 0)
    agree = np.nanmean(np.sign(d) == np.sign(med), 0)
    order = np.argsort(-np.abs(med))[:10]
    for j in order:
        print(f"  {names[j]:<22} median d {med[j]:+.2f}  same sign in {agree[j]:.0%} of {len(cycles)} cycles")

    os.makedirs(os.path.join(OUT, "five_states"), exist_ok=True)
    path = os.path.join(OUT, "five_states", "diagnose_wash_heating.json")
    with open(path, "w") as f:
        json.dump({"window": args.window, "step": args.step, "k": args.k, "timeline": tl,
                   "leave_one_cycle_out": {"pooled_balanced_accuracy": round(pooled, 3), "per_cycle": per_cycle},
                   "top_features": [{"feature": names[j], "median_d": round(float(med[j]), 3),
                                     "sign_agreement": round(float(agree[j]), 3)} for j in order]}, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
