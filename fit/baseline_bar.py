#!/usr/bin/env python3
"""Stage 4a: the bar. Simple baselines Omega has to beat, on exactly the role files and split.

Library: the 1,600 library windows (data/roles/library/, 400 per state). Validation:
windows of 1,024 rows at step 1,024 from the start of each validation file, each
labelled by its last row (the osm blueprint's `last_record` pairing). Scored as the
platform scores: macro-F1 over all four states, a state with no windows and no
predictions counting as 0 (fit/probe_timestamps.py). Features, on the z-scored files:

  level      per-channel log RMS of the window (mean removed): what a threshold uses
  fft        per-channel log power in 16 log-spaced bands, each window standardised
  level+fft  both

each classified by kNN (features z-scored on the library; l1 or l2; k in 1, 5, 15, 51).
The best setting is chosen on the 6 search-validation cycles only, then reported on
all 21 validation cycles. Test is not touched here (Stage 5), except by --test.

--test (Stage 5) scores the bar's chosen setting for that window/step (from its
validation file, never re-chosen) once on the 18 test cycles: pooled, without the
exploration-seen cycle cold_cotton_40_2, per program family and per cycle. --delivery
(Stage 7) does the same on the 106 becken-flt cycles, against the labels held back,
listing cold_cotton_40_2 and cold_cotton_30_4 separately. Writes
fit/out/bar_<test|delivery>_w<window>_s<step>.json.

Writes fit/out/bar_validation_w<window>_s<step>.json, one per window/step pair, so
fit/compare_bar.py can judge each Omega trial against the bar on the same windows.

    python fit/baseline_bar.py                          # window 1024, step 1024 (~90 s)
    python fit/baseline_bar.py --window 512 --step 512  # another pair
    python fit/baseline_bar.py --window 512 --step 512 --test       # Stage 5: the bar on test
    python fit/baseline_bar.py --window 512 --step 512 --delivery   # Stage 7: the bar on delivery
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "prep"))
from states import CHANNELS, STATES  # noqa: E402

ROLES = os.path.join(ROOT, "data", "roles")
OUT = os.path.join(ROOT, "fit", "out")
WINDOW = STEP = 1024       # overridden by --window / --step
KS = [1, 5, 15, 51]
METRICS = ["l1", "l2"]


def level_features(w):
    return np.log(np.sqrt(((w - w.mean(0)) ** 2).mean(0)) + 1e-6)


def fft_features(w, n_bands=16):
    z = (w - w.mean(0)) / np.maximum(w.std(0), 1e-9)
    p = np.abs(np.fft.rfft(z, axis=0)) ** 2
    edges = np.unique(np.geomspace(1, p.shape[0] - 1, n_bands + 1).astype(int))
    return np.concatenate([np.log1p(p[edges[i]:edges[i + 1]].sum(0)) for i in range(len(edges) - 1)])


def windows_of(path, labelled, window, step, pieces=None):
    # window and step are passed in: worker processes don't see globals set in main().
    # A library file holds several continuous pieces (manifest): windows are cut within
    # each piece, never across the time jump between two, which gives exactly the
    # windows the earlier one-file-per-piece library gave.
    d = pd.read_csv(path, engine="pyarrow")
    x = d[CHANNELS].to_numpy(np.float64)
    spans = [(p["row"], p["row"] + p["rows"]) for p in pieces] if pieces else [(0, len(x))]
    starts = [a + o for a, b in spans for o in range(0, b - a - window + 1, step)]
    lev = np.array([level_features(x[s:s + window]) for s in starts])
    fft = np.array([fft_features(x[s:s + window]) for s in starts])
    if labelled == "sidecar":      # delivery: the labels held back, row for row
        side = path.replace(f"{os.sep}delivery{os.sep}", f"{os.sep}delivery_labels{os.sep}")
        lab = pd.read_csv(side, usecols=["label"], engine="pyarrow").label.to_numpy()[[s + window - 1 for s in starts]]
    elif labelled:
        lab = d.label.to_numpy()[[s + window - 1 for s in starts]]
    else:
        lab = np.array([os.path.basename(path).split("__")[0]] * len(lev))
    return lev, fft, lab


def load(paths, labelled, pieces=None):
    pieces = pieces or [None] * len(paths)
    with ProcessPoolExecutor(6) as pool:
        parts = list(pool.map(windows_of, paths, [labelled] * len(paths), [WINDOW] * len(paths), [STEP] * len(paths),
                              pieces))
    lev = np.concatenate([p[0] for p in parts])
    fft = np.concatenate([p[1] for p in parts])
    lab = np.concatenate([p[2] for p in parts])
    src = np.concatenate([[os.path.basename(pth)] * len(p[2]) for pth, p in zip(paths, parts)])
    return {"level": lev, "fft": fft, "level+fft": np.hstack([lev, fft])}, lab, src


def knn(Xtr, ytr, Xte, k, metric):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    pred = np.empty(len(B), dtype=object)
    # block size keeps the distance work near 250 MB whatever the library size
    block = max(16, int(2.5e8 / (8 * len(A) * A.shape[1]))) if metric == "l1" else 4096
    aa = (A ** 2).sum(1)
    for i in range(0, len(B), block):
        b = B[i:i + block]
        if metric == "l1":
            d = np.abs(b[:, None, :] - A[None]).sum(2)
        else:
            d = aa[None, :] - 2 * b @ A.T          # squared l2, minus the constant |b|^2
        nn = np.argsort(d, axis=1, kind="stable")[:, :k]
        votes = np.stack([(ytr[nn] == s).sum(1) for s in STATES], 1)
        pred[i:i + block] = np.array(STATES, dtype=object)[votes.argmax(1)]
    return pred


def scores(y, p):
    """Platform-style: F1 per state over all four (0 when a state has no windows and no predictions)."""
    f1 = {}
    for s in STATES:
        tp = int(((y == s) & (p == s)).sum()); fp = int(((y != s) & (p == s)).sum()); fn = int(((y == s) & (p != s)).sum())
        f1[s] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    cm = [[int(((y == a) & (p == b)).sum()) for b in STATES] for a in STATES]
    return {"macro_f1": round(float(np.mean(list(f1.values()))), 4), "f1": {s: round(v, 4) for s, v in f1.items()},
            "confusion": cm, "windows": int(len(y))}


def main():
    global WINDOW, STEP
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--step", type=int, default=1024)
    ap.add_argument("--test", action="store_true", help="Stage 5: score the chosen setting once on the test cycles")
    ap.add_argument("--delivery", action="store_true",
                    help="Stage 7: score it on the delivery cycles, against the labels held back")
    args = ap.parse_args()
    WINDOW, STEP = args.window, args.step
    manifest = json.load(open(os.path.join(ROLES, "manifest.json")))
    if args.test or args.delivery:
        return score_role(manifest, "test" if args.test else "delivery")
    search = {n[3:-4] for n in manifest["search_validation"]}
    lib_paths = sorted(glob.glob(os.path.join(ROLES, "library", "*.csv")))
    val_paths = sorted(glob.glob(os.path.join(ROLES, "validation", "*.csv")))
    lib_pieces = {os.path.join(ROLES, f["file"]): f.get("pieces") for f in manifest["library"]["files"]}
    Ftr, ytr, _ = load(lib_paths, labelled=False, pieces=[lib_pieces.get(p) for p in lib_paths])
    Fva, yva, src = load(val_paths, labelled=True)
    in_search = np.array([s.split("__seg")[0] in search for s in src])
    print(f"window {WINDOW}, step {STEP}")
    print(f"library {len(ytr):,} windows {dict(zip(*np.unique(ytr, return_counts=True)))}")
    print(f"validation {len(yva):,} windows ({in_search.sum():,} in the 6 search cycles)")

    results = []
    for feat, k, metric in product(Ftr, KS, METRICS):
        p = knn(Ftr[feat], ytr, Fva[feat][in_search], k, metric)
        r = {"features": feat, "k": k, "metric": metric, **scores(yva[in_search], p)}
        results.append(r)
        print(f"  {feat:<9} k={k:<3} {metric}  search macro-F1 {r['macro_f1']:.4f}  "
              + "  ".join(f"{s} {v:.2f}" for s, v in r["f1"].items()))

    best = max(results, key=lambda r: r["macro_f1"])
    best_per_feature = {f: max((r for r in results if r["features"] == f), key=lambda r: r["macro_f1"]) for f in Ftr}
    confirm = {}
    for f, r in best_per_feature.items():
        p = knn(Ftr[f], ytr, Fva[f], r["k"], r["metric"])
        c = scores(yva, p)
        c["per_cycle"] = {}
        for cyc in sorted({s.split("__seg")[0] for s in src}):
            m = np.array([s.split("__seg")[0] == cyc for s in src])
            c["per_cycle"][cyc] = scores(yva[m], p[m])["macro_f1"]
        confirm[f] = {"setting": {k: r[k] for k in ("features", "k", "metric")}, **c}

    print(f"\nbest on the 6 search cycles: {best['features']} k={best['k']} {best['metric']}, macro-F1 {best['macro_f1']:.4f}")
    print("each feature set's best setting, then scored once on all 21 validation cycles:")
    for f, c in confirm.items():
        s = best_per_feature[f]
        print(f"  {f:<9} k={s['k']:<3} {s['metric']}  search {s['macro_f1']:.4f}  all 21: {c['macro_f1']:.4f}  "
              + "  ".join(f"{st} {v:.2f}" for st, v in c["f1"].items()))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"bar_validation_w{WINDOW}_s{STEP}.json")
    with open(path, "w") as fh:
        json.dump({"window": WINDOW, "step": STEP, "pairing": "last_record", "search_cycles": sorted(search),
                   "grid": results, "best": best, "best_per_feature": best_per_feature, "all_21": confirm}, fh, indent=1)
    print(f"\nwrote {path}")


def family(cycle):
    program = cycle.split("_")[4]      # wm_becken_BWM5381IX_<room>_<program>_...
    return program if program in ("cotton", "eco") else "other"


# cycles reported on their own lines (plan.md): the one test cycle seen in exploration, and
# in delivery becken-flt's exploration cycle and its 42%-coverage cycle
SEPARATE = {"test": ["wm_becken_BWM5381IX_cold_cotton_40_2.csv"],
            "delivery": ["wm_becken-flt_BWM5381IX_cold_cotton_40_2.csv", "wm_becken-flt_BWM5381IX_cold_cotton_30_4.csv"]}


def score_role(manifest, role):
    """Stages 5 and 7: the bar's validation-chosen setting, scored once on test or delivery."""
    val = os.path.join(OUT, f"bar_validation_w{WINDOW}_s{STEP}.json")
    if not os.path.exists(val):
        sys.exit(f"no bar for window {WINDOW} / step {STEP} yet: run without --{role} first")
    best = json.load(open(val))["best"]
    lib_paths = sorted(glob.glob(os.path.join(ROLES, "library", "*.csv")))
    paths = sorted(glob.glob(os.path.join(ROLES, role, "*.csv")))
    lib_pieces = {os.path.join(ROLES, f["file"]): f.get("pieces") for f in manifest["library"]["files"]}
    Ftr, ytr, _ = load(lib_paths, labelled=False, pieces=[lib_pieces.get(p) for p in lib_paths])
    Fte, yte, src = load(paths, labelled="sidecar" if role == "delivery" else True)
    p = knn(Ftr[best["features"]], ytr, Fte[best["features"]], best["k"], best["metric"])
    cyc = np.array(["wm_" + s.split("__seg")[0] + ".csv" for s in src])
    sep = SEPARATE[role]
    rest = ~np.isin(cyc, sep)
    fam = np.array([family(c) for c in cyc])
    out = {"role": role, "window": WINDOW, "step": STEP, "setting": {k: best[k] for k in ("features", "k", "metric")},
           "cycles": len(set(cyc)), "all": scores(yte, p), "without_separate": scores(yte[rest], p[rest]),
           "separate": {c: scores(yte[cyc == c], p[cyc == c]) for c in sep},
           "per_family": {f: scores(yte[fam == f], p[fam == f]) for f in ("cotton", "eco", "other")},
           "per_cycle": {c: scores(yte[cyc == c], p[cyc == c])["macro_f1"] for c in sorted(set(cyc))}}
    print(f"window {WINDOW}, step {STEP}: the bar ({best['features']} k={best['k']} {best['metric']}, "
          f"chosen on validation) on {len(yte):,} {role} windows")
    rows = ([(f"all {out['cycles']} {role} cycles", out["all"]), (f"without {len(sep)} listed below", out["without_separate"])]
            + [(f"{c[3:-4]} alone", r) for c, r in out["separate"].items()]
            + [(f"family {f}", r) for f, r in out["per_family"].items()])
    for name, r in rows:
        print(f"  {name:<44} macro-F1 {r['macro_f1']:.4f}  " + "  ".join(f"{s} {v:.2f}" for s, v in r["f1"].items())
              + f"  windows {r['windows']:,}")
    path = os.path.join(OUT, f"bar_{role}_w{WINDOW}_s{STEP}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
