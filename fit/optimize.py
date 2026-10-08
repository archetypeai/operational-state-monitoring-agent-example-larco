#!/usr/bin/env python3
"""Stages 4b and 4c: Omega on the Optimize API, trained on the library, scored on validation.

Platform only: it needs no baseline. Comparing trials with one is a separate, optional
step (fit/compare_bar.py), since a user with their own data may have no sensible
baseline at all.

Every library file (data/roles/library/, one state each, named <state>__...) becomes a
training example labelled by its state; each of the 6 search-validation files becomes
a validation example scored on its `label` column (`last_record` pairing). Stage 4c
(--validation all) scores one chosen setting on all 21 validation cycles (25 files) instead. Files are
uploaded once; their ids are cached in fit/out/uploads.json (keyed by path, size and
mtime), so re-runs and later searches reuse them.

The search space is a product of the values given (flags, or --pool full). With
max_trials below its size, the platform samples configurations: a random search.
Default: the single measurement trial (window 1024, step 1024, k 5, l1, uniform).
Step > window skips records and needs --allow-gaps. After the run it prints every
trial's scores, flags configurations sampled twice, and reports trial time (a fixed cost
plus a cost per 1,000 windows).

Writes fit/out/optimize_<optimization id>.json. Long runs: add --background, which
relaunches the same command under nohup + caffeinate with output to fit/out/optimize.log.

    python fit/optimize.py --dry-run                            # the plan: no uploads, no jobs
    python fit/optimize.py --background                         # the measurement trial, detached
    python fit/optimize.py --pool full --allow-gaps --max-trials 16 --background   # random search (Stage 4b)
    python fit/optimize.py --resume opt_...                     # collect a run whose poller died
    python fit/optimize.py --windows 512 --steps 512 --k 31 --metrics cosine --validation all   # Stage 4c
"""
import argparse
import datetime
import glob
import itertools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atai import agents, api_base, load_dotenv, request, states_override, trial_f1, trial_setting, upload_file, wait_optimization  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prep"))
from background import add_background_flag, maybe_detach  # noqa: E402
from states import STATES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES = os.path.join(ROOT, "data", "roles")
OUT = os.path.join(ROOT, "fit", "out")
CACHE = os.path.join(OUT, "uploads.json")
BLUEPRINT_ID = "osm"     # the canonical blueprint's key: each deployment resolves it to its own blp_ id
# The exhaustive pool (Stage 4b), limited to values known to work on the platform.
FULL_POOL = {"windows": [256, 512, 1024], "steps": [256, 512, 1024], "k": [1, 3, 5, 7, 9, 15, 21, 31],
             "metrics": ["l1", "cosine"], "weights": ["uniform", "distance"]}


def log(msg):
    print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def upload_all(paths, jobs):
    """Platform file ids for `paths`, uploading only what the cache doesn't already hold."""
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    def key(p):
        st = os.stat(p)
        # keyed by deployment too: file ids from one deployment don't exist on another
        return f"{api_base()}|{os.path.relpath(p, ROLES)}|{st.st_size}|{int(st.st_mtime)}"

    todo = [p for p in paths if key(p) not in cache]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if todo:
        mb = sum(os.path.getsize(p) for p in todo) / 1e6
        log(f"uploading {len(todo):,} of {len(paths):,} files ({mb:,.0f} MB; {len(paths) - len(todo):,} cached)")
        done, t0 = 0, time.time()

        def one(p):
            name = "larco-" + os.path.relpath(p, ROLES).replace("/", "__").replace(".csv", f"-{stamp}.csv")
            return key(p), upload_file(p, rename=name)["file_id"]

        with ThreadPoolExecutor(jobs) as pool:
            for k, fid in pool.map(one, todo):
                cache[k] = fid
                done += 1
                os.makedirs(OUT, exist_ok=True)
                with open(CACHE, "w") as f:         # save as we go, so an interrupted upload resumes
                    json.dump(cache, f, indent=1)
                if done % 100 == 0 or done == len(todo):
                    log(f"  uploaded {done:,}/{len(todo):,} ({time.time() - t0:.0f} s)")
    return {p: cache[key(p)] for p in paths}


def windows_in(rows, window, step):
    return max(0, (rows - window) // step + 1)


def library_windows(lib_files, window, step):
    """Training windows the platform keeps: those within a piece. Any window across the time
    jump between two pieces fails the sampling-rate check and is skipped (fit/probe_gaps.py)."""
    return sum(windows_in(p["rows"], window, step) for f in lib_files for p in (f.get("pieces") or [f]))


def summarize(opt, trials, val_files, lib_files):
    """Print every trial's scores, duplicates and the platform rate; return the rows saved."""
    # Every trial is created with the optimization, and they run one after another, so a
    # trial's own time runs from the previous completion (or its creation) to its completion.
    parse = lambda v: datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))  # noqa: E731
    minutes, prev = {}, None
    for t in sorted((t for t in trials if t.get("completed_at") and t.get("created_at")),
                    key=lambda t: t["completed_at"]):
        start = max(parse(t["created_at"]), prev) if prev else parse(t["created_at"])
        prev = parse(t["completed_at"])
        minutes[t["trial_number"]] = round((prev - start).total_seconds() / 60, 1)
    rows = []
    for t in sorted(trials, key=lambda t: t["trial_number"]):
        f1, n = trial_f1(t)
        mins = minutes.get(t["trial_number"])
        w, st, k, metric, weights = trial_setting(t)
        windows = library_windows(lib_files, w, st) + sum(windows_in(f["rows"], w, st) for f in val_files)
        rows.append({"trial": t["trial_number"], "window": w, "step": st, "k": k, "metric": metric, "weights": weights,
                     "status": t["status"], "macro_f1": t.get("objective_value"), "f1": f1,
                     "windows_scored": n, "windows_total": windows, "minutes": mins})
    print(f"\noptimization {opt['id']}: {opt['status']}, {sum(r['macro_f1'] is not None for r in rows)}/{len(rows)} trials scored")
    for r in sorted(rows, key=lambda r: -(r["macro_f1"] if r["macro_f1"] is not None else -1)):
        m = "   -  " if r["macro_f1"] is None else f"{r['macro_f1']:.4f}"
        print(f"  #{r['trial']:<3} w={r['window']:<4} step={r['step']:<4} k={r['k']:<3} {r['metric']:<6} {r['weights']:<8} "
              f"{r['status']:<9} macro-F1 {m}  " + "  ".join(f"{s_} {r['f1'].get(s_, 0):.2f}" for s_ in STATES)
              + f"  windows {r['windows_scored']:,}  {r['minutes']} min")
    seen = {}
    for r in rows:
        seen.setdefault((r["window"], r["step"], r["k"], r["metric"], r["weights"]), []).append(r["trial"])
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    if dup:
        print(f"WARNING: {len(dup)} configuration(s) sampled more than once: {dup}")
    # Trial time is mostly a fixed cost, so report minutes = fixed + per 1,000 windows
    # (least squares over the finished trials), not one windows-per-minute rate.
    done = [r for r in rows if r["minutes"]]
    xs = [r["windows_total"] / 1000 for r in done]
    if len(set(xs)) > 1:
        mx, my = sum(xs) / len(xs), sum(r["minutes"] for r in done) / len(done)
        slope = sum((x - mx) * (r["minutes"] - my) for x, r in zip(xs, done)) / sum((x - mx) ** 2 for x in xs)
        log(f"trial time: about {my - slope * mx:.1f} min + {slope:.2f} min per 1,000 windows "
            f"(library + validation), over {len(done)} trials; {sum(r['minutes'] for r in done) / 60:.1f} h in all")
    elif done:
        log(f"trial time: {done[0]['minutes']} min for {done[0]['windows_total']:,} windows")
    return rows, {str(k): v for k, v in dup.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", choices=["full"], help="start from the exhaustive pool; other flags override its values")
    ap.add_argument("--windows", type=int, nargs="+", default=[1024])
    ap.add_argument("--steps", type=int, nargs="+", default=[1024])
    ap.add_argument("--k", type=int, nargs="+", default=[5])
    ap.add_argument("--metrics", nargs="+", default=["l1"])
    ap.add_argument("--weights", nargs="+", default=["uniform"])
    ap.add_argument("--max-trials", type=int)
    ap.add_argument("--allow-gaps", action="store_true",
                    help="allow step > window (skips the records between windows)")
    ap.add_argument("--blueprint", default=BLUEPRINT_ID)
    ap.add_argument("--validation", choices=["search", "all"], default="search",
                    help="score on the 6 search cycles (Stage 4b) or all 21 validation cycles (Stage 4c)")
    ap.add_argument("--name", default="LARCO Stage 4b")
    ap.add_argument("--upload-jobs", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", metavar="OPT_ID")
    add_background_flag(ap, default_log="fit/out/optimize.log")
    args = ap.parse_args()
    maybe_detach(args)
    if args.pool == "full":
        given = {a.lstrip("-").split("=")[0].replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
        for key, vals in FULL_POOL.items():
            if key not in given:
                setattr(args, key, vals)

    manifest = json.load(open(os.path.join(ROLES, "manifest.json")))
    search = set(manifest["search_validation"])
    library = sorted(glob.glob(os.path.join(ROLES, "library", "*.csv")))
    val_files = [f for f in manifest["validation"]["files"] if args.validation == "all" or f["cycle"] in search]
    validation = [os.path.join(ROLES, f["file"]) for f in val_files]
    lib_files = manifest["library"]["files"]

    grid = list(itertools.product(args.windows, args.steps, args.k, args.metrics, args.weights))
    gaps = sorted({(w, st) for w, st, *_ in grid if st > w})
    if gaps and args.allow_gaps:
        log(f"--allow-gaps: step > window for {gaps}; the records between windows are skipped "
            f"(validation scored on a regular sample; each library file gives its first window only)")
    elif gaps:
        # step > window skips the records between windows: unscored validation, and
        # (library files being whole 1,024-row windows) half-used training files
        sys.exit(f"step > window would skip records: {gaps}. Use step <= window, pass --allow-gaps to "
                 f"do it deliberately, or run a second optimization for another window size (the search space is a product)")
    one = lambda vals: {"type": "categorical", "values": vals}
    space = {"parameters": {
        "window_size": {"kind": "value", "spec": one(args.windows)},
        "step_size": {"kind": "value", "spec": one(args.steps)},
        "k_neighbors": {"kind": "fitting", "spec": one(args.k)},
        "metric": {"kind": "fitting", "spec": one(args.metrics)},
        "weights": {"kind": "fitting", "spec": one(args.weights)},
    }}
    if not args.resume:          # the plan comes from this command's flags, not the optimization being collected
        for w, st in {(g[0], g[1]) for g in grid}:
            lw = library_windows(lib_files, w, st)
            vw = sum(windows_in(f["rows"], w, st) for f in val_files)
            log(f"window {w}, step {st}: library {len(library):,} files / {lw:,} windows kept; "
                f"validation {len(validation)} files / {vw:,} windows")
        n = min(args.max_trials or len(grid), len(grid))
        log(f"{n} trial(s) of a {len(grid)}-point space ({'random search' if n < len(grid) else 'every point'}); "
            f"blueprint {args.blueprint}")
    if args.dry_run:
        print(json.dumps(space, indent=1))
        return

    load_dotenv()
    if args.resume:
        opt_id = args.resume
        log(f"collecting {opt_id} on {api_base()}")
    else:
        ids = upload_all(library + validation, args.upload_jobs)
        training = [{"name": os.path.basename(p)[:-4],
                     "inputs": [{"type": "file", "id": ids[p], "format": "csv"}],
                     "ground_truth": {"state": {"from": {"constant": os.path.basename(p).split("__")[0]}}}}
                    for p in library]
        validation_examples = [{"name": os.path.basename(p)[:-4],
                                "inputs": [{"type": "file", "id": ids[p], "format": "csv"}],
                                "ground_truth": {"state": {"from": {"column": "label"}, "downsampling": "last_record"}}}
                               for p in validation]
        bp = request("GET", f"{agents()}/blueprints/{args.blueprint}")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        opt = request("POST", f"{agents()}/optimizations", body={
            "name": f"{args.name} {stamp}", "blueprint_id": bp["id"], "objective": "macro_f1",
            "search_space": space, "budget": {"max_trials": args.max_trials or len(grid)},
            "training_examples": training, "validation_examples": validation_examples,
            **states_override(bp, training)})
        opt_id = opt["id"]
        log(f"optimization {opt_id} created (collect later with --resume {opt_id}) on {api_base()}")

    opt, trials = wait_optimization(opt_id, label="", every_s=60, log=log)
    rows, dup = summarize(opt, trials, val_files, lib_files)
    path = os.path.join(OUT, f"optimize_{opt_id}.json")
    with open(path, "w") as f:
        json.dump({"optimization": opt, "trials": trials, "summary": rows, "duplicates": dup,
                   "endpoint": api_base(), "search_space": space, "blueprint": args.blueprint, "allow_gaps": bool(args.allow_gaps),
                   "validation_set": args.validation, "validation": [f["file"] for f in val_files]}, f, indent=1)
    log(f"results: {path}")
    log("to compare with a baseline (optional): python fit/compare_bar.py")


if __name__ == "__main__":
    main()
