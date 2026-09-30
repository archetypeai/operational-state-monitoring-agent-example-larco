#!/usr/bin/env python3
"""Stage 0a: split becken's cycles into roles, by setting group, before any download.

A setting group is every cycle of one program × wash temperature × load:
cotton and eco groups are a cold-room + hot-room pair of near-twins, the
other programs one warm-room cycle each. Whole groups go to library,
validation or test at random, stratified by family (cotton / eco / other),
so no twin straddles two roles. becken-flt is delivery only.

The seed is fixed in plan.md and never re-rolled. Writes data/split.json.

    python3 prep/split.py
"""
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from larco import DATA, DELIVERY_UNIT, LIBRARY_UNIT, listing  # noqa: E402

SEED = 20260928
# groups per role and family: library / validation / test (plan.md)
ALLOCATION = {"cotton": (16, 6, 6), "eco": (4, 2, 1), "other": (14, 5, 4)}
ROLES = ("library", "validation", "test")


def family(program):
    return program if program in ("cotton", "eco") else "other"


def main():
    cycles = listing()["cycles"]
    becken = [c for c in cycles if c["unit"] == LIBRARY_UNIT and c["vibration"]]
    groups = collections.defaultdict(list)
    for c in becken:
        groups[c["setting"]].append(c["file"])

    rng = random.Random(SEED)
    role_of = {}
    for fam, counts in ALLOCATION.items():
        names = sorted(g for g in groups if family(g.split("_")[0]) == fam)
        if len(names) != sum(counts):
            sys.exit(f"{fam}: {len(names)} groups in the listing, allocation expects {sum(counts)}")
        rng.shuffle(names)
        start = 0
        for role, n in zip(ROLES, counts):
            for g in names[start:start + n]:
                role_of[g] = role
            start += n

    split = {role: {} for role in ROLES}
    for g in sorted(groups):
        split[role_of[g]][g] = sorted(groups[g])
    split["delivery"] = sorted(c["file"] for c in cycles if c["unit"] == DELIVERY_UNIT and c["vibration"])
    out = {"seed": SEED, "allocation": ALLOCATION, "library_unit": LIBRARY_UNIT,
           "delivery_unit": DELIVERY_UNIT, **split}
    with open(os.path.join(DATA, "split.json"), "w") as f:
        json.dump(out, f, indent=1)

    for role in ROLES:
        fams = collections.Counter(family(g.split("_")[0]) for g in split[role])
        n = sum(len(v) for v in split[role].values())
        print(f"{role:<10} {len(split[role]):>2} groups, {n:>2} cycles  {dict(fams)}")
    print(f"{'delivery':<10} {len(split['delivery']):>3} cycles of {DELIVERY_UNIT}")
    print(f"wrote {os.path.join(DATA, 'split.json')} (seed {SEED})")


if __name__ == "__main__":
    main()
