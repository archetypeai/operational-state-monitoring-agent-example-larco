"""Preflight FAILs the user has looked at and decided to keep for now (plan.md).

Keyed by (cycle file, check name). Every preflight still prints them, tagged
with the reason, but they don't block the next stage.
"""
ACKNOWLEDGED = {
    ("wm_becken-flt_BWM5381IX_cold_cotton_30_4.csv", "coverage"):
        "kept in delivery for now (2026-09-29); revisit whether to drop the uncovered part",
}
