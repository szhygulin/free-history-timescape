#!/usr/bin/env python3
"""Task 2 -- commensurable NR watershed void-fraction decline test (feasibility).

The availability section (Sec. IV A) notes that the void-fraction redshift
evolution in the numerical-relativity watershed catalogs of Williams et al.
(arXiv:2403.15134) would be the definition-free commensurable test of the required
x3.31 decline. This probe records whether that test is FEASIBLE from the PUBLISHED
paper alone: it needs the watershed void volume fraction reported at >= 2 distinct
redshifts/snapshots so a decline ratio can be formed.

Finding (full-text read of the published paper via ar5iv, 2026-07-10): the paper
reports the watershed void filling fraction at ONLY the single final snapshot
z ~ 0 (Sec. 4.2), at two smoothing resolutions. No table or figure gives the void
fraction versus redshift/time, so no NR decline ratio can be formed from the paper.
Values below are the numbers PRINTED in the text (not figure eyeballing).

Verdict: NOT-FEASIBLE from the published paper. No manuscript number is drawn from
this probe; the availability-section sentence already hedges ("we claim no such
recomputation here, only that the measurement is sharpenable") and claims
measurability in the NR *catalogs* (the simulation is a time evolution), which this
finding does not contradict -- so the tex is left unchanged.

Run from repo root:  .venv/bin/python src/probes/nr_decline_test.py
"""
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R2_FINAL = os.path.join(ROOT, "probes_out", "R2_final.json")
OUT_JSON = os.path.join(ROOT, "tensions", "probes_out", "nr_decline_test.json")


def rel(p):
    return os.path.relpath(p, ROOT)


# Published watershed void filling fractions from Williams et al. 2024.
# Source: arXiv:2403.15134, Sec. 4.2, verbatim: "In the 4 h^-1 Mpc resolution
# simulation, 61.5% of the volume is in cells marked as being part of a void, and
# for the 12 h^-1 Mpc resolution simulation this fraction is 50%." Both at the
# final z~0 snapshot. Printed in text; NOT read off a figure.
WILLIAMS_PUBLISHED = [
    {"z_approx": 0.0, "resolution": "4 h^-1 Mpc", "void_volume_fraction": 0.615},
    {"z_approx": 0.0, "resolution": "12 h^-1 Mpc", "void_volume_fraction": 0.50},
]


def main():
    with open(R2_FINAL) as fh:
        R = json.load(fh)
    req_decline = R["two_part_test"]["part2_shape_decline"]["req_total_decline_x"]

    distinct_z = sorted({round(p["z_approx"], 3) for p in WILLIAMS_PUBLISHED})
    n_distinct_z = len(distinct_z)
    feasible = n_distinct_z >= 2

    record = {
        "probe": "Task 2 -- NR watershed void-fraction decline test (feasibility)",
        "question": "Does Williams et al. 2024 (arXiv:2403.15134) report the "
                    "watershed void volume fraction at >1 redshift/snapshot, so an "
                    "NR decline ratio can be compared to the required x3.31 decline?",
        "source": {
            "arxiv": "2403.15134",
            "citation": "Williams, Macpherson, Wiltshire, Stevens 2024 "
                        "(MNRAS 536, 2645; arXiv:2403.15134)",
            "read_method": "full-text via ar5iv HTML, 2026-07-10",
            "section": "4.2",
            "verbatim_quote": "In the 4 h^-1 Mpc resolution simulation, 61.5% of "
                              "the volume is in cells marked as being part of a "
                              "void, and for the 12 h^-1 Mpc resolution simulation "
                              "this fraction is 50%.",
            "figure_eyeballed": False,
            "note": "Both values are at the single final z~0 snapshot. No table or "
                    "figure in the paper gives the void fraction versus "
                    "redshift/time; the abstract's z~0 statistics (void expansion "
                    "10-30% above global average; curvature density 60-80% in void "
                    "centres) are also single-snapshot.",
        },
        "williams_published_points": WILLIAMS_PUBLISHED,
        "distinct_redshift_snapshots_reported": distinct_z,
        "n_distinct_redshift_snapshots": n_distinct_z,
        "needed_for_decline_ratio": 2,
        "feasible_from_published_paper": feasible,
        "nr_decline_ratio_over_covered_range": None,
        "required_decline_for_reference": {
            "source": rel(R2_FINAL),
            "req_total_decline_x_0_to_2p33": round(req_decline, 4),
            "note": "Reference only: the required LA decline over 0<z<2.33. No NR "
                    "counterpart can be formed because the NR paper covers a single "
                    "redshift (z~0), so no over-range comparison is possible.",
        },
        "verdict": "NOT-FEASIBLE",
        "verdict_reason": "The published Williams et al. paper reports the watershed "
                          "void volume fraction only at the single z~0 snapshot "
                          "(61.5% at 4 h^-1 Mpc, 50% at 12 h^-1 Mpc; Sec. 4.2). A "
                          "decline ratio needs >=2 redshift snapshots, so the "
                          "commensurable NR decline test cannot be run from the "
                          "paper alone. The simulation is itself a time evolution, "
                          "so f_v(z) is extractable in principle from its earlier "
                          "snapshots (re-analysis of the catalogs, not done here) -- "
                          "consistent with the tex claim that the measurement is "
                          "'within reach' in the NR catalogs and 'sharpenable', "
                          "which this finding does not contradict.",
        "tex_change": "NONE. The availability-section sentence does not overpromise: "
                      "it claims measurability in the NR catalogs (true; a time "
                      "evolution) and explicitly hedges 'we claim no such "
                      "recomputation here, only that the measurement is sharpenable'.",
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"[nr] wrote {rel(OUT_JSON)}")
    print(f"[nr] distinct z snapshots reported: {distinct_z} -> feasible={feasible}")
    print(f"[nr] verdict: {record['verdict']}")


if __name__ == "__main__":
    main()
