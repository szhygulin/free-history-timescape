#!/usr/bin/env python3
"""One-shot: add the pre-registered k=4 BIC sensitivity accounting alongside the primary
k=0 forced prediction and the T1 ceiling in threephase_forced_geometry.json (PLAN sec 3
accounting: 'primary k=0 ...; sensitivity row k=4 reported alongside').  No recompute -- it
is an algebraic BIC re-weighting of the already-computed joint chi2 values."""
import json
import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
P = os.path.join(_REPO, "probes_out", "threephase_forced_geometry.json")

d = json.load(open(P))
chi2_lcdm = d["bic_bar_dr2"]["chi2_LCDM_dr2"]
lnN = d["bic_bar_dr2"]["lnN"]
jc_k0 = d["STEP3_k0_forced_prediction"]["joint_chi2"]
jc_t1 = d["STEP4_T1_diagnostic_ceiling"]["joint_chi2"]

# k=4 bar: LCDM has k=1 (Om). A model with k params beats LCDM (BIC) iff
#   chi2_model + k lnN <= chi2_LCDM + 1 lnN  <=>  chi2_model <= chi2_LCDM - (k-1) lnN.
bar_k4 = chi2_lcdm - 3.0 * lnN          # k=4 vs LCDM k=1

d["k4_sensitivity_accounting"] = {
    "note": "PLAN sec 3 sensitivity row: if the four constants were charged as cosmology-"
            "fitted (k=4) rather than structure-pinned (k=0), the BIC bar drops by 3 lnN. "
            "Both the forced k=0 prediction and the T1 ceiling still miss it; the verdict is "
            "accounting-robust.",
    "bar_k4_vs_LCDM_k1": bar_k4,
    "forced_prediction_k0_joint_chi2": jc_k0,
    "forced_prediction_miss_vs_k4_bar": jc_k0 - bar_k4,
    "T1_ceiling_joint_chi2": jc_t1,
    "T1_ceiling_miss_vs_k4_bar": jc_t1 - bar_k4,
    "T1_deltaBIC_vs_LCDM": (jc_t1 - chi2_lcdm) + 3.0 * lnN,
}
json.dump(d, open(P, "w"), indent=2)
print(f"patched {P}: bar_k4={bar_k4:.3f}  T1_miss_k4={jc_t1-bar_k4:.3f}  "
      f"T1_deltaBIC={ (jc_t1-chi2_lcdm)+3*lnN:.3f}")
