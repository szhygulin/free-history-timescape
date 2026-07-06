#!/usr/bin/env python3
"""Independent adversarial recompute of probes_out/R2_final.json.

Does NOT import R2_final.py. Reads only the five committed input artifacts and
recomputes every load-bearing number from scratch (Phi(sigma/2), decline ratios,
band-membership, floor sigma) to confirm or refute the R2-final verdict.
"""
import json, os
from math import erf, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "probes_out")


def Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def Phi_inv(p):
    # bisection on the monotone Phi; ample range for p in (0,1)
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


tel = load("telescope_fvobs.json")
pr = load("modelV_probeR.json")
prlb = load("modelV_probeR_LB.json")
mdf = load("mapping_decline_forecast.json")
r2 = load("R2.json")
rf = load("R2_final.json")

results = {}


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ---------- CHECK 1: Part 1 level gate ----------
sigma0 = tel["provenance"]["sigma0_anchor"]              # 0.7344797420042518
bm_central = tel["provenance"]["below_mean_z0_anchor"]   # 0.643279949444014
# band from R2.json below-mean band (narrow, NOT the wide definition band)
band = r2["overlap_at_z0"]["vs_below_mean_band"]["band"]
band_lo, band_hi = band[0], band[1]
width = band_hi - band_lo

LA_fv0 = pr["V"]["fv0"]
LB_fv0 = prlb["V"]["fv0"]
V0_fv0 = pr["V0"]["fv0"]

# independent: band symmetric about below-mean central?
band_mid = 0.5 * (band_lo + band_hi)

def frac_height(x):
    return (x - band_lo) / width

LA_inside = band_lo <= LA_fv0 <= band_hi
LB_inside = band_lo <= LB_fv0 <= band_hi
V0_inside = band_lo <= V0_fv0 <= band_hi

results["check1_part1_level"] = {
    "band_used": [band_lo, band_hi],
    "band_symmetric_about_below_mean_central": approx(band_mid, bm_central),
    "band_mid": band_mid, "below_mean_central": bm_central,
    "LA_fv0": LA_fv0, "LA_inside": LA_inside, "LA_frac_height": frac_height(LA_fv0),
    "LB_fv0": LB_fv0, "LB_inside": LB_inside, "LB_frac_height": frac_height(LB_fv0),
    "V0_fv0": V0_fv0, "V0_inside": V0_inside, "V0_frac_height": frac_height(V0_fv0),
    # cross-check against R2_final claims
    "matches_rf_band": approx(band_lo, rf["two_part_test"]["part1_level_z0"]["measured_below_mean_band"][0])
                       and approx(band_hi, rf["two_part_test"]["part1_level_z0"]["measured_below_mean_band"][1]),
    "matches_rf_LA_frac": approx(frac_height(LA_fv0), 0.4457, tol=1e-2),
}
results["check1_part1_level"]["PASS"] = (
    LA_inside and (not LB_inside) and LB_fv0 < band_lo and (not V0_inside) and V0_fv0 < band_lo
    and results["check1_part1_level"]["band_symmetric_about_below_mean_central"]
    and results["check1_part1_level"]["matches_rf_band"]
)

# ---------- CHECK 2: Part 2 decline ratios, independent from sigma0*D(z) ----------
D = mdf["inputs"]["D_z_LCDM"]           # z = 0,0.3,0.7,1.3,2.33
znodes = mdf["inputs"]["z_nodes"]
obs_fv = [Phi(sigma0 * d / 2.0) for d in D]
obs_ratio = [f / obs_fv[0] for f in obs_fv]

req_lo = mdf["required"]["decline_ratio_lo"]
req_hi = mdf["required"]["decline_ratio_hi"]
req_ctr = mdf["required"]["decline_ratio"]

per_node = []
all_above = True
for i in range(1, len(znodes)):  # skip z=0
    above = obs_ratio[i] > req_hi[i]
    all_above = all_above and above
    per_node.append({
        "z": znodes[i],
        "obs_fv_recomputed": obs_fv[i],
        "obs_decline_ratio_recomputed": obs_ratio[i],
        "req_band": [req_lo[i], req_hi[i]],
        "obs_above_req_hi": above,
    })

obs_total = obs_fv[0] / obs_fv[-1]
req_total = mdf["required"]["total_decline_x"]

results["check2_part2_shape"] = {
    "per_node": per_node,
    "all_z_gt0_above_req_hi": all_above,
    "obs_total_decline_x": obs_total,
    "req_total_decline_x": req_total,
    # cross-check recomputed obs against telescope PRIMARY node values
    "obs_fv_matches_telescope_z03": approx(obs_fv[1], 0.6229320569236853),
    "obs_fv_matches_telescope_z07": approx(obs_fv[2], 0.6009407462040413),
    "obs_ratio_matches_rf_z03": approx(obs_ratio[1], rf["two_part_test"]["part2_shape_decline"]["per_node"][0]["obs_decline_ratio"]),
    "obs_ratio_matches_rf_z07": approx(obs_ratio[2], rf["two_part_test"]["part2_shape_decline"]["per_node"][1]["obs_decline_ratio"]),
    "obs_total_matches_rf": approx(obs_total, 1.1589, tol=1e-3),
}
results["check2_part2_shape"]["PASS"] = (
    all_above
    and results["check2_part2_shape"]["obs_fv_matches_telescope_z03"]
    and results["check2_part2_shape"]["obs_fv_matches_telescope_z07"]
    and results["check2_part2_shape"]["obs_total_matches_rf"]
    and req_total > obs_total
)

# ---------- CHECK 3: z=0.7 gap + floor theorem ----------
req_fv07 = pr["V"]["fv_nodes"][2]        # 0.39578
obs_fv07 = obs_fv[2]                      # recomputed
gap_abs = obs_fv07 - req_fv07
gap_rel = gap_abs / req_fv07
sigma_needed = 2.0 * Phi_inv(req_fv07)   # req_fv = Phi(sigma/2) -> sigma = 2 Phi^-1
# floor: below-mean Phi(sigma/2) >= 0.5 for sigma>=0; req 0.396<0.5 -> needs sigma<0
floor_impossible = (req_fv07 < 0.5) and (sigma_needed < 0.0)

# conservative Rs=8 bound from telescope
rs8 = [b for b in tel["Rs_sensitivity_band"]["band"] if b["R_s"] == 8.0][0]
fv07_rs8 = rs8["fv_below_mean_z07"]
rs_range = tel["Rs_sensitivity_band"]["fv_z07_range"]

results["check3_z07_floor"] = {
    "req_fv07": req_fv07, "obs_fv07_recomputed": obs_fv07,
    "gap_absolute": gap_abs, "gap_relative": gap_rel,
    "sigma_needed_for_req_fv": sigma_needed,
    "req_fv_below_half": req_fv07 < 0.5,
    "floor_impossible": floor_impossible,
    "fv07_Rs8": fv07_rs8, "gap_abs_Rs8": fv07_rs8 - req_fv07,
    "Rs_range": rs_range, "Rs_floor_holds_all_ge_half": all(v >= 0.5 for v in rs_range),
    "matches_rf_gap_abs": approx(gap_abs, 0.20516074620404123),
    "matches_rf_gap_rel": approx(gap_rel, 0.5183706761434161),
    "matches_rf_sigma_needed": approx(sigma_needed, -0.528570824830728, tol=1e-4),
}
results["check3_z07_floor"]["PASS"] = (
    floor_impossible
    and results["check3_z07_floor"]["matches_rf_gap_abs"]
    and results["check3_z07_floor"]["matches_rf_gap_rel"]
    and results["check3_z07_floor"]["matches_rf_sigma_needed"]
    and results["check3_z07_floor"]["Rs_floor_holds_all_ge_half"]
    and (fv07_rs8 >= 0.5)
)

# ---------- CHECK 4: verdict token + no wide-band envelope re-entry ----------
verdict = rf["verdict"]
vocab = set(rf["verdict_vocabulary"].keys())
pre_reg = {"SUPPLIED", "SHAPE-UNAVAILABLE", "MAPPING-UNDERIVABLE"}
# wide definition band from R2.json
wide_band = r2["overlap_at_z0"]["vs_definition_band"]["band"]  # [0.2215..., 0.6723...]
# Does R2_final's Part-1 use the narrow below-mean lower edge (0.614) not the wide (0.222)?
rf_band = rf["two_part_test"]["part1_level_z0"]["measured_below_mean_band"]
uses_narrow_lower = approx(rf_band[0], band_lo) and not approx(rf_band[0], wide_band[0])
# Under the WIDE band LB and V0 would both pass -> confirm discrimination is lost with wide band
lb_would_pass_wide = wide_band[0] <= LB_fv0 <= wide_band[1]
v0_would_pass_wide = wide_band[0] <= V0_fv0 <= wide_band[1]
# deep-void edge documented as EXPOSITION-only, fails Part 1 at fv0~0.414
deep_void_fv0 = mdf["shape_matched"]["fv0_implied"]
deep_void_fails_part1 = not (band_lo <= deep_void_fv0 <= band_hi)

results["check4_verdict_token"] = {
    "verdict": verdict,
    "verdict_is_pre_registered": verdict in pre_reg,
    "vocab_matches_pre_registered": vocab == pre_reg,
    "part1_uses_narrow_below_mean_lower_edge": uses_narrow_lower,
    "narrow_lower": band_lo, "wide_lower": wide_band[0],
    "LB_would_pass_under_wide_band": lb_would_pass_wide,
    "V0_would_pass_under_wide_band": v0_would_pass_wide,
    "discrimination_requires_narrow_band": lb_would_pass_wide and v0_would_pass_wide,
    "deep_void_edge_fv0": deep_void_fv0,
    "deep_void_edge_fails_part1_level": deep_void_fails_part1,
    "supersedes_old": rf["supersedes"]["old_verdict"] == "MARGINAL_top_edge"
                      and r2["verdict"] == "MARGINAL_top_edge",
}
results["check4_verdict_token"]["PASS"] = (
    verdict == "SHAPE-UNAVAILABLE"
    and vocab == pre_reg
    and uses_narrow_lower
    and lb_would_pass_wide and v0_would_pass_wide   # wide band would have masked discrimination
    and deep_void_fails_part1
    and results["check4_verdict_token"]["supersedes_old"]
)

# ---------- BONUS: Q prefactor degeneracy at z=0.7 ----------
pref_req = req_fv07 * (1.0 - req_fv07)
pref_obs = obs_fv07 * (1.0 - obs_fv07)
q_req_z07 = pr["derived_backreaction_V"]["Q_over_Hbar0sq"][4]  # z=0.7 index
results["check_bonus_Q"] = {
    "prefactor_req": pref_req, "prefactor_obs": pref_obs,
    "ratio_obs_over_req": pref_obs / pref_req,
    "Q_req_z07_over_Hbar0sq": q_req_z07,
    "matches_rf_ratio": approx(pref_obs / pref_req, 1.0028133279392555, tol=1e-4),
    "matches_rf_Qreq": approx(q_req_z07, 1.284162, tol=1e-4),
}

print(json.dumps(results, indent=2))

with open(os.path.join(OUT, "verify_R2_final.json"), "w") as f:
    json.dump({
        "probe": "verify_R2_final -- independent adversarial recompute of R2_final.json (no import of R2_final.py)",
        "method": "recomputed Phi(sigma/2), decline ratios, band-membership, floor sigma from sigma0 + LCDM D(z); "
                  "erf-based Phi and bisection Phi_inv (no scipy).",
        "inputs_read": ["telescope_fvobs.json", "modelV_probeR.json", "modelV_probeR_LB.json",
                        "mapping_decline_forecast.json", "R2.json", "R2_final.json"],
        "checks": results,
        "verdict_of_verification": "SURVIVES" if all(
            results[k].get("PASS", True) for k in ("check1_part1_level", "check2_part2_shape",
                                                   "check3_z07_floor", "check4_verdict_token")
        ) else "REFUTED",
    }, f, indent=2)
print("WROTE", os.path.join(OUT, "verify_R2_final.json"))
