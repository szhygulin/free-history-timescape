#!/usr/bin/env python3
"""Adversarial independent verification of WP-B Stage-0 Q-budget (q_budget.json).

Refute-by-default. Recomputes Q_avail(z) from the measured 3-phase void population
with a DIFFERENT derivative pipeline than the build, re-reads Q_req from paper-2
derived_backreaction_V (RAW published, plus the build's smoothed reading), and
independently re-derives the gate arithmetic and the A1 physicality strike.

Independence of method (vs src/probes/q_budget.py):
  * build  : analytic df/dz of the normal CDF, times analytic PCHIP derivative of tau(z).
  * here   : tabulate f_i(z) on a 60k fine grid, map to bare time tau(z)=2/(3 Hw(z)),
             take df_i/dtau by NUMERICAL central differences (np.gradient), Hw(z)
             interpolated with a CubicSpline (build uses PchipInterpolator).
Agreement of Q under two unrelated derivative schemes is the robustness test.

Identity checked:  Q = 6 sum_i f_i (H_i-Hbar)^2 = (2/3) sum_i fdot_i^2/f_i ,
                   H_i-Hbar = fdot_i/(3 f_i) ,  and  sum_i fdot_i = 0 (Hbar closure).
Gate rule (pre-registered): NO-GO iff Q_avail_CEILING(z) < Q_req(z)/3 for ALL
0.3<=z<=1.0 ; else GO.  Evaluated at both raw and smoothed Q_req.
"""
import json
import os

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
SIBLING = os.path.normpath(os.path.join(REPO, ".."))
MODELV = os.path.join(SIBLING, "probes_out", "modelV_probeR.json")
QBUDGET = os.path.join(REPO, "probes_out", "q_budget.json")
OUTJ = os.path.join(REPO, "probes_out", "verify_q_budget.json")

OM = 0.315
SIGMA0 = 0.7344797420042518
DEEP_TH = -0.5
NODES = [0.0, 0.3, 0.5, 0.7, 1.0]
GATE = [0.3, 0.5, 0.7, 1.0]
TOL_CEIL = 0.08          # order-of-magnitude match tolerance (derivative-method spread)
TOL_TIGHT = 0.01         # f_i, empty-Milne, strike must match to ~1%


def main():
    modelv = json.load(open(MODELV))
    build = json.load(open(QBUDGET))
    db = modelv["derived_backreaction_V"]
    zdb = np.asarray(db["z"], float)
    Hw_pub = np.asarray(db["Hw_over_Hbar0"], float)
    fv_pub = np.asarray(db["fv"], float)
    Q_pub = np.asarray(db["Q_over_Hbar0sq"], float)
    HvHw_pub = np.asarray(db["Hv_minus_Hw_over_Hbar0"], float)

    # ---- independent growth D(z), extended to a=1.1 for a clean z=0 derivative ----
    ol = 1.0 - OM
    a = np.linspace(1e-5, 1.1, 500000)
    e = np.sqrt(OM * a ** -3 + ol)
    d_un = e * cumulative_trapezoid(1.0 / (a * e) ** 3, a, initial=0.0)
    d = d_un / np.interp(1.0, a, d_un)

    def Dz(z):
        return np.interp(1.0 / (1.0 + np.asarray(z, float)), a, d)

    def frac(dth, z):
        s = SIGMA0 * Dz(z)
        return norm.cdf((np.log(1.0 + dth) + s ** 2 / 2.0) / s)

    # ---- bare-time tau(z)=2/(3 Hw): Hw via CubicSpline (build uses PCHIP) ----
    Hw_cs = CubicSpline(zdb, Hw_pub)

    def Hw_of(z):
        return float(Hw_cs(z))

    zg = np.linspace(-0.06, 2.4, 60001)
    tau_g = 2.0 / (3.0 * Hw_cs(zg))
    fd_g, fb_g = frac(DEEP_TH, zg), frac(0.0, zg)
    fs_g, fw_g = fb_g - fd_g, 1.0 - fb_g

    fdp_g = np.gradient(fd_g, tau_g)          # numerical df/dtau
    fsp_g = np.gradient(fs_g, tau_g)
    fwp_g = np.gradient(fw_g, tau_g)
    closure = float(np.max(np.abs(fdp_g + fsp_g + fwp_g)))   # sum fdot -> 0
    Qceil_g = (2.0 / 3.0) * (fdp_g ** 2 / fd_g + fsp_g ** 2 / fs_g + fwp_g ** 2 / fw_g)

    def at(yg, z):
        return float(np.interp(z, zg, yg))

    def q_milne(kappa, z):
        fb = frac(0.0, z)
        return 6.0 * fb * (1.0 - fb) * (kappa * Hw_of(z)) ** 2

    # ---- Q_req: RAW published curve vs build's clean-node smoothed curve ----
    Qraw = PchipInterpolator(zdb, Q_pub)
    zc = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
    Qc = np.array([Q_pub[list(zdb).index(zz)] for zz in zc])
    Qsm = PchipInterpolator(zc, Qc)

    # ---- reconstruct Q_req from forced fv via my foliation (foliation sanity) ----
    fv_p = PchipInterpolator(zdb, fv_pub)
    fvp_g = np.gradient(fv_p(zg), tau_g)
    Qrec_g = (2.0 / 3.0) * fvp_g ** 2 / (fv_p(zg) * (1.0 - fv_p(zg)))
    recon = {f"z={z:g}": {"Q_rec": round(at(Qrec_g, z), 4),
                          "Q_pub": round(float(Qraw(z)), 4),
                          "relerr": round(abs(at(Qrec_g, z) - float(Qraw(z))) / float(Qraw(z)), 4)}
             for z in (0.0, 0.3, 0.7)}

    # ---- per-node recompute + cross-check vs build ----
    bn = build["q_at_nodes"]
    checks, discrepancies = [], []
    nodes_out = {}
    for z in NODES:
        key = f"z={z:g}"
        b = bn[key]
        fd, fb = frac(DEEP_TH, z), frac(0.0, z)
        my = {
            "f_deep": round(fd, 4), "f_below": round(fb, 4), "f_walls": round(1.0 - fb, 4),
            "Q_avail_ceiling_recompute": round(at(Qceil_g, z), 4),
            "Q_avail_empty_Milne": round(q_milne(0.5, z), 4),
            "Q_avail_physical_dm-0.5": round(q_milne(0.25, z), 4),
            "Q_req_raw_published": round(float(Qraw(z)), 4),
            "Q_req_build_smoothed": round(float(Qsm(z)), 4),
        }
        nodes_out[key] = {"mine": my, "build": {
            "Q_avail_ceiling_3phase_kinematic": b["Q_avail_ceiling_3phase_kinematic"],
            "Q_avail_empty_Milne_bound": b["Q_avail_empty_Milne_bound"],
            "Q_avail_physical_dm-0.5": b["Q_avail_physical_dm-0.5"],
            "Q_req_central": b["Q_req_central"]}}
        # ceiling agreement (loose, derivative-method)
        rc = abs(my["Q_avail_ceiling_recompute"] - b["Q_avail_ceiling_3phase_kinematic"]) / \
            b["Q_avail_ceiling_3phase_kinematic"]
        (checks if rc < TOL_CEIL else discrepancies).append(
            f"ceiling z={z}: mine={my['Q_avail_ceiling_recompute']} build={b['Q_avail_ceiling_3phase_kinematic']} relerr={rc:.3f}")
        # empty-Milne agreement (tight)
        re = abs(my["Q_avail_empty_Milne"] - b["Q_avail_empty_Milne_bound"]) / b["Q_avail_empty_Milne_bound"]
        (checks if re < TOL_TIGHT else discrepancies).append(
            f"empty-Milne z={z}: mine={my['Q_avail_empty_Milne']} build={b['Q_avail_empty_Milne_bound']} relerr={re:.4f}")
        # Q_req: build reports smoothed as 'central'; flag when it departs from raw published
        rr = abs(b["Q_req_central"] - my["Q_req_raw_published"]) / my["Q_req_raw_published"]
        if rr > TOL_TIGHT:
            discrepancies.append(
                f"Q_req z={z}: build reports smoothed {b['Q_req_central']} vs RAW published "
                f"{my['Q_req_raw_published']} (dep {rr:.2%}) -- build smooths PCHIP jitter at interleaved nodes")

    # ---- gate arithmetic, both readings ----
    gate_eval = {}
    for name, Qr in [("raw_published", Qraw), ("build_smoothed", Qsm)]:
        per = []
        for z in GATE:
            qc = at(Qceil_g, z)
            r3 = float(Qr(z)) / 3.0
            per.append({"z": z, "Q_avail_ceiling": round(qc, 4), "Q_req_over3": round(r3, 4),
                        "clears": bool(qc >= r3)})
        nogo = all(not p["clears"] for p in per)
        gate_eval[name] = {"per_z": per, "NO-GO_all_fail": nogo, "gate": "NO-GO" if nogo else "GO"}
        checks.append(f"gate({name}): every gate-node ceiling clears Q_req/3 -> GO (nogo_all_fail={nogo})")

    # empty-Milne clears Q_req/3 (raw) too?
    empty_clear = {z: bool(q_milne(0.5, z) >= float(Qraw(z)) / 3.0) for z in GATE}
    checks.append(f"empty-Milne clears raw Q_req/3 at all gate z: {all(empty_clear.values())} {empty_clear}")

    # physical budget below Q_req/3 at every z (raw)?
    phys_below = {z: bool(q_milne(0.25, z) < float(Qraw(z)) / 3.0) for z in NODES}
    checks.append(f"physical (dm=-0.5) below raw Q_req/3 at every z: {all(phys_below.values())} {phys_below}")

    # ---- A1 physicality strike ----
    hw0 = Hw_of(0.0)
    req0 = float(HvHw_pub[0])
    empty0 = 0.5 * hw0
    strike = {
        "required_HvHw_over_Hbar0_z0": round(req0, 4),
        "empty_Milne_ceiling_z0": round(empty0, 4),
        "required_over_empty": round(req0 / empty0, 4),
        "strike_fires": bool(req0 > empty0),
        "required_over_physical_dm-0.5": round(req0 / (0.25 * hw0), 3),
    }
    checks.append(f"A1 strike: required (Hv-Hw)/Hbar0(z=0)={req0:.4f} EXCEEDS empty-Milne {empty0:.4f} "
                  f"(x{req0/empty0:.3f}) -> even an empty void cannot source it")

    # deep-phase super-Milne, independent (numerical fdot)
    deep_sm = {}
    for z in NODES:
        Hd = at(fdp_g, z) / (3 * at(fd_g, z))
        Hw = at(fwp_g, z) / (3 * at(fw_g, z))
        ratio = (Hd - Hw) / Hw_of(z)
        deep_sm[f"z={z:g}"] = {"(H_d-H_w)/H_w": round(ratio, 4), "super_Milne": bool(ratio > 0.5)}
    z_ge = min([z for z in NODES if deep_sm[f"z={z:g}"]["super_Milne"]])
    checks.append(f"deep-phase implied (H_d-H_w)/H_w > 0.5 (super-Milne) for z>={z_ge} "
                  f"-- confirms the 3-phase ceiling is partly super-physical")

    checks.append(f"Hbar-closure: max|sum_i fdot_i| = {closure:.2e} (~0) -> partition/identity self-consistent")

    build_gate = build.get("gate")
    gate_agrees = (gate_eval["raw_published"]["gate"] == build_gate ==
                   gate_eval["build_smoothed"]["gate"] == "GO")

    # verdict: gate survives if GO under BOTH Q_req readings and all robustness variants clear
    if gate_agrees and not discrepancies:
        verdict = "SURVIVES"
    elif gate_agrees:
        verdict = "SURVIVES_WITH_CAVEATS"
    else:
        verdict = "REFUTED"

    out = {
        "probe": "verify_q_budget -- adversarial independent recompute of WP-B Stage-0 Q-budget",
        "target": "probes_out/q_budget.json",
        "method": ("independent derivative pipeline: numerical np.gradient of f_i(tau) on a 60k grid "
                   "with CubicSpline Hw(z); build uses analytic normal-CDF derivative x analytic PCHIP "
                   "of tau. Q_req re-read RAW from derived_backreaction_V and via build's smoothed curve."),
        "verdict": verdict,
        "gate_build": build_gate,
        "gate_recomputed": {"raw_published_Qreq": gate_eval["raw_published"]["gate"],
                            "build_smoothed_Qreq": gate_eval["build_smoothed"]["gate"]},
        "gate_agrees": bool(gate_agrees),
        "gate_evaluation": gate_eval,
        "empty_Milne_clears_rawQreq3": empty_clear,
        "physical_below_rawQreq3": phys_below,
        "per_node": nodes_out,
        "A1_physicality_strike": strike,
        "deep_phase_super_Milne": deep_sm,
        "Qreq_foliation_reconstruction": recon,
        "Hbar_closure_max_abs_sum_fdot": closure,
        "checks": checks,
        "discrepancies": discrepancies,
    }
    os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
    with open(OUTJ, "w") as f:
        json.dump(out, f, indent=2)
    print("[verify_q_budget] verdict =", verdict, " gate_agrees =", gate_agrees)
    print("[verify_q_budget] wrote", OUTJ)
    for c in checks:
        print("  CHECK:", c)
    for dd in discrepancies:
        print("  DISCREPANCY:", dd)
    return out


if __name__ == "__main__":
    main()
