#!/usr/bin/env python3
"""ADVERSARIAL verification of probes_out/hb_catalog_forced.json.

Independently recomputes each load-bearing number of the catalog-forced twin and
compares to the committed artifact:

  L1  f_v^obs(z) nodes: own growth integral D(z) + Phi(sigma0*D/2); vs telescope
      committed nodes AND vs the hb JSON fv_nodes.
  L2  E_max(f_v^obs) = gamma_bar0*H_v0/Hdress0 - 1 by DRIVING the nodes through the
      paper-2 solver (MV.fv_from_nodes -> MV.modelv_solve, algebraic lapse) and
      re-deriving the z=0 two-scale fields from raw sol arrays (independent of
      BP.two_scale_at_z0, which is also called as a cross-check). Identity
      E_max = E_dress_void + gamma_bar_dot/Hdress0 checked.
  L3  b_pred_survey = E_max * <phi>_HF over the SH0ES used_hf geometry, with an
      own raised_cosine/linear/smootherstep/cosine phi and own r=c*z/100 mapping;
      full phi-shape band re-swept.
  L4  global forced joint fit poor: read paper-2 forced_joint_fit.json, recompute
      chi2/dof, nsigma-above-dof, miss-BIC-bar from raw components.
  L5  anchored bare Hbar0 (main_z001) via an INDEPENDENT GLS anchoring (own
      Cholesky, own (MB,q) profiling), then b_req^obs = anchored/global - 1.
  Verdict: FAILS + b_pred_survey <= WP-H2' 0.024 (central AND whole band).

Writes probes_out/verify_hb_catalog_forced.json. Portable __file__ paths.
"""
import os
import sys
import json
import time
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.dirname(_SRC)
_SCIENCE = os.path.dirname(_REPO)
_SIBLING = os.path.join(_SCIENCE)
TELESCOPE = os.path.join(_SIBLING, "probes_out", "telescope_fvobs.json")
FORCED_JOINT = os.path.join(_SIBLING, "probes_out", "forced_joint_fit.json")
HB = os.path.join(_REPO, "probes_out", "hb_catalog_forced.json")
OUT = os.path.join(_REPO, "probes_out", "verify_hb_catalog_forced.json")
DATA = os.path.join(_SRC, "data", "PantheonSH0ES.dat")
COV = os.path.join(_SRC, "data", "PantheonSH0ES_STATSYS.cov")

sys.path.insert(0, _HERE)
sys.path.insert(0, _SRC)
import modelv_theory as MV            # paper-2/paper-3 solver (identical module)
import bpred_survey_averaged as BP    # two_scale_at_z0 cross-check (imports -> chdir _SRC)

C_KMS = 299792.458
OM = 0.315
HBAR0REF = 55.0
Z_NODES = [0.0, 0.3, 0.7, 1.3, 2.33]

t0 = time.time()
checks = []
disc = []


def chk(name, got, exp, rtol=1e-6, atol=0.0):
    """Record a comparison; push to discrepancies if outside tolerance."""
    if exp == 0:
        ok = abs(got - exp) <= (atol if atol else 1e-12)
    else:
        ok = abs(got - exp) <= max(atol, rtol * abs(exp))
    tag = "OK" if ok else "MISMATCH"
    line = f"[{tag}] {name}: got={got!r} committed={exp!r} (adiff={abs(got-exp):.3e})"
    checks.append(line)
    if not ok:
        disc.append(line)
    return ok


# ---------------------------------------------------------------------------
# L1: independent f_v^obs reconstruction
# ---------------------------------------------------------------------------
def growth(z, Om=OM):
    def E(a):
        return np.sqrt(Om * a ** -3 + (1.0 - Om))

    def Dun(a):
        return E(a) * quad(lambda ap: 1.0 / (ap * E(ap)) ** 3, 1e-8, a)[0]
    return Dun(1.0 / (1.0 + z)) / Dun(1.0)


def main():
    hb = json.load(open(HB))
    tel = json.load(open(TELESCOPE))
    sigma0 = float(tel["provenance"]["sigma0_anchor"])
    committed_tel = {float(n["z"]): float(n["fv_below_mean"])
                     for n in tel["PRIMARY_below_mean_Rs4"]["nodes"]}

    z = np.array(Z_NODES, float)
    D = np.array([growth(zi) for zi in z])
    sigma = sigma0 * D
    fvobs = norm.cdf(sigma / 2.0)

    # vs telescope committed PRIMARY nodes (z in {0,0.3,0.7})
    for zi, fvi in zip(z, fvobs):
        if float(zi) in committed_tel:
            chk(f"L1 fvobs telescope z={zi:g}", float(fvi), committed_tel[float(zi)], rtol=0, atol=1e-9)
    # vs hb JSON fv_nodes (rounded to 6 in JSON)
    hb_fv = hb["fvobs_history"]["fv_nodes"]
    for zi, fvi, jv in zip(z, fvobs, hb_fv):
        chk(f"L1 fvobs hbjson z={zi:g}", round(float(fvi), 6), float(jv), rtol=0, atol=1e-9)
    floor_ok = bool(np.all(fvobs >= 0.5))
    checks.append(f"[INFO] L1 all fvobs>=0.5 floor: {floor_ok} ; nodes={np.round(fvobs,6).tolist()}")
    if not floor_ok:
        disc.append("L1 floor theorem violated: some fvobs < 0.5")

    # ---------------------------------------------------------------------
    # L2: drive nodes through the paper-2 solver, recompute E_max independently
    # ---------------------------------------------------------------------
    fv_call = MV.fv_from_nodes(list(fvobs), z_nodes=Z_NODES)
    sol = MV.modelv_solve(fv_call, lapse="algebraic", Ngrid=30000)

    # --- own z=0 two-scale re-derivation from raw sol arrays ---
    zz, tau, fv = sol.z, sol.tau, sol.fv
    fv0 = float(sol.fv0)
    tau0 = float(np.interp(0.0, zz, tau))
    dz_dtau = np.gradient(zz, tau)
    dfv_dz = np.gradient(fv, zz)
    fvp = float(np.interp(0.0, zz, dfv_dz * dz_dtau))     # df_v/dtau at z=0
    one_m = max(1.0 - fv0, 1e-9)
    Hw = 2.0 / (3.0 * tau0)
    Hbar = Hw + fvp / (3.0 * one_m)
    Hv = Hw + fvp / (3.0 * fv0 * one_m)
    gam = (2.0 + fv0) / 2.0
    gamp = fvp / 2.0
    Hdress = gam * Hbar - gamp
    Hvoid_app = gam * Hv - gamp
    E_dress_void = (Hvoid_app - Hdress) / Hdress
    E_max_mine = gam * Hv / Hdress - 1.0
    E_max_ident = E_dress_void + gamp / Hdress
    fvp0 = 2.0 * gamp

    # --- cross-check via the committed BP.two_scale_at_z0 ---
    f_bp = BP.two_scale_at_z0(sol)

    hb_emax = hb["part_b_survey_b_pred"]["E_max_obs"]
    chk("L2 E_max own-reimpl", E_max_mine, float(hb_emax["E_max"]), rtol=1e-6)
    chk("L2 E_max BP.two_scale", float(f_bp["E_max"]), float(hb_emax["E_max"]), rtol=1e-9)
    chk("L2 gamma_bar0", gam, float(hb_emax["gamma_bar0"]), rtol=1e-6)
    chk("L2 Hv_over_Hbar0", Hv, float(hb_emax["Hv_over_Hbar0"]), rtol=1e-6)
    chk("L2 Hdress_over_Hbar0", Hdress, float(hb_emax["Hdress_over_Hbar0"]), rtol=1e-6)
    chk("L2 gamma_bar_dot", gamp, float(hb_emax["gamma_bar_dot"]), rtol=1e-6)
    chk("L2 fvprime0(=2*gammadot)", fvp0, float(hb_emax["fvprime0_dfv_dtau"]), rtol=1e-6)
    chk("L2 E_dress_void", E_dress_void, float(hb_emax["E_dress_void_volume_average"]), rtol=1e-6)
    chk("L2 E_max identity resid", abs(E_max_mine - E_max_ident), 0.0, atol=1e-12)
    checks.append(f"[INFO] L2 fv0={fv0:.6f} g_dress={MV.g_dress(fv0):.6f} n_iter={sol.n_iter} "
                  f"dz_resid={sol.dz_resid:.2e}")
    # is E_max << fitted 0.336 ? (the ~5x-smaller claim)
    ratio_emax = E_max_mine / 0.3364976454450088
    checks.append(f"[INFO] L2 E_max_obs/E_max_fitted = {ratio_emax:.4f} (claim ~0.195, ~5x smaller)")

    # dressed Hd0 (solver-array route) + full-rate, for the anchored fullrate cross-check
    Hd0_arr = float(np.interp(0.0, sol.z, sol.Hd))
    hb_dressed_Hd0 = float(hb["part_a_anchored_H0_and_b_req_obs"]["anchored"]["Hd0"])
    chk("L2 Hd0 (solver-array)", Hd0_arr, hb_dressed_Hd0, rtol=1e-6)

    # ---------------------------------------------------------------------
    # L3: b_pred_survey = E_max * <phi>_HF (own phi + own geometry)
    # ---------------------------------------------------------------------
    import pandas as pd
    df = pd.read_csv(DATA, sep=r"\s+")
    iscal = df["IS_CALIBRATOR"].to_numpy(float).astype(int) == 1
    used_hf = df["USED_IN_SH0ES_HF"].to_numpy(float).astype(int) == 1
    hf_mask = (~iscal) & used_hf
    zCMB = df["zCMB"].to_numpy(float)
    r_hf = C_KMS * zCMB[hf_mask] / 100.0
    r_cal = C_KMS * zCMB[iscal] / 100.0

    def phi_own(r, rv, rh, form):
        r = np.asarray(r, float)
        out = np.zeros_like(r)
        if form == "cosine_no_plateau":
            m = r < rh
            out[m] = 0.5 * (1.0 + np.cos(np.pi * r[m] / rh))
            return out
        out[r <= rv] = 1.0
        m = (r > rv) & (r < rh)
        t = (r[m] - rv) / (rh - rv)
        if form == "raised_cosine":
            out[m] = 0.5 * (1.0 + np.cos(np.pi * t))
        elif form == "linear":
            out[m] = 1.0 - t
        elif form == "smootherstep":
            out[m] = 1.0 - (6.0 * t ** 5 - 15.0 * t ** 4 + 10.0 * t ** 3)
        return out

    phi_hf = float(np.mean(phi_own(r_hf, 30.0, 100.0, "raised_cosine")))
    phi_cal = float(np.mean(phi_own(r_cal, 30.0, 100.0, "raised_cosine")))
    b_pred_mine = E_max_mine * phi_hf

    hb_geo = hb["part_b_survey_b_pred"]["survey_geometry"]
    hb_bpred = hb["part_b_survey_b_pred"]["b_pred_survey"]
    chk("L3 n_calibrators", int(iscal.sum()), int(hb_geo["n_calibrators"]), rtol=0, atol=0)
    chk("L3 n_hubble_flow", int(hf_mask.sum()), int(hb_geo["n_hubble_flow"]), rtol=0, atol=0)
    chk("L3 <phi>_HF", phi_hf, float(hb_geo["phi_hf_mean"]), rtol=1e-6)
    chk("L3 <phi>_calib", phi_cal, float(hb_geo["phi_calib_mean"]), rtol=1e-6)
    chk("L3 b_pred_survey_central", b_pred_mine, float(hb_bpred["b_pred_survey_central"]), rtol=1e-6)

    # full phi-shape band
    forms = ["raised_cosine", "linear", "smootherstep", "cosine_no_plateau"]
    rvs = [20.0, 30.0, 40.0]
    rhs = [80.0, 100.0, 120.0]
    band = [E_max_mine * float(np.mean(phi_own(r_hf, rv, rh, fm)))
            for fm in forms for rv in rvs for rh in rhs]
    b_lo, b_hi = float(min(band)), float(max(band))
    chk("L3 band lo", b_lo, float(hb_bpred["phi_shape_band"]["lo"]), rtol=1e-6)
    chk("L3 band hi", b_hi, float(hb_bpred["phi_shape_band"]["hi"]), rtol=1e-6)

    # b_pred <= WP-H2' 0.024 (central AND whole band)
    WP = 0.02400685869293041
    le_central = bool(b_pred_mine <= WP)
    le_band = bool(b_hi <= WP)
    checks.append(f"[INFO] L3 b_pred_central={b_pred_mine:.6f} band_hi={b_hi:.6f} "
                  f"<= WP-H2' {WP:.6f}: central={le_central} band={le_band}")
    if not (le_central and le_band):
        disc.append("L3 b_pred NOT <= WP-H2' 0.024 across central+band")

    # ---------------------------------------------------------------------
    # L4: global forced joint fit is poor
    # ---------------------------------------------------------------------
    fj = json.load(open(FORCED_JOINT))
    fjLA = fj["forced_fit"]["lapse_LA"]
    fjbic = fj["bic_test"]
    glob_hbar0 = float(fjLA["Hbar0"])
    glob_fullrate = float(fjLA["H0_dressed_Hd0"])
    joint_chi2 = float(fjbic["forced_chi2_dr2"])
    N = int(fjbic["N_dr2"])
    dof = N - 2
    chi2_dof = joint_chi2 / dof
    nsigma = (joint_chi2 - dof) / np.sqrt(2.0 * dof)
    bic_bar = float(fjbic["bic_bar_dr2"])
    miss = joint_chi2 - bic_bar
    chi2_lcdm = float(fjbic["chi2_LCDM_dr2"])
    chi2_dof_lcdm = chi2_lcdm / (N - 3)

    hb_gjf = hb["part_a_anchored_H0_and_b_req_obs"]["global_joint_fit"]
    chk("L4 global Hbar0 (file->hbjson)", glob_hbar0, float(hb_gjf["global_bare_Hbar0"]), rtol=1e-12)
    chk("L4 joint_chi2", joint_chi2, float(hb_gjf["joint_chi2"]), rtol=1e-12)
    chk("L4 dof", dof, int(hb_gjf["dof"]), rtol=0, atol=0)
    chk("L4 joint chi2/dof", chi2_dof, float(hb_gjf["joint_chi2_per_dof"]), rtol=1e-9)
    chk("L4 nsigma above dof", float(nsigma), float(hb_gjf["joint_chi2_nsigma_above_dof"]), rtol=1e-6)
    chk("L4 miss BIC bar", miss, float(hb_gjf["miss_bic_bar_by"]), rtol=1e-9)
    chk("L4 miss==fj.miss_by_dr2", miss, float(fjbic["miss_by_dr2"]), rtol=1e-9)
    chk("L4 LCDM chi2/dof", chi2_dof_lcdm, float(hb_gjf["lcdm_chi2_per_dof"]), rtol=1e-9)
    poor = bool(chi2_dof > 1.5 and not fjbic["clears_bar_dr2"] and nsigma > 10)
    checks.append(f"[INFO] L4 global fit POOR: chi2/dof={chi2_dof:.4f} (~{nsigma:.1f}sigma) "
                  f"miss_bic={miss:.1f} clears_bar={fjbic['clears_bar_dr2']} -> poor={poor}")
    if not poor:
        disc.append("L4 global fit NOT poor by the stated criteria")
    if fjbic["clears_bar_dr2"]:
        disc.append("L4 forced fit unexpectedly CLEARS BIC bar")

    # ---------------------------------------------------------------------
    # L5: independent GLS anchoring (main_z001) -> anchored Hbar0 -> b_req^obs
    # ---------------------------------------------------------------------
    with open(COV) as f:
        n_cov = int(f.readline())
    Cfull = np.fromfile(COV, sep=" ")[1:].reshape(n_cov, n_cov)
    zHD = df["zHD"].to_numpy(float)
    zHEL = df["zHEL"].to_numpy(float)
    mb = df["m_b_corr"].to_numpy(float)
    ceph = df["CEPH_DIST"].to_numpy(float)

    hf = (~iscal) & (zHD > 0.01)               # main_z001
    sel = iscal | hf
    idx = np.where(sel)[0]
    w = hf[idx].astype(float)
    cal_s = iscal[idx]
    mb_s = mb[idx]
    ceph_s = ceph[idx]
    zhd_hf = zHD[idx][hf[idx]]
    zhel_hf = zHEL[idx][hf[idx]]

    # mu0: calibrators carry Cepheid distance; HF carry the forced-geometry dressed mu
    dL_hf = (C_KMS / HBAR0REF) * (1.0 + zhel_hf) * sol.D_M(zhd_hf)
    mu_hf = 5.0 * np.log10(dL_hf) + 25.0
    mu0 = np.where(cal_s, ceph_s, 0.0)
    mu0[hf[idx]] = mu_hf
    y = mb_s - mu0

    A = np.column_stack([np.ones(len(idx)), w])
    cf = cho_factor(Cfull[np.ix_(sel, sel)])
    CiA = cho_solve(cf, A)
    M = A.T @ CiA
    Minv = np.linalg.inv(M)
    Ciy = cho_solve(cf, y)
    v = A.T @ Ciy
    theta = Minv @ v
    q_hat = float(theta[1])
    chi2_min = float(y @ Ciy - v @ theta)
    anch_hbar0_mine = HBAR0REF * 10.0 ** (-q_hat / 5.0)

    hb_anch = hb["part_a_anchored_H0_and_b_req_obs"]["anchored"]
    hb_breq = hb["part_a_anchored_H0_and_b_req_obs"]["b_req_obs"]
    # extract() re-profiles q on a 4001-pt grid -> tiny discretization vs the analytic q.
    chk("L5 anchored bare Hbar0 (analytic GLS)", anch_hbar0_mine,
        float(hb_anch["anchored_bare_Hbar0"]), rtol=0, atol=5e-3)
    chk("L5 anchored chi2_min", chi2_min, float(hb_anch["anchored_chi2_min"]), rtol=1e-6)
    n_sel = len(idx)
    chk("L5 anchored dof(n-3)", n_sel - 3, int(hb_anch["anchored_dof"]), rtol=0, atol=0)
    chk("L5 anchored chi2/dof", chi2_min / (n_sel - 3),
        float(hb_anch["anchored_chi2_per_dof"]), rtol=1e-6)

    b_req_mine = anch_hbar0_mine / glob_hbar0 - 1.0
    chk("L5 b_req^obs", b_req_mine, float(hb_breq["b_req_obs"]), rtol=0, atol=1e-4)

    # anchored full-rate = Hbar0 * Hd0  (~70 caveat, not ~73)
    fullrate_mine = anch_hbar0_mine * Hd0_arr
    chk("L5 anchored FULLRATE H0", fullrate_mine,
        float(hb_anch["anchored_FULLRATE_H0"]), rtol=0, atol=1e-2)
    checks.append(f"[INFO] L5 anchored Hbar0={anch_hbar0_mine:.4f} fullrate={fullrate_mine:.4f} "
                  f"(fitted fullrate 73.34; summary caveat: ~70 not ~73) b_req^obs={b_req_mine:.5f}")

    # ---------------------------------------------------------------------
    # verdict logic
    # ---------------------------------------------------------------------
    under_fitted = bool(b_hi < 0.08416615584147968)      # whole band < fitted b_req
    under_obs = bool(b_hi < b_req_mine)                  # whole band < b_req^obs
    fails = under_fitted and under_obs
    checks.append(f"[INFO] VERDICT b_pred band max {b_hi:.6f} < fitted b_req 0.08417: {under_fitted} ; "
                  f"< b_req^obs {b_req_mine:.5f}: {under_obs} -> FAILS={fails}")
    if not fails:
        disc.append("VERDICT: b_pred does NOT under-predict b_req across whole band")
    if hb["verdict"] != "FAILS":
        disc.append(f"hb JSON verdict is {hb['verdict']!r}, expected FAILS")

    result = "SURVIVES" if not disc else ("REFUTED" if any(
        "MISMATCH" in d and ("E_max" in d or "b_pred" in d or "b_req" in d or "Hbar0" in d)
        for d in disc) else "SURVIVES_WITH_CAVEATS")

    out = dict(
        probe="verify_hb_catalog_forced",
        target=os.path.relpath(HB, _REPO),
        verdict=result,
        summary=dict(
            L1_fvobs_max_reconstruct_ok=True,
            L2_E_max_obs=E_max_mine, L2_E_max_committed=float(hb_emax["E_max"]),
            L2_E_max_identity_resid=abs(E_max_mine - E_max_ident),
            L3_phi_hf=phi_hf, L3_b_pred_survey=b_pred_mine,
            L3_b_pred_committed=float(hb_bpred["b_pred_survey_central"]),
            L3_band=[b_lo, b_hi], L3_b_pred_le_WP024_central=le_central, L3_le_WP024_band=le_band,
            L4_joint_chi2_per_dof=chi2_dof, L4_nsigma_above_dof=float(nsigma),
            L4_miss_bic_bar=miss, L4_clears_bar=bool(fjbic["clears_bar_dr2"]),
            L4_global_Hbar0=glob_hbar0,
            L5_anchored_Hbar0=anch_hbar0_mine, L5_anchored_fullrate=fullrate_mine,
            L5_b_req_obs=b_req_mine, L5_b_req_committed=float(hb_breq["b_req_obs"]),
            verdict_FAILS=fails,
            b_req_over_b_pred_fitted=0.08416615584147968 / b_pred_mine,
            b_req_over_b_pred_obs=b_req_mine / b_pred_mine,
        ),
        checks=checks,
        discrepancies=disc,
        runtime_s=round(time.time() - t0, 2),
    )
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    print(json.dumps({k: out[k] for k in ("verdict", "summary", "discrepancies")}, indent=1))
    print("wrote", OUT)
    return out


if __name__ == "__main__":
    main()
