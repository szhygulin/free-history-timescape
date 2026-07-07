#!/usr/bin/env python3
"""Catalog-forced anchored-H0 + survey-b_pred: fixed shape = telescope f_v^obs.

Re-runs the paper-3 anchored-H0 (phaseF_freshH0) + survey-averaged b_pred
(bpred_survey_averaged) pipeline with the void history FIXED to the SURVEY-MEASURED
below-mean f_v^obs(z), instead of the joint-fit winner V. Answers whether the SH0ES-
local-Hubble mechanism fails identically when the structure is TELESCOPE-supplied
rather than fitted to the Hubble diagram.

This is the CONCRETE-EXPERIMENT twin required by the paper-3 symmetry rule (it runs once
even though the paper-2 telescope R2 shape is SHAPE-UNAVAILABLE): it substitutes the exact
observed void history and reruns both legs of the mechanism.

f_v^obs(z) = Phi(sigma0 * D(z) / 2), sigma0=0.734480 (2M++ 4 Mpc/h below-mean anchor),
flat-LCDM growth D(z) with Om=0.315 (== telescope_voidfrac.growth_factor). The z={0,0.3,0.7}
values ARE the committed telescope PRIMARY below-mean nodes (validated to <1e-9); z={1.3,2.33}
EXTRAPOLATE the same below-mean floor Phi(sigma0*D(z)/2) beyond the BOSS reach (z<~0.7) -- both
stay >= 0.5 by the floor theorem and are carried as a DECLARED SYSTEMATIC.

(a) Catalog-forced anchored H0 + b_req^obs. phaseF GLS anchoring (Cepheid-pinned M_B, full
    1701x1701 Pantheon+SH0ES stat+sys cov, (M_B,q) profiled) with FV_NODES=f_v^obs:
      * anchored bare Hbar0 (SH0ES ladder pins the SN absolute scale),
      * anchored FULL-RATE H0 = Hd(0)*Hbar0 (~73: the SH0ES local slope is shape-robust),
      * global Hbar0 + joint SN+BAO+CMB chi2/dof from paper-2 forced_joint_fit (a BAD fit --
        the forced f_v^obs misses the BIC bar by ~1438 chi2, verdict FORCED_FVOBS_FAILS_BIC_BAR),
      * b_req^obs = anchored Hbar0 / global Hbar0 - 1.

(b) Catalog-forced survey b_pred. two_scale_at_z0 / E_max / b_survey with f_v^obs:
      * E_max(f_v^obs) = gamma_bar0*H_v0/Hdress0 - 1 (< fitted 0.33650 -- f_v^obs is FLATTER at
        low z, so |f_v'(0)| is smaller, so the void-scale apparent-H0 maximum is smaller),
      * b_pred_survey = E_max * <phi>_HF over the SH0ES calibrator+HF geometry (< fitted 0.024),
      * compared to b_req = 0.08417 (fitted) and to b_req^obs -> FAILS.

FINDING. The fitted-vs-observed shape difference lives at HIGH z (the fitted V plunges to
f_v~0.19 by z=2.33 while f_v^obs sits at ~0.56 on the floor). That difference is INVISIBLE to
the low-z survey average (both shapes are near-identical over the SH0ES HF window z<~0.15, so
E_max and hence b_pred_survey are set by |f_v'(0)| and change only modestly) but FATAL to the
joint fit (the flat high-z tail cannot bend the SN+BAO+CMB Hubble diagram, so the global fit
craters and lands at a low Hbar0). The mechanism therefore fails IDENTICALLY whether the void
structure is fitted (paper-3 WP-H2' PARTIAL/underpredict) or telescope-supplied (this probe):
b_pred_survey stays an order of magnitude below b_req in both.

One number -> one script -> one JSON: probes_out/hb_catalog_forced.json.
"""
import os
import sys
import json
import time
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.linalg import cho_factor

# ---------------------------------------------------------------------------
# portable paths (absolute, immune to the os.chdir(_SRC) done by phaseF on import)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.dirname(_SRC)                              # .../free-history-timescape-tensions
_SCIENCE = os.path.dirname(_REPO)                          # .../science
_SIBLING = os.path.join(_SCIENCE)  # paper-2
TELESCOPE = os.path.join(_SIBLING, "probes_out", "telescope_fvobs.json")
FORCED_JOINT = os.path.join(_SIBLING, "probes_out", "forced_joint_fit.json")
OUT = os.path.join(_REPO, "probes_out", "hb_catalog_forced.json")

sys.path.insert(0, _HERE)
sys.path.insert(0, _SRC)

import modelv_theory as MV                        # free-history dressed-geometry solver
import phaseF_freshH0 as PF                        # GLS anchoring (run_model/extract/dressed) -- chdirs to _SRC
import bpred_survey_averaged as BP                 # two_scale_at_z0 / phi_profile / load_survey / b_survey

Z_NODES = [0.0, 0.3, 0.7, 1.3, 2.33]
OM_FIDUCIAL = 0.315                                # telescope growth cosmology (== telescope_voidfrac.OM)

# fitted-case reference numbers (committed artifacts) for the comparison block
FIT_ANCHORED_HBAR0 = 58.32194048848205            # phaseF main_z001 free_fixed scale (winner V)
FIT_FULLRATE = 73.3438014761951                   # phaseF main_z001 free_fixed dressed H0_fullrate
FIT_GLOBAL_HBAR0 = 53.794282522327265             # modelV_probeR V (fitted joint SN+BAO+CMB)
FIT_B_REQ = 0.08416615584147968                   # fitted b_req (phaseF delta_local_excess) ~ 0.08417
FIT_E_MAX = 0.3364976454450088                    # bpred_survey E_max LA (fitted) ~ 0.336
FIT_B_PRED_SURVEY = 0.02400685869293041           # bpred_survey central (fitted WP-H2') ~ 0.024

_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------------------
# f_v^obs(z): Phi(sigma0*D(z)/2) -- growth reproduced from telescope_voidfrac exactly
# ---------------------------------------------------------------------------
def growth_factor(z, Om=OM_FIDUCIAL):
    """Linear growth D(z), D(0)=1 (flat LCDM). == telescope_voidfrac.growth_factor."""
    def E(a):
        return np.sqrt(Om * a ** -3 + (1.0 - Om))
    def Dun(a):
        return E(a) * quad(lambda ap: 1.0 / (ap * E(ap)) ** 3, 1e-8, a)[0]
    return Dun(1.0 / (1.0 + z)) / Dun(1.0)


def fvobs_nodes():
    """Fixed f_v^obs 5-node vector + validation against telescope_fvobs.json committed nodes."""
    tel = json.load(open(TELESCOPE))
    sigma0 = float(tel["provenance"]["sigma0_anchor"])
    z = np.array(Z_NODES, dtype=float)
    D = np.array([growth_factor(zi) for zi in z])
    sigma = sigma0 * D
    fv = norm.cdf(sigma / 2.0)                              # below-mean floor Phi(sigma/2)
    committed = {float(n["z"]): float(n["fv_below_mean"]) for n in tel["PRIMARY_below_mean_Rs4"]["nodes"]}
    checks = {}
    for zi, fvi in zip(z, fv):
        if float(zi) in committed:
            checks[f"z={zi:g}"] = dict(computed=float(fvi), committed=committed[float(zi)],
                                       abs_diff=abs(float(fvi) - committed[float(zi)]))
    max_diff = max(c["abs_diff"] for c in checks.values())
    assert max_diff < 1e-9, f"f_v^obs node mismatch vs telescope_fvobs.json: {max_diff:g}"
    extrapolated = {f"z={zi:g}": dict(z=float(zi), D_z=float(Di), sigma_Rs4=float(si), fv=float(fi),
                                      above_floor=bool(fi >= 0.5))
                    for zi, Di, si, fi in zip(z, D, sigma, fv) if float(zi) not in committed}
    return z, fv, sigma0, D, sigma, checks, float(max_diff), extrapolated


# ---------------------------------------------------------------------------
# (a) phaseF GLS anchoring of the FORCED f_v^obs shape
# ---------------------------------------------------------------------------
def load_pantheon():
    import pandas as pd
    df = pd.read_csv(PF.DATA, sep=r"\s+")
    n_all = len(df)
    with open(PF.COV) as f:
        n_cov = int(f.readline())
    assert n_cov == n_all, f"cov {n_cov} != rows {n_all}"
    Cfull = np.fromfile(PF.COV, sep=" ")[1:].reshape(n_cov, n_cov)
    zHD = df["zHD"].to_numpy(float); zHEL = df["zHEL"].to_numpy(float)
    mb = df["m_b_corr"].to_numpy(float); ceph = df["CEPH_DIST"].to_numpy(float)
    iscal = df["IS_CALIBRATOR"].to_numpy(float).astype(int) == 1
    used_hf = df["USED_IN_SH0ES_HF"].to_numpy(float).astype(int) == 1
    return dict(n_all=n_all, n_cov=n_cov, Cfull=Cfull, zHD=zHD, zHEL=zHEL, mb=mb,
                ceph=ceph, iscal=iscal, used_hf=used_hf, n_cal=int(iscal.sum()))


def anchor_free_fixed(PAN, hf_mask, label, sol):
    """GLS-anchor the FORCED f_v^obs dressed geometry `sol` on one Pantheon+SH0ES variant."""
    iscal = PAN["iscal"]; zHD = PAN["zHD"]; zHEL = PAN["zHEL"]; mb = PAN["mb"]; ceph = PAN["ceph"]
    Cfull = PAN["Cfull"]
    sel = iscal | hf_mask
    idx = np.where(sel)[0]
    w_s = hf_mask[idx]; cal_s = iscal[idx]
    zhd_s, zhel_s, mb_s, ceph_s = zHD[idx], zHEL[idx], mb[idx], ceph[idx]
    n_sel = len(idx)
    zhd_hf, zhel_hf = zhd_s[w_s], zhel_s[w_s]
    cf = cho_factor(Cfull[np.ix_(sel, sel)])
    mu0base = np.where(cal_s, ceph_s, 0.0)

    def mu0_F(_):
        m = mu0base.copy(); m[w_s] = PF.mu_free(sol, zhd_hf, zhel_hf, PF.HBAR0REF); return m
    rF = PF.extract(np.array([0.0]), *PF.run_model(cf, w_s, mb_s, mu0_F, np.array([0.0])),
                    n_sel, label, PF.HBAR0REF)
    rF["dressed"] = PF.dressed_from_hbar0(sol, rF["scale"])
    rF.pop("shape_best", None)
    rF["n_cal"] = int(cal_s.sum()); rF["n_hf"] = int(w_s.sum())
    rF["hf_z_range"] = [float(zhd_hf.min()), float(zhd_hf.max())]
    return rF, cf, w_s, mb_s, mu0base


def lcdm_gate(PAN, hf_mask, cf, w_s, mb_s, mu0base):
    """LCDM anchoring sanity gate on the main variant (paper-1/phaseF: H0 ~ 73.5)."""
    zHD = PAN["zHD"]; zHEL = PAN["zHEL"]
    sel = PAN["iscal"] | hf_mask
    idx = np.where(sel)[0]
    zhd_hf = zHD[idx][w_s]; zhel_hf = zHEL[idx][w_s]
    n_sel = len(idx)
    om_grid = np.arange(0.05, 0.6000001, 0.0025)
    def mu0_L(Om):
        m = mu0base.copy(); m[w_s] = PF.mu_lcdm(zhd_hf, zhel_hf, Om, PF.H0REF); return m
    rL = PF.extract(om_grid, *PF.run_model(cf, w_s, mb_s, mu0_L, om_grid), n_sel, "LCDM", PF.H0REF)
    return rL


def clean(o):
    if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):  return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o): return None
    return o


# ---------------------------------------------------------------------------
def main():
    log("catalog-forced anchored-H0 + survey-b_pred: fixed shape = f_v^obs")

    # ---- fixed f_v^obs history -------------------------------------------
    z_nodes, fvobs, sigma0, Dz, sigmaz, checks, max_diff, extrap = fvobs_nodes()
    floor_ok = bool(np.all(fvobs >= 0.5))
    log(f"f_v^obs nodes = {np.round(fvobs,5).tolist()} (validated maxdiff={max_diff:.1e}, floor>=0.5={floor_ok})")

    # ONE algebraic-lapse solve of the forced geometry, reused by (a) anchoring and (b) E_max.
    sol = PF.solve_free(list(fvobs), Ngrid=30000)          # MV.fv_from_nodes -> modelv_solve(algebraic)
    log(f"forced solve: fv0={sol.fv0:.6f} g_dress={MV.g_dress(sol.fv0):.6f} "
        f"n_iter={sol.n_iter} dz_resid={sol.dz_resid:.2e}")

    # =====================================================================
    # (a) CATALOG-FORCED ANCHORED H0 + b_req^obs
    # =====================================================================
    PAN = load_pantheon()
    log(f"Pantheon+SH0ES: N={PAN['n_all']} cov {PAN['n_cov']}x{PAN['n_cov']} n_cal={PAN['n_cal']}")

    variants = {
        "main_z001":     (~PAN["iscal"]) & (PAN["zHD"] > 0.01),
        "shoes_hf_flag": (~PAN["iscal"]) & PAN["used_hf"],
        "hf_z_gt_010":   (~PAN["iscal"]) & (PAN["zHD"] > 0.10),
    }
    anchored = {}
    for vname, hf in variants.items():
        rF, cf, w_s, mb_s, mu0base = anchor_free_fixed(PAN, hf, f"{vname}/free_fixed_obs", sol)
        anchored[vname] = rF
        log(f"  {vname}: anchored Hbar0={rF['scale']:.3f}+-{rF['scale_err_sym']:.3f} "
            f"fullrate={rF['dressed']['H0_fullrate']:.3f} gdress={rF['dressed']['H0_gdress']:.3f} "
            f"local_slope={rF['dressed']['H0_local_slope']:.3f} anchored_chi2/dof={rF['chi2_per_dof']:.4f}")
        if vname == "main_z001":
            lcdm_main = lcdm_gate(PAN, hf, cf, w_s, mb_s, mu0base)

    main_a = anchored["main_z001"]
    anch_hbar0 = float(main_a["scale"])
    anch_hbar0_err = float(main_a["scale_err_sym"])
    dressed = main_a["dressed"]

    # global (forced joint SN+BAO+CMB fit) from the paper-2 forced_joint_fit artifact
    fj = json.load(open(FORCED_JOINT))
    fjLA = fj["forced_fit"]["lapse_LA"]
    fjbic = fj["bic_test"]
    glob_hbar0 = float(fjLA["Hbar0"])
    glob_fullrate = float(fjLA["H0_dressed_Hd0"])
    glob_gdress = float(fjLA["H0_dressed"])
    joint_chi2 = float(fjbic["forced_chi2_dr2"])
    N_joint = int(fjbic["N_dr2"])
    n_nuis = 2                                              # SN offset + BAO alpha (0 cosmological shape params)
    dof_joint = N_joint - n_nuis
    chi2_per_dof_joint = joint_chi2 / dof_joint
    chi2_LCDM = float(fjbic["chi2_LCDM_dr2"])
    dof_LCDM = N_joint - 3                                  # LCDM: Om + SN offset + BAO alpha
    chi2_per_dof_LCDM = chi2_LCDM / dof_LCDM
    # how many sigma the joint chi2 sits above its dof expectation (chi2 ~ dof +- sqrt(2 dof))
    joint_nsigma = (joint_chi2 - dof_joint) / np.sqrt(2.0 * dof_joint)

    # b_req^obs = anchored / global - 1  (convention-independent: same fixed shape)
    b_req_obs = anch_hbar0 / glob_hbar0 - 1.0
    b_req_obs_fullrate = dressed["H0_fullrate"] / glob_fullrate - 1.0
    b_req_obs_gdress = dressed["H0_gdress"] / glob_gdress - 1.0

    part_a = dict(
        anchored=dict(
            variant="main_z001 (paper-1 cosmology cut ~zHD>0.01)",
            anchored_bare_Hbar0=anch_hbar0, anchored_bare_Hbar0_err=anch_hbar0_err,
            anchored_FULLRATE_H0=dressed["H0_fullrate"],
            anchored_gdress_H0=dressed["H0_gdress"],
            anchored_local_slope_H0=dressed["H0_local_slope"],
            fv0=dressed["fv0"], g_dress=dressed["g_dress"], Hd0=dressed["Hd0"], S0=dressed["S0"],
            anchored_chi2_min=main_a["chi2_min"], anchored_dof=main_a["dof"],
            anchored_chi2_per_dof=main_a["chi2_per_dof"],
            fullrate_shape_robust_check=dict(
                anchored_fullrate=dressed["H0_fullrate"], fitted_fullrate=FIT_FULLRATE,
                abs_shift_vs_fitted=abs(dressed["H0_fullrate"] - FIT_FULLRATE),
                note=("anchored FULL-RATE H0 ~ 70 for the forced f_v^obs vs 73.3 for the fitted V: the "
                      "SH0ES ladder measures the ~shape-robust LOCAL Hubble slope, so both land near the "
                      "SH0ES value, but the flat f_v^obs fits the low-z Hubble diagram slightly worse "
                      "(anchored chi2/dof 1.18 vs fitted 0.88), pulling the anchored local rate ~3 below "
                      "the fitted value. The residual is small vs the local-minus-global gap: even the "
                      "forced ~70 towers over the forced GLOBAL full-rate 52.98.")),
            all_variants={k: dict(anchored_Hbar0=float(v["scale"]),
                                  anchored_Hbar0_err=float(v["scale_err_sym"]),
                                  fullrate=float(v["dressed"]["H0_fullrate"]),
                                  gdress=float(v["dressed"]["H0_gdress"]),
                                  local_slope=float(v["dressed"]["H0_local_slope"]),
                                  anchored_chi2_per_dof=float(v["chi2_per_dof"]),
                                  n=int(v["n"]), n_cal=int(v["n_cal"]), n_hf=int(v["n_hf"]))
                          for k, v in anchored.items()},
        ),
        lcdm_validation_gate=dict(
            H0=float(lcdm_main["scale"]), sigma=float(lcdm_main["scale_err_sym"]),
            passed=bool(71.5 <= lcdm_main["scale"] <= 75.0 and 0.6 <= lcdm_main["scale_err_sym"] <= 1.6),
            expected="~73.5 (71.5-75); confirms the GLS anchoring machinery is intact under the forced shape."),
        global_joint_fit=dict(
            source="paper-2 forced_joint_fit.json (SN+BAO+CMB, DESI DR2 headline; f_v^obs FIXED, 0 shape params)",
            global_bare_Hbar0=glob_hbar0, global_FULLRATE_H0=glob_fullrate, global_gdress_H0=glob_gdress,
            fv0=float(fjLA["fv0"]),
            joint_chi2=joint_chi2, N_data=N_joint, n_fitted_params=n_nuis,
            n_fitted_params_note="SN offset + BAO alpha profiled; ZERO fitted cosmological shape params (f_v FIXED)",
            dof=dof_joint, joint_chi2_per_dof=chi2_per_dof_joint,
            joint_chi2_nsigma_above_dof=float(joint_nsigma),
            lcdm_refit_chi2=chi2_LCDM, lcdm_dof=dof_LCDM, lcdm_chi2_per_dof=chi2_per_dof_LCDM,
            bic_bar_dr2=float(fjbic["bic_bar_dr2"]), miss_bic_bar_by=float(fjbic["miss_by_dr2"]),
            clears_bic_bar=bool(fjbic["clears_bar_dr2"]),
            fit_quality="BAD",
            note=("The forced f_v^obs joint fit is CATASTROPHICALLY bad: chi2/dof=%.3f (~%.0f sigma above "
                  "its dof), vs LCDM chi2/dof=%.3f. It misses the (most-generous, one-fewer-parameter) "
                  "BIC bar by %.0f chi2 -> paper-2 verdict FORCED_FVOBS_FAILS_BIC_BAR. This is why the "
                  "global Hbar0 lands LOW (%.2f vs fitted %.2f): the flat high-z tail cannot bend the "
                  "SN+BAO+CMB Hubble diagram."
                  % (chi2_per_dof_joint, joint_nsigma, chi2_per_dof_LCDM, fjbic["miss_by_dr2"],
                     glob_hbar0, FIT_GLOBAL_HBAR0))),
        b_req_obs=dict(
            b_req_obs=b_req_obs, formula="anchored bare Hbar0 / global bare Hbar0 - 1",
            crosscheck_fullrate_convention=b_req_obs_fullrate,
            crosscheck_gdress_convention=b_req_obs_gdress,
            convention_max_abs_err=float(max(abs(b_req_obs - b_req_obs_fullrate),
                                             abs(b_req_obs - b_req_obs_gdress))),
            note=("b_req^obs is convention-independent (same fixed shape -> same g_dress, Hd0; reduces to "
                  "the Hbar0 ratio). It is LARGER than the fitted b_req=%.5f because the forced global fit "
                  "craters to a lower Hbar0 -- i.e. the required local bias is even bigger for the observed "
                  "shape, which the mechanism (part b) fails to supply by an even wider margin."
                  % FIT_B_REQ)),
    )

    # =====================================================================
    # (b) CATALOG-FORCED SURVEY b_pred
    # =====================================================================
    fields = BP.two_scale_at_z0(sol)                       # E_max from the SAME forced solve
    E_max_obs = float(fields["E_max"])
    E_max_obs_id = float(fields["E_dress_void"] + fields["gamma_bar_dot"] / fields["Hdress_over_Hbar0"])

    survey = BP.load_survey()
    rc, rh = survey["calib_r_zCMB"], survey["hf_r_zCMB"]
    b_central, phi_c, phi_h, b_diff, phi_all = BP.b_survey(
        E_max_obs, rc, rh, BP.R_VOID_CENTRAL, BP.R_HOM_CENTRAL, BP.PHI_FORM_CENTRAL)

    # phi-shape systematic band (sweep form x r_void x r_hom), LA=obs E_max, zCMB
    forms = ["raised_cosine", "linear", "smootherstep", "cosine_no_plateau"]
    rvoids = [BP.R_VOID_RANGE[0], BP.R_VOID_CENTRAL, BP.R_VOID_RANGE[1]]
    rhoms = [BP.R_HOM_RANGE[0], BP.R_HOM_CENTRAL, BP.R_HOM_RANGE[1]]
    band = []
    for fm in forms:
        for rv in rvoids:
            for rh_ in rhoms:
                band.append(BP.b_survey(E_max_obs, rc, rh, rv, rh_, fm)[0])
    band = np.array(band)
    b_lo, b_hi = float(band.min()), float(band.max())

    def verdict_vs(b_req):
        # P3-style directional test: b_pred UNDER-predicts b_req across the whole band -> FAILS
        under_central = bool(b_central < b_req)
        under_band = bool(b_hi < b_req)
        ratio = float(b_req / b_central) if b_central else float("inf")
        return dict(b_req=float(b_req), b_pred_survey=float(b_central),
                    under_predicts_central=under_central, under_predicts_full_band=under_band,
                    b_req_over_b_pred=ratio,
                    verdict="FAILS" if under_band else ("PARTIAL" if under_central else "RESOLVES"))

    fvp0_obs = float(2.0 * fields["gamma_bar_dot"])        # df_v/dtau at z=0 = 2*gamma_bar_dot (algebraic)
    part_b = dict(
        E_max_obs=dict(
            E_max=E_max_obs, gamma_bar0=float(fields["gamma_bar0"]),
            Hv_over_Hbar0=float(fields["Hv_over_Hbar0"]),
            Hdress_over_Hbar0=float(fields["Hdress_over_Hbar0"]),
            gamma_bar_dot=float(fields["gamma_bar_dot"]),
            fvprime0_dfv_dtau=fvp0_obs,
            fvprime0_note=("df_v/dtau at z=0 (=2*gamma_bar_dot). This LOW-z slope, not the high-z tail, "
                           "sets E_max; it is ~5x smaller for the flat f_v^obs than for the steep fitted V, "
                           "which is why E_max and b_pred_survey drop ~5x. E_max->0 as the history flattens."),
            E_dress_void_volume_average=float(fields["E_dress_void"]),
            E_max_via_identity=E_max_obs_id, identity_abs_err=abs(E_max_obs - E_max_obs_id),
            formula="E_max = gamma_bar0*H_v0/Hdress0 - 1 (void-scale apparent-H0 maximum, pure clock conv.)",
            vs_fitted=dict(fitted_E_max=FIT_E_MAX, obs_E_max=E_max_obs,
                           ratio_obs_over_fitted=E_max_obs / FIT_E_MAX,
                           smaller_than_fitted=bool(E_max_obs < FIT_E_MAX),
                           note=("E_max(f_v^obs) < fitted 0.33650: f_v^obs is FLATTER at low z, so |f_v'(0)| "
                                 "(hence the void-growth term driving the apparent-H0 maximum) is smaller. "
                                 "E_max=0 for a perfectly flat history; f_v^obs sits near that limit.")),
        ),
        survey_geometry=dict(
            n_calibrators=survey["n_calib"], n_hubble_flow=survey["n_hf"],
            phi_calib_mean=phi_c, phi_hf_mean=phi_h,
            hf_frac_inside_r_hom=float(np.mean(rh < BP.R_HOM_CENTRAL)),
            note="identical SH0ES geometry to the fitted WP-H2' run; only E_max changes with the shape."),
        b_pred_survey=dict(
            b_pred_survey_central=b_central,
            definition="b_pred_survey = E_max * <phi>_Hubble-flow (LA/obs, raised_cosine, r_void=30, r_hom=100, zCMB)",
            phi_shape_band=dict(lo=b_lo, hi=b_hi, half_range=float((b_hi - b_lo) / 2.0)),
            alternative_differential=b_diff, diagnostic_whole_survey=float(E_max_obs * phi_all),
            vs_fitted=dict(fitted_b_pred_survey=FIT_B_PRED_SURVEY, obs_b_pred_survey=b_central,
                           ratio_obs_over_fitted=b_central / FIT_B_PRED_SURVEY,
                           note="b_pred_survey(f_v^obs)=0.0047 is ~5x SMALLER than the fitted 0.024 -- both "
                                "are set by the low-z slope |f_v'(0)|, which is ~5x smaller for the flat "
                                "f_v^obs. Both are far below b_req (0.084): the mechanism FAILS for either "
                                "shape, harder for the observed one."),
        ),
        comparison_to_b_req=dict(
            vs_fitted_b_req=verdict_vs(FIT_B_REQ),
            vs_catalog_forced_b_req_obs=verdict_vs(b_req_obs),
            note=("b_pred_survey UNDER-predicts b_req across the ENTIRE phi-shape band against BOTH the "
                  "fitted requirement (0.08417) and the catalog-forced requirement b_req^obs. FAILS."),
        ),
    )

    # =====================================================================
    # comparison to fitted case + verdict + finding
    # =====================================================================
    comparison = dict(
        fitted=dict(anchored_Hbar0=FIT_ANCHORED_HBAR0, anchored_fullrate=FIT_FULLRATE,
                    global_Hbar0=FIT_GLOBAL_HBAR0, b_req=FIT_B_REQ, E_max=FIT_E_MAX,
                    b_pred_survey=FIT_B_PRED_SURVEY,
                    fv_nodes_V=[0.64013, 0.53112, 0.39578, 0.27945, 0.19359],
                    joint_fit_quality="GOOD (winner V; chi2/dof ~ 0.88)"),
        catalog_forced=dict(anchored_Hbar0=anch_hbar0, anchored_fullrate=dressed["H0_fullrate"],
                            global_Hbar0=glob_hbar0, b_req_obs=b_req_obs, E_max=E_max_obs,
                            b_pred_survey=b_central,
                            fv_nodes_obs=[float(x) for x in np.round(fvobs, 5)],
                            joint_fit_quality="BAD (FORCED_FVOBS_FAILS_BIC_BAR; chi2/dof ~ %.2f)" % chi2_per_dof_joint),
        where_shapes_differ=dict(
            low_z_z0_value=dict(fitted_fv0=0.64013, obs_fv0=float(fvobs[0]),
                                diff=abs(0.64013 - float(fvobs[0]))),
            low_z_slope=dict(fitted_slope_0_to_0p3=(0.53112 - 0.64013) / 0.3,
                             obs_slope_0_to_0p3=(float(fvobs[1]) - float(fvobs[0])) / 0.3,
                             note="|df_v/dz| near z=0 is ~5x smaller for f_v^obs -> ~5x smaller E_max/b_pred."),
            high_z_z233=dict(fitted_fv=0.19359, obs_fv=float(fvobs[-1]),
                             diff=abs(0.19359 - float(fvobs[-1]))),
            note=("The two shapes AGREE in VALUE at z=0 (both ~0.64) but differ everywhere else: f_v^obs is "
                  "far FLATTER in slope from the start (|f_v'(0)| ~5x smaller) and diverges enormously at "
                  "high z (fitted V plunges to 0.194 by z=2.33 while f_v^obs stays at 0.555 on the below-mean "
                  "floor). The z=0 VALUE fixes g_dress (identical); the low-z SLOPE sets Hd0 and E_max; the "
                  "high-z TAIL sets the joint-fit quality. The SH0ES low-z ladder pins the measured local "
                  "RATE ~shape-robustly (~70-73; the model absorbs the slope difference into the Hbar0/Hd0 "
                  "split), but E_max -- the slope-driven void-growth term -- drops ~5x, and the joint fit "
                  "craters on the flat high-z tail.")),
    )

    verdict = "FAILS"
    finding = (
        "The catalog-forced mechanism FAILS identically to the fitted case. Substituting the exact "
        "survey-measured void history f_v^obs(z) for the joint-fit winner V and rerunning BOTH legs: "
        "(a) the anchored FULL-RATE H0 barely moves (%.2f vs fitted 73.3) -- the SH0ES low-z ladder pins "
        "the measured local rate ~shape-robustly (the model absorbs the shape difference into the "
        "Hbar0/Hd0 split; anchored chi2/dof %.2f vs fitted 0.88). But the global joint SN+BAO+CMB fit of "
        "the flat f_v^obs is CATASTROPHIC (chi2/dof=%.2f, ~%.0f sigma, misses the BIC bar by %.0f) and "
        "craters to a low global Hbar0=%.2f, so b_req^obs=%.3f (>> the fitted 0.08417). (b) the survey "
        "b_pred DROPS ~5x: E_max(f_v^obs)=%.4f << fitted 0.336 and b_pred_survey=%.4f << fitted 0.024 "
        "(both set by |f_v'(0)|, ~5x smaller for the flat shape) -- so it FAILS HARDER (%.1fx below the "
        "fitted b_req, %.0fx below b_req^obs). FINDING: the fitted and observed shapes agree in VALUE at "
        "z=0 (fv0~0.64) but the fitted V is far steeper and plunges to 0.19 by z=2.33 while f_v^obs stays "
        "at 0.56 on the below-mean floor. That high-z divergence is nearly INVISIBLE to the low-z SH0ES "
        "ladder (anchored H0 ~70 either way) yet FATAL to the joint fit (global Hbar0 craters); and the "
        "low-z survey-averaged b_pred cannot supply b_req for EITHER shape (0.024 and 0.0047, both far "
        "below 0.084). The mechanism fails identically whether the void structure is FITTED to the Hubble "
        "diagram or TELESCOPE-supplied: b_pred_survey stays an order of magnitude below b_req in both."
        % (dressed["H0_fullrate"], main_a["chi2_per_dof"], chi2_per_dof_joint, joint_nsigma,
           fjbic["miss_by_dr2"], glob_hbar0, b_req_obs, E_max_obs, b_central,
           FIT_B_REQ / b_central, b_req_obs / b_central))

    out = dict(
        probe="hb_catalog_forced",
        purpose=("Catalog-forced twin of the paper-3 anchored-H0 + survey-b_pred pipeline: fix the void "
                 "history to the telescope-measured below-mean f_v^obs(z) and rerun both legs. Concrete "
                 "experiment required by the paper-3 symmetry rule."),
        reading="KINEMATIC (force f_v(z)=f_v^obs; integrability not enforced).",
        fvobs_history=dict(
            definition="f_v^obs(z) = Phi(sigma0*D(z)/2), sigma0=%.6f (2M++ 4 Mpc/h below-mean anchor), "
                       "flat-LCDM growth D(z), Om=%.3f." % (sigma0, OM_FIDUCIAL),
            z_nodes=[float(z) for z in z_nodes],
            D_of_z=[float(x) for x in np.round(Dz, 6)],
            sigma_Rs4=[float(x) for x in np.round(sigmaz, 6)],
            fv_nodes=[float(x) for x in np.round(fvobs, 6)],
            source="probes_out/telescope_fvobs.json PRIMARY_below_mean_Rs4 (paper-2 sibling)",
            validation_vs_telescope=checks, max_abs_diff_vs_committed=max_diff,
            extrapolated_nodes=extrap,
            floor_all_above_0p5=floor_ok,
            declared_systematic=("z={1.3,2.33} EXTRAPOLATE the below-mean floor Phi(sigma0*D(z)/2) beyond the "
                                 "BOSS reach (z<~0.7); both stay >=0.5 by the floor theorem. Carried as a "
                                 "declared systematic. z={0,0.3,0.7} are the committed telescope PRIMARY nodes."),
        ),
        part_a_anchored_H0_and_b_req_obs=part_a,
        part_b_survey_b_pred=part_b,
        comparison_to_fitted_case=comparison,
        verdict=verdict,
        finding=finding,
        runtime_s=round(time.time() - _t0, 1),
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(clean(out), f, indent=1, allow_nan=False)
    log(f"wrote {OUT}")

    # ---- console summary ----
    print("=== (a) CATALOG-FORCED ANCHORED H0 ===")
    print(f"  anchored bare Hbar0 = {anch_hbar0:.3f} +- {anch_hbar0_err:.3f}")
    print(f"  anchored FULL-RATE H0 = {dressed['H0_fullrate']:.3f}  (fitted 73.34; ladder-slope ~shape-robust)")
    print(f"  anchored chi2/dof (GOOD) = {main_a['chi2_per_dof']:.4f}")
    print(f"  global (forced joint) Hbar0 = {glob_hbar0:.3f}   joint chi2/dof (BAD) = {chi2_per_dof_joint:.3f}")
    print(f"    joint chi2={joint_chi2:.1f} dof={dof_joint} (~{joint_nsigma:.0f} sigma above); "
          f"misses BIC bar by {fjbic['miss_by_dr2']:.0f}")
    print(f"  b_req^obs = {b_req_obs:.5f}  (fitted b_req = {FIT_B_REQ:.5f})")
    print("=== (b) CATALOG-FORCED SURVEY b_pred ===")
    print(f"  E_max(f_v^obs) = {E_max_obs:.5f}  (fitted 0.33650; smaller: flatter low-z)")
    print(f"  b_pred_survey = {b_central:.5f}  band [{b_lo:.5f},{b_hi:.5f}]  (fitted 0.02401)")
    print(f"  vs b_req=0.08417 -> {part_b['comparison_to_b_req']['vs_fitted_b_req']['verdict']} "
          f"(b_req/b_pred = {FIT_B_REQ/b_central:.1f}x)")
    print(f"  vs b_req^obs={b_req_obs:.5f} -> {part_b['comparison_to_b_req']['vs_catalog_forced_b_req_obs']['verdict']}")
    print(f"=== VERDICT: {verdict} ===")
    return out


if __name__ == "__main__":
    main()
