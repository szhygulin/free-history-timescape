#!/usr/bin/env python3
"""Survey-averaged b_pred via a radial apparent-H0 profile (paper-3 WP-H2', PLAN P2/P3/P4).

THE DECISIVE COMPUTATION. Turns the void-scale apparent-H0 MAXIMUM (E_max) into the
survey-weighted local Hubble bias the SH0ES ladder actually samples, by averaging a radial
apparent-H0 excess profile dH(<r)/Hdress = E_max * phi(r) over the SH0ES calibrator +
Hubble-flow survey geometry, and compares it once to b_req.

WHAT CHANGED FROM THE PRIOR COMMITTED b_pred (bpred_local_excess.json).
  The prior b_pred used E_dress_void = (Hvoid_app - Hdress)/Hdress with
  Hvoid_app = gamma_bar0*H_v - gamma_bar_dot. That SUBTRACTS the global dressed-scale-factor
  term gamma_bar_dot -> the VOLUME-AVERAGE excess (0.19480 on LA). That is NOT the void-scale
  maximum. The wall-observer void-crossing rate is a PURE clock conversion gamma_bar0*H_v
  (NO subtraction), so the void-scale MAXIMUM is
        E_max = gamma_bar0*H_v0/Hdress0 - 1.
  Reconciliation (exact identity, verified in gate b):
        E_max = E_dress_void + gamma_bar_dot/Hdress.
  ON THE TRACKER this reduces to E_max(fv0) = (3/2)/g_dress(fv0) - 1
        = 3(2+fv0)/(4 fv0^2 + fv0 + 4) - 1,   which reproduces Wiltshire's window (gate a).
  ON A FREE HISTORY (LA/LB) it does NOT reduce to (3/2)/g_dress -- the ACTUAL solution values
  gamma_bar0, H_v0, Hdress0 are used (gate b: E_max(LA)=0.33650).

THREE VALIDATION GATES (must PASS before any survey number; PLAN sec 10 discipline):
  (a) DERIVE the Wiltshire 17-22% window on the tracker: E_max(0.76)=0.171, E_max(0.695)=0.220;
      down-side wall floor 1-1/g_dress(0.76)=0.219; "75 km/s/Mpc" check 61.7*(1+E_max).
      Refs: Wiltshire 2009 (arXiv:0909.0749); review (arXiv:0912.5234).
  (b) small-r ceiling: phi(0)=1 -> dH(<r->0)/Hdress = E_max(LA) = 0.33650.
  (c) large-r limit: phi(r>=r_hom)=0 -> dH -> 0 (global dressed rate past homogeneity).

RADIAL PROFILE. phi(r): plateau at the void-interior maximum out to the dominant-void scale
r_void ~ 30 h^-1 Mpc, smooth monotone decline to 0 by the homogeneity scale r_hom ~ 100 h^-1
Mpc (Wiltshire's radial prescription; e.g. Wiltshire, Smale, Mattsson & Watson 2013,
arXiv:1201.5371). (r_void, r_hom) and the functional form are DECLARED SYSTEMATICS (Wiltshire's
values -- NOT fitted); the survey-averaged b_pred's sensitivity to them is the DOMINANT
systematic band. Optional 2M++ (Carrick+15) cross-check of the homogeneity scale (not a fit).

SURVEY AVERAGING. Each SN -> r_i = c*z_i/100 (h^-1 Mpc; zCMB first-order primary, zHD systematic
per PLAN P4). Two populations: 77 Cepheid calibrators (z~0.001-0.017, r<~51, inside the bump)
and the Hubble-flow SNe (z~0.023-0.15, r~69-450, mostly past r_hom). The survey-averaged local
Hubble bias is the offset between the calibrator-pinned local rate and the HF global rate:
        b_pred_survey = E_max * ( <phi>_calibrators - <phi>_Hubble-flow ).
EXPECT strong dilution: ~all HF SNe sit past r_hom (phi->0), so b_pred_survey << E_max.

Writes probes_out/bpred_survey_averaged.json. One number -> one script -> one JSON.
"""
import os
import sys
import json
import warnings
import numpy as np
from scipy.optimize import brentq

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.dirname(_SRC)                        # .../free-history-timescape-tensions
_SCIENCE = os.path.dirname(_REPO)                    # .../science
os.chdir(_SRC)                                       # so fit_timescape's data/ paths resolve
sys.path.insert(0, _HERE)
sys.path.insert(0, _SRC)
import modelv_theory as MV                            # free-history dressed-geometry solver

_POUT = os.path.join(_REPO, "probes_out")
OUT = os.path.join(_POUT, "bpred_survey_averaged.json")
PHASEF = os.path.join(_POUT, "phaseF_freshH0.json")
ADV = os.path.join(_POUT, "adv_anchored_h0.json")
DATA = os.path.join(_SRC, "data", "PantheonSH0ES.dat")
LB_ARTIFACT = os.path.join(_SCIENCE, "probes_out",
                           "modelV_probeR_LB.json")
# audit sec.3: paper-2 two-sided void-wall contrast budget (cross-repo, portable).
CONTRAST_BUDGET = os.path.join(_SCIENCE, "probes_out",
                               "contrast_budget.json")
TWOMPP = os.path.join(_REPO, "external_data", "twompp_density.npy")

C_KMS = 299792.458

# Probe-R free-history histories (LA winner, LB winner) on the default nodes.
Z_NODES = [0.0, 0.3, 0.7, 1.3, 2.33]
FV_NODES_V = [0.64013, 0.53112, 0.39578, 0.27945, 0.19359]     # LA (algebraic) winner
FV_NODES_LB = [0.58786, 0.49709, 0.38473, 0.2503, 0.12145]     # LB (rate_ratio) winner

# LB two-scale fields embedded fallback (verbatim from 1b-A modelV_probeR_LB.json two_scale_z0_LB);
# the paper-3 rate_ratio solver diverges on the LB nodes, so LB is READ, not recomputed.
LB_FALLBACK = {
    "fv0": 0.5878612293793102, "Hv_over_Hbar0": 1.1983344389865043,
    "Hbar_over_Hbar0": 1.022971258376305,
    "gamma_bar0_LB": 1.3236247572964668, "gamma_bar_dot_LB": 0.04140610935253841,
    "Hdress_over_Hbar0": 1.3126239742370596,
    "E_dress_void_PRIMARY": 0.17683285688029907,
}

# Wiltshire radial prescription anchors (DECLARED SYSTEMATICS -- not fitted).
R_VOID_CENTRAL = 30.0      # dominant-void scale [h^-1 Mpc]
R_HOM_CENTRAL = 100.0      # statistical-homogeneity scale [h^-1 Mpc]
R_VOID_RANGE = (20.0, 40.0)
R_HOM_RANGE = (80.0, 120.0)
PHI_FORM_CENTRAL = "raised_cosine"

# Wiltshire (2009) tracker window endpoints (gate a targets, 3 sig figs).
WILT_TARGET_076 = 0.171
WILT_TARGET_0695 = 0.220
WILT_DOWN_FLOOR_076 = 0.219   # 1 - 1/g_dress(0.76): "present epoch variance 22%"
WILT_DRESSED_GLOBAL_H0 = 61.7 # km/s/Mpc, dressed global rate for the "75 km/s/Mpc" check


# ---------------------------------------------------------------------------
# tracker algebraic identity (gate a)
# ---------------------------------------------------------------------------
def g_dress(fv0):
    fv0 = float(fv0)
    return (4.0 * fv0 ** 2 + fv0 + 4.0) / (2.0 * (2.0 + fv0))


def E_max_tracker(fv0):
    """Void-scale maximum on the tracker: (3/2)/g_dress - 1 = 3(2+fv0)/(4fv0^2+fv0+4) - 1."""
    fv0 = float(fv0)
    return 3.0 * (2.0 + fv0) / (4.0 * fv0 ** 2 + fv0 + 4.0) - 1.0


# ---------------------------------------------------------------------------
# free-history two-scale decomposition at z=0 (recompute LA; read LB)
# ---------------------------------------------------------------------------
def two_scale_at_z0(sol):
    """Exact two-scale decomposition at z=0 for the algebraic lapse (units of Hbar0).

    Returns gamma_bar0, H_v0, Hbar0, Hdress0, gamma_bar_dot, E_dress_void, E_max.
    Reproduces bpred_local_excess.two_scale_at_z0 / modelv_probeR.derived_curves at z=0.
    """
    z, tau, fv = sol.z, sol.tau, sol.fv
    fv0 = float(sol.fv0)
    tau0 = float(np.interp(0.0, z, tau))
    dz_dtau = np.gradient(z, tau)
    dfv_dz = np.gradient(fv, z)
    fvp = float(np.interp(0.0, z, dfv_dz * dz_dtau))            # df_v/dtau (>0)
    one_m = max(1.0 - fv0, 1e-9)

    Hw = 2.0 / (3.0 * tau0)                                     # H_w/Hbar0
    dHvw = fvp / (3.0 * fv0 * one_m)                            # (H_v - H_w)/Hbar0
    Hbar = Hw + fvp / (3.0 * one_m)                            # <H>/Hbar0
    Hv = Hw + dHvw                                             # H_v/Hbar0

    gam = (2.0 + fv0) / 2.0                                     # algebraic lapse
    gamp = fvp / 2.0                                            # dgamma_bar/dtau
    Hdress = gam * Hbar - gamp                                 # dressed present rate = Hd(0)
    Hvoid_app = gam * Hv - gamp                                # dressed void rate (subtracted term)

    E_dress_void = (Hvoid_app - Hdress) / Hdress               # VOLUME-AVERAGE excess (prior b_pred)
    E_max = gam * Hv / Hdress - 1.0                           # VOID-SCALE MAXIMUM (pure clock conv.)
    return dict(fv0=fv0, gamma_bar0=gam, gamma_bar_dot=gamp,
                Hv_over_Hbar0=Hv, Hbar_over_Hbar0=Hbar, Hdress_over_Hbar0=Hdress,
                Hvoid_app_over_Hbar0=Hvoid_app,
                E_dress_void=E_dress_void, E_max=E_max)


def load_lb_fields():
    """LB two-scale fields: prefer the live 1b-A artifact, else the embedded fallback."""
    if os.path.exists(LB_ARTIFACT):
        try:
            art = json.load(open(LB_ARTIFACT))
            b = art["two_scale_z0_LB"]
            blk = dict(fv0=float(b["fv0"]), Hv_over_Hbar0=float(b["Hv_over_Hbar0"]),
                       Hbar_over_Hbar0=float(b["Hbar_over_Hbar0"]),
                       gamma_bar0=float(b["gamma_bar0_LB"]),
                       gamma_bar_dot=float(b["gamma_bar_dot_LB"]),
                       Hdress_over_Hbar0=float(b["Hdress_over_Hbar0"]),
                       E_dress_void=float(b["E_dress_void_PRIMARY"]))
            blk["_source"] = "sibling_artifact:" + os.path.relpath(LB_ARTIFACT, _REPO)
            return blk
        except Exception as e:  # pragma: no cover
            warnings.warn(f"LB artifact unreadable ({e!r}); using embedded fallback")
    blk = dict(fv0=LB_FALLBACK["fv0"], Hv_over_Hbar0=LB_FALLBACK["Hv_over_Hbar0"],
               Hbar_over_Hbar0=LB_FALLBACK["Hbar_over_Hbar0"],
               gamma_bar0=LB_FALLBACK["gamma_bar0_LB"],
               gamma_bar_dot=LB_FALLBACK["gamma_bar_dot_LB"],
               Hdress_over_Hbar0=LB_FALLBACK["Hdress_over_Hbar0"],
               E_dress_void=LB_FALLBACK["E_dress_void_PRIMARY"])
    blk["_source"] = "embedded_fallback"
    return blk


def E_max_from_fields(f):
    """E_max = gamma_bar0*H_v0/Hdress0 - 1 (void-scale maximum) with cross-check identity."""
    E_max = f["gamma_bar0"] * f["Hv_over_Hbar0"] / f["Hdress_over_Hbar0"] - 1.0
    E_via_id = f["E_dress_void"] + f["gamma_bar_dot"] / f["Hdress_over_Hbar0"]
    return E_max, E_via_id


# ---------------------------------------------------------------------------
# audit sec.3: PHYSICAL void rate from the paper-2 two-sided contrast budget
# ---------------------------------------------------------------------------
# The CEILING E_max above uses the model's DERIVED/EMPTY-void rate Hv_over_Hbar0 (=1.273 on
# LA, a sec.2 void-contrast h_v=H_v/H_bg=1.273, i.e. a void deeper than lensing measures).
# The PHYSICAL row CAPS that void rate by sec.2's lensing-measured two-sided void-wall
# contrast (H_v-H_w)/<H>: substitute the sec.2 spherical-patch void rate h_v=H_v/H_bg for
# Hv_over_Hbar0 (identifying sec.2's background H_bg with the model's Hbar0 -- the *background*
# normalization; the model's <H>/Hbar0=1.088 differs only via backreaction), keeping the SAME
# gamma_bar0, Hdress0 and the SAME survey dilution <phi>_HF. Only the void-scale ceiling changes.
#
# Embedded sec.2 fallback (verbatim from ../free-history-timescape/probes_out/contrast_budget.json)
# so this stays runnable if the paper-2 sibling is absent. h_v = H_v/H_bg spherical-patch void
# rates; two_sided = (H_v-H_w)/<H> at z=0 they imply.
SEC2_FALLBACK = {
    "field_mean_hv_r200": 1.1527073299823296,        # self-consistent field-mean void (r200), delta~-0.45
    "field_mean_hv_r100": 1.1757957893985875,        # self-consistent field-mean void (r100)
    "deep_void_center_m0p8_hv": 1.3361087210110663,  # lensing deepest void center delta_m=-0.8
    "field_mean_two_sided_z0_r200": 0.375,
    "field_mean_two_sided_z0_r100": 0.40102,
    "deep_void_center_m0p8_two_sided_z0": 0.50223,   # delta_m=-0.8 void x field-effective wall
    "required_two_sided_z0_central": 0.47409,
    "empty_void_ceiling_hv": 1.5,                    # EdS empty-void h_v ceiling (sec.2 reference)
}


def load_sec2_contrast():
    """sec.2 physical void rates h_v=H_v/H_bg + two-sided contrasts (H_v-H_w)/<H>: prefer the
    paper-2 contrast_budget.json sibling (portable __file__-derived path), else the embedded
    fallback so this artifact stays runnable stand-alone."""
    if os.path.exists(CONTRAST_BUDGET):
        try:
            cb = json.load(open(CONTRAST_BUDGET))
            fa = cb["available_field_averaged"]
            grid = {round(float(g["delta_void_central"]), 1): g for g in cb["available_grid"]}
            g08 = grid[-0.8]
            blk = dict(
                field_mean_hv_r200=float(fa["r200"]["hv"]),
                field_mean_hv_r100=float(fa["r100"]["hv"]),
                deep_void_center_m0p8_hv=float(g08["h_v_central"]),
                field_mean_two_sided_z0_r200=float(fa["r200"]["excess_z0"]),
                field_mean_two_sided_z0_r100=float(fa["r100"]["excess_z0"]),
                deep_void_center_m0p8_two_sided_z0=float(g08["wall_effective"]["excess_z0"]),
                required_two_sided_z0_central=float(cb["required"]["central"][0]),
                empty_void_ceiling_hv=float(cb["spherical_model"]["empty_void_ceiling_hv_over_bg"]),
            )
            blk["_source"] = "sibling_artifact:" + os.path.relpath(CONTRAST_BUDGET, _REPO)
            return blk
        except Exception as e:  # pragma: no cover
            warnings.warn(f"contrast_budget unreadable ({e!r}); using embedded sec.2 fallback")
    blk = dict(SEC2_FALLBACK)
    blk["_source"] = "embedded_fallback"
    return blk


def E_max_physical(fields, h_v_phys):
    """Void-scale maximum with the void rate CAPPED by sec.2's physical h_v=H_v/H_bg:
    E_max_phys = gamma_bar0 * h_v_phys / Hdress0 - 1, substituting the lensing-physical void
    rate for the model's DERIVED Hv_over_Hbar0 while keeping gamma_bar0, Hdress0 from the SAME
    free-history solution as the ceiling."""
    return fields["gamma_bar0"] * float(h_v_phys) / fields["Hdress_over_Hbar0"] - 1.0


# ---------------------------------------------------------------------------
# radial apparent-H0 profile phi(r): phi(0)=1, monotone decline, phi(r>=r_hom)=0
# ---------------------------------------------------------------------------
def phi_profile(r, r_void=R_VOID_CENTRAL, r_hom=R_HOM_CENTRAL, form="raised_cosine"):
    """Radial apparent-H0 excess shape. phi(0)=1 (void-interior maximum), phi(r>=r_hom)=0.

    Forms (all pinned ONLY by (r_void, r_hom), monotone non-increasing):
      raised_cosine    : plateau to r_void, C1 cosine taper to r_hom            (central)
      linear           : plateau to r_void, linear taper to r_hom              (Wiltshire shells)
      smootherstep     : plateau to r_void, quintic (6t^5-15t^4+10t^3) taper    (softest knee)
      cosine_no_plateau: cosine taper from r=0 to r_hom (r_void unused)         (earliest decline)
    """
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    if form == "cosine_no_plateau":
        m = r < r_hom
        out[m] = 0.5 * (1.0 + np.cos(np.pi * r[m] / r_hom))
        return out
    plateau = r <= r_void
    out[plateau] = 1.0
    taper = (r > r_void) & (r < r_hom)
    t = (r[taper] - r_void) / (r_hom - r_void)
    if form == "raised_cosine":
        out[taper] = 0.5 * (1.0 + np.cos(np.pi * t))
    elif form == "linear":
        out[taper] = 1.0 - t
    elif form == "smootherstep":
        out[taper] = 1.0 - (6.0 * t ** 5 - 15.0 * t ** 4 + 10.0 * t ** 3)
    else:
        raise ValueError(f"unknown phi form {form!r}")
    return out


# ---------------------------------------------------------------------------
# SH0ES survey geometry
# ---------------------------------------------------------------------------
def load_survey():
    """Parse PantheonSH0ES.dat -> (r_calib, r_hf) for zCMB and zHD, plus counts and z-ranges."""
    import pandas as pd
    df = pd.read_csv(DATA, sep=r"\s+")
    iscal = df["IS_CALIBRATOR"].to_numpy(float).astype(int) == 1
    used_hf = df["USED_IN_SH0ES_HF"].to_numpy(float).astype(int) == 1
    hf = (~iscal) & used_hf
    zCMB = df["zCMB"].to_numpy(float)
    zHD = df["zHD"].to_numpy(float)

    def r_of(z):
        return C_KMS * z / 100.0  # h^-1 Mpc

    survey = dict(
        n_calib=int(iscal.sum()), n_hf=int(hf.sum()),
        calib_zCMB=zCMB[iscal], calib_zHD=zHD[iscal],
        hf_zCMB=zCMB[hf], hf_zHD=zHD[hf],
        calib_r_zCMB=r_of(zCMB[iscal]), calib_r_zHD=r_of(zHD[iscal]),
        hf_r_zCMB=r_of(zCMB[hf]), hf_r_zHD=r_of(zHD[hf]),
    )
    return survey


def b_survey(E_max, r_calib, r_hf, r_void=R_VOID_CENTRAL, r_hom=R_HOM_CENTRAL,
             form="raised_cosine"):
    """Survey-averaged local Hubble bias the SH0ES ladder measures.

    PRIMARY (physically derived): b_pred_survey = E_max * <phi>_HF.
      The Cepheid geometric distances that fix M_B are LOCAL wall-frame physical distances,
      unaffected by the local apparent EXPANSION rate, so M_B is recovered true. The ladder's
      H0 then follows from the Hubble-flow SNe: H0_meas = cz/d_L = the LOCAL apparent rate AT
      the HF SNe = Hdress*(1+E_max*phi(r_hf)). Hence the ladder's fractional bias vs the global
      dressed rate is E_max*<phi>_HF -- the calibrator apparent rate does NOT enter (its only
      role, fixing M_B, is unbiased). "HF past r_hom -> phi->0" therefore DILUTES the bias.

    Returns (b_primary, phi_calib_mean, phi_hf_mean, b_differential, phi_all_mean).
      b_differential = E_max*(<phi>_calib - <phi>_HF) is reported as a labelled ALTERNATIVE
      (the value IF the calibrator anchoring were itself biased by the local rate -- it is not,
      for geometric Cepheid distances); b_all = E_max*<phi>_survey is a diagnostic only.
    """
    phi_c = float(np.mean(phi_profile(r_calib, r_void, r_hom, form)))
    phi_h = float(np.mean(phi_profile(r_hf, r_void, r_hom, form)))
    phi_all = float(np.mean(phi_profile(np.concatenate([r_calib, r_hf]), r_void, r_hom, form)))
    b_primary = E_max * phi_h
    b_diff = E_max * (phi_c - phi_h)
    return b_primary, phi_c, phi_h, b_diff, phi_all


# ---------------------------------------------------------------------------
# 2M++ homogeneity-scale cross-check (declared systematic; NOT a fit)
# ---------------------------------------------------------------------------
def twompp_homogeneity_scale():
    """Spherically-averaged cumulative overdensity delta(<r) around the Local Group.

    The scale where the cumulative volume-averaged density contrast settles to ~0 is an
    external anchor for r_hom; the innermost shells anchor the local-void extent. Cross-check
    only -- (r_void, r_hom) are Wiltshire's declared values, never fitted to this field.
    """
    try:
        d = np.load(TWOMPP)  # (257,257,257), Galactic Cartesian, 1.5625 h^-1 Mpc spacing
        n = d.shape[0]
        c = n // 2                                  # LG at central voxel [128,128,128]
        dx = 400.0 / 256.0                          # h^-1 Mpc per cell
        ax = (np.arange(n) - c) * dx
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        R = np.sqrt(X * X + Y * Y + Z * Z)
        edges = np.arange(0.0, 160.0 + dx, 5.0)
        rc = 0.5 * (edges[:-1] + edges[1:])
        # cumulative volume-average delta within r (equal-volume voxels -> plain mean)
        cum = np.array([float(d[R <= e].mean()) if np.any(R <= e) else np.nan
                        for e in edges[1:]])
        # shell-average delta
        shell = np.array([float(d[(R > lo) & (R <= hi)].mean())
                          if np.any((R > lo) & (R <= hi)) else np.nan
                          for lo, hi in zip(edges[:-1], edges[1:])])
        # homogeneity scale: first r where |cum| < 0.02 and stays (approach to mean density)
        r_hom_est = None
        for i in range(len(rc)):
            if np.isfinite(cum[i]) and abs(cum[i]) < 0.02:
                r_hom_est = float(rc[i])
                break
        return dict(
            available=True, lg_overdensity=float(d[c, c, c]),
            radii=[round(x, 3) for x in rc.tolist()],
            cumulative_delta=[None if not np.isfinite(x) else round(x, 5) for x in cum.tolist()],
            shell_delta=[None if not np.isfinite(x) else round(x, 5) for x in shell.tolist()],
            r_homogeneity_est_cum_lt_0p02=r_hom_est,
            note=("Cumulative volume-averaged 2M++ (Carrick+15) overdensity around the LG. "
                  "|delta(<r)|<0.02 first reached near r_homogeneity_est -- consistent with the "
                  "adopted r_hom~100 h^-1 Mpc to order-of-magnitude. CROSS-CHECK ONLY; r_void/r_hom "
                  "are Wiltshire's declared values, not fitted here."))
    except Exception as e:  # pragma: no cover
        return dict(available=False, error=repr(e))


# ---------------------------------------------------------------------------
# global-fit Hbar0 SCALE error: sigma_global = sigma(Hbar0_glob)/Hbar0_glob
# ---------------------------------------------------------------------------
def compute_sigma_global(sol):
    """Fractional Hbar0 SCALE error of the joint SN+BAO+CMB fit at the LA best-fit
    history (replaces the prior 0.010 ESTIMATE).

    Route: Hbar0_glob = c/(alpha r_d) with alpha the BAO/CMB scale, so
    sigma_global = sigma(Hbar0_glob)/Hbar0_glob = sigma(alpha)/alpha at the fixed LA
    history. sigma(alpha) is read from the JOINT chi2 curvature vs alpha: the SN term
    marginalizes M_B and is FLAT in alpha, so the absolute scale is pinned by BAO+CMB
    (dominated by the Planck acoustic-scale point). chi2_joint(alpha) is fit by a
    parabola about the best alpha; sigma(alpha) = 1/sqrt(0.5 d2chi2/dalpha2). Uses the
    paper-2 harness (byte-identical sibling of paper-3's shared harness).
    """
    import io
    import contextlib
    _p2 = os.path.join(_SCIENCE, "src")
    if _p2 not in sys.path:
        sys.path.insert(0, _p2)
    with contextlib.redirect_stdout(io.StringIO()):
        import harness as H
    rows = H.bao_cmb_rows()
    g = np.array([sol.predict(z, k) for (z, k, _v, _e, _c) in rows])
    Cinv, DV = H._CINV, H._DV
    _, alpha_best = H.bao_cmb_chi2(sol.predict)       # analytic best alpha at fixed shape
    zHD = H.load_sn()[0]
    chi2_sn = float(H.sn_chi2(sol.D_M(zHD)))          # constant in alpha (M_B marginalized)

    def chi2_joint(a):
        r = DV - a * g
        return chi2_sn + float(r @ (Cinv @ r))

    da = 0.02 * alpha_best
    alphas = alpha_best + da * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    chi2s = np.array([chi2_joint(a) for a in alphas])
    d2chi2 = 2.0 * float(np.polyfit(alphas, chi2s, 2)[0])   # d2chi2/dalpha2 from the parabola
    sigma_alpha = 1.0 / np.sqrt(0.5 * d2chi2)
    Hbar0_glob = float(H.H0_from_alpha(alpha_best))
    return dict(
        sigma_global=float(sigma_alpha / alpha_best),
        alpha_best=float(alpha_best), sigma_alpha=float(sigma_alpha),
        Hbar0_glob_from_alpha=Hbar0_glob, d2chi2_dalpha2=d2chi2,
        method=("sigma(alpha)/alpha from the JOINT SN+BAO+CMB chi2 curvature vs the BAO/CMB "
                "scale alpha at the LA best-fit history (SN flat in alpha; parabola about "
                "alpha_best; sigma(alpha)=1/sqrt(0.5 d2chi2/dalpha2)); paper-2 harness "
                "bao_cmb_chi2. Replaces the prior 0.010 ESTIMATE."))


def solve_rhom_star(E_max, r_hf, b_req_val, r_void=R_VOID_CENTRAL, form=PHI_FORM_CENTRAL):
    """r_hom that WOULD make b_pred_survey = b_req: <phi>_HF(r_void, r_hom) = b_req/E_max.

    <phi>_HF is monotone increasing in r_hom, so a bracketed root exists whenever the
    target b_req/E_max is reachable (< 1). Returns (r_hom_star, target); r_hom_star is
    None if the target exceeds the reachable maximum.
    """
    target = float(b_req_val) / float(E_max)

    def resid(r_hom):
        return float(np.mean(phi_profile(r_hf, r_void, r_hom, form))) - target

    lo, hi = float(r_void) + 1e-6, 5000.0
    if resid(hi) < 0.0:                     # target unreachable even for a very large r_hom
        return None, target
    return float(brentq(resid, lo, hi, xtol=1e-4)), target


# ---------------------------------------------------------------------------
def main():
    out = {
        "probe": "bpred_survey_averaged",
        "purpose": ("Survey-averaged b_pred: the void-scale apparent-H0 MAXIMUM E_max averaged over "
                    "a radial apparent-H0 profile dH(<r)/Hdress = E_max*phi(r) with the SH0ES "
                    "calibrator+Hubble-flow survey geometry, its systematic band, and the "
                    "pre-registered P3 verdict vs b_req."),
        "reading": "KINEMATIC (forced f_v(z); integrability not enforced).",
    }

    # =====================================================================
    # GATE (a): DERIVE the Wiltshire 17-22% window on the tracker
    # =====================================================================
    e076 = E_max_tracker(0.76)
    e0695 = E_max_tracker(0.695)
    down076 = 1.0 - 1.0 / g_dress(0.76)
    h0_up_076 = WILT_DRESSED_GLOBAL_H0 * (1.0 + e076)
    h0_down_076 = WILT_DRESSED_GLOBAL_H0 * (1.0 + down076)
    a_pass = (round(e076, 3) == WILT_TARGET_076) and (round(e0695, 3) == WILT_TARGET_0695) \
        and (round(down076, 3) == WILT_DOWN_FLOOR_076)
    out["gate_a_tracker_window"] = dict(
        PASS=bool(a_pass),
        E_max_tracker_fv0_0p76=e076, target_0p76=WILT_TARGET_076,
        E_max_tracker_fv0_0p695=e0695, target_0p695=WILT_TARGET_0695,
        down_side_wall_floor_0p76=down076, target_down=WILT_DOWN_FLOOR_076,
        pairing_up_endpoints=dict(up_at_0p76=e076, up_at_0p695=e0695,
                                  reading="up-side excess at the two tracker endpoints -> the "
                                          "17-22% window as an fv0-range [0.76,0.695]"),
        pairing_up_down_at_0p76=dict(up_at_0p76=e076, down_at_0p76=-down076,
                                     reading="up/down apparent-Hubble variance at a single fv0=0.76 "
                                             "-> +0.171 / -0.219 (the '22% present-epoch variance')"),
        km_s_Mpc_check=dict(dressed_global_H0=WILT_DRESSED_GLOBAL_H0,
                            H0_up_side_0p76=h0_up_076, H0_down_side_0p76=h0_down_076,
                            note="61.7*(1+E_max): up-side->%.1f, down-side->%.1f km/s/Mpc; the "
                                 "down-side matches Wiltshire's ~75 km/s/Mpc maximal local value."
                                 % (h0_up_076, h0_down_076)),
        identity="E_max(fv0) = (3/2)/g_dress(fv0) - 1 = 3(2+fv0)/(4 fv0^2 + fv0 + 4) - 1",
        references=["Wiltshire 2009 arXiv:0909.0749", "review arXiv:0912.5234"],
        note=("Both pairings reported, neither chosen: the historical '17-22%' is EITHER the "
              "up-side excess across the tracker fv0 endpoints [0.76->0.171, 0.695->0.220] OR "
              "the up/down apparent-Hubble variance at fv0=0.76 [+0.171/-0.219]."),
    )

    # =====================================================================
    # free-history E_max (LA recomputed via solver; LB read from 1b-A artifact)
    # =====================================================================
    fvA = MV.fv_from_nodes(FV_NODES_V, z_nodes=Z_NODES)
    solA = MV.modelv_solve(fvA, lapse="algebraic", Ngrid=30000)
    fields_LA = two_scale_at_z0(solA)
    E_max_LA = fields_LA["E_max"]
    E_max_LA_id = fields_LA["E_dress_void"] + fields_LA["gamma_bar_dot"] / fields_LA["Hdress_over_Hbar0"]

    fields_LB = load_lb_fields()
    E_max_LB, E_max_LB_id = E_max_from_fields(fields_LB)

    out["E_max"] = dict(
        tracker=dict(fv0_0p76=e076, fv0_0p695=e0695),
        LA=dict(E_max=E_max_LA, gamma_bar0=fields_LA["gamma_bar0"],
                Hv_over_Hbar0=fields_LA["Hv_over_Hbar0"],
                Hdress_over_Hbar0=fields_LA["Hdress_over_Hbar0"],
                gamma_bar_dot=fields_LA["gamma_bar_dot"],
                E_dress_void_volume_average=fields_LA["E_dress_void"],
                E_max_via_identity=E_max_LA_id,
                identity_abs_err=abs(E_max_LA - E_max_LA_id),
                formula="E_max = gamma_bar0*H_v0/Hdress0 - 1 (pure clock conversion, NO gamma_bar_dot "
                        "subtraction)"),
        LB=dict(E_max=E_max_LB, gamma_bar0=fields_LB["gamma_bar0"],
                Hv_over_Hbar0=fields_LB["Hv_over_Hbar0"],
                Hdress_over_Hbar0=fields_LB["Hdress_over_Hbar0"],
                gamma_bar_dot=fields_LB["gamma_bar_dot"],
                E_dress_void_volume_average=fields_LB["E_dress_void"],
                E_max_via_identity=E_max_LB_id,
                identity_abs_err=abs(E_max_LB - E_max_LB_id),
                source=fields_LB["_source"]),
    )

    # =====================================================================
    # GATE (b): small-r ceiling recovers E_max(LA) = 0.33650
    # =====================================================================
    phi0 = float(phi_profile(0.0, form=PHI_FORM_CENTRAL))
    small_r_limit = E_max_LA * phi0
    b_pass = (round(small_r_limit, 5) == 0.33650) and (abs(E_max_LA - E_max_LA_id) < 1e-9)
    out["gate_b_small_r_ceiling"] = dict(
        PASS=bool(b_pass),
        phi_at_r0=phi0,
        small_r_limit=small_r_limit,
        target_E_max_LA=0.33650,
        E_max_LA=E_max_LA,
        identity_crosscheck=dict(
            E_max=E_max_LA, E_dress_void_plus_gammadot_over_Hdress=E_max_LA_id,
            abs_err=abs(E_max_LA - E_max_LA_id),
            identity="E_max = E_dress_void + gamma_bar_dot/Hdress"),
        reconciliation=("0.19480 = VOLUME-AVERAGE excess E_dress_void = (Hvoid_app-Hdress)/Hdress "
                        "with Hvoid_app = gamma_bar0*H_v - gamma_bar_dot (subtracts the global "
                        "dressed-scale-factor term). 0.33650 = VOID-SCALE MAXIMUM E_max = "
                        "gamma_bar0*H_v0/Hdress0 - 1 (pure wall-observer clock conversion, NO "
                        "subtraction). They differ by exactly gamma_bar_dot/Hdress = %.5f. The "
                        "committed Wave-1b b_pred (bpred_local_excess.json) mislabeled the "
                        "volume-average 0.19480 as 'the maximum'; the robust-FAILS verdict there is "
                        "UNCHANGED and STRENGTHENED by the larger true ceiling."
                        % (fields_LA["gamma_bar_dot"] / fields_LA["Hdress_over_Hbar0"])),
    )

    # =====================================================================
    # GATE (c): large-r limit -> 0
    # =====================================================================
    phi_rhom = float(phi_profile(R_HOM_CENTRAL, form=PHI_FORM_CENTRAL))
    phi_beyond = float(phi_profile(R_HOM_CENTRAL * 1.5, form=PHI_FORM_CENTRAL))
    large_r_limit = E_max_LA * phi_rhom
    c_pass = (abs(large_r_limit) < 1e-9) and (abs(phi_beyond) < 1e-12)
    out["gate_c_large_r_limit"] = dict(
        PASS=bool(c_pass),
        phi_at_r_hom=phi_rhom, phi_beyond_r_hom=phi_beyond,
        large_r_limit=large_r_limit,
        r_hom=R_HOM_CENTRAL,
        note="dH(<r)/Hdress -> 0 past the homogeneity scale r_hom~100 h^-1 Mpc: the survey averages "
             "over the global dressed rate there (no local void bump).",
    )

    gates_all_pass = a_pass and b_pass and c_pass
    out["gates_summary"] = dict(gate_a=bool(a_pass), gate_b=bool(b_pass), gate_c=bool(c_pass),
                                all_pass=bool(gates_all_pass))

    # STOP if any gate fails: no survey number is emitted without all three PASSing.
    if not gates_all_pass:
        out["survey_bpred"] = dict(
            status="NOT_COMPUTED",
            reason="A validation gate FAILED; per PLAN sec 10 no survey-averaged b_pred is emitted "
                   "until gates a/b/c all PASS. Inspect gates_summary.")
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
        print("!! GATE FAILURE -- see gates_summary; survey number withheld.")
        print(json.dumps(out["gates_summary"], indent=1))
        print(f"wrote {OUT}")
        return

    # =====================================================================
    # radial profile + SH0ES survey averaging
    # =====================================================================
    survey = load_survey()
    rc_cmb, rh_cmb = survey["calib_r_zCMB"], survey["hf_r_zCMB"]
    rc_hd, rh_hd = survey["calib_r_zHD"], survey["hf_r_zHD"]

    # central point prediction (LA, raised_cosine, r_void=30, r_hom=100, zCMB)
    b_central, phi_c, phi_h, b_diff, phi_all = b_survey(
        E_max_LA, rc_cmb, rh_cmb, R_VOID_CENTRAL, R_HOM_CENTRAL, PHI_FORM_CENTRAL)

    out["radial_profile"] = dict(
        form_central=PHI_FORM_CENTRAL, r_void_central=R_VOID_CENTRAL, r_hom_central=R_HOM_CENTRAL,
        r_void_range=list(R_VOID_RANGE), r_hom_range=list(R_HOM_RANGE),
        forms_tested=["raised_cosine", "linear", "smootherstep", "cosine_no_plateau"],
        prescription=("phi(0)=1 (void-interior maximum), plateau to r_void, smooth monotone decline "
                      "to 0 by r_hom; Wiltshire radial apparent-H0 prescription (cf. Wiltshire, "
                      "Smale, Mattsson & Watson 2013 arXiv:1201.5371). (r_void,r_hom) & form are "
                      "DECLARED SYSTEMATICS, not fitted."),
        phi_at_check_radii=dict(
            r0=float(phi_profile(0.0, form=PHI_FORM_CENTRAL)),
            r_void=float(phi_profile(R_VOID_CENTRAL, form=PHI_FORM_CENTRAL)),
            r50=float(phi_profile(50.0, form=PHI_FORM_CENTRAL)),
            r_hom=float(phi_profile(R_HOM_CENTRAL, form=PHI_FORM_CENTRAL))),
    )

    out["survey_geometry"] = dict(
        n_calibrators=survey["n_calib"], n_hubble_flow=survey["n_hf"],
        calib_z_range_zCMB=[float(survey["calib_zCMB"].min()), float(survey["calib_zCMB"].max())],
        hf_z_range_zCMB=[float(survey["hf_zCMB"].min()), float(survey["hf_zCMB"].max())],
        calib_r_range_zCMB=[float(rc_cmb.min()), float(rc_cmb.max())],
        hf_r_range_zCMB=[float(rh_cmb.min()), float(rh_cmb.max())],
        calib_frac_inside_r_hom=float(np.mean(rc_cmb < R_HOM_CENTRAL)),
        hf_frac_inside_r_hom=float(np.mean(rh_cmb < R_HOM_CENTRAL)),
        hf_frac_inside_r_void=float(np.mean(rh_cmb < R_VOID_CENTRAL)),
        r_mapping="r_i = c*z_i/100 [h^-1 Mpc]; zCMB first-order (primary), zHD (P4 systematic).",
    )

    out["b_pred_survey"] = dict(
        b_pred_survey_central=b_central,
        E_max_used=E_max_LA, lapse="LA",
        phi_hf_mean=phi_h, phi_calib_mean=phi_c, phi_survey_mean=phi_all,
        definition="b_pred_survey = E_max * <phi>_Hubble-flow. The ladder's H0 = cz/d_L over the "
                   "Hubble-flow SNe traces the LOCAL apparent rate at the HF SNe = "
                   "Hdress*(1+E_max*phi(r_hf)); M_B is fixed by geometric Cepheid distances "
                   "(unbiased by the local apparent rate), so the calibrator apparent rate does "
                   "NOT enter. Fractional bias vs the global dressed rate = E_max*<phi>_HF, the "
                   "analogue of b_req = Hbar0_anchored/Hbar0_global - 1.",
        alternative_differential=dict(
            b_calib_minus_hf=b_diff,
            note="ALTERNATIVE, NOT primary: E_max*(<phi>_calib - <phi>_HF). Would apply only IF the "
                 "calibrator anchoring were itself biased by the local apparent rate; it is not "
                 "(Cepheid distances are geometric). Reported for transparency: because calibrators "
                 "sit deep in the void (<phi>_calib=%.3f) this OVERSHOOTS to %.5f -- physically "
                 "inapplicable, shown only to bracket the definitional choice." % (phi_c, b_diff)),
        diagnostic_whole_survey=dict(
            b_all=E_max_LA * phi_all,
            note="Diagnostic only: E_max*<phi> over calibrators+HF together (no physical basis for "
                 "co-averaging the two populations)."),
        conventions="LA E_max, raised_cosine phi, r_void=30, r_hom=100 h^-1 Mpc, zCMB.",
    )

    # =====================================================================
    # audit sec.3: PHYSICAL-void sensitivity row (the CEILING row above is UNCHANGED).
    # Cap the void rate H_v by sec.2's LENSING-measured two-sided contrast band instead of
    # the model's derived/empty-void maximum. b_pred_physical = E_max_physical * <phi>_HF with
    # the SAME survey dilution <phi>_HF (phi_h) -- only the void-scale ceiling changes.
    # PRIMARY: the self-consistent field-mean void (h_v = field-averaged value). Band spans the
    # sec.2 modeling range: field-mean (conservative) -> deep-void-center delta_m=-0.8 (generous).
    # nsigma filled in below once b_req + sigma are computed. Ceiling/physical PAIRING is the point.
    # =====================================================================
    sec2 = load_sec2_contrast()
    phi_hf_used = phi_h                          # same survey dilution as the ceiling row
    hv_fm_r200 = sec2["field_mean_hv_r200"]      # conservative field-mean anchor
    hv_fm_r100 = sec2["field_mean_hv_r100"]
    hv_fm_mid = 0.5 * (hv_fm_r200 + hv_fm_r100)  # field-mean central (PRIMARY)
    hv_deep = sec2["deep_void_center_m0p8_hv"]   # generous deep-void-center delta_m=-0.8

    E_phys_fm_r200 = E_max_physical(fields_LA, hv_fm_r200)
    E_phys_fm_r100 = E_max_physical(fields_LA, hv_fm_r100)
    E_phys_fm_mid = E_max_physical(fields_LA, hv_fm_mid)
    E_phys_deep = E_max_physical(fields_LA, hv_deep)

    b_phys_fm_r200 = E_phys_fm_r200 * phi_hf_used
    b_phys_fm_r100 = E_phys_fm_r100 * phi_hf_used
    b_phys_fm_mid = E_phys_fm_mid * phi_hf_used   # PRIMARY central b_pred_physical
    b_phys_deep = E_phys_deep * phi_hf_used

    # band spans conservative field-mean (lowest realistic b_pred) -> generous deep-void
    E_phys_band = [E_phys_fm_r200, E_phys_deep]
    b_phys_band = [b_phys_fm_r200, b_phys_deep]

    out["b_pred_survey_PHYSICAL"] = dict(
        role="LENSING-PHYSICAL sensitivity row (audit sec.3). The CEILING row (E_max=%.5f, "
             "b_pred_survey=%.5f) above is UNCHANGED -- the ceiling/physical PAIRING is the point."
             % (E_max_LA, b_central),
        formula="E_max_physical = gamma_bar0 * h_v_phys / Hdress0 - 1 ; b_pred_physical = "
                "E_max_physical * <phi>_HF. h_v_phys = sec.2 spherical-patch void rate H_v/H_bg "
                "(paper-2 contrast_budget.json), substituted for the model's DERIVED Hv_over_Hbar0 "
                "(=%.5f). gamma_bar0=%.5f, Hdress0=%.5f, <phi>_HF=%.6f held FIXED from the ceiling "
                "row (survey dilution unchanged)."
                % (fields_LA["Hv_over_Hbar0"], fields_LA["gamma_bar0"],
                   fields_LA["Hdress_over_Hbar0"], phi_hf_used),
        sec2_source=sec2["_source"],
        physical_contrast_used=dict(
            two_sided_contrast_definition="(H_v-H_w)/<H> at z=0 (paper-2 audit sec.2)",
            field_mean_two_sided_z0_band=[sec2["field_mean_two_sided_z0_r200"],
                                          sec2["field_mean_two_sided_z0_r100"]],
            deep_void_center_m0p8_two_sided_z0=sec2["deep_void_center_m0p8_two_sided_z0"],
            required_two_sided_z0_central=sec2["required_two_sided_z0_central"],
            empty_void_ceiling_hv=sec2["empty_void_ceiling_hv"],
            note="sec.2 two-sided budget: self-consistent field-mean 0.375-0.401 (r200-r100), deep "
                 "delta_m=-0.8 void x field-effective wall 0.502; required 0.474; strike MARGINAL. "
                 "The PHYSICAL row caps the void RATE h_v by these measured contents; the model's "
                 "derived void rate h_v=%.5f is deeper than the field-mean (h_v~%.3f-%.3f), so the "
                 "field-mean cap REDUCES E_max and STRENGTHENS the FAILS."
                 % (fields_LA["Hv_over_Hbar0"], hv_fm_r200, hv_fm_r100)),
        void_anchors=dict(
            field_mean_r200_conservative=dict(h_v_phys=hv_fm_r200, E_max_physical=E_phys_fm_r200,
                                              b_pred_physical=b_phys_fm_r200),
            field_mean_midpoint_PRIMARY=dict(h_v_phys=hv_fm_mid, E_max_physical=E_phys_fm_mid,
                                             b_pred_physical=b_phys_fm_mid),
            field_mean_r100=dict(h_v_phys=hv_fm_r100, E_max_physical=E_phys_fm_r100,
                                 b_pred_physical=b_phys_fm_r100),
            deep_void_center_m0p8_generous=dict(h_v_phys=hv_deep, E_max_physical=E_phys_deep,
                                                b_pred_physical=b_phys_deep),
        ),
        E_max_physical_central=E_phys_fm_mid,
        E_max_physical_band=E_phys_band,
        E_max_physical_band_note="[field-mean r200 conservative, deep-void delta_m=-0.8 generous]",
        b_pred_physical_central=b_phys_fm_mid,
        b_pred_physical_band=b_phys_band,
        b_pred_physical_band_note="[field-mean r200 conservative, deep-void delta_m=-0.8 generous]",
        ceiling_reference=dict(E_max_ceiling=E_max_LA, b_pred_survey_ceiling=b_central,
                               note="derived/EMPTY-void maximum -- UNCHANGED by this row."),
        # nsigma_physical_* + verdict_* filled in after b_req + sigma are computed (see below).
    )

    # =====================================================================
    # systematic band: sweep form x (r_void, r_hom) x z-choice x lapse
    # =====================================================================
    forms = ["raised_cosine", "linear", "smootherstep", "cosine_no_plateau"]
    rvoids = [R_VOID_RANGE[0], R_VOID_CENTRAL, R_VOID_RANGE[1]]
    rhoms = [R_HOM_RANGE[0], R_HOM_CENTRAL, R_HOM_RANGE[1]]

    def sweep(E_max, rc, rh, forms_=forms, rvs=rvoids, rhs=rhoms):
        vals = []
        for fm in forms_:
            for rv in rvs:
                for rh_ in rhs:
                    b = b_survey(E_max, rc, rh, rv, rh_, fm)[0]
                    vals.append(b)
        return np.array(vals)

    # phi-shape systematic (DOMINANT): LA, zCMB, sweep form + (r_void, r_hom)
    band_phi = sweep(E_max_LA, rc_cmb, rh_cmb)
    phi_lo, phi_hi = float(band_phi.min()), float(band_phi.max())
    sig_phi = float((phi_hi - phi_lo) / 2.0)

    # zHD-vs-zCMB systematic (P4): central form/scales, LA
    b_zhd = b_survey(E_max_LA, rc_hd, rh_hd, R_VOID_CENTRAL, R_HOM_CENTRAL, PHI_FORM_CENTRAL)[0]
    sig_z = float(abs(b_central - b_zhd))

    # lapse LA/LB systematic (V0 EXCLUDED -- no-dressing null control): central form/scales, zCMB
    b_lb = b_survey(E_max_LB, rc_cmb, rh_cmb, R_VOID_CENTRAL, R_HOM_CENTRAL, PHI_FORM_CENTRAL)[0]
    sig_lapse = float(abs(b_central - b_lb) / 2.0)

    # full band over everything (report as the honest envelope)
    band_full = np.concatenate([sweep(E_max_LA, rc_cmb, rh_cmb),
                                sweep(E_max_LB, rc_cmb, rh_cmb),
                                sweep(E_max_LA, rc_hd, rh_hd),
                                sweep(E_max_LB, rc_hd, rh_hd)])
    out["systematic_band"] = dict(
        phi_shape_band=dict(lo=phi_lo, hi=phi_hi, half_range=sig_phi,
                            note="DOMINANT systematic: sweep phi form {raised_cosine, linear, "
                                 "smootherstep, cosine_no_plateau} x r_void{20,30,40} x r_hom"
                                 "{80,100,120}, LA E_max, zCMB."),
        zHD_vs_zCMB_shift=dict(b_zCMB=b_central, b_zHD=b_zhd, abs_shift=sig_z,
                               note="P4 systematic: zHD carries LCDM/2M++ peculiar-velocity "
                                    "corrections; zCMB is the first-order primary."),
        lapse_LA_LB=dict(b_LA=b_central, b_LB=b_lb, half_range=sig_lapse,
                         note="LA/LB lapse-reading spread; V0 EXCLUDED (no-dressing null control)."),
        full_envelope=dict(lo=float(band_full.min()), hi=float(band_full.max())),
    )

    # =====================================================================
    # b_req (recompute) + sigma + P3 verdict
    # =====================================================================
    phaseF = json.load(open(PHASEF))
    adv = json.load(open(ADV))
    Hbar0_anch = float(phaseF["variants"]["main_z001"]["free_fixed"]["scale"])
    scale_err = float(phaseF["variants"]["main_z001"]["free_fixed"]["scale_err_sym"])
    Hbar0_glob = float(phaseF["global_reference"]["Hbar0"])
    b_req = Hbar0_anch / Hbar0_glob - 1.0
    b_req_adv = float(adv["local_excess"]["excess_Hbar0"])
    out["b_req"] = dict(
        b_req=b_req, Hbar0_anchored=Hbar0_anch, Hbar0_global=Hbar0_glob,
        formula="Hbar0_anchored / Hbar0_global - 1",
        crosscheck_adv_anchored_h0=b_req_adv, crosscheck_abs_err=abs(b_req - b_req_adv),
        crosscheck_phaseF=float(phaseF["delta_local_excess"]["excess_Hbar0_local_over_global"]),
    )

    # =====================================================================
    # r_hom_star: the homogeneity scale that WOULD make b_pred_survey = b_req, and its
    # physical exclusion. Closes the soft "resolves-at-a-corner" reading: to reach b_req
    # the profile decay scale must be pushed OUTSIDE the declared r_hom band [80,120] and
    # to/beyond the 2M++ homogeneity scale, where the density field is already homogeneous.
    # =====================================================================
    twompp = twompp_homogeneity_scale()
    twompp_rhom = twompp.get("r_homogeneity_est_cum_lt_0p02")
    r_hom_star, r_hom_star_target = solve_rhom_star(E_max_LA, rh_cmb, b_req)
    r_hom_star_by_form = {}
    for _fm in ["raised_cosine", "linear", "smootherstep"]:
        _rs, _ = solve_rhom_star(E_max_LA, rh_cmb, b_req, form=_fm)
        r_hom_star_by_form[_fm] = _rs
    r_hom_star_outside_declared = bool(r_hom_star is not None and r_hom_star > R_HOM_RANGE[1])
    r_hom_star_excluded = bool(
        r_hom_star is not None and twompp_rhom is not None and r_hom_star >= float(twompp_rhom))
    out["r_hom_star"] = dict(
        r_hom_star=r_hom_star,
        target_phi_hf=r_hom_star_target,
        formula="<phi>_HF(r_void=30, r_hom_star) = b_req/E_max(LA); monotone root-find over "
                "the 277 USED_IN_SH0ES_HF SNe (zCMB, raised_cosine primary).",
        by_form=r_hom_star_by_form,
        declared_r_hom_range=list(R_HOM_RANGE),
        outside_declared_range=r_hom_star_outside_declared,
        twompp_homogeneity_scale=twompp_rhom,
        excluded_by_2Mpp=r_hom_star_excluded,
        note=("r_hom_star=%.1f h^-1 Mpc is the profile decay scale that would lift b_pred_survey "
              "to b_req. It lies OUTSIDE Wiltshire's declared r_hom band [%.0f,%.0f] (the robust "
              "exclusion) and at/beyond the 2M++ (Carrick+15) homogeneity scale (~%s h^-1 Mpc, "
              "where the cumulative overdensity |delta(<r)|<0.02, margin thin): the density field "
              "is already homogeneous there, so an apparent-H0 bump persisting to r_hom_star has "
              "no source. The admissible r_hom (~100, band [80,120]) keeps the ENTIRE phi band "
              "below b_req."
              % ((r_hom_star if r_hom_star is not None else float('nan')),
                 R_HOM_RANGE[0], R_HOM_RANGE[1], str(twompp_rhom))),
    )

    sig_anch = scale_err / Hbar0_anch
    # sigma_global: real fractional Hbar0 SCALE error of the joint SN+BAO+CMB fit at the LA
    # best-fit history (Fix: replaces the prior 0.010 ESTIMATE). Subdominant to sigma_phi.
    glob = compute_sigma_global(solA)
    sig_glob = float(glob["sigma_global"])
    sigma_total = float(np.sqrt(sig_anch ** 2 + sig_glob ** 2 + sig_lapse ** 2 + sig_phi ** 2))
    # measurement-only sigma EXCLUDES the theoretical phi-shape band -- exposes whether the
    # pre-registered verdict is carried by agreement or by an inflated theoretical systematic.
    sigma_meas_only = float(np.sqrt(sig_anch ** 2 + sig_glob ** 2 + sig_lapse ** 2))
    out["sigma_components"] = dict(
        anchored_scale=float(sig_anch),
        anchored_scale_note="phaseF free_fixed scale_err_sym / Hbar0_anchored (SH0ES/SN anchoring).",
        global_fit=float(sig_glob),
        global_fit_method=glob["method"],
        global_fit_detail=dict(alpha_best=glob["alpha_best"], sigma_alpha=glob["sigma_alpha"],
                               Hbar0_glob_from_alpha=glob["Hbar0_glob_from_alpha"],
                               phaseF_Hbar0_global=Hbar0_glob,
                               d2chi2_dalpha2=glob["d2chi2_dalpha2"]),
        global_fit_note="COMPUTED (was ESTIMATE 0.010): fractional Hbar0 scale error from the joint "
                        "chi2 curvature vs alpha; subdominant to sigma_phi and sigma_lapse.",
        lapse_LA_LB_halfrange=float(sig_lapse),
        lapse_note="LA/LB lapse-reading spread on b_pred_survey; V0 excluded (no-dressing control).",
        phi_shape_halfrange=float(sig_phi),
        phi_shape_flag="DOMINANT",
        phi_shape_note="Dominant systematic: the radial-profile form + (r_void, r_hom) band.",
    )
    out["sigma_total"] = sigma_total
    out["sigma_measurement_only"] = sigma_meas_only

    diff = float(b_central - b_req)
    nsigma = float(abs(diff) / sigma_total)
    nsigma_meas = float(abs(diff) / sigma_meas_only)
    out["diff"] = dict(b_pred_survey_minus_b_req=diff)
    out["nsigma"] = nsigma
    out["nsigma_measurement_only"] = nsigma_meas

    def verdict_of(ns):
        return "RESOLVES" if ns <= 1.0 else ("PARTIAL" if ns <= 2.0 else "FAILS")

    v_central = verdict_of(nsigma)        # pre-registered, sigma_total (phi-inflated)
    v_meas = verdict_of(nsigma_meas)      # measurement-only sigma (excludes the phi band)

    # ---- audit sec.3: fill the PHYSICAL row's nsigma / verdict ------------------------
    # Reuse the ceiling's sigma budget (measurement-only + total). It is computed at the LARGER
    # ceiling E_max, so it CONSERVATIVELY over-estimates the physical row's systematics (sigma_phi
    # and sigma_lapse both scale ~ with E_max) -> an upper bound on sigma -> a lower bound on the
    # strengthening. Direction-preserving: b_pred_physical < b_pred_ceiling -> |diff| grows ->
    # nsigma grows -> the FAILS only strengthens.
    def _phys_ns(bp):
        d = float(bp) - b_req
        nm = float(abs(d) / sigma_meas_only)
        nt = float(abs(d) / sigma_total)
        return dict(b_pred_physical=float(bp), diff_vs_b_req=d,
                    nsigma_measurement_only=nm, verdict_measurement_only=verdict_of(nm),
                    nsigma_preregistered=nt, verdict_preregistered=verdict_of(nt))
    ph = out["b_pred_survey_PHYSICAL"]
    ns_central = _phys_ns(b_phys_fm_mid)
    ns_fm_r200 = _phys_ns(b_phys_fm_r200)
    ns_fm_r100 = _phys_ns(b_phys_fm_r100)
    ns_deep = _phys_ns(b_phys_deep)
    # FALSIFIER: even the GENEROUS deep-void edge must under-predict b_req; report loudly if not.
    deep_reaches_breq = bool(b_phys_deep >= b_req)
    ph["nsigma_physical"] = dict(
        sigma_measurement_only=sigma_meas_only, sigma_total=sigma_total,
        sigma_note="ceiling sigma budget reused (measurement-only + total); computed at the larger "
                   "ceiling E_max so it CONSERVATIVELY over-estimates the physical row's systematics "
                   "(sigma_phi, sigma_lapse ~ E_max) -> upper-bound sigma -> lower-bound strengthening.",
        central_field_mean=ns_central,
        field_mean_r200_conservative=ns_fm_r200,
        field_mean_r100=ns_fm_r100,
        deep_void_center_m0p8_generous=ns_deep,
        ceiling_comparison=dict(
            nsigma_measurement_only_ceiling=nsigma_meas,
            nsigma_preregistered_ceiling=nsigma,
            note="ceiling: meas-only nsigma=%.3f (%s), pre-reg nsigma=%.3f (%s)."
                 % (nsigma_meas, v_meas, nsigma, v_central)),
    )
    ph["verdict_physical"] = dict(
        verdict_measurement_only=ns_central["verdict_measurement_only"],
        verdict_preregistered=ns_central["verdict_preregistered"],
        whole_band_measurement_verdict=("FAILS" if all(
            n["verdict_measurement_only"] == "FAILS"
            for n in (ns_fm_r200, ns_fm_r100, ns_deep, ns_central)) else "MIXED"),
        deep_void_generous_reaches_b_req=deep_reaches_breq,
        direction_preserving="FAILS only strengthens",
        headline=("LENSING-PHYSICAL row: capping the void rate by sec.2's two-sided contrast gives "
                  "b_pred_physical central %.5f (field-mean, band [%.5f,%.5f] conservative->generous) "
                  "vs ceiling %.5f vs b_req %.5f. Measurement-only nsigma central=%.2f (%s), "
                  "STRENGTHENED from the ceiling's %.2f; the ENTIRE physical band FAILS on "
                  "measurement-only sigma (generous deep-void edge %.2f sigma). The pre-registered "
                  "nsigma stays %s (softened only by the wide phi theoretical systematic). Even the "
                  "GENEROUS deep-void delta_m=-0.8 edge (b_pred %.5f) UNDER-predicts b_req by %.1fx "
                  "-- dilution dominates; the falsifier (generous >= b_req) is NOT tripped."
                  % (b_phys_fm_mid, b_phys_fm_r200, b_phys_deep, b_central, b_req,
                     ns_central["nsigma_measurement_only"], ns_central["verdict_measurement_only"],
                     nsigma_meas, ns_deep["nsigma_measurement_only"],
                     ns_central["verdict_preregistered"], b_phys_deep,
                     (b_req / b_phys_deep) if b_phys_deep else float("inf"))),
        note="Ceiling row (E_max=%.5f, b_pred_survey=%.5f) UNCHANGED; this is the lensing-physical "
             "sensitivity row that pairs with it." % (E_max_LA, b_central),
    )
    if deep_reaches_breq:  # pragma: no cover -- falsifier tripwire (dilution should preclude this)
        ph["verdict_physical"]["FALSIFIER_TRIPPED"] = (
            "!! GENEROUS deep-void physical b_pred %.5f REACHED b_req %.5f -- REPORT LOUDLY."
            % (b_phys_deep, b_req))

    # ONE-SIDED envelope. The ENTIRE admissible phi band [b_env_lo, b_env_hi] lies BELOW
    # b_req, so the mechanism under-predicts for EVERY admissible phi. We therefore do NOT
    # form a per-corner nsigma by dividing a phi-band EDGE by sigma_total -- that double-
    # counts sigma_phi (the phi variation IS the band edge) and spuriously printed RESOLVES
    # (~0.31 sigma) at the high corner. The robust statement is the measurement-only sigma
    # (which excludes the phi band): FAILS at nsigma_meas. Reaching b_req would need r_hom_star,
    # outside the declared r_hom band [80,120] and at/beyond the 2M++ homogeneity scale.
    b_env_lo = float(band_full.min())
    b_env_hi = float(band_full.max())
    envelope_one_sided = bool(b_env_hi < b_req)
    v_robust = v_meas                     # measurement-only sigma -> FAILS at ~4 sigma
    robust = bool(envelope_one_sided and v_meas == "FAILS")
    primary_token = v_central             # no "_FRAGILE": the RESOLVES corner was a sigma_phi
                                          # double-count; the one-sided band admits no resolution.

    out["verdict"] = dict(
        verdict_preregistered=primary_token,
        verdict_robust=v_robust,
        verdict_measurement_sigma_only=v_meas,
        robust=bool(robust),
        envelope_one_sided=envelope_one_sided,
        r_hom_star=r_hom_star,
        r_hom_star_excluded_by_2Mpp=r_hom_star_excluded,
        headline=("UNDER-PREDICTS -> FAILS. b_pred_survey (central %.5f) < b_req (%.5f) at the "
                  "central value AND across the ENTIRE admissible phi band (max %.5f < b_req): the "
                  "mechanism under-predicts the required survey-averaged local bias for EVERY "
                  "admissible phi (one-sided). ROBUST statement = measurement-only sigma (%.5f, "
                  "excluding the theoretical phi band) -> nsigma=%.2f -> %s. The pre-registered "
                  "nsigma=%.2f (%s) is softened ONLY by the wide phi-shape THEORETICAL systematic "
                  "(half-range %.5f) inflating sigma, not by any agreement -- and NO per-corner "
                  "nsigma is formed by dividing a phi-band edge by sigma_total (that double-counts "
                  "sigma_phi and is what spuriously printed RESOLVES). Matching b_req would require "
                  "r_hom_star=%.1f h^-1 Mpc, outside the declared r_hom band [%.0f,%.0f] and "
                  "at/beyond the 2M++ homogeneity scale (~%s) -- physically excluded."
                  % (b_central, b_req, b_env_hi, sigma_meas_only, nsigma_meas, v_meas, nsigma,
                     primary_token, sig_phi,
                     (r_hom_star if r_hom_star is not None else float('nan')),
                     R_HOM_RANGE[0], R_HOM_RANGE[1], str(twompp_rhom))),
        basis=("P3 equality test |b_pred_survey - b_req| = %.5f. ROBUST (measurement-only) sigma = "
               "%.5f -> nsigma = %.3f -> %s. Pre-registered (phi-inflated) sigma = %.5f -> nsigma = "
               "%.3f -> %s. b_pred_survey = %.5f (LA, raised_cosine, r_void=30, r_hom=100, zCMB); "
               "b_req = %.5f."
               % (abs(diff), sigma_meas_only, nsigma_meas, v_meas, sigma_total, nsigma,
                  primary_token, b_central, b_req)),
        thresholds="RESOLVES nsigma<=1 ; PARTIAL <=2 ; FAILS >2 (PLAN P3).",
        envelope_check=dict(
            b_env_lo=b_env_lo, b_env_hi=b_env_hi, b_req=b_req,
            envelope_one_sided=envelope_one_sided,
            r_hom_star=r_hom_star, r_hom_star_target_phi_hf=r_hom_star_target,
            r_hom_star_by_form=r_hom_star_by_form,
            declared_r_hom_range=list(R_HOM_RANGE),
            r_hom_star_outside_declared_range=r_hom_star_outside_declared,
            twompp_homogeneity_scale=twompp_rhom,
            r_hom_star_excluded_by_2Mpp=r_hom_star_excluded,
            note=("ONE-SIDED: the ENTIRE admissible phi band [%.5f, %.5f] lies below b_req=%.5f, so "
                  "the mechanism under-predicts for EVERY admissible phi. NO per-corner nsigma is "
                  "formed from a band-edge / sigma_total ratio -- that double-counts sigma_phi (the "
                  "phi variation already IS the band edge) and spuriously printed RESOLVES (~0.31 "
                  "sigma) at the high corner. Robust statement = measurement-only sigma -> %s at "
                  "nsigma=%.2f. Matching b_req needs r_hom_star=%.1f h^-1 Mpc, OUTSIDE the declared "
                  "r_hom band [%.0f,%.0f] and at/beyond the 2M++ homogeneity scale (~%s)."
                  % (b_env_lo, b_env_hi, b_req, v_meas, nsigma_meas,
                     (r_hom_star if r_hom_star is not None else float('nan')),
                     R_HOM_RANGE[0], R_HOM_RANGE[1], str(twompp_rhom)))),
        under_prediction=dict(
            b_pred_below_b_req_at_central=bool(b_central < b_req),
            b_pred_below_b_req_across_full_band=bool(float(band_full.max()) < b_req),
            note=("b_pred_survey UNDER-predicts b_req at the central value AND across the ENTIRE "
                  "systematic band (max %.5f < b_req %.5f): one-sided under-prediction for every "
                  "admissible phi. This is a DIRECTIONAL FAILS; the pre-registered nsigma is "
                  "softened only by the wide phi-shape systematic, not by agreement, and the "
                  "measurement-only sigma gives FAILS at %.2f sigma."
                  % (float(band_full.max()), b_req, nsigma_meas))),
        dilution_finding=("The void-scale MAXIMUM E_max(LA)=%.5f is diluted to b_pred_survey=%.5f "
                          "(%.1fx) because the ladder's H0 traces the apparent rate at the "
                          "Hubble-flow SNe: <phi>_HF=%.3f (r~69-450, ~%.0f%% past r_hom -> phi->0). "
                          "The calibrators sit deep in the bump (<phi>_calib=%.3f) but only fix M_B "
                          "via geometric distances, so their enhancement does not enter the bias."
                          % (E_max_LA, b_central,
                             (E_max_LA / b_central) if b_central else float("inf"),
                             phi_h, 100.0 * (1.0 - out["survey_geometry"]["hf_frac_inside_r_hom"]),
                             phi_c)),
    )

    # =====================================================================
    # 2M++ homogeneity cross-check (optional; declared systematic) -- computed above for
    # the r_hom_star exclusion; reused here.
    # =====================================================================
    out["twompp_crosscheck"] = twompp

    # =====================================================================
    # machinery status
    # =====================================================================
    out["machinery_status"] = dict(
        gates_all_pass=bool(gates_all_pass),
        window_IS_derivable=True,
        supersedes=("The Wiltshire 17-22% window IS DERIVABLE from the two-scale structure via "
                    "E_max(fv0)=(3/2)/g_dress-1 (gate a) -- this SUPERSEDES the paper-1-issue "
                    "premise that freshH0 has no window-computing machinery, FOR THE DERIVATION. "
                    "The prior finding that freshH0's WINDOW=(0.17,0.22) is a HARD-CODED literal "
                    "remains a provenance issue for that script, but the number itself is now "
                    "reproduced from first principles."),
        prior_bpred_note=("bpred_local_excess.json used the VOLUME-AVERAGE E_dress_void (0.19480 LA) "
                          "and mislabeled it 'the maximum'. Corrected here: E_max(LA)=%.5f is the "
                          "void-scale maximum; its robust-FAILS conclusion is unchanged and "
                          "strengthened by the larger ceiling." % E_max_LA),
        survey_averaged_bpred="COMPUTED (this artifact) -- was OUT_OF_SCOPE in Wave-1b.",
    )

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    # ---- console summary ----
    print("=== GATE (a) tracker window ===")
    print(f"  E_max(0.76)  = {e076:.5f}  (target 0.171)")
    print(f"  E_max(0.695) = {e0695:.5f}  (target 0.220)")
    print(f"  down floor(0.76) = {down076:.5f}  (target 0.219)")
    print(f"  61.7*(1+E_max): up={h0_up_076:.1f}  down={h0_down_076:.1f} km/s/Mpc")
    print(f"  GATE (a) PASS = {a_pass}")
    print("=== GATE (b) small-r ceiling ===")
    print(f"  E_max(LA) = {E_max_LA:.5f}  via identity = {E_max_LA_id:.5f}  (target 0.33650)")
    print(f"  small-r limit (E_max*phi(0)) = {small_r_limit:.5f}")
    print(f"  reconcile: E_dress_void(0.19480) + gamma_dot/Hdress = E_max")
    print(f"  GATE (b) PASS = {b_pass}")
    print("=== GATE (c) large-r limit ===")
    print(f"  phi(r_hom)={phi_rhom:.2e}  E_max*phi(r_hom)={large_r_limit:.2e}")
    print(f"  GATE (c) PASS = {c_pass}")
    print(f"=== ALL GATES PASS = {gates_all_pass} ===")
    print("=== SURVEY-AVERAGED b_pred ===")
    print(f"  n_calib={survey['n_calib']}  n_HF={survey['n_hf']}")
    print(f"  <phi>_calib={phi_c:.4f}  <phi>_HF={phi_h:.4f}")
    print(f"  b_pred_survey (central, LA) = {b_central:.5f}")
    print(f"  phi-shape band = [{phi_lo:.5f}, {phi_hi:.5f}]  (half-range {sig_phi:.5f})")
    print(f"  full envelope  = [{band_full.min():.5f}, {band_full.max():.5f}]")
    print(f"  E_max(LB) reading b_pred_survey = {b_lb:.5f}")
    print("=== b_req / sigma / verdict ===")
    print(f"  b_req = {b_req:.5f}  (adv xcheck err {abs(b_req-b_req_adv):.1e})")
    print(f"  sigma: anch={sig_anch:.5f} glob={sig_glob:.6f}(COMPUTED) lapse={sig_lapse:.5f} "
          f"phi={sig_phi:.5f}(DOM) -> total={sigma_total:.5f}")
    print(f"  diff = {diff:.5f}  nsigma = {nsigma:.3f}  (pre-registered {primary_token})")
    print(f"  measurement-only sigma (no phi band) = {sigma_meas_only:.5f} -> nsigma = "
          f"{nsigma_meas:.3f} ({v_meas})  [ROBUST]")
    print(f"  ONE-SIDED: entire phi band < b_req (band max {band_full.max():.5f} < b_req {b_req:.5f})")
    print(f"  r_hom_star = {r_hom_star:.1f} h^-1 Mpc  (declared band [80,120], 2M++ {twompp_rhom}); "
          f"outside_declared={r_hom_star_outside_declared}  excluded_by_2Mpp={r_hom_star_excluded}")
    print(f"  VERDICT robust = {v_robust}   pre-registered = {primary_token}")
    print(f"  dilution: E_max {E_max_LA:.4f} -> b_pred_survey {b_central:.4f} "
          f"({(E_max_LA/b_central) if b_central else float('inf'):.1f}x)")
    print("=== audit sec.3: PHYSICAL-void row (ceiling row UNCHANGED) ===")
    print(f"  CEILING: E_max={E_max_LA:.5f}  b_pred={b_central:.5f}  (derived/empty-void max)")
    print(f"  PHYSICAL field-mean h_v: r200={hv_fm_r200:.4f} r100={hv_fm_r100:.4f} "
          f"deep(-0.8)={hv_deep:.4f}  (sec.2 {sec2['_source']})")
    print(f"  E_max_physical central(field-mean)={E_phys_fm_mid:.5f}  band[{E_phys_fm_r200:.5f},"
          f"{E_phys_deep:.5f}]")
    print(f"  b_pred_physical central={b_phys_fm_mid:.5f}  band[{b_phys_fm_r200:.5f},"
          f"{b_phys_deep:.5f}]  (vs ceiling {b_central:.5f}, b_req {b_req:.5f})")
    print(f"  nsigma_physical central: meas-only={ns_central['nsigma_measurement_only']:.3f}"
          f"({ns_central['verdict_measurement_only']}, ceiling {nsigma_meas:.3f})  "
          f"pre-reg={ns_central['nsigma_preregistered']:.3f}({ns_central['verdict_preregistered']})")
    print(f"  whole physical band meas-only verdict = "
          f"{ph['verdict_physical']['whole_band_measurement_verdict']}; generous deep-void "
          f"reaches b_req? {deep_reaches_breq}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
