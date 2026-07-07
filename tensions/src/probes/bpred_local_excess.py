#!/usr/bin/env python3
"""b_pred + the P3 Hubble-tension verdict (paper-3 WP-H, PLAN P2/P3).

The local Hubble bias a wall observer SHOULD measure, from the free-history model's
OWN two-scale structure, with ZERO new parameters, compared once to the required bias
b_req and turned into the pre-registered P3 equality-test verdict.

WHAT THIS SCRIPT IS (and honestly is NOT).
  The handoff framing was "generalize paper-1's freshH0 expansion-variance WINDOW
  machinery". THAT MACHINERY DOES NOT EXIST: in paper-1's freshH0.py the 17-22% window
  is a HARD-CODED literal (WINDOW=(0.17,0.22)) -- a Wiltshire(2009) apparent-Hubble-
  variance LITERATURE citation used only as a membership-test target. freshH0 never
  COMPUTES a window from survey geometry or f_v0 (see machinery_status below). So this
  script does the honest computable thing: the EXACT two-scale kinematic decomposition
  (NOTES K1-K3 / modelv_probeR.derived_curves) at z=0, in its THREE lapse readings.

THE TWO-SCALE EXCESS (zero new parameters).
      K1  Hbar = f_w H_w + f_v H_v                     (volume-avg expansion)
      K2  fv'  = 3 f_v (1-f_v)(H_v - H_w)              (fraction evolution)
      dressed:  H_dress = gamma_bar Hbar - dgamma_bar/dt
  PRIMARY excess:  E_dress_void = gamma_bar (H_v - Hbar)/H_dress  at z=0
                 = (H_void_app - H_dress)/H_dress,  H_void_app = gamma_bar H_v - gamma_bar'
  read under three lapses:
      LA  algebraic   gamma_bar = (2+f_v)/2         (primary, adopted)  -> recomputed here
      LB  rate_ratio  gamma_bar = Hbar/H_w          (systematic)        -> READ from the
          1b-A artifact modelV_probeR_LB.json (the paper-3 solver's rate_ratio path
          diverges on the LB free-history nodes -- Hd0 overflows; see lb_recompute_status)
      V0  none        gamma_bar = 1                 (no-lapse control)  -> recomputed here

  E_dress_void is a KINEMATIC two-scale BOUND (the void-scale apparent-H0 MAXIMUM at
  z->0), NOT the survey-averaged local bias the P3 equality test ultimately needs. It
  fails the tracker window validation (gives ~9-12%, not the cited 17-22%). A decisive
  survey-averaged b_pred requires the radial apparent-H0 / spatial expansion-variance
  profile -- declared OUT OF SCOPE (PLAN.md sec 8, P4 flow-correction re-derivation).

Writes probes_out/bpred_local_excess.json.
"""
import os, sys, json, warnings
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.dirname(_SRC)                       # .../free-history-timescape-tensions
_SCIENCE = os.path.dirname(_REPO)                   # .../science
os.chdir(_SRC)
sys.path.insert(0, _HERE)
sys.path.insert(0, _SRC)
import modelv_theory as MV

_POUT = os.path.join(_REPO, "probes_out")
OUT = os.path.join(_POUT, "bpred_local_excess.json")
PHASEF = os.path.join(_POUT, "phaseF_freshH0.json")
PROBER = os.path.join(_POUT, "modelV_probeR.json")
ADV = os.path.join(_POUT, "adv_anchored_h0.json")
# The LB two-scale reading lives in the sibling repo (Wave 1b-A output). Portable,
# __file__-derived relative path (no absolute /home/... hardcode).
LB_ARTIFACT = os.path.join(_SCIENCE, "probes_out",
                           "modelV_probeR_LB.json")

Z_NODES = [0.0, 0.3, 0.7, 1.3, 2.33]
FV_NODES_V = [0.64013, 0.53112, 0.39578, 0.27945, 0.19359]     # Probe R free-history winner (LA)
FV_NODES_V0 = [0.38292, 0.25043, 0.12798, 0.07152, 0.01699]    # V0 no-lapse control
FV_NODES_LB = [0.58786, 0.49709, 0.38473, 0.2503, 0.12145]     # Probe R LB (rate_ratio) winner

# Embedded fallback for the LB two-scale block (verbatim from the 1b-A artifact
# modelV_probeR_LB.json two_scale_z0_LB), so this script stays runnable if the sibling
# repo is absent. The live artifact, when present, is authoritative and is verified
# against this fallback.
FREE_HISTORY_LB_FALLBACK = {
    "fv0": 0.5878612293793102, "tau0": 0.8626204097931035, "fvp_dtau": 0.3092676448241328,
    "Hw_over_Hbar0": 0.7728389672887109, "Hv_over_Hbar0": 1.1983344389865043,
    "Hbar_over_Hbar0": 1.022971258376305, "Hv_minus_Hw_over_Hbar0": 0.4254954716977933,
    "gamma_bar": 1.3236247572964668, "gamma_bar_dot": 0.04140610935253841,
    "Hdress_over_Hbar0": 1.3126239742370596, "Hd0_solver_check": 1.3126239742370596,
    "Hvoid_app_over_Hbar0": 1.5447390216109709,
    "E_dress_void_PRIMARY": 0.17683285688029907, "E_bare_void": 0.17142532517339892,
    "E_contrast_HvHw_over_Hbar": 0.4159407883787021, "E_lapse_boost": 0.3236247572964668,
}

# Wiltshire (2009) apparent-Hubble-variance window paper 1 cites for the tracker.
WILT_WINDOW = (0.17, 0.22)
# The paper-3 target (convention-independent bare ratio); recomputed from artifacts below.
B_REQ_ROADMAP = 0.0842


def two_scale_at_z0(sol, lapse):
    """Exact two-scale decomposition at z=0 (all rates in units of Hbar0).

    Reproduces modelv_probeR.derived_curves at the z=0 node, independently, for the
    STABLE lapses (algebraic, none). The rate_ratio (LB) lapse is NOT recomputed here:
    the paper-3 solver's coupled z<->tau Picard diverges on the LB free-history nodes
    (Hd0 overflows), so the LB reading is READ from the 1b-A artifact instead.
    """
    z = sol.z
    tau = sol.tau
    fv = sol.fv
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

    if lapse == "algebraic":
        gam = (2.0 + fv0) / 2.0
        gamp = fvp / 2.0                                        # dgamma_bar/dtau
    elif lapse == "none":
        gam = 1.0
        gamp = 0.0
    else:
        raise ValueError("two_scale_at_z0 handles algebraic|none only; LB comes from the "
                         "1b-A artifact (paper-3 rate_ratio solver diverges)")

    Hdress = gam * Hbar - gamp                                 # = Hd(0), dressed present rate
    Hvoid_app = gam * Hv - gamp                                # dressed void rate (void-scale max)

    E_dress_void = (Hvoid_app - Hdress) / Hdress               # PRIMARY: apparent-Hubble window max
    E_bare_void = (Hv - Hbar) / Hbar                           # bare void-over-mean
    E_contrast = dHvw / Hbar                                   # (H_v-H_w)/<H>
    E_lapse = gam - 1.0                                        # pure lapse boost

    Hd0_solver = float(np.interp(0.0, z, sol.Hd))
    return dict(
        fv0=fv0, tau0=tau0, fvp_dtau=fvp,
        Hw_over_Hbar0=Hw, Hv_over_Hbar0=Hv, Hbar_over_Hbar0=Hbar,
        Hv_minus_Hw_over_Hbar0=dHvw,
        gamma_bar=gam, gamma_bar_dot=gamp,
        Hdress_over_Hbar0=Hdress, Hd0_solver_check=Hd0_solver,
        Hvoid_app_over_Hbar0=Hvoid_app,
        E_dress_void_PRIMARY=E_dress_void,
        E_bare_void=E_bare_void,
        E_contrast_HvHw_over_Hbar=E_contrast,
        E_lapse_boost=E_lapse,
    )


def solve_free(nodes, lapse):
    fv = MV.fv_from_nodes(nodes, z_nodes=Z_NODES)
    return MV.modelv_solve(fv, lapse=lapse, Ngrid=30000)


def solve_tracker(fv0, lapse="algebraic"):
    return MV.modelv_solve(MV.tracker_fv_of_z(fv0), lapse=lapse, Ngrid=30000)


def load_lb():
    """Return (lb_block, source, recompute_status). Prefer the live 1b-A artifact;
    fall back to the embedded copy. Verify agreement when both are available.
    Also record that the paper-3 solver CANNOT recompute LB (rate_ratio diverges)."""
    recompute = _try_lb_recompute()
    if os.path.exists(LB_ARTIFACT):
        art = json.load(open(LB_ARTIFACT))
        blk = dict(art["two_scale_z0_LB"])
        blk["Hd0_solver_check"] = blk.get("Hdress_over_Hbar0")
        # verify against embedded fallback
        d = abs(blk["E_dress_void_PRIMARY"] - FREE_HISTORY_LB_FALLBACK["E_dress_void_PRIMARY"])
        if d > 1e-9:
            warnings.warn(f"LB artifact E disagrees with embedded fallback by {d:.2e}")
        blk["_source"] = "sibling_artifact:" + os.path.relpath(LB_ARTIFACT, _REPO)
        blk["_fallback_agrees_abs"] = float(d)
        return blk, "sibling_artifact", recompute
    blk = dict(FREE_HISTORY_LB_FALLBACK)
    blk["_source"] = "embedded_fallback (1b-A modelV_probeR_LB.json unavailable at runtime)"
    return blk, "embedded_fallback", recompute


def _try_lb_recompute():
    """Best-effort in-repo recompute of the LB PRIMARY excess via the paper-3 solver,
    purely to DOCUMENT that it diverges. E_dress_void = gamma_bar0 (H_v-Hbar)/Hd0 does
    not need gamma_bar_dot, so a converged solver would suffice -- but it does not."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_free(FV_NODES_LB, "rate_ratio")
        z = sol.z
        Hd0 = float(np.interp(0.0, z, sol.Hd))
        finite = float(np.isfinite(sol.Hd).mean())
        ok = np.isfinite(Hd0) and 0.5 < Hd0 < 5.0 and finite > 0.999
        return dict(converged=bool(ok), Hd0_recomputed=Hd0, Hd_finite_fraction=finite,
                    note=("paper-3 modelv_solve(lapse='rate_ratio') on the LB free-history "
                          "nodes: the coupled z<->tau Picard DIVERGES (Hd0 overflows, non-"
                          "finite entries in Hd). LB is therefore READ from the 1b-A "
                          "artifact, not recomputed here." if not ok else
                          "paper-3 solver reproduces LB (unexpected -- verify)."))
    except Exception as e:  # pragma: no cover
        return dict(converged=False, error=repr(e),
                    note="paper-3 rate_ratio recompute raised; LB read from 1b-A artifact.")


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


def main():
    phaseF = json.load(open(PHASEF))
    adv = json.load(open(ADV))

    out = {"probe": "bpred_local_excess",
           "purpose": "b_pred (predicted local Hubble bias, wall observer) from the free-history "
                      "two-scale structure in its three lapse readings (LA/LB/V0), the recomputed "
                      "b_req, and the pre-registered P3 equality-test verdict with its sigma "
                      "sensitivity.",
           "reading": "KINEMATIC (forced f_v(z); integrability not enforced).",
           "wiltshire_window": list(WILT_WINDOW)}

    # ---- VALIDATION: tracker apparent-Hubble-variance window ----
    trk = {}
    for fv0 in [0.695, 0.76, 0.80, 0.85]:
        d = two_scale_at_z0(solve_tracker(fv0, "algebraic"), "algebraic")
        trk[f"fv0={fv0}"] = d
    out["tracker_validation"] = trk
    prim_trk = {k: v["E_dress_void_PRIMARY"] for k, v in trk.items()}
    in_win = {k: bool(WILT_WINDOW[0] <= p <= WILT_WINDOW[1]) for k, p in prim_trk.items()}
    out["tracker_validation_primary"] = prim_trk
    out["tracker_validation_in_window"] = in_win
    tracker_gate_pass = any(in_win.values())

    # ---- the THREE lapse readings of the two-scale excess at z=0 ----
    solLA = solve_free(FV_NODES_V, "algebraic")
    dLA = two_scale_at_z0(solLA, "algebraic")                                 # LA (recomputed)
    dV0 = two_scale_at_z0(solve_free(FV_NODES_V0, "none"), "none")            # V0 (recomputed)
    dLB, lb_source, lb_recompute = load_lb()                                  # LB (read, 1b-A)
    out["free_history_LA"] = dLA
    out["free_history_LB"] = dLB
    out["free_history_V0_nolapse"] = dV0
    out["lb_recompute_status"] = lb_recompute

    LA = dLA["E_dress_void_PRIMARY"]
    LB = dLB["E_dress_void_PRIMARY"]
    V0 = dV0["E_dress_void_PRIMARY"]

    # confirm the recomputed LA/V0 match the committed values
    out["recompute_checks"] = dict(
        LA_recomputed=LA, LA_committed=0.19480180520559792,
        LA_abs_err=abs(LA - 0.19480180520559792),
        V0_recomputed=V0, V0_committed=0.4350050123598255,
        V0_abs_err=abs(V0 - 0.4350050123598255),
        LB_source=lb_source, LB_value=LB,
    )

    # ---- b_req (recomputed independently as anchored/global Hbar0 ratio - 1) ----
    Hbar0_anch = float(phaseF["variants"]["main_z001"]["free_fixed"]["scale"])
    scale_err = float(phaseF["variants"]["main_z001"]["free_fixed"]["scale_err_sym"])
    Hbar0_glob = float(phaseF["global_reference"]["Hbar0"])
    b_req = Hbar0_anch / Hbar0_glob - 1.0
    b_req_adv = float(adv["local_excess"]["excess_Hbar0"])
    out["b_req"] = dict(
        b_req=b_req,
        Hbar0_anchored=Hbar0_anch, Hbar0_global=Hbar0_glob,
        formula="Hbar0_anchored / Hbar0_global - 1 (same fixed shape -> dressing cancels; "
                "a pure ratio of the SH0ES-anchored vs BAO+CMB-anchored bare scales)",
        crosscheck_adv_anchored_h0=b_req_adv,
        crosscheck_abs_err=abs(b_req - b_req_adv),
        crosscheck_phaseF=float(phaseF["delta_local_excess"]["excess_Hbar0_local_over_global"]),
    )

    # ---- b_pred point prediction: the LA reading (primary), a KINEMATIC BOUND ----
    b_pred = LA
    out["b_pred"] = dict(
        b_pred=b_pred,
        reading="LA (algebraic lapse) two-scale excess at z=0 -- the primary point prediction.",
        caveat_BOUND=("KINEMATIC two-scale BOUND: the void-scale apparent-H0 MAXIMUM at z->0 "
                      "from the smooth volume-average, NOT the survey-averaged local bias the "
                      "P3 equality test needs. The SH0ES ladder samples mostly ABOVE the void "
                      "scale, so the true survey-averaged bias is a dilution of this maximum "
                      "toward b_req; computing that dilution needs the radial apparent-H0 decay "
                      "profile (out of scope)."),
        caveat_tracker_window_FAIL=("This BOUND is NOT validated: fed the tracker at f_v0~0.7-0.8 "
                                    "the same construction yields ~9-12%% (decreasing with f_v0), "
                                    "never the cited Wiltshire 17-22%% -> it fails the window "
                                    "validation gate, so it MIS-represents the survey-averaged "
                                    "local bias."),
        alt_measures_LA=dict(bare_void=dLA["E_bare_void"],
                             contrast_HvHw_over_Hbar=dLA["E_contrast_HvHw_over_Hbar"],
                             lapse_boost=dLA["E_lapse_boost"]),
        three_lapse_readings=dict(LA=LA, LB=LB, V0_nolapse=V0),
    )

    # ---- sigma (PLAN P3): anchored (+) global (+) lapse-reading spread, in quadrature ----
    sig_anch = scale_err / Hbar0_anch
    # sigma_global: real fractional Hbar0 SCALE error of the joint SN+BAO+CMB fit at the LA
    # best-fit history (Fix: replaces the prior 0.010 ESTIMATE). Subdominant to sigma_lapse.
    glob = compute_sigma_global(solLA)
    sig_glob = float(glob["sigma_global"])
    lapse3 = np.array([LA, LB, V0])
    lapse_LALBV0_halfrange = float((lapse3.max() - lapse3.min()) / 2.0)
    lapse_LALBV0_std_pop = float(np.std(lapse3, ddof=0))
    lapse_LALBV0_std_samp = float(np.std(lapse3, ddof=1))
    lapse_LALBonly_halfrange = float(abs(LA - LB) / 2.0)

    # PRIMARY convention for the pre-registered spread: half-range (max-min)/2 -- matches the
    # prior committed methodology (abs(hi-lo)/2) and is the conservative envelope of the readings.
    sig_lapse_prereg = lapse_LALBV0_halfrange
    sig_lapse_LALBonly = lapse_LALBonly_halfrange

    def quad(sl):
        return float(np.sqrt(sig_anch**2 + sig_glob**2 + sl**2))

    out["sigma_components"] = dict(
        anchored_scale=float(sig_anch),
        anchored_scale_note="phaseF free_fixed scale_err_sym / Hbar0_anchored (SH0ES/SN anchoring).",
        global_fit=float(sig_glob),
        global_fit_method=glob["method"],
        global_fit_detail=dict(alpha_best=glob["alpha_best"], sigma_alpha=glob["sigma_alpha"],
                               Hbar0_glob_from_alpha=glob["Hbar0_glob_from_alpha"],
                               phaseF_Hbar0_global=Hbar0_glob,
                               d2chi2_dalpha2=glob["d2chi2_dalpha2"]),
        global_fit_note=("COMPUTED (was ESTIMATE 0.010): the Hbar0 SCALE error of the joint "
                         "SN+BAO+CMB fit. b_req uses the SAME fixed shape in numerator and "
                         "denominator, so the dressing/shape (g_dress, Hd0, S0) CANCELS in the "
                         "ratio; the relevant residual uncertainty is this scale error, read from "
                         "the joint chi2 curvature vs alpha. Subdominant to sigma_lapse."),
        lapse_LA_LB_V0=dict(
            half_range=lapse_LALBV0_halfrange,
            std_population=lapse_LALBV0_std_pop,
            std_sample=lapse_LALBV0_std_samp,
            primary_convention="half_range",
            note="Pre-registered PLAN P3 spread of the three lapse readings LA/LB/V0. Half-range "
                 "adopted as primary (matches prior committed abs(hi-lo)/2); std reported too. "
                 "DOMINATED by V0 (LA-V0 gap 0.240 vs LA-LB gap 0.018)."),
        lapse_LA_LB_only=dict(
            half_range=lapse_LALBonly_halfrange,
            note="SENSITIVITY: the genuine algebraic-vs-rate-ratio lapse ambiguity ONLY (V0, the "
                 "no-lapse control, is arguably not a lapse-reading of the DRESSING mechanism). "
                 "Small because LA=%.5f ~ LB=%.5f." % (LA, LB)),
    )

    sigma_total_prereg = quad(sig_lapse_prereg)
    sigma_total_prereg_std = quad(lapse_LALBV0_std_pop)
    sigma_total_LALBonly = quad(sig_lapse_LALBonly)
    out["sigma_total"] = dict(
        preregistered_LA_LB_V0_halfrange=sigma_total_prereg,
        preregistered_LA_LB_V0_stdpop=sigma_total_prereg_std,
        sensitivity_LA_LB_only_halfrange=sigma_total_LALBonly,
    )

    diff = float(b_pred - b_req)
    ns_prereg = float(abs(diff) / sigma_total_prereg)
    ns_prereg_std = float(abs(diff) / sigma_total_prereg_std)
    ns_lalbonly = float(abs(diff) / sigma_total_LALBonly)
    out["diff"] = dict(b_pred_minus_b_req=diff,
                       note="|b_pred(LA bound) - b_req| ; b_pred is a BOUND, not the survey-averaged value.")

    def verdict_of(ns):
        return "RESOLVES" if ns <= 1.0 else ("PARTIAL" if ns <= 2.0 else "FAILS")

    out["nsigma"] = dict(
        preregistered_LA_LB_V0_halfrange=ns_prereg,
        preregistered_LA_LB_V0_stdpop=ns_prereg_std,
        sensitivity_LA_LB_only_halfrange=ns_lalbonly,
    )

    # ---- P3 verdict (pre-registered primary + sensitivity) ----
    # Integrity fix (adversarial review): the primary token must itself carry the fragility so a
    # bare "RESOLVES" cannot be lifted out of context. When the pre-registered verdict disagrees
    # with the robust (LA/LB-only, V0 excluded) verdict, brand the token _FRAGILE and surface the
    # robust verdict as a first-class field.
    _pre_v = verdict_of(ns_prereg)
    _robust_v = verdict_of(ns_lalbonly)
    _primary_token = _pre_v + ("_FRAGILE" if _pre_v != _robust_v else "") + "_phenomenological"
    out["verdict"] = dict(
        verdict_preregistered=_primary_token,
        verdict_robust_lapse_only=_robust_v,
        basis=("P3 equality test |b_pred - b_req| = %.5f vs 1sigma = %.5f (nsigma = %.3f) under "
               "the pre-registered LA/LB/V0 lapse spread (half-range). Std convention gives "
               "nsigma = %.3f -- also RESOLVES." % (abs(diff), sigma_total_prereg, ns_prereg,
                                                    ns_prereg_std)),
        reading="KINEMATIC; b_pred is the LA two-scale BOUND (void-scale maximum).",
        verdict_sensitivity=(
            "FRAGILE -- the RESOLVES is an ARTIFACT of an inflated sigma, not a robust prediction:\n"
            "  (1) FLIPS TO FAILS at nsigma = %.2f (> 2sigma) under the LA/LB-only lapse "
            "systematic, i.e. as soon as the V0 no-lapse control is excluded. V0 is the "
            "no-dressing control, arguably NOT a genuine lapse-reading of the dressing mechanism; "
            "including it inflates sigma_lapse ~14x (half-range 0.129 vs LA/LB-only 0.009).\n"
            "  (2) sigma_lapse dominates sigma_total (~10x sigma_anch, ~13x sigma_glob) and is "
            "itself dominated by the LA-V0 gap (0.240) not the genuine LA-LB gap (0.018).\n"
            "  (3) b_pred is a void-scale-MAXIMUM two-scale BOUND that FAILS the tracker window "
            "validation (gives ~9-12%% on the tracker, not the cited 17-22%%), so it OVER-states "
            "the survey-averaged local bias. The near-1sigma 'agreement' is not evidence the "
            "mechanism predicts b_req." % ns_lalbonly),
        what_is_established=(
            "controls reproduce (LCDM anchored 73.53+-1.02, tracker 73.06); b_req=%.5f recomputed "
            "exactly (adv/phaseF cross-checks agree to <1e-9) and is convention-independent; the "
            "three lapse readings LA=%.5f, LB=%.5f, V0=%.5f bracket b_req, i.e. the two-scale "
            "expansion variance is AMPLE (mechanism not shown incapable -> NOT a hard FAILS of the "
            "mechanism). But the survey-averaged point prediction that would make RESOLVES robust "
            "is not computable here (machinery_status)." % (b_req, LA, LB, V0)),
    )

    # ---- machinery status: freshH0 has NO window machinery; decisive b_pred out of scope ----
    out["machinery_status"] = dict(
        tracker_validation_gate_pass=bool(tracker_gate_pass),
        freshH0_window_machinery="NONE",
        finding=("Paper-1's freshH0.py has NO window-COMPUTING machinery: WINDOW=(0.17,0.22) is a "
                 "HARD-CODED Wiltshire(2009) apparent-Hubble-variance LITERATURE citation, used "
                 "only as a membership-test target; freshH0 never derives a window from survey "
                 "geometry or f_v0. The one computable expansion-variance object -- the exact "
                 "two-scale kinematic decomposition -- does NOT reproduce it (gives ~9-12%% on the "
                 "tracker, never in [0.17,0.22]). The smooth volume-average carries no void-scale "
                 "apparent-H0 bump; that bump is a SPATIAL-inhomogeneity effect (radial apparent-"
                 "H0 profile / observer environment)."),
        survey_averaged_bpred="OUT_OF_SCOPE",
        out_of_scope_note=("A DECISIVE survey-averaged b_pred -- the point prediction that turns "
                           "the two-scale bound into a survey-weighted local bias -- requires the "
                           "radial apparent-H0 / spatial expansion-variance profile and the SH0ES "
                           "calibrator+Hubble-flow survey geometry. Declared OUT OF SCOPE: PLAN.md "
                           "sec 8; P4 flow-correction re-derivation (first-order treatment only)."),
    )

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    # ---- console summary ----
    print("=== TRACKER VALIDATION (E_dress_void vs Wiltshire 17-22%) ===")
    for k in trk:
        print(f"  {k}: E={prim_trk[k]:.4f}  in_window={in_win[k]}")
    print(f"  tracker gate pass = {tracker_gate_pass}  (freshH0 window machinery = NONE)")
    print("=== THREE LAPSE READINGS of the two-scale excess (z=0) ===")
    print(f"  LA (algebraic, recomputed) = {LA:.5f}   [committed 0.19480, err {abs(LA-0.19480180520559792):.1e}]")
    print(f"  LB (rate_ratio, {lb_source:16s}) = {LB:.5f}")
    print(f"  V0 (no-lapse,  recomputed) = {V0:.5f}   [committed 0.43501, err {abs(V0-0.4350050123598255):.1e}]")
    print(f"  LB paper-3 solver recompute: converged={lb_recompute.get('converged')} "
          f"(Hd0={lb_recompute.get('Hd0_recomputed')})")
    print("=== b_pred / b_req ===")
    print(f"  b_pred (LA BOUND)          = {b_pred:.5f}   [BOUND; fails tracker window]")
    print(f"  b_req  (recomputed)        = {b_req:.5f}   [adv xcheck err {abs(b_req-b_req_adv):.1e}]")
    print(f"  diff                       = {diff:.5f}")
    print("=== sigma ===")
    print(f"  sig_anch={sig_anch:.5f}  sig_glob={sig_glob:.6f}(COMPUTED)")
    print(f"  sig_lapse LA/LB/V0 half-range = {lapse_LALBV0_halfrange:.5f}  (std_pop {lapse_LALBV0_std_pop:.5f})")
    print(f"  sig_lapse LA/LB-only half-rng = {lapse_LALBonly_halfrange:.5f}")
    print(f"  sigma_total prereg (LA/LB/V0) = {sigma_total_prereg:.5f}  -> nsigma={ns_prereg:.3f} -> {verdict_of(ns_prereg)}")
    print(f"  sigma_total sens  (LA/LB only)= {sigma_total_LALBonly:.5f}  -> nsigma={ns_lalbonly:.3f} -> {verdict_of(ns_lalbonly)}")
    print("=== P3 VERDICT ===")
    print(f"  PRE-REGISTERED (primary): {out['verdict']['verdict_preregistered']}  (nsigma {ns_prereg:.2f})")
    print(f"  FRAGILE: flips to FAILS at nsigma {ns_lalbonly:.2f} under LA/LB-only (V0 excluded); "
          f"b_pred is a BOUND failing the tracker window.")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
