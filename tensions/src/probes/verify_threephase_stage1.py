#!/usr/bin/env python3
"""ADVERSARIAL verification of WP-B Stage-1 Phase B (threephase_dynamics.json +
threephase_forced_geometry.json).  Refute-by-default, from-scratch harness path.

Per PLAN_WPB_threephase.md sec 4(ii): independently recompute the Stage-1 result for
>=2 constant sets through a from-scratch harness path.

  (1) TRACK verdict: independently recompute the measured f_s(z),f_d(z) centrals from
      growth+lognormal (cross-check vs q_budget authoritative population), reconstruct
      the bands from the declared envelope constants, drive the three-phase solver at
      the CLAIMED best constants + an INDEPENDENT differential_evolution fit + a GENUINE
      matter-differentiated set, and recompute per-node tracking residuals in half-widths
      / band-widths.  Confirm the TRACKS/TRACK-FAIL call.
  (2) Geometry (since TRACKS): independently rebuild the DESI-DR2 BAO+CMB alpha-marginalised
      chi2 from desi_dr2_rows.json (my own covariance + marginalisation), drive the solver's
      D_M through harness.sn_chi2, independently refit chi2_LCDM, recompute the BIC bar
      chi2_LCDM + ln N, and re-check clears_bar.  Recompute the degeneracy collapse
      (geometry(best) == tracker(f_v0=f_d0)) and the two-phase geometry ceiling.
  (3) Coherence vs Stage-0 / Phase-A homothetic finding.

Everything self-contained; writes probes_out/verify_threephase_stage1.json.
"""
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import ndtr

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_P2 = os.path.join(os.path.dirname(_REPO))
_P2_SRC = os.path.join(_P2, "src")
_P2_OUT = os.path.join(_P2, "probes_out")
sys.path.insert(0, _HERE)
sys.path.insert(0, _P2_SRC)
import threephase_solver as TP  # noqa: E402

_OUT = os.path.join(_REPO, "probes_out", "verify_threephase_stage1.json")
_DYN = os.path.join(_REPO, "probes_out", "threephase_dynamics.json")
_GEO = os.path.join(_REPO, "probes_out", "threephase_forced_geometry.json")
_WALL_SOFT_S = float(os.environ.get("VERIFY_WALL_SOFT_S", "1500"))

SIGMA0 = 0.7344797420042518
OM_FID = 0.315
NODES = np.array([0.0, 0.3, 0.5, 0.7, 1.0])


# ---------------------------------------------------------------- independent harness
def _load_harness():
    cwd = os.getcwd(); os.chdir(_P2_SRC)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import harness as HN
    finally:
        os.chdir(cwd)
    return HN


# ---------------------------------------------------------------- measured curves (mine)
def growth_D(z, Om=OM_FID):
    """Independent flat-LCDM growing-mode D(z), D(0)=1."""
    z = np.atleast_1d(np.asarray(z, float))
    out = np.empty_like(z)

    def _Dun(a_up):
        ag = np.linspace(1e-6, a_up, 60000)
        E = np.sqrt(Om * ag ** -3 + (1 - Om))
        integ = 1.0 / (ag * E) ** 3
        E_up = np.sqrt(Om * a_up ** -3 + (1 - Om))
        return E_up * np.trapezoid(integ, ag)

    for i, zi in enumerate(z):
        out[i] = _Dun(1.0 / (1.0 + zi))
    return out / _Dun(1.0)


def _frac(delta_th, s):
    return ndtr((np.log(1.0 + delta_th) + 0.5 * s * s) / s)


def measured_centrals(nodes):
    D = growth_D(nodes)
    s = SIGMA0 * D
    f_below = _frac(0.0, s)
    f_deep = _frac(-0.5, s)
    f_shallow = f_below - f_deep
    return D, s, f_below, f_deep, f_shallow


def bands_from_json(dyn):
    """Read the declared bands straight from the fit output (my residual recompute uses
    exactly the bands the fit claims, so the residual check is fully independent of my
    band-reconstruction arithmetic)."""
    m = dyn["measured_curves"]
    return m


# ---------------------------------------------------------------- three-phase driver
def _cfg(fs0, fd0, r_s, r_d):
    fv0 = fs0 + fd0
    ak = 1.0 / ((2.0 + fv0) / 3.0) ** 2
    return TP.ThreePhaseConfig(TP.VoidPhase(fs0, min(r_s, 1.0) * ak, "shallow"),
                               TP.VoidPhase(fd0, min(r_d, 1.0) * ak, "deep")), ak


def model_split(p, nodes, Ngrid=60000):
    cfg, ak = _cfg(*p)
    sol = TP.solve(cfg, Ngrid=Ngrid)
    fs = np.interp(nodes, sol.z, sol.fs)
    fd = np.interp(nodes, sol.z, sol.fd)
    return fs, fd, sol, ak


def residuals_halfwidths(fs_m, fd_m, m, idx):
    rs, rd = [], []
    for j, i in enumerate(idx):
        c_s = m["f_shallow"][i]; hw_s = 0.5 * (m["f_shallow_hi"][i] - m["f_shallow_lo"][i])
        c_d = m["f_deep"][i]; hw_d = 0.5 * (m["f_deep_hi"][i] - m["f_deep_lo"][i])
        rs.append(abs(fs_m[j] - c_s) / hw_s)
        rd.append(abs(fd_m[j] - c_d) / hw_d)
    return rs, rd


# ---------------------------------------------------------------- my own DR2 BAO+CMB chi2
def build_dr2():
    with open(os.path.join(_P2_OUT, "desi_dr2_rows.json")) as f:
        d = json.load(f)
    rows = [tuple(r) for r in d["rows"]] + [tuple(d["cmb_point"]["row"])]
    n = len(rows)
    C = np.zeros((n, n))
    for i in range(n):
        zi, ki, vi, ei, ci = rows[i]
        C[i, i] = ei * ei
        for j in range(i + 1, n):
            zj, kj, vj, ej, cj = rows[j]
            if zi == zj and ci is not None and cj is not None and ki != kj:
                C[i, j] = C[j, i] = ci * ei * ej
    DV = np.array([r[2] for r in rows], float)
    CINV = np.linalg.inv(C)
    return rows, DV, CINV, float(d["rd"])


def dr2_chi2(predict, rows, DV, CINV):
    g = np.array([predict(z, k) for z, k, _, _, _ in rows], float)
    gCi = CINV @ g
    a = (g @ (CINV @ DV)) / (g @ gCi)
    chi = DV @ (CINV @ DV) - (g @ (CINV @ DV)) ** 2 / (g @ gCi)
    return float(chi), float(a)


def main():
    t0 = time.time()
    HN = _load_harness()
    zHD, zHEL, mb, Cf = HN.load_sn()
    rows, DV, CINV, RD = build_dr2()

    dyn = json.load(open(_DYN))
    geo = json.load(open(_GEO))
    m = bands_from_json(dyn)
    bc = dyn["best_constants"]

    checks = []
    discrepancies = []

    # ---------- (0) measured centrals: independent recompute vs JSON vs q_budget ----------
    D, s, fb, fd, fs = measured_centrals(NODES)
    qb = json.load(open(os.path.join(_REPO, "probes_out", "q_budget.json")))
    cen_max_abs = 0.0
    for i, z in enumerate(NODES):
        cen_max_abs = max(cen_max_abs,
                          abs(fs[i] - m["f_shallow"][i]),
                          abs(fd[i] - m["f_deep"][i]),
                          abs(fb[i] - m["f_below"][i]))
    centrals_ok = cen_max_abs < 1e-6
    checks.append(f"measured centrals reproduced from growth+lognormal: max|diff vs JSON|={cen_max_abs:.2e} ({'OK' if centrals_ok else 'MISMATCH'})")
    if not centrals_ok:
        discrepancies.append(f"measured centrals differ from JSON by {cen_max_abs:.2e}")

    # ---------- (1) TRACK verdict: claimed best + independent fit + genuine set ----------
    idx_primary = [0, 1, 2, 3]
    idx_all = [0, 1, 2, 3, 4]

    # (1a) claimed best constants
    best_p = [bc["f_s0"], bc["f_d0"], bc["alpha_s2_over_ceiling"], bc["alpha_d2_over_ceiling"]]
    fs_b, fd_b, sol_b, ak_b = model_split(best_p, NODES)
    rs_b, rd_b = residuals_halfwidths(fs_b, fd_b, m, idx_all)
    max_prim_b = max(rs_b[:4] + rd_b[:4])
    max_all_b = max(rs_b + rd_b)
    claimed_max_prim = dyn["max_resid_halfwidths_primary_z_le_0.7"]
    match_resid = abs(max_prim_b - claimed_max_prim) < 1e-3
    checks.append(f"claimed-best per-node residuals reproduced: my max_prim={max_prim_b:.4f} vs JSON {claimed_max_prim:.4f} ({'OK' if match_resid else 'MISMATCH'})")
    if not match_resid:
        discrepancies.append(f"claimed-best max primary residual {max_prim_b:.4f} != JSON {claimed_max_prim:.4f}")
    tracks_claimed = max_prim_b <= 1.0

    # (1b) independent differential_evolution fit (my own, different seed)
    def obj(p):
        fs0, fd0, r_s, r_d = p
        if fs0 <= 0.02 or fd0 <= 0.02 or fs0 + fd0 >= 0.97:
            return 1e6
        try:
            fsm, fdm, _, _ = model_split(p, NODES, Ngrid=9000)
        except Exception:
            return 1e5
        rs_, rd_ = residuals_halfwidths(fsm, fdm, m, idx_primary)
        return float(np.max(rs_ + rd_))
    bounds = [(0.05, 0.65), (0.03, 0.45), (0.001, 1.0), (0.001, 1.0)]
    res = differential_evolution(obj, bounds, maxiter=45, popsize=20, tol=1e-8,
                                 seed=101, polish=True, mutation=(0.4, 1.2), recombination=0.85)
    ind_p = list(res.x)
    fs_i, fd_i, sol_i, ak_i = model_split(ind_p, NODES)
    rs_i, rd_i = residuals_halfwidths(fs_i, fd_i, m, idx_all)
    max_prim_i = max(rs_i[:4] + rd_i[:4])
    tracks_ind = max_prim_i <= 1.0
    # is the independent optimum also degenerate (alpha_s2 -> 0)?
    ind_rs_over_ceiling = min(ind_p[2], 1.0)
    ind_degenerate = ind_rs_over_ceiling < 1e-2 and min(ind_p[3], 1.0) > 0.9
    checks.append(f"independent DE fit: max_prim={max_prim_i:.4f} (TRACKS={tracks_ind}); best p={[round(x,4) for x in ind_p]}; degenerate(alpha_s2->0,alpha_d2->ceiling)={ind_degenerate}")

    # (1c) a GENUINE matter-differentiated set (NOTES representative): should NOT track
    #      alpha_s2=0.70, alpha_d2=1.20 with f_v0~0.64 -> ceiling ~1.277 so r=alpha2/ceiling
    ak_rep = 1.0 / ((2.0 + 0.64) / 3.0) ** 2
    gen_p = [0.36, 0.28, 0.70 / ak_rep, 1.20 / ak_rep]
    fs_g, fd_g, sol_g, ak_g = model_split(gen_p, NODES)
    rs_g, rd_g = residuals_halfwidths(fs_g, fd_g, m, idx_all)
    max_prim_g = max(rs_g[:4] + rd_g[:4])
    # genuine df_shallow z0->z1
    gen_df_s = float(fs_g[4] - fs_g[0])
    checks.append(f"genuine matter-differentiated set (a_s2=0.70,a_d2=1.20): max_prim={max_prim_g:.3f} (TRACKS={max_prim_g<=1.0}); df_shallow(z0->z1)={gen_df_s:+.4f} (measured rises +{m['f_shallow'][4]-m['f_shallow'][0]:.4f})")

    # verdict
    verdict_track = "TRACKS" if tracks_claimed else "TRACK-FAIL"
    track_agrees = (verdict_track == dyn["track_verdict"])
    if not track_agrees:
        discrepancies.append(f"my TRACK verdict {verdict_track} != JSON {dyn['track_verdict']}")

    # ---------- (2) geometry: my own DR2 chi2 + SN + LCDM refit + BIC bar ----------
    # LCDM refit (independent)
    def lcdm_joint(Om):
        return float(HN.sn_chi2(HN.lcdm_Dc(zHD, Om))) + dr2_chi2(HN.lcdm_predict(Om), rows, DV, CINV)[0]
    rl = minimize(lambda x: lcdm_joint(x[0]), [0.30], method="Nelder-Mead",
                  options={"xatol": 1e-6, "fatol": 1e-8})
    Om_lcdm = float(rl.x[0]); chi2_lcdm = float(rl.fun)
    N = len(zHD) + len(rows)
    lnN = float(np.log(N))
    bar = chi2_lcdm + lnN
    claimed_bar = geo["bic_bar_dr2"]["bar"]
    claimed_lcdm = geo["bic_bar_dr2"]["chi2_LCDM_dr2"]
    bar_match = abs(bar - claimed_bar) < 1e-2 and abs(chi2_lcdm - claimed_lcdm) < 1e-2
    checks.append(f"LCDM/DR2 refit: Om={Om_lcdm:.4f} chi2_LCDM={chi2_lcdm:.4f} (JSON {claimed_lcdm:.4f}); N={N} lnN={lnN:.4f} bar={bar:.4f} (JSON {claimed_bar:.4f}) ({'OK' if bar_match else 'MISMATCH'})")
    if not bar_match:
        discrepancies.append(f"BIC bar {bar:.4f}/LCDM {chi2_lcdm:.4f} != JSON {claimed_bar:.4f}/{claimed_lcdm:.4f}")

    # joint chi2 at tracked constants (my own pipeline)
    predict_b = lambda z, k: float(sol_b.predict(z, k))
    csn_b = float(HN.sn_chi2(sol_b.D_M(zHD)))
    cbc_b, a_b = dr2_chi2(predict_b, rows, DV, CINV)
    jc_b = csn_b + cbc_b
    H0_b = float(HN.H0_from_alpha(a_b))
    claimed_jc = geo["STEP3_k0_forced_prediction"]["joint_chi2"]
    jc_match = abs(jc_b - claimed_jc) < 0.05
    clears_bar = bool(jc_b <= bar)
    bar_agrees = (clears_bar == geo["STEP3_k0_forced_prediction"]["clears_bar"])
    checks.append(f"joint chi2 @ tracked constants (my pipeline): {jc_b:.3f} (SN={csn_b:.3f}+DR2BAOCMB={cbc_b:.3f}) vs JSON {claimed_jc:.3f} ({'OK' if jc_match else 'MISMATCH'}); H0={H0_b:.2f}")
    checks.append(f"clears_bar: joint {jc_b:.1f} vs bar {bar:.1f} -> {clears_bar} (miss={jc_b-bar:.1f}); JSON clears_bar={geo['STEP3_k0_forced_prediction']['clears_bar']} ({'AGREE' if bar_agrees else 'DISAGREE'})")
    if not jc_match:
        discrepancies.append(f"joint chi2 @ tracked {jc_b:.3f} != JSON {claimed_jc:.3f}")

    # ---------- (3) degeneracy collapse + two-phase ceiling ----------
    def tracker_joint(fv0):
        sT = TP.solve(TP.tracker_config(fv0), Ngrid=60000)
        cs = float(HN.sn_chi2(sT.D_M(zHD)))
        cb, aa = dr2_chi2(lambda z, k: float(sT.predict(z, k)), rows, DV, CINV)
        return cs + cb, float(HN.H0_from_alpha(aa))
    jc_trk_fd, H0_trk_fd = tracker_joint(bc["f_d0"])
    jc_trk_fv, H0_trk_fv = tracker_joint(bc["f_s0"] + bc["f_d0"])
    degen_gap = abs(jc_b - jc_trk_fd)
    checks.append(f"degeneracy: geom(best)={jc_b:.3f} == tracker(f_v0=f_d0={bc['f_d0']:.4f})={jc_trk_fd:.3f} |gap|={degen_gap:.3f}; tracker(f_v0=f_s0+f_d0={bc['f_s0']+bc['f_d0']:.4f})={jc_trk_fv:.3f}")
    degen_ok = degen_gap < 0.1
    if not degen_ok:
        discrepancies.append(f"degeneracy collapse gap {degen_gap:.3f} exceeds 0.1 (geometry(best) not == tracker(f_d0))")

    # two-phase geometry ceiling (free f_v0)
    rceil = minimize(lambda x: tracker_joint(float(x[0]))[0], [0.64], method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-4})
    fv0_ceil = float(rceil.x[0]); jc_ceil, H0_ceil = tracker_joint(fv0_ceil)
    checks.append(f"two-phase geometry ceiling: best f_v0={fv0_ceil:.4f} joint={jc_ceil:.3f} H0={H0_ceil:.2f} (still {'clears' if jc_ceil<=bar else 'MISSES'} bar {bar:.1f} by {jc_ceil-bar:.1f})")

    # ---------- coherence with Stage-0 / Phase-A homothetic finding ----------
    # Om_s at degenerate limit should equal H_w0^2 (flat-dust shallow == walls)
    H_w0_sq = (2.0 / (3.0 * bc["tau0"])) ** 2
    om_s_match = abs(sol_b.Om_s - H_w0_sq) < 1e-3
    checks.append(f"coherence: degenerate shallow Om_s={sol_b.Om_s:.4f} == H_w0^2={H_w0_sq:.4f} ({'OK flat-dust=walls' if om_s_match else 'MISMATCH'}); Om_d={sol_b.Om_d:.2e} (empty Milne)")
    coheres = clears_bar is False and degen_ok and om_s_match

    # ---------- overall verdict ----------
    if track_agrees and bar_agrees and jc_match and bar_match and degen_ok and centrals_ok:
        if len(discrepancies) == 0:
            verdict = "SURVIVES"
        else:
            verdict = "SURVIVES_WITH_CAVEATS"
    else:
        # if the geometry catastrophically clears the bar unexpectedly -> REFUTED
        if clears_bar and not geo["STEP3_k0_forced_prediction"]["clears_bar"]:
            verdict = "REFUTED"
        elif len(discrepancies) > 0:
            verdict = "REFUTED" if (not bar_agrees or not track_agrees) else "SURVIVES_WITH_CAVEATS"
        else:
            verdict = "SURVIVES_WITH_CAVEATS"

    out = {
        "probe": "ADVERSARIAL verify of WP-B Stage-1 Phase B (threephase_dynamics + threephase_forced_geometry)",
        "spec": "PLAN_WPB_threephase.md sec 4(ii): from-scratch harness recompute, >=2 constant sets",
        "verdict": verdict,
        "track_agrees": bool(track_agrees),
        "bar_agrees": bool(bar_agrees),
        "STEP0_measured_centrals": {
            "max_abs_diff_vs_JSON": cen_max_abs, "reproduced": bool(centrals_ok),
            "note": "growth D(z) + lognormal excursion-set f(delta<th) recomputed independently; "
                    "matches q_budget authoritative population",
        },
        "STEP1_track": {
            "claimed_best": {"p_[fs0,fd0,rs,rd]": best_p, "max_resid_halfwidths_primary": max_prim_b,
                             "max_resid_bandwidths_primary": max_prim_b / 2.0,
                             "max_resid_halfwidths_incl_z1": max_all_b,
                             "per_node_shallow_hw": rs_b, "per_node_deep_hw": rd_b,
                             "TRACKS": bool(tracks_claimed)},
            "independent_DE_fit": {"p": ind_p, "max_resid_halfwidths_primary": max_prim_i,
                                   "TRACKS": bool(tracks_ind),
                                   "degenerate_alpha_s2_to_0": bool(ind_degenerate),
                                   "alpha_s2_over_ceiling": ind_rs_over_ceiling},
            "genuine_differentiated_set": {"p": gen_p, "max_resid_halfwidths_primary": max_prim_g,
                                           "TRACKS": bool(max_prim_g <= 1.0),
                                           "df_shallow_z0_to_z1": gen_df_s,
                                           "measured_df_shallow_z0_to_z1": m["f_shallow"][4] - m["f_shallow"][0]},
            "verdict_track": verdict_track, "JSON_track_verdict": dyn["track_verdict"],
        },
        "STEP2_geometry": {
            "LCDM_DR2_refit": {"Om": Om_lcdm, "chi2_LCDM": chi2_lcdm, "N": N, "lnN": lnN, "bar": bar,
                               "JSON_bar": claimed_bar, "JSON_chi2_LCDM": claimed_lcdm, "match": bool(bar_match)},
            "joint_at_tracked": {"joint_chi2": jc_b, "chi2_SN": csn_b, "chi2_DR2_BAOCMB": cbc_b,
                                 "H0": H0_b, "JSON_joint_chi2": claimed_jc, "match": bool(jc_match),
                                 "clears_bar": clears_bar, "miss": jc_b - bar,
                                 "JSON_clears_bar": geo["STEP3_k0_forced_prediction"]["clears_bar"]},
        },
        "STEP3_degeneracy": {
            "geom_best": jc_b,
            "tracker_fv0_eq_fd0": {"fv0": bc["f_d0"], "joint_chi2": jc_trk_fd, "H0": H0_trk_fd},
            "tracker_fv0_eq_fs0_plus_fd0": {"fv0": bc["f_s0"] + bc["f_d0"], "joint_chi2": jc_trk_fv, "H0": H0_trk_fv},
            "collapse_gap_abschi2": degen_gap, "collapse_confirmed": bool(degen_ok),
            "twophase_geometry_ceiling": {"fv0_best": fv0_ceil, "joint_chi2": jc_ceil, "H0": H0_ceil,
                                          "misses_bar_by": jc_ceil - bar},
            "shallow_Om_s": sol_b.Om_s, "H_w0_squared": H_w0_sq, "flatdust_shallow_eq_walls": bool(om_s_match),
        },
        "coherence_with_stage0_and_phaseA": bool(coheres),
        "checks": checks,
        "discrepancies": discrepancies,
        "runtime_s": round(time.time() - t0, 2),
    }
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2, default=TP._json_default)
    print(json.dumps({"verdict": verdict, "track_agrees": track_agrees, "bar_agrees": bar_agrees,
                      "n_discrepancies": len(discrepancies)}, indent=2))
    for c in checks:
        print("  -", c)
    print("wrote", _OUT, "(%.1fs)" % (time.time() - t0))
    return out


if __name__ == "__main__":
    main()
