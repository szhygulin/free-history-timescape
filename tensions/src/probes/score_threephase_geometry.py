#!/usr/bin/env python3
"""WP-B Stage-1 Phase-B STEP 3+4: geometry scoring of the three-phase dressed geometry.

STEP 3 (k=0 forced prediction): ONLY if STEP 1-2 returned TRACKS -- at the structure-
tracked constants the geometry is a ZERO-cosmology-parameter prediction; score joint
SN + DESI-DR2 BAO + Planck CMB against the BIC bar (chi2 <= chi2_LCDM + lnN ~ 1407.2).
If TRACK-FAIL, the k=0 prediction is 'not scored: TRACK-FAIL' -- but we still report the
best-tracking-constants geometry chi2 as an informational number.

STEP 4 (T1 diagnostic ceiling): the SAME machinery with the four constants FREED to fit
the cosmology data -> the lowest joint chi2 the three-phase dressed geometry can reach.
A CEILING only, never a claim (it uses the cosmology data the structure fit forbids).

Reuses the paper-2 shared harness (sn_chi2, alpha-marginalised BAO+CMB); the DR2 BAO+CMB
chi2 mirror is the same block-diagonal alpha-marginalised construction as
forced_joint_fit.py / phaseF_joint_ampsplit.py (DESI DR2 rows + Planck acoustic point).
"""
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_P2_SRC = os.path.join(os.path.dirname(_REPO), "src")
_P2_OUT = os.path.join(os.path.dirname(_REPO), "probes_out")
sys.path.insert(0, _HERE)
sys.path.insert(0, _P2_SRC)
import threephase_solver as TP  # noqa: E402

_DYN = os.path.join(_REPO, "probes_out", "threephase_dynamics.json")
_OUT = os.path.join(_REPO, "probes_out", "threephase_forced_geometry.json")
_WALL_SOFT_S = float(os.environ.get("THREEPHASE_GEOM_WALL_SOFT_S", "1800"))


def _load_harness():
    cwd = os.getcwd(); os.chdir(_P2_SRC)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import harness as HN
    finally:
        os.chdir(cwd)
    return HN


# ----------------------- DR2 BAO+CMB chi2 (mirror of harness.bao_cmb_chi2) -------
def _build_dr2():
    with open(os.path.join(_P2_OUT, "desi_dr2_rows.json")) as f:
        d = json.load(f)
    rows = [tuple(r) for r in d["rows"]] + [tuple(d["cmb_point"]["row"])]
    n = len(rows)
    C = np.zeros((n, n))
    for i, (zi, ki, vi, ei, ci) in enumerate(rows):
        C[i, i] = ei * ei
        for j in range(i + 1, n):
            zj, kj, vj, ej, cj = rows[j]
            if zi == zj and ci is not None and cj is not None and ki != kj:
                C[i, j] = C[j, i] = ci * ei * ej
    DV = np.array([r[2] for r in rows])
    CINV = np.linalg.inv(C)
    return rows, DV, CINV, float(d["rd"])


_DR2_ROWS, _DR2_DV, _DR2_CINV, _RD = _build_dr2()


def bao_cmb_chi2_dr2(predict):
    g = np.array([predict(z, k) for z, k, _, _, _ in _DR2_ROWS])
    gCi = _DR2_CINV @ g
    a = (g @ (_DR2_CINV @ _DR2_DV)) / (g @ gCi)
    chi = _DR2_DV @ (_DR2_CINV @ _DR2_DV) - (g @ (_DR2_CINV @ _DR2_DV)) ** 2 / (g @ gCi)
    return float(chi), float(a)


# ------------------------------------------------------------- three-phase driver
def _cfg_from_params(p):
    fs0, fd0, r_s, r_d = p
    fv0 = fs0 + fd0
    ak = 1.0 / ((2.0 + fv0) / 3.0) ** 2
    return TP.ThreePhaseConfig(TP.VoidPhase(fs0, min(r_s, 1.0) * ak, "shallow"),
                               TP.VoidPhase(fd0, min(r_d, 1.0) * ak, "deep")), ak


def _joint_chi2(p, HN, zHD, Ngrid=30000):
    fs0, fd0, r_s, r_d = p
    if fs0 <= 0.02 or fd0 <= 0.02 or fs0 + fd0 >= 0.97:
        return 1e9, None, None, None, None
    try:
        cfg, ak = _cfg_from_params(p)
        sol = TP.solve(cfg, Ngrid=Ngrid)
        predict = lambda z, k: float(sol.predict(z, k))
        csn = float(HN.sn_chi2(sol.D_M(zHD)))
        cbc, a = bao_cmb_chi2_dr2(predict)
        return csn + cbc, csn, cbc, a, sol
    except Exception as e:
        return 1e8, None, None, None, None


def main():
    t0 = time.time()
    HN = _load_harness()
    zHD, zHEL, mb, Cf = HN.load_sn()

    dyn = json.load(open(_DYN))
    verdict = dyn["track_verdict"]
    tracks = dyn.get("tracks", False)
    bc = dyn["best_constants"]
    best_p = [bc["f_s0"], bc["f_d0"], bc["alpha_s2_over_ceiling"], bc["alpha_d2_over_ceiling"]]

    # ---- BIC bar (DR2): LCDM refit on the SAME data (self-contained recompute) ----
    def lcdm_joint(Om):
        return float(HN.sn_chi2(HN.lcdm_Dc(zHD, Om))) + bao_cmb_chi2_dr2(HN.lcdm_predict(Om))[0]
    r = minimize(lambda x: lcdm_joint(x[0]), [0.30], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-8})
    Om_lcdm = float(r.x[0]); chi2_lcdm = float(r.fun)
    N = len(zHD) + len(_DR2_ROWS)      # 1580 SN + 13 BAO + 1 CMB = 1594
    lnN = float(np.log(N))
    bic_bar = chi2_lcdm + lnN
    print(f"[LCDM/DR2] Om={Om_lcdm:.4f} chi2={chi2_lcdm:.3f}  N={N} lnN={lnN:.3f}  bar={bic_bar:.3f}")

    # ---- geometry AT the structure-best constants (informational; k=0 if TRACKS) ----
    jc, csn, cbc, a, sol = _joint_chi2(best_p, HN, zHD, Ngrid=60000)
    H0 = HN.H0_from_alpha(a) if a else None
    print(f"[structure-best geometry] joint chi2={jc:.3f} (SN={csn:.3f} + BAOCMB_DR2={cbc:.3f})  "
          f"vs bar={bic_bar:.3f}  H0={H0:.2f}" if a else f"[structure-best] invalid")

    # ---- DEGENERACY CHECK: the tracking config has alpha_s2->0 (shallow = flat dust =
    # walls) + deep = empty Milne, i.e. the two-phase EMPTY-void tracker whose single
    # genuine void fraction is f_d0.  Verify geometry(best) == tracker(f_v0=f_d0), and
    # contrast tracker(f_v0=f_s0+f_d0). ----
    def _tracker_joint(fv0):
        s = TP.solve(TP.tracker_config(fv0), Ngrid=60000)
        csn_ = float(HN.sn_chi2(s.D_M(zHD)))
        cbc_, a_ = bao_cmb_chi2_dr2(lambda z, k: float(s.predict(z, k)))
        return csn_ + cbc_, csn_, cbc_, float(HN.H0_from_alpha(a_))
    trk_fd = _tracker_joint(bc["f_d0"])
    trk_fv = _tracker_joint(bc["f_s0"] + bc["f_d0"])
    # paper-2 two-phase tracker best (free f_v0 to the DR2 geometry) -- the geometry ceiling
    def _trk_obj(fv0):
        return _tracker_joint(float(fv0))[0]
    rtrk = minimize(lambda x: _trk_obj(x[0]), [0.64], method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-4})
    fv0_best = float(rtrk.x[0]); trk_best = _tracker_joint(fv0_best)
    degeneracy = {
        "finding": "The structure-tracking constants collapse the model to the two-phase "
                   "empty-void tracker: alpha_s2->0 (shallow spatially FLAT, Om_s=H_w0^2, "
                   "dynamically walls) and alpha_d2=ceiling (deep EMPTY Milne). The single "
                   "genuine void fraction is f_d0, NOT f_s0+f_d0. This is the homothetic "
                   "ceiling (NOTES_threephase sec 5) realized by the fit.",
        "geometry_best_config_joint_chi2": jc,
        "tracker_fv0_eq_fd0": {"fv0": bc["f_d0"], "joint_chi2": trk_fd[0], "H0": trk_fd[3]},
        "tracker_fv0_eq_fs0_plus_fd0": {"fv0": bc["f_s0"] + bc["f_d0"], "joint_chi2": trk_fv[0], "H0": trk_fv[3]},
        "match_best_vs_tracker_fd0_abschi2": abs(jc - trk_fd[0]),
        "note_match": "geometry(best 3-phase) == tracker(f_v0=f_d0) confirms the flat-dust "
                      "shallow is geometrically walls; the effective void fraction is ~f_d0.",
        "twophase_tracker_geometry_ceiling": {
            "fv0_best_fit_to_DR2": fv0_best, "joint_chi2": trk_best[0], "H0": trk_best[3],
            "note": "best the two-phase empty-void tracker can do vs SN+DR2+CMB (free f_v0); "
                    "the structure fit forbids this f_v0 (~0.6-0.7) because the measured deep "
                    "fraction pins f_d0~0.27 and the shallow supplies NO backreaction.",
        },
    }
    print(f"[degeneracy] geom(best)={jc:.3f}  tracker(fv0=fd0={bc['f_d0']:.3f})={trk_fd[0]:.3f}  "
          f"|diff|={abs(jc-trk_fd[0]):.3f}   tracker(fv0=fs0+fd0)={trk_fv[0]:.3f}  "
          f"tracker_best(fv0={fv0_best:.3f})={trk_best[0]:.3f}")

    if tracks:
        k0 = {
            "scored": True, "k": 0,
            "joint_chi2": jc, "chi2_SN": csn, "chi2_BAOCMB_DR2": cbc,
            "bic_bar_dr2": bic_bar, "chi2_LCDM_dr2": chi2_lcdm, "lnN": lnN, "N": N,
            "clears_bar": bool(jc <= bic_bar),
            "margin_or_miss": float(jc - bic_bar),
            "H0_dressed_from_alpha": H0,
        }
    else:
        k0 = {
            "scored": False, "status": "not scored: TRACK-FAIL",
            "reason": "STEP 1-2 returned TRACK-FAIL; the geometry is NOT a valid zero-parameter "
                      "prediction because no constant set tracks the measured f_s(z),f_d(z) within "
                      "their bands. The structure-best-constants geometry chi2 is reported below "
                      "for information only (it does NOT track the structure data).",
            "informational_structure_best_geometry": {
                "joint_chi2": jc, "chi2_SN": csn, "chi2_BAOCMB_DR2": cbc,
                "bic_bar_dr2": bic_bar, "would_clear_bar": bool(jc <= bic_bar),
                "margin_or_miss": float(jc - bic_bar), "H0_dressed_from_alpha": H0,
            },
        }

    # ---- STEP 4: T1 diagnostic ceiling (constants FREED to fit cosmology data) ----
    print("[T1 ceiling] freeing the four constants to fit SN+BAO(DR2)+CMB ...")
    bounds = [(0.05, 0.70), (0.02, 0.45), (0.05, 1.0), (0.05, 1.0)]

    def objT1(p):
        return _joint_chi2(p, HN, zHD, Ngrid=12000)[0]

    resT1 = differential_evolution(objT1, bounds, maxiter=40, popsize=16, tol=1e-7,
                                   seed=7, polish=True, mutation=(0.4, 1.2), recombination=0.8)
    # polish at high Ngrid
    resT1nm = minimize(lambda p: _joint_chi2(p, HN, zHD, Ngrid=30000)[0], resT1.x,
                       method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})
    bestT1 = resT1nm.x if resT1nm.fun < resT1.fun else resT1.x
    jcT1, csnT1, cbcT1, aT1, solT1 = _joint_chi2(bestT1, HN, zHD, Ngrid=60000)
    fs0T1, fd0T1, rsT1, rdT1 = bestT1
    fv0T1 = fs0T1 + fd0T1
    akT1 = 1.0 / ((2 + fv0T1) / 3.0) ** 2
    H0T1 = HN.H0_from_alpha(aT1)
    print(f"[T1 ceiling] best joint chi2={jcT1:.3f} (SN={csnT1:.3f}+BAOCMB={cbcT1:.3f})  "
          f"vs bar={bic_bar:.3f}  clears={jcT1<=bic_bar}  H0={H0T1:.2f}")

    t1 = {
        "note": "CEILING ONLY -- constants freed to fit the cosmology data (the structure fit "
                "forbids this); reported to bound what the three-phase dressed geometry could "
                "reach, never as a claim.",
        "best_constants": {"f_s0": float(fs0T1), "f_d0": float(fd0T1),
                           "alpha_s2": float(min(rsT1, 1.0) * akT1),
                           "alpha_d2": float(min(rdT1, 1.0) * akT1),
                           "alpha_s2_over_ceiling": float(min(rsT1, 1.0)),
                           "alpha_d2_over_ceiling": float(min(rdT1, 1.0)),
                           "f_w0": float(1 - fv0T1),
                           "Om_shallow": float(solT1.Om_s), "Om_deep": float(solT1.Om_d)},
        "joint_chi2": jcT1, "chi2_SN": csnT1, "chi2_BAOCMB_DR2": cbcT1,
        "bic_bar_dr2": bic_bar, "clears_bar": bool(jcT1 <= bic_bar),
        "margin_or_miss": float(jcT1 - bic_bar), "H0_dressed_from_alpha": float(H0T1),
        "k_sensitivity_row_note": "T1 uses k=4 fitted constants; even the ceiling (best possible) "
                                  "does not license a claim -- it is the geometry's floor on chi2.",
    }

    out = {
        "probe": "WP-B Stage-1 Phase-B STEP 3+4: three-phase geometry scoring + T1 ceiling",
        "spec": "PLAN_WPB_threephase.md sec 3 (outcomes 2-3, accounting; T1 diagnostic)",
        "track_verdict_from_step12": verdict,
        "data": {"sn": "harness.sn_chi2 (1580 Pantheon+, full stat+sys cov, offset marginalised)",
                 "bao_cmb": "DESI DR2 (13 BAO, desi_dr2_rows.json) + Planck acoustic point, "
                            "alpha marginalised, rd=147.09 (mirror of harness.bao_cmb_chi2)"},
        "bic_bar_dr2": {"chi2_LCDM_dr2": chi2_lcdm, "Om_lcdm": Om_lcdm, "N": N, "lnN": lnN,
                        "bar": bic_bar,
                        "crosscheck_forced_joint_fit_chi2_LCDM_dr2": 1399.805009110365,
                        "crosscheck_forced_joint_fit_bar_dr2": 1407.1790109697151},
        "accounting": "primary k=0 (four constants pinned by structure data, external-calibration "
                      "status; band-widths -> systematic on predicted chi2); sensitivity row k=4.",
        "STEP3_k0_forced_prediction": k0,
        "degeneracy_check": degeneracy,
        "STEP4_T1_diagnostic_ceiling": t1,
        "runtime_s": round(time.time() - t0, 2),
    }
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2, default=TP._json_default)
    print(f"[geometry] wrote {_OUT}  ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
