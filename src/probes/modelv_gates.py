#!/usr/bin/env python3
"""Model V validation gates (PLAN_void_history.md sec 2). Run from src/:

    python probes/modelv_gates.py

Feeds the TRACKER f_v(t) through the general Model V solver (`modelv_theory.py`) and
checks it reproduces the tracker oracle + the non-FLRW audit signature:

  G-T (tracker limit):
    (a) D_M(z) matches the tracker oracle to <1e-6 rel over z in [1e-3, 1100];
    (b) SN full-covariance chi2 == 1391.545176 (<0.01) at the tracker f_v(0.853);
    (c) gamma_bar0 = (2+fv0)/2 = 1.4265;
    (d) dressed H0/Hbar0 = g_dress(fv0): 1.3606 at 0.853, 1.2295 at 0.695.
  G-A (audit regression): dD_M/dz / D_H - 1 (D_M, D_H computed INDEPENDENTLY) equals
    {z=.01:+.0026, .1:+.0249, .5:+.0982, 1:+.1488, 2:+.1752} to <0.002 abs.
  G-N (numerics): halving the integration step moves the tracker JOINT chi2 by <0.01;
    the z<->tau iteration residual is reported.

High-z reference note: the committed `fit_timescape.D_shape_TS` uses a linear tau
grid + np.interp and itself loses precision above z~100 (rel ~8e-5 at z=1090 vs the
brentq oracle), so the accurate reference over the full range is
`timescape_baocmb.DM/DH` (brentq tau + closed-form F). D_shape_TS is used as the
cross-check only where it is accurate (z < 10).
"""
import os
import sys
import io
import contextlib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
os.chdir(_SRC)

import fit_timescape as F
import modelv_theory as MV
# these import-and-print at module level (a fit); silence for a clean gate report
with contextlib.redirect_stdout(io.StringIO()):
    import timescape_baocmb as T
    import harness as HN

FV0 = 0.853
GA_TARGETS = {0.01: 0.0026, 0.1: 0.0249, 0.5: 0.0982, 1.0: 0.1488, 2.0: 0.1752}


def joint_chi2_tracker(fv0, Ngrid):
    """Tracker joint SN + BAO+CMB chi2 through the general Model V solver."""
    trk = MV.tracker_fv_of_z(fv0)
    sol = MV.modelv_solve(trk, Ngrid=Ngrid, lapse="algebraic")
    zHD, zHEL, mb, Cf = F.load()
    csn = HN.sn_chi2(sol.D_M(zHD))
    cbc, _ = HN.bao_cmb_chi2(lambda z, k: float(sol.predict(z, k)))
    return csn + cbc, csn, cbc, sol


def gate_GT():
    print("=" * 74)
    print("G-T  TRACKER LIMIT")
    print("=" * 74)
    trk = MV.tracker_fv_of_z(FV0)
    sol = MV.modelv_solve(trk, Ngrid=30000, lapse="algebraic")

    # (a) distances over z in [1e-3, 1100] vs the accurate brentq oracle
    zq = np.geomspace(1e-3, 1100.0, 600)
    dm_oracle = np.array([T.DM(z, FV0) for z in zq])
    relDM = np.abs(sol.D_M(zq) / dm_oracle - 1.0)
    max_relDM = float(relDM.max())
    zmax = float(zq[relDM.argmax()])
    # cross-check vs D_shape_TS where it is accurate (z < 10)
    lowz = zq[zq < 10.0]
    rel_fit = np.abs(sol.D_M(lowz) / F.D_shape_TS(lowz, FV0) - 1.0).max()
    a_pass = max_relDM < 1e-6
    print(f"  (a) max|D_M/oracle_DM - 1| over z in [1e-3,1100] = {max_relDM:.3e} "
          f"(at z={zmax:.4g})   [<1e-6]  {'PASS' if a_pass else 'FAIL'}")
    print(f"      cross-check vs D_shape_TS (z<10)             = {rel_fit:.3e}")

    # (b) SN full-covariance chi2
    zHD, zHEL, mb, Cf = F.load()
    chi2 = F.make_chi2(zHD, zHEL, mb, Cf)
    c_general = float(chi2(sol.D_M(zHD)))
    c_harness = float(HN.sn_chi2(sol.D_M(zHD)))
    ref = 1391.545176
    b_pass = abs(c_general - ref) < 0.01 and abs(c_harness - ref) < 0.01
    print(f"  (b) SN chi2 (make_chi2)={c_general:.6f}  (harness)={c_harness:.6f}  "
          f"ref={ref:.6f}  |d|={abs(c_general-ref):.2e}   [<0.01]  "
          f"{'PASS' if b_pass else 'FAIL'}")

    # (c) present lapse gamma_bar0 = (2+fv0)/2
    gam0 = (2.0 + sol.fv0) / 2.0
    c_pass = abs(gam0 - 1.4265) < 1e-3
    print(f"  (c) gamma_bar0 = (2+fv0)/2 = {gam0:.4f}   [1.4265]  "
          f"{'PASS' if c_pass else 'FAIL'}")

    # (d) dressed H0/Hbar0 = g_dress(fv0); cross-check against Hd at z=0
    g853 = float(MV.g_dress(0.853))
    g695 = float(MV.g_dress(0.695))
    hd_z0 = float(sol.Hd[np.argmin(np.abs(sol.z))])   # H/Hbar0 at z~=0 from the solve
    d_pass = abs(g853 - 1.3606) < 1e-3 and abs(g695 - 1.2295) < 1e-3 and abs(hd_z0 - g853) < 1e-4
    print(f"  (d) g_dress(0.853)={g853:.4f} [1.3606]  g_dress(0.695)={g695:.4f} [1.2295]  "
          f"Hd(z=0)={hd_z0:.6f}   {'PASS' if d_pass else 'FAIL'}")

    # (e) cross-check: tracker JOINT SN+BAO+CMB at the committed best-fit fv0=0.6426
    #     validates the full BAO+CMB predict path (DM/DH/DV) beyond the G-A ratio.
    solj = MV.modelv_solve(MV.tracker_fv_of_z(0.6426), Ngrid=30000, lapse="algebraic")
    zHDj, _, _, _ = F.load()
    cj = float(HN.sn_chi2(solj.D_M(zHDj)))
    cbcj, aj = HN.bao_cmb_chi2(lambda z, k: float(solj.predict(z, k)))
    joint_ref = cj + cbcj
    H0j = float(MV.g_dress(0.6426) * HN.H0_from_alpha(aj))
    e_pass = abs(joint_ref - 1469.2926) < 0.05
    print(f"  (e) tracker JOINT @ fv0=0.6426 = {joint_ref:.4f} (ref 1469.2926)  "
          f"dressed H0={H0j:.3f} (ref 63.142)   {'PASS' if e_pass else 'FAIL'}")

    return a_pass and b_pass and c_pass and d_pass and e_pass, dict(
        max_relDM=max_relDM, relDM_zmax=zmax, cross_fit=rel_fit,
        sn_chi2=c_general, gamma0=gam0, g853=g853, g695=g695, Hd_z0=hd_z0,
        tracker_joint=joint_ref, tracker_H0=H0j)


def gate_GA():
    print("=" * 74)
    print("G-A  AUDIT REGRESSION  (dD_M/dz / D_H - 1, computed INDEPENDENTLY)")
    print("=" * 74)
    trk = MV.tracker_fv_of_z(FV0)
    sol = MV.modelv_solve(trk, Ngrid=30000, lapse="algebraic")
    ratios = {}
    ok = True
    print("   z      dD_M/dz / D_H - 1     target      |diff|")
    for z, tgt in GA_TARGETS.items():
        h = 1e-4 * (1.0 + z)
        dDMdz = (float(sol.D_M(z + h)) - float(sol.D_M(z - h))) / (2.0 * h)
        ratio = dDMdz / float(sol.D_H(z)) - 1.0
        diff = abs(ratio - tgt)
        ok = ok and (diff < 0.002)
        ratios[z] = ratio
        print(f"  {z:5.2f}   {ratio:+.4f}              {tgt:+.4f}     {diff:.2e}"
              f"   {'ok' if diff < 0.002 else 'FAIL'}")
    print(f"  -> {'PASS' if ok else 'FAIL'}  (all |diff| < 0.002 abs)")
    return ok, ratios


def gate_GN():
    print("=" * 74)
    print("G-N  NUMERICS  (step halving; z<->tau iteration residual)")
    print("=" * 74)
    j1, csn1, cbc1, sol1 = joint_chi2_tracker(FV0, Ngrid=30000)
    j2, csn2, cbc2, sol2 = joint_chi2_tracker(FV0, Ngrid=60000)
    dj = abs(j2 - j1)
    n_pass = dj < 0.01
    print(f"  joint chi2 @ Ngrid=30000 (N={sol1.z.size}) = {j1:.6f}  "
          f"(SN {csn1:.4f} + BAO/CMB {cbc1:.4f})")
    print(f"  joint chi2 @ Ngrid=60000 (N={sol2.z.size}) = {j2:.6f}  "
          f"(SN {csn2:.4f} + BAO/CMB {cbc2:.4f})")
    print(f"  |delta joint chi2 under step halving| = {dj:.2e}   [<0.01]  "
          f"{'PASS' if n_pass else 'FAIL'}")
    print(f"  z<->tau iteration: n_iter={sol1.n_iter}, max|dz| residual={sol1.dz_resid:.2e} "
          f"(target <1e-6)")
    return n_pass, dict(dj=dj, dz_resid=sol1.dz_resid, n_iter=sol1.n_iter,
                        joint=j1)


def main():
    gt_pass, gt = gate_GT()
    print()
    ga_pass, ga = gate_GA()
    print()
    gn_pass, gn = gate_GN()
    print()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  G-T (tracker limit)      : {'PASS' if gt_pass else 'FAIL'}")
    print(f"  G-A (audit regression)   : {'PASS' if ga_pass else 'FAIL'}")
    print(f"  G-N (numerics)           : {'PASS' if gn_pass else 'FAIL'}")
    print()
    print(f"  chi2_TS (SN, tracker)    = {gt['sn_chi2']:.6f}  (ref 1391.545176)")
    print(f"  max distance rel err     = {gt['max_relDM']:.3e}  (<1e-6)")
    print(f"  G-A ratios               = " +
          ", ".join(f"{z}:{ga[z]:+.4f}" for z in GA_TARGETS))
    print(f"  joint chi2 (tracker)     = {gn['joint']:.4f}")
    print(f"  step-halving d(chi2)     = {gn['dj']:.2e}  ; iter residual = {gn['dz_resid']:.2e}")
    return gt_pass, ga_pass, gn_pass


if __name__ == "__main__":
    main()
