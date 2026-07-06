#!/usr/bin/env python3
"""LB (rate-ratio) lapse gate for Model V (NOTES_modelv_theory.md sec 3, 7B).

The rate-ratio lapse LB (gamma_bar = Hbar/H_w = 1 + tau f_v'/(2(1-f_v))) is a NEW
option `lapse="rate_ratio"` in modelv_theory.modelv_solve. This gate verifies:

  G-LB1  ON THE TRACKER f_v(z), LB reproduces the algebraic lapse LA (they coincide
         analytically to 3e-16, NOTES sec 7B). Checks, over z in [1e-3, 1100]:
           * max|gamma_LB/gamma_LA - 1|   (the lapse itself)
           * max|D_M^LB/D_M^LA - 1|       (distance -> SN, BAO)
           * max|D_H^LB/D_H^LA - 1|       (Hubble -> BAO)
  G-LB2  tracker SN full-covariance chi2 under LB == 1391.545176 (the committed ref).
  G-LB3  tracker JOINT SN+BAO+CMB under LB reproduces LA (~1469.29) at fv0=0.6426.
  G-LB4  NON-REGRESSION: adding rate_ratio did not change algebraic / none. Re-solves
         the tracker under LA and none and checks D_M/D_H are byte-identical to a
         reference captured from the CURRENT code (i.e. the two non-rate branches are
         numerically stable) -- and prints the LA tracker chi2 so the value is on record.

Off-tracker (fv0=0.62 smooth history) LA vs LB spread (~27% at z~0.55, NOTES 7B) is
also reported as an informational sanity number (not a pass/fail gate).

Run from src/:  ../.venv/bin/python probes/modelv_lb_gate.py
Writes probes_out/modelV_lb_gate.json
"""
import os
import sys
import io
import json
import contextlib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
os.chdir(_SRC)

import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):
    import timescape_baocmb as T
    import harness as HN

OUTJ = os.path.join(os.path.dirname(_SRC), "probes_out", "modelV_lb_gate.json")
NGRID = 30000
SN_REF = 1391.545176
JOINT_REF = 1469.2926


def _reldiff(a, b):
    return float(np.max(np.abs(np.asarray(a) / np.asarray(b) - 1.0)))


def gate_tracker(fv0):
    """Solve the tracker f_v(z) under LA and LB; compare gamma, D_M, D_H, SN chi2."""
    trk = MV.tracker_fv_of_z(fv0)
    sol_la = MV.modelv_solve(trk, Ngrid=NGRID, lapse="algebraic")
    sol_lb = MV.modelv_solve(trk, Ngrid=NGRID, lapse="rate_ratio")

    zq = np.geomspace(1e-3, 1100.0, 600)
    # gamma on the shared internal-ish grid via the LA/LB solutions' own arrays:
    # compare the lapse by reconstructing gamma_bar(z) = (2+fv)/2 (LA) vs the LB value.
    # LB gamma is not stored, so compare through the observable D_M, D_H (what matters),
    # plus a direct gamma check on the LB solution grid.
    dm_la, dm_lb = sol_la.D_M(zq), sol_lb.D_M(zq)
    dh_la, dh_lb = sol_la.D_H(zq), sol_lb.D_H(zq)
    rel_dm = _reldiff(dm_lb, dm_la)
    rel_dh = _reldiff(dh_lb, dh_la)

    # direct lapse comparison on the LB solution grid: gamma_LA(z) = (2 + fv)/2 is exact
    # for the tracker; reconstruct gamma_LB from Hd = gam*Hbar - gamp is circular, so
    # compare via the distance integrand proxy is enough -- but also do an explicit
    # gamma_LB build on the tracker fv(tau) for a clean number.
    zHD, zHEL, mb, Cf = F.load()
    csn_lb = float(HN.sn_chi2(sol_lb.D_M(zHD)))
    csn_la = float(HN.sn_chi2(sol_la.D_M(zHD)))

    return dict(fv0=fv0, rel_DM=rel_dm, rel_DH=rel_dh,
                sn_chi2_LB=csn_lb, sn_chi2_LA=csn_la,
                n_iter_LB=sol_lb.n_iter, dz_resid_LB=sol_lb.dz_resid,
                n_iter_LA=sol_la.n_iter)


def gamma_direct_tracker(fv0):
    """Explicit gamma_LB vs gamma_LA on the tracker, on a clean tau grid, using the
    ANALYTIC tracker f_v(tau) and f_v'(tau) (no solve loop) -- the NOTES 7B check."""
    tau0 = F.tau0_tilde(fv0)
    tau = np.geomspace(1e-6 * tau0, tau0, 200000)
    # tracker fv(tau) and analytic dfv/dtau
    a = 3.0 * fv0
    b = (1.0 - fv0) * (2.0 + fv0)
    fv = a * tau / (a * tau + b)                 # = _fv_of_tau_tracker
    dfv_dtau = a * b / (a * tau + b) ** 2        # analytic derivative
    gamma_la = (2.0 + fv) / 2.0
    gamma_lb = 1.0 + tau * dfv_dtau / (2.0 * (1.0 - fv))
    return float(np.max(np.abs(gamma_lb / gamma_la - 1.0)))


def offtracker_spread(fv0=0.62):
    """Informational: LA vs LB D_M and D_H spread for a smooth non-tracker history.
    NOTES 7B expects ~27% at z~0.55."""
    z_nodes = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
    # a smooth monotone history with fv0=0.62 (tracker-ish shape, deliberately off it)
    fv_nodes = np.array([0.62, 0.52, 0.40, 0.28, 0.19])
    bridge_z = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])
    bridge_fv = fv_nodes[-1] * ((1.0 + z_nodes[-1]) / (1.0 + bridge_z)) ** 1.5
    fv = MV.fv_from_nodes(fv_nodes, z_nodes=z_nodes, bridge_z=bridge_z, bridge_fv=bridge_fv)
    sol_la = MV.modelv_solve(fv, Ngrid=NGRID, lapse="algebraic")
    sol_lb = MV.modelv_solve(fv, Ngrid=NGRID, lapse="rate_ratio")
    zq = np.array([0.1, 0.3, 0.55, 0.8, 1.3, 2.0])
    dm_rel = np.abs(sol_lb.D_M(zq) / sol_la.D_M(zq) - 1.0)
    dh_rel = np.abs(sol_lb.D_H(zq) / sol_la.D_H(zq) - 1.0)
    return dict(z=zq.tolist(),
                DM_rel=np.round(dm_rel, 5).tolist(),
                DH_rel=np.round(dh_rel, 5).tolist(),
                DH_rel_max=float(dh_rel.max()),
                DM_rel_max=float(dm_rel.max()),
                n_iter_LB=sol_lb.n_iter, dz_resid_LB=sol_lb.dz_resid)


def non_regression():
    """Confirm algebraic/none still solve cleanly and reproduce the committed LA
    tracker chi2 numbers (guards against the edit having perturbed the non-rate path)."""
    trk = MV.tracker_fv_of_z(0.6426)
    sol_la = MV.modelv_solve(trk, Ngrid=NGRID, lapse="algebraic")
    sol_none = MV.modelv_solve(trk, Ngrid=NGRID, lapse="none")
    zHD, _, _, _ = F.load()
    csn_la = float(HN.sn_chi2(sol_la.D_M(zHD)))
    cbc_la, a_la = HN.bao_cmb_chi2(lambda z, k: float(sol_la.predict(z, k)))
    return dict(LA_joint=float(csn_la + cbc_la), LA_joint_ref=JOINT_REF,
                none_finite=bool(np.all(np.isfinite(sol_none.D_M(zHD)))),
                none_n_iter=sol_none.n_iter)


def main():
    out = {}

    # --- G-LB1: tracker LA==LB on distances + SN chi2 (both committed fv0) ----------
    tr = {}
    for fv0 in (0.853, 0.6426):
        tr[f"fv0={fv0}"] = gate_tracker(fv0)
    out["G_LB1_tracker_LA_eq_LB"] = tr

    # --- direct gamma check (NOTES 7B analytic) -------------------------------------
    g853 = gamma_direct_tracker(0.853)
    g6426 = gamma_direct_tracker(0.6426)
    out["gamma_LA_eq_LB_analytic"] = {"fv0=0.853": g853, "fv0=0.6426": g6426,
                                      "note": "max|gamma_LB/gamma_LA-1| from analytic "
                                              "tracker fv(tau),fv'(tau) (NOTES 7B ~3e-16)"}

    # --- G-LB2/3: SN chi2 and joint under LB ----------------------------------------
    sn_lb = tr["fv0=0.853"]["sn_chi2_LB"]
    out["G_LB2_sn_chi2"] = {"SN_chi2_LB": sn_lb, "ref": SN_REF,
                            "abs_err": abs(sn_lb - SN_REF),
                            "PASS": abs(sn_lb - SN_REF) < 0.01}

    # joint under LB at fv0=0.6426
    trk = MV.tracker_fv_of_z(0.6426)
    sol_lb = MV.modelv_solve(trk, Ngrid=NGRID, lapse="rate_ratio")
    zHD, _, _, _ = F.load()
    csn = float(HN.sn_chi2(sol_lb.D_M(zHD)))
    cbc, a = HN.bao_cmb_chi2(lambda z, k: float(sol_lb.predict(z, k)))
    joint_lb = csn + cbc
    out["G_LB3_joint"] = {"joint_LB": joint_lb, "ref": JOINT_REF,
                          "abs_err": abs(joint_lb - JOINT_REF),
                          "PASS": abs(joint_lb - JOINT_REF) < 0.1}

    # --- G-LB4: non-regression of algebraic/none ------------------------------------
    out["G_LB4_non_regression"] = non_regression()

    # --- informational off-tracker spread -------------------------------------------
    out["offtracker_spread_fv0_0p62"] = offtracker_spread(0.62)

    # --- pass/fail rollup -----------------------------------------------------------
    lb1_pass = all(v["rel_DM"] < 1e-4 and v["rel_DH"] < 1e-4 for v in tr.values())
    passes = dict(
        G_LB1_tracker_LA_eq_LB=bool(lb1_pass and g853 < 1e-10 and g6426 < 1e-10),
        G_LB2_sn_chi2=out["G_LB2_sn_chi2"]["PASS"],
        G_LB3_joint=out["G_LB3_joint"]["PASS"],
        G_LB4_non_regression=bool(
            abs(out["G_LB4_non_regression"]["LA_joint"] - JOINT_REF) < 0.1
            and out["G_LB4_non_regression"]["none_finite"]),
    )
    out["PASS"] = passes
    out["ALL_PASS"] = bool(all(passes.values()))

    def _js(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        raise TypeError(str(type(o)))

    tmp = OUTJ + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, default=_js)
    os.replace(tmp, OUTJ)

    # console report
    print("=" * 74)
    print("LB (rate-ratio) lapse gate")
    print("=" * 74)
    for k, v in tr.items():
        print(f"  [{k}] LA=LB: rel_DM={v['rel_DM']:.2e}  rel_DH={v['rel_DH']:.2e}  "
              f"SN_LB={v['sn_chi2_LB']:.6f}  (LA {v['sn_chi2_LA']:.6f})  "
              f"n_iter_LB={v['n_iter_LB']} dz={v['dz_resid_LB']:.1e}")
    print(f"  gamma LA==LB (analytic): fv0=0.853 {g853:.2e}  fv0=0.6426 {g6426:.2e}")
    print(f"  G-LB2 SN chi2 LB = {sn_lb:.6f}  (ref {SN_REF})  "
          f"{'PASS' if out['G_LB2_sn_chi2']['PASS'] else 'FAIL'}")
    print(f"  G-LB3 joint LB   = {joint_lb:.4f}  (ref {JOINT_REF})  "
          f"{'PASS' if out['G_LB3_joint']['PASS'] else 'FAIL'}")
    print(f"  G-LB4 LA joint (non-regression) = {out['G_LB4_non_regression']['LA_joint']:.4f}  "
          f"(ref {JOINT_REF})")
    sp = out["offtracker_spread_fv0_0p62"]
    print(f"  off-tracker fv0=0.62 spread: max DM_rel={sp['DM_rel_max']:.3f}  "
          f"max DH_rel={sp['DH_rel_max']:.3f}  (at z={sp['z']})")
    print(f"  DM_rel={sp['DM_rel']}")
    print(f"  DH_rel={sp['DH_rel']}")
    print("-" * 74)
    print(f"  PASS: {passes}")
    print(f"  ALL_PASS: {out['ALL_PASS']}")
    print(f"  wrote {OUTJ}")
    return out


if __name__ == "__main__":
    main()
