#!/usr/bin/env python3
"""ADVERSARIAL, from-scratch re-derivation of the Phase-F amplitude split.

Independent reviewer's verification (does NOT import phaseF_joint_ampsplit or
modelv_probeR). Imports ONLY:
  * modelv_theory (MV)  -- the dressed-geometry solver + tracker f_v(z) oracle
  * harness       (H)   -- SN chi2 (Pantheon+) + DR1 BAO+CMB (for a cross-check)
  * fit_timescape (F)   -- SN loader

and re-implements independently:
  * the DR2 BAO+CMB alpha-marginalised chi2 (block-diagonal cov) from the raw
    probes_out/desi_dr2_rows.json  (validated to match H.bao_cmb_chi2 on DR1 rows);
  * the monotone-node -> f_v(z) history glue + the tracker-shaped high-z bridge;
  * the 1-D amplitude/f_v0 profiles + the profile parameter-shift tension sigma.

Checks (refute-by-default):
  (A) chi2-impl cross-check: my DR2-style formula on the DR1 rows reproduces
      H.bao_cmb_chi2 to <1e-8 for an LCDM predict  -> my machinery is the harness's.
  (B) DR2 joint chi2 for LCDM / tracker / free (re-eval at reported nodes).
  (C) FREE-history amplitude split: fix shape=free nodes/free[0], fit ONE A to
      SN-only and BAO+CMB-only.  Expect A_SN~A_BC~0.635, sigma<<2 (dissolves).
  (D) TRACKER sanity: same statistic on the tracker's own f_v0 (shape==amplitude).
      MUST reproduce paper 1's ~0.85 (SN) vs ~0.64 (BAO+CMB) at 4.8-6.6 sigma,
      else the harness is wrong and the headline is REFUTED.

Run from src/:  python probes/verify_ampsplit_adversarial.py
"""
import os
import sys
import io
import json
import time
import contextlib
import numpy as np
from scipy.optimize import minimize_scalar

np.seterr(all="ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SRC)

import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):
    import harness as H

_OUT = os.path.join(os.path.dirname(_SRC), "probes_out")
DR2J = os.path.join(_OUT, "desi_dr2_rows.json")
PROBERJ = os.path.join(_OUT, "modelV_probeR.json")
PHASEFJ = os.path.join(_OUT, "phaseF_joint_ampsplit.json")
OUTJ = os.path.join(_OUT, "verify_ampsplit_adversarial.json")

_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)

zHD, zHEL, mb, Cf = F.load()

# --------------------------------------------------------------------------
# (0) independent alpha-marginalised BAO+CMB chi2, block-diagonal covariance.
#     Written from the harness convention; parametrised by the row list so the
#     SAME code runs on DR1 (cross-check vs H.bao_cmb_chi2) and DR2.
# --------------------------------------------------------------------------
def build_cov(rows):
    n = len(rows)
    C = np.zeros((n, n))
    for i, (z, k, v, e, c) in enumerate(rows):
        C[i, i] = e * e
    for i in range(n):
        for j in range(i + 1, n):
            zi, ki, _, ei, ci = rows[i]
            zj, kj, _, ej, cj = rows[j]
            if zi == zj and {ki, kj} == {"DM", "DH"} and ci is not None:
                C[i, j] = C[j, i] = ci * ei * ej
    return C


def make_baocmb_chi2(rows):
    dv = np.array([r[2] for r in rows], dtype=float)
    cinv = np.linalg.inv(build_cov(rows))
    dCd = dv @ (cinv @ dv)

    def chi2(predict):
        g = np.array([predict(z, k) for z, k, _, _, _ in rows], dtype=float)
        gCi = cinv @ g
        num = g @ (cinv @ dv)
        a = num / (g @ gCi)
        return float(dCd - num * num / (g @ gCi)), float(a)
    return chi2


# ---- DR2 rows (independent read) ----
_DR2 = json.load(open(DR2J))
DR2_BAO = [tuple(r) for r in _DR2["rows"]]
CMB_ROW = tuple(_DR2["cmb_point"]["row"])
DR2_ROWS = DR2_BAO + [CMB_ROW]
RD_DR2 = float(_DR2["rd"])
dr2_baocmb_chi2 = make_baocmb_chi2(DR2_ROWS)


def H0_from_alpha_dr2(a):
    return 299792.458 / (a * RD_DR2)


# --------------------------------------------------------------------------
# (A) cross-check: my formula on the DR1 rows must equal H.bao_cmb_chi2.
# --------------------------------------------------------------------------
def crosscheck_dr1():
    dr1_rows = H.bao_cmb_rows()
    my = make_baocmb_chi2([tuple(r) for r in dr1_rows])
    pred = H.lcdm_predict(0.31)
    c_mine, a_mine = my(pred)
    c_harn, a_harn = H.bao_cmb_chi2(pred)
    dchi = abs(c_mine - c_harn)
    da = abs(a_mine - a_harn)
    log(f"(A) DR1 chi2 cross-check: mine={c_mine:.8f} harness={c_harn:.8f} "
        f"|d|={dchi:.2e}  |dalpha|={da:.2e}")
    return dict(chi2_mine=c_mine, chi2_harness=c_harn, dchi2=dchi, dalpha=da,
               ok=bool(dchi < 1e-6 and da < 1e-9))


# --------------------------------------------------------------------------
# (1) independent node -> f_v(z) glue + tracker-shaped bridge (Probe-R spec)
# --------------------------------------------------------------------------
Z_NODES = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
BRIDGE_Z = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])


def bridge_fv(fv_last):
    return fv_last * ((1.0 + Z_NODES[-1]) / (1.0 + BRIDGE_Z)) ** 1.5


def solve_nodes(v, Ngrid):
    v = np.asarray(v, dtype=float)
    fv = MV.fv_from_nodes(v, z_nodes=Z_NODES, bridge_z=BRIDGE_Z,
                          bridge_fv=bridge_fv(float(v[-1])))
    return MV.modelv_solve(fv, lapse="algebraic", Ngrid=Ngrid)


def node_parts(v, Ngrid):
    sol = solve_nodes(v, Ngrid)
    csn = float(H.sn_chi2(sol.D_M(zHD)))
    cbc, a = dr2_baocmb_chi2(lambda z, k: float(sol.predict(z, k)))
    return csn, cbc, a, sol


def tracker_parts(fv0, Ngrid, ntau):
    trk = MV.tracker_fv_of_z(fv0, ntau=ntau)
    sol = MV.modelv_solve(trk, lapse="algebraic", Ngrid=Ngrid)
    csn = float(H.sn_chi2(sol.D_M(zHD)))
    cbc, a = dr2_baocmb_chi2(lambda z, k: float(sol.predict(z, k)))
    return csn, cbc, a


# --------------------------------------------------------------------------
# robust 1-D minimiser: coarse grid guard + bounded Brent
# --------------------------------------------------------------------------
def min1d(f, lo, hi, ngrid=30):
    xs = np.linspace(lo, hi, ngrid)
    ys = np.array([f(x) for x in xs], dtype=float)
    ys = np.where(np.isfinite(ys), ys, np.inf)
    i = int(np.argmin(ys))
    a = xs[max(i - 1, 0)]
    b = xs[min(i + 1, ngrid - 1)]
    r = minimize_scalar(f, bounds=(a, b), method="bounded",
                        options=dict(xatol=1e-5))
    rf = float(r.fun) if np.isfinite(r.fun) else np.inf
    if rf <= ys[i]:
        return float(r.x), rf
    return float(xs[i]), float(ys[i])


NGRID_FIT = 6000
NGRID_FINE = 30000
TRK_NTAU_FIT = 90000
TRK_NTAU_FINE = 200000


def main():
    out = dict(probe="adversarial independent re-derivation of Phase-F amplitude split")

    # ---- (A) chi2 cross-check ----
    out["chi2_impl_crosscheck_DR1"] = crosscheck_dr1()

    # ---- reference nodes from the committed artifacts ----
    PF = json.load(open(PHASEFJ))
    v_free_reported = np.array(PF["dr2_joint_refit"]["free_history"]["fv_nodes"], float)
    log(f"reported free-history DR2 nodes: {v_free_reported.tolist()}")

    # ============================================================
    # (B) DR2 joint chi2 for LCDM / tracker / free
    # ============================================================
    # LCDM: profile Om
    def lcdm_joint(Om):
        return float(H.sn_chi2(H.lcdm_Dc(zHD, Om))) + dr2_baocmb_chi2(H.lcdm_predict(Om))[0]
    Om_dr2, chi2_lcdm = min1d(lcdm_joint, 0.20, 0.45, ngrid=26)
    _, a_lcdm = dr2_baocmb_chi2(H.lcdm_predict(Om_dr2))
    log(f"(B) LCDM DR2:    Om={Om_dr2:.4f} chi2={chi2_lcdm:.4f} H0={H0_from_alpha_dr2(a_lcdm):.2f}")

    # tracker: profile fv0 (joint)
    fv0_trk, _ = min1d(lambda x: sum(tracker_parts(x, NGRID_FIT, TRK_NTAU_FIT)[:2]),
                       0.30, 0.95, ngrid=28)
    csn_tj, cbc_tj, a_tj = tracker_parts(fv0_trk, NGRID_FINE, TRK_NTAU_FINE)
    chi2_trk = csn_tj + cbc_tj
    log(f"(B) tracker DR2: fv0={fv0_trk:.4f} chi2={chi2_trk:.4f} (SN={csn_tj:.3f} BC={cbc_tj:.3f})")

    # free: re-eval at the reported nodes (a true min can only be <= this)
    csn_f, cbc_f, a_f, sol_f = node_parts(v_free_reported, NGRID_FINE)
    chi2_free = csn_f + cbc_f
    log(f"(B) free DR2 (reported nodes): chi2={chi2_free:.4f} (SN={csn_f:.3f} BC={cbc_f:.3f}) "
        f"fv0={sol_f.fv0:.4f}")

    out["dr2_joint_chi2"] = dict(
        lcdm=dict(Om=Om_dr2, chi2=chi2_lcdm, H0=H0_from_alpha_dr2(a_lcdm)),
        tracker=dict(fv0=fv0_trk, chi2=chi2_trk, chi2_SN=csn_tj, chi2_BAOCMB=cbc_tj),
        free_at_reported_nodes=dict(chi2=chi2_free, chi2_SN=csn_f, chi2_BAOCMB=cbc_f,
                                    fv0=float(sol_f.fv0)),
        free_minus_lcdm=chi2_free - chi2_lcdm,
        free_minus_tracker=chi2_free - chi2_trk)

    # ============================================================
    # (C) FREE amplitude split (shape = reported free nodes / node0)
    # ============================================================
    shape = v_free_reported / v_free_reported[0]

    def amp_parts(A, Ngrid):
        return node_parts(A * shape, Ngrid)[:2]

    A_LO, A_HI = 0.20, 0.90
    A_SN, _ = min1d(lambda A: amp_parts(A, NGRID_FIT)[0], A_LO, A_HI, ngrid=30)
    A_BC, _ = min1d(lambda A: amp_parts(A, NGRID_FIT)[1], A_LO, A_HI, ngrid=30)
    A_J, _ = min1d(lambda A: sum(amp_parts(A, NGRID_FIT)), A_LO, A_HI, ngrid=30)
    csn_ASN, cbc_ASN = amp_parts(A_SN, NGRID_FINE)
    csn_ABC, cbc_ABC = amp_parts(A_BC, NGRID_FINE)
    csn_AJ, cbc_AJ = amp_parts(A_J, NGRID_FINE)
    joint_min = csn_AJ + cbc_AJ
    sep_min = csn_ASN + cbc_ABC
    delta_join = joint_min - sep_min
    sigma_free = float(np.sqrt(max(delta_join, 0.0)))
    log(f"(C) FREE amp-split: A_SN={A_SN:.6f} A_BC={A_BC:.6f} A_J={A_J:.6f} "
        f"delta_join={delta_join:.5f} sigma={sigma_free:.4f}")

    out["free_amplitude_split"] = dict(
        shape_norm=np.round(shape, 5).tolist(),
        A_SN=A_SN, A_BAOCMB=A_BC, A_joint=A_J,
        chi2_SN_at_A_SN=csn_ASN, chi2_BC_at_A_BC=cbc_ABC,
        joint_min=joint_min, separate_min=sep_min,
        delta_chi2_join=delta_join, sigma=sigma_free,
        split_dissolves=bool(sigma_free < 2.0))

    # ============================================================
    # (D) TRACKER sanity split (shape == amplitude, one param f_v0)
    # ============================================================
    fv0_SN, _ = min1d(lambda x: tracker_parts(x, NGRID_FIT, TRK_NTAU_FIT)[0],
                      0.45, 0.94, ngrid=26)
    fv0_BC, _ = min1d(lambda x: tracker_parts(x, NGRID_FIT, TRK_NTAU_FIT)[1],
                      0.30, 0.90, ngrid=26)
    fv0_J, _ = min1d(lambda x: sum(tracker_parts(x, NGRID_FIT, TRK_NTAU_FIT)[:2]),
                     0.30, 0.95, ngrid=28)
    csn_tSN, cbc_tSN, _ = tracker_parts(fv0_SN, NGRID_FINE, TRK_NTAU_FINE)
    csn_tBC, cbc_tBC, _ = tracker_parts(fv0_BC, NGRID_FINE, TRK_NTAU_FINE)
    csn_tJ, cbc_tJ, _ = tracker_parts(fv0_J, NGRID_FINE, TRK_NTAU_FINE)
    trk_joint_min = csn_tJ + cbc_tJ
    trk_sep_min = csn_tSN + cbc_tBC
    trk_delta = trk_joint_min - trk_sep_min
    sigma_trk = float(np.sqrt(max(trk_delta, 0.0)))
    log(f"(D) TRACKER sanity: fv0_SN={fv0_SN:.4f} fv0_BC={fv0_BC:.4f} fv0_J={fv0_J:.4f} "
        f"delta_join={trk_delta:.4f} sigma={sigma_trk:.4f}")

    out["tracker_sanity_split"] = dict(
        fv0_SN=fv0_SN, fv0_BAOCMB=fv0_BC, fv0_joint=fv0_J,
        chi2_SN_at_fv0SN=csn_tSN, chi2_BC_at_fv0BC=cbc_tBC,
        joint_min=trk_joint_min, separate_min=trk_sep_min,
        delta_chi2_join=trk_delta, sigma=sigma_trk,
        reproduces_paper1=bool(4.8 <= sigma_trk <= 6.7 and fv0_SN > 0.80 and fv0_BC < 0.66))

    out["runtime_s"] = round(time.time() - _t0, 1)
    with open(OUTJ, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {OUTJ}")
    log("SUMMARY  " + json.dumps({
        "A_SN": round(A_SN, 5), "A_BC": round(A_BC, 5), "sigma_free": round(sigma_free, 3),
        "fv0_SN": round(fv0_SN, 4), "fv0_BC": round(fv0_BC, 4), "sigma_trk": round(sigma_trk, 3),
        "free_chi2": round(chi2_free, 2), "lcdm_chi2": round(chi2_lcdm, 2),
        "tracker_chi2": round(chi2_trk, 2)}))
    return out


if __name__ == "__main__":
    main()
