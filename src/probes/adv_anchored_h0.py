#!/usr/bin/env python3
"""ADVERSARIAL clean-room recomputation of the calibrator-anchored H0.

Two independent checks against phaseF_freshH0.json:

(A) LCDM anchored gate: my own flat-LCDM comoving distance + my own GLS (MB,q)
    profiling against Pantheon+SH0ES with Cepheid calibrators pinning M_B. Should
    reproduce ~73.5 (SH0ES ladder). I do NOT call phaseF's run_model/extract or
    F.D_shape_LCDM; I re-implement the distance and the linear algebra.

(B) Free-history anchored fullrate H0: drive the Probe-R V fixed history through the
    modelv solver, get g_dress, Hd(0), S0 independently; then GLS-anchor Hbar0 the
    same way and form H0_fullrate = Hd(0)*Hbar0, H0_gdress = g_dress*Hbar0. Confirm
    the direction (near 73) and the local-over-global excess = Hbar0_anch/Hbar0_glob-1.
"""
import os, sys, json
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
os.chdir(_SRC)
sys.path.insert(0, _HERE)
sys.path.insert(0, _SRC)
import modelv_theory as MV

C_KMS = 299792.458
H0REF = 70.0
HBAR0REF = 55.0
DATA = "data/PantheonSH0ES.dat"
COV = "data/PantheonSH0ES_STATSYS.cov"

Z_NODES = [0.0, 0.3, 0.7, 1.3, 2.33]
FV_NODES_V = [0.64013, 0.53112, 0.39578, 0.27945, 0.19359]


def lcdm_Dc_dimensionless(z, Om, n=4000):
    """Clean-room flat-LCDM dimensionless comoving distance int_0^z dz'/E(z')."""
    z = np.asarray(z, float)
    zmax = float(z.max())
    zg = np.linspace(0.0, zmax * 1.0001 + 1e-9, n)
    E = np.sqrt(Om * (1 + zg) ** 3 + (1.0 - Om))
    from numpy import trapz
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / E[1:] + 1.0 / E[:-1]) * np.diff(zg))])
    return np.interp(z, zg, cum)


def gls_MB_q(C, w, y):
    """GLS fit of y = MB*1 + q*w with full covariance C. Returns (MB, q, chi2).
    Solved via Cholesky; A=[1,w]."""
    from scipy.linalg import cho_factor, cho_solve
    n = len(y)
    A = np.column_stack([np.ones(n), w.astype(float)])
    cf = cho_factor(C)
    CiA = cho_solve(cf, A)
    Ciy = cho_solve(cf, y)
    M = A.T @ CiA
    v = A.T @ Ciy
    th = np.linalg.solve(M, v)
    chi2 = float(y @ Ciy - v @ th)
    return float(th[0]), float(th[1]), chi2


def main():
    df = pd.read_csv(DATA, sep=r"\s+")
    n_all = len(df)
    with open(COV) as f:
        n_cov = int(f.readline())
    Cfull = np.fromfile(COV, sep=" ")[1:].reshape(n_cov, n_cov)
    assert n_cov == n_all

    zHD = df["zHD"].to_numpy(float)
    zHEL = df["zHEL"].to_numpy(float)
    mb = df["m_b_corr"].to_numpy(float)
    ceph = df["CEPH_DIST"].to_numpy(float)
    iscal = df["IS_CALIBRATOR"].to_numpy(float).astype(int) == 1

    # main_z001 variant: calibrators + non-calibrator Hubble-flow with zHD>0.01
    hf = (~iscal) & (zHD > 0.01)
    sel = iscal | hf
    idx = np.where(sel)[0]
    w_s = hf[idx].astype(float)          # 1 for HF SNe, 0 for calibrators
    cal_s = iscal[idx]
    zhd_s, zhel_s, mb_s, ceph_s = zHD[idx], zHEL[idx], mb[idx], ceph[idx]
    C = Cfull[np.ix_(sel, sel)]
    hfmask = w_s > 0.5
    zhd_hf, zhel_hf = zhd_s[hfmask], zhel_s[hfmask]

    out = {"n_sel": int(len(idx)), "n_cal": int(cal_s.sum()), "n_hf": int(hfmask.sum())}

    # base mu0: Cepheid geometric distance modulus on calibrators, 0 on HF (filled per shape)
    mu0base = np.where(cal_s, ceph_s, 0.0)

    # ---------------- (A) LCDM anchored gate ----------------
    om_grid = np.arange(0.05, 0.6000001, 0.0025)
    best = (np.inf, None, None, None)
    for Om in om_grid:
        mu0 = mu0base.copy()
        dL = (C_KMS / H0REF) * (1.0 + zhel_hf) * lcdm_Dc_dimensionless(zhd_hf, Om)
        mu0[hfmask] = 5.0 * np.log10(dL) + 25.0
        y = mb_s - mu0
        MB, q, chi2 = gls_MB_q(C, w_s, y)
        if chi2 < best[0]:
            best = (chi2, Om, q, MB)
    chi2L, OmL, qL, MBL = best
    H0_lcdm = H0REF * 10 ** (-qL / 5.0)
    out["lcdm_gate"] = {"Om_best": float(OmL), "q": float(qL), "MB": float(MBL),
                        "chi2_min": float(chi2L), "H0_anchored": float(H0_lcdm),
                        "phaseF_value": 73.5274431482745, "expected": "~73.5"}

    # ---------------- (B) free-history anchored fullrate ----------------
    fv = MV.fv_from_nodes(FV_NODES_V, z_nodes=Z_NODES)
    sol = MV.modelv_solve(fv, lapse="algebraic", Ngrid=30000)
    fv0 = float(sol.fv0)
    g_dress = float(MV.g_dress(fv0))
    Hd0 = float(np.interp(0.0, sol.z, sol.Hd))
    # S0 = dD_M/dz|_0 via low-z linear fit (independent of phaseF's exact z-nodes)
    zs = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    S0 = float(np.polyfit(zs, sol.D_M(zs), 1)[0])

    # anchor Hbar0 to the SH0ES ladder with the SAME free-history shape
    mu0 = mu0base.copy()
    dL = (C_KMS / HBAR0REF) * (1.0 + zhel_hf) * sol.D_M(zhd_hf)
    mu0[hfmask] = 5.0 * np.log10(dL) + 25.0
    y = mb_s - mu0
    MBf, qf, chi2f = gls_MB_q(C, w_s, y)
    Hbar0 = HBAR0REF * 10 ** (-qf / 5.0)

    out["free_fixed"] = {
        "fv0": fv0, "g_dress": g_dress, "Hd0": Hd0, "S0": S0,
        "Hbar0_anchored": float(Hbar0),
        "H0_gdress": float(g_dress * Hbar0),
        "H0_fullrate": float(Hd0 * Hbar0),
        "H0_local_slope": float(Hbar0 / S0),
        "chi2_min": float(chi2f),
        "phaseF": {"g_dress": 1.189183045456095, "Hd0": 1.2575679214699604,
                   "S0": 0.7946988394415567, "Hbar0": 58.32194048848205,
                   "H0_gdress": 69.35546280700221, "H0_fullrate": 73.3438014761951,
                   "H0_local_slope": 73.38873242782824},
    }

    # ---------------- local-over-global excess ----------------
    # global reference: Probe R V joint SN+BAO+CMB Hbar0
    Hbar0_glob = 53.794282522327265
    H0_fullrate_glob = Hd0 * Hbar0_glob   # same fixed shape -> same Hd0
    H0_gdress_glob = g_dress * Hbar0_glob
    out["local_excess"] = {
        "Hbar0_anchored": float(Hbar0),
        "Hbar0_global": Hbar0_glob,
        "excess_Hbar0": float(Hbar0 / Hbar0_glob - 1.0),
        "excess_fullrate": float((Hd0 * Hbar0) / H0_fullrate_glob - 1.0),
        "excess_gdress": float((g_dress * Hbar0) / H0_gdress_glob - 1.0),
        "phaseF_excess_fullrate": 0.08419761917976598,
        "note": "excess reduces to Hbar0_anchored/Hbar0_global-1 (same fixed shape).",
    }

    # direction summary
    out["direction"] = {
        "free_anchored_fullrate": float(Hd0 * Hbar0),
        "paper1_tracker_fresh_H0": 73.0,
        "SH0ES_published": 73.04,
        "moved_from_73": bool(abs(Hd0 * Hbar0 - 73.0) > 1.0),
    }

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "probes_out", "adv_anchored_h0.json"), "w") as fo:
        json.dump(out, fo, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
