#!/usr/bin/env python3
"""Diagnostic v5: LB solver with z as the independent variable (the robust one).

All tau-space schemes (Picard v1, opsplit v2, F-ODE v3, smoothing v4) blow up: they
either differentiate a noisy map or invert f_v near the floor. Switching the independent
variable to z removes BOTH pathologies. The LB redshift relation

    1+z = (tau0/tau)^{2/3} ((1-f_v)/(1-fv0))^{1/3} gamma_bar/gamma_bar0,
    gamma_bar = 1 + tau f_v'/(2(1-f_v)),  f_v' = df_v/dtau = (df_v/dz)/(dtau/dz)

solves algebraically for sigma = dtau/dz:
    dtau/dz = tau (df_v/dz) / ( 2 (1-f_v) (Phi - 1) ),
    Phi(z,tau) = (1+z) gamma_bar0 (tau/tau0)^{2/3} ((1-fv0)/(1-f_v))^{1/3}  (= gamma_bar).
A well-conditioned first-order ODE for tau(z), tau(0)=tau0, gamma_bar0=(2+fv0)/2 (the
LA-matched present-lapse normalisation; on the tracker it reproduces LA exactly). No
map differentiation, no f_v inversion. f_v' and gamma_bar' come from the ANALYTIC RHS.

GATE: reproduce LA on the tracker (D_M full z incl CMB z=1089.8, D_H, SN chi2). Then
report the off-tracker LB systematic. Run from src/:
  ../.venv/bin/python probes/_diag_lb_zode.py
"""
import os
import sys
import io
import time
import contextlib
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
os.chdir(_SRC)

import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):
    import harness as HN

Z_MAX = 1200.0


def solve_lb_zode(fv_of_z, *, n_z=6000, z_max=Z_MAX, method="LSODA",
                  rtol=1e-10, atol=1e-13):
    fv0 = float(fv_of_z(0.0))
    tau0 = (2.0 + fv0) / 3.0
    gam0 = (2.0 + fv0) / 2.0
    floor = float(getattr(fv_of_z, "floor", 0.0))
    ceil = float(getattr(fv_of_z, "ceil", 1.0))

    def sigma(z, tau):
        fv = float(fv_of_z(z))
        if fv <= floor * (1.0 + 1e-9) or fv >= ceil * (1.0 - 1e-9):
            return -1.5 * tau / (1.0 + z)              # EdS tail (gamma_bar->1)
        fvz = float(fv_of_z.deriv(z))
        Phi = (1.0 + z) * gam0 * (tau / tau0) ** (2.0 / 3.0) \
            * ((1.0 - fv0) / (1.0 - fv)) ** (1.0 / 3.0)
        d = Phi - 1.0
        if abs(d) < 1e-12:
            return -1.5 * tau / (1.0 + z)
        return tau * fvz / (2.0 * (1.0 - fv) * d)

    def rhs(z, y):
        return [sigma(z, y[0])]

    z_grid = np.concatenate([[0.0], np.geomspace(1e-4, z_max, n_z)])
    sol = solve_ivp(rhs, (0.0, z_max), [tau0], method=method, t_eval=z_grid,
                    rtol=rtol, atol=atol)
    if not sol.success:
        return None
    z = z_grid
    tau = sol.y[0]
    fv = fv_of_z(z)
    fvz = fv_of_z.deriv(z)
    cm = (fv <= floor * (1.0 + 1e-9)) | (fv >= ceil * (1.0 - 1e-9))
    Phi = (1.0 + z) * gam0 * (tau / tau0) ** (2.0 / 3.0) \
        * ((1.0 - fv0) / (1.0 - fv)) ** (1.0 / 3.0)
    d = Phi - 1.0
    sig = np.where(cm | (np.abs(d) < 1e-12),
                   -1.5 * tau / (1.0 + z),
                   tau * fvz / (2.0 * (1.0 - fv) * np.where(np.abs(d) < 1e-30, 1e-30, d)))
    gam = np.where(cm, 1.0, Phi)
    fvp = np.where(cm, 0.0, fvz / sig)                 # df_v/dtau
    t23 = tau ** (2.0 / 3.0)

    # d_A(z) = tau^{2/3} int_0^z (-sigma)/(gam tau^{2/3}) dz'
    integ = (-sig) / (gam * t23)
    Jz = cumulative_trapezoid(integ, z, initial=0.0)
    dA = t23 * Jz
    DM = (1.0 + z) * dA

    # gamma_bar'(tau) = (dgam/dz)/sigma ; dgam/dz = dPhi/dz + dPhi/dtau * sigma
    dgam_dz = Phi * (1.0 / (1.0 + z) + (1.0 / 3.0) * fvz / (1.0 - fv)) \
        + Phi * (2.0 / (3.0 * tau)) * sig
    dgam_dz = np.where(cm, 0.0, dgam_dz)
    gamp = dgam_dz / sig
    Hbar = 2.0 / (3.0 * tau) + fvp / (3.0 * (1.0 - fv))
    Hd = gam * Hbar - gamp
    DH = 1.0 / Hd
    return dict(z=z, tau=tau, fv=fv, DM=DM, DH=DH, gam=gam, Hd=Hd,
                gamma0=float(gam[0]), nfev=int(sol.nfev))


def compare(fv_of_z, label, sol_la, n_zs=(3000, 6000, 12000)):
    zq = np.geomspace(1e-3, 1089.0, 500)
    dm_la, dh_la = sol_la.D_M(zq), sol_la.D_H(zq)
    zHD, _, _, _ = F.load()
    print(f"\n== {label} ==")
    for nz in n_zs:
        t0 = time.time()
        r = solve_lb_zode(fv_of_z, n_z=nz)
        dt = time.time() - t0
        if r is None:
            print(f"  n_z={nz:6d} -> FAILED")
            continue
        dm = np.interp(zq, r["z"], r["DM"])
        dh = np.interp(zq, r["z"], r["DH"])
        rel_dm = float(np.max(np.abs(dm / dm_la - 1.0)))
        rel_dh = float(np.max(np.abs(dh / dh_la - 1.0)))
        # split low/high z to see where any error sits
        lo = zq < 3.0
        rel_dm_lo = float(np.max(np.abs(dm[lo] / dm_la[lo] - 1.0)))
        rel_dh_lo = float(np.max(np.abs(dh[lo] / dh_la[lo] - 1.0)))
        dmHD = np.interp(zHD, r["z"], r["DM"])
        csn = float(HN.sn_chi2(dmHD))

        def predict(zz, k, rr=r):
            dm = float(np.interp(zz, rr["z"], rr["DM"]))
            dh = float(np.interp(zz, rr["z"], rr["DH"]))
            if k == "DM":
                return dm
            if k == "DH":
                return dh
            return (zz * dm * dm * dh) ** (1.0 / 3.0)
        cbc, a = HN.bao_cmb_chi2(predict)
        print(f"  n_z={nz:6d} nfev={r['nfev']:5d} {dt:.2f}s  relDM={rel_dm:.2e}(lo {rel_dm_lo:.1e}) "
              f"relDH={rel_dh:.2e}(lo {rel_dh_lo:.1e}) g0={r['gamma0']:.6f} SN={csn:.6f} joint={csn+cbc:.4f}")


def main():
    for fv0 in (0.6426, 0.853):
        trk = MV.tracker_fv_of_z(fv0)
        sol_la = MV.modelv_solve(trk, Ngrid=30000, lapse="algebraic")
        zHD, _, _, _ = F.load()
        cbc, a = HN.bao_cmb_chi2(lambda z, k: float(sol_la.predict(z, k)))
        csn = float(HN.sn_chi2(sol_la.D_M(zHD)))
        compare(trk, f"tracker fv0={fv0} (LA joint={csn+cbc:.4f}, gamma0 target={(2+fv0)/2:.6f})", sol_la)

    z_nodes = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
    fv_nodes = np.array([0.62, 0.52, 0.40, 0.28, 0.19])
    bz = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])
    bfv = fv_nodes[-1] * ((1.0 + z_nodes[-1]) / (1.0 + bz)) ** 1.5
    fv = MV.fv_from_nodes(fv_nodes, z_nodes=z_nodes, bridge_z=bz, bridge_fv=bfv)
    sol_la = MV.modelv_solve(fv, Ngrid=30000, lapse="algebraic")
    compare(fv, "off-tracker fv0=0.62 (LB SHOULD differ)", sol_la)
    r = solve_lb_zode(fv, n_z=6000)
    if r is not None:
        zk = np.array([0.1, 0.3, 0.55, 0.8, 1.3, 2.0])
        dm_rel = np.abs(np.interp(zk, r["z"], r["DM"]) / sol_la.D_M(zk) - 1.0)
        dh_rel = np.abs(np.interp(zk, r["z"], r["DH"]) / sol_la.D_H(zk) - 1.0)
        print(f"  off-tracker spread z={zk.tolist()}")
        print(f"    DM_rel={np.round(dm_rel,4).tolist()}")
        print(f"    DH_rel={np.round(dh_rel,4).tolist()}  (NOTES ~0.27 at z~0.55)")


if __name__ == "__main__":
    main()
