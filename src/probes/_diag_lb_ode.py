#!/usr/bin/env python3
"""Diagnostic v3: ODE solver for the LB (rate-ratio) coupled redshift map.

Both Picard (v1) and operator-split (v2) diverge because gamma_bar depends on f_v'
and any scheme that differentiates the iterated map re-injects high-frequency noise.
Robust fix: recognise the LB redshift relation as a first-order ODE for F(tau)=f_v(z(tau))
whose RHS gives f_v' ANALYTICALLY (no numerical differentiation of a noisy map):

  1+Z(F) = (tau0/tau)^{2/3} ((1-F)/(1-fv0))^{1/3} gamma_bar/gamma_bar0,
  gamma_bar = 1 + tau F'/(2(1-F))        [LB lapse]
  =>  F' = (2(1-F)/tau) [ B (1+Z(F)) tau^{2/3} (1-F)^{-1/3} - 1 ],
      B = gamma_bar0 (1-fv0)^{1/3} / tau0^{2/3},   Z(F)=f_v^{-1}(F).

Normalisation gamma_bar0: the redshift relation fixes the map only up to the present
lapse gamma_bar0 (a scale, NOTES sec 6.1); on the tracker gamma_bar0=(2+fv0)/2, and that
LA-matched value is the convention here. GATE: on the tracker this ODE must reproduce
LA (D_M,D_H,SN chi2). Off-tracker it yields the LB systematic.

Run from src/:  ../.venv/bin/python probes/_diag_lb_ode.py
"""
import os
import sys
import io
import time
import contextlib
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import PchipInterpolator

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
os.chdir(_SRC)

import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):
    import harness as HN

NGRID = 30000


def tau_grid(tau0, tau_lo_frac, Ngrid):
    tlo = tau_lo_frac * tau0
    return np.unique(np.concatenate([
        np.linspace(tlo, tau0, int(Ngrid)),
        np.geomspace(tlo, tau0, int(Ngrid)),
    ]))


def build_Zinv(fv_of_z, z_hi=2.0e4, n=8000):
    """Z(F) = f_v^{-1}(F): invert the monotone-decreasing forced history.
    Sample z (geomspace-ish) -> F=f_v(z) descending; build PCHIP F->z on the strictly
    decreasing part (above the floor)."""
    zs = np.concatenate([[0.0], np.geomspace(1e-4, z_hi, n)])
    Fs = fv_of_z(zs)
    # keep strictly decreasing F (drop the clipped-floor tail where F is flat)
    keep = np.concatenate([[True], np.diff(Fs) < -1e-15])
    Fk, zk = Fs[keep], zs[keep]
    order = np.argsort(Fk)               # ascending F for PCHIP
    Zinv = PchipInterpolator(Fk[order], zk[order], extrapolate=True)
    return Zinv, float(Fk.min()), float(Fk.max())


def solve_lb_ode(fv_of_z, *, Ngrid=NGRID, tau_lo_frac=1e-6, method="LSODA",
                 rtol=1e-9, atol=1e-12):
    fv0 = float(fv_of_z(0.0))
    tau0 = (2.0 + fv0) / 3.0
    gamma0 = (2.0 + fv0) / 2.0                      # LA-matched normalisation
    tau = tau_grid(tau0, tau_lo_frac, Ngrid)
    t23 = tau ** (2.0 / 3.0)
    floor = float(getattr(fv_of_z, "floor", 0.0))

    Zinv, Fmin, Fmax = build_Zinv(fv_of_z)
    B = gamma0 * (1.0 - fv0) ** (1.0 / 3.0) / tau0 ** (2.0 / 3.0)
    Ffloor = max(Fmin, floor)

    def rhs(t, y):
        Fv = float(np.clip(y[0], Ffloor, Fmax))
        z = float(Zinv(Fv))
        Fp = (2.0 * (1.0 - Fv) / t) * (B * (1.0 + z) * t ** (2.0 / 3.0)
                                       * (1.0 - Fv) ** (-1.0 / 3.0) - 1.0)
        return [Fp]

    # integrate tau0 -> tau_lo (decreasing); evaluate on the reversed grid
    t_eval = tau[::-1]
    sol = solve_ivp(rhs, (tau0, tau[0]), [fv0], method=method, t_eval=t_eval,
                    rtol=rtol, atol=atol, dense_output=False)
    if not sol.success:
        return None
    Farr = np.clip(sol.y[0][::-1], Ffloor, Fmax)     # back to ascending tau
    z = Zinv(Farr)
    # f_v' from the ODE RHS (analytic; no map differentiation)
    fvp = (2.0 * (1.0 - Farr) / tau) * (B * (1.0 + z) * t23
                                        * (1.0 - Farr) ** (-1.0 / 3.0) - 1.0)
    fvp = np.where(Farr <= Ffloor * (1.0 + 1e-9), 0.0, fvp)
    gam = 1.0 + tau * fvp / (2.0 * (1.0 - Farr))
    gamp = np.gradient(gam, tau, edge_order=2)
    integrand = 1.0 / (gam * t23)
    J = cumulative_trapezoid(integrand, tau, initial=0.0)
    dA = t23 * (J[-1] - J)
    DM = (1.0 + z) * dA
    Hbar = 2.0 / (3.0 * tau) + fvp / (3.0 * (1.0 - Farr))
    Hd = gam * Hbar - gamp
    DH = 1.0 / Hd
    return dict(z=z, tau=tau, fv=Farr, DM=DM, DH=DH, gam=gam, gamma0=gam[-1], nfev=sol.nfev)


def compare(fv_of_z, label, sol_la, methods=("LSODA", "RK45")):
    zq = np.geomspace(1e-3, 1100.0, 400)
    dm_la, dh_la = sol_la.D_M(zq), sol_la.D_H(zq)
    zHD, _, _, _ = F.load()
    print(f"\n== {label} ==")
    for m in methods:
        t0 = time.time()
        r = solve_lb_ode(fv_of_z, method=m)
        dt = time.time() - t0
        if r is None:
            print(f"  method={m:6s} -> FAILED")
            continue
        order = np.argsort(r["z"])
        dm_lb = np.interp(zq, r["z"][order], r["DM"][order])
        dh_lb = np.interp(zq, r["z"][order], r["DH"][order])
        rel_dm = float(np.max(np.abs(dm_lb / dm_la - 1.0)))
        rel_dh = float(np.max(np.abs(dh_lb / dh_la - 1.0)))
        dmHD = np.interp(zHD, r["z"][order], r["DM"][order])
        csn = float(HN.sn_chi2(dmHD))
        print(f"  method={m:6s} -> nfev={r['nfev']:5d} {dt:.2f}s  rel_DM={rel_dm:.2e} "
              f"rel_DH={rel_dh:.2e}  gamma0={r['gamma0']:.6f}  SN={csn:.6f}")


def main():
    for fv0 in (0.6426, 0.853):
        trk = MV.tracker_fv_of_z(fv0)
        sol_la = MV.modelv_solve(trk, Ngrid=NGRID, lapse="algebraic")
        compare(trk, f"tracker fv0={fv0}  (gamma0 target={(2+fv0)/2:.6f})", sol_la)

    z_nodes = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
    fv_nodes = np.array([0.62, 0.52, 0.40, 0.28, 0.19])
    bz = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])
    bfv = fv_nodes[-1] * ((1.0 + z_nodes[-1]) / (1.0 + bz)) ** 1.5
    fv = MV.fv_from_nodes(fv_nodes, z_nodes=z_nodes, bridge_z=bz, bridge_fv=bfv)
    sol_la = MV.modelv_solve(fv, Ngrid=NGRID, lapse="algebraic")
    # print the off-tracker LA vs LB spread at the key redshifts
    r = solve_lb_ode(fv)
    if r is not None:
        order = np.argsort(r["z"])
        zk = np.array([0.1, 0.3, 0.55, 0.8, 1.3, 2.0])
        dm_lb = np.interp(zk, r["z"][order], r["DM"][order])
        dh_lb = np.interp(zk, r["z"][order], r["DH"][order])
        dm_rel = np.abs(dm_lb / sol_la.D_M(zk) - 1.0)
        dh_rel = np.abs(dh_lb / sol_la.D_H(zk) - 1.0)
        print("\n== off-tracker fv0=0.62 LA-vs-LB spread ==")
        print(f"  z       = {zk.tolist()}")
        print(f"  DM_rel  = {np.round(dm_rel,4).tolist()}")
        print(f"  DH_rel  = {np.round(dh_rel,4).tolist()}   (NOTES: ~27% at z~0.55)")
    compare(fv, "off-tracker fv0=0.62 (LB should DIFFER from LA)", sol_la)


if __name__ == "__main__":
    main()
