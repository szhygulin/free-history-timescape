#!/usr/bin/env python3
"""Diagnostic v2: operator-split solver for the LB (rate-ratio) coupled map.

Naive Picard diverges (see _diag_lb_iteration.py): gamma_bar depends on f_v', f_v'
depends on dz/dtau, dz/dtau depends on gamma_bar -> high-frequency feedback blows up.

Operator split breaks the feedback:
  OUTER (few, damped): from a CONVERGED smooth map z(tau), form a SMOOTH gamma_bar(tau)
    = 1 + tau f_v'/(2(1-f_v)), with f_v' = (analytic df_v/dz)(PCHIP dz/dtau). Freeze it.
  INNER (stable, LA-like): solve 1+z = (abar0/abar)(gamma_bar(tau)/gamma_bar0) for z(tau)
    with gamma_bar held as a FIXED function of tau -- only abar depends on z, so this is
    as stable as the algebraic-lapse solve.
Warm-start the outer loop from the LA map (LA==LB on the tracker).

Tests on the tracker (LB must reproduce LA) and an off-tracker history.
Run from src/:  ../.venv/bin/python probes/_diag_lb_opsplit.py
"""
import os
import sys
import io
import contextlib
import numpy as np
from scipy.integrate import cumulative_trapezoid
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


def solve_lb_opsplit(fv_of_z, *, deriv_mode="pchip_z", n_outer=60, w=0.5,
                     Ngrid=NGRID, tau_lo_frac=1e-6, inner_tol=1e-10, inner_max=200,
                     outer_tol=1e-9, verbose=False):
    fv0 = float(fv_of_z(0.0))
    tau0 = (2.0 + fv0) / 3.0
    tau = tau_grid(tau0, tau_lo_frac, Ngrid)
    t23 = tau ** (2.0 / 3.0)
    floor = float(getattr(fv_of_z, "floor", 0.0))
    ceil = float(getattr(fv_of_z, "ceil", 1.0))

    def clip_mask(fv):
        return (fv <= floor * (1.0 + 1e-9)) | (fv >= ceil * (1.0 - 1e-9))

    def fvp_smooth(fv, z):
        cm = clip_mask(fv)
        if deriv_mode == "pchip_z":
            dzt = PchipInterpolator(tau, z).derivative()(tau)
            fvp = np.where(cm, 0.0, fv_of_z.deriv(z)) * dzt
        elif deriv_mode == "pchip_fv":
            fvp = PchipInterpolator(tau, fv).derivative()(tau)
        elif deriv_mode == "grad":
            fvp = np.where(cm, 0.0, fv_of_z.deriv(z)) * np.gradient(z, tau, edge_order=2)
        else:
            raise ValueError(deriv_mode)
        fvp = np.where(np.isfinite(fvp), fvp, 0.0)
        return np.where(cm, 0.0, fvp)

    def inner_solve(G_tau, z0):
        """Solve the map with gamma_bar(tau)=G_tau FIXED (LA-like, stable)."""
        z = z0.copy()
        for _ in range(inner_max):
            fv = fv_of_z(z)
            abar = t23 * (1.0 - fv) ** (-1.0 / 3.0)
            znew = (abar[-1] / abar) * (G_tau / G_tau[-1]) - 1.0
            d = float(np.max(np.abs(znew - z)))
            z = znew
            if d < inner_tol:
                break
        return z

    # warm start: LA map (gamma_bar=(2+fv)/2)
    z = (tau0 / tau) ** (2.0 / 3.0) - 1.0
    G_la = None
    for _ in range(inner_max):
        fv = fv_of_z(z)
        gam = (2.0 + fv) / 2.0
        abar = t23 * (1.0 - fv) ** (-1.0 / 3.0)
        znew = (abar[-1] / abar) * (gam / gam[-1]) - 1.0
        d = float(np.max(np.abs(znew - z)))
        z = znew
        if d < inner_tol:
            break

    # outer operator-split loop
    dz = np.inf
    ko = 0
    for ko in range(1, n_outer + 1):
        fv = fv_of_z(z)
        fvp = fvp_smooth(fv, z)
        G_tau = 1.0 + tau * fvp / (2.0 * (1.0 - fv))
        z_new = inner_solve(G_tau, z)
        dz = float(np.max(np.abs(z_new - z)))
        z = z + w * (z_new - z)
        if verbose and (ko <= 5 or ko % 10 == 0):
            print(f"      outer={ko:3d}  dz={dz:.3e}  gamma0={G_tau[-1]:.6f}")
        if not np.all(np.isfinite(z)):
            return None
        if dz < outer_tol:
            break

    # final observables
    fv = fv_of_z(z)
    fvp = fvp_smooth(fv, z)
    gam = 1.0 + tau * fvp / (2.0 * (1.0 - fv))
    gamp = np.gradient(gam, tau, edge_order=2)
    integrand = 1.0 / (gam * t23)
    J = cumulative_trapezoid(integrand, tau, initial=0.0)
    dA = t23 * (J[-1] - J)
    DM = (1.0 + z) * dA
    Hbar = 2.0 / (3.0 * tau) + fvp / (3.0 * (1.0 - fv))
    Hd = gam * Hbar - gamp
    DH = 1.0 / Hd
    return dict(z=z, tau=tau, DM=DM, DH=DH, gam=gam, n_outer=ko, dz=dz, gamma0=gam[-1])


def compare(fv_of_z, label, sol_la):
    zq = np.geomspace(1e-3, 1100.0, 400)
    dm_la, dh_la = sol_la.D_M(zq), sol_la.D_H(zq)
    zHD, _, _, _ = F.load()
    print(f"\n== {label} ==")
    for dm in ("pchip_z", "pchip_fv", "grad"):
        for w in (0.5, 0.7, 1.0):
            r = solve_lb_opsplit(fv_of_z, deriv_mode=dm, w=w)
            if r is None:
                print(f"  deriv={dm:8s} w={w:.1f}  -> DIVERGED")
                continue
            order = np.argsort(r["z"])
            dm_lb = np.interp(zq, r["z"][order], r["DM"][order])
            dh_lb = np.interp(zq, r["z"][order], r["DH"][order])
            rel_dm = float(np.max(np.abs(dm_lb / dm_la - 1.0)))
            rel_dh = float(np.max(np.abs(dh_lb / dh_la - 1.0)))
            dmHD = np.interp(zHD, r["z"][order], r["DM"][order])
            csn = float(HN.sn_chi2(dmHD))
            print(f"  deriv={dm:8s} w={w:.1f}  -> n_outer={r['n_outer']:3d} dz={r['dz']:.1e}  "
                  f"rel_DM={rel_dm:.2e} rel_DH={rel_dh:.2e}  gamma0={r['gamma0']:.6f}  SN={csn:.6f}")


def main():
    # tracker: LB must reproduce LA
    for fv0 in (0.6426, 0.853):
        trk = MV.tracker_fv_of_z(fv0)
        sol_la = MV.modelv_solve(trk, Ngrid=NGRID, lapse="algebraic")
        compare(trk, f"tracker fv0={fv0}  (gamma0 target (2+fv0)/2={(2+fv0)/2:.6f})", sol_la)

    # off-tracker smooth history fv0=0.62
    z_nodes = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
    fv_nodes = np.array([0.62, 0.52, 0.40, 0.28, 0.19])
    bz = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])
    bfv = fv_nodes[-1] * ((1.0 + z_nodes[-1]) / (1.0 + bz)) ** 1.5
    fv = MV.fv_from_nodes(fv_nodes, z_nodes=z_nodes, bridge_z=bz, bridge_fv=bfv)
    sol_la = MV.modelv_solve(fv, Ngrid=NGRID, lapse="algebraic")
    compare(fv, "off-tracker fv0=0.62 (LB should DIFFER from LA by ~27%)", sol_la)


if __name__ == "__main__":
    main()
