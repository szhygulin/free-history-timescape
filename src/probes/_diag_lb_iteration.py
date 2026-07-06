#!/usr/bin/env python3
"""Diagnostic: stabilise the LB (rate-ratio) coupled z<->tau iteration.

The LB lapse gamma_bar = 1 + tau f_v'/(2(1-f_v)) makes the dressed redshift map depend
on f_v' = df_v/dtau. The naive Picard iteration (EdS start, np.gradient dz/dtau, no
damping) DIVERGES (high-frequency iteration noise is amplified by the derivative). This
script tries stabilisation strategies on the TRACKER f_v(z) (where LB must reproduce LA)
and reports convergence + accuracy vs the LA solution. It does NOT write an artifact --
it just prints; the winning strategy is folded into modelv_theory.modelv_solve.

Run from src/:  ../.venv/bin/python probes/_diag_lb_iteration.py
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


def solve_lb(fv_of_z, *, deriv_mode, warm, damp, Ngrid=NGRID, tau_lo_frac=1e-6,
             tol=1e-8, max_iter=400, verbose=False):
    """LB solve with pluggable stabilisation.
      deriv_mode : 'grad' (np.gradient) | 'pchip_z' (PCHIP of z(tau)) | 'pchip_fv'
      warm       : warm-start from the algebraic (LA) map before the LB iteration
      damp       : under-relaxation w in z <- z + w (znew - z)
    Returns (z, tau, DM, DH, n_iter, dz_resid).
    """
    fv0 = float(fv_of_z(0.0))
    tau0 = (2.0 + fv0) / 3.0
    tau = tau_grid(tau0, tau_lo_frac, Ngrid)
    t23 = tau ** (2.0 / 3.0)
    floor = float(getattr(fv_of_z, "floor", 0.0))
    ceil = float(getattr(fv_of_z, "ceil", 1.0))

    def clip_mask(fv):
        return (fv <= floor * (1.0 + 1e-9)) | (fv >= ceil * (1.0 - 1e-9))

    def fvp_of(fv, z):
        cm = clip_mask(fv)
        if deriv_mode == "grad":
            fvp = fv_of_z.deriv(z) * np.gradient(z, tau, edge_order=2)
        elif deriv_mode == "pchip_z":
            dzt = PchipInterpolator(tau, z).derivative()(tau)
            dfz = np.where(cm, 0.0, fv_of_z.deriv(z))
            fvp = dfz * dzt
        elif deriv_mode == "pchip_fv":
            fvp = PchipInterpolator(tau, fv).derivative()(tau)
        else:
            raise ValueError(deriv_mode)
        fvp = np.where(np.isfinite(fvp), fvp, 0.0)
        return np.where(cm, 0.0, fvp)

    def gamma_lb(fv, fvp):
        return 1.0 + tau * fvp / (2.0 * (1.0 - fv))

    z = (tau0 / tau) ** (2.0 / 3.0) - 1.0

    if warm:
        for _ in range(200):
            fv = fv_of_z(z)
            gam = (2.0 + fv) / 2.0
            abar = t23 * (1.0 - fv) ** (-1.0 / 3.0)
            znew = (abar[-1] / abar) * (gam / gam[-1]) - 1.0
            d = float(np.max(np.abs(znew - z)))
            z = znew
            if d < tol:
                break

    dz = np.inf
    it = 0
    for it in range(1, int(max_iter) + 1):
        fv = fv_of_z(z)
        gam = gamma_lb(fv, fvp_of(fv, z))
        abar = t23 * (1.0 - fv) ** (-1.0 / 3.0)
        znew = (abar[-1] / abar) * (gam / gam[-1]) - 1.0
        dz = float(np.max(np.abs(znew - z)))
        z = z + damp * (znew - z)
        if verbose and (it <= 5 or it % 25 == 0):
            print(f"      it={it:3d}  dz={dz:.3e}  z_max={np.max(z):.3e}")
        if not np.all(np.isfinite(z)):
            return None, tau, None, None, it, np.nan
        if dz < tol:
            break

    fv = fv_of_z(z)
    fvp = fvp_of(fv, z)
    gam = gamma_lb(fv, fvp)
    gamp = np.gradient(gam, tau, edge_order=2)
    integrand = 1.0 / (gam * t23)
    J = cumulative_trapezoid(integrand, tau, initial=0.0)
    dA = t23 * (J[-1] - J)
    DM = (1.0 + z) * dA
    Hbar = 2.0 / (3.0 * tau) + fvp / (3.0 * (1.0 - fv))
    Hd = gam * Hbar - gamp
    DH = 1.0 / Hd
    return z, tau, DM, DH, it, dz


def main():
    fv0 = 0.6426
    trk = MV.tracker_fv_of_z(fv0)
    sol_la = MV.modelv_solve(trk, Ngrid=NGRID, lapse="algebraic")
    zq = np.geomspace(1e-3, 1100.0, 400)
    dm_la, dh_la = sol_la.D_M(zq), sol_la.D_H(zq)

    strategies = [
        dict(deriv_mode="grad", warm=False, damp=1.0),
        dict(deriv_mode="grad", warm=True, damp=1.0),
        dict(deriv_mode="grad", warm=True, damp=0.3),
        dict(deriv_mode="pchip_z", warm=True, damp=1.0),
        dict(deriv_mode="pchip_z", warm=True, damp=0.5),
        dict(deriv_mode="pchip_z", warm=True, damp=0.3),
        dict(deriv_mode="pchip_z", warm=False, damp=0.3),
        dict(deriv_mode="pchip_fv", warm=True, damp=0.5),
        dict(deriv_mode="pchip_fv", warm=True, damp=0.3),
        dict(deriv_mode="pchip_fv", warm=False, damp=1.0),
    ]
    print(f"tracker fv0={fv0}  (LB must reproduce LA)")
    print("-" * 90)
    for st in strategies:
        z, tau, DM, DH, nit, dz = solve_lb(trk, **st)
        tag = f"deriv={st['deriv_mode']:8s} warm={str(st['warm']):5s} damp={st['damp']:.2f}"
        if z is None or DM is None or not np.all(np.isfinite(DM)):
            print(f"  {tag}  ->  DIVERGED (n_iter={nit}, dz={dz})")
            continue
        # interpolate onto zq (z descending -> ascending)
        order = np.argsort(z)
        dm = np.interp(zq, z[order], DM[order])
        dh = np.interp(zq, z[order], DH[order])
        rel_dm = float(np.max(np.abs(dm / dm_la - 1.0)))
        rel_dh = float(np.max(np.abs(dh / dh_la - 1.0)))
        zHD, _, _, _ = F.load()
        dmHD = np.interp(zHD, z[order], DM[order])
        csn = float(HN.sn_chi2(dmHD))
        print(f"  {tag}  ->  n_iter={nit:3d} dz={dz:.1e}  rel_DM={rel_dm:.2e}  "
              f"rel_DH={rel_dh:.2e}  SN={csn:.6f}")
    print("-" * 90)
    print(f"  (LA SN chi2 reference at fv0=0.853 is 1391.545176; here fv0={fv0})")


if __name__ == "__main__":
    main()
