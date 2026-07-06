#!/usr/bin/env python3
"""Diagnostic v4: LB Picard with a LOW-PASS (coarse-PCHIP) map derivative.

The LB instability is numerical: differentiating grid-scale iteration noise in dz/dtau
blows up gamma_bar. Fix: form dz/dtau from a COARSE PchipInterpolator of ln(1+z) vs
ln(tau) (a smooth, near-linear monotone function), evaluated back on the fine grid --
this low-passes the grid noise while the analytic df_v/dz (MonotoneFv.deriv) restores
the physical node kinks. gamma_bar' (for D_H) is likewise a coarse derivative of the
converged, smooth gamma_bar(tau). Warm-start from the LA map.

Tests reproduction of LA on the tracker (D_M over full z incl CMB, D_H, SN chi2) and
the off-tracker LB systematic. Run from src/:
  ../.venv/bin/python probes/_diag_lb_smooth.py
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
    import timescape_baocmb as T
    import harness as HN

NGRID = 30000


def tau_grid(tau0, tau_lo_frac, Ngrid):
    tlo = tau_lo_frac * tau0
    return np.unique(np.concatenate([
        np.linspace(tlo, tau0, int(Ngrid)),
        np.geomspace(tlo, tau0, int(Ngrid)),
    ]))


def _coarse_ddtau(y, tau, ncoarse):
    """d y/d tau via a coarse PCHIP of y vs tau on a log-tau subgrid (low-pass)."""
    lt = np.log(tau)
    lc = np.linspace(lt[0], lt[-1], int(ncoarse))
    # nearest fine indices for the coarse log-tau nodes (strictly increasing, unique)
    idx = np.unique(np.searchsorted(lt, lc).clip(0, len(tau) - 1))
    p = PchipInterpolator(tau[idx], y[idx])
    return p.derivative()(tau)


def solve_lb_smooth(fv_of_z, *, ncoarse=400, w=1.0, n_iter=200, tol=1e-9,
                    Ngrid=NGRID, tau_lo_frac=1e-6, warm=True):
    fv0 = float(fv_of_z(0.0))
    tau0 = (2.0 + fv0) / 3.0
    tau = tau_grid(tau0, tau_lo_frac, Ngrid)
    t23 = tau ** (2.0 / 3.0)
    floor = float(getattr(fv_of_z, "floor", 0.0))
    ceil = float(getattr(fv_of_z, "ceil", 1.0))

    def clip_mask(fv):
        return (fv <= floor * (1.0 + 1e-9)) | (fv >= ceil * (1.0 - 1e-9))

    def fvp_of(fv, z):
        # dz/dtau from a coarse (low-pass) PCHIP; analytic df_v/dz restores node kinks
        dzt = _coarse_ddtau(z, tau, ncoarse)
        fvp = np.where(clip_mask(fv), 0.0, fv_of_z.deriv(z)) * dzt
        return np.where(np.isfinite(fvp), fvp, 0.0)

    z = (tau0 / tau) ** (2.0 / 3.0) - 1.0
    if warm:
        for _ in range(200):
            fv = fv_of_z(z)
            gam = (2.0 + fv) / 2.0
            abar = t23 * (1.0 - fv) ** (-1.0 / 3.0)
            znew = (abar[-1] / abar) * (gam / gam[-1]) - 1.0
            d = float(np.max(np.abs(znew - z)))
            z = znew
            if d < 1e-11:
                break

    dz = np.inf
    it = 0
    for it in range(1, n_iter + 1):
        fv = fv_of_z(z)
        fvp = fvp_of(fv, z)
        gam = 1.0 + tau * fvp / (2.0 * (1.0 - fv))
        abar = t23 * (1.0 - fv) ** (-1.0 / 3.0)
        znew = (abar[-1] / abar) * (gam / gam[-1]) - 1.0
        dz = float(np.max(np.abs(znew - z)))
        z = z + w * (znew - z)
        if not np.all(np.isfinite(z)):
            return None
        if dz < tol:
            break

    fv = fv_of_z(z)
    fvp = fvp_of(fv, z)
    gam = 1.0 + tau * fvp / (2.0 * (1.0 - fv))
    gamp = _coarse_ddtau(gam, tau, ncoarse)
    integrand = 1.0 / (gam * t23)
    J = cumulative_trapezoid(integrand, tau, initial=0.0)
    dA = t23 * (J[-1] - J)
    DM = (1.0 + z) * dA
    Hbar = 2.0 / (3.0 * tau) + fvp / (3.0 * (1.0 - fv))
    Hd = gam * Hbar - gamp
    DH = 1.0 / Hd
    return dict(z=z, tau=tau, fv=fv, DM=DM, DH=DH, gam=gam, gamma0=gam[-1], n_iter=it, dz=dz)


def report(fv_of_z, label, sol_la, ncoarses=(200, 400, 800, 1500)):
    zq = np.geomspace(1e-3, 1100.0, 400)
    dm_la, dh_la = sol_la.D_M(zq), sol_la.D_H(zq)
    zHD, _, _, _ = F.load()
    print(f"\n== {label} ==")
    for nc in ncoarses:
        r = solve_lb_smooth(fv_of_z, ncoarse=nc)
        if r is None:
            print(f"  ncoarse={nc:5d} -> DIVERGED")
            continue
        order = np.argsort(r["z"])
        dm = np.interp(zq, r["z"][order], r["DM"][order])
        dh = np.interp(zq, r["z"][order], r["DH"][order])
        rel_dm = float(np.max(np.abs(dm / dm_la - 1.0)))
        rel_dh = float(np.max(np.abs(dh / dh_la - 1.0)))
        dmHD = np.interp(zHD, r["z"][order], r["DM"][order])
        csn = float(HN.sn_chi2(dmHD))
        print(f"  ncoarse={nc:5d} -> n_iter={r['n_iter']:3d} dz={r['dz']:.1e}  "
              f"rel_DM={rel_dm:.2e} rel_DH={rel_dh:.2e} gamma0={r['gamma0']:.6f} SN={csn:.6f}")


def main():
    for fv0 in (0.6426, 0.853):
        trk = MV.tracker_fv_of_z(fv0)
        sol_la = MV.modelv_solve(trk, Ngrid=NGRID, lapse="algebraic")
        report(trk, f"tracker fv0={fv0} (target gamma0={(2+fv0)/2:.6f}; LB must == LA)", sol_la)

    z_nodes = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
    fv_nodes = np.array([0.62, 0.52, 0.40, 0.28, 0.19])
    bz = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])
    bfv = fv_nodes[-1] * ((1.0 + z_nodes[-1]) / (1.0 + bz)) ** 1.5
    fv = MV.fv_from_nodes(fv_nodes, z_nodes=z_nodes, bridge_z=bz, bridge_fv=bfv)
    sol_la = MV.modelv_solve(fv, Ngrid=NGRID, lapse="algebraic")
    report(fv, "off-tracker fv0=0.62 (LB SHOULD differ from LA)", sol_la)
    r = solve_lb_smooth(fv, ncoarse=400)
    if r is not None:
        order = np.argsort(r["z"])
        zk = np.array([0.1, 0.3, 0.55, 0.8, 1.3, 2.0])
        dm_rel = np.abs(np.interp(zk, r["z"][order], r["DM"][order]) / sol_la.D_M(zk) - 1.0)
        dh_rel = np.abs(np.interp(zk, r["z"][order], r["DH"][order]) / sol_la.D_H(zk) - 1.0)
        print(f"  off-tracker spread z={zk.tolist()}")
        print(f"    DM_rel={np.round(dm_rel,4).tolist()}")
        print(f"    DH_rel={np.round(dh_rel,4).tolist()}  (NOTES ~0.27 at z~0.55)")


if __name__ == "__main__":
    main()
