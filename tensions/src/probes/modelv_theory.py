#!/usr/bin/env python3
"""Model V — general (non-tracker) dressed-geometry solver for a FORCED void history.

Phase 4, `significance-audit`. Production solver behind Probe R. Given an ARBITRARY
void history f_v(z) (Probe R: monotone PCHIP through nodes + a high-z bridge to
f_v -> 0), this module computes the KINEMATIC-reading dressed timescape observables:

    D_M(z)   transverse comoving distance shape         (units c/Hbar0)  -- SN & BAO
    D_H(z)   = c / H_dressed(z), computed INDEPENDENTLY  (units c/Hbar0)  -- BAO
    D_V(z)   = (z D_M^2 D_H)^(1/3)                                        -- BAO
    dressed H0 / Hbar0 = g_dress(f_v(0))

Theory (KINEMATIC / phenomenological reading; see NOTES_modelv_theory.md):
  * walls = spatially flat dust in BARE time:  a_w propto tau^{2/3},  H_w = 2/(3 tau);
  * volume closure a_w^3 = (1-f_v) abar^3  =>  abar propto tau^{2/3} (1-f_v)^{-1/3}
    and, crucially, a_w propto tau^{2/3} for ANY forced f_v (the wall ruler is
    f_v-independent -- the bare abar carries the (1-f_v)^{-1/3}, and it enters ONLY
    the redshift, never the distance ruler);
  * lapse (adopted: ALGEBRAIC)  gamma_bar = (2 + f_v)/2   (Wiltshire09 tracker value,
    generalised as a pure function of the forced f_v);
  * dressed redshift (Wiltshire09 Eq 37):
        1 + z = (abar0/abar)(gamma_bar/gamma_bar0);
  * dressed d_A (DHW17 App. A, generalised f_v):
        d_A(z) = a_w(tau_e) * int_{tau_e}^{tau0} dtau / (gamma_bar a_w)
               = tau_e^{2/3} * int_{tau_e}^{tau0} 2 / ((2+f_v) tau^{2/3}) dtau,
        D_M = (1+z) d_A;
  * dressed Hubble (Wiltshire09 Eq 27):  H = gamma_bar Hbar - dgamma_bar/dt, so
        H/Hbar0 = gamma_bar [ 2/(3 tau) + f_v'/(3(1-f_v)) ] - f_v'/2   (f_v' = df_v/dtau),
        D_H = 1 / (H/Hbar0).   [NEVER derived from dD_M/dz -- the non-FLRW
                                dD_M/dz != D_H signature (gate G-A) is reproduced.]

The tracker limit is exact: fed the tracker f_v(z) this reproduces
`fit_timescape.D_shape_TS` / `timescape_baocmb.DM,DH` to ~1e-9 and the committed
SN chi2 = 1391.545176 (gates G-T, G-A; run `modelv_gates.py`).

Variants:
  lapse="algebraic"   (primary, adopted)  gamma_bar = (2 + f_v)/2   -- pure fn of f_v
  lapse="none"        (V0 control: gamma_bar == 1, pure Buchert, no clock dressing)
  lapse="rate_ratio"  (LB systematic)     gamma_bar = Hbar/H_w = 1 + tau f_v'/(2(1-f_v))
                      (Wiltshire09 Eq 14 / DNW13 Eq 16; NOTES_modelv_theory.md sec 3, LB).

The rate-ratio lapse LB is f_v'-DEPENDENT, so the dressed redshift map itself depends
on f_v' (Wiltshire09 Eq 37 with gamma_bar carrying the void-growth term). It COINCIDES
with the algebraic lapse ON the tracker to machine precision (NOTES sec 7B) and diverges
by up to ~27% off-tracker (z~0.55) -- the declared Probe-R lapse-reading systematic.
Robustness (NOTES risk 2): f_v' is formed from the ANALYTIC PchipInterpolator derivative
(df_v/dz, exposed by MonotoneFv.deriv) composed with a numerical dz/dtau on the CURRENT
z map inside a guarded coupled z<->tau iteration -- never np.gradient of a piecewise-linear
interpolant (which blew up in the theory prototype). df_v/dtau is zeroed where f_v is
clipped at its floor/ceil (the PCHIP extrapolated slope there is meaningless), which keeps
gamma_bar -> 1 in the flat-dust high-z tail. dgamma_bar/dtau (needed for the dressed H,
Wiltshire09 Eq 27) is a 2nd-order numerical derivative of the converged, smooth gamma_bar.

Units: distances in c/Hbar0; tau = Hbar0 t dimensionless. The overall scale is
degenerate with the SN offset and the BAO alpha = c/(Hbar0 r_d), both profiled by
the harness, so the distance SHAPE (not an absolute Hbar0) is what this returns.

Import is clean (only `fit_timescape`, no heavy module-level side effects).
"""
import os
import sys
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import fit_timescape as F   # clean import: functions only, no module-level exec

# Probe R default nodes (PLAN_void_history.md sec 3)
Z_NODES_DEFAULT = np.array([0.0, 0.3, 0.7, 1.3, 2.33])

_FV_FLOOR = 1e-5
_FV_CEIL = 1.0 - 1e-9


# ---------------------------------------------------------------------------
# f_v <-> tracker helpers (local; avoid importing timescape_baocmb, which has a
# heavy module-level fit that prints on import)
# ---------------------------------------------------------------------------
def g_dress(fv0):
    """dressed H0 / Hbar0 for present void fraction fv0 (Wiltshire tracker value)."""
    fv0 = np.asarray(fv0, dtype=float)
    return (4.0 * fv0 ** 2 + fv0 + 4.0) / (2.0 * (2.0 + fv0))


def _fv_of_tau_tracker(tau, fv0):
    """Tracker void fraction as a function of bare time tau (timescape_baocmb.fv_of_tau)."""
    return 3.0 * fv0 * tau / (3.0 * fv0 * tau + (1.0 - fv0) * (2.0 + fv0))


# ---------------------------------------------------------------------------
# monotone f_v(z) callable with an analytic derivative
# ---------------------------------------------------------------------------
class MonotoneFv:
    """f_v as a monotone PCHIP of z, clipped to (floor, ceil), with df_v/dz.

    Callable: fv(z) -> f_v (scalar/array).  .deriv(z) -> df_v/dz (analytic, from the
    underlying PCHIP; the smooth-derivative path recommended in NOTES sec 6.5).
    """

    def __init__(self, z_pts, fv_pts, floor=_FV_FLOOR, ceil=_FV_CEIL):
        z_pts = np.asarray(z_pts, dtype=float)
        fv_pts = np.asarray(fv_pts, dtype=float)
        order = np.argsort(z_pts)
        z_pts, fv_pts = z_pts[order], fv_pts[order]
        # drop duplicate z (PCHIP requires strictly increasing x)
        keep = np.concatenate([[True], np.diff(z_pts) > 0])
        self._p = PchipInterpolator(z_pts[keep], fv_pts[keep], extrapolate=True)
        self._dp = self._p.derivative()
        self.floor, self.ceil = float(floor), float(ceil)
        self.fv0 = float(np.clip(self._p(0.0), self.floor, self.ceil))

    def __call__(self, z):
        return np.clip(self._p(np.asarray(z, dtype=float)), self.floor, self.ceil)

    def deriv(self, z):
        return self._dp(np.asarray(z, dtype=float))


def _default_bridge(z_last, fv_last):
    """Cheap, monotone high-z bridge continuing f_v -> ~0 above the data nodes.

    Fixed z-nodes with a geometric decay anchored just below the last data value.
    This is a sensible default so the CMB integral sees flat-dust+radiation early
    physics (f_v -> 0). Probe R should pass its explicit V-a / V-b bridge (a declared
    systematic) via `fv_from_nodes(..., bridge_z=, bridge_fv=)`; the gates bypass the
    bridge entirely (they feed the analytic tracker f_v(z)).
    """
    z_last = float(z_last)
    fv_last = float(fv_last)
    bz = np.array([z_last * 1.7, z_last * 3.0, 10.0, 30.0, 100.0, 1100.0])
    frac = np.array([0.60, 0.35, 0.15, 0.05, 0.012, 1e-4])
    bfv = fv_last * frac
    # keep strictly above the fixed z_last and strictly decreasing
    m = bz > z_last
    return bz[m], np.minimum.accumulate(np.minimum(bfv[m], 0.999 * fv_last))


def fv_from_nodes(fv_nodes, z_nodes=Z_NODES_DEFAULT, bridge_z=None, bridge_fv=None,
                  floor=_FV_FLOOR, ceil=_FV_CEIL):
    """Build a MonotoneFv from node values + a high-z bridge to f_v -> 0.

    fv_nodes : f_v values at z_nodes (default nodes {0, .3, .7, 1.3, 2.33}).
    bridge_z, bridge_fv : optional explicit high-z bridge nodes; if omitted a cheap
        monotone default (`_default_bridge`) is used.
    """
    z_nodes = np.asarray(z_nodes, dtype=float)
    fv_nodes = np.asarray(fv_nodes, dtype=float)
    if bridge_z is None or bridge_fv is None:
        bridge_z, bridge_fv = _default_bridge(z_nodes[-1], fv_nodes[-1])
    z_pts = np.concatenate([z_nodes, np.asarray(bridge_z, dtype=float)])
    fv_pts = np.concatenate([fv_nodes, np.asarray(bridge_fv, dtype=float)])
    return MonotoneFv(z_pts, fv_pts, floor=floor, ceil=ceil)


def tracker_fv_of_z(fv0, ntau=300000, tau_lo_frac=1e-8):
    """The TRACKER f_v as a MonotoneFv of z, from the oracle kinematics.

    Used by the gates to drive the general solver with the tracker history. A dense
    geomspace tau grid resolves small tau (high z) so f_v(z) is accurate up to the
    CMB point.
    """
    tau0 = F.tau0_tilde(fv0)
    tau = np.geomspace(tau_lo_frac * tau0, tau0, int(ntau))
    z = F.z_of_tau(tau, fv0)                 # decreasing in tau
    fv = _fv_of_tau_tracker(tau, fv0)
    return MonotoneFv(z, fv, floor=1e-12, ceil=_FV_CEIL)


# ---------------------------------------------------------------------------
# core solver
# ---------------------------------------------------------------------------
_LAPSES = ("algebraic", "none", "rate_ratio")


def _lapse_gamma(fv, lapse):
    """Dressing lapse gamma_bar for the derivative-free lapses (pure functions of f_v).

    The rate-ratio lapse (lapse="rate_ratio") is f_v'-dependent; it is formed inside
    modelv_solve (it needs the tau grid + df_v/dtau), never here.
    """
    if lapse == "algebraic":
        return (2.0 + fv) / 2.0
    if lapse == "none":
        return np.ones_like(fv)
    if lapse == "rate_ratio":
        raise ValueError("rate_ratio lapse is f_v'-dependent; formed in modelv_solve")
    raise ValueError(f"unknown lapse {lapse!r} (use one of {_LAPSES})")


class ModelVSolution:
    """Solved dressed geometry for one forced f_v(z).

    Attributes (arrays over the internal tau grid, z ASCENDING for interpolation):
        z, tau, fv, DM, DH, Hd  (Hd = H/Hbar0)
    Query methods (vectorised, linear interpolation on the dense grid):
        D_M(z), D_H(z), D_V(z), predict(z, kind)   kind in {"DM","DH","DV"}
    Scalars:
        fv0, dressed_H0_over_Hbar0, n_iter, dz_resid
    """

    def __init__(self, z, tau, fv, DM, DH, Hd, fv0, n_iter, dz_resid):
        order = np.argsort(z)
        self.z = z[order]
        self.tau = tau[order]
        self.fv = fv[order]
        self.DM = DM[order]
        self.DH = DH[order]
        self.Hd = Hd[order]
        self.fv0 = float(fv0)
        self.n_iter = int(n_iter)
        self.dz_resid = float(dz_resid)
        self.dressed_H0_over_Hbar0 = float(g_dress(fv0))

    def D_M(self, z):
        return np.interp(np.asarray(z, dtype=float), self.z, self.DM)

    def D_H(self, z):
        return np.interp(np.asarray(z, dtype=float), self.z, self.DH)

    def D_V(self, z):
        z = np.asarray(z, dtype=float)
        dM = self.D_M(z)
        dH = self.D_H(z)
        return (z * dM * dM * dH) ** (1.0 / 3.0)

    def predict(self, z, kind):
        if kind == "DM":
            return self.D_M(z)
        if kind == "DH":
            return self.D_H(z)
        if kind == "DV":
            return self.D_V(z)
        raise ValueError(f"unknown kind {kind!r}")


def _tau_grid(tau0, tau_lo_frac, Ngrid):
    """Hybrid tau grid dense at BOTH ends: linspace (dense near tau0, for the small
    difference-of-integrals at low z) UNION geomspace (dense near tau_lo, for high-z
    / CMB placement). Either end alone fails the <1e-6 distance gate (NOTES sec 6.2)."""
    tlo = tau_lo_frac * tau0
    return np.unique(np.concatenate([
        np.linspace(tlo, tau0, int(Ngrid)),
        np.geomspace(tlo, tau0, int(Ngrid)),
    ]))


def modelv_solve(fv_of_z, *, lapse="algebraic", Ngrid=30000, tau0=None,
                 tau_lo_frac=1e-6, tol=1e-8, max_iter=100):
    """Solve the dressed geometry for a forced f_v(z) callable.

    fv_of_z : callable f_v(z); if it exposes .deriv(z) (a MonotoneFv), df_v/dtau is
        formed analytically in the z-direction (kinky at nodes taken analytically)
        and numerically in the smooth tau-direction; otherwise np.gradient(fv, tau).
    lapse   : "algebraic" (adopted) | "none" (V0 control, gamma_bar==1) |
        "rate_ratio" (LB systematic: gamma_bar = Hbar/H_w = 1 + tau f_v'/(2(1-f_v));
        the redshift map itself becomes f_v'-dependent, solved by a guarded coupled
        z<->tau iteration -- see the module docstring / NOTES sec 3).
    Ngrid   : per-component tau-grid size (total ~2*Ngrid after the linspace-geomspace
        union); 30000 gives ~5e-8 distance error on the tracker (passes the gates).
    tau0    : present bare time; default (2+fv0)/3 (tracker value). The distance
        SHAPE is invariant under tau -> lambda tau, so this only sets the scale that
        the SN offset / BAO alpha absorb (NOTES sec 6.1).
    tol, max_iter : z<->tau fixed-point convergence on max|dz|.

    Returns a ModelVSolution.
    """
    if lapse not in _LAPSES:
        raise ValueError(f"unknown lapse {lapse!r} (use one of {_LAPSES})")
    rate = (lapse == "rate_ratio")
    fv0 = float(fv_of_z(0.0))
    if tau0 is None:
        tau0 = (2.0 + fv0) / 3.0
    tau = _tau_grid(tau0, tau_lo_frac, Ngrid)
    t23 = tau ** (2.0 / 3.0)
    has_deriv = hasattr(fv_of_z, "deriv")
    floor = float(getattr(fv_of_z, "floor", 0.0))
    ceil = float(getattr(fv_of_z, "ceil", 1.0))

    def _fvp(fv, z, edge=1):
        # df_v/dtau = (df_v/dz)(dz/dtau); analytic df_v/dz (PCHIP) when available.
        if has_deriv:
            return fv_of_z.deriv(z) * np.gradient(z, tau, edge_order=edge)
        return np.gradient(fv, tau, edge_order=edge)

    def _fvp_lb(fv, z):
        # LB is f_v'-fragile: 2nd-order dz/dtau, sanitised of non-finite entries and
        # zeroed where f_v is CLIPPED (the extrapolated PCHIP slope is meaningless in
        # the flat-dust high-z tail) so gamma_bar -> 1 there, as the tracker limit needs.
        fvp = _fvp(fv, z, edge=2)
        fvp = np.where(np.isfinite(fvp), fvp, 0.0)
        clipped = (fv <= floor * (1.0 + 1e-9)) | (fv >= ceil * (1.0 - 1e-9))
        return np.where(clipped, 0.0, fvp)

    def _gamma_lb(fv, fvp):
        # rate-ratio lapse gamma_bar = Hbar/H_w = 1 + tau f_v'/(2(1-f_v))  (NOTES eq LB)
        return 1.0 + tau * fvp / (2.0 * (1.0 - fv))

    # ---- z <-> tau fixed-point iteration -----------------------------------
    # algebraic/none: gamma_bar is a pure function of f_v, so the redshift map is
    # stable. rate_ratio (LB): gamma_bar carries f_v', so the map is COUPLED to
    # df_v/dtau on the current iterate -- guarded (masked f_v', bounded fixed point).
    z = (tau0 / tau) ** (2.0 / 3.0) - 1.0        # EdS-like initial guess
    dz = np.inf
    it = 0
    for it in range(1, int(max_iter) + 1):
        fv = fv_of_z(z)
        if rate:
            gam = _gamma_lb(fv, _fvp_lb(fv, z))
        else:
            gam = _lapse_gamma(fv, lapse)
        abar = t23 * (1.0 - fv) ** (-1.0 / 3.0)
        onepz = (abar[-1] / abar) * (gam / gam[-1])   # pinned so z(tau0)=0 exactly
        znew = onepz - 1.0
        dz = float(np.max(np.abs(znew - z)))
        z = znew
        if dz < tol:
            break

    # ---- final dressed observables -----------------------------------------
    fv = fv_of_z(z)
    if rate:
        fvp = _fvp_lb(fv, z)                   # df_v/dtau (masked, for gamma + Hubble)
        gam = _gamma_lb(fv, fvp)
        gamp = np.gradient(gam, tau, edge_order=2)  # dgamma_bar/dtau: 2nd-order deriv of
                                                    # the converged, smooth gamma_bar
    else:
        fvp = _fvp(fv, z)                       # df_v/dtau (for the Hubble rate)
        gam = _lapse_gamma(fv, lapse)
        gamp = fvp / 2.0 if lapse == "algebraic" else np.zeros_like(fv)  # dgamma_bar/dtau

    # dressed angular-diameter distance: d_A = a_w(tau) int_tau^{tau0} dtau/(gam a_w)
    integrand = 1.0 / (gam * t23)                 # a_w = tau^{2/3} (const absorbed)
    J = cumulative_trapezoid(integrand, tau, initial=0.0)
    dA = t23 * (J[-1] - J)
    DM = (1.0 + z) * dA

    # dressed Hubble (independent of DM): H/Hbar0 = gam Hbar - dgam/dt
    Hbar = 2.0 / (3.0 * tau) + fvp / (3.0 * (1.0 - fv))
    Hd = gam * Hbar - gamp
    DH = 1.0 / Hd

    return ModelVSolution(z, tau, fv, DM, DH, Hd, fv0, it, dz)


# ---------------------------------------------------------------------------
# convenience wrappers (build history + solve + evaluate). For a fit, call
# modelv_solve ONCE per parameter vector and reuse the solution for all z.
# ---------------------------------------------------------------------------
def _solve_from_nodes(fv_nodes, *, z_nodes=Z_NODES_DEFAULT, bridge_z=None,
                      bridge_fv=None, **solve_kw):
    fv = fv_from_nodes(fv_nodes, z_nodes=z_nodes, bridge_z=bridge_z, bridge_fv=bridge_fv)
    return modelv_solve(fv, **solve_kw)


def modelv_D_M(z_array, fv_nodes, **kw):
    """Dressed transverse comoving distance shape at z_array (units c/Hbar0)."""
    return _solve_from_nodes(fv_nodes, **kw).D_M(z_array)


def modelv_D_H(z_array, fv_nodes, **kw):
    """Dressed c/H_dressed at z_array (units c/Hbar0), computed INDEPENDENTLY of D_M."""
    return _solve_from_nodes(fv_nodes, **kw).D_H(z_array)


def modelv_D_V(z_array, fv_nodes, **kw):
    """Dressed volume-average distance (z D_M^2 D_H)^(1/3) at z_array (units c/Hbar0)."""
    return _solve_from_nodes(fv_nodes, **kw).D_V(z_array)


def modelv_dressed_H0(fv_nodes):
    """dressed H0 / Hbar0 = g_dress(f_v(0)); multiply by Hbar0=c/(alpha r_d) for H0.

    Tracker-consistent normalisation (matches timescape_baocmb and the tau0=(2+fv0)/3
    default). The fully-physical general value additionally carries an f_v'(tau0)
    correction (NOTES sec 6.1); this convention is what Probe R reports.
    """
    fv0 = float(np.asarray(fv_nodes, dtype=float).reshape(-1)[0])
    return float(g_dress(fv0))


if __name__ == "__main__":
    # tiny smoke test (not a gate; run modelv_gates.py for the real gates)
    trk = tracker_fv_of_z(0.853)
    sol = modelv_solve(trk, Ngrid=40000)
    print(f"[smoke] tracker fv0=0.853  n_iter={sol.n_iter}  dz_resid={sol.dz_resid:.2e}")
    print(f"[smoke] D_M(0.5)={sol.D_M(0.5):.6f}  D_H(0.5)={sol.D_H(0.5):.6f}  "
          f"H0/Hbar0={sol.dressed_H0_over_Hbar0:.6f}")
