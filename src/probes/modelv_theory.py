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

The rate-ratio lapse LB is f_v'-DEPENDENT, so the dressed redshift map itself depends on
f_v' (Wiltshire09 Eq 37 with gamma_bar carrying the void-growth term). The naive tau-space
fixed point is UNSTABLE (differentiating the iterated map amplifies grid noise -> it blew
up in the prototype, NOTES risk 2), so `_solve_rate_ratio` recasts the map as a first-order
ODE for tau(z) and integrates it with an ANALYTIC df_v/dz (MonotoneFv.deriv) -- the ODE
trajectory is repelling forward but ATTRACTING backward, so a fixed-step RK4 run BACKWARD
from a high-z EdS boundary is stable and kink-robust; the flat-dust tail (f_v below a
threshold, gamma_bar -> LA) uses the closed-form LA map. LB COINCIDES with the algebraic
lapse ON the tracker to machine precision (NOTES sec 7B) so the gate is reproduced exactly.
Crucially the present lapse gamma_bar0 = gamma_bar(z=0) is NOT the tracker value (2+fv0)/2
for a non-tracker history -- self-consistency (Wiltshire09 Eq 37) fixes it by shooting so
that tau(0)=tau0. Because the observables carry the RATIO gamma_bar/gamma_bar0, this
self-adjustment cancels most of the naive frozen-gamma_bar0 lapse-reading spread (the ~27%
prototype figure), leaving a much smaller off-tracker systematic -- report the value found.

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
from scipy.optimize import brentq

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import fit_timescape as F   # clean import: functions only, no module-level exec

# Probe R default nodes (PLAN_void_history.md sec 3)
Z_NODES_DEFAULT = np.array([0.0, 0.3, 0.7, 1.3, 2.33])

_FV_FLOOR = 1e-5
_FV_CEIL = 1.0 - 1e-9

# rate-ratio (LB) solver: z-grid ceiling (covers the CMB acoustic point z*=1089.8 with
# headroom) and the f_v below which the LB lapse is replaced by the stable closed-form LA
# map (the flat-dust tail where gamma_bar -> LA; sits ABOVE every BAO redshift, since
# f_v(z<=2.33) >~ 0.19 for reconciling histories).
_Z_MAX_RATE = 1200.0
_FV_SWITCH_RATE = 0.10


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


def _solve_rate_ratio(fv_of_z, fv0, floor, ceil, *, Ngrid, tau0,
                      z_max=_Z_MAX_RATE, fv_switch=_FV_SWITCH_RATE):
    """Self-consistent rate-ratio (LB) dressed geometry, solved with z as the variable.

    The LB lapse gamma_bar = 1 + tau f_v'/(2(1-f_v)) makes the dressed redshift map
    depend on f_v', so the tau-space fixed point is unstable (differentiating the iterated
    map amplifies grid noise -> it blew up in the prototype, NOTES risk 2). With z as the
    independent variable the map is a first-order ODE for tau(z):

        1+z = (tau0/tau)^{2/3} ((1-f_v)/(1-fv0))^{1/3} gamma_bar/gamma_bar0,
        gamma_bar = Phi(z,tau) = (1+z) gamma_bar0 (tau/tau0)^{2/3} ((1-fv0)/(1-f_v))^{1/3}
        =>  dtau/dz = tau (df_v/dz) / (2 (1-f_v)(Phi-1))                       [LB region].

    This trajectory is REPELLING forward but ATTRACTING backward, so it is integrated by a
    fixed-step RK4 BACKWARD from a high-z EdS boundary (robust to the PCHIP node kinks;
    cannot hang). Below f_v = fv_switch (the flat-dust tail, gamma_bar -> LA) the stable
    CLOSED-FORM LA map tau_LA(z) is used -- also exact for the CMB D_M (a DM-only point).

    Normalisation (the crux): the present lapse gamma_bar0 = gamma_bar(z=0) is NOT the
    tracker value (2+fv0)/2 for a non-tracker history -- it is fixed by the present
    void-growth rate f_v'(tau0). Self-consistency (Wiltshire09 Eq 37, gamma_bar0 =
    gamma_bar(t0)) requires tau(0)=tau0, so gamma_bar0 is SHOT: root-find gamma_bar0 s.t.
    the backward integration lands on tau(0)=tau0. On the tracker gamma_bar0=(2+fv0)/2
    already yields tau(0)=tau0 (LB==LA), so the shot is a no-op and the gate holds exactly.
    Because the observable distances carry the RATIO gamma_bar/gamma_bar0, this
    self-adjustment CANCELS most of the naive (frozen-gamma_bar0) lapse-reading spread.
    """
    n_z = int(Ngrid)
    z_grid = np.concatenate([[0.0], np.geomspace(1e-4, z_max, n_z)])
    zc = z_grid
    zm = 0.5 * (z_grid[:-1] + z_grid[1:])
    fv_c = fv_of_z(zc)
    fvz_c = fv_of_z.deriv(zc)
    fv_m = fv_of_z(zm)
    fvz_m = fv_of_z.deriv(zm)
    cm_c = (fv_c <= floor * (1.0 + 1e-9)) | (fv_c >= ceil * (1.0 - 1e-9))
    cm_m = (fv_m <= floor * (1.0 + 1e-9)) | (fv_m >= ceil * (1.0 - 1e-9))
    fvz_c = np.where(cm_c, 0.0, fvz_c)
    fvz_m = np.where(cm_m, 0.0, fvz_m)
    # exact closed-form LA (algebraic-lapse) map: the tail and the backward-BC seed
    tau_la_arr = tau0 * ((1.0 + zc) * (1.0 - fv0) ** (1.0 / 3.0) * (2.0 + fv0)
                         / ((1.0 - fv_c) ** (1.0 / 3.0) * (2.0 + fv_c))) ** (-1.5)
    use_lb = (fv_c >= fv_switch) & (~cm_c)
    i_sw = int(np.max(np.nonzero(use_lb)[0])) if np.any(use_lb) else 0

    def _sig(z, tau, fv, fvz, cm, g0):
        if fv < fv_switch or cm:                      # LA lapse (stable linear ODE)
            return -1.5 * tau * (1.0 / (1.0 + z) + fvz / (3.0 * (1.0 - fv)) - fvz / (2.0 + fv))
        Phi = (1.0 + z) * g0 * (tau / tau0) ** (2.0 / 3.0) * ((1.0 - fv0) / (1.0 - fv)) ** (1.0 / 3.0)
        d = Phi - 1.0
        if abs(d) < 1e-12:
            return -1.5 * tau / (1.0 + z)
        return tau * fvz / (2.0 * (1.0 - fv) * d)

    def _integrate(g0):
        tau = tau_la_arr.copy()                       # tail (f_v < fv_switch) = exact LA
        if i_sw > 0:
            t = tau_la_arr[i_sw]                      # BC at the switch (exact LA)
            tau[i_sw] = t
            for i in range(i_sw, 0, -1):              # RK4 backward (attracting)
                h = z_grid[i - 1] - z_grid[i]
                k1 = _sig(zc[i], t, fv_c[i], fvz_c[i], cm_c[i], g0)
                k2 = _sig(zm[i - 1], t + 0.5 * h * k1, fv_m[i - 1], fvz_m[i - 1], cm_m[i - 1], g0)
                k3 = _sig(zm[i - 1], t + 0.5 * h * k2, fv_m[i - 1], fvz_m[i - 1], cm_m[i - 1], g0)
                k4 = _sig(zc[i - 1], t + h * k3, fv_c[i - 1], fvz_c[i - 1], cm_c[i - 1], g0)
                t = t + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                tau[i - 1] = t
        return tau

    def _r(g0):
        tau = _integrate(g0)
        if not np.all(np.isfinite(tau)) or tau[0] <= 0.0:
            return np.nan
        return tau[0] / tau0

    # --- shoot gamma_bar0 so tau(0)=tau0 (self-consistency Phi(0,tau0)=gamma_bar0) -----
    g0_la = (2.0 + fv0) / 2.0
    r0 = _r(g0_la)
    if not np.isfinite(r0):
        raise FloatingPointError("rate_ratio backward integration diverged")
    n_shoot = 1
    if abs(r0 - 1.0) < 1e-7:
        g0_star = g0_la                               # tracker / no LB region: no-op
    else:
        if r0 > 1.0:                                  # r decreases in gamma_bar0: root above
            hi = g0_la
            while _r(hi) > 1.0 and hi < 3.0:
                hi *= 1.03
                n_shoot += 1
            lo = hi / 1.03
        else:                                         # root below
            lo = g0_la
            while _r(lo) < 1.0 and lo > 0.5:
                lo /= 1.03
                n_shoot += 1
            hi = lo * 1.03
        if not (np.isfinite(_r(lo)) and np.isfinite(_r(hi)) and (_r(lo) - 1.0) * (_r(hi) - 1.0) <= 0.0):
            raise FloatingPointError("rate_ratio gamma_bar0 shot found no bracket")
        g0_star = brentq(lambda g: _r(g) - 1.0, lo, hi, xtol=1e-10, rtol=1e-13)
        n_shoot += 12

    tau = _integrate(g0_star)
    resid = abs(tau[0] / tau0 - 1.0)
    tau = tau * (tau0 / tau[0])                        # cleanup; ~no-op at g0_star

    # --- dressed observables on the z-grid ------------------------------------------
    z = z_grid
    fv = fv_c
    fvz = fvz_c
    t23 = tau ** (2.0 / 3.0)
    Phi = (1.0 + z) * g0_star * (tau / tau0) ** (2.0 / 3.0) * ((1.0 - fv0) / (1.0 - fv)) ** (1.0 / 3.0)
    use_lb2 = (fv >= fv_switch) & (~cm_c)
    gam = np.where(use_lb2, Phi, (2.0 + fv) / 2.0)
    d_safe = np.where(np.abs(Phi - 1.0) < 1e-30, 1e-30, Phi - 1.0)
    sig = np.where(use_lb2,
                   tau * fvz / (2.0 * (1.0 - fv) * d_safe),
                   -1.5 * tau * (1.0 / (1.0 + z) + fvz / (3.0 * (1.0 - fv)) - fvz / (2.0 + fv)))
    fvp = fvz / sig                                    # df_v/dtau (analytic, no map deriv)
    # d_A(z) = tau^{2/3} int_0^z (-sigma)/(gam tau'^{2/3}) dz'
    integ = (-sig) / (gam * t23)
    Jz = cumulative_trapezoid(integ, z, initial=0.0)
    dA = t23 * Jz
    DM = (1.0 + z) * dA
    # gamma_bar'(tau) = (dgam/dz)/sigma ; dgam/dz analytic (LB: from Phi; LA: fvz/2)
    dgam_dz = np.where(use_lb2,
                       Phi * (1.0 / (1.0 + z) + fvz / (3.0 * (1.0 - fv))) + Phi * (2.0 / (3.0 * tau)) * sig,
                       0.5 * fvz)
    gamp = dgam_dz / sig
    Hbar = 2.0 / (3.0 * tau) + fvp / (3.0 * (1.0 - fv))
    Hd = gam * Hbar - gamp
    DH = 1.0 / Hd

    sol = ModelVSolution(z, tau, fv, DM, DH, Hd, fv0, n_shoot, resid)
    sol.gamma_bar0 = float(g0_star)                    # self-consistent present lapse (LB)
    return sol


def modelv_solve(fv_of_z, *, lapse="algebraic", Ngrid=30000, tau0=None,
                 tau_lo_frac=1e-6, tol=1e-8, max_iter=100):
    """Solve the dressed geometry for a forced f_v(z) callable.

    fv_of_z : callable f_v(z); if it exposes .deriv(z) (a MonotoneFv), df_v/dtau is
        formed analytically in the z-direction (kinky at nodes taken analytically)
        and numerically in the smooth tau-direction; otherwise np.gradient(fv, tau).
    lapse   : "algebraic" (adopted) | "none" (V0 control, gamma_bar==1) |
        "rate_ratio" (LB systematic: gamma_bar = Hbar/H_w = 1 + tau f_v'/(2(1-f_v));
        the redshift map becomes f_v'-dependent and is solved in z by a backward RK4 with
        a self-consistent gamma_bar0 shot -- see `_solve_rate_ratio` / NOTES sec 3). For
        rate_ratio, Ngrid is the z-grid size and tau_lo_frac / tol / max_iter are unused.
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
    fv0 = float(fv_of_z(0.0))
    if tau0 is None:
        tau0 = (2.0 + fv0) / 3.0

    # rate-ratio (LB): f_v'-dependent lapse -> unstable tau-space fixed point; solved in
    # z with a backward RK4 + gamma_bar0 shot (self-consistent present lapse). Requires
    # an analytic df_v/dz (MonotoneFv.deriv).
    if lapse == "rate_ratio":
        if not hasattr(fv_of_z, "deriv"):
            raise ValueError("rate_ratio lapse needs a callable with .deriv(z) (MonotoneFv)")
        floor = float(getattr(fv_of_z, "floor", 0.0))
        ceil = float(getattr(fv_of_z, "ceil", 1.0))
        return _solve_rate_ratio(fv_of_z, fv0, floor, ceil, Ngrid=Ngrid, tau0=tau0)

    # algebraic / none: gamma_bar is a pure function of f_v, so the redshift map is a
    # stable tau-space fixed point.
    tau = _tau_grid(tau0, tau_lo_frac, Ngrid)
    t23 = tau ** (2.0 / 3.0)
    has_deriv = hasattr(fv_of_z, "deriv")

    def _fvp(fv, z):
        # df_v/dtau = (df_v/dz)(dz/dtau); analytic df_v/dz (PCHIP) when available.
        if has_deriv:
            return fv_of_z.deriv(z) * np.gradient(z, tau)
        return np.gradient(fv, tau)

    z = (tau0 / tau) ** (2.0 / 3.0) - 1.0        # EdS-like initial guess
    dz = np.inf
    it = 0
    for it in range(1, int(max_iter) + 1):
        fv = fv_of_z(z)
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
    fvp = _fvp(fv, z)                          # df_v/dtau (for the Hubble rate)
    gam = _lapse_gamma(fv, lapse)
    gamp = fvp / 2.0 if lapse == "algebraic" else np.zeros_like(fv)   # dgamma_bar/dtau

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
