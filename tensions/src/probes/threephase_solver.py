#!/usr/bin/env python3
"""WP-B Stage-1 Phase-A: dynamically-consistent THREE-PHASE Buchert solver.

Three genuine Buchert phases sharing one volume-average (bare) time t:
  walls  w : flat dust             a_w propto t^{2/3},  H_w = 2/(3t)   (<R>_w = 0)
  shallow s: dust + neg curvature  constant alpha_s^2                  (a Buchert void phase)
  deep    d: dust + neg curvature  constant alpha_d^2                  (a Buchert void phase)
Each void phase is an exact homogeneous-isotropic (FLRW) dust+curvature patch with a
CONSTANT spatial-curvature parameter -- the property an exact multi-phase Buchert solution
requires, and exactly what the paper-2 forced kinematic history (81% alpha^2 drift) fails.

DYNAMICS (Wiegand-Buchert 2010 arXiv:1002.3912 multiscale partition; Duley-Nazer-Wiltshire
2013 arXiv:1306.3208 two-scale template -- see NOTES_threephase.md for the equation-by-
equation derivation):

  Eq (WB/ DNW 8, generalised):  abar^3 = f_wi a_w^3 + f_si a_s^3 + f_di a_d^3
  Eq (DNW 9):                   f_i(t) = f_ii a_i^3 / abar^3 ,  sum_i f_i = 1
  Eq (DNW 14):                  Hbar   = sum_i f_i H_i          (H_i = d ln a_i/dt)
  multiscale backreaction (WB): Q = 6 sum_i f_i (H_i - Hbar)^2  (each phase homogeneous,
                                    Q_i = 0), and the GLOBAL Hamiltonian constraint
                                    3 Hbar^2 = 8 pi G <rho> - <R>/2 - Q/2 then holds as an
                                    IDENTITY of the partition (NOTES sec 2) -- this is the
                                    dynamical feedback that makes the phase split observable.

DERIVED LAPSE (the theory task; NOT the algebraic (2+f_v)/2 ansatz):
  Eq (DNW 16):  gamma_bar = 1 + f_v (1 - h_r)/h_r = Hbar / H_w ,   h_r = H_w/H_v
  generalised to three phases as the WALL clock lapse gamma_bar = Hbar/H_w = (sum_i f_i H_i)/H_w
  (the wall observer's clock relative to the volume-average observer; walls are the flat-dust
  phase in which observers sit). On the empty-void tracker gamma_bar -> (2+f_v)/2 (gate G3).
  Because Hbar = sum f_i H_i is read straight off the phase Hubble rates, no fragile
  derivative of a forced f_v is needed (contrast paper-2's rate-ratio LB path).

DRESSED OBSERVABLES (Wiltshire 2009 arXiv:0909.0749; the same construction paper-2 validated):
  redshift  1+z = gamma_bar abar0/(gamma_bar0 abar)   (DNW redshift; abar propto t^{2/3}(1-f_v)^{-1/3})
  distance  d_A(z) = a_w(t_e) int_{t_e}^{t0} dt/(gamma_bar a_w) ,  D_M = (1+z) d_A   (wall ruler a_w propto t^{2/3})
  dressed H = gamma_bar Hbar - dgamma_bar/dt ,  D_H = 1/H         (computed INDEPENDENTLY of dD_M/dz)

Because a_w propto t^{2/3} exactly (flat dust) and the volume closure gives
abar^3 propto t^2/(1-f_v), the bare abar, Hbar, gamma_bar and the distances depend on the phase
split ONLY through the total f_v(t) it PRODUCES -- but a genuine (integrable) three-phase
dynamics produces a total f_v(t) that a single-void dynamics cannot, which is how the split
becomes observable (NOTES sec 3). The solver therefore evolves the void phases, forms
f_v(t) = f_s(t)+f_d(t), and runs the dressed machinery on it.

GATES (run: python src/probes/threephase_solver.py -> probes_out/threephase_gates.json):
  G1 tracker limit   : two-phase EMPTY-void limit reproduces the paper-2 tracker oracle
                       (SN chi2 = 1391.545 +- 0.01; D_M matches to <1e-6; dD_M/dz != D_H).
  G2 integrability   : recovered alpha_s^2(t), alpha_d^2(t) constant to <= 1e-6 fractional.
  G3 lapse limit     : derived gamma_bar -> (2+f_v)/2 on the tracker to machine precision.
  G4 numerics        : D_M(z), Hbar(z) converge under step-halving to < 1e-4.

Portable __file__ paths; each homogeneous phase is solved in closed form (walls: a_w propto
tau^{2/3}; voids: exact common-bang open-dust FLRW), so there is no unbounded inner loop; a
checkpoint (save/load of the solved grid) and a wall-clock soft-limit are provided.
"""
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# portable paths: this repo (paper 3) + the paper-2 sibling checkout (oracle/harness)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))          # free-history-timescape-tensions
_PARENT = os.path.dirname(_REPO)
_P2 = os.path.join(_PARENT)             # paper-2 sibling
_P2_SRC = os.path.join(_P2, "src")
_OUT_JSON = os.path.join(_REPO, "probes_out", "threephase_gates.json")
_CKPT_DIR = os.path.join(_REPO, "probes_out", "_threephase_ckpt")

# soft wall-clock limit for any inner loop (seconds); overridable via env
_WALL_SOFT_S = float(os.environ.get("THREEPHASE_WALL_SOFT_S", "300"))

_FV_FLOOR = 1e-12
_FV_CEIL = 1.0 - 1e-12


def _json_default(o):
    """JSON encoder for numpy scalars/arrays."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


# ---------------------------------------------------------------------------
# model configuration
# ---------------------------------------------------------------------------
class VoidPhase:
    """One dust + constant-negative-curvature Buchert void phase (a homogeneous FLRW patch)
    with a COMMON BIG BANG at t=0 (a_j -> 0 as t -> 0, hence f_j -> 0 early: DNW13's f_vi << 1,
    voids negligible at last scattering).

    Friedmann first integral (bare time, H_j = d ln a_j/dt; present normalisation a_j0 = 1):
        H_j^2 = Om_j a_j^{-3} + alpha2_j a_j^{-2}          (Om_j >= 0, alpha2_j > 0 open)
    The CONSTANT curvature parameter alpha2_j (proportional to DNW13's alpha^2 = -k_v f_vi^{2/3})
    is the phase's single dynamical free constant; the common bang at t=0 with a_j0 = 1 then
    FIXES the matter content Om_j given (alpha2_j, tau0):

        empty/Milne   alpha2_j = 1/tau0^2  =>  Om_j = 0,  a_j = tau/tau0
                      (the deepest, youngest, maximal-contrast void = the mechanism's CEILING);
        matter void   alpha2_j < 1/tau0^2  =>  Om_j > 0 derived, exact open-dust parametric
                      solution (slower, EdS->Milne transition later -> a genuinely DISTINCT
                      history).

    Two empty (Milne) voids are homothetic (both a_v propto tau), collapsing to the two-scale
    tracker; a genuine three-phase split therefore REQUIRES matter differentiation between the
    phases (deep emptier/faster, shallow more matter/slower). See NOTES_threephase.md sec 5.

    f0 = present (t0) volume fraction of this phase.  Om is derived and stored at solve time.
    """

    def __init__(self, f0, alpha2, name="void"):
        self.f0 = float(f0)
        self.alpha2 = float(alpha2)
        self.name = str(name)
        self.Om = None                                       # derived from (alpha2, tau0) at solve

    def profile(self, tau0, tau):
        """(a_j(tau), h_j(tau)=d ln a_j/dtau, Om_j): exact common-bang open-dust FLRW, a_j0=1."""
        a, h, Om = _void_profile(self.alpha2, tau0, tau)
        self.Om = float(Om)
        return a, h, Om


def _void_profile(alpha2, tau0, tau):
    """Exact closed-form common-bang (t=0) dust+negative-curvature FLRW patch, a(tau0)=1.

    empty Milne (alpha2 = 1/tau0^2): a = tau/tau0, H = 1/tau, Om = 0.
    matter void  (alpha2 < 1/tau0^2): standard open-dust parametric solution with development
        angle eta:  a(eta) = (Om/(2 alpha2))(cosh eta - 1),
                    tau(eta) = (Om/(2 alpha2^{3/2}))(sinh eta - eta),  bang at eta=0 (tau=0).
        Present a=1, tau=tau0 fix eta0 from  tau0 sqrt(alpha2) = (sinh eta0 - eta0)/(cosh eta0 - 1)
        and then Om = 2 alpha2/(cosh eta0 - 1).  tau(eta) is inverted for eta(tau) on a fine
        eta grid (a smooth monotone map -> converges fast).
    """
    ak = 1.0 / tau0 ** 2
    if alpha2 > ak * (1.0 + 1e-10):
        raise ValueError(f"alpha2={alpha2} exceeds common-bang ceiling 1/tau0^2={ak}")
    if alpha2 >= ak * (1.0 - 1e-12):                          # empty Milne
        a = tau / tau0
        return a, 1.0 / tau, 0.0
    from scipy.optimize import brentq
    s = tau0 * np.sqrt(alpha2)                                # in (0,1); age relation RHS target

    def _rhs(eta):
        if eta < 1e-3:                                       # series: avoid cosh(eta)-1 cancellation
            return eta / 3.0 - s
        return (np.sinh(eta) - eta) / (np.cosh(eta) - 1.0) - s

    hi = 1.0
    _t0 = time.time()
    while _rhs(hi) < 0.0 and hi < 1e4:                        # bounded bracket search (soft-limited)
        hi *= 1.5
        if time.time() - _t0 > _WALL_SOFT_S:
            raise TimeoutError(f"_void_profile bracket search exceeded soft limit {_WALL_SOFT_S}s")
    eta0 = brentq(_rhs, 1e-3, hi, xtol=1e-14, rtol=1e-15)
    Om = 2.0 * alpha2 / (np.cosh(eta0) - 1.0)
    C_a = Om / (2.0 * alpha2)
    C_t = Om / (2.0 * alpha2 ** 1.5)
    eta_g = np.linspace(0.0, eta0, 800000)
    tau_g = C_t * (np.sinh(eta_g) - eta_g)                    # monotone; tau_g[0]=0
    tau_g[-1] = tau0                                          # pin present endpoint exactly
    eta = np.interp(np.asarray(tau, dtype=float), tau_g, eta_g)
    a = np.maximum(C_a * (np.cosh(eta) - 1.0), 1e-30)
    h = np.sqrt(Om * a ** -3 + alpha2 * a ** -2)              # H_j = d ln a_j/dtau
    return a, h, Om


def recover_alpha2(phase, tau0, tau_lo, tau_hi, n=2000, delta_frac=1e-4):
    """Integrability self-check (gate G2): recover the phase's curvature parameter
        alpha_j^2 = a_j^2 (d ln a_j/dtau)^2 - Om_j/a_j
    from the SOLVED a_j(tau), with d ln a_j/dtau by an INDEPENDENT central difference (step
    delta = delta_frac*tau0, a_j evaluated from the closed-form solution) -- a fixed near-optimal
    step decouples the derivative accuracy from grid roundoff.  For a genuine constant-curvature
    Buchert phase this is constant along the solution; the fractional spread quantifies the
    dynamical consistency (contrast the FORCED kinematic history's ~81 pct alpha^2 drift)."""
    tau_u = np.linspace(tau_lo, tau_hi, int(n))
    d = delta_frac * tau0
    a, _h, Om = _void_profile(phase.alpha2, tau0, tau_u)
    ap, _, _ = _void_profile(phase.alpha2, tau0, tau_u + d)
    am, _, _ = _void_profile(phase.alpha2, tau0, tau_u - d)
    h_num = (np.log(ap) - np.log(am)) / (2.0 * d)            # d ln a/dtau (central difference)
    a2 = a ** 2 * h_num ** 2 - Om / a
    return tau_u, a2, float(Om)


class ThreePhaseConfig:
    """Free constants of the three-phase model.

    walls: flat dust (a_w = (t/t0)^{2/3}); its fraction f_w0 = 1 - f_s0 - f_d0 is derived.
    shallow, deep: VoidPhase instances.  tau0 = present bare time (H_w0 = 2/(3 tau0)); the
    distance SHAPE is invariant under tau -> lambda tau (paper-2 NOTES sec 6.1), so tau0 only
    sets the scale absorbed by the SN offset / BAO alpha.  Default tau0 = (2 + f_v0)/3 (the
    tracker value) for a clean present-day normalisation.
    """

    def __init__(self, shallow, deep, tau0=None):
        self.shallow = shallow
        self.deep = deep
        self.f_v0 = shallow.f0 + deep.f0
        self.f_w0 = 1.0 - self.f_v0
        if not (0.0 < self.f_w0 < 1.0):
            raise ValueError(f"wall fraction f_w0={self.f_w0} out of (0,1)")
        self.tau0 = (2.0 + self.f_v0) / 3.0 if tau0 is None else float(tau0)


def tracker_config(f_v0):
    """The two-phase EMPTY-void limit: deep+shallow collapsed into ONE empty (Milne) void
    phase, walls flat dust.  This is the exact Wiltshire tracker (NOTES sec 4): the empty
    void has alpha^2 = 1/tau0^2 (common big bang, a_v propto tau) and tau0 = (2+f_v0)/3.
    The 'deep' phase carries the whole void fraction; 'shallow' is a null placeholder.
    """
    tau0 = (2.0 + f_v0) / 3.0
    deep = VoidPhase(f0=f_v0, alpha2=1.0 / tau0 ** 2, name="void(tracker)")
    shallow = VoidPhase(f0=0.0, alpha2=1.0 / tau0 ** 2, name="null")
    return ThreePhaseConfig(shallow, deep, tau0=tau0)


# ---------------------------------------------------------------------------
# tau grid
# ---------------------------------------------------------------------------
def _tau_grid(tau0, tau_lo_frac, Ngrid):
    """Hybrid grid dense at BOTH ends (paper-2 NOTES sec 6.2): linspace (dense near tau0,
    for the small difference-of-integrals at low z) UNION geomspace (dense near tau_lo, for
    high-z / CMB placement)."""
    tlo = tau_lo_frac * tau0
    return np.unique(np.concatenate([
        np.linspace(tlo, tau0, int(Ngrid)),
        np.geomspace(tlo, tau0, int(Ngrid)),
    ]))


# ---------------------------------------------------------------------------
# solution container
# ---------------------------------------------------------------------------
class ThreePhaseSolution:
    """Solved dressed geometry of a three-phase config.  Arrays are z-ASCENDING for
    interpolation.  Query: D_M(z), D_H(z), D_V(z), predict(z, kind in {DM,DH,DV})."""

    _ARR = ("z", "tau", "fv", "fs", "fd", "abar", "Hbar", "gamma", "DM", "DH", "Hd")

    def __init__(self, **kw):
        order = np.argsort(kw["z"])
        for k in self._ARR:
            setattr(self, k, np.asarray(kw[k])[order])
        self.f_v0 = float(kw["f_v0"])
        self.gamma0 = float(kw["gamma0"])
        self.tau0 = float(kw["tau0"])
        self.alpha2_s = np.asarray(kw["alpha2_s"])[order]
        self.alpha2_d = np.asarray(kw["alpha2_d"])[order]
        self.selfcons_maxfrac = float(kw["selfcons_maxfrac"])
        self.Om_s = float(kw["Om_s"])
        self.Om_d = float(kw["Om_d"])
        self.Ngrid = int(kw["Ngrid"])

    def D_M(self, z):
        return np.interp(np.asarray(z, dtype=float), self.z, self.DM)

    def D_H(self, z):
        return np.interp(np.asarray(z, dtype=float), self.z, self.DH)

    def D_V(self, z):
        z = np.asarray(z, dtype=float)
        dM = self.D_M(z)
        return (z * dM * dM * self.D_H(z)) ** (1.0 / 3.0)

    def predict(self, z, kind):
        if kind == "DM":
            return self.D_M(z)
        if kind == "DH":
            return self.D_H(z)
        if kind == "DV":
            return self.D_V(z)
        raise ValueError(f"unknown kind {kind!r}")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **{k: getattr(self, k) for k in self._ARR},
                 alpha2_s=self.alpha2_s, alpha2_d=self.alpha2_d,
                 f_v0=self.f_v0, gamma0=self.gamma0, tau0=self.tau0,
                 selfcons_maxfrac=self.selfcons_maxfrac, Om_s=self.Om_s, Om_d=self.Om_d,
                 Ngrid=self.Ngrid)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(**{k: d[k] for k in d.files})


# ---------------------------------------------------------------------------
# core solve
# ---------------------------------------------------------------------------
def solve(config, Ngrid=30000, tau_lo_frac=1e-6, checkpoint=None):
    """Solve the dressed three-phase geometry.

    Returns a ThreePhaseSolution.  If `checkpoint` is a path and exists, it is loaded
    (skips the integration); otherwise the solution is written there after solving.
    """
    if checkpoint and os.path.exists(checkpoint):
        return ThreePhaseSolution.load(checkpoint)

    tau0 = config.tau0
    tau = _tau_grid(tau0, tau_lo_frac, Ngrid)
    fw0, fs0, fd0 = config.f_w0, config.shallow.f0, config.deep.f0

    # walls: flat dust, analytic (no integration needed -- a_w propto tau^{2/3} exactly)
    a_w = (tau / tau0) ** (2.0 / 3.0)
    h_w = 2.0 / (3.0 * tau)                       # H_w = d ln a_w/dt (bare time, 1/tau units)

    # void phases: exact common-bang open-dust FLRW patches (a_j -> 0 as tau -> 0)
    if fs0 > 0:
        a_s, h_s, Om_s = config.shallow.profile(tau0, tau)
    else:
        a_s, h_s, Om_s = np.zeros_like(tau), np.zeros_like(tau), 0.0
    if fd0 > 0:
        a_d, h_d, Om_d = config.deep.profile(tau0, tau)
    else:
        a_d, h_d, Om_d = np.zeros_like(tau), np.zeros_like(tau), 0.0

    # global volume closure (Eq 8) + fractions (Eq 9) + bare Hubble (Eq 14)
    abar3 = fw0 * a_w ** 3 + fs0 * a_s ** 3 + fd0 * a_d ** 3
    abar = abar3 ** (1.0 / 3.0)
    fv = np.clip((fs0 * a_s ** 3 + fd0 * a_d ** 3) / abar3, _FV_FLOOR, _FV_CEIL)
    fs = fs0 * a_s ** 3 / abar3
    fd = fd0 * a_d ** 3 / abar3
    fw = 1.0 - fv
    Hbar = fw * h_w + fs * h_s + fd * h_d          # Eq 14

    # DERIVED lapse (Eq 16 generalised): gamma_bar = Hbar/H_w (wall clock)
    gamma = Hbar / h_w
    abar0 = abar[-1]
    gamma0 = gamma[-1]

    # self-consistency: Hbar (=sum f_i H_i) must equal the volume identity D1
    #   Hbar_D1 = 2/(3 tau) + fv'/(3(1-fv)) since abar^3 propto tau^2/(1-fv) (a_w flat dust).
    #   fv' via a C2 spline (clean on the non-uniform grid; np.gradient of it is noisy).
    fv_cb = CubicSpline(tau, fv)
    Hbar_D1 = 2.0 / (3.0 * tau) + fv_cb(tau, 1) / (3.0 * (1.0 - fv))
    m_in = (tau > 30.0 * tau_lo_frac * tau0) & (tau < tau0 * (1 - 1e-4))
    selfcons_maxfrac = float(np.max(np.abs(Hbar[m_in] / Hbar_D1[m_in] - 1.0)))

    # dressed redshift 1+z = gamma_bar abar0/(gamma_bar0 abar)
    onepz = gamma * abar0 / (gamma0 * abar)
    z = onepz - 1.0

    # dressed distance d_A = a_w(tau_e) int_{tau_e}^{tau0} dtau/(gamma_bar a_w); D_M=(1+z)d_A
    integrand = 1.0 / (gamma * a_w)
    J = cumulative_trapezoid(integrand, tau, initial=0.0)
    dA = a_w * (J[-1] - J)
    DM = (1.0 + z) * dA

    # dressed Hubble H = gamma_bar Hbar - dgamma_bar/dt  (D_H = 1/H, INDEPENDENT of dD_M/dz)
    gamma_prime = np.gradient(gamma, tau, edge_order=2)          # dgamma_bar/dtau
    Hd = gamma * Hbar - gamma_prime                              # (Hbar0 units)
    DH = 1.0 / Hd

    # recovered per-phase constant curvature parameter alpha_j^2(tau) (gate G2):
    #   alpha_j^2 = a_j^2 H_j^2 - Om_j/a_j, with H_j = d ln a_j/dtau recovered NUMERICALLY
    #   (C2 spline derivative) from the SOLVED a_j(tau) -- the integrability self-check that
    #   the dynamical solution is a genuine constant-curvature Buchert phase (contrast the
    #   forced kinematic history: 81% alpha^2 drift, paper-2 NOTES sec 8 / wpb_integrability).
    def _alpha2_recovered(a_j, Om_j, f0):
        if f0 <= 0:
            return np.full_like(tau, np.nan)
        good = a_j > 1e-12
        aj = np.where(good, a_j, 1.0)
        hj = CubicSpline(tau, np.log(aj))(tau, 1)             # d ln a_j/dtau (numeric, C2)
        out = aj ** 2 * hj ** 2 - Om_j / aj
        return np.where(good, out, np.nan)

    alpha2_s = _alpha2_recovered(a_s, Om_s, fs0)
    alpha2_d = _alpha2_recovered(a_d, Om_d, fd0)

    sol = ThreePhaseSolution(
        z=z, tau=tau, fv=fv, fs=fs, fd=fd, abar=abar, Hbar=Hbar, gamma=gamma,
        DM=DM, DH=DH, Hd=Hd, f_v0=config.f_v0, gamma0=gamma0, tau0=tau0,
        alpha2_s=alpha2_s, alpha2_d=alpha2_d, selfcons_maxfrac=selfcons_maxfrac,
        Om_s=Om_s, Om_d=Om_d, Ngrid=Ngrid)
    if checkpoint:
        sol.save(checkpoint)
    return sol


# ---------------------------------------------------------------------------
# paper-2 oracle / harness (imported lazily; they print a fit on import -> silence)
# ---------------------------------------------------------------------------
def _load_p2():
    if _P2_SRC not in sys.path:
        sys.path.insert(0, _P2_SRC)
    cwd = os.getcwd()
    os.chdir(_P2_SRC)
    try:
        import fit_timescape as F
        with contextlib.redirect_stdout(io.StringIO()):
            import timescape_baocmb as T
            import harness as HN
    finally:
        os.chdir(cwd)
    return F, T, HN


# ---------------------------------------------------------------------------
# GATES
# ---------------------------------------------------------------------------
_GA_TARGETS = {0.01: 0.0026, 0.1: 0.0249, 0.5: 0.0982, 1.0: 0.1488, 2.0: 0.1752}
_SN_REF = 1391.545176


def gate_G1(F, T, HN):
    """G1 (LOAD-BEARING): two-phase EMPTY-void limit reproduces the paper-2 tracker oracle."""
    FV0 = 0.853
    cwd = os.getcwd(); os.chdir(_P2_SRC)
    try:
        sol = solve(tracker_config(FV0), Ngrid=200000)
        # (a) distances vs the accurate brentq oracle over z in [1e-3, 1100]
        zq = np.geomspace(1e-3, 1100.0, 600)
        dm_oracle = np.array([T.DM(z, FV0) for z in zq])
        relDM = np.abs(sol.D_M(zq) / dm_oracle - 1.0)
        max_relDM = float(relDM.max()); zmax = float(zq[relDM.argmax()])
        # (b) SN full-covariance chi2 at the tracker fv0=0.853
        zHD, zHEL, mb, Cf = F.load()
        chi2 = F.make_chi2(zHD, zHEL, mb, Cf)
        c_general = float(chi2(sol.D_M(zHD)))
        c_harness = float(HN.sn_chi2(sol.D_M(zHD)))
        # (c) non-FLRW audit signature dD_M/dz / D_H - 1 (D_M, D_H computed INDEPENDENTLY)
        ga = {}
        for z, tgt in _GA_TARGETS.items():
            h = 1e-4 * (1.0 + z)
            dDMdz = (float(sol.D_M(z + h)) - float(sol.D_M(z - h))) / (2.0 * h)
            ga[z] = dDMdz / float(sol.D_H(z)) - 1.0
        ga_max = max(abs(ga[z] - _GA_TARGETS[z]) for z in _GA_TARGETS)
        ga_nonflrw = all(ga[z] > 1e-3 for z in _GA_TARGETS)      # genuinely != D_H
        # cross-check at a second fv0 (the committed BAO+CMB best fit) via the joint predict path
        solj = solve(tracker_config(0.6426), Ngrid=200000)
        cj = float(HN.sn_chi2(solj.D_M(zHD)))
        cbcj, aj = HN.bao_cmb_chi2(lambda z, k: float(solj.predict(z, k)))
        joint = cj + cbcj
    finally:
        os.chdir(cwd)
    a_pass = max_relDM < 1e-6
    b_pass = abs(c_general - _SN_REF) < 0.01 and abs(c_harness - _SN_REF) < 0.01
    c_pass = ga_max < 0.002 and ga_nonflrw
    j_pass = abs(joint - 1469.2926) < 0.5                        # loose: validates full predict path
    passed = a_pass and b_pass and c_pass and j_pass
    return passed, {
        "max_relDM_over_1e-3_1100": max_relDM, "relDM_argmax_z": zmax, "(a)_pass": a_pass,
        "SN_chi2_make_chi2": c_general, "SN_chi2_harness": c_harness, "SN_chi2_ref": _SN_REF,
        "SN_chi2_absdiff": abs(c_general - _SN_REF), "(b)_pass": b_pass,
        "GA_ratios": {str(z): ga[z] for z in _GA_TARGETS},
        "GA_targets": {str(z): _GA_TARGETS[z] for z in _GA_TARGETS},
        "GA_max_absdiff": ga_max, "dDMdz_ne_DH_nonFLRW": ga_nonflrw, "(c)_pass": c_pass,
        "tracker_joint_fv0_0.6426": joint, "joint_ref": 1469.2926, "(joint)_pass": j_pass,
        "note": "empty-void two-scale = exact Wiltshire tracker (walls a_w~t^{2/3}, void a_v~t Milne)",
    }


def gate_G2():
    """G2 (integrability): recovered alpha_s^2(t), alpha_d^2(t) constant along the solution
    to <= 1e-6 fractional -- the dynamical three-phase solution is a genuine constant-curvature
    Buchert configuration (contrast the FORCED kinematic history, ~81 pct drift, paper-2 sec 8).
    Two genuine three-phase configs (matter-differentiated: deep near-empty/fast, shallow more
    matter/slow).  tau0=(2+f_v0)/3 with f_v0=0.64 gives the common-bang ceiling 1/tau0^2."""
    # observable-range window (avoid the tau_lo boundary where the numeric d/dtau is noisy)
    ZLO, ZHI = 0.05, 2.4
    ak = 1.0 / ((2 + 0.64) / 3) ** 2                         # common-bang ceiling for f_v0=0.64

    def _tau_window(sol):
        return float(np.interp(ZHI, sol.z, sol.tau)), float(np.interp(ZLO, sol.z, sol.tau))

    def _drift(a2):
        a = np.abs(a2)
        a = a[np.isfinite(a) & (a > 0)]
        return float(np.max(a) / np.min(a) - 1.0)

    def _eval(name, alpha2_s, alpha2_d):
        cfg = ThreePhaseConfig(VoidPhase(0.36, alpha2_s, "shallow"),
                               VoidPhase(0.28, alpha2_d, "deep"))
        sol = solve(cfg, Ngrid=60000)
        tlo, thi = _tau_window(sol)
        _, a2s, Om_s = recover_alpha2(cfg.shallow, cfg.tau0, tlo, thi)
        _, a2d, Om_d = recover_alpha2(cfg.deep, cfg.tau0, tlo, thi)
        ds, dd = _drift(a2s), _drift(a2d)
        return sol, {"alpha2_shallow": alpha2_s, "alpha2_deep": alpha2_d,
                     "Om_shallow": Om_s, "Om_deep": Om_d,
                     "alpha2_shallow_recovered_mean": float(np.nanmean(a2s)),
                     "alpha2_deep_recovered_mean": float(np.nanmean(a2d)),
                     "drift_alpha2_shallow": ds, "drift_alpha2_deep": dd,
                     "fv0": sol.f_v0, "selfcons_Hbar_vs_D1_maxfrac": sol.selfcons_maxfrac}, [ds, dd]

    out = {}
    passes = []
    # config A: strongly matter-differentiated (deep near-empty alpha^2=1.20, shallow alpha^2=0.70)
    _, out["matter_differentiated_3phase"], p = _eval("A", 0.70, 1.20)
    out["matter_differentiated_3phase"]["common_bang_ceiling_1/tau0^2"] = ak
    passes += [x <= 1e-6 for x in p]
    # config B: deep EMPTY Milne (alpha^2 = ceiling, Om=0) + shallow matter (alpha^2 = 1.10)
    _, out["deep_empty_shallow_matter_3phase"], p = _eval("B", 1.10, ak)
    passes += [x <= 1e-6 for x in p]
    return all(passes), out


def gate_G3():
    """G3 (lapse limit): derived gamma_bar = Hbar/H_w reduces to (2+f_v)/2 on the tracker."""
    res = {}
    passed = True
    for fv0 in (0.853, 0.6426, 0.5):
        sol = solve(tracker_config(fv0), Ngrid=60000)
        gam_alg = (2.0 + sol.fv) / 2.0
        m = (sol.z >= 0.0) & (sol.z <= 1100.0)
        maxfrac = float(np.max(np.abs(sol.gamma[m] / gam_alg[m] - 1.0)))
        gam0_alg = (2.0 + fv0) / 2.0
        res[f"fv0={fv0}"] = {"max_frac_gamma_vs_(2+fv)/2": maxfrac,
                             "gamma0": sol.gamma0, "(2+fv0)/2": gam0_alg,
                             "gamma0_absdiff": abs(sol.gamma0 - gam0_alg)}
        # machine precision: full-range (z up to 1100) accumulated float error over a 1e6 grid
        passed = passed and (maxfrac < 1e-10) and (abs(sol.gamma0 - gam0_alg) < 1e-13)
    return passed, res


def gate_G4():
    """G4 (numerics): D_M(z), Hbar(z) converge under step-halving to < 1e-4.  Genuine
    matter-differentiated three-phase config (deep near-empty, shallow more matter)."""
    cfg = ThreePhaseConfig(VoidPhase(0.36, 0.70, "shallow"), VoidPhase(0.28, 1.20, "deep"))
    zc = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 2.0, 2.33])
    s1 = solve(cfg, Ngrid=30000)
    s2 = solve(cfg, Ngrid=60000)                     # step halved (grid doubled)
    dDM = np.abs(s2.D_M(zc) / s1.D_M(zc) - 1.0)
    # Hbar(z): interpolate bare Hubble onto z
    def Hbar_of(s, zq):
        return np.interp(zq, s.z, s.Hbar)
    dHb = np.abs(Hbar_of(s2, zc) / Hbar_of(s1, zc) - 1.0)
    maxDM, maxHb = float(dDM.max()), float(dHb.max())
    passed = maxDM < 1e-4 and maxHb < 1e-4
    return passed, {"z": zc.tolist(),
                    "max_rel_dDM_stephalving": maxDM,
                    "max_rel_dHbar_stephalving": maxHb,
                    "Ngrid_coarse": 30000, "Ngrid_fine": 60000,
                    "dDM_per_z": dDM.tolist(), "dHbar_per_z": dHb.tolist()}


def run_gates():
    t0 = time.time()
    F, T, HN = _load_p2()
    g1_pass, g1 = gate_G1(F, T, HN)
    g2_pass, g2 = gate_G2()
    g3_pass, g3 = gate_G3()
    g4_pass, g4 = gate_G4()
    all_pass = g1_pass and g2_pass and g3_pass and g4_pass

    # solver key outputs at a few z (tracker fv0=0.853 + a representative three-phase)
    sol_trk = solve(tracker_config(0.853), Ngrid=60000)
    cfg_rep = ThreePhaseConfig(VoidPhase(0.36, 0.70, "shallow"),
                               VoidPhase(0.28, 1.20, "deep"))
    sol_rep = solve(cfg_rep, Ngrid=60000)
    zshow = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    key_outputs = {
        "tracker_fv0_0.853": {
            "z": zshow,
            "D_M": [float(sol_trk.D_M(z)) for z in zshow],
            "D_H": [float(sol_trk.D_H(z)) for z in zshow],
            "Hbar_over_Hbar0": [float(np.interp(z, sol_trk.z, sol_trk.Hbar)) for z in zshow],
            "gamma_bar": [float(np.interp(z, sol_trk.z, sol_trk.gamma)) for z in zshow],
            "fv": [float(np.interp(z, sol_trk.z, sol_trk.fv)) for z in zshow],
        },
        "representative_3phase": {
            "config": {"f_s0": 0.36, "alpha2_s": 0.70, "Om_s": float(sol_rep.Om_s),
                       "f_d0": 0.28, "alpha2_d": 1.20, "Om_d": float(sol_rep.Om_d),
                       "f_w0": 1.0 - 0.36 - 0.28},
            "z": zshow,
            "D_M": [float(sol_rep.D_M(z)) for z in zshow],
            "D_H": [float(sol_rep.D_H(z)) for z in zshow],
            "Hbar_over_Hbar0": [float(np.interp(z, sol_rep.z, sol_rep.Hbar)) for z in zshow],
            "gamma_bar": [float(np.interp(z, sol_rep.z, sol_rep.gamma)) for z in zshow],
            "fv": [float(np.interp(z, sol_rep.z, sol_rep.fv)) for z in zshow],
            "fs": [float(np.interp(z, sol_rep.z, sol_rep.fs)) for z in zshow],
            "fd": [float(np.interp(z, sol_rep.z, sol_rep.fd)) for z in zshow],
        },
    }

    out = {
        "probe": "WP-B Stage-1 Phase-A: dynamically-consistent three-phase Buchert solver gates",
        "spec": "PLAN_WPB_threephase.md sec 3 (G1-G4)",
        "lapse_path": "DERIVED (DNW13 Eq 16 gamma_bar=Hbar/H_w generalised to the three-phase "
                      "wall clock); NOT the algebraic (2+f_v)/2 ansatz. See NOTES_threephase.md.",
        "sources": {
            "multiscale_partition_and_backreaction": "Wiegand & Buchert 2010 (arXiv:1002.3912)",
            "two_scale_template_and_lapse": "Duley, Nazer & Wiltshire 2013 (arXiv:1306.3208), "
                                            "Eqs 8, 9, 10-11, 14, 16, 24",
            "dressed_distance_construction": "Wiltshire 2009 (arXiv:0909.0749); paper-2 "
                                             "modelv_theory (oracle it must reproduce)",
        },
        "gates": {
            "G1_tracker_limit": {"PASS": bool(g1_pass), **g1},
            "G2_integrability": {"PASS": bool(g2_pass), **g2},
            "G3_lapse_limit": {"PASS": bool(g3_pass), **g3},
            "G4_numerics": {"PASS": bool(g4_pass), **g4},
        },
        "ALL_GATES_PASS": bool(all_pass),
        "phase_B_verdict": "GO" if all_pass else "NO-GO",
        "key_outputs": key_outputs,
        "runtime_s": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(_OUT_JSON), exist_ok=True)
    with open(_OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)

    # console report
    print("=" * 74)
    print("THREE-PHASE BUCHERT SOLVER -- Phase-A gates")
    print("=" * 74)
    print(f"  lapse path: DERIVED (DNW13 Eq 16, gamma_bar = Hbar/H_w)")
    print(f"  G1 tracker limit : {'PASS' if g1_pass else 'FAIL'}   "
          f"SN chi2={g1['SN_chi2_harness']:.6f} (ref {_SN_REF}) |d|={g1['SN_chi2_absdiff']:.2e}  "
          f"max|D_M/oracle-1|={g1['max_relDM_over_1e-3_1100']:.2e}")
    _a = g2["matter_differentiated_3phase"]
    _b = g2["deep_empty_shallow_matter_3phase"]
    print(f"  G2 integrability : {'PASS' if g2_pass else 'FAIL'}   "
          f"cfgA drift(s,d)=({_a['drift_alpha2_shallow']:.2e},{_a['drift_alpha2_deep']:.2e})  "
          f"cfgB drift(s,d)=({_b['drift_alpha2_shallow']:.2e},{_b['drift_alpha2_deep']:.2e})  "
          f"selfcons={_a['selfcons_Hbar_vs_D1_maxfrac']:.2e}")
    print(f"  G3 lapse limit   : {'PASS' if g3_pass else 'FAIL'}   "
          f"max|gamma/((2+fv)/2)-1|={g3['fv0=0.853']['max_frac_gamma_vs_(2+fv)/2']:.2e}")
    print(f"  G4 numerics      : {'PASS' if g4_pass else 'FAIL'}   "
          f"max rel dD_M={g4['max_rel_dDM_stephalving']:.2e}  max rel dHbar={g4['max_rel_dHbar_stephalving']:.2e}")
    print("-" * 74)
    print(f"  ALL GATES: {'PASS' if all_pass else 'FAIL'}   -> Phase B: {'GO' if all_pass else 'NO-GO'}")
    print(f"  wrote {_OUT_JSON}   ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    run_gates()
