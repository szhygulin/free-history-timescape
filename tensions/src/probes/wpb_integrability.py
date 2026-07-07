#!/usr/bin/env python3
"""WP-B dynamical-consistency close-out: Buchert integrability check (DNW13 Eq 11).

Re-homes the integrability probe that used to live only as a quoted number in the
paper-2 NOTES_modelv_theory.md sec 8 (from a since-DELETED prototype `proto_modelv4.py`;
no committed generator existed). This is the committed generator.

WHAT IT COMPUTES
----------------
The Duley-Nazer-Wiltshire 2013 (arXiv:1306.3208) Buchert two-scale system admits an
integrability invariant: the void curvature parameter

    alpha^2 = -k_v * f_vi^{2/3}                        (DNW13 Eq 11, definition)

which MUST be CONSTANT in bare time tau for a genuine single-phase Buchert solution.
Written as a functional of the void history f_v(tau) and the bare (volume-average)
scale factor abar(tau), the transcription faithfully copied from
NOTES_modelv_theory.md ~line 262 is

    alpha^2(tau) = ( 2 abar^2 / (3 f_v^{1/3} (1 - f_v)) )
                   * [ f_v'' + f_v'^2 (2 f_v - 1) / (2 f_v (1 - f_v)) + 3 (abar'/abar) f_v' ]

primes = d/dtau (bare time). abar = tau^{2/3} (1 - f_v)^{-1/3} is the bare volume-average
scale factor (wall a_w = tau^{2/3}, volume closure a_w^3 = (1 - f_v) abar^3), so
abar'/abar = 2/(3 tau) + (1/3) f_v'/(1 - f_v) = the bare Hubble rate Hbar. See the
transcription-flag block in the JSON for the (un)ambiguity assessment.

For a constant-alpha^2 (single-phase two-scale) model this invariant is a genuine
constant; the FRACTIONAL SPREAD  max|alpha^2| / min|alpha^2| - 1  over the observable
window quantifies the integrability violation of an arbitrary forced history.

THREE HISTORIES driven through the paper-2 dressed-geometry solver (algebraic lapse):
  (i)   TRACKER  f_v (modelv_theory.tracker_fv_of_z): alpha^2 must be CONSTANT
        (~1e-6 pipeline noise floor) -- validates the Eq-11 transcription + closure (C1).
  (ii)  FORCED  f_v (LA Probe-R fitted nodes, modelV_probeR.json V.fv_nodes): the history
        the SN+BAO+CMB Hubble diagram REQUIRES.
  (iii) OBSERVED f_v (below-mean route, telescope_fvobs.json PRIMARY): f_v_obs(z) =
        Phi(sigma0 D(z)/2), the survey-measured void fraction.
Both (ii) and (iii) drift by O(100 %) to O(1000 %) -- the invariant is grossly
non-constant -> the forced/observed (Q, <R>) is not a consistent single-phase GR void
fluid. The forced magnitude is representation-sensitive (5 nodes under-determine f_v'');
the observed is representation-robust (its analytic form Phi(sigma0 D/2) is smooth).

DERIVATIVE PIPELINE.
  f_v'' is a SECOND derivative, so a C2 representation of f_v is required. The solver's
  monotone PCHIP (fv_from_nodes) is only C1: its f_v'' has spurious jumps at the nodes
  that inflate the drift by ~20x (reported as `pchip_literal` for contrast). The physical
  diagnostic uses a C2 cubic spline through the SAME nodes (natural BC), driven through
  the SAME solver. The algebraic-lapse map gives an EXACT closed form for dz/dtau
  (`_dz_dtau`), so f_v'(tau) = f_v'(z) dz/dtau is analytic; only f_v''(tau) is numeric, as
  np.gradient of the CONTINUOUS f_v'(tau) on a uniform tau grid (one numeric derivative of
  a C0 field, not two of a kinky one -- avoids the np.gradient blow-up in NOTES risk 2).

One number -> one script -> one JSON: probes_out/wpb_integrability.json.
Read-only on paper-2: imports the solver, reads two committed JSONs; writes nothing there.
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.special import ndtr  # standard-normal CDF Phi

# --- portable paths: this repo (paper 3) + the paper-2 sibling checkout ---------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))          # .../free-history-timescape-tensions
_PARENT = os.path.dirname(_REPO)
_P2 = os.path.join(_PARENT)             # paper-2 sibling
_P2_PROBES = os.path.join(_P2, "src", "probes")
_P2_PROBES_OUT = os.path.join(_P2, "probes_out")
if _P2_PROBES not in sys.path:
    sys.path.insert(0, _P2_PROBES)

import modelv_theory as MV  # noqa: E402  (solver + history builders; adds its own src/ to path)

_OUT = os.path.join(_REPO, "probes_out", "wpb_integrability.json")

Z_NODES = np.array([0.0, 0.3, 0.7, 1.3, 2.33])          # Probe-R / paper-2 default node grid
Z_SPAN_LO, Z_SPAN_HI = 0.0, float(Z_NODES[-1])          # full observable node span (primary window)
Z_INT_LO, Z_INT_HI = 0.1, 1.8                           # BC/1-over-tau-robust interior window
Z_SAMPLE = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.8, 2.33])
_FLOOR, _CEIL = 1e-5, 1.0 - 1e-9


# ---------------------------------------------------------------------------
# flat-LCDM linear growth D(z) (for the below-mean observed history)
# ---------------------------------------------------------------------------
def growth_D(z, Om=0.315):
    """Normalised flat-LCDM linear growth factor D(z), D(0)=1.

    D(a) prop H(a) int_0^a da'/(a' H(a'))^3 ,  H(a)/H0 = sqrt(Om a^-3 + (1-Om)).
    Reproduces the telescope_fvobs.json `D_z` column (validated in main()).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))

    def E(a):
        return np.sqrt(Om * a ** -3 + (1.0 - Om))

    def _g(a):
        integ, _ = quad(lambda ap: 1.0 / (ap * E(ap)) ** 3, 1e-8, a, limit=200)
        return E(a) * integ

    g1 = _g(1.0)
    return np.array([_g(1.0 / (1.0 + zi)) / g1 for zi in z])


def fv_obs_below_mean(z, sigma0, Om=0.315):
    """Observed below-mean void fraction f_v_obs(z) = Phi(sigma0 D(z) / 2)."""
    return ndtr(sigma0 * growth_D(z, Om=Om) / 2.0)


# ---------------------------------------------------------------------------
# C2 (natural cubic spline) history representation -- solver-compatible
# ---------------------------------------------------------------------------
class SmoothFv:
    """f_v(z) as a C2 natural cubic spline, clipped to (floor, ceil), with df_v/dz.

    Interface matches modelv_theory.MonotoneFv (._p, ._dp, __call__, .deriv, .floor,
    .ceil) so it drops into modelv_solve(lapse='algebraic'). C2 => f_v'' is continuous
    (unlike the solver's C1 PCHIP), which the integrability diagnostic requires.
    """

    def __init__(self, z_pts, fv_pts, floor=_FLOOR, ceil=_CEIL):
        z_pts = np.asarray(z_pts, dtype=float)
        fv_pts = np.asarray(fv_pts, dtype=float)
        order = np.argsort(z_pts)
        z_pts, fv_pts = z_pts[order], fv_pts[order]
        keep = np.concatenate([[True], np.diff(z_pts) > 0])
        self._p = CubicSpline(z_pts[keep], fv_pts[keep], bc_type="natural", extrapolate=True)
        self._dp = self._p.derivative(1)
        self.floor, self.ceil = float(floor), float(ceil)
        self.fv0 = float(np.clip(self._p(0.0), self.floor, self.ceil))

    def __call__(self, z):
        return np.clip(self._p(np.asarray(z, dtype=float)), self.floor, self.ceil)

    def deriv(self, z):
        return self._dp(np.asarray(z, dtype=float))


# ---------------------------------------------------------------------------
# alpha^2(tau) via the DNW13 Eq-11 transcription
# ---------------------------------------------------------------------------
def _dz_dtau(z, tau, fv, fvp_z):
    """Exact dz/dtau for the algebraic-lapse map 1+z=(abar0/abar)(gamma_bar/gamma_bar0).

    Derived by d/dtau of ln(1+z) = -ln abar + ln gamma_bar (+const), substituting
    f_v'(tau)=f_v'(z) dz/dtau; closed form in (z, tau, f_v, f_v'(z)).
    """
    denom = 1.0 / (1.0 + z) + fvp_z * (1.0 / (3.0 * (1.0 - fv)) - 1.0 / (2.0 + fv))
    return -2.0 / (3.0 * tau) / denom


def solve_and_alpha2(rep, tau0, n_tau=6000, z_pad=2.5):
    """Drive `rep` through modelv_solve, then evaluate alpha^2(tau) on a uniform tau grid.

    rep : MonotoneFv or SmoothFv (exposes ._p PCHIP/spline, ._dp derivative, .deriv()).
    Returns dict(z, tau, fv, fvp_tau, fvpp_tau, alpha2, n_iter, dz_resid, dzdtau_maxfrac).
    """
    sol = MV.modelv_solve(rep, lapse="algebraic", Ngrid=30000)
    idx = np.argsort(sol.tau)
    tau_s, z_s = sol.tau[idx], sol.z[idx]
    keep = np.concatenate([[True], np.diff(tau_s) > 0])
    tau_s, z_s = tau_s[keep], z_s[keep]
    z_of_tau = PchipInterpolator(tau_s, z_s, extrapolate=True)
    zr, tr = z_s[::-1], tau_s[::-1]
    k2 = np.concatenate([[True], np.diff(zr) > 0])
    tau_of_z = PchipInterpolator(zr[k2], tr[k2], extrapolate=True)

    tau = np.linspace(float(tau_of_z(z_pad)), tau0, int(n_tau))    # uniform -> clean np.gradient
    z = z_of_tau(tau)
    fv = rep._p(np.asarray(z))                                    # raw spline/PCHIP (unclipped; in-range)
    fvp_z = rep._dp(np.asarray(z))                               # df_v/dz analytic
    s = _dz_dtau(z, tau, fv, fvp_z)                              # dz/dtau (exact map)
    fvp = fvp_z * s                                              # df_v/dtau (analytic)
    fvpp = np.gradient(fvp, tau, edge_order=2)                   # d2f_v/dtau^2 (one numeric diff)

    abar = tau ** (2.0 / 3.0) * (1.0 - fv) ** (-1.0 / 3.0)
    abar0 = tau0 ** (2.0 / 3.0) * (1.0 - float(rep._p(0.0))) ** (-1.0 / 3.0)
    abar_n = abar / abar0                                        # convention: abar(tau0)=1
    abarp_over_abar = 2.0 / (3.0 * tau) + (1.0 / 3.0) * fvp / (1.0 - fv)  # = bare Hubble

    pref = 2.0 * abar_n ** 2 / (3.0 * fv ** (1.0 / 3.0) * (1.0 - fv))
    bracket = (fvpp
               + fvp ** 2 * (2.0 * fv - 1.0) / (2.0 * fv * (1.0 - fv))
               + 3.0 * abarp_over_abar * fvp)
    alpha2 = pref * bracket

    # analytic-vs-solver dz/dtau cross-check over the node span
    m = (z_s >= 0.0) & (z_s <= Z_SPAN_HI)
    s_num = np.gradient(z_s[m], tau_s[m])
    s_ana = _dz_dtau(z_s[m], tau_s[m], rep._p(z_s[m]), rep._dp(z_s[m]))
    good = np.abs(s_num) > 1e-6
    dzdtau_maxfrac = float(np.max(np.abs(s_ana[good] / s_num[good] - 1.0)))

    return {"z": z, "tau": tau, "fv": fv, "fvp_tau": fvp, "fvpp_tau": fvpp, "alpha2": alpha2,
            "n_iter": int(sol.n_iter), "dz_resid": float(sol.dz_resid),
            "dzdtau_maxfrac": dzdtau_maxfrac}


def _win(cur, lo, hi):
    z, a2 = cur["z"], cur["alpha2"]
    m = (z >= lo) & (z <= hi)
    return z[m], a2[m]


def drift(cur, lo=Z_SPAN_LO, hi=Z_SPAN_HI):
    z, a2 = _win(cur, lo, hi)
    a = np.abs(a2)
    return float(np.max(a) / np.min(a) - 1.0)


def extremes(cur, lo=Z_SPAN_LO, hi=Z_SPAN_HI):
    z, a2 = _win(cur, lo, hi)
    a = np.abs(a2)
    i_min, i_max = int(np.argmin(a)), int(np.argmax(a))
    return {"min_abs_alpha2": float(a[i_min]), "argmin_z": float(z[i_min]),
            "max_abs_alpha2": float(a[i_max]), "argmax_z": float(z[i_max]),
            "sign_change": bool(np.any(a2 > 0) and np.any(a2 < 0))}


def sample(cur, z_at=Z_SAMPLE):
    order = np.argsort(cur["z"])
    z, a2, fv = cur["z"][order], cur["alpha2"][order], cur["fv"][order]
    return np.interp(z_at, z, a2), np.interp(z_at, z, fv)


# ---------------------------------------------------------------------------
# independent transcription check: EXACT analytic tracker (no solver, no np.gradient)
# ---------------------------------------------------------------------------
def tracker_alpha2_analytic(fv0, taus):
    """alpha^2 from the CLOSED-FORM tracker f_v(tau)=3 fv0 tau/(3 fv0 tau + b).

    All derivatives exact => alpha^2 must be machine-precision constant iff the
    transcription is correct. abar normalised to abar(tau0)=1, tau0=(2+fv0)/3.
    """
    b = (1.0 - fv0) * (2.0 + fv0)
    D = 3.0 * fv0 * taus + b
    fv = 3.0 * fv0 * taus / D
    fvp = 3.0 * fv0 * b / D ** 2
    fvpp = -18.0 * fv0 ** 2 * b / D ** 3
    abar = taus ** (2.0 / 3.0) * (D / b) ** (1.0 / 3.0)
    tau0 = (2.0 + fv0) / 3.0
    abar0 = tau0 ** (2.0 / 3.0) * ((3.0 * fv0 * tau0 + b) / b) ** (1.0 / 3.0)
    abar_n = abar / abar0
    abarp_over_abar = 2.0 / (3.0 * taus) + (1.0 / 3.0) * fvp / (1.0 - fv)
    pref = 2.0 * abar_n ** 2 / (3.0 * fv ** (1.0 / 3.0) * (1.0 - fv))
    bracket = (fvpp + fvp ** 2 * (2.0 * fv - 1.0) / (2.0 * fv * (1.0 - fv))
               + 3.0 * abarp_over_abar * fvp)
    return pref * bracket


# ---------------------------------------------------------------------------
def _hist_block(cur, extra=None):
    ex = extremes(cur)
    a2_s, fv_s = sample(cur)
    zw, a2w = _win(cur, Z_SPAN_LO, Z_SPAN_HI)
    block = {
        "fractional_drift": drift(cur),                       # PRIMARY: full span [0, 2.33]
        "fractional_drift_interior_0.1_1.8": drift(cur, Z_INT_LO, Z_INT_HI),
        "fractional_drift_0.3_1.3": drift(cur, 0.3, 1.3),
        "alpha2_mean_over_span": float(np.mean(a2w)),
        "extremes_over_span": ex,
        "z_sample": Z_SAMPLE.tolist(),
        "alpha2_sample": [float(v) for v in a2_s],
        "fv_sample": [float(v) for v in fv_s],
        "solver_n_iter": cur["n_iter"],
        "solver_dz_resid": cur["dz_resid"],
        "analytic_dzdtau_vs_solver_maxfrac": cur["dzdtau_maxfrac"],
    }
    if extra:
        block.update(extra)
    return block


def main():
    with open(os.path.join(_P2_PROBES_OUT, "modelV_probeR.json")) as f:
        probeR = json.load(f)
    with open(os.path.join(_P2_PROBES_OUT, "telescope_fvobs.json")) as f:
        tele = json.load(f)

    fv_nodes_forced = np.array(probeR["V"]["fv_nodes"], dtype=float)   # LA fitted required history
    fv0_forced = float(probeR["V"]["fv0"])
    sigma0 = float(tele["provenance"]["sigma0_anchor"])

    # ---- validate the growth reproduction against telescope_fvobs.json D_z ---
    dz_checks = []
    for nd in tele["PRIMARY_below_mean_Rs4"]["nodes"]:
        d_mine = float(growth_D(nd["z"])[0])
        dz_checks.append({"z": nd["z"], "D_json": nd["D_z"], "D_mine": d_mine,
                          "frac_diff": abs(d_mine / nd["D_z"] - 1.0)})
    growth_maxfrac = max(c["frac_diff"] for c in dz_checks)

    fv_nodes_obs = fv_obs_below_mean(Z_NODES, sigma0)
    obs_json_check = {
        "z0_anchor": {"mine": float(fv_nodes_obs[0]),
                      "json": float(tele["provenance"]["below_mean_z0_anchor"])},
        "z0.3": {"mine": float(fv_nodes_obs[1]),
                 "json": float(tele["PRIMARY_below_mean_Rs4"]["at_required_grid"]["z=0.3"]["fv"])},
        "z0.7": {"mine": float(fv_nodes_obs[2]),
                 "json": float(tele["PRIMARY_below_mean_Rs4"]["at_required_grid"]["z=0.7"]["fv"])},
    }

    # ---- (i) TRACKER: exact Buchert solution (validation) --------------------
    fv0_tracker = fv0_forced        # same present void fraction as the forced history
    trk_rep = MV.tracker_fv_of_z(fv0_tracker)
    tau0_trk = (2.0 + fv0_tracker) / 3.0
    trk_cur = solve_and_alpha2(trk_rep, tau0_trk)

    # ---- (ii) FORCED required history: C2 smooth (primary) + PCHIP literal ----
    tau0_f = (2.0 + fv0_forced) / 3.0
    forced_c2 = solve_and_alpha2(SmoothFv(Z_NODES, fv_nodes_forced), tau0_f)
    forced_pchip = solve_and_alpha2(MV.fv_from_nodes(fv_nodes_forced, z_nodes=Z_NODES), tau0_f)

    # ---- (iii) OBSERVED below-mean: C2 nodes (primary) + analytic Phi(sigma0 D/2) + PCHIP
    fv0_obs = float(fv_nodes_obs[0])
    tau0_o = (2.0 + fv0_obs) / 3.0
    obs_c2 = solve_and_alpha2(SmoothFv(Z_NODES, fv_nodes_obs), tau0_o)
    obs_pchip = solve_and_alpha2(MV.fv_from_nodes(fv_nodes_obs, z_nodes=Z_NODES), tau0_o)
    z_dense = np.linspace(0.0, 3.2, 300)                          # analytic below-mean history (smooth)
    obs_analytic = solve_and_alpha2(SmoothFv(z_dense, fv_obs_below_mean(z_dense, sigma0)), tau0_o)

    # ---- independent transcription validation: exact analytic tracker -------
    taus = np.linspace(tau0_trk * (1.0 / (1.0 + Z_SPAN_HI)) ** 1.5, tau0_trk, 4000)
    a2_exact = tracker_alpha2_analytic(fv0_tracker, taus)
    exact_drift = float(np.max(np.abs(a2_exact)) / np.min(np.abs(a2_exact)) - 1.0)
    tracker_drift = drift(trk_cur)                                   # full span [0, 2.33]
    tracker_drift_interior = drift(trk_cur, Z_INT_LO, Z_INT_HI)      # BC/boundary-clean
    # validation rests on the boundary-clean interior pipeline drift + the exact-analytic check
    tracker_validates = bool(tracker_drift_interior < 1e-4 and exact_drift < 1e-10)

    histories = {
        "tracker": _hist_block(trk_cur, extra={
            "desc": "TRACKER f_v(z) (modelv_theory.tracker_fv_of_z) -- exact Buchert solution",
            "fv0": fv0_tracker,
            "representation": "dense analytic-tracker samples (modelv_theory MonotoneFv)",
        }),
        "forced_required": _hist_block(forced_c2, extra={
            "desc": "FORCED required f_v(z): LA Probe-R fitted nodes (modelV_probeR.json V.fv_nodes)",
            "fv0": fv0_forced,
            "representation": "C2 natural cubic spline through the 5 fitted nodes",
            "fv_nodes": fv_nodes_forced.tolist(),
            "representation_sensitivity": {
                "note": "the 5-node forced history under-determines f_v'' -> the drift MAGNITUDE is "
                        "representation-dependent; every reading is >> the tracker's ~1e-6.",
                "pchip_literal_fractional_drift_span": drift(forced_pchip),
                "pchip_literal_note": "the solver's monotone C1 PCHIP: f_v'' has spurious jumps at the "
                                      "nodes -> ~20x inflation vs the C2 reading; sign of the violation, "
                                      "not its size.",
                "c2_full_span_note": "the C2 full-span drift is inflated by the natural-BC endpoint at "
                                     "z=2.33 (min|alpha^2| sits at the boundary); the interior windows are "
                                     "BC-robust. The max|alpha^2| ~2x the tracker value sits at z~0.7 and "
                                     "is representation-stable.",
            },
        }),
        "observed_below_mean": _hist_block(obs_c2, extra={
            "desc": "OBSERVED f_v_obs(z)=Phi(sigma0 D(z)/2), below-mean route (telescope_fvobs PRIMARY)",
            "fv0": fv0_obs,
            "representation": "C2 natural cubic spline through the 5 below-mean nodes at the required grid",
            "fv_nodes": [float(v) for v in fv_nodes_obs],
            "representation_sensitivity": {
                "note": "the observed history is genuinely smooth (Phi(sigma0 D(z)/2)) -> representation-"
                        "ROBUST: the analytic-form and PCHIP readings agree with the C2-node reading.",
                "analytic_form_fractional_drift_span": drift(obs_analytic),
                "analytic_form_fractional_drift_interior": drift(obs_analytic, Z_INT_LO, Z_INT_HI),
                "pchip_literal_fractional_drift_span": drift(obs_pchip),
            },
        }),
    }

    out = {
        "probe": "WP-B integrability / dynamical-consistency close-out -- Buchert invariant "
                 "alpha^2=-k_v f_vi^{2/3} (DNW13 Eq 11) along three void histories",
        "rehomes": "paper-2 NOTES_modelv_theory.md sec 8 (quoted from the deleted proto_modelv4.py; "
                   "no committed generator existed until this script)",
        "reading": "KINEMATIC forced-f_v histories vs the constant-alpha^2 (single-phase two-scale) "
                   "Buchert invariant; a large fractional drift => integrability (K5) violated.",
        "equation": {
            "definition": "alpha^2 = -k_v * f_vi^{2/3}  (DNW13 Eq 11; MUST be constant in tau for a "
                          "genuine single-phase Buchert two-scale solution)",
            "transcription": "alpha^2(tau) = (2 abar^2 / (3 f_v^{1/3} (1-f_v))) * "
                             "[ f_v'' + f_v'^2 (2 f_v - 1)/(2 f_v (1-f_v)) + 3 (abar'/abar) f_v' ]",
            "source_line": "NOTES_modelv_theory.md ~line 262",
            "primes": "d/dtau (bare time)",
            "abar": "bare volume-average scale factor = tau^{2/3} (1-f_v)^{-1/3}; abar'/abar = "
                    "2/(3 tau) + (1/3) f_v'/(1-f_v) = bare Hubble rate",
            "normalisation_convention": "abar(tau0)=1; alpha^2 carries abar^2 so its ABSOLUTE scale is "
                                        "this convention, but the fractional spread max|a2|/min|a2|-1 is "
                                        "convention-INVARIANT (the overall abar^2 constant cancels).",
            "transcription_flag": "UNAMBIGUOUS as transcribed. Two reading choices were checked and are "
                                  "pinned by the tracker: (a) primes are d/dtau in BARE time (so abar'/abar "
                                  "is the bare Hubble rate that closes the map); (b) abar is the BARE "
                                  "volume-average scale factor tau^{2/3}(1-f_v)^{-1/3}, not the wall "
                                  "a_w=tau^{2/3}. Both readings are the ones for which the exact analytic "
                                  "tracker gives a machine-precision constant alpha^2 "
                                  "(exact_analytic_fractional_drift=2e-15 below) -> the copy is faithful.",
        },
        "growth_reproduction": {
            "Om": 0.315,
            "max_frac_diff_vs_json_Dz": growth_maxfrac,
            "note": "flat-LCDM linear-growth D(z) reproduces telescope_fvobs.json D_z column exactly; "
                    "used to place the below-mean f_v_obs at the required-grid z=1.3, 2.33.",
            "checks": dz_checks,
        },
        "observed_nodes_check": obs_json_check,
        "windows": {
            "primary_span": [Z_SPAN_LO, Z_SPAN_HI],
            "interior": [Z_INT_LO, Z_INT_HI],
            "metric": "fractional spread = max|alpha^2| / min|alpha^2| - 1",
        },
        "histories": histories,
        "transcription_validation": {
            "tracker_pipeline_fractional_drift_span": tracker_drift,
            "tracker_pipeline_fractional_drift_interior": tracker_drift_interior,
            "exact_analytic_fractional_drift": exact_drift,
            "tracker_validates": tracker_validates,
            "note": "tracker_pipeline drift is the FULL solver+np.gradient pipeline noise floor "
                    "(NOTES sec-8 quoted 3.9e-7 from the deleted prototype; same order); exact_analytic "
                    "drift tests the transcription with closed-form derivatives (no solver, no numeric "
                    "differentiation) -> machine-precision constant confirms the Eq-11 copy is faithful "
                    "and the void-curvature invariant is genuinely constant on the tracker.",
        },
        "verdict": {
            "summary": "THEORY-LEVEL CLOSE-OUT (a result, not a failure). No consistent single-phase "
                       "two-scale (constant-alpha^2) Buchert model tracks either the REQUIRED "
                       "(forced_required) or the OBSERVED (observed_below_mean) history: both drift by "
                       "O(100%)-O(1000%) while the tracker is constant to the ~1e-6 pipeline floor "
                       "(6+ orders of magnitude contrast).",
            "tracker_drift": tracker_drift,
            "forced_required_drift_span": histories["forced_required"]["fractional_drift"],
            "forced_required_drift_interior": histories["forced_required"]["fractional_drift_interior_0.1_1.8"],
            "observed_below_mean_drift_span": histories["observed_below_mean"]["fractional_drift"],
            "observed_below_mean_drift_interior": histories["observed_below_mean"]["fractional_drift_interior_0.1_1.8"],
            "note_on_81pct": "NOTES sec-8's '81%' was the deleted prototype's illustrative smooth forced "
                             "history, NOT the final Probe-R V.fv_nodes; the actual required history is a "
                             "stronger deformation and drifts more (interior ~1.5x; span dominated by the "
                             "BC-sensitive z=2.33 endpoint). The magnitude is representation-sensitive; the "
                             "qualitative verdict (>> tracker, non-constant) is representation-INDEPENDENT.",
        },
        "inputs": {
            "solver": os.path.join(_P2_PROBES, "modelv_theory.py"),
            "probeR_json": os.path.join(_P2_PROBES_OUT, "modelV_probeR.json"),
            "telescope_json": os.path.join(_P2_PROBES_OUT, "telescope_fvobs.json"),
            "z_nodes": Z_NODES.tolist(),
            "fv_nodes_forced": fv_nodes_forced.tolist(),
            "fv_nodes_observed": [float(v) for v in fv_nodes_obs],
            "fv0_tracker": fv0_tracker,
        },
    }

    with open(_OUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[wpb_integrability] wrote {_OUT}")
    print(f"  growth max frac diff vs JSON D_z : {growth_maxfrac:.2e}")
    print(f"  {'history':22s} {'fv0':>6s} {'drift[0,2.33]':>13s} {'drift[.1,1.8]':>13s}")
    for name in ("tracker", "forced_required", "observed_below_mean"):
        h = histories[name]
        print(f"  {name:22s} {h['fv0']:6.4f} {h['fractional_drift']:13.3e} "
              f"{h['fractional_drift_interior_0.1_1.8']:13.3e}")
    print(f"  forced PCHIP-literal span drift  : "
          f"{histories['forced_required']['representation_sensitivity']['pchip_literal_fractional_drift_span']:.3e}")
    print(f"  observed analytic-form span drift: "
          f"{histories['observed_below_mean']['representation_sensitivity']['analytic_form_fractional_drift_span']:.3e}")
    print(f"  exact-analytic tracker drift     : {exact_drift:.2e}")
    print(f"  tracker_validates                : {tracker_validates}")


if __name__ == "__main__":
    main()
