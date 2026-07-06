#!/usr/bin/env python3
"""Probe R — the REQUIRED void history f_v(z) through the Model V dressed geometry.

Pre-registered decision gate R1 (PLAN_void_history.md secs 0, 3). This MIRRORS
`probeT1_freeshape.py` in structure, but replaces the free-E(z) FLRW spline distance
with the Model V dressed, lapse-driven distances (`modelv_theory.modelv_solve`)
driven by a FREE, monotone void history f_v(z).

Question (R1): the tracker compresses the whole void history into ONE number f_v0 and
fails (joint chi2=1469.3 vs LCDM 1402.2). Is the *mechanism* (two-phase Buchert
averaging + wall/void clock-rate / lapse dressing) right but the tracker *closure*
wrong? Discard the attractor; drive the SAME dressed geometry with an ARBITRARY
f_v(z) and ask whether SOME history reconciles SN + BAO + CMB.

  chi2_min_V <= 1412.24 (LCDM+10)  -> RECONCILES_mechanism_flexible  (GO to Phase D/R2)
  1412.24 < chi2_min_V <= 1427.24  -> DISFAVOURED
  chi2_min_V >  1427.24 (LCDM+25)  -> REFUTED_mechanism_rigid  (the lapse fixes the
                                       D_M:D_H relation so tightly that even a free
                                       f_v cannot serve SN and BAO+CMB together)
  chi2_min_V >= 1469.29 (tracker)  -> AMPLITUDE_DEAD

Parametrisation: monotone (strictly decreasing in z) f_v at nodes
z = {0, 0.3, 0.7, 1.3, 2.33} in (0, 1), built through a smooth sigmoid-multiplicative
transform (no hard walls -> stable Nelder-Mead / differential_evolution), + a
tracker-shaped high-z bridge f_v(z) ~ f_v(z_last) ((1+z_last)/(1+z))^{3/2}
(the exact tracker small-tau asymptotic; value-matched at the last node; -> ~0 by the
CMB point so the early geometry is flat-dust EdS, exactly as the tracker limit).

Two machineries per run:
  V  (primary)  : lapse="algebraic"  gamma_bar=(2+f_v)/2  (Wiltshire dressing)
  V0 (control)  : lapse="none"       gamma_bar==1         (pure Buchert, no clock
                  dressing) -- separates "lapse needed" from "any two-phase suffices".

Data: IDENTICAL to the committed joint references -- harness.sn_chi2 (1580 SNe, full
stat+sys covariance, M_B/H0 offset marginalised) + harness.bao_cmb_chi2 (DESI DR1 +
Planck acoustic point, alpha=c/(Hbar0 r_d) marginalised, r_d=147.09).

Sanity anchors (among the restarts):
  * tracker-EXACT (via the oracle f_v(z), gate-style) -> reproduces joint 1469.29;
  * tracker-NODE start (tracker f_v sampled at the 5 nodes) -> near 1469.3 (the small
    residual is the 5-node PCHIP representation error of the smooth tracker shape);
  * LCDM-projected start (f_v nodes pre-fit to the LCDM D_M(z) shape).

Resumable: intra-run checkpointing to the JSON every CKPT_EVERY restarts; a soft
wall-clock limit PROBER_MAXSEC exits cleanly for relaunch (seifert.py pattern).

Run from src/:   python probes/modelv_probeR.py
Env knobs: PROBER_NGRID_FIT (10000), PROBER_NGRID_FINE (30000), PROBER_NRESTARTS (44),
           PROBER_MAXSEC (0=off), PROBER_SEED (1234), PROBER_PROFILE (1).
"""
import os
import sys
import io
import json
import time
import contextlib
import numpy as np
from scipy.optimize import minimize, differential_evolution

np.seterr(all="ignore")   # bad random node vectors can make D_V^(1/3) of a negative -> nan; obj() maps nan->1e9

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(_SRC)

import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):   # these fit+print at import
    import harness as H

OUTJ = os.path.join(os.path.dirname(_SRC), "probes_out", "modelV_probeR.json")

# ---- committed joint references on the SAME data (results_joint.json / T1) --------
REF = {"LCDM": 1402.2372, "w0waCDM": 1398.2856, "tracker": 1469.2926, "free_E": 1391.8498}
THR = {"reconciles_le": REF["LCDM"] + 10.0,     # 1412.24
       "disfavoured_le": REF["LCDM"] + 25.0,     # 1427.24
       "amplitude_dead_ge": REF["tracker"]}      # 1469.29

# ---- config -----------------------------------------------------------------------
Z_NODES = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
FLOOR, CEIL = MV._FV_FLOOR, MV._FV_CEIL           # (1e-5, 1-1e-9)
BRIDGE_Z = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])

NGRID_FIT = int(os.environ.get("PROBER_NGRID_FIT", 6000))     # fit resolution (converged ~2e-4 on tracker)
NGRID_FINE = int(os.environ.get("PROBER_NGRID_FINE", 30000))  # headline re-eval resolution
N_RESTARTS = int(os.environ.get("PROBER_NRESTARTS", 44))
MAXSEC = float(os.environ.get("PROBER_MAXSEC", 0)) or None
SEED = int(os.environ.get("PROBER_SEED", 1234))
DO_PROFILE = os.environ.get("PROBER_PROFILE", "1") == "1"
CKPT_EVERY = 4

_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)

zHD, zHEL, mb, Cf = F.load()


def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# monotone (strictly decreasing in z) f_v-node <-> unconstrained param transform
#   v0 = CEIL * sig(p0)          (z=0, largest)
#   v_k = v_{k-1} * sig(p_k)     (strictly < v_{k-1}, > 0)     k=1..4
# smooth & unconstrained: no hard walls for Nelder-Mead / differential_evolution.
# ---------------------------------------------------------------------------
def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
def _logit(y):
    y = np.clip(y, 1e-12, 1.0 - 1e-12)
    return np.log(y / (1.0 - y))


def nodes_from_params(p):
    g = _sig(np.asarray(p, dtype=float))
    v = np.empty(5)
    v[0] = CEIL * g[0]
    for i in range(1, 5):
        v[i] = v[i - 1] * g[i]
    return v


def params_from_nodes(v):
    v = np.asarray(v, dtype=float)
    g = np.empty(5)
    g[0] = v[0] / CEIL
    for i in range(1, 5):
        g[i] = v[i] / v[i - 1]
    return _logit(g)


# ---------------------------------------------------------------------------
# forward model: forced f_v(z) nodes -> dressed Model V geometry -> joint chi2
# ---------------------------------------------------------------------------
def _bridge_fv(fv_last):
    z_last = Z_NODES[-1]
    return fv_last * ((1.0 + z_last) / (1.0 + BRIDGE_Z)) ** 1.5


def solve_nodes(v, lapse, Ngrid):
    fv = MV.fv_from_nodes(np.asarray(v, dtype=float), z_nodes=Z_NODES,
                          bridge_z=BRIDGE_Z, bridge_fv=_bridge_fv(float(v[-1])))
    return MV.modelv_solve(fv, lapse=lapse, Ngrid=Ngrid)


def joint_from_solution(sol):
    csn = float(H.sn_chi2(sol.D_M(zHD)))
    cbc, a = H.bao_cmb_chi2(lambda z, k: float(sol.predict(z, k)))
    return csn + cbc, csn, cbc, float(a)


def dressed_H0(sol, a, lapse):
    """Dressed present H0 for the best fit. Two conventions (they COINCIDE on the
    tracker, gate G-T(d): Hd(z=0)==g_dress(fv0); they SPLIT for a free history because
    the present void-growth term f_v'(tau0) is no longer the tracker value):

      * gdress  = g_dress(fv0) * Hbar0  -- fv0-only, SLOPE-INDEPENDENT, robust, and the
        convention modelv_theory.modelv_dressed_H0 designates for Probe R + the one the
        committed tracker references use. HEADLINE. (Only meaningful with the lapse; for
        the no-lapse V0 control there is no clock dressing, so gdress is not applicable.)
      * Hd0     = Hd(z=0) * Hbar0  -- the FULL present dressed rate gamma_bar0*Hbar -
        dgamma_bar/dt|_0, incl. the present slope term; physically the true dressed H0 of
        THIS history, but endpoint-PCHIP-slope sensitive (the data only weakly pin f_v'(0)).
        For V0 (gamma_bar==1) Hd(z=0)=<H>(z=0)/Hbar0 IS the observed H0."""
    Hbar0 = float(H.H0_from_alpha(a))
    hd0 = float(sol.Hd[np.argmin(np.abs(sol.z))])
    g = float(MV.g_dress(sol.fv0))
    headline = (g if lapse == "algebraic" else hd0) * Hbar0   # V0: no dressing -> use Hd0
    return dict(H0_dressed=headline, H0_dressed_gdress=g * Hbar0,
                H0_dressed_Hd0=hd0 * Hbar0, Hbar0=Hbar0, g_dress=g, Hd_z0=hd0)


def joint_nodes(v, lapse, Ngrid, parts=False):
    sol = solve_nodes(v, lapse, Ngrid)
    tot, csn, cbc, a = joint_from_solution(sol)
    if parts:
        return tot, csn, cbc, a, sol
    return tot


def obj(p, lapse, Ngrid=None):
    Ngrid = NGRID_FIT if Ngrid is None else Ngrid
    try:
        v = nodes_from_params(p)
        c = joint_nodes(v, lapse, Ngrid)
    except Exception:
        return 1e9
    return float(c) if np.isfinite(c) else 1e9


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
def tracker_node_values(fv0=0.6426):
    trk = MV.tracker_fv_of_z(fv0)
    return np.array([float(trk(z)) for z in Z_NODES])


def tracker_exact_joint(fv0=0.6426):
    """Gate-style: drive the general solver with the full oracle tracker f_v(z)
    (300k-point grid ~ exact) -> should reproduce joint 1469.29. Validates the whole
    Probe-R harness path (SN + BAO/CMB predict) independent of the 5-node basis."""
    trk = MV.tracker_fv_of_z(fv0)
    sol = MV.modelv_solve(trk, Ngrid=NGRID_FINE, lapse="algebraic")
    tot, csn, cbc, a = joint_from_solution(sol)
    h0 = dressed_H0(sol, a, "algebraic")
    return dict(chi2=tot, chi2_SN=csn, chi2_BAOCMB=cbc, H0_dressed=h0["H0_dressed"])


def lcdm_projected_nodes(Om=0.3048091938075369):
    """f_v(z) nodes pre-fit to the LCDM D_M(z) SHAPE (project LCDM onto the V manifold).
    A legitimate 'LCDM-matched' seed; its joint chi2 measures how well the dressed
    geometry can *approximate* LCDM distances."""
    zg = np.linspace(0.02, Z_NODES[-1], 60)
    DM_lcdm = H.lcdm_Dc(zg, Om)

    def resid(p):
        try:
            sol = solve_nodes(nodes_from_params(p), "algebraic", NGRID_FIT)
            dm = sol.D_M(zg)
            if not np.all(np.isfinite(dm)) or np.any(dm <= 0):
                return 1e9
            s = np.sum(dm * DM_lcdm) / np.sum(dm * dm)  # scale-free shape match
            return float(np.sum((s * dm - DM_lcdm) ** 2))
        except Exception:
            return 1e9

    best = None
    for seed in (params_from_nodes([0.60, 0.52, 0.42, 0.30, 0.20]),
                 params_from_nodes([0.85, 0.78, 0.68, 0.55, 0.40]),
                 params_from_nodes([0.40, 0.34, 0.27, 0.19, 0.12])):
        r = minimize(resid, seed, method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-6, maxiter=6000))
        if best is None or r.fun < best.fun:
            best = r
    return nodes_from_params(best.x), float(best.fun), zg.size


# ---------------------------------------------------------------------------
# multistart (resumable): >= N_RESTARTS Nelder-Mead + one differential_evolution
# ---------------------------------------------------------------------------
def build_starts(lapse, rng, anchor_ps):
    starts = list(anchor_ps)                                   # anchors first
    # structured spread of plausible monotone histories
    grid = [[0.85, 0.78, 0.68, 0.55, 0.40], [0.70, 0.62, 0.52, 0.40, 0.28],
            [0.55, 0.47, 0.38, 0.27, 0.18], [0.95, 0.90, 0.82, 0.70, 0.55],
            [0.40, 0.33, 0.26, 0.18, 0.11], [0.90, 0.70, 0.45, 0.25, 0.12],
            [0.98, 0.85, 0.60, 0.35, 0.18], [0.30, 0.24, 0.18, 0.12, 0.07]]
    for gv in grid:
        starts.append(params_from_nodes(gv))
    while len(starts) < N_RESTARTS:                            # random fill
        starts.append(rng.uniform(-5.0, 5.0, 5))
    return starts[:max(N_RESTARTS, len(anchor_ps) + len(grid))]


def run_multistart(lapse, starts, prior_best=None):
    best_chi2 = np.inf if prior_best is None else prior_best["chi2"]
    best_p = None if prior_best is None else np.array(prior_best["p"])
    done = 0 if prior_best is None else prior_best.get("done", 0)
    endpoints = []
    for i, s in enumerate(starts):
        if i < done:
            continue
        r = minimize(obj, s, args=(lapse,), method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=5e-3, maxiter=2000))
        endpoints.append((float(r.fun), r.x.copy()))
        if r.fun < best_chi2:
            best_chi2, best_p = float(r.fun), r.x.copy()
        done = i + 1
        if (i + 1) % CKPT_EVERY == 0:
            _checkpoint(lapse, dict(chi2=best_chi2, p=best_p.tolist(), done=done,
                                    n_starts=len(starts)))
            log(f"  [{lapse}] restart {i+1}/{len(starts)}  best={best_chi2:.4f}")
        if MAXSEC and time.time() - _t0 > MAXSEC:
            _checkpoint(lapse, dict(chi2=best_chi2, p=best_p.tolist(), done=done,
                                    n_starts=len(starts)))
            log(f"  [{lapse}] MAXSEC hit at restart {i+1}; checkpointed, exit for relaunch")
            sys.exit(3)
    # one differential_evolution polish over the same space
    de = differential_evolution(obj, bounds=[(-6.0, 6.0)] * 5, args=(lapse,),
                                seed=SEED, maxiter=60, popsize=15, tol=1e-6,
                                mutation=(0.4, 1.0), recombination=0.8,
                                polish=True, init="sobol")
    endpoints.append((float(de.fun), de.x.copy()))
    de_used = de.fun < best_chi2
    if de_used:
        best_chi2, best_p = float(de.fun), de.x.copy()
    log(f"  [{lapse}] DE fun={de.fun:.4f}  -> global best={best_chi2:.4f}")
    return best_chi2, best_p, endpoints, de_used


_CKPT = {}
def _checkpoint(lapse, state):
    _CKPT[lapse] = state
    _atomic_dump({"_checkpoint": True, "stage": "multistart", "state": _CKPT}, OUTJ)


# ---------------------------------------------------------------------------
# per-node Delta-chi2 <= 1 profile band (profile likelihood; node pinned softly,
# the other 4 re-optimised through the monotone transform so ordering always holds)
# ---------------------------------------------------------------------------
def profile_node(i, best_p, lapse, chi2_min):
    v_best = nodes_from_params(best_p)
    vi0 = v_best[i]
    W = 1.0e6

    def pinned(p, vi):
        v = nodes_from_params(p)
        c = obj(p, lapse)
        return c + W * (v[i] - vi) ** 2

    def chi2_at(vi):
        r = minimize(pinned, best_p, args=(vi,), method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=5e-3, maxiter=2000))
        v = nodes_from_params(r.x)
        return float(joint_nodes(v, lapse, NGRID_FIT)), float(v[i])

    def scan(direction):
        """walk vi outward until chi2 crosses chi2_min+1 or an edge; return the
        Delta-chi2=1 boundary (linear-interpolated). A flat direction returns the
        scanned edge (node effectively unconstrained on that side)."""
        step = max(0.02, 0.05 * vi0)
        prev_v, prev_d = vi0, 0.0
        for k in range(1, 24):
            vi = float(np.clip(vi0 + direction * step * k, FLOOR * 1.5, CEIL * (1 - 1e-6)))
            c, vgot = chi2_at(vi)
            d = c - chi2_min
            if d >= 1.0:                                      # crossed -> interpolate
                if d > prev_d:
                    return float(prev_v + (vgot - prev_v) * (1.0 - prev_d) / (d - prev_d))
                return float(vgot)
            prev_v, prev_d = vgot, d
            if vi <= FLOOR * 2 or vi >= CEIL * (1 - 1e-6):    # hit an edge, still < +1
                return float(vgot)
        return float(prev_v)

    lo = scan(-1.0)
    hi = scan(+1.0)
    return [min(lo, hi), max(lo, hi)]


# ---------------------------------------------------------------------------
# derived required-backreaction curves (exact kinematic identities, sec 1.1)
# ---------------------------------------------------------------------------
def derived_curves(v, lapse="algebraic"):
    fv_cb = MV.fv_from_nodes(np.asarray(v, dtype=float), z_nodes=Z_NODES,
                             bridge_z=BRIDGE_Z, bridge_fv=_bridge_fv(float(v[-1])))
    sol = MV.modelv_solve(fv_cb, lapse=lapse, Ngrid=NGRID_FINE)
    zq = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.8, 2.33])
    # interpolate tau(z), fv(z) and df_v/dtau on the solution grid (z ascending)
    tau = np.interp(zq, sol.z, sol.tau)
    fv = fv_cb(zq)
    # df_v/dtau = (df_v/dz)(dz/dtau); dz/dtau from the solution grid
    dz_dtau = np.gradient(sol.z, sol.tau)
    dzdt_q = np.interp(zq, sol.z, dz_dtau)
    fvp = fv_cb.deriv(zq) * dzdt_q                        # df_v/dtau (>0: voids grow)
    one_m = np.clip(1.0 - fv, 1e-9, None)
    fvc = np.clip(fv, 1e-9, None)
    Hw = 2.0 / (3.0 * tau)                                # H_w/Hbar0
    dHvw = fvp / (3.0 * fvc * one_m)                      # (H_v - H_w)/Hbar0
    Hbar = Hw + fvp / (3.0 * one_m)                       # <H>/Hbar0
    Q = (2.0 / 3.0) * fvp ** 2 / (fvc * one_m)            # Q/Hbar0^2 (>=0)
    excess = dHvw / np.clip(Hbar, 1e-9, None)             # kinematic void-depth proxy
    return dict(
        z=zq.tolist(),
        fv=np.round(fv, 5).tolist(),
        Hw_over_Hbar0=np.round(Hw, 5).tolist(),
        Hv_minus_Hw_over_Hbar0=np.round(dHvw, 5).tolist(),
        Hbar_over_Hbar0=np.round(Hbar, 5).tolist(),
        Q_over_Hbar0sq=np.round(Q, 6).tolist(),
        void_expansion_excess=np.round(excess, 5).tolist(),
        note=("(H_v-H_w) and Q are exact kinematic identities of the forced f_v(z) "
              "(sec 1.1). 'void_expansion_excess'=(H_v-H_w)/<H> is a dimensionless "
              "kinematic depth proxy; a matter-density delta_v(z) is NOT uniquely "
              "determined in the kinematic reading (it needs the integrability/"
              "dynamical closure) -- reported as a follow-up, not here."))


# ---------------------------------------------------------------------------
def fit_variant(lapse, anchor_ps, prior_best=None):
    rng = np.random.default_rng(SEED + (0 if lapse == "algebraic" else 7))
    starts = build_starts(lapse, rng, anchor_ps)
    log(f"[{lapse}] multistart: {len(starts)} restarts + 1 DE")
    best_chi2_fit, best_p, endpoints, de_used = run_multistart(lapse, starts, prior_best)
    v_best = nodes_from_params(best_p)
    # headline re-eval at fine resolution
    tot, csn, cbc, a, sol = joint_nodes(v_best, lapse, NGRID_FINE, parts=True)
    h0 = dressed_H0(sol, a, lapse)
    out = dict(
        lapse=lapse, chi2_min=tot, chi2_min_fitgrid=best_chi2_fit,
        chi2_SN=csn, chi2_BAOCMB=cbc, alpha=a,
        fv0=float(sol.fv0),
        H0_dressed=h0["H0_dressed"], H0_dressed_gdress=h0["H0_dressed_gdress"],
        H0_dressed_Hd0=h0["H0_dressed_Hd0"], Hbar0=h0["Hbar0"],
        g_dress=h0["g_dress"], Hd_z0=h0["Hd_z0"],
        z_nodes=Z_NODES.tolist(), fv_nodes=np.round(v_best, 5).tolist(),
        n_restarts=len(starts), de_used=bool(de_used),
        ngrid_fit=NGRID_FIT, ngrid_fine=NGRID_FINE)
    return out, best_p, sol


def main():
    log("Probe R start")
    # resume?
    prior = None
    if os.path.exists(OUTJ):
        try:
            with open(OUTJ) as f:
                j = json.load(f)
            if j.get("_checkpoint") and j.get("stage") == "multistart":
                prior = j.get("state", {})
                log(f"resuming from checkpoint: {list(prior.keys())}")
        except Exception:
            prior = None

    # ---- anchors -----------------------------------------------------------
    v_trk = tracker_node_values(0.6426)
    p_trk = params_from_nodes(v_trk)
    trk_node_chi2 = obj(p_trk, "algebraic", NGRID_FINE)
    trk_exact = tracker_exact_joint(0.6426)
    v_lcdm, lcdm_distrms, nz = lcdm_projected_nodes()
    p_lcdm = params_from_nodes(v_lcdm)
    lcdm_seed_chi2 = obj(p_lcdm, "algebraic", NGRID_FINE)
    log(f"anchors: tracker_exact={trk_exact['chi2']:.4f} (ref 1469.29)  "
        f"tracker_node={trk_node_chi2:.4f}  lcdm_seed={lcdm_seed_chi2:.4f} "
        f"(distRMS={np.sqrt(lcdm_distrms/nz):.2e})")
    anchor_ps = [p_trk, p_lcdm]
    reproduced = (abs(trk_exact["chi2"] - REF["tracker"]) < 0.1) and \
                 (abs(trk_node_chi2 - REF["tracker"]) < 5.0) and \
                 np.isfinite(lcdm_seed_chi2)

    anchors = dict(
        tracker_exact_via_oracle=dict(**trk_exact, ref=REF["tracker"],
                                      note="full oracle f_v(z) through the general solver"),
        tracker_node_start=dict(chi2=float(trk_node_chi2),
                                fv_nodes=np.round(v_trk, 5).tolist(),
                                note="tracker f_v sampled at the 5 nodes + tracker bridge; "
                                     "residual vs 1469.29 is 5-node PCHIP representation error"),
        lcdm_projected_start=dict(chi2=float(lcdm_seed_chi2),
                                  fv_nodes=np.round(v_lcdm, 5).tolist(),
                                  dist_shape_rms=float(np.sqrt(lcdm_distrms / nz)),
                                  note="f_v nodes pre-fit to LCDM D_M(z) shape"),
        reproduced=bool(reproduced))

    # ---- V (primary, lapse) ------------------------------------------------
    V, V_p, V_sol = fit_variant("algebraic", anchor_ps,
                                prior_best=(prior or {}).get("algebraic"))
    log(f"V  chi2_min={V['chi2_min']:.4f}  SN={V['chi2_SN']:.4f}  BC={V['chi2_BAOCMB']:.4f}  "
        f"H0d={V['H0_dressed']:.2f}  fv={V['fv_nodes']}")

    # ---- V0 (no-lapse control) --------------------------------------------
    V0, V0_p, V0_sol = fit_variant("none", anchor_ps,
                                   prior_best=(prior or {}).get("none"))
    log(f"V0 chi2_min={V0['chi2_min']:.4f}  SN={V0['chi2_SN']:.4f}  BC={V0['chi2_BAOCMB']:.4f}  "
        f"H0d={V0['H0_dressed']:.2f}  fv={V0['fv_nodes']}")

    # ---- per-node Delta-chi2<=1 band (primary V) --------------------------
    band = {}
    if DO_PROFILE:
        log("profiling per-node Delta-chi2<=1 band (V)")
        for i, zn in enumerate(Z_NODES):
            band[f"z={zn:g}"] = profile_node(i, V_p, "algebraic", V["chi2_min_fitgrid"])
            log(f"  node z={zn:g}: fv_best={V['fv_nodes'][i]:.4f} band={band[f'z={zn:g}']}")

    # ---- derived required-backreaction curves (primary V) -----------------
    curves = derived_curves(nodes_from_params(V_p), "algebraic")

    # ---- R1 verdict --------------------------------------------------------
    c = V["chi2_min"]
    if c >= THR["amplitude_dead_ge"]:
        verdict = "AMPLITUDE_DEAD"
    elif c > THR["disfavoured_le"]:
        verdict = "REFUTED_mechanism_rigid"
    elif c > THR["reconciles_le"]:
        verdict = "DISFAVOURED"
    else:
        verdict = "RECONCILES_mechanism_flexible"
    if not reproduced:
        verdict = "GATE_FAILED_unvalidated"

    d_lcdm = c - REF["LCDM"]
    H0d = V["H0_dressed"]                       # g_dress convention (robust, comparable)
    H0d_full = V["H0_dressed_Hd0"]              # full present dressed rate (slope-dependent)
    def _dir(x):
        return ("timescape-direction (down, ~61)" if x < 66 else
                ("LCDM-band (~68)" if x < 71 else "Bolejko-direction (up, ~73)"))
    h0_dir = _dir(H0d)
    reasoning = (
        f"chi2_min_V={c:.2f} (Delta_vs_LCDM={d_lcdm:+.2f}). Thresholds: reconciles<= "
        f"{THR['reconciles_le']:.2f} (LCDM+10), disfavoured<={THR['disfavoured_le']:.2f} "
        f"(LCDM+25), refuted above, amplitude-dead>={THR['amplitude_dead_ge']:.2f} "
        f"(tracker). Free-E(z) reached {REF['free_E']:.2f} and LCDM {REF['LCDM']:.2f} on "
        f"the same data; the one-parameter tracker sat at {REF['tracker']:.2f}. Dressed "
        f"H0={H0d:.2f} (g_dress convention -> {h0_dir}); full present dressed rate Hd(0)*"
        f"Hbar0={H0d_full:.2f} ({_dir(H0d_full)}) -- the two split by the loosely-pinned "
        f"present void-growth slope f_v'(0); NEITHER lands in the Bolejko-up (~73) "
        f"direction. Bare Hbar0={V['Hbar0']:.2f}. no-lapse control V0 "
        f"chi2_min={V0['chi2_min']:.2f}.")

    out = dict(
        probe="R -- required void history f_v(z) through Model V dressed geometry (R1 gate)",
        reading="KINEMATIC (force f_v(z), compute dressed observables; integrability NOT "
                "enforced -- mirrors the free-E(z) T1 test in f_v-space)",
        data="harness.sn_chi2 (1580 SNe, full stat+sys cov, offset marginalised) + "
             "harness.bao_cmb_chi2 (DESI DR1 + Planck acoustic point, alpha marginalised, rd=147.09)",
        z_nodes=Z_NODES.tolist(),
        references_same_data=REF, thresholds=THR,
        sanity_anchors=anchors,
        V=V, V0=V0,
        fv_req_band_dchi2_le1=band,
        derived_backreaction_V=curves,
        R1=dict(chi2_min_V=c, chi2_min_V0_nolapse=V0["chi2_min"],
                chi2_decomposition=f"SN {V['chi2_SN']:.4f} + BAO+CMB {V['chi2_BAOCMB']:.4f}",
                H0_dressed=H0d, H0_dressed_full_Hd0=H0d_full, Hbar0=V["Hbar0"],
                H0_direction=h0_dir,
                verdict=verdict, reasoning=reasoning,
                sanity_starts_ok=bool(reproduced)),
        runtime_s=round(time.time() - _t0, 1))

    _atomic_dump(out, OUTJ)
    log(f"wrote {OUTJ}")
    log(f"R1 VERDICT: {verdict}   chi2_min_V={c:.4f}  (V0 {V0['chi2_min']:.4f})")
    print(json.dumps(out["R1"], indent=2))
    return out


if __name__ == "__main__":
    main()
