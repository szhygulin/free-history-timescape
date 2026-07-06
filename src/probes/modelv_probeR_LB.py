#!/usr/bin/env python3
"""Probe R (LB reading) -- the REQUIRED void history f_v(z) through the Model V
dressed geometry under the RATE-RATIO lapse (LB), the kinematic lapse-B systematic.

This is the LB-reading sibling of `modelv_probeR.py` (which fits the ALGEBRAIC lapse
LA, the adopted reading). SAME data, SAME nodes, SAME thresholds, SAME monotone
transform and multistart machinery -- the ONLY change is lapse="rate_ratio":

    gamma_bar = Hbar/H_w = 1 + tau f_v'/(2(1-f_v))   (Wiltshire09 Eq 14 / DNW13 Eq 16),

an f_v'-DEPENDENT lapse whose dressed-redshift map is solved in z by a backward RK4 with
a self-consistent gamma_bar0 shot (modelv_theory._solve_rate_ratio; NOTES sec 3, LB).
LB COINCIDES with LA on the tracker to ~1e-7 (modelV_lb_gate.json PASS, joint 1469.29),
so the tracker anchor reproduces the committed number; off the tracker LB is a modest
distance systematic whose f_v^req band this probe measures.

Only ONE variant is fit here (V under rate_ratio). The no-lapse control V0 is lapse-
independent and already lives in the LA run (probes_out/modelV_probeR.json); it is NOT
recomputed. The result is a KINEMATIC (lapse-B) reading -- integrability NOT enforced,
mirroring the free-E(z) T1 test in f_v-space, exactly as the LA Probe R.

Extra field for paper-3 b_pred: `two_scale_excess_z0_LB` = the LB best-fit history's
dressed-void expansion excess E_dress_void = (H_void_app - H_dress)/H_dress at z=0
(bpred_local_excess.two_scale_at_z0 PRIMARY analogue), with gamma_bar and gamma_bar_dot
taken from the LB SOLUTION (gamma_bar0 shot, Hd0 solver value), not the algebraic formula.

Decision gate R1 (same thresholds as LA):
  chi2_min_V <= 1412.24 (LCDM+10)  -> RECONCILES_mechanism_flexible
  1412.24 < chi2_min_V <= 1427.24  -> DISFAVOURED
  chi2_min_V >  1427.24 (LCDM+25)  -> REFUTED_mechanism_rigid
  chi2_min_V >= 1469.29 (tracker)  -> AMPLITUDE_DEAD

Resumable: checkpoints to probes_out/modelV_probeR_LB.ckpt.json every CKPT_EVERY
restarts; soft wall-clock PROBER_MAXSEC exits(3) for relaunch (seifert pattern).

Run from anywhere:  .venv/bin/python src/probes/modelv_probeR_LB.py
Env knobs: PROBER_NGRID_FIT (3000), PROBER_NGRID_FINE (30000), PROBER_NRESTARTS (32),
           PROBER_MAXSEC (0=off), PROBER_SEED (1234), PROBER_PROFILE (1),
           PROBER_DO_DE (0=skip; the structured grid + anchors already span the space).
"""
import os
import sys
import io
import json
import time
import contextlib
import numpy as np
from scipy.optimize import minimize, differential_evolution

np.seterr(all="ignore")   # bad node vectors -> nan/FloatingPointError; obj() maps -> 1e9

# ---- portable paths (derive repo root from __file__; NEVER hardcode /home/...) -----
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(_SRC)

import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):   # these fit+print at import
    import harness as H

OUTJ = os.path.join(_ROOT, "probes_out", "modelV_probeR_LB.json")
CKPT = os.path.join(_ROOT, "probes_out", "modelV_probeR_LB.ckpt.json")

LAPSE = "rate_ratio"   # LB reading -- the ONLY difference from modelv_probeR.py

# ---- committed joint references on the SAME data (identical to the LA script) ------
REF = {"LCDM": 1402.2372, "w0waCDM": 1398.2856, "tracker": 1469.2926, "free_E": 1391.8498}
THR = {"reconciles_le": REF["LCDM"] + 10.0,     # 1412.24
       "disfavoured_le": REF["LCDM"] + 25.0,     # 1427.24
       "amplitude_dead_ge": REF["tracker"]}      # 1469.29

# ---- config -----------------------------------------------------------------------
Z_NODES = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
FLOOR, CEIL = MV._FV_FLOOR, MV._FV_CEIL           # (1e-5, 1-1e-9)
BRIDGE_Z = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])

# LB solve is ~10-45x costlier per eval than LA (pure-Python backward RK4 + gamma_bar0
# shoot). NGRID_FIT=3000 reproduces the tracker joint to ~4e-3 (validated in
# sanity_anchors); NGRID_FINE=30000 (~3e-4) is used only for the headline re-eval,
# the anchors, the derived curves and the z=0 two-scale excess.
NGRID_FIT = int(os.environ.get("PROBER_NGRID_FIT", 3000))    # tracker joint err ~4e-3
NGRID_FINE = int(os.environ.get("PROBER_NGRID_FINE", 30000))  # headline / anchors / two-scale
# per-node Delta-chi2 band grid: a constant grid bias cancels in Delta-chi2, so the band
# can run coarser than the fit. Its chi2_min reference is recomputed on THIS grid so the
# threshold is self-consistent.
NGRID_BAND = int(os.environ.get("PROBER_NGRID_BAND", 1000))
N_RESTARTS = int(os.environ.get("PROBER_NRESTARTS", 12))      # 2 anchors + 8 grid + 2 random
MAXSEC = float(os.environ.get("PROBER_MAXSEC", 0)) or None
SEED = int(os.environ.get("PROBER_SEED", 1234))
DO_PROFILE = os.environ.get("PROBER_PROFILE", "1") == "1"
DO_DE = os.environ.get("PROBER_DO_DE", "0") == "1"
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
# (identical to the LA script -- no hard walls for Nelder-Mead / DE)
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
# forward model: forced f_v(z) nodes -> LB dressed geometry -> joint chi2
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
    """Dressed present H0 for the LB best fit. Two conventions (they SPLIT for a free
    history; on the tracker they coincide, and LB==LA there):

      * gdress  = g_dress(fv0) * Hbar0 -- fv0-only, slope-independent tracker-closure
        convention; the HEADLINE (kept identical to the LA V headline for an apples-to-
        apples R1 comparison).
      * Hd0     = Hd(z=0) * Hbar0 -- the FULL present dressed rate of THIS LB history,
        gamma_bar0*Hbar - dgamma_bar/dt|_0, with the SELF-CONSISTENT LB present lapse
        gamma_bar0 (shot so tau(0)=tau0), NOT the tracker (2+fv0)/2. This is the
        physically-true LB dressed H0; endpoint-slope sensitive."""
    Hbar0 = float(H.H0_from_alpha(a))
    hd0 = float(sol.Hd[np.argmin(np.abs(sol.z))])
    g = float(MV.g_dress(sol.fv0))
    g0lb = float(getattr(sol, "gamma_bar0", np.nan))   # LB self-consistent present lapse
    headline = g * Hbar0
    return dict(H0_dressed=headline, H0_dressed_gdress=g * Hbar0,
                H0_dressed_Hd0=hd0 * Hbar0, Hbar0=Hbar0, g_dress=g, Hd_z0=hd0,
                gamma_bar0_LB=g0lb)


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
    """Gate-style LB anchor: drive the general solver with the full oracle tracker
    f_v(z) under the RATE-RATIO lapse -> must reproduce joint ~1469.29 (LB gate). This
    is the load-bearing standing control: LB==LA on the tracker."""
    trk = MV.tracker_fv_of_z(fv0)
    sol = MV.modelv_solve(trk, Ngrid=NGRID_FINE, lapse=LAPSE)
    tot, csn, cbc, a = joint_from_solution(sol)
    h0 = dressed_H0(sol, a, LAPSE)
    return dict(chi2=tot, chi2_SN=csn, chi2_BAOCMB=cbc, H0_dressed=h0["H0_dressed"])


def lcdm_projected_nodes(Om=0.3048091938075369):
    """f_v(z) nodes pre-fit to the LCDM D_M(z) SHAPE. The shape-match uses the fast
    ALGEBRAIC solve (the seed is a lapse-agnostic f_v-node history); its reported chi2
    is then evaluated under LB. A legitimate LCDM-matched LB seed."""
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
# multistart (resumable): N_RESTARTS Nelder-Mead (+ optional DE polish)
# ---------------------------------------------------------------------------
def build_starts(rng, anchor_ps):
    starts = list(anchor_ps)                                   # anchors first
    grid = [[0.85, 0.78, 0.68, 0.55, 0.40], [0.70, 0.62, 0.52, 0.40, 0.28],
            [0.55, 0.47, 0.38, 0.27, 0.18], [0.95, 0.90, 0.82, 0.70, 0.55],
            [0.40, 0.33, 0.26, 0.18, 0.11], [0.90, 0.70, 0.45, 0.25, 0.12],
            [0.98, 0.85, 0.60, 0.35, 0.18], [0.30, 0.24, 0.18, 0.12, 0.07]]
    for gv in grid:
        starts.append(params_from_nodes(gv))
    while len(starts) < N_RESTARTS:                            # random fill
        starts.append(rng.uniform(-5.0, 5.0, 5))
    return starts[:max(N_RESTARTS, len(anchor_ps) + len(grid))]


_CKPT = {}
def _checkpoint(lapse, state):
    _CKPT[lapse] = state
    _atomic_dump({"_checkpoint": True, "stage": "multistart", "state": _CKPT}, CKPT)


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
    de_used = False
    if DO_DE:
        de = differential_evolution(obj, bounds=[(-6.0, 6.0)] * 5, args=(lapse,),
                                    seed=SEED, maxiter=40, popsize=12, tol=1e-6,
                                    mutation=(0.4, 1.0), recombination=0.8,
                                    polish=True, init="sobol")
        endpoints.append((float(de.fun), de.x.copy()))
        de_used = de.fun < best_chi2
        if de_used:
            best_chi2, best_p = float(de.fun), de.x.copy()
        log(f"  [{lapse}] DE fun={de.fun:.4f}  -> global best={best_chi2:.4f}")
    else:
        log(f"  [{lapse}] DE skipped (PROBER_DO_DE=0); best={best_chi2:.4f}")
    return best_chi2, best_p, endpoints, de_used


# ---------------------------------------------------------------------------
# per-node Delta-chi2 <= 1 profile band (node pinned softly; the other 4 re-optimised
# through the monotone transform so ordering always holds). Soft MAXSEC-aware: on a
# time-out it stops profiling further nodes (this is the LAST stage; a complete JSON is
# still written with the bands computed so far).
# ---------------------------------------------------------------------------
def profile_node(i, best_p, lapse, chi2_min):
    v_best = nodes_from_params(best_p)
    vi0 = v_best[i]
    W = 1.0e6

    def pinned(p, vi):
        v = nodes_from_params(p)
        c = obj(p, lapse, NGRID_BAND)
        return c + W * (v[i] - vi) ** 2

    def chi2_at(vi):
        r = minimize(pinned, best_p, args=(vi,), method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=5e-3, maxiter=1200))
        v = nodes_from_params(r.x)
        return float(joint_nodes(v, lapse, NGRID_BAND)), float(v[i])

    def scan(direction):
        step = max(0.02, 0.05 * vi0)
        prev_v, prev_d = vi0, 0.0
        for k in range(1, 24):
            vi = float(np.clip(vi0 + direction * step * k, FLOOR * 1.5, CEIL * (1 - 1e-6)))
            c, vgot = chi2_at(vi)
            d = c - chi2_min
            if d >= 1.0:
                if d > prev_d:
                    return float(prev_v + (vgot - prev_v) * (1.0 - prev_d) / (d - prev_d))
                return float(vgot)
            prev_v, prev_d = vgot, d
            if vi <= FLOOR * 2 or vi >= CEIL * (1 - 1e-6):
                return float(vgot)
        return float(prev_v)

    lo = scan(-1.0)
    hi = scan(+1.0)
    return [min(lo, hi), max(lo, hi)]


# ---------------------------------------------------------------------------
# derived required-backreaction curves under LB (exact kinematic identities; the
# tau(z) map is the LB solution's, so Hw/Hbar/Q at each z are the LB kinematics)
# ---------------------------------------------------------------------------
def derived_curves(v, lapse=LAPSE):
    fv_cb = MV.fv_from_nodes(np.asarray(v, dtype=float), z_nodes=Z_NODES,
                             bridge_z=BRIDGE_Z, bridge_fv=_bridge_fv(float(v[-1])))
    sol = MV.modelv_solve(fv_cb, lapse=lapse, Ngrid=NGRID_FINE)
    zq = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.8, 2.33])
    tau = np.interp(zq, sol.z, sol.tau)
    fv = fv_cb(zq)
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
        note=("LB (rate-ratio) kinematics: (H_v-H_w) and Q are exact kinematic identities "
              "of the forced f_v(z) (sec 1.1); the tau(z) map is the LB backward-RK4 "
              "solution's, so these differ from the LA curves off the tracker. "
              "'void_expansion_excess'=(H_v-H_w)/<H> is a dimensionless kinematic depth "
              "proxy; a matter-density delta_v(z) is NOT uniquely determined kinematically."))


# ---------------------------------------------------------------------------
# z=0 two-scale dressed-void expansion excess under LB (bpred_local_excess PRIMARY
# analogue). gamma_bar and gamma_bar_dot come from the LB SOLUTION (gamma_bar0 shot,
# Hd0 solver value), NOT the algebraic formula. E_dress_void = gamma_bar (H_v-Hbar)/H_dress
# (gamma_bar_dot cancels), so it needs only gamma_bar0, H_dress=Hd0, and f_v'(tau0).
# ---------------------------------------------------------------------------
def two_scale_at_z0_LB(sol):
    z, tau, fv = sol.z, sol.tau, sol.fv
    fv0 = float(sol.fv0)
    tau0 = float(np.interp(0.0, z, tau))
    dz_dtau = np.gradient(z, tau)
    dfv_dz = np.gradient(fv, z)
    fvp = float(np.interp(0.0, z, dfv_dz * dz_dtau))     # df_v/dtau at z=0 (grid, as LA)
    one_m = max(1.0 - fv0, 1e-9)

    Hw = 2.0 / (3.0 * tau0)                              # H_w/Hbar0
    dHvw = fvp / (3.0 * fv0 * one_m)                     # (H_v - H_w)/Hbar0
    Hbar = Hw + fvp / (3.0 * one_m)                      # <H>/Hbar0
    Hv = Hw + dHvw                                       # H_v/Hbar0

    gam = float(getattr(sol, "gamma_bar0"))              # LB self-consistent present lapse
    Hdress = float(np.interp(0.0, z, sol.Hd))            # Hd(0): dressed present rate (LB)
    gamp = gam * Hbar - Hdress                           # gamma_bar_dot recovered from sol
    Hvoid_app = gam * Hv - gamp                          # dressed void-scale rate

    E_dress_void = (Hvoid_app - Hdress) / Hdress         # PRIMARY (gamma_bar_dot cancels)
    E_bare_void = (Hv - Hbar) / Hbar
    E_contrast = dHvw / Hbar
    E_lapse = gam - 1.0
    return dict(
        fv0=fv0, tau0=tau0, fvp_dtau=fvp,
        Hw_over_Hbar0=Hw, Hv_over_Hbar0=Hv, Hbar_over_Hbar0=Hbar,
        Hv_minus_Hw_over_Hbar0=dHvw,
        gamma_bar0_LB=gam, gamma_bar_dot_LB=gamp,
        Hdress_over_Hbar0=Hdress, Hvoid_app_over_Hbar0=Hvoid_app,
        E_dress_void_PRIMARY=E_dress_void,
        E_bare_void=E_bare_void,
        E_contrast_HvHw_over_Hbar=E_contrast,
        E_lapse_boost=E_lapse,
        note=("LB two-scale excess at z=0; gamma_bar0 (present lapse, shot) and Hd0 "
              "(dressed present rate) taken from the LB solution. E_dress_void_PRIMARY = "
              "gamma_bar0*(H_v-Hbar)/Hd0 is the field paper-3 b_pred reads."))


# ---------------------------------------------------------------------------
def fit_variant(lapse, anchor_ps, prior_best=None):
    rng = np.random.default_rng(SEED)
    starts = build_starts(rng, anchor_ps)
    log(f"[{lapse}] multistart: {len(starts)} restarts"
        f"{' + 1 DE' if DO_DE else ' (DE skipped)'}")
    best_chi2_fit, best_p, endpoints, de_used = run_multistart(lapse, starts, prior_best)
    v_best = nodes_from_params(best_p)
    tot, csn, cbc, a, sol = joint_nodes(v_best, lapse, NGRID_FINE, parts=True)
    h0 = dressed_H0(sol, a, lapse)
    out = dict(
        lapse=lapse, chi2_min=tot, chi2_min_fitgrid=best_chi2_fit,
        chi2_SN=csn, chi2_BAOCMB=cbc, alpha=a,
        fv0=float(sol.fv0),
        H0_dressed=h0["H0_dressed"], H0_dressed_gdress=h0["H0_dressed_gdress"],
        H0_dressed_Hd0=h0["H0_dressed_Hd0"], Hbar0=h0["Hbar0"],
        g_dress=h0["g_dress"], Hd_z0=h0["Hd_z0"], gamma_bar0_LB=h0["gamma_bar0_LB"],
        z_nodes=Z_NODES.tolist(), fv_nodes=np.round(v_best, 5).tolist(),
        n_restarts=len(starts), de_used=bool(de_used),
        ngrid_fit=NGRID_FIT, ngrid_fine=NGRID_FINE)
    return out, best_p, sol


def main():
    log(f"Probe R LB (lapse={LAPSE}) start  NGRID_FIT={NGRID_FIT} N_RESTARTS={N_RESTARTS} "
        f"MAXSEC={MAXSEC} DO_DE={DO_DE}")
    # resume?
    prior = None
    if os.path.exists(CKPT):
        try:
            with open(CKPT) as f:
                j = json.load(f)
            if j.get("_checkpoint") and j.get("stage") == "multistart":
                prior = j.get("state", {})
                log(f"resuming from checkpoint: {list(prior.keys())} "
                    f"done={prior.get(LAPSE, {}).get('done')}")
        except Exception:
            prior = None

    # ---- anchors -----------------------------------------------------------
    v_trk = tracker_node_values(0.6426)
    p_trk = params_from_nodes(v_trk)
    trk_node_chi2 = obj(p_trk, LAPSE, NGRID_FINE)
    trk_exact = tracker_exact_joint(0.6426)
    v_lcdm, lcdm_distrms, nz = lcdm_projected_nodes()
    p_lcdm = params_from_nodes(v_lcdm)
    lcdm_seed_chi2 = obj(p_lcdm, LAPSE, NGRID_FINE)
    log(f"anchors: tracker_exact_LB={trk_exact['chi2']:.4f} (ref 1469.29)  "
        f"tracker_node={trk_node_chi2:.4f}  lcdm_seed={lcdm_seed_chi2:.4f} "
        f"(distRMS={np.sqrt(lcdm_distrms/nz):.2e})")
    anchor_ps = [p_trk, p_lcdm]

    # Standing control (load-bearing): the tracker joint via the DENSE ORACLE f_v(z)
    # under LB must reproduce ~1469.29 (the LB gate value). The 5-node-sampled tracker
    # (trk_node_chi2) is NOT part of the control: its residual is amplified under LB
    # because the rate-ratio lapse is f_v'-DEPENDENT and the 5-node PCHIP's node
    # derivatives differ from the smooth tracker's (a representation error, ~21 under
    # LB vs ~2 under LA) -- reported as a diagnostic, not a pass/fail.
    tracker_anchor_reproduced = abs(trk_exact["chi2"] - REF["tracker"]) < 0.1
    reproduced = tracker_anchor_reproduced and np.isfinite(lcdm_seed_chi2)

    anchors = dict(
        tracker_exact_via_oracle_LB=dict(**trk_exact, ref=REF["tracker"],
            note="full oracle f_v(z) through the general solver under rate_ratio (LB); "
                 "LB==LA on the tracker, so this reproduces the committed 1469.29"),
        tracker_joint_LB=float(trk_exact["chi2"]),
        tracker_joint_LB_ref=REF["tracker"],
        tracker_joint_LB_abs_err=float(abs(trk_exact["chi2"] - REF["tracker"])),
        tracker_joint_LB_PASS=bool(tracker_anchor_reproduced),
        tracker_node_start=dict(chi2=float(trk_node_chi2),
            fv_nodes=np.round(v_trk, 5).tolist(),
            note="tracker f_v sampled at the 5 nodes + tracker bridge, under LB (a seed / "
                 "diagnostic, NOT the standing control). Its residual vs 1469.29 is the "
                 "5-node PCHIP representation error, AMPLIFIED under LB (~21 vs ~2 under "
                 "LA) because the rate-ratio lapse is f_v'-dependent and the PCHIP node "
                 "derivatives differ from the smooth tracker's. The dense-oracle anchor "
                 "(tracker_exact_via_oracle_LB) is the control and reproduces 1469.29"),
        lcdm_projected_start=dict(chi2=float(lcdm_seed_chi2),
            fv_nodes=np.round(v_lcdm, 5).tolist(),
            dist_shape_rms=float(np.sqrt(lcdm_distrms / nz)),
            note="f_v nodes pre-fit to LCDM D_M(z) shape (algebraic shape-match), "
                 "evaluated under LB"),
        reproduced=bool(reproduced))

    # ---- V (LB, rate-ratio lapse) -----------------------------------------
    V, V_p, V_sol = fit_variant(LAPSE, anchor_ps,
                                prior_best=(prior or {}).get(LAPSE))
    log(f"V(LB) chi2_min={V['chi2_min']:.4f}  SN={V['chi2_SN']:.4f}  BC={V['chi2_BAOCMB']:.4f}  "
        f"H0d={V['H0_dressed']:.2f}  g0_LB={V['gamma_bar0_LB']:.4f}  fv={V['fv_nodes']}")

    # ---- per-node Delta-chi2<=1 band --------------------------------------
    band = {}
    band_time_limited = False
    if DO_PROFILE:
        # band chi2_min reference recomputed at NGRID_BAND so the Delta-chi2 threshold is
        # self-consistent with the band-grid scans (a constant grid bias then cancels).
        chi2_min_band = float(joint_nodes(nodes_from_params(V_p), LAPSE, NGRID_BAND))
        log(f"profiling per-node Delta-chi2<=1 band (V, LB)  "
            f"ngrid_band={NGRID_BAND} chi2_min_band={chi2_min_band:.4f}")
        for i, zn in enumerate(Z_NODES):
            if MAXSEC and time.time() - _t0 > MAXSEC:
                band_time_limited = True
                log(f"  MAXSEC hit before node z={zn:g}; band partial ({len(band)}/5 nodes)")
                break
            band[f"z={zn:g}"] = profile_node(i, V_p, LAPSE, chi2_min_band)
            log(f"  node z={zn:g}: fv_best={V['fv_nodes'][i]:.4f} band={band[f'z={zn:g}']}")

    # ---- derived backreaction curves + z=0 two-scale excess (LB) ----------
    curves = derived_curves(nodes_from_params(V_p), LAPSE)
    two_scale = two_scale_at_z0_LB(V_sol)   # V_sol is at NGRID_FINE

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
    H0d_full = V["H0_dressed_Hd0"]              # full LB present dressed rate

    def _dir(x):
        return ("timescape-direction (down, ~61)" if x < 66 else
                ("LCDM-band (~68)" if x < 71 else "Bolejko-direction (up, ~73)"))
    h0_dir = _dir(H0d)
    reasoning = (
        f"LB (rate-ratio lapse) reading. chi2_min_V={c:.2f} (Delta_vs_LCDM={d_lcdm:+.2f}). "
        f"Thresholds: reconciles<={THR['reconciles_le']:.2f} (LCDM+10), disfavoured<="
        f"{THR['disfavoured_le']:.2f} (LCDM+25), refuted above, amplitude-dead>="
        f"{THR['amplitude_dead_ge']:.2f} (tracker). Free-E(z) reached {REF['free_E']:.2f} "
        f"and LCDM {REF['LCDM']:.2f} on the same data; the one-parameter tracker sat at "
        f"{REF['tracker']:.2f}. Dressed H0={H0d:.2f} (g_dress convention -> {h0_dir}); "
        f"full LB present dressed rate Hd(0)*Hbar0={H0d_full:.2f} ({_dir(H0d_full)}) with "
        f"self-consistent present lapse gamma_bar0_LB={V['gamma_bar0_LB']:.4f} (vs tracker "
        f"(2+fv0)/2={(2.0+V['fv0'])/2.0:.4f}). Bare Hbar0={V['Hbar0']:.2f}. Standing "
        f"control: tracker joint under LB={trk_exact['chi2']:.4f} (ref {REF['tracker']}, "
        f"PASS={tracker_anchor_reproduced}).")

    out = dict(
        probe="R (LB reading) -- required void history f_v(z) through Model V dressed "
              "geometry under the rate-ratio lapse (R1 gate)",
        reading="KINEMATIC (lapse-B / rate-ratio): force f_v(z), compute dressed "
                "observables; integrability NOT enforced -- the LB sibling of the LA "
                "Probe R (modelV_probeR.json)",
        lapse=LAPSE,
        data="harness.sn_chi2 (1580 SNe, full stat+sys cov, offset marginalised) + "
             "harness.bao_cmb_chi2 (DESI DR1 + Planck acoustic point, alpha marginalised, "
             "rd=147.09)",
        z_nodes=Z_NODES.tolist(),
        references_same_data=REF, thresholds=THR,
        ngrid_fit=NGRID_FIT, ngrid_band=NGRID_BAND, ngrid_fine=NGRID_FINE,
        n_restarts=V["n_restarts"],
        time_limited=bool(band_time_limited),
        time_limited_note=("false = the multistart ran all restarts to completion and the "
                           "per-node band finished; true = the wall-clock (PROBER_MAXSEC) "
                           "cut the band short (bands computed so far are reported, the rest "
                           "omitted) -- reported at full volume per program rule"),
        sanity_anchors=anchors,
        V=V,
        fv_req_band_dchi2_le1=band,
        fv_req_band_time_limited=bool(band_time_limited),
        derived_backreaction_V=curves,
        two_scale_excess_z0_LB=float(two_scale["E_dress_void_PRIMARY"]),
        two_scale_z0_LB=two_scale,
        R1=dict(chi2_min_V=c,
                chi2_decomposition=f"SN {V['chi2_SN']:.4f} + BAO+CMB {V['chi2_BAOCMB']:.4f}",
                H0_dressed=H0d, H0_dressed_full_Hd0=H0d_full, Hbar0=V["Hbar0"],
                gamma_bar0_LB=V["gamma_bar0_LB"], H0_direction=h0_dir,
                verdict=verdict, reasoning=reasoning,
                sanity_starts_ok=bool(reproduced)),
        runtime_s=round(time.time() - _t0, 1))

    _atomic_dump(out, OUTJ)
    log(f"wrote {OUTJ}")
    log(f"R1 VERDICT (LB): {verdict}   chi2_min_V={c:.4f}")
    log(f"two_scale_excess_z0_LB (E_dress_void PRIMARY) = {two_scale['E_dress_void_PRIMARY']:.5f}")
    print(json.dumps(out["R1"], indent=2))
    return out


if __name__ == "__main__":
    main()
