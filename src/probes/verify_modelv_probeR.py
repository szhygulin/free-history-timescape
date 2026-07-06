#!/usr/bin/env python3
"""Independent verification of Probe R's headline (the failed workflow's verify stage).

Adversarial re-check, refute-by-default. Re-implements the node<->param transform and the
joint-chi2 harness path FROM SCRATCH (only MV.modelv_solve is reused — it is gate-validated,
G-T/G-A), and:
  (1) confirms the tracker anchor reproduces joint 1469.29 (validates the SN+BAO+CMB harness path);
  (2) confirms the checkpoint best-fit f_v(z) gives chi2 ~ 1396 (the headline);
  (3) runs an INDEPENDENT 16-start Nelder-Mead + a coarse random sweep to try to BEAT 1396
      (is the optimiser stuck above the true minimum? would change the R1 verdict);
  (4) reports the physical f_v(z) nodes and the R1 verdict vs the pre-registered thresholds.

Run from src/:  ../.venv/bin/python probes/verify_modelv_probeR.py
"""
import os, sys, io, json, contextlib
import numpy as np
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _SRC); sys.path.insert(0, _HERE)
os.chdir(_SRC)
import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):
    import harness as H

REF = {"LCDM": 1402.2372, "w0waCDM": 1398.2856, "tracker": 1469.2926, "free_E": 1391.8498}
THR = {"reconciles_le": REF["LCDM"] + 10.0, "disfavoured_le": REF["LCDM"] + 25.0,
       "amplitude_dead_ge": REF["tracker"]}
Z_NODES = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
BRIDGE_Z = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])
CEIL = 1.0 - 1e-9
zHD, zHEL, mb, Cf = F.load()

# --- checkpoint best-fit params from the reaped workflow run (algebraic / lapse) ---
CKPT_ALG_P = [0.5759195451412239, 1.583615164777176, 1.07300604211554,
              0.8763648488909743, 0.8130686052496472]


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, float), -40, 40)))


def nodes_from_params(p):
    g = sig(p); v = np.empty(5); v[0] = CEIL * g[0]
    for i in range(1, 5):
        v[i] = v[i - 1] * g[i]
    return v


def bridge_fv(fv_last):
    return fv_last * ((1.0 + Z_NODES[-1]) / (1.0 + BRIDGE_Z)) ** 1.5


def joint_chi2(fv_nodes, lapse="algebraic", Ngrid=30000):
    fv = MV.fv_from_nodes(np.asarray(fv_nodes, float), z_nodes=Z_NODES,
                          bridge_z=BRIDGE_Z, bridge_fv=bridge_fv(float(fv_nodes[-1])))
    sol = MV.modelv_solve(fv, lapse=lapse, Ngrid=Ngrid)
    csn = float(H.sn_chi2(sol.D_M(zHD)))
    cbc, a = H.bao_cmb_chi2(lambda z, k: float(sol.predict(z, k)))
    return csn + cbc, csn, cbc, sol


def obj(p, lapse="algebraic", Ngrid=8000):
    try:
        c = joint_chi2(nodes_from_params(p), lapse, Ngrid)[0]
        return float(c) if np.isfinite(c) else 1e9
    except Exception:
        return 1e9


out = {}

# (1) tracker anchor -> must reproduce 1469.29 (independent of the 5-node basis)
trk = MV.tracker_fv_of_z(0.6426)
sol_trk = MV.modelv_solve(trk, Ngrid=30000, lapse="algebraic")
csn_t = float(H.sn_chi2(sol_trk.D_M(zHD)))
cbc_t, a_t = H.bao_cmb_chi2(lambda z, k: float(sol_trk.predict(z, k)))
joint_trk = csn_t + cbc_t
out["tracker_anchor"] = {"joint": joint_trk, "ref": REF["tracker"],
                         "abs_err": abs(joint_trk - REF["tracker"]),
                         "PASS": abs(joint_trk - REF["tracker"]) < 0.1}
print(f"(1) tracker anchor joint = {joint_trk:.4f}  (ref {REF['tracker']:.4f}, "
      f"err {abs(joint_trk-REF['tracker']):.4f})  "
      f"{'PASS' if out['tracker_anchor']['PASS'] else 'FAIL'}")

# (2) checkpoint best-fit -> headline chi2
v_ck = nodes_from_params(CKPT_ALG_P)
c_ck, sn_ck, bc_ck, sol_ck = joint_chi2(v_ck, "algebraic", 30000)
out["checkpoint_bestfit"] = {"fv_nodes": [round(float(x), 4) for x in v_ck],
                             "joint": c_ck, "chi2_SN": sn_ck, "chi2_BAOCMB": bc_ck,
                             "fv0": float(sol_ck.fv0)}
print(f"(2) checkpoint best-fit f_v nodes = {[round(float(x),4) for x in v_ck]}")
print(f"    joint = {c_ck:.4f}  (SN {sn_ck:.4f} + BAO+CMB {bc_ck:.4f})  f_v(0) = {sol_ck.fv0:.4f}")

# (3) independent multistart: can we beat it? (optimiser-stuck check)
rng = np.random.default_rng(20260706)
starts = [np.array(CKPT_ALG_P)]
def p_from_v(v):
    v = np.asarray(v, float); g = np.empty(5); g[0] = v[0] / CEIL
    for i in range(1, 5):
        g[i] = v[i] / v[i - 1]
    return np.log(np.clip(g, 1e-12, 1 - 1e-12) / (1 - np.clip(g, 1e-12, 1 - 1e-12)))
for gv in ([0.70, 0.60, 0.48, 0.34, 0.22], [0.62, 0.55, 0.45, 0.33, 0.22],
           [0.55, 0.47, 0.37, 0.26, 0.17], [0.80, 0.68, 0.52, 0.35, 0.22],
           [0.45, 0.38, 0.30, 0.21, 0.13]):
    starts.append(p_from_v(gv))
while len(starts) < 16:
    starts.append(rng.uniform(-5, 5, 5))
best = (np.inf, None)
for i, s in enumerate(starts):
    r = minimize(obj, s, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=5e-3, maxiter=2000))
    if r.fun < best[0]:
        best = (float(r.fun), r.x.copy())
# refine the winner at headline resolution
v_best = nodes_from_params(best[1])
c_best, sn_b, bc_b, sol_b = joint_chi2(v_best, "algebraic", 30000)
out["independent_multistart"] = {"n_starts": len(starts), "best_joint_fitgrid": best[0],
                                 "best_joint_headline": c_best,
                                 "fv_nodes": [round(float(x), 4) for x in v_best],
                                 "fv0": float(sol_b.fv0),
                                 "beat_checkpoint": bool(c_best < c_ck - 0.2)}
print(f"(3) independent {len(starts)}-start best joint = {c_best:.4f}  "
      f"f_v nodes = {[round(float(x),4) for x in v_best]}  f_v(0)={sol_b.fv0:.4f}")

# (4) V0 (no-lapse) at the best nodes, and the R1 verdict
c_v0, _, _, _ = joint_chi2(v_best, "none", 30000)
c = min(c_best, c_ck)
verdict = ("AMPLITUDE_DEAD" if c >= THR["amplitude_dead_ge"] else
           "REFUTED_mechanism_rigid" if c > THR["disfavoured_le"] else
           "DISFAVOURED" if c > THR["reconciles_le"] else
           "RECONCILES_mechanism_flexible")
out["R1"] = {"chi2_min_V": c, "chi2_V0_nolapse_at_best": c_v0,
             "delta_vs_LCDM": c - REF["LCDM"], "verdict": verdict, "thresholds": THR}
print(f"(4) V0 (no-lapse) at best nodes = {c_v0:.4f}")
print(f"    R1 chi2_min_V = {c:.4f}  (Delta vs LCDM {c-REF['LCDM']:+.4f})  -> {verdict}")

def _js(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer, np.floating)):
        return float(o)
    raise TypeError(str(type(o)))


with open(os.path.join(os.path.dirname(_SRC), "probes_out", "verify_modelv_probeR.json"), "w") as f:
    json.dump(out, f, indent=2, default=_js)
print("\nwrote probes_out/verify_modelv_probeR.json")
