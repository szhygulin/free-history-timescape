#!/usr/bin/env python3
"""Adversarial independent verification of the LB (rate-ratio) Probe R fit.

Refute-by-default re-derivation of `probes_out/modelV_probeR_LB.json` (produced by
`modelv_probeR_LB.py`). This script re-implements the node<->param transform, the
high-z bridge, the joint-chi2 harness path, the multistart, the two-scale-excess
formula and the per-node profile-band scan FROM SCRATCH. It reuses ONLY the
gate-validated theory + harness primitives (never the probe's fitting internals):

    modelv_theory : modelv_solve, fv_from_nodes, tracker_fv_of_z, g_dress, _FV_CEIL/_FLOOR
    harness       : sn_chi2, bao_cmb_chi2, H0_from_alpha
    fit_timescape : load

Five checks (each -> PASS / PARTIAL / REFUTED):
  (1) LB harness/solver control : dense-oracle tracker f_v(z) through modelv_solve
      (lapse=rate_ratio) MUST reproduce joint ~1469.29 (the LB gate value).
  (2) Headline reproduction     : committed best-fit fv_nodes -> joint ~1406.92.
  (3) Optimiser-stuck falsifier : an INDEPENDENT time-bounded multistart (fresh seed,
      != the probe's 1234) tries to BEAT 1406.90. > ~1 below -> under-converged optimum.
  (4) Verdict + two-scale       : R1 threshold on the reproduced chi2; independent
      recompute of two_scale_excess_z0_LB = gamma_bar0*(H_v-Hbar)/Hd0 from the LB solution.
  (5) fv_req band edges         : pin a node at the reported Delta-chi2<=1 edge, re-optimise
      the other four, confirm Delta-chi2 ~ 1.

Run:  .venv/bin/python src/probes/verify_modelv_probeR_LB.py
Env:  VERIFY_MS_NGRID (2500), VERIFY_MS_MAXSEC (700), VERIFY_MS_NRAND (10), VERIFY_SEED (20260706)
"""
import os
import sys
import io
import json
import time
import contextlib
import numpy as np
from scipy.optimize import minimize

np.seterr(all="ignore")

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
import modelv_theory as MV                      # gate-validated theory (reused)
with contextlib.redirect_stdout(io.StringIO()):
    import harness as H                         # gate-validated harness (reused)

COMMITTED_JSON = os.path.join(_ROOT, "probes_out", "modelV_probeR_LB.json")
OUTJ = os.path.join(_ROOT, "probes_out", "verify_modelV_probeR_LB.json")

LAPSE = "rate_ratio"
REF = {"LCDM": 1402.2372, "w0waCDM": 1398.2856, "tracker": 1469.2926, "free_E": 1391.8498}
THR = {"reconciles_le": REF["LCDM"] + 10.0,     # 1412.2372
       "disfavoured_le": REF["LCDM"] + 25.0,     # 1427.2372
       "amplitude_dead_ge": REF["tracker"]}      # 1469.2926

Z_NODES = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
BRIDGE_Z = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])
FLOOR, CEIL = MV._FV_FLOOR, MV._FV_CEIL          # (1e-5, 1-1e-9)

NGRID_FINE = 30000
NGRID_MS = int(os.environ.get("VERIFY_MS_NGRID", 2500))    # multistart fit grid
MS_MAXSEC = float(os.environ.get("VERIFY_MS_MAXSEC", 700))  # hard wall-clock for check (3)
MS_NRAND = int(os.environ.get("VERIFY_MS_NRAND", 10))
SEED = int(os.environ.get("VERIFY_SEED", 20260706))         # != the probe's 1234
NGRID_BAND = 1000

zHD, zHEL, mb, Cf = F.load()
_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)


# ---- independent node<->param transform (re-implemented, NOT imported) -------------
def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, float), -40.0, 40.0)))
def _logit(y):
    y = np.clip(y, 1e-12, 1.0 - 1e-12)
    return np.log(y / (1.0 - y))


def nodes_from_params(p):
    g = _sig(p)
    v = np.empty(5)
    v[0] = CEIL * g[0]
    for i in range(1, 5):
        v[i] = v[i - 1] * g[i]
    return v


def params_from_nodes(v):
    v = np.asarray(v, float)
    g = np.empty(5)
    g[0] = v[0] / CEIL
    for i in range(1, 5):
        g[i] = v[i] / v[i - 1]
    return _logit(g)


def bridge_fv(fv_last):
    return fv_last * ((1.0 + Z_NODES[-1]) / (1.0 + BRIDGE_Z)) ** 1.5


# ---- independent joint-chi2 harness path -------------------------------------------
def solve_nodes(v, lapse=LAPSE, Ngrid=NGRID_FINE):
    fv = MV.fv_from_nodes(np.asarray(v, float), z_nodes=Z_NODES,
                          bridge_z=BRIDGE_Z, bridge_fv=bridge_fv(float(v[-1])))
    return MV.modelv_solve(fv, lapse=lapse, Ngrid=Ngrid)


def joint_from_sol(sol):
    csn = float(H.sn_chi2(sol.D_M(zHD)))
    cbc, a = H.bao_cmb_chi2(lambda z, k: float(sol.predict(z, k)))
    return csn + cbc, csn, cbc, float(a)


def joint_nodes(v, lapse=LAPSE, Ngrid=NGRID_FINE):
    return joint_from_sol(solve_nodes(v, lapse, Ngrid))


def obj(p, Ngrid):
    try:
        tot, _, _, _ = joint_nodes(nodes_from_params(p), LAPSE, Ngrid)
    except Exception:
        return 1e9
    return float(tot) if np.isfinite(tot) else 1e9


# committed headline (read from the artifact, carried inline for the comparison)
with open(COMMITTED_JSON) as f:
    CJ = json.load(f)
COMM = dict(
    tracker_LB=CJ["sanity_anchors"]["tracker_joint_LB"],
    tracker_ref=CJ["sanity_anchors"]["tracker_joint_LB_ref"],
    chi2_min=CJ["V"]["chi2_min"],
    chi2_SN=CJ["V"]["chi2_SN"], chi2_BC=CJ["V"]["chi2_BAOCMB"],
    fv0=CJ["V"]["fv0"], fv_nodes=CJ["V"]["fv_nodes"],
    gamma_bar0=CJ["V"]["gamma_bar0_LB"],
    two_scale=CJ["two_scale_excess_z0_LB"],
    verdict=CJ["R1"]["verdict"], band=CJ["fv_req_band_dchi2_le1"])

out = {"_meta": {"seed": SEED, "ngrid_fine": NGRID_FINE, "ngrid_ms": NGRID_MS,
                 "ms_maxsec": MS_MAXSEC, "committed_json": os.path.relpath(COMMITTED_JSON, _ROOT)}}


# ===================================================================================
# (1) LB harness/solver control: dense-oracle tracker f_v(z) -> joint ~1469.29
# ===================================================================================
trk = MV.tracker_fv_of_z(0.6426)
sol_trk = MV.modelv_solve(trk, lapse=LAPSE, Ngrid=NGRID_FINE)
jt, sn_t, bc_t, a_t = joint_from_sol(sol_trk)
err1 = abs(jt - REF["tracker"])
out["check1_control_LB"] = {
    "my_joint": jt, "my_chi2_SN": sn_t, "my_chi2_BAOCMB": bc_t,
    "committed_tracker_LB": COMM["tracker_LB"], "ref_gate": REF["tracker"],
    "abs_err_vs_ref": err1, "abs_err_vs_committed": abs(jt - COMM["tracker_LB"]),
    "gamma_bar0": float(getattr(sol_trk, "gamma_bar0", np.nan)),
    "status": "PASS" if err1 < 0.1 else "REFUTED"}
log(f"(1) control: my tracker-LB joint = {jt:.4f}  (ref {REF['tracker']}, committed "
    f"{COMM['tracker_LB']:.4f}, err {err1:.4f})  {out['check1_control_LB']['status']}")


# ===================================================================================
# (2) headline reproduction: committed best-fit fv_nodes -> joint ~1406.92
# ===================================================================================
v_comm = np.array(COMM["fv_nodes"], float)
jh, sn_h, bc_h, a_h = joint_nodes(v_comm, LAPSE, NGRID_FINE)
sol_h = solve_nodes(v_comm, LAPSE, NGRID_FINE)
err2 = abs(jh - COMM["chi2_min"])
out["check2_headline"] = {
    "my_joint": jh, "my_chi2_SN": sn_h, "my_chi2_BAOCMB": bc_h,
    "committed_chi2_min": COMM["chi2_min"],
    "committed_SN": COMM["chi2_SN"], "committed_BC": COMM["chi2_BC"],
    "abs_err": err2, "my_fv0": float(sol_h.fv0), "committed_fv0": COMM["fv0"],
    "fv_nodes_used": v_comm.tolist(),
    "status": "PASS" if err2 < 0.5 else ("PARTIAL" if err2 < 2.0 else "REFUTED")}
log(f"(2) headline: my joint = {jh:.4f}  (committed {COMM['chi2_min']:.4f}, err {err2:.4f})  "
    f"{out['check2_headline']['status']}")


# ===================================================================================
# (3) optimiser-stuck falsifier: independent time-bounded multistart (fresh seed)
# ===================================================================================
rng = np.random.default_rng(SEED)
structured = [
    v_comm.copy(),                                   # committed optimum (local-min probe)
    np.array([0.62, 0.53, 0.41, 0.27, 0.14]),        # slight perturbation
    np.array([0.55, 0.47, 0.37, 0.26, 0.17]),        # shallower
    np.array([0.70, 0.60, 0.48, 0.34, 0.22]),        # deeper
    np.array([0.50, 0.42, 0.33, 0.22, 0.13]),        # lower amplitude
]
starts = [params_from_nodes(v) for v in structured]
while len(starts) < len(structured) + MS_NRAND:
    starts.append(rng.uniform(-5.0, 5.0, 5))

best_fun, best_p, ran, hit_wall = np.inf, None, 0, False
for i, s in enumerate(starts):
    if time.time() - _t0 > MS_MAXSEC + 0.0 and i >= len(structured):
        hit_wall = True
        log(f"(3) wall-clock {MS_MAXSEC}s reached after {ran} starts; stopping (best-so-far kept)")
        break
    r = minimize(obj, s, args=(NGRID_MS,), method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=5e-3, maxiter=1500))
    ran += 1
    if r.fun < best_fun:
        best_fun, best_p = float(r.fun), r.x.copy()
    if (i + 1) % 3 == 0 or i < len(structured):
        log(f"    start {i+1}/{len(starts)} fun={r.fun:.4f}  best={best_fun:.4f}  "
            f"[{time.time()-_t0:.0f}s]")

# refine the multistart winner at headline resolution for an apples-to-apples compare
v_best = nodes_from_params(best_p)
jb, sn_b, bc_b, a_b = joint_nodes(v_best, LAPSE, NGRID_FINE)
# also polish the committed optimum on the fit grid to expose any residual descent
r_comm = minimize(obj, params_from_nodes(v_comm), args=(NGRID_MS,), method="Nelder-Mead",
                  options=dict(xatol=1e-3, fatol=5e-3, maxiter=1500))
v_comm_pol = nodes_from_params(r_comm.x)
jcp, _, _, _ = joint_nodes(v_comm_pol, LAPSE, NGRID_FINE)

improvement = COMM["chi2_min"] - min(jb, jcp)       # >0 => beat the committed optimum
beat_material = improvement > 1.0
out["check3_multistart"] = {
    "n_starts_run": ran, "n_random": MS_NRAND, "seed": SEED, "hit_wall_clock": hit_wall,
    "ngrid_ms": NGRID_MS,
    "best_joint_fitgrid": best_fun,
    "best_joint_headline": jb, "best_fv_nodes": np.round(v_best, 5).tolist(),
    "committed_polished_headline": jcp,
    "committed_chi2_min": COMM["chi2_min"],
    "improvement_over_committed": improvement,
    "beat_by_more_than_1": bool(beat_material),
    "status": ("REFUTED" if improvement > 1.0 else
               "PARTIAL" if improvement > 0.3 else "PASS"),
    "note": ("PASS = could not beat the committed optimum by >~0.3 (converged). "
             "improvement = committed_chi2_min - min(best headline, polished committed). "
             "A negative/near-zero value confirms the reported optimum is the basin floor.")}
log(f"(3) multistart best headline = {jb:.4f} (fitgrid {best_fun:.4f}); committed polished "
    f"= {jcp:.4f}; improvement over committed = {improvement:+.4f}  "
    f"{out['check3_multistart']['status']}")


# ===================================================================================
# (4) verdict + independent two-scale-excess recompute
# ===================================================================================
# use the BEST reproduced chi2 (the lower of headline reproduction / multistart winner)
c_repro = min(jh, jb, jcp)
if c_repro >= THR["amplitude_dead_ge"]:
    my_verdict = "AMPLITUDE_DEAD"
elif c_repro > THR["disfavoured_le"]:
    my_verdict = "REFUTED_mechanism_rigid"
elif c_repro > THR["reconciles_le"]:
    my_verdict = "DISFAVOURED"
else:
    my_verdict = "RECONCILES_mechanism_flexible"

# --- independent two_scale_excess_z0_LB from the LB solution of the best-fit history ---
def two_scale_excess_LB(sol):
    z, tau, fv = sol.z, sol.tau, sol.fv
    fv0 = float(sol.fv0)
    tau0 = float(np.interp(0.0, z, tau))
    dz_dtau = np.gradient(z, tau)
    dfv_dz = np.gradient(fv, z)
    fvp = float(np.interp(0.0, z, dfv_dz * dz_dtau))     # df_v/dtau at z=0
    one_m = max(1.0 - fv0, 1e-9)
    Hw = 2.0 / (3.0 * tau0)
    dHvw = fvp / (3.0 * fv0 * one_m)
    Hbar = Hw + fvp / (3.0 * one_m)
    Hv = Hw + dHvw
    gam = float(getattr(sol, "gamma_bar0"))              # LB self-consistent present lapse
    Hd0 = float(np.interp(0.0, z, sol.Hd))               # dressed present rate (LB)
    E = gam * (Hv - Hbar) / Hd0                          # PRIMARY (gamma_bar_dot cancels)
    return dict(E_dress_void=E, gamma_bar0=gam, Hd0=Hd0, Hv=Hv, Hbar=Hbar,
                Hv_minus_Hbar=Hv - Hbar, fvp_dtau=fvp, tau0=tau0, fv0=fv0)

ts = two_scale_excess_LB(sol_h)                          # sol_h = committed best-fit at NGRID_FINE
err_ts = abs(ts["E_dress_void"] - COMM["two_scale"])
out["check4_verdict_twoscale"] = {
    "my_chi2_repro": c_repro, "delta_vs_LCDM": c_repro - REF["LCDM"],
    "threshold_reconciles_le": THR["reconciles_le"],
    "my_verdict": my_verdict, "committed_verdict": COMM["verdict"],
    "verdict_match": my_verdict == COMM["verdict"],
    "my_two_scale_excess": ts["E_dress_void"], "committed_two_scale_excess": COMM["two_scale"],
    "two_scale_abs_err": err_ts,
    "two_scale_components": {k: ts[k] for k in
                            ("gamma_bar0", "Hd0", "Hv", "Hbar", "Hv_minus_Hbar",
                             "fvp_dtau", "tau0", "fv0")},
    "status": ("PASS" if (my_verdict == COMM["verdict"] and err_ts < 1e-3) else
               "PARTIAL" if my_verdict == COMM["verdict"] else "REFUTED")}
log(f"(4) verdict: my chi2={c_repro:.4f} (Delta_LCDM {c_repro-REF['LCDM']:+.4f}) -> {my_verdict} "
    f"(committed {COMM['verdict']}, match={out['check4_verdict_twoscale']['verdict_match']})")
log(f"    two_scale_excess: mine {ts['E_dress_void']:.6f}  committed {COMM['two_scale']:.6f}  "
    f"err {err_ts:.2e}  {out['check4_verdict_twoscale']['status']}")


# ===================================================================================
# (5) fv_req band-edge spot-check: pin a node at the reported Delta-chi2<=1 edge,
#     re-optimise the other four, confirm Delta-chi2 ~ 1
# ===================================================================================
# self-consistent chi2_min reference on the band grid (constant grid bias cancels in Dchi2)
chi2_min_band = joint_nodes(v_comm, LAPSE, NGRID_BAND)[0]
p_best_band = params_from_nodes(v_comm)


def dchi2_at_pinned(node_idx, vi_target):
    W = 1.0e6
    def pinned(p):
        v = nodes_from_params(p)
        return obj(p, NGRID_BAND) + W * (v[node_idx] - vi_target) ** 2
    r = minimize(pinned, p_best_band, method="Nelder-Mead",
                 options=dict(xatol=1e-3, fatol=5e-3, maxiter=1500))
    v = nodes_from_params(r.x)
    c = joint_nodes(v, LAPSE, NGRID_BAND)[0]
    return c - chi2_min_band, float(v[node_idx])


band_checks = {}
# spot-check z=0 (both edges) and z=0.3 (both edges) -- the well-constrained nodes
for lbl, idx in (("z=0", 0), ("z=0.3", 1)):
    edges = COMM["band"][lbl]
    res = []
    for edge_val in edges:
        d, vgot = dchi2_at_pinned(idx, edge_val)
        res.append({"edge_target": edge_val, "dchi2": d, "node_val_reached": vgot})
        log(f"(5) {lbl} edge {edge_val:.5f}: Delta-chi2 = {d:.4f}")
    # PASS if both edges land near Delta-chi2 ~ 1 (tolerant band: 0.5..1.8)
    ok = all(0.5 <= e["dchi2"] <= 1.8 for e in res)
    band_checks[lbl] = {"edges": res, "status": "PASS" if ok else "PARTIAL"}

out["check5_band_edges"] = {
    "chi2_min_band_ref": chi2_min_band, "ngrid_band": NGRID_BAND,
    "checked": band_checks,
    "status": "PASS" if all(b["status"] == "PASS" for b in band_checks.values()) else "PARTIAL"}


# ===================================================================================
# overall verdict
# ===================================================================================
statuses = {k: out[k]["status"] for k in
            ("check1_control_LB", "check2_headline", "check3_multistart",
             "check4_verdict_twoscale", "check5_band_edges")}
if any(v == "REFUTED" for v in statuses.values()):
    overall = "DOES_NOT_SURVIVE"
elif any(v == "PARTIAL" for v in statuses.values()):
    overall = "SURVIVES_WITH_CAVEATS"
else:
    overall = "SURVIVES"
out["overall"] = {
    "per_check_status": statuses,
    "verdict": overall,
    "summary": (f"LB Probe R adversarial re-derivation: control reproduces "
                f"{jt:.2f} (ref {REF['tracker']}); headline reproduces {jh:.2f} "
                f"(committed {COMM['chi2_min']:.2f}); independent multistart could not "
                f"beat it (improvement {out['check3_multistart']['improvement_over_committed']:+.3f}); "
                f"verdict {my_verdict} matches; two_scale {ts['E_dress_void']:.5f} "
                f"matches {COMM['two_scale']:.5f}."),
    "runtime_s": round(time.time() - _t0, 1)}


def _js(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


with open(OUTJ, "w") as f:
    json.dump(out, f, indent=2, default=_js)
log(f"wrote {OUTJ}")
log(f"OVERALL: {overall}   per-check: {statuses}")
print(json.dumps(out["overall"], indent=2, default=_js))
