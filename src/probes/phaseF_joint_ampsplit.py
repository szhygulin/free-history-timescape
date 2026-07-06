#!/usr/bin/env python3
"""Phase F -- DESI DR2 joint refit + amplitude-split under the free-history Model V.

Phase 4/F, `significance-audit` -> paper 2. Two decision-grade results, both on the
SAME DESI DR2 BAO + Planck-CMB data (a DR2 mirror of Probe R, which ran on DR1):

  (1) DR2 JOINT REFIT.  Repeat Probe R's free-f_v(z) joint fit (monotone PCHIP nodes
      at z={0,.3,.7,1.3,2.33} through the Model V dressed geometry) against DESI DR2
      instead of DR1, and refit LCDM and the one-parameter timescape TRACKER on the
      same DR2 data. Honest parameter accounting: the free history has 5 node params
      (4 extra vs LCDM's 1) -> Wilks p for the joint-chi2 improvement (cf the T1 free
      -E(z) spline p=0.065).

  (2) AMPLITUDE-SPLIT (the headline, PLAN_void_history.md sec 0).  FIX the SHAPE of
      the free-fit f_v_req(z) (normalise the DR2 joint best-fit nodes to shape(0)=1),
      introduce ONE amplitude A with f_v(z)=clip(A*shape(z)) -- so A is exactly the
      present void fraction f_v(0). Fit A SEPARATELY to (i) SNe only and (ii) BAO+CMB
      only. The tracker analogue is the 0.85-vs-0.64 f_v0 split at 4.8-6.6 sigma
      (paper 1). Does freeing the history SHAPE make A_SN ~ A_BAOCMB (the split
      DISSOLVES -- the tracker shape was the culprit) or does the amplitude split
      persist (history-shape-independent, amplitude-level conflict)?
      Statistic: profile parameter-shift, sigma = sqrt(delta chi2_join), where
        delta chi2_join = [chi2_SN(A_joint)+chi2_BC(A_joint)]
                          - [min_A chi2_SN(A) + min_A chi2_BC(A)]  (>=0, 1 dof).
      SANITY: the SAME statistic applied to the tracker's own parameter f_v0 (its
      shape==amplitude one-parameter family) must reproduce paper 1's large split.

Data: harness.sn_chi2 (1580 Pantheon+ SNe, full stat+sys cov, M_B/H0 offset
marginalised) + a DR2 bao_cmb_chi2 built HERE that mirrors harness.bao_cmb_chi2
(same alpha=c/(Hbar0 rd) marginalisation, same block-diagonal covariance
construction, same Planck acoustic point value 94.31613=94.316 that src/harness.py
hard-codes) but with the DESI DR2 rows/cov from probes_out/desi_dr2_rows.json.

Forward model reused verbatim from Probe R (modelv_probeR.solve_nodes /
nodes_from_params / params_from_nodes / Z_NODES / BRIDGE_Z / dressed_H0) so the
dressed geometry, node transform and high-z bridge are identical -- only the BAO
dataset changes (DR1 -> DR2).

Resumable: the free-history multistart checkpoints to the output JSON every
CKPT_EVERY restarts; a soft wall-clock PHASEF_MAXSEC exits cleanly for relaunch.

Run from src/:   python probes/phaseF_joint_ampsplit.py
Env knobs: PHASEF_NGRID_FIT (6000), PHASEF_NGRID_FINE (30000),
           PHASEF_NRESTARTS (12), PHASEF_MAXSEC (0=off), PHASEF_SEED (1234).
"""
import os
import sys
import io
import json
import time
import contextlib
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scipy.stats import chi2 as _chi2dist

np.seterr(all="ignore")  # bad node vectors -> nan; objectives map nan -> 1e9

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SRC)

import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):   # these fit + print on import
    import harness as H
    import modelv_probeR as PR                     # reuse Probe R's forward-model glue

_OUTDIR = os.path.join(os.path.dirname(_SRC), "probes_out")
OUTJ = os.path.join(_OUTDIR, "phaseF_joint_ampsplit.json")
DR2J = os.path.join(_OUTDIR, "desi_dr2_rows.json")
PROBERJ = os.path.join(_OUTDIR, "modelV_probeR.json")

# ---- config -----------------------------------------------------------------------
NGRID_FIT = int(os.environ.get("PHASEF_NGRID_FIT", 6000))     # fit resolution
NGRID_FINE = int(os.environ.get("PHASEF_NGRID_FINE", 30000))  # headline re-eval
N_RESTARTS = int(os.environ.get("PHASEF_NRESTARTS", 12))
MAXSEC = float(os.environ.get("PHASEF_MAXSEC", 0)) or None
SEED = int(os.environ.get("PHASEF_SEED", 1234))
CKPT_EVERY = 3
TRK_NTAU_FIT = 90000     # tracker f_v(z) grid for fits (fast); fine re-eval uses 300000

_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)

zHD, zHEL, mb, Cf = F.load()

# committed DR1 joint references (paper 1 / Probe R, SAME machinery, DR1 data)
REF_DR1 = {"LCDM": 1402.2372, "w0waCDM": 1398.2856, "tracker": 1469.2926, "free_E": 1391.8498}


def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# DR2 BAO + Planck-CMB chi2 -- mirror of harness.bao_cmb_chi2 with DR2 rows/cov.
# build_cov is inlined verbatim from timescape_baocmb.build_cov (block-diagonal
# over z bins; a DM/DH pair at one z shares a 2x2 block with off-diagonal
# corr*err_DM*err_DH; DV bins 1x1; no cross-bin correlation), so the covariance
# construction is identical to paper 1 -- only the numbers are DR2.
# ---------------------------------------------------------------------------
with open(DR2J) as _f:
    _DR2 = json.load(_f)
_DR2_BAO = [tuple(r) for r in _DR2["rows"]]              # 13 rows (z,kind,val,err,corr)
_CMB_ROW = tuple(_DR2["cmb_point"]["row"])               # (1089.8,"DM",94.31613...,0.05,None)
ROWS_DR2 = _DR2_BAO + [_CMB_ROW]
RD_DR2 = float(_DR2["rd"])


def _build_cov(rows):
    n = len(rows)
    C = np.zeros((n, n))
    for i, (z, k, v, e, c) in enumerate(rows):
        C[i, i] = e * e
    for i in range(n):
        for j in range(i + 1, n):
            zi, ki, _, ei, ci = rows[i]
            zj, kj, _, ej, cj = rows[j]
            if zi == zj and {ki, kj} == {"DM", "DH"} and ci is not None:
                C[i, j] = C[j, i] = ci * ei * ej
    return C


_DV2 = np.array([r[2] for r in ROWS_DR2])
_CINV2 = np.linalg.inv(_build_cov(ROWS_DR2))
_DV2_BAO = np.array([r[2] for r in _DR2_BAO])
_CINV2_BAO = np.linalg.inv(_build_cov(_DR2_BAO))


def bao_cmb_chi2_dr2(predict):
    """alpha-marginalised DR2 BAO + Planck-CMB chi2 (mirror of harness.bao_cmb_chi2)."""
    g = np.array([predict(z, k) for z, k, _, _, _ in ROWS_DR2])
    gCi = _CINV2 @ g
    a = (g @ (_CINV2 @ _DV2)) / (g @ gCi)
    chi = _DV2 @ (_CINV2 @ _DV2) - (g @ (_CINV2 @ _DV2)) ** 2 / (g @ gCi)
    return float(chi), float(a)


def bao_only_chi2_dr2(predict):
    """alpha-marginalised DR2 BAO-only chi2 (drops the CMB point)."""
    g = np.array([predict(z, k) for z, k, _, _, _ in _DR2_BAO])
    gCi = _CINV2_BAO @ g
    a = (g @ (_CINV2_BAO @ _DV2_BAO)) / (g @ gCi)
    chi = _DV2_BAO @ (_CINV2_BAO @ _DV2_BAO) - (g @ (_CINV2_BAO @ _DV2_BAO)) ** 2 / (g @ gCi)
    return float(chi), float(a)


def H0_from_alpha_dr2(a):
    return 299792.458 / (a * RD_DR2)


# ---------------------------------------------------------------------------
# forward models (all reuse Probe R's dressed geometry glue)
# ---------------------------------------------------------------------------
def free_joint(v, Ngrid):
    """free-history nodes v -> (tot, SN, BC, alpha, sol) on DR2."""
    sol = PR.solve_nodes(np.asarray(v, dtype=float), "algebraic", Ngrid)
    csn = float(H.sn_chi2(sol.D_M(zHD)))
    cbc, a = bao_cmb_chi2_dr2(lambda z, k: float(sol.predict(z, k)))
    return csn + cbc, csn, cbc, a, sol


def obj_free(p, Ngrid=None):
    Ngrid = NGRID_FIT if Ngrid is None else Ngrid
    try:
        v = PR.nodes_from_params(p)
        tot, _, _, _, _ = free_joint(v, Ngrid)
    except Exception:
        return 1e9
    return float(tot) if np.isfinite(tot) else 1e9


def tracker_sol(fv0, Ngrid, ntau=TRK_NTAU_FIT):
    trk = MV.tracker_fv_of_z(fv0, ntau=ntau)
    return MV.modelv_solve(trk, lapse="algebraic", Ngrid=Ngrid)


def tracker_parts(fv0, Ngrid, ntau=TRK_NTAU_FIT):
    """tracker f_v0 -> (tot, SN, BC, alpha, sol) on DR2."""
    sol = tracker_sol(fv0, Ngrid, ntau=ntau)
    csn = float(H.sn_chi2(sol.D_M(zHD)))
    cbc, a = bao_cmb_chi2_dr2(lambda z, k: float(sol.predict(z, k)))
    return csn + cbc, csn, cbc, a, sol


# ---------------------------------------------------------------------------
# (1a) free-history DR2 joint multistart (resumable)
# ---------------------------------------------------------------------------
def _probeR_seed_nodes():
    """Probe-R (DR1) best-fit nodes -- the strong seed for the DR2 optimum."""
    try:
        j = json.load(open(PROBERJ))
        return np.array(j["V"]["fv_nodes"], dtype=float)
    except Exception:
        return np.array([0.64013, 0.53112, 0.39578, 0.27945, 0.19359])


def build_free_starts(rng):
    seeds = [_probeR_seed_nodes(),
             [0.70, 0.62, 0.52, 0.40, 0.28], [0.55, 0.47, 0.38, 0.27, 0.18],
             [0.85, 0.78, 0.68, 0.55, 0.40], [0.40, 0.33, 0.26, 0.18, 0.11],
             [0.62, 0.50, 0.36, 0.24, 0.15], [0.90, 0.70, 0.45, 0.25, 0.12]]
    starts = [PR.params_from_nodes(np.asarray(s, dtype=float)) for s in seeds]
    while len(starts) < N_RESTARTS:
        starts.append(rng.uniform(-5.0, 5.0, 5))
    return starts[:max(N_RESTARTS, len(seeds))]


def free_multistart(prior=None):
    rng = np.random.default_rng(SEED)
    starts = build_free_starts(rng)
    best_chi2 = np.inf if prior is None else prior["chi2"]
    best_p = None if prior is None else np.array(prior["p"])
    done = 0 if prior is None else prior.get("done", 0)
    log(f"[free] multistart: {len(starts)} restarts + 1 DE  (resume from {done})")
    for i, s in enumerate(starts):
        if i < done:
            continue
        r = minimize(obj_free, s, method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=5e-3, maxiter=2000))
        if r.fun < best_chi2:
            best_chi2, best_p = float(r.fun), r.x.copy()
        done = i + 1
        if (i + 1) % CKPT_EVERY == 0:
            _atomic_dump({"_checkpoint": True, "stage": "free_multistart",
                          "state": dict(chi2=best_chi2, p=best_p.tolist(),
                                        done=done, n_starts=len(starts))}, OUTJ)
            log(f"  [free] restart {i+1}/{len(starts)}  best={best_chi2:.4f}")
        if MAXSEC and time.time() - _t0 > MAXSEC:
            _atomic_dump({"_checkpoint": True, "stage": "free_multistart",
                          "state": dict(chi2=best_chi2, p=best_p.tolist(),
                                        done=done, n_starts=len(starts))}, OUTJ)
            log(f"  [free] MAXSEC hit at restart {i+1}; checkpointed, exit for relaunch")
            sys.exit(3)
    de = differential_evolution(obj_free, bounds=[(-6.0, 6.0)] * 5, seed=SEED,
                                maxiter=50, popsize=15, tol=1e-6, mutation=(0.4, 1.0),
                                recombination=0.8, polish=True, init="sobol")
    de_used = de.fun < best_chi2
    if de_used:
        best_chi2, best_p = float(de.fun), de.x.copy()
    log(f"  [free] DE fun={de.fun:.4f} -> global best={best_chi2:.4f}")
    return best_p, de_used, len(starts)


# ---------------------------------------------------------------------------
# 1D profile helper: minimise f(x) on [lo,hi] with a coarse grid guard + Brent
# ---------------------------------------------------------------------------
def min1d(f, lo, hi, ngrid=25):
    xs = np.linspace(lo, hi, ngrid)
    ys = np.array([f(x) for x in xs], dtype=float)
    ys = np.where(np.isfinite(ys), ys, np.inf)   # nan/inf (degenerate geometry) -> +inf;
    i = int(np.argmin(ys))                        # np.argmin returns the NaN index otherwise
    a = xs[max(i - 1, 0)]
    b = xs[min(i + 1, ngrid - 1)]
    r = minimize_scalar(f, bounds=(a, b), method="bounded",
                        options=dict(xatol=1e-4))
    rfun = float(r.fun) if np.isfinite(r.fun) else np.inf
    if rfun <= ys[i]:
        return float(r.x), rfun
    return float(xs[i]), float(ys[i])


def main():
    log("Phase F start")
    prior = None
    if os.path.exists(OUTJ):
        try:
            j = json.load(open(OUTJ))
            if j.get("_checkpoint") and j.get("stage") == "free_multistart":
                prior = j.get("state")
                log(f"resuming free multistart from checkpoint: done={prior.get('done')}")
        except Exception:
            prior = None

    # =====================================================================
    # (1) DR2 JOINT REFIT
    # =====================================================================
    # ---- LCDM on DR2 ----
    def lcdm_joint(Om):
        return float(H.sn_chi2(H.lcdm_Dc(zHD, Om))) + bao_cmb_chi2_dr2(H.lcdm_predict(Om))[0]
    Om_dr2, chi2_lcdm = min1d(lcdm_joint, 0.20, 0.45, ngrid=26)
    _, a_lcdm = bao_cmb_chi2_dr2(H.lcdm_predict(Om_dr2))
    H0_lcdm = H0_from_alpha_dr2(a_lcdm)
    log(f"LCDM DR2:    Om={Om_dr2:.4f}  chi2={chi2_lcdm:.3f}  H0={H0_lcdm:.2f}")

    # ---- tracker on DR2 (one-parameter f_v0) ----
    fv0_trk, chi2_trk = min1d(lambda x: tracker_parts(x, NGRID_FIT)[0], 0.30, 0.95, ngrid=28)
    tot_t, csn_t, cbc_t, a_t, sol_t = tracker_parts(fv0_trk, NGRID_FINE, ntau=300000)
    chi2_trk = tot_t
    H0_trk_dressed = float(MV.g_dress(fv0_trk)) * H0_from_alpha_dr2(a_t)
    log(f"tracker DR2: fv0={fv0_trk:.4f}  chi2={chi2_trk:.3f} (SN={csn_t:.2f} BC={cbc_t:.2f})  "
        f"H0d={H0_trk_dressed:.2f}")

    # ---- free history on DR2 ----
    best_p, de_used, n_starts = free_multistart(prior)
    v_free = PR.nodes_from_params(best_p)
    tot_f, csn_f, cbc_f, a_f, sol_f = free_joint(v_free, NGRID_FINE)
    chi2_free = tot_f
    h0_free = PR.dressed_H0(sol_f, a_f, "algebraic")
    fv0_free = float(sol_f.fv0)
    log(f"free DR2:    chi2={chi2_free:.3f} (SN={csn_f:.2f} BC={cbc_f:.2f})  fv0={fv0_free:.4f}  "
        f"H0d={h0_free['H0_dressed']:.2f}  nodes={np.round(v_free,5).tolist()}")

    # ---- honest parameter accounting (Wilks) ----
    # free history: 5 node params (cosmological) vs LCDM 1 (Om) -> 4 extra dof.
    dchi_free_vs_lcdm = chi2_lcdm - chi2_free
    p_wilks = float(_chi2dist.sf(max(dchi_free_vs_lcdm, 0.0), df=4))
    dchi_free_E_dr1 = REF_DR1["LCDM"] - REF_DR1["free_E"]     # 10.39 (T1, DR1)
    p_wilks_T1 = float(_chi2dist.sf(max(dchi_free_E_dr1, 0.0), df=4))

    # =====================================================================
    # (2) AMPLITUDE-SPLIT (headline)
    # =====================================================================
    # Fix the SHAPE = DR2 free-fit nodes normalised so shape(z=0)=1; A == f_v(0).
    shape_norm = v_free / v_free[0]

    def amp_parts(A, Ngrid):
        nodes = A * shape_norm
        sol = PR.solve_nodes(nodes, "algebraic", Ngrid)
        csn = float(H.sn_chi2(sol.D_M(zHD)))
        cbc, aa = bao_cmb_chi2_dr2(lambda z, k: float(sol.predict(z, k)))
        return csn, cbc, aa

    # A == f_v(0); cap below the f_v->1 dressed-geometry breakdown (chi2 explodes
    # far above any minimum there anyway -- both SN and BC minima sit near ~0.64).
    A_LO, A_HI = 0.20, 0.90
    A_SN, _ = min1d(lambda A: amp_parts(A, NGRID_FIT)[0], A_LO, A_HI, ngrid=30)
    A_BC, _ = min1d(lambda A: amp_parts(A, NGRID_FIT)[1], A_LO, A_HI, ngrid=30)
    A_J, _ = min1d(lambda A: sum(amp_parts(A, NGRID_FIT)[:2]), A_LO, A_HI, ngrid=30)
    # fine re-eval of the three anchor points
    csn_ASN, cbc_ASN, _ = amp_parts(A_SN, NGRID_FINE)
    csn_ABC, cbc_ABC, _ = amp_parts(A_BC, NGRID_FINE)
    csn_AJ, cbc_AJ, aJ = amp_parts(A_J, NGRID_FINE)
    joint_min_A = csn_AJ + cbc_AJ
    sep_min_A = csn_ASN + cbc_ABC
    delta_join_A = joint_min_A - sep_min_A
    sigma_A = float(np.sqrt(max(delta_join_A, 0.0)))
    log(f"AMP-SPLIT (free shape): A_SN={A_SN:.4f} A_BC={A_BC:.4f} A_joint={A_J:.4f}  "
        f"delta_join={delta_join_A:.3f}  sigma={sigma_A:.2f}")

    # ---- SANITY: same statistic on the TRACKER's own f_v0 (shape==amplitude) ----
    fv0_SN_trk, _ = min1d(lambda x: tracker_parts(x, NGRID_FIT)[1], 0.45, 0.94, ngrid=26)
    fv0_BC_trk, _ = min1d(lambda x: tracker_parts(x, NGRID_FIT)[2], 0.30, 0.90, ngrid=26)
    fv0_J_trk, _ = min1d(lambda x: tracker_parts(x, NGRID_FIT)[0], 0.30, 0.95, ngrid=28)
    _, csn_tSN, cbc_tSN, _, _ = tracker_parts(fv0_SN_trk, NGRID_FINE, ntau=300000)
    _, csn_tBC, cbc_tBC, _, _ = tracker_parts(fv0_BC_trk, NGRID_FINE, ntau=300000)
    tot_tJ, csn_tJ, cbc_tJ, _, _ = tracker_parts(fv0_J_trk, NGRID_FINE, ntau=300000)
    delta_join_trk = tot_tJ - (csn_tSN + cbc_tBC)
    sigma_trk = float(np.sqrt(max(delta_join_trk, 0.0)))
    log(f"AMP-SPLIT SANITY (tracker f_v0): SN={fv0_SN_trk:.4f} BC={fv0_BC_trk:.4f} "
        f"joint={fv0_J_trk:.4f}  delta_join={delta_join_trk:.3f}  sigma={sigma_trk:.2f}")

    # ---- cross-check: amplitude split on the literal DR1 Probe-R shape ----
    v_pr = _probeR_seed_nodes()
    shape_pr = v_pr / v_pr[0]

    def amp_parts_pr(A, Ngrid):
        sol = PR.solve_nodes(A * shape_pr, "algebraic", Ngrid)
        csn = float(H.sn_chi2(sol.D_M(zHD)))
        cbc, aa = bao_cmb_chi2_dr2(lambda z, k: float(sol.predict(z, k)))
        return csn, cbc
    A_SN_pr, _ = min1d(lambda A: amp_parts_pr(A, NGRID_FIT)[0], A_LO, A_HI, ngrid=30)
    A_BC_pr, _ = min1d(lambda A: amp_parts_pr(A, NGRID_FIT)[1], A_LO, A_HI, ngrid=30)
    A_J_pr, _ = min1d(lambda A: sum(amp_parts_pr(A, NGRID_FIT)), A_LO, A_HI, ngrid=30)
    csn_prSN, cbc_prSN = amp_parts_pr(A_SN_pr, NGRID_FINE)
    csn_prBC, cbc_prBC = amp_parts_pr(A_BC_pr, NGRID_FINE)
    csn_prJ, cbc_prJ = amp_parts_pr(A_J_pr, NGRID_FINE)
    delta_pr = (csn_prJ + cbc_prJ) - (csn_prSN + cbc_prBC)
    sigma_pr = float(np.sqrt(max(delta_pr, 0.0)))
    log(f"AMP-SPLIT (DR1 Probe-R shape): A_SN={A_SN_pr:.4f} A_BC={A_BC_pr:.4f} "
        f"delta={delta_pr:.3f} sigma={sigma_pr:.2f}")

    split_dissolves = bool(sigma_A < 2.0)

    # =====================================================================
    # assemble + write
    # =====================================================================
    out = dict(
        probe="Phase F -- DESI DR2 joint refit + amplitude-split under free-history Model V",
        reading="KINEMATIC (force f_v(z), dressed observables; integrability NOT enforced)",
        data=dict(
            sn="harness.sn_chi2 (1580 Pantheon+ SNe, full stat+sys cov, offset marginalised)",
            bao_cmb=f"DESI DR2 (probes_out/desi_dr2_rows.json, {len(_DR2_BAO)} BAO points) + "
                    f"Planck acoustic point (DM/rd={_CMB_ROW[2]:.5f}, err={_CMB_ROW[3]}), "
                    f"alpha=c/(Hbar0 rd) marginalised, rd={RD_DR2}",
            note="DR2 bao_cmb_chi2 built here mirrors harness.bao_cmb_chi2 exactly "
                 "(same alpha marginalisation, same block-diagonal build_cov, same CMB value); "
                 "only the DESI rows differ (DR1 -> DR2).",
        ),
        references_DR1_same_machinery=REF_DR1,

        dr2_joint_refit=dict(
            free_history=dict(
                chi2=chi2_free, chi2_SN=csn_f, chi2_BAOCMB=cbc_f, alpha=a_f,
                fv0=fv0_free, z_nodes=PR.Z_NODES.tolist(),
                fv_nodes=np.round(v_free, 5).tolist(),
                H0_dressed=h0_free["H0_dressed"],
                H0_dressed_gdress=h0_free["H0_dressed_gdress"],
                H0_dressed_Hd0=h0_free["H0_dressed_Hd0"], Hbar0=h0_free["Hbar0"],
                n_restarts=n_starts, de_used=bool(de_used),
                ngrid_fit=NGRID_FIT, ngrid_fine=NGRID_FINE),
            lcdm=dict(chi2=chi2_lcdm, Om=Om_dr2, H0=H0_lcdm),
            tracker=dict(chi2=chi2_trk, chi2_SN=csn_t, chi2_BAOCMB=cbc_t,
                         fv0=fv0_trk, H0_dressed=H0_trk_dressed),
            deltas=dict(
                free_minus_lcdm=chi2_free - chi2_lcdm,
                free_minus_tracker=chi2_free - chi2_trk,
                lcdm_minus_tracker=chi2_lcdm - chi2_trk),
            param_accounting=dict(
                free_history_node_params=5, lcdm_cosmo_params=1, extra_dof=4,
                dchi2_free_vs_lcdm=dchi_free_vs_lcdm,
                wilks_p_free_vs_lcdm=p_wilks,
                reference_T1_free_E_DR1=dict(
                    dchi2=dchi_free_E_dr1, extra_dof=4, wilks_p=p_wilks_T1,
                    reported_p_paper="0.065"),
                note="free history has 5 monotone f_v nodes (4 extra vs LCDM's 1 Om); "
                     "Wilks p = chi2.sf(dchi2, df=4). Both share 2 profiled nuisances "
                     "(SN offset, BAO alpha)."),
        ),

        amplitude_split=dict(
            definition="f_v(z)=clip(A*shape(z)), shape=free DR2 nodes normalised to "
                       "shape(0)=1, so A == f_v(0). A fit separately to SN-only and "
                       "BAO+CMB-only. sigma = sqrt(delta chi2_join) (profile "
                       "parameter-shift, 1 dof).",
            shape_nodes_norm=np.round(shape_norm, 5).tolist(),
            A_SN=A_SN, A_BAOCMB=A_BC, A_joint=A_J,
            chi2_SN_at_A_SN=csn_ASN, chi2_BC_at_A_BC=cbc_ABC,
            chi2_SN_at_A_joint=csn_AJ, chi2_BC_at_A_joint=cbc_AJ,
            joint_min=joint_min_A, separate_min=sep_min_A,
            delta_chi2_join=delta_join_A, sigma=sigma_A,
            split_dissolves=split_dissolves,
            tracker_sanity=dict(
                definition="same statistic on the tracker's own parameter f_v0 "
                           "(its shape==amplitude one-parameter family); reproduces "
                           "paper 1's SN(high f_v0) vs BAO+CMB(low f_v0) split.",
                fv0_SN=fv0_SN_trk, fv0_BAOCMB=fv0_BC_trk, fv0_joint=fv0_J_trk,
                chi2_SN_at_fv0SN=csn_tSN, chi2_BC_at_fv0BC=cbc_tBC,
                joint_min=tot_tJ, separate_min=csn_tSN + cbc_tBC,
                delta_chi2_join=delta_join_trk, sigma=sigma_trk,
                paper1_reference="0.85-vs-0.64 f_v0 split, 4.8-6.6 sigma"),
            crosscheck_DR1_probeR_shape=dict(
                note="amplitude split using the LITERAL DR1 Probe-R nodes as the shape",
                shape_nodes=np.round(v_pr, 5).tolist(),
                A_SN=A_SN_pr, A_BAOCMB=A_BC_pr, A_joint=A_J_pr,
                delta_chi2_join=delta_pr, sigma=sigma_pr),
        ),
        runtime_s=round(time.time() - _t0, 1),
    )
    _atomic_dump(out, OUTJ)
    log(f"wrote {OUTJ}")
    log(f"SUMMARY  free={chi2_free:.2f}  lcdm={chi2_lcdm:.2f}  tracker={chi2_trk:.2f} | "
        f"amp-split free sigma={sigma_A:.2f} (dissolves={split_dissolves})  "
        f"tracker sigma={sigma_trk:.2f}")
    return out


if __name__ == "__main__":
    main()
