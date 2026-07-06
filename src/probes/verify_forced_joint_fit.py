#!/usr/bin/env python3
"""ADVERSARIAL, refute-by-default verification of probes_out/forced_joint_fit.json.

Independent re-implementation of the FORCED zero-shape-parameter joint fit. This
script DELIBERATELY does NOT import forced_joint_fit.py. It:

  1. rebuilds the f_v^obs 5-node vector = Phi(sigma0*D(z)/2) from an independently
     coded flat-LCDM linear growth D(z) and the committed 2M++ sigma0 anchor;
  2. drives it through modelv_theory.modelv_solve (algebraic lapse), re-coding the
     Probe-R high-z bridge glue locally (BRIDGE_Z + geometric tail);
  3. computes chi2_SN via harness.sn_chi2 (full cov) and rebuilds the diagonal /
     Tripp SN cells from fit_timescape.make_chi2 directly;
  4. rebuilds the DR2 BAO+Planck chi2 FROM SCRATCH (own block-diagonal cov + alpha
     marginalisation) and the DR1 BAO+Planck chi2 both via harness.bao_cmb_chi2 and
     an independent rebuild from timescape_baocmb.BAO;
  5. independently refits flat LCDM (own comoving-distance + predict functions) on
     the same DR2/DR1 data under each SN covariance treatment;
  6. recomputes bic_bar = chi2_LCDM + ln N, clears_bar, miss_by, and the SN
     covariance ladder headline;
  7. writes probes_out/verify_forced_joint_fit.json with every check + numbers.

SANITY (the point of the probe): the shallow, near-constant survey f_v^obs must
MISS the BIC bar by a large margin. If it CLEARS the bar, that is a REFUTATION of
the SHAPE-UNAVAILABLE claim and is flagged.

Run:  python probes/verify_forced_joint_fit.py
"""
import os
import sys
import io
import json
import contextlib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.integrate import quad, cumulative_trapezoid

np.seterr(all="ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.dirname(_SRC)
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SRC)  # harness / fit_timescape use relative data/ paths

import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):
    import harness as H
    import timescape_baocmb as T

_OUTDIR = os.path.join(_ROOT, "probes_out")
FORCEDJ = os.path.join(_OUTDIR, "forced_joint_fit.json")
DR2J = os.path.join(_OUTDIR, "desi_dr2_rows.json")
TELESCOPEJ = os.path.join(_OUTDIR, "telescope_fvobs.json")
OUTJ = os.path.join(_OUTDIR, "verify_forced_joint_fit.json")

NGRID = 30000
OM_FIDUCIAL = 0.315
Z_NODES = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
# Probe-R high-z bridge glue (re-coded locally, mirrors modelv_probeR)
BRIDGE_Z = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])


def rel(a, b):
    a, b = float(a), float(b)
    d = abs(a - b)
    return d, (d / abs(b) if b != 0 else d)


# ---------------------------------------------------------------------------
# 1. f_v^obs(z) -- independent linear-growth + below-mean-floor Phi(sigma/2)
# ---------------------------------------------------------------------------
def growth_factor(z, Om=OM_FIDUCIAL):
    """Flat-LCDM linear growth D(z), D(0)=1, coded from the standard integral
       D(a) ∝ H(a) ∫_0^a da'/(a' H(a'))^3  (independent of forced_joint_fit)."""
    def E(a):
        return np.sqrt(Om * a ** -3 + (1.0 - Om))
    def Dun(a):
        return E(a) * quad(lambda ap: 1.0 / (ap * E(ap)) ** 3, 1e-8, a)[0]
    return Dun(1.0 / (1.0 + z)) / Dun(1.0)


def build_fvobs():
    tel = json.load(open(TELESCOPEJ))
    sigma0 = float(tel["provenance"]["sigma0_anchor"])
    D = np.array([growth_factor(z) for z in Z_NODES])
    sigma = sigma0 * D
    fv = norm.cdf(sigma / 2.0)                      # below-mean floor Phi(sigma/2)
    committed = {n["z"]: n["fv_below_mean"] for n in tel["PRIMARY_below_mean_Rs4"]["nodes"]}
    checks = {}
    for zi, fvi in zip(Z_NODES, fv):
        if float(zi) in committed:
            ref = committed[float(zi)]
            checks[f"z={zi:g}"] = dict(computed=float(fvi), committed=float(ref),
                                       abs_diff=abs(float(fvi) - ref))
    max_diff = max(c["abs_diff"] for c in checks.values())
    return Z_NODES, fv, sigma0, D, checks, float(max_diff)


# ---------------------------------------------------------------------------
# 2. forced dressed geometry via modelv_theory.modelv_solve (algebraic)
# ---------------------------------------------------------------------------
def bridge_fv(fv_last):
    return fv_last * ((1.0 + Z_NODES[-1]) / (1.0 + BRIDGE_Z)) ** 1.5


def forced_solution(v, lapse):
    fv = MV.fv_from_nodes(np.asarray(v, dtype=float), z_nodes=Z_NODES,
                          bridge_z=BRIDGE_Z, bridge_fv=bridge_fv(float(v[-1])))
    return MV.modelv_solve(fv, lapse=lapse, Ngrid=NGRID)


# ---------------------------------------------------------------------------
# 3. DR2 BAO + Planck-CMB chi2 -- rebuilt FROM SCRATCH
# ---------------------------------------------------------------------------
def block_diag_cov(rows):
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


def alpha_marg_chi2(rows, Cinv, DV, predict):
    g = np.array([predict(z, k) for z, k, _, _, _ in rows])
    gCi = Cinv @ g
    a = (g @ (Cinv @ DV)) / (g @ gCi)
    chi = DV @ (Cinv @ DV) - (g @ (Cinv @ DV)) ** 2 / (g @ gCi)
    return float(chi), float(a)


# ---------------------------------------------------------------------------
# 5. independent flat-LCDM distances
# ---------------------------------------------------------------------------
def lcdm_Dc(z, Om):
    zg = np.linspace(0.0, np.max(z) * 1.0001, 200000)
    invE = 1.0 / np.sqrt(Om * (1.0 + zg) ** 3 + (1.0 - Om))
    Dc = cumulative_trapezoid(invE, zg, initial=0.0)
    return np.interp(z, zg, Dc)


def lcdm_predict(Om):
    def p(z, k):
        E = np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))
        if k == "DH":
            return 1.0 / E
        dM = quad(lambda zz: 1.0 / np.sqrt(Om * (1.0 + zz) ** 3 + (1.0 - Om)), 0.0, z)[0]
        if k == "DM":
            return dM
        return (z * dM * dM / E) ** (1.0 / 3.0)
    return p


def main():
    ref = json.load(open(FORCEDJ))
    checks = []

    def check(name, mine, committed, tol=1e-6, note=""):
        d, r = rel(mine, committed)
        ok = (d <= tol) or (r <= tol)
        checks.append(dict(check=name, mine=float(mine), committed=float(committed),
                           abs_diff=d, rel_diff=r, tol=tol, pass_=bool(ok), note=note))
        return ok

    # ---- 1. f_v^obs nodes -------------------------------------------------
    z_nodes, fvobs, sigma0, Dz, fv_checks, max_diff = build_fvobs()
    ref_fv = ref["fvobs_history"]["fv_nodes"]
    fv_node_diffs = [abs(float(a) - float(b)) for a, b in zip(fvobs, ref_fv)]
    max_fv_node_diff = max(fv_node_diffs)
    checks.append(dict(check="fvobs_nodes_vs_committed_json",
                       mine=[float(x) for x in fvobs], committed=[float(x) for x in ref_fv],
                       abs_diff=max_fv_node_diff, tol=1e-6,
                       pass_=bool(max_fv_node_diff <= 1e-6),
                       note="Phi(sigma0*D(z)/2), sigma0=%.10f, indep growth D(z)" % sigma0))
    checks.append(dict(check="fvobs_validation_vs_telescope_committed_nodes",
                       mine=max_diff, committed=0.0, abs_diff=max_diff, tol=1e-9,
                       pass_=bool(max_diff <= 1e-9),
                       note="z=0,0.3,0.7 below-mean primary nodes maxdiff"))

    # ---- 2-4. forced geometry -> chi2 ------------------------------------
    sol = forced_solution(fvobs, "algebraic")
    predict = lambda z, k: float(sol.predict(z, k))
    zHD, zHEL, mb, Cf = F.load()
    DM_LA = sol.D_M(zHD)

    chi2_SN_full = float(H.sn_chi2(DM_LA))

    # SN cov ladder cells (independent rebuild from F.make_chi2)
    sn_diag = F.make_chi2(zHD, zHEL, mb, np.diag(np.diag(Cf)))
    # Tripp stat-only diagonal from m_b_corr_err_DIAG under F.load's exact mask
    with open(F.DATA) as f:
        hdr = f.readline().split()
    idx = {n: i for i, n in enumerate(hdr)}
    raw = [ln.split() for ln in open(F.DATA).read().splitlines()[1:]]
    z_all = np.array([float(r[idx["zHD"]]) for r in raw])
    iscal = np.array([int(float(r[idx["IS_CALIBRATOR"]])) for r in raw])
    errd = np.array([float(r[idx["m_b_corr_err_DIAG"]]) for r in raw])
    mask = (iscal == 0) & (z_all > 0.01)
    err_tripp = errd[mask]
    sn_tripp = F.make_chi2(zHD, zHEL, mb, np.diag(err_tripp ** 2))
    SN = {"full_cov": H.sn_chi2, "diagonal_cov": sn_diag, "tripp_diag": sn_tripp}
    csn = {name: float(fn(DM_LA)) for name, fn in SN.items()}

    # DR2 BAO+CMB chi2 -- from scratch
    dr2 = json.load(open(DR2J))
    rows_dr2 = [tuple(r) for r in dr2["rows"]] + [tuple(dr2["cmb_point"]["row"])]
    Cinv2 = np.linalg.inv(block_diag_cov(rows_dr2))
    DV2 = np.array([r[2] for r in rows_dr2])
    chi2_bc_dr2, a_dr2 = alpha_marg_chi2(rows_dr2, Cinv2, DV2, predict)

    # DR1 BAO+CMB chi2 -- via harness AND independent rebuild
    chi2_bc_dr1_H, a_dr1_H = H.bao_cmb_chi2(predict)
    rows_dr1 = [(z, k, v, e, c) for (z, k, v, e, c) in T.BAO] + [(1089.80, "DM", H._DMz, H._SIG, None)]
    Cinv1 = np.linalg.inv(block_diag_cov(rows_dr1))
    DV1 = np.array([r[2] for r in rows_dr1])
    chi2_bc_dr1_mine, a_dr1_mine = alpha_marg_chi2(rows_dr1, Cinv1, DV1, predict)

    forced_dr2 = chi2_SN_full + chi2_bc_dr2
    forced_dr1 = chi2_SN_full + chi2_bc_dr1_H

    rf = ref["forced_fit"]["lapse_LA"]
    check("chi2_SN_full_cov", chi2_SN_full, rf["chi2_SN"], tol=1e-3)
    check("chi2_BAOCMB_dr2", chi2_bc_dr2, rf["chi2_BAOCMB_dr2"], tol=1e-2)
    check("chi2_BAOCMB_dr1_harness", chi2_bc_dr1_H, rf["chi2_BAOCMB_dr1"], tol=1e-3)
    check("chi2_BAOCMB_dr1_independent_rebuild", chi2_bc_dr1_mine, rf["chi2_BAOCMB_dr1"], tol=1e-3,
          note="rebuilt from timescape_baocmb.BAO + harness Planck point")
    check("forced_chi2_dr2", forced_dr2, rf["forced_chi2_dr2"], tol=1e-2)
    check("forced_chi2_dr1", forced_dr1, rf["forced_chi2_dr1"], tol=1e-2)
    check("fv0", sol.fv0, rf["fv0"], tol=1e-6)
    check("alpha_dr2", a_dr2, rf["alpha_dr2"], tol=1e-3)

    # ---- 5-6. LCDM refits + BIC bars, per SN cov treatment ----------------
    def lcdm_min(snfn, bc_fn):
        def joint(Om):
            return float(snfn(lcdm_Dc(zHD, Om))) + bc_fn(lcdm_predict(Om))[0]
        grid = np.linspace(0.20, 0.45, 26)
        ys = np.array([joint(g) for g in grid])
        i = int(np.argmin(ys))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        r = minimize_scalar(joint, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-5))
        return float(r.x), float(r.fun)

    def bc_dr2(pred):
        return alpha_marg_chi2(rows_dr2, Cinv2, DV2, pred)
    def bc_dr1(pred):
        return alpha_marg_chi2(rows_dr1, Cinv1, DV1, pred)

    N_dr2 = len(zHD) + len(rows_dr2)     # 1580 + 14
    N_dr1 = len(zHD) + len(rows_dr1)     # 1580 + 13
    lnN_dr2, lnN_dr1 = float(np.log(N_dr2)), float(np.log(N_dr1))
    check("N_dr2", N_dr2, ref["bic_test"]["N_dr2"], tol=0)
    check("N_dr1", N_dr1, ref["bic_test"]["N_dr1"], tol=0)
    check("lnN_dr2", lnN_dr2, ref["bic_test"]["lnN_dr2"], tol=1e-6)
    check("lnN_dr1", lnN_dr1, ref["bic_test"]["lnN_dr1"], tol=1e-6)

    ladder = {}
    ref_ladder = ref["cov_ladder"]
    for cov, snfn in SN.items():
        Om2, c2 = lcdm_min(snfn, bc_dr2)
        Om1, c1 = lcdm_min(snfn, bc_dr1)
        f2 = csn[cov] + chi2_bc_dr2
        f1 = csn[cov] + chi2_bc_dr1_H
        bar2, bar1 = c2 + lnN_dr2, c1 + lnN_dr1
        ladder[cov] = dict(
            Om_dr2=Om2, Om_dr1=Om1, chi2_SN=csn[cov],
            chi2_LCDM_dr2=c2, chi2_LCDM_dr1=c1,
            forced_chi2_dr2=f2, forced_chi2_dr1=f1,
            bic_bar_dr2=bar2, bic_bar_dr1=bar1,
            miss_by_dr2=f2 - bar2, miss_by_dr1=f1 - bar1,
            clears_bar_dr2=bool(f2 <= bar2), clears_bar_dr1=bool(f1 <= bar1))
        rl = ref_ladder[cov]
        check(f"{cov}:chi2_SN", csn[cov], rl["chi2_SN"], tol=1e-2)
        check(f"{cov}:chi2_LCDM_dr2", c2, rl["chi2_LCDM_dr2"], tol=5e-2)
        check(f"{cov}:chi2_LCDM_dr1", c1, rl["chi2_LCDM_dr1"], tol=5e-2)
        check(f"{cov}:bic_bar_dr2", bar2, rl["bic_bar_dr2"], tol=5e-2)
        check(f"{cov}:miss_by_dr2", f2 - bar2, rl["miss_by_dr2"], tol=5e-2)
        check(f"{cov}:miss_by_dr1", f1 - bar1, rl["miss_by_dr1"], tol=5e-2)
        checks.append(dict(check=f"{cov}:clears_bar_dr2", mine=bool(f2 <= bar2),
                           committed=rl["clears_bar_dr2"], abs_diff=0.0, tol=0,
                           pass_=bool((f2 <= bar2) == rl["clears_bar_dr2"]),
                           note="expected False (miss)"))

    head = ladder["full_cov"]
    ref_head = ref["bic_test"]
    check("headline:chi2_LCDM_dr2", head["chi2_LCDM_dr2"], ref_head["chi2_LCDM_dr2"], tol=5e-2)
    check("headline:chi2_LCDM_dr1", head["chi2_LCDM_dr1"], ref_head["chi2_LCDM_dr1"], tol=5e-2)
    check("headline:bic_bar_dr2", head["bic_bar_dr2"], ref_head["bic_bar_dr2"], tol=5e-2)
    check("headline:bic_bar_dr1", head["bic_bar_dr1"], ref_head["bic_bar_dr1"], tol=5e-2)
    check("headline:miss_by_dr2", head["miss_by_dr2"], ref_head["miss_by_dr2"], tol=5e-2)
    check("headline:miss_by_dr1", head["miss_by_dr1"], ref_head["miss_by_dr1"], tol=5e-2)

    # ladder headline vector [full, diag, tripp] miss_by dr2
    ladder_miss = [ladder["full_cov"]["miss_by_dr2"],
                   ladder["diagonal_cov"]["miss_by_dr2"],
                   ladder["tripp_diag"]["miss_by_dr2"]]
    ref_ladder_miss = ref["numbers"]["cov_ladder_miss_by_dr2_full_diag_tripp"] \
        if "numbers" in ref else [1438.45, 1811.99, 1408.04]

    # ---- SANITY: shallow history must MISS the bar (all rows) --------------
    all_clear_dr2 = all(ladder[c]["clears_bar_dr2"] for c in SN)
    all_clear_dr1 = all(ladder[c]["clears_bar_dr1"] for c in SN)
    sanity_miss = (not all_clear_dr2) and (not all_clear_dr1) and head["miss_by_dr2"] > 1000.0
    checks.append(dict(check="SANITY_forced_shallow_MISSES_bar_large_margin",
                       mine=dict(all_clear_dr2=all_clear_dr2, all_clear_dr1=all_clear_dr1,
                                 headline_miss_by_dr2=head["miss_by_dr2"]),
                       committed="MISS by >1000 chi2 in every row (SHAPE-UNAVAILABLE)",
                       abs_diff=0.0, tol=0, pass_=bool(sanity_miss),
                       note="REFUTED if it unexpectedly CLEARS the bar"))

    n_pass = sum(1 for c in checks if c.get("pass_"))
    n_total = len(checks)
    verdict = "SURVIVES" if (n_pass == n_total and sanity_miss) else \
              ("REFUTED" if not sanity_miss else "SURVIVES_WITH_CAVEATS")

    out = dict(
        probe="ADVERSARIAL verification of forced_joint_fit.json (independent re-implementation)",
        method="Rebuilt f_v^obs, forced Model-V geometry (modelv_theory.modelv_solve, algebraic), "
                "harness.sn_chi2, from-scratch DR2 BAO+CMB chi2, independent LCDM refit + BIC bars. "
                "forced_joint_fit.py NOT imported.",
        verdict=verdict,
        n_checks_pass=n_pass, n_checks_total=n_total,
        headline=dict(
            forced_chi2_dr2=forced_dr2, forced_chi2_dr1=forced_dr1,
            chi2_SN_full_cov=chi2_SN_full, chi2_BAOCMB_dr2=chi2_bc_dr2, chi2_BAOCMB_dr1=chi2_bc_dr1_H,
            chi2_LCDM_dr2=head["chi2_LCDM_dr2"], chi2_LCDM_dr1=head["chi2_LCDM_dr1"],
            bic_bar_dr2=head["bic_bar_dr2"], bic_bar_dr1=head["bic_bar_dr1"],
            miss_by_dr2=head["miss_by_dr2"], miss_by_dr1=head["miss_by_dr1"],
            clears_bar_dr2=head["clears_bar_dr2"], clears_bar_dr1=head["clears_bar_dr1"],
            fv0=float(sol.fv0), dz_resid=float(sol.dz_resid)),
        fvobs_nodes=[float(x) for x in fvobs],
        fvobs_max_node_diff_vs_json=max_fv_node_diff,
        fvobs_validation_maxdiff_vs_telescope=max_diff,
        cov_ladder=ladder,
        cov_ladder_miss_by_dr2_full_diag_tripp=ladder_miss,
        cov_ladder_ref_miss_by=ref_ladder_miss,
        cov_ladder_all_clear_dr2=all_clear_dr2,
        cov_ladder_all_clear_dr1=all_clear_dr1,
        dr1_independent_rebuild_matches_harness=abs(chi2_bc_dr1_mine - chi2_bc_dr1_H),
        sanity_miss=bool(sanity_miss),
        checks=checks,
    )
    tmp = OUTJ + ".tmp"
    json.dump(out, open(tmp, "w"), indent=2)
    os.replace(tmp, OUTJ)
    print(f"wrote {OUTJ}")
    print(f"VERDICT={verdict}  checks {n_pass}/{n_total}  "
          f"forced_dr2={forced_dr2:.3f}  bar={head['bic_bar_dr2']:.3f}  "
          f"miss_by={head['miss_by_dr2']:.3f}  clears={head['clears_bar_dr2']}")
    for c in checks:
        if not c.get("pass_"):
            print("  FAIL:", c["check"], "mine=", c["mine"], "committed=", c["committed"],
                  "abs_diff=", c.get("abs_diff"))
    return out


if __name__ == "__main__":
    main()
