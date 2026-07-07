#!/usr/bin/env python3
# phaseF_freshH0: CALIBRATOR-ANCHORED late-time dressed H0 under FREE-HISTORY timescape.
#
# Paper-2 analogue of paper-1's freshH0 (SH0ES-calibrator-anchored decisive-test-(ii)).
# Paper-1 freshH0 fit the ONE-PARAMETER TRACKER timescape shape (D_shape_TS(z;fv0)) to
# Pantheon+SH0ES with the Cepheid calibrators pinning M_B, and found the anchored late-time
# dressed H0 = 73.0 (the SH0ES ladder measures the LOCAL Hubble slope, which is model-shape
# independent). Here we replace the tracker shape with the FREE-HISTORY dressed geometry
# (modelv_theory: an arbitrary monotone f_v(z) driven through two-phase Buchert + algebraic
# wall/void clock dressing) and ask: does the calibrator-anchored H0 still sit near 73, or
# does freeing the history move it?
#
# Machinery (mirrors paper-1 freshH0.py verbatim where possible):
#   * full 1701x1701 Pantheon+SH0ES stat+sys covariance;
#   * GLS-analytic profiling of (M_B, q) with q = -5 log10(scale/REF) applied ONLY to the
#     Hubble-flow indicator w (calibrators carry the geometric Cepheid distance CEPH_DIST and
#     no cosmological scale) -- so the anchored scale is the offset between the calibrator-
#     pinned M_B and the Hubble-flow SNe, i.e. the SH0ES local-ladder H0;
#   * the shape parameter is gridded (LCDM: Om; tracker: fv0) or FIXED (free-history: the
#     Probe R joint SN+BAO+CMB best-fit f_v(z) node vector) or RE-FIT (Part D robustness).
#
# CONVENTIONS. modelv D_M(z) is the dressed transverse comoving distance in units c/Hbar0,
# so d_L = (c/Hbar0)(1+zHEL) D_M(zHD) is the physical luminosity distance in Mpc. Anchoring
# the SN absolute magnitude with the Cepheid calibrators therefore pins Hbar0 (the BARE
# Hubble constant) in km/s/Mpc. The two dressed-H0 conventions Probe R reports both follow:
#     anchored_H0_gdress   = g_dress(fv0) * Hbar0        (Wiltshire algebraic tracker scale)
#     anchored_H0_fullrate = Hd(0)        * Hbar0        (instantaneous present dressed rate)
# and the directly-measured local Hubble slope of the ladder is Hbar0 / S0 with
# S0 = dD_M/dz|_0. For the TRACKER S0 = 1/g_dress = 1/Hd(0) (all three coincide -> paper-1's
# single number 73.0). For a FREE history f_v'(0)!=0 makes Hd(0) != g_dress, and the low-z
# slope tracks Hd(0) (dD_M/dz|_0 = D_H(0) = 1/Hd(0)); so the locally-measured (full-rate)
# dressed H0 stays pinned at ~73 while the g_dress convention splits off.
#
# Outputs probes_out/phaseF_freshH0.json (checkpointed; honours PHASEF_WALL_S soft limit).
import os, sys, json, hashlib, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
os.chdir(_SRC)                                   # so fit_timescape's data/ relative paths work
sys.path.insert(0, _HERE)                        # probes/
sys.path.insert(0, _SRC)                         # src/

import modelv_theory as MV                        # free-history dressed-geometry solver
import fit_timescape as F                         # D_shape_LCDM (paper-1, unmodified)
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import pandas as pd

C_KMS = 299792.458
H0REF = 70.0                                       # reference scale for LCDM/tracker local H0
HBAR0REF = 55.0                                    # reference bare scale for free-history fit
DATA = "data/PantheonSH0ES.dat"
COV = "data/PantheonSH0ES_STATSYS.cov"
OUT = os.path.join(_SRC, "probes", "..", "..", "probes_out", "phaseF_freshH0.json")
OUT = os.path.abspath(OUT)
PROBER = os.path.abspath(os.path.join(_SRC, "..", "probes_out", "modelV_probeR.json"))

# free-history best-fit history (Probe R joint SN+BAO+CMB winner V)
Z_NODES = [0.0, 0.3, 0.7, 1.3, 2.33]
FV_NODES_V = [0.64013, 0.53112, 0.39578, 0.27945, 0.19359]

# paper-1 claim-A anchors / references
H0_TS_CMB, SIG_CMB_STAT, SIG_CMB_SYS = 61.0, 0.79, 4.88
H0_PLANCK, SIG_PLANCK = 67.36, 0.54
H0_SH0ES, SIG_SH0ES = 73.04, 1.04
WINDOW = (0.17, 0.22)                              # predicted apparent-Hubble variance window
PAPER1_TRACKER_FRESH_H0 = 73.0                     # paper-1 freshH0 tracker anchored dressed H0

WALL_S = float(os.environ.get("PHASEF_WALL_S", "780"))  # soft wall-clock budget (s)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


# ----------------------------------------------------------------------
# GLS: r = y - MB*1 - q*w ; (MB, q) profiled analytically, shape gridded
# (verbatim structure from paper-1 freshH0.run_model / package)
# ----------------------------------------------------------------------
def run_model(cf, w, mb_sel, mu0_of_shape, grid):
    n = len(mb_sel)
    A = np.column_stack([np.ones(n), w.astype(float)])
    CiA = cho_solve(cf, A)
    M = A.T @ CiA
    Minv = np.linalg.inv(M)
    chi2_g = np.empty(len(grid)); MB_g = np.empty(len(grid)); q_g = np.empty(len(grid))
    for k, s in enumerate(grid):
        y = mb_sel - mu0_of_shape(s)
        Ciy = cho_solve(cf, y)
        v = A.T @ Ciy
        th = Minv @ v
        chi2_g[k] = y @ Ciy - v @ th
        MB_g[k], q_g[k] = th
    return chi2_g, MB_g, q_g, Minv


def extract(grid, chi2_g, MB_g, q_g, Minv, n, label, ref):
    """Profile the scale (=ref*10^(-q/5)) and shape errors. `ref` sets the scale units:
    ref=H0REF -> local Hubble H0 (LCDM/tracker); ref=HBAR0REF -> bare Hbar0 (free)."""
    i = int(np.argmin(chi2_g)); cmin = float(chi2_g[i])
    railed = bool(i == 0 or i == len(grid) - 1)
    d = chi2_g - cmin
    sh_lo = float(grid[i] - np.interp(1.0, d[:i+1][::-1], grid[:i+1][::-1])) if (i > 0 and d[:i+1].max() >= 1.0) else float("nan")
    sh_hi = float(np.interp(1.0, d[i:], grid[i:]) - grid[i]) if (i < len(grid)-1 and d[i:].max() >= 1.0) else float("nan")
    sq2 = float(Minv[1, 1])
    span = 12.0 * np.sqrt(sq2)
    qg = np.linspace(q_g[i]-span, q_g[i]+span, 4001); dprof = np.zeros_like(qg); j = 0
    for _ in range(8):
        qg = np.linspace(q_g[i]-span, q_g[i]+span, 4001)
        prof = np.min(chi2_g[None, :] + (qg[:, None]-q_g[None, :])**2/sq2, axis=1)
        j = int(np.argmin(prof)); dprof = prof - prof[j]
        if dprof[0] >= 1.0 and dprof[-1] >= 1.0 and 0 < j < len(qg)-1:
            break
        span *= 2.0
    qhat = float(qg[j])
    qlo = float(np.interp(1.0, dprof[:j+1][::-1], qg[:j+1][::-1]))
    qhi = float(np.interp(1.0, dprof[j:], qg[j:]))
    scale = ref * 10 ** (-qhat / 5.0)
    scale_hi = ref * 10 ** (-qlo / 5.0) - scale
    scale_lo = scale - ref * 10 ** (-qhi / 5.0)
    sig = 0.5 * (scale_hi + scale_lo)
    res = dict(label=label, n=n, dof=n-3, ref=ref,
               shape_best=float(grid[i]), shape_err_lo=sh_lo, shape_err_hi=sh_hi,
               shape_railed=railed,
               scale=float(scale), scale_err_lo=float(scale_lo), scale_err_hi=float(scale_hi),
               scale_err_sym=float(sig), MB=float(MB_g[i]), MB_err_cond=float(np.sqrt(Minv[0, 0])),
               chi2_min=cmin, chi2_per_dof=cmin/(n-3))
    return res


# ----------------------------------------------------------------------
# modelv shape helpers: solve once, cache D_M(HF z), S0, Hd(0), g_dress
# ----------------------------------------------------------------------
def solve_free(fv_nodes, z_nodes=Z_NODES, lapse="algebraic", Ngrid=6000):
    fv = MV.fv_from_nodes(fv_nodes, z_nodes=z_nodes)
    return MV.modelv_solve(fv, lapse=lapse, Ngrid=Ngrid)

def solve_tracker(fv0, Ngrid=6000):
    return MV.modelv_solve(MV.tracker_fv_of_z(fv0), lapse="algebraic", Ngrid=Ngrid)

def s0_slope(sol):
    zs = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    return float(np.polyfit(zs, sol.D_M(zs), 1)[0])

def dressed_from_hbar0(sol, Hbar0):
    fv0 = sol.fv0
    gd = float(MV.g_dress(fv0))
    hd0 = float(np.interp(0.0, sol.z, sol.Hd))
    s0 = s0_slope(sol)
    return dict(fv0=float(fv0), g_dress=gd, Hd0=hd0, S0=s0, Hbar0=float(Hbar0),
                H0_gdress=float(gd*Hbar0), H0_fullrate=float(hd0*Hbar0),
                H0_local_slope=float(Hbar0/s0))


def mu_free(sol, zhd_hf, zhel_hf, Hbar0ref):
    dL = (C_KMS/Hbar0ref) * (1.0+zhel_hf) * sol.D_M(zhd_hf)
    return 5.0*np.log10(dL) + 25.0

def mu_lcdm(zhd_hf, zhel_hf, Om, H0ref):
    dL = (C_KMS/H0ref) * (1.0+zhel_hf) * F.D_shape_LCDM(zhd_hf, Om)
    return 5.0*np.log10(dL) + 25.0


def clean(o):
    if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):  return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o): return None
    return o


# ----------------------------------------------------------------------
def main():
    log("loading Pantheon+SH0ES...")
    df = pd.read_csv(DATA, sep=r"\s+")
    n_all = len(df)
    with open(COV) as f:
        n_cov = int(f.readline())
    assert n_cov == n_all, f"cov {n_cov} != rows {n_all}"
    Cfull = np.fromfile(COV, sep=" ")[1:].reshape(n_cov, n_cov)
    md5_dat = hashlib.md5(open(DATA, "rb").read()).hexdigest()
    md5_cov = hashlib.md5(open(COV, "rb").read()).hexdigest()
    log(f"N={n_all}, cov {n_cov}x{n_cov}")

    zHD = df["zHD"].to_numpy(float); zHEL = df["zHEL"].to_numpy(float)
    mb = df["m_b_corr"].to_numpy(float); ceph = df["CEPH_DIST"].to_numpy(float)
    iscal = df["IS_CALIBRATOR"].to_numpy(float).astype(int) == 1
    used_hf = df["USED_IN_SH0ES_HF"].to_numpy(float).astype(int) == 1

    prober = json.load(open(PROBER))
    glob = prober["V"]                              # free-history joint SN+BAO+CMB best fit
    log(f"Probe R global (V): Hbar0={glob['Hbar0']:.3f} gdress={glob['H0_dressed_gdress']:.3f} "
        f"fullrate={glob['H0_dressed_Hd0']:.3f} fv0={glob['fv0']:.5f}")

    variants = {
        "main_z001":     (~iscal) & (zHD > 0.01),  # paper-1 cosmology cut
        "shoes_hf_flag": (~iscal) & used_hf,        # Riess-style 0.0233<z<0.15 HF sample
        "hf_z_gt_010":   (~iscal) & (zHD > 0.10),   # beyond homogeneity/void scales
    }
    om_grid = np.arange(0.05, 0.6000001, 0.0025)
    fv0_grid = np.arange(0.50, 0.9000001, 0.005)   # tracker sanity grid (ModelV)

    results = {}
    for vname, hf in variants.items():
        sel = iscal | hf
        idx = np.where(sel)[0]
        w_s = hf[idx]; cal_s = iscal[idx]
        zhd_s, zhel_s, mb_s, ceph_s = zHD[idx], zHEL[idx], mb[idx], ceph[idx]
        n_sel = len(idx)
        zhd_hf, zhel_hf = zhd_s[w_s], zhel_s[w_s]
        log(f"variant {vname}: n={n_sel} (cal={int(cal_s.sum())}, HF={int(w_s.sum())}), "
            f"HF z {zhd_hf.min():.4f}..{zhd_hf.max():.4f}; Cholesky...")
        cf = cho_factor(Cfull[np.ix_(sel, sel)])
        mu0base = np.where(cal_s, ceph_s, 0.0)

        # -- LCDM sanity (paper-1 anchored gate ~73.5) --
        def mu0_L(Om):
            m = mu0base.copy(); m[w_s] = mu_lcdm(zhd_hf, zhel_hf, Om, H0REF); return m
        rL = extract(om_grid, *run_model(cf, w_s, mb_s, mu0_L, om_grid), n_sel,
                     f"{vname}/LCDM", H0REF)
        rL["H0_local"] = rL["scale"]

        # -- tracker sanity via ModelV (reproduce paper-1 fresh ~73.0) --
        trk_sols = {fv0: solve_tracker(fv0) for fv0 in fv0_grid}
        def mu0_T(fv0):
            m = mu0base.copy(); m[w_s] = mu_free(trk_sols[fv0], zhd_hf, zhel_hf, HBAR0REF); return m
        rT = extract(fv0_grid, *run_model(cf, w_s, mb_s, mu0_T, fv0_grid), n_sel,
                     f"{vname}/tracker", HBAR0REF)
        solTb = trk_sols[rT["shape_best"]]
        rT["dressed"] = dressed_from_hbar0(solTb, rT["scale"])   # Hbar0 = scale

        # -- free-history FIXED shape (Probe R V) anchored --
        solV = solve_free(FV_NODES_V, Ngrid=30000)
        def mu0_F(_):
            m = mu0base.copy(); m[w_s] = mu_free(solV, zhd_hf, zhel_hf, HBAR0REF); return m
        rF = extract(np.array([0.0]), *run_model(cf, w_s, mb_s, mu0_F, np.array([0.0])),
                     n_sel, f"{vname}/free_fixed", HBAR0REF)
        rF["dressed"] = dressed_from_hbar0(solV, rF["scale"])    # Hbar0 = scale
        rF.pop("shape_best", None)

        results[vname] = dict(n=n_sel, n_cal=int(cal_s.sum()), n_hf=int(w_s.sum()),
                              lcdm=rL, tracker=rT, free_fixed=rF)
        log(f"  {vname}: LCDM H0={rL['scale']:.2f}+-{rL['scale_err_sym']:.2f} | "
            f"tracker fv0={rT['shape_best']:.3f} local={rT['dressed']['H0_local_slope']:.2f} | "
            f"free Hbar0={rF['scale']:.2f} gdress={rF['dressed']['H0_gdress']:.2f} "
            f"fullrate={rF['dressed']['H0_fullrate']:.2f} local={rF['dressed']['H0_local_slope']:.2f}")

    # ---- LCDM validation gate (paper-1: main ~73.5) ----
    H0L = results["main_z001"]["lcdm"]["scale"]; sL = results["main_z001"]["lcdm"]["scale_err_sym"]
    gate = (71.5 <= H0L <= 75.0) and (0.6 <= sL <= 1.6)
    log(f"GATE LCDM main: H0={H0L:.2f}+-{sL:.2f} -> {'PASS' if gate else 'FAIL'}")

    # ---- tension + delta_local_excess (fixed-shape free history, main variant) ----
    mv = results["main_z001"]
    dF = mv["free_fixed"]["dressed"]
    def T(a, sa, b, sb): return abs(a-b)/np.hypot(sa, sb)
    sig_cmb_tot = float(np.hypot(SIG_CMB_STAT, SIG_CMB_SYS))
    tension = dict(
        free_local_fullrate_vs_SH0ES=float(abs(dF["H0_fullrate"]-H0_SH0ES)),
        free_local_fullrate=dF["H0_fullrate"], free_local_gdress=dF["H0_gdress"],
        paper1_tracker_fresh_H0=PAPER1_TRACKER_FRESH_H0,
        lcdm_main_anchored_H0=H0L,
        shoes_published_vs_Planck=float(T(H0_SH0ES, SIG_SH0ES, H0_PLANCK, SIG_PLANCK)))

    # local (ladder) vs global (Probe R BAO+CMB joint) dressed excess -- convention-consistent
    def excess(a, b): return float(a/b - 1.0)
    exc_hbar0 = excess(dF["Hbar0"], glob["Hbar0"])
    exc_gdress = excess(dF["H0_gdress"], glob["H0_dressed_gdress"])
    exc_fullrate = excess(dF["H0_fullrate"], glob["H0_dressed_Hd0"])
    # for context: excess vs the fixed 61.0 CMB timescape value paper-1 used
    exc_vs_cmb61_full = excess(dF["H0_fullrate"], H0_TS_CMB)
    exc_vs_cmb61_gd = excess(dF["H0_gdress"], H0_TS_CMB)
    delta_local = dict(
        window=list(WINDOW),
        excess_Hbar0_local_over_global=exc_hbar0,
        excess_gdress_local_over_global=exc_gdress,
        excess_fullrate_local_over_global=exc_fullrate,
        global_ref="Probe R V joint SN+BAO+CMB (Hbar0=%.3f)" % glob["Hbar0"],
        excess_fullrate_vs_cmb61=exc_vs_cmb61_full,
        excess_gdress_vs_cmb61=exc_vs_cmb61_gd,
        in_window_local_over_global=bool(WINDOW[0] <= exc_gdress <= WINDOW[1]),
        note=("local-over-global excess is convention-independent (same fixed shape -> same "
              "g_dress, Hd(0); it reduces to Hbar0_anchored/Hbar0_global-1). It measures the "
              "residual local-vs-early Hubble offset of the free-history model once the SH0ES "
              "ladder pins the SN absolute scale."))

    out = dict(
        probe="phaseF_freshH0",
        purpose=("calibrator-anchored late-time dressed H0 for FREE-HISTORY timescape from "
                 "Pantheon+SH0ES (Cepheid calibrators pin M_B; full stat+sys cov; (MB,q) "
                 "GLS-profiled). Answers whether freeing f_v(z) moves the anchored H0 off "
                 "paper-1's tracker value 73.0."),
        data=dict(dat_rows=n_all, cov_dim=n_cov, md5_dat=md5_dat, md5_cov=md5_cov,
                  n_calibrators=int(iscal.sum())),
        conventions=dict(
            free_scale="d_L=(c/Hbar0)(1+zHEL)D_M(zHD); anchoring pins Hbar0 (bare) in km/s/Mpc",
            H0_gdress="g_dress(fv0)*Hbar0 (Wiltshire algebraic tracker scale)",
            H0_fullrate="Hd(0)*Hbar0 (instantaneous present dressed rate; = local ladder slope)",
            H0_local_slope="Hbar0/S0, S0=dD_M/dz|_0; equals H0_fullrate since dD_M/dz|_0=1/Hd(0)"),
        global_reference=dict(Hbar0=glob["Hbar0"], H0_gdress=glob["H0_dressed_gdress"],
                              H0_fullrate=glob["H0_dressed_Hd0"], fv0=glob["fv0"]),
        lcdm_validation_gate=dict(passed=bool(gate), H0=H0L, sigma=sL, expected="~73.5 (71.5-75)"),
        variants=results, tension=tension, delta_local_excess=delta_local)
    json.dump(clean(out), open(OUT, "w"), indent=1, allow_nan=False)
    log(f"wrote CORE {OUT}")

    # ==================================================================
    # PART D (robustness): RE-FIT the free-history node vector to Pantheon+SH0ES
    # (SN ladder only, no BAO+CMB) on the main variant, multi-start; confirms the
    # anchored local H0 is stable under re-fitting the shape to the anchored SN data.
    # ==================================================================
    hf = variants["main_z001"]; sel = iscal | hf; idx = np.where(sel)[0]
    w_s = hf[idx]; cal_s = iscal[idx]
    zhd_s, zhel_s, mb_s, ceph_s = zHD[idx], zHEL[idx], mb[idx], ceph[idx]
    zhd_hf, zhel_hf = zhd_s[w_s], zhel_s[w_s]
    cf = cho_factor(Cfull[np.ix_(sel, sel)])
    mu0base = np.where(cal_s, ceph_s, 0.0)

    # parametrise 5 monotone-decreasing nodes: fv0 in (0.05,0.95), ratios s_i in (0.05,0.999)
    def unpack(p):
        fv0 = 0.05 + 0.90/(1.0+np.exp(-p[0]))
        s = 0.05 + 0.949/(1.0+np.exp(-p[1:]))
        nodes = [fv0]
        for si in s:
            nodes.append(nodes[-1]*si)
        return np.array(nodes)

    def objective(p):
        nodes = unpack(p)
        try:
            sol = solve_free(nodes, Ngrid=6000)
        except Exception:
            return 1e9
        def mu0_(_):
            m = mu0base.copy(); m[w_s] = mu_free(sol, zhd_hf, zhel_hf, HBAR0REF); return m
        r = extract(np.array([0.0]), *run_model(cf, w_s, mb_s, mu0_, np.array([0.0])),
                    len(idx), "refit", HBAR0REF)
        return r["chi2_min"]

    def pack_from_nodes(nodes):
        nodes = np.asarray(nodes, float)
        fv0 = nodes[0]
        p0 = np.log((fv0-0.05)/(0.95-fv0))
        ps = []
        for i in range(1, len(nodes)):
            si = np.clip(nodes[i]/nodes[i-1], 0.051, 0.998)
            ps.append(np.log((si-0.05)/(0.999-si)))
        return np.array([p0]+ps)

    rng = np.random.default_rng(20260706)
    starts = [pack_from_nodes(FV_NODES_V)]
    for _ in range(11):
        fv0 = rng.uniform(0.45, 0.85)
        ratios = rng.uniform(0.55, 0.9, 4)
        nodes = [fv0]
        for rr in ratios: nodes.append(nodes[-1]*rr)
        starts.append(pack_from_nodes(nodes))

    best = dict(chi2=np.inf, nodes=list(FV_NODES_V), n_done=0, railed=False)
    ckpt = dict(refit=dict(status="running", **best))
    def write_ckpt():
        out2 = json.load(open(OUT)); out2["refit"] = ckpt["refit"]
        json.dump(clean(out2), open(OUT, "w"), indent=1, allow_nan=False)

    log(f"Part D: re-fitting free history to Pantheon+SH0ES (main), {len(starts)} starts, "
        f"soft limit {WALL_S:.0f}s")
    for si, p0 in enumerate(starts):
        if time.time() - t0 > WALL_S:
            log(f"soft limit hit after {si} starts; stopping re-fit"); break
        res = minimize(objective, p0, method="Nelder-Mead",
                       options=dict(maxiter=1200, maxfev=1200, xatol=1e-4, fatol=1e-4))
        nodes = unpack(res.x)
        if res.fun < best["chi2"]:
            best = dict(chi2=float(res.fun), nodes=[float(x) for x in nodes], n_done=si+1,
                        railed=bool(nodes[0] > 0.945 or nodes[0] < 0.055))
        best["n_done"] = si + 1
        ckpt["refit"] = dict(status="running", **best)
        write_ckpt()
        log(f"  start {si+1}/{len(starts)}: chi2={res.fun:.3f} fv0={nodes[0]:.4f} "
            f"best={best['chi2']:.3f}")

    # anchored dressed H0 for the re-fit best shape (fine solve)
    solB = solve_free(best["nodes"], Ngrid=30000)
    def mu0_B(_):
        m = mu0base.copy(); m[w_s] = mu_free(solB, zhd_hf, zhel_hf, HBAR0REF); return m
    rB = extract(np.array([0.0]), *run_model(cf, w_s, mb_s, mu0_B, np.array([0.0])),
                 len(idx), "refit_best", HBAR0REF)
    dB = dressed_from_hbar0(solB, rB["scale"])
    ckpt["refit"] = dict(status="done", chi2=best["chi2"], nodes=best["nodes"],
                         n_starts=len(starts), n_done=best["n_done"], railed=best["railed"],
                         fv0=dB["fv0"], Hbar0=dB["Hbar0"], H0_gdress=dB["H0_gdress"],
                         H0_fullrate=dB["H0_fullrate"], H0_local_slope=dB["H0_local_slope"],
                         chi2_at_best=rB["chi2_min"],
                         note=("SN-ladder-only re-fit of the full 5-node history is under-"
                               "constrained at high z (no BAO/CMB anchor); the low-z slope "
                               "(hence anchored local H0) is the robust output."))
    write_ckpt()
    log(f"Part D done: refit fv0={dB['fv0']:.4f} Hbar0={dB['Hbar0']:.2f} "
        f"gdress={dB['H0_gdress']:.2f} fullrate={dB['H0_fullrate']:.2f} "
        f"local={dB['H0_local_slope']:.2f} chi2={best['chi2']:.2f}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
