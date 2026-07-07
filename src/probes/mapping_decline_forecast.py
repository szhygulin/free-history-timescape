#!/usr/bin/env python3
"""Pre-registration FORECAST for the derived f_v <-> observable mapping (NOTES_mapping.md).

This is NOT telescope-data reduction. It uses only (i) the LCDM linear growth factor D(z)
and growth rate f(z) [the structure-growth NULL], (ii) the committed z~0 smoothed-field
variance sigma0 from Phase D (2M++, 4 Mpc/h), and (iii) the committed required history
f_v^req(z) from Probe R. It demonstrates that the derived expansion-margin mapping

    void  ==  { x : (H_local(x) - Hbar)/Hbar  >=  eps }             (theory: expansion excess)
    linear velocity-density:  (H_local - Hbar)/Hbar = -(f(z)/3) delta
    =>  void  ==  { x : delta(z) <= delta_th(z) },   delta_th(z) = -3 eps / f(z)   (observable: density threshold)
    f_v(z; eps) = P[ delta(z) < delta_th(z) ]   (lognormal field, s(z) = s0 D(z))

(a) dissolves the below-mean floor (f_v -> 0 as z grows, for ANY eps>0), and
(b) makes the decline ratio f_v(z)/f_v(0) a ONE-parameter (eps) prediction, so that
    matching the z=0 LEVEL and the DECLINE SHAPE simultaneously is falsifiable.

It sweeps the single void-margin parameter eps and reports, for each, the predicted z=0
level and the decline ratios, plus the eps that best matches the required DECLINE SHAPE and
the z=0 level it then implies. Output: probes_out/mapping_decline_forecast.json

Run from src/ :  python probes/mapping_decline_forecast.py
"""
import os
import json
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(WT, "probes_out", "mapping_decline_forecast.json")
PROBER = os.path.join(WT, "probes_out", "modelV_probeR.json")
PHASED = os.path.join(WT, "probes_out", "phaseD_fvobs.json")

OM = 0.315                       # matches Phase D growth_factor()
Z_NODES = [0.0, 0.3, 0.7, 1.3, 2.33]


# --- LCDM structure-growth null -------------------------------------------------
def growth_factor(z, Om=OM):
    """Linear growth factor D(z), D(0)=1 (flat LCDM)."""
    def E(a):
        return np.sqrt(Om * a ** -3 + (1.0 - Om))
    def Dun(a):
        return E(a) * quad(lambda ap: 1.0 / (ap * E(ap)) ** 3, 1e-8, a)[0]
    return Dun(1.0 / (1.0 + z)) / Dun(1.0)


def growth_rate(z, Om=OM):
    """Linear growth rate f = dlnD/dlna ~= Omega_m(z)^0.55 (Linder)."""
    a = 1.0 / (1.0 + z)
    Omz = Om / (Om + (1.0 - Om) * a ** 3)
    return Omz ** 0.55


# --- the derived mapping --------------------------------------------------------
def fv_expansion_margin(z, eps, s0, D=None, f=None):
    """Expansion-margin mapping, lognormal field. delta_th(z) = -3 eps / f(z)."""
    if D is None:
        D = growth_factor(z)
    if f is None:
        f = growth_rate(z)
    dth = -3.0 * eps / f
    if dth <= -1.0:
        return 0.0                       # threshold below the empty-void floor delta=-1
    s = s0 * D
    return float(norm.cdf((np.log(1.0 + dth) + s * s / 2.0) / s))


def fv_fixed_density(z, dth, s0, D=None):
    """Comparison: FIXED nonlinear density threshold delta_th const (no f(z) factor)."""
    if D is None:
        D = growth_factor(z)
    s = s0 * D
    if dth <= -1.0:
        return 0.0
    return float(norm.cdf((np.log(1.0 + dth) + s * s / 2.0) / s))


def main():
    R = json.load(open(PROBER))
    Dd = json.load(open(PHASED))
    s0 = Dd["shape_model"]["sigma0"]                       # 0.7345, measured 2M++ 4 Mpc/h
    bm0 = Dd["z0_anchor"]["below_mean_central"]            # 0.6433 (below-mean z=0 level)

    req = R["V"]["fv_nodes"]
    band = R["fv_req_band_dchi2_le1"]
    req_lo = [band[f"z={z:g}"][0] for z in Z_NODES]
    req_hi = [band[f"z={z:g}"][1] for z in Z_NODES]

    # required decline ratios r_i = f_v^req(z_i)/f_v^req(0), with a conservative band
    req_ratio = [req[i] / req[0] for i in range(len(Z_NODES))]
    req_ratio_lo = [req_lo[i] / req_hi[0] for i in range(len(Z_NODES))]
    req_ratio_hi = [req_hi[i] / req_lo[0] for i in range(len(Z_NODES))]

    Dz = [growth_factor(z) for z in Z_NODES]
    fz = [growth_rate(z) for z in Z_NODES]

    # --- 1. sweep the single void-margin eps ----------------------------------
    eps_grid = np.linspace(0.0, 0.17, 35)                 # eps < f0/3 ~ 0.177 keeps delta_th>-1
    sweep = []
    for eps in eps_grid:
        fv = [fv_expansion_margin(z, eps, s0, D=Dz[i], f=fz[i])
              for i, z in enumerate(Z_NODES)]
        if fv[0] <= 0:
            continue
        ratio = [v / fv[0] for v in fv]
        sweep.append({
            "eps": float(eps),
            "delta_th_z0": float(-3.0 * eps / fz[0]),
            "fv0": fv[0],
            "fv": fv,
            "decline_ratio": ratio,
            "total_decline_fv0_over_fv233": float(fv[0] / fv[-1]) if fv[-1] > 0 else None,
        })

    # --- 2. eps that best matches the required DECLINE SHAPE (z>0 ratios) -------
    def shape_cost(eps):
        fv = [fv_expansion_margin(z, eps, s0, D=Dz[i], f=fz[i])
              for i, z in enumerate(Z_NODES)]
        if fv[0] <= 0:
            return 1e9
        ratio = [v / fv[0] for v in fv]
        return sum((ratio[i] - req_ratio[i]) ** 2 for i in range(1, len(Z_NODES)))

    res = minimize_scalar(shape_cost, bounds=(1e-3, 0.176), method="bounded")
    eps_best = float(res.x)
    fv_best = [fv_expansion_margin(z, eps_best, s0, D=Dz[i], f=fz[i])
               for i, z in enumerate(Z_NODES)]
    ratio_best = [v / fv_best[0] for v in fv_best]

    # --- 3. eps that best matches the required z=0 LEVEL (fv0 = required 0.640) -
    #        and the decline shape it then predicts
    def level_cost(eps):
        return (fv_expansion_margin(0.0, eps, s0, D=1.0, f=fz[0]) - req[0]) ** 2
    # required level 0.640 needs delta_th_z0 >= 0 (above mean) -> no eps>0 reaches it; report boundary
    fv0_at_eps0 = fv_expansion_margin(0.0, 0.0, s0, D=1.0, f=fz[0])   # = P(delta<0), the floor

    # --- 4. the two incompatible edges made explicit --------------------------
    # (a) match level 0.640 -> eps ~ 0 -> below-mean floor -> decline:
    fv_levelmatch = [fv_expansion_margin(z, 1e-6, s0, D=Dz[i], f=fz[i])
                     for i, z in enumerate(Z_NODES)]
    ratio_levelmatch = [v / fv_levelmatch[0] for v in fv_levelmatch]

    # comparison mapping: fixed nonlinear density threshold (steeper decline, no f(z))
    # anchored so its z=0 level equals the shape-matched eps level, for contrast
    dth_fixed = brentq(lambda d: fv_fixed_density(0.0, d, s0, D=1.0) - fv_best[0],
                       -0.999, -1e-4)
    fv_fixed = [fv_fixed_density(z, dth_fixed, s0, D=Dz[i]) for i, z in enumerate(Z_NODES)]
    ratio_fixed = [v / fv_fixed[0] for v in fv_fixed]

    result = {
        "probe": "mapping decline FORECAST (pre-registration support for NOTES_mapping.md)",
        "not_data_reduction": ("Uses only the LCDM growth NULL D(z),f(z), the committed z~0 "
                               "field variance sigma0 (Phase D 2M++), and the committed required "
                               "history. No BOSS/eBOSS/DESI void data touched. Replaces D(z) with "
                               "measured sigma(z)/sigma0 at execution."),
        "mapping": ("void = {(H_local-Hbar)/Hbar >= eps}; linear map -3eps/f(z)=delta_th(z); "
                    "f_v(z)=P[delta(z)<delta_th(z)], lognormal s(z)=s0 D(z)."),
        "inputs": {
            "s0_measured_2Mpp_4Mpch": s0,
            "below_mean_z0_level": bm0,
            "OM_growth": OM,
            "z_nodes": Z_NODES,
            "D_z_LCDM": Dz,
            "f_growthrate_z": fz,
        },
        "required": {
            "fv": req, "fv_lo": req_lo, "fv_hi": req_hi,
            "decline_ratio": req_ratio,
            "decline_ratio_lo": req_ratio_lo,
            "decline_ratio_hi": req_ratio_hi,
            "total_decline_x": req[0] / req[-1],
        },
        "floor_demonstration": {
            "fv0_at_eps0_is_below_mean": fv0_at_eps0,
            "note": ("eps=0 (below-mean threshold) reproduces the Phase-D floor level and "
                     "does NOT decline to 0; any eps>0 forces delta_th<0 and f_v->0 as z grows."),
        },
        "eps_sweep": sweep,
        "shape_matched": {
            "eps": eps_best,
            "delta_th_z0": float(-3.0 * eps_best / fz[0]),
            "fv0_implied": fv_best[0],
            "fv0_required": req[0],
            "fv0_shortfall_vs_required": req[0] - fv_best[0],
            "fv": fv_best,
            "decline_ratio": ratio_best,
            "note": ("eps tuned to the required DECLINE SHAPE; the z=0 LEVEL it then predicts "
                     "is fv0_implied. If fv0_implied << required 0.640, the level and shape are "
                     "incompatible under the derived mapping -> SHAPE-UNAVAILABLE lean."),
        },
        "level_matched": {
            "fv": fv_levelmatch,
            "decline_ratio": ratio_levelmatch,
            "total_decline_x": fv_levelmatch[0] / fv_levelmatch[-1],
            "note": ("eps->0 to match the required z=0 LEVEL 0.640 (below-mean); the decline it "
                     "predicts is far too flat vs required (floor)."),
        },
        "fixed_density_comparison": {
            "delta_th_const": dth_fixed,
            "fv": fv_fixed,
            "decline_ratio": ratio_fixed,
            "note": ("FIXED nonlinear density threshold (no f(z) factor), anchored to the same "
                     "z=0 level as shape_matched; declines STEEPER than the expansion-margin "
                     "mapping -- shows the f(z) factor makes the theory-faithful mapping shallower."),
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[forecast] wrote {OUT}")
    print(f"[forecast] s0={s0:.4f}  below-mean z0 level={bm0:.4f}")
    print(f"[forecast] required decline x{req[0]/req[-1]:.2f}  ratios={['%.3f'%r for r in req_ratio]}")
    print(f"[forecast] floor: f_v0 at eps=0 (below-mean) = {fv0_at_eps0:.4f}")
    print(f"[forecast] SHAPE-matched eps={eps_best:.4f} -> f_v0={fv_best[0]:.4f} "
          f"(required 0.640; shortfall {req[0]-fv_best[0]:.3f})")
    print(f"[forecast]   shape-matched ratios = {['%.3f'%r for r in ratio_best]}")
    print(f"[forecast] LEVEL-matched (eps~0) decline x{fv_levelmatch[0]/fv_levelmatch[-1]:.2f} "
          f"ratios={['%.3f'%r for r in ratio_levelmatch]}")
    return result


if __name__ == "__main__":
    main()
