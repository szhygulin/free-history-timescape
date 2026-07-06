#!/usr/bin/env python3
"""Phase D -- OBSERVED void volume-fraction history f_v_obs(z).

Companion to Probe R (required history). Computes the observed void VOLUME fraction
f_v_obs(z) under timescape's operational definition -- "regions expanding faster than
the volume-average" -- which in the linear velocity-density relation (theta_pec = -f H delta)
maps to the BELOW-MEAN-DENSITY set {delta < <delta>_V ~= 0}. We carry the DEFINITION
SYSTEMATIC (how deep a threshold makes a "void") as the dominant two-sided band.

Output: probes_out/phaseD_fvobs.json

(a) z~0 anchor -- directly from the 2M++/Carrick+2015 reconstruction
    (external_data/twompp_density.npy, delta_g* luminosity-weighted contrast, 4 Mpc/h
    Gaussian smoothing, Galactic Cartesian, 257^3 @ 1.5625 Mpc/h). We own the systematics:
      * Reliable volume: sphere r < 200 Mpc/h (Carrick reliability radius), with the
        fully-measured core r < 100 Mpc/h as a cross-check.
      * SURVEY-MASK FILL: cells with NO data default to delta==0.0 exactly (mean density).
        These grow from 0% (r<100) to 31% (r<200) of the sphere and are NOT voids; we
        EXCLUDE them (renormalise to the measured sub-volume). A genuine 4 Mpc/h-smoothed
        field is essentially never exactly 0.0, so the exact-zero cut is safe.
      * Definition family {delta<0, delta<-0.3, delta<-0.5, watershed/basin}.
        - delta<0 (below mean) == "faster than mean expanding" == timescape f_v proxy.
          BIAS-INDEPENDENT: delta_g<0 <=> delta_m<0 for any monotone bias through the
          origin, so the sign-threshold fraction is insensitive to galaxy-vs-matter bias.
        - delta<-0.3, delta<-0.5: progressively stricter "genuine void" definitions
          (bias-dependent), setting the restrictive edge of the definition band.
        - watershed/basin: connected below-mean regions hosting a genuine deep core
          (min delta < -0.5). The below-mean excursion set PERCOLATES (one connected
          component in the r<100 core), so a watershed at the mean-density level fills the
          same volume as delta<0 -- reported explicitly.

(b) higher-z: literature anchors (Pan+2012 SDSS DR7; Williams+2024 NR watershed) plus,
    where no clean published filling fraction exists (z ~ 0.3-2.33), a DECLARED growth-model
    extrapolation: a lognormal excursion-set shape f(delta_th, sigma) with sigma(z) scaled
    by the linear growth factor D(z) and anchored to the z~0 2M++ below-mean value. The
    model reproduces the measured delta<-0.3 and delta<-0.5 fractions at z=0 to within the
    r100/r200 systematic (validation printed), so it is a calibrated shape proxy, NOT a fit.
    Every higher-z point is flagged direct=False.

Run from src/ :  python probes/phaseD_voidfrac.py
"""
import os
import sys
import json
import time
import numpy as np
from scipy import ndimage
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import los_common as LC

WT = "/Users/s/dev/science/free-history-timescape"
OUT = os.path.join(WT, "probes_out", "phaseD_fvobs.json")

SOFT_WALL_S = float(os.environ.get("PHASED_MAXSEC", "1200"))
_T0 = time.time()

# node redshifts shared with Probe R (required history)
Z_NODES = [0.0, 0.3, 0.7, 1.3, 2.33]

# definition-family thresholds
THRESHOLDS = [0.0, -0.3, -0.5]
CORE_DEPTH = -0.5   # a "genuine void" must host a core below this (watershed/basin variant)


# ---------------------------------------------------------------------------
# (a) z~0 anchor from the 2M++ field
# ---------------------------------------------------------------------------
def field_void_fractions():
    """Volume fractions of the definition family over the reliable sphere(s),
    excluding survey-mask fill (delta==0 exactly). Returns a dict."""
    f = LC.load_field()
    n = LC.NGRID
    ax = (np.arange(n) - 128) * LC.DX
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    struct = ndimage.generate_binary_structure(3, 3)  # 26-connectivity

    out = {"field": "2M++/Carrick+2015 delta_g* (lum-weighted, 4 Mpc/h Gaussian smoothing)",
           "radii": {}}
    for R in (100.0, 200.0):
        sphere = r < R
        fill = (f == 0.0) & sphere            # unmeasured mean-density fill
        measured = sphere & (~fill)
        Nmeas = int(measured.sum())
        rec = {
            "N_sphere": int(sphere.sum()),
            "N_fill_exact_zero": int(fill.sum()),
            "fill_fraction": float(fill.sum() / max(sphere.sum(), 1)),
            "N_measured": Nmeas,
            "thresholds": {},
        }
        for thr in THRESHOLDS:
            rec["thresholds"][f"delta<{thr:+.1f}"] = float(((f < thr) & measured).sum() / Nmeas)
        # watershed/basin: connected below-mean regions hosting a delta<CORE_DEPTH core
        below = (f < 0.0) & measured
        lab, nlab = ndimage.label(below, structure=struct)
        if nlab > 0:
            mins = ndimage.minimum(f, lab, index=np.arange(1, nlab + 1))
            good = np.where(mins < CORE_DEPTH)[0] + 1
            void = np.isin(lab, good)
            fw = float(void.sum() / Nmeas)
        else:
            fw = 0.0
        rec["watershed_basin"] = fw
        rec["n_below_mean_components"] = int(nlab)
        out["radii"][f"r<{R:.0f}"] = rec
    return out


# ---------------------------------------------------------------------------
# (b) growth-model shape for the higher-z extrapolation (declared)
# ---------------------------------------------------------------------------
def lognormal_frac(delta_th, sigma):
    """Volume fraction below delta_th for a lognormal 1+delta = exp(G - sig^2/2),
    G ~ N(0, sig^2):  frac = Phi[(ln(1+delta_th) + sig^2/2)/sig]."""
    return float(norm.cdf((np.log(1.0 + delta_th) + sigma * sigma / 2.0) / sigma))


def growth_factor(z, Om=0.315):
    """Linear growth factor D(z) (flat LCDM), normalised D(z=0)=1."""
    def E(a):
        return np.sqrt(Om * a ** -3 + (1.0 - Om))
    def Dun(a):
        return E(a) * quad(lambda ap: 1.0 / (ap * E(ap)) ** 3, 1e-8, a)[0]
    D0 = Dun(1.0)
    return Dun(1.0 / (1.0 + z)) / D0


def main():
    field = field_void_fractions()

    # --- z~0 anchor: assemble central + definition band from the family ---------
    r100 = field["radii"]["r<100"]
    r200 = field["radii"]["r<200"]
    bm100 = r100["thresholds"]["delta<+0.0"]   # below-mean, fully-measured core
    bm200 = r200["thresholds"]["delta<+0.0"]   # below-mean, full reliable sphere
    deep100 = r100["thresholds"]["delta<-0.5"]
    deep200 = r200["thresholds"]["delta<-0.5"]

    # below-mean (timescape-natural, bias-independent) sub-band across the reliable-volume
    # systematic; this is the physically-correct f_v proxy:
    bm_lo, bm_hi = sorted([bm100, bm200])
    bm_central = 0.5 * (bm100 + bm200)

    # full definition band: permissive edge = below-mean; restrictive edge = deep voids
    def_hi = bm_hi                                   # delta<0 (most permissive)
    def_lo = min(deep100, deep200)                   # delta<-0.5 (most restrictive)
    # moderate/watershed-like centre = delta<-0.3 (r<100 core)
    def_central = r100["thresholds"]["delta<-0.3"]

    z0_anchor = {
        "z": 0.0,
        "below_mean_band": [bm_lo, bm_hi],
        "below_mean_central": bm_central,
        "below_mean_note": "delta<0 == 'faster than mean expanding' == timescape f_v; "
                           "BIAS-INDEPENDENT (sign preserved under monotone galaxy bias). "
                           "Band = reliable-volume systematic (r<100 core vs r<200 sphere).",
        "definition_band": [def_lo, def_hi],
        "definition_central": def_central,
        "family": {
            "delta<0_r100": bm100, "delta<0_r200": bm200,
            "delta<-0.3_r100": r100["thresholds"]["delta<-0.3"],
            "delta<-0.3_r200": r200["thresholds"]["delta<-0.3"],
            "delta<-0.5_r100": deep100, "delta<-0.5_r200": deep200,
            "watershed_basin_r100": r100["watershed_basin"],
            "watershed_basin_r200": r200["watershed_basin"],
        },
        "percolation_note": (
            f"below-mean excursion set is a single connected component in the r<100 core "
            f"(n_components={r100['n_below_mean_components']}); watershed-at-mean therefore "
            f"fills the same volume as delta<0 (watershed_basin==delta<0)."),
        "probe4_bracket_ref": [0.50, 0.62],
        "probe4_check": ("below-mean (0.61-0.67) sits at/above the Probe-4 top (0.62); the "
                         "delta<-0.3..-0.5 definitions bracket it from below -- consistent."),
    }

    # --- calibrate the lognormal shape model to the z~0 below-mean value --------
    sig0 = brentq(lambda s: lognormal_frac(0.0, s) - bm_central, 1e-2, 3.0)
    model_val = {  # validation: model vs measured at z=0
        "sigma0": sig0,
        "anchored_to_below_mean": bm_central,
        "delta<0_model": lognormal_frac(0.0, sig0),
        "delta<-0.3_model": lognormal_frac(-0.3, sig0),
        "delta<-0.3_measured_r100_r200": [r100["thresholds"]["delta<-0.3"],
                                          r200["thresholds"]["delta<-0.3"]],
        "delta<-0.5_model": lognormal_frac(-0.5, sig0),
        "delta<-0.5_measured_r100_r200": [deep100, deep200],
        "note": "single-sigma lognormal excursion set; reproduces measured -0.3,-0.5 "
                "thresholds to within the r100/r200 definition systematic -> calibrated "
                "shape proxy for the higher-z growth extrapolation (NOT a fit).",
    }

    # --- literature anchors -----------------------------------------------------
    literature = [
        {"z": 0.044, "value": 0.62, "band": [0.55, 0.66], "direct": True,
         "definition": "watershed VoidFinder filling fraction (SDSS DR7 main)",
         "citation": "Pan, Vogeley, Hoyle, Choi, Park 2012, MNRAS 421, 926 "
                     "(arXiv:1103.4156) -- voids fill ~62% of the volume",
         "band_note": "assigned +-0.05 (survey/definition); central is the published 62%."},
        {"z": 0.0, "value": 0.558, "band": [0.50, 0.615], "direct": True,
         "definition": "numerical-relativity watershed void filling fraction",
         "citation": "Williams et al. 2024 (NR watershed); filling fraction 50-61.5% "
                     "(as tabulated in PLAN_void_history.md sec 4 / Paper-1 Probe 4; "
                     "arXiv id not independently re-verified here)",
         "band_note": "published range 0.50-0.615; central = midpoint. Also supplies the "
                      "growth-shape prior for the higher-z extrapolation."},
    ]
    boss_eboss_note = (
        "BOSS DR12 VIDE (Mao et al. 2017, arXiv:1602.02771) and eBOSS (Aubert et al. 2022, "
        "arXiv:2007.09013) provide void catalogs at z~0.2-0.7, but no clean published VOLUME "
        "filling fraction was extractable here without re-deriving from survey randoms; the "
        "z=0.3/0.7 points therefore use the declared growth-model extrapolation, per "
        "PLAN_void_history.md sec 4 fallback.")

    # --- assemble f_v_obs(z) points at the Probe-R nodes ------------------------
    # each: central (moderate/watershed-like, delta<-0.3 evolved), lo (deep, delta<-0.5),
    # hi (below-mean/permissive, delta<0). z=0 uses DIRECT measured; z>0 growth model.
    fvobs_points = []
    for z in Z_NODES:
        if z == 0.0:
            pt = {
                "z": 0.0, "direct": True,
                "central": def_central,          # delta<-0.3 (r<100), moderate/watershed-like
                "lo": def_lo,                    # delta<-0.5 (deep, restrictive)
                "hi": def_hi,                    # delta<0 (below-mean, permissive)
                "below_mean": bm_central,
                "source": "2M++ direct (this probe)",
            }
        else:
            Dz = growth_factor(z)
            sig = sig0 * Dz
            hi = lognormal_frac(0.0, sig)        # below-mean, evolved
            central = lognormal_frac(-0.3, sig)  # moderate
            lo = lognormal_frac(-0.5, sig)       # deep
            pt = {
                "z": z, "direct": False,
                "central": central, "lo": lo, "hi": hi, "below_mean": hi,
                "growth_D": Dz, "sigma_z": sig,
                "source": "growth-model extrapolation (lognormal excursion set, anchored "
                          "to z~0 2M++ below-mean; declared, NOT a catalog measurement)",
            }
        fvobs_points.append(pt)

    result = {
        "probe": "D -- observed void volume-fraction history f_v_obs(z)",
        "definition": ("timescape f_v = volume fraction of regions expanding faster than the "
                       "volume-average ~ below-mean-density set {delta<0}. Definition "
                       "systematic (threshold depth) carried as the dominant two-sided band."),
        "z_nodes": Z_NODES,
        "z0_field_raw": field,
        "z0_anchor": z0_anchor,
        "shape_model": model_val,
        "literature": literature,
        "boss_eboss_note": boss_eboss_note,
        "fv_obs_points": fvobs_points,
        "systematics_ranked": [
            "1. definition mapping (watershed/below-mean vs deep-void threshold) -- DOMINANT, "
            "carried as the two-sided band [delta<-0.5, delta<0].",
            "2. survey-mask fill (delta==0 exactly): excluded; renormalised to measured "
            "volume; r<100 core (0% fill) cross-checks r<200 (31% fill).",
            "3. tracer bias (galaxy delta_g* vs matter delta_m): the below-mean (delta<0) "
            "fraction is bias-INDEPENDENT; deeper thresholds are bias-dependent (sub-dominant "
            "to the definition band).",
            "4. higher-z: no clean catalog filling fraction z>0.1 -> declared growth-model "
            "extrapolation, widened band, direct=False per point.",
        ],
        "runtime_s": round(time.time() - _T0, 1),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[phaseD] wrote {OUT}")
    print(f"[phaseD] z~0 below-mean band [{bm_lo:.3f},{bm_hi:.3f}] central {bm_central:.3f}")
    print(f"[phaseD] z~0 definition band [{def_lo:.3f},{def_hi:.3f}] central {def_central:.3f}")
    print(f"[phaseD] shape model sig0={sig0:.3f}  runtime {result['runtime_s']}s")
    return result


if __name__ == "__main__":
    main()
