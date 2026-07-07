#!/usr/bin/env python3
"""
WP-C : the sound horizon r_d computed IN-MODEL (radiation-era bare frame),
replacing the external r_d = 147.09 Mpc adopted as a "disclosed crutch" at
   src/harness.py:35,54            RD / H0_from_alpha
   src/timescape_baocmb.py:73-75   RD_STD, the Planck acoustic point

PHYSICS  (Duley, Nazer & Wiltshire 2013, "Timescape cosmology with radiation
fluid", arXiv:1306.3208 = DNW13).  In the radiation era the void fraction
f_v -> 0, so the two-scale backreaction (Q, curvature) is negligible and every
timescape history -- tracker OR free -- reduces to the SAME bare two-fluid
(matter + radiation) Friedmann problem.  r_d is then fixed by the bare density
parameters (Omega_M0-bar, Omega_R0-bar, Omega_B0-bar) alone; it is NOT sensitive
to the late-time forced f_v(z) shape.  Hence a single bare integration suffices.

EQUATIONS (transcribed verbatim from the ar5iv HTML of arXiv:1306.3208; the raw
LaTeX alttext was extracted and each equation number VERIFIED against the source
-- see the module docstring notes and the JSON "provenance" block):

  Bare Friedmann, radiation era (DNW13 Eq 10, drop Q & curvature as f_v->0):
     H_bar(x)^2 = H_bar0^2 [ Omega_M0_bar x^-3 + Omega_R0_bar x^-4 ],   x = a_bar/a_bar0

  Tracker bare density parameters (DNW13, matter era):
     Omega_M0_bar = 4(1-f_v0)/(2+f_v0)^2                 [VERIFIED Eq-region 20-21]
     Omega_k0_bar = 9 f_v0/(2+f_v0)^2
     gamma_bar0   = (2+f_v0)/2                           [tracker lapse; identically
                    (2+f_v0)/2 given Omega_M0_bar above and Om_dressed=1/2(1-f_v0)(2+f_v0)]

  Bare radiation density (DNW13, def. of Omega_R0_bar):
     Omega_R0_bar = kappa g_* T0^4 / (H_bar0^2 gamma_bar0^4),
        kappa = 4 pi^3 G kB^4 /(45 hbar^3 c^5),  g_* = 3.36,  T0 = 2.725 K
     Omega_gamma0_bar = 2 g_*^-1 Omega_R0_bar            (photons only)
     => bare CMB temperature  T_bar = gamma_bar0^-1 T0   (DNW13: T_bar = gamma_bar^-1 T)
        i.e. the volume-average frame is COOLER; radiation density carries a
        gamma_bar0^-4 suppression.

  Baryons from BBN (DNW13: eta_Bgamma=(5.1+/-0.5)e-10):
     n_bar_B = 3 H_bar0^2 Omega_B0_bar/(8 pi G m_p) (T_bar/T_bar_gamma0)^3
     Physical bare baryon density:  omega_b_bar = omega_b_std(eta)/gamma_bar0^3,
        omega_b_std(eta) = eta_10/273.9   (reproduces DNW13 Omega_B0_bar=0.0303 at
        eta=5.1e-10, H_bar0=50.1 -- self-check in JSON).

  Sound horizon (DNW13 Eq 31, VERIFIED):
     D_bar_s = (a_bar(t)/a_bar0)(c/sqrt3) INT_0^{x_dec} dx /
                 [ x^2 H_bar sqrt(1 + 0.75 x Omega_B0_bar/Omega_gamma0_bar) ]
     The COMOVING ruler r_s (BAO/CMB standard scale) drops the a_bar(t)/a_bar0
     prefactor.  With x^2 H_bar = H_bar0 sqrt(Omega_M0_bar x + Omega_R0_bar) this is
     the standard c_s/H integral; R = 0.75 x Omega_B0_bar/Omega_gamma0_bar (DNW13
     "R = 0.75 rho_B/rho_gamma").

  Drag epoch (DNW13 Eq 49, VERIFIED): c tau_d ~= 1, with
     tau_d(t) = INT_t^t0 sigma_T n_bar_e/(a_bar R) dt   (drag optical depth).
     The full ionization history n_bar_e (Saha+Peebles ODE) is OUT OF SCOPE at
     the compression level (WP-B gate); we adopt STANDARD recombination redshifts
     -- z is set by atomic physics + eta (both frame- and cosmology-robust), and
     eta is fixed by BBN.  z_star (recomb) and z_drag are taken at Planck values;
     the bare integration limit uses x_dec = 1/[gamma_bar0 (1+z_dec)] (DNW13:
     x_dec = z_bar_dec+1 = gamma_bar0 (1+z_dec)/gamma_bar_dec, gamma_bar_dec~=1).

Compression observables (task spec):
     l_A     = pi D_M(z*)/r_s(z*)
     shift R = sqrt(Omega_M0_bar H_bar0^2) D_M(z*)/c
     omega_b (in-model, from BBN eta)

SELF-CONSISTENCY.  The harness marginalises the ruler amplitude alpha=c/(H_bar0 r_d)
analytically, so the BAO+CMB chi2 (hence the R1 verdict) is r_d-INDEPENDENT.  What
r_d fixes is the ABSOLUTE calibration H_bar0 = c/(alpha r_d).  We report whether the
in-model r_d reproduces 147.09 (=> H_bar0 stays 53.79 => R1 reconciliation intact)
or shifts it.

One number -> one script -> one JSON.  Portable __file__ paths.  No fabricated
precision: every adopted-vs-derived choice is tagged; the residual uncertainties
are listed in the JSON.
"""
import os, sys, json
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
_OUT  = os.path.join(_ROOT, "probes_out", "wpc_sound_horizon.json")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

C = 299792.458  # km/s

# ------------------------------------------------------------------ inputs
# Free-history global fit (SN+BAO+CMB, no calibrators): probes_out/phaseF_freshH0.json
# global_reference AND modelV_probeR.json "V"/"R1"
FV0        = 0.6401279536221243     # global reference f_v0
HBAR0      = 53.794282522327265     # global reference bare Hubble (km/s/Mpc)
ALPHA_V    = 37.88797860347935      # BAO+CMB ruler amplitude from the joint fit V (probeR)
G_DRESS    = 1.1891815949755753     # H0_dressed/Hbar0 (Wiltshire tracker scale)
HD0_RATE   = 1.257531426937754      # full present dressed rate factor Hd(0)
RD_EXTERNAL = 147.09                # the external ruler being replaced

# radiation / BBN constants (DNW13 conventions)
GSTAR      = 3.36                   # DNW13 relativistic dof (g_* = 2 rho_R/rho_gamma)
T0_CMB     = 2.725                  # K (DNW13)
OMEGA_GAMMA_STD = 2.4728e-5         # Omega_gamma h^2 (photons, T0=2.7255); std tabulated
ETA10      = 5.1                    # 10^10 eta_Bgamma (DNW13 central BBN value)
ETA_TO_OMB = 273.9                  # omega_b_std = eta10/273.9  (reproduces DNW13 Omega_B0_bar)
OMEGA_B_PLANCK = 0.02237            # alternative anchor (stated, not used for headline)

# adopted STANDARD recombination redshifts (atomic physics + eta; Saha-Peebles ODE
# out of scope -- see docstring).  z_star = repo ZSTAR (Planck); z_drag = Planck 2018.
Z_STAR = 1089.80
Z_DRAG = 1059.94


# ------------------------------------------------------------- bare parameters
def bare_params(fv0, Hbar0):
    """Bare density parameters (physical, in units of rho_crit,100) from (fv0, Hbar0)."""
    gam0   = (2.0 + fv0) / 2.0                       # tracker lapse gamma_bar0
    OmM0   = 4.0 * (1.0 - fv0) / (2.0 + fv0) ** 2     # bare matter density parameter
    Omk0   = 9.0 * fv0 / (2.0 + fv0) ** 2             # bare curvature (negligible early)
    hbar2  = (Hbar0 / 100.0) ** 2
    # physical densities (rho/rho_crit,100), so H_bar(x)=100 sqrt(om x^-3 + or x^-4)
    om_m   = OmM0 * hbar2
    or_std = OMEGA_GAMMA_STD * GSTAR / 2.0            # total radiation, no dressing
    og_std = OMEGA_GAMMA_STD                          # photons, no dressing
    om_r   = or_std / gam0 ** 4                       # bare radiation: gamma_bar0^-4 (T_bar=T0/gamma0)
    om_g   = og_std / gam0 ** 4                       # bare photons
    ob_std = ETA10 / ETA_TO_OMB                       # omega_b from BBN eta (no dressing)
    om_b   = ob_std / gam0 ** 3                       # bare baryon: gamma_bar0^-3
    OmB0_bare = om_b / hbar2                           # bare baryon density PARAMETER (for DNW13 check)
    return dict(gam0=gam0, OmM0=OmM0, Omk0=Omk0, hbar2=hbar2,
                om_m=om_m, om_r=om_r, om_g=om_g, om_b=om_b,
                ob_over_og=om_b / om_g, OmB0_bare=OmB0_bare,
                z_eq_bare=om_m / om_r, ob_std=ob_std)


def sound_horizon(om_m, om_r, ob_over_og, z_dec, gam0):
    """DNW13 Eq 31 comoving sound horizon (Mpc) to observed-frame z_dec.

    r_s = (c/sqrt3) INT_0^{x_dec} dx /[x^2 H_bar sqrt(1+R)],  x=a_bar/a_bar0
        = (c/sqrt3) INT_0^{x_dec} dx /[100 sqrt(om_m x + om_r) sqrt(1+R)]
    R   = 0.75 x (om_b/om_gamma).
    Bare integration limit x_dec = 1/[gamma_bar0 (1+z_dec)]  (gamma_bar_dec~=1).
    """
    x_dec = 1.0 / (gam0 * (1.0 + z_dec))
    integrand = lambda x: 1.0 / (100.0 * np.sqrt(om_m * x + om_r)
                                 * np.sqrt(1.0 + 0.75 * x * ob_over_og))
    val, _ = quad(integrand, 0.0, x_dec, limit=400)
    return (C / np.sqrt(3.0)) * val


# ----------------------------------------------------------- LCDM validation gate
def _lcdm_validation():
    """Same integral with gamma_bar0=1 and standard LCDM densities must reproduce
    the known Planck r_star~144.4 / r_drag~147.1 Mpc -- proves the quadrature."""
    om_m = 0.1430
    Neff = 3.046
    om_r = OMEGA_GAMMA_STD * (1.0 + 0.2271 * Neff)
    ob_over_og = OMEGA_B_PLANCK / OMEGA_GAMMA_STD
    r_star = sound_horizon(om_m, om_r, ob_over_og, Z_STAR, 1.0)
    r_drag = sound_horizon(om_m, om_r, ob_over_og, Z_DRAG, 1.0)
    ok = (143.0 < r_star < 146.0) and (145.5 < r_drag < 148.5)
    return dict(passed=bool(ok), r_star=r_star, r_drag=r_drag,
                expected="r_star~144.4, r_drag~147.1 (Planck 2018)")


def main():
    val = _lcdm_validation()

    bp = bare_params(FV0, HBAR0)
    r_dec = sound_horizon(bp["om_m"], bp["om_r"], bp["ob_over_og"], Z_STAR, bp["gam0"])
    r_d   = sound_horizon(bp["om_m"], bp["om_r"], bp["ob_over_og"], Z_DRAG, bp["gam0"])

    # ---- DNW13 self-check: at their canonical params, reproduce Omega_B0_bar=0.0303
    dnw = bare_params(0.698, 50.1)
    dnw_rd = sound_horizon(dnw["om_m"], dnw["om_r"], dnw["ob_over_og"], 1100.0, dnw["gam0"])

    # ---- compression observables (task spec) ----
    import timescape_baocmb as T
    DM_dim = float(T.DM(Z_STAR, FV0))          # dimensionless comoving distance (units c/Hbar0)
    DM_star_Mpc = (C / HBAR0) * DM_dim         # Mpc
    l_A = np.pi * DM_star_Mpc / r_dec
    shift_R = np.sqrt(bp["OmM0"]) * DM_dim     # sqrt(OmM0 Hbar0^2) D_M/c = sqrt(OmM0) * DM_dim
    omega_b = bp["ob_std"]                       # compression omega_b: BBN eta->omega_b (std/observed frame)
    omega_b_bare = bp["om_b"]                    # bare-frame baryon density (enters Friedmann loading)
    # external-ruler references for the same distance (what the harness fit actually matches)
    l_A_ext = np.pi * DM_star_Mpc / val["r_drag"]   # if the LCDM-like r_star were used
    shift_R_dressed = np.sqrt(bp["OmM0"] * bp["gam0"]**3) * DM_dim  # dressed OmM0 alt (flagged)

    # ---- external comparison ----
    ratio = r_d / RD_EXTERNAL
    delta_pct = 100.0 * (ratio - 1.0)

    # ---- self-consistency ----
    # chi2 is r_d-independent (alpha marginalised) => R1 verdict chi2 UNCHANGED.
    # Absolute calibration: Hbar0 = c/(alpha_V r_d).  Invariant c/alpha_V = Hbar0*r_d.
    inv = C / ALPHA_V                                  # = Hbar0_global * RD_EXTERNAL
    Hbar0_implied = C / (ALPHA_V * r_d)                # holding shape+alpha, swap ruler
    # fixed point: Hbar0* with r_d(Hbar0*) self-consistent:  Hbar0* r_d(Hbar0*) = inv
    def _resid(Hb):
        b = bare_params(FV0, Hb)
        return Hb * sound_horizon(b["om_m"], b["om_r"], b["ob_over_og"], Z_DRAG, b["gam0"]) - inv
    try:
        Hbar0_fixed = brentq(_resid, 20.0, 90.0, xtol=1e-4)
        bfx = bare_params(FV0, Hbar0_fixed)
        r_d_fixed = sound_horizon(bfx["om_m"], bfx["om_r"], bfx["ob_over_og"], Z_DRAG, bfx["gam0"])
    except Exception as e:
        Hbar0_fixed, r_d_fixed = None, None

    H0_dressed_global = G_DRESS * HBAR0
    H0_dressed_implied = G_DRESS * Hbar0_implied
    H0_fullrate_implied = HD0_RATE * Hbar0_implied

    # R1 survives?  Reconciliation needs Hbar0 ~ 53.8 (dressed H0 ~ 64-68). Threshold:
    # accept if in-model r_d within +/-3% of 147.09 (=> Hbar0 within +/-3%).
    r1_survives = bool(abs(delta_pct) <= 3.0)

    out = {
        "probe": "wpc_sound_horizon",
        "purpose": ("In-model radiation-era bare-frame sound horizon r_d (DNW13 "
                    "arXiv:1306.3208), replacing the external r_d=147.09 Mpc used at "
                    "harness.py:35,54 and timescape_baocmb.py:73-75. Compression-level "
                    "CMB (l_A, shift R, omega_b) + self-consistency of the R1 joint fit."),
        "inputs": {
            "fv0": FV0, "Hbar0": HBAR0, "alpha_V": ALPHA_V,
            "g_dress": G_DRESS, "Hd0_rate": HD0_RATE,
            "eta_Bgamma": ETA10 * 1e-10, "gstar": GSTAR, "T0_CMB_K": T0_CMB,
            "z_star_adopted": Z_STAR, "z_drag_adopted": Z_DRAG,
            "omega_b_anchor": "BBN eta=5.1e-10 -> omega_b_std=eta10/273.9 "
                              "(NOT the Planck omega_b=0.0224; stated per task)",
        },
        "bare_parameters": {
            "gamma_bar0": bp["gam0"],
            "Omega_M0_bar": bp["OmM0"], "Omega_k0_bar": bp["Omk0"],
            "omega_m_bar_phys": bp["om_m"], "omega_r_bar_phys": bp["om_r"],
            "omega_gamma_bar_phys": bp["om_g"], "omega_b_bar_phys": bp["om_b"],
            "Omega_B0_bar": bp["OmB0_bare"],
            "z_eq_bare": bp["z_eq_bare"],
            "note": ("bare matter density omega_m_bar=0.060 is ~0.42x the LCDM 0.143 "
                     "(low bare matter density is the dominant driver of a large r_d); "
                     "radiation carries a gamma_bar0^-4 suppression (bare frame cooler, "
                     "T_bar=T0/gamma_bar0=%.3f K) which partly compensates via earlier "
                     "matter-radiation equality." % (T0_CMB / bp["gam0"])),
        },
        "r_d_in_model": r_d,
        "r_dec": r_dec,
        "z_star": Z_STAR,
        "z_drag": Z_DRAG,
        "l_A": l_A,
        "shift_R": shift_R,
        "omega_b": omega_b,
        "compression_detail": {
            "DM_star_dimensionless": DM_dim,
            "DM_star_Mpc": DM_star_Mpc,
            "l_A_in_model_rs": l_A,
            "l_A_if_external_rs": l_A_ext,
            "l_A_Planck": 301.7,
            "shift_R_bare_OmM0": shift_R,
            "shift_R_dressed_OmM0_alt": shift_R_dressed,
            "shift_R_Planck": 1.7448,
            "omega_b_BBN_eta": omega_b,
            "omega_b_bare_frame": omega_b_bare,
            "omega_b_Planck": OMEGA_B_PLANCK,
            "note": ("l_A and R use the BARE Omega_M0 / in-model r_s per the task spec; "
                     "both are convention-dependent in timescape (bare vs dressed) and "
                     "sit well below Planck because r_s is inflated / Omega_M0_bar is low. "
                     "DM_star is the tracker-limit distance; the node history shifts it "
                     "<2%."),
        },
        "external_147p09_comparison": {
            "r_d_external": RD_EXTERNAL,
            "r_d_in_model": r_d,
            "ratio_inmodel_over_external": ratio,
            "delta_percent": delta_pct,
            "r_star_over_r_drag_in_model": r_dec / r_d,
            "r_star_over_r_drag_Planck": 144.43 / 147.09,
            "ratio_note": ("the in-model r_star/r_drag=%.5f matches Planck 0.98192 to 0.03%%; "
                           "the acoustic RATIO (what enters the CMB data point) is preserved, "
                           "only the ABSOLUTE scale shifts -- this is why the marginalised chi2 "
                           "is unmoved." % (r_dec / r_d)),
            "verdict": ("in-model r_d is %.1f%% LARGER than 147.09 -- the external ruler "
                        "is NOT reproduced from the model's own bare densities" % delta_pct),
        },
        "self_consistency": {
            "chi2_is_rd_independent": True,
            "explanation": ("harness.bao_cmb_chi2 marginalises alpha=c/(Hbar0 r_d) "
                            "analytically, so the BAO+CMB chi2 and the R1 verdict "
                            "(RECONCILES, chi2_min_V=1396.06) DO NOT change when r_d is "
                            "swapped; r_d only sets the ABSOLUTE Hbar0=c/(alpha r_d)."),
            "invariant_c_over_alphaV_Mpc_kms": inv,
            "Hbar0_global_with_147": HBAR0,
            "Hbar0_implied_with_inmodel_rd": Hbar0_implied,
            "Hbar0_self_consistent_fixed_point": Hbar0_fixed,
            "r_d_at_fixed_point": r_d_fixed,
            "H0_dressed_global": H0_dressed_global,
            "H0_dressed_implied": H0_dressed_implied,
            "H0_fullrate_implied": H0_fullrate_implied,
            "R1_survives": r1_survives,
            "verdict": (
                "R1 RECONCILES (the chi2 fit) SURVIVES trivially -- it is r_d-independent. "
                "But the ABSOLUTE-H0 reconciliation, the actual point of WP-C/H-C, does NOT: "
                "the in-model r_d=%.0f Mpc drives the implied bare Hbar0 from 53.79 down to "
                "%.1f (dressed H0 %.1f g_dress / %.1f full-rate), and the fully self-consistent "
                "fixed point sits at Hbar0*=%.1f, r_d*=%.0f Mpc. Both are far below the "
                "reconciling band (~54 / dressed ~64-68) and catastrophically below SH0ES 73. "
                "The external r_d=147.09 was doing real work pinning H0; the model's own "
                "early-Universe physics predicts a ~35%% larger sound horizon."
                % (r_d, Hbar0_implied, H0_dressed_implied, H0_fullrate_implied,
                   Hbar0_fixed if Hbar0_fixed else -1, r_d_fixed if r_d_fixed else -1)),
        },
        "lcdm_validation_gate": val,
        "dnw13_selfcheck": {
            "params": "fv0=0.698, Hbar0=50.1 (DNW13 canonical: dressed H0=61.7, OmM0=0.410)",
            "Omega_M0_bar": dnw["OmM0"], "Omega_B0_bar": dnw["OmB0_bare"],
            "Omega_B0_bar_DNW13_quoted": 0.0303,
            "gamma_bar0": dnw["gam0"],
            "r_drag_at_DNW13_params": dnw_rd,
            "note": ("baryon relation reproduces DNW13 Omega_B0_bar=0.0303 at eta=5.1e-10 "
                     "(check: %.4f). At DNW13's own params r_drag=%.0f Mpc, i.e. their "
                     "timescape also has a large physical r_d; DNW13 reconcile the "
                     "DIMENSIONLESS acoustic scale theta*/D_V-r_d ratios (as the harness "
                     "does via alpha), not a physical r_d=147." % (dnw["OmB0_bare"], dnw_rd)),
        },
        "provenance": {
            "source": "arXiv:1306.3208 (Duley, Nazer & Wiltshire 2013), ar5iv HTML",
            "verified_equations": {
                "Eq10_bare_Friedmann_with_radiation": "confirmed",
                "Eq20_21_tracker_OmM0_Omk0": "OmM0_bar=4(1-fv0)/(2+fv0)^2, Omk0_bar=9fv0/(2+fv0)^2",
                "Eq22_region_gamma_bar0": "gamma_bar0=(2+fv0)/2 (tracker); T_bar=gamma_bar^-1 T",
                "Eq31_sound_horizon": "D_bar_s integral -- transcribed verbatim, label (31) verified",
                "Eq49_drag_depth": "c tau_d ~=1, tau_d=INT sigma_T n_bar_e/(a_bar R)dt -- label (49) verified",
                "Omega_R0_bar": "kappa g_* T0^4/(Hbar0^2 gamma_bar0^4), g_*=3.36, T0=2.725K",
            },
            "residual_uncertainties": [
                "z_star/z_drag adopted at Planck values; the full Saha-Peebles ionization "
                "ODE (needed to evaluate c tau_d=1 rigorously) is OUT OF SCOPE (WP-B gate). "
                "Impact on r_d: ~1-2% (r_drag/r_star ratio).",
                "omega_b from BBN eta=5.1e-10 (DNW13), NOT Planck 0.0224; a Planck omega_b "
                "would raise baryon loading and lower r_d by a few %.",
                "DM_star uses the tracker-limit timescape distance, not the 5-node V history "
                "(<2% shift); l_A/R are compression-level.",
                "l_A and shift R are convention-dependent (bare vs dressed Omega_M0); reported "
                "with the bare parameters per the task spec, dressed alternative flagged.",
                "At DNW13's own params my faithful Eq-31 integral gives r_drag~207-213 Mpc, "
                "consistent with the low-bare-matter mechanism; DNW13's Fig-2 'matching' is in "
                "dimensionless acoustic-scale space, so no direct physical-Mpc cross-check exists.",
            ],
        },
    }

    with open(_OUT, "w") as f:
        json.dump(out, f, indent=1)

    # ---- console summary ----
    print("=" * 74)
    print("WP-C  in-model sound horizon (DNW13 bare radiation-era integration)")
    print("=" * 74)
    print("LCDM validation gate: r_star=%.2f r_drag=%.2f Mpc  -> %s"
          % (val["r_star"], val["r_drag"], "PASS" if val["passed"] else "FAIL"))
    print("DNW13 self-check: Omega_B0_bar=%.4f (quoted 0.0303)  r_drag@DNW13=%.0f Mpc"
          % (dnw["OmB0_bare"], dnw_rd))
    print("-" * 74)
    print("bare: gamma0=%.4f OmM0=%.4f omega_m=%.4f omega_r=%.3e omega_b=%.5f z_eq=%.0f"
          % (bp["gam0"], bp["OmM0"], bp["om_m"], bp["om_r"], bp["om_b"], bp["z_eq_bare"]))
    print("r_d_in_model = %.1f Mpc   r_dec = %.1f Mpc   (external 147.09; %+.1f%%)"
          % (r_d, r_dec, delta_pct))
    print("l_A=%.1f (Planck 301.7)  shift_R=%.3f (Planck 1.745)  omega_b=%.5f"
          % (l_A, shift_R, omega_b))
    print("-" * 74)
    print("SELF-CONSISTENCY:")
    print("  chi2 r_d-independent (alpha marginalised) -> R1 fit verdict UNCHANGED")
    print("  Hbar0: global(147)=%.2f  implied(in-model r_d)=%.2f  fixed-point=%.2f"
          % (HBAR0, Hbar0_implied, Hbar0_fixed if Hbar0_fixed else -1))
    print("  dressed H0 implied: %.1f (g_dress) / %.1f (full-rate)  [global %.1f]"
          % (H0_dressed_implied, H0_fullrate_implied, H0_dressed_global))
    print("  R1 absolute-H0 reconciliation SURVIVES=%s" % r1_survives)
    print("wrote", _OUT)
    return out


if __name__ == "__main__":
    main()
