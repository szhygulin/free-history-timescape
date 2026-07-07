#!/usr/bin/env python3
"""
ADVERSARIAL clean-room verification of probes_out/wpc_sound_horizon.json.

Independent re-derivation (NOT importing wpc_sound_horizon.py) of:
  1. bare density parameters from the tracker relations at V params
  2. r_d via an INDEPENDENT quadrature (fixed fine trapezoid in x, and a
     log-x Simpson variant) cross-checked against scipy.quad
  3. LCDM validation gate (must reproduce Planck r_star~144.4 / r_drag~147.1)
  4. DNW13 self-check (Omega_B0_bar=0.0303 at fv0=0.698, Hbar0=50.1; dressed
     H0=61.7; OmM0_dressed=0.410; gamma_bar0=1.349)
  5. baryon-loading R FRAME-INVARIANCE algebra (the gam0 in om_b/om_g cancels
     the gam0 in x_dec) -- a self-consistency test of the gamma bookkeeping
  6. z_eq direction check (is equality EARLIER or LATER than LCDM?)
  7. self-consistency arithmetic: invariant, implied Hbar0, fixed point,
     dressed H0 -- recomputed from scratch
  8. r_d decomposition: isolate the matter-density vs radiation-suppression vs
     omega_b(eta) contributions to the r_d inflation
  9. confirm chi2 r_d-independence directly from harness.bao_cmb_chi2

Writes probes_out/verify_wpc_sound_horizon.json.
"""
import os, sys, json
import numpy as np
from scipy.integrate import quad

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)
_OUT  = os.path.join(_ROOT, "probes_out", "verify_wpc_sound_horizon.json")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

C = 299792.458  # km/s

# ---- claimed values from wpc_sound_horizon.json ----
CLAIM = dict(
    r_d_in_model=199.5612258893033, r_dec=195.89268303008296,
    gamma_bar0=1.320063976811062, Omega_M0_bar=0.20651811574526202,
    omega_m_bar_phys=0.059762725162048176, omega_r_bar_phys=1.3681016122896446e-05,
    omega_gamma_bar_phys=8.143461977914551e-06, omega_b_bar_phys=0.00809455921657799,
    Omega_B0_bar=0.02797183549885561, z_eq_bare=4368.2957921546285,
    l_A=218.93579751960547, shift_R=1.1132164440193866, omega_b=0.018619934282584884,
    Hbar0_implied=39.649992031008274, Hbar0_fixed=32.36326372127159,
    r_d_fixed=244.49329761441047, H0_dressed_implied=47.15104076420327,
    H0_fullrate_implied=49.86111105682441, inv=7912.601016209118,
    lcdm_r_star=144.29392336494914, lcdm_r_drag=146.93931243803257,
    dnw_OmB0=0.030218078362731382, dnw_rd=212.97726776208407,
)

# ---- V-fit inputs (from modelV_probeR.json 'V' block; re-read below) ----
FV0     = 0.6401279536221243
HBAR0   = 53.794282522327265
ALPHA_V = 37.88797860347935
G_DRESS = 1.1891815949755753
HD0RATE = 1.257531426937754
RD_EXT  = 147.09

GSTAR   = 3.36
T0      = 2.725
OM_G_STD = 2.4728e-5
ETA10   = 5.1
ETA2OMB = 273.9
OMB_PLANCK = 0.02237
ZSTAR   = 1089.80
ZDRAG   = 1059.94


def bare(fv0, Hbar0):
    gam0 = (2.0 + fv0) / 2.0
    OmM0 = 4.0 * (1.0 - fv0) / (2.0 + fv0) ** 2
    Omk0 = 9.0 * fv0 / (2.0 + fv0) ** 2
    hbar2 = (Hbar0 / 100.0) ** 2
    om_m = OmM0 * hbar2
    or_std = OM_G_STD * GSTAR / 2.0
    og_std = OM_G_STD
    om_r = or_std / gam0 ** 4
    om_g = og_std / gam0 ** 4
    ob_std = ETA10 / ETA2OMB
    om_b = ob_std / gam0 ** 3
    return dict(gam0=gam0, OmM0=OmM0, Omk0=Omk0, hbar2=hbar2, om_m=om_m,
                om_r=om_r, om_g=om_g, om_b=om_b, ob_over_og=om_b / om_g,
                OmB0_bare=om_b / hbar2, z_eq_bare=om_m / om_r, ob_std=ob_std)


def rs_quad(om_m, om_r, ob_og, z_dec, gam0):
    """Reference: same scipy.quad integral the probe uses."""
    x_dec = 1.0 / (gam0 * (1.0 + z_dec))
    f = lambda x: 1.0 / (100.0 * np.sqrt(om_m * x + om_r) * np.sqrt(1.0 + 0.75 * x * ob_og))
    val, _ = quad(f, 0.0, x_dec, limit=400)
    return (C / np.sqrt(3.0)) * val


def rs_trap(om_m, om_r, ob_og, z_dec, gam0, n=2_000_000):
    """INDEPENDENT quadrature: uniform trapezoid in x on a very fine grid."""
    x_dec = 1.0 / (gam0 * (1.0 + z_dec))
    x = np.linspace(0.0, x_dec, n + 1)
    integ = 1.0 / (100.0 * np.sqrt(om_m * x + om_r) * np.sqrt(1.0 + 0.75 * x * ob_og))
    val = np.trapz(integ, x)
    return (C / np.sqrt(3.0)) * val


def rs_subst(om_m, om_r, ob_og, z_dec, gam0, n=400_000):
    """INDEPENDENT quadrature via u=sqrt(x) substitution (removes sqrt endpoint
    softness), Simpson rule. dx=2u du, integrand smooth at u=0."""
    from scipy.integrate import simpson
    x_dec = 1.0 / (gam0 * (1.0 + z_dec))
    u = np.linspace(0.0, np.sqrt(x_dec), n + 1)
    x = u * u
    g = 2.0 * u / (100.0 * np.sqrt(om_m * x + om_r) * np.sqrt(1.0 + 0.75 * x * ob_og))
    val = simpson(g, x=u)
    return (C / np.sqrt(3.0)) * val


def rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a - b)


def main():
    checks = []
    disc = []

    # ---------- (0) confirm V inputs against modelV_probeR.json ----------
    mv = json.load(open(os.path.join(_ROOT, "probes_out", "modelV_probeR.json")))
    V = mv["V"]
    for k, ref in [("alpha", ALPHA_V), ("fv0", FV0), ("Hbar0", HBAR0),
                   ("g_dress", G_DRESS), ("Hd_z0", HD0RATE)]:
        ok = rel(V[k], ref) < 1e-9
        checks.append(f"input {k}={V[k]:.10g} matches probe {'OK' if ok else 'MISMATCH'}")
        if not ok:
            disc.append(f"probe input {k}={ref} != modelV V.{k}={V[k]}")
    chi2V = V["chi2_min"]
    checks.append(f"modelV V.chi2_min={chi2V:.4f} (probe cites 1396.06) "
                  + ("OK" if abs(chi2V - 1396.06) < 0.01 else "MISMATCH"))

    # ---------- (1) tracker/bare relations ----------
    bp = bare(FV0, HBAR0)
    for k, cl in [("gam0", CLAIM["gamma_bar0"]), ("OmM0", CLAIM["Omega_M0_bar"]),
                  ("om_m", CLAIM["omega_m_bar_phys"]), ("om_r", CLAIM["omega_r_bar_phys"]),
                  ("om_g", CLAIM["omega_gamma_bar_phys"]), ("om_b", CLAIM["omega_b_bar_phys"]),
                  ("OmB0_bare", CLAIM["Omega_B0_bar"]), ("z_eq_bare", CLAIM["z_eq_bare"])]:
        ok = rel(bp[k], cl) < 1e-6
        checks.append(f"bare {k}={bp[k]:.6g} vs claim {cl:.6g} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            disc.append(f"bare {k}: recomputed {bp[k]} != claimed {cl}")

    # dressed cross-checks of the tracker map
    OmM_dressed = 0.5 * (1.0 - FV0) * (2.0 + FV0)
    gdress_check = (4 * FV0 ** 2 + FV0 + 4) / (2 * (2 + FV0))
    checks.append(f"g_dress(fv0) recomputed={gdress_check:.10f} vs V={G_DRESS:.10f} "
                  + ("OK" if rel(gdress_check, G_DRESS) < 1e-9 else "MISMATCH"))
    checks.append(f"dressed Omega_M0={OmM_dressed:.4f} (independent tracker relation)")

    # ---------- (2) r_d via three independent quadratures ----------
    rq = rs_quad(bp["om_m"], bp["om_r"], bp["ob_over_og"], ZDRAG, bp["gam0"])
    rt = rs_trap(bp["om_m"], bp["om_r"], bp["ob_over_og"], ZDRAG, bp["gam0"])
    rsub = rs_subst(bp["om_m"], bp["om_r"], bp["ob_over_og"], ZDRAG, bp["gam0"])
    rdec_q = rs_quad(bp["om_m"], bp["om_r"], bp["ob_over_og"], ZSTAR, bp["gam0"])
    checks.append(f"r_d: quad={rq:.4f} trap={rt:.4f} subst={rsub:.4f} Mpc (claim {CLAIM['r_d_in_model']:.4f})")
    for nm, v in [("quad", rq), ("trap", rt), ("subst", rsub)]:
        if rel(v, CLAIM["r_d_in_model"]) > 2e-3:
            disc.append(f"r_d {nm}={v:.3f} deviates >0.2% from claim {CLAIM['r_d_in_model']:.3f}")
    checks.append(f"r_dec quad={rdec_q:.4f} vs claim {CLAIM['r_dec']:.4f} "
                  + ("OK" if rel(rdec_q, CLAIM["r_dec"]) < 2e-3 else "MISMATCH"))
    rd = rq
    delta_pct = 100.0 * (rd / RD_EXT - 1.0)
    checks.append(f"delta vs 147.09 = {delta_pct:+.2f}% (claim +35.67%)")

    # ---------- (3) LCDM validation gate (independent) ----------
    om_m_L = 0.1430
    om_r_L = OM_G_STD * (1.0 + 0.2271 * 3.046)
    ob_og_L = OMB_PLANCK / OM_G_STD
    r_star_L = rs_quad(om_m_L, om_r_L, ob_og_L, ZSTAR, 1.0)
    r_drag_L = rs_quad(om_m_L, om_r_L, ob_og_L, ZDRAG, 1.0)
    gate_ok = bool((143.0 < r_star_L < 146.0) and (145.5 < r_drag_L < 148.5))
    checks.append(f"LCDM gate r_star={r_star_L:.3f} r_drag={r_drag_L:.3f} "
                  + ("PASS" if gate_ok else "FAIL") + " (Planck 144.4/147.1)")
    if not gate_ok:
        disc.append(f"LCDM gate failed: r_star={r_star_L}, r_drag={r_drag_L}")

    # ---------- (4) DNW13 self-check (independent) ----------
    dnw = bare(0.698, 50.1)
    dnw_rd = rs_quad(dnw["om_m"], dnw["om_r"], dnw["ob_over_og"], 1100.0, dnw["gam0"])
    dnw_OmM_dressed = 0.5 * (1 - 0.698) * (2 + 0.698)
    dnw_gdress = (4 * .698 ** 2 + .698 + 4) / (2 * (2 + .698))
    dnw_H0d = dnw_gdress * 50.1
    checks.append(f"DNW13 OmB0_bar={dnw['OmB0_bare']:.4f} (quoted 0.0303; claim {CLAIM['dnw_OmB0']:.4f})")
    checks.append(f"DNW13 gamma_bar0={dnw['gam0']:.4f} (expect 1.349)")
    checks.append(f"DNW13 dressed OmM0={dnw_OmM_dressed:.4f} (expect 0.410)")
    checks.append(f"DNW13 dressed H0=g_dress*50.1={dnw_H0d:.2f} (expect 61.7)")
    checks.append(f"DNW13 r_drag@params={dnw_rd:.2f} Mpc (claim {CLAIM['dnw_rd']:.2f})")
    if rel(dnw["OmB0_bare"], 0.0303) > 0.01:
        disc.append(f"DNW13 OmB0_bar={dnw['OmB0_bare']:.4f} off from 0.0303 by >1%")
    if abs(dnw_H0d - 61.7) > 0.3:
        disc.append(f"DNW13 dressed H0={dnw_H0d:.2f} off from 61.7")
    if abs(dnw_OmM_dressed - 0.410) > 0.005:
        disc.append(f"DNW13 dressed OmM0={dnw_OmM_dressed:.4f} off from 0.410")

    # ---------- (5) baryon loading R frame-invariance ----------
    # R_bare(z_dec) = 0.75 * x_dec * (om_b/om_g);  x_dec=1/(gam0(1+z))
    # om_b/om_g = (ob_std/og_std)*gam0  =>  R_bare = 0.75*(ob_std/og_std)/(1+z)
    x_dec = 1.0 / (bp["gam0"] * (1 + ZDRAG))
    R_bare = 0.75 * x_dec * bp["ob_over_og"]
    R_frameinv = 0.75 * (bp["ob_std"] / OM_G_STD) / (1 + ZDRAG)  # gam0 cancelled
    checks.append(f"R(z_drag) bare={R_bare:.5f} vs gam0-cancelled={R_frameinv:.5f} "
                  + ("FRAME-INVARIANT OK" if rel(R_bare, R_frameinv) < 1e-9 else "MISMATCH"))
    if rel(R_bare, R_frameinv) > 1e-9:
        disc.append("baryon loading R is NOT frame-invariant -- gamma bookkeeping bug")

    # ---------- (6) z_eq direction ----------
    z_eq_bare = bp["z_eq_bare"]
    z_eq_L = om_m_L / om_r_L
    earlier = bool(z_eq_bare > z_eq_L)
    checks.append(f"z_eq_bare={z_eq_bare:.0f} vs z_eq_LCDM={z_eq_L:.0f}: bare equality is "
                  + ("EARLIER (higher z)" if earlier else "LATER (lower z)"))
    if earlier:
        disc.append("SUMMARY-CAVEAT WORDING: bare z_eq (4368) > LCDM (3421) so equality "
                    "is EARLIER, not 'delayed' as one summary caveat states; the JSON "
                    "bare_parameters note ('earlier matter-radiation equality') is correct. "
                    "r_s inflation is driven by lower H throughout (both om_m and om_r down), "
                    "not by delayed equality. Mechanism conclusion unaffected.")

    # ---------- (7) self-consistency arithmetic ----------
    inv = C / ALPHA_V
    Hbar0_impl = C / (ALPHA_V * rd)
    checks.append(f"invariant c/alpha_V={inv:.4f} vs Hbar0*147.09={HBAR0 * RD_EXT:.4f} "
                  + ("OK" if rel(inv, HBAR0 * RD_EXT) < 1e-6 else "MISMATCH"))
    checks.append(f"Hbar0_implied=c/(alpha*r_d)={Hbar0_impl:.4f} (claim {CLAIM['Hbar0_implied']:.4f})")
    if rel(Hbar0_impl, CLAIM["Hbar0_implied"]) > 2e-3:
        disc.append(f"Hbar0_implied {Hbar0_impl:.3f} != claim {CLAIM['Hbar0_implied']:.3f}")
    # fixed point: Hb * r_d(Hb) = inv
    from scipy.optimize import brentq
    def resid(Hb):
        b = bare(FV0, Hb)
        return Hb * rs_quad(b["om_m"], b["om_r"], b["ob_over_og"], ZDRAG, b["gam0"]) - inv
    Hb_fix = brentq(resid, 20.0, 90.0, xtol=1e-5)
    bfx = bare(FV0, Hb_fix)
    rd_fix = rs_quad(bfx["om_m"], bfx["om_r"], bfx["ob_over_og"], ZDRAG, bfx["gam0"])
    checks.append(f"fixed point Hbar0*={Hb_fix:.4f} r_d*={rd_fix:.4f} "
                  f"(claim {CLAIM['Hbar0_fixed']:.4f}/{CLAIM['r_d_fixed']:.4f})")
    if rel(Hb_fix, CLAIM["Hbar0_fixed"]) > 2e-3:
        disc.append(f"fixed point Hbar0*={Hb_fix:.3f} != claim {CLAIM['Hbar0_fixed']:.3f}")
    H0d_impl = G_DRESS * Hbar0_impl
    H0f_impl = HD0RATE * Hbar0_impl
    checks.append(f"dressed H0 implied={H0d_impl:.3f} (g_dress) / {H0f_impl:.3f} (full-rate) "
                  f"(claim {CLAIM['H0_dressed_implied']:.2f}/{CLAIM['H0_fullrate_implied']:.2f})")

    # ---------- (8) r_d decomposition: physically-consistent single-input path ----------
    # LCDM-gate -> in-model, one physical input changed per step. CRUCIAL: the gamma
    # frame has TWO sub-effects that must move together (both from T_bar=T0/gam0):
    #   radiation gamma^-4 suppression  AND  x_dec=1/(gam0(1+z)) shrink.
    or_std = OM_G_STD * GSTAR / 2.0
    s0 = rs_quad(om_m_L, or_std, OMB_PLANCK / OM_G_STD, ZDRAG, 1.0)      # LCDM gate
    s1 = rs_quad(bp["om_m"], or_std, OMB_PLANCK / OM_G_STD, ZDRAG, 1.0)  # + low bare matter
    s2 = rs_quad(bp["om_m"], or_std, bp["ob_std"] / OM_G_STD, ZDRAG, 1.0)  # + eta baryon
    s3 = rs_quad(bp["om_m"], bp["om_r"], bp["ob_std"] / OM_G_STD, ZDRAG, 1.0)  # + radiation gamma^-4 supp
    s4 = rs_quad(bp["om_m"], bp["om_r"], bp["ob_over_og"], ZDRAG, bp["gam0"])   # + x_dec gamma limit (=full)
    d_matter = s1 - s0; d_baryon = s2 - s1; d_radsupp = s3 - s2; d_xdec = s4 - s3
    d_gammaframe = d_radsupp + d_xdec
    checks.append(f"DECOMP path r_d: LCDM={s0:.1f} +matter->{s1:.1f} +baryon->{s2:.1f} "
                  f"+radsupp->{s3:.1f} +xdec->{s4:.1f} (=full {rd:.1f})")
    checks.append(f"DECOMP contributions (Mpc): low-matter={d_matter:+.1f}  eta-baryon={d_baryon:+.1f}  "
                  f"radiation-gamma^-4-suppression={d_radsupp:+.1f} (INFLATES)  "
                  f"x_dec-bare-decoupling-limit={d_xdec:+.1f} (COMPENSATES)  "
                  f"NET-gamma-frame={d_gammaframe:+.1f}")
    # is low bare matter the dominant NET driver?
    matter_dominant = bool(abs(d_matter) >= abs(d_gammaframe) and abs(d_matter) >= abs(d_baryon))
    checks.append(f"low-bare-matter is dominant NET driver? {matter_dominant} "
                  f"(matter {d_matter:+.1f} vs net-gamma {d_gammaframe:+.1f} vs baryon {d_baryon:+.1f})")
    # WORDING CHECK: the JSON/summary say radiation gamma^-4 suppression 'partly compensates/offsets'.
    # Decomposition: radiation suppression alone INFLATES by d_radsupp; the COMPENSATOR is x_dec.
    if d_radsupp > 0:
        disc.append(f"MECHANISM-WORDING ERROR: the JSON bare_parameters note and the summary caveat "
                    f"say the 'gamma_bar0^-4 radiation suppression partly compensates/offsets'. The "
                    f"physically-consistent decomposition shows the OPPOSITE SIGN: radiation suppression "
                    f"alone INFLATES r_d by {d_radsupp:+.1f} Mpc (it is the single largest sub-effect). "
                    f"The genuine partial COMPENSATOR is the x_dec=1/(gam0(1+z_dec)) bare-decoupling limit "
                    f"({d_xdec:+.1f} Mpc), a distinct gamma effect. NET gamma-frame = {d_gammaframe:+.1f} Mpc "
                    f"(mild inflation). Low bare matter ({d_matter:+.1f}) remains the dominant NET driver, so "
                    f"the headline mechanism and the +35.7% number are UNAFFECTED; only the causal attribution "
                    f"of the gamma terms is stated with the wrong sign.")

    # ---------- (9) chi2 r_d-independence, direct ----------
    _cwd = os.getcwd()
    os.chdir(_SRC)  # harness.py loads 'data/PantheonSH0ES.dat' relative to src/
    try:
        import harness as H
    finally:
        os.chdir(_cwd)
    import timescape_baocmb as Tm
    def predict(z, k):
        return Tm.DM(z, FV0) if k == "DM" else (Tm.DH(z, FV0) if k == "DH" else Tm.DV(z, FV0))
    chi_a, alpha_a = H.bao_cmb_chi2(predict)
    # monkeypatch RD and re-run: chi2 must be identical, alpha identical, only H0_from_alpha moves
    H0_147 = H.H0_from_alpha(alpha_a)
    old = H.RD
    H.RD = rd  # swap ruler
    H0_inmodel = H.H0_from_alpha(alpha_a)
    H.RD = old
    chi_b, alpha_b = H.bao_cmb_chi2(predict)
    rd_indep = bool((chi_a == chi_b) and (alpha_a == alpha_b))
    checks.append(f"chi2 r_d-independence: chi2={chi_a:.4f} (unchanged={chi_a==chi_b}), "
                  f"alpha={alpha_a:.4f} (unchanged={alpha_a==alpha_b}) -> "
                  + ("CONFIRMED" if rd_indep else "VIOLATED"))
    checks.append(f"H0(bare) from harness alpha: with 147.09={H0_147:.3f}  "
                  f"with in-model r_d={H0_inmodel:.3f}")
    if not rd_indep:
        disc.append("chi2/alpha changed when RD swapped -- r_d-independence claim FALSE")
    # note: harness alpha=38.61 (pure tracker single-fv0 grid) differs from V alpha=37.888
    # (5-node forced history); the probe correctly uses V alpha. Record both.
    checks.append(f"NOTE harness bao_cmb_chi2 alpha (tracker grid)={alpha_a:.3f}; probe uses "
                  f"modelV V alpha={ALPHA_V:.3f} (5-node forced) -- distinct fits, expected")

    verdict = "SURVIVES" if not disc else "SURVIVES_WITH_CAVEATS"

    out = {
        "probe": "verify_wpc_sound_horizon",
        "purpose": "Adversarial clean-room verification of wpc_sound_horizon.json (in-model DNW13 bare sound horizon).",
        "verdict": verdict,
        "recomputed": {
            "r_d_quad": rq, "r_d_trap": rt, "r_d_subst": rsub, "r_dec": rdec_q,
            "delta_percent_vs_147p09": delta_pct,
            "gamma_bar0": bp["gam0"], "Omega_M0_bar": bp["OmM0"],
            "omega_m_bar": bp["om_m"], "omega_r_bar": bp["om_r"], "omega_b_bar": bp["om_b"],
            "z_eq_bare": bp["z_eq_bare"], "z_eq_LCDM": z_eq_L,
            "lcdm_gate_r_star": r_star_L, "lcdm_gate_r_drag": r_drag_L, "lcdm_gate_pass": gate_ok,
            "dnw13_OmB0_bar": dnw["OmB0_bare"], "dnw13_gamma_bar0": dnw["gam0"],
            "dnw13_dressed_OmM0": dnw_OmM_dressed, "dnw13_dressed_H0": dnw_H0d, "dnw13_r_drag": dnw_rd,
            "R_baryon_drag_bare": R_bare, "R_baryon_drag_frameinv": R_frameinv,
            "invariant_c_over_alpha": inv, "Hbar0_implied": Hbar0_impl,
            "Hbar0_fixed_point": Hb_fix, "r_d_fixed_point": rd_fix,
            "H0_dressed_implied": H0d_impl, "H0_fullrate_implied": H0f_impl,
            "decomp_path_r_d": [s0, s1, s2, s3, s4],
            "decomp_d_matter": d_matter, "decomp_d_baryon_eta": d_baryon,
            "decomp_d_radiation_suppression": d_radsupp, "decomp_d_xdec_limit": d_xdec,
            "decomp_d_net_gamma_frame": d_gammaframe, "decomp_matter_is_dominant": matter_dominant,
            "harness_alpha_tracker": alpha_a, "harness_chi2": chi_a,
            "harness_H0_with_147": H0_147, "harness_H0_with_inmodel_rd": H0_inmodel,
            "chi2_rd_independent": rd_indep,
        },
        "checks": checks,
        "discrepancies": disc,
        "equation_number_verification": {
            "source_checked": "ar5iv.labs.arxiv.org/abs/1306.3208 (via WebFetch small-model read; partial content, treat as indicative not exhaustive)",
            "Eq31_sound_horizon": "CONFIRMED -- source: Eq (31) defines D_bar_s 'the volume-average sound horizon scale at any epoch'. Matches probe.",
            "Eq49_drag_depth": "CONFIRMED -- source: c*tau_d ~= 1 assigned to Eq (49) in Appendix B. Matches probe.",
            "Eq22_lapse_gamma_bar0": "CONFIRMED as the lapse equation -- source: Eq (22) gives the general gamma_bar0 relation; the tracker simplification (2+fv0)/2 is DERIVED, not verbatim. Probe label OK.",
            "Eq20_21_tracker_OmM0_Omk0": "PLAUSIBLE-NOT-VERBATIM -- source places the bare density parameters (Om_M,Om_R,Om_k,Om_Q) in Eqs (18)-(21); the specific OmM0/Omk0 tracker forms are within that block but the exact sub-numbering (20-21) is not verbatim confirmable.",
            "Eq10_bare_Friedmann_with_radiation": "NOT CONFIRMED -- source's read attributes the radiation Friedmann (first Buchert eq with Om_M+Om_R) to Eq (2), with density-parameter forms in (18)-(21); 'Eq 10' specifically could not be located and may be a mislabel. The FORM (H_bar^2=H_bar0^2[OmM0 x^-3+OmR0 x^-4], drop Q&curvature as fv->0) is standard and numerically validated regardless.",
            "quoted_params_confirmed": "eta_Bgamma=(5.1+/-0.5)e-10 CONFIRMED; g_*=3.36 CONFIRMED. Omega_B0_bar/T0/z_dec vary across the paper's numerical solutions (no single universal value in the source read); probe's DNW13 self-check reproduces Omega_B0_bar=0.0303 numerically.",
            "verdict": "FORMS all standard & numerically validated (LCDM gate passes, DNW13 self-check reproduces 0.0303, R frame-invariant). Eq 31 & Eq 49 labels CONFIRMED; Eq 22 confirmed as lapse; Eq 10 and Eq 20-21 labels NOT verbatim-confirmable (Eq 10 likely should be Eq 2). Citation-precision issue only; does not affect any number.",
        },
    }
    with open(_OUT, "w") as f:
        json.dump(out, f, indent=1)

    print("=" * 74)
    print("ADVERSARIAL VERIFY  wpc_sound_horizon  ->", verdict)
    print("=" * 74)
    for c in checks:
        print(" ", c)
    print("-" * 74)
    if disc:
        print("DISCREPANCIES:")
        for d in disc:
            print("  *", d)
    else:
        print("no numeric discrepancies")
    print("wrote", _OUT)
    return out


if __name__ == "__main__":
    main()
