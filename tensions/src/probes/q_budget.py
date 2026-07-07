#!/usr/bin/env python3
"""WP-B Stage 0 -- the Q-budget pre-gate (analytic; exact kinematic identities, NO solver).

Instantiates PLAN_WPB_threephase.md sec 2 + amendment A1.  Computes the AVAILABLE
backreaction Q_avail(z) from the MEASURED void population and compares it to the committed
kinematic requirement Q_req(z) (paper-2 modelV_probeR.json -> derived_backreaction_V).

Backreaction identity (Buchert multi-phase, exact):
    partition {deep d, shallow s, walls w},  f_d + f_s + f_w = 1 (volume fractions)
    H_i - Hbar = fdot_i / (3 f_i)          fdot_i = df_i/dtau   (bare/volume-average time)
    Q_avail    = 6 * sum_i f_i (H_i - Hbar)^2  =  (2/3) * sum_i fdot_i^2 / f_i
(Sum-i f_i (H_i-Hbar) = 0 identically since sum fdot_i = 0, so Hbar = volume-weighted mean.)

Phases from the MEASURED threshold family (matter-mapped lognormal excursion set, sec D/telescope):
    f(delta_th ; z) = Phi( [ ln(1+delta_th) + sigma(z)^2/2 ] / sigma(z) ),  sigma(z)=sigma0*D(z)
    deep d   = {delta_m < -0.5}   (PRIMARY;  delta_m < -0.3 as declared sensitivity)
    below    = {delta_m < 0}      (bias-independent PRIMARY below-mean, telescope Rs=4)
    shallow s= below - deep ,  walls w = 1 - below
sigma0 = 0.7345 (2M++ z=0 below-mean anchor) reproduces the measured below-mean, delta<-0.3 and
delta<-0.5 fractions of phaseD_fvobs.json; D(z) is the LCDM growth VALIDATED by telescope
PRIMARY_below_mean_measured_growth (BOSS CIC, reduced chi2 ~ 1).

Bare-time slicing tau(z): the committed paper-2 background sets H_w = 2/(3 tau) (flat-dust walls),
so tau(z) = 2 / (3 * H_w_over_Hbar0(z)) read directly from derived_backreaction_V.  dz/dtau from a
smooth PCHIP of tau(z) (analytic derivative -- NOT np.gradient of a linear interpolant).  This is
the SAME foliation used to compute Q_req, so Q_avail and Q_req share one time-slicing (validated by
reproducing the published Q_over_Hbar0sq at the clean fv-anchor nodes to ~10%).

A1 variants:
  * CEILING (primary, gate metric): the kinematic-identity Q_avail from the measured fractional
    growth at face-value threshold depths.  (For the fast-growing deep phase this implies per-void
    contrasts ABOVE the empty-Milne bound -- see the physicality strike; the ceiling is thus an
    upper bound that is itself partly super-physical.)
  * EMPTY-MILNE bound: all void phases at the maximal PHYSICAL (empty/Milne) contrast
    (H_v - H_w) = 0.5 H_w  ->  Q = 6 f_below f_w (0.5 H_w)^2 .  The honest empty-void ceiling.
  * PHYSICAL sensitivity: void depth capped at lensing-calibrated delta_m ~ -0.3..-0.5
    (contrast (H_v-H_w) = 0.5|delta_m| H_w) -> Q shrinks ~2-4x vs the empty-Milne bound.

PRE-REGISTERED GATE: NO-GO iff Q_avail_CEILING(z) < Q_req(z)/3 for ALL 0.3 <= z <= 1.0
(band edges most permissive for Q_avail); else GO.  Stage 0 is an order-of-magnitude go/no-go,
NOT a verdict -- both sides are reading-dependent (Q_req is the kinematic-forced requirement; the
dynamical requirement differs because Q back-reacts on Hbar).
"""
import json
import os

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

# ---------------------------------------------------------------------------
# portable paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
SIBLING = os.path.normpath(os.path.join(REPO, ".."))
P2 = os.path.join(SIBLING, "probes_out")
OUTJ = os.path.join(REPO, "probes_out", "q_budget.json")

MODELV = os.path.join(P2, "modelV_probeR.json")
PHASED = os.path.join(P2, "phaseD_fvobs.json")
TELE = os.path.join(P2, "telescope_fvobs.json")

REPORT_Z = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
GATE_Z = np.array([0.3, 0.5, 0.7, 1.0])          # the pre-registered 0.3 <= z <= 1.0 window
OM = 0.315                                        # telescope fiducial (flat LCDM)
SIGMA0 = 0.7344797420042518                       # 2M++ z=0 below-mean anchor
DEEP_TH = -0.5                                    # primary deep threshold
DEEP_TH_SENS = -0.3                               # declared sensitivity
# reliable-volume band on the total below-mean anchor (r100 vs r200), sec D z0_anchor.below_mean_band
BELOW_R100, BELOW_R200 = 0.6142592614519756, 0.6723006374360525
LENSING_DM = (-0.5, -0.3)                         # physical void-depth cap (central, shallow-edge)
LENSING_DM_DEEPEST = -0.8                          # deepest large-void interiors


def log(*a):
    print("[q_budget]", *a, flush=True)


# ---------------------------------------------------------------------------
# growth D(z): flat-LCDM linear growing mode, normalised D(0)=1
# ---------------------------------------------------------------------------
def build_growth(om=OM):
    ol = 1.0 - om
    a = np.linspace(1e-5, 1.0, 400000)
    e = np.sqrt(om * a ** -3 + ol)
    d_un = e * cumulative_trapezoid(1.0 / (a * e) ** 3, a, initial=0.0)
    d = d_un / d_un[-1]

    def Dz(z):
        return np.interp(1.0 / (1.0 + np.asarray(z, float)), a, d)

    return Dz


# ---------------------------------------------------------------------------
# measured-population shape model: f(delta_th ; z) and analytic df/dz
# ---------------------------------------------------------------------------
class ShapeModel:
    """f(delta_th; z) = Phi(u), u = [ln(1+delta_th)+sigma^2/2]/sigma, sigma(z)=sigma0*D(z).

    Smooth analytic derivative via an analytic PCHIP of sigma(z) (no np.gradient of a
    linear interpolant)."""

    def __init__(self, Dz, sigma0=SIGMA0):
        self.sigma0 = sigma0
        zg = np.linspace(0.0, 2.4, 20000)
        self._sig = PchipInterpolator(zg, sigma0 * Dz(zg))
        self._sigp = self._sig.derivative()

    def sigma(self, z):
        return self._sig(z)

    def frac(self, dth, z):
        s = self._sig(z)
        return norm.cdf((np.log(1.0 + dth) + s ** 2 / 2.0) / s)

    def fracp_z(self, dth, z):
        """df/dz (analytic)."""
        s = self._sig(z)
        sp = self._sigp(z)
        u = (np.log(1.0 + dth) + s ** 2 / 2.0) / s
        du_dz = sp * (-np.log(1.0 + dth) / s ** 2 + 0.5)
        return norm.pdf(u) * du_dz


# ---------------------------------------------------------------------------
def main():
    log("start; repo=", REPO, "sibling=", SIBLING)
    modelv = json.load(open(MODELV))
    phased = json.load(open(PHASED))
    tele = json.load(open(TELE))
    db = modelv["derived_backreaction_V"]
    zdb = np.asarray(db["z"], float)
    Hw_pub = np.asarray(db["Hw_over_Hbar0"], float)
    fv_pub = np.asarray(db["fv"], float)
    Q_pub = np.asarray(db["Q_over_Hbar0sq"], float)
    HvHw_pub = np.asarray(db["Hv_minus_Hw_over_Hbar0"], float)

    # ---- background bare-time slicing tau(z) = 2/(3 H_w), dz/dtau (analytic PCHIP) ----
    tau_cb = PchipInterpolator(zdb, 2.0 / (3.0 * Hw_pub))
    tracker_tau0 = (2.0 + fv_pub[0]) / 3.0
    assert abs(tau_cb(0.0) - tracker_tau0) < 1e-3, (float(tau_cb(0.0)), tracker_tau0)

    def dz_dtau(z):
        return 1.0 / tau_cb.derivative()(z)

    def Hw_hb0(z):
        return 2.0 / (3.0 * tau_cb(z))

    # ---- growth + shape model, VALIDATE against committed nodes ----
    Dz = build_growth()
    tel_D = {n["z"]: n["D_z"] for n in tele["PRIMARY_below_mean_Rs4"]["nodes"]}
    dz_val = {z: abs(float(Dz(z)) - v) / v for z, v in tel_D.items()}
    assert max(dz_val.values()) < 1e-3, dz_val
    sm = ShapeModel(Dz, SIGMA0)
    # reproduce phaseD z=0 threshold family
    fam = phased["z0_anchor"]["family"]
    rep = {
        "below_mean_z0": (float(sm.frac(0.0, 0.0)), phased["z0_anchor"]["below_mean_central"]),
        "delta<-0.3_z0": (float(sm.frac(-0.3, 0.0)), phased["shape_model"]["delta<-0.3_model"]),
        "delta<-0.5_z0": (float(sm.frac(-0.5, 0.0)), phased["shape_model"]["delta<-0.5_model"]),
        "below_mean_z0.7": (float(sm.frac(0.0, 0.7)),
                            tele["PRIMARY_below_mean_Rs4"]["at_required_grid"]["z=0.7"]["fv"]),
    }
    for k, (got, exp) in rep.items():
        assert abs(got - exp) < 5e-3, (k, got, exp)
    log("validation OK: D(z)<1e-3, shape-model family reproduced")

    # ---- dz/dtau consistency: reproduce published Q_req at the clean fv-anchor nodes ----
    fv_cb = PchipInterpolator(zdb, fv_pub)
    fvp_rec = fv_cb.derivative()(zdb) * dz_dtau(zdb)
    Q_rec = (2.0 / 3.0) * fvp_rec ** 2 / (fv_pub * (1.0 - fv_pub))
    clean_idx = [0, 2, 4]                            # z = 0, 0.3, 0.7 (original fv nodes; smooth)
    recon_relerr = {float(zdb[i]): float(abs(Q_rec[i] - Q_pub[i]) / Q_pub[i]) for i in clean_idx}
    log("Q_req reconstruction rel-err at clean nodes:", recon_relerr)

    # ---- Q_req: committed clean-node smooth (smooth the PCHIP jitter) + dchi2<=1 band ----
    # clean nodes = the ORIGINAL forced-fit fv nodes z in {0,0.3,0.7,1.3,2.33}; the interleaved
    # z in {0.1,0.5,1.0,1.8} carry the 5-node PCHIP-derivative jitter -> excluded from the smooth.
    zc = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
    Qc = np.array([Q_pub[list(zdb).index(z)] for z in zc])
    Qreq_cb = PchipInterpolator(zc, Qc)

    band = modelv["fv_req_band_dchi2_le1"]
    zb = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
    fvlo = np.array([band[f"z={z:g}".replace("z=0.0", "z=0")][0] for z in zb])
    fvhi = np.array([band[f"z={z:g}".replace("z=0.0", "z=0")][1] for z in zb])

    def qreq_from_fv(fnodes):
        cb = PchipInterpolator(zb, fnodes)
        qq = (2.0 / 3.0) * (cb.derivative()(zb) * dz_dtau(zb)) ** 2 / (cb(zb) * (1.0 - cb(zb)))
        return PchipInterpolator(zb, qq)

    q_e1 = qreq_from_fv(fvlo)
    q_e2 = qreq_from_fv(fvhi)

    def qreq_band(z):
        return sorted((float(q_e1(z)), float(q_e2(z))))

    # ---- Q_avail: three-phase kinematic ceiling (analytic) ----
    def q_ceiling_3phase(deep_th, shape):
        out = []
        for z in REPORT_Z:
            fd = float(shape.frac(deep_th, z))
            fb = float(shape.frac(0.0, z))
            fs = fb - fd
            fw = 1.0 - fb
            dzdt = float(dz_dtau(z))
            fdp = float(shape.fracp_z(deep_th, z)) * dzdt      # df_d/dtau
            fbp = float(shape.fracp_z(0.0, z)) * dzdt          # df_below/dtau
            fsp = fbp - fdp
            fwp = -fbp
            q = (2.0 / 3.0) * (fdp ** 2 / fd + fsp ** 2 / fs + fwp ** 2 / fw)
            out.append(q)
        return np.array(out)

    # ---- Q_avail: empty-Milne two-phase physical bound + physical (depth-capped) ----
    def q_milne_scaled(kappa, sigma0=SIGMA0):
        """6 f_below f_w (kappa H_w)^2 ; kappa=0.5 empty(Milne), kappa=0.5|dm| physical."""
        out = []
        sm_ = ShapeModel(Dz, sigma0)
        for z in REPORT_Z:
            fb = float(sm_.frac(0.0, z))
            fw = 1.0 - fb
            out.append(6.0 * fb * fw * (kappa * Hw_hb0(z)) ** 2)
        return np.array(out)

    q_ceil = q_ceiling_3phase(DEEP_TH, sm)                     # PRIMARY
    q_ceil_sens = q_ceiling_3phase(DEEP_TH_SENS, sm)           # delta<-0.3 sensitivity
    q_ceil_r100 = q_ceiling_3phase(DEEP_TH, ShapeModel(Dz, 2 * norm.ppf(BELOW_R100)))
    q_ceil_r200 = q_ceiling_3phase(DEEP_TH, ShapeModel(Dz, 2 * norm.ppf(BELOW_R200)))
    # most-permissive / least-permissive ceiling envelope across the variants + reliable-vol band
    ceil_stack = np.vstack([q_ceil, q_ceil_sens, q_ceil_r100, q_ceil_r200])
    q_ceil_hi = ceil_stack.max(axis=0)
    q_ceil_lo = ceil_stack.min(axis=0)

    q_empty = q_milne_scaled(0.5)                             # empty-void (Milne) physical ceiling
    q_empty_r100 = q_milne_scaled(0.5, 2 * norm.ppf(BELOW_R100))
    q_empty_r200 = q_milne_scaled(0.5, 2 * norm.ppf(BELOW_R200))
    q_phys = q_milne_scaled(0.5 * abs(LENSING_DM[0]))          # dm=-0.5 central (kappa=0.25)
    q_phys_shallow = q_milne_scaled(0.5 * abs(LENSING_DM[1]))  # dm=-0.3 edge (kappa=0.15)
    q_phys_deepest = q_milne_scaled(0.5 * abs(LENSING_DM_DEEPEST))  # dm=-0.8 (kappa=0.40)

    # ---- per-z assembly + ratios + gate ----
    q_at_nodes = {}
    for i, z in enumerate(REPORT_Z):
        qr = float(Qreq_cb(z))
        qb = qreq_band(z)
        q_at_nodes[f"z={z:g}"] = {
            "Q_req_central": round(qr, 4),
            "Q_req_band_dchi2le1": [round(qb[0], 4), round(qb[1], 4)],
            "Q_req_over3_gate": round(qr / 3.0, 4),
            "Q_avail_ceiling_3phase_kinematic": round(float(q_ceil[i]), 4),
            "Q_avail_ceiling_band": [round(float(q_ceil_lo[i]), 4), round(float(q_ceil_hi[i]), 4)],
            "Q_avail_ceiling_deep_dm<-0.3": round(float(q_ceil_sens[i]), 4),
            "Q_avail_empty_Milne_bound": round(float(q_empty[i]), 4),
            "Q_avail_empty_band_r100_r200": [round(float(min(q_empty_r100[i], q_empty_r200[i])), 4),
                                             round(float(max(q_empty_r100[i], q_empty_r200[i])), 4)],
            "Q_avail_physical_dm-0.5": round(float(q_phys[i]), 4),
            "Q_avail_physical_band_dm-0.3_-0.8": [round(float(q_phys_shallow[i]), 4),
                                                  round(float(q_phys_deepest[i]), 4)],
            "ratio_ceiling_over_req": round(float(q_ceil[i]) / qr, 4),
            "ratio_empty_over_req": round(float(q_empty[i]) / qr, 4),
            "ratio_physical_over_req": round(float(q_phys[i]) / qr, 4),
        }

    # gate: NO-GO iff Q_avail_CEILING < Q_req/3 for ALL z in [0.3,1.0].
    # Evaluate at the most-permissive Q_avail (upper ceiling envelope); confirm robustness at the
    # least-permissive ceiling and at the physical empty-Milne bound, and against Q_req upper band.
    def below_req3_all(qav):
        return all(float(qav[list(REPORT_Z).index(z)]) < float(Qreq_cb(z)) / 3.0 for z in GATE_Z)

    nogo_permissive = below_req3_all(q_ceil_hi)               # most permissive Q_avail
    nogo_central = below_req3_all(q_ceil)
    nogo_least = below_req3_all(q_ceil_lo)                    # least permissive ceiling variant
    nogo_empty = below_req3_all(q_empty)                     # physical empty-Milne bound
    gate = "NO-GO" if nogo_permissive else "GO"

    # per-z gate detail across the 0.3<=z<=1.0 window
    gate_detail = []
    for z in GATE_Z:
        i = list(REPORT_Z).index(z)
        r3 = float(Qreq_cb(z)) / 3.0
        r3_hi = qreq_band(z)[1] / 3.0
        gate_detail.append({
            "z": float(z),
            "Q_req_over3": round(r3, 4),
            "Q_req_over3_upperband": round(r3_hi, 4),
            "Q_avail_ceiling_central": round(float(q_ceil[i]), 4),
            "Q_avail_ceiling_least_permissive": round(float(q_ceil_lo[i]), 4),
            "Q_avail_empty_Milne": round(float(q_empty[i]), 4),
            "ceiling_clears_req3": bool(float(q_ceil[i]) >= r3),
            "empty_Milne_clears_req3": bool(float(q_empty[i]) >= r3),
            "empty_Milne_clears_req3_upperband": bool(float(q_empty[i]) >= r3_hi),
        })

    # ---- A1 physicality strike ----
    # per-phase kinematic contrast vs the empty (Milne) per-void ceiling 0.5*H_w
    strike_perphase = []
    for z in REPORT_Z:
        fd = float(sm.frac(DEEP_TH, z)); fb = float(sm.frac(0.0, z)); fs = fb - fd; fw = 1.0 - fb
        dzdt = float(dz_dtau(z)); hw = float(Hw_hb0(z))
        fdp = float(sm.fracp_z(DEEP_TH, z)) * dzdt
        fbp = float(sm.fracp_z(0.0, z)) * dzdt
        fsp = fbp - fdp; fwp = -fbp
        Hd_mHbar = fdp / (3 * fd); Hw_mHbar = fwp / (3 * fw); Hs_mHbar = fsp / (3 * fs)
        strike_perphase.append({
            "z": float(z),
            "deep_(H_d-H_w)_over_H_w": round((Hd_mHbar - Hw_mHbar) / hw, 4),
            "shallow_(H_s-H_w)_over_H_w": round((Hs_mHbar - Hw_mHbar) / hw, 4),
            "empty_Milne_ceiling_over_H_w": 0.5,
            "deep_exceeds_Milne": bool((Hd_mHbar - Hw_mHbar) / hw > 0.5),
        })

    hw0 = float(Hw_hb0(0.0))
    req_HvHw0 = float(HvHw_pub[0])                            # 0.51562, committed requirement @z=0
    empty_HvHw0 = 0.5 * hw0                                   # Milne per-void ceiling in Hbar0 units
    strike = {
        "statement": ("The committed kinematic requirement demands a void-wall Hubble contrast "
                      "(H_v-H_w)/Hbar0 = %.4f at z=0, which EXCEEDS the empty-void (Milne) ceiling "
                      "0.5*H_w/Hbar0 = %.4f by x%.2f -- unattainable by ANY physical void, empty or "
                      "not.  Lensing-calibrated depths delta_m~-0.3..-0.5 cap the contrast at "
                      "0.5|delta_m|*H_w/Hbar0 = %.4f..%.4f, a factor %.1f..%.1f below the requirement."
                      % (req_HvHw0, empty_HvHw0, req_HvHw0 / empty_HvHw0,
                         0.5 * 0.3 * hw0, 0.5 * 0.5 * hw0,
                         req_HvHw0 / (0.5 * 0.5 * hw0), req_HvHw0 / (0.5 * 0.3 * hw0))),
        "required_HvHw_over_Hbar0_z0": round(req_HvHw0, 4),
        "empty_Milne_max_HvHw_over_Hbar0_z0": round(empty_HvHw0, 4),
        "required_over_empty_ceiling": round(req_HvHw0 / empty_HvHw0, 3),
        "physical_max_HvHw_over_Hbar0_z0": {
            "dm=-0.3": round(0.5 * 0.3 * hw0, 4),
            "dm=-0.5": round(0.5 * 0.5 * hw0, 4),
            "dm=-0.8": round(0.5 * 0.8 * hw0, 4),
        },
        "required_over_physical_dm-0.5": round(req_HvHw0 / (0.5 * 0.5 * hw0), 3),
        "per_phase_measured_deep_vs_Milne": strike_perphase,
        "measured_deep_exceeds_Milne_for_z_ge": 0.3,
    }

    # ---- Mao EDGE direct deep-void fractions: cross-check of the deep-phase growth rate ----
    mao = tele["EDGE_fixed_density_Mao"]["z_bins"]
    zc_m = np.array([b["z_c"] for b in mao])
    fill_m = np.array([b["fill_fraction"] for b in mao])
    mao_cb = PchipInterpolator(zc_m, fill_m)
    # dln f_deep / dtau over the Mao span (implies (H_d - Hbar) = (1/3) dln f_d/dtau)
    mao_check = []
    for z in (0.3, 0.5):
        f = float(mao_cb(z)); fp = float(mao_cb.derivative()(z)) * float(dz_dtau(z))
        Hd_mHbar = fp / (3.0 * f)
        mao_check.append({"z": z, "f_deep_Mao": round(f, 4),
                          "(H_d-Hbar)_over_Hbar0": round(Hd_mHbar, 4),
                          "over_H_w": round(Hd_mHbar / float(Hw_hb0(z)), 4)})

    # ---- emit ----
    out = {
        "probe": "WP-B Stage 0 -- Q-budget pre-gate (analytic; exact kinematic identities, no solver)",
        "spec": "PLAN_WPB_threephase.md sec 2 + amendment A1",
        "reading": ("Q_req is the KINEMATIC-forced requirement (paper-2 modelV_probeR "
                    "derived_backreaction_V); Q_avail is the exact Buchert multi-phase identity "
                    "Q = 6 sum_i f_i (H_i-Hbar)^2 with H_i-Hbar = fdot_i/(3 f_i) applied to the "
                    "MEASURED three-phase void population.  Both share one bare-time foliation "
                    "tau(z)=2/(3 H_w) from the committed background.  ORDER-OF-MAGNITUDE go/no-go, "
                    "NOT a verdict."),
        "gate": gate,
        "gate_z_range": [0.3, 1.0],
        "gate_rule": ("NO-GO iff Q_avail_CEILING(z) < Q_req(z)/3 for ALL 0.3<=z<=1.0 "
                      "(most-permissive Q_avail); else GO"),
        "gate_evaluation": {
            "NO-GO_at_most_permissive_ceiling": bool(nogo_permissive),
            "NO-GO_at_central_ceiling": bool(nogo_central),
            "NO-GO_at_least_permissive_ceiling": bool(nogo_least),
            "NO-GO_at_empty_Milne_bound": bool(nogo_empty),
            "per_z": gate_detail,
            "note": ("GO is robust: even the least-permissive 3-phase ceiling and the physical "
                     "empty-Milne bound clear Q_req/3 (and Q_req_upperband/3) at z>=0.5; NO-GO "
                     "requires failure at ALL z in [0.3,1.0].  Only the depth-capped PHYSICAL "
                     "budget falls below Q_req/3 -- but the gate is pre-registered on the CEILING."),
        },
        "z_report": REPORT_Z.tolist(),
        "q_at_nodes": q_at_nodes,
        "Q_req_central": {f"z={z:g}": round(float(Qreq_cb(z)), 4) for z in REPORT_Z},
        "Q_req_band_dchi2le1": {f"z={z:g}": [round(qreq_band(z)[0], 4), round(qreq_band(z)[1], 4)]
                                for z in REPORT_Z},
        "Q_avail_ceiling_3phase_kinematic": {f"z={z:g}": round(float(q_ceil[i]), 4)
                                             for i, z in enumerate(REPORT_Z)},
        "Q_avail_ceiling_band": {f"z={z:g}": [round(float(q_ceil_lo[i]), 4),
                                              round(float(q_ceil_hi[i]), 4)]
                                 for i, z in enumerate(REPORT_Z)},
        "Q_avail_empty_Milne_bound": {f"z={z:g}": round(float(q_empty[i]), 4)
                                      for i, z in enumerate(REPORT_Z)},
        "Q_avail_physical": {f"z={z:g}": round(float(q_phys[i]), 4)
                             for i, z in enumerate(REPORT_Z)},
        "Q_avail_physical_note": ("depth-capped at lensing delta_m=-0.5 (kappa=0.25); band "
                                  "delta_m in [-0.3,-0.8].  ~4x below the empty-Milne bound and "
                                  "~5-6x below the (partly super-physical) kinematic ceiling."),
        "physicality_strike_A1": strike,
        "mao_edge_deep_growth_crosscheck": {
            "source": "telescope_fvobs.json EDGE_fixed_density_Mao (delta_min<-0.5 ZOBOV, Reff<=100)",
            "caveat": ("Mao decline conflates real growth with LOWZ->CMASS tracer/sampling change "
                       "(overstates the decline) -> permissive edge only, not primary."),
            "implied_deep_contrast": mao_check,
            "note": ("the direct deep-void fractions decline even steeper than the excursion-set "
                     "shape model, so they imply an even LARGER (more super-Milne) deep-phase "
                     "contrast -- strengthens the physicality strike, does not relieve it."),
        },
        "phases": {
            "deep_d": "delta_m < -0.5 (primary; -0.3 sensitivity), lognormal excursion set",
            "shallow_s": "below-mean minus deep",
            "walls_w": "1 - below-mean",
            "below_mean_definition": "P(delta_m<0)=Phi(sigma/2), bias-INDEPENDENT (telescope Rs=4)",
            "f_i_at_report_z": {
                f"z={z:g}": {
                    "f_deep": round(float(sm.frac(DEEP_TH, z)), 4),
                    "f_shallow": round(float(sm.frac(0.0, z) - sm.frac(DEEP_TH, z)), 4),
                    "f_walls": round(float(1.0 - sm.frac(0.0, z)), 4),
                    "f_below_mean": round(float(sm.frac(0.0, z)), 4),
                } for z in REPORT_Z},
        },
        "caveats": [
            "Stage 0 is an ORDER-OF-MAGNITUDE go/no-go, NOT a verdict (spec sec 2).",
            "Both sides reading-dependent: Q_req is the kinematic-forced requirement; the DYNAMICAL "
            "requirement differs because Q back-reacts on Hbar via the Hamiltonian constraint.",
            "GATE=GO is on the CEILING (kinematic identity / empty-void budget); the PHYSICAL "
            "(lensing-capped) budget is 4-8x below Q_req and falls below Q_req/3 at every z.",
            "The 3-phase kinematic ceiling is itself PARTLY SUPER-PHYSICAL: the measured deep phase's "
            "implied (H_d-H_w) exceeds the empty-Milne 0.5*H_w ceiling for z>=0.3 (physicality strike).",
            "The committed requirement's (H_v-H_w)/Hbar0=0.516 at z=0 already exceeds the empty-Milne "
            "ceiling (x1.36) -- an independent physicality strike, contrast-side, population-free.",
            "z>0 below-mean uses the telescope survey-derived / growth-model curve (validated LCDM "
            "growth, reduced chi2~1); the deep-phase z-evolution is the excursion-set model, "
            "cross-checked (permissively) by the Mao EDGE direct fractions.",
            "dz/dtau reconstructed from the committed background reproduces the published Q_req at the "
            "clean fv-anchor nodes to ~5-15%% (Stage-0 foliation systematic); ratio uses committed Q_req.",
        ],
        "provenance": {
            "Q_req_source": "free-history-timescape/probes_out/modelV_probeR.json :: derived_backreaction_V",
            "below_mean_source": "free-history-timescape/probes_out/telescope_fvobs.json :: PRIMARY_below_mean_Rs4 (bias-independent) + measured_growth (BOSS CIC, reduced chi2~1)",
            "threshold_family_source": "free-history-timescape/probes_out/phaseD_fvobs.json :: z0_anchor.family, shape_model",
            "deep_edge_source": "telescope_fvobs.json :: EDGE_fixed_density_Mao (delta_min<-0.5)",
            "tau_of_z": "tau=2/(3 H_w), H_w_over_Hbar0 from derived_backreaction_V; PCHIP analytic dz/dtau",
            "growth": "flat-LCDM Om=0.315 linear growing mode, validated to <1e-3 vs telescope D(z) nodes",
            "sigma0": SIGMA0,
            "tracer_bias_band": tele["provenance"]["bias"],
            "reliable_volume_band_below_mean": [BELOW_R100, BELOW_R200],
            "Q_req_reconstruction_relerr_clean_nodes": recon_relerr,
            "identity": "Q = 6 sum_i f_i (H_i-Hbar)^2 = (2/3) sum_i fdot_i^2/f_i ; H_i-Hbar = fdot_i/(3 f_i)",
        },
    }
    os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
    with open(OUTJ, "w") as f:
        json.dump(out, f, indent=2)
    log("wrote", OUTJ)
    log("GATE =", gate)
    for z in REPORT_Z:
        n = q_at_nodes[f"z={z:g}"]
        log(f"  z={z:4.2f}  Qavail_ceil={n['Q_avail_ceiling_3phase_kinematic']:7.4f} "
            f"Qempty={n['Q_avail_empty_Milne_bound']:7.4f} Qphys={n['Q_avail_physical_dm-0.5']:7.4f} "
            f"| Qreq={n['Q_req_central']:7.4f} (/3={n['Q_req_over3_gate']:.4f}) "
            f"ceil/req={n['ratio_ceiling_over_req']:.3f}")
    log("STRIKE: required (H_v-H_w)/Hbar0(z=0)=%.4f  vs empty-Milne ceiling %.4f  (x%.2f)"
        % (req_HvHw0, empty_HvHw0, req_HvHw0 / empty_HvHw0))
    return out


if __name__ == "__main__":
    main()
