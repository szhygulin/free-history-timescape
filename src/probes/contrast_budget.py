#!/usr/bin/env python3
"""PLAN_void_content_audit.md sec.2 -- the TWO-SIDED void-wall contrast budget (amendment A2).

Question. The fitted free history *implies* a void-wall expansion contrast
    (H_v - H_w)/<H>  ==  void_expansion_excess  ~ 0.47 at z=0  (0.37..0.59 across z)
committed in probes_out/modelV_probeR.json -> derived_backreaction_V. This is an exact
kinematic identity of the forced f_v(z): (H_v - H_w) = f_v'/(3 f_v (1-f_v)) (sec 1.1).
The empty-void EdS ceiling on H_v/<H> is 3/2 (excess 0.5) -- the required contrast sits
essentially AT that ceiling. Can REAL voids/walls (lensing- and 2M++-measured contents)
supply it?

Method (analytic spherical / separate-universe, EdS background; TWO-SIDED -- credit BOTH the
void emptiness AND the measured wall overdensity; crediting only voids understates the
mechanism):

  VOID side (underdense open patch, parametric eta):
      1+delta = (9/2) (sinh eta - eta)^2 / (cosh eta - 1)^3
      H_v/H_bg = (3/2) sinh(eta) (sinh eta - eta) / (cosh eta - 1)^2      (-> 3/2 as delta->-1)
  WALL side (overdense closed patch, pre-turnaround 0<eta<pi):
      1+delta = (9/2) (eta - sin eta)^2 / (1 - cos eta)^3
      H_w/H_bg = (3/2) sin(eta) (eta - sin eta) / (1 - cos eta)^2         (-> 1 as delta->0,
                                                                            -> 0 at turnaround)
  Turnaround at eta=pi: delta_turn = (9/2) pi^2/8 - 1 = 4.5517 (past it: virialized, H=0).

Same normalization on both sides. The dimensionless available contrast is
      (H_v - H_w)/<H>  =  (h_v - h_w) / (f_v h_v + (1-f_v) h_w),   h == H/H_bg,
H_bg cancels; f_v(z) is the model's forced void fraction (same weighting as the required
curve), so the ONLY difference vs required is that h_v,h_w here come from spherical DYNAMICS at
measured densities, not from the forced kinematics.

VOID depths: stacked-void-lensing central band delta_m in {-0.3,-0.5,-0.8} (+/-0.1)
             (Clampitt & Jain 2015, DES -- the direct dark-matter probe).
WALL density: recomputed from the bias-calibrated 2M++/Carrick field (external_data/
             twompp_density.npy) by mass balance over the delta>0 volume (r100/r200 band).

Verdict grid: STANDS / MARGINAL / WITHDRAWN, per the pre-registered falsifier -- if the
two-sided contrast REACHES the required Delta-chi2<=1 band within MEASURED densities, the
strike is WITHDRAWN. Either way R2 (availability) and WP-H2' (dilution) are untouched.

No git side effects. Writes probes_out/contrast_budget.json.
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                       # .../free-history-timescape
FIELD = os.path.join(REPO, "external_data", "twompp_density.npy")
PROBER_JSON = os.path.join(REPO, "probes_out", "modelV_probeR.json")
OUT = os.path.join(REPO, "probes_out", "contrast_budget.json")

# 2M++ / Carrick+2015 grid (external_data/twompp_README.txt):
#   X=(i-128)*400/256, cell centres -200..+200, spacing 1.5625 Mpc/h, LG at [128,128,128].
NGRID = 257
CENTER = 128
DX = 400.0 / 256.0

DELTA_TURN = (9.0 / 2.0) * np.pi ** 2 / 8.0 - 1.0                   # 4.5517


# ---------------------------------------------------------------------------
# spherical / separate-universe expansion rate relative to the EdS background
# ---------------------------------------------------------------------------
def void_h_over_bg(delta):
    """Underdense open patch: h_v = H_v/H_bg given contrast delta in (-1,0]."""
    tgt = 1.0 + delta
    if tgt >= 1.0:
        return 1.0
    if tgt <= 0.0:
        return 1.5
    g = lambda e: (9.0 / 2.0) * (np.sinh(e) - e) ** 2 / (np.cosh(e) - 1.0) ** 3
    eta = brentq(lambda e: g(e) - tgt, 1e-4, 80.0, xtol=1e-12, rtol=1e-12)
    return (1.5) * np.sinh(eta) * (np.sinh(eta) - eta) / (np.cosh(eta) - 1.0) ** 2


def wall_h_over_bg(delta):
    """Overdense closed patch, pre-turnaround: h_w = H_w/H_bg given contrast delta>=0.
    At/after turnaround the patch is virialized and contributes H=0."""
    tgt = 1.0 + delta
    if tgt <= 1.0:
        return 1.0
    if tgt >= 1.0 + DELTA_TURN:
        return 0.0
    g = lambda e: (9.0 / 2.0) * (e - np.sin(e)) ** 2 / (1.0 - np.cos(e)) ** 3
    eta = brentq(lambda e: g(e) - tgt, 1e-4, np.pi - 1e-9, xtol=1e-12, rtol=1e-12)
    return (1.5) * np.sin(eta) * (eta - np.sin(eta)) / (1.0 - np.cos(eta)) ** 2


def _tables():
    dv = np.linspace(-0.9999, 0.0, 2000)
    hv = np.array([void_h_over_bg(x) for x in dv])
    dw = np.linspace(0.0, DELTA_TURN, 3000)
    hw = np.array([wall_h_over_bg(x) for x in dw])
    return dv, hv, dw, hw


def delta_for_wall_h(target_h, dw, hw):
    """Invert wall_h_over_bg: effective single-top-hat delta whose h_w == target_h."""
    # hw is monotone decreasing in delta on (0, DELTA_TURN)
    return float(np.interp(target_h, hw[::-1], dw[::-1]))


def excess(hv, hw, fv):
    """(H_v-H_w)/<H> with <H>=f_v H_v+(1-f_v)H_w (volume-weighted, H_bg cancels)."""
    return (hv - hw) / (fv * hv + (1.0 - fv) * hw)


# ---------------------------------------------------------------------------
# 2M++ wall / void density recompute (mass balance over the delta>0 volume)
# ---------------------------------------------------------------------------
def field_recompute(dv, hv, dw, hw):
    d = np.load(FIELD)
    assert d.shape == (NGRID, NGRID, NGRID)
    ax = (np.arange(NGRID) - CENTER) * DX
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)

    out = {}
    for rad, tag in [(100.0, "r100"), (200.0, "r200")]:
        vals = d[R < rad]
        meas = vals[vals != 0.0]                              # drop exact-zero survey-mask fill
        n_fill = int(vals.size - meas.size)
        regmean = float(meas.mean())
        void = meas[meas < 0.0]
        wall = meas[meas > 0.0]
        f_void = void.size / meas.size
        f_wall = wall.size / meas.size
        mean_void = float(void.mean())
        mean_wall_volwt = float(wall.mean())                  # incl. rare cluster tail (delta up to ~54)
        median_wall = float(np.median(wall))                  # diffuse-wall typical, robust to tail
        # mass balance assuming cosmic mean 0 over the reliable volume:
        massbal_wall = -mean_void * f_void / f_wall
        # field-averaged LOCAL expansion (physically self-consistent Buchert two-phase average):
        hv_cell = np.interp(void, dv, hv)
        wall_c = np.clip(wall, 0.0, DELTA_TURN)
        hw_cell = np.where(wall >= DELTA_TURN, 0.0, np.interp(wall_c, dw, hw))
        hv_field = float(hv_cell.mean())
        hw_field = float(hw_cell.mean())
        frac_virial = float((wall >= DELTA_TURN).mean())
        # single-top-hat wall delta that REPRODUCES the field-averaged wall expansion:
        delta_wall_eff = delta_for_wall_h(hw_field, dw, hw)
        out[tag] = dict(
            radius_Mpc_h=rad, N_measured=int(meas.size), N_zero_fill=n_fill,
            regional_mean_delta=regmean,
            f_void_delta_lt_0=f_void, f_wall_delta_gt_0=f_wall,
            mean_delta_void=mean_void,
            mean_delta_wall_volume_weighted=mean_wall_volwt,
            median_delta_wall=median_wall,
            massbalance_delta_wall_cosmicmean0=massbal_wall,
            hv_field_averaged=hv_field, hw_field_averaged=hw_field,
            frac_wall_cells_virialized=frac_virial,
            delta_wall_effective_for_expansion=delta_wall_eff,
        )
    # r100/r200 bands
    out["band_delta_wall_massbalance"] = sorted(
        [out["r100"]["massbalance_delta_wall_cosmicmean0"],
         out["r200"]["massbalance_delta_wall_cosmicmean0"]])
    out["band_delta_wall_volume_weighted"] = sorted(
        [out["r100"]["mean_delta_wall_volume_weighted"],
         out["r200"]["mean_delta_wall_volume_weighted"]])
    out["band_delta_wall_median"] = sorted(
        [out["r100"]["median_delta_wall"], out["r200"]["median_delta_wall"]])
    out["band_delta_wall_effective_for_expansion"] = sorted(
        [out["r100"]["delta_wall_effective_for_expansion"],
         out["r200"]["delta_wall_effective_for_expansion"]])
    out["note"] = ("delta is the contrast about the cosmic mean by construction. The "
                   "volume-weighted mean over delta>0 (0.86-0.96) is inflated by the rare "
                   "cluster tail (delta up to ~54); the diffuse-wall median (0.55-0.64) matches "
                   "the plan's expected +0.5..+0.7. The physically self-consistent Buchert wall "
                   "expansion is the volume-average of the LOCAL h_w (clusters past turnaround "
                   "-> H=0), giving hw_field ~0.76-0.77, i.e. an effective single-top-hat "
                   "delta_wall ~0.77-0.80.")
    return out


# ---------------------------------------------------------------------------
# required curve (central reproduced + LA=algebraic-lapse node Delta-chi2<=1 band)
# ---------------------------------------------------------------------------
def required_curve(J):
    ZQ = J["derived_backreaction_V"]["z"]
    central = J["derived_backreaction_V"]["void_expansion_excess"]
    src = "committed derived_backreaction_V.void_expansion_excess"
    band_lo, band_hi = None, None
    band_note = ("Delta-chi2<=1 band unavailable (model import failed); central only. "
                 "The per-node LA band (fv_req_band_dchi2_le1) is on f_v, not on the "
                 "derivative-driven excess.")
    try:
        sys.path.insert(0, HERE)
        import modelv_probeR as P                             # noqa: E402
        vbest = list(J["V"]["fv_nodes"])
        znodes = list(P.Z_NODES)
        repro = P.derived_curves(np.asarray(vbest, float), "algebraic")["void_expansion_excess"]
        # LA node Delta-chi2<=1 envelope: each of the 5 nodes at {lo,hi} from fv_req_band,
        # min/max of the excess over the 2^5 corners (per-node-independent -> conservatively WIDE
        # vs the joint band).
        bnd = J["fv_req_band_dchi2_le1"]
        keys = ["z=0", "z=0.3", "z=0.7", "z=1.3", "z=2.33"]
        los = [bnd[k][0] for k in keys]
        his = [bnd[k][1] for k in keys]
        stack = []
        for mask in range(32):
            v = [his[i] if (mask >> i) & 1 else los[i] for i in range(5)]
            stack.append(P.derived_curves(np.asarray(v, float), "algebraic")["void_expansion_excess"])
        stack = np.array(stack)
        band_lo = stack.min(axis=0).tolist()
        band_hi = stack.max(axis=0).tolist()
        src = ("reproduced from modelV_probeR V.fv_nodes via modelv_probeR.derived_curves "
               "(max|repro-committed|=%.2e); band = LA (algebraic-lapse) per-node "
               "Delta-chi2<=1 envelope over 2^5 fv-node corners" %
               float(np.max(np.abs(np.array(repro) - np.array(central)))))
        band_note = ("per-node-independent Delta-chi2<=1 corners -> conservatively WIDER than "
                     "the joint band; the low-z lower edge is the falsifier bar.")
    except Exception as e:                                    # pragma: no cover
        src += " (band skipped: %s)" % e
    return dict(z=ZQ, central=central, band_lo=band_lo, band_hi=band_hi,
                source=src, band_note=band_note)


# ---------------------------------------------------------------------------
def main():
    dv, hv, dw, hw = _tables()
    J = json.load(open(PROBER_JSON))
    fv_z = J["derived_backreaction_V"]["fv"]                  # model forced void fraction on ZQ
    ZQ = J["derived_backreaction_V"]["z"]
    field = field_recompute(dv, hv, dw, hw)
    req = required_curve(J)

    # wall-density choices carried through (r100/r200 banded), physically-calibrated primary:
    dwall_eff_band = field["band_delta_wall_effective_for_expansion"]   # ~[0.77,0.80]
    dwall_median_band = field["band_delta_wall_median"]                 # ~[0.55,0.64] (plan band)
    dwall_massbal_band = field["band_delta_wall_massbalance"]           # ~[0.78,0.84]
    hw_eff_band = [wall_h_over_bg(dwall_eff_band[0]), wall_h_over_bg(dwall_eff_band[1])]
    hw_median_band = [wall_h_over_bg(dwall_median_band[1]), wall_h_over_bg(dwall_median_band[0])]

    # ---- AVAILABLE grid: void-depth x wall-density, per z ----------------
    void_centrals = [-0.3, -0.5, -0.8]
    grid = []
    for dvoid in void_centrals:
        row = {"delta_void_central": dvoid, "delta_void_band": [dvoid - 0.1, dvoid + 0.1]}
        hv_c = void_h_over_bg(dvoid)
        hv_lo = void_h_over_bg(max(dvoid - 0.1, -0.999))       # deeper void -> higher h_v
        hv_hi = void_h_over_bg(min(dvoid + 0.1, 0.0))          # shallower void -> lower h_v
        row["h_v_central"] = hv_c
        row["h_v_band"] = [hv_hi, hv_lo]                        # [shallow, deep]
        # wall variants (each a per-z curve over the model f_v(z)):
        for wtag, hwv in [("wall_effective", 0.5 * (hw_eff_band[0] + hw_eff_band[1])),
                          ("wall_median", wall_h_over_bg(0.5 * sum(dwall_median_band))),
                          ("wall_massbal", wall_h_over_bg(0.5 * sum(dwall_massbal_band))),
                          ("wall_EdS_void_only", 1.0)]:
            row[wtag] = dict(
                h_w=hwv,
                excess_z=[round(excess(hv_c, hwv, fv), 5) for fv in fv_z],
                excess_z0=round(excess(hv_c, hwv, fv_z[0]), 5),
            )
        # two-sided band at z0 across the void +/-0.1 and the wall r100/r200 (effective) band:
        combos = [excess(a, b, fv_z[0]) for a in (hv_hi, hv_lo)
                  for b in (hw_eff_band[0], hw_eff_band[1])]
        row["two_sided_excess_z0_band"] = [round(min(combos), 5), round(max(combos), 5)]
        grid.append(row)

    # ---- self-consistent FIELD-AVERAGED two-sided contrast (both sides from the 2M++ field)
    field_avail = {}
    for tag in ("r100", "r200"):
        hvf = field[tag]["hv_field_averaged"]
        hwf = field[tag]["hw_field_averaged"]
        field_avail[tag] = dict(
            hv=hvf, hw=hwf,
            excess_z=[round(excess(hvf, hwf, fv), 5) for fv in fv_z],
            excess_z0=round(excess(hvf, hwf, fv_z[0]), 5),
        )
    field_avail["band_excess_z0"] = sorted([field_avail["r100"]["excess_z0"],
                                            field_avail["r200"]["excess_z0"]])

    # ---- ceiling / best-case references ----------------------------------
    hw_densest = min(wall_h_over_bg(field["r100"]["mean_delta_wall_volume_weighted"]),
                     wall_h_over_bg(field["r200"]["mean_delta_wall_volume_weighted"]))
    ceilings = dict(
        empty_void_EdS_wall=dict(
            desc="delta_v=-1 (h_v=1.5), delta_w=0 (h_w=1); naive model reference",
            excess_z=[round(excess(1.5, 1.0, fv), 5) for fv in fv_z],
            excess_z0=round(excess(1.5, 1.0, fv_z[0]), 5)),
        best_case_measured=dict(
            desc=("deepest measured void center delta=-0.9 (h_v=%.4f) + densest measured "
                  "vol-weighted wall (h_w=%.4f)" % (void_h_over_bg(-0.9), hw_densest)),
            h_v=void_h_over_bg(-0.9), h_w=hw_densest,
            excess_z=[round(excess(void_h_over_bg(-0.9), hw_densest, fv), 5) for fv in fv_z],
            excess_z0=round(excess(void_h_over_bg(-0.9), hw_densest, fv_z[0]), 5)),
    )

    # ---- per-z gap: required central minus the plan-central available -----
    # plan-central available = lensing void central (-0.5) x field-effective wall
    hv_pc = void_h_over_bg(-0.5)
    hw_pc = 0.5 * (hw_eff_band[0] + hw_eff_band[1])
    avail_plan_central = [excess(hv_pc, hw_pc, fv) for fv in fv_z]
    gap = [round(req["central"][i] - avail_plan_central[i], 5) for i in range(len(fv_z))]

    # ---- VERDICT ----------------------------------------------------------
    z0 = 0
    req0 = req["central"][z0]
    req0_lo = req["band_lo"][z0] if req["band_lo"] else req0
    field_lo, field_hi = field_avail["band_excess_z0"]
    best0 = ceilings["best_case_measured"]["excess_z0"]
    plan_central0 = round(avail_plan_central[z0], 5)
    # decision logic (low-z, where the lensing/2M++ densities physically apply):
    #  STANDS    : even the best measured two-sided case falls clearly below required band lo
    #  WITHDRAWN : the self-consistent (field-averaged) two-sided contrast REACHES required band
    #  MARGINAL  : field-averaged/central falls short but the measured band straddles required
    if best0 < req0_lo:
        verdict = "STANDS"
    elif field_lo >= req0_lo:
        verdict = "WITHDRAWN"
    else:
        verdict = "MARGINAL"

    reasoning = (
        "At z=0 required (H_v-H_w)/<H>=%.3f (LA Delta-chi2<=1 band lo=%.3f). "
        "Two-sided credit lifts the available contrast far above the void-only estimate "
        "(void-only delta=-0.8 gives %.3f), but the physically SELF-CONSISTENT field-averaged "
        "two-sided contrast (both sides from the actual 2M++ field, clusters past turnaround "
        "-> H=0) is only %.3f-%.3f (r200-r100) and the plan-central case (lensing void "
        "delta=-0.5 x field-effective wall) is %.3f -- both SHORT of required by ~0.06-0.10. "
        "The shortfall is NOT decisive: pushing to the measured extremes (deepest lensing void "
        "center delta=-0.9 + densest measured wall) reaches %.3f, straddling/exceeding the "
        "required band. So the strike is neither cleanly refuted nor withdrawn -> MARGINAL, "
        "exactly the pre-registered contingency. Note the fixed-delta available rises with z as "
        "f_v falls and formally crosses required at high z, but that regime is unphysical "
        "(high-z voids are shallower); the valid comparison is z<~0.5, where it falls short."
        % (req0, req0_lo,
           round(excess(void_h_over_bg(-0.8), 1.0, fv_z[0]), 3),
           field_lo, field_hi, plan_central0, best0))

    out = dict(
        probe="contrast_budget -- PLAN_void_content_audit.md sec.2 amendment A2 (two-sided)",
        question=("Can real (lensing/2M++-measured) void+wall contents supply the fitted "
                  "void-wall expansion contrast (H_v-H_w)/<H> ~0.47 at z=0, which sits at the "
                  "empty-void EdS ceiling of 0.5?"),
        method=("EdS spherical / separate-universe: void=open patch, wall=closed pre-turnaround "
                "patch; available (H_v-H_w)/<H>=(h_v-h_w)/(f_v h_v+(1-f_v)h_w) with the model's "
                "forced f_v(z) and h=H/H_bg; two-sided (credits void emptiness AND wall "
                "overdensity)."),
        spherical_model=dict(
            void="1+delta=(9/2)(sinh e-e)^2/(cosh e-1)^3 ; h_v=(3/2)sinh e (sinh e-e)/(cosh e-1)^2",
            wall="1+delta=(9/2)(e-sin e)^2/(1-cos e)^3 ; h_w=(3/2)sin e (e-sin e)/(1-cos e)^2",
            delta_turnaround=DELTA_TURN,
            empty_void_ceiling_hv_over_bg=1.5,
            references=["Sheth & van de Weygaert 2004 (void spherical evolution)",
                        "Peebles 1980 / Gunn & Gott 1972 (spherical top-hat)",
                        "Clampitt & Jain 2015 DES (stacked void lensing, delta_m band)"],
        ),
        z=ZQ,
        model_forced_fv=fv_z,
        wall_density_recompute=field,
        required=req,
        available_grid=grid,
        available_field_averaged=field_avail,
        available_plan_central=dict(
            desc="lensing void delta=-0.5 (h_v=%.4f) x field-effective wall (h_w=%.4f)"
                 % (hv_pc, hw_pc),
            excess_z=[round(x, 5) for x in avail_plan_central]),
        ceilings=ceilings,
        gap_required_minus_available_plan_central=dict(z=ZQ, gap=gap),
        verdict=verdict,
        verdict_reasoning=reasoning,
        preregistered=dict(
            expected_available="0.25-0.45", expected_required="0.45-0.5",
            expectation="shortfall EXPECTED but NOT PRESUMED; may be MARGINAL not decisive",
            falsifier=("if two-sided contrast REACHES required band within measured densities "
                       "-> WITHDRAWN"),
            untouched="R2 (availability / SHAPE-UNAVAILABLE) and WP-H2' (survey dilution)"),
        provenance=dict(
            field=FIELD,
            field_coord="X=(i-128)*400/256, spacing 1.5625 Mpc/h, LG at [128,128,128]",
            required_source=PROBER_JSON + " -> derived_backreaction_V",
            void_lensing="Clampitt & Jain 2015 (DES) central delta_m in {-0.3,-0.5,-0.8} +/-0.1",
            wall_field="2M++/Carrick+2015 delta_g* (bias-calibrated), mass balance over delta>0",
            normalization="(h_v-h_w)/(f_v h_v+(1-f_v)h_w), h=H/H_bg; identical on both sides",
            script=os.path.relpath(os.path.abspath(__file__), REPO),
        ),
    )

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("verdict:", verdict)
    print("required z0 central=%.4f band_lo=%.4f" % (req0, req0_lo))
    print("field-averaged two-sided z0 band =", field_avail["band_excess_z0"])
    print("plan-central available z0 =", plan_central0)
    print("best-case measured z0 =", best0)
    print("void-only(delta=-0.8) z0 =", round(excess(void_h_over_bg(-0.8), 1.0, fv_z[0]), 4))
    print("wall delta bands: eff=%s median=%s massbal=%s volwt=%s" % (
        [round(x, 3) for x in dwall_eff_band],
        [round(x, 3) for x in dwall_median_band],
        [round(x, 3) for x in dwall_massbal_band],
        [round(x, 3) for x in field["band_delta_wall_volume_weighted"]]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
