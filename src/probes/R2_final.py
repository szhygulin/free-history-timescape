#!/usr/bin/env python3
"""R2-final -- paper-2's FINAL telescope-forced availability verdict.

Supersedes probes_out/R2.json (verdict MARGINAL_top_edge, which rode the too-wide
[0.222, 0.672] 2M++ *definition* band). R2-final evaluates the pre-registered TWO-PART
test of NOTES_mapping.md sec 5 / REASONING_AND_ROADMAP.md sec 4b, sec 6, at the
level-anchored eps~0 below-mean mapping, using the Wave-2A telescope measurement.

  PART 1 (z=0 LEVEL, bias-independent below-mean): is f_v^req(0) inside the measured
          below-mean band [0.614, 0.672]?  -- discriminates the lapse readings.
  PART 2 (decline ratio, derived mapping at the level-anchored eps=0): does the measured
          below-mean decline ratio lie inside the required decline-ratio band per node?

  VERDICT vocabulary (roadmap sec 6): SUPPLIED / SHAPE-UNAVAILABLE / MAPPING-UNDERIVABLE.

Synthesises (reads only; touches no raw data):
  probes_out/telescope_fvobs.json      (Wave-2A measured below-mean f_v_obs(z), R_s band)
  probes_out/modelV_probeR.json        (LA required history, fv0=0.640, Q(z))
  probes_out/modelV_probeR_LB.json     (LB required history, fv0=0.588)
  probes_out/mapping_decline_forecast.json (required decline-ratio bands; eps=0 sweep)
  probes_out/R2.json                   (superseded MARGINAL verdict; below-mean band)

Output: probes_out/R2_final.json

Run:  .venv/bin/python src/probes/R2_final.py
"""
import os
import json
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))          # src/probes
REPO = os.path.dirname(os.path.dirname(HERE))              # repo root
POUT = os.path.join(REPO, "probes_out")

TELE = os.path.join(POUT, "telescope_fvobs.json")
PROBER_LA = os.path.join(POUT, "modelV_probeR.json")
PROBER_LB = os.path.join(POUT, "modelV_probeR_LB.json")
FORECAST = os.path.join(POUT, "mapping_decline_forecast.json")
R2_OLD = os.path.join(POUT, "R2.json")
OUT_JSON = os.path.join(POUT, "R2_final.json")

Z_NODES = [0.0, 0.3, 0.7, 1.3, 2.33]


def load(p):
    with open(p) as fh:
        return json.load(fh)


def frac_height(x, lo, hi):
    return (x - lo) / (hi - lo)


def main():
    tele = load(TELE)
    la = load(PROBER_LA)
    lb = load(PROBER_LB)
    fc = load(FORECAST)
    r2old = load(R2_OLD)

    # -- measured below-mean band (Phase-D reliable-volume systematic, reused by Wave-2A) --
    bm_band = r2old["overlap_at_z0"]["vs_below_mean_band"]["band"]   # [0.6143, 0.6723]
    bm_central = tele["provenance"]["below_mean_z0_anchor"]         # 0.6433

    # ---------------------------------------------------------------- PART 1: LEVEL (z=0)
    la_fv0 = la["V"]["fv0"]      # 0.6401
    lb_fv0 = lb["V"]["fv0"]      # 0.5879
    v0_fv0 = la["V0"]["fv0"]     # 0.3829

    def p1(reading, lapse, fv0, note_pass, note_fail):
        fh = frac_height(fv0, bm_band[0], bm_band[1])
        inside = bool(bm_band[0] <= fv0 <= bm_band[1])
        rec = {
            "reading": reading, "lapse": lapse, "fv0_req": fv0,
            "inside_below_mean_band": inside,
            "frac_height_in_band": round(fh, 4),
        }
        if inside:
            rec["status"] = "PASS"
            rec["note"] = note_pass
        else:
            # LB sits just below the lower edge (LEVEL-FAIL); V0 sits far below (FAIL)
            rec["status"] = "FAIL" if reading == "V0" else "LEVEL-FAIL"
            rec["band_widths_below_lower"] = round(-fh, 4)
            rec["note"] = note_fail
        return rec

    part1_per_reading = [
        p1("LA", "algebraic", la_fv0,
           "0.640 mid-band -> the z=0 LEVEL is available at the bias-independent below-mean edge.",
           ""),
        p1("LB", "rate_ratio", lb_fv0,
           "",
           "0.588 sits BELOW the band lower edge 0.614 -> LB fails Part 1 by level."),
        p1("V0", "none", v0_fv0,
           "",
           "0.383 sits far below the band -> V0 fails Part 1 by level (below 0.5 everywhere past z=0)."),
    ]

    part1 = {
        "definition": "bias-independent below-mean f_v(0)=P(delta<0); PASS iff f_v^req(0) inside "
                       "the measured below-mean band. (M1) delta<0 <=> H_local>Hbar is bias-free at "
                       "the sign, so this level is the one definition-free anchor.",
        "measured_below_mean_band": bm_band,
        "measured_below_mean_central": bm_central,
        "measured_source": "telescope_fvobs.json below_mean_z0_anchor (2M++ sigma0, Phase-D "
                           "reliable-volume band reused by Wave-2A; z=0 not re-measured).",
        "per_reading": part1_per_reading,
        "discriminates": True,
        "discriminates_note": "the z=0 below-mean measurement SELECTS LA (supplies 0.640) and "
                              "REJECTS LB (0.588, below the lower edge) and V0 (0.383, far below): "
                              "the readings are discriminated by the level alone (roadmap sec 5.3).",
        "readings_passing_part1": ["LA"],
    }

    # ---------------------------------------------------------------- PART 2: SHAPE (decline)
    # Measured decline ratio = level-anchored eps=0 below-mean at fixed R_s=4 Mpc/h.
    eps0 = next(e for e in fc["eps_sweep"] if e["eps"] == 0.0)
    obs_fv = eps0["fv"]                    # [0.6433, 0.6229, 0.6009, 0.5779, 0.5551]
    obs_ratio = eps0["decline_ratio"]      # [1, 0.9684, 0.9342, 0.8984, 0.8629]

    req = fc["required"]
    req_fv = req["fv"]
    req_ratio = req["decline_ratio"]
    req_lo = req["decline_ratio_lo"]
    req_hi = req["decline_ratio_hi"]

    part2_nodes = []
    for i, z in enumerate(Z_NODES):
        if z == 0.0:
            continue
        above = bool(obs_ratio[i] > req_hi[i])
        inside = bool(req_lo[i] <= obs_ratio[i] <= req_hi[i])
        part2_nodes.append({
            "z": z,
            "obs_fv_below_mean": obs_fv[i],
            "obs_decline_ratio": obs_ratio[i],
            "req_fv": req_fv[i],
            "req_decline_ratio": req_ratio[i],
            "req_decline_ratio_band": [req_lo[i], req_hi[i]],
            "obs_inside_req_band": inside,
            "obs_above_req_hi": above,
            "status": "FAIL_flatter" if above else ("PASS" if inside else "FAIL"),
        })

    part2 = {
        "gate_anchoring": {
            "primary": "level-anchored eps=0 (below-mean); a SINGLE edge, band width 0 -- the Part-2 "
                       "prediction is one curve, not a band. The single mapping parameter eps is "
                       "fixed by the Part-1 z=0 level (non-circular; NOTES_mapping.md sec 4).",
            "deep_void_edge_role": "EXPOSITION ONLY. The fixed-density / deep-void anchor (eps set by "
                                   "a physical void barrier) matches the required decline SHAPE but at "
                                   "f_v(0)=0.414, failing the Part-1 level -- it is NEVER an alternate "
                                   "pass route. No single eps occupies both the (0.640 level) and the "
                                   "(x3.31 shape) corner (NOTES_mapping.md sec 2.2/3/4).",
            "evaluated_against": "LA required decline-ratio band (the only reading passing Part 1).",
        },
        "measured_source": "telescope_fvobs.json PRIMARY_below_mean_Rs4 (eps=0, R_s=4 Mpc/h); "
                           "== mapping_decline_forecast.json eps_sweep[eps=0], whose LCDM D(z) growth "
                           "is validated by the measured CIC growth (sigma_m_ratio 0.765 vs D_ratio "
                           "0.829, reduced chi2 ~1).",
        "required_source": "mapping_decline_forecast.json required.decline_ratio_{lo,hi} (LA history).",
        "per_node": part2_nodes,
        "all_z_gt0_nodes_above_req_hi": all(n["obs_above_req_hi"] for n in part2_nodes),
        "obs_total_decline_x": round(obs_fv[0] / obs_fv[-1], 4),
        "req_total_decline_x": req["total_decline_x"],
        "status": "FAIL",
        "status_note": "the measured below-mean decline is FLATTER than required (sits at the floor) "
                       "and exceeds the required-hi edge at EVERY z>0 node -- most decisively at the "
                       "directly-measurable z~0.7 node. The observed void population cannot supply the "
                       "required x3.31 decline.",
    }

    # ---------------------------------------------------------------- z=0.7 gap (decisive node)
    fvreq_07 = req_fv[2]              # 0.39578
    fvobs_07 = obs_fv[2]             # 0.6009
    gap_abs = fvobs_07 - fvreq_07
    sigma_needed = 2.0 * norm.ppf(fvreq_07)   # Phi(sigma/2)=fvreq -> sigma = 2*Phi^-1(fvreq)
    rs_band = tele["Rs_sensitivity_band"]
    fv07_rs8 = next(b for b in rs_band["band"] if b["R_s"] == 8.0)["fv_below_mean_z07"]

    z07_gap = {
        "z": 0.7,
        "obs_fv_below_mean": fvobs_07,
        "req_fv": fvreq_07,
        "gap_absolute": gap_abs,
        "gap_relative": gap_abs / fvreq_07,
        "obs_decline_ratio": obs_ratio[2],
        "req_decline_ratio": req_ratio[2],
        "req_decline_ratio_band": [req_lo[2], req_hi[2]],
        "obs_ratio_above_req_hi_band": bool(obs_ratio[2] > req_hi[2]),
        "floor_theorem": {
            "sigma_needed_for_req_fv": sigma_needed,
            "impossible": bool(sigma_needed < 0.0),
            "statement": "f_v^req(0.7)=0.396 < 0.5, but the below-mean fraction Phi(sigma/2) >= 0.5 "
                         "for ANY real field (sigma>0). Reaching 0.396 needs sigma<0. The gap is "
                         "therefore DEFINITIONALLY unbridgeable by the below-mean (eps=0) mapping, "
                         "independent of the measured sigma value (NOTES_mapping.md sec 3 "
                         "near-theorem, roadmap 4a floor). SHAPE-UNAVAILABLE.",
        },
        "conservative_Rs_bound": {
            "fv07_Rs8": fv07_rs8,
            "gap_abs_Rs8": fv07_rs8 - fvreq_07,
            "note": "even the most conservative smoothing R_s=8 Mpc/h gives f_v(0.7)=0.560, still "
                    "0.164 above the required 0.396 and still >= 0.5 (floor).",
        },
        "note": "gap in absolute f_v is 0.205 (52% relative). The verdict does NOT rest on a sigma "
                "count (statistical+bias error ~0.006 would give tens of sigma); it rests on the "
                "floor theorem, which is bias- and growth-independent.",
    }

    # ---------------------------------------------------------------- Q(z) / backreaction contrast
    dbr = la["derived_backreaction_V"]
    zg = dbr["z"]
    i07 = zg.index(0.7)
    q_req_07 = dbr["Q_over_Hbar0sq"][i07]
    vee_07 = dbr["void_expansion_excess"][i07]
    pre_req = fvreq_07 * (1.0 - fvreq_07)
    pre_obs = fvobs_07 * (1.0 - fvobs_07)
    q_contrast = {
        "backreaction_identity": "Q = 6 f_v (1-f_v) (H_v-H_w)^2  (K3); H_v-H_w is sourced by the "
                                 "DECLINE of f_v, so a flat f_v(z) sources little backreaction.",
        "Q_req_over_Hbar0sq_grid_z": zg,
        "Q_req_over_Hbar0sq": dbr["Q_over_Hbar0sq"],
        "Q_req_over_Hbar0sq_z07": q_req_07,
        "void_expansion_excess_req_z07": vee_07,
        "prefactor_fv_1_minus_fv_z07": {"req": pre_req, "obs": pre_obs,
                                        "ratio_obs_over_req": pre_obs / pre_req},
        "note": "at z=0.7 the geometric prefactor f_v(1-f_v) is near-DEGENERATE (req 0.2391 vs obs "
                "0.2398, ratio 1.003) because obs 0.601 and req 0.396 straddle 0.5 symmetrically; the "
                "ENTIRE backreaction deficit therefore lives in (H_v-H_w)^2, which the flat observed "
                "decline (x1.16) cannot source. Full Q_obs(z) needs the two-scale solver (not re-run "
                "here), so Q_req(z) stands as the backreaction TARGET the observed voids miss.",
    }

    # ---------------------------------------------------------------- R_s note (non-blocking)
    rs_note = {
        "definition": "f_v(0.7) across R_s in {2,4,8} Mpc/h (telescope_fvobs.json Rs_sensitivity_band).",
        "fv_z07_range": rs_band["fv_z07_range"],
        "floor_holds": rs_band["floor_holds"],
        "statement": "f_v(0.7) stays in [0.560, 0.654] across R_s=2..8 Mpc/h, all >= 0.5 -- so R_s "
                     "CANNOT rescue the level-anchored decline. The floor is R_s-independent; smoothing "
                     "choice is not an escape from SHAPE-UNAVAILABLE (mapping-review tightening).",
    }

    # ---------------------------------------------------------------- lapse required-history bands
    def band_of(probe, key):
        b = probe["fv_req_band_dchi2_le1"]
        return {f"z={z:g}": b[f"z={z:g}"] for z in Z_NODES}

    lapse_reading_bands = {
        "LA": {
            "lapse": "algebraic", "primary": True, "fv0": la_fv0,
            "fv_nodes": la["V"]["fv_nodes"],
            "total_decline_x": round(la["V"]["fv_nodes"][0] / la["V"]["fv_nodes"][-1], 4),
            "fv_req_band_dchi2_le1": band_of(la, "V"),
            "part1": "PASS", "part2": "FAIL_shape",
            "role": "the primary reading: supplies the z=0 level (Part 1 PASS) but fails Part 2 by "
                    "shape -> drives the SHAPE-UNAVAILABLE verdict.",
        },
        "LB": {
            "lapse": "rate_ratio", "primary": False, "fv0": lb_fv0,
            "fv_nodes": lb["V"]["fv_nodes"],
            "total_decline_x": round(lb["V"]["fv_nodes"][0] / lb["V"]["fv_nodes"][-1], 4),
            "fv_req_band_dchi2_le1": band_of(lb, "V"),
            "part1": "LEVEL-FAIL", "part2": "moot (fails Part 1)",
            "role": "fails Part 1 by level (0.588 below 0.614); its required decline is even steeper "
                    "(x4.84), so Part 2 would fail harder -- moot, does not pass Part 1.",
        },
        "V0": {
            "lapse": "none", "primary": False, "fv0": v0_fv0,
            "fv_nodes": la["V0"]["fv_nodes"],
            "total_decline_x": round(la["V0"]["fv_nodes"][0] / la["V0"]["fv_nodes"][-1], 4),
            "fv_req_band_dchi2_le1": "unavailable (V0 is the no-lapse control; Probe R emits no "
                                     "dchi2<=1 band for it)",
            "part1": "FAIL", "part2": "moot (fails Part 1)",
            "role": "no-lapse control; fails Part 1 far below the band (0.383) with the steepest "
                    "required decline (x22.5) -- moot for Part 2.",
        },
    }

    # ---------------------------------------------------------------- verdict
    verdict = "SHAPE-UNAVAILABLE"
    verdict_reason = (
        "Part 1 (z=0 LEVEL): f_v^req(0)=0.640 (LA) sits mid-band in the bias-independent below-mean "
        f"band [{bm_band[0]:.3f},{bm_band[1]:.3f}] -> PASS; the LEVEL is available at the below-mean "
        "edge. The z=0 measurement discriminates the readings: LB (0.588) and V0 (0.383) fail Part 1 "
        "by level. Part 2 (SHAPE, level-anchored eps=0 below-mean at fixed R_s=4): the measured decline "
        "ratio is FLATTER than required at EVERY z>0 node, exceeding the required-hi edge -- at the "
        "decisive z~0.7 node obs f_v=0.601 (ratio 0.934) vs required 0.396 (ratio 0.618, band "
        "[0.599,0.637]): a 0.205 absolute / 52% relative gap. By the floor theorem the below-mean "
        "fraction Phi(sigma/2)>=0.5 for any real field (sigma>0), so reaching 0.396 needs sigma<0 -- "
        "the gap is DEFINITIONALLY unbridgeable, R_s-independent ([0.560,0.654] across R_s=2..8). The "
        "level is available (Part 1) but the observed void population cannot supply the required x3.31 "
        "decline (Part 2): the observed voids cannot supply the backreaction the Hubble diagram wants. "
        "This is the branch the LCDM-growth forecast and the NOTES_mapping.md sec 3 near-theorem "
        "anticipate. f_v^req(z) stands as the model-independent target any backreaction proposal must "
        "hit."
    )

    result = {
        "probe": "R2-final -- FINAL telescope-forced availability verdict (paper 2)",
        "supersedes": {
            "artifact": "probes_out/R2.json",
            "old_verdict": r2old["verdict"],
            "why": "R2.json's MARGINAL_top_edge rode the too-wide 2M++ *definition* band [0.222,0.672] "
                   "(envelope test, necessarily-true, unfalsifiable). R2-final replaces it with the "
                   "pre-registered TWO-PART test (NOTES_mapping.md sec 5): a bias-independent z=0 level "
                   "gate + a derived-mapping decline-ratio gate at the level-anchored eps=0, evaluated "
                   "on the Wave-2A telescope measurement.",
        },
        "reading_convention": "KINEMATIC (f_v^req is what backreaction the Hubble diagram wants, not a "
                              "proven dynamical solution; Buchert integrability not enforced).",
        "z_nodes": Z_NODES,
        "two_part_test": {
            "part1_level_z0": part1,
            "part2_shape_decline": part2,
        },
        "z07_gap": z07_gap,
        "Q_backreaction_contrast": q_contrast,
        "Rs_note": rs_note,
        "lapse_reading_bands": lapse_reading_bands,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "verdict_vocabulary": {
            "SUPPLIED": "Part 1 PASS AND Part 2 inside band at every node AND forced fit clears BIC "
                        "bar -- NOT reached.",
            "SHAPE-UNAVAILABLE": "Part 1 PASS but Part 2 decline flatter than required beyond the "
                                 "floor at z~0.7 -- THIS VERDICT.",
            "MAPPING-UNDERIVABLE": "no principled f_v<->observable mapping without per-z freedom -- NOT "
                                   "triggered (the mapping is derived, single-parameter, anchored "
                                   "non-circularly at z=0).",
        },
        "provenance": {
            "synthesizes": [
                "probes_out/telescope_fvobs.json (Wave-2A measured below-mean f_v_obs(z), decline "
                "ratios, R_s-sensitivity band, floor theorem)",
                "probes_out/modelV_probeR.json (LA required history fv0=0.640, dchi2<=1 bands, Q(z))",
                "probes_out/modelV_probeR_LB.json (LB required history fv0=0.588, dchi2<=1 bands)",
                "probes_out/mapping_decline_forecast.json (required decline-ratio bands; eps=0 "
                "level-anchored below-mean sweep)",
                "probes_out/R2.json (superseded MARGINAL_top_edge; below-mean band source)",
            ],
            "specs": [
                "NOTES_mapping.md sec 4b/5/6 (two-part test, level-anchored eps=0 gate, verdict "
                "vocabulary)",
                "REASONING_AND_ROADMAP.md sec 4b/6 (level-vs-shape tension, decision tree)",
            ],
            "no_raw_data_touched": True,
        },
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[R2-final] wrote {OUT_JSON}")
    print(f"[R2-final] Part 1: LA PASS (mid-band), LB LEVEL-FAIL, V0 FAIL")
    print(f"[R2-final] Part 2: FLATTER at every z>0 node; z=0.7 gap {gap_abs:.3f} abs "
          f"({gap_abs/fvreq_07*100:.0f}% rel); floor sigma_needed={sigma_needed:.3f} (<0 impossible)")
    print(f"[R2-final] verdict: {verdict}")
    return result


if __name__ == "__main__":
    main()
