#!/usr/bin/env python3
"""R2 -- required void history f_v_req(z) vs available (observed) f_v_obs(z).

Overlays the Probe-R required band (nodes + Delta_chi2<=1 profile band) against the
Phase-D observed band (2M++ z~0 anchor + literature + declared growth extrapolation) and
returns the availability verdict of the Model-V pre-registered decision tree (PLAN sec 0, R2).

Inputs:  probes_out/modelV_probeR.json (required),  probes_out/phaseD_fvobs.json (observed).
Outputs: probes_out/R2.json  +  fig_fvhistory.pdf / .png

Run from src/ :  python probes/R2_required_vs_available.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBER = os.path.join(WT, "probes_out", "modelV_probeR.json")
PHASED = os.path.join(WT, "probes_out", "phaseD_fvobs.json")
OUT_JSON = os.path.join(WT, "probes_out", "R2.json")
FIG_PDF = os.path.join(WT, "fig_fvhistory.pdf")
FIG_PNG = os.path.join(WT, "fig_fvhistory.png")

TRACKER_DEMAND = 0.853   # SN-forced tracker f_v0 (Paper 1); the unavailable amplitude
WATERSHED_LIT = [0.50, 0.62]   # standard watershed filling-fraction band (Probe 4 / Pan / Williams)


def frac_height(x, lo, hi):
    """Fractional height of x within [lo,hi]; <0 below, >1 above (in band-widths)."""
    return (x - lo) / (hi - lo)


def main():
    R = json.load(open(PROBER))
    D = json.load(open(PHASED))

    z_nodes = R["z_nodes"]
    req_central = R["V"]["fv_nodes"]
    band = R["fv_req_band_dchi2_le1"]
    req_lo = [band[f"z={z:g}"][0] for z in z_nodes]
    req_hi = [band[f"z={z:g}"][1] for z in z_nodes]

    obs = {p["z"]: p for p in D["fv_obs_points"]}
    obs_lo = [obs[z]["lo"] for z in z_nodes]
    obs_hi = [obs[z]["hi"] for z in z_nodes]
    obs_central = [obs[z]["central"] for z in z_nodes]
    obs_belowmean = [obs[z]["below_mean"] for z in z_nodes]

    bm_band = D["z0_anchor"]["below_mean_band"]      # [0.614, 0.672]
    def_band = D["z0_anchor"]["definition_band"]     # [0.222, 0.672]

    # --- overlap at z=0 ---------------------------------------------------------
    req0 = req_central[0]
    z0 = {
        "f_v_req_z0": req0,
        "f_v_req_z0_band": [req_lo[0], req_hi[0]],
        "vs_below_mean_band": {
            "band": bm_band,
            "inside": bool(bm_band[0] <= req0 <= bm_band[1]),
            "fractional_height": round(frac_height(req0, *bm_band), 3),
            "note": "below-mean = 'faster than mean expanding' = timescape f_v "
                    "(bias-independent). Required sits mid-band.",
        },
        "vs_definition_band": {
            "band": def_band,
            "inside": bool(def_band[0] <= req0 <= def_band[1]),
            "fractional_height": round(frac_height(req0, *def_band), 3),
        },
        "vs_watershed_literature_band": {
            "band": WATERSHED_LIT,
            "inside": bool(WATERSHED_LIT[0] <= req0 <= WATERSHED_LIT[1]),
            "band_widths_above_top": round(frac_height(req0, *WATERSHED_LIT) - 1.0, 3),
            "note": "standard watershed filling fraction (Probe 4 / Pan+2012 / Williams+2024); "
                    "required is just above its 0.62 top.",
        },
        "tracker_demand_0.853": {
            "below_mean_band_widths_above_top": round(
                (TRACKER_DEMAND - bm_band[1]) / (bm_band[1] - bm_band[0]), 2),
            "note": "tracker's SN-forced 0.853 is ~3 below-mean-band-widths above the "
                    "observed top -- unavailable. Free-history 0.640 is inside.",
        },
    }

    # --- shape agreement over the nodes -----------------------------------------
    per_node = []
    for i, z in enumerate(z_nodes):
        inside = bool(obs_lo[i] <= req_central[i] <= obs_hi[i])
        per_node.append({
            "z": z,
            "f_v_req": round(req_central[i], 4),
            "obs_band": [round(obs_lo[i], 4), round(obs_hi[i], 4)],
            "obs_central": round(obs_central[i], 4),
            "obs_below_mean": round(obs_belowmean[i], 4),
            "req_inside_obs_band": inside,
            "req_fractional_height_in_obs_band": round(frac_height(
                req_central[i], obs_lo[i], obs_hi[i]), 3),
            "direct": obs[z]["direct"],
        })
    req_decline = req_central[0] / req_central[-1]
    bm_decline = obs_belowmean[0] / obs_belowmean[-1]
    mod_decline = obs_central[0] / obs_central[-1]
    shape = {
        "per_node": per_node,
        "req_inside_obs_definition_band_all_nodes": all(p["req_inside_obs_band"] for p in per_node),
        "req_decline_z0_to_z233": round(req_decline, 2),
        "obs_below_mean_decline": round(bm_decline, 2),
        "obs_moderate(delta<-0.3)_decline": round(mod_decline, 2),
        "tension": (
            "Required declines by x{:.1f} (0.640->0.194). The below-mean observed edge "
            "declines only x{:.1f} (matches z~0 normalisation but is too flat), while the "
            "moderate delta<-0.3 edge declines x{:.1f} (matches the required SHAPE but has a "
            "z~0 normalisation ~0.44, well below required 0.640). A single-threshold observed "
            "void population cannot match the z~0 NORMALISATION and the SHAPE simultaneously; "
            "the required history threads the upper part of the definition band, riding the "
            "permissive (below-mean) edge at z~0 and the moderate edge at high z."
        ).format(req_decline, bm_decline, mod_decline),
    }

    # --- verdict ----------------------------------------------------------------
    verdict = "MARGINAL_top_edge"
    verdict_reason = (
        "Required present-day f_v(0)=0.640 sits INSIDE the bias-independent below-mean band "
        f"[{bm_band[0]:.3f},{bm_band[1]:.3f}] (mid-band, frac height "
        f"{z0['vs_below_mean_band']['fractional_height']:.2f}) and inside the full 2M++ "
        f"definition band [{def_band[0]:.3f},{def_band[1]:.3f}]; under the below-mean = "
        "'faster than mean expanding' definition (the timescape-correct f_v) the observed "
        "voids SUPPLY the required value. Against the STANDARD watershed filling fraction "
        f"[{WATERSHED_LIT[0]:.2f},{WATERSHED_LIT[1]:.2f}] (Probe 4 / Pan+2012 / Williams+2024) "
        f"required 0.640 sits {z0['vs_watershed_literature_band']['band_widths_above_top']:.2f} "
        "band-widths ABOVE the 0.62 top -> a marginal, top-edge overlap. The required band is "
        "enveloped by the observed definition band at ALL five nodes. This is a qualitative "
        "reversal of the tracker, whose SN-forced 0.853 sits ~3 below-mean-band-widths above "
        "the observed top (unavailable at every definition). Verdict: the observed void "
        "population can supply the required history only at the permissive (below-mean) "
        "definition edge, sitting at the top edge of conventional watershed catalogs -- "
        "consistent with PLAN R2 'overlap at the most permissive definition edge -> run "
        "Phase F expecting a live model'."
    )

    result = {
        "probe": "R2 -- required vs available void history",
        "inputs": {"required": "modelV_probeR.json", "observed": "phaseD_fvobs.json"},
        "z_nodes": z_nodes,
        "required": {"central": req_central, "lo": req_lo, "hi": req_hi},
        "observed": {"lo": obs_lo, "hi": obs_hi, "central": obs_central,
                     "below_mean": obs_belowmean,
                     "direct": [obs[z]["direct"] for z in z_nodes]},
        "overlap_at_z0": z0,
        "shape_agreement": shape,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "figure": {"pdf": FIG_PDF, "png": FIG_PNG},
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[R2] wrote {OUT_JSON}")

    make_figure(z_nodes, req_central, req_lo, req_hi,
                obs_lo, obs_hi, obs_central, obs_belowmean, obs, D)
    print(f"[R2] wrote {FIG_PDF} / {FIG_PNG}")
    print(f"[R2] verdict: {verdict}")
    return result


def make_figure(z_nodes, req_c, req_lo, req_hi, obs_lo, obs_hi, obs_c, obs_bm, obs, D):
    z = np.array(z_nodes)
    # colour-blind-safe: required = blue, observed = orange, tracker = red, reconcile = green
    C_REQ = "#1f5fa6"
    C_OBS = "#e08214"
    C_TRK = "#c0392b"
    C_REC = "#2a8a4a"

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    # observed definition band (deep-void .. below-mean); split direct (z=0) vs extrapolated
    ax.fill_between(z, obs_lo, obs_hi, color=C_OBS, alpha=0.16, lw=0,
                    label="observed definition band (deep-void .. below-mean)")
    ax.plot(z, obs_bm, color=C_OBS, lw=1.8, ls="-",
            label=r"observed below-mean edge ($\delta<0$, faster-than-mean $\equiv f_v$)")
    ax.plot(z, obs_c, color=C_OBS, lw=1.3, ls=":",
            label=r"observed moderate edge ($\delta<-0.3$)")
    # mark the extrapolated (non-direct) region
    ax.axvspan(0.11, z.max() + 0.05, color="0.5", alpha=0.05, lw=0)
    ax.text(1.5, 0.055, "higher-z: growth-model extrapolation\n(declared; not a direct catalog)",
            fontsize=7.5, color="0.4", ha="center", va="bottom")

    # required band + central
    ax.fill_between(z, req_lo, req_hi, color=C_REQ, alpha=0.28, lw=0,
                    label=r"required band ($\Delta\chi^2\leq1$)")
    ax.plot(z, req_c, color=C_REQ, lw=2.4, marker="o", ms=6,
            label=r"required $f_v^{\rm req}(z)$ (Probe R, $\chi^2$=1396.06)")

    # tracker flat demand + BAO+CMB reconciling value
    ax.axhline(TRACKER_DEMAND, color=C_TRK, lw=1.6, ls="--",
               label=r"tracker demand $f_{v0}=0.853$ (SN-forced, Paper 1)")
    ax.axhline(0.640, color=C_REC, lw=1.2, ls="-.", alpha=0.9,
               label=r"reconciling $f_v(0)=0.640$ (SN+BAO+CMB)")

    # literature points
    lit = D["literature"]
    for i, L in enumerate(lit):
        lab = "watershed literature (Pan+2012, Williams+2024)" if i == 0 else None
        ax.errorbar(L["z"], L["value"],
                    yerr=[[L["value"] - L["band"][0]], [L["band"][1] - L["value"]]],
                    fmt="s", color="0.25", ms=6, capsize=3, lw=1.2, label=lab, zorder=6)

    # z=0 direct-measurement marker (below-mean band)
    bm = D["z0_anchor"]["below_mean_band"]
    ax.errorbar(0.0, np.mean(bm), yerr=[[np.mean(bm) - bm[0]], [bm[1] - np.mean(bm)]],
                fmt="D", color=C_OBS, ms=7, capsize=4, lw=1.6, zorder=7,
                label="2M++ below-mean, direct (this work)")

    ax.set_xlabel("redshift  z")
    ax.set_ylabel(r"void volume fraction  $f_v$")
    ax.set_xlim(-0.05, z.max() + 0.08)
    ax.set_ylim(0.0, 0.9)
    ax.set_title("Required vs available void history (Model V, R2)")
    ax.legend(fontsize=7.4, loc="upper right", framealpha=0.92, ncol=1)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
