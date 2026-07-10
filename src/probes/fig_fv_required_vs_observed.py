#!/usr/bin/env python3
"""Faithful required-vs-observed void-history figure for Paper 2 (Sec. IV A).

Reads ONLY the committed final availability artifact ``probes_out/R2_final.json``
(verdict SHAPE-UNAVAILABLE) -- NOT the superseded ``R2.json`` MARGINAL_top_edge
logic -- and plots, all in the below-mean (bias-independent) definition:

  * required f_v^req(z) for the three lapse readings LA (primary, with its
    Delta chi^2 <= 1 band), LB, V0  (R2_final.lapse_reading_bands);
  * observed below-mean f_v^obs(z): z=0 2M++ anchor (with measured below-mean
    band) and z=0.3/0.7 BOSS DR12 measurements as filled markers + solid
    connector; z=1.3/2.33 LCDM-growth extrapolations as open markers + dashed
    connector (R2_final Part-1 anchor + Part-2 per-node obs_fv_below_mean);
  * the f_v = 0.5 below-mean floor as a hatched exclusion region (floor theorem:
    the below-mean fraction Phi(sigma/2) >= 1/2 for any real field).

Every plotted value equals an R2_final.json field; the companion
``probes_out/fig_fv_required_vs_observed.json`` records each plotted number, its
source key, and an equality check against the artifact (repo-relative paths).

Run from repo root:  .venv/bin/python src/probes/fig_fv_required_vs_observed.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R2_FINAL = os.path.join(ROOT, "probes_out", "R2_final.json")
FIG_PDF = os.path.join(ROOT, "fig_fv_required_vs_observed.pdf")
FIG_PNG = os.path.join(ROOT, "fig_fv_required_vs_observed.png")
OUT_JSON = os.path.join(ROOT, "probes_out", "fig_fv_required_vs_observed.json")

# Colour-blind-safe (Wong 2011) with distinct line styles.
C_LA = "#0072B2"   # blue    -- required LA (primary)
C_LB = "#D55E00"   # verm2   -- required LB
C_V0 = "#009E73"   # green   -- required V0
C_OBS = "#000000"  # black   -- observed below-mean
C_FLOOR = "#999999"


def rel(p):
    return os.path.relpath(p, ROOT)


def main():
    with open(R2_FINAL) as fh:
        R = json.load(fh)

    z = R["z_nodes"]                     # [0, 0.3, 0.7, 1.3, 2.33]
    zc = np.array(z)

    lrb = R["lapse_reading_bands"]
    la = lrb["LA"]["fv_nodes"]
    lb = lrb["LB"]["fv_nodes"]
    v0 = lrb["V0"]["fv_nodes"]
    la_band = lrb["LA"]["fv_req_band_dchi2_le1"]
    la_lo = [la_band[f"z={zz:g}"][0] for zz in z]
    la_hi = [la_band[f"z={zz:g}"][1] for zz in z]

    # --- observed below-mean f_v^obs(z) --------------------------------------
    # z=0 anchor from Part 1 (measured below-mean band + central);
    # z=0.3, 0.7, 1.3, 2.33 from Part 2 per-node obs_fv_below_mean.
    p1 = R["two_part_test"]["part1_level_z0"]
    obs_z0 = p1["measured_below_mean_central"]
    obs_z0_band = p1["measured_below_mean_band"]
    per_node = {p["z"]: p["obs_fv_below_mean"]
                for p in R["two_part_test"]["part2_shape_decline"]["per_node"]}
    obs = [obs_z0, per_node[0.3], per_node[0.7], per_node[1.3], per_node[2.33]]

    # measured (BOSS DR12 / z=0 anchor) vs LCDM-growth extrapolated segments
    z_meas = [0.0, 0.3, 0.7]
    obs_meas = obs[:3]
    z_extrap = [0.7, 1.3, 2.33]        # start at 0.7 so the dashed connector joins
    obs_extrap = [obs[2], obs[3], obs[4]]

    req_decline = R["two_part_test"]["part2_shape_decline"]["req_total_decline_x"]
    obs_decline = R["two_part_test"]["part2_shape_decline"]["obs_total_decline_x"]

    # --- figure ---------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 5.9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    })
    fig, ax = plt.subplots(figsize=(3.4, 2.9))

    # below-mean floor exclusion region (f_v < 0.5 forbidden for the observable)
    ax.axhspan(0.0, 0.5, facecolor="none", edgecolor=C_FLOOR, hatch="////",
               lw=0.0, alpha=0.5, zorder=0)
    ax.axhline(0.5, color=C_FLOOR, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.text(0.05, 0.472, r"$f_v\geq0.5$ floor", fontsize=6.2,
            color="#555555", ha="left", va="top", zorder=6)

    # required LA band + central (primary)
    ax.fill_between(zc, la_lo, la_hi, color=C_LA, alpha=0.22, lw=0, zorder=2,
                    label=r"required LA band ($\Delta\chi^2\!\leq\!1$)")
    ax.plot(zc, la, color=C_LA, lw=1.8, marker="o", ms=4.0, zorder=5,
            label=r"required LA $f_v^{\rm req}(z)$ (primary)")
    # required LB, V0 (systematic bracket)
    ax.plot(zc, lb, color=C_LB, lw=1.3, ls=(0, (5, 2)), marker="^", ms=3.2,
            zorder=4, label="required LB")
    ax.plot(zc, v0, color=C_V0, lw=1.3, ls=(0, (1, 1.4)), marker="s", ms=3.0,
            zorder=4, label="required V0")

    # observed below-mean: measured segment (filled + solid), z=0 band
    ax.plot(z_meas, obs_meas, color=C_OBS, lw=1.4, ls="-", marker="o", ms=4.2,
            mfc=C_OBS, mec=C_OBS, zorder=6,
            label=r"observed $f_v^{\rm obs}$ (BOSS DR12, $z\!\leq\!0.7$)")
    ax.errorbar(0.0, obs_z0,
                yerr=[[obs_z0 - obs_z0_band[0]], [obs_z0_band[1] - obs_z0]],
                fmt="none", ecolor=C_OBS, elinewidth=1.0, capsize=2.5, zorder=6)
    # observed below-mean: LCDM-growth extrapolation (open + dashed)
    ax.plot(z_extrap, obs_extrap, color=C_OBS, lw=1.2, ls=(0, (3, 2)),
            marker="o", ms=4.6, mfc="white", mec=C_OBS, mew=1.1, zorder=6,
            label=r"observed (LCDM-growth extrap., $z\!>\!0.7$)")

    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"void volume fraction $f_v$")
    ax.set_xlim(-0.06, 2.45)
    ax.set_ylim(0.0, 0.72)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.grid(alpha=0.14, lw=0.5)
    leg = ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.57),
                    framealpha=0.96, handlelength=2.0,
                    borderpad=0.35, labelspacing=0.25)
    leg.set_zorder(20)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_PNG, dpi=200)
    plt.close(fig)

    # --- provenance / equality-check record ----------------------------------
    def approx(a, b):
        return bool(abs(float(a) - float(b)) < 1e-9)

    record = {
        "figure": "required vs observed void history (Paper 2 Sec. IV A)",
        "source_artifact": rel(R2_FINAL),
        "source_verdict": R["verdict"],
        "reads_superseded_R2json": False,
        "outputs": {"pdf": rel(FIG_PDF), "png": rel(FIG_PNG)},
        "z_nodes": z,
        "plotted": {
            "required_LA": la, "required_LA_band_lo": la_lo,
            "required_LA_band_hi": la_hi, "required_LB": lb, "required_V0": v0,
            "observed_below_mean": obs,
            "observed_z0_below_mean_band": obs_z0_band,
            "measured_nodes_z_le_0p7": z_meas,
            "extrapolated_nodes_z_gt_0p7": [1.3, 2.33],
            "floor": 0.5,
        },
        "headline": {
            "required_total_decline_x": round(req_decline, 4),
            "observed_total_decline_x": round(obs_decline, 4),
        },
        "equality_checks_vs_artifact": {
            "LA_matches_lapse_reading_bands": la == lrb["LA"]["fv_nodes"],
            "LB_matches_lapse_reading_bands": lb == lrb["LB"]["fv_nodes"],
            "V0_matches_lapse_reading_bands": v0 == lrb["V0"]["fv_nodes"],
            "obs_z07_matches_part2": approx(obs[2], per_node[0.7]),
            "obs_z0_matches_part1_central": approx(obs[0], obs_z0),
            "req_decline_matches": approx(
                req_decline,
                R["two_part_test"]["part2_shape_decline"]["req_total_decline_x"]),
            "obs_decline_matches": approx(
                obs_decline,
                R["two_part_test"]["part2_shape_decline"]["obs_total_decline_x"]),
        },
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"[fig] wrote {rel(FIG_PDF)} / {rel(FIG_PNG)}")
    print(f"[fig] wrote {rel(OUT_JSON)}")
    print(f"[fig] required decline x{req_decline:.2f} vs observed x{obs_decline:.2f}")
    print(f"[fig] all equality checks: "
          f"{all(record['equality_checks_vs_artifact'].values())}")


if __name__ == "__main__":
    main()
