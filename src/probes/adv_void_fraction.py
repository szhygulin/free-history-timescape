#!/usr/bin/env python3
"""ADVERSARIAL clean-room recomputation of the observed z~0 void volume fraction
from the 2M++ / Carrick+2015 delta_g* field.

Independent of phaseD_fvobs.py: I build my own radial mask over the raw cube and
count the below-mean (delta<0) volume fraction, with exact-zero (survey-mask fill)
cells excluded and the remaining volume renormalised. Deeper thresholds computed as
cross-checks. No import of the authors' probe script.
"""
import os
import json
import numpy as np

FIELD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external_data", "twompp_density.npy")

# Coordinate system straight from external_data/twompp_README.txt:
#   X=(i-128)*400/256, cell centres -200..+200, spacing 1.5625 Mpc/h.
N = 257
CENTER = 128
DX = 400.0 / 256.0  # 1.5625


def main():
    d = np.load(FIELD)
    assert d.shape == (N, N, N)

    # Build cell-centre coordinates (symmetric in all three axes, so which axis is
    # X/Y/Z is irrelevant for a sphere about the origin).
    ax = (np.arange(N) - CENTER) * DX
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)

    out = {"field": FIELD, "coord": "X=(i-128)*400/256, spacing 1.5625, LG at [128,128,128]"}

    for rad, tag in [(100.0, "r<100"), (200.0, "r<200")]:
        for cmp_name, cmp_mask in [("center_dist_lt", R < rad), ("center_dist_le", R <= rad)]:
            insph = cmp_mask
            vals = d[insph]
            n_sphere = int(vals.size)
            n_zero = int((vals == 0.0).sum())
            n_meas = n_sphere - n_zero
            # below-mean = delta<0 (strictly). exact-zero cells are the mask fill:
            # excluded from numerator (not <0) and from the renormalised denominator.
            n_below0 = int((vals < 0.0).sum())
            n_below_m3 = int((vals < -0.3).sum())
            n_below_m5 = int((vals < -0.5).sum())
            meas = vals[vals != 0.0]
            # regional mean of the contrast over measured cells: if ~0, delta<0 is a
            # faithful 'below cosmic-mean' threshold (delta is a contrast about the
            # global mean by construction). Also report fraction below the REGIONAL mean.
            reg_mean = float(meas.mean()) if meas.size else 0.0
            n_below_regmean = int((meas < reg_mean).sum())
            rec = {
                "N_sphere": n_sphere,
                "N_zero_fill": n_zero,
                "fill_fraction": n_zero / n_sphere if n_sphere else 0.0,
                "N_measured": n_meas,
                "regional_mean_delta": reg_mean,
                "frac_below_regional_mean": n_below_regmean / n_meas if n_meas else 0.0,
                "frac_delta_lt_0": n_below0 / n_meas if n_meas else 0.0,
                "frac_delta_lt_-0.3": n_below_m3 / n_meas if n_meas else 0.0,
                "frac_delta_lt_-0.5": n_below_m5 / n_meas if n_meas else 0.0,
            }
            out[f"{tag}__{cmp_name}"] = rec

    # headline: below-mean band from the two radii (using strict < radius, the natural
    # cell-in-sphere convention), central = mean of the two.
    r100 = out["r<100__center_dist_lt"]["frac_delta_lt_0"]
    r200 = out["r<200__center_dist_lt"]["frac_delta_lt_0"]
    out["below_mean_band_r100_r200"] = [r100, r200]
    out["below_mean_central"] = 0.5 * (r100 + r200)
    out["paper1_bracket"] = [0.50, 0.62]
    out["required_fv0"] = 0.64013

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "probes_out", "adv_void_fraction.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
