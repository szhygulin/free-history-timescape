#!/usr/bin/env python3
"""Emit a self-contained DESI DR2 BAO + Planck-CMB data file for Model V Phase F,
so paper 2 does not depend on paper 1's tree at runtime.

Primary source (embedded below, no external read): the official DESI DR2 Gaussian BAO
likelihood shipped for Cobaya/CosmoMC -- github.com/CobayaSampler/bao_data,
desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt @ commit b7b8a36e9bcc
(2025-03-20), fetched 2026-07-05; paper arXiv:2503.14738. This reproduces exactly the
DR2 rows paper 1 used (its probes_out/dr2.json `dr2_rows_used`), cross-checked inline.

Output: probes_out/desi_dr2_rows.json

Row format matches the shared harness (harness.bao_cmb_rows / timescape_baocmb.build_cov):
each row is (z, kind, value, err, corr) with kind in {DM, DH, DV}; the DR2 covariance is
block diagonal by redshift bin, so a (DM,DH) pair at one z shares a 2x2 block whose
off-diagonal is corr*err_DM*err_DH, DV bins are 1x1, and there is no cross-bin correlation
-- exactly what build_cov reconstructs from these rows.
"""
import os
import json
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "probes_out", "desi_dr2_rows.json")

# ----------------------------------------------------------------- DR2 primary --
# Exact DESI DR2 means (Cobaya bao_data, desi_gaussian_bao_ALL_GCcomb_mean.txt)
DR2_MEAN = [
    (0.295, "DV", 7.94167639),
    (0.510, "DM", 13.58758434), (0.510, "DH", 21.86294686),
    (0.706, "DM", 17.35069094), (0.706, "DH", 19.45534918),
    (0.934, "DM", 21.57563956), (0.934, "DH", 17.64149464),
    (1.321, "DM", 27.60085612), (1.321, "DH", 14.17602155),
    (1.484, "DM", 30.51190063), (1.484, "DH", 12.81699964),
    (2.330, "DM", 38.988973961958784), (2.330, "DH", 8.631545674846294),
]
# z -> (var of first entry [DM or DV], var_DH, cov_DM_DH); from
# desi_gaussian_bao_ALL_GCcomb_cov.txt (block diagonal 2x2 DM/DH blocks per z)
DR2_VARCOV = {
    0.295: (5.78998687e-03, None, None),
    0.510: (2.83473742e-02, 1.83928040e-01, -3.26062007e-02),
    0.706: (3.23752442e-02, 1.11469198e-01, -2.37445646e-02),
    0.934: (2.61732816e-02, 4.04183878e-02, -1.12938006e-02),
    1.321: (1.05336516e-01, 5.04233092e-02, -2.90308418e-02),
    1.484: (5.83020277e-01, 2.68336193e-01, -1.95215562e-01),
    2.330: (2.82685779e-01, 1.02136194e-02, -2.31395216e-02),
}
# Rounded journal-table values (arXiv:2503.14738v2 summary table) -- sensitivity variant
DR2_TABLE = [
    (0.295, "DV", 7.942, 0.075, None),
    (0.510, "DM", 13.588, 0.167, -0.459), (0.510, "DH", 21.863, 0.425, -0.459),
    (0.706, "DM", 17.351, 0.177, -0.404), (0.706, "DH", 19.455, 0.330, -0.404),
    (0.934, "DM", 21.576, 0.152, -0.416), (0.934, "DH", 17.641, 0.193, -0.416),
    (1.321, "DM", 27.601, 0.318, -0.434), (1.321, "DH", 14.176, 0.221, -0.434),
    (1.484, "DM", 30.512, 0.760, -0.500), (1.484, "DH", 12.817, 0.516, -0.500),
    (2.330, "DM", 38.988, 0.531, -0.431), (2.330, "DH", 8.632, 0.101, -0.431),
]


def build_rows():
    """(z, kind, value, err, corr) rows from the exact Cobaya mean+cov."""
    rows, blocks = [], []
    for z, k, v in DR2_MEAN:
        vM, vH, cMH = DR2_VARCOV[z]
        if k == "DV":
            rows.append([z, "DV", v, float(np.sqrt(vM)), None])
        else:
            var = vM if k == "DM" else vH
            corr = None if cMH is None else float(cMH / np.sqrt(vM * vH))
            rows.append([z, k, v, float(np.sqrt(var)), corr])
    for z, (vM, vH, cMH) in DR2_VARCOV.items():
        blocks.append(dict(z=z, var_first=vM, var_DH=vH, cov_DM_DH=cMH))
    return rows, blocks


# Planck 2018 acoustic scale -> D_M(z*)/r_d.  100 theta* = 1.04109(30).
THETA100, SIG100 = 1.04109, 0.00030
ZSTAR, RD = 1089.80, 147.09
# Value paper 1 pairs with the DR2 headline dr2_bao_cmb fit (r*/r_d = 144.39/147.09,
# "identical to verify_and_extend.py"); this is the mixed-column pairing.
CMB_VALUE_HEADLINE = (100.0 / THETA100) * (144.39 / 147.09)          # = 94.29001253...
# Consistent single-Planck-column pairing (r*/r_d = 144.43/147.09); this is the value
# hard-coded in paper 2's src/harness.py (bao_cmb_chi2), so Phase F's harness uses THIS.
CMB_VALUE_HARNESS = (100.0 / THETA100) * (144.43 / 147.09)           # = 94.31614...
SIG_CMB = max(CMB_VALUE_HEADLINE * (SIG100 / THETA100), 0.05)        # = 0.05 floor


def main():
    rows, blocks = build_rows()

    out = dict(
        name="desi_dr2_rows",
        description="Self-contained DESI DR2 BAO + Planck-CMB data for Model V Phase F. "
                    "Rows are (z, kind, value, err, corr) in shared-harness format; the "
                    "covariance is block diagonal by redshift bin.",
        provenance=dict(
            primary="DESI DR2 official Gaussian BAO likelihood: github.com/CobayaSampler/"
                    "bao_data desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt @ "
                    "commit b7b8a36e9bcc (2025-03-20), fetched 2026-07-05; paper arXiv:2503.14738",
            crosscheck="arXiv:2503.14738v2 summary table (rounded journal values), stored as "
                       "table_rows; all 13 central values agree to <=0.001, errors to <=0.004",
            rd_for_H0=RD,
            note_rd_pivot="DESI DR2 (eq. 2 of 2503.14738) pivots its rd fitting formula on "
                          "147.05 Mpc; using 147.09 shifts H0 by +0.03% (~0.02 km/s/Mpc), negligible",
            extracted_from="paper 1 probes_out/dr2.json dr2_rows_used (reproduced from the "
                           "embedded Cobaya mean+cov; cross-checked equal in this script)",
        ),
        rd=RD,
        # ---- primary DR2 BAO rows (harness format) ----
        rows=rows,
        n_bao=len(rows),
        # exact Cobaya raw inputs, for reconstruction / audit
        mean_raw=[[z, k, v] for (z, k, v) in DR2_MEAN],
        cov_blocks=blocks,
        cov_note="Full BAO covariance = block diagonal over redshift bins in `rows` order. "
                 "For a (DM,DH) pair at one z: C[DM,DM]=err_DM^2, C[DH,DH]=err_DH^2, "
                 "C[DM,DH]=corr*err_DM*err_DH. DV bins are 1x1 (err^2). No cross-bin "
                 "correlation. timescape_baocmb.build_cov(rows) reproduces this exactly.",
        # ---- Planck CMB acoustic point ----
        cmb_point=dict(
            zstar=ZSTAR, kind="DM", err=SIG_CMB,
            value_headline=CMB_VALUE_HEADLINE,
            value_harness=CMB_VALUE_HARNESS,
            value=CMB_VALUE_HARNESS,
            row=[ZSTAR, "DM", CMB_VALUE_HARNESS, SIG_CMB, None],
            row_headline=[ZSTAR, "DM", CMB_VALUE_HEADLINE, SIG_CMB, None],
            note="`value`/`row` use the consistent single-Planck-column pairing "
                 "144.43/147.09 = 94.316, which is what paper 2's src/harness.py "
                 "bao_cmb_chi2 already hard-codes, so Phase F using that harness is "
                 "consistent by construction. `value_headline`/`row_headline` "
                 "(144.39/147.09 = 94.290) is the mixed-column value paper 1 reported "
                 "its DR2+CMB headline fit with (its dr2_bao_cmb; the 144.43 pairing is "
                 "paper 1's dr2_bao_cmb_planckfix variant). theta*=1.04109(30); "
                 "error = theta*-propagated, floored at 0.05.",
        ),
        # ---- rounded-table sensitivity variant ----
        table_rows=[list(r) for r in DR2_TABLE],
    )

    # ---- self-check: reconstructed rows must equal paper 1's dr2_rows_used ----
    P1 = [
        [0.295, "DV", 7.94167639, 0.07609196324185624, None],
        [0.51, "DM", 13.58758434, 0.16836678472905517, -0.45156451597511],
        [0.51, "DH", 21.86294686, 0.42886832478046216, -0.45156451597511],
        [0.706, "DM", 17.35069094, 0.17993122074837373, -0.39525761496067274],
        [0.706, "DH", 19.45534918, 0.33387003159912393, -0.39525761496067274],
        [0.934, "DM", 21.57563956, 0.1617815860968114, -0.3472334325670801],
        [0.934, "DH", 17.64149464, 0.20104324858099562, -0.3472334325670801],
        [1.321, "DM", 27.60085612, 0.32455587500459765, -0.39834051635094414],
        [1.321, "DH", 14.17602155, 0.22455135092000672, -0.39834051635094414],
        [1.484, "DM", 30.51190063, 0.763557644844186, -0.49355207107146415],
        [1.484, "DH", 12.81699964, 0.5180117691713191, -0.49355207107146415],
        [2.33, "DM", 38.988973961958784, 0.5316820280957407, -0.4306382085569688],
        [2.33, "DH", 8.631545674846294, 0.10106245296844917, -0.4306382085569688],
    ]
    assert len(rows) == len(P1), (len(rows), len(P1))
    for r, p in zip(rows, P1):
        assert r[0] == p[0] and r[1] == p[1], (r, p)
        assert abs(r[2] - p[2]) < 1e-12, (r, p)
        assert abs(r[3] - p[3]) < 1e-12, (r, p)
        if p[4] is None:
            assert r[4] is None, (r, p)
        else:
            assert abs(r[4] - p[4]) < 1e-12, (r, p)
    assert abs(CMB_VALUE_HEADLINE - 94.29001253445936) < 1e-9, CMB_VALUE_HEADLINE

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[make_desi_dr2_rows] wrote {OUT}")
    print(f"  n_bao={len(rows)}  rd={RD}  cmb value(harness)={CMB_VALUE_HARNESS:.6f} "
          f"headline={CMB_VALUE_HEADLINE:.6f} err={SIG_CMB}")
    print("  self-check vs paper-1 dr2_rows_used: PASS")


if __name__ == "__main__":
    main()
