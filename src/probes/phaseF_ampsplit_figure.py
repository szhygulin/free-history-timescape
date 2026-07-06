#!/usr/bin/env python3
"""Headline figure for Phase F: the SN-vs-BAO+CMB amplitude split under the
timescape tracker (persists, ~6.5 sigma) vs the free void history (dissolves, ~0.2
sigma).

Two panels of profile Delta chi2 (each dataset relative to its own minimum):
  LEFT  (tracker)     : x = f_v0 (the tracker's shape==amplitude parameter). The
                        SN-only minimum (~0.85) and the BAO+CMB-only minimum (~0.64)
                        sit far apart -> the 0.85-vs-0.64 split (paper 1).
  RIGHT (free shape)  : x = A = f_v(0), amplitude on the fixed free-fit shape. The
                        SN-only and BAO+CMB-only minima coincide (~0.64) -> the
                        amplitude split dissolves; the tracker SHAPE was the culprit.

Reads the fitted nodes from probes_out/phaseF_joint_ampsplit.json (run
phaseF_joint_ampsplit.py first); recomputes the 1D profiles fresh. Saves
probes_out/fig_ampsplit.png and the profile arrays to
probes_out/phaseF_ampsplit_profiles.json.

Run from src/:   python probes/phaseF_ampsplit_figure.py
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_SRC)

import phaseF_joint_ampsplit as PF   # module-level setup only (no main())
import modelv_probeR as PR
import harness as H

_OUT = os.path.join(os.path.dirname(_SRC), "probes_out")
OUTJ = os.path.join(_OUT, "phaseF_joint_ampsplit.json")
FIG = os.path.join(_OUT, "fig_ampsplit.png")
PROFJ = os.path.join(_OUT, "phaseF_ampsplit_profiles.json")

R = json.load(open(OUTJ))
v_free = np.array(R["dr2_joint_refit"]["free_history"]["fv_nodes"], dtype=float)
shape = v_free / v_free[0]
amp = R["amplitude_split"]
trk = amp["tracker_sanity"]


def tracker_profiles(fv0_grid):
    csn, cbc = [], []
    for x in fv0_grid:
        _, s, b, _, _ = PF.tracker_parts(float(x), PF.NGRID_FINE, ntau=200000)
        csn.append(s); cbc.append(b)
    return np.array(csn), np.array(cbc)


def free_amp_profiles(A_grid):
    csn, cbc = [], []
    for A in A_grid:
        sol = PR.solve_nodes(float(A) * shape, "algebraic", PF.NGRID_FINE)
        s = float(H.sn_chi2(sol.D_M(PF.zHD)))
        b, _ = PF.bao_cmb_chi2_dr2(lambda z, k: float(sol.predict(z, k)))
        csn.append(s); cbc.append(b)
    return np.array(csn), np.array(cbc)


fv0_grid = np.linspace(0.55, 0.94, 60)
A_grid = np.linspace(0.48, 0.82, 60)
t_sn, t_bc = tracker_profiles(fv0_grid)
a_sn, a_bc = free_amp_profiles(A_grid)

SN_C = "#1f6feb"    # SN colour
BC_C = "#d1495b"    # BAO+CMB colour
YMAX = 25.0

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)

for ax, x, sn, bc, xlab, title, xsn, xbc, sig in [
    (axL, fv0_grid, t_sn, t_bc, r"$f_{v0}$  (tracker shape$\equiv$amplitude)",
     "Tracker: SN vs BAO+CMB amplitude split",
     trk["fv0_SN"], trk["fv0_BAOCMB"], trk["sigma"]),
    (axR, A_grid, a_sn, a_bc, r"$A \equiv f_v(0)$  (fixed free-fit shape)",
     "Free history: split dissolves",
     amp["A_SN"], amp["A_BAOCMB"], amp["sigma"]),
]:
    dsn = sn - np.nanmin(sn)
    dbc = bc - np.nanmin(bc)
    ax.plot(x, dsn, color=SN_C, lw=2.2, label="SNe (Pantheon+)")
    ax.plot(x, dbc, color=BC_C, lw=2.2, label="BAO+CMB (DESI DR2+Planck)")
    ax.axvline(xsn, color=SN_C, ls="--", lw=1.2, alpha=0.8)
    ax.axvline(xbc, color=BC_C, ls="--", lw=1.2, alpha=0.8)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax.axvspan(min(xsn, xbc), max(xsn, xbc), color="0.6", alpha=0.12)
    ax.annotate("", xy=(xsn, 0.6 * YMAX), xytext=(xbc, 0.6 * YMAX),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.3))
    ax.text(0.5 * (xsn + xbc), 0.63 * YMAX,
            rf"$\sqrt{{\Delta\chi^2_{{\rm join}}}}={sig:.2f}\,\sigma$",
            ha="center", va="bottom", fontsize=11, color="0.15")
    ax.text(xsn, 0.02 * YMAX, f" SN\n {xsn:.3f}", color=SN_C, fontsize=9,
            ha="center", va="bottom")
    ax.text(xbc, 0.02 * YMAX, f"BAO+CMB\n{xbc:.3f}", color=BC_C, fontsize=9,
            ha="center", va="bottom")
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, YMAX)
    ax.set_xlim(x.min(), x.max())
    ax.grid(alpha=0.25)

axL.set_ylabel(r"$\Delta\chi^2$ (each vs its own minimum)", fontsize=11)
axL.legend(loc="upper center", fontsize=9, framealpha=0.9)
fig.suptitle("Freeing the void-history shape collapses the timescape SN vs BAO+CMB "
             "tension (DESI DR2)", fontsize=12.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG, dpi=150, bbox_inches="tight")
print(f"wrote {FIG}")

json.dump(dict(
    note="Profile Delta chi2 curves behind fig_ampsplit.png. Delta chi2 is each "
         "dataset relative to its own 1D minimum over the panel's parameter.",
    tracker=dict(fv0_grid=fv0_grid.tolist(),
                 chi2_SN=t_sn.tolist(), chi2_BC=t_bc.tolist(),
                 fv0_SN=trk["fv0_SN"], fv0_BAOCMB=trk["fv0_BAOCMB"], sigma=trk["sigma"]),
    free=dict(A_grid=A_grid.tolist(),
              chi2_SN=a_sn.tolist(), chi2_BC=a_bc.tolist(),
              A_SN=amp["A_SN"], A_BAOCMB=amp["A_BAOCMB"], sigma=amp["sigma"]),
), open(PROFJ, "w"), indent=1)
print(f"wrote {PROFJ}")
