#!/usr/bin/env python3
"""Exploration probe (WP-B Stage-1 Phase-B): what f_s(z), f_d(z) SHAPES can the
three-phase dynamical Buchert solver produce?  The decisive structural question for the
TRACK-FAIL vs TRACKS fork is whether the SHALLOW void fraction can RISE with z (the
measured f_shallow = below-mean - deep rises 0.361 -> 0.497 over z=0..1, because the
below-mean floor is nearly flat while the deep fraction declines steeply).

Not a fit -- a shape scan.  Prints f_s(z), f_d(z) at nodes for a spread of configs.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import threephase_solver as TP  # noqa: E402

NODES = np.array([0.0, 0.3, 0.5, 0.7, 1.0])

MEAS = {  # authoritative measured population (q_budget.json phases.f_i_at_report_z)
    "f_d": {0.0: 0.2821, 0.3: 0.2138, 0.5: 0.1723, 0.7: 0.1358, 1.0: 0.0913},
    "f_s": {0.0: 0.3611, 0.3: 0.4092, 0.5: 0.4388, 0.7: 0.4651, 1.0: 0.4968},
}


def fs_fd_at(cfg, nodes=NODES, Ngrid=40000):
    sol = TP.solve(cfg, Ngrid=Ngrid)
    fs = np.interp(nodes, sol.z, sol.fs)
    fd = np.interp(nodes, sol.z, sol.fd)
    fv = np.interp(nodes, sol.z, sol.fv)
    return fs, fd, fv, sol


def main():
    print("MEASURED (target):")
    print("  z      ", "  ".join(f"{z:6.2f}" for z in NODES))
    print("  f_s    ", "  ".join(f"{MEAS['f_s'][z]:6.3f}" for z in NODES),
          " (RISES with z)")
    print("  f_d    ", "  ".join(f"{MEAS['f_d'][z]:6.3f}" for z in NODES),
          " (declines)")
    print()

    # a spread of configs: vary (alpha_s2, alpha_d2, f_s0, f_d0)
    # ceiling ak = 1/tau0^2 with tau0=(2+f_v0)/3 depends on f_v0
    configs = []
    for fs0 in (0.30, 0.36, 0.42, 0.50):
        for fd0 in (0.10, 0.20, 0.28):
            fv0 = fs0 + fd0
            if not (0.0 < 1 - fv0 < 1):
                continue
            tau0 = (2 + fv0) / 3.0
            ak = 1.0 / tau0 ** 2
            # shallow: matter-rich (small alpha2) ... near-empty (ak); deep: near-empty
            for a_s in (0.30 * ak, 0.60 * ak, 0.90 * ak, 0.999 * ak):
                for a_d in (0.90 * ak, 0.999 * ak):
                    configs.append((fs0, fd0, a_s, a_d, ak))

    print(f"scanning {len(configs)} configs; showing those with the LARGEST df_s(z=0->1):\n")
    rows = []
    for fs0, fd0, a_s, a_d, ak in configs:
        try:
            cfg = TP.ThreePhaseConfig(TP.VoidPhase(fs0, a_s, "s"),
                                      TP.VoidPhase(fd0, a_d, "d"))
            fs, fd, fv, sol = fs_fd_at(cfg)
            dfs = fs[-1] - fs[0]         # f_s(z=1) - f_s(z=0): >0 means RISES with z
            dfd = fd[-1] - fd[0]
            rows.append((dfs, dfd, fs0, fd0, a_s / ak, a_d / ak, fs, fd, sol.Om_s, sol.Om_d))
        except Exception as e:
            print(f"  skip fs0={fs0} fd0={fd0} a_s/ak={a_s/ak:.2f} a_d/ak={a_d/ak:.3f}: {e}")

    rows.sort(key=lambda r: -r[0])   # most-rising shallow first
    print("  df_s = f_s(1)-f_s(0)  [>0 = shallow RISES with z, as data demands]")
    print("  df_s   df_d   fs0   fd0  aS/ak aD/ak   Om_s   Om_d   f_s(nodes)              f_d(nodes)")
    for dfs, dfd, fs0, fd0, ras, rad, fs, fd, oms, omd in rows[:12]:
        print(f"  {dfs:+.3f} {dfd:+.3f}  {fs0:.2f}  {fd0:.2f}  {ras:.2f}  {rad:.3f}  "
              f"{oms:.3f}  {omd:.3f}  [{','.join(f'{x:.3f}' for x in fs)}]  "
              f"[{','.join(f'{x:.3f}' for x in fd)}]")
    print("\n  ... and the LEAST-rising (most-declining) shallow:")
    for dfs, dfd, fs0, fd0, ras, rad, fs, fd, oms, omd in rows[-4:]:
        print(f"  {dfs:+.3f} {dfd:+.3f}  {fs0:.2f}  {fd0:.2f}  {ras:.2f}  {rad:.3f}  "
              f"{oms:.3f}  {omd:.3f}  [{','.join(f'{x:.3f}' for x in fs)}]  "
              f"[{','.join(f'{x:.3f}' for x in fd)}]")

    maxdfs = max(r[0] for r in rows)
    print(f"\n  MAX achievable df_s (shallow rise) over scan = {maxdfs:+.4f}")
    print(f"  data demands df_s = {MEAS['f_s'][1.0]-MEAS['f_s'][0.0]:+.4f}")


if __name__ == "__main__":
    main()
