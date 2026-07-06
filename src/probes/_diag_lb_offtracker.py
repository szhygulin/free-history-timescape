#!/usr/bin/env python3
"""Debug why the hybrid LB solve breaks for the off-tracker node history.
Print fv(z), fvz(z), tau_LB(z) vs tau_LA(z), Phi(z), sigma(z) at low z.
Run from src/:  ../.venv/bin/python probes/_diag_lb_offtracker.py
"""
import os
import sys
import io
import contextlib
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _SRC)
os.chdir(_SRC)
import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):
    import harness as HN

z_nodes = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
fv_nodes = np.array([0.62, 0.52, 0.40, 0.28, 0.19])
bz = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])
bfv = fv_nodes[-1] * ((1.0 + z_nodes[-1]) / (1.0 + bz)) ** 1.5
fv_cb = MV.fv_from_nodes(fv_nodes, z_nodes=z_nodes, bridge_z=bz, bridge_fv=bfv)
sol_la = MV.modelv_solve(fv_cb, Ngrid=30000, lapse="algebraic")
tau_la = lambda z: np.interp(z, sol_la.z, sol_la.tau)

fv0 = float(fv_cb(0.0))
tau0 = (2 + fv0) / 3
gam0 = (2 + fv0) / 2
print(f"off-tracker fv0={fv0:.4f} tau0={tau0:.5f} gam0={gam0:.5f}")
print("check monotonicity + fvz sign on a fine z-grid:")
zz = np.linspace(0, 3.0, 400)
fvv = fv_cb(zz)
fvzz = fv_cb.deriv(zz)
print(f"  fv monotone decreasing? {np.all(np.diff(fvv) <= 1e-12)}   max fvz (should be <0): {fvzz.max():.3e}  min: {fvzz.min():.3e}")

# reproduce the RK4 step-by-step and watch Phi-1 and sigma
def sigma(z, tau, fv, fvz, fv_switch=0.15):
    if fv < fv_switch:
        return -1.5 * tau * (1 / (1 + z) + (1 / 3) * fvz / (1 - fv) - fvz / (2 + fv))
    Phi = (1 + z) * gam0 * (tau / tau0) ** (2 / 3) * ((1 - fv0) / (1 - fv)) ** (1 / 3)
    d = Phi - 1
    return tau * fvz / (2 * (1 - fv) * d), Phi

print(f"\n{'z':>8}{'fv':>9}{'fvz':>11}{'tauLA':>11}{'Phi=gamLB':>11}{'gamLA':>9}{'sigma':>12}")
for zt in [1e-3, 0.05, 0.2, 0.5, 0.9, 1.3, 1.8, 2.33, 2.6, 2.9]:
    fv = float(fv_cb(zt)); fvz = float(fv_cb.deriv(zt)); tl = float(tau_la(zt))
    gl = (2 + fv) / 2
    if fv < 0.15:
        s = sigma(zt, tl, fv, fvz); Phi = np.nan
    else:
        s, Phi = sigma(zt, tl, fv, fvz)
    print(f"{zt:8.3g}{fv:9.4f}{fvz:11.4f}{tl:11.5f}{Phi:11.5f}{gl:9.5f}{s:12.5f}")
