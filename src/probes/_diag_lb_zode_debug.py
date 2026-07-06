#!/usr/bin/env python3
"""Debug the z-ODE: forward-in-z is unstable at high z (tau freezes, Phi blows up).
The true trajectory is repelling forward => attracting BACKWARD. Test backward
integration from a high-z EdS boundary condition and check it (a) matches LA tau(z)
at all z and (b) lands on tau(0)=tau0. Run from src/:
  ../.venv/bin/python probes/_diag_lb_zode_debug.py
"""
import os
import sys
import io
import contextlib
import numpy as np
from scipy.integrate import solve_ivp

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _SRC)
os.chdir(_SRC)
import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):
    import harness as HN


def run_backward(fv0, trk, z_max=1200.0, n_z=6000, rtol=1e-11, atol=1e-14):
    tau0 = (2.0 + fv0) / 3.0
    gam0 = (2.0 + fv0) / 2.0
    floor = float(getattr(trk, "floor", 0.0))
    ceil = float(getattr(trk, "ceil", 1.0))
    fv_zmax = float(trk(z_max))
    tau_bc = tau0 * ((1.0 + z_max) * gam0 * ((1.0 - fv0) / (1.0 - fv_zmax)) ** (1.0 / 3.0)) ** (-1.5)

    def sigma(z, tau):
        fv = float(trk(z))
        if fv <= floor * (1.0 + 1e-9) or fv >= ceil * (1.0 - 1e-9):
            return -1.5 * tau / (1.0 + z)
        fvz = float(trk.deriv(z))
        Phi = (1.0 + z) * gam0 * (tau / tau0) ** (2.0 / 3.0) * ((1.0 - fv0) / (1.0 - fv)) ** (1.0 / 3.0)
        d = Phi - 1.0
        if abs(d) < 1e-13:
            return -1.5 * tau / (1.0 + z)
        return tau * fvz / (2.0 * (1.0 - fv) * d)

    z_grid = np.concatenate([[0.0], np.geomspace(1e-4, z_max, n_z)])
    sol = solve_ivp(lambda z, y: [sigma(z, y[0])], (z_max, 0.0), [tau_bc],
                    method="LSODA", t_eval=z_grid[::-1], rtol=rtol, atol=atol)
    tau = sol.y[0][::-1]
    return z_grid, tau, sol.nfev, sol.success, tau_bc


def main():
    for fv0 in (0.853, 0.6426):
        trk = MV.tracker_fv_of_z(fv0)
        sol_la = MV.modelv_solve(trk, Ngrid=30000, lapse="algebraic")

        def tau_la(z):
            return np.interp(z, sol_la.z, sol_la.tau)

        z, tau, nfev, ok, tbc = run_backward(fv0, trk)
        print(f"\nfv0={fv0} nfev={nfev} ok={ok}  tau(0)/tau0-1={tau[0]/((2+fv0)/3)-1:.2e}  tau_bc={tbc:.4e}")
        print(f"{'z':>9}{'reltau_vs_LA':>16}")
        for zt in [0.01, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1089.8]:
            i = int(np.argmin(np.abs(z - zt)))
            print(f"{z[i]:9.4g}{tau[i]/tau_la(z[i]) - 1:16.2e}")


if __name__ == "__main__":
    main()
