#!/usr/bin/env python3
"""Diagnostic v6: hybrid z-ODE LB solver (the robust design for the module).

Independent variable z. Two regimes in one forward pass:
  z <= z_switch : LB lapse. sigma_LB = dtau/dz from
        1+z = (tau0/tau)^{2/3}((1-fv)/(1-fv0))^{1/3} Phi/gamma0,  Phi=gamma_bar,
        => sigma_LB = tau fvz / (2(1-fv)(Phi-1)).   (accurate & stable at low z)
  z >  z_switch : LA lapse. sigma_LA = -(3/2) tau [1/(1+z) + (1/3)fvz/(1-fv) - fvz/(2+fv)]
        (a stable LINEAR ODE; the LB/LA maps have merged by z_switch since gamma_bar->1
         in the flat-dust tail, so this is accurate for the CMB D_M and cannot blow up).
z_switch is set ABOVE every BAO point (z<=2.33) and BELOW the high-z forward instability
onset (~z=15-30). Single continuous forward integration -> no map differentiation, no
inversion, no backward stiffness. GATE: reproduce LA on the tracker. Off-tracker: the LB
systematic. Run from src/:  ../.venv/bin/python probes/_diag_lb_hybrid.py
"""
import os
import sys
import io
import time
import contextlib
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _SRC)
os.chdir(_SRC)
import fit_timescape as F
import modelv_theory as MV
with contextlib.redirect_stdout(io.StringIO()):
    import harness as HN

Z_MAX = 1200.0
FV_SWITCH = 0.15


def solve_hybrid(fv_of_z, *, fv_switch=FV_SWITCH, z_max=Z_MAX, n_z=6000,
                 gam0_override=None, **_ignore):
    fv0 = float(fv_of_z(0.0))
    tau0 = (2.0 + fv0) / 3.0
    gam0 = (2.0 + fv0) / 2.0 if gam0_override is None else float(gam0_override)
    floor = float(getattr(fv_of_z, "floor", 0.0))
    ceil = float(getattr(fv_of_z, "ceil", 1.0))

    z_grid = np.concatenate([[0.0], np.geomspace(1e-4, z_max, n_z)])
    # precompute f_v and df_v/dz on the grid + midpoints for a fast RK4 RHS
    zc = z_grid
    zm = 0.5 * (z_grid[:-1] + z_grid[1:])
    fv_c = fv_of_z(zc)
    fvz_c = fv_of_z.deriv(zc)
    fv_m = fv_of_z(zm)
    fvz_m = fv_of_z.deriv(zm)
    cm_c = (fv_c <= floor * (1.0 + 1e-9)) | (fv_c >= ceil * (1.0 - 1e-9))
    cm_m = (fv_m <= floor * (1.0 + 1e-9)) | (fv_m >= ceil * (1.0 - 1e-9))
    fvz_c = np.where(cm_c, 0.0, fvz_c)
    fvz_m = np.where(cm_m, 0.0, fvz_m)

    def sig1(z, tau, fv, fvz, cm):
        if fv < fv_switch or cm:
            return -1.5 * tau * (1.0 / (1.0 + z) + (1.0 / 3.0) * fvz / (1.0 - fv)
                                 - fvz / (2.0 + fv))
        Phi = (1.0 + z) * gam0 * (tau / tau0) ** (2.0 / 3.0) * ((1.0 - fv0) / (1.0 - fv)) ** (1.0 / 3.0)
        d = Phi - 1.0
        if abs(d) < 1e-12:
            return -1.5 * tau / (1.0 + z)
        return tau * fvz / (2.0 * (1.0 - fv) * d)

    # EXACT closed-form LA map (stable everywhere); used for the high-z tail where
    # gamma_bar -> LA and as the BC for the backward LB integration at the switch point.
    tau_la_arr = tau0 * ((1.0 + zc) * (1.0 - fv0) ** (1.0 / 3.0) * (2.0 + fv0)
                         / ((1.0 - fv_c) ** (1.0 / 3.0) * (2.0 + fv_c))) ** (-1.5)

    use_lb = (fv_c >= fv_switch) & (~cm_c)             # LB-active grid points (low z)
    tau = tau_la_arr.copy()                            # tail (fv<fv_switch) = exact LA
    if np.any(use_lb):
        i_sw = int(np.max(np.nonzero(use_lb)[0]))      # highest-z LB-active index
        # integrate LB BACKWARD from z[i_sw] (BC = exact LA) to z=0: the LB trajectory
        # is repelling forward but ATTRACTING backward -> numerically stable; robust to
        # the PCHIP node kinks; cannot fail/hang.
        t = tau_la_arr[i_sw]
        tau[i_sw] = t
        for i in range(i_sw, 0, -1):
            h = z_grid[i - 1] - z_grid[i]              # negative
            k1 = sig1(zc[i], t, fv_c[i], fvz_c[i], cm_c[i])
            k2 = sig1(zm[i - 1], t + 0.5 * h * k1, fv_m[i - 1], fvz_m[i - 1], cm_m[i - 1])
            k3 = sig1(zm[i - 1], t + 0.5 * h * k2, fv_m[i - 1], fvz_m[i - 1], cm_m[i - 1])
            k4 = sig1(zc[i - 1], t + h * k3, fv_c[i - 1], fvz_c[i - 1], cm_c[i - 1])
            t = t + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            tau[i - 1] = t
    if not np.all(np.isfinite(tau)):
        return None
    r_pre = float(tau[0] / tau0)                        # scale before renorm (want ~1)
    # renormalise to the scale convention tau(z=0)=tau0 (distance SHAPE is scale-free,
    # NOTES sec 6.1); tracker: tau[0]==tau0 already so this is a no-op.
    tau = tau * (tau0 / tau[0])
    z = z_grid
    fv = fv_of_z(z)
    fvz = fv_of_z.deriv(z)
    cm = (fv <= floor * (1.0 + 1e-9)) | (fv >= ceil * (1.0 - 1e-9))
    fvz = np.where(cm, 0.0, fvz)
    Phi = (1.0 + z) * gam0 * (tau / tau0) ** (2.0 / 3.0) * ((1.0 - fv0) / (1.0 - fv)) ** (1.0 / 3.0)
    use_lb = (fv >= fv_switch) & (~cm)
    gam = np.where(use_lb, Phi, (2.0 + fv) / 2.0)
    t23 = tau ** (2.0 / 3.0)
    # sigma on the grid (one-time analytic; smooth converged map)
    sig = np.where(
        use_lb,
        tau * fvz / (2.0 * (1.0 - fv) * np.where(np.abs(Phi - 1.0) < 1e-30, 1e-30, Phi - 1.0)),
        -1.5 * tau * (1.0 / (1.0 + z) + (1.0 / 3.0) * fvz / (1.0 - fv) - fvz / (2.0 + fv)))
    fvp = fvz / sig                                    # df_v/dtau
    integ = (-sig) / (gam * t23)
    Jz = cumulative_trapezoid(integ, z, initial=0.0)
    dA = t23 * Jz
    DM = (1.0 + z) * dA
    # gamma_bar'(tau) = (dgam/dz)/sigma, dgam/dz analytic in the LB region
    dgam_dz = np.where(
        use_lb,
        Phi * (1.0 / (1.0 + z) + (1.0 / 3.0) * fvz / (1.0 - fv)) + Phi * (2.0 / (3.0 * tau)) * sig,
        0.5 * fvz)                                     # LA: gamma=(2+fv)/2 -> dgam/dz=fvz/2
    gamp = dgam_dz / sig
    Hbar = 2.0 / (3.0 * tau) + fvp / (3.0 * (1.0 - fv))
    Hd = gam * Hbar - gamp
    DH = 1.0 / Hd
    return dict(z=z, tau=tau, fv=fv, DM=DM, DH=DH, gam=gam, Hd=Hd,
                gamma0=float(gam[0]), nfev=len(z_grid), r_pre=r_pre)


def predict_of(r):
    def predict(zz, k):
        dm = float(np.interp(zz, r["z"], r["DM"]))
        dh = float(np.interp(zz, r["z"], r["DH"]))
        if k == "DM":
            return dm
        if k == "DH":
            return dh
        return (zz * dm * dm * dh) ** (1.0 / 3.0)
    return predict


def compare(fv_of_z, label, sol_la, la_joint, n_zs=(3000, 6000, 12000)):
    zq = np.geomspace(1e-3, 1089.0, 500)
    dm_la, dh_la = sol_la.D_M(zq), sol_la.D_H(zq)
    zHD, _, _, _ = F.load()
    print(f"\n== {label} ==")
    for nz in n_zs:
        t0 = time.time()
        r = solve_hybrid(fv_of_z, n_z=nz)
        dt = time.time() - t0
        if r is None:
            print(f"  n_z={nz:6d} -> FAILED")
            continue
        dm = np.interp(zq, r["z"], r["DM"])
        dh = np.interp(zq, r["z"], r["DH"])
        bao = zq <= 2.4
        rel_dm = float(np.max(np.abs(dm[bao] / dm_la[bao] - 1.0)))
        rel_dh = float(np.max(np.abs(dh[bao] / dh_la[bao] - 1.0)))
        dm_cmb = float(np.interp(1089.8, r["z"], r["DM"]))
        dm_cmb_la = float(sol_la.D_M(1089.8))
        csn = float(HN.sn_chi2(np.interp(zHD, r["z"], r["DM"])))
        cbc, a = HN.bao_cmb_chi2(predict_of(r))
        print(f"  n_z={nz:6d} {dt:.3f}s relDM(z<2.4)={rel_dm:.2e} relDH(z<2.4)={rel_dh:.2e} "
              f"DM_cmb/LA-1={dm_cmb/dm_cmb_la-1:.2e} g0={r['gamma0']:.6f} r_pre={r['r_pre']:.6f} "
              f"SN={csn:.6f} joint={csn+cbc:.4f}(LA {la_joint:.4f})")


def main():
    for fv0 in (0.6426, 0.853):
        trk = MV.tracker_fv_of_z(fv0)
        sol_la = MV.modelv_solve(trk, Ngrid=30000, lapse="algebraic")
        zHD, _, _, _ = F.load()
        cbc, a = HN.bao_cmb_chi2(lambda z, k: float(sol_la.predict(z, k)))
        csn = float(HN.sn_chi2(sol_la.D_M(zHD)))
        compare(trk, f"tracker fv0={fv0} (gamma0 target={(2+fv0)/2:.6f})", sol_la, csn + cbc)

    z_nodes = np.array([0.0, 0.3, 0.7, 1.3, 2.33])
    fv_nodes = np.array([0.62, 0.52, 0.40, 0.28, 0.19])
    bz = np.array([3.5, 5.0, 8.0, 15.0, 40.0, 120.0, 400.0, 1100.0])
    bfv = fv_nodes[-1] * ((1.0 + z_nodes[-1]) / (1.0 + bz)) ** 1.5
    fv = MV.fv_from_nodes(fv_nodes, z_nodes=z_nodes, bridge_z=bz, bridge_fv=bfv)
    sol_la = MV.modelv_solve(fv, Ngrid=30000, lapse="algebraic")
    zHD, _, _, _ = F.load()
    cbc, a = HN.bao_cmb_chi2(lambda z, k: float(sol_la.predict(z, k)))
    csn = float(HN.sn_chi2(sol_la.D_M(zHD)))
    compare(fv, "off-tracker fv0=0.62 (LB SHOULD differ from LA)", sol_la, csn + cbc)
    zk = np.array([0.1, 0.3, 0.55, 0.8, 1.3, 2.0])
    fv0 = float(fv(0.0))
    g0_la = (2.0 + fv0) / 2.0
    print(f"  G0 scan (self-consistent gamma_bar0: r_pre=tau[0]/tau0 must ->1). LA G0={g0_la:.5f}")
    for g0 in np.linspace(g0_la * 0.85, g0_la * 1.15, 9):
        r = solve_hybrid(fv, n_z=6000, gam0_override=g0)
        print(f"    G0={g0:.5f}  r_pre={r['r_pre']:.5f}")
    # root-find G0 so r_pre=1 (adaptive bracket from g0_la in the monotone direction)
    from scipy.optimize import brentq
    def rootfn(g0):
        return solve_hybrid(fv, n_z=6000, gam0_override=g0)["r_pre"] - 1.0
    f0 = rootfn(g0_la)
    lo, hi = g0_la, g0_la
    if f0 > 0:                       # r>1: root above (r decreasing in G0)
        hi = g0_la
        while rootfn(hi) > 0 and hi < 3.0:
            hi *= 1.05
        lo = hi / 1.05
    else:
        lo = g0_la
        while rootfn(lo) < 0 and lo > 0.5:
            lo /= 1.05
        hi = lo * 1.05
    g0_star = brentq(rootfn, lo, hi, xtol=1e-9)
    r = solve_hybrid(fv, n_z=6000, gam0_override=g0_star)
    dm_rel = np.abs(np.interp(zk, r["z"], r["DM"]) / sol_la.D_M(zk) - 1.0)
    dh_rel = np.abs(np.interp(zk, r["z"], r["DH"]) / sol_la.D_H(zk) - 1.0)
    print(f"  self-consistent G0*={g0_star:.5f} (LA {g0_la:.5f})  r_pre={r['r_pre']:.6f}")
    print(f"    DM_rel={np.round(dm_rel,4).tolist()}")
    print(f"    DH_rel={np.round(dh_rel,4).tolist()}  (NOTES ~0.27 at z~0.55)")


if __name__ == "__main__":
    main()
