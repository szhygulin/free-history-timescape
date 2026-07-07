#!/usr/bin/env python3
"""Scratch diagnostic: measure the low-z slope S0 = dD_M/dz|_0 of the ModelV
dressed distance, and compare to 1/Hd(0) and 1/g_dress(fv0), for the tracker
and for the Probe R free-history best fit. This fixes the convention mapping
between the SH0ES-anchored local Hubble slope and the two dressed-H0 conventions.
"""
import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                 # probes/
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))  # src/
import modelv_theory as MV

def s0_of_sol(sol):
    # low-z slope of D_M(z) via a tight finite difference near z=0
    zs = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    dm = sol.D_M(zs)
    # linear fit through origin-ish: slope from polyfit (intercept ~0)
    A = np.polyfit(zs, dm, 1)
    return float(A[0]), float(A[1])

def report(label, sol):
    fv0 = sol.fv0
    gd = float(MV.g_dress(fv0))
    hd0 = float(sol.Hd[0])          # z ascending, index 0 is z~0
    # Hd interpolated exactly at 0
    hd0_i = float(np.interp(0.0, sol.z, sol.Hd))
    s0, icept = s0_of_sol(sol)
    print(f"== {label} ==")
    print(f"  fv0            = {fv0:.6f}")
    print(f"  g_dress(fv0)   = {gd:.6f}   1/g_dress = {1/gd:.6f}")
    print(f"  Hd(0)          = {hd0_i:.6f}   1/Hd(0)  = {1/hd0_i:.6f}")
    print(f"  S0=dD_M/dz|0   = {s0:.6f}   (intercept {icept:.2e})")
    print(f"  S0 vs 1/g_dress ratio = {s0*gd:.6f}")
    print(f"  S0 vs 1/Hd(0)   ratio = {s0*hd0_i:.6f}")
    print(f"  local-rate/Hbar0 = 1/S0 = {1/s0:.6f}")
    print()

# tracker fv0=0.695 (near Seifert Pantheon+)
trk = MV.tracker_fv_of_z(0.695)
sol_trk = MV.modelv_solve(trk, lapse="algebraic", Ngrid=30000)
report("TRACKER fv0=0.695", sol_trk)

# Probe R free-history best fit V
fv_nodes = [0.64013, 0.53112, 0.39578, 0.27945, 0.19359]
z_nodes = [0.0, 0.3, 0.7, 1.3, 2.33]
fvfree = MV.fv_from_nodes(fv_nodes, z_nodes=z_nodes)
sol_free = MV.modelv_solve(fvfree, lapse="algebraic", Ngrid=30000)
report("FREE-HISTORY V (Probe R)", sol_free)

# also the no-lapse V0 fv0=0.383
fv_nodes0 = [0.38292, 0.25043, 0.12798, 0.07152, 0.01699]
fvfree0 = MV.fv_from_nodes(fv_nodes0, z_nodes=z_nodes)
sol_free0 = MV.modelv_solve(fvfree0, lapse="none", Ngrid=30000)
report("FREE-HISTORY V0 no-lapse (Probe R)", sol_free0)
