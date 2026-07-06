#!/usr/bin/env python3
"""
Fit the timescape and spatially-flat LCDM models to the Pantheon+ Type Ia
supernova compilation, with the full statistical+systematic covariance and
analytic marginalisation over the absolute-magnitude / H0 offset.

Distance formulae for timescape are implemented verbatim from
Dam, Heinesen & Wiltshire 2017 (MNRAS 472, 835; arXiv:1706.07236),
Appendix A, eqs (dLTS),(FF),(redshift) and the tracker relations:

    d_L = (1+z)^2 d_A
    d_A = c t^{2/3} [ F(t0) - F(t) ]
    F(t) = 2 t^{1/3}
           + (b^{1/3}/6) ln[ (t^{1/3}+b^{1/3})^2 /
                             (t^{2/3} - b^{1/3} t^{1/3} + b^{2/3}) ]
           + (b^{1/3}/sqrt 3) arctan[ (2 t^{1/3} - b^{1/3})/(sqrt3 b^{1/3}) ]
    1+z = 2^{4/3} t^{1/3} (t+b) / [ fv0^{1/3} Hbar t (2t+3b)^{4/3} ]
    b   = 2 (1-fv0)(2+fv0) / (9 fv0 Hbar)
    fv(t) = 3 fv0 Hbar t / [ 3 fv0 Hbar t + (1-fv0)(2+fv0) ]
    Hbar0 = 2(2+fv0) H0 / (4 fv0^2 + fv0 + 4)   (bare <-> dressed)

Working in dimensionless time tau = Hbar*t, the bare Hubble constant and c
drop out of the *shape* of mu(z); the absolute scale is degenerate with M_B
and is marginalised analytically.  Only fv0 (timescape) / Om0 (LCDM) shape
the Hubble diagram -- consistent with Dam et al.'s statement that SNe
constrain exactly one cosmological parameter per model.
"""
import sys, numpy as np
try:
    from scipy.integrate import cumulative_trapezoid as cumtrap
except ImportError:
    from scipy.integrate import cumtrapz as cumtrap

C_KMS = 299792.458
DATA = "data/PantheonSH0ES.dat"
COV  = "data/PantheonSH0ES_STATSYS.cov"

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
def load():
    import csv
    rows = []
    with open(DATA) as f:
        header = f.readline().split()
        idx = {n: i for i, n in enumerate(header)}
        for line in f:
            p = line.split()
            rows.append(p)
    arr = np.array(rows, dtype=object)
    def col(name, t=float):
        return np.array([t(r[idx[name]]) for r in rows])
    zHD = col("zHD"); zHEL = col("zHEL")
    mb  = col("m_b_corr")
    iscal = col("IS_CALIBRATOR", lambda x: int(float(x)))
    # cosmology sample: drop Cepheid calibrators and z < 0.01
    mask = (iscal == 0) & (zHD > 0.01)
    # covariance (row-major, first line = N), subset to mask
    with open(COV) as f:
        n = int(f.readline())
    flat = np.loadtxt(COV, skiprows=1)
    C = flat.reshape(n, n)
    C = C[np.ix_(mask, mask)]
    return zHD[mask], zHEL[mask], mb[mask], C

# ----------------------------------------------------------------------
# Timescape shape:  D_comoving_shape(zHD)  (up to const c/Hbar0)
# ----------------------------------------------------------------------
def b_tilde(fv0):   return 2.0*(1.0-fv0)*(2.0+fv0)/(9.0*fv0)
def tau0_tilde(fv0):return (2.0+fv0)/3.0

def z_of_tau(tau, fv0):
    b = b_tilde(fv0)
    num = 2.0**(4.0/3.0) * tau**(1.0/3.0) * (tau + b)
    den = fv0**(1.0/3.0) * tau * (2.0*tau + 3.0*b)**(4.0/3.0)
    return num/den - 1.0

def Ftilde(tau, fv0):
    b = b_tilde(fv0); b13 = b**(1.0/3.0); t13 = np.cbrt(tau)
    a = (t13 + b13)**2
    d = tau**(2.0/3.0) - b13*t13 + b**(2.0/3.0)
    return 2.0*t13 + (b13/6.0)*np.log(a/d) + (b13/np.sqrt(3.0))*np.arctan((2.0*t13 - b13)/(np.sqrt(3.0)*b13))

def D_shape_TS(zHD, fv0):
    tau0 = tau0_tilde(fv0)
    grid = np.linspace(1e-7*tau0, tau0, 400000)
    zg = z_of_tau(grid, fv0)                  # decreasing in tau
    order = np.argsort(zg)
    tau_of_z = np.interp(zHD, zg[order], grid[order])
    F0 = Ftilde(tau0, fv0)
    dA = tau_of_z**(2.0/3.0) * (F0 - Ftilde(tau_of_z, fv0))   # x c/Hbar0
    return (1.0 + zHD) * dA                                    # comoving D = (1+z) dA

# ----------------------------------------------------------------------
# Flat LCDM shape:  D_comoving_shape(zHD) = int_0^z dz'/E(z')
# ----------------------------------------------------------------------
def D_shape_LCDM(zHD, Om):
    OL = 1.0 - Om
    zg = np.linspace(0.0, zHD.max()*1.0001, 600000)
    invE = 1.0/np.sqrt(Om*(1.0+zg)**3 + OL)
    Dc = cumtrap(invE, zg, initial=0.0)
    return np.interp(zHD, zg, Dc)

# ----------------------------------------------------------------------
# chi^2 with full covariance, analytic offset (M_B/H0) marginalisation
# ----------------------------------------------------------------------
def make_chi2(zHD, zHEL, mb, C):
    Cinv = np.linalg.inv(C)
    one = np.ones_like(mb)
    Cinv1 = Cinv @ one
    s11 = one @ Cinv1
    def chi2(D_shape):
        mu_shape = 5.0*np.log10((1.0+zHEL)*D_shape)   # +const (marginalised)
        r = mb - mu_shape
        Cinvr = Cinv @ r
        return r @ Cinvr - (one @ Cinvr)**2 / s11
    return chi2

# ----------------------------------------------------------------------
# Grid scan + parabolic 1-sigma
# ----------------------------------------------------------------------
def scan(chi2, D_shape_fn, grid, label):
    chis = np.array([chi2(D_shape_fn(g)) for g in grid])
    i = int(np.argmin(chis)); best = grid[i]; cmin = chis[i]
    # 1-sigma from delta chi^2 = 1
    dchi = chis - cmin
    lo = np.interp(1.0, dchi[:i+1][::-1], grid[:i+1][::-1]) if i > 0 else best
    hi = np.interp(1.0, dchi[i:], grid[i:]) if i < len(grid)-1 else best
    return best, cmin, best-lo, hi-best, grid, chis

def main():
    zHD, zHEL, mb, C = load()
    N = len(zHD)
    print(f"# Pantheon+ cosmology sample: N = {N} SNe (zHD>0.01, non-calibrators)")
    print(f"# redshift range: {zHD.min():.4f} .. {zHD.max():.4f}")
    chi2 = make_chi2(zHD, zHEL, mb, C)

    fv_grid = np.linspace(0.500, 0.960, 461)
    om_grid = np.linspace(0.050, 0.600, 300)

    fv0, cTS, fvlo, fvhi, g1, ch1 = scan(chi2, lambda fv: D_shape_TS(zHD, fv), fv_grid, "TS")
    om0, cL,  omlo, omhi, g2, ch2 = scan(chi2, lambda om: D_shape_LCDM(zHD, om), om_grid, "LCDM")

    # k = 2 fitted params each (cosmological + marginalised offset) -> dBIC = dchi2
    dof = N - 2
    Om_from_fv = 0.5*(1.0-fv0)*(2.0+fv0)
    print()
    print("="*64)
    print("RESULTS")
    print("="*64)
    print(f"TIMESCAPE : fv0   = {fv0:.4f}  (+{fvhi:.4f} / -{fvlo:.4f})")
    print(f"            chi2  = {cTS:.2f}   chi2/dof = {cTS/dof:.4f}")
    print(f"            implied dressed Omega_M0 = 0.5(1-fv0)(2+fv0) = {Om_from_fv:.4f}")
    print(f"LCDM(flat): Om0   = {om0:.4f}  (+{omhi:.4f} / -{omlo:.4f})")
    print(f"            chi2  = {cL:.2f}   chi2/dof = {cL/dof:.4f}")
    print()
    dchi = cTS - cL
    dBIC = dchi   # equal k
    print(f"Delta chi2 (TS - LCDM) = {dchi:+.2f}   (negative favours timescape)")
    print(f"Delta BIC  (TS - LCDM) = {dBIC:+.2f}")
    print()
    # ---- validation gate ----
    ok = 0.70 <= fv0 <= 0.78
    print("VALIDATION GATE  (expect fv0 in 0.73-0.74 cf. Seifert+2024 Pantheon+,")
    print(f"                  0.778 cf. Dam+2017 JLA):  fv0={fv0:.3f}  -> "
          + ("PASS" if ok else "CHECK"))
    np.savez("scan_results.npz", fv_grid=g1, chi2_TS=ch1, om_grid=g2, chi2_LCDM=ch2,
             fv0=fv0, om0=om0, cTS=cTS, cL=cL, N=N)
    print("\n# saved scan_results.npz")

if __name__ == "__main__":
    main()
