#!/usr/bin/env python3
"""
Timescape inverse-distance-ladder: fit DESI 2024 BAO + Planck CMB acoustic
scale, using the sound horizon r_d as the standard ruler, and test consistency
with the supernova-preferred void fraction and with H0.

Timescape distances (dressed; Dam et al. 2017 / Wiltshire 2009), dimensionless
(units of c/Hbar0):
  D_M(z) = (1+z) tau^{2/3} [F(tau0)-F(tau)]                 (transverse comoving)
  D_H(z) = 1 / [ H(z)/Hbar0 ],  H/Hbar0 = g(fv) (2+fv) tau0 / [(2+fv0) tau]
  D_V(z) = [ z D_M^2 D_H ]^{1/3}
with g(fv) = (4fv^2+fv+4)/(2(2+fv)),  H0 = g(fv0) Hbar0.

Observables: DM/rd = alpha * D_M,  DH/rd = alpha * D_H,  DV/rd = alpha * D_V,
with alpha = c/(Hbar0 rd).  We fit (fv0, alpha); rd is then external.
"""
import numpy as np
from scipy.optimize import brentq
import fit_timescape as F   # reuse b_tilde, tau0_tilde, Ftilde, z_of_tau

C = 299792.458  # km/s

# ---------------- timescape dimensionless distances ----------------
def tau_of_z(z, fv0):
    tau0 = F.tau0_tilde(fv0)
    f = lambda t: F.z_of_tau(t, fv0) - z
    # z decreases as tau increases; bracket (tiny, tau0)
    return brentq(f, 1e-9*tau0, tau0, xtol=1e-14, rtol=1e-12)

def fv_of_tau(tau, fv0):
    return 3*fv0*tau/(3*fv0*tau+(1-fv0)*(2+fv0))

def DM(z, fv0):
    tau0=F.tau0_tilde(fv0); tau=tau_of_z(z,fv0)
    return (1+z)*tau**(2/3)*(F.Ftilde(tau0,fv0)-F.Ftilde(tau,fv0))

def H_over_Hbar0(z, fv0):
    tau0=F.tau0_tilde(fv0); tau=tau_of_z(z,fv0); fv=fv_of_tau(tau,fv0)
    g=(4*fv**2+fv+4)/(2*(2+fv))
    return g*(2+fv)*tau0/((2+fv0)*tau)

def DH(z, fv0): return 1.0/H_over_Hbar0(z, fv0)
def DV(z, fv0):
    dM=DM(z,fv0); dH=DH(z,fv0); return (z*dM*dM*dH)**(1/3)
def g_dress(fv0): return (4*fv0**2+fv0+4)/(2*(2+fv0))

# ---------------- LCDM dimensionless distances (units c/H0) ----------------
def DM_L(z, Om):
    from scipy.integrate import quad
    OL=1-Om; E=lambda zz: np.sqrt(Om*(1+zz)**3+OL)
    return quad(lambda zz:1.0/E(zz),0,z)[0]
def DH_L(z, Om):
    OL=1-Om; return 1.0/np.sqrt(Om*(1+z)**3+OL)
def DV_L(z, Om):
    dM=DM_L(z,Om); dH=DH_L(z,Om); return (z*dM*dM*dH)**(1/3)

# ---------------- data: DESI DR1 (2404.03002) + Planck acoustic scale --------
# (z, kind, value, err[, corr with the paired DM/DH]); pairs share a 2x2 cov
BAO = [
 (0.295,"DV",7.93,0.15,None),
 (0.510,"DM",13.62,0.25,-0.445),(0.510,"DH",20.98,0.61,-0.445),
 (0.706,"DM",16.85,0.32,-0.420),(0.706,"DH",20.08,0.60,-0.420),
 (0.930,"DM",21.71,0.28,-0.389),(0.930,"DH",17.88,0.35,-0.389),
 (1.317,"DM",27.79,0.69,-0.444),(1.317,"DH",13.82,0.42,-0.444),
 (1.491,"DV",26.07,0.67,None),
 (2.330,"DM",39.71,0.94,-0.477),(2.330,"DH",8.52,0.17,-0.477),
]
# Planck 2018 acoustic scale -> D_M(z*)/r_d.  100 theta* = 1.04109(30);
# D_M(z*)/r_* = 1/theta* = 96.053; r_*/r_drag = 144.43/147.09 = 0.98192
# (single, consistent Planck 2018 TT,TE,EE+lowE+lensing column for both r_star
#  and r_drag -- previously this mixed the no-lensing r_star=144.39 with the
#  +lensing r_drag=147.09, an internally inconsistent pairing)
ZSTAR=1089.80; DM_zstar_over_rd=96.053*(144.43/147.09)   # = 94.316 -> 94.32
CMB=(ZSTAR,"DM",DM_zstar_over_rd,0.05)
RD_STD=147.09   # Planck drag sound horizon (Mpc), standard early physics

def build_data():
    rows=[];
    for z,k,v,e,c in BAO: rows.append((z,k,v,e,c))
    rows.append((CMB[0],CMB[1],CMB[2],CMB[3],None))
    return rows

def model_vec(fv0, rows, lcdm=None):
    out=[]
    for z,k,v,e,c in rows:
        if lcdm is None:
            d = DM(z,fv0) if k=="DM" else (DH(z,fv0) if k=="DH" else DV(z,fv0))
        else:
            d = DM_L(z,lcdm) if k=="DM" else (DH_L(z,lcdm) if k=="DH" else DV_L(z,lcdm))
        out.append(d)
    return np.array(out)

def build_cov(rows):
    n=len(rows); C=np.zeros((n,n))
    for i,(z,k,v,e,c) in enumerate(rows): C[i,i]=e*e
    # correlated DM/DH pairs (same z, both have corr c)
    for i in range(n):
        for j in range(i+1,n):
            zi,ki,_,ei,ci=rows[i]; zj,kj,_,ej,cj=rows[j]
            if zi==zj and {ki,kj}=={"DM","DH"} and ci is not None:
                C[i,j]=C[j,i]=ci*ei*ej
    return C

def fit(shape_param_grid, rows, Cinv, d, lcdm=False):
    best=None
    for p in shape_param_grid:
        g = model_vec(p, rows, lcdm=(p if lcdm else None))
        gCi=Cinv@g; a=(g@(Cinv@d))/(g@gCi); chi2=d@(Cinv@d)-(g@(Cinv@d))**2/(g@gCi)
        if best is None or chi2<best[2]: best=(p,a,chi2)
    return best  # (param, alpha, chi2)

rows=build_data(); d=np.array([r[2] for r in rows]); Cmat=build_cov(rows); Cinv=np.linalg.inv(Cmat)
dof=len(rows)-2

print("="*72); print("TIMESCAPE inverse-distance-ladder: DESI BAO + Planck acoustic scale")
print("="*72)
fvg=np.linspace(0.30,0.995,696)
fv0,a_ts,chi_ts=fit(fvg,rows,Cinv,d,lcdm=False)
Hbar0=C/(a_ts*RD_STD); H0_ts=g_dress(fv0)*Hbar0
print(f"TIMESCAPE best: fv0={fv0:.3f}  alpha=c/(Hbar0 rd)={a_ts:.2f}  chi2/dof={chi_ts:.1f}/{dof}={chi_ts/dof:.2f}")
print(f"  -> with rd={RD_STD} Mpc:  Hbar0={Hbar0:.1f}  dressed H0={H0_ts:.1f} km/s/Mpc")
# 1-sigma on fv0
chis=np.array([fit([p],rows,Cinv,d)[2] for p in fvg]);
il=np.argmin(chis); dch=chis-chis[il]
lo=np.interp(1,dch[:il+1][::-1],fvg[:il+1][::-1]) if il>0 else fvg[il]
hi=np.interp(1,dch[il:],fvg[il:]) if il<len(fvg)-1 else fvg[il]
print(f"  fv0 = {fv0:.3f} (+{hi-fv0:.3f}/-{fv0-lo:.3f})  [BAO+CMB]")

# consistency with SN-preferred fv0=0.85
g85=model_vec(0.85,rows); gCi=Cinv@g85; a85=(g85@(Cinv@d))/(g85@gCi)
chi85=d@(Cinv@d)-(g85@(Cinv@d))**2/(g85@gCi)
print(f"\nFix fv0=0.85 (SN-preferred): chi2/dof={chi85:.1f}/{len(rows)-1}={chi85/(len(rows)-1):.2f}  "
      f"H0={g_dress(0.85)*C/(a85*RD_STD):.1f}  -> "+("consistent" if chi85/(len(rows)-1)<2 else "POOR FIT"))

print("\n"+"-"*72)
omg=np.linspace(0.20,0.45,251)
Om,a_l,chi_l=fit(omg,rows,Cinv,d,lcdm=True)
H0_l=C/(a_l*RD_STD)
print(f"LCDM reference: Om0={Om:.3f}  chi2/dof={chi_l:.1f}/{dof}={chi_l/dof:.2f}  "
      f"H0={H0_l:.1f} km/s/Mpc (rd={RD_STD})")
print("="*72)
print("INTERPRETATION:")
print(f"  BAO+CMB prefer fv0={fv0:.3f}; SN prefer fv0~0.85.  "
      +("AGREE" if lo<=0.85<=hi else f"DISAGREE (SN 0.85 outside [{lo:.2f},{hi:.2f}])"))
print(f"  timescape chi2/dof={chi_ts/dof:.2f} vs LCDM {chi_l/dof:.2f}")
