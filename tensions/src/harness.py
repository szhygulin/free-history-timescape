"""
SHARED HARNESS for fitting any cosmological model to SN + BAO + CMB on the
SAME footing as the timescape analysis. Reuse these; implement ONLY your
model's distance functions.

INTERFACES
----------
load_sn() -> (zHD, zHEL, mb, Cf)            Pantheon+ cosmology sample (1580 SNe)
sn_chi2(Dc) -> chi2                          Dc = comoving distance to each SN,
                                             array in units c/H0 (abs scale /
                                             M_B offset marginalised internally).
bao_cmb_rows() -> list of (z, kind, value, err, corr)   DESI DR1 + Planck theta*
bao_cmb_chi2(predict) -> (chi2, alpha)       predict(z,kind)->dimensionless
                                             distance (units c/H0); kind in
                                             {"DM","DH","DV"}; alpha=c/(H0 rd)
                                             marginalised. rd=147.09 Mpc.
bao_only_chi2(predict) -> (chi2, alpha)      same but drops the CMB point.
H0_from_alpha(alpha) -> H0 [km/s/Mpc]
joint = sn_chi2(Dc) + bao_cmb_chi2(predict)[0]

CONVENTIONS
-----------
DM = transverse comoving distance (units c/H0);  DH = c/H(z) (units c/H0,
i.e. 1/E(z) for FLRW);  DV = (z DM^2 DH)^(1/3).  For SN, Dc is the line-of-sight
comoving distance to zHD (=DM for flat); the harness forms mu=5log10((1+zHEL)Dc).

A WORKED EXAMPLE (flat LCDM) is provided: copy lcdm_Dc / lcdm_predict for your
model. Reference timescape lives in timescape_baocmb.py (import it).
"""
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid as cumtrap
import fit_timescape as F, timescape_baocmb as T

C_KM = 299792.458
RD   = 147.09  # Planck drag sound horizon (Mpc), standard early physics

_zHD,_zHEL,_mb,_Cf = F.load()
_SN = F.make_chi2(_zHD,_zHEL,_mb,_Cf)
def load_sn(): return _zHD,_zHEL,_mb,_Cf
def sn_chi2(Dc): return _SN(np.asarray(Dc,dtype=float))

# BAO+CMB data (DESI DR1 + Planck acoustic scale, correct error)
_DMz=(100.0/1.04109)*(144.43/147.09); _SIG=max(_DMz*(0.00030/1.04109),0.05)  # Planck 2018 +lensing column (consistent r_star/r_drag; was mixed 144.39/147.09)
ROWS=[(z,k,v,e,c) for (z,k,v,e,c) in T.BAO]+[(1089.80,"DM",_DMz,_SIG,None)]
_DV=np.array([r[2] for r in ROWS]); _CINV=np.linalg.inv(T.build_cov(ROWS))
def bao_cmb_rows(): return list(ROWS)
def bao_cmb_chi2(predict):
    g=np.array([predict(z,k) for z,k,_,_,_ in ROWS]); gCi=_CINV@g
    a=(g@(_CINV@_DV))/(g@gCi); chi=_DV@(_CINV@_DV)-(g@(_CINV@_DV))**2/(g@gCi); return chi,a
def bao_only_chi2(predict):
    r=ROWS[:-1]; dv=np.array([x[2] for x in r]); ci=np.linalg.inv(T.build_cov(r))
    g=np.array([predict(z,k) for z,k,_,_,_ in r]); gCi=ci@g
    a=(g@(ci@dv))/(g@gCi); return dv@(ci@dv)-(g@(ci@dv))**2/(g@gCi), a
def H0_from_alpha(a): return C_KM/(a*RD)

# ----------------- worked example: flat LCDM -----------------
def _invE_lcdm(z,Om): return 1.0/np.sqrt(Om*(1+z)**3+1-Om)
def lcdm_Dc(z,Om):
    zg=np.linspace(0,np.max(z)*1.0001,200000); Dc=cumtrap(_invE_lcdm(zg,Om),zg,initial=0)
    return np.interp(z,zg,Dc)
def lcdm_predict(Om):
    def p(z,k):
        if k=="DH": return _invE_lcdm(z,Om)
        dM=quad(lambda zz:_invE_lcdm(zz,Om),0,z)[0]
        if k=="DM": return dM
        return (z*dM*dM*_invE_lcdm(z,Om))**(1/3)
    return p

if __name__=="__main__":
    from scipy.optimize import minimize_scalar
    def joint(Om): return sn_chi2(lcdm_Dc(_zHD,Om))+bao_cmb_chi2(lcdm_predict(Om))[0]
    r=minimize_scalar(joint,bounds=(0.15,0.45),method="bounded")
    _,a=bao_cmb_chi2(lcdm_predict(r.x))
    print(f"[harness self-test] LCDM joint: Om={r.x:.3f} chi2={r.fun:.1f} H0={H0_from_alpha(a):.1f}")
    print("  expect Om~0.305, chi2~1402, H0~68.6")
