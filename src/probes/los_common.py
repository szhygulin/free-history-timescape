#!/usr/bin/env python3
"""Phase 3 (LOS voids) — shared geometry for Probes 1-5.

Coordinate transforms and density-field sampling used by every LOS probe, validated in
Probe 0 (`probe0_sample.py`: NGP/Galactic-Centre + Virgo/Coma/Local-Void checks). Kept in
one module so the probes cannot drift apart on the field-reading convention.

Field: 2M++/Carrick+2015 reconstruction, external_data/twompp_density.npy, 257^3 float64,
Galactic Cartesian comoving Mpc/h, cell centres -200..+200 (spacing 1.5625), LG at
[128,128,128], value = delta_g* (luminosity-weighted contrast, real space, 4 Mpc/h smoothing).
"""
import os
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import map_coordinates

WT = "/Users/s/dev/science/free-history-timescape"
FIELD = os.path.join(WT, "external_data", "twompp_density.npy")
DATA = os.path.join(WT, "src", "data", "PantheonSH0ES.dat")

C_KM = 299792.458
OM_FID = 0.315
H100 = 100.0
NGRID = 257
LMIN, LMAX = -200.0, 200.0
DX = (LMAX - LMIN) / (NGRID - 1)   # 1.5625 Mpc/h
R_RELIABLE = 200.0

# J2000 equatorial -> Galactic rotation (Hipparcos), validated in Probe 0.
R_EQ2GAL = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [+0.4941094279, -0.4448296300, +0.7469822445],
    [-0.8676661490, -0.1980763734, +0.4559837762],
])


def eq_to_gal_lb(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    r_eq = np.array([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)])
    xg, yg, zg = R_EQ2GAL @ r_eq
    l = np.degrees(np.arctan2(yg, xg)) % 360.0
    b = np.degrees(np.arcsin(np.clip(zg, -1, 1)))
    return l, b


def unit_gal(l_deg, b_deg):
    """Unit direction vector(s) in Galactic Cartesian from (l, b) in degrees."""
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def gal_lbr_to_xyz(l_deg, b_deg, r):
    u = unit_gal(l_deg, b_deg)
    return r * u[0], r * u[1], r * u[2]


def comoving_dist_mpc_h(z, om=OM_FID, n=200000):
    zg = np.linspace(0.0, float(np.max(z)) * 1.0001 + 1e-6, n)
    invE = 1.0 / np.sqrt(om * (1 + zg) ** 3 + (1 - om))
    chi = (C_KM / H100) * cumulative_trapezoid(invE, zg, initial=0.0)
    return np.interp(z, zg, chi)


def load_field():
    return np.load(FIELD)


def _to_index(X, Y, Z):
    return (X - LMIN) / DX, (Y - LMIN) / DX, (Z - LMIN) / DX


def sample_points(field, X, Y, Z):
    """Trilinear (order-1) density contrast at Galactic Cartesian points (Mpc/h)."""
    ii, jj, kk = _to_index(np.asarray(X), np.asarray(Y), np.asarray(Z))
    return map_coordinates(field, [ii, jj, kk], order=1, mode="nearest")


def sample_ray(field, l_deg, b_deg, r_end, ds=0.5):
    """Density contrast sampled along observer->SN sightline, from s=ds to r_end.

    Returns (s, delta) with uniform step ds (so path-length weighting == sample count).
    """
    n = max(2, int(np.ceil(r_end / ds)))
    s = np.linspace(ds, r_end, n)
    ux, uy, uz = unit_gal(l_deg, b_deg)
    d = sample_points(field, s * ux, s * uy, s * uz)
    return s, d


# spherical top-hat stencil (grid offsets within R_loc of a point), for delta_loc
def sphere_stencil(R_loc):
    m = int(np.ceil(R_loc / DX))
    g = np.arange(-m, m + 1) * DX
    OX, OY, OZ = np.meshgrid(g, g, g, indexing="ij")
    keep = (OX ** 2 + OY ** 2 + OZ ** 2) <= R_loc ** 2
    return OX[keep], OY[keep], OZ[keep]


def delta_local(field, X0, Y0, Z0, R_loc=8.0):
    """Spherical top-hat mean contrast within R_loc of each endpoint (the SN monopole)."""
    ox, oy, oz = sphere_stencil(R_loc)
    X0 = np.atleast_1d(X0); Y0 = np.atleast_1d(Y0); Z0 = np.atleast_1d(Z0)
    out = np.empty(len(X0))
    for i in range(len(X0)):
        out[i] = sample_points(field, X0[i] + ox, Y0[i] + oy, Z0[i] + oz).mean()
    return out


# ----------------------------------------------------------------------
# Catalog + sample definition (deterministic mask over the full 1701 rows), so every
# probe subsets the covariance identically and LOS covariates align by catalog index.
# ----------------------------------------------------------------------
COV = os.path.join(WT, "src", "data", "PantheonSH0ES_STATSYS.cov")
_CAT_COLS = ["zHD", "zHDERR", "zCMB", "zHEL", "m_b_corr", "m_b_corr_err_DIAG",
             "x1", "c", "mB", "RA", "DEC", "VPEC", "VPECERR", "HOST_LOGMASS", "IDSURVEY"]


def load_catalog():
    """Full 1701-row Pantheon+ catalog: numeric columns + CID + IS_CALIBRATOR."""
    with open(DATA) as f:
        header = f.readline().split()
        idx = {n: i for i, n in enumerate(header)}
        rows = [ln.split() for ln in f]
    cat = {n: np.array([float(r[idx[n]]) for r in rows]) for n in _CAT_COLS}
    cat["CID"] = np.array([r[idx["CID"]] for r in rows])
    cat["IS_CALIBRATOR"] = np.array([int(float(r[idx["IS_CALIBRATOR"]])) for r in rows])
    # Galactic direction + fiducial comoving distance for every row (zCMB placement)
    cat["l"], cat["b"] = eq_to_gal_lb(cat["RA"], cat["DEC"])
    cat["r"] = comoving_dist_mpc_h(np.maximum(cat["zCMB"], 0.0))
    return cat


def usable_mask(cat):
    """Probe-0 usable sample: non-calibrator, zHD>0.01, inside the reliable sphere."""
    return (cat["IS_CALIBRATOR"] == 0) & (cat["zHD"] > 0.01) & (cat["r"] < R_RELIABLE)


def load_cov_subset(mask):
    """Subset the full 1701^2 STAT+SYS covariance to the masked rows."""
    with open(COV) as f:
        n = int(f.readline())
    C = np.loadtxt(COV, skiprows=1).reshape(n, n)
    return C[np.ix_(mask, mask)]
