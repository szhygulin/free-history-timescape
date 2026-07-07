#!/usr/bin/env python3
"""Phase 3 (LOS voids), Probe 0 — fetch the external 2M++ / Carrick+2015 density field.

Downloads the reconstructed 2M++ density field (and, optionally, the velocity field)
from the Cosmic Flows site into ``external_data/`` (git-ignored — the file is 136 MB and
regenerable, so it is not redistributed with the repo).

Source: Carrick, Turnbull, Lavaux & Hudson 2015, MNRAS 450, 317 (arXiv:1504.04627),
        distributed at https://cosmicflows.iap.fr/download/ .

Field format (from twompp_README.txt, mirrored below):
  * shape (257, 257, 257) float64, indexed [i, j, k], Galactic Cartesian comoving Mpc/h;
  * cell centres run -200 .. +200 Mpc/h, spacing 400/256 = 1.5625 Mpc/h;
  * X = (i-128)*400/256, Y = (j-128)*400/256, Z = (k-128)*400/256;
  * Local Group at the central voxel [128, 128, 128];
  * value is delta_g* — the luminosity-weighted galaxy density contrast, real space,
    Gaussian-smoothed at 4 Mpc/h; reconstruction reliable to ~200 Mpc/h (z ~ 0.067).

Idempotent: skips a file already present at the correct size. Run:
    python src/probes/fetch_twompp.py            # density + README (default)
    python src/probes/fetch_twompp.py --velocity # also the 407 MB velocity field
"""
import os
import sys
import subprocess

WT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "timescape-hubble-tension")
DEST = os.path.join(WT, "external_data")
BASE = "https://cosmicflows.iap.fr/assets/data"

# name -> expected Content-Length in bytes (from the server, 2026-07-06)
FILES = {
    "twompp_README.txt": 1549,
    "twompp_density.npy": 135796824,
}
VELOCITY = ("twompp_velocity.npy", 407390328)


def fetch(name, expected):
    out = os.path.join(DEST, name)
    if os.path.exists(out) and os.path.getsize(out) == expected:
        print(f"  ok (cached, {expected} bytes)   {name}")
        return
    url = f"{BASE}/{name}"
    print(f"  downloading {name} <- {url}")
    # -C - resumes a partial file; --fail turns HTTP errors into non-zero exit.
    subprocess.run(
        ["curl", "-fSL", "-C", "-", "--retry", "3", "-o", out, url], check=True
    )
    got = os.path.getsize(out)
    if got != expected:
        raise SystemExit(f"SIZE MISMATCH {name}: got {got}, expected {expected}")
    print(f"  ok ({got} bytes)   {name}")


def main():
    os.makedirs(DEST, exist_ok=True)
    for name, sz in FILES.items():
        fetch(name, sz)
    if "--velocity" in sys.argv:
        fetch(*VELOCITY)
    print(f"\nDensity field ready at {os.path.join(DEST, 'twompp_density.npy')}")


if __name__ == "__main__":
    main()
