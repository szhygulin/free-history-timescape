#!/usr/bin/env python3
"""One-shot: patch the structural_diagnosis note in threephase_dynamics.json to match the
corrected wording in fit_threephase_structure.py (the model DOES reproduce the rising
shallow, but only in the degenerate alpha_s2->0 flat-dust limit = two-phase tracker), so the
committed artifact and its generator agree without re-running the 262 s DE fit."""
import json
import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
P = os.path.join(_REPO, "probes_out", "threephase_dynamics.json")

d = json.load(open(P))
sd = d["structural_diagnosis"]
sd.pop("note", None)
sd["how_it_tracks"] = (
    "The measured shallow fraction RISES with z. A GENUINE matter-differentiated three-phase "
    "split (both phases actual under-dense voids, alpha_j^2 within [0,ceiling]) reproduces at "
    "most df_s=+0.055 (shape scan) and FAILS. The fit tracks ONLY in the DEGENERATE limit "
    "alpha_s2->0: the shallow phase becomes spatially FLAT (Om_s=H_w0^2), dynamically "
    "indistinguishable from the walls, so f_s rises as the empty deep void dilutes the volume "
    "at low z. This is the homothetic ceiling (NOTES sec 5): the tracking solution collapses "
    "the three-phase model to the two-phase empty-void tracker with a single genuine void of "
    "fraction f_d0~0.27. Tracking is therefore real but PHYSICALLY DEGENERATE, and it pins the "
    "effective void fraction to f_d0 (see threephase_forced_geometry.json)."
)
sd["genuine_threephase_max_df_shallow_rise"] = 0.055
json.dump(d, open(P, "w"), indent=2)
print(f"patched {P}")
