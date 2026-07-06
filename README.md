# Free-history timescape

**Does a data-driven void history — rather than the one-parameter tracker — let timescape fit supernovae, BAO, and the CMB together?**

Author: **Viacheslav Zhygulin**

Reproduction code and paper (in progress) for *free-history timescape*: a generalisation of David Wiltshire's [timescape cosmology](https://en.wikipedia.org/wiki/Inhomogeneous_cosmology) in which the void volume-fraction history `f_v(z)` is **freed from the tracker attractor** and instead driven by the observed void population, then run through the same two-phase Buchert averaging and wall/void clock-rate ("dressing") mechanism.

> **Status:** working draft / active analysis. Companion to [*Can eliminating dark energy resolve the Hubble tension?*](https://github.com/szhygulin/timescape-hubble-tension) (paper 1), whose uniform SN+BAO+CMB harness this reuses. Analysis is AI-assisted (disclosed in the paper).

## Why this exists

Paper 1 found that standard timescape does **not** resolve the Hubble tension: its one-parameter *tracker* solution is forced to a present void fraction `f_v0 ≈ 0.85` by the supernovae while BAO+CMB demand `≈ 0.64` — a 4.8–6.6σ split. A model-independent check in that paper (a free smooth expansion history) showed the SN, BAO, and CMB data are **not** internally contradictory; it is specifically the *rigidity of the one-parameter tracker history* that fails.

Free-history timescape asks the natural follow-up: **is the timescape mechanism right but its tracker closure wrong?** Discard the attractor; let the void history be an arbitrary `f_v(z)` (later: the *observed* one), keep the dressing, and re-test.

## Result so far (the decision gate)

The general non-tracker dressed-geometry solver ([`src/probes/modelv_theory.py`](src/probes/modelv_theory.py)) is validated to reproduce the tracker limit exactly (SN χ² = 1391.545, distances to 6×10⁻⁸, and the non-FLRW `dD_M/dz ≠ D_H` signature; run [`modelv_gates.py`](src/probes/modelv_gates.py)).

**Probe R** — the required void history — fits a free `f_v(z)` jointly to Pantheon+ SNe, DESI BAO, and the Planck acoustic scale:

| Model | joint χ² (same data) |
|---|---|
| free `E(z)` spline (model-independent) | 1391.85 |
| **free-history timescape (free `f_v(z)`)** | **1396.06** |
| flat ΛCDM | 1402.24 |
| timescape **tracker** (1 parameter) | 1469.29 |

A free void history reconciles the three datasets **below ΛCDM** and far below the tracker, at a **physically plausible present void fraction `f_v(0) = 0.640`** (band [0.638, 0.641]) — matching the BAO+CMB value, not the tracker's runaway 0.85. So the mechanism is flexible; the tracker parametrisation was the bottleneck.

**Important caveat (why this is a *kinematic* result).** This forced-`f_v` history is not yet a dynamically self-consistent Buchert solution — it violates the integrability condition (the void curvature parameter varies ~81%, versus constant to 4×10⁻⁷ on the tracker). It is *"the backreaction the Hubble diagram wants,"* a well-defined target — not a proven GR solution. Establishing the dynamically-consistent version is part of this paper (see plan).

## Plan (this paper)

See [`PLAN_void_history.md`](PLAN_void_history.md) and [`NOTES_modelv_theory.md`](NOTES_modelv_theory.md).

1. **Probe R** — required `f_v(z)` (done).
2. **Phase D** — the *observed* void history `f_v_obs(z)` from public void catalogs (2M++, SDSS, BOSS/eBOSS).
3. **R2** — required vs available: does the observed void population supply `f_v_req(z)`?
4. **Phase F** — repeat paper 1's SN covariance ladder, DESI DR2 BAO+CMB, joint fit, and calibrator-anchored H₀ under free-history timescape.
5. **Reading B** — the dynamically-consistent solve: enforce Buchert integrability, re-derive the lapse, and refit, so the reconciling history is a genuine GR solution.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python probes/modelv_gates.py        # tracker-limit + non-FLRW gates (must PASS)
python probes/modelv_probeR.py       # required void history f_v(z) -> ../probes_out/modelV_probeR.json
python probes/verify_modelv_probeR.py # independent re-derivation of the headline
```

Scripts read `src/data/` and write to `probes_out/`; run them from inside `src/`.

## Data

Supernovae: public **Pantheon+** release (Scolnic et al. 2022; Brout et al. 2022), redistributed under `src/data/`. BAO: **DESI** DR1/DR2. CMB: **Planck 2018** acoustic scale. Void catalogs (Phase D) are fetched by committed scripts into a git-ignored `external_data/`, not redistributed.

## License

Code, paper, and figures released under the [MIT License](LICENSE) © 2026 Viacheslav Zhygulin. Bundled Pantheon+ data remains under its original public-release terms.
