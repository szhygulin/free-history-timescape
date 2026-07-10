# Free-history timescape

**A data-driven void history reconciles the supernova–BAO–CMB geometry but does not resolve the Hubble tension.**

Author: **Viacheslav Zhygulin**

Reproduction code and manuscript for *free-history timescape*: a generalisation of David Wiltshire's [timescape cosmology](https://en.wikipedia.org/wiki/Inhomogeneous_cosmology) in which the void volume-fraction history `f_v(z)` is **freed from the one-parameter tracker attractor**, kept inside the same two-phase Buchert averaging and wall/void clock-rate ("dressing") mechanism, and then tested end to end — first against the distance geometry, then against every deeper physical requirement the reconciling history must satisfy.

The manuscript is [`free-history-timescape.tex`](free-history-timescape.tex) (REVTeX). Companion to [*Can eliminating dark energy resolve the Hubble tension?*](https://github.com/szhygulin/timescape-hubble-tension) (paper 1), whose uniform SN+BAO+CMB harness this reuses. Analysis is AI-assisted (disclosed in the paper).

> **This repository combines two phases of the program into one paper.** Part I (geometry reconciliation) lives at the repo root; Part II (the tension tests, formerly the separate `free-history-timescape-tensions` repo) lives under [`tensions/`](tensions/). That repository is now **archived read-only** on GitHub as the per-commit history of record.

## The question

Paper 1 found that standard timescape does **not** resolve the Hubble tension: its one-parameter *tracker* history is forced to a present void fraction `f_v0 ≈ 0.85` by the supernovae while BAO+CMB demand `≈ 0.64` — a 4.8–6.6σ split. A model-independent free-`E(z)` check there showed the SN, BAO, and CMB data are **not** internally contradictory; it is the *rigidity of the one-parameter tracker* that fails. So: **is the timescape mechanism right but its tracker closure wrong?** Discard the attractor; let `f_v(z)` be free; re-test.

## Part I — the geometry reconciles (kinematic)

The general non-tracker dressed-geometry solver ([`src/probes/modelv_theory.py`](src/probes/modelv_theory.py)) reproduces the tracker limit exactly (SN χ² = 1391.545 and the non-FLRW `dD_M/dz ≠ D_H` signature; run [`modelv_gates.py`](src/probes/modelv_gates.py)). A **free `f_v(z)`** then fits Pantheon+ SNe, DESI DR2 BAO, and the Planck acoustic scale jointly:

| Model | joint χ² (same data) |
|---|---|
| free `E(z)` spline (model-independent) | 1391.85 |
| **free-history timescape (free `f_v(z)`)** | **1396.06** |
| flat ΛCDM | 1402.24 |
| timescape **tracker** (1 parameter) | 1469.29 |

A free void history reconciles the three datasets **below ΛCDM** and far below the tracker, at a **physically plausible `f_v(0) = 0.640`** — the BAO+CMB value, not the tracker's runaway 0.85 — and the amplitude split that defeated the tracker dissolves from 6.5σ to 0.16σ. The mechanism is flexible; the tracker parametrisation was the bottleneck.

**But this is a *kinematic* reconciliation:** the forced `f_v(z)` violates the Buchert integrability condition. It is *"the backreaction the Hubble diagram wants"* — a well-defined target, not a proven GR solution.

## Part II — the reconciliation does not survive

Four independent tests ask whether the kinematic history is physically **available**, dynamically **consistent**, and early-Universe **calibratable**. All four fail (code under [`tensions/`](tensions/)):

| Test | Result |
|---|---|
| **Availability** (observed void population vs required) | **SHAPE-UNAVAILABLE** — required decline ×3.31 vs observed ×1.16 (floor theorem) |
| **Local bias** `b_pred` (SH0ES-ladder H₀ excess) | **PARTIAL** (pre-registered, 1.57σ) **/ FAILS** (one-sided envelope + catalog-forced) — predicts +2.4% vs the +8.4% required; 4.16σ measurement-only (systematic-excluded) |
| **Dynamical consistency** (three-phase Buchert solve) | **CLOSED** — collapses to the two-phase tracker; k=0 geometry misses the BIC bar by 4953 |
| **Sound horizon** `r_d` (in-model early-Universe) | **FAILS** — r_d = 199.6 Mpc (+36%), driving the bare H̄₀ to ~40 km/s/Mpc |

**Reconciliation without resolution.** Removing the tracker's rigidity reconciles the distance geometry, but the flexibility is kinematic and closes at every physical test — corroborating and sharpening paper 1.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Part I (geometry reconciliation) — from the repo root:
cd src
python probes/modelv_gates.py         # tracker-limit + non-FLRW gates (must PASS)
python probes/modelv_probeR.py        # required void history f_v(z) -> ../probes_out/modelV_probeR.json
python probes/verify_modelv_probeR.py # independent re-derivation of the headline

# Part II (the tension tests) — from tensions/:
cd ../tensions/src
python probes/verify_bpred_survey.py      # b_pred survey-averaged -> SURVIVES
python probes/verify_wpb_integrability.py # three-phase / integrability close-out
```

One number → one committed script → one result JSON, against pre-registered thresholds. Part-I scripts read `src/data/` and write `probes_out/`; Part-II scripts under `tensions/` are self-contained (their own `src/data/`) and read the Part-I artifacts they depend on from the repo root. Void-catalog density fields (Phase D / telescope) are fetched by committed scripts into a git-ignored `external_data/`, not redistributed.

## Data

Supernovae: public **Pantheon+** release (Scolnic et al. 2022; Brout et al. 2022), redistributed under `src/data/`. BAO: **DESI DR2**. CMB: **Planck 2018** acoustic scale. Observed voids: **BOSS DR12** (Mao et al. 2017) and the **2M++/Carrick+2015** field, fetched by committed scripts, not redistributed.

## License

Code, manuscript, and figures released under the [MIT License](LICENSE) © 2026 Viacheslav Zhygulin. Bundled Pantheon+ data remains under its original public-release terms.
