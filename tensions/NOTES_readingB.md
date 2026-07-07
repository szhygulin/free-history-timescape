# NOTES — Reading B (WP-B): dynamical-consistency / Buchert-integrability close-out

**Status: THEORY-LEVEL CLOSE-OUT — a result, not a failure.**

Work package B asked whether the KINEMATIC forced-`f_v` machinery behind Probe R (paper 2)
can be upgraded to a DYNAMICALLY-consistent Buchert two-scale solution — i.e. whether the
required (and separately, the observed) void history is a genuine GR backreaction fluid, not
just "what the Hubble diagram wants." It cannot, at the single-phase two-scale level. This memo
records the close-out and the committed generator that now backs the paper-2 §8 caveat.

## What was re-homed

The Buchert-integrability check previously existed only as a quoted pair of numbers in
paper-2 `NOTES_modelv_theory.md` §8, produced by a since-deleted prototype `proto_modelv4.py`
with no committed generator. It is now re-homed as:

- generator: [`src/probes/wpb_integrability.py`](src/probes/wpb_integrability.py)
- output: [`probes_out/wpb_integrability.json`](probes_out/wpb_integrability.json)

The probe imports the paper-2 production solver (`modelv_theory.py`: `fv_from_nodes`,
`tracker_fv_of_z`, `modelv_solve`) read-only and writes nothing in the paper-2 tree.

## The invariant and the transcription

DNW13 (arXiv:1306.3208) Eq 11 gives a void-curvature invariant `α² = −k_v f_vi^{2/3}` that
MUST be constant in bare time `τ` for a genuine single-phase Buchert two-scale solution.
Transcribed from `NOTES_modelv_theory.md ~line 262` (copied faithfully; verified below):

```
α²(τ) = ( 2 ā² / (3 f_v^{1/3} (1−f_v)) )
        · [ f_v″ + f_v′²(2f_v−1)/(2f_v(1−f_v)) + 3 (ā′/ā) f_v′ ]
```

primes = `d/dτ`; `ā = τ^{2/3}(1−f_v)^{-1/3}` is the bare volume-average scale factor, so
`ā′/ā = 2/(3τ) + (1/3)f_v′/(1−f_v)` is the bare Hubble rate. **The equation form is
unambiguous** as transcribed; the two latent reading choices (primes in BARE time; `ā` the
volume-average, not the wall `a_w=τ^{2/3}`) are both pinned by the tracker check below.

`α²` carries `ā²`, so its absolute scale is a normalisation convention (`ā(τ₀)=1` here); the
reported metric, the fractional spread `max|α²|/min|α²|−1`, is convention-invariant.

## Result — three histories through the paper-2 solver (algebraic lapse)

| history | source | `α²` fractional spread |
|---|---|---|
| (i) **tracker** `f_v` | `tracker_fv_of_z` | **2.2e-6** (constant to the pipeline floor) |
| (ii) **required** forced `f_v` | `modelV_probeR.json V.fv_nodes` | **~1.5 interior / 15.5 span** |
| (iii) **observed** `f_v_obs` | `telescope_fvobs.json` PRIMARY, `Φ(σ₀D(z)/2)` | **~2.9 interior / 5.4 span** |

**Transcription validated.** The exact closed-form tracker `f_v(τ)` fed through the equation
with analytic derivatives gives `α²` constant to **2.4e-15** (machine precision) — the copy is
faithful and the invariant is genuinely constant on the tracker. Driven through the full
solver + `np.gradient` pipeline the tracker holds to **2.2e-6** (paper-2 §8 quoted `3.9e-7`
from the deleted prototype; same order — pipeline noise floor, not physics). This also
cross-validates the kinematic closure (C1).

**No consistent single-phase two-scale (constant-`α²`) model tracks the required or the
observed history.** Both drift by `O(100 %)–O(1000 %)` — 6+ orders of magnitude above the
tracker floor — so the forced/observed `(Q, ⟨R⟩)` is not a consistent GR void fluid. The
forced-history magnitude is representation-sensitive (5 fitted nodes under-determine the
second derivative `f_v″`; the solver's monotone PCHIP is only C¹ and its `f_v″` jumps at
nodes — reported as `pchip_literal ≈ 21.5` for contrast; the physical C² cubic-spline reading
gives interior `~1.5`). The observed history is representation-ROBUST (its analytic form
`Φ(σ₀D(z)/2)` is smooth: analytic, C²-node, and PCHIP readings all agree at `~5`). The
paper-2 §8 "81 %" figure was the prototype's illustrative smooth forced history, NOT the final
Probe-R `V.fv_nodes`; the actual required history is a stronger deformation and drifts more.
The **qualitative verdict is representation-independent**: `≫` tracker, grossly non-constant.

## Why the violation cannot be fixed by initial conditions

The tracker is the late-time ATTRACTOR of the Buchert two-scale system. A genuine constant-`α²`
solution launched from any generic initial condition flows TOWARD the tracker `f_v(z)` — which
is not the required (nor the observed) history. So the integrability violation cannot be
patched by re-choosing initial data: **the tracker attractor blocks IC fixes.** Enforcing
integrability (K5) / DNW13 Eqs 10–11 constrains `f_v` back onto (a neighbourhood of) the
tracker, precisely the amplitude-dead one-parameter history Probe R already rejected.

## What is deferred, and what is not built

- **Deferred (not attempted here):** the fully dynamical reading — impose (K5)/DNW13 Eqs
  10–11 so `f_v` is dynamically constrained, with the lapse re-derived from **DNW13 Eq 22**
  (dynamical lapse) rather than the adopted algebraic `γ̄=(2+f_v)/2`. This is the deeper
  follow-up if Probe R ever reconciles; it is out of scope for the close-out.
- **Not built (and it need not be): the multi-phase / multi-scale void alternative.** A
  many-`α²`-phase construction could in principle bend `f_v(z)` further, but its falsifier has
  **already fired independently** at the observational layer:
  - paper-2 `telescope_fvobs.json` verdict **SHAPE-UNAVAILABLE** (below-mean floor theorem:
    `Φ(σ/2) ≥ 0.5` for any real field, so the required `f_v^req(0.7)=0.396 < 0.5` is
    definitionally unreachable — a `~52 %` gap at `z=0.7`, `R_s`-independent);
  - the **level ⊥ shape near-theorem**, paper-2 [`NOTES_mapping.md` §3](../free-history-timescape/NOTES_mapping.md):
    no single-threshold void population of a field whose fluctuations shrink with `z` can
    occupy the required `(f_v(0)=0.64, ×3.3 decline)` corner.

  Building a multi-phase model to match the required history would only re-derive a backreaction
  the observed void population cannot supply. The observational falsifier caps the exercise; no
  amount of theory-side phase structure recovers a history the data have already excluded.

## Mapping sub-item — already discharged, not rebuilt

The Reading-B mapping sub-item (expansion-excess margin ↔ density threshold, and the
level-vs-shape structural result) is fully discharged in paper-2
[`NOTES_mapping.md`](../free-history-timescape/NOTES_mapping.md) — §2 (the derived mapping),
§3 (the level ⊥ shape near-theorem + ΛCDM-growth forecast), §5 (the pre-registered
SHAPE-UNAVAILABLE test). **Cited, not rebuilt here.**

## Verdict

WP-B closes at the theory level as a **result**: the DNW13 Eq-11 integrability invariant is
validated on the tracker (constant to `2e-15` analytically) and is grossly non-constant along
both the required and the observed histories, so neither is a consistent single-phase Buchert
two-scale solution; the tracker attractor forbids an initial-condition escape; the dynamical
Eq-22 lapse derivation is a deferred follow-up; and the multi-phase alternative is left
unbuilt because its observational falsifier (SHAPE-UNAVAILABLE + level ⊥ shape) has already
fired. Probe R's forced-`f_v` outputs remain the correct KINEMATIC reading — "what
backreaction the Hubble diagram wants," to be compared against observed voids — and are not to
be silently upgraded to a proven dynamical solution.
