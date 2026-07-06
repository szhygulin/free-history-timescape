# NOTES — the f_v ↔ observable mapping (pre-registration memo, R2-final gate)

*Paper 2, `significance-audit`. Standalone pre-registration of the mapping between timescape's
volume fraction `f_v` and a survey-measurable void statistic, and of the two-part R2-final test
that mapping governs. Written before any BOSS/eBOSS/DESI void data are reduced (roadmap §5 item 1).
This document is part of the pre-registration record: the telescope data reduction (roadmap §5
item 2) must not begin until this memo is reviewed. Every equation is sourced; every threshold and
statistic is fixed here, in advance. KINEMATIC-reading caveat rides on `f_v^req` throughout
(NOTES_modelv_theory.md §8): the required history violates Buchert integrability and is "what
backreaction the Hubble diagram wants," not a proven dynamical solution.*

Inputs it reconciles: `REASONING_AND_ROADMAP.md` §4 (the below-mean floor 4a, the level-vs-shape
tension 4b, the verdict tree §6); `NOTES_modelv_theory.md` §§1–3 (identities, lapse readings);
committed artifacts `probes_out/modelV_probeR.json` (required history), `probes_out/phaseD_fvobs.json`
(z≈0 field), `probes_out/R2.json` (current MARGINAL verdict). Quantitative forecast that backs the
numbers below: `src/probes/mapping_decline_forecast.py` → `probes_out/mapping_decline_forecast.json`.

---

## 1. What must be mapped, and why a fixed density threshold cannot do it

**Timescape's `f_v`.** The two-phase partition defines `f_v` as the volume fraction of regions
*expanding faster than the volume-average expansion rate* `H̄` (Wiltshire 2009; NOTES_modelv_theory.md
§0, K1: `H̄ = f_w H_w + f_v H_v`). It is an **expansion-rate** fraction, not a density fraction. The
observable is a **density** field. The mapping is the bridge between the two, and it is the bridge —
not the partition — that carries the z-dependence.

**The bridge (linear velocity–density identity).** In linear perturbation theory the peculiar
velocity divergence obeys `∇·v = −f H δ` (physical divergence of the physical peculiar velocity;
continuity + `δ ∝ D(t)`, so `δ̇ = f H δ`), where `f(z) ≡ d ln D/d ln a ≈ Ω_m(z)^0.55` is the linear
growth rate (Peebles 1980 §14; Linder 2005 for the `0.55` index). The local volume-expansion rate is
therefore

- **(M1)**  `H_local(x)/H̄ − 1 = −(f(z)/3) · δ(x)`.

A region expands faster than the volume-average (`H_local > H̄`) iff `δ < 0`. This is the identity
Phase D already used to call the below-mean set `{δ<0}` the "faster-than-mean-expanding" set and hence
the operational `f_v` proxy (`phaseD_voidfrac.py` docstring; `R2.json` `overlap_at_z0.note`). **(M1) is
bias-independent at the sign**: `δ_g < 0 ⇔ δ_m < 0` for any monotone bias through the origin, so
`P(δ<0)` is insensitive to galaxy-vs-matter bias — the one clean, definition-free anchor we have.

**The below-mean floor (roadmap 4a, derived).** Gravitational collapse concentrates mass, so the
volume PDF of `δ` is right-skewed at every epoch (mode below the mean; lognormal is the standard model,
Coles & Jones 1991). Hence

- **(M2)**  `f_v^{bm}(z) ≡ P(δ(z) < 0) ≥ 0.5 for all z,  → 0.5 as the field Gaussianises (σ→0) at high z.`

This is structural, not observational, and it holds for **any** growth history (right-skewness is
guaranteed by collapse, independent of the growth *rate*). But `f_v^req` crosses `0.5` at `z≈0.4`,
reaches `0.396` at `z=0.7`, and any timescape history has `f_v → 0` as `z → ∞` (voids form late). So:

> **No fixed density threshold — least of all the below-mean `δ<0` threshold — can supply `f_v^req(z)`
> beyond `z≈0.4`. A `δ<0` mapping is pinned at/above `0.5` and → `0.5`, never → 0.**

The mapping must therefore be **z-dependent**, and the void-defining margin must be one that → 0 early.
This memo derives that margin from the partition definition (M1), not by per-z fitting.

---

## 2. The derived mapping: a fixed expansion-excess margin → an evolving density threshold

The physically invariant object in the timescape partition is the **expansion excess**, not a density
percentile (the excess is what sources the backreaction `Q = 6 f_v(1−f_v)(H_v−H_w)²`, K3). Define the
void phase by a fixed fractional expansion-excess margin `ε ≥ 0`:

- **(M3)**  `void ≡ { x : (H_local(x) − H̄)/H̄ ≥ ε }`.

Insert the bridge (M1). The margin in expansion space maps to a **redshift-dependent density threshold**:

- **(M4)**  `void ≡ { x : δ(z) ≤ δ_th(z) },  δ_th(z) = − 3ε / f(z).`

The mapped volume fraction, for a smoothed field with volume PDF of variance `σ²(z) = σ_0² D(z)²`
(single comoving smoothing scale `R_s`; lognormal `1+δ = exp(G − s²/2)`, `G ~ N(0,s²)`, following
Phase D):

- **(M5)**  `f_v^{map}(z; ε) = P(δ(z) < δ_th(z)) = Φ[ ( ln(1+δ_th(z)) + s(z)²/2 ) / s(z) ],  s(z)=s_0 D(z).`

This is **one** free parameter (`ε`), fixed by **one** anchor (§4), after which the entire `z`-shape is
determined by the *observable* field evolution `{σ(z), f(z)}` — no per-z freedom. That is the distinction
the roadmap demands (§4a: "chosen per-z to fit" is what makes threading unfalsifiable; a single
`ε` fixed once is not that).

### 2.1 The z-dependence, and why it dissolves the floor

Two evolving quantities drive `f_v^{map}(z)`:

1. **`σ(z) = σ_0 D(z)` shrinks** toward high z (`D`: 1 → 0.377 at `z=2.33`, ΛCDM). At any *fixed
   negative* threshold this drives `P(δ<δ_th) → 0`.
2. **`δ_th(z) = −3ε/f(z)` moves toward the mean** as `f(z)` grows (`f`: 0.53 → 0.97; `|δ_th|` shrinks by
   `f(0)/f(z)`: 1 → 0.547). This partially opposes (1).

For **any `ε>0`**, `δ_th(z) < 0` at every z and `σ(z)→0`, so `f_v^{map}(z) → 0` as `z→∞`. **Floor
dissolved** (roadmap 4a requirement met): unlike the fixed below-mean threshold (`ε=0`, floored at
`0.5`), every `ε>0` mapping vanishes early. The net decline steepness is set by the single number
`ν_0 ≡ |δ_th(0)|/σ_0 = 3ε/(f(0)σ_0)` — the void threshold's "height" in units of the z=0 field width.

### 2.2 The mapping systematic band (two-sided, pre-registered)

Two modelling choices are carried as the mapping's systematic band, exactly as Phase D carried the
definition systematic:

- **Primary — expansion-margin (M4/M5), theory-faithful.** Timescape `f_v` *is* an expansion fraction,
  so holding the expansion margin `ε` fixed is the faithful reading. The `f(z)` factor is intrinsic to it.
- **Edge — fixed nonlinear density threshold**, `δ_th = const` (drop the `f(z)` factor). This is what
  void catalogs operationally threshold on (VoidFinder/watershed underdensity). It declines *steeper*
  than the expansion-margin mapping (no opposing `f(z)` term), so it is the permissive edge for the shape.

The velocity–density relation (M1) is linear; deep voids are nonlinear (`δ ≲ −0.5`). The linear `f`
in (M4) is therefore an approximation whose nonlinear correction (voids expand faster than linear at
fixed `δ`; e.g. spherical-void / Bernardeau expansion) is a sub-dominant systematic inside this band.
The structural verdict of §3 is **anchor- and band-independent**, so this uncertainty does not move it.

---

## 3. The structural result: level ⊥ shape (near-theorem), and the ΛCDM-growth forecast

**Near-theorem (level–shape incompatibility).** For any unimodal volume PDF whose fluctuation
amplitude `σ(z)` decreases monotonically with z (standard growth) and any single threshold `δ_th`:

- if `f_v(0) ≥ 0.5` the threshold sits at/above the field mode; as `σ` shrinks the PDF narrows toward
  its mode and `f_v(z)` stays pinned near its z=0 value (→ `0.5` for the right-skewed field) — **it
  cannot decline to `≈0.19`**;
- a steep decline to `→0` requires `δ_th` well below the mode, i.e. `f_v(0) < 0.5`.

`f_v^req(0)=0.640 > 0.5` forces the first branch. **The required combination (high level `0.64` AND
steep decline `×3.3`) is not jointly realizable by any single-threshold void population of a field whose
fluctuations shrink with z.** The below-mean floor (M2) is the sharpest case and is *growth-independent*
(right-skewness alone), so this incompatibility survives even if backreaction alters the growth rate.

**ΛCDM-growth forecast** (the structure-growth null; `mapping_decline_forecast.json`; uses only ΛCDM
`D(z), f(z)`, the committed `σ_0=0.734` from 2M++, and the committed `f_v^req` — **no telescope void
data**). Required decline ratios `f_v^req(z)/f_v^req(0)` = `[1, 0.830, 0.618, 0.437, 0.302]` (`×3.31`).
Under the derived mapping:

| anchoring of the single `ε` | `f_v^{map}(0)` | predicted decline `f_v(z)/f_v(0)` | total |
|---|---|---|---|
| match z=0 **level** (`ε→0`, below-mean) | 0.643 | `[1, 0.968, 0.934, 0.898, 0.863]` | ×1.16 |
| match decline **shape** (best `ε`, expansion-margin) | **0.414** | `[1, 1.027, 0.974, 0.851, 0.642]` | ×1.56 |
| fixed-density edge, anchored to 0.414 | 0.414 | `[1, 0.857, 0.676, 0.448, 0.191]` | ×5.23 |

- The **expansion-margin** mapping's *maximum achievable* total decline over **all** `ε` is **×1.95**
  (attained only at `f_v(0)≈0.18`) — it can never reach the required `×3.31`. Being theory-faithful
  (keeping the `f(z)` factor) makes it *shallower*, so the faithful reading fails hardest.
- Matching the **level** (`0.64`, `ε→0`) gives the below-mean floor, decline `×1.16`.
- Matching the **shape** forces `f_v(0)=0.41`, a 0.23 shortfall below the required `0.640`.
- Only the **fixed-density edge** declines enough (`×5.23`), and only at `f_v(0)=0.41`.

Every row confirms the near-theorem: no single `ε` occupies the `(0.64, ×3.3)` corner. This is the
quantitative form of roadmap 4b's level-vs-shape tension and of `R2.json`'s `MARGINAL_top_edge`.

---

## 4. Anchoring the single parameter (pre-registered, non-circular)

The margin `ε` is fixed **once**, before the shape data, by matching the mapping's z=0 level to the
**bias-independent below-mean measurement** `f_v^{bm}(0) = 0.643` (Phase D, `phaseD_fvobs.json`
`below_mean_central`; band `[0.614, 0.672]`), which independently coincides with `f_v^req(0)=0.640`
(mid-band). This is the *fairest* anchor to the theory (it grants the level the data already supply)
and it is non-circular: it uses only the z=0 measurement, nothing from the z>0 required shape. It
forces `ε≈0` (`δ_th(0)≈0`), and — because `0.643 > 0.5` — that lands in the floor regime, so the
predicted decline is bounded near `×1.16` **regardless of the measured `σ(z)`** (the below-mean floor
is growth-independent). The z>0 decline is then a **parameter-free prediction**.

The deep-void anchor (`ε` set by a physical void barrier, e.g. `δ_th ≈ −0.8` nonlinear / shell-crossing,
Sheth & van de Weygaert 2004) is carried as the **shape edge**: it matches the required decline shape
but at `f_v(0)≈0.41`, missing the level. Reporting both anchors makes explicit that **no single `ε`
delivers both** — which is the finding, not a tuning knob.

---

## 5. Pre-registered R2-final test (falsifiable; measure Y, calculation Z, falsifier W)

A **two-part** test (roadmap 4b), fixed here in advance. Survey nodes: the required nodes
`z ∈ {0, 0.3, 0.7, 1.3, 2.33}`; the *decisive* directly-measurable node is `z ≈ 0.5–0.7`
(BOSS DR12 / eBOSS / DESI-if-public), per PLAN_void_history.md §5.2.

**Part 1 — LEVEL (z=0), bias-independent below-mean definition.**
- **Y**: `f_v^{bm}(0) = P(δ<0)` in the 2M++/Carrick field.
- **Z**: Phase D, committed: `0.643`, band `[0.614, 0.672]` (reliable-volume systematic).
- **W (pass)**: `f_v^req(0) ∈` below-mean band. **Status: PASS** for the LA reading (`0.640` mid-band).
  Documents that the *level* is available at the below-mean edge — but says nothing about the shape.

**Part 2 — SHAPE (decline ratio), derived mapping at the anchored threshold.**
- **Y**: the decline ratio `r_obs(z_k) = f_v^{obs}(z_k) / f_v^{obs}(0)` at each survey node, with the
  void volume fraction measured under the derived evolving threshold `δ_th(z)=−3ε/f(z)` (§2) at a
  **single, fixed comoving smoothing scale `R_s`** (the same `R_s` used for the z=0 anchor), from survey
  randoms (volume filling fraction) or a reconstructed density field.
- **Z**: `f_v^{obs}(z_k)` = Σ(void volume)/(survey volume) at the pre-registered threshold; ratio taken
  against the same-mapping z=0 value. Carry the `{σ(z), f(z)}` inputs as **measured** (replace the ΛCDM
  `D(z)` of the forecast with the survey-measured `σ(z)/σ_0`).
- **W (SUPPLIED)**: `r_obs(z_k)` lies inside the required decline-ratio band at **every** node:
  `z=0.3: [0.812, 0.836]`, `z=0.7: [0.599, 0.637]`, `z=1.3: [0.361, 0.499]`, `z=2.33: [0.231, 0.363]`
  (`Δχ²≤1` propagated, `mapping_decline_forecast.json` `required.decline_ratio_{lo,hi}`).
- **W (SHAPE-UNAVAILABLE)**: at the level-anchored threshold, `r_obs(z_k)` exceeds the required-hi edge
  (flatter than required — the floor) by `≥ 2σ_obs` at the decisive node `z≈0.7`. Concretely: the
  below-mean floor predicts `f_v^{obs}(0.7) ≈ 0.60` (ratio `0.93`); the required-derived value is
  `≈ 0.40` (ratio `0.62`); a ~0.20 absolute / ~50% relative gap that a void filling fraction pinned to
  `±0.05` resolves decisively. Quantify the gap in `σ` and in the backreaction deficit `Q_obs(z) − Q_req(z)`.

**Lapse-reading multiplicity (roadmap §5.3).** `f_v^req` is lapse-dependent. Register all three required
bands: **LA (primary)** `f_v^req(0)=0.640`, decline `×3.3`; **V0 (no-lapse)** `f_v^req(0)=0.383`
(*below* 0.5), decline `×22`; **LB (rate-ratio)** — not yet run, up to 27% off LA (`NOTES_modelv_theory.md`
§7B). The below-mean measurement `0.643` *supplies* the LA level but is *too high* for the V0 level
(0.383) — V0 fails Part 1 by level, LA fails Part 2 by shape; the data already discriminate the readings
(roadmap §5.3 bonus). Run Part 2 against whichever readings pass Part 1.

**Forced-fit gate (roadmap §1, §6).** SUPPLIED additionally requires the zero-shape-parameter forced
joint fit at `f_v^{obs}(z)` to clear the BIC bar `χ² ≤ χ²_ΛCDM + ln N` (≈ `χ²_ΛCDM + 7.4` on DR2).

---

## 6. Verdict vocabulary (roadmap §6; assigned only after §5 telescope data)

- **SUPPLIED** — the derived mapping (single pre-registered `ε`, §4) reproduces `f_v^req(z)` at all
  nodes: Part 1 PASS **and** Part 2 `r_obs` inside the required band at every node, **and** the forced
  fit clears the BIC bar. Headline: *the observed void population supplies the required history; a
  zero-shape-parameter void-forced cosmology fits SN+BAO+CMB with one parameter fewer than ΛCDM*
  (kinematic caveat rides; "better than ΛCDM" stays out — paper 3's adjudication).
- **SHAPE-UNAVAILABLE** — Part 1 PASS but Part 2 measured decline is flatter than required beyond `2σ`
  at `z≈0.7` (the floor scenario). Report the gap in `σ` and `Q(z)`: *the observed void population
  cannot supply the backreaction the Hubble diagram wants.* `f_v^req(z)` stands as the model-independent
  target any backreaction proposal must hit. **This is the branch the ΛCDM-growth forecast and the §3
  near-theorem anticipate.**
- **MAPPING-UNDERIVABLE** — reserved for the case in which no `f_v ↔ observable` relation can be fixed
  without per-z freedom. **This memo does not trigger it**: a principled one-parameter, z-dependent
  mapping is derived (M4/M5), its single parameter is anchored non-circularly at z=0 (§4), and its z>0
  shape is a parameter-free, falsifiable prediction (§5). The mapping *is* derivable; the open question
  is only whether the data land SUPPLIED or SHAPE-UNAVAILABLE.

---

## 7. Systematics and open items (carry into the data reduction)

1. **Smoothing scale `R_s`.** `f_v` and its evolution depend on the comoving smoothing scale (`σ_0`
   depends on `R_s`; Phase D used 4 Mpc/h). The z=0 anchor and every z>0 point **must** use the same
   `R_s`. Report the `R_s`-sensitivity of the decline as a systematic band.
2. **Measured vs ΛCDM `σ(z)`.** The forecast uses ΛCDM `D(z)` as the null; the test replaces it with the
   survey-measured `σ(z)/σ_0`. The timescape premise is that backreaction alters growth, so the
   measurement can in principle deviate from the null — but the below-mean floor (M2) is
   growth-independent, so the level-anchored Part-2 prediction is robust to this.
3. **Linear vs nonlinear velocity–density (M1).** Deep voids expand faster than linear at fixed `δ`;
   the nonlinear correction to `f` in (M4) is a sub-dominant systematic, inside the §2.2 band, and does
   not move the §3 verdict.
4. **Bias.** Part 1 (below-mean) is bias-independent by sign. Part 2 deep thresholds are bias-dependent;
   use the Carrick `δ_g*→δ` mapping (Phase-3 experience) and carry the residual as a systematic.
5. **Lognormal vs Gaussian PDF.** (M5) uses the lognormal (nonlinear z=0); at high z the field
   linearises (`s→0`, lognormal→Gaussian). Cross-check the decisive `z≈0.7` point under both PDFs.
6. **High-z bridge.** `z=1.3, 2.33` are extrapolation-dominated (no clean catalog filling fraction);
   the decisive test lives at `z≈0.5–0.7` where catalogs exist. Do not let the bridge drive the verdict.
7. **LB reading** (roadmap §5.3) still owes its own `f_v^req` band; fold in when run.

---

## 8. Sources

- Wiltshire 2009, *Average observational quantities in the timescape cosmology*, PRD 80, 123512
  (arXiv:0909.0749) — `f_v` as the faster-than-average-expanding fraction; two-phase partition; lapse.
- Duley, Nazer & Wiltshire 2013 (arXiv:1306.3208) — general non-tracker two-scale system; identities
  K1–K5 (NOTES_modelv_theory.md §1).
- Peebles 1980, *The Large-Scale Structure of the Universe*, §14 — linear velocity–density `∇·v=−fHδ`.
- Linder 2005, PRD 72, 043529 — growth index `f ≈ Ω_m(z)^0.55`.
- Coles & Jones 1991, MNRAS 248, 1 — lognormal model of the density field (M5 form).
- Sheth & van de Weygaert 2004, MNRAS 350, 517 — void shell-crossing barrier (deep-void anchor, §4).
- Committed repo artifacts: `modelV_probeR.json` (required history + lapse readings), `phaseD_fvobs.json`
  (z≈0 below-mean field, `σ_0`), `R2.json` (current MARGINAL verdict), and this memo's forecast
  `src/probes/mapping_decline_forecast.py` → `probes_out/mapping_decline_forecast.json`.
