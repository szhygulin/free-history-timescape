# Reasoning & roadmap — paper 2: the theory test, and the road to the telescope-forced verdict

*Strategy layer, written 2026-07-06 by the review session, after the committed Probe R artifacts
and a read of the then-uncommitted Phase D / R2 / Phase F first passes. Preliminary numbers below
are quoted from in-flight artifacts and are **not verified** — the §7 verification pass governs
them. Execution mechanics live in [`PLAN_void_history.md`](PLAN_void_history.md); equations and
their provenance in [`NOTES_modelv_theory.md`](NOTES_modelv_theory.md). This file records why the
program is shaped the way it is, what each result can and cannot claim, and the bars the remaining
tests must clear.*

---

## 0. Program map (three papers)

1. [timescape-hubble-tension](https://github.com/szhygulin/timescape-hubble-tension) — **finished.**
   Standard (tracker) timescape does not resolve the Hubble tension: its one-parameter history is
   forced to f_v0 ≈ 0.85 by supernovae while BAO+CMB demand ≈ 0.64 (4.8–6.6σ split); a free smooth
   E(z) shows the data are not internally contradictory — the tracker's rigidity is the failure.
2. **This repo — the theory test.** Two questions, one per rung of the evidence ladder (§1):
   *(a)* does the timescape mechanism fit at all once the history is freed? (**answered:
   R1 = RECONCILES**); *(b)* can the void history actually measured by surveys supply the required
   history with **zero fitted shape parameters**? — the telescope-forced test, the grand final
   target of this paper.
3. [free-history-timescape-tensions](https://github.com/szhygulin/free-history-timescape-tensions)
   — **the tensions.** Whether free-history timescape resolves the Hubble tension end-to-end (on
   fitted *and* catalog-forced structure) and survives the wider tension suite (growth/S₈, CMB
   beyond the acoustic point, BBN/sound horizon). Plan there: `PLAN.md`. The anchored-H₀
   (freshH0) analysis belongs to paper 3 and re-homes there.

## 1. The evidence ladder — why "telescope-forced" is the whole game

Same equations, same solver, same data at every rung; the only change is **who supplies f_v(z)**:

| rung | f_v(z) supplied by | fitted shape params | what it can prove |
|---|---|---|---|
| fitted (Probe R — done) | optimizer | 5 | existence of a reconciling history; the target band |
| amplitude variant (done, first pass) | shape from fit/data + fitted amplitude A | 1 | whether SN and BAO+CMB agree on A — the split test |
| **forced (grand final)** | **surveys** | **0** | **evidence: prediction, not accommodation** |

Accounting (ΛCDM has one shape parameter, Ωm; SN offset and BAO α are common profiled nuisances):

- **Fitted rung earns nothing.** DR1: Δχ² = −6.2 vs ΛCDM for +4 parameters (Wilks p ≈ 0.19,
  ΔBIC ≈ +23 against). DR2 first pass: Δχ² = −1.5 (p ≈ 0.82). As expected — never quotable as
  evidence for the theory.
- **Forced rung flips the comparison.** With zero shape parameters the model has *one fewer* than
  ΛCDM, so it wins the BIC comparison iff joint **χ² ≤ χ²_ΛCDM + ln N** — ≈ 1409.6 on the DR1
  vintage (N ≈ 1593), ≈ 1407.2 on DR2 (ΛCDM 1399.8 + 7.4). It may fit up to ~7 points *worse*
  than ΛCDM in raw χ² and still win. This is the bar the telescope history must clear.
- **Fit-space ceiling.** Free E(z) reached 1391.85 → the whole dataset holds only ~10 χ² points of
  non-ΛCDM structure. No model can beat ΛCDM decisively in fit-space; decisive wins live in
  prediction-space (paper 3).
- **Band freedom is freedom.** The observed-history band (definition systematic above all)
  re-admits flexibility through the side door. Any within-band choice must be accounted like a
  parameter; §4–5 exist to shrink exactly this.

## 2. Established (committed artifacts)

- Solver gates: tracker limit reproduced exactly (SN χ² = 1391.545 to six digits; distances to
  6×10⁻⁸; non-FLRW dD_M/dz ≠ D_H signature to <5×10⁻⁵) — `modelv_gates.py`, G-T/G-A PASS.
- **R1 = RECONCILES** (`modelV_probeR.json`, independently re-derived in
  `verify_modelv_probeR.json`): joint χ² on identical DR1-vintage data — free E(z) 1391.85 ·
  free f_v(z) **1396.06** · w₀wₐCDM 1398.29 · ΛCDM 1402.24 · tracker 1469.29.
- Required history (algebraic lapse), nodes z = {0, 0.3, 0.7, 1.3, 2.33}:
  f_v^req = [0.640, 0.531, 0.396, 0.279, 0.194]; Δχ²≤1 bands [0.638–0.641], [0.521–0.533],
  [0.384–0.407], [0.231–0.318], [0.148–0.231].
- No-lapse control V0 also reconciles: χ² = 1396.49 at f_v = [0.383, 0.250, 0.128, 0.072, 0.017].
- Probe R localization: the joint-best *tracker* already sits at f_v0 ≈ 0.643 (paying +40 on SN);
  the freed history keeps that amplitude and steepens the decline — worth Δχ² = −73 (SN −45,
  BAO+CMB −28). The tracker fails by **decline rate, not amplitude**. A history mimicking ΛCDM's
  D_M(z) scores 1410.3 (*worse* than ΛCDM: the dressed geometry cannot copy D_M and D_H at once);
  the optimum sits 6 *below* — the non-FLRW wedge does positive work.

**Not established (ride with every quoted number):** (i) no evidence for the theory from any
fitted rung (§1); (ii) not a theory — the forced history violates Buchert integrability (α² drifts
81%; tracker constant to 4×10⁻⁷), lies off the entire exact two-scale empty-void family, and the
tracker attractor means initial conditions cannot buy it; (iii) mechanism untested — V ≈ V0, so a
free history cannot distinguish clock dressing from none; (iv) all bands conditional on the
algebraic (LA) lapse; the rate-ratio (LB) reading (27% off-tracker) not yet run.

## 3. In-flight first passes (PRELIMINARY — unverified, uncommitted at time of writing)

- **Phase D** (`phaseD_fvobs.json`): z = 0 anchored directly in the 2M++ field — below-mean
  (δ<0) volume fraction band **[0.614, 0.672]** (bias-independent; single percolating component),
  threshold family down to 0.222 (δ<−0.5). **Only z = 0 is telescope-direct**: BOSS/eBOSS volume
  filling fractions were not extractable without re-deriving from survey randoms, so z > 0 points
  are a declared lognormal-excursion growth extrapolation anchored at z = 0.
- **R2** (`R2.json`): verdict `MARGINAL_top_edge`. Required f_v(0) = 0.640 sits mid-band in the
  below-mean definition (✓), above the standard watershed literature bracket [0.50, 0.62] by
  0.17 band-widths, and inside the (very permissive) full definition band at all five nodes.
  Tracker contrast: its 0.853 sits ~3 band-widths above every definition — the qualitative
  reversal is real. But: **required decline ×3.3 over the node range vs below-mean edge ×1.2
  (right level, too flat) vs δ<−0.3 edge ×3.5 (right shape, level 0.44 not 0.64)** — no single
  threshold matches level *and* shape (§4).
- **DR2 joint refit** (`phaseF_joint_ampsplit.json`): free-history 1398.28 (f_v0 = 0.636 — stable
  vs DR1's 0.640) vs ΛCDM 1399.81 vs tracker 1514.10 (DR2 sharpens the tracker's failure to
  +114 vs ΛCDM).
- **Amplitude split — the headline robustness number: it dissolves.** A_SN = 0.6351 vs
  A_BAO+CMB = 0.6363, σ = **0.16** (the same statistic on the tracker's own parameter reproduces
  paper 1's split at 6.5σ). The 0.85-vs-0.64 conflict was entirely the tracker *shape*'s fault.
- **freshH0** (`phaseF_freshH0.json`): paper-3 subject matter; re-homes there. For the record:
  ΛCDM anchored gate passed (73.53 ± 1.02); tracker control reproduces paper 1 (73.00);
  free-history fixed-shape anchored fullrate 73.34; **convention-independent anchored/global bare
  ratio = +8.4%** — paper 3's target number.

## 4. Two structural findings that reshape the final R2 (registered before the BOSS/eBOSS data work)

**4a. The below-mean floor.** For any right-skewed density field (gravity guarantees it: collapse
concentrates mass, most volume is underdense), the below-mean volume fraction obeys
**P(δ<0) ≥ 0.5 at every z**, → 0.5 as the field → Gaussian at high z. But f_v^req crosses 0.5 at
z ≈ 0.4 and reaches 0.396 by z = 0.7 (V0's required history is below 0.5 *everywhere past z = 0*).
So **no fixed below-mean mapping can supply the required history beyond z ≈ 0.4 — measured or
extrapolated**. This is not an observational question; it is structural. It also cuts the other
way: the timescape f_v → 0 at early times while below-mean → 0.5, so a fixed below-mean mapping is
wrong at high z for *any* timescape history, the tracker included. Consequence: the
f_v ↔ observable mapping is necessarily **z-dependent and must be derived from the two-phase
partition** (voids = regions past a finite expansion-excess / underdensity margin, which naturally
→ 0 early), not chosen per-z to fit. R2's current "threads the definition band, riding the
permissive edge at z≈0 and the moderate edge at high z" is exactly what a physically-derived
mapping *would* look like — but until the mapping is derived, threading is unfalsifiable freedom.

**4b. Level-vs-shape tension at fixed threshold** (the empirical mirror of 4a): below-mean matches
the z=0 level (0.64 ✓) but is flat and floor-bounded; δ<−0.3 matches the decline (×3.5 ≈ required
×3.3) but not the level (0.44). The final R2 verdict must therefore be a **two-part test**:
level at z = 0 under the bias-independent below-mean definition, **and** decline ratio
f_v(z)/f_v(0) under the pre-registered derived mapping. The envelope test ("inside the definition
band at all nodes") is necessary but cannot falsify anything — the band [0.22, 0.67] is too wide.

## 5. Remaining work for paper 2 (critical path, in order)

1. **Derive the mapping** (theory memo → NOTES addendum): the observable proxy for timescape's
   f_v as a function of z, from the partition definition. Pre-register it **before** step 2's
   data are reduced.
2. **The telescope numbers**: BOSS DR12 VIDE / eBOSS (DESI if public) volume filling fractions
   from survey randoms at z ≈ 0.2–0.7. The single decisive measurement: the void volume fraction
   at z ≈ 0.5–0.7 under the pre-registered mapping — required ≈ 0.40–0.45 there vs a
   flat/floor-bound ≈ 0.55–0.60 if the below-mean extrapolation is right. Replace the growth-model
   extrapolation with data; until then the "telescope-forced" claim exists only at z = 0.
3. **Run LB (rate-ratio) Probe R** → its own f_v^req band; carry LA/LB/V0 as three required-history
   bands into R2-final. Bonus: the data adjudicate the readings (LA needs 0.64 at z=0, V0 needs
   0.38 — the below-mean measurement already discriminates).
4. **R2-final** per §4b, verdict vocabulary fixed in advance: SUPPLIED / SHAPE-UNAVAILABLE /
   MAPPING-UNDERIVABLE (see §6).
5. **Forced joint fit** (zero shape params) vs the BIC bar (§1); SN covariance ladder (Table-I
   analogue); DR1 sensitivity row.
6. **Adversarial verification** of every in-flight artifact (refute-by-default, fresh agents),
   plus the NOTES equation-number audit against source PDFs.
7. **Presentation discipline**: parameter-count column + kinematic-reading disclaimer with every
   χ² table (README table currently lacks the params column — pending fix).

## 6. Decision tree → publishable claims (pre-registered)

- **SUPPLIED**: derived mapping + measured decline consistent with f_v^req, forced fit clears the
  BIC bar → headline: *the observed void population supplies the required history; a
  zero-shape-parameter void-forced cosmology fits SN+BAO+CMB with one parameter fewer than ΛCDM.*
  (Kinematic caveat rides; "better than ΛCDM" stays out of the abstract — that adjudication is
  paper 3's.) Paper 3 upgrades to decisive.
- **SHAPE-UNAVAILABLE**: measured decline flat where f_v^req dives (the floor scenario) →
  quantified refutation in σ and in Q(z): *the observed void population cannot supply the
  backreaction the Hubble diagram wants.* Strongest possible close of paper 1's argument;
  f_v^req(z) stands as the model-independent target any backreaction proposal must hit.
- **MAPPING-UNDERIVABLE**: no principled f_v ↔ observable mapping exists without per-z freedom →
  report exactly that: *the comparison is unfalsifiable at current theory maturity* — an honest
  negative about the mechanism's testability, and a sharp task statement for paper 3's WP-B.

## 7. Claims ceiling

Paper 2 never claims: better-than-ΛCDM overall (breadth untested — paper 3); theory status
(integrability violated — paper 3 WP-B); mechanism support from any fitted rung (V ≈ V0). The
amplitude-split dissolution is an *autopsy of the tracker's shape*, not evidence for dressing.
What paper 2 *can* claim at maximum: the two-question answer — the mechanism is flexible enough
(R1), and the observed void population does / does not / cannot-yet-be-tested-to supply the
required history (R2-final + forced fit).

## 8. Discipline

One number, one committed script, one artifact. Adversarial verification before any headline.
Thresholds, statistics, and mappings fixed before the data are reduced (this document is part of
that record). Failures reported at the same volume as successes.
