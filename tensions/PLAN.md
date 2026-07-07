# PLAN — free-history timescape vs. the cosmological tensions (paper 3)

*Drafted 2026-07-06 by the program's review session. Scope per [`README.md`](README.md): the
Hubble tension end-to-end — on **fitted** and on **catalog-forced** void structure — then the
wider tension suite. Inputs come from
[paper 1](https://github.com/szhygulin/timescape-hubble-tension) (harness, anchored-H₀ machinery,
tracker baselines) and
[paper 2](https://github.com/szhygulin/free-history-timescape) (general solver + gates, required
history f_v^req(z), observed history f_v^obs(z); strategy in its `REASONING_AND_ROADMAP.md`).
A fitted-structure anchored-H₀ **first pass already exists** (in paper 2's working tree, to be
re-homed here); its numbers are quoted below as PRELIMINARY and unverified. The pre-commitments
in §3 are registered now — they govern the verification of that first pass and every re-run.*

---

## 0. The question, and why the Hubble tension is the leverage point

Papers 1–2 are geometry fits. This paper asks the question that decides whether the model
*matters*: does free-history timescape resolve the Hubble tension as a **prediction**, and does
it survive the joint tension suite that ΛCDM already fits?

The H₀ tension is the one major observable where ΛCDM has no mechanism at all — its options are
"~5σ fluke" or "ladder systematics", i.e. the SH0ES datum costs it Δχ² ≈ 25. A model that
*predicts* the local value from independently fixed ingredients converts that cost to ~0: an
evidence swing of order e^{12} from a single datum, dwarfing anything achievable in fit-space
(paper 2 §1: the whole joint dataset holds only ~10 χ² points of non-ΛCDM structure). The entire
difference between "predicted 73" and "can reach 73" is whether the ingredients were fixed before
the comparison — hence the pre-commitments in §3.

Four conditions separate a decisive resolution from a worthless one:

1. **Prediction, not accommodation** — zero new knobs; conventions pinned in advance (§3).
2. **Right z-profile, not just the right number** — the bias must decay past the homogeneity
   scale as the actual ladder samples it (§3, P4).
3. **Nothing else breaks** — growth, CMB beyond one point, BBN/sound horizon (§4–5, §7).
4. **It is a theory when it does it** — dynamical consistency (§6). Until WP-B lands, every
   positive result here is phenomenology and is labeled as such.

## 1. Inputs and artifact re-homing

- From paper 2: the gate-validated general dressed-geometry solver (tracker limit to 1e-9;
  non-FLRW dD_M/dz ≠ D_H reproduced); f_v^req(z) bands per lapse reading (LA committed; LB to be
  produced — paper 2 critical-path item 3); f_v^obs(z) with the derived-mapping band (paper 2
  items 1–2); the DR1/DR2 joint-fit machinery.
- From paper 1: the Pantheon+SH0ES calibrator-anchored machinery (`freshH0` pattern: M_B pinned
  by 77 Cepheid calibrators, full stat+sys covariance, GLS-profiled), its ΛCDM gate and tracker
  baseline; the 17–22% tracker local-excess window computation (to be generalized, §3 P2).
- Re-home here: `phaseF_freshH0.py`, `phaseF_freshH0.json`, `_diag_conventions.py` (currently in
  paper 2's working tree — they are paper-3 subject matter). Every quoted number gets a committed
  generating script + JSON artifact in this repo.

## 2. First-pass state (PRELIMINARY — quoted from the unverified in-flight artifact)

- Controls passed: ΛCDM anchored H₀ = 73.53 ± 1.02 (expected 71.5–75); tracker anchored
  H₀ = 73.00 (reproduces paper 1 exactly — "agreement at 61" was calibration-relative).
- Free-history, shape **fixed** to the Probe-R global fit (no refit to the ladder — correct
  discipline): anchored bare H̄₀ = 58.32 ± 0.80 → full-rate dressed H₀ = **73.34** (0.3σ from
  SH0ES), g_dress convention 69.36.
- Global (SN+BAO+CMB, no calibrators): H̄₀ = 53.79 → full-rate 67.65 (Planck band), g_dress 63.97.
- **The paper's target number:** the convention-independent anchored/global bare ratio
  b_req ≡ H̄₀(anchored)/H̄₀(global) − 1 = **+8.4%** — the model's internal ladder-vs-global offset.
  It replaces both ΛCDM's 9.1% (73.5/67.4) and the tracker's required 17–22% window. The model
  resolves the tension iff it *predicts* this number (§3).
- Robustness note from the first pass: an SN-ladder-only refit of the 5-node history is
  under-constrained at high z (f_v0 rails to 0.92) but the anchored local slope is shape-stable
  (full-rate stays ≈ 73.2) — the anchored H₀ is a robust output, the high-z nodes are not.

## 3. WP-H — the Hubble tension end-to-end (the core; fitted AND catalog-forced)

**Pre-commitments (registered 2026-07-06, before any tension verdict is computed):**

- **P1 — primary statistic.** The convention-independent bare ratio b_req (above). The
  g_dress-vs-full-rate convention split (anchored 69.36 vs 73.34) is thereby sidestepped for the
  verdict; both conventions are still reported, and the f_v′(0)-sensitivity is a declared
  systematic, never a post-hoc choice.
- **P2 — the decisive computation.** b_pred: the local Hubble bias a wall observer should measure,
  computed from the model's own structure — f_v(z→0), (H_v−H_w)(0), the dressing, and the actual
  survey geometry of the SH0ES calibrator + Hubble-flow sets — with **zero new parameters**, by
  generalizing paper 1's expansion-variance window machinery (which gave 17–22% for the tracker at
  f_v0 ≈ 0.76) to the free history (f_v(0) = 0.640, (H_v−H_w)/H̄₀ ≈ 0.52 at z=0). This number
  exists before looking at b_req and is compared once.
- **P3 — verdict form.** Equality test: |b_pred − b_req| ≤ 1σ ⇒ **RESOLVES** (phenomenological);
  ≤ 2σ ⇒ PARTIAL; else ⇒ **FAILS**. σ combines the anchored-scale error, the global-fit error, and
  the lapse-reading spread (LA/LB/V0) in quadrature. A FAILS verdict kills the tension claim for
  this model family regardless of joint-fit quality — and is published at full volume.
- **P4 — z-profile falsifier.** b_pred(z_max) must decay past ~100 Mpc/h the way the ladder
  samples it: compare against SH0ES redshift-subsample splits at first order. Carry the
  ladder-correction caveat explicitly: published zHD contains ΛCDM/2M++ peculiar-velocity
  corrections (paper 1's zCMB-vs-zHD discipline applies); full re-derivation of the ladder's flow
  corrections inside the model is **declared out of scope** (§8) — first-order treatment only.
- **P5 — standing controls.** Every run reproduces the ΛCDM anchored gate (73.5 ± 1) and the
  tracker baseline (73.0) before its number is quoted.

**Tracks:**

- **H-A (fitted structure).** Verify the first pass adversarially; add the LB-lapse variant; DR2
  vintage; compute b_pred (P2) and issue the P3 verdict; z-profile check (P4).
- **WP-H2′ (scope amendment, 2026-07-06) — the survey-averaged b_pred.** Supersedes §8's
  out-of-scope line for this one item. Built as a radial apparent-H₀ profile
  ⟨H⟩(<r) = E_max·φ(r): the void-scale maximum is a pure wall-observer clock conversion
  E_max = γ̄₀H_v0/H_dress0 − 1 = (3/2)/g_dress(fv0) − 1 on the tracker (gate a reproduces the
  Wiltshire 17–22% window: 0.171@fv0=0.76, 0.220@fv0=0.695, verified vs arXiv:0909.0749 /
  0912.5234); small-r ceiling E_max(LA)=0.33650 (gate b — this corrects the Wave-1b b_pred, which
  mis-identified the *volume-average* excess 0.19480 = E_dress_void as the maximum; the maximum
  drops the γ̄̇ subtraction, E_max = E_dress_void + γ̄̇/H_dress); large-r → 0 (gate c). The SH0ES
  ladder bias is b_pred_survey = E_max·⟨φ⟩_HF — only the Hubble-flow SNe carry it (M_B is fixed by
  *geometric* Cepheid distances, so the calibrator apparent rate does not enter) — giving 0.024, a
  14× dilution (58% of HF SNe sit past the ~100 h⁻¹Mpc homogeneity scale). **Verdict FAILS**:
  b_pred_survey under-predicts b_req=0.08417 by 3.5σ (measurement-only); the pre-registered
  PARTIAL_FRAGILE holds only because the dominant φ-shape *theoretical* systematic inflates σ, and
  b_pred stays below b_req across the entire systematic band. The dressing mechanism does not
  produce the required local bias even with the corrected (larger) void-scale maximum. Artifacts
  `bpred_survey_averaged.json`, verified `verify_bpred_survey.json`.
- **H-B (catalog-forced structure) — the decisive version.** Re-run the identical pipeline with
  f_v(z) = f_v^obs(z) from paper 2's final Phase D (zero fitted shape parameters). Gated on
  paper 2's R2-final verdict (SUPPLIED or at minimum a derived-mapping band). If paper 2 lands
  SHAPE-UNAVAILABLE, H-B still runs once at f_v^obs as the confirmation of failure, per the
  program's symmetry rule (failures get the same compute as successes).
- **H-C (early side).** The model's own early-Universe determination of the absolute scale —
  currently the CMB acoustic point with r_d = 147.09 adopted as an external ruler (a disclosed
  crutch). Owned by WP-C; H₀ "end-to-end" is claimable only when r_d is computed, not assumed.

## 4. WP-C — CMB beyond one point, sound horizon, BBN

- Generalize the radiation-era solution (Duley–Nazer–Wiltshire 2013 template) to the free
  history; compute r_d *within* the model + BBN consistency (ω_b), replacing the external 147.09.
- Acoustic-scale compression done consistently (at minimum: ℓ_A + shift parameter + ω_b derived
  in-model); the full Planck likelihood requires the perturbation layer and is gated on WP-B.
- Self-consistency check the current harness cannot see: the same f_v(z) must serve the distance
  to z_dec *and* the early expansion history that sets r_d.

## 5. WP-G — growth / S₈ (gated on WP-B)

Growth observables (fσ₈, RSD, lensing amplitude) require perturbation theory on the averaged
background; **no rigorous version exists in the kinematic reading**. Do not fake it with FLRW
growth formulas. Deliverable before WP-B lands: a scoping memo stating exactly which growth
observables are well-defined in which reading — itself a publishable clarification, and the
honest placeholder for the S₈ column of the tension table.

## 6. WP-B — dynamical consistency (Reading B / V2)

The forced history violates Buchert integrability (α² drifts 81%; any exact two-scale empty-void
solution keeps it constant, and the tracker attractor blocks initial-condition fixes) — so
consistency requires new physical freedom: multi-phase void populations binned by observed
depth/size (Wiegand–Buchert multi-scale), evolving depth, non-empty voids / wall curvature /
phase exchange. Derive the lapse dynamically (DNW13 Eq 22 generalization) — this kills the LA/LB
ambiguity — and re-validate the tracker limit. WP-B also owns the **derived f_v ↔ observable
mapping** flagged in paper 2 §4a (the below-mean floor: P(δ<0) ≥ 0.5 at all z while f_v^req and
any timescape f_v → 0 early — the fixed below-mean proxy is structurally wrong at high z).
Falsifier: no consistent population model tracks f_v^obs within its band → the mechanism dies at
theory level; that close-out is a result, not a failure of the paper.

**Execution plan: [`PLAN_WPB_threephase.md`](PLAN_WPB_threephase.md)** (pre-registered
2026-07-06, post-double-refutation): Stage-0 Q-budget pre-gate → Stage-1 dynamically consistent
three-phase solve (walls + measured shallow + measured deep void phases, constants fit to
structure data only, k=0 geometry prediction vs the BIC bar). Includes the total-f_v collapse
lemma showing all *kinematic* multi-phase variants are already covered by paper 2's forced-fit
verdict — only the dynamical route remains. Ceiling: success cannot reopen the WP-H2′ tension
verdict (population-independent dilution); it bears only on the dark-energy-free geometry claim.

## 7. WP-N — tensions the model itself introduces

The README's fourth bullet, made concrete: (i) the required-vs-floor shape conflict (paper 2
§4a) if it survives the derived mapping; (ii) the b_pred z-profile vs local survey data;
(iii) anything the forced history breaks that ΛCDM does not (ages, ISW-like integrated effects,
the void catalogs themselves via the self-consistency loop of §4). A tension table with a
"self-inflicted" column is part of the deliverable.

## 8. Decision tree → claims (pre-registered)

- **H-A and H-B both RESOLVE (P3), WP-C consistent at compression level:** headline — *free-history
  timescape predicts the SH0ES–Planck offset as a wall-observer bias, on a void history supplied
  by surveys, with one parameter fewer than ΛCDM.* Better-than-ΛCDM language is used **only** in
  that bounded form; the full title requires WP-B + WP-G to pass, which this paper does not claim.
- **H-A RESOLVES, H-B unavailable/fails:** the mechanism can predict the offset but nature does
  not supply the history — tension claim dies with the telescope; quantified.
- **P3 FAILS on H-A:** the tension claim is dead for this family at the first gate; the paper
  becomes the quantified close-out ("the dressing mechanism cannot produce the required 8.4%
  local bias"), plus the WP-C/WP-N results.
- Any middle outcome: PARTIAL, quantified, with the failing condition named.

## 9. What this paper must not claim

Better-than-ΛCDM overall while WP-C is at compression level and WP-G is gated; theory status
before WP-B; ladder-grade rigor (the flow-correction re-derivation is declared future work).
Every positive phrasing carries the kinematic-reading label until WP-B replaces it.

## 10. Verification & discipline

Adversarial, refute-by-default verification of every headline (fresh agents; independent
re-derivation of b_req and b_pred from scratch); standing controls (P5) in every run;
one number — one committed script — one artifact; parameter-count column in every comparison
table; pre-registered thresholds only (this document is the record); failures reported at the
same volume as successes.
