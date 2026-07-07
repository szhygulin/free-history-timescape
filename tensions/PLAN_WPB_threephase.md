# PLAN — WP-B execution: the three-phase (depth-resolved) dynamical test

*Drafted 2026-07-06 by the program's review session, after the double refutation
(paper 2 `SHAPE-UNAVAILABLE` + `FORCED_FVOBS_FAILS_BIC_BAR`; paper 3 WP-H2′ `FAILS`).
This instantiates [`PLAN.md`](PLAN.md) §6 (WP-B) — the last untested door. **Pre-registered:
no Stage-0 or Stage-1 number has been computed at the time of this commit.** The §1 lemma and
the Stage-0 result are paper-2 discussion material (pointer in its `REASONING_AND_ROADMAP.md`);
the Stage-1 test and its verdict belong to this paper.*

---

## 0. Question, ceiling, and prior

**Question.** The two-phase tests refuted a conjunction: the data demand voids that are
*abundant today* (level 0.64, for the dressed geometry) *and young* (×3.3 volume growth since
z ≈ 2, for the backreaction engine), while every single-threshold reading of the sky gives
abundant-but-old (below-mean: level ✓, growth ✗) or young-but-scarce (deep: growth ✓, level ✗).
A **three-phase** structure — walls + abundant slowly-evolving *shallow* voids + a small
fast-growing *deep* phase — is the one configuration that could satisfy both at once, because
the backreaction Q = 6 Σ f_i (H_i − H̄)² is variance-weighted: a small phase with large ḟ can
carry the engine while the shallow phase carries the volume. Both ingredient histories are now
**measured** (the threshold family in paper 2's `telescope_fvobs.json` / `phaseD_fvobs.json`).

**Ceiling (pre-registered).** Even full success does **not** reopen the Hubble-tension claim:
the WP-H2′ dilution verdict is population-independent (the SH0ES Hubble-flow SNe sit beyond the
homogeneity scale whatever the phase structure inside it). Success rescues only the
dark-energy-free *geometric* claim, and would then face WP-C (full CMB). Failure completes the
§6 falsifier and closes the mechanism at the population level — the strongest available close.

**Prior on record:** ~5% that Stage 1 clears the BIC bar. For: a strategy-session
order-of-magnitude estimate puts the available Q from the measured deep phase within a factor
~1–2 of the requirement at z ≈ 0. Against: the same estimate falls short ×3–5 at z ≈ 0.3–0.7;
and the two-phase forced history violated integrability by 81%, so the dynamics may simply
refuse to track the measured curves. Stage 0 settles the budget question cheaply before any
solver is built.

## 1. Why there is no kinematic three-phase test (lemma — route to paper 2)

**Lemma (total-f_v collapse).** Under the kinematic dressing closure — flat-dust walls
(P1: H_w = 2/(3t), a_w ∝ t^{2/3}) and either committed lapse ansatz (LA: γ̄ = (2+f_v)/2;
LB: γ̄ = H̄/H_w) — the dressed observables z(t), D_M(z), D_H(z) depend on the phase
decomposition **only through the total void fraction** f_v(t) ≡ Σ_voids f_i(t).
*Proof sketch:* the volume identity gives ā³ ∝ t²/(1−f_v) (C1) and
H̄ = 2/(3t) + f_v′/(3(1−f_v)) (D1), both functions of the total; the redshift map (E1), the
dressed rate (E2), and the distance integrand (E3) use only ā, a_w, γ̄, f_v′ — and both lapse
ansätze are functions of the total alone. The per-phase split enters **only** Q and ⟨R⟩, which
the kinematic reading never uses. ∎

**Corollary.** Every multi-phase *kinematic* model whose total below-mean history matches the
measurement is observationally identical to the already-run forced fit
(`forced_joint_fit.json`: χ² = 2213.7 DR1 / 2845.6 DR2 vs bar ≈ 1409.6 / 1407.2 —
`FORCED_FVOBS_FAILS_BIC_BAR`). The two-phase verdict therefore already covers **all** kinematic
depth-resolved variants; "your closure was too crude" is not an open rebuttal at the kinematic
level. The only live route is **dynamical**: with Buchert integrability enforced, Q feeds back
into H̄ through the Hamiltonian constraint (3H̄² = 8πG⟨ρ⟩ − ½⟨R⟩ − ½Q), and the phase split
becomes observable. → Paper 2 discussion, sealing `SHAPE-UNAVAILABLE`; the dynamical test below
belongs here (paper 3).

## 2. Stage 0 — the Q-budget pre-gate (hours; go/no-go)

Compute the **available** backreaction analytically from the measured curves, using only exact
kinematic identities (no solver):

- Phases from the measured threshold family (matter-mapped): deep d = {δ_m < −0.5} (primary;
  δ_m < −0.3 as declared sensitivity), shallow s = {below mean} − d, walls w = the rest. Carry
  the r100/r200 reliable-volume band and the tracer-bias band (LOWZ 1.85 / CMASS 2.0, ±10%,
  as in `telescope_fvobs.json`) into every curve.
- H_i − H̄ = ḟ_i / (3 f_i) from the measured f_i(z) (τ(z) from the committed background;
  smooth/band the derivatives — no np.gradient of linear interpolants);
  Q_avail(z) = 6 Σ f_i (H_i − H̄)².
- Compare against the committed kinematic requirement Q_req(z)
  (`modelV_probeR.json → derived_backreaction_V`: Q/H̄₀² ≈ 0.37 / 0.84 / 1.28 at
  z = 0 / 0.3 / 0.7; smooth the PCHIP jitter and band it with the LA node band).
- **Declared caveat:** both sides are reading-dependent (Q_req is the *kinematic-forced*
  requirement; the dynamical requirement differs because Q back-reacts on H̄). Stage 0 is an
  order-of-magnitude go/no-go, not a verdict.

**Pre-registered gate:** if Q_avail(z) < Q_req(z)/3 across all of 0.3 ≤ z ≤ 1.0 (band edges at
their most permissive), report **NO-GO** and stop — the measured population cannot plausibly
supply the engine even before dynamics, and the close-out is written from Stage 0 alone.
Otherwise **GO** to Stage 1. Either outcome → `probes_out/q_budget.json` (+ adversarial verify).

## 3. Stage 1 — the dynamically consistent three-phase solve (the test)

**Model.** Walls: flat dust. Two void phases, each a genuine Buchert phase with **constant**
curvature parameter (α_s², α_d² — constancy is what an exact solution requires; the 81% drift
of the forced two-phase history is exactly what this construction must not reproduce). Dust +
radiation era spliced per DNW13. No Λ. Formal basis: Wiegand & Buchert 2010 (arXiv:1002.3912)
multi-scale partitions; Duley–Nazer–Wiltshire 2013 (arXiv:1306.3208) as the implementation
template. **Lapse: derived** (DNW13 Eq 22 generalized to the three-phase wall clock), not
assumed — this is the plan's main theory task. *Declared fallback if the derivation stalls:*
hybrid = dynamical background (Q feeding H̄ via the constraint) + LA(f_v_total) dressing,
labeled an approximation, all gates still required.

**Free constants (constants, never functions):** {α_s², α_d², f_s(z_i), f_d(z_i)} at
z_i ≈ 100, plus the standard profiled nuisances (SN offset, BAO α). **Fit them to the measured
structure curves only** — f_s(z), f_d(z) with their full bands — never to SN/BAO/CMB.

**Void-content variants (amendment A1, pre-registered before Stage 0 — voids are not empty):**
Galaxy-traced voids overstate matter emptiness (bias); stacked void *lensing* measures the total
matter and finds central underdensities δ_m ≈ −0.3…−0.5 for typical watershed voids (deepest
large-void interiors ≈ −0.8), never the Milne δ = −1 the base model assumes. Since matter inside
voids slows them, the empty-void configuration is the mechanism's mathematical **best case**.
Therefore:
- **CEILING variant** (base plan): empty (Milne) void phases — maximal contrast per unit volume.
  If the CEILING fails the bar, every physical variant fails *a fortiori* — stop there.
- **PHYSICAL variant**: void-phase matter content Ω_m^(i) pinned by lensing-calibrated profiles
  (fixed, not fit). Run only if the CEILING clears the bar.
- **Stage-0 sensitivity row**: recompute Q_avail with lensing-calibrated depths (expected to
  shrink it ~2–4×) so the go/no-go states both the ceiling and the physical budget.
- Consistency note for the write-up: the committed kinematic requirement already demands
  (H_v−H_w)/H̄₀ ≈ 0.5 at z = 0, which only near-empty voids can supply — real void contents cap
  the contrast well below this; state it as an independent physicality strike.

**Gates (all must PASS before any cosmology is scored):**
- **G1** two-phase empty-void limit reproduces the tracker oracle (SN χ² = 1391.545 ± 0.01;
  distances < 1e-6; the non-FLRW dD_M/dz ≠ D_H signature).
- **G2** α_s²(t), α_d²(t) constant along the solution to ≤ 1e-6 fractional (integrability
  self-check at the two-phase 3.9e-7 standard).
- **G3** the derived lapse reduces to (2+f_v)/2 on the tracker limit to machine precision.
- **G4** numerics: step-halving moves the joint χ² by < 0.01.

**Pre-registered outcomes:**
1. **TRACK-FAIL** — no constant set tracks the measured f_s(z), f_d(z) within their bands
   (report the best-achievable tracking residual in band-widths). The mechanism is closed at the
   population level; the §6 falsifier fires. This is a complete, publishable close-out.
2. **Tracks → predict.** The geometry is then a zero-cosmology-parameter prediction: score
   joint SN + BAO(DR2 primary, DR1 row) + CMB against the BIC bar. **Accounting (pre-registered):**
   primary k = 0 — the four constants are pinned by structure data (external-calibration status,
   like Ω_b from BBN), with their band-widths propagated as a systematic on the predicted χ²;
   sensitivity row k = 4 reported alongside. Bar (primary): χ² ≤ χ²_ΛCDM + ln N ≈ 1407.2 (DR2).
3. **If the bar is cleared:** recompute this paper's b_req under the three-phase background and
   re-state the tension section; the WP-H2′ b_pred verdict stands regardless (§0 ceiling).
   Then WP-C is mandatory before any claim language.

Also run the T1-style diagnostic ceiling: the same dynamical machinery with the constants freed
to fit the cosmology data — reported as a ceiling only, never a claim.

## 4. Verification (mandatory, before any number is quoted)

Adversarial, refute-by-default, fresh agents: (i) equation-by-equation audit of the three-phase
system against Wiegand–Buchert 2010 / DNW13 (every adopted equation with its source, NOTES
style); (ii) independent recompute of Stage-0 Q_avail(z) and of the Stage-1 joint χ² for ≥ 2
constant sets through a from-scratch harness path; (iii) gates re-run in the parent session's
own hands. Standing controls (ΛCDM and tracker reproduction) in every run.

## 5. Artifacts & paper routing

Stage 0 → `probes_out/q_budget.json` (+ `verify_q_budget.json`). Stage 1 →
`probes_out/threephase_dynamics.json` (tracking + gates), `threephase_forced_geometry.json`
(the prediction + BIC verdict), + verifies. The §1 lemma and the Stage-0 number go into
paper 2's discussion (its roadmap §5 item 8 points here). Outcome 1 or a failed bar closes
paper 3 §6 and strengthens paper 2's verdict; outcomes 2+3 with a cleared bar become paper 3's
geometry headline and trigger WP-C.

## 6. Budget & fleet

Stage 0: one agent, hours, committed data only. Stage 1: theory memo + solver (≤ 2 agents) →
gates → structure-fit + one geometry evaluation (1 agent) → adversarial verification (2 agents).
≤ 6 agents total, phase-by-phase with a parent review between stages; every long fit
checkpointed and resumable.
