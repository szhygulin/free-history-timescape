# PLAN: Model V — void-population-forced backreaction ("right mechanism, wrong parameter set?")

*Phase 4 of the improvement programme. Designed 2026-07-06 on `significance-audit`, after the
full audit + Phase 3. DESIGN ONLY — execution is delegated to a separate session. Read
`PLAN_improve_paper.md` (master plan), `REPORT_los_phase3.md` (Phase 3 verdicts) and
`probes_out/verify_audit_20260706.json` (independent audit) first.*

---

## 0. Hypothesis, and what already constrains it

**H_V:** The timescape *mechanism* (two-phase Buchert averaging + wall/void clock-rate
dressing) is right, but the *tracker solution* — which compresses the whole void history
f_v(t) into one parameter f_v0 — is the wrong closure. Replace it with the **observed void
population**: the measured void volume fraction history f_v(z) (Model V), and the measured
void depth/size *distribution* (Model V2), fed through the same averaging + dressing. Then
re-run the paper's entire SN + BAO + CMB experiment under that theory.

**What is already settled (do not re-litigate):**
- Phase 3 killed the *directional* version of "use the real voids": per-sightline measured
  void content adds nothing (Probe 2 GLS null at −0.9σ vs rotation null; Probe 3 coupling
  Δχ²=0.63, p=0.55, f̂_v0 unmoved; Probe 5 ceiling ≤0.05 mag RMS). Model V is the
  *temporal/amplitude* version — a different, untested dimension.
- T1 rider: a free 5-node spline E(z) fits SN+BAO+CMB jointly at χ²=1391.8 vs ΛCDM 1402.2
  (p=0.065, not significant) vs timescape tracker 1469.3. So a reconciling smooth history
  EXISTS; the tracker isn't it. Model V asks whether the *observed voids* generate it.
- Probe 4: observed watershed void filling today is ≈0.50–0.62 (Williams+2024 NR, Pan+2012
  SDSS); the SN-preferred tracker f_v0=0.85 exceeds every structure-based definition.
- Audit note (verify_audit_20260706, check A4b/A5): in the dressed timescape geometry
  dD_M/dz ≠ D_H (+0.3% at z=0.01 → +17.5% at z=2 at f_v0=0.853) and the distance-effective
  q0 (−0.22…−0.30) ≠ dressed-H q0 (−0.04). Any general solver must compute D_M and D_H from
  their *definitions* independently, never derive one from the other via FLRW identities.

**Prior art to cite and differentiate (positioning, not blockers):** Duley–Nazer–Wiltshire
2013 (arXiv:1306.3208; general non-tracker timescape ODEs — our equation source); Wiegand &
Buchert 2010 (arXiv:1002.3912; multi-scale two-phase partitions — V2's formal basis);
Larena et al. 2009 (arXiv:0808.1161; template-metric observables); Bolejko 2018
(arXiv:1712.02967; emerging spatial curvature from structure, claims H0 shifts *up* toward
73 — the opposite direction to timescape's 61, so the sign of the answer is genuinely open);
Räsänen peak model (arXiv:0801.2692; already in Table III at ΔBIC≈+72). Novelty of Model V:
*observed void catalogs* (not simulations) closing the system, run through Wiltshire's lapse
dressing, inside this paper's uniform SN+BAO+CMB harness, with a pre-registered
required-vs-available comparison.

**Pre-registered decision tree** (write the verdict against these, whatever it is):
- **R1 (mechanism flexibility):** fit f_v(z) as free nodes through the Model-V machinery
  (mirror of the T1 spline fit, but in f_v-space, lapse dressing on). If even the *best*
  f_v(z) cannot approach ΛCDM's joint χ² (χ²_min > χ²_ΛCDM + 25), the mechanism itself —
  not the tracker — is too rigid: **hypothesis dead at mechanism level**; skip Phase F,
  report. (This can happen: the dressing fixes the D_M–D_H relation even with free f_v.)
- **R2 (availability):** if R1 reconciles (χ²_min ≤ χ²_ΛCDM + 10), compare the required band
  f_v^req(z) (Δχ²≤1 node ensemble) against the observed band f_v^obs(z) (Phase D, with the
  definition systematic). Overlap at the most permissive definition edge → **run Phase F
  expecting a live model**. No overlap → quantify the gap in σ and in Q(z) (required vs
  available backreaction) → run Phase F once at f_v^obs as confirmation; verdict
  "**mechanism flexible, amplitude unavailable**".
- **Amplitude split test (always run in Phase F):** with the *shape* of f_v^obs(z) fixed,
  fit one amplitude A (f_v = A·f_v^obs) separately to SN and to BAO+CMB. The tracker analogue
  is the 0.85-vs-0.64 split (4.8–6.6σ). If the split persists in A, the conflict is
  amplitude-level and history-shape-independent; if it dissolves, the tracker *shape* was the
  culprit all along — either way this is the headline robustness number.
- Thresholds (joint χ², 1593 points, vs ΛCDM's 1402.2): ≤ +10 viable; +10…+25 disfavoured;
  ≥ +25 refuted; ≥ tracker's +67 amplitude-dead. Also report ΔBIC with the honest k
  (Model V pure has k=2 nuisances only — *fewer* than ΛCDM's 3 — count it).

---

## 1. Theory specification

### 1.1 Exact kinematic identities (safe foundation — use as written)
For a volume partition into disjoint phases j with volume fractions f_j (Σf_j = 1),
intra-phase homogeneity, no shear:
- ⟨H⟩ ≡ H̄ = Σ f_j H_j  (volume-weighted),
- ḟ_j = 3 f_j (H_j − H̄)  ⇒ two-phase: ḟ_v = 3 f_v (1−f_v) (H_v − H_w),
- kinematical backreaction Q = 6 [ Σ f_j H_j² − H̄² ]  ⇒ two-phase: Q = 6 f_v(1−f_v)(H_v−H_w)²,
- Buchert (dust): 3H̄² = 8πG⟨ρ⟩ − ½⟨R⟩ − ½Q;  3ǟ/ā = −4πG⟨ρ⟩ + Q;
  integrability ∂_t(ā⁶Q) + ā⁴ ∂_t(ā²⟨R⟩) = 0. ā³ ∝ comoving volume; f_v = f_vi a_v³/ā³.

### 1.2 Model V closure (the new theory)
Keep walls = spatially flat dust regions, voids = underdense (depth from data, §4; empty
Milne as a limiting variant), and keep Wiltshire's wall-observer dressing (lapse γ̄(t),
dressed redshift/distances). **Discard the tracker attractor. Close the system with the
observed f_v(z) instead:** given f_v(t) (after the z↔t iteration below), H_v − H_w is fixed
by the exact identity ḟ_v = 3f_v(1−f_v)(H_v−H_w); the wall Friedmann equation fixes H_w;
Q and ⟨R⟩ follow from §1.1 + integrability. The void-phase density/curvature is then
*derived*, and must be checked against the observed void depths (consistency output, not
input, for Model V; input for V2).

**Lapse and dressed observables — lift, do not re-derive:** take the general (non-tracker)
lapse construction and the dressed z, D_M, D_H definitions verbatim from Wiltshire 2009
(arXiv:0909.0749, App. B, eqs. B1–B8 general; tracker limit γ̄ = (2+f_v)/2) and
Duley–Nazer–Wiltshire 2013 (arXiv:1306.3208, §2 — they solved the general system numerically
with matter+radiation; that is the implementation template). D_H ≡ c/H_dressed per the
DHW17 convention the paper already uses. **Write every adopted equation with its source into
`NOTES_modelV_theory.md`** so the verification pass can audit the derivation chain.

**z↔t iteration:** f_v^obs is measured against redshift; the solver runs in volume time.
Iterate: solve with f_v(t) guess → get z(t) → re-grid f_v^obs(z(t)) → resolve. Converges in
a few passes (distances change <5% between iterations); gate on |Δz(t)| < 1e-6.

**High-z bridge (declared systematic, two variants):** catalogs constrain f_v only at
z ≲ 0.8. Above the data: (V-a) tracker-shaped f_v(t) matched in value+slope at the last data
point; (V-b) simulation-shaped growth curve (Williams+2024 NR evolution) matched the same
way. Run both; the bridge choice must not drive the verdict (it mostly affects the CMB
point; report the spread). f_v → 0 by z ~ 100 in both; above that the model is flat
dust+radiation FLRW — splice the analytic early solution, r_d stays the external standard
ruler exactly as in the paper (α = c/(H̄0 r_d) profiled).

### 1.3 Variants
- **V0 (no-lapse control, cheap):** same forced f_v(z), γ̄ ≡ 1 (pure Buchert, no clock
  dressing). Isolates how much of the fit the lapse mechanism does.
- **V (primary):** two-phase + lapse, as above.
- **V2 (the literal "exact observed distribution"):** N void phases binned by observed
  depth/size (§4), one wall phase; identities of §1.1 in multi-phase form. Tests whether the
  *distribution's variance* (Q grows with Var(H) across phases) buys expansion history beyond
  the two-phase mean. Pre-registered expectation: the deep-void tail raises Q at low z;
  quantify ΔQ(z) = Q_V2 − Q_V and its Δχ².
- Parameter count: pure V/V2 have zero fitted cosmological parameters (all forced) + the two
  standard nuisances (SN offset, BAO α). The amplitude variant adds A (one parameter).

---

## 2. Validation gates (all must PASS before any result is quoted)

- **G-T (tracker limit):** force f_v(t) = tracker history (fit_timescape.py eq., f_v0=0.853)
  through the general solver. Must reproduce: (a) D_M(z) from `fit_timescape.D_shape_TS` /
  `timescape_baocmb.DM` to <1e-6 relative over z ∈ [1e-3, 1100]; (b) the committed SN
  χ² = 1391.545 ± 0.01 at f_v0 = 0.853 (full stat+sys covariance, offset marginalised);
  (c) γ̄0 = (2+f_v0)/2 = 1.4265; (d) H0_dressed/H̄0 = (4f_v0²+f_v0+4)/(2(2+f_v0)) — at
  f_v0=0.695 this is 1.2295 (Duley+2013 Table 1: H̄0=50.1 → H0=61.6).
- **G-A (audit regression):** reproduce the non-FLRW signature on the tracker:
  dD_M/dz / D_H − 1 = {+0.0026, +0.0249, +0.0982, +0.1488, +0.1752} at
  z = {0.01, 0.1, 0.5, 1.0, 2.0}, f_v0 = 0.853 (from `verify_audit_20260706.json`). This
  proves D_M and D_H are computed independently from definitions.
- **G-N (numerics):** ODE tolerances such that halving steps moves joint χ² by <0.01;
  integrability condition residual <1e-8 along the solution; energy bookkeeping closed.
- **G-P3 (Phase-3 consistency):** whatever Model V predicts for per-sightline scatter must
  respect the Phase 3 nulls (LOS modulation ≤0.05 mag RMS at z<0.067). V/V0 are isotropic —
  trivially pass; V2 must check.
- **G-D (data sanity, Phase D):** f_v^obs(z=0) must land inside the Probe 4 bracket
  [0.50, 0.62] under the watershed mapping before any cosmology is run against it.

---

## 3. Probe R — required void history (run FIRST; cheap; decides continuation)

Mirror `src/probes/probeT1_freeshape.py`, but parametrise **f_v(z)** (not E(z)):
monotone PCHIP nodes at z = {0, 0.3, 0.7, 1.3, 2.33} (5 free values in [0,1)) + the V-a
bridge; observables through the full Model-V dressed machinery; joint SN (full cov, offset
profiled) + DESI BAO + Planck CMB point (α profiled). Optimise with ≥40 Nelder–Mead restarts
+ one differential-evolution run (the T1-verified recipe); LCDM-matched and tracker-matched
starts must be among the restarts (sanity anchors).

Outputs → `probes_out/modelV_probeR.json`:
- χ²_min, per-dataset decomposition, H0 (from α), vs the committed references
  {ΛCDM 1402.2, w0waCDM ~1398.5, tracker 1469.3, free-E(z) 1391.8};
- f_v^req(z) best nodes + Δχ²≤1 ensemble band (profile each node);
- the implied (H_v−H_w)(z), Q(z), and derived void depth δ_v(z) — the "required
  backreaction" curves;
- R1 verdict per §0 thresholds. **If R1 kills the mechanism, stop after writing the report
  section — that is a complete, publishable answer.**

Also run Probe R with γ̄≡1 (V0 machinery) — separates "lapse needed" from "any Buchert
two-phase suffices".

## 4. Phase D — the observed void population f_v^obs(z) and its distribution

Deliverable: `probes_out/modelV_fvobs.json` — z-grid, f_v^obs central + band, per-point
provenance, plus the binned depth/size distribution for V2. Keep raw catalogs OUT of the
repo in git-ignored `external_data/` with a committed fetch script per source
(`src/probes/fetch_voidcats.py`), following the Phase-3 `fetch_twompp.py` pattern.

Sources (acquisition ladder; take what exists, record what doesn't):
1. z ≲ 0.067: compute the watershed-like void filling fraction directly from the already
   acquired 2M++ field (`external_data/twompp_density.npy`, `src/probes/los_common.py`
   loaders) — threshold family δ < {−0.3,−0.5,−0.7} plus a watershed-style basin variant;
   this anchors z≈0 with our own systematics control.
2. z ≈ 0.02–0.11: SDSS DR7 VoidFinder/V² catalogs (VAST project, Douglass et al.;
   Pan et al. 2012 arXiv:1103.4156 — published filling fraction 62%).
3. z ≈ 0.2–0.7: BOSS DR12 VIDE voids (Mao et al. 2017, arXiv:1602.02771 — public); eBOSS
   voids (Aubert et al. 2022, arXiv:2007.09013). Filling fraction = Σ void volumes / survey
   volume from the survey random catalogs — if a published filling fraction exists, prefer
   it; else derive and document.
4. Optional if public by execution time: DESI DR1/DR2 void catalog.
5. Fallback if catalog-derived fractions prove unreliable: simulation/NR-calibrated *shape*
   (Williams+2024) anchored to the z≈0 observed value, with a widened band. Declare which
   path was used per z-point.

Systematics to carry explicitly (dominant first): (i) **definition mapping** (watershed vs
below-mean vs timescape's "faster-than-mean-expanding" f_v) — carry as a two-sided band, run
R2 at the most permissive edge (Probe 4 logic); (ii) fiducial-cosmology volume conversion in
catalogs (ΛCDM-assumed; model-V-vs-ΛCDM distance differences <5% here → f_v shift ≲7%, add
in quadrature); (iii) survey completeness/mask; (iv) tracer bias (galaxy voids vs matter
voids — use the Carrick δ_g*→δ mapping experience from Phase 3). Depth distribution for V2:
stacked void density profiles (e.g. Hamaus-type) or per-catalog mean enclosed contrast.

## 5. Phase F — repeat the paper's experiment under Model V

"The whole experiment" = the paper's decision-grade numbers, each with a Model-V analogue
artifact (`probes_out/modelV_*.json`), leaving the family Table III alone:

1. **SN covariance ladder** (Table I analogue): standard m_b_corr × {full, diagonal} and
   Tripp-marginalised × full — χ²/ΔBIC vs ΛCDM *and* vs tracker timescape. Lesson from the
   paper: quote the full-cov cell as headline, show the ladder. → `modelV_sn.json`
2. **BAO+CMB (DESI DR2 primary, DR1 sensitivity row)** via the existing
   `src/probes/dr2.py` machinery with Model-V distances: χ²/dof, ΔBIC, H̄0 and dressed H0.
   → `modelV_baocmb.json`
3. **Joint fit** (analogue of `joint_w0wa.py`): pure V (k=2), amplitude variant (k=3), V0,
   V2; ΔBIC vs {ΛCDM, w0waCDM, tracker}; the amplitude split test A_SN vs A_BAO+CMB with
   profile-likelihood σ (fvsplit.py machinery). → `modelV_joint.json`
4. **Fresh calibrator-anchored H0** (freshH0.py machinery): does Model V move the anchored
   late-time H0 off 73, and what does it predict for the local-excess window (the paper's
   17–22% δ_req check)? → `modelV_freshH0.json`
5. **Dressed-H0 story**: early(CMB-side) vs late dressed H0 under Model V — does the
   "both ≈ 61" recasting survive a data-driven f_v history, and where does the pure-V H0
   land (Bolejko direction up, or timescape direction down)? → part of `modelV_joint.json`.
6. **Stretch (optional, only if everything above is green):** rerun the Seifert
   cosmology-independent-covariance comparison (src/probes/seifert.py; re-fetch Zenodo
   doi:10.5281/zenodo.12729746, ~650MB, keep out of repo) with Model V distances at the
   three z-cuts. Expectation on record: the low-z preference structure is covariance-driven
   and should persist for any smooth history.

## 6. Verification pass (mandatory before any number is believed)

Adversarial re-derivation, same protocol as Phases 1–3 (fresh agents, refute-by-default):
(i) audit `NOTES_modelV_theory.md` equation-by-equation against the cited sources;
(ii) re-derive G-T/G-A gates independently; (iii) re-implement the Probe-R χ² for ≥3 node
vectors from scratch; (iv) spot-check f_v^obs points against raw catalogs; (v) re-run the
amplitude split with an independent statistic (Gaussian gap cross-check as in dr2.json).
No headline enters the report un-verified. Rotation-null lesson from Phase 3 applies to any
directional claim V2 makes.

## 7. Reporting & integration

`REPORT_void_history.md`: decision-tree verdict up front (R1/R2/amplitude split), then the
required-vs-available figure — f_v^req(z) band vs f_v^obs(z) band — as the headline plot
(`fig_fvhistory.pdf`), then the Phase-F table. Paper integration is a USER decision after
results: (a) new section "Is any void history enough?" in this paper, or (b) companion paper
("Can the observed void population supply the backreaction the Hubble diagram wants?").
Do not edit the tex without that decision.

## 8. Delegation notes (execution session, read first)

- Worktree: `.claude/worktrees/significance-audit` (branch `significance-audit`); venv:
  `.venv` at worktree root (numpy 2.0.2 / scipy 1.13.1) — reuse, don't rebuild. Run scripts
  from `src/` (data paths are relative); probes import via `los_common.py` absolute paths.
- Committed reference numbers for gates: SN full-cov χ²_TS(0.853)=1391.545,
  χ²_ΛCDM(0.3333)=1386.407 (results.json); joint {ΛCDM 1402.2, tracker 1469.3, free-E
  1391.8} (results_joint.json, probeT1_freeshape.json); DR2 rows in probes_out/dr2.json.
- Order: theory module + gates → Probe R (→ STOP/GO per §0) → Phase D (can start in
  parallel with Probe R; it does not depend on R) → Phase F → verification → report.
- Fleet ≤10 agents total per run; phase-by-phase dispatch with a parent review between
  phases; surface the planned agent count before launching. Suggested: 2 (theory+gates,
  Opus-tier) → 1 (Probe R) → 2 (Phase D, Sonnet-tier) → 2 (Phase F) → 3 (verification).
- Long runs: the harness reaps background jobs ~20 min — checkpoint intra-run (the
  `seifert.py` SEIFERT_MAXSEC pattern) and make every script resumable.
- One-number-one-artifact rule: every quoted number must have a committed generating script
  and a `probes_out/*.json` artifact. Version-control scope: commit locally on
  `significance-audit`; do NOT push or open PRs without user instruction.
- Compute budget guess: theory module is ODE work (fast); Probe R ~1e5 χ² evals — hours,
  checkpoint; Phase D is mostly I/O + bookkeeping; Phase F reuses existing fit machinery.
  External data ~50–200 MB (void catalogs) + optional 650 MB (Seifert stretch).
