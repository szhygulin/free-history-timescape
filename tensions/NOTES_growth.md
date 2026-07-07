# NOTES_growth.md — WP-G (growth / S₈): scoping memo

*Deliverable for [`PLAN.md`](PLAN.md) §5 (WP-G, gated on WP-B §6). This is a scoping memo, not a
computation: there is **nothing rigorous to compute in the kinematic reading** — no growth
equation exists — so no script and no number is produced here. Producing an FLRW growth number
(`fσ₈ = Ωm^0.55 · σ₈ · D(z)`) would be a fabrication, not a placeholder; §1 states why. The one
growth-relevant quantity that does exist is a **measurement fed in as an input**, not a model
output (§3). Cites: paper-2 strategy [`REASONING_AND_ROADMAP.md`](../free-history-timescape/REASONING_AND_ROADMAP.md)
(§4a below-mean floor, §5 critical path, §6 verdict tree) and the telescope growth measurement
[`probes_out/telescope_fvobs.json`](../free-history-timescape/probes_out/telescope_fvobs.json)
key `PRIMARY_below_mean_measured_growth`.*

---

## Thesis

**In the kinematic reading (a forced/free `f_v(z)` on the two-scale Buchert-averaged background,
with clock-rate dressing but no perturbed metric), no growth observable — fσ₈, RSD β,
weak-lensing/S₈, cluster counts, ISW — is rigorously defined, because there is no perturbation
theory and hence no growth equation on the averaged background.** What paper 2 already measured
(the below-mean field amplitude σ_m(z) via count-in-cells, consistent with an ΛCDM linear-growth
shape) is an **observable used as an input** to the void-fraction mapping, with its z-shape *assumed*
FLRW — it is not a timescape growth prediction. A genuine prediction requires the dynamical reading
(WP-B: perturbing the two-scale system), which is not built. Therefore the **S₈ column of the
tension table is `UNDEFINED-IN-KINEMATIC-READING`** — and that entry is itself the publishable
clarification, and the honest placeholder until WP-B lands.

---

## 1. Why the FLRW growth formula is not an option (the "do NOT fake it" directive, made explicit)

The standard closure `f ≡ d ln δ / d ln a ≈ Ωm(z)^γ` with `γ ≈ 0.55`, and `fσ₈(z) = f(z)·σ₈·D(z)`,
is **inadmissible here** for three independent reasons, each fatal on its own:

1. **No δ, no D, no f.** `D(z)` and `f` are solutions of the linear growth ODE
   `δ'' + (2 + Ḣ/H²) δ' − (3/2) Ωm(a) δ = 0`, which is derived from *perturbing the FLRW metric*.
   The kinematic reading averages the background (Buchert) but never writes a perturbed metric on
   top of that average — so δ, and therefore D and f, are undefined objects, not merely
   hard-to-compute ones. There is no equation to solve.
2. **γ ≈ 0.55 is an FLRW fit, not a law.** The growth index is a fitting form calibrated to the
   FLRW growth ODE across dark-energy models. It carries no meaning on a backreaction-averaged
   background whose expansion is not an FLRW `H(a)`; importing it would smuggle in exactly the
   ΛCDM dynamics the model replaces.
3. **Ωm is convention-ambiguous.** In the timescape framework the matter density parameter splits
   into *bare* vs *dressed* (wall-observer) versions differing by the lapse/dressing (the same
   split that produces the g_dress-vs-full-rate H₀ convention issue, PLAN §3 P1). `Ωm^0.55` has no
   unique value to evaluate. Any number printed would be a choice masquerading as a prediction.

Conclusion: the correct kinematic-reading output for every growth cell is **"undefined — no growth
equation"**, not a plausible-looking FLRW surrogate.

---

## 2. The three readings

- **(a) KINEMATIC reading — the current model.** Two-phase (wall/void) Buchert spatial average with
  a forced or fitted `f_v(z)` and clock-rate dressing. Fixes the *background* expansion history and
  distances (paper 2's R1 = RECONCILES; paper 3's H₀ tracks). It carries **no perturbed metric** on
  the averaged background, so **no growth equation** — growth observables are not defined. Every
  positive result in this reading is phenomenology, labelled as such (PLAN §0 condition 4, §9).
- **(b) AVAILABLE INPUT (not a prediction).** Paper 2 measured the below-mean matter field amplitude
  `σ_m(L = 20 Mpc/h, z)` from BOSS DR12 count-in-cells (bias- and shot-deconvolved), and found it
  consistent with an ΛCDM linear-growth shape `A·D(z)` (§3). This is a *measurement fed into* the
  `f_v ↔ observable` mapping to set the void-fraction z-shape — an **observable-as-input**, whose
  evolution template is itself FLRW. It is not a growth *prediction* of the timescape model and
  cannot be quoted as one.
- **(c) DYNAMICAL reading — WP-B (not built).** With Buchert integrability enforced (the forced
  history violates it — α² drifts 81%; REASONING §2, PLAN §6), growth would follow from
  **perturbing the two-scale system**: multi-phase void populations binned by depth/size, evolving
  depth, wall curvature / phase exchange (Wiegand–Buchert multi-scale), with the lapse derived
  dynamically (DNW13 Eq. 22 generalization). Only in this reading do δ, a growth rate, and hence
  fσ₈/S₈/cluster counts/ISW acquire model-internal definitions. **This is the honest S₈
  placeholder**: the column is defined only after WP-B.

---

## 3. The one input that exists: measured below-mean growth σ_m(z)

Source: `telescope_fvobs.json` → `PRIMARY_below_mean_measured_growth`. Count-in-cells on the BOSS
DR12 galaxy catalog (LOWZ + CMASS), L = 20 Mpc/h cells, five z-shells (z_c = 0.25, 0.35, 0.47,
0.545, 0.615), shot-noise- and bias-deconvolved to the matter-field amplitude σ_m, then fit to the
FLRW linear-growth template `σ_m(z) = A·D(z)`:

- amplitude `A = 0.4443`
- goodness of fit `χ² = 0.879` for `dof = 4` (five shells, one amplitude). **Reduced χ² ≈ 0.22** —
  a good fit; the source note's phrase "reduced χ² ∼ 1" is loose, but the conclusion it draws holds:
  the measured field amplitude is fully consistent with the ΛCDM growth shape.
- measured decline across the span: `σ_m` ratio `= 0.765` over z ∈ [0.25, 0.615], vs the LCDM
  `D(z)` ratio `= 0.829` — mildly steeper than LCDM but absorbed into `A` within the errors.
- dominant systematic: the cross-tracer (LOWZ b≈1.85 vs CMASS b≈2.0) bias step on the absolute
  stitch, carried; the bias values themselves are **external inputs** to the deconvolution.

**What this is, and is not.** It validated the growth-*shape* assumption used for σ(R_s, z) inside
paper 2's void-fraction mapping (route (i) count-in-cells confirming route (ii) LCDM-null σ₀·D(z)).
It is doubly an input: (i) it feeds the mapping rather than emerging from the model, and (ii) its
z-evolution was *fit to* an FLRW template, so it can never be reported as a non-FLRW growth result.
It supplies **no growth rate f** (only an amplitude), so it cannot even assemble fσ₈ or β. It sits
against the below-mean floor `P(δ<0) ≥ 0.5` (REASONING §4a) — the reason only a below-mean amplitude,
not a full growth history, is extractable this way.

---

## 4. Per-observable table

Reading of each cell: **(a)** = rigorous status in the kinematic reading; **(b)** = what data
exists as an *input* (never a prediction); **(c)** = where a real prediction would come from (WP-B).

| Observable | What defining it *requires* | (a) Kinematic reading | (b) Available input (NOT a prediction) | (c) Dynamical reading (WP-B) |
|---|---|---|---|---|
| **fσ₈** | growth rate `f = d ln δ/d ln a` **and** amplitude σ₈ — i.e. a linear δ(z) from a growth ODE | **NONE.** No perturbed metric ⇒ no δ, no ODE, no f. FLRW `f=Ωm^0.55` inadmissible (§1). | Only the σ piece: measured σ_m(L=20,z), and its z-shape is FLRW-fit (§3). **f is wholly absent** ⇒ fσ₈ not assemblable even from data. | f and the growth amplitude from perturbing the two-scale Buchert system; not built. |
| **RSD β = f/b** | growth rate f **and** galaxy bias b | **NONE.** Same missing f. | b enters only as an **external input** (LOWZ 1.85 / CMASS 2.0, used to deconvolve σ_g→σ_m). f absent ⇒ β not assemblable. | f from WP-B; b remains an astrophysical input, not model-predicted. |
| **Weak lensing / S₈** (`S₈ = σ₈√(Ωm/0.3)`) | σ₈ (growth-normalized linear P(k)), a well-defined Ωm, **and** the lensing potential (Φ+Ψ) = perturbed metric | **NONE — this is the S₈ column.** σ₈ needs the growth-normalized linear spectrum (absent); Ωm is bare-vs-dressed ambiguous (§1.3); no lensing kernel without the perturbed metric. | Measured σ_m(z) is a single-scale *below-mean* amplitude, FLRW-shape-fit — not a σ₈, and there is **no lensing kernel** to map it through. | Full WP-B perturbation layer: σ₈-analogue + (Φ+Ψ) on the averaged background. |
| **Cluster counts** (`dN/dz dM`) | halo mass function: σ(M,z) evolved by growth **and** a spherical-collapse threshold δ_c(z) | **NONE.** No growth to evolve σ(M,z); no defined δ_c on a two-phase averaged background. | σ_m(L≈20,z) is a single-scale variance used as input; δ_c and the growth extrapolation are FLRW-borrowed ⇒ mass function not defined in-model. | σ(M,z) + a derived two-phase collapse threshold from WP-B; not built. |
| **ISW** (`∝ ∫ d(Φ+Ψ)/dz dχ`) | time-varying perturbed-metric potentials Φ+Ψ along the line of sight | **NONE.** No perturbed metric ⇒ no potential ⇒ no ISW kernel. (An averaged-model integrated effect is a qualitative flag, §5 — not a growth prediction.) | **None** — no measured proxy is fed into the current pipeline for ISW. | Potentials from WP-B; PLAN §7 (WP-N) lists "ISW-like integrated effects" as a **self-inflicted-tension** candidate to evaluate once the potential exists. |

---

## 5. Consequence for the tension table, and the WP-N flag

- **S₈ / growth column ⇒ `UNDEFINED-IN-KINEMATIC-READING`.** This is the correct, publishable entry.
  It states a real property of the theory at its current maturity — the mechanism fixes the
  *background* (distances, H₀) but has not yet been given a *perturbation* sector — rather than
  hiding that gap behind an FLRW number the model cannot own. It is the honest placeholder PLAN §5
  asks for.
- **Not a claim that the model passes S₈.** `UNDEFINED` is not `CONSISTENT`. Whether free-history
  timescape eases or worsens the S₈ tension is a WP-B question; until then the model neither
  claims the S₈ tension nor is credited with resolving it (PLAN §9: no better-than-ΛCDM breadth
  claim while WP-G is gated).
- **ISW is also a self-inflicted-tension watch item (PLAN §7 / WP-N).** A backreaction-averaged
  expansion with an evolving two-phase partition may generate integrated line-of-sight effects once
  the perturbed potential exists; that is a *risk to check* under WP-B, not a present prediction.
- **The measured σ_m(z) input (§3) travels with the mapping, not the growth column.** It belongs to
  paper 2's `f_v ↔ observable` mapping validation; it must not be re-labelled as a paper-3 growth
  result.

---

## 6. Status / open on WP-B

Every cell above is defined **only** in reading (c). WP-B (PLAN §6) is the gate: derive the lapse
dynamically, enforce Buchert integrability, and build the multi-phase perturbation sector; only then
do fσ₈, β, S₈, cluster counts, and ISW become model-internal predictions and the table's growth
column can be filled with computed numbers (one number → one committed script → one artifact, per
PLAN §10). Until then this memo — and the `UNDEFINED-IN-KINEMATIC-READING` cell — is the WP-G
deliverable.
