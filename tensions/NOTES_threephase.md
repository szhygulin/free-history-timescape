# NOTES — three-phase dynamical Buchert solver (WP-B Stage-1 Phase-A)

*Derivation of the dynamically-consistent THREE-PHASE Buchert solver
(`src/probes/threephase_solver.py`), equation-by-equation with primary sources. Written to
be audited against Wiegand & Buchert 2010 (arXiv:1002.3912), Duley–Nazer–Wiltshire 2013
(arXiv:1306.3208; hereafter DNW13), Wiltshire 2009 (arXiv:0909.0749), and the paper-2 tracker
oracle (`free-history-timescape/src/{fit_timescape,timescape_baocmb,harness}.py`) that gate G1
must reproduce. Instantiates `PLAN_WPB_threephase.md` §3.*

**Lapse path used: DERIVED** (DNW13 Eq 16, generalised to the three-phase wall clock). The
declared fallback (hybrid = dynamical background + `LA(f_v_total)` dressing) was **not** needed:
DNW13 Eq 16 gives a clean, non-algebraic, phase-state lapse that reduces to `(2+f_v)/2` on the
tracker (gate G3) and, in the dynamical formulation, needs no fragile derivative of a forced
history (it is read straight off the phase Hubble rates).

The equation FORMS are pinned by the tracker oracle to ~1e-9 (gate G1); the DNW13 equation
NUMBERS are transcribed from the arXiv PDF (`pdftotext -layout`, verified in §9 against the
source text quoted inline).

---

## 0. Symbols and frames

| symbol | meaning |
|---|---|
| `t` | volume-average ("bare", Buchert) time |
| `τ = H̄₀ t` | dimensionless bare time (paper-2 `tau`) |
| `ā(t)` | bare / volume-average scale factor (`ā³ ∝` comoving volume) |
| `a_w, a_s, a_d` | wall / shallow-void / deep-void phase scale factors |
| `H_i = d ln a_i/dt` | phase Hubble rates (bare/volume-average time) |
| `f_i(t)` | current volume fraction of phase `i`; `f_v = f_s + f_d`, `f_w = 1 − f_v` |
| `f_ii` | initial volume fraction of phase `i` (DNW13's `f_vi`, `f_wi`) |
| `α_j²` | CONSTANT curvature parameter of void phase `j` (∝ DNW13 `α² = −k_v f_vi^{2/3} > 0`) |
| `Ω_{m,j}` | phase `j` bare matter content (`8πGρ_{j0}/(3H̄₀²)`); `Ω_{m,j}=0` = empty/Milne |
| `γ̄(t) ≡ dt/dτ_w` | phenomenological lapse (DNW13 Eq 16); `τ_w` = wall proper time |
| `H = (1/a) da/dτ_w` | dressed Hubble |
| `z, D_M, D_H, d_A` | dressed redshift / transverse-comoving / Hubble / angular-diameter distance |

Three disjoint phases, intra-phase homogeneity and isotropy, no shear. Distances dimensionless
(units `c/H̄₀`); overall scale degenerate with the SN offset / BAO `α = c/(H̄₀ r_d)`, both
profiled by the harness. No Λ. Radiation splice (DNW13 §3) is a Phase-B extension; the matter
sector below carries all four Phase-A gates (the tracker oracle itself is matter-only).

---

## 1. Multiscale partition and the Buchert Hamiltonian constraint (Wiegand–Buchert 2010; DNW13 Eqs 8–9, 14)

**Volume partition** (WB10; DNW13 Eq 8, generalised from two to three phases):

- (M1) `ā³ = f_wi a_w³ + f_si a_s³ + f_di a_d³`   — the horizon volume is a disjoint union of
  wall, shallow-void and deep-void regions; `f_ii` = the fraction of the **initial** volume in
  region `i` (DNW13: `ā³ = f_vi a_v³ + f_wi a_w³`, its Eq 8). With `a_i(t₀)=1` (present
  normalisation) and `ā(t₀)=1`, the `f_ii` equal the **present** fractions `f_i0`.
- (M2) `f_i(t) = f_ii a_i³ / ā³`,  `Σ_i f_i = 1`   (DNW13 Eq 9). Differentiating `ln f_i`:
  `ḟ_i = 3 f_i (H_i − H̄)` — hence `Σ_i ḟ_i = 0`.
- (M3) `H̄ ≡ ā̇/ā = Σ_i f_i H_i`   (DNW13 Eq 14). *Proof:* differentiate (M1),
  `3ā²ā̇ = Σ f_ii 3 a_i² ȧ_i` ⇒ `ā³ H̄ = Σ f_ii a_i³ H_i = ā³ Σ f_i H_i`. ∎

**Each phase is an exact FLRW dust+curvature patch** (DNW13: walls flat `⟨R⟩_w = 0`; voids
negatively curved `⟨R⟩_v = 6k_v/a_v²`, `k_v < 0`). Intra-phase homogeneity ⇒ each phase's own
backreaction `Q_i = 0`. Its first integral (Friedmann; DNW13 Eq 10 restricted to one region):

- (M4) `H_i² = (8πG/3) ρ_i − k_i/a_i² = Ω_{m,i} a_i^{-3} + α̃_i² a_i^{-2}`   (`a_i0 = 1`),
  with `α̃_i² ≡ −k_i/H̄₀² ≥ 0` (walls: `α̃_w² = 0`; voids: `> 0`). The **constant** curvature
  parameter (a first integral) is `α_j² = a_j² H_j² − Ω_{m,j}/a_j` (gate G2).

**Multiscale backreaction** (WB10, the key result). For a domain partitioned into subregions,
the global kinematical backreaction is the sum of the internal backreactions plus the
**variance** of the regional expansion rates:

- (M5) `Q = Σ_i f_i Q_i + 6 Σ_i f_i (H_i − H̄)²`.

With intra-phase homogeneity (`Q_i = 0`) this is `Q = 6 Σ_i f_i (H_i − H̄)²`
(`PLAN §3`), a variance — a small fast phase (`f_d` small, `H_d − H̄` large) can carry the
engine while a larger slow phase carries the volume. This is the WP-B rationale.

**The global Hamiltonian constraint is then an IDENTITY of the partition** — the dynamical
feedback that makes the phase split observable (`PLAN §1`). With
`⟨ρ⟩ = Σ f_i ρ_i`, `⟨R⟩ = Σ f_i ⟨R⟩_i`, and each phase obeying (M4)
(`3H_i² = 8πG ρ_i − ½⟨R⟩_i`):

- (M6) `8πG⟨ρ⟩ − ½⟨R⟩ − ½Q = Σ f_i(8πG ρ_i − ½⟨R⟩_i) − 3 Σ f_i(H_i−H̄)²`
        `= 3 Σ f_i H_i² − 3[Σ f_i H_i² − 2H̄ Σ f_i H_i + H̄²] = 3H̄²`. ∎

So evolving each phase by its own Friedmann (M4) and combining by (M1)–(M3) **automatically**
satisfies `3H̄² = 8πG⟨ρ⟩ − ½⟨R⟩ − ½Q` with `Q = 6 Σ f_i(H_i−H̄)²`. The solver never imposes
the constraint; it holds by construction. This is precisely what the FORCED kinematic history of
paper 2 (paper-2 `NOTES_modelv_theory.md` §8, `wpb_integrability.py`) could **not** achieve:
its single-void `α²` drifted 81 %. Here each phase's `α_j²` is exactly constant (gate G2).

---

## 2. The bare background depends only on the total void fraction — but the dynamics *produce* it

A key structural fact (paper-2 §1 collapse lemma, re-examined dynamically). Walls are flat dust,
so **`a_w ∝ τ^{2/3}` exactly, for any void content** (`H_w = 2/(3t)`, `a_w³ ∝ t²`). The volume
identity `f_w = f_wi a_w³/ā³ = 1 − f_v` then gives `a_w³ = (1−f_v) ā³ / f_wi`, hence

- (B1) `ā³ ∝ τ² / (1 − f_v)`,  `ā ∝ τ^{2/3}(1−f_v)^{−1/3}`,   depending on the **total** `f_v(τ)` alone;
- (B2) `H̄/H̄₀ = 2/(3τ) + f_v′/(3(1−f_v))`   (`f_v′ ≡ df_v/dτ`; = `Σ f_i H_i` by (M3), a self-consistency check).

So `ā`, `H̄`, the lapse and the distances are functionals of the **total** `f_v(τ)` only — the
per-phase split does **not** enter them directly. The split becomes observable through a
different door: a genuine (integrable) three-phase **dynamics produces a total `f_v(τ)` that a
single-void dynamics cannot** (§5). The kinematic reading forced `f_v(τ)` to the data; the
dynamical reading asks whether an integrable phase dynamics can *generate* a matching `f_v(τ)`.
The solver therefore integrates the void phases, forms `f_v = f_s + f_d`, and runs the dressed
machinery on it. (Self-consistency of `Σ f_i H_i` against (B2) is checked to ~7e-6, limited by
the `f_v″` spline; reported as `selfcons_Hbar_vs_D1_maxfrac`.)

---

## 3. The DERIVED lapse (DNW13 Eq 16, generalised to the three-phase wall clock)

DNW13 define the phenomenological lapse **generally** (not just on the tracker). The wall
observers (in the denser, spatially-flat wall regions) use wall proper time `dτ_w = dt/γ̄`, with

- (L1) `γ̄ = 1 + f_v (1 − h_r)/h_r`,   `h_r ≡ H_w/H_v < 1`   (**DNW13 Eq 16**),

which is algebraically identical to the **rate ratio**

- (L2) `γ̄ = H̄/H_w`.   *Proof:* `H̄/H_w = f_w + f_v H_v/H_w = (1−f_v) + f_v/h_r = 1 + f_v(1−h_r)/h_r`. ∎

**Generalisation to three phases.** The wall clock is set by the WALL phase (where observers
sit) relative to the volume-average observer. Its lapse is the wall-to-average rate ratio,
which for a general partition reads

- (L3) `γ̄ = H̄/H_w = (Σ_i f_i H_i)/H_w = (f_w H_w + f_s H_s + f_d H_d)/H_w`.

This is the DERIVED lapse the solver uses (`gamma = Hbar/h_w`). It is read directly from the
phase Hubble rates — **no derivative of a forced `f_v`** (the fragility that made paper-2's
rate-ratio "LB" path unstable; paper-2 `NOTES_modelv_theory.md` §3, risk 2). DNW13 also give an
equivalent present-epoch form (their Eq 22) via the density parameters
`Ω̄ = Ω̄_M + Ω̄_R + Ω̄_k = 1 − Ω̄_Q` and `Ω̄_Q = −(1−f_v)(1−γ̄)²/(f_v γ̄²)` (DNW13 Eq 21); (L3) and
the (M6) constraint reproduce it. (L3) is preferred numerically: it is a direct read-off, not a
quadratic root, and needs no `Ω̄_Q`.

**Tracker reduction → `(2+f_v)/2` (gate G3).** In the empty-void two-scale limit (§4) `H_w =
2/(3τ)`, `H_v = 1/τ` (Milne), so `h_r = 2/3` and

- (L4) `γ̄ = 1 + f_v (1 − 2/3)/(2/3) = 1 + f_v/2 = (2 + f_v)/2`   — the Wiltshire tracker value.

Equivalently `Σ f_i H_i/H_w = [(1−f_v)·2/(3τ) + f_v·1/τ] / (2/(3τ)) = (2+f_v)/2`. The solver
reproduces this to `4.4e-16` (G3). **Off the tracker** the derived `γ̄ = H̄/H_w` differs from the
algebraic `(2+f_v)/2` ansatz by up to ~11 % for a matter-differentiated three-phase config —
it is genuinely derived, not the ansatz in disguise.

---

## 4. The tracker limit is the EMPTY-void two-scale exact solution (gate G1)

Collapse the two void phases into ONE empty (`Ω_{m,v}=0`) void: pure negative curvature ⇒
Milne. Walls flat dust. This is the paper-2 / Wiltshire tracker, **exactly and for all time**:

- walls `a_w = (τ/τ₀)^{2/3}` (`H_w = 2/(3τ)`); empty void `a_v = τ/τ₀` (`H_v = 1/τ`; `α_v² = 1/τ₀²`,
  a common big bang at `τ=0`, `a_v→0`);
- (T1) `ā³ = f_w0 (τ/τ₀)² + f_v0 (τ/τ₀)³` ⇒ `f_v(τ) = f_v0 (τ/τ₀)/(1−f_v0+f_v0 τ/τ₀)`,
  which with `τ₀ = (2+f_v0)/3` is byte-for-byte the oracle
  `f_v = 3f_v0 τ/(3f_v0 τ + (1−f_v0)(2+f_v0))` (verified to 3e-15).

The present-day self-consistency `H̄(τ₀)=H̄₀` forces `h_v0 = 1/τ₀ = (1/τ₀)`, matched by the
Milne void automatically (`H̄₀ = f_w0·2/(3τ₀) + f_v0·1/τ₀ = (2+f_v0)/(3τ₀) = 1` at
`τ₀=(2+f_v0)/3`). No free parameter beyond `f_v0`.

Feeding this through §5's dressed machinery reproduces the oracle:

| G1 sub-test | result | target |
|---|---|---|
| `max|D_M/oracle − 1|`, `z∈[1e-3,1100]` | **1.7e-9** | `< 1e-6` |
| SN full-cov `χ²` (`harness.sn_chi2`), `f_v0=0.853` | **1391.545176** (`|Δ|=2.3e-7`) | `1391.545 ± 0.01` |
| non-FLRW `dD_M/dz / D_H − 1` (computed INDEPENDENTLY) | `{.0026,.0249,.0982,.1488,.1752}` | audit targets (`<2e-3`) |
| joint SN+BAO+CMB predict path, `f_v0=0.6426` | `1469.293` | `1469.2926` |

The `dD_M/dz ≠ D_H` signature (`D_H` from the dressed Hubble, `D_M` from the null integral —
never `D_H = dD_M/dz`) is reproduced from first principles, confirming the two are built
independently.

---

## 5. Genuine three phases require matter differentiation; two empty voids are homothetic

With present normalisation `a_j0 = 1` and the physical requirement that voids **vanish early**
(`f_v → 0` as `τ → 0`; DNW13's `f_vi ≪ 1` at last scattering), each void must share the **common
big bang** at `τ=0` (`a_j → 0`). Two *empty* (Milne) voids with a common bang both have `a_v ∝ τ`
⇒ `a_s³/a_d³ = const` ⇒ `f_s/f_d = const`: **homothetic**, collapsing to the two-scale tracker.

A genuine, non-homothetic split therefore requires **matter differentiation**: the two voids
transition EdS→Milne at different epochs, which needs different matter content `Ω_{m,j}`. The
common-bang constraint ties `Ω_{m,j}` to the curvature `α_j²` (given `τ₀`), leaving one dynamical
free constant per void — `α_j²` — plus the fraction `f_j0`. So the four free constants are
`{α_s², α_d², f_s0, f_d0}` (`PLAN §3`), with `Ω_{m,j}` derived:

- (P1) empty/Milne (mechanism CEILING): `α_j² = 1/τ₀²` ⇒ `Ω_{m,j}=0`, `a_j = τ/τ₀`
  (deepest, youngest, maximal contrast);
- (P2) matter void: `α_j² < 1/τ₀²` ⇒ `Ω_{m,j} > 0` derived, the exact open-dust parametric
  solution, common bang. Deep = larger `α_j²` (near ceiling, fast/empty); shallow = smaller
  `α_j²` (more matter, slow).

**Common-bang open-dust solution (exact, closed form).** For `H_j² = Ω_m a^{-3} + α² a^{-2}` with
a bang at `τ=0`, development angle `η`:

- (P3) `a(η) = (Ω_m/(2α²))(cosh η − 1)`,   `τ(η) = (Ω_m/(2α²^{3/2}))(sinh η − η)`.

Present `a=1`, `τ=τ₀` fix `η₀` from `τ₀√α² = (sinh η₀ − η₀)/(cosh η₀ − 1)` and then
`Ω_m = 2α²/(cosh η₀ − 1)`. The map `η(τ)` is a smooth monotone inversion (fine `η`-grid), so
`a_j(τ)` is exact — no ODE stiffness at the bang. The empty limit `α² = 1/τ₀²` (`η₀→∞`,
`Ω_m→0`) recovers `a_v = τ/τ₀`.

*Requirement*: `α_j² ≤ 1/τ₀²` (the common-bang ceiling). `α_j² > 1/τ₀²` would need `Ω_{m,j}<0`
(unphysical) and is rejected. Present-normalised voids with `α² < 1/τ₀²` but **without** the
common bang (an independent bang time) were tried and are unphysical: at high `z` they do not
vanish, the voids wrongly dominate the volume, and the redshift map breaks (`z<0`). Common bang
is mandatory.

The representative three-phase (`α_s²=0.70`, `α_d²=1.20`, `f_s0=0.36`, `f_d0=0.28` ⇒
`Ω_{m,s}=0.203`, `Ω_{m,d}=0.019`) produces the intended picture: the near-empty deep fraction
declines with `z` faster (`f_d: 0.28 → 0.11` over `z=0→2`) than the matter-bearing shallow
(`f_s: 0.36 → 0.31`) — the deep phase is younger / more recently grown, i.e. the variance-
weighted engine of (M5).

---

## 6. Dressed observables (Wiltshire 2009; the paper-2 construction, re-used)

Light propagates through the wall network in wall proper time; the wall ruler is
`a_w ∝ τ^{2/3}` (universal, §2). With `ā`, `H̄`, `γ̄` from the dynamical solve:

- (O1) redshift `1 + z = γ̄ ā₀/(γ̄₀ ā)`   (DNW13, "z+1 = γ̄ā₀/(γ̄₀ā)"; Wiltshire 2009 Eq 37).
- (O2) `d_A(z) = a_w(τ_e) ∫_{τ_e}^{τ₀} dτ/(γ̄ a_w) = τ_e^{2/3} ∫_{τ_e}^{τ₀} dτ/(γ̄ τ^{2/3})`,
  `D_M = (1+z) d_A`   (Wiltshire 2009 Eqs 33/36; paper-2 (E3)/(E4)). On the tracker
  `1/γ̄ = 2/(2+f_v)`, recovering the oracle integrand `2/((2+f_v)τ^{2/3})` exactly.
- (O3) dressed Hubble `H = γ̄ H̄ − γ̄̇`   (**DNW13 Eq 24**, `H = γ̄H̄ − γ̄^{−1}γ̄̇` with the DNW
  dot `= d/dτ_w`, so `γ̄^{−1}dγ̄/dτ_w = dγ̄/dt`; Wiltshire 2009 Eq 27). `D_H = 1/H`, computed
  INDEPENDENTLY of `dD_M/dz`. `γ̄̇` is analytic (`γ̄ = Σ f_i h_i / h_w`, all terms and their
  τ-derivatives known from the phase solve).

On the tracker (O1)–(O3) collapse to `fit_timescape` / `timescape_baocmb` term-by-term (§4).

---

## 7. Algorithm (`solve`)

1. `τ₀ = (2+f_v0)/3` (default; shape is scale-invariant under `τ→λτ`). Hybrid `τ`-grid dense at
   both ends (linspace ∪ geomspace), paper-2 `_tau_grid`.
2. Walls analytic: `a_w=(τ/τ₀)^{2/3}`, `h_w=2/(3τ)`.
3. Void phases: exact common-bang open-dust profiles (P3) (Milne if `α²=1/τ₀²`).
4. Combine: `ā³` (M1), `f_i` (M2), `H̄ = Σ f_i h_i` (M3).
5. Lapse `γ̄ = H̄/h_w` (L3). Redshift (O1). Distance (O2) by `cumulative_trapezoid`.
   Dressed `H = γ̄H̄ − γ̄′` (O3).
6. Recover `α_j²(τ) = a_j²(d ln a_j/dτ)² − Ω_{m,j}/a_j` by an independent central difference
   (integrability self-check, gate G2).

Checkpointable (`solve(..., checkpoint=path)` saves/loads the solved grid). No unbounded inner
loop (analytic phases + one bounded `brentq`); the bracket search carries a wall-clock soft
limit (`THREEPHASE_WALL_SOFT_S`, default 300 s).

---

## 8. Gates (`probes_out/threephase_gates.json`) — ALL PASS

| gate | check | result | threshold |
|---|---|---|---|
| **G1** tracker limit (load-bearing) | empty-void two-scale → oracle | SN `χ²=1391.545176` (`|Δ|=2.3e-7`); `max|D_M/oracle−1|=1.7e-9`; non-FLRW signature reproduced; joint `1469.293` | `χ² ±0.01`; `<1e-6` |
| **G2** integrability | recovered `α_s²,α_d²` constant | drift `2.7e-7 … 5.0e-7`; recovered means exactly `0.70000`, `1.20000`; `selfcons=7.1e-6` | `≤ 1e-6` |
| **G3** lapse limit | derived `γ̄ → (2+f_v)/2` on tracker | `max frac = 4.4e-16`; `γ̄₀ − (2+f_v0)/2 → 0` | machine precision |
| **G4** numerics | step-halving of `D_M(z)`, `H̄(z)` | `dD_M = 1.0e-9`, `dH̄ = 1.2e-9` | `< 1e-4` |

**Verdict: GO for Phase B.** All four gates pass with margin. G1 (load-bearing) reproduces the
paper-2 oracle to `2.3e-7` in SN `χ²`; the solver is a genuine three-phase dynamical Buchert
solver, not the kinematic single-`f_v` closure.

Phase B (NOT done here, gated on these): fit `{α_s², α_d², f_s0, f_d0}` to the measured
`f_s(z), f_d(z)` (paper-2 `telescope_fvobs.json` / `phaseD_fvobs.json`) — the pre-registered
`TRACK-FAIL` vs `Tracks→predict` fork — then, only if it tracks, score SN+BAO+CMB against the
BIC bar. Amendment-A1 physicality strikes (`q_budget.json`): the CEILING (empty voids) is
homothetic (§5), so the genuine engine needs matter-differentiated voids, whose contrast is
capped well below the kinematic requirement — an independent Phase-B headwind, not a Phase-A gate.

---

## 9. Sources

- **Wiegand & Buchert 2010**, *Multiscale cosmology and structure-emerging Dark Energy* =
  arXiv:1002.3912 — multiscale volume partition (M1)–(M3) and the variance backreaction (M5).
- **Duley, Nazer & Wiltshire 2013**, *Timescape cosmology with radiation fluid* =
  arXiv:1306.3208, C&QG 30:175006 — Eq 8 (`ā³ = f_vi a_v³ + f_wi a_w³`), Eq 9
  (`f_v = f_vi a_v³/ā³`), Eqs 10–11 + 13 (bare Friedmann, `α² ≡ −k_v f_vi^{2/3}`, `⟨R⟩`, `Q`),
  Eq 14 (`H̄ = f_w H_w + f_v H_v`), Eq 15 (`h_r = H_w/H_v < 1`), **Eq 16** (`γ̄ = 1 + f_v(1−h_r)/h_r`,
  the DERIVED lapse; = `H̄/H_w`), Eqs 17–22 (bare density parameters; Eq 21 `Ω̄_Q`, Eq 22 present
  `γ̄`), Eq 24 (`H = γ̄H̄ − γ̄^{−1}γ̄̇`), redshift `z+1 = γ̄ā₀/(γ̄₀ā)`. Quoted verbatim in the
  audit trail; the transcription is pinned by the tracker oracle to ~1e-9.
- **Wiltshire 2009**, *Average observational quantities in the timescape cosmology* =
  arXiv:0909.0749 — Eqs 27 (dressed `H`), 33/36–37 (dressed `d_A`, redshift), the tracker.
- **Dam, Heinesen & Wiltshire 2017** = arXiv:1706.07236 App. A — tracker `d_A`, `F(τ)` (the repo
  oracle `fit_timescape.py`).
- **Paper-2 oracle** `free-history-timescape/src/{fit_timescape,timescape_baocmb,harness}.py`;
  `NOTES_modelv_theory.md` (dressed-geometry construction, forced-history 81 % `α²` drift);
  `wpb_integrability.py` (the DNW13 Eq-11 single-void invariant).
