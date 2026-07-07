# NOTES — Model V theory: general (non-tracker) dressed-distance construction

*Phase 4, `significance-audit`. Derivation + prototype validation for the Model V
"forced void history" dressed geometry. This file is written to be audited
equation-by-equation against the cited primary sources and against the repo's
tracker oracle (`src/fit_timescape.py`, `src/timescape_baocmb.py`).*

**Status of the equation provenance.** The equation FORMS below are validated
numerically against the repo tracker oracle to ~1e-9 relative (the authoritative
check — see §7). The paper equation NUMBERS are provenance pointers obtained by
reading the arXiv sources (Wiltshire 2009 = arXiv:0909.0749; Duley–Nazer–Wiltshire
2013 = arXiv:1306.3208); the verification pass should confirm each number against
the source PDF. Where the CTX/PLAN quotes an identity directly it is cited as
`PLAN §x`.

---

## 0. Symbols and frames

| symbol | meaning |
|---|---|
| `t` | volume-average ("bare", Buchert) time |
| `τ = H̄₀ t` | dimensionless bare time (matches `fit_timescape.py`'s `tau`) |
| `τ_w` | wall-observer ("dressed") proper time |
| `ā(t)` | bare / volume-average scale factor (`ā³ ∝` comoving volume) |
| `a_w(t)` | wall-region scale factor (`a_w³ = (1−f_v) ā³`) |
| `a(τ_w)` | dressed scale factor `= γ̄⁻¹ ā` |
| `H̄ = ā̇/ā` | bare Hubble (dot = d/dt) |
| `H_w, H_v` | wall / void phase Hubble rates (`d ln a_j/dt`) |
| `f_v(t)` | void volume fraction; `f_w = 1−f_v` |
| `γ̄(t) ≡ dt/dτ_w` | phenomenological lapse (Wiltshire09 Eq 13) |
| `H` | dressed Hubble `≡ (1/a) da/dτ_w` |
| `z` | dressed (observed) redshift |
| `D_M, D_H, d_A` | transverse comoving / Hubble / angular-diameter distance |

Two-phase disjoint partition, intra-phase homogeneity, no shear. Distances are
dimensionless (units `c/H̄₀`); the overall scale is degenerate with the SN offset
and the BAO `α = c/(H̄₀ r_d)`, both profiled by the harness.

---

## 1. Exact kinematic identities (safe foundation — `PLAN §1.1`, from Buchert averaging)

- (K1) `H̄ = f_w H_w + f_v H_v`                         (volume-weighted mean expansion)
- (K2) `ḟ_v = 3 f_v (1−f_v)(H_v − H_w)`                 (volume-fraction evolution)
- (K3) `Q = 6 f_v(1−f_v)(H_v − H_w)²`                   (kinematical backreaction)
- (K4) `3H̄² = 8πG⟨ρ⟩ − ½⟨R⟩ − ½Q`                      (Buchert Hamiltonian constraint)
- (K5) integrability: `∂_t(ā⁶ Q) + ā⁴ ∂_t(ā²⟨R⟩) = 0`  (DNW13 Eq 6)

These are identities of the averaging; they do NOT by themselves close the system.

---

## 2. Model V closure decision — KINEMATIC reading (this is the crux)

**Decision: Probe R uses the KINEMATIC (phenomenological) reading.** Force `f_v(z)`
arbitrarily; impose flat-dust walls; compute dressed observables; do NOT enforce the
Buchert integrability condition (K5). This mirrors the free-`E(z)` T1 test
(`probeT1_freeshape.py`): a well-posed ansatz for the expansion history, a cheap
first cut to ask whether ANY `f_v(z)` reconciles SN+BAO+CMB. The dynamically-
consistent reading (integrability enforced, `f_v` then constrained by DNW13 Eqs
10–11) is flagged in §8 as the deeper follow-up.

**Closure postulate (P1) — flat-dust walls in BARE time** (`PLAN §3` / CTX §3):

- (P1) `H_w = 2/(3t)`,  `a_w ∝ t^{2/3}`.

Combined with the volume identity `a_w³ = (1−f_v) ā³`:

- (C1) `ā³ ∝ t² / (1 − f_v(t))`     ⇔   `ā ∝ τ^{2/3}(1−f_v)^{−1/3}`.

A key consequence used throughout: since `a_w³ = (1−f_v) ā³ ∝ τ²`,

- (C2) `a_w ∝ τ^{2/3}` **for ANY forced `f_v`** — the wall scale factor is
  `f_v`-independent. (This is the resolution of the CTX hint: the DISTANCE
  integrand carries `a_w ∝ s^{2/3}`, which looks like the naive `ā ∝ s^{2/3}` but
  is actually the WALL scale factor; the BARE `ā` is `∝ s^{2/3}(1−f_v)^{−1/3}` and
  is used only in the REDSHIFT.)

From (C1) the bare Hubble follows by differentiating `ln ā`:

- (D1) `H̄/H̄₀ = 2/(3τ) + f_v′/(3(1−f_v))`,   `f_v′ ≡ df_v/dτ`.

and (K1)+(K2) give the phase split `H_v − H_w = ḟ_v/(3 f_v(1−f_v))`, hence from (K3):

- (D2) `Q = 2 ḟ_v² / (3 f_v(1−f_v))`   (dimensionless: `Q/H̄₀² = 2 f_v′²/(3 f_v(1−f_v))`).

No extra ODE is needed: the kinematics close on `f_v(t)` alone (CTX §3). The void
density/curvature is a DERIVED consistency output (see §8), not an input.

---

## 3. The lapse γ̄(t): general relation vs the value we adopt

**The general object** is the clock lapse `γ̄ ≡ dt/dτ_w` (Wiltshire09 Eq 13). This
is general; what is NOT general is any algebraic formula for it.

**The tracker value.** On the tracker one finds (validated §7)

- (L0) `γ̄_tracker(τ) = 3(τ+b)/(2τ+3b) = (2 + f_v)/2`,   with `f_v = 2τ/(2τ+3b)`,

so the widely-quoted present value `γ̄₀ = (2+f_v0)/2` (Wiltshire09; `PLAN §2` G-T(c))
is the tracker value of the clock lapse, **not** a general identity.

**Two candidate generalisations** (they COINCIDE on the tracker, see §7B):

- (LA) **Algebraic** `γ̄ = (2 + f_v)/2` — a pure function of the forced `f_v`.
- (LB) **Rate-ratio** `γ̄ = H̄/H_w = 1 + f_v(1−h_r)/h_r`, `h_r ≡ H_w/H_v`
  (DNW13 Eq 16 / Wiltshire09 Eq 14). With (P1), `H_w = 2/(3t)`, this is
  `γ̄ = 1 + τ f_v′/(2(1−f_v))`.

**Decision: adopt the ALGEBRAIC lapse (LA) for Probe R.** Justification:

1. It is a pure function of the FORCED quantity `f_v` (no derivative, no dynamical
   solve) — maximally faithful to the "force the history" philosophy, and
   numerically robust with PCHIP nodes. (LB) differentiates `f_v` and is fragile:
   fed a piecewise-linear `f_v` interpolant it blew up by orders of magnitude in the
   prototype — a practical argument for (LA).
2. It makes the dressed distance the ORACLE formula with `f_v(τ)` generalised
   (§5, (E3)), so the tracker limit is manifest and exact.
3. It reproduces the oracle `D_M, D_H` to ~1e-9 and the SN χ² exactly (§7).

**Why this is a genuine choice, not a fact.** (LA) and (LB) are identical ALONG the
tracker to machine precision (3e-16, §7B), so the tracker gate CANNOT discriminate
them. For a realistic non-tracker history (`f_v0=0.62`) they differ by up to **27%**
at `z≈0.55` — a distance/`D_H`-level ambiguity. **Probe R must therefore run (LA)
as primary and (LB) as a declared systematic**, and neither is the fully consistent
dynamical lapse (DNW13 Eq 22), which is the §8 follow-up.

---

## 4. Dressed relations (lifted, then validated)

- (R1) dressed scale factor: `a ≡ γ̄⁻¹ ā`              (Wiltshire09, after Eq 25)
- (R2) dressed density:      `ρ = γ̄³ ρ̄`, `Ω_M = γ̄³ Ω̄_M`   (Wiltshire09 Eq 26)
- (R3) dressed Hubble:       `H = γ̄ H̄ − γ̄̇`  (dot = d/dt)  (Wiltshire09 Eq 27).
       This exact form is validated to 1e-9 against the oracle (§7). DNW13 Eq 24
       states the same relation in a different time-derivative convention; the two
       are equivalent iff DNW's dot is `d/dτ_w` (so `γ̄⁻¹ dγ̄/dτ_w = dγ̄/dt`) — the
       verification pass should confirm the exact DNW form against the PDF. The
       oracle-validated form (Wiltshire09 Eq 27) is authoritative here.
- (R4) dressed redshift:     `1 + z = (ā₀ γ̄)/(ā γ̄₀)`     (Wiltshire09 Eq 37)

Derivation of (R3) from (R1), for the audit: with `a = γ̄⁻¹ā` and
`d/dτ_w = γ̄ d/dt`, `H = (1/a) da/dτ_w = γ̄ (ā̇/ā) − γ̄̇ = γ̄ H̄ − γ̄̇`. ∎

Substituting (C1) into (R4) and using (LA):

- (E1) `1 + z = (τ₀/τ)^{2/3} · ((1−f_v)/(1−f_v0))^{1/3} · (2+f_v)/(2+f_v0)`.

Substituting (D1) and `γ̄̇/H̄₀ = f_v′/2` into (R3):

- (E2) `H/H̄₀ = (2+f_v)/2 · [2/(3τ) + f_v′/(3(1−f_v))] − f_v′/2`.

On the tracker (E2) collapses to `(4f_v²+f_v+4)/(6τ)` (validated analytically and
numerically), i.e. `timescape_baocmb.H_over_Hbar0`. **`D_H = 1/(H/H̄₀)` is computed
from (E2) directly — never from `dD_M/dz` — so the non-FLRW `dD_M/dz ≠ D_H` signature
(gate G-A) is reproduced from first principles (§7).**

---

## 5. Dressed distance: angular-diameter / transverse-comoving (the integrand)

The dressed comoving distance is built from the null geodesic in the wall geometry.
Wiltshire09 Eq 33 gives `d_A = D/(1+z)`; Eq 36 writes the comoving distance with a
`(1−f_v)^{1/3}` in the denominator of the radial integrand — which is exactly the
wall scale factor `a_w = (1−f_v)^{1/3} ā ∝ τ^{2/3}` (by (C2)). The general integrand
is lapse-weighted over the WALL scale factor:

- (E3) `d_A(z) = a_w(τ_e) ∫_{τ_e}^{τ₀} dτ / (γ̄(τ) a_w(τ))`
        `= τ_e^{2/3} ∫_{τ_e}^{τ₀} 2 dτ / ((2+f_v(τ)) τ^{2/3})`   [using (LA), (C2)]
- (E4) `D_M(z) = (1+z) d_A(z)`.

Note `dτ/γ̄ = H̄₀ dτ_w`, so (E3) is the flat-wall FLRW angular-diameter integral
`d_A = a_w(τ_e) ∫ c dτ_w / a_w` — light propagates through the wall network in wall
proper time. The `f_v`-dependence enters distances ONLY through the lapse `γ̄` in
(E3) and through the redshift map (E1); the wall ruler `a_w ∝ τ^{2/3}` is universal.

For the tracker `f_v(τ)`, (E3) is byte-for-byte `fit_timescape.D_shape_TS`'s
`d_A = τ^{2/3}[F(τ₀)−F(τ)]` with `F′(τ) = 2/((2+f_v)τ^{2/3})` (DHW17 App. A;
Wiltshire09 Eqs 39–40). Model V simply swaps the tracker `f_v(τ)` for an arbitrary
one in the SAME formula.

---

## 6. Algorithm — dressed `D_M(z), D_H(z)` for arbitrary `f_v(z)`

Input: a callable `f_v(z)` on `[0, z_dec]` with `f_v(0)=f_v0` (Probe R: monotone
PCHIP through nodes + a high-z bridge to `f_v→0`).

1. **Scale choice.** Pick `τ₀` (present bare time). The distance SHAPE and all
   `α`-profiled BAO/CMB ratios are invariant under `τ→λτ, τ₀→λτ₀` (scale absorbed
   by the SN offset / BAO `α`), so `τ₀ = (2+f_v0)/3` (the tracker value) is a clean
   default. For the DRESSED-`H₀` OUTPUT only, fix `τ₀` physically by
   `H̄(τ₀)=H̄₀` ⇔ `2/(3τ₀) + f_v′(τ₀)/(3(1−f_v0)) = 1` (D1); then
   `H₀/H̄₀ = (4f_v0²+f_v0+4)/(6τ₀)`, which recovers `g_dress(f_v0)` iff
   `τ₀=(2+f_v0)/3` (the tracker).
2. **τ grid.** `τ ∈ (τ_min, τ₀]`, dense (prototype: 4×10⁵ linspace; integrand
   `∝ τ^{−2/3}` is integrable, but sample finely near `τ_min` for the CMB `z`).
3. **z↔t iteration** (CTX §4). Initialise `z(τ)` with the EdS guess
   `z = (τ₀/τ)^{2/3} − 1`. Repeat: evaluate `f_v(τ) = f_v(z(τ))` (the forced history
   on the current map) → recompute `z(τ)` from (E1) with `f_v0=f_v(z=0)` → until
   `max|Δz| < 1e-6` (a few passes; monotone contraction in practice; prototype
   converged to `<1e-10` in ≤80 iters).
4. **Distance.** `d_A(τ)` by (E3): `integrand = 2/((2+f_v)τ^{2/3})`; cumulative
   trapezoid to get `∫_τ^{τ₀}`; `d_A = τ^{2/3}·∫`; `D_M = (1+z)d_A`.
5. **Hubble.** `f_v′ = df_v/dτ`; `H/H̄₀` by (E2); `D_H = 1/(H/H̄₀)`. (For robustness
   with PCHIP nodes, prefer an analytic `f_v′` via `PchipInterpolator.derivative()`
   composed with `dz/dτ`, not `np.gradient` of a piecewise-linear interpolant.)
6. **Interpolate** `(z(τ), D_M(τ), D_H(τ))` onto the SN `zHD`, the DESI BAO `z`, and
   `z_dec`; feed `D_M` to `harness.sn_chi2` and `predict(z,kind)` to
   `harness.bao_cmb_chi2` (α profiled). `D_V = (z D_M² D_H)^{1/3}`.

**How γ̄ enters (audit summary):** twice, in two DIFFERENT scale factors —
(a) REDSHIFT (E1) via `ā` (bare, carries `(1−f_v)^{−1/3}`) and the ratio `γ̄/γ̄₀`;
(b) DISTANCE (E3) via `1/γ̄` weighting the WALL ruler `a_w ∝ τ^{2/3}`. Conflating the
two scale factors is the easy error the CTX warns about.

---

## 7. Tracker-limit validation (prototype, non-negotiable gate)

Prototype (throwaway, in scratchpad): `proto_modelv.py`, `proto_modelv2.py`,
`proto_modelv3.py`, `proto_modelv4.py`. The general recipe of §6 was fed the tracker
`f_v(z)` (built from the oracle kinematics) and compared to BOTH oracle
implementations over `z ∈ [1e-3, 3]`.

**7A. Distances (both `f_v0`, both oracles):**

| quantity | fv0=0.853 | fv0=0.695 |
|---|---|---|
| `max| D_M,gen / D_shape_TS − 1|`          | 9.9e-10 | 1.0e-09 |
| `max| D_M,gen / timescape_baocmb.DM − 1|` | 2.9e-10 | 3.1e-10 |
| `max| D_H,gen / timescape_baocmb.DH − 1|` | 1.2e-10 | 9.4e-11 |

Worst over all checks **1.0e-09 ≪ 1e-4 required — PASS.**

**SN full-covariance χ² gate (G-T b):** general `D_M` through
`fit_timescape.make_chi2` gives χ² = **1391.545176** at `f_v0=0.853`, matching the
committed reference `1391.545176` to all quoted digits.

**G-A audit regression (non-FLRW signature), `dD_M/dz / D_H − 1` at `f_v0=0.853`:**
computed `{+0.0026, +0.0249, +0.0982, +0.1488, +0.1752}` at `z={0.01,0.1,0.5,1,2}`,
reproducing the audit targets exactly — confirming `D_M` and `D_H` are computed
INDEPENDENTLY (D_M from (E3), D_H from (E2)), never via `D_H = dD_M/dz`.

**7B. Lapse-reading equivalence on the tracker.** Using the exact analytic tracker
`f_v(τ)` and `f_v′(τ)`, `max|γ̄_rate/γ̄_alg − 1| = 3e-16` (both `f_v0`) — (LA) and (LB)
are identical along the tracker. **Off the tracker** (smooth `f_v0=0.62` history)
they diverge by up to **27%** at `z≈0.55` — the size of the lapse-reading systematic
Probe R must carry.

---

## 8. Integrability / dynamical-consistency caveat (quantified)

The kinematic closure (C1) does NOT enforce the Buchert integrability condition (K5).
Demonstration (`proto_modelv4.py`): solve DNW13 Eq 11 (the Buchert `f_v`-evolution
ODE) for the void curvature parameter `α² = −k_v f_vi^{2/3}`, which MUST be constant
for a genuine Buchert solution:

`α²(τ) = (2ā²/(3 f_v^{1/3}(1−f_v))) [ f_v″ + f_v′²(2f_v−1)/(2f_v(1−f_v)) + 3(ā′/ā)f_v′ ]`

- TRACKER `f_v`: `α²(τ)` constant to fractional spread **3.9e-7** — an exact Buchert
  solution (integrability satisfied). This also cross-validates the closure (C1) and
  the Eq-11 transcription.
- FORCED non-tracker `f_v`: `α²(τ)` varies by **81%** — the forced history violates
  the Buchert dynamics/integrability.

  > **Reconciliation with the paper's ~152%.** This 81% is the (now-deleted) prototype
  > `proto_modelv4.py` evaluated on an *illustrative smooth* forced history; the production
  > paper quotes **~152%**, the interior drift (`τ∈[0.1,1.8]`) of the *actual fitted* Probe-R
  > `V.fv_nodes` (`tensions/probes_out/wpb_integrability.json`, adjudicated in
  > `verify_wpb_integrability.json §7`), which in fact drift *more* than the prototype (the
  > specific 81% does not reproduce for the fitted nodes). The gap is a difference of input
  > history, not a contradiction: the load-bearing claim — forced drift ≫ tracker, grossly
  > non-constant — is representation-independent.

**Consequence.** Probe R's forced-`f_v` machinery is the KINEMATIC reading: it yields
a well-posed dressed expansion history and dressed observables, but the implied
`(Q, ⟨R⟩)` do not satisfy (K5), so the void phase's density/curvature is not a
consistent GR fluid. The required `(H_v−H_w)(z), Q(z), δ_v(z)` curves (Probe R
outputs) should be read as "what backreaction the Hubble diagram wants," to be
compared to observed voids — NOT as a proven dynamical solution. The
dynamically-consistent version (impose (K5)/DNW13 Eqs 10–11, then `f_v` is
constrained, lapse via DNW13 Eq 22) is the deeper follow-up if Probe R reconciles.

---

## 9. Risks / open items for the production solver and verification pass

1. **Lapse reading is a theory choice, not fixed by the tracker** (§3, §7B): run (LA)
   primary + (LB) systematic; a 27% off-tracker `D_M/D_H` spread can move the verdict.
   The V0 no-lapse control (`γ̄≡1`) is a third, independent point.
2. **`D_H` needs a smooth `f_v′`** (§6.5): use analytic PCHIP derivatives, not
   `np.gradient` of a piecewise-linear interpolant (which blew up in the prototype).
3. **Paper equation numbers** are from an automated read of the arXiv sources; confirm
   each against the source PDFs in the verification pass. The equation FORMS are
   pinned by the 1e-9 oracle match, which is authoritative.
4. **Dressed-`H₀` normalisation** (§6.1): only the `H₀` OUTPUT needs the physical
   `τ₀` fix; the fit SHAPE does not. Keep these separate to avoid a spurious `H₀`.
5. **High-z bridge** (`f_v→0` by `z~100`) and the r_d splice are unchanged from the
   paper; the CMB point is most sensitive to (LA)/(LB) via the `(1−f_v)^{1/3}` factor
   in `ā` — report the bridge spread.
6. **Integrability** (§8): kinematic reading is deliberate and pre-registered
   (`PLAN §0`, CTX §3); do not silently upgrade to the dynamical reading without
   re-deriving the lapse (DNW13 Eq 22) and re-validating the tracker limit.

## 10. Sources

- Wiltshire 2009, *Average observational quantities in the timescape cosmology*,
  PRD 80, 123512 = arXiv:0909.0749 — Eqs 13, 14, 25–27, 33, 36, 37, 38–40, App. A/B.
- Duley, Nazer & Wiltshire 2013, *Timescape cosmology with radiation fluid* =
  arXiv:1306.3208 — Eqs 4–6, 10–11, 16, 22, 24 (general two-scale system;
  implementation template).
- Dam, Heinesen & Wiltshire 2017, MNRAS 472, 835 = arXiv:1706.07236, App. A —
  tracker `d_A`, `F(τ)`, redshift (the repo oracle `fit_timescape.py` implements this).
- Repo oracle: `src/fit_timescape.py`, `src/timescape_baocmb.py`, `src/harness.py`;
  gates `PLAN_void_history.md §2`, `probes_out/verify_audit_20260706.json`.
