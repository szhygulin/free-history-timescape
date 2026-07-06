# EXECUTION STATUS & HANDOFF — free-history timescape program

Central handoff to continue the three-paper program on a fresh (higher-CPU) machine.
Read this, then the plan files, then execute the remaining waves. Written 2026-07-06.

## Repos (all `github.com/szhygulin`, clone as SIBLINGS in one directory)
- **timescape-hubble-tension** — paper 1, FINISHED (merged to `main`).
- **free-history-timescape** — paper 2 (THIS repo, the hub). Strategy: `REASONING_AND_ROADMAP.md`;
  mechanics: `PLAN_void_history.md`; theory: `NOTES_modelv_theory.md`, `NOTES_mapping.md`.
- **free-history-timescape-tensions** — paper 3. Plan: `PLAN.md`.

## Setup on the new machine (do this first)
1. Clone the three repos as siblings.
2. Per repo needing compute (2 and 3): `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
   (numpy 2.0.2, scipy 1.13.1, pandas, matplotlib).
3. **Repoint absolute paths.** Scripts carry origin-machine paths. From EACH repo root:
   `grep -rl "/Users/s/dev/science/<this-repo-name>" src | xargs sed -i '' "s|/Users/s/dev/science/<this-repo-name>|$PWD|g"`
   (affects `los_common.py`, `fetch_twompp.py`, `phaseD_voidfrac.py`, `R2_required_vs_available.py`,
   `adv_*.py`, `mapping_decline_forecast.py`, `make_desi_dr2_rows.py`). Or make them relative.
4. **Re-fetch the 2M++ density field** (135 MB, gitignored — too big for git):
   `cd src && ../.venv/bin/python probes/fetch_twompp.py` → `external_data/twompp_density.npy`.
   Needed for Phase D / the telescope void-fraction work.
5. Sanity gate: `cd src && ../.venv/bin/python probes/modelv_gates.py` → G-T / G-A / G-N must PASS
   (tracker χ²=1391.545178, non-FLRW dD_M/dz≠D_H reproduced).

## Execution discipline (standing rules — non-negotiable)
- **One experiment per subagent** (owns it end-to-end: setup, compute, artifact) + several **adversarial
  reviewers**. Never split one experiment across separate setup/author/compute agents.
- **CHECKPOINT every fit** to its output JSON + a HARD wall-clock soft-limit that exits cleanly and
  resumes — a runaway LB fit burned an hour here. ≤10 agents per wave; phase-by-phase.
- **Commit + push each COMPLETED wave to `main`** before starting the next (do not strand work).
- One number → one committed script → one JSON artifact. Pre-registered thresholds only. Report
  failures at the same volume as successes. Label every positive result KINEMATIC-reading.

## Established — do NOT redo
- Solver gate-validated (tracker limit to 1e-9; non-FLRW dD_M/dz≠D_H). **R1 = RECONCILES**: free f_v(z)
  joint χ² = 1396.06 (ΛCDM 1402.24, tracker 1469.29).
- f_v^req (LA lapse), z={0,.3,.7,1.3,2.33}: **[0.640, 0.531, 0.396, 0.279, 0.194]**.
- **Amplitude-split DISSOLVES** (A_SN ≈ A_BAO+CMB, 0.16σ) — the tracker's 0.85-vs-0.64 split was its
  shape's fault, not a data conflict.
- **b_req = +8.4%** (anchored/global bare-H₀ ratio). Controls: ΛCDM anchored 73.5, tracker 73.0.
- The f_v↔observable **mapping IS derivable** (`NOTES_mapping.md`) — the pre-registration gate for the
  telescope R2. REVIEW it before reducing any telescope data under it.

## Wave 1 status (PARTIAL — finish in wave 1b)
- **DONE:** mapping memo (`NOTES_mapping.md`); paper-1 cross-reference (merged to paper-1 main).
- **BLOCKED — redo:** paper-3 b_pred (`free-history-timescape-tensions/src/probes/bpred_local_excess.py`).
  b_req reproduced (+8.4%) and controls pass, but **b_pred used a FALLBACK**, not the real machinery.
  REDO by generalizing paper 1's actual expansion-variance / local-excess window computation
  (`timescape-hubble-tension .../src/probes/freshH0.py`) to the free history; then issue the P3 verdict
  |b_pred − b_req| (RESOLVES ≤1σ / PARTIAL ≤2σ / else FAILS).
- **FAILED — fix + rerun:** LB rate-ratio Probe R. The `rate_ratio` lapse is added to `modelv_theory.py`
  and its gate passes (`modelV_lb_gate.json`), but the Probe-R fit HUNG on the f_v′ numerical
  instability (see the `_diag_lb_*.py` exploration files). FIX: cap the z↔τ iterations, return `inf` on
  non-convergence so a pathological f_v candidate can't stall the optimiser, hard soft-limit; then run
  the LB Probe R → `modelV_probeR_LB.json`. If LB is genuinely intractable in the kinematic solver,
  bound the lapse-reading systematic another way and report that as the finding.
- **NOT RUN:** the two reviewers (LB band; b_pred/verdict) — were behind the failed LB barrier.

## Remaining waves (in order)
- **WAVE 1b:** fix+run LB Probe R; redo b_pred properly; run the 2 reviewers. Commit+push.
- **WAVE 2** (gated on reviewing `NOTES_mapping.md`): the telescope numbers — BOSS DR12 VIDE / eBOSS
  (DESI if public) void VOLUME filling fractions from survey randoms at z≈0.2–0.7, reduced under the
  derived mapping, replacing the z>0 growth-extrapolation in `phaseD_fvobs.json`; **R2-final** (two-part
  test, roadmap §4b; verdict SUPPLIED / SHAPE-UNAVAILABLE / MAPPING-UNDERIVABLE); **forced
  zero-shape-parameter joint fit** vs the BIC bar + the SN covariance ladder (roadmap §5). Commit+push.
- **PAPER 3** (`PLAN.md`): finish H-A (LB variant + DR2 + z-profile P4); H-B catalog-forced anchored-H₀
  (gated on R2-final); WP-C sound horizon / CMB / BBN; WP-B Reading-B / dynamical consistency;
  WP-G growth / S₈; WP-N self-inflicted tensions. Each per `PLAN.md` pre-commitments; commit+push each.

## First-pass artifacts (PRELIMINARY, unverified — in `probes_out/`)
`phaseD_fvobs.json`, `R2.json` (verdict MARGINAL_top_edge), `phaseF_joint_ampsplit.json`,
`phaseF_freshH0.json` (re-homed to paper 3), `adv_*.json`, `verify_*.json`. Roadmap §6 mandates
adversarial re-derivation of each before any headline enters a paper.
