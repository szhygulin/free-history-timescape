# PLAN — submission wave (post-review; 2026-07-08)

*From the review session's read of the combined manuscript (`free-history-timescape.pdf`,
commit fa8ffcd) and the new WP-C / WP-B-integrability / H-B artifacts. Verdict of the review:
the manuscript is sound — every headline number cross-checks against its committed artifact,
the claims-ceiling discipline holds, and the erroneous mechanism wording flagged by
`verify_wpc_sound_horizon` did NOT leak into the tex. Two tex edits were applied directly by
the reviewer (accurate AI-verification disclosure in the acknowledgments; the void-systematics
immunity sentence in §IV A promised by `PLAN_void_content_audit.md`). This plan is the
remaining work, ordered. W1 is the only new computation.*

## W0 — recompile (blocking; the committed PDF is now stale)

Rebuild `free-history-timescape.pdf` from the edited tex (review edits touch the §IV A
measurement paragraph and the acknowledgments). Confirm page count and that Tables I–IV are
unchanged.

## W1 — Seifert-covariance sensitivity row for Part I (the one new computation)

**Why:** the paper's refs [17,18] (Lane et al. 2025; Seifert et al. 2025) are the timescape
school's strongest live argument and rest on their cosmology-independent SN covariance. Part I's
comparisons currently use the Pantheon+ stat+sys covariance only. The likeliest hostile referee
question is: *"does your free-history reconciliation — and the ΛCDM-vs-free-history ordering —
survive under our covariance?"* Paper 1 already reproduced that covariance (task U2); the
machinery exists.

**Compute:** the SN-side (and, where the covariance permits, joint) χ² ladder
{free-history (fitted nodes fixed from Probe R), ΛCDM, tracker} under the Seifert covariance,
at the paper's three z-cuts if cheap. **Pre-registered expectations:** the *reconciliation*
statement (shape-sufficiency: free-history ≤ ΛCDM + 10) is covariance-robust; the
ΛCDM-vs-free-history ΔBIC ordering MAY flip on the SN side (paper 1's stat-only corner
reproduced Seifert's ln B > 5 for timescape) — report either way; the Part II verdicts do not
depend on this row at all (state that explicitly). Adversarial verify mandatory. Then one
sentence + a Table I footnote in §III. **Stop and report the numbers before editing the tex.**

## W2 — artifact hygiene (fix before Zenodo freeze)

1. `tensions/probes_out/wpc_sound_horizon.json`: correct the two mechanism-wording errors per
   its own verify: (a) bare z_eq = 4368 > ΛCDM's 3421 — equality is EARLIER, not "delayed";
   (b) the γ̄₀⁻⁴ radiation suppression does not "partly compensate" — it INFLATES r_d by
   +65.2 Mpc (the largest single sub-effect); the genuine partial compensator is the x_dec
   term. Numbers unchanged; notes only. Re-run the verify to confirm it upgrades to SURVIVES.
2. Repo-relative paths: `wpn_tension_table.json` (17 absolute `/home/szhygulin/...` artifact
   paths) and `verify_wpb_integrability.json` (`notes_source`). Public artifacts must not leak
   machine paths.
3. Standardize the verify verdict key (`verdict_of_verification`) across every `verify_*.json`
   in both repos (final pass on a long-standing nag; keep old keys as duplicates if removal
   would break provenance readers).

## W3 — repo/paper coherence note

`NOTES_modelv_theory.md` §8 quotes the prototype integrability drift (81%, interval measure);
the paper quotes the production value (~152%, full-interior drift, `wpb_integrability.json`).
Add one reconciling sentence to the NOTES so a referee reading both sees measurement-definition,
not contradiction.

## W4 — submission packaging (prepare only; every external action is user-only)

1. Zenodo deposit drafts for `timescape-hubble-tension`, `free-history-timescape` (with
   `tensions/`), and the archived `free-history-timescape-tensions` — DOIs minted at the
   user's push of the button, not before.
2. arXiv metadata drafts for both papers (astro-ph.CO; abstracts within limits); insert
   paper 1's arXiv ID into ref [1] of this paper once paper 1 is posted (ordering: paper 1
   first).
3. Endorsement-outreach shortlist with one-paragraph rationale each, for the user to choose
   from and send personally. Do NOT contact anyone, open no external accounts.

## Order and stop points

W0 → W2 → W3 (autonomous) → W1 (report numbers, wait) → tex integration + W0 again → W4 prep →
STOP. Fleet ≤ 4 agents; adversarial verify on W1; failures at full volume; no external-facing
action without the user.
