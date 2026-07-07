# Submission prep — Paper II (free-history timescape) + Paper III (tensions)

**Prepare-only.** Every external action below (minting DOIs, posting to arXiv, contacting
anyone) is the author's to perform. Nothing here has been submitted.

Submission **ordering** (hard dependency): post **Paper I first**, obtain its arXiv ID,
insert that ID into this paper's ref `[1]` (`\bibitem{Paper1}`), rebuild the PDF, *then*
post this paper. Zenodo DOIs can be reserved in parallel but are minted on the author's
"publish" click.

---

## 1. Zenodo deposit — `free-history-timescape` (code + data record for Paper II, incl. Paper III under `tensions/`)

- **Upload type:** Software
- **Title:** Free-history timescape: code and data (Paper II + Paper III)
- **Authors / creators:** Zhygulin, Viacheslav
- **License:** MIT
- **Version:** v1.0
- **Description (draft):**
  > Analysis code, committed numerical artifacts, and manuscript for *"Free-history
  > timescape: a data-driven void history reconciles the supernova–BAO–CMB geometry but
  > does not resolve the Hubble tension"* (Paper II), with Paper III (the cosmological-
  > tensions close-out) nested under `tensions/`. Freeing the void volume-fraction history
  > f_v(z) from the timescape tracker attractor (Buchert two-phase averaging + wall/void
  > clock-rate dressing) reconciles Pantheon+, DESI BAO, and the Planck acoustic scale at a
  > physically plausible present void fraction, but the reconciliation is kinematic
  > (violates Buchert integrability) and covariance-dependent, and four independent
  > availability/dynamics/calibration tests each fail. Every headline number is reproduced
  > by a committed probe script + JSON artifact and cross-checked by an adversarial
  > verification pass. Reproducibility: Python 3.12, numpy 2.0.2 / scipy 1.13.1; large
  > external inputs (2M++ field; Seifert P+1690 covariance, Zenodo 12729746) are fetched,
  > not redistributed.
- **Keywords:** timescape cosmology; cosmological backreaction; Buchert averaging; Hubble
  tension; type Ia supernovae; baryon acoustic oscillations; CMB acoustic scale; void
  fraction; inhomogeneous cosmology
- **Related identifiers:**
  - `isSupplementTo` → Paper II arXiv ID (once posted)
  - `isContinuationOf` → Paper I Zenodo DOI (`timescape-hubble-tension`)
  - `references` → Seifert P+1690 covariance, `10.5281/zenodo.12729746`

## 2. Zenodo deposit — `free-history-timescape-tensions` (archived Paper III repo, GitHub-only)

- **Upload type:** Software · **License:** MIT · **Version:** v1.0-archived
- **Title:** Free-history timescape vs. the cosmological tensions (archived per-commit history)
- **Description (draft):**
  > Archived per-commit development history of Paper III, now merged into
  > `free-history-timescape` under `tensions/`. Retained read-only as the history of record.
- **Related identifiers:** `isPartOf` → `free-history-timescape` Zenodo DOI; `isPreviousVersionOf` → the merged `tensions/` tree.

---

## 3. arXiv — Paper II

- **Primary category:** astro-ph.CO
- **Title:** Free-history timescape: a data-driven void history reconciles the
  supernova–BAO–CMB geometry but does not resolve the Hubble tension
- **Authors:** Viacheslav Zhygulin
- **Comments:** 9 pages, 4 tables, no figures. Companion to Paper I (arXiv:XXXX.XXXXX).
  Code and artifacts: (Zenodo DOI once minted).
- **Report-no / MSC / ACM:** none
- **Abstract (condensed to ≤1920 chars for the arXiv field; the manuscript's full 2630-char
  abstract stays in the PDF):**

> Paper I of this program showed that timescape cosmology in its one-parameter tracker
> closure cannot fit type Ia supernovae (SNe), baryon acoustic oscillations (BAO), and the
> CMB acoustic scale together—the SNe drive the present void fraction to f_v0≈0.85 while
> BAO+CMB demand ≈0.64—yet a model-independent free-E(z) check shows the
> data are not internally contradictory: the split is the tracker shape's rigidity. Here we
> free the void-fraction history f_v(z) from the attractor, keeping the Buchert two-phase
> average and the wall/void clock-rate dressing, and re-test end to end. Part I (kinematic):
> the freed f_v(z) reconciles Pantheon+, DESI BAO, and the Planck acoustic scale at joint
> χ²=1396.1, below flat ΛCDM (1402.2) and far below the tracker (1469.3), at a plausible
> f_v0=0.640; the amplitude split dissolves to 0.16σ. This holds under the Pantheon+
> stat+sys covariance; under the cosmology-independent Lane–Seifert covariance (bias
> correction removed) the SNe instead prefer ΛCDM to the fixed free history at every
> redshift cut (ΔBIC=+11 to +27), so even the kinematic shape-sufficiency is
> covariance-dependent. And the freed history is the backreaction the Hubble diagram
> wants—it violates Buchert integrability, not a proven solution. Part II: four independent
> falsification tests each fail—the required void-fraction decline is unavailable in the
> BOSS DR12 population (by a floor theorem); the predicted local expansion-rate bias falls
> far short of what SH0ES needs; enforcing dynamical consistency collapses the
> solve to the tracker; and the in-model sound horizon is +36% too large. Removing the
> tracker's rigidity reconciles the distance geometry, but the flexibility is purely
> kinematic: free-history timescape does not resolve the Hubble tension. Consistent with
> Paper I, eliminating dark energy through inhomogeneity does not resolve the tension on
> present public data.

*(Char count verified in the accompanying commit; trim further if the arXiv form rejects.)*
