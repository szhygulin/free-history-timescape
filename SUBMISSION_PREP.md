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
  > (violates Buchert integrability) and covariance-dependent, and four
  > availability/dynamics/calibration tests---facets of one kinematic-dynamical
  > gap---each fail. Every headline number is reproduced
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
- **Comments:** 10 pages, 4 tables, no figures. Companion to Paper I (arXiv:XXXX.XXXXX).
  Code and artifacts: (Zenodo DOI once minted).
- **Report-no / MSC / ACM:** none
- **Abstract (condensed to ≤1920 chars for the arXiv field; the manuscript's full 2630-char
  abstract stays in the PDF):**

> Paper I showed that timescape cosmology's one-parameter tracker closure cannot jointly
> fit type Ia supernovae (SNe), baryon acoustic oscillations (BAO), and the CMB acoustic
> scale: the SNe pull the present void fraction to f_v0≈0.85 while BAO+CMB demand ≈0.64,
> though a model-independent free-E(z) check finds the data consistent—the split is the
> tracker's rigidity. Here we free the void history f_v(z), keeping the Buchert two-phase
> average and clock-rate dressing. The freed f_v(z) reconciles Pantheon+, DESI BAO, and the
> Planck acoustic scale at joint χ²=1396.1, below flat ΛCDM (1402.2) and far below the
> tracker (1469.3), at a plausible f_v0=0.640; the amplitude split dissolves to 0.16σ—a
> single-amplitude check at fixed jointly-fit shape, not a proof of data consistency. The
> reconciliation is covariance-dependent: under the cosmology-independent covariance of
> Lane, Seifert et al. (bias correction removed) the SNe prefer ΛCDM at every redshift cut
> (ΔBIC=+11 to +27). It is moreover kinematic—a backreaction target the Hubble diagram
> wants, violating Buchert integrability—and fails four tests of independent physical
> character (availability, local-ladder bias, dynamical consistency, early-Universe
> calibration), facets of one kinematic–dynamical gap: the required history declines by
> ×3.31 over 0<z<2.33 against an observed ×1.16 (a floor theorem; level matched, shape
> unavailable); the predicted local bias is +2.4% against the +8.4% SH0ES requires
> (pre-registered PARTIAL, failing when the observed catalog is forced in); enforced
> consistency collapses the three-phase Buchert solve to the tracker, missing the geometry
> bar by 4953; and the in-model sound horizon r_d=199.6 Mpc (+36%) drives the bare H0 to
> ~32–40 km/s/Mpc. The distance geometry reconciles, but the flexibility is kinematic:
> consistent with Paper I, free-history timescape does not resolve the Hubble tension on
> present public data.

*(Char count verified in the accompanying commit; trim further if the arXiv form rejects.)*
