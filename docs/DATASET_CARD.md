# KPAC dataset card

## Summary

KPAC converts public Kathakalipadam story catalogues into an auditable metadata corpus and a bounded padam-recognition benchmark. Its principal contribution is corpus construction and evaluation infrastructure, not a novel classifier.

## Composition

- Source manifest: 615 rows from seven complete story catalogues.
- Validated corpus: 400 decoded, byte-unique recordings.
- Canonical targets: 88 story-qualified padam classes.
- Locked benchmark: 112 recordings, 12 classes, 4 stories, 26 main musicians.
- Class support in the lock: 8–16 recordings.
- Outer folds: 41, 39, and 32 recordings.

The target is `story_label + padam_canonical_key`. Story qualification prevents same-title padams in different narratives from being silently merged.

## Collection and validation

Catalogue pages were scraped completely for the selected stories. Each admitted file was downloaded, decoded, hashed with SHA-256, and inspected for duration, sample rate, and channel count. Missing and undecodable items were excluded with reasons. All 400 retained files have distinct byte hashes.

The benchmark’s 496 within-class pairs were additionally screened with time-ordered 2-second chroma/spectral shingles. No pair exceeded Jaccard 0.20 together with duration ratio 0.85; the maximum observed Jaccard was 0.016679. This is a candidate screen for likely transcodes, not proof of independently staged performances.

## Canonicalization

Unicode, case, punctuation, and whitespace are normalized mechanically. Explicit story-local aliases merge documented spelling and spacing variants such as `ghora_vipinam` and `ghoravipinam`. Raw, normalized, canonical, method, confidence, and note fields are retained. Phrase extensions remain distinct.

Canonical labels have not yet been independently adjudicated by multiple Kathakali practitioners. `PRACTITIONER_ADJUDICATION_PROTOCOL.md` defines a prospective process; it does not imply that practitioner adjudication has already occurred.

## Evaluation

The locked primary result is a recording-held-out three-fold nested evaluation of frozen MERT features and class-balanced linear SVMs. Standardization and classifier fitting occur afresh inside each inner training split. Feature representation and `C` are selected only within the outer training partition; the outer test fold is predicted once.

Primary reference: 29.46% accuracy, 26.76% macro-F1, 27.02% balanced accuracy, 60.71% top-3 accuracy, and a 21.43–37.50% recording-bootstrap interval.

The split is not artist-disjoint. A 13-recording unseen-main-musician subset and known-story results are post-hoc diagnostics, not alternative primary estimates.

## Intended uses

- Low-resource representation evaluation.
- Retrieval-assistance research with a human cataloguer in the loop.
- Canonicalization and archival-data quality studies.
- Preregistered hierarchical or grouped evaluation on a future lock.

## Out-of-scope uses

- Automatic musicological adjudication.
- Ranking performers or renditions.
- Claims of singer-, event-, or archive-independent generalization.
- Treating the corpus as comprehensive coverage of Kathakali.
- Redistributing source audio without permission.

## Known limitations

Small classes, artist overlap, incomplete provenance, fixed excerpts not aligned to dramatic boundaries, dead source links, non-expert canonicalization, and bootstrap intervals that omit model-selection and performer-cluster uncertainty. Hashes, URLs, durations, and acoustic fingerprints reduce duplicate risk but cannot establish performance-event independence.

## Maintenance

Use semantic versions. Metadata corrections require a metadata release. Added performances require a corpus release. Any changed test membership requires a new benchmark version. Preserve the v1 split so data improvements can be distinguished from modeling improvements.
