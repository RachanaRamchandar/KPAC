# KPAC: Kathakali Padam Audio Corpus

KPAC is a reproducibly curated metadata corpus and locked reference benchmark for story-qualified Kathakali padam recognition.

## What is in this repository

- `outputs/`: manifests, audits, the frozen benchmark, predictions, metrics, reports, and summary workbook.
- `work/`: collection, validation, canonicalization, locking, feature extraction, evaluation, and figure-generation programs.
- `docs/`: dataset card, field definitions, reconstruction and validation instructions, practitioner-adjudication protocol, and access conditions.

Audio is intentionally absent. The recordings are third-party archival material. Benchmark rows retain source URLs, SHA-256 digests, and durations so eligible users can reconstruct and verify the same corpus subject to the source archive’s availability and terms.

## Headline results

| System or diagnostic | Accuracy | Macro-F1 | Top-3 |
|---|---:|---:|---:|
| Locked MERT level 6 | 29.46% | 26.76% | 60.71% |
| Handcrafted MFCC/chroma/spectral | 24.11% | 21.76% | 51.79% |
| Frozen YAMNet | 22.32% | 20.08% | 40.18% |
| Performer metadata only | 8.04% | 3.98% | 28.57% |
| Known-story majority, post-hoc | 38.39% | 20.20% | — |
| Known-story MERT, post-hoc | 46.43% | 44.26% | — |

Only the first row is the preregistered primary reference evaluation. The exact-fold comparison systems are additional diagnostics. The known-story rows change the task by supplying the true story and must not be presented as the audio-only score.

## Quick verification

From the repository root:

```bash
python3 work/validate_release.py
```

The validator checks required files, row counts, unique recording identifiers, byte-level audio digests in the validated index, benchmark membership, outer-fold counts, class support, and the published assignment digest.

For full reconstruction and experimental commands, see `docs/REPRODUCTION.md`. The feature caches and downloaded model weights are deliberately excluded because they are regenerable and large.

## Licensing

- Original software under `work/` is licensed under the MIT License in `LICENSE`.
- Author-created annotations, canonicalization mappings, benchmark assignments, predictions, aggregate results, and documentation are available under CC BY 4.0 as described in `DATA_LICENSE.md`.
- Source audio is not included or licensed. Third-party catalogue fields and source material remain subject to their rightsholders and the source archive's terms. See `DATA_ACCESS.md`.

## Locked benchmark identity

- Recordings: 112
- Classes: 12
- Stories: 4
- Outer folds: 3 (41, 39, 32 recordings)
- Assignment seed: `kathakalipadam-mert-outer-v1-2026-08-09`
- Assignment SHA-256: `a558f3324b7051726b89f6678bce32df2a1cd549f0cc40a0206826956685bb50`
- MERT model: `m-a-p/MERT-v1-95M`
- Pinned MERT revision: `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`

Do not change test membership under the same benchmark version. Corrections to labels or provenance should be versioned, and a changed test set should receive a new benchmark identifier.

## Scope and limitations

KPAC is a corpus and reference evaluation, not a new classifier. Its present split is recording-held-out, not singer-, event-, or archive-disjoint. Distinct URLs, durations, hashes, and a 496-pair acoustic fingerprint screen reduce duplicate risk but do not prove independently staged performances. Canonical labels have not yet been independently adjudicated by multiple Kathakali practitioners. See `docs/DATASET_CARD.md` and `docs/PRACTITIONER_ADJUDICATION_PROTOCOL.md` before extending the corpus.
