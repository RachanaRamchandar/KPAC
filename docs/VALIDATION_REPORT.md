# Release validation report

Validated on 2026-10-27 with `python3 work/validate_release.py`.

- Source manifest: 615 rows, unique `record_id`.
- Validated corpus: 400 rows, 88 canonical story-qualified classes.
- Audio identity fields: 400 syntactically valid, unique SHA-256 digests.
- Locked benchmark: 112 rows, 12 classes, 4 stories.
- Outer-fold sizes: 41, 39, and 32.
- Class support: minimum 8, maximum 16.
- Locked rows are a subset of the validated corpus.
- Locked benchmark CSV SHA-256: `f49ae18cecd97f330771f888cb9d2c7d9f11758408b04b2d42660ed146cf14dd`.
- Published assignment SHA-256: `a558f3324b7051726b89f6678bce32df2a1cd549f0cc40a0206826956685bb50`.
- Primary recording accuracy: 0.29464285714285715.
- Acoustic screen: 496 within-class pairs, 0 above-threshold candidates.
- Licensing files: MIT software license, CC BY 4.0 data notice, and explicit third-party audio exclusion present.
`MANIFEST.sha256` covers every release file except itself. Run `shasum -a 256 -c MANIFEST.sha256` after downloading or extracting the archive.
