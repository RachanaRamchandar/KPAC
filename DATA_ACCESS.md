# Data access, rights, and reconstruction

## What is redistributed

This package redistributes author-created scripts, canonicalization decisions, split assignments, audits, predictions, aggregate results, and documentation. It does **not** redistribute the source audio.

The CSV files contain public catalogue fields and author-created annotations derived from the Kathakalipadam archive. The inclusion of a URL, title, artist name, or checksum does not transfer ownership of the corresponding recording or source metadata.

## Licensing boundary

The programs under `work/` are licensed under the MIT License in `LICENSE`.
The KPAC authors' original annotations, canonicalization mappings, benchmark
assignments, predictions, aggregate results, and documentation are available
under CC BY 4.0 as described in `DATA_LICENSE.md`. These grants apply only to
material for which the KPAC authors hold the necessary rights. They do not
license source audio, performances, or third-party catalogue content.

## Audio access

Each admitted recording is identified by its public source URL, SHA-256 digest, duration, sample rate, and channel count in `outputs/kathakalipadam_repeat_rich_validated_audio.csv`. The locked subset is in `outputs/kathakalipadam_mert_locked_benchmark.csv`.

Users must obtain recordings from the source archive themselves and comply with the archive’s current terms, performer and rightsholder rights, and applicable law. Links may expire or change. A digest mismatch means the retrieved file is not byte-identical to the file used in the reported experiment and should not silently replace it.

## Responsible use

KPAC is intended for research on low-resource MIR, retrieval assistance, corpus curation, and transparent evaluation. It is not a comprehensive Kathakali catalogue or a musicological authority. Preserve attribution and provenance, document corrections, solicit practitioner consultation, and do not treat model outputs as judgments about artistic quality or cultural authenticity.
