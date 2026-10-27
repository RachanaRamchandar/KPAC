# Data dictionary

## Primary files

### `outputs/kathakalipadam_repeat_rich_expanded.csv`

One row per catalogue recording candidate (615 rows). Important fields:

- `record_id`: stable corpus row identifier.
- `source_page_record_id`: identifier retained from the source-page collection stage.
- `recording_scope`: original or expansion collection scope.
- `padam_title_raw`: source title as collected.
- `padam_title_key`: mechanically normalized title.
- `padam_canonical_key`: canonical spelling/spacing key.
- `padam_canonical_id`: story-qualified target identifier.
- `padam_canonical_label`: display label.
- `canonicalization_method`, `canonicalization_confidence`, `canonicalization_note`: decision provenance.
- `main_musician_raw`, `supporting_musician_raw`: source strings.
- `main_musician`, `supporting_musician`: normalized performer fields.
- `story_raw`, `ragam_raw`, `thalam_raw`: source strings.
- `story_label`, `ragam_label`, `thalam_label`: normalized fields.
- `missing_*`: explicit missingness indicators.
- `audio_url`, `detail_url`, `source_listing_url`: reconstruction provenance.
- `source_retrieved_on`: catalogue retrieval date.
- `split_group_id`: grouping field used during data preparation.
- `dataset_iteration`: collection iteration.

### `outputs/kathakalipadam_repeat_rich_validated_audio.csv`

The 400 decoded recordings. It contains the manifest fields plus:

- `audio_sha256`: digest of the exact retrieved file.
- `audio_duration_seconds`: decoded duration.
- `audio_sample_rate`: decoded sampling rate.
- `audio_channels`: decoded channel count.
- `audio_validation_status`: validation disposition.

### `outputs/kathakalipadam_mert_locked_benchmark.csv`

The 112-row immutable v1 benchmark assignment:

- `record_id`, `padam_canonical_id`, `padam_canonical_label`, `story_label`.
- musician, ragam, and thalam metadata retained for audit/diagnostics.
- `audio_url`, `audio_sha256`, `audio_duration_seconds` for identity checks.
- `locked_outer_fold`: one of 1, 2, or 3.
- `fold_assignment_seed`: seed naming the frozen assignment procedure.

## Audit files

- `kathakalipadam_repeat_rich_canonicalization_audit.csv`: spelling-variant decisions and support.
- `kathakalipadam_expansion_exclusions.csv`: excluded candidates and reasons.
- `kathakalipadam_recording_independence_audit.csv`: class-level URL, duration, hash, and musician evidence.
- `kpac_acoustic_near_duplicate_screen.csv`: all 496 within-class pairwise acoustic-screen results.
- `kathakalipadam_remaining_story_depth_audit.csv`: remaining archive depth by story.
- `kathakalipadam_mert_targeted_collection_priority.csv`: prioritized weak classes for independent collection.

## Results

Predictions are out-of-fold. Metrics in `outputs/kathakalipadam_mert_classification_results.json` correspond to the locked primary experiment. Files prefixed `kpac_additional_baseline_` contain additional exact-fold diagnostics and are not preregistered primary results.
