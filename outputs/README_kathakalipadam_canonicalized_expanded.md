# Kathakalipadam canonicalized expanded dataset

## Deliverables

- `kathakalipadam_canonicalized_expanded_425.csv`: analysis table with raw labels, canonical labels, provenance, and audio links.
- `kathakalipadam_canonicalization_audit.csv`: one row per canonical padam, showing every merged raw title and recording counts.
- `kathakalipadam_story_selection_audit.csv`: comparison used to select the two added stories.

## Coverage

- Total archive rows: 425
- Individual padam rows: 423
- Non-padam compilations: 2
- Story counts: {'Duryodhana Vadham': 53, 'Nalacharitham Moonam Divasam': 137, 'Santhana Gopalam': 131, 'Keechaka Vadham': 104}
- Rows without an exposed audio URL: 6
- Rows changed by a curated orthographic/transliteration alias: 107
- Canonical padams with at least three distinct exposed audio URLs: 51
- Distinct audio recordings inside those 3+ classes: 261

## Canonicalization policy

The source spelling is never overwritten: `padam_title_raw` and `padam_title_key` remain available. The modeling label is `padam_canonical_id`, which is story-qualified so similarly named padams in different stories are not merged. `padam_canonical_key` merges only documented punctuation, spacing, and conservative transliteration variants. For example, `ghora_vipinam` and `ghoravipinam` map to the same canonical padam. Phrase extensions such as `anthike` versus `anthike_vannitenam` remain separate.

Each row records the method, confidence, and note used for canonicalization. Medium-confidence curated merges should be checked by a Kathakali domain expert before publication.

## Story selection

Seven high-volume story listings were profiled. Santhana Gopalam and Keechaka Vadham were selected because they contributed the largest numbers of records already belonging to repeated exact-title groups (86 and 61 respectively). This favors repeated, independent performances rather than additional singleton labels.

## Modeling use

Use only `recording_scope == padam` and non-empty `audio_url` for audio classification. Use `padam_canonical_id` as the target. Keep `split_group_id` at recording level and use artist-aware or performance-aware grouped evaluation where metadata allows it. The canonicalization audit is the authoritative record of label merges.
