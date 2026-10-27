# KathakaliPadam combined story scrape

This deliverable contains **190 records** scraped from two Kathakalipadam.com “By Story” listings on 2026-08-09. It is an analysis-oriented metadata manifest for music-information-retrieval experiments, with performer, story, ragam, thalam, provenance, audio references, missingness flags, and leakage-safe recording groups.

## Files

- `kathakalipadam_combined_stories_190.csv`: UTF-8 CSV with 190 rows and 26 columns.

## Included stories

- Duryodhana Vadham: 53 rendered records.
- Nalacharitham Moonam Divasam: 137 rendered records.
- The Nalacharitham page’s sidebar label says 138, but its rendered table exposed 137 complete title/metadata pairs and no pagination controls on the retrieval date. The CSV includes only verifiable rendered rows.

## Validation summary

- 190/190 detail pages parsed without scrape errors.
- No duplicate detail-page URLs were found across the two story listings.
- 189 rows are individual padam upload pages.
- 1 row (record 023) is a full-story compilation and is flagged for exclusion or manual segmentation.
- 188 rows expose normal downloadable audio links.
- 2 rows expose legacy/indirect media references: record 058 uses an embedded SoundCloud track reference, and record 159 uses a Copy.com audio source reference. Both require resolution and verification before analysis.
- 31 rows have missing/unknown ragam.
- 120 rows have missing/unknown thalam.
- 22 rows have missing/unknown supporting-musician metadata.
- Listing and detail-page story/ragam values showed no contradictions where detail values were present.

## Modeling guidance

1. **Split before chunking.** Use `split_group_id` so every fixed-length chunk from the same source recording stays in one train/validation/test fold. Random chunk-level splitting would leak recording-specific acoustics.
2. **Curate padam class labels.** `padam_title_key` performs Unicode-preserving lowercase and punctuation normalization only. It does not merge spelling variants, translated/transliterated forms, ragam suffixes, or culturally equivalent titles.
3. **Exclude or separately segment record 023.** It is a full-story compilation rather than a single padam recording.
4. **Resolve records 058 and 159 before feature extraction.** Their source pages provide legacy media references instead of normal direct audio-file URLs.
5. **Account for story imbalance.** The two story groups contain 53 and 137 rows. Story-stratified analyses and any story-conditioned evaluation should report this imbalance.
6. **Treat missing fields as missing.** Analysis fields are blank when the source says Unknown, --unknown--, --none--, or omits a heading; original strings remain in the matching `*_raw` columns.
7. **Verify audio and rights before experiments.** The CSV links to public archive or third-party-hosted media; it does not redistribute audio or establish a reuse license. Resolve URLs, record retrieval failures, store checksums, and confirm permissions before constructing a research corpus.
8. **Use the exact dataset size.** This manifest contains 190 rows, or 189 individual padam candidates after excluding the full-story compilation—not approximately 150 unless additional documented filtering is applied.

## Column notes

- `record_id`: stable row ID for this combined export; the original 001–053 IDs are preserved.
- `source_page_record_id`: numeric identifier from the detail-page slug.
- `recording_scope`: `padam` or `full_story_compilation`.
- `padam_title_raw`: title exactly as displayed in the listing.
- `padam_title_key`: mechanical Unicode-preserving lowercase/punctuation key, not an expert-curated class label.
- `*_raw`: source value preserved verbatim.
- `main_musician`, `supporting_musician`, `story_label`, `ragam_label`, and `thalam_label`: analysis fields; source unknown/none values become blank.
- `missing_*`: binary missingness flags.
- `audio_url`: direct audio URL when exposed; otherwise the legacy media reference described in `modeling_note`.
- `audio_format`: file extension for ordinary audio files or a reference-type marker for legacy media.
- `detail_url` and `source_listing_url`: row-level provenance.
- `split_group_id`: grouping key for leakage-safe evaluation.
- `modeling_note`: row-specific caution.

Sources:

- https://www.kathakalipadam.com/index.php/en/padam-list/by-story?option=com_musicsearch&limitstart=0&sragam=-1&sartist=-1&sstory=78&limit=53
- https://www.kathakalipadam.com/index.php/en/padam-list/by-story?option=com_musicsearch&limitstart=0&sragam=-1&sartist=-1&sstory=83&limit=138
