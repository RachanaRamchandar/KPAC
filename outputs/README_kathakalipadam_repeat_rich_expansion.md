# Repeat-rich Kathakalipadam corpus expansion

The previous 425-row corpus is retained. Three reserve stories were scraped in full, but only new canonical classes with at least 3 distinct exposed audio URLs were admitted. New singleton and two-recording padams remain in the exclusion audit rather than the modeling CSV.

- Previous rows retained: 425
- New story listing rows audited: 311
- New repeat-rich recordings admitted: 190
- New repeat-rich canonical classes admitted: 41
- Final CSV rows: 615
- New-story distinct-audio class-size distribution: {1: 73, 2: 18, 3: 15, 4: 11, 5: 3, 6: 4, 7: 4, 8: 2, 9: 2}

After downloading and decoding the admitted audio:

- New decoded recordings retained: 167
- New audio URLs excluded after download/decode validation: 23 (all returned HTTP 404)
- Final validated analysis recordings: 400 (233 previous + 167 new)
- Final validated canonical classes: 88
- Global duplicate SHA-256 digests: 0
- Canonical classes with at least 8 validated recordings: 12 (112 recordings)
- Canonical/story label inconsistencies: 0

The 615-row manifest preserves metadata for unavailable files; use `kathakalipadam_repeat_rich_validated_audio.csv` as the audio-analysis corpus and `kathakalipadam_mert_locked_benchmark.csv` for the preregistered MERT evaluation cohort.

Canonicalization is story-local and conservative. It merges punctuation, spacing, and documented transliteration variants, while phrase extensions remain distinct. Distinct hashes, URLs, millisecond durations, and musician combinations support recording independence; these signals cannot by themselves prove that two differently encoded sources were independently staged performances.
