# Kathakalipadam repeat-rich expansion and MERT evaluation

## Executive result

The preregistered audio-only MERT experiment did **not** reach 40%: it achieved **29.46% recording accuracy** (33/112; recording bootstrap 95% interval 21.43%–37.50%), **26.76% macro-F1**, **27.02% balanced accuracy**, and **60.71% top-3 accuracy** across 12 canonical padam classes.

A clearly labeled post-hoc hybrid experiment that supplies the known story at inference time and fits a nested MERT classifier within each story reached **46.43% accuracy** and **44.26% macro-F1**. This is useful evidence that story-aware classification is viable, but it is not a replacement for the locked audio-only result.

## Corpus expansion

Three repeat-rich reserve stories were scraped in full:

- Karna Shapadham: 97 rendered listing rows; 45 repeat-rich recordings selected.
- Kuchela Vritham: 96 rendered rows; 65 selected.
- Nalacharitham Nalaam Divasam: 118 rendered rows; 80 selected.

This produced 190 candidate recordings across 41 non-singleton canonical classes. After download and decode validation, 167 new recordings were retained and 23 dead URLs (HTTP 404) were excluded. Combined with the previous 233 recordings, the validated corpus contains **400 byte-unique recordings across 88 canonical classes**.

Canonicalization remained story-local and conservative. The audit found zero canonical label/story inconsistencies and zero duplicate SHA-256 digests. Distinct URLs, millisecond durations, and artist combinations support performance independence; these signals cannot conclusively prove that every differently encoded source was an independently staged performance.

## Locked MERT design

- Eligibility: canonical classes with at least 8 validated recordings.
- Locked cohort: 112 recordings, 12 classes, 4 stories.
- Outer evaluation: three deterministic stratified folds, frozen before MERT download or feature extraction.
- Lock digest: `a558f3324b7051726b89f6678bce32df2a1cd549f0cc40a0206826956685bb50`.
- Model: [`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M), pinned to commit `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.
- Input: five 8-second mono chunks per recording at 24 kHz.
- Candidate representations: layers 6, 9, 12, and mean of the last four layers; recording vectors use chunk mean and standard deviation.
- Classifier: standardized class-balanced linear SVM with C in {0.01, 0.1, 1, 10}.
- Selection: representation and C chosen only by three-fold CV inside each outer training partition.

The MERT architecture and pretraining procedure are described in the official [MERT paper](https://arxiv.org/abs/2306.00107).

## Ablations

| Experiment | Accuracy | Macro-F1 | Balanced accuracy | Top-3 |
|---|---:|---:|---:|---:|
| Preregistered nested selection | 29.46% | 26.76% | 27.02% | 60.71% |
| Layer 6 | 29.46% | 26.76% | 27.02% | 60.71% |
| Last four layers | 26.79% | 25.12% | 25.26% | 57.14% |
| Layer 9 | 25.00% | 22.61% | 23.58% | 52.68% |
| Layer 12 | 24.11% | 21.81% | 22.27% | 48.21% |
| Training-fold majority label | 14.29% | 2.08% | 8.33% | — |
| Known-story majority (metadata only) | 38.39% | 20.25% | 32.64% | — |
| Known story + nested MERT (post hoc) | 46.43% | 44.26% | 44.55% | 97.32% |

All three outer folds selected layer 6. The deeper layers were consistently worse, so further tuning should begin from the mid-level MERT representation rather than layer 12.

## Collection contingency

Because the audio-only result is below 40%, the locked 12 classes would need **36 additional independent recordings** to reach 12 per class, or 69 to reach 15 per class. The same-story Kathakalipadam pages have already been exhausted.

An archive-wide audit of every remaining story with at least 30 listed recordings found **no unselected story containing an exact-normalized padam group of 12 or more**. The deepest remaining groups are:

- `devayani_charitham__sundara_kalebara`: 9 listing rows.
- `nalacharitham_randam_divasam__kuvalaya_vilochane`: 9 listing rows.
- `lavanasuravadham__sukhamo_devi`: 8 listing rows.

Adding these stories would enlarge the label space without meeting the 12–15-per-class contingency, so they were audited but not added to the locked experiment. Further collection should instead seek external archives or newly digitized performances for the ranked existing classes, starting with `santhana_gopalam__sreeman_sakhe`, `nalacharitham_moonam_divasam__ghora_vipinam`, `nalacharitham_moonam_divasam__swaira_vachanam`, and `santhana_gopalam__vidhimatham`.

## Recommended interpretation

Use **29.46%** as the defensible audio-only result. Use **46.43%** only when the deployment task legitimately knows the story in advance. For a genuinely story-agnostic classifier above 40%, the evidence still favors acquiring independent performances for the weakest existing canonical classes rather than adding singleton or low-depth padams.
