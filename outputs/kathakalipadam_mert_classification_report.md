# Locked MERT padam-classification evaluation

## Result

- Recordings: **112**
- Canonical padam classes: **12** across **4** stories
- Accuracy: **29.46%** (recording bootstrap 95% interval: 21.43%–37.50%)
- Macro-F1: **26.76%**
- Balanced accuracy: **27.02%**
- Top-3 accuracy: **60.71%**
- 40% target reached: **no**

## Locked design

The 112-recording outer evaluation assignment (SHA-256 `a558f3324b7051726b89f6678bce32df2a1cd549f0cc40a0206826956685bb50`) was written before MERT was downloaded or any MERT feature was extracted. For each outer fold, MERT representation and linear-SVM C were chosen only by three-fold CV inside the outer training partition. Each outer test partition was predicted once.

## Fold selections

- Fold 1: `layer_6_mean_std_across_chunks`, C=0.1; inner accuracy 21.13%; n=41
- Fold 2: `layer_6_mean_std_across_chunks`, C=0.01; inner accuracy 21.92%; n=39
- Fold 3: `layer_6_mean_std_across_chunks`, C=0.01; inner accuracy 25.00%; n=32

## Weakest classes (out-of-fold recall)

- `santhana_gopalam__sreeman_sakhe`: 0.00% (0/8)
- `nalacharitham_moonam_divasam__lokapalanmaare`: 11.11% (1/9)
- `nalacharitham_moonam_divasam__ghora_vipinam`: 12.50% (1/8)
- `nalacharitham_moonam_divasam__swaira_vachanam`: 12.50% (1/8)
- `santhana_gopalam__vidhimatham`: 12.50% (1/8)

The known-story majority diagnostic is not an acoustic result and is reported only to quantify how much story metadata can simplify this benchmark.

## Post-hoc known-story diagnostic

When the true story is supplied as inference-time metadata and a separate nested MERT classifier is fitted within each story, accuracy is **46.43%** and macro-F1 is **44.26%**. This crosses 40%, but it is a post-hoc hybrid metadata+audio result, not the preregistered audio-only result.
