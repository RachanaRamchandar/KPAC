# Reproduction guide

Run all commands from the repository root. Python 3.10 or 3.11 is recommended for the frozen MERT environment. Network access is needed only to reconstruct source audio and obtain pretrained model weights.

## 1. Verify the release

```bash
python3 work/validate_release.py
shasum -a 256 -c MANIFEST.sha256
```

## 2. Reconstruct audio

Create an ignored `audio/` directory. Use `outputs/kathakalipadam_repeat_rich_validated_audio.csv` as the allow-list. Retrieve each `audio_url`, preserve the `record_id` mapping, decode it, and require the resulting source-file digest to match `audio_sha256`. Do not silently substitute a changed file.

The original acquisition/validation path is implemented by:

```bash
python3 work/scrape_expansion_stories.py
python3 work/build_repeat_rich_expansion.py
python3 work/download_validate_repeat_rich_audio.py
python3 work/audit_repeat_rich_independence.py
python3 work/create_mert_locked_benchmark.py
```

Scripts were preserved from the working study and expect the repository layout documented in their constants. Check paths before rerunning destructive or networked acquisition steps. Public links are volatile, so a later reconstruction may report fewer available items.

## 3. Locked MERT evaluation

Install the MERT dependencies in their own environment:

```bash
python -m venv .venv-mert
source .venv-mert/bin/activate
pip install numpy==1.26.4 scipy==1.14.1 soundfile==0.14.0 scikit-learn==1.5.2 torch==2.4.1 transformers==4.44.2 librosa==0.11.0
python work/extract_mert_embeddings.py
python work/evaluate_mert_locked.py
```

The extraction script pins `m-a-p/MERT-v1-95M` to commit `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`. Confirm that the generated assignment digest is `a558f3324b7051726b89f6678bce32df2a1cd549f0cc40a0206826956685bb50` before comparing metrics.

## 4. Additional exact-fold baselines

Handcrafted features:

```bash
python work/additional_baselines/extract_locked_handcrafted.py
```

YAMNet is best isolated because TensorFlow and the MERT stack have different dependency constraints:

```bash
python -m venv .venv-yamnet
source .venv-yamnet/bin/activate
pip install tensorflow==2.16.2 tensorflow-hub==0.16.1 soundfile==0.14.0
python work/additional_baselines/extract_locked_yamnet.py
```

Evaluate the handcrafted, YAMNet, and performer-only references on the frozen folds:

```bash
python work/additional_baselines/evaluate_locked_baselines.py
```

Run the acoustic duplicate-candidate screen:

```bash
python work/additional_baselines/acoustic_near_duplicate_screen.py
```

## Determinism boundary

Fold membership and tabular evaluation are deterministic given the checked files and stated package versions. Audio reconstruction can fail because remote files are third-party and mutable. GPU kernels and pretrained-model library changes can also introduce small numerical differences; retain the pinned model revision and record the full environment for a new release.
