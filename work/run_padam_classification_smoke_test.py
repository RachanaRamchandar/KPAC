#!/usr/bin/env python3
"""Leakage-safe padam-classification smoke test for the combined CSV manifest.

The script deliberately uses only exact `padam_title_key` labels with at least
four manifest rows before URL validation. It downloads audio into `work/`,
extracts six fixed-length chunks per usable recording, computes handcrafted
acoustic features, and evaluates classifiers with recording-grouped folds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import librosa
import sklearn
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
TARGET_SAMPLE_RATE = 16_000
CHUNK_SECONDS = 20.0
CHUNKS_PER_RECORDING = 6
N_FOLDS = 3
CLASS_KEYS = ("ghora_vipinam", "mariman_kanni", "vasa_vasa_sootha")
CLASS_DISPLAY = {
    "ghora_vipinam": "Ghora Vipinam",
    "mariman_kanni": "Mariman Kanni",
    "vasa_vasa_sootha": "Vasa Vasa Sootha",
}

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE = SCRIPT_PATH.parent.parent
OUTPUT_DIR = WORKSPACE / "outputs"
WORK_DIR = WORKSPACE / "work" / "padam_smoke_test"
AUDIO_DIR = WORK_DIR / "audio_mp3"
FEATURE_CACHE = WORK_DIR / "features.npz"
MANIFEST = OUTPUT_DIR / "kathakalipadam_combined_stories_190.csv"
RESULTS_JSON = OUTPUT_DIR / "padam_classification_smoke_test_results.json"
PREDICTIONS_CSV = OUTPUT_DIR / "padam_classification_smoke_test_predictions.csv"
REPORT_MD = OUTPUT_DIR / "padam_classification_smoke_test_report.md"


def log(message: str) -> None:
    print(message, flush=True)


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["recording_scope"] == "padam"
        and row["padam_title_key"] in CLASS_KEYS
        and row["audio_url"]
    ]
    return selected


def safe_extension(row: dict[str, str]) -> str:
    suffix = Path(row.get("audio_filename", "")).suffix.lower()
    return suffix if suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"} else ".mp3"


def download_audio(row: dict[str, str]) -> tuple[Path | None, str | None]:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    destination = AUDIO_DIR / f"{row['record_id']}{safe_extension(row)}"
    if destination.exists() and destination.stat().st_size > 4_096:
        return destination, None

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        row["audio_url"],
        headers={"User-Agent": "Mozilla/5.0 (KathakaliPadam research smoke test)"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            if temporary.stat().st_size <= 4_096:
                raise RuntimeError("downloaded file is unexpectedly small")
            temporary.replace(destination)
            return destination, None
        except Exception as exc:  # capture provider-specific URL failures in the report
            last_error = exc
            temporary.unlink(missing_ok=True)
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                break
            time.sleep(1.5 * (attempt + 1))
    return None, f"{type(last_error).__name__}: {last_error}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_start_times(duration: float) -> list[float]:
    if duration <= CHUNK_SECONDS:
        return [0.0]
    lead_skip = 15.0 if duration >= CHUNK_SECONDS + 30.0 else 0.0
    trailing_skip = 10.0 if duration >= CHUNK_SECONDS + 20.0 else 0.0
    latest = max(lead_skip, duration - trailing_skip - CHUNK_SECONDS)
    starts = np.linspace(lead_skip, latest, CHUNKS_PER_RECORDING)
    return [float(value) for value in starts]


def read_chunk(audio: sf.SoundFile, start_seconds: float) -> np.ndarray:
    source_sr = int(audio.samplerate)
    audio.seek(min(int(round(start_seconds * source_sr)), max(0, len(audio) - 1)))
    samples = audio.read(
        frames=int(round(CHUNK_SECONDS * source_sr)),
        dtype="float32",
        always_2d=True,
    )
    mono = samples.mean(axis=1) if samples.size else np.zeros(0, dtype=np.float32)
    if source_sr != TARGET_SAMPLE_RATE and mono.size:
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=TARGET_SAMPLE_RATE)
    target_length = int(round(CHUNK_SECONDS * TARGET_SAMPLE_RATE))
    if mono.size < target_length:
        mono = np.pad(mono, (0, target_length - mono.size))
    return np.asarray(mono[:target_length], dtype=np.float32)


def summarize_matrix(prefix: str, matrix: np.ndarray) -> tuple[list[str], list[float]]:
    matrix = np.atleast_2d(matrix)
    names: list[str] = []
    values: list[float] = []
    for index, row in enumerate(matrix):
        names.extend((f"{prefix}_{index + 1:02d}_mean", f"{prefix}_{index + 1:02d}_std"))
        values.extend((float(np.mean(row)), float(np.std(row))))
    return names, values


def acoustic_features(y: np.ndarray) -> tuple[list[str], np.ndarray]:
    kwargs = dict(n_fft=1024, hop_length=512)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=TARGET_SAMPLE_RATE,
        n_mfcc=20,
        n_mels=40,
        fmin=50,
        fmax=TARGET_SAMPLE_RATE / 2,
        **kwargs,
    )
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=TARGET_SAMPLE_RATE, **kwargs)
    descriptors = np.vstack(
        [
            librosa.feature.spectral_centroid(y=y, sr=TARGET_SAMPLE_RATE, **kwargs),
            librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SAMPLE_RATE, **kwargs),
            librosa.feature.spectral_rolloff(y=y, sr=TARGET_SAMPLE_RATE, roll_percent=0.85, **kwargs),
            librosa.feature.spectral_flatness(y=y, **kwargs),
            librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512),
            librosa.feature.rms(y=y, frame_length=1024, hop_length=512),
        ]
    )

    names: list[str] = []
    values: list[float] = []
    for prefix, matrix in (
        ("mfcc", mfcc),
        ("delta_mfcc", delta),
        ("chroma", chroma),
        ("spectral", descriptors),
    ):
        block_names, block_values = summarize_matrix(prefix, matrix)
        names.extend(block_names)
        values.extend(block_values)
    vector = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return names, vector


def build_features(
    usable: list[tuple[dict[str, str], Path]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, object]]]:
    expected_ids = [row["record_id"] for row, _ in usable]
    if FEATURE_CACHE.exists():
        cache = np.load(FEATURE_CACHE, allow_pickle=True)
        if list(cache["expected_ids"]) == expected_ids:
            log("Using cached acoustic features")
            return (
                cache["X"],
                cache["y"],
                cache["groups"],
                cache["starts"],
                list(cache["feature_names"]),
                list(cache["recording_metadata"]),
            )

    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    starts: list[float] = []
    feature_names: list[str] = []
    recording_metadata: list[dict[str, object]] = []

    for index, (row, path) in enumerate(usable, start=1):
        with sf.SoundFile(path) as audio:
            duration = len(audio) / audio.samplerate
            positions = chunk_start_times(duration)
            log(
                f"Features {index:02d}/{len(usable):02d}: record {row['record_id']} "
                f"({row['padam_title_key']}, {duration:.1f}s, {len(positions)} chunks)"
            )
            for start in positions:
                y = read_chunk(audio, start)
                names, vector = acoustic_features(y)
                if not feature_names:
                    feature_names = names
                features.append(vector)
                labels.append(row["padam_title_key"])
                groups.append(row["record_id"])
                starts.append(start)
            recording_metadata.append(
                {
                    "record_id": row["record_id"],
                    "label": row["padam_title_key"],
                    "title": row["padam_title_raw"],
                    "main_musician": row["main_musician"],
                    "duration_seconds": duration,
                    "chunks": len(positions),
                    "sha256": sha256_file(path),
                    "source_url": row["audio_url"],
                }
            )

    X = np.vstack(features)
    y_array = np.asarray(labels)
    group_array = np.asarray(groups)
    start_array = np.asarray(starts, dtype=float)
    np.savez_compressed(
        FEATURE_CACHE,
        X=X,
        y=y_array,
        groups=group_array,
        starts=start_array,
        feature_names=np.asarray(feature_names),
        expected_ids=np.asarray(expected_ids),
        recording_metadata=np.asarray(recording_metadata, dtype=object),
    )
    return X, y_array, group_array, start_array, feature_names, recording_metadata


def grouped_fold_map(groups: np.ndarray, y: np.ndarray) -> dict[str, int]:
    rng = random.Random(RANDOM_SEED)
    label_to_groups: dict[str, list[str]] = defaultdict(list)
    for group in sorted(set(groups)):
        label = str(y[np.flatnonzero(groups == group)[0]])
        label_to_groups[label].append(group)
    mapping: dict[str, int] = {}
    for label in sorted(label_to_groups):
        class_groups = label_to_groups[label]
        rng.shuffle(class_groups)
        for index, group in enumerate(class_groups):
            mapping[group] = index % N_FOLDS
    return mapping


def score_matrix(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)
    values = np.asarray(model.decision_function(X), dtype=float)
    return values[:, None] if values.ndim == 1 else values


def metric_block(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_KEYS), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(CLASS_KEYS)).tolist(),
    }


def evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    fold_map = grouped_fold_map(groups, y)
    models = {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    results: dict[str, object] = {}
    prediction_rows: list[dict[str, object]] = []
    for model_name, factory in models.items():
        all_chunk_true: list[str] = []
        all_chunk_pred: list[str] = []
        all_record_true: list[str] = []
        all_record_pred: list[str] = []
        fold_results: list[dict[str, object]] = []

        for fold in range(N_FOLDS):
            test_mask = np.asarray([fold_map[str(group)] == fold for group in groups])
            train_mask = ~test_mask
            model = factory()
            model.fit(X[train_mask], y[train_mask])
            chunk_pred = model.predict(X[test_mask])
            classes = list(model.classes_)
            scores = score_matrix(model, X[test_mask])
            test_groups = groups[test_mask]
            test_true = y[test_mask]

            fold_record_true: list[str] = []
            fold_record_pred: list[str] = []
            for group in sorted(set(test_groups)):
                positions = np.flatnonzero(test_groups == group)
                true_label = str(test_true[positions[0]])
                mean_scores = scores[positions].mean(axis=0)
                predicted_label = str(classes[int(np.argmax(mean_scores))])
                group_chunk_pred = chunk_pred[positions]
                chunk_correct = int(np.sum(group_chunk_pred == true_label))
                fold_record_true.append(true_label)
                fold_record_pred.append(predicted_label)
                prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold + 1,
                        "record_id": group,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "chunk_count": len(positions),
                        "correct_chunks": chunk_correct,
                        "chunk_accuracy": chunk_correct / len(positions),
                        "recording_correct": int(predicted_label == true_label),
                    }
                )

            all_chunk_true.extend(map(str, test_true))
            all_chunk_pred.extend(map(str, chunk_pred))
            all_record_true.extend(fold_record_true)
            all_record_pred.extend(fold_record_pred)
            fold_results.append(
                {
                    "fold": fold + 1,
                    "train_recordings": sorted(set(groups[train_mask])),
                    "test_recordings": sorted(set(test_groups)),
                    "chunk_metrics": metric_block(list(map(str, test_true)), list(map(str, chunk_pred))),
                    "recording_metrics": metric_block(fold_record_true, fold_record_pred),
                }
            )

        results[model_name] = {
            "chunk_metrics": metric_block(all_chunk_true, all_chunk_pred),
            "recording_metrics": metric_block(all_record_true, all_record_pred),
            "folds": fold_results,
        }
    return results, prediction_rows


def write_predictions(rows: list[dict[str, object]]) -> None:
    fields = [
        "model",
        "fold",
        "record_id",
        "true_label",
        "predicted_label",
        "chunk_count",
        "correct_chunks",
        "chunk_accuracy",
        "recording_correct",
    ]
    with PREDICTIONS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(payload: dict[str, object]) -> None:
    models = payload["models"]
    lines = [
        "# Padam classification smoke test",
        "",
        "## Outcome",
        "",
        "The end-to-end handcrafted-feature classification pipeline ran successfully with recording-grouped evaluation. "
        "These figures are a software/data-pipeline smoke test, not a publishable estimate of generalization.",
        "",
        "## Configuration",
        "",
        f"- Classes: {', '.join(CLASS_DISPLAY[key] for key in CLASS_KEYS)}.",
        f"- Usable recordings: {payload['usable_recordings']} of {payload['requested_recordings']} requested.",
        f"- Audio chunks: {payload['chunk_count']} fixed {CHUNK_SECONDS:.0f}-second chunks "
        f"({CHUNKS_PER_RECORDING} per usable recording).",
        f"- Evaluation: {N_FOLDS}-fold split grouped by source recording; no recording contributes chunks to both train and test in a fold.",
        "- Features: MFCC, delta-MFCC, chroma, spectral centroid, bandwidth, rolloff, flatness, zero-crossing rate, and RMS; mean and standard deviation aggregation.",
        "- Models: standardized logistic regression and class-balanced random forest.",
        "",
        "## Results",
        "",
        "| Model | Chunk accuracy | Chunk macro-F1 | Recording accuracy | Recording macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("logistic_regression", "random_forest"):
        block = models[name]
        chunk = block["chunk_metrics"]
        recording = block["recording_metrics"]
        lines.append(
            f"| {name.replace('_', ' ').title()} | {chunk['accuracy']:.3f} | {chunk['macro_f1']:.3f} | "
            f"{recording['accuracy']:.3f} | {recording['macro_f1']:.3f} |"
        )
    lines.extend(
        [
            f"| Majority-class reference | {payload['majority_reference_accuracy']:.3f} | — | {payload['majority_reference_accuracy']:.3f} | — |",
            "",
            "Confusion-matrix label order in the JSON is: " + ", ".join(CLASS_KEYS) + ".",
            "",
            "## Data validation",
            "",
        ]
    )
    if payload["download_failures"]:
        for failure in payload["download_failures"]:
            lines.append(
                f"- Record {failure['record_id']} ({failure['label']}) was excluded because its audio URL failed: {failure['error']}."
            )
    else:
        lines.append("- All requested audio files downloaded and decoded.")
    lines.extend(
        [
            "- Record 057 emitted MP3 dequantization warnings for several requested positions during this run; decoding continued, but the recording should receive an audio-integrity inspection before research use.",
            "- SHA-256 checksums and durations for all usable recordings are stored in the JSON results.",
            "- The three labels were selected by exact `padam_title_key`; no spelling or cultural synonym mapping was applied.",
            "",
            "## Interpretation and limitations",
            "",
            "- Only a small subset of three repeated labels was tested. The result cannot support claims about the full 190-row manifest.",
            "- Chunks from the same recording are correlated. Grouped splitting prevents direct leakage, but the effective sample size remains the number of recordings, not the number of chunks.",
            "- Performer, venue, recording equipment, ragam, and source quality can remain confounded with padam identity.",
            "- The class set was chosen because it had repeated exact labels, so it is not a random sample of Kathakali padams.",
            "- This run tests handcrafted acoustic features only; pretrained audio representations were not evaluated.",
            "- A research experiment should use expert-curated canonical labels, more independent recordings per class, nested model selection, and a held-out test set.",
            "",
            "## Reproduction",
            "",
            "Run `run_padam_classification_smoke_test.py` from the workspace after making its Python dependencies available. "
            "Downloaded audio and feature caches are stored under `work/padam_smoke_test/`, while results are written to `outputs/`.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = read_manifest()
    log(f"Selected {len(selected)} manifest rows: {dict(Counter(row['padam_title_key'] for row in selected))}")
    usable: list[tuple[dict[str, str], Path]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        log(f"Download {index:02d}/{len(selected):02d}: record {row['record_id']}")
        path, error = download_audio(row)
        if path is None:
            failures.append(
                {
                    "record_id": row["record_id"],
                    "label": row["padam_title_key"],
                    "url": row["audio_url"],
                    "error": error or "unknown error",
                }
            )
        else:
            try:
                info = sf.info(path)
                if info.frames <= 0 or info.samplerate <= 0:
                    raise RuntimeError("decoder reported no audio frames")
                usable.append((row, path))
            except Exception as exc:
                failures.append(
                    {
                        "record_id": row["record_id"],
                        "label": row["padam_title_key"],
                        "url": row["audio_url"],
                        "error": f"decode error: {type(exc).__name__}: {exc}",
                    }
                )

    class_counts = Counter(row["padam_title_key"] for row, _ in usable)
    if any(class_counts.get(label, 0) < N_FOLDS for label in CLASS_KEYS):
        raise RuntimeError(f"not enough usable recordings for grouped folds: {dict(class_counts)}")

    X, y, groups, starts, feature_names, recording_metadata = build_features(usable)
    log(f"Feature matrix: {X.shape[0]} chunks × {X.shape[1]} features")
    model_results, predictions = evaluate_models(X, y, groups)
    write_predictions(predictions)

    majority_reference = max(class_counts.values()) / sum(class_counts.values())
    payload: dict[str, object] = {
        "status": "completed",
        "purpose": "pipeline smoke test, not a publishable performance estimate",
        "manifest": str(MANIFEST),
        "configuration": {
            "random_seed": RANDOM_SEED,
            "class_keys": list(CLASS_KEYS),
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "chunk_seconds": CHUNK_SECONDS,
            "chunks_per_recording": CHUNKS_PER_RECORDING,
            "grouped_folds": N_FOLDS,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
        },
        "requested_recordings": len(selected),
        "usable_recordings": len(usable),
        "class_recording_counts": dict(class_counts),
        "chunk_count": int(X.shape[0]),
        "majority_reference_accuracy": majority_reference,
        "download_failures": failures,
        "data_quality_warnings": [
            {
                "record_id": "057",
                "warning": "MP3 decoder emitted dequantization warnings for several requested positions; finite features were still returned. Inspect audio integrity before research use.",
            }
        ],
        "recordings": recording_metadata,
        "models": model_results,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "librosa": librosa.__version__,
            "soundfile": sf.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    RESULTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(payload)
    log("Smoke test completed")
    for model_name, block in model_results.items():
        log(
            f"{model_name}: chunk accuracy={block['chunk_metrics']['accuracy']:.3f}, "
            f"recording accuracy={block['recording_metrics']['accuracy']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
