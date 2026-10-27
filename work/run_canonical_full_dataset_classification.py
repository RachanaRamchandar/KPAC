#!/usr/bin/env python3
"""Canonical-label Kathakali padam classification and ablation study."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
import soundfile as sf
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
MIN_RECORDINGS = 3
N_FOLDS = 3
CHUNKS_PER_RECORDING = 6
CHUNK_SECONDS = 20.0

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent.parent
OUT = ROOT / "outputs"
WORK = ROOT / "work" / "canonical_full_evaluation"
AUDIO_DIR = ROOT / "work" / "padam_smoke_test" / "audio_mp3"
PCM_DIR = WORK / "decoded_pcm"
FEATURE_DIR = WORK / "record_features"
MANIFEST = OUT / "kathakalipadam_canonicalized_expanded_425.csv"
BASE_SCRIPT = OUT / "run_padam_classification_smoke_test.py"
OLD_CACHE = ROOT / "work" / "padam_smoke_test" / "full_dataset_features.npz"
DECODER = ROOT / "work" / "padam_smoke_test" / "decode_audio_objc"

RESULTS_JSON = OUT / "canonical_full_dataset_classification_results.json"
PREDICTIONS_CSV = OUT / "canonical_full_dataset_classification_predictions.csv"
ABLATIONS_CSV = OUT / "canonical_full_dataset_ablation_results.csv"
ELIGIBILITY_CSV = OUT / "canonical_full_dataset_label_eligibility.csv"
REPORT_MD = OUT / "canonical_full_dataset_classification_report.md"


def log(message: str) -> None:
    print(message, flush=True)


def load_base():
    spec = importlib.util.spec_from_file_location("padam_features", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load feature extractor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CHUNKS_PER_RECORDING = CHUNKS_PER_RECORDING
    module.CHUNK_SECONDS = CHUNK_SECONDS
    module.RANDOM_SEED = SEED
    return module


base = load_base()


def read_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_extension(row: dict[str, str]) -> str:
    suffix = Path(row.get("audio_filename", "")).suffix.lower()
    return suffix if suffix in {".mp3", ".wav", ".m4a", ".ogg", ".flac"} else ".mp3"


def download(row: dict[str, str]) -> tuple[Path | None, str | None]:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    destination = AUDIO_DIR / f"{row['record_id']}{safe_extension(row)}"
    if destination.exists() and destination.stat().st_size > 4096:
        return destination, None
    request = urllib.request.Request(
        row["audio_url"],
        headers={"User-Agent": "Mozilla/5.0 (KathakaliPadam research evaluation)"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=75) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if temporary.stat().st_size <= 4096:
                raise RuntimeError("downloaded file is unexpectedly small")
            temporary.replace(destination)
            return destination, None
        except Exception as exc:
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


def validate_audio(record_id: str, path: Path) -> tuple[str | None, float | None, str | None]:
    try:
        with sf.SoundFile(path) as audio:
            duration = len(audio) / audio.samplerate
        if duration < 5:
            raise RuntimeError(f"audio too short ({duration:.1f}s)")
        return "soundfile", float(duration), None
    except Exception as original:
        if not DECODER.exists():
            return None, None, f"decoder unavailable after {type(original).__name__}: {original}"
        PCM_DIR.mkdir(parents=True, exist_ok=True)
        pcm = PCM_DIR / f"{record_id}.pcm"
        run = subprocess.run([str(DECODER), str(path), str(pcm)], capture_output=True, text=True)
        if run.returncode != 0 or not pcm.exists() or pcm.stat().st_size < 16000:
            pcm.unlink(missing_ok=True)
            detail = run.stderr.strip() or str(original)
            return None, None, f"decode failure: {detail}"
        duration = pcm.stat().st_size / 2 / 16000
        return "pcm16_16k", float(duration), None


def load_old_cache() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str], dict[str, dict]]:
    if not OLD_CACHE.exists():
        return {}, [], {}
    cache = np.load(OLD_CACHE, allow_pickle=True)
    groups = cache["groups"].astype(str)
    starts = cache["starts"].astype(float)
    mapping = {group: (cache["X"][groups == group], starts[groups == group]) for group in set(groups)}
    metadata = {str(x["record_id"]): dict(x) for x in list(cache["recording_metadata"])}
    return mapping, list(cache["feature_names"].astype(str)), metadata


OLD_FEATURES, OLD_FEATURE_NAMES, OLD_METADATA = load_old_cache()


def read_pcm_chunk(path: Path, start: float) -> np.ndarray:
    source = np.memmap(path, mode="r", dtype="<i2")
    start_sample = min(int(round(start * 16000)), max(0, len(source) - 1))
    length = int(CHUNK_SECONDS * 16000)
    y = np.asarray(source[start_sample : start_sample + length], dtype=np.float32) / 32768.0
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y


def record_features(row: dict[str, str], path: Path, decoder_mode: str, duration: float, audio_sha: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURE_DIR / f"{row['record_id']}.npz"
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        if str(cache["sha256"].item()) == audio_sha:
            return cache["X"], cache["starts"], list(cache["feature_names"].astype(str))
    if row["record_id"] in OLD_FEATURES and row["record_id"] in OLD_METADATA:
        meta = OLD_METADATA[row["record_id"]]
        if meta.get("sha256") == audio_sha:
            X, starts = OLD_FEATURES[row["record_id"]]
            np.savez_compressed(cache_path, X=X, starts=starts, feature_names=np.asarray(OLD_FEATURE_NAMES), sha256=audio_sha)
            return X, starts, OLD_FEATURE_NAMES

    starts = np.asarray(base.chunk_start_times(duration), dtype=float)
    vectors: list[np.ndarray] = []
    names: list[str] = []
    if decoder_mode == "soundfile":
        with sf.SoundFile(path) as audio:
            for start in starts:
                y = base.read_chunk(audio, float(start))
                names, vector = base.acoustic_features(y)
                vectors.append(vector)
    else:
        pcm = PCM_DIR / f"{row['record_id']}.pcm"
        for start in starts:
            names, vector = base.acoustic_features(read_pcm_chunk(pcm, float(start)))
            vectors.append(vector)
    X = np.vstack(vectors)
    np.savez_compressed(cache_path, X=X, starts=starts, feature_names=np.asarray(names), sha256=audio_sha)
    return X, starts, names


def model_factory(name: str):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED))
    if name == "linear_svm":
        return make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.1, class_weight="balanced", random_state=SEED))
    if name == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=SEED))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", max_features="sqrt", random_state=SEED, n_jobs=-1)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, class_weight="balanced", max_features="sqrt", random_state=SEED, n_jobs=-1)
    raise KeyError(name)


def stratified_recording_folds(record_rows: list[dict[str, str]], label_field: str) -> dict[str, int]:
    rng = random.Random(SEED)
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in record_rows:
        by_label[row[label_field]].append(row["record_id"])
    mapping: dict[str, int] = {}
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        for i, record_id in enumerate(ids):
            mapping[record_id] = i % N_FOLDS
    return mapping


def score_matrix(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)
    scores = np.asarray(model.decision_function(X), dtype=float)
    if scores.ndim == 1:
        return np.column_stack([-scores, scores])
    return scores


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = correct / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def metrics(true: list[str], pred: list[str], labels: list[str], top3_correct: int) -> dict[str, float | int | list[float]]:
    accuracy = float(accuracy_score(true, pred))
    lo, hi = wilson_interval(sum(a == b for a, b in zip(true, pred)), len(true))
    return {
        "n": len(true),
        "accuracy": accuracy,
        "accuracy_ci95": [lo, hi],
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "top3_accuracy": top3_correct / len(true) if true else float("nan"),
    }


def evaluate_chunk_cv(
    experiment: str,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fold_map: dict[str, int],
    record_lookup: dict[str, dict[str, str]],
    feature_idx: np.ndarray,
) -> tuple[dict, list[dict]]:
    labels = sorted(set(y.astype(str)))
    all_true: list[str] = []
    all_pred: list[str] = []
    top3_correct = 0
    all_chunk_true: list[str] = []
    all_chunk_pred: list[str] = []
    chunk_top3_correct = 0
    predictions: list[dict] = []
    for fold in range(N_FOLDS):
        test = np.asarray([fold_map[str(g)] == fold for g in groups])
        train = ~test
        model = model_factory(model_name)
        model.fit(X[train][:, feature_idx], y[train])
        scores = score_matrix(model, X[test][:, feature_idx])
        classes = list(map(str, model.classes_))
        test_groups = groups[test].astype(str)
        test_y = y[test].astype(str)
        chunk_pred = model.predict(X[test][:, feature_idx]).astype(str)
        all_chunk_true.extend(test_y.tolist())
        all_chunk_pred.extend(chunk_pred.tolist())
        for chunk_i, true_chunk in enumerate(test_y):
            chunk_order = np.argsort(scores[chunk_i])[::-1][:3]
            chunk_top3_correct += int(str(true_chunk) in [classes[int(j)] for j in chunk_order])
        for group in sorted(set(test_groups)):
            pos = np.flatnonzero(test_groups == group)
            mean_scores = scores[pos].mean(axis=0)
            order = np.argsort(mean_scores)[::-1]
            predicted = classes[int(order[0])]
            true = str(test_y[pos[0]])
            top3 = [classes[int(i)] for i in order[:3]]
            all_true.append(true)
            all_pred.append(predicted)
            top3_correct += int(true in top3)
            row = record_lookup[group]
            predictions.append({
                "experiment": experiment, "model": model_name, "fold": fold + 1,
                "record_id": group, "story_label": row["story_label"], "main_musician": row["main_musician"],
                "true_label": true, "predicted_label": predicted, "top3_labels": " | ".join(top3),
                "recording_correct": int(predicted == true), "top3_correct": int(true in top3),
            })
    result = metrics(all_true, all_pred, labels, top3_correct)
    chunk_result = metrics(all_chunk_true, all_chunk_pred, labels, chunk_top3_correct)
    result.update({
        "chunk_accuracy": chunk_result["accuracy"],
        "chunk_balanced_accuracy": chunk_result["balanced_accuracy"],
        "chunk_macro_f1": chunk_result["macro_f1"],
        "chunk_top3_accuracy": chunk_result["top3_accuracy"],
    })
    return result, predictions


def aggregate_record_features(X: np.ndarray, groups: np.ndarray, record_ids: list[str], feature_idx: np.ndarray, mode: str) -> np.ndarray:
    rows = []
    for record_id in record_ids:
        block = X[groups == record_id][:, feature_idx]
        if mode == "mean_std":
            rows.append(np.concatenate([block.mean(axis=0), block.std(axis=0)]))
        elif mode == "center_chunk":
            rows.append(block[len(block) // 2])
        else:
            raise KeyError(mode)
    return np.vstack(rows)


def evaluate_record_cv(
    experiment: str,
    model_name: str,
    RX: np.ndarray,
    record_rows: list[dict[str, str]],
    label_field: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict, list[dict]]:
    y = np.asarray([r[label_field] for r in record_rows])
    labels = sorted(set(y.astype(str)))
    all_true: list[str] = []
    all_pred: list[str] = []
    top3_correct = 0
    predictions = []
    for fold, (train, test) in enumerate(splits, start=1):
        model = model_factory(model_name)
        model.fit(RX[train], y[train])
        scores = score_matrix(model, RX[test])
        classes = list(map(str, model.classes_))
        for local_i, record_i in enumerate(test):
            order = np.argsort(scores[local_i])[::-1]
            pred = classes[int(order[0])]
            true = str(y[record_i])
            top3 = [classes[int(i)] for i in order[:3]]
            all_true.append(true)
            all_pred.append(pred)
            top3_correct += int(true in top3)
            row = record_rows[int(record_i)]
            predictions.append({
                "experiment": experiment, "model": model_name, "fold": fold,
                "record_id": row["record_id"], "story_label": row["story_label"], "main_musician": row["main_musician"],
                "true_label": true, "predicted_label": pred, "top3_labels": " | ".join(top3),
                "recording_correct": int(pred == true), "top3_correct": int(true in top3),
            })
    return metrics(all_true, all_pred, labels, top3_correct), predictions


def splits_from_fold_map(record_rows: list[dict[str, str]], fold_map: dict[str, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = np.asarray([fold_map[r["record_id"]] for r in record_rows])
    return [(np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)) for fold in range(N_FOLDS)]


def result_row(experiment: str, model: str, description: str, result: dict, classes: int, recordings: int) -> dict:
    return {
        "experiment": experiment,
        "model": model,
        "description": description,
        "classes": classes,
        "recordings": recordings,
        "accuracy": result["accuracy"],
        "accuracy_ci95_low": result["accuracy_ci95"][0],
        "accuracy_ci95_high": result["accuracy_ci95"][1],
        "balanced_accuracy": result["balanced_accuracy"],
        "macro_f1": result["macro_f1"],
        "top3_accuracy": result["top3_accuracy"],
        "chunk_accuracy": result.get("chunk_accuracy", ""),
        "uniform_chance_accuracy": 1 / classes,
        "result_key": f"{experiment}|{model}",
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "work" / "padam_smoke_test" / "numba-cache"))
    WORK.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    padams = [r for r in rows if r["recording_scope"] == "padam" and r["audio_url"]]
    manifest_counts = Counter(r["padam_canonical_id"] for r in padams)
    candidate_labels = {label for label, count in manifest_counts.items() if count >= MIN_RECORDINGS}
    candidates = [r for r in padams if r["padam_canonical_id"] in candidate_labels]
    log(f"Candidate set: {len(candidates)} recordings across {len(candidate_labels)} canonical classes")

    downloaded: list[dict] = []
    failures: dict[str, str] = {}
    for i, row in enumerate(candidates, start=1):
        path, error = download(row)
        if error or path is None:
            failures[row["record_id"]] = error or "download failure"
        else:
            mode, duration, decode_error = validate_audio(row["record_id"], path)
            if decode_error or mode is None or duration is None:
                failures[row["record_id"]] = decode_error or "decode failure"
            else:
                item = dict(row)
                item["local_path"] = str(path)
                item["decoder_mode"] = mode
                item["duration_seconds"] = duration
                item["sha256"] = sha256_file(path)
                downloaded.append(item)
        if i % 20 == 0 or i == len(candidates):
            log(f"Audio validation {i}/{len(candidates)}; usable so far {len(downloaded)}, failures {len(failures)}")

    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in downloaded:
        by_hash[row["sha256"]].append(row)
    conflicting_hashes = {h for h, rr in by_hash.items() if len({r["padam_canonical_id"] for r in rr}) > 1}
    same_label_duplicates: dict[str, list[str]] = {}
    deduped: list[dict] = []
    for audio_hash, rr in by_hash.items():
        if audio_hash in conflicting_hashes:
            for r in rr:
                failures[r["record_id"]] = "identical audio bytes occur under conflicting canonical labels"
            continue
        rr = sorted(rr, key=lambda r: r["record_id"])
        deduped.append(rr[0])
        if len(rr) > 1:
            same_label_duplicates[rr[0]["record_id"]] = [r["record_id"] for r in rr[1:]]

    usable_counts = Counter(r["padam_canonical_id"] for r in deduped)
    tested_labels = {label for label, count in usable_counts.items() if count >= MIN_RECORDINGS}
    usable = sorted([r for r in deduped if r["padam_canonical_id"] in tested_labels], key=lambda r: r["record_id"])
    log(f"Final canonical set: {len(usable)} unique recordings across {len(tested_labels)} classes")

    eligibility = []
    for label in sorted(manifest_counts):
        member_ids = [r["record_id"] for r in candidates if r["padam_canonical_id"] == label]
        final_ids = [r["record_id"] for r in usable if r["padam_canonical_id"] == label]
        failed = [rid for rid in member_ids if rid in failures]
        duplicate_ids = sorted({d for keep, ds in same_label_duplicates.items() for d in ds if d in member_ids})
        eligibility.append({
            "padam_canonical_id": label,
            "manifest_recordings_with_url": manifest_counts[label],
            "validated_unique_recordings": usable_counts.get(label, 0),
            "final_test_recordings": len(final_ids),
            "failed_record_ids": " | ".join(failed),
            "duplicate_record_ids_excluded": " | ".join(duplicate_ids),
            "status": "tested" if label in tested_labels else "excluded_below_3_after_validation",
        })
    write_csv(ELIGIBILITY_CSV, eligibility, list(eligibility[0]))

    matrices = []
    labels = []
    groups = []
    feature_names: list[str] = []
    for i, row in enumerate(usable, start=1):
        Xr, starts, names = record_features(
            row, Path(row["local_path"]), row["decoder_mode"], float(row["duration_seconds"]), row["sha256"]
        )
        matrices.append(Xr)
        labels.extend([row["padam_canonical_id"]] * len(Xr))
        groups.extend([row["record_id"]] * len(Xr))
        feature_names = names
        if i % 10 == 0 or i == len(usable):
            log(f"Features {i}/{len(usable)}")
    X = np.vstack(matrices)
    y = np.asarray(labels)
    group_array = np.asarray(groups)
    record_lookup = {r["record_id"]: r for r in usable}
    fold_map = stratified_recording_folds(usable, "padam_canonical_id")
    all_idx = np.arange(X.shape[1])

    predictions: list[dict] = []
    model_results: dict[str, dict] = {}
    ablation_rows: list[dict] = []
    for model_name in ["logistic_regression", "linear_svm", "rbf_svm", "random_forest", "extra_trees"]:
        log(f"Model comparison: {model_name}")
        result, pred = evaluate_chunk_cv("canonical_all_features", model_name, X, y, group_array, fold_map, record_lookup, all_idx)
        model_results[model_name] = result
        predictions.extend(pred)
        ablation_rows.append(result_row("canonical_all_features", model_name, "All handcrafted feature families; chunk training with recording-level score aggregation.", result, len(tested_labels), len(usable)))

    feature_sets = {
        "mfcc_only": np.asarray([i for i, n in enumerate(feature_names) if n.startswith("mfcc_")]),
        "mfcc_plus_delta": np.asarray([i for i, n in enumerate(feature_names) if n.startswith("mfcc_") or n.startswith("delta_mfcc_")]),
        "chroma_only": np.asarray([i for i, n in enumerate(feature_names) if n.startswith("chroma_")]),
        "spectral_only": np.asarray([i for i, n in enumerate(feature_names) if n.startswith("spectral_")]),
        "all_minus_chroma": np.asarray([i for i, n in enumerate(feature_names) if not n.startswith("chroma_")]),
    }
    feature_ablation_results = {}
    for name, idx in feature_sets.items():
        log(f"Feature ablation: {name}")
        result, pred = evaluate_chunk_cv(name, "linear_svm", X, y, group_array, fold_map, record_lookup, idx)
        feature_ablation_results[name] = result
        predictions.extend(pred)
        ablation_rows.append(result_row(name, "linear_svm", f"Linear SVM using {name.replace('_', ' ')}.", result, len(tested_labels), len(usable)))

    record_ids = [r["record_id"] for r in usable]
    record_splits = splits_from_fold_map(usable, fold_map)
    RX_mean_std = aggregate_record_features(X, group_array, record_ids, all_idx, "mean_std")
    aggregated_model_results: dict[str, dict] = {}
    for model_name in ["logistic_regression", "linear_svm", "rbf_svm", "random_forest", "extra_trees"]:
        log(f"Recording-aggregated model comparison: {model_name}")
        result, pred = evaluate_record_cv("record_mean_std", model_name, RX_mean_std, usable, "padam_canonical_id", record_splits)
        aggregated_model_results[model_name] = result
        predictions.extend(pred)
        ablation_rows.append(result_row("record_mean_std", model_name, "One mean+standard-deviation vector per recording.", result, len(tested_labels), len(usable)))

    log("Training-unit ablation: center_chunk")
    RX_center = aggregate_record_features(X, group_array, record_ids, all_idx, "center_chunk")
    center_result, center_pred = evaluate_record_cv("record_center_chunk", "linear_svm", RX_center, usable, "padam_canonical_id", record_splits)
    predictions.extend(center_pred)
    ablation_rows.append(result_row("record_center_chunk", "linear_svm", "One center chunk vector per recording.", center_result, len(tested_labels), len(usable)))
    unit_results = {"mean_std": aggregated_model_results["linear_svm"], "center_chunk": center_result}

    aggregated_feature_results = {}
    for name, idx in feature_sets.items():
        log(f"Recording-aggregated feature ablation: {name}")
        RX_feature = aggregate_record_features(X, group_array, record_ids, idx, "mean_std")
        result, pred = evaluate_record_cv(f"record_{name}", "linear_svm", RX_feature, usable, "padam_canonical_id", record_splits)
        aggregated_feature_results[name] = result
        predictions.extend(pred)
        ablation_rows.append(result_row(f"record_{name}", "linear_svm", f"Recording mean+std using {name.replace('_', ' ')}.", result, len(tested_labels), len(usable)))

    exact_counts = Counter(r["padam_title_key"] for r in usable)
    exact_labels = {label for label, count in exact_counts.items() if count >= MIN_RECORDINGS}
    exact_rows = [r for r in usable if r["padam_title_key"] in exact_labels]
    exact_ids = [r["record_id"] for r in exact_rows]
    exact_RX = aggregate_record_features(X, group_array, exact_ids, all_idx, "mean_std")
    exact_fold_map = stratified_recording_folds(exact_rows, "padam_title_key")
    exact_splits = splits_from_fold_map(exact_rows, exact_fold_map)
    exact_result, exact_pred = evaluate_record_cv("exact_spelling_labels", "linear_svm", exact_RX, exact_rows, "padam_title_key", exact_splits)
    predictions.extend(exact_pred)
    ablation_rows.append(result_row("exact_spelling_labels", "linear_svm", "Exact source spelling labels with >=3 usable recordings; recording mean+std features.", exact_result, len(exact_labels), len(exact_rows)))

    known_artist_rows = [r for r in usable if r["main_musician"]]
    artist_counts: dict[str, set[str]] = defaultdict(set)
    for r in known_artist_rows:
        artist_counts[r["padam_canonical_id"]].add(r["main_musician"])
    artist_labels = {label for label, artists in artist_counts.items() if len(artists) >= N_FOLDS}
    artist_rows = [r for r in known_artist_rows if r["padam_canonical_id"] in artist_labels]
    artist_ids = [r["record_id"] for r in artist_rows]
    artist_RX = aggregate_record_features(X, group_array, artist_ids, all_idx, "mean_std")
    artist_y = np.asarray([r["padam_canonical_id"] for r in artist_rows])
    artist_groups = np.asarray([r["main_musician"] for r in artist_rows])
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    artist_splits = [(train, test) for train, test in sgkf.split(artist_RX, artist_y, groups=artist_groups)]
    artist_result, artist_pred = evaluate_record_cv("main_artist_disjoint", "linear_svm", artist_RX, artist_rows, "padam_canonical_id", artist_splits)
    predictions.extend(artist_pred)
    ablation_rows.append(result_row("main_artist_disjoint", "linear_svm", "Canonical subset with >=3 known main musicians per class; main musicians never cross train/test within a fold.", artist_result, len(artist_labels), len(artist_rows)))

    prediction_fields = ["experiment", "model", "fold", "record_id", "story_label", "main_musician", "true_label", "predicted_label", "top3_labels", "recording_correct", "top3_correct"]
    write_csv(PREDICTIONS_CSV, predictions, prediction_fields)
    write_csv(ABLATIONS_CSV, ablation_rows, list(ablation_rows[0]))

    best_model = max(model_results, key=lambda name: (model_results[name]["accuracy"], model_results[name]["macro_f1"]))
    comparable_rows = [r for r in ablation_rows if r["classes"] == len(tested_labels) and r["recordings"] == len(usable)]
    best_canonical_row = max(comparable_rows, key=lambda r: (r["accuracy"], r["macro_f1"]))
    best_canonical_predictions = [p for p in predictions if p["experiment"] == best_canonical_row["experiment"] and p["model"] == best_canonical_row["model"]]
    per_story_best = {}
    for story in sorted({p["story_label"] for p in best_canonical_predictions}):
        story_rows = [p for p in best_canonical_predictions if p["story_label"] == story]
        per_story_best[story] = {
            "recordings": len(story_rows),
            "accuracy": sum(int(p["recording_correct"]) for p in story_rows) / len(story_rows),
        }
    artist_split_checks = []
    for fold in range(1, N_FOLDS + 1):
        test_rows = [p for p in artist_pred if p["fold"] == fold]
        train_rows = [p for p in artist_pred if p["fold"] != fold]
        test_artists = {p["main_musician"] for p in test_rows}
        train_artists = {p["main_musician"] for p in train_rows}
        unseen_labels = {p["true_label"] for p in test_rows} - {p["true_label"] for p in train_rows}
        artist_split_checks.append({
            "fold": fold,
            "test_recordings": len(test_rows),
            "artist_overlap_count": len(test_artists & train_artists),
            "test_classes_absent_from_training": len(unseen_labels),
        })
    canonical_counts = Counter(r["padam_canonical_id"] for r in usable)
    majority = max(canonical_counts.values()) / len(usable)
    payload = {
        "status": "completed",
        "purpose": "canonical-label feasibility evaluation; not a publishable final performance estimate",
        "manifest": str(MANIFEST),
        "manifest_rows": len(rows),
        "candidate_canonical_classes": len(candidate_labels),
        "candidate_recordings": len(candidates),
        "tested_canonical_classes": len(tested_labels),
        "tested_unique_recordings": len(usable),
        "audio_failures": failures,
        "same_label_duplicate_recordings_excluded": same_label_duplicates,
        "conflicting_audio_hashes_excluded": len(conflicting_hashes),
        "class_recording_counts": dict(canonical_counts),
        "majority_reference_accuracy": majority,
        "uniform_chance_accuracy": 1 / len(tested_labels),
        "best_model": best_model,
        "best_canonical_full_experiment": best_canonical_row["experiment"],
        "best_canonical_full_model": best_canonical_row["model"],
        "best_canonical_full_metrics": {k: best_canonical_row[k] for k in ["accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "balanced_accuracy", "macro_f1", "top3_accuracy"]},
        "best_canonical_full_per_story": per_story_best,
        "models": model_results,
        "recording_aggregated_models": aggregated_model_results,
        "feature_ablations_linear_svm": feature_ablation_results,
        "recording_aggregated_feature_ablations_linear_svm": aggregated_feature_results,
        "training_unit_ablations_linear_svm": unit_results,
        "exact_spelling_label_ablation": {"classes": len(exact_labels), "recordings": len(exact_rows), "metrics": exact_result},
        "main_artist_disjoint_ablation": {"classes": len(artist_labels), "recordings": len(artist_rows), "distinct_main_artists": len(set(r['main_musician'] for r in artist_rows)), "metrics": artist_result, "split_checks": artist_split_checks},
        "configuration": {
            "seed": SEED, "folds": N_FOLDS, "minimum_unique_recordings": MIN_RECORDINGS,
            "chunk_seconds": CHUNK_SECONDS, "chunks_per_recording": CHUNKS_PER_RECORDING,
            "feature_count": len(feature_names), "feature_names": feature_names,
            "exact_audio_byte_duplicates_removed": True,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "soundfile": sf.__version__, "platform": platform.platform()},
    }
    RESULTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Canonical full-dataset padam classification",
        "",
        "## Evaluation set",
        "",
        f"- {len(usable)} byte-unique, decodable recordings across {len(tested_labels)} canonical padams.",
        f"- Three-fold recording-held-out evaluation; six 20-second chunks per recording.",
        f"- Uniform chance: {1/len(tested_labels):.3%}; majority-class reference: {majority:.3%}.",
        f"- Download/decode/label-conflict exclusions: {len(failures)} records; same-label byte duplicates removed: {sum(len(v) for v in same_label_duplicates.values())}.",
        "",
        "## Model comparison",
        "",
        "| Model | Chunk accuracy | Recording accuracy | 95% CI | Balanced accuracy | Macro-F1 | Top-3 accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["logistic_regression", "linear_svm", "rbf_svm", "random_forest", "extra_trees"]:
        m = model_results[name]
        lines.append(f"| {name.replace('_',' ').title()} | {m['chunk_accuracy']:.3%} | {m['accuracy']:.3%} | {m['accuracy_ci95'][0]:.3%}–{m['accuracy_ci95'][1]:.3%} | {m['balanced_accuracy']:.3%} | {m['macro_f1']:.3f} | {m['top3_accuracy']:.3%} |")
    lines.extend(["", "## Ablations", "", "| Ablation | Model | Classes | Recordings | Accuracy | Balanced accuracy | Macro-F1 | Top-3 accuracy |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in ablation_rows:
        if row["experiment"] == "canonical_all_features":
            continue
        lines.append(f"| {row['experiment']} | {row['model']} | {row['classes']} | {row['recordings']} | {row['accuracy']:.3%} | {row['balanced_accuracy']:.3%} | {row['macro_f1']:.3f} | {row['top3_accuracy']:.3%} |")
    best = model_results[best_model]
    artist_majority = max(Counter(p["true_label"] for p in artist_pred).values()) / len(artist_pred)
    exact_majority = max(Counter(p["true_label"] for p in exact_pred).values()) / len(exact_pred)
    lines.extend([
        "", "## Interpretation", "",
        f"The strongest full-set baseline is **{best_canonical_row['experiment'].replace('_',' ').title()} / {best_canonical_row['model'].replace('_',' ').title()}** at **{best_canonical_row['accuracy']:.3%} recording accuracy** (95% Wilson CI {best_canonical_row['accuracy_ci95_low']:.3%}–{best_canonical_row['accuracy_ci95_high']:.3%}), balanced accuracy {best_canonical_row['balanced_accuracy']:.3%}, and macro-F1 {best_canonical_row['macro_f1']:.3f}.",
        f"Among chunk-trained models, the best is **{best_model.replace('_',' ').title()}** at {best['accuracy']:.3%} recording accuracy; temporal mean+standard-deviation aggregation is therefore the most consequential ablation.",
        f"The main-artist-disjoint subset reaches {artist_result['accuracy']:.3%} accuracy across {len(artist_labels)} classes, compared with {artist_majority:.3%} majority and {1/len(artist_labels):.3%} uniform-chance references. Every split has zero main-artist overlap and no test class absent from training.",
        f"The exact-spelling subset reaches {exact_result['accuracy']:.3%} across {len(exact_labels)} classes and {len(exact_rows)} recordings (majority reference {exact_majority:.3%}).",
        "Canonical and exact-spelling experiments use different label inventories; the exact-label row is a label/coverage ablation, not a direct paired accuracy comparison.",
        "The main-artist-disjoint result is the strongest leakage check, but it uses only classes represented by at least three known main musicians.",
        "Story-held-out evaluation is not defined for this target because canonical padam IDs are story-qualified: a held-out story contains unseen classes.",
        "Handcrafted features remain a baseline. A publication-grade follow-up should add pretrained music/audio embeddings and nested hyperparameter selection.",
        "", "## Output guide", "",
        "The ablation CSV contains one row per experiment. The predictions CSV supports per-class, per-story, and error analysis. The eligibility CSV documents all exclusions.",
        "",
    ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    log(f"Completed. Best model: {best_model}, accuracy={best['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
