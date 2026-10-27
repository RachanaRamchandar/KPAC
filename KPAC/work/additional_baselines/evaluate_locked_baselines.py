#!/usr/bin/env python3
"""Evaluate additional baselines on the frozen 112-recording split."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
LOCKED = OUT / "kathakalipadam_mert_locked_benchmark.csv"
MERT_PREDICTIONS = OUT / "kathakalipadam_mert_classification_predictions.csv"
HANDCRAFTED = ROOT / "work" / "canonical_full_evaluation" / "record_features"
YAMNET = ROOT / "work" / "additional_baselines" / "yamnet"
RESULTS = OUT / "kpac_additional_baseline_results.json"
SUMMARY = OUT / "kpac_additional_baseline_summary.csv"
PREDICTIONS = OUT / "kpac_additional_baseline_predictions.csv"
CS = (0.01, 0.1, 1.0, 10.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def model(c: float):
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=c, dual="auto", class_weight="balanced", max_iter=30_000),
    )


def metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
    }


def bootstrap_accuracy(
    y: np.ndarray, predicted: np.ndarray, seed: int = 20261027
) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    values = np.empty(10_000, dtype=float)
    for index in range(len(values)):
        sample = rng.integers(0, n, n)
        values[index] = np.mean(y[sample] == predicted[sample])
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def inner_choice(x: np.ndarray, y: np.ndarray, outer_fold: int) -> tuple[float, float, float]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=18_000 + outer_fold)
    choices: list[tuple[float, float, float]] = []
    for c in CS:
        prediction = np.full(len(y), -1, dtype=int)
        for train, test in splitter.split(x, y):
            estimator = model(c)
            estimator.fit(x[train], y[train])
            prediction[test] = estimator.predict(x[test])
        score = metrics(y, prediction)
        choices.append((score["accuracy"], score["macro_f1"], -c))
    best = max(choices)
    return -best[2], best[0], best[1]


def evaluate(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    classes: np.ndarray,
    rows: list[dict[str, str]],
) -> tuple[dict, list[dict]]:
    predicted = np.full(len(y), -1, dtype=int)
    top3 = np.empty((len(y), 3), dtype=int)
    selections: list[dict] = []
    for outer_fold in sorted(np.unique(folds)):
        train = np.where(folds != outer_fold)[0]
        test = np.where(folds == outer_fold)[0]
        c, inner_accuracy, inner_macro_f1 = inner_choice(x[train], y[train], int(outer_fold))
        estimator = model(c)
        estimator.fit(x[train], y[train])
        predicted[test] = estimator.predict(x[test])
        decision = np.asarray(estimator.decision_function(x[test]), dtype=float)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        class_ids = np.asarray(estimator.named_steps["linearsvc"].classes_, dtype=int)
        order = np.argsort(decision, axis=1)[:, ::-1]
        top3[test] = class_ids[order[:, :3]]
        selections.append({
            "outer_fold": int(outer_fold),
            "C": c,
            "inner_accuracy": inner_accuracy,
            "inner_macro_f1": inner_macro_f1,
            "outer_test_size": int(len(test)),
        })
    result = metrics(y, predicted)
    result.update({
        "experiment": name,
        "recordings": int(len(y)),
        "classes": int(len(classes)),
        "top_3_accuracy": float(np.mean([target in choices for target, choices in zip(y, top3)])),
        "accuracy_bootstrap_95": bootstrap_accuracy(y, predicted),
        "outer_fold_selections": selections,
    })
    prediction_rows: list[dict] = []
    for index, row in enumerate(rows):
        prediction_rows.append({
            "experiment": name,
            "record_id": row["record_id"],
            "outer_fold": row["locked_outer_fold"],
            "true_class": classes[y[index]],
            "predicted_class": classes[predicted[index]],
            "correct": int(y[index] == predicted[index]),
            "top_3_classes": " | ".join(classes[top3[index]]),
        })
    return result, prediction_rows


def artist_metadata_baseline(
    rows: list[dict[str, str]], y: np.ndarray, folds: np.ndarray, classes: np.ndarray
) -> tuple[dict, list[dict]]:
    predicted = np.full(len(y), -1, dtype=int)
    top3 = np.empty((len(y), 3), dtype=int)
    for outer_fold in sorted(np.unique(folds)):
        train = np.where(folds != outer_fold)[0]
        test = np.where(folds == outer_fold)[0]
        global_counts = Counter(y[train])
        artist_counts: dict[str, Counter] = defaultdict(Counter)
        for index in train:
            artist_counts[rows[index]["main_musician"].strip()][y[index]] += 1
        global_order = [label for label, _ in sorted(global_counts.items(), key=lambda item: (-item[1], item[0]))]
        for index in test:
            artist = rows[index]["main_musician"].strip()
            local = artist_counts.get(artist, Counter())
            order = [label for label, _ in sorted(local.items(), key=lambda item: (-item[1], item[0]))]
            order.extend(label for label in global_order if label not in order)
            predicted[index] = order[0]
            top3[index] = order[:3]
    result = metrics(y, predicted)
    result.update({
        "experiment": "main_artist_metadata_only",
        "recordings": int(len(y)),
        "classes": int(len(classes)),
        "top_3_accuracy": float(np.mean([target in choices for target, choices in zip(y, top3)])),
        "accuracy_bootstrap_95": bootstrap_accuracy(y, predicted, seed=20261028),
        "rule": "Training-fold label mode for the test record's main musician; training-fold global mode for unseen or missing musicians.",
    })
    predictions = [{
        "experiment": "main_artist_metadata_only",
        "record_id": row["record_id"],
        "outer_fold": row["locked_outer_fold"],
        "true_class": classes[y[index]],
        "predicted_class": classes[predicted[index]],
        "correct": int(y[index] == predicted[index]),
        "top_3_classes": " | ".join(classes[top3[index]]),
    } for index, row in enumerate(rows)]
    return result, predictions


def unseen_artist_subset(rows: list[dict[str, str]]) -> dict:
    predictions = {row["record_id"]: row for row in read_csv(MERT_PREDICTIONS)}
    selected: list[dict[str, str]] = []
    fold_details: list[dict] = []
    for outer_fold in (1, 2, 3):
        training_artists = {
            row["main_musician"].strip()
            for row in rows
            if int(row["locked_outer_fold"]) != outer_fold and row["main_musician"].strip()
        }
        fold_rows = [
            row for row in rows
            if int(row["locked_outer_fold"]) == outer_fold
            and row["main_musician"].strip()
            and row["main_musician"].strip() not in training_artists
        ]
        selected.extend(fold_rows)
        fold_details.append({
            "outer_fold": outer_fold,
            "unseen_artist_recordings": len(fold_rows),
            "unseen_main_musicians": sorted({row["main_musician"].strip() for row in fold_rows}),
        })
    y_true = np.asarray([row["padam_canonical_id"] for row in selected])
    y_pred = np.asarray([predictions[row["record_id"]]["predicted_class"] for row in selected])
    return {
        "definition": "Test recordings whose non-empty main musician is absent from that outer fold's training partition; this is a diagnostic subset, not an artist-disjoint resplit.",
        "recordings": len(selected),
        "main_musicians": len({row["main_musician"] for row in selected}),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(selected) else None,
        "macro_f1_observed_classes": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(selected) else None,
        "folds": fold_details,
        "record_ids": [row["record_id"] for row in selected],
    }


def main() -> int:
    rows = sorted(read_csv(LOCKED), key=lambda row: row["record_id"])
    encoder = LabelEncoder()
    y = encoder.fit_transform([row["padam_canonical_id"] for row in rows])
    classes = encoder.classes_
    folds = np.asarray([int(row["locked_outer_fold"]) for row in rows])

    handcrafted_vectors: list[np.ndarray] = []
    yamnet_vectors: list[np.ndarray] = []
    for row in rows:
        record_id = row["record_id"]
        handcrafted = np.load(HANDCRAFTED / f"{record_id}.npz", allow_pickle=True)
        chunks = np.asarray(handcrafted["X"], dtype=float)
        handcrafted_vectors.append(np.concatenate([chunks.mean(axis=0), chunks.std(axis=0)]))
        yamnet = np.load(YAMNET / f"{record_id}.npz", allow_pickle=True)
        embedding = np.asarray(yamnet["embedding_mean"], dtype=float)
        yamnet_vectors.append(np.concatenate([embedding.mean(axis=0), embedding.std(axis=0)]))

    results: list[dict] = []
    prediction_rows: list[dict] = []
    for name, matrix in (
        ("handcrafted_mfcc_chroma_spectral", np.vstack(handcrafted_vectors)),
        ("yamnet_frozen_embedding", np.vstack(yamnet_vectors)),
    ):
        result, predictions = evaluate(name, matrix, y, folds, classes, rows)
        results.append(result)
        prediction_rows.extend(predictions)
    artist_result, artist_predictions = artist_metadata_baseline(rows, y, folds, classes)
    results.append(artist_result)
    prediction_rows.extend(artist_predictions)

    payload = {
        "status": "complete_additional_exact_fold_diagnostics",
        "locked_assignment_sha256": "a558f3324b7051726b89f6678bce32df2a1cd549f0cc40a0206826956685bb50",
        "recordings": len(rows),
        "classes": len(classes),
        "folds": 3,
        "preprocessing_note": "For every inner split, StandardScaler and LinearSVC are instantiated as one pipeline and fitted only on that inner training subset.",
        "feature_protocol": {
            "handcrafted": "Six deterministic 20-second excerpts; MFCC, delta-MFCC, chroma and spectral summaries; record-level mean and standard deviation.",
            "yamnet": "Frozen TF Hub YAMNet/1 on the same six excerpts; record-level mean and standard deviation of chunk embedding means.",
        },
        "results": results,
        "mert_unseen_artist_subset": unseen_artist_subset(rows),
        "interpretation": "These additional diagnostics use the frozen outer folds. They broaden reference baselines but do not convert the original recording-held-out design into an artist- or performance-disjoint benchmark.",
    }
    RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_rows = [{
        "experiment": result["experiment"],
        "recordings": result["recordings"],
        "classes": result["classes"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "balanced_accuracy": result["balanced_accuracy"],
        "top_3_accuracy": result["top_3_accuracy"],
        "accuracy_ci95_low": result["accuracy_bootstrap_95"][0],
        "accuracy_ci95_high": result["accuracy_bootstrap_95"][1],
        "status": "additional_exact_fold_diagnostic",
    } for result in results]
    write_csv(SUMMARY, summary_rows, list(summary_rows[0]))
    write_csv(
        PREDICTIONS,
        prediction_rows,
        ["experiment", "record_id", "outer_fold", "true_class", "predicted_class", "correct", "top_3_classes"],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
