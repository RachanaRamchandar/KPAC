#!/usr/bin/env python3
"""Run the preregistered nested evaluation on the locked MERT benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
WORK = ROOT / "work"
LOCKED = OUT / "kathakalipadam_mert_locked_benchmark.csv"
LOCK_META = OUT / "kathakalipadam_mert_locked_benchmark.json"
PROTOCOL = OUT / "kathakalipadam_mert_evaluation_protocol.json"
EMBEDDINGS = WORK / "mert_eval" / "embeddings"
RESULTS = OUT / "kathakalipadam_mert_classification_results.json"
PREDICTIONS = OUT / "kathakalipadam_mert_classification_predictions.csv"
ABLATION = OUT / "kathakalipadam_mert_ablation_results.csv"
CLASS_METRICS = OUT / "kathakalipadam_mert_per_class_metrics.csv"
CONFUSION = OUT / "kathakalipadam_mert_confusion_matrix.csv"
REPORT = OUT / "kathakalipadam_mert_classification_report.md"

REPRESENTATIONS = (
    "layer_6_mean_std_across_chunks",
    "layer_9_mean_std_across_chunks",
    "layer_12_mean_std_across_chunks",
    "last_4_layers_mean_std_across_chunks",
)
CS = (0.01, 0.1, 1.0, 10.0)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def representation(cache: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    layers = np.asarray(cache["layer_chunk_means"], dtype=np.float32)
    if name.startswith("layer_6"):
        chunks = layers[6]
    elif name.startswith("layer_9"):
        chunks = layers[9]
    elif name.startswith("layer_12"):
        chunks = layers[12]
    elif name.startswith("last_4"):
        chunks = layers[9:13].mean(axis=0)
    else:
        raise KeyError(name)
    return np.concatenate([chunks.mean(axis=0), chunks.std(axis=0)]).astype(np.float32)


def model(c: float):
    return make_pipeline(
        StandardScaler(),
        LinearSVC(C=c, dual="auto", class_weight="balanced", max_iter=30_000),
    )


def inner_score(x: np.ndarray, y: np.ndarray, c: float, outer_fold: int) -> tuple[float, float]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=18_000 + outer_fold)
    predictions = np.empty_like(y)
    for train, test in splitter.split(x, y):
        estimator = model(c)
        estimator.fit(x[train], y[train])
        predictions[test] = estimator.predict(x[test])
    return float(accuracy_score(y, predictions)), float(f1_score(y, predictions, average="macro"))


def nested_predictions(
    matrices: dict[str, np.ndarray], y: np.ndarray, folds: np.ndarray, allowed: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    predicted = np.full(len(y), -1, dtype=int)
    decisions = np.full((len(y), len(np.unique(y))), np.nan, dtype=float)
    selections: list[dict] = []
    for outer_fold in sorted(np.unique(folds)):
        train = folds != outer_fold
        test = folds == outer_fold
        candidates: list[tuple[float, float, str, float]] = []
        for rep in allowed:
            for c in CS:
                acc, macro = inner_score(matrices[rep][train], y[train], c, int(outer_fold))
                candidates.append((acc, macro, rep, c))
        # Accuracy, macro-F1, then preregistered representation order and smaller C.
        best = max(
            candidates,
            key=lambda item: (
                item[0], item[1], -allowed.index(item[2]), -CS.index(item[3])
            ),
        )
        inner_acc, inner_macro, rep, c = best
        estimator = model(c)
        estimator.fit(matrices[rep][train], y[train])
        predicted[test] = estimator.predict(matrices[rep][test])
        raw = estimator.decision_function(matrices[rep][test])
        decisions[np.ix_(np.flatnonzero(test), np.asarray(estimator.classes_, dtype=int))] = raw
        selections.append(
            {
                "outer_fold": int(outer_fold),
                "representation": rep,
                "C": c,
                "inner_accuracy": inner_acc,
                "inner_macro_f1": inner_macro,
                "outer_test_size": int(test.sum()),
            }
        )
    if np.any(predicted < 0) or np.any(~np.isfinite(decisions)):
        raise RuntimeError("Incomplete nested predictions")
    return predicted, decisions, selections


def metrics(y: np.ndarray, predicted: np.ndarray, decisions: np.ndarray) -> dict[str, float]:
    labels = np.arange(len(np.unique(y)))
    return {
        "recording_accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "top_3_accuracy": float(top_k_accuracy_score(y, decisions, k=3, labels=labels)),
    }


def bootstrap_interval(y: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260809)
    values = np.empty(10_000, dtype=float)
    for i in range(len(values)):
        sample = rng.integers(0, len(y), size=len(y))
        values[i] = np.mean(y[sample] == predicted[sample])
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def story_conditioned_predictions(
    matrices: dict[str, np.ndarray], y: np.ndarray, folds: np.ndarray, stories: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Post-hoc diagnostic: assume story metadata is known at inference time."""
    predicted = np.full(len(y), -1, dtype=int)
    decisions = np.full((len(y), len(np.unique(y))), -1e9, dtype=float)
    selections: list[dict] = []
    for story in sorted(np.unique(stories)):
        in_story = stories == story
        for outer_fold in sorted(np.unique(folds)):
            train = in_story & (folds != outer_fold)
            test = in_story & (folds == outer_fold)
            candidates: list[tuple[float, float, str, float]] = []
            for rep in REPRESENTATIONS:
                for c in CS:
                    acc, macro = inner_score(matrices[rep][train], y[train], c, int(outer_fold))
                    candidates.append((acc, macro, rep, c))
            inner_acc, inner_macro, rep, c = max(
                candidates,
                key=lambda item: (
                    item[0], item[1], -REPRESENTATIONS.index(item[2]), -CS.index(item[3])
                ),
            )
            estimator = model(c)
            estimator.fit(matrices[rep][train], y[train])
            predicted[test] = estimator.predict(matrices[rep][test])
            raw = estimator.decision_function(matrices[rep][test])
            if raw.ndim == 1:
                # LinearSVC emits one score for binary classification.
                raw = np.column_stack([-raw, raw])
            decisions[np.ix_(np.flatnonzero(test), np.asarray(estimator.classes_, dtype=int))] = raw
            selections.append(
                {
                    "story": story,
                    "outer_fold": int(outer_fold),
                    "representation": rep,
                    "C": c,
                    "inner_accuracy": inner_acc,
                    "inner_macro_f1": inner_macro,
                    "outer_test_size": int(test.sum()),
                }
            )
    if np.any(predicted < 0):
        raise RuntimeError("Incomplete story-conditioned predictions")
    return predicted, decisions, selections


def main() -> int:
    lock_meta = json.loads(LOCK_META.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = read_rows(LOCKED)
    assignment = "\n".join(
        f'{row["record_id"]},{row["padam_canonical_id"]},{row["audio_sha256"]},{row["locked_outer_fold"]}'
        for row in rows
    )
    digest = hashlib.sha256(assignment.encode()).hexdigest()
    if digest != lock_meta["assignment_sha256"]:
        raise RuntimeError("Locked assignment digest changed")
    if protocol["benchmark_assignment_sha256"] != digest:
        raise RuntimeError("Protocol does not reference the locked assignment")

    encoder = LabelEncoder()
    labels = np.asarray([row["padam_canonical_id"] for row in rows])
    y = encoder.fit_transform(labels)
    folds = np.asarray([int(row["locked_outer_fold"]) for row in rows])
    stories = np.asarray([row["story_label"] for row in rows])
    matrices: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATIONS}
    commits: set[str] = set()
    for row in rows:
        path = EMBEDDINGS / f'{row["record_id"]}.npz'
        if not path.exists():
            raise RuntimeError(f"Missing MERT embedding cache: {path}")
        cache = np.load(path, allow_pickle=False)
        if str(cache["source_sha256"].item()) != row["audio_sha256"]:
            raise RuntimeError(f'Embedding/audio mismatch for {row["record_id"]}')
        commits.add(str(cache["model_commit"].item()))
        for name in REPRESENTATIONS:
            matrices[name].append(representation(cache, name))
    if len(commits) != 1:
        raise RuntimeError(f"Expected one model commit, found {commits}")
    x_by_rep = {name: np.vstack(values) for name, values in matrices.items()}

    primary_pred, primary_decisions, selections = nested_predictions(
        x_by_rep, y, folds, REPRESENTATIONS
    )
    primary = metrics(y, primary_pred, primary_decisions)
    ci_low, ci_high = bootstrap_interval(y, primary_pred)
    primary["accuracy_bootstrap_95_low"] = ci_low
    primary["accuracy_bootstrap_95_high"] = ci_high

    ablations: list[dict] = [
        {
            "experiment": "preregistered_nested_representation_selection",
            "recording_accuracy": primary["recording_accuracy"],
            "macro_f1": primary["macro_f1"],
            "balanced_accuracy": primary["balanced_accuracy"],
            "top_3_accuracy": primary["top_3_accuracy"],
            "outer_fold_selected_C": ";".join(
                f'{item["outer_fold"]}:{item["C"]}' for item in selections
            ),
            "status": "primary_locked_result",
        }
    ]
    ablation_predictions: dict[str, np.ndarray] = {}
    for name in REPRESENTATIONS:
        pred, decision, selected = nested_predictions(x_by_rep, y, folds, (name,))
        score = metrics(y, pred, decision)
        ablation_predictions[name] = pred
        ablations.append(
            {
                "experiment": name,
                **score,
                "outer_fold_selected_C": ";".join(
                    f'{item["outer_fold"]}:{item["C"]}' for item in selected
                ),
                "status": "secondary_nested_ablation",
            }
        )

    story_mert_pred, story_mert_decisions, story_mert_selections = story_conditioned_predictions(
        x_by_rep, y, folds, stories
    )
    story_mert_metrics = metrics(y, story_mert_pred, story_mert_decisions)
    ablations.append(
        {
            "experiment": "known_story_plus_nested_MERT",
            **story_mert_metrics,
            "outer_fold_selected_C": "story-specific nested selection",
            "status": "posthoc_metadata_conditioned_diagnostic",
        }
    )

    majority_pred = np.empty_like(y)
    story_pred = np.empty_like(y)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        majority = Counter(y[train]).most_common(1)[0][0]
        majority_pred[test] = majority
        for story in np.unique(stories[test]):
            subset = train & (stories == story)
            fallback = majority
            prediction = Counter(y[subset]).most_common(1)[0][0] if subset.any() else fallback
            story_pred[test & (stories == story)] = prediction
    for name, pred, status in (
        ("training_majority_label", majority_pred, "non_audio_baseline"),
        ("known_story_training_majority", story_pred, "metadata_oracle_diagnostic"),
    ):
        ablations.append(
            {
                "experiment": name,
                "recording_accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "top_3_accuracy": "",
                "outer_fold_selected_C": "",
                "status": status,
            }
        )

    prediction_rows: list[dict] = []
    for i, row in enumerate(rows):
        ranked = np.argsort(primary_decisions[i])[::-1]
        prediction_rows.append(
            {
                **row,
                "true_class": labels[i],
                "predicted_class": encoder.inverse_transform([primary_pred[i]])[0],
                "correct": int(y[i] == primary_pred[i]),
                "top_3_classes": " | ".join(encoder.inverse_transform(ranked[:3])),
                "top_3_correct": int(y[i] in ranked[:3]),
            }
        )
    write_rows(PREDICTIONS, prediction_rows, list(prediction_rows[0]))
    write_rows(ABLATION, ablations, list(ablations[0]))

    precision, recall, f1, support = precision_recall_fscore_support(
        y, primary_pred, labels=np.arange(len(encoder.classes_)), zero_division=0
    )
    class_rows = []
    for i, class_name in enumerate(encoder.classes_):
        class_rows.append(
            {
                "padam_canonical_id": class_name,
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "correct": int(((y == i) & (primary_pred == i)).sum()),
            }
        )
    write_rows(CLASS_METRICS, class_rows, list(class_rows[0]))

    matrix = confusion_matrix(y, primary_pred, labels=np.arange(len(encoder.classes_)))
    confusion_rows = []
    for i, true_class in enumerate(encoder.classes_):
        confusion_rows.append(
            {"true_class": true_class, **{name: int(matrix[i, j]) for j, name in enumerate(encoder.classes_)}}
        )
    write_rows(CONFUSION, confusion_rows, list(confusion_rows[0]))

    result = {
        "status": "complete_locked_nested_evaluation",
        "target_accuracy": 0.40,
        "target_reached": primary["recording_accuracy"] >= 0.40,
        "recordings": len(rows),
        "classes": len(encoder.classes_),
        "stories": len(np.unique(stories)),
        "locked_assignment_sha256": digest,
        "mert_model": "m-a-p/MERT-v1-95M",
        "mert_resolved_commit": next(iter(commits)),
        "primary_metrics": primary,
        "outer_fold_selections": selections,
        "class_support": {name: int((labels == name).sum()) for name in encoder.classes_},
        "ablation_results": ablations,
        "posthoc_known_story_mert_metrics": story_mert_metrics,
        "posthoc_known_story_mert_selections": story_mert_selections,
        "methodological_note": "The outer folds were frozen before MERT download or extraction; all representation and C choices were made inside each outer training partition.",
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    weakest = sorted(class_rows, key=lambda row: (row["recall"], row["support"]))[:5]
    report = f"""# Locked MERT padam-classification evaluation

## Result

- Recordings: **{len(rows)}**
- Canonical padam classes: **{len(encoder.classes_)}** across **{len(np.unique(stories))}** stories
- Accuracy: **{primary['recording_accuracy']:.2%}** (recording bootstrap 95% interval: {ci_low:.2%}–{ci_high:.2%})
- Macro-F1: **{primary['macro_f1']:.2%}**
- Balanced accuracy: **{primary['balanced_accuracy']:.2%}**
- Top-3 accuracy: **{primary['top_3_accuracy']:.2%}**
- 40% target reached: **{'yes' if result['target_reached'] else 'no'}**

## Locked design

The 112-recording outer evaluation assignment (SHA-256 `{digest}`) was written before MERT was downloaded or any MERT feature was extracted. For each outer fold, MERT representation and linear-SVM C were chosen only by three-fold CV inside the outer training partition. Each outer test partition was predicted once.

## Fold selections

"""
    for item in selections:
        report += (
            f"- Fold {item['outer_fold']}: `{item['representation']}`, C={item['C']}; "
            f"inner accuracy {item['inner_accuracy']:.2%}; n={item['outer_test_size']}\n"
        )
    report += "\n## Weakest classes (out-of-fold recall)\n\n"
    for item in weakest:
        report += f"- `{item['padam_canonical_id']}`: {item['recall']:.2%} ({item['correct']}/{item['support']})\n"
    report += f"""

## Post-hoc known-story diagnostic

When the true story is supplied as inference-time metadata and a separate nested MERT classifier is fitted within each story, accuracy is **{story_mert_metrics['recording_accuracy']:.2%}** and macro-F1 is **{story_mert_metrics['macro_f1']:.2%}**. This crosses 40%, but it is a post-hoc hybrid metadata+audio result, not the preregistered audio-only result.

The known-story majority diagnostic is not an acoustic result and is reported only to quantify how much story metadata can simplify this benchmark.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
