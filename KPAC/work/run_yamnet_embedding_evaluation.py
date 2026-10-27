#!/usr/bin/env python3
"""Evaluate YAMNet and fused features on the exact prior Kathakali splits."""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
WORK = ROOT / "work"
OLD_PREDICTIONS = OUT / "canonical_full_dataset_classification_predictions.csv"
OLD_RESULTS = OUT / "canonical_full_dataset_classification_results.json"
MANIFEST = OUT / "kathakalipadam_canonicalized_expanded_425.csv"
ELIGIBILITY = OUT / "canonical_full_dataset_label_eligibility.csv"
HANDCRAFTED_DIR = WORK / "canonical_full_evaluation" / "record_features"
EMBEDDING_DIR = WORK / "pretrained_embedding_eval" / "yamnet_record_embeddings"

RESULTS_JSON = OUT / "yamnet_embedding_classification_results.json"
PREDICTIONS_CSV = OUT / "yamnet_embedding_classification_predictions.csv"
ABLATIONS_CSV = OUT / "yamnet_embedding_ablation_results.csv"
PRIORITY_CSV = OUT / "targeted_scraping_priority.csv"
REPORT_MD = OUT / "yamnet_embedding_classification_report.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_rows(experiment: str) -> list[dict[str, str]]:
    rows = [
        row for row in read_csv(OLD_PREDICTIONS)
        if row["experiment"] == experiment and row["model"] == "linear_svm"
    ]
    if len(rows) != len({row["record_id"] for row in rows}):
        raise RuntimeError(f"Non-unique split rows for {experiment}")
    return sorted(rows, key=lambda row: row["record_id"])


def model_factory(name: str):
    if name == "linear_svm":
        return make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.1, class_weight="balanced", random_state=SEED))
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=SEED),
        )
    if name == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=SEED))
    raise KeyError(name)


def score_matrix(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)
    scores = np.asarray(model.decision_function(X), dtype=float)
    return np.column_stack([-scores, scores]) if scores.ndim == 1 else scores


def wilson(correct: int, total: int) -> tuple[float, float]:
    z = 1.96
    p = correct / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def record_features(record_id: str) -> dict[str, np.ndarray]:
    pretrained = np.load(EMBEDDING_DIR / f"{record_id}.npz")
    embedding = np.asarray(pretrained["embedding_mean"], dtype=np.float32)
    frame_std = np.asarray(pretrained["embedding_frame_std"], dtype=np.float32)
    scores = np.asarray(pretrained["score_mean"], dtype=np.float32)
    handcrafted = np.asarray(np.load(HANDCRAFTED_DIR / f"{record_id}.npz")["X"], dtype=np.float32)
    embed_mean = embedding.mean(axis=0)
    embed_mean_std = np.concatenate([embed_mean, embedding.std(axis=0)])
    frame_stats = np.concatenate([embed_mean_std, frame_std.mean(axis=0)])
    score_stats = np.concatenate([scores.mean(axis=0), scores.std(axis=0)])
    hand_stats = np.concatenate([handcrafted.mean(axis=0), handcrafted.std(axis=0)])
    return {
        "yamnet_mean": embed_mean,
        "yamnet_mean_std": embed_mean_std,
        "yamnet_frame_stats": frame_stats,
        "yamnet_scores_mean_std": score_stats,
        "handcrafted_mean_std": hand_stats,
        "yamnet_plus_handcrafted": np.concatenate([embed_mean_std, hand_stats]),
    }


def evaluate(
    experiment: str,
    model_name: str,
    X: np.ndarray,
    rows: list[dict[str, str]],
) -> tuple[dict, list[dict]]:
    y = np.asarray([row["true_label"] for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows])
    labels = sorted(set(y))
    true_all: list[str] = []
    pred_all: list[str] = []
    predictions: list[dict] = []
    top3_correct = 0
    for fold in sorted(set(folds)):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        model = model_factory(model_name)
        model.fit(X[train], y[train])
        scores = score_matrix(model, X[test])
        classes = list(map(str, model.classes_))
        for local_i, record_i in enumerate(test):
            order = np.argsort(scores[local_i])[::-1]
            pred = classes[int(order[0])]
            true = str(y[record_i])
            top3 = [classes[int(i)] for i in order[:3]]
            true_all.append(true)
            pred_all.append(pred)
            top3_correct += int(true in top3)
            source = rows[int(record_i)]
            predictions.append({
                "experiment": experiment,
                "model": model_name,
                "fold": fold,
                "record_id": source["record_id"],
                "story_label": source["story_label"],
                "main_musician": source["main_musician"],
                "true_label": true,
                "predicted_label": pred,
                "top3_labels": " | ".join(top3),
                "recording_correct": int(pred == true),
                "top3_correct": int(true in top3),
            })
    accuracy = float(accuracy_score(true_all, pred_all))
    lo, hi = wilson(sum(a == b for a, b in zip(true_all, pred_all)), len(true_all))
    result = {
        "n": len(true_all),
        "classes": len(labels),
        "accuracy": accuracy,
        "accuracy_ci95": [lo, hi],
        "balanced_accuracy": float(balanced_accuracy_score(true_all, pred_all)),
        "macro_f1": float(f1_score(true_all, pred_all, labels=labels, average="macro", zero_division=0)),
        "top3_accuracy": top3_correct / len(true_all),
        "uniform_chance_accuracy": 1 / len(labels),
    }
    return result, predictions


def ablation_row(experiment: str, model: str, description: str, result: dict) -> dict:
    return {
        "experiment": experiment,
        "model": model,
        "description": description,
        "classes": result["classes"],
        "recordings": result["n"],
        "accuracy": result["accuracy"],
        "accuracy_ci95_low": result["accuracy_ci95"][0],
        "accuracy_ci95_high": result["accuracy_ci95"][1],
        "balanced_accuracy": result["balanced_accuracy"],
        "macro_f1": result["macro_f1"],
        "top3_accuracy": result["top3_accuracy"],
        "uniform_chance_accuracy": result["uniform_chance_accuracy"],
    }


def paired_comparison(candidate: list[dict], control: list[dict]) -> dict:
    candidate_by_id = {row["record_id"]: int(row["recording_correct"]) for row in candidate}
    control_by_id = {row["record_id"]: int(row["recording_correct"]) for row in control}
    if candidate_by_id.keys() != control_by_id.keys():
        raise RuntimeError("Paired comparison requires identical recording IDs")
    candidate_only = sum(candidate_by_id[rid] == 1 and control_by_id[rid] == 0 for rid in candidate_by_id)
    control_only = sum(candidate_by_id[rid] == 0 and control_by_id[rid] == 1 for rid in candidate_by_id)
    discordant = candidate_only + control_only
    p_value = float(binomtest(min(candidate_only, control_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "candidate_only_correct": candidate_only,
        "control_only_correct": control_only,
        "discordant_recordings": discordant,
        "exact_mcnemar_p_value": p_value,
    }


def build_priority_rows(
    exact_rows: list[dict[str, str]],
    best_predictions: list[dict],
    old_best_predictions: list[dict[str, str]],
) -> list[dict]:
    manifest = read_csv(MANIFEST)
    label_lookup = {}
    for row in manifest:
        label_lookup.setdefault(row["padam_canonical_id"], row["padam_canonical_label"])
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in exact_rows:
        by_class[row["true_label"]].append(row)
    new_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in best_predictions:
        new_by_class[row["true_label"]].append(row)
    old_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in old_best_predictions:
        old_by_class[row["true_label"]].append(row)

    output = []
    for canonical_id, members in by_class.items():
        new_rows = new_by_class[canonical_id]
        old_rows = old_by_class[canonical_id]
        new_recall = sum(int(row["recording_correct"]) for row in new_rows) / len(new_rows)
        old_recall = sum(int(row["recording_correct"]) for row in old_rows) / len(old_rows)
        n = len(members)
        musicians = sorted({row["main_musician"] for row in members if row["main_musician"]})
        target_total = max(8, n)
        target_additional = max(0, target_total - n)
        if new_recall == 0 and old_recall == 0:
            tier, weakness = "1-critical", "zero recall with both pretrained and handcrafted features"
        elif n <= 4 or len(musicians) < 3:
            tier, weakness = "1-critical", "few recordings or fewer than three independent main musicians"
        elif new_recall < 0.20:
            tier, weakness = "2-high", "pretrained-embedding recall below 20%"
        elif n < 8:
            tier, weakness = "2-high", "fewer than eight validated independent recordings"
        else:
            tier, weakness = "3-medium", "class remains weaker than the preferred collection depth"
        priority_score = (2.0 - new_recall - old_recall) + max(0, 8 - n) / 8 + max(0, 3 - len(musicians)) / 3
        output.append({
            "priority_tier": tier,
            "priority_score": round(priority_score, 6),
            "padam_canonical_id": canonical_id,
            "padam_canonical_label": label_lookup.get(canonical_id, canonical_id),
            "story_label": members[0]["story_label"],
            "validated_unique_recordings": n,
            "distinct_main_musicians": len(musicians),
            "main_musicians": " | ".join(musicians),
            "handcrafted_recall": round(old_recall, 6),
            "best_transfer_model_recall": round(new_recall, 6),
            "target_additional_recordings_to_8": target_additional,
            "collection_priority_reason": weakness,
            "collection_rule": "Add a byte-distinct recording, preferably by a main musician not already represented in this class.",
        })
    return sorted(output, key=lambda row: (row["priority_tier"], -row["priority_score"], row["padam_canonical_id"]))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> int:
    exact_rows = split_rows("record_mean_std")
    if len(exact_rows) != 233 or len({row["true_label"] for row in exact_rows}) != 47:
        raise RuntimeError("The preserved canonical split is not the expected 233 recordings / 47 classes")
    feature_by_id = {row["record_id"]: record_features(row["record_id"]) for row in exact_rows}
    feature_names = list(next(iter(feature_by_id.values())))
    matrices = {
        name: np.vstack([feature_by_id[row["record_id"]][name] for row in exact_rows])
        for name in feature_names
    }

    descriptions = {
        "yamnet_mean": "Mean of six chunk-level YAMNet embeddings (1,024 dimensions).",
        "yamnet_mean_std": "Mean and standard deviation of six chunk-level YAMNet embeddings (2,048 dimensions).",
        "yamnet_frame_stats": "YAMNet chunk mean/std plus mean within-chunk frame variability (3,072 dimensions).",
        "yamnet_scores_mean_std": "Mean and standard deviation of YAMNet's 521 AudioSet class scores (1,042 dimensions).",
        "handcrafted_mean_std": "Recomputed control: prior handcrafted recording mean/std vector (232 dimensions).",
        "yamnet_plus_handcrafted": "Concatenated YAMNet mean/std and handcrafted mean/std features (2,280 dimensions).",
    }
    all_results: dict[str, dict] = {}
    predictions: list[dict] = []
    ablations: list[dict] = []
    model_grid = {
        "yamnet_mean": ["linear_svm"],
        "yamnet_mean_std": ["linear_svm", "logistic_regression", "rbf_svm"],
        "yamnet_frame_stats": ["linear_svm"],
        "yamnet_scores_mean_std": ["linear_svm"],
        "handcrafted_mean_std": ["linear_svm"],
        "yamnet_plus_handcrafted": ["linear_svm", "logistic_regression", "rbf_svm"],
    }
    for experiment, models in model_grid.items():
        for model_name in models:
            result, pred = evaluate(experiment, model_name, matrices[experiment], exact_rows)
            key = f"{experiment}|{model_name}"
            all_results[key] = result
            predictions.extend(pred)
            ablations.append(ablation_row(experiment, model_name, descriptions[experiment], result))
            print(f"{key}: accuracy={pct(result['accuracy'])}, macro-F1={pct(result['macro_f1'])}", flush=True)

    comparable = [row for row in ablations if row["experiment"] != "handcrafted_mean_std"]
    best_row = max(comparable, key=lambda row: (row["accuracy"], row["macro_f1"], row["top3_accuracy"]))
    best_key = f"{best_row['experiment']}|{best_row['model']}"
    best_predictions = [
        row for row in predictions
        if row["experiment"] == best_row["experiment"] and row["model"] == best_row["model"]
    ]

    control_predictions = [
        row for row in predictions
        if row["experiment"] == "handcrafted_mean_std" and row["model"] == "linear_svm"
    ]
    paired = paired_comparison(best_predictions, control_predictions)

    embedding_only_rows = [
        row for row in ablations
        if row["experiment"].startswith("yamnet_") and row["experiment"] != "yamnet_plus_handcrafted"
    ]
    best_embedding_only_row = max(
        embedding_only_rows,
        key=lambda row: (row["accuracy"], row["macro_f1"], row["top3_accuracy"]),
    )
    best_embedding_only_key = f"{best_embedding_only_row['experiment']}|{best_embedding_only_row['model']}"

    artist_rows = split_rows("main_artist_disjoint")
    artist_specs = [
        (best_row["experiment"], best_row["model"]),
        (best_embedding_only_row["experiment"], best_embedding_only_row["model"]),
        ("handcrafted_mean_std", "linear_svm"),
    ]
    artist_results = {}
    seen_specs = set()
    for experiment, model_name in artist_specs:
        if (experiment, model_name) in seen_specs:
            continue
        seen_specs.add((experiment, model_name))
        artist_matrix = np.vstack([feature_by_id[row["record_id"]][experiment] for row in artist_rows])
        exp_name = f"artist_disjoint__{experiment}"
        result, pred = evaluate(exp_name, model_name, artist_matrix, artist_rows)
        artist_results[f"{experiment}|{model_name}"] = result
        predictions.extend(pred)
        ablations.append(ablation_row(exp_name, model_name, f"Exact prior main-artist-disjoint folds using {descriptions[experiment]}", result))
        print(f"{exp_name}|{model_name}: accuracy={pct(result['accuracy'])}", flush=True)

    old_predictions = read_csv(OLD_PREDICTIONS)
    old_best_predictions = [
        row for row in old_predictions
        if row["experiment"] == "record_mean_std" and row["model"] == "linear_svm"
    ]
    priority_rows = build_priority_rows(exact_rows, best_predictions, old_best_predictions)
    artist_best_predictions = [
        row for row in predictions
        if row["experiment"] == f"artist_disjoint__{best_row['experiment']}" and row["model"] == best_row["model"]
    ]
    artist_control_predictions = [
        row for row in predictions
        if row["experiment"] == "artist_disjoint__handcrafted_mean_std" and row["model"] == "linear_svm"
    ]
    artist_paired = paired_comparison(artist_best_predictions, artist_control_predictions)

    write_csv(PREDICTIONS_CSV, predictions)
    write_csv(ABLATIONS_CSV, ablations)
    write_csv(PRIORITY_CSV, priority_rows)
    with OLD_RESULTS.open(encoding="utf-8") as handle:
        old_results = json.load(handle)
    payload = {
        "status": "completed",
        "purpose": "pretrained-audio transfer baseline on exact preserved splits",
        "pretrained_model": {
            "name": "YAMNet",
            "source": "https://tfhub.dev/google/yamnet/1",
            "training_taxonomy": "AudioSet (521 classes)",
            "embedding_dimensions": 1024,
            "input": "16 kHz mono waveform",
        },
        "exact_recording_split": {"recordings": 233, "classes": 47, "folds": 3},
        "exact_artist_disjoint_split": {"recordings": 198, "classes": 38, "folds": 3},
        "best_pretrained_experiment": best_key,
        "best_pretrained_result": all_results[best_key],
        "best_embedding_only_experiment": best_embedding_only_key,
        "best_embedding_only_result": all_results[best_embedding_only_key],
        "paired_comparison_best_vs_handcrafted": paired,
        "all_exact_split_results": all_results,
        "artist_disjoint_results": artist_results,
        "paired_comparison_artist_best_vs_handcrafted": artist_paired,
        "prior_best_reported": {
            "experiment": old_results.get("best_canonical_full_experiment"),
            "model": old_results.get("best_canonical_full_model"),
            "result": old_results.get("best_canonical_full_metrics"),
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "random_seed": SEED,
        "outputs": {
            "predictions_csv": str(PREDICTIONS_CSV),
            "ablations_csv": str(ABLATIONS_CSV),
            "targeted_scraping_priority_csv": str(PRIORITY_CSV),
        },
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    old_acc = all_results["handcrafted_mean_std|linear_svm"]["accuracy"]
    best_result = all_results[best_key]
    delta = best_result["accuracy"] - old_acc
    critical = [row for row in priority_rows if row["priority_tier"] == "1-critical"]
    artist_best = artist_results[best_key]
    artist_embedding_only = artist_results[best_embedding_only_key]
    artist_control = artist_results["handcrafted_mean_std|linear_svm"]
    report = f"""# Pretrained-audio embedding experiment

## Result

YAMNet was evaluated on the exact prior three-fold recording split: 233 byte-unique recordings in 47 canonical padam classes. The best pretrained configuration was `{best_key}`.

| Measure | Result |
|---|---:|
| Accuracy | {pct(best_result['accuracy'])} |
| 95% Wilson CI | {pct(best_result['accuracy_ci95'][0])}–{pct(best_result['accuracy_ci95'][1])} |
| Balanced accuracy | {pct(best_result['balanced_accuracy'])} |
| Macro-F1 | {pct(best_result['macro_f1'])} |
| Top-3 accuracy | {pct(best_result['top3_accuracy'])} |
| Uniform chance | {pct(best_result['uniform_chance_accuracy'])} |
| Absolute accuracy change vs matched handcrafted control | {delta * 100:+.2f} percentage points |
| Paired exact McNemar p-value | {paired['exact_mcnemar_p_value']:.4f} |

The matched handcrafted control reproduced {pct(old_acc)} accuracy on the same rows and folds. The fused model gained one net correct recording ({paired['candidate_only_correct']} candidate-only correct versus {paired['control_only_correct']} control-only correct), and the paired test does not establish a reliable improvement. YAMNet alone peaked at {pct(all_results[best_embedding_only_key]['accuracy'])} accuracy with `{best_embedding_only_key}`. This is a transfer-learning feasibility result, not a publishable estimate: the corpus remains small and class-imbalanced, and YAMNet was trained for general AudioSet event recognition rather than Kathakali or Indian classical music.

## Main-artist-disjoint stress test

The exact prior artist-disjoint split contains 198 recordings in 38 classes. The selected fused model achieved {pct(artist_best['accuracy'])} accuracy, embedding-only achieved {pct(artist_embedding_only['accuracy'])}, and the matched handcrafted control achieved {pct(artist_control['accuracy'])}. The fused-versus-control paired exact McNemar p-value is {artist_paired['exact_mcnemar_p_value']:.4f}; this stress-test difference is therefore also descriptive rather than established.

## Ablations

The ablation table compares embedding pooling, AudioSet-score features, three classifiers for the principal embedding, and fusion with handcrafted features. See `{ABLATIONS_CSV.name}` for every metric and `{PREDICTIONS_CSV.name}` for every held-out prediction.

## Data-collection decision

The collection priority file ranks all 47 existing canonical classes using held-out recall, recording count, and main-musician diversity. There are {len(critical)} critical-priority classes. New scraping should first add byte-distinct performances for these existing classes, preferably by main musicians not already represented in the class, until each has at least eight validated recordings and at least three musicians. Singleton padams should not be added for the next experiment cycle.

## Reproduction

1. Extract YAMNet features with `work/extract_yamnet_embeddings.py` in the isolated TensorFlow environment.
2. Run `outputs/run_yamnet_embedding_evaluation.py` in the existing scikit-learn environment.
3. Preserve fold assignments from `canonical_full_dataset_classification_predictions.csv`; do not regenerate splits.
"""
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"Wrote {RESULTS_JSON.name}, {ABLATIONS_CSV.name}, {PREDICTIONS_CSV.name}, {PRIORITY_CSV.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
