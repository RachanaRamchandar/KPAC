#!/usr/bin/env python3
"""Manifest-wide padam-classification test using every eligible exact label.

All 189 individual-padam rows are audited. A label is eligible only when the
manifest contains at least three exact `padam_title_key` rows and at least three
audio files remain usable after download/decoder validation. The evaluation
uses recording-grouped folds and never splits chunks from one recording across
train and test.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
import soundfile as sf
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE = SCRIPT_PATH.parent.parent
OUTPUT_DIR = WORKSPACE / "outputs"
WORK_DIR = WORKSPACE / "work" / "padam_smoke_test"
MANIFEST = OUTPUT_DIR / "kathakalipadam_combined_stories_190.csv"
BASE_SCRIPT = OUTPUT_DIR / "run_padam_classification_smoke_test.py"
RESULTS_JSON = OUTPUT_DIR / "full_dataset_padam_classification_results.json"
PREDICTIONS_CSV = OUTPUT_DIR / "full_dataset_padam_classification_predictions.csv"
ELIGIBILITY_CSV = OUTPUT_DIR / "full_dataset_padam_label_eligibility.csv"
REPORT_MD = OUTPUT_DIR / "full_dataset_padam_classification_report.md"

MIN_RECORDINGS = 3
N_FOLDS = 3
CHUNKS_PER_RECORDING = 6
RANDOM_SEED = 42


def load_base_module():
    spec = importlib.util.spec_from_file_location("padam_smoke_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base smoke-test module: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FEATURE_CACHE = WORK_DIR / "full_dataset_features.npz"
    module.CHUNKS_PER_RECORDING = CHUNKS_PER_RECORDING
    module.N_FOLDS = N_FOLDS
    module.RANDOM_SEED = RANDOM_SEED
    return module


base = load_base_module()


def log(message: str) -> None:
    print(message, flush=True)


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_keys: tuple[str, ...],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, int]]:
    base.CLASS_KEYS = class_keys
    fold_map = base.grouped_fold_map(groups, y)
    models = {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        ),
        "linear_svm": lambda: make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=0.1, class_weight="balanced", random_state=RANDOM_SEED),
        ),
        "rbf_svm": lambda: make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=RANDOM_SEED),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    results: dict[str, object] = {}
    prediction_rows: list[dict[str, object]] = []
    for model_name, factory in models.items():
        log(f"Evaluating {model_name}")
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
            scores = base.score_matrix(model, X[test_mask])
            test_groups = groups[test_mask]
            test_true = y[test_mask]

            fold_record_true: list[str] = []
            fold_record_pred: list[str] = []
            for group in sorted(set(test_groups)):
                positions = np.flatnonzero(test_groups == group)
                true_label = str(test_true[positions[0]])
                predicted_label = str(classes[int(np.argmax(scores[positions].mean(axis=0)))])
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
                    "chunk_metrics": base.metric_block(list(map(str, test_true)), list(map(str, chunk_pred))),
                    "recording_metrics": base.metric_block(fold_record_true, fold_record_pred),
                }
            )

        results[model_name] = {
            "chunk_metrics": base.metric_block(all_chunk_true, all_chunk_pred),
            "recording_metrics": base.metric_block(all_record_true, all_record_pred),
            "folds": fold_results,
        }
    return results, prediction_rows, fold_map


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(payload: dict[str, object]) -> None:
    model_order = ["logistic_regression", "linear_svm", "rbf_svm", "random_forest", "extra_trees"]
    lines = [
        "# Full-manifest padam classification test",
        "",
        "## Decision",
        "",
    ]
    best_name = payload["best_model"]
    best = payload["models"][best_name]["recording_metrics"]
    best_display = {
        "linear_svm": "Linear SVM",
        "rbf_svm": "RBF SVM",
    }.get(best_name, best_name.replace("_", " ").title())
    majority = payload["majority_reference_accuracy"]
    if best["accuracy"] <= majority:
        lines.append(
            "Accuracy did not exceed the majority-class reference. More data is needed, but new stories should be chosen to add independent recordings of the same canonical padams—not merely more singleton titles."
        )
    else:
        lines.append(
            "The strongest fixed baseline exceeded the majority-class reference, but the eligible sample remains too small and sparse for a reliable research claim. Targeted additional recordings are still recommended."
        )
    lines.extend(
        [
            "",
            "## Dataset coverage",
            "",
            f"- Manifest rows: {payload['manifest_rows']} total, including {payload['individual_padam_rows']} individual padam rows.",
            f"- Exact normalized labels: {payload['unique_exact_labels']}.",
            f"- Labels with at least {MIN_RECORDINGS} manifest recordings: {payload['candidate_classes']} ({payload['candidate_recordings']} recordings).",
            f"- Final tested set after URL/decoder validation: {payload['tested_classes']} classes and {payload['usable_recordings']} recordings.",
            f"- Excluded for fewer than {MIN_RECORDINGS} manifest examples: {payload['labels_below_minimum']} labels covering {payload['rows_below_minimum']} recordings.",
            f"- Candidate classes lost after URL/decoder validation: {', '.join(payload['classes_lost_after_audio_validation']) or 'none'}.",
            f"- Audio chunks: {payload['chunk_count']} fixed {payload['configuration']['chunk_seconds']:.0f}-second chunks, grouped by source recording across {N_FOLDS} folds.",
            "",
            "## Results",
            "",
            "| Model | Chunk accuracy | Chunk macro-F1 | Recording accuracy | Recording macro-F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in model_order:
        block = payload["models"][name]
        chunk = block["chunk_metrics"]
        record = block["recording_metrics"]
        display_name = {
            "linear_svm": "Linear SVM",
            "rbf_svm": "RBF SVM",
        }.get(name, name.replace("_", " ").title())
        lines.append(
            f"| {display_name} | {chunk['accuracy']:.3f} | {chunk['macro_f1']:.3f} | "
            f"{record['accuracy']:.3f} | {record['macro_f1']:.3f} |"
        )
    lines.extend(
        [
            f"| Majority-class reference | {majority:.3f} | — | {majority:.3f} | — |",
            f"| Uniform {payload['tested_classes']}-class chance reference | {payload['uniform_chance_accuracy']:.3f} | — | {payload['uniform_chance_accuracy']:.3f} | — |",
            "",
            f"Best recording-level result: **{best_display}**, accuracy {best['accuracy']:.3f}, balanced accuracy {best['balanced_accuracy']:.3f}, macro-F1 {best['macro_f1']:.3f}.",
            "",
            "## Audio validation",
            "",
            "- Records 103, 121, 135, and 138 returned HTTP 404.",
            "- Record 136 is an ISO/3GPP container stored with an `.mp3` suffix and was not decodable by the test environment.",
            "- Records 057, 061, and 180 emitted MP3 frame-decoding warnings; records 119, 137, 153, and 154 emitted malformed-ID3 warnings. Features remained finite, but these files should receive integrity inspection.",
            "",
            "## Interpretation",
            "",
            "- This is the largest defensible exact-label supervised subset of the current manifest, not all 189 padam rows: singletons and doubletons cannot support recording-held-out classification.",
            "- Raw accuracy is not directly comparable with the earlier three-class test because this run has more classes and a much lower chance level.",
            "- Several exact keys appear to be orthographic variants of the same padam (for example `ghora_vipinam`/`ghoravipinam` and `mariman_kanni`/`marimaan_kanni`). Treating them as different classes can depress accuracy and should be resolved by an expert label map.",
            "- Chunks remain correlated within recordings; grouped folds prevent leakage, but the effective sample size is the number of recordings.",
            "- Performer, ragam, venue, recording device, and archive source may remain confounded with padam identity.",
            "",
            "## Recommendation for expansion",
            "",
            "1. Curate canonical padam IDs before scraping further stories, merging confirmed spelling/transliteration variants while preserving raw titles.",
            "2. Add stories selectively where the same canonical padams have multiple independent performers or performances; aim for at least 8–10 usable recordings per class.",
            "3. Repair or replace dead audio references and run integrity checks before feature extraction.",
            "4. After label curation and targeted expansion, evaluate pretrained audio embeddings alongside these handcrafted features.",
            "",
            "The eligibility CSV records every exact label, its manifest count, usable count, failed IDs, and inclusion decision.",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = manifest_rows()
    padam_rows = [row for row in rows if row["recording_scope"] == "padam"]
    exact_counts = Counter(row["padam_title_key"] for row in padam_rows)
    candidate_keys = {label for label, count in exact_counts.items() if count >= MIN_RECORDINGS}
    candidates = [row for row in padam_rows if row["padam_title_key"] in candidate_keys]
    log(f"Candidate set: {len(candidates)} recordings across {len(candidate_keys)} exact labels")

    usable_pre_filter: list[tuple[dict[str, str], Path]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(candidates, start=1):
        log(f"Download/validate {index:02d}/{len(candidates):02d}: record {row['record_id']} ({row['padam_title_key']})")
        path, error = base.download_audio(row)
        if path is None:
            failures.append(
                {
                    "record_id": row["record_id"],
                    "label": row["padam_title_key"],
                    "url": row["audio_url"],
                    "error": error or "unknown error",
                }
            )
            continue
        try:
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate <= 0:
                raise RuntimeError("decoder reported no audio frames")
            usable_pre_filter.append((row, path))
        except Exception as exc:
            header = path.read_bytes()[:32] if path.exists() else b""
            if b"ftyp3gp" in header:
                detail = "actual file is an ISO/3GPP container stored with an .mp3 suffix"
            else:
                detail = f"{type(exc).__name__}: {exc}"
            failures.append(
                {
                    "record_id": row["record_id"],
                    "label": row["padam_title_key"],
                    "url": row["audio_url"],
                    "error": f"decode error: {detail}",
                }
            )

    usable_counts = Counter(row["padam_title_key"] for row, _ in usable_pre_filter)
    tested_keys = tuple(sorted(label for label, count in usable_counts.items() if count >= MIN_RECORDINGS))
    usable = [(row, path) for row, path in usable_pre_filter if row["padam_title_key"] in tested_keys]
    classes_lost = sorted(candidate_keys - set(tested_keys))
    base.CLASS_KEYS = tested_keys

    eligibility_rows: list[dict[str, object]] = []
    failure_ids: dict[str, list[str]] = defaultdict(list)
    for failure in failures:
        failure_ids[failure["label"]].append(failure["record_id"])
    for label in sorted(exact_counts):
        manifest_count = exact_counts[label]
        usable_count = usable_counts.get(label, 0) if manifest_count >= MIN_RECORDINGS else 0
        if label in tested_keys:
            status = "eligible_and_tested"
        elif manifest_count >= MIN_RECORDINGS:
            status = "excluded_after_url_or_decode_failures"
        else:
            status = f"excluded_fewer_than_{MIN_RECORDINGS}_recordings"
        eligibility_rows.append(
            {
                "padam_title_key": label,
                "manifest_recordings": manifest_count,
                "usable_recordings": usable_count,
                "failed_record_ids": ";".join(sorted(failure_ids.get(label, []))),
                "status": status,
            }
        )
    write_csv(
        ELIGIBILITY_CSV,
        eligibility_rows,
        ["padam_title_key", "manifest_recordings", "usable_recordings", "failed_record_ids", "status"],
    )

    log(f"Final tested set: {len(usable)} recordings across {len(tested_keys)} classes")
    X, y, groups, starts, feature_names, recording_metadata = base.build_features(usable)
    log(f"Feature matrix: {X.shape[0]} chunks × {X.shape[1]} features")
    model_results, predictions, fold_map = evaluate_models(X, y, groups, tested_keys)
    write_csv(
        PREDICTIONS_CSV,
        predictions,
        [
            "model",
            "fold",
            "record_id",
            "true_label",
            "predicted_label",
            "chunk_count",
            "correct_chunks",
            "chunk_accuracy",
            "recording_correct",
        ],
    )

    final_counts = Counter(row["padam_title_key"] for row, _ in usable)
    majority_reference = max(final_counts.values()) / sum(final_counts.values())
    best_model = max(
        model_results,
        key=lambda name: (
            model_results[name]["recording_metrics"]["accuracy"],
            model_results[name]["recording_metrics"]["macro_f1"],
        ),
    )
    labels_below = {label for label, count in exact_counts.items() if count < MIN_RECORDINGS}
    payload: dict[str, object] = {
        "status": "completed",
        "purpose": "manifest-wide exact-label feasibility test, not a publishable performance estimate",
        "manifest": str(MANIFEST),
        "manifest_rows": len(rows),
        "individual_padam_rows": len(padam_rows),
        "unique_exact_labels": len(exact_counts),
        "candidate_classes": len(candidate_keys),
        "candidate_recordings": len(candidates),
        "tested_classes": len(tested_keys),
        "usable_recordings": len(usable),
        "labels_below_minimum": len(labels_below),
        "rows_below_minimum": sum(exact_counts[label] for label in labels_below),
        "classes_lost_after_audio_validation": classes_lost,
        "class_recording_counts": dict(final_counts),
        "chunk_count": int(X.shape[0]),
        "majority_reference_accuracy": majority_reference,
        "uniform_chance_accuracy": 1.0 / len(tested_keys),
        "best_model": best_model,
        "configuration": {
            "random_seed": RANDOM_SEED,
            "minimum_recordings_per_exact_label": MIN_RECORDINGS,
            "class_keys": list(tested_keys),
            "target_sample_rate": base.TARGET_SAMPLE_RATE,
            "chunk_seconds": base.CHUNK_SECONDS,
            "chunks_per_recording": CHUNKS_PER_RECORDING,
            "grouped_folds": N_FOLDS,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "model_parameters_fixed_before_outer_evaluation": True,
        },
        "download_or_decode_failures": failures,
        "data_quality_warnings": [
            {
                "record_ids": ["057", "061", "180"],
                "warning": "MP3 frame-decoding warnings occurred during feature extraction; finite features were returned, but audio integrity should be inspected.",
            },
            {
                "record_ids": ["119", "137", "153", "154"],
                "warning": "Malformed ID3 metadata warnings occurred; audio features were still extracted.",
            },
        ],
        "recordings": recording_metadata,
        "fold_assignment_by_recording": fold_map,
        "models": model_results,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "soundfile": sf.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    RESULTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(payload)
    log("Full-manifest test completed")
    for name, block in model_results.items():
        log(
            f"{name}: chunk accuracy={block['chunk_metrics']['accuracy']:.3f}, "
            f"recording accuracy={block['recording_metrics']['accuracy']:.3f}, "
            f"recording macro-F1={block['recording_metrics']['macro_f1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
