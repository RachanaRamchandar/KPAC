#!/usr/bin/env python3
"""Freeze the outer evaluation folds for the repeat-rich MERT benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
VALIDATED = OUT / "kathakalipadam_repeat_rich_validated_audio.csv"
LOCKED_CSV = OUT / "kathakalipadam_mert_locked_benchmark.csv"
LOCK_JSON = OUT / "kathakalipadam_mert_locked_benchmark.json"
MIN_RECORDINGS = 8
N_FOLDS = 3
SEED_TEXT = "kathakalipadam-mert-outer-v1-2026-08-09"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_rank(row: dict[str, str]) -> str:
    payload = f"{SEED_TEXT}|{row['padam_canonical_id']}|{row['record_id']}|{row['audio_sha256']}"
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> int:
    rows = read_csv(VALIDATED)
    counts = Counter(row["padam_canonical_id"] for row in rows)
    labels = {label for label, count in counts.items() if count >= MIN_RECORDINGS}
    selected = [row for row in rows if row["padam_canonical_id"] in labels]
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_label[row["padam_canonical_id"]].append(row)
    fold_map = {}
    for label, members in sorted(by_label.items()):
        ordered = sorted(members, key=stable_rank)
        for index, row in enumerate(ordered):
            fold_map[row["record_id"]] = index % N_FOLDS + 1
    output = []
    for row in sorted(selected, key=lambda item: item["record_id"]):
        output.append({
            "record_id": row["record_id"],
            "padam_canonical_id": row["padam_canonical_id"],
            "padam_canonical_label": row["padam_canonical_label"],
            "story_label": row["story_label"],
            "main_musician": row["main_musician"],
            "supporting_musician": row["supporting_musician"],
            "ragam_label": row["ragam_label"],
            "thalam_label": row["thalam_label"],
            "audio_url": row["audio_url"],
            "audio_sha256": row["audio_sha256"],
            "audio_duration_seconds": row["audio_duration_seconds"],
            "dataset_iteration": row["dataset_iteration"],
            "locked_outer_fold": fold_map[row["record_id"]],
            "fold_assignment_seed": SEED_TEXT,
        })
    with LOCKED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    digest_payload = "\n".join(
        f"{row['record_id']},{row['padam_canonical_id']},{row['audio_sha256']},{row['locked_outer_fold']}"
        for row in output
    )
    digest = hashlib.sha256(digest_payload.encode()).hexdigest()
    payload = {
        "status": "locked_before_MERT_feature_extraction",
        "minimum_validated_recordings_per_class": MIN_RECORDINGS,
        "recordings": len(output),
        "classes": len(labels),
        "folds": N_FOLDS,
        "fold_sizes": dict(sorted(Counter(row["locked_outer_fold"] for row in output).items())),
        "class_sizes": dict(sorted(Counter(row["padam_canonical_id"] for row in output).items())),
        "assignment_seed": SEED_TEXT,
        "assignment_sha256": digest,
        "protocol": "Nested evaluation: classifier and MERT layer/pooling choices are selected only inside each outer training fold; outer test predictions are emitted once.",
        "created_before_embedding_extraction": True,
    }
    LOCK_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
