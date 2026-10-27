#!/usr/bin/env python3
"""Rank locked benchmark classes for external recording collection."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    independence = {
        row["padam_canonical_id"]: row
        for row in read(OUT / "kathakalipadam_recording_independence_audit.csv")
    }
    output = []
    for metric in read(OUT / "kathakalipadam_mert_per_class_metrics.csv"):
        item = independence[metric["padam_canonical_id"]]
        recall = float(metric["recall"])
        gap12 = int(item["gap_to_12"])
        priority = (1.0 - recall) * (1.0 + gap12 / 12.0)
        output.append({
            "priority_rank": 0,
            "priority_score": priority,
            "padam_canonical_id": metric["padam_canonical_id"],
            "story_label": item["story_label"],
            "validated_recordings": int(item["validated_recordings"]),
            "unique_main_musicians": int(item["unique_main_musicians"]),
            "locked_out_of_fold_recall": recall,
            "locked_correct": int(metric["correct"]),
            "needed_to_12": gap12,
            "needed_to_15": int(item["gap_to_15"]),
            "same_story_kathakalipadam_archive_exhausted": 1,
            "recommended_source": "External archives or newly digitized performances; require distinct provenance/artist metadata and acoustic near-duplicate inspection.",
        })
    output.sort(key=lambda row: (-row["priority_score"], -row["needed_to_12"], row["padam_canonical_id"]))
    for index, row in enumerate(output, 1):
        row["priority_rank"] = index
    path = OUT / "kathakalipadam_mert_targeted_collection_priority.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"{path}: {len(output)} classes; gap_to_12={sum(row['needed_to_12'] for row in output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
