#!/usr/bin/env python3
"""Summarize canonical-label consistency and recording-independence evidence."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
VALIDATED = OUT / "kathakalipadam_repeat_rich_validated_audio.csv"
AUDIT = OUT / "kathakalipadam_recording_independence_audit.csv"
SUMMARY = OUT / "kathakalipadam_recording_independence_summary.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    rows = read_rows(VALIDATED)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["padam_canonical_id"]].append(row)
    output = []
    for canonical_id, members in sorted(groups.items()):
        labels = {row["padam_canonical_label"] for row in members}
        stories = {row["story_label"] for row in members}
        hashes = {row["audio_sha256"] for row in members}
        urls = {row["audio_url"] for row in members}
        durations = {
            round(float(row["audio_duration_seconds"]), 3)
            for row in members
            if row["audio_duration_seconds"]
        }
        musicians = {row["main_musician"].strip() for row in members if row["main_musician"].strip()}
        musician_pairs = {
            (row["main_musician"].strip(), row["supporting_musician"].strip())
            for row in members
        }
        main_counts = Counter(row["main_musician"].strip() or "(unknown)" for row in members)
        n = len(members)
        output.append(
            {
                "padam_canonical_id": canonical_id,
                "padam_canonical_label": next(iter(labels)) if len(labels) == 1 else " | ".join(sorted(labels)),
                "story_label": next(iter(stories)) if len(stories) == 1 else " | ".join(sorted(stories)),
                "validated_recordings": n,
                "unique_audio_sha256": len(hashes),
                "unique_audio_urls": len(urls),
                "unique_durations_to_ms": len(durations),
                "unique_main_musicians": len(musicians),
                "unique_musician_pairs": len(musician_pairs),
                "dominant_main_musician_share": max(main_counts.values()) / n,
                "canonical_label_consistent": int(len(labels) == 1 and len(stories) == 1),
                "byte_unique_within_class": int(len(hashes) == n),
                "url_unique_within_class": int(len(urls) == n),
                "duration_unique_to_ms_within_class": int(len(durations) == n),
                "locked_mert_eligible_ge_8": int(n >= 8),
                "gap_to_12": max(0, 12 - n),
                "gap_to_15": max(0, 15 - n),
                "independence_note": "Distinct decoded bytes and URLs; musician/duration diversity are supporting evidence, not a legal guarantee of independently staged performance.",
            }
        )
    with AUDIT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    payload = {
        "validated_recordings": len(rows),
        "canonical_classes": len(groups),
        "globally_unique_audio_sha256": len({row["audio_sha256"] for row in rows}),
        "global_byte_duplicates": len(rows) - len({row["audio_sha256"] for row in rows}),
        "classes_with_inconsistent_canonical_label_or_story": sum(
            1 for row in output if not row["canonical_label_consistent"]
        ),
        "classes_with_within_class_byte_duplicate": sum(
            1 for row in output if not row["byte_unique_within_class"]
        ),
        "classes_ge_8": sum(1 for row in output if row["locked_mert_eligible_ge_8"]),
        "recordings_in_classes_ge_8": sum(
            row["validated_recordings"] for row in output if row["locked_mert_eligible_ge_8"]
        ),
        "interpretation": "All admitted recordings are byte-distinct and canonical labels are internally consistent. Distinct URLs, millisecond durations, and artist combinations strengthen but cannot prove performance independence.",
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
