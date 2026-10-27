#!/usr/bin/env python3
"""Validate the structure and published invariants of the KPAC v1 release."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
EXPECTED_ASSIGNMENT_SHA256 = (
    "a558f3324b7051726b89f6678bce32df2a1cd549f0cc40a0206826956685bb50"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    for required in ("LICENSE", "DATA_LICENSE.md", "DATA_ACCESS.md"):
        require((ROOT / required).is_file(), f"missing required file: {required}")

    software_license = (ROOT / "LICENSE").read_text(encoding="utf-8")
    data_license = (ROOT / "DATA_LICENSE.md").read_text(encoding="utf-8")
    require(software_license.startswith("MIT License"), "software license is not MIT")
    require("Creative Commons Attribution 4.0 International" in data_license, "data license is not CC BY 4.0")
    require("Source audio is not included" in data_license, "data license lacks the audio exclusion")

    manifest = read_rows(OUTPUTS / "kathakalipadam_repeat_rich_expanded.csv")
    validated = read_rows(OUTPUTS / "kathakalipadam_repeat_rich_validated_audio.csv")
    locked = read_rows(OUTPUTS / "kathakalipadam_mert_locked_benchmark.csv")
    acoustic = read_rows(OUTPUTS / "kpac_acoustic_near_duplicate_screen.csv")

    require(len(manifest) == 615, f"manifest rows: {len(manifest)} != 615")
    require(len(validated) == 400, f"validated rows: {len(validated)} != 400")
    require(len(locked) == 112, f"locked rows: {len(locked)} != 112")
    require(len(acoustic) == 496, f"acoustic pairs: {len(acoustic)} != 496")

    for name, rows in (("manifest", manifest), ("validated", validated), ("locked", locked)):
        ids = [row["record_id"] for row in rows]
        require(len(ids) == len(set(ids)), f"duplicate record_id in {name}")

    digests = [row["audio_sha256"] for row in validated]
    require(all(len(value) == 64 for value in digests), "invalid SHA-256 length")
    require(len(digests) == len(set(digests)), "validated audio digests are not unique")

    validated_ids = {row["record_id"] for row in validated}
    require({row["record_id"] for row in locked} <= validated_ids, "lock not subset of validated")
    require(len({row["padam_canonical_id"] for row in validated}) == 88, "validated class count != 88")
    require(len({row["padam_canonical_id"] for row in locked}) == 12, "locked class count != 12")
    require(len({row["story_label"] for row in locked}) == 4, "locked story count != 4")

    folds = Counter(row["locked_outer_fold"] for row in locked)
    require(folds == Counter({"1": 41, "2": 39, "3": 32}), f"fold sizes changed: {folds}")
    support = Counter(row["padam_canonical_id"] for row in locked)
    require(min(support.values()) >= 8 and max(support.values()) == 16, "class support changed")

    lock_meta = json.loads((OUTPUTS / "kathakalipadam_mert_locked_benchmark.json").read_text())
    require(lock_meta["assignment_sha256"] == EXPECTED_ASSIGNMENT_SHA256, "assignment digest changed")

    primary = json.loads((OUTPUTS / "kathakalipadam_mert_classification_results.json").read_text())
    accuracy = primary["primary_metrics"]["recording_accuracy"]
    require(abs(float(accuracy) - 0.29464285714285715) < 1e-12, "primary accuracy changed")

    candidates = sum(row["candidate_near_duplicate"].strip().lower() in {"1", "true", "yes"} for row in acoustic)
    require(candidates == 0, f"unexpected acoustic candidates: {candidates}")

    manifest_digest = hashlib.sha256((OUTPUTS / "kathakalipadam_mert_locked_benchmark.csv").read_bytes()).hexdigest()
    print("KPAC release validation passed")
    print(f"manifest={len(manifest)} validated={len(validated)} locked={len(locked)}")
    print(f"classes_validated=88 classes_locked=12 folds={dict(sorted(folds.items()))}")
    print(f"locked_csv_sha256={manifest_digest}")
    print(f"assignment_sha256={EXPECTED_ASSIGNMENT_SHA256}")


if __name__ == "__main__":
    main()
