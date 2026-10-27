#!/usr/bin/env python3
"""Download and byte-validate audio for the repeat-rich corpus expansion."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
WORK = ROOT / "work"
CORPUS = OUT / "kathakalipadam_repeat_rich_expanded.csv"
OLD_PREDICTIONS = OUT / "canonical_full_dataset_classification_predictions.csv"
OLD_AUDIO = WORK / "padam_smoke_test" / "audio_mp3"
OLD_FEATURES = WORK / "canonical_full_evaluation" / "record_features"
NEW_AUDIO = WORK / "corpus_expansion" / "audio"
VALIDATED_CSV = OUT / "kathakalipadam_repeat_rich_validated_audio.csv"
VALIDATION_CSV = OUT / "kathakalipadam_repeat_rich_audio_validation.csv"
SUMMARY_JSON = OUT / "kathakalipadam_repeat_rich_audio_validation.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def client() -> requests.Session:
    out = requests.Session()
    retry = Retry(total=4, connect=4, read=4, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    out.mount("https://", HTTPAdapter(max_retries=retry))
    out.headers.update({"User-Agent": "Mozilla/5.0 (KathakaliPadam research audio validation)"})
    return out


def destination(row: dict[str, str]) -> Path:
    suffix = Path(row.get("audio_filename", "")).suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
        suffix = ".mp3"
    return NEW_AUDIO / f"{row['record_id']}{suffix}"


def download(http: requests.Session, row: dict[str, str]) -> Path:
    path = destination(row)
    if path.exists() and path.stat().st_size > 4096:
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    with http.get(row["audio_url"], timeout=90, stream=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)
    if temporary.stat().st_size <= 4096:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("downloaded audio is unexpectedly small")
    temporary.replace(path)
    return path


def audio_info(path: Path) -> tuple[float, int, int]:
    with sf.SoundFile(path) as audio:
        duration = len(audio) / audio.samplerate
        if duration < 5:
            raise RuntimeError(f"audio shorter than five seconds ({duration:.2f}s)")
        return float(duration), int(audio.samplerate), int(audio.channels)


def main() -> int:
    NEW_AUDIO.mkdir(parents=True, exist_ok=True)
    rows = read_csv(CORPUS)
    by_id = {row["record_id"]: row for row in rows}
    old_split_rows = [
        row for row in read_csv(OLD_PREDICTIONS)
        if row["experiment"] == "record_mean_std" and row["model"] == "linear_svm"
    ]
    old_ids = sorted({row["record_id"] for row in old_split_rows})
    new_rows = [row for row in rows if row.get("dataset_iteration") == "repeat_rich_story_expansion"]
    validation = []
    valid_rows = []

    for record_id in old_ids:
        row = by_id[record_id]
        candidates = [path for path in OLD_AUDIO.glob(f"{record_id}.*") if not path.name.endswith(".part")]
        if len(candidates) != 1:
            raise RuntimeError(f"Missing unique cached prior audio for {record_id}")
        path = candidates[0]
        feature_cache = np.load(OLD_FEATURES / f"{record_id}.npz", allow_pickle=True)
        audio_sha = str(feature_cache["sha256"].item())
        duration, sample_rate, channels = audio_info(path)
        enriched = dict(row)
        enriched.update({
            "audio_sha256": audio_sha,
            "audio_duration_seconds": duration,
            "audio_sample_rate": sample_rate,
            "audio_channels": channels,
            "audio_validation_status": "validated_unique_prior_corpus",
        })
        valid_rows.append(enriched)
        validation.append({
            "record_id": record_id, "padam_canonical_id": row["padam_canonical_id"],
            "story_label": row["story_label"], "audio_url": row["audio_url"],
            "local_audio_path": str(path), "audio_sha256": audio_sha,
            "duration_seconds": duration, "sample_rate": sample_rate, "channels": channels,
            "status": "validated_unique_prior_corpus", "exclusion_reason": "",
        })

    http = client()
    new_candidates = []
    for index, row in enumerate(new_rows, start=1):
        try:
            path = download(http, row)
            duration, sample_rate, channels = audio_info(path)
            audio_sha = sha256(path)
            item = {
                "row": row, "path": path, "sha": audio_sha, "duration": duration,
                "sample_rate": sample_rate, "channels": channels,
            }
            new_candidates.append(item)
            status, reason = "downloaded_and_decoded", ""
        except Exception as exc:
            path = destination(row)
            audio_sha = ""
            duration = sample_rate = channels = ""
            status, reason = "excluded", f"{type(exc).__name__}: {exc}"
        validation.append({
            "record_id": row["record_id"], "padam_canonical_id": row["padam_canonical_id"],
            "story_label": row["story_label"], "audio_url": row["audio_url"],
            "local_audio_path": str(path), "audio_sha256": audio_sha,
            "duration_seconds": duration, "sample_rate": sample_rate, "channels": channels,
            "status": status, "exclusion_reason": reason,
        })
        if index % 10 == 0 or index == len(new_rows):
            print(f"New audio {index}/{len(new_rows)}; decoded {len(new_candidates)}", flush=True)
        time.sleep(0.03)

    by_hash: dict[str, list[dict]] = defaultdict(list)
    for item in new_candidates:
        by_hash[item["sha"]].append(item)
    validation_by_id = {row["record_id"]: row for row in validation}
    duplicate_summary = {"same_label": {}, "conflicting_label": {}}
    for audio_sha, items in by_hash.items():
        labels = {item["row"]["padam_canonical_id"] for item in items}
        if len(labels) > 1:
            ids = [item["row"]["record_id"] for item in items]
            duplicate_summary["conflicting_label"][audio_sha] = ids
            for item in items:
                audit = validation_by_id[item["row"]["record_id"]]
                audit["status"] = "excluded"
                audit["exclusion_reason"] = "identical audio bytes occur under conflicting canonical labels"
            continue
        ordered = sorted(items, key=lambda item: item["row"]["record_id"])
        if len(ordered) > 1:
            duplicate_summary["same_label"][audio_sha] = [item["row"]["record_id"] for item in ordered]
        keep = ordered[0]
        for item in ordered[1:]:
            audit = validation_by_id[item["row"]["record_id"]]
            audit["status"] = "excluded"
            audit["exclusion_reason"] = f"same-label byte duplicate of record {keep['row']['record_id']}"
        enriched = dict(keep["row"])
        enriched.update({
            "audio_sha256": keep["sha"],
            "audio_duration_seconds": keep["duration"],
            "audio_sample_rate": keep["sample_rate"],
            "audio_channels": keep["channels"],
            "audio_validation_status": "validated_unique_new_recording",
        })
        valid_rows.append(enriched)
        validation_by_id[keep["row"]["record_id"]]["status"] = "validated_unique_new_recording"

    validated_fields = list(rows[0]) + [
        "audio_sha256", "audio_duration_seconds", "audio_sample_rate", "audio_channels", "audio_validation_status",
    ]
    valid_rows.sort(key=lambda row: row["record_id"])
    write_csv(VALIDATED_CSV, valid_rows, validated_fields)
    validation.sort(key=lambda row: row["record_id"])
    write_csv(VALIDATION_CSV, validation)
    class_counts = Counter(row["padam_canonical_id"] for row in valid_rows)
    payload = {
        "prior_validated_recordings": len(old_ids),
        "new_repeat_rich_manifest_recordings": len(new_rows),
        "new_downloaded_and_decoded": len(new_candidates),
        "new_validated_unique_recordings": sum(row["audio_validation_status"] == "validated_unique_new_recording" for row in valid_rows),
        "total_validated_unique_recordings": len(valid_rows),
        "total_canonical_classes": len(class_counts),
        "classes_with_at_least_8_validated_recordings": sum(count >= 8 for count in class_counts.values()),
        "recordings_in_8plus_classes": sum(count for count in class_counts.values() if count >= 8),
        "class_count_distribution": dict(sorted(Counter(class_counts.values()).items())),
        "duplicates": duplicate_summary,
        "output_csv": str(VALIDATED_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
