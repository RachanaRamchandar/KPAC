#!/usr/bin/env python3
"""Screen locked KPAC recordings for possible acoustic near-duplicates.

This is a conservative candidate-generation audit, not proof of performance
independence. It compares time-ordered chroma/spectral shingles only within the
same canonical class, where accidental musical similarity is most plausible.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "outputs" / "kathakalipadam_mert_locked_benchmark.csv"
OUT_CSV = ROOT / "outputs" / "kpac_acoustic_near_duplicate_screen.csv"
OUT_JSON = ROOT / "outputs" / "kpac_acoustic_near_duplicate_summary.json"
CACHE = ROOT / "work" / "additional_baselines" / "acoustic_signatures"
DECODER = ROOT / "work" / "padam_smoke_test" / "decode_audio_objc"
AUDIO_DIRS = (
    ROOT / "work" / "padam_smoke_test" / "audio_mp3",
    ROOT / "work" / "corpus_expansion" / "audio",
)
TARGET_SR = 8_000
N_FFT = 4_096
HOP = 4_000
WINDOW_FRAMES = 4
WINDOW_STRIDE = 1
CANDIDATE_JACCARD = 0.20
CANDIDATE_DURATION_RATIO = 0.85


def read_rows() -> list[dict[str, str]]:
    with LOCKED.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audio_path(record_id: str) -> Path:
    found = [
        path
        for folder in AUDIO_DIRS
        for path in folder.glob(f"{record_id}.*")
        if not path.name.endswith(".part")
    ]
    if len(found) != 1:
        raise RuntimeError(f"Expected one audio file for {record_id}; found {found}")
    return found[0]


def signature(record_id: str) -> tuple[set[str], float]:
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = CACHE / f"v2_{record_id}.npz"
    if destination.exists():
        cached = np.load(destination, allow_pickle=True)
        return set(cached["shingles"].astype(str)), float(cached["duration"].item())
    path = audio_path(record_id)
    try:
        with sf.SoundFile(path) as audio:
            source_sr = int(audio.samplerate)
            duration = len(audio) / source_sr
            samples = audio.read(dtype="float32", always_2d=True).mean(axis=1)
    except Exception:
        blocks: list[np.ndarray] = []
        try:
            with sf.SoundFile(path) as audio:
                source_sr = int(audio.samplerate)
                duration = len(audio) / source_sr
                while True:
                    try:
                        block = audio.read(
                            frames=30 * source_sr,
                            dtype="float32",
                            always_2d=True,
                        )
                    except Exception:
                        break
                    if not len(block):
                        break
                    blocks.append(block.mean(axis=1))
            if sum(len(block) for block in blocks) < 60 * source_sr:
                raise RuntimeError("less than 60 seconds decoded blockwise")
            samples = np.concatenate(blocks)
            duration = len(samples) / source_sr
        except Exception:
            pcm = CACHE / f"{record_id}.pcm"
            run = subprocess.run(
                [str(DECODER), str(path), str(pcm)], capture_output=True, text=True
            )
            if run.returncode != 0 or not pcm.exists():
                raise RuntimeError(
                    f"Fallback decode failed for {record_id}: {run.stderr.strip()}"
                )
            source_sr = 16_000
            raw = np.memmap(pcm, mode="r", dtype="<i2")
            samples = np.asarray(raw, dtype=np.float32) / 32768.0
            duration = len(samples) / source_sr
    if source_sr != TARGET_SR:
        samples = librosa.resample(samples, orig_sr=source_sr, target_sr=TARGET_SR)
    chroma = librosa.feature.chroma_stft(
        y=samples, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP
    )
    centroid = librosa.feature.spectral_centroid(
        y=samples, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP
    )[0]
    rms = librosa.feature.rms(y=samples, frame_length=N_FFT, hop_length=HOP)[0]
    frames = min(chroma.shape[1], len(centroid), len(rms))
    chroma = chroma[:, :frames]
    top = np.argsort(chroma, axis=0)[-2:, :]
    centroid_bin = np.clip(
        np.floor(np.log2(np.maximum(centroid[:frames], 30.0) / 30.0) * 2), 0, 15
    ).astype(int)
    delta = np.diff(np.log1p(rms[:frames]), prepend=np.log1p(rms[0]))
    energy_bin = np.where(delta > 0.02, 2, np.where(delta < -0.02, 0, 1))
    tokens = [
        f"{int(top[1, i]):02d}{centroid_bin[i]:02d}{energy_bin[i]}"
        for i in range(frames)
    ]
    shingles: set[str] = set()
    for start in range(0, max(0, len(tokens) - WINDOW_FRAMES + 1), WINDOW_STRIDE):
        payload = "|".join(tokens[start : start + WINDOW_FRAMES]).encode()
        shingles.add(hashlib.blake2b(payload, digest_size=8).hexdigest())
    np.savez_compressed(
        destination,
        shingles=np.asarray(sorted(shingles)),
        duration=duration,
        target_sample_rate=TARGET_SR,
        window_frames=WINDOW_FRAMES,
        hop_seconds=HOP / TARGET_SR,
    )
    return shingles, duration


def main() -> int:
    rows = read_rows()
    signatures: dict[str, tuple[set[str], float]] = {}
    for index, row in enumerate(rows, 1):
        signatures[row["record_id"]] = signature(row["record_id"])
        print(f"signature {index}/{len(rows)}: {row['record_id']}", flush=True)

    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["padam_canonical_id"]].append(row)
    pairs: list[dict] = []
    for canonical_id, members in sorted(by_class.items()):
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                left_set, left_duration = signatures[left["record_id"]]
                right_set, right_duration = signatures[right["record_id"]]
                union = left_set | right_set
                intersection = left_set & right_set
                jaccard = len(intersection) / len(union) if union else 0.0
                duration_ratio = min(left_duration, right_duration) / max(left_duration, right_duration)
                candidate = int(
                    jaccard >= CANDIDATE_JACCARD
                    and duration_ratio >= CANDIDATE_DURATION_RATIO
                )
                pairs.append({
                    "padam_canonical_id": canonical_id,
                    "left_record_id": left["record_id"],
                    "right_record_id": right["record_id"],
                    "left_main_musician": left["main_musician"],
                    "right_main_musician": right["main_musician"],
                    "left_duration_seconds": left_duration,
                    "right_duration_seconds": right_duration,
                    "duration_ratio": duration_ratio,
                    "acoustic_shingle_jaccard": jaccard,
                    "shared_shingles": len(intersection),
                    "candidate_near_duplicate": candidate,
                    "inspection_status": "manual_provenance_inspection_required" if candidate else "screen_below_threshold",
                })
    pairs.sort(key=lambda row: (-row["candidate_near_duplicate"], -row["acoustic_shingle_jaccard"], row["left_record_id"], row["right_record_id"]))
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    candidates = [row for row in pairs if row["candidate_near_duplicate"]]
    payload = {
        "status": "complete_candidate_screen",
        "recordings": len(rows),
        "within_class_pairs": len(pairs),
        "candidate_pairs": len(candidates),
        "thresholds": {
            "acoustic_shingle_jaccard_at_least": CANDIDATE_JACCARD,
            "duration_ratio_at_least": CANDIDATE_DURATION_RATIO,
        },
        "maximum_observed_jaccard": max(row["acoustic_shingle_jaccard"] for row in pairs),
        "candidate_record_pairs": [
            [row["left_record_id"], row["right_record_id"]] for row in candidates
        ],
        "interpretation": "A below-threshold result reduces concern about direct transcodes under this fingerprint, but cannot prove that two files come from independently staged performances. Candidate pairs require provenance and listening inspection before any grouped resplit.",
        "method": "Within-class Jaccard similarity of 2-second time-ordered shingles built from dominant chroma, coarse spectral-centroid bins and energy direction on 0.5-second frames.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
