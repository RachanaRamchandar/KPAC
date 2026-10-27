#!/usr/bin/env python3
"""Extract cached YAMNet embeddings for the exact canonical evaluation recordings."""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
WORK = ROOT / "work"
PREDICTIONS = OUT / "canonical_full_dataset_classification_predictions.csv"
MANIFEST = OUT / "kathakalipadam_canonicalized_expanded_425.csv"
HANDCRAFTED_DIR = WORK / "canonical_full_evaluation" / "record_features"
AUDIO_DIR = WORK / "padam_smoke_test" / "audio_mp3"
EMBEDDING_DIR = WORK / "pretrained_embedding_eval" / "yamnet_record_embeddings"
MODEL_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_SR = 16_000
CHUNK_SECONDS = 20.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def exact_record_ids() -> list[str]:
    rows = read_csv(PREDICTIONS)
    ids = {
        row["record_id"]
        for row in rows
        if row["experiment"] == "record_mean_std" and row["model"] == "linear_svm"
    }
    if len(ids) != 233:
        raise RuntimeError(f"Expected 233 exact-split recordings, found {len(ids)}")
    return sorted(ids)


def audio_path(record_id: str) -> Path:
    candidates = sorted(
        path for path in AUDIO_DIR.glob(f"{record_id}.*") if not path.name.endswith(".part")
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one cached audio file for {record_id}, found {candidates}")
    return candidates[0]


def read_chunk(path: Path, start_seconds: float) -> np.ndarray:
    with sf.SoundFile(path) as audio:
        audio.seek(min(int(round(start_seconds * audio.samplerate)), max(0, len(audio) - 1)))
        frames = int(round(CHUNK_SECONDS * audio.samplerate))
        block = audio.read(frames=frames, dtype="float32", always_2d=True)
        mono = block.mean(axis=1)
        if len(mono) < frames:
            mono = np.pad(mono, (0, frames - len(mono)))
        if audio.samplerate != TARGET_SR:
            divisor = math.gcd(audio.samplerate, TARGET_SR)
            mono = resample_poly(mono, TARGET_SR // divisor, audio.samplerate // divisor)
    target_length = int(CHUNK_SECONDS * TARGET_SR)
    if len(mono) < target_length:
        mono = np.pad(mono, (0, target_length - len(mono)))
    return np.asarray(mono[:target_length], dtype=np.float32)


def main() -> int:
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {row["record_id"]: row for row in read_csv(MANIFEST)}
    record_ids = exact_record_ids()
    print(f"Loading YAMNet from {MODEL_URL}", flush=True)
    model = hub.load(MODEL_URL)
    started = time.time()
    for index, record_id in enumerate(record_ids, start=1):
        source_cache = np.load(HANDCRAFTED_DIR / f"{record_id}.npz", allow_pickle=True)
        source_sha = str(source_cache["sha256"].item())
        starts = np.asarray(source_cache["starts"], dtype=float)
        destination = EMBEDDING_DIR / f"{record_id}.npz"
        if destination.exists():
            cached = np.load(destination, allow_pickle=True)
            if (
                str(cached["sha256"].item()) == source_sha
                and str(cached["model_url"].item()) == MODEL_URL
                and np.array_equal(np.asarray(cached["starts"], dtype=float), starts)
            ):
                if index % 10 == 0 or index == len(record_ids):
                    print(f"Embeddings {index}/{len(record_ids)} (cached)", flush=True)
                continue

        chunk_embedding_means: list[np.ndarray] = []
        chunk_embedding_stds: list[np.ndarray] = []
        chunk_score_means: list[np.ndarray] = []
        path = audio_path(record_id)
        for start in starts:
            waveform = read_chunk(path, float(start))
            scores, embeddings, _ = model(tf.convert_to_tensor(waveform, dtype=tf.float32))
            embedding_array = np.asarray(embeddings, dtype=np.float32)
            score_array = np.asarray(scores, dtype=np.float32)
            chunk_embedding_means.append(embedding_array.mean(axis=0))
            chunk_embedding_stds.append(embedding_array.std(axis=0))
            chunk_score_means.append(score_array.mean(axis=0))

        np.savez_compressed(
            destination,
            embedding_mean=np.vstack(chunk_embedding_means),
            embedding_frame_std=np.vstack(chunk_embedding_stds),
            score_mean=np.vstack(chunk_score_means),
            starts=starts,
            sha256=source_sha,
            model_url=MODEL_URL,
            record_id=record_id,
            canonical_id=manifest[record_id]["padam_canonical_id"],
        )
        if index % 5 == 0 or index == len(record_ids):
            elapsed = time.time() - started
            print(
                f"Embeddings {index}/{len(record_ids)}; elapsed {elapsed / 60:.1f} min; "
                f"mean {elapsed / index:.1f} s/record",
                flush=True,
            )
    print(f"Complete: {len(record_ids)} recording caches in {EMBEDDING_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
