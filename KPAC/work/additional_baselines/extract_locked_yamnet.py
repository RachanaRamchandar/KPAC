#!/usr/bin/env python3
"""Extract YAMNet summaries for the 112-row locked KPAC benchmark."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "outputs" / "kathakalipadam_mert_locked_benchmark.csv"
HANDCRAFTED = ROOT / "work" / "canonical_full_evaluation" / "record_features"
CACHE = ROOT / "work" / "additional_baselines" / "yamnet"
TFHUB_CACHE = ROOT / "work" / "pretrained_embedding_eval" / "tfhub_cache"
AUDIO_DIRS = (
    ROOT / "work" / "padam_smoke_test" / "audio_mp3",
    ROOT / "work" / "corpus_expansion" / "audio",
)
MODEL_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_SR = 16_000
CHUNK_SECONDS = 20.0


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_chunk(path: Path, start: float) -> np.ndarray:
    with sf.SoundFile(path) as audio:
        audio.seek(min(int(round(start * audio.samplerate)), max(0, len(audio) - 1)))
        frames = int(round(CHUNK_SECONDS * audio.samplerate))
        block = audio.read(frames=frames, dtype="float32", always_2d=True)
        mono = block.mean(axis=1)
        if len(mono) < frames:
            mono = np.pad(mono, (0, frames - len(mono)))
        if audio.samplerate != TARGET_SR:
            divisor = math.gcd(audio.samplerate, TARGET_SR)
            mono = resample_poly(mono, TARGET_SR // divisor, audio.samplerate // divisor)
    target = int(CHUNK_SECONDS * TARGET_SR)
    if len(mono) < target:
        mono = np.pad(mono, (0, target - len(mono)))
    return np.asarray(mono[:target], dtype=np.float32)


def main() -> int:
    os.environ.setdefault("TFHUB_CACHE_DIR", str(TFHUB_CACHE))
    CACHE.mkdir(parents=True, exist_ok=True)
    model = hub.load(MODEL_URL)
    locked = read_rows()
    for index, row in enumerate(locked, 1):
        record_id = row["record_id"]
        path = audio_path(record_id)
        digest = file_sha256(path)
        if digest != row["audio_sha256"]:
            raise RuntimeError(f"Checksum mismatch for record {record_id}")
        source = np.load(HANDCRAFTED / f"{record_id}.npz", allow_pickle=True)
        starts = np.asarray(source["starts"], dtype=float)
        destination = CACHE / f"{record_id}.npz"
        if destination.exists():
            cached = np.load(destination, allow_pickle=True)
            if (
                str(cached["sha256"].item()) == digest
                and np.array_equal(np.asarray(cached["starts"], dtype=float), starts)
            ):
                continue
        chunk_means: list[np.ndarray] = []
        chunk_stds: list[np.ndarray] = []
        for start in starts:
            scores, embeddings, _ = model(
                tf.convert_to_tensor(read_chunk(path, float(start)), dtype=tf.float32)
            )
            values = np.asarray(embeddings, dtype=np.float32)
            chunk_means.append(values.mean(axis=0))
            chunk_stds.append(values.std(axis=0))
        np.savez_compressed(
            destination,
            embedding_mean=np.vstack(chunk_means),
            embedding_frame_std=np.vstack(chunk_stds),
            starts=starts,
            sha256=digest,
            model_url=MODEL_URL,
        )
        print(f"yamnet {index}/{len(locked)}: {record_id}", flush=True)
    print(f"complete: {len(list(CACHE.glob('*.npz')))} locked YAMNet caches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
