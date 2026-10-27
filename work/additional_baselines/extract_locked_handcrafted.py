#!/usr/bin/env python3
"""Extract the established handcrafted features for missing locked records.

The extractor matches the 20 s, six-excerpt protocol used by the earlier
canonical feasibility study. Existing checksum-matched caches are preserved.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "outputs" / "kathakalipadam_mert_locked_benchmark.csv"
CACHE = ROOT / "work" / "canonical_full_evaluation" / "record_features"
AUDIO_DIRS = (
    ROOT / "work" / "padam_smoke_test" / "audio_mp3",
    ROOT / "work" / "corpus_expansion" / "audio",
)
TARGET_SR = 16_000
CHUNK_SECONDS = 20.0
CHUNKS = 6


def rows() -> list[dict[str, str]]:
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


def starts(duration: float) -> np.ndarray:
    if duration <= CHUNK_SECONDS:
        return np.asarray([0.0])
    lead = 15.0 if duration >= CHUNK_SECONDS + 30.0 else 0.0
    trailing = 10.0 if duration >= CHUNK_SECONDS + 20.0 else 0.0
    latest = max(lead, duration - trailing - CHUNK_SECONDS)
    return np.linspace(lead, latest, CHUNKS, dtype=float)


def read_chunk(audio: sf.SoundFile, start_seconds: float) -> np.ndarray:
    source_sr = int(audio.samplerate)
    audio.seek(min(int(round(start_seconds * source_sr)), max(0, len(audio) - 1)))
    samples = audio.read(
        frames=int(round(CHUNK_SECONDS * source_sr)),
        dtype="float32",
        always_2d=True,
    )
    mono = samples.mean(axis=1) if samples.size else np.zeros(0, dtype=np.float32)
    if source_sr != TARGET_SR and mono.size:
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=TARGET_SR)
    target = int(round(CHUNK_SECONDS * TARGET_SR))
    if mono.size < target:
        mono = np.pad(mono, (0, target - mono.size))
    return np.asarray(mono[:target], dtype=np.float32)


def summarize(prefix: str, matrix: np.ndarray) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    for index, row in enumerate(np.atleast_2d(matrix)):
        names.extend((f"{prefix}_{index + 1:02d}_mean", f"{prefix}_{index + 1:02d}_std"))
        values.extend((float(np.mean(row)), float(np.std(row))))
    return names, values


def features(y: np.ndarray) -> tuple[list[str], np.ndarray]:
    kwargs = dict(n_fft=1024, hop_length=512)
    mfcc = librosa.feature.mfcc(
        y=y, sr=TARGET_SR, n_mfcc=20, n_mels=40, fmin=50,
        fmax=TARGET_SR / 2, **kwargs,
    )
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=TARGET_SR, **kwargs)
    spectral = np.vstack([
        librosa.feature.spectral_centroid(y=y, sr=TARGET_SR, **kwargs),
        librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR, **kwargs),
        librosa.feature.spectral_rolloff(
            y=y, sr=TARGET_SR, roll_percent=0.85, **kwargs
        ),
        librosa.feature.spectral_flatness(y=y, **kwargs),
        librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512),
        librosa.feature.rms(y=y, frame_length=1024, hop_length=512),
    ])
    names: list[str] = []
    values: list[float] = []
    for prefix, matrix in (
        ("mfcc", mfcc), ("delta_mfcc", delta),
        ("chroma", chroma), ("spectral", spectral),
    ):
        block_names, block_values = summarize(prefix, matrix)
        names.extend(block_names)
        values.extend(block_values)
    vector = np.nan_to_num(np.asarray(values, dtype=np.float64))
    return names, vector


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    locked = rows()
    created = 0
    for index, row in enumerate(locked, 1):
        record_id = row["record_id"]
        path = audio_path(record_id)
        digest = file_sha256(path)
        if digest != row["audio_sha256"]:
            raise RuntimeError(f"Checksum mismatch for record {record_id}")
        destination = CACHE / f"{record_id}.npz"
        if destination.exists():
            cached = np.load(destination, allow_pickle=True)
            if str(cached["sha256"].item()) == digest:
                continue
        with sf.SoundFile(path) as audio:
            duration = len(audio) / audio.samplerate
            positions = starts(duration)
            vectors: list[np.ndarray] = []
            feature_names: list[str] = []
            for start in positions:
                feature_names, vector = features(read_chunk(audio, float(start)))
                vectors.append(vector)
        np.savez_compressed(
            destination,
            X=np.vstack(vectors),
            starts=positions,
            feature_names=np.asarray(feature_names),
            sha256=digest,
        )
        created += 1
        print(f"handcrafted {index}/{len(locked)}: {record_id}", flush=True)
    print(f"complete: created {created}; available {len(list(CACHE.glob('*.npz')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
