#!/usr/bin/env python3
"""Extract preregistered MERT layer summaries for the locked benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModel, Wav2Vec2FeatureExtractor


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
WORK = ROOT / "work"
LOCKED = OUT / "kathakalipadam_mert_locked_benchmark.csv"
PROTOCOL = OUT / "kathakalipadam_mert_evaluation_protocol.json"
EMBEDDINGS = WORK / "mert_eval" / "embeddings"
MODEL_ID = "m-a-p/MERT-v1-95M"
MODEL_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
WEIGHT_NORM_COMPAT = "legacy_checkpoint_weight_g_weight_v"
SAMPLE_RATE = 24_000
CHUNK_SECONDS = 8.0
CHUNKS = 5


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def local_audio(record_id: str) -> Path:
    candidates: list[Path] = []
    for folder in (
        WORK / "padam_smoke_test" / "audio_mp3",
        WORK / "corpus_expansion" / "audio",
    ):
        candidates.extend(
            path for path in folder.glob(f"{record_id}.*") if not path.name.endswith(".part")
        )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one audio file for {record_id}, found {candidates}")
    return candidates[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_starts(path: Path) -> np.ndarray:
    with sf.SoundFile(path) as audio:
        duration = len(audio) / audio.samplerate
    max_start = max(0.0, duration - CHUNK_SECONDS)
    return np.linspace(0.05 * max_start, 0.85 * max_start, CHUNKS, dtype=np.float64)


def read_chunk(path: Path, start: float) -> np.ndarray:
    with sf.SoundFile(path) as audio:
        frames = int(round(CHUNK_SECONDS * audio.samplerate))
        audio.seek(min(int(round(start * audio.samplerate)), max(0, len(audio) - 1)))
        block = audio.read(frames=frames, dtype="float32", always_2d=True)
        mono = block.mean(axis=1)
        if len(mono) < frames:
            mono = np.pad(mono, (0, frames - len(mono)))
        if audio.samplerate != SAMPLE_RATE:
            divisor = math.gcd(audio.samplerate, SAMPLE_RATE)
            mono = resample_poly(mono, SAMPLE_RATE // divisor, audio.samplerate // divisor)
    target = int(CHUNK_SECONDS * SAMPLE_RATE)
    if len(mono) < target:
        mono = np.pad(mono, (0, target - len(mono)))
    return np.asarray(mono[:target], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "preregistered_before_MERT_model_download_or_feature_extraction":
        raise RuntimeError("Evaluation protocol is not in the expected preregistered state")

    rows = read_rows(LOCKED)
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise ValueError("Require 0 <= shard < num-shards")
    rows = rows[args.shard :: args.num_shards]
    if args.limit:
        rows = rows[: args.limit]
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cpu" if args.cpu or not torch.backends.mps.is_available() else "mps"
    )
    print(f"Loading {MODEL_ID}; device={device}", flush=True)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    # The MERT checkpoint stores the legacy PyTorch ``weight_g``/``weight_v``
    # keys. Transformers 4.44 otherwise constructs the HuBERT positional
    # convolution with the newer parametrization API and silently leaves those
    # two pretrained tensors unused. Construct this one module with the legacy
    # API so every checkpoint weight maps to its intended parameter.
    parametrizations = torch.nn.utils.parametrizations
    modern_weight_norm = getattr(parametrizations, "weight_norm", None)
    if modern_weight_norm is not None:
        delattr(parametrizations, "weight_norm")
    try:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            output_hidden_states=True,
        )
    finally:
        if modern_weight_norm is not None:
            setattr(parametrizations, "weight_norm", modern_weight_norm)
    model.eval().to(device)
    commit = str(getattr(model.config, "_commit_hash", "unknown"))
    hidden_size = int(model.config.hidden_size)
    print(f"Resolved commit={commit}; hidden_size={hidden_size}", flush=True)

    started = time.time()
    for index, row in enumerate(rows, start=1):
        record_id = row["record_id"]
        path = local_audio(record_id)
        source_sha = sha256(path)
        if source_sha != row["audio_sha256"]:
            raise RuntimeError(f"Audio digest mismatch for {record_id}")
        destination = EMBEDDINGS / f"{record_id}.npz"
        if destination.exists():
            cached = np.load(destination, allow_pickle=False)
            if (
                str(cached["source_sha256"].item()) == source_sha
                and str(cached["model_commit"].item()) == commit
                and "weight_norm_compat" in cached.files
                and str(cached["weight_norm_compat"].item()) == WEIGHT_NORM_COMPAT
                and cached["layer_chunk_means"].shape == (13, CHUNKS, hidden_size)
            ):
                print(f"MERT {index}/{len(rows)} {record_id} (cached)", flush=True)
                continue

        starts = chunk_starts(path)
        waveforms = np.stack([read_chunk(path, float(start)) for start in starts])
        inputs = feature_extractor(
            list(waveforms), sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(device)
        with torch.inference_mode():
            output = model(input_values, output_hidden_states=True)
        if output.hidden_states is None or len(output.hidden_states) != 13:
            raise RuntimeError(
                f"Expected 13 hidden-state levels for {record_id}; "
                f"found {0 if output.hidden_states is None else len(output.hidden_states)}"
            )
        layer_chunk_means = np.stack(
            [state.mean(dim=1).float().cpu().numpy() for state in output.hidden_states]
        ).astype(np.float32)
        np.savez_compressed(
            destination,
            layer_chunk_means=layer_chunk_means,
            starts_seconds=starts,
            source_sha256=np.asarray(source_sha),
            model_id=np.asarray(MODEL_ID),
            model_commit=np.asarray(commit),
            weight_norm_compat=np.asarray(WEIGHT_NORM_COMPAT),
            sample_rate_hz=np.asarray(SAMPLE_RATE),
            chunk_seconds=np.asarray(CHUNK_SECONDS),
            record_id=np.asarray(record_id),
            canonical_id=np.asarray(row["padam_canonical_id"]),
        )
        del output, input_values, inputs
        if device.type == "mps":
            torch.mps.empty_cache()
        elapsed = time.time() - started
        print(
            f"MERT {index}/{len(rows)} {record_id}; elapsed={elapsed / 60:.1f} min; "
            f"mean={elapsed / index:.1f} s/record",
            flush=True,
        )

    metadata = {
        "model_id": MODEL_ID,
        "resolved_commit": commit,
        "requested_revision": MODEL_REVISION,
        "weight_norm_compat": WEIGHT_NORM_COMPAT,
        "device": str(device),
        "sample_rate_hz": SAMPLE_RATE,
        "chunk_seconds": CHUNK_SECONDS,
        "chunks_per_recording": CHUNKS,
        "expected_recordings": len(read_rows(LOCKED)),
        "process_shard": args.shard,
        "process_num_shards": args.num_shards,
        "completed_caches": len(list(EMBEDDINGS.glob("*.npz"))),
    }
    (WORK / "mert_eval" / "embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
