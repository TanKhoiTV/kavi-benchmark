"""Materialize FLEURS subset to WAV for VAD manifest (Checkpoint A/B)."""

from __future__ import annotations

import io
import random
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch


def materialize(
    source_parquet: str, out_dir: str, n: int = 30, seed: int = 42, lang: str = "vi"
) -> list[str]:
    wd = Path(out_dir)
    wd.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(source_parquet)
    meta = pf.read_row_groups(
        list(range(pf.metadata.num_row_groups)), columns=["id", "transcription"]
    ).to_pylist()
    total = len(meta)
    idxs = (
        list(range(total))
        if total <= n
        else random.Random(seed).sample(range(total), n)
    )
    sampled_ids = {str(meta[i]["id"]) for i in idxs}
    audio_by_id: dict[str, dict] = {}
    for batch in pf.iter_batches(columns=["id", "audio"], batch_size=64):
        for r in batch.to_pylist():
            rid = str(r.get("id"))
            if rid in sampled_ids:
                audio_by_id[rid] = r.get("audio") or {}
    out_paths = []
    sr = 16000
    for i in idxs:
        r = meta[i]
        uid = str(r.get("id"))
        a = audio_by_id.get(uid) or {}
        arr = a.get("array")
        if arr is None:
            blob = a.get("bytes")
            if blob is None:
                continue
            data, sr_file = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
            audio = torch.from_numpy(data).transpose(0, 1)
            if sr_file != sr:
                import torchaudio

                audio = torchaudio.functional.resample(audio, sr_file, sr)
        else:
            audio = torch.tensor(arr).float().unsqueeze(0)
            sr_file = a.get("sampling_rate") or sr
            if sr_file != sr and audio.shape[-1] > 0:
                import torchaudio

                audio = torchaudio.functional.resample(audio, sr_file, sr)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        wav_path = wd / f"{lang}_{uid}.wav"
        sf.write(str(wav_path), audio.transpose(0, 1).numpy(), sr)
        out_paths.append(str(wav_path))
    return out_paths
