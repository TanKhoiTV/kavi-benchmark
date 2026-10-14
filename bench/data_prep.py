"""Phase 1 data prep: build the lean eval set + noise/SNR variants -> manifest.

The eval set is built from locally-available FLEURS parquets when present
(fetch them once with `make data`, which pins the dataset revision and verifies
digests against `assets.lock.toml`). When they are not present, a runnable
**offline fallback** set is emitted instead: an authored VI<->EN
factory/logistics gold set scored as MT (vi->en) + TTS items, using synthetic
noise for the SNR recipe.

Speech corpora (FLEURS `vi_vn`/`en_us`, CC BY 4.0) give:
  - VI ASR + EN ASR items (real speech, transcript references)
  - VI<->EN speech-translation items via FLEURS' shared sentence IDs (plan S3.3)
The authored gold set adds domain-flavored MT/TTS references that need no
download. ASR items are only emitted when FLEURS parquets are available; the
offline fallback covers MT + TTS, which the v0 CPU candidates score fully
offline (Opus-MT vi->en + Piper EN).

Noise is applied at eval time (plan S3.5): synthetic steady/impulsive noise mixed
to a target SNR via torchaudio.add_noise. Swap in MUSAN/RIRS_NOISES later by
replacing `_synth_noise` with real clips (the manifest schema is unchanged).
"""

from __future__ import annotations

import random
from pathlib import Path

import soundfile as sf

from .schema import EvalItem, RunManifest

SNR_LEVELS = [None, 15.0, 10.0, 5.0, 0.0]  # clean + mild/moderate/hard/severe
NOISE_TYPES = ["steady", "impulsive"]

# Authored VI<->EN factory/logistics gold set (text only, no download needed).
# Domain-skewed per the benchmarking plan (S8): equipment, safety, numbers,
# short imperatives. Use as MT (vi->en) + TTS reference texts.
GOLD_SET: list[tuple[str, str]] = [
    (
        "Xin hãy đeo kính bảo hộ trước khi vào khu vực làm việc.",
        "Please wear safety goggles before entering the work area.",
    ),
    (
        "Cần kiểm tra áp suất lốp xe nâng mỗi ca làm việc.",
        "Check the forklift tire pressure at the start of every shift.",
    ),
    (
        "Máy nén khí đang quá nhiệt, hãy tắt ngay lập tức.",
        "The air compressor is overheating, switch it off immediately.",
    ),
    (
        "Kho hàng số tám đã nhận đủ năm mươi thùng hàng.",
        "Warehouse number eight has received the full fifty cartons.",
    ),
    (
        "Cảnh báo: khu vực này có nguy cơ rơi vật thể từ trên cao.",
        "Warning: this area has a risk of falling objects from above.",
    ),
    (
        "Vui lòng giữ khoảng cách hai mét với băng chuyền.",
        "Please keep a two meter distance from the conveyor belt.",
    ),
    (
        "Đèn báo hiệu đang nhấp nháy, không được đi qua.",
        "The signal light is flashing, do not walk through.",
    ),
    (
        "Hôm nay chúng ta xuất kho tổng cộng một trăm hai mươi đơn.",
        "Today we shipped a total of one hundred and twenty orders.",
    ),
    (
        "Mặt hàng dễ vỡ, xếp nhẹ tay và dán nhãn cẩn thận.",
        "Fragile goods, handle with care and label them properly.",
    ),
    (
        "Cần bổ sung ba kiện hàng vào chuyến giao buổi chiều.",
        "We need to add three packages to the afternoon delivery.",
    ),
    (
        "Van an toàn đã mở, xả áp suất dư thừa ra ngoài.",
        "The relief valve opened and vented the excess pressure outside.",
    ),
    (
        "Cấm sử dụng điện thoại di động gần trạm biến áp.",
        "Mobile phones are prohibited near the transformer station.",
    ),
]


def _synth_noise(kind: str, length: int, sample_rate: int, seed: int):
    import torch
    import torchaudio

    g = torch.Generator().manual_seed(seed)
    if kind == "steady":
        noise = torch.randn(1, length, generator=g)
        noise = torchaudio.functional.lowpass_biquad(noise, sample_rate, 1200)
    else:  # impulsive: periodic bursts -> forklift/clatter impulses
        noise = torch.zeros(1, length)
        burst = round(0.02 * sample_rate)
        spacing = round(0.25 * sample_rate)
        rng = random.Random(seed)
        pos = rng.randint(0, max(1, burst))
        while pos < length:
            end = min(pos + burst, length)
            noise[0, pos:end] = torch.randn(end - pos, generator=g) * 4.0
            pos += spacing
    return noise


def _load_real_noise(dir_path: str, sample_rate: int) -> dict[str, list]:
    """Load real noise clips from `dir_path/{steady,impulsive}/*.wav`.

    Returns NOISE_TYPES -> list[Tensor] (mono, at sample_rate). Used by
    build_lean_manifest when a real noise bank is supplied, replacing the
    synthetic `_synth_noise`. Requires real assets (e.g. MUSAN/RIRS_NOISES
    from openslr); those are NOT fetched automatically -- acquisition is
    large-file downloads in some CI environments, so the
    synthetic fallback remains the v0 default.
    """
    import torch
    import torchaudio

    out: dict[str, list] = {kind: [] for kind in NOISE_TYPES}
    base = Path(dir_path)
    for kind in NOISE_TYPES:
        d = base / kind
        if not d.is_dir():
            continue
        for wav in sorted(d.glob("*.wav")):
            try:
                data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            except Exception:
                continue
            t = torch.from_numpy(data).mean(0, keepdim=True)
            if sr != sample_rate:
                t = torchaudio.functional.resample(t, sr, sample_rate)
            out[kind].append(t)
    return out


def _mix(clean, noise, snr_db: float):
    import torch
    import torchaudio

    if noise.shape[-1] < clean.shape[-1]:
        reps = (clean.shape[-1] + noise.shape[-1] - 1) // noise.shape[-1]
        noise = noise.repeat(1, reps)
    noise = noise[:, : clean.shape[-1]]
    try:
        return torchaudio.functional.add_noise(clean, noise, torch.tensor([snr_db]))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"noise mix failed (snr={snr_db}): {exc}") from exc


def mix_noise(clean_wav: str, noise_wav: str, snr_db: float, out_path: str) -> None:
    """Mix `noise_wav` into `clean_wav` at `snr_db` dB and write to `out_path`."""
    import torch
    import torchaudio

    clean, sr = sf.read(clean_wav, dtype="float32", always_2d=True)
    clean = torch.from_numpy(clean).transpose(0, 1)
    noise, nsr = sf.read(noise_wav, dtype="float32", always_2d=True)
    noise = torch.from_numpy(noise).transpose(0, 1)
    if nsr != sr:
        noise = torchaudio.functional.resample(noise, nsr, sr)
    mixed = _mix(clean, noise, snr_db)
    sf.write(out_path, mixed.transpose(0, 1).numpy(), sr)


def _load_fleurs(path: str, n: int, sample_rate: int, seed: int = 42):
    """Return list of (id, transcript, audio_tensor) from a FLEURS parquet.

    FLEURS stores the `audio` column as encoded bytes (WAV); decode via
    soundfile and resample/downmix to `sample_rate`. Sampling is seeded so
    the lean set is reproducible and unbiased by file order. Peak memory is
    bounded: the audio column is streamed in small batches and only the
    `n` sampled rows are retained, rather than materializing the full
    (~690 MB) audio column up front.
    """
    import io
    import random

    import pyarrow.parquet as pq
    import torch
    import torchaudio

    try:
        pf = pq.ParquetFile(path)
        num_rg = pf.metadata.num_row_groups
        # Cheap pass: ids + transcripts for ALL row groups (no audio bytes),
        # so seeded sampling draws from the full utterance pool regardless of
        # row-group layout (no per-row-group sampling bias).
        meta = pf.read_row_groups(
            list(range(num_rg)), columns=["id", "transcription"]
        ).to_pylist()
        total = len(meta)
        idxs = (
            list(range(total))
            if total <= n
            else random.Random(seed).sample(range(total), n)
        )
        # Select by id, not by position. The audio pass below uses a
        # different pyarrow API (iter_batches) than this metadata pass
        # (read_row_groups); relying on matching row order between the two
        # is unsafe (no cross-version/engine guarantee) and would silently
        # drop or mis-align rows. Keying on the stable `id` makes the audio
        # read order-independent.
        sampled_ids = {str(meta[i]["id"]) for i in idxs}
        audio_by_id: dict[str, dict] = {}
        for batch in pf.iter_batches(columns=["id", "audio"], batch_size=64):
            for r in batch.to_pylist():
                rid = str(r.get("id"))
                if rid in sampled_ids:
                    audio_by_id[rid] = r.get("audio") or {}
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot read FLEURS parquet {path}: {exc}") from exc
    out = []
    for i in idxs:
        r = meta[i]
        uid = str(r.get("id"))
        a = r.get("audio") or audio_by_id.get(uid) or {}
        arr = a.get("array")
        if arr is None:
            blob = a.get("bytes")
            if blob is None:
                continue
            data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
            audio = torch.from_numpy(data).transpose(0, 1)
        else:
            audio = torch.tensor(arr).float().unsqueeze(0)
            sr = a.get("sampling_rate") or sample_rate
        if audio.shape[0] > 1:  # downmix to mono
            audio = audio.mean(0, keepdim=True)
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        out.append((uid, (r.get("transcription") or "").strip(), audio))
    return out


def _load_fleurs_ids_texts(path: str) -> dict[str, str]:
    """Full id->transcription map (no audio decode) for VI->EN MT pairing."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "transcription"])
    return {
        str(r["id"]): (r.get("transcription") or "").strip() for r in table.to_pylist()
    }


def _emit_asr_items(
    items, mixed_dir, lang, idx, uid, transcript, audio, sample_rate, real_noise=None
):
    """Emit the SNR-sweep ASR items for one utterance (clean + noisy).

    `real_noise`, when provided, maps NOISE_TYPES -> list[Tensor] of real
    clips (already at `sample_rate`); those replace the synthetic
    `_synth_noise` generator for the sweep. This is the documented swap point
    for MUSAN/RIRS_NOISES once those assets are available.
    """
    # Clean reference condition: identical regardless of noise type, so emit once.
    clean_wav = mixed_dir / f"{lang}_{uid}_clean.wav"
    sf.write(str(clean_wav), audio.transpose(0, 1).numpy(), sample_rate)
    items.append(
        EvalItem(
            id=f"{lang}-asr-{uid}-clean",
            stage="ASR",
            language=lang,
            direction="",
            audio_ref=str(clean_wav),
            reference_text=transcript,
            snr=None,
            noise_type="clean",
        )
    )
    for kind in NOISE_TYPES:
        clips = (real_noise or {}).get(kind)
        for snr in SNR_LEVELS:
            if snr is None:
                continue
            if clips:
                noise = clips[idx % len(clips)]
            else:
                noise = _synth_noise(kind, audio.shape[-1], sample_rate, seed=idx + 1)
            mixed = _mix(audio, noise, snr)
            wav = mixed_dir / f"{lang}_{uid}_{kind}_{snr:g}.wav"
            sf.write(str(wav), mixed.transpose(0, 1).numpy(), sample_rate)
            items.append(
                EvalItem(
                    id=f"{lang}-asr-{uid}-{kind}-{snr:g}",
                    stage="ASR",
                    language=lang,
                    direction="",
                    audio_ref=str(wav),
                    reference_text=transcript,
                    snr=snr,
                    noise_type=kind,
                )
            )


def _evenly_spaced_indices(n_available: int, k: int) -> list[int]:
    """k deterministic, evenly spaced, distinct indices from ``range(n_available)``.

    The naive expression ``round(step / 2 + i * step)`` can collapse two
    different ``i`` onto the same index, and clamping to ``n_available - 1`` can
    fold several onto the last one. Once ``k`` approaches ``n_available`` that
    returns fewer than k picks and the caller emits duplicate utterances while
    still reporting the requested count. Deduplicate, then top up in index
    order so the caller always gets exactly k distinct picks.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > n_available:
        raise ValueError(f"cannot pick {k} distinct indices from {n_available}")
    step = n_available / k
    picks: list[int] = []
    seen: set[int] = set()
    for i in range(k):
        p = min(round(step / 2 + i * step), n_available - 1)
        if p not in seen:
            seen.add(p)
            picks.append(p)
    for p in range(n_available):
        if len(picks) >= k:
            break
        if p not in seen:
            seen.add(p)
            picks.append(p)
    picks.sort()
    assert len(picks) == k == len(set(picks)), (
        f"expected {k} distinct indices, got {len(set(picks))}"
    )
    return picks


def _load_vivos_utterances(
    vivos_root: str | Path,
    n_items: int = 100,
    sample_rate: int = 16000,
) -> list:
    """Load (uid, transcript, audio_tensor) triples from the VIVOS test split.

    Same speaker-stratified deterministic selection as
    ``bench/scripts/convert_vivos_manifest.py`` (19 speakers, round-robin
    base + remainder allocation, evenly-spaced picks per speaker, sorted by
    uid), so the sampled utterances match the existing
    ``vivos_vi_test_manifest.json``. Audio is decoded to mono (1, N) float32
    at ``sample_rate``.
    """
    import torch
    import torchaudio

    test_dir = Path(vivos_root) / "test"
    waves_dir = test_dir / "waves"

    # prompts.txt: "<ID> <TRANSCRIPT>" per line
    prompts: dict[str, str] = {}
    for line in (
        (test_dir / "prompts.txt").read_text(encoding="utf-8").strip().split("\n")
    ):
        parts = line.split(" ", 1)
        if len(parts) == 2:
            prompts[parts[0]] = parts[1].strip()

    # Group by speaker (VIVOSDEV01..19), preserve file order; drop missing audio
    speakers: dict[str, list[str]] = {}
    for uid in prompts:
        speakers.setdefault(uid.rsplit("_", 1)[0], []).append(uid)
    for spk in list(speakers):
        speakers[spk] = [
            uid for uid in speakers[spk] if (waves_dir / spk / f"{uid}.wav").exists()
        ]
        if not speakers[spk]:
            del speakers[spk]
    n_total = sum(len(v) for v in speakers.values())
    n_speakers = len(speakers)
    if n_total < n_items:
        raise ValueError(
            f"Only {n_total} VIVOS test utterances available, need {n_items}"
        )
    if n_items < n_speakers:
        raise ValueError(
            f"n_items ({n_items}) must be >= number of VIVOS speakers ({n_speakers})"
        )

    # Allocate counts per speaker: base + remainder to first speakers
    base, rem = divmod(n_items, n_speakers)
    counts = {
        spk: base + (1 if i < rem else 0) for i, spk in enumerate(sorted(speakers))
    }
    # Evenly-spaced picks within each speaker's ordered list
    selected: list[str] = []
    for spk in sorted(speakers):
        uids = speakers[spk]
        k = counts[spk]
        # Evenly-spaced distinct picks within each speaker's ordered list.
        for p in _evenly_spaced_indices(len(uids), k):
            selected.append(uids[p])
    selected.sort()  # deterministic output order

    out: list = []
    for uid in selected:
        spk = uid.rsplit("_", 1)[0]
        data, sr = sf.read(
            str(waves_dir / spk / f"{uid}.wav"), dtype="float32", always_2d=True
        )
        audio = torch.from_numpy(data).transpose(0, 1).mean(0, keepdim=True)
        if sr != sample_rate:
            audio = torchaudio.functional.resample(audio, sr, sample_rate)
        out.append((uid, prompts[uid], audio))
    return out


def build_vivos_manifest(
    out_path: str,
    workdir: str = "eval_data",
    n_items: int = 100,
    sample_rate: int = 16000,
    real_noise_dir: str | None = None,
    mixed_dir: str | Path | None = None,
) -> None:
    """Build a full-schema ASR eval manifest from the VIVOS test split (clean vi).

    Mirror of the FLEURS ASR path in :func:`build_lean_manifest`: each sampled
    utterance is emitted as 9 EvalItems (clean + steady/impulsive @ 15/10/5/0 dB)
    via :func:`_emit_asr_items`, with the mixed audio written to ``mixed_dir``
    (default ``{workdir}/mixed``; pass e.g. ``eval_data/vivos-mixed`` to keep
    VIVOS audio separate from the FLEURS bank). Uses the same synthetic noise
    recipe, or real clips from ``real_noise_dir`` when supplied.
    """
    wd = Path(workdir)
    mixed_dir = Path(mixed_dir) if mixed_dir else wd / "mixed"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    real_noise = (
        _load_real_noise(real_noise_dir, sample_rate) if real_noise_dir else None
    )

    utterances = _load_vivos_utterances(wd / "raw" / "vivos", n_items, sample_rate)
    items: list[EvalItem] = []
    for idx, (uid, transcript, audio) in enumerate(utterances):
        _emit_asr_items(
            items,
            mixed_dir,
            "vi",
            idx,
            uid,
            transcript,
            audio,
            sample_rate,
            real_noise=real_noise,
        )
    RunManifest(items=items).to_json(out_path)
    print(
        f"VIVOS: added {len(utterances)} VI ASR utterances -> {len(items)} items "
        f"({sum(1 for it in items if it.stage == 'ASR')} ASR)"
    )


def build_mt_manifest(
    out_path: str,
    workdir: str = "eval_data",
    n_items: int | None = None,
) -> None:
    """Build a text-only vi->en MT eval manifest from the full FLEURS parallel pool.

    FLEURS is a parallel corpus: the vi and en test parquets share utterance
    ids, so every id present in both parquets is a vi->en translation pair.
    MT items need no audio (``input_text`` + ``reference_text`` only), so this
    is decoupled from ASR sampling and cheap to build. Deterministic (sorted
    by id); ``n_items`` optionally subsamples with even spacing. The authored
    gold items are intentionally NOT included -- they stay in the FLEURS
    manifest flow (:func:`build_lean_manifest`).
    """
    wd = Path(workdir)
    vi_path = wd / "raw" / "fleurs_vi_vn_test.parquet"
    en_path = wd / "raw" / "fleurs_en_us_test.parquet"
    if not (vi_path.exists() and en_path.exists()):
        raise FileNotFoundError(f"need both FLEURS parquets under {wd / 'raw'}")

    import pyarrow.parquet as pq

    def load(path: Path) -> dict[str, str]:
        table = pq.read_table(str(path), columns=["id", "transcription"])
        return {
            str(r["id"]): (r.get("transcription") or "").strip()
            for r in table.to_pylist()
        }

    vi_text = load(vi_path)
    en_text = load(en_path)
    pairs = [
        (uid, vi_text[uid], en_text[uid])
        for uid in sorted(set(vi_text) & set(en_text))
        if vi_text[uid] and en_text[uid]
    ]
    if n_items is not None and 0 < n_items < len(pairs):
        # Reuse the shared helper: the naive `round(step / 2 + i * step)` collapses
        # onto duplicate indices once n_items approaches len(pairs), emitting the
        # same utterance twice while still reporting the requested count.
        picks = _evenly_spaced_indices(len(pairs), n_items)
        pairs = [pairs[p] for p in picks]

    items = [
        EvalItem(
            id=f"vi-en-mt-{uid}",
            stage="MT",
            language="vi",
            direction="vi->en",
            input_text=vi_t,
            reference_text=en_t,
        )
        for uid, vi_t, en_t in pairs
    ]
    RunManifest(items=items).to_json(out_path)
    print(f"MT: added {len(items)} VI->EN parallel pairs -> {out_path}")


def build_lean_manifest(
    out_path: str,
    workdir: str = "eval_data",
    n_per_lang: int = 30,
    sample_rate: int = 16000,
    use_fleurs: bool = True,
    real_noise_dir: str | None = None,
) -> None:

    wd = Path(workdir)
    mixed_dir = wd / "mixed"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    items: list[EvalItem] = []
    real_noise = (
        _load_real_noise(real_noise_dir, sample_rate) if real_noise_dir else None
    )

    vi_path = wd / "raw" / "fleurs_vi_vn_test.parquet"
    en_path = wd / "raw" / "fleurs_en_us_test.parquet"
    if use_fleurs and vi_path.exists() and en_path.exists():
        vi = _load_fleurs(str(vi_path), n_per_lang, sample_rate)
        en_audio = _load_fleurs(str(en_path), n_per_lang, sample_rate)
        en_text_by_id = _load_fleurs_ids_texts(str(en_path))
        for idx, (uid, transcript, audio) in enumerate(vi):
            _emit_asr_items(
                items,
                mixed_dir,
                "vi",
                idx,
                uid,
                transcript,
                audio,
                sample_rate,
                real_noise=real_noise,
            )
            en_tr = en_text_by_id.get(uid)
            if en_tr:
                items.append(
                    EvalItem(
                        id=f"vi-en-mt-{uid}",
                        stage="MT",
                        language="vi",
                        direction="vi->en",
                        input_text=transcript,
                        reference_text=en_tr,
                    )
                )
        for idx, (uid, transcript, audio) in enumerate(en_audio):
            _emit_asr_items(
                items,
                mixed_dir,
                "en",
                idx,
                uid,
                transcript,
                audio,
                sample_rate,
                real_noise=real_noise,
            )
        print(
            f"FLEURS: added {len(vi)} VI + {len(en_audio)} EN ASR speakers "
            f"+ VI->EN MT items"
        )
    elif use_fleurs:
        print(
            f"FLEURS parquets not found in {wd / 'raw'} "
            f"-- emitting offline fallback (MT + TTS only)."
        )

    for i, (vi_text, en_text) in enumerate(GOLD_SET):
        items.append(
            EvalItem(
                id=f"gold-mt-{i:02d}",
                stage="MT",
                language="vi",
                direction="vi->en",
                input_text=vi_text,
                reference_text=en_text,
            )
        )
        items.append(
            EvalItem(
                id=f"gold-tts-{i:02d}",
                stage="TTS",
                language="en",
                direction="",
                input_text=en_text,
            )
        )

    RunManifest(items=items).to_json(out_path)
    print(
        f"Wrote {out_path} with {len(items)} items "
        f"({sum(1 for it in items if it.stage == 'ASR')} ASR, "
        f"{sum(1 for it in items if it.stage == 'MT')} MT, "
        f"{sum(1 for it in items if it.stage == 'TTS')} TTS)"
    )


def _resolve_out(out: str | None, default_name: str) -> str:
    """Honour ``--out`` in every mode, falling back to that mode's default.

    ``--vivos`` and ``--mt-only`` used to hard-code their output path under
    ``eval_data/`` and ignore ``--out`` while honouring ``--workdir``, so
    ``--workdir /tmp/x --vivos`` read from ``/tmp/x`` and wrote into the
    repository. None of the builders create the parent directory either, so a
    caller pointing at a fresh path would have hit FileNotFoundError.
    """
    path = Path(out) if out else Path("eval_data") / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Kavi bench data prep (Phase 1)")
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "manifest output path (default: eval_data/eval_manifest_v1.json; "
            "eval_data/vivos_vi_eval_manifest.json with --vivos; "
            "eval_data/mt_vi_en_eval_manifest.json with --mt-only)"
        ),
    )
    ap.add_argument("--workdir", default="eval_data")
    ap.add_argument("--n-per-lang", type=int, default=30)
    ap.add_argument(
        "--no-fleurs",
        action="store_true",
        help="skip FLEURS even if present (offline fallback only)",
    )
    ap.add_argument(
        "--noise-dir",
        help="dir with {steady,impulsive}/*.wav real noise clips (else synthetic)",
    )
    ap.add_argument(
        "--vivos",
        nargs="?",
        const=100,
        type=int,
        default=None,
        metavar="N",
        help="build a VIVOS ASR eval manifest instead (optional N items, default 100)",
    )
    ap.add_argument(
        "--vivos-mixed-dir",
        default=None,
        help=(
            "write VIVOS mixed wavs here instead of {workdir}/mixed "
            "(e.g. eval_data/vivos-mixed)"
        ),
    )
    ap.add_argument(
        "--mt-only",
        nargs="?",
        const=0,
        type=int,
        default=None,
        metavar="N",
        help=(
            "build a text-only vi->en MT manifest from ALL FLEURS parallel pairs "
            "instead (optional N to subsample deterministically)"
        ),
    )
    args = ap.parse_args()
    if args.vivos is not None:
        build_vivos_manifest(
            _resolve_out(args.out, "vivos_vi_eval_manifest.json"),
            workdir=args.workdir,
            n_items=args.vivos,
            real_noise_dir=args.noise_dir,
            mixed_dir=args.vivos_mixed_dir,
        )
        return
    if args.mt_only is not None:
        build_mt_manifest(
            _resolve_out(args.out, "mt_vi_en_eval_manifest.json"),
            workdir=args.workdir,
            n_items=(args.mt_only or None),
        )
        return
    build_lean_manifest(
        _resolve_out(args.out, "eval_manifest_v1.json"),
        workdir=args.workdir,
        n_per_lang=args.n_per_lang,
        use_fleurs=not args.no_fleurs,
        real_noise_dir=args.noise_dir,
    )


if __name__ == "__main__":
    main()
