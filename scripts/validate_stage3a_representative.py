"""Representative Stage 3A audio validation (one item per condition x SNR cell).

Builds each item with bench.vad_noise.build_timeline_audio, writes the WAV,
reads it back, and verifies structure, noise coverage and SNR from the file.
Run: PYTHONPATH=. uv run python scripts/validate_stage3a_representative.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from bench.vad_manifest import VadItem, VadManifest
from bench.vad_noise import (
    LEAD_SAMPLES,
    PEAK_LIMIT,
    SR,
    TARGET_P99_PEAK,
    TRAIL_SAMPLES,
    align_noise,
    build_timeline_audio,
    frame_peak_p99,
    load_mono,
    rms,
    speech_mask,
)

MANIFEST_PATH = "eval_data/vad_manifest_v2.json"
OUT_DIR = Path("/tmp/kavi_stage3a_rep")
SNR_TOL_DB = 0.05
CORR_MIN = 0.999
LEVEL_TOL = 0.10
RIR_CONDITIONS = ("indoors", "near-field", "far-field")

CASES = [
    ("vi", "quiet", None, 200),
    ("en", "quiet", 15.0, 350),
    ("vi", "street", 10.0, 500),
    ("en", "street", 5.0, 700),
    ("vi", "indoors", 10.0, 200),
    ("en", "indoors", 5.0, 350),
    ("vi", "near-field", 5.0, 500),
    ("en", "near-field", 0.0, 700),
    ("vi", "far-field", 0.0, 200),
    ("en", "far-field", -5.0, 500),
]


def find_item(m: VadManifest, lang: str, cond: str, snr, pause: int) -> VadItem:
    for it in m.items:
        if (
            it.language == lang
            and it.condition == cond
            and it.snr_db == snr
            and it.pause_ms == pause
            and it.id.endswith("-000")
        ):
            return it
    raise LookupError(f"no manifest item for {lang} {cond} {snr} {pause}")


def _bounds(value) -> tuple[int, int] | None:
    return None if value is None else (int(value[0]), int(value[1]))


def check_item(item: VadItem) -> tuple[list[str], dict]:
    fails: list[str] = []

    expect_noise = item.noise_type != "clean"
    expect_rir = item.condition in RIR_CONDITIONS
    if expect_noise != (item.noise_path is not None):
        fails.append("noise_path presence does not match noise_type")
    if expect_rir != (item.rir_path is not None):
        fails.append("rir_path presence does not match condition")
    if item.noise_path and "musan" not in item.noise_path:
        fails.append(f"noise_path is not MUSAN: {item.noise_path}")
    if item.rir_path and "rirs" not in item.rir_path:
        fails.append(f"rir_path is not RIRS: {item.rir_path}")
    if item.speech_b_path is None:
        raise ValueError(f"speech_b_path missing for {item.id}")

    a = load_mono(item.clean_audio_path)
    b = load_mono(item.speech_b_path)
    noise = load_mono(item.noise_path) if item.noise_path else None
    rir = load_mono(item.rir_path) if item.rir_path else None

    if item.speech_a_bounds is None or item.speech_b_bounds is None:
        fails.append("item has no source bounds; regenerate the manifest")
    bounds_a = _bounds(item.speech_a_bounds)
    bounds_b = _bounds(item.speech_b_bounds)

    built = build_timeline_audio(
        a,
        b,
        item.pause_ms,
        noise=noise,
        rir=rir,
        snr_db=item.snr_db,
        bounds_a=bounds_a,
        bounds_b=bounds_b,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{item.id}.wav"
    sf.write(str(out), built.audio, SR)
    wav, sr = sf.read(str(out), dtype="float64")

    pause = int(item.pause_ms * 16)
    (a0, a1), (b0, b1) = built.gt_segments
    la = bounds_a[1] - bounds_a[0] if bounds_a else len(a)
    lb = bounds_b[1] - bounds_b[0] if bounds_b else len(b)
    total = LEAD_SAMPLES + la + pause + lb + TRAIL_SAMPLES

    if sr != SR:
        fails.append(f"sample rate {sr}")
    if len(wav) != total:
        fails.append(f"length {len(wav)} != {total}")
    if [tuple(s) for s in item.ground_truth_segments] != built.gt_segments:
        fails.append("manifest GT != construction GT")
    if a0 != LEAD_SAMPLES or b0 - a1 != pause:
        fails.append("GT start/gap wrong")
    if float(np.max(np.abs(wav))) > PEAK_LIMIT + 1e-4:
        fails.append("peak above limit")

    level = frame_peak_p99(built.reference[speech_mask(len(wav), built.gt_segments)])
    if not (built.gain_a > 0 and built.gain_b > 0):
        fails.append("non-positive level gain")
    level_off = abs(level / TARGET_P99_PEAK - 1.0) > LEVEL_TOL
    if level_off and not expect_rir and not built.peak_limited:
        fails.append(f"speech level p99 {level:.3f} != {TARGET_P99_PEAK}")

    lead = wav[:LEAD_SAMPLES]
    gap = wav[a1:b0]
    trail = wav[-TRAIL_SAMPLES:]
    snr_wav = None
    err = None
    corr = None

    if not expect_noise:
        if float(np.max(np.abs(wav - built.reference))) > 2e-4:
            fails.append("clean item differs from reference")
        if max(rms(lead), rms(gap), rms(trail)) != 0.0:
            fails.append("clean item has energy in silence")
    else:
        residual = wav - built.reference
        mask = speech_mask(len(wav), built.gt_segments)
        snr_wav = float(20 * np.log10(rms(built.reference[mask]) / rms(residual)))
        err = snr_wav - float(item.snr_db)
        ref_noise = np.asarray(align_noise(noise, len(wav)), dtype=np.float64)
        corr = float(np.corrcoef(residual, ref_noise)[0, 1])
        if abs(err) > SNR_TOL_DB:
            fails.append(f"SNR error {err:.4f} dB")
        if corr < CORR_MIN:
            fails.append(f"whole-timeline noise correlation {corr:.5f}")

    if expect_rir and float(np.max(np.abs(built.reference[a1:b0]))) == 0.0:
        fails.append("no reverb tail in pause")

    info = {
        "snr_wav": snr_wav,
        "err": err,
        "corr": corr,
        "peak_limited": built.peak_limited,
        "level": level,
        "out": str(out),
    }
    return fails, info


def main() -> int:
    m = VadManifest.from_json(MANIFEST_PATH)
    bad = 0
    for lang, cond, snr, pause in CASES:
        item = find_item(m, lang, cond, snr, pause)
        fails, info = check_item(item)
        status = "FAIL" if fails else "PASS"
        bad += bool(fails)
        err = "n/a" if info["err"] is None else f"{info['err']:+.5f}"
        corr = "n/a" if info["corr"] is None else f"{info['corr']:.6f}"
        print(
            f"{status} {item.id} | target={item.snr_db} | err_dB={err} | "
            f"corr={corr} | lvl_p99={info['level']:.3f} | "
            f"peak_limited={info['peak_limited']} | {info['out']}"
        )
        for f in fails:
            print(f"    - {f}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} cases PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
