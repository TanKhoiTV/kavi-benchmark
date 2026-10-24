"""Render the 50 gold clips with their condition and SNR applied, for hand labelling.

Run: make vad-gold-audio
Writes eval_data/vad_gold_audio/<id>.wav, a render_meta.json, and (only if it does
not exist) a blank eval_data/vad_gold_labels.csv with onset_s / offset_s to fill in.
Each WAV has 0.5 s of silence before and after the clip. Label the boundaries of the
speech you hear, in seconds from the start of that WAV, without looking at the
template's prefilled values or at any VAD output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import soundfile as sf

from bench.vad_label_error import GOLD_AUDIO_DIR, GOLD_LABELS, GOLD_TEMPLATE
from bench.vad_noise import SR, build_clip_audio, load_mono

CONDITION_ASSETS = {
    "quiet": ("steady", False),
    "street": ("impulsive", False),
    "indoors": ("steady", True),
    "near-field": ("steady", True),
    "far-field": ("impulsive", True),
}


def readable(paths: list[Path]) -> list[Path]:
    ok = []
    for p in paths:
        try:
            if sf.info(str(p)).frames > 0:
                ok.append(p)
        except (RuntimeError, OSError):
            continue
    return ok


def parse_snr(text: str) -> float | None:
    if text == "clean":
        return None
    return -5.0 if text == "m5" else float(text)


def pick(files: list[Path], k: int) -> Path:
    return files[k % len(files)]


def render_row(
    row: dict,
    k: int,
    steady: list[Path],
    impulsive: list[Path],
    rirs: list[Path],
) -> tuple[object, dict]:
    kind, uses_rir = CONDITION_ASSETS[row["condition"]]
    snr = parse_snr(row["snr_db"])
    noise_path = None
    if snr is not None:
        noise_path = pick(steady if kind == "steady" else impulsive, k)
    rir_path = pick(rirs, k) if uses_rir else None
    built = build_clip_audio(
        load_mono(row["audio_path"]),
        noise=load_mono(str(noise_path)) if noise_path else None,
        rir=load_mono(str(rir_path)) if rir_path else None,
        snr_db=snr,
    )
    meta = {
        "source": row["audio_path"],
        "condition": row["condition"],
        "snr_db": snr,
        "noise": noise_path.as_posix() if noise_path else None,
        "rir": rir_path.as_posix() if rir_path else None,
        "gain": built.gain_a,
        "peak_limited": built.peak_limited,
    }
    return built, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--template", default=GOLD_TEMPLATE)
    ap.add_argument("--out-dir", default=GOLD_AUDIO_DIR)
    ap.add_argument("--labels", default=GOLD_LABELS)
    args = ap.parse_args(argv)

    found = {
        "steady": sorted(Path("assets/noise/musan/steady").glob("*.wav")),
        "impulsive": sorted(Path("assets/noise/musan/impulsive").glob("*.wav")),
        "rirs": sorted(Path("assets/noise/rirs").glob("*.wav")),
    }
    steady, impulsive, rirs = (readable(found[k]) for k in found)
    for name, usable in zip(found, (steady, impulsive, rirs), strict=True):
        skipped = len(found[name]) - len(usable)
        if skipped:
            print(f"skipped {skipped} unreadable {name} file(s)")
    if not steady or not impulsive or not rirs:
        print("real MUSAN / RIRS assets missing", file=sys.stderr)
        return 1

    with open(args.template, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    for k, row in enumerate(rows):
        built, info = render_row(row, k, steady, impulsive, rirs)
        wav = out_dir / f"{row['id']}.wav"
        sf.write(str(wav), built.audio, SR)
        info["wav"] = wav.as_posix()
        meta[row["id"]] = info
    (out_dir / "render_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    labels = Path(args.labels)
    if labels.exists():
        print(f"{labels} exists; left untouched")
    else:
        with open(labels, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["id", "audio_path", "onset_s", "offset_s", "annotator"])
            for row in rows:
                w.writerow(
                    [row["id"], (out_dir / f"{row['id']}.wav").as_posix(), "", "", ""]
                )
        print(f"wrote blank {labels}")
    print(f"rendered {len(rows)} clips to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
