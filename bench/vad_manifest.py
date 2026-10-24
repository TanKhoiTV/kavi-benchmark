"""VAD manifest schema and builder (separate from eval_manifest_v1.json contract).

Ground truth spans the reference-VAD speech bounds of each FLEURS clip
(bench/vad_source_bounds.json), not the whole clip. The builder also writes the
gold-template CSV for manual labelling and stops if FLEURS audio, real noise
assets or the source-bounds file are unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from bench.vad_reference import SOURCE_BOUNDS_PATH, load_source_bounds

MANIFEST_VERSION = "vad_manifest_v2"

CONDITIONS = [
    "quiet",
    "street",
    "indoors",
    "near-field",
    "far-field",
]

PAUSE_LENGTHS_MS = [200, 350, 500, 700]

CONDITION_SNR_LEVELS = {
    "quiet": [None, 15.0],
    "street": [10.0, 5.0],
    "indoors": [10.0, 5.0],
    "near-field": [5.0, 0.0],
    "far-field": [0.0, -5.0],
}


@dataclass
class VadItem:
    id: str
    language: str  # "vi" | "en"
    condition: str
    clean_audio_path: str
    noise_type: str  # "clean" | "steady" | "impulsive"
    noise_path: str | None = None
    snr_db: float | None = None
    rir_path: str | None = None
    speech_b_path: str | None = None
    pause_ms: int = 500
    ground_truth_segments: list[tuple[int, int]] = field(default_factory=list)
    energy_threshold: float = 0.05
    speech_timeout_ms: int = 500
    speech_a_bounds: tuple[int, int] | None = None
    speech_b_bounds: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.language not in {"vi", "en"}:
            raise ValueError(f"Invalid language: {self.language}")

        if self.condition not in CONDITIONS:
            raise ValueError(f"Invalid condition: {self.condition}")

        if self.noise_type not in {"clean", "steady", "impulsive"}:
            raise ValueError(f"Invalid noise_type: {self.noise_type}")

        if self.pause_ms not in PAUSE_LENGTHS_MS:
            raise ValueError(f"Invalid pause_ms: {self.pause_ms}")

        allowed_snr = CONDITION_SNR_LEVELS[self.condition]
        if self.snr_db not in allowed_snr:
            raise ValueError(
                f"SNR {self.snr_db} is not valid for condition {self.condition}"
            )


@dataclass
class VadManifest:
    version: str = MANIFEST_VERSION
    items: list[VadItem] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {"version": self.version, "items": [asdict(i) for i in self.items]}
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> VadManifest:
        import json

        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        items = [VadItem(**i) for i in data.get("items", [])]
        return cls(version=data.get("version", MANIFEST_VERSION), items=items)


def build_vad_manifest(
    out_path: str = "eval_data/vad_manifest_v2.json",
    workdir: str = "eval_data",
    n_per_lang: int = 10,
    use_fleurs: bool = True,
    real_noise_dir: str | None = None,
    seed: int = 42,
    bounds_path: str = SOURCE_BOUNDS_PATH,
) -> VadManifest:
    """Build VAD manifest v2 (800 baseline benchmark items)."""
    # Asset validation
    musan_steady = sorted(Path("assets/noise/musan/steady").glob("*.wav"))
    musan_impulsive = sorted(Path("assets/noise/musan/impulsive").glob("*.wav"))
    # Filter to readable WAV RIRs only (bad format files excluded)
    rirs_all = sorted(Path("assets/noise/rirs").glob("*.wav"))
    import soundfile as sf

    rirs_files_readable = []
    for p in rirs_all:
        try:
            info = sf.info(str(p))
            if info.frames > 0:
                rirs_files_readable.append(p)
        except Exception:
            continue
    rirs_files = sorted(rirs_files_readable)
    if not musan_steady or not musan_impulsive or not rirs_files:
        raise FileNotFoundError(
            "Real MUSAN/RIRS assets missing for VAD (no synthetic substitute)."
        )

    import random

    import soundfile as sf

    rng = random.Random(seed)

    vi_dir = Path(workdir) / "vad_audio" / "fleurs" / "vi"
    en_dir = Path(workdir) / "vad_audio" / "fleurs" / "en"
    vi_all = sorted(vi_dir.glob("*.wav"))
    en_all = sorted(en_dir.glob("*.wav"))
    if len(vi_all) < n_per_lang or len(en_all) < n_per_lang:
        raise FileNotFoundError(f"Need >= {n_per_lang} WAVs per language")
    vi_sel = sorted(rng.sample(vi_all, n_per_lang), key=lambda p: str(p))
    en_sel = sorted(rng.sample(en_all, n_per_lang), key=lambda p: str(p))

    bounds = load_source_bounds(bounds_path)
    missing = [p.as_posix() for p in vi_sel + en_sel if p.as_posix() not in bounds]
    if missing:
        raise FileNotFoundError(
            f"No reference speech bounds in {bounds_path} for: {missing}; "
            "run `make vad-bounds`"
        )

    leading_samples = 16000  # 1000 ms

    def make_gt(src_a, src_b, pause_ms):
        a_samples = bounds[src_a.as_posix()][1] - bounds[src_a.as_posix()][0]
        b_samples = bounds[src_b.as_posix()][1] - bounds[src_b.as_posix()][0]
        pause_samples = int(pause_ms * 16)
        a_start = leading_samples
        a_end = a_start + a_samples
        b_start = a_end + pause_samples
        b_end = b_start + b_samples
        return [(a_start, a_end), (b_start, b_end)]

    items = []
    for lang, pool in [("vi", vi_sel), ("en", en_sel)]:
        for cond in CONDITIONS:
            snrs = CONDITION_SNR_LEVELS[cond]
            for snr in snrs:
                for pause_ms in PAUSE_LENGTHS_MS:
                    for i in range(n_per_lang):
                        src_a = pool[i]
                        src_b = pool[(i + 1) % len(pool)]
                        str(src_a)

                        if cond == "quiet" and snr is None:
                            noise_type = "clean"
                            noise_path = None
                            rir_path = None
                        elif cond == "quiet" and snr == 15.0:
                            noise_type = "steady"
                            noise_path = str(musan_steady[i % len(musan_steady)])
                            rir_path = None
                        elif cond == "street" and snr in (10.0, 5.0):
                            noise_type = "impulsive"
                            noise_path = str(musan_impulsive[i % len(musan_impulsive)])
                            rir_path = None
                        elif (
                            cond == "indoors"
                            and snr in (10.0, 5.0)
                            or cond == "near-field"
                            and snr in (5.0, 0.0)
                        ):
                            noise_type = "steady"
                            noise_path = str(musan_steady[i % len(musan_steady)])
                            rir_path = str(rirs_files[i % len(rirs_files)])
                        elif cond == "far-field" and snr in (0.0, -5.0):
                            noise_type = "impulsive"
                            noise_path = str(musan_impulsive[i % len(musan_impulsive)])
                            rir_path = str(rirs_files[i % len(rirs_files)])
                        else:
                            raise ValueError(f"Unexpected: {cond} {snr}")

                        if snr is None:
                            snr_part = "clean"
                        else:
                            snr_part = "m5" if snr == -5.0 else f"{int(snr)}"
                        item_id = f"{lang}-{cond}-{snr_part}-{pause_ms}-{i:03d}"
                        gt = make_gt(src_a, src_b, pause_ms)
                        items.append(
                            VadItem(
                                id=item_id,
                                language=lang,
                                condition=cond,
                                clean_audio_path=str(src_a),
                                noise_type=noise_type,
                                noise_path=noise_path,
                                snr_db=snr,
                                rir_path=rir_path,
                                speech_b_path=str(src_b),
                                pause_ms=pause_ms,
                                ground_truth_segments=gt,
                                speech_a_bounds=bounds[src_a.as_posix()],
                                speech_b_bounds=bounds[src_b.as_posix()],
                            )
                        )
    manifest = VadManifest(items=items)
    manifest.to_json(out_path)
    return manifest


def build_gold_template(
    path: str = "eval_data/vad_gold_template.csv", seed: int = 42
) -> None:
    import random
    from collections import Counter

    import soundfile as sf

    # Grid: 5 conditions x 2 SNR = 10 cells
    # Need 50 rows: 25 VI + 25 EN, >=2 per cell
    # Allocation: 10 cells x 2 (VI+EN) = 20; + 5 cells x 3 (VI+EN+extra) = 30 => wait need exactly 50
    # Simpler: 10 cells x 5 rows = 50 (3 VI + 2 EN or similar balanced)
    # For exact balance 25/25: 5 cells get 3 VI + 2 EN, 5 cells get 2 VI + 3 EN -> 25/25
    cells = [
        ("quiet", None),
        ("quiet", 15.0),
        ("street", 10.0),
        ("street", 5.0),
        ("indoors", 10.0),
        ("indoors", 5.0),
        ("near-field", 5.0),
        ("near-field", 0.0),
        ("far-field", 0.0),
        ("far-field", -5.0),
    ]
    # Balanced allocation: odd index cells get 3 VI / 2 EN; even get 2 VI / 3 EN
    # Total VI = 5*3 + 5*2 = 25; EN = 5*2 + 5*3 = 25
    vi_dir = Path("eval_data/vad_audio/fleurs/vi")
    en_dir = Path("eval_data/vad_audio/fleurs/en")
    vi_all = sorted(vi_dir.glob("*.wav"))
    en_all = sorted(en_dir.glob("*.wav"))
    if len(vi_all) < 10 or len(en_all) < 10:
        raise FileNotFoundError("Need >=10 WAVs per lang for gold template")
    rng = random.Random(seed)
    vi_sel = sorted(rng.sample(vi_all, 10), key=lambda p: str(p))
    en_sel = sorted(rng.sample(en_all, 10), key=lambda p: str(p))

    # Noise / RIR mapping per condition for reference paths (not required in CSV but useful for validation)
    sorted(Path("assets/noise/musan/steady").glob("*.wav"))
    sorted(Path("assets/noise/musan/impulsive").glob("*.wav"))
    sorted(Path("assets/noise/rirs").glob("*.wav"))

    rows = []
    row_id = 1
    for idx, (cond, snr) in enumerate(cells):
        vi_count = 3 if idx % 2 == 1 else 2
        en_count = 2 if idx % 2 == 1 else 3
        # Pick deterministic source files from selection
        vi_pool = vi_sel
        en_pool = en_sel
        # We need to pick specific files; rotate deterministically
        vi_picks = [vi_pool[(row_id + k) % len(vi_pool)] for k in range(vi_count)]
        en_picks = [en_pool[(row_id + k) % len(en_pool)] for k in range(en_count)]
        for src in vi_picks + en_picks:
            lang = "vi" if src in vi_pool else "en"
            info = sf.info(str(src))
            onset = 16000  # leading silence
            offset = 16000 + info.frames
            snr_str = (
                "clean" if snr is None else ("m5" if snr == -5.0 else f"{int(snr)}")
            )
            rows.append(
                {
                    "id": f"gold-{row_id:03d}",
                    "audio_path": str(src),
                    "onset_sample": onset,
                    "offset_sample": offset,
                    "condition": cond,
                    "snr_db": snr_str,
                    "annotator": "stage3a-baseline",
                    "language": lang,
                }
            )
            row_id += 1

    # Ensure exactly 50 rows
    if len(rows) != 50:
        raise ValueError(f"Expected 50 gold rows, got {len(rows)}")

    # Coverage check >= 2 per cell
    Counter(
        r["condition"] + ("_" + r["snr_db"] if r["snr_db"] != "clean" else "_clean")
        for r in rows
    )
    # More precise: per condition+snr
    cell_counts = Counter((r["condition"], r["snr_db"]) for r in rows)
    for (cond, snr_str), count in cell_counts.items():
        if count < 2:
            raise ValueError(f"Gold coverage <2 for {cond}/{snr_str}: {count}")

    # Balance check
    vi_rows = [r for r in rows if r["language"] == "vi"]
    en_rows = [r for r in rows if r["language"] == "en"]
    if len(vi_rows) != 25 or len(en_rows) != 25:
        raise ValueError(
            f"Gold must be 25 VI / 25 EN; got {len(vi_rows)}/{len(en_rows)}"
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("id,audio_path,onset_sample,offset_sample,condition,snr_db,annotator\n")
        for r in rows:
            f.write(
                f"{r['id']},{r['audio_path']},{r['onset_sample']},{r['offset_sample']},{r['condition']},{r['snr_db']},{r['annotator']}\n"
            )
