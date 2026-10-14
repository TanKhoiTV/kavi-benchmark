"""Host-side v0 runner: load manifest -> fan out candidates -> score -> table."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .adapters import Candidate
from .registry import build_candidate, default_candidate_id_for_stage
from .schema import RunManifest, StageResult
from .scorer import score_item
from .smoke_sample import build_smoke_manifest


def audio_duration(path: str | None) -> float | None:
    if path is None:
        return None
    if not Path(path).exists():
        return None
    try:
        import soundfile as sf

        with sf.SoundFile(path) as f:
            return f.frames / f.samplerate
    except Exception:  # noqa: BLE001
        return None


def run_manifest(
    manifest: RunManifest,
    out_dir: Path,
    candidate_filter: str | None,
    config_override: dict | None = None,
) -> list[dict]:
    records: list[dict] = []
    # Cache one candidate instance per cid so model weights load once per run,
    # not once per item (which would reload e.g. Whisper for every ASR clip).
    candidates: dict[str, Candidate] = {}
    for item in manifest.items:
        # Merge config override into the item BEFORE building the candidate so
        # e.g. beam_size takes effect from the first Translator construction.
        if config_override:
            item.config = {**(item.config or {}), **config_override}

        # --candidate overrides the item's candidate_id (not just filters).
        if candidate_filter is not None:
            item.candidate_id = candidate_filter
        cid = item.candidate_id
        if cid is None:
            cid = default_candidate_id_for_stage(item.stage)
        if cid is None:
            print(f"  skipped item {item.id}: no candidate resolved")
            continue
        try:
            cached = None
            if cid is not None:
                cached = candidates.get(cid)
            if cached is not None:
                candidate = cached
            else:
                candidate = build_candidate(item)
                if hasattr(candidate, "out_dir"):
                    candidate.out_dir = str(out_dir / "outputs")
                if cid is not None:
                    candidates[cid] = candidate
            result = candidate.run(item)
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not abort the run
            result = StageResult(
                candidate_id=cid,
                item_id=item.id,
                stage=item.stage,
                error=f"{type(exc).__name__}: {exc}",
            )

        duration = None
        if item.stage == "ASR":
            duration = audio_duration(item.audio_ref)
        elif item.stage == "TTS":
            duration = audio_duration(result.output_audio_path)

        metrics = score_item(item, result, duration)
        records.append(
            {
                "item": asdict(item),
                "result": asdict(result),
                "metrics": asdict(metrics),
            }
        )
    return records


def print_table(records: list[dict]) -> None:
    headers = [
        "item",
        "stage",
        "cand",
        "lat_s",
        "ram_mb",
        "WER",
        "BLEU",
        "RTF",
        "out/err",
    ]
    rows = []
    for rec in records:
        it, res, met = rec["item"], rec["result"], rec["metrics"]
        out = res.get("output_audio_path")
        if out is None:
            out = res.get("error")
        if out is None:
            out = res.get("output_text")
        if out is None:
            out = ""
        rows.append(
            [
                it["id"],
                it["stage"],
                res["candidate_id"].split("-")[0],
                f"{res['latency_s']:.2f}",
                f"{res['peak_ram_mb']:.0f}" if res.get("peak_ram_mb") else "-",
                f"{met['wer']:.3f}" if met.get("wer") is not None else "-",
                f"{met['bleu']:.1f}" if met.get("bleu") is not None else "-",
                f"{met['rtf']:.2f}" if met.get("rtf") is not None else "-",
                out[:36],
            ]
        )
    if not rows:
        print("(no items matched)")
        return
    widths = [
        max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))
    ]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths, strict=True))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        # Windows cp1252 console cannot encode Vietnamese diacritics; print the
        # table with replacement chars instead of crashing (JSON output is
        # unaffected).
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description="Kavi benchmark harness (host-side v0)")
    ap.add_argument("--manifest", help="path to eval_manifest_v1.json")
    ap.add_argument(
        "--smoke", action="store_true", help="generate + run a tiny offline sample"
    )
    ap.add_argument("--out", default="bench-results", help="output directory")
    ap.add_argument(
        "--candidate",
        help="force this candidate_id for every item (overrides the manifest)",
    )
    ap.add_argument(
        "--config-override",
        help="JSON dict merged into every item.config, e.g. '{\"beam_size\": 4}'",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        manifest = build_smoke_manifest(workdir=out_dir / "smoke-inputs")
    elif args.manifest:
        manifest = RunManifest.from_json(args.manifest)
    else:
        ap.error("either --manifest PATH or --smoke required")

    config_override: dict | None = None
    if args.config_override:
        try:
            config_override = json.loads(args.config_override)
        except json.JSONDecodeError as exc:
            ap.error(f"--config-override is not valid JSON: {exc}")
        # A non-dict (e.g. `--config-override 4`) would raise inside
        # run_manifest's merge, which sits outside the per-item try — so it
        # would abort the whole run instead of degrading one item. Reject it
        # here, where the message can name the fix.
        if not isinstance(config_override, dict):
            ap.error(
                "--config-override must be a JSON object, "
                "e.g. '{\"beam_size\": 4}' — got " + type(config_override).__name__
            )
    records = run_manifest(manifest, out_dir, args.candidate, config_override)
    (out_dir / "run_results.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_table(records)
    print(f"\nWrote {out_dir / 'run_results.json'}")


if __name__ == "__main__":
    main()
