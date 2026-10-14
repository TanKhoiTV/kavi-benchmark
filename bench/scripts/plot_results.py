"""Plot benchmark results from docs/benchmark/raw-results/*.csv.

Produces PNG charts into docs/benchmark/report/charts/:
  1. asr-wer-vs-snr-fleurs.png  — per-candidate WER degradation lines (FLEURS)
  2. asr-wer-vs-snr-vivos.png   — per-candidate WER degradation lines (VIVOS)
  3. asr-latency-vs-wer.png     — latency (log) vs WER bubble scatter, vi only
  4. mt-bleu-vs-latency.png     — BLEU vs latency (log) bubble scatter (MT)
  5. asr-wer-heatmap-fleurs.png — candidate x condition WER heatmap (FLEURS)
  6. asr-wer-heatmap-vivos.png  — candidate x condition WER heatmap (VIVOS)

Reads the same cleaned CSVs as bench/scripts/aggregate_results.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Anchored to the repo root so the script reads and writes the same place from
# any working directory. OUT is the directory the report actually reads: the
# .typ sections reference image("../charts/…") from report/sections/, which
# resolves to report/charts/. Writing to docs/benchmark/charts produced a tree
# nothing referenced, so the checked-in charts could go stale unnoticed.
DOCS = Path(__file__).resolve().parents[2] / "docs" / "benchmark"
RAW = DOCS / "raw-results"
OUT = DOCS / "report" / "charts"

# Display labels for the x-axis / heatmap columns.
SNR_TICKS = ["clean", "15", "10", "5", "0 dB"]

# Lookup keys corresponding 1:1 with SNR_TICKS, decoupled from display text.
# None = clean condition; otherwise the numeric SNR value in dB.
SNR_KEYS: list[float | None] = [None, 15.0, 10.0, 5.0, 0.0]


def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float(x) if x not in ("", None) else None


def short_label(cid: str) -> str:
    """Candidate id -> short display label."""
    return (
        cid.replace("whisper-small-faster-whisper-cpu", "whisper-small")
        .replace("moonshine-tiny-vi-hf-cpu", "moonshine-vi")
        .replace("moonshine-tiny-en-hf-cpu", "moonshine-en")
        .replace("zipformer-vi-30m-sherpa-onnx-cpu", "zipformer-vi")
        .replace("zipformer-en-sherpa-onnx-cpu", "zipformer-en")
        .replace("hy-mt1.5-1.8b-hf-cpu", "hy-mt")
        .replace("m2m100-vi-en-ct2-cpu", "m2m100")
        .replace("opus-mt-vi-en-ct2-cpu", "opus-mt")
    )


def _candidate_of(run: str) -> str:
    return run.split("/")[-1]


def _grid(rows, lang: str | None):
    """{(noise_type, snr_key): mean WER} for one candidate x language.

    snr_key is None for clean, else the float SNR value (15.0/10.0/5.0/0.0).
    Averages over all matching items instead of keeping only the last one.
    """
    buckets: dict[tuple[str, float | None], list[float]] = {}
    for r in rows:
        if lang is not None and r["language"] != lang:
            continue
        w = fnum(r["wer"])
        if w is None:
            continue
        snr = fnum(r["snr"])
        key = (r["noise_type"], snr)
        buckets.setdefault(key, []).append(w)
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def _line_chart(ax, grid, title: str) -> None:
    """Steady + impulsive WER lines over clean/15/10/5/0 dB with clean marker."""
    x = np.arange(len(SNR_TICKS))
    for noise, color, marker in (
        ("steady", "tab:red", "o"),
        ("impulsive", "tab:blue", "s"),
    ):
        ys = [grid.get((noise, k)) for k in SNR_KEYS[1:]]
        if any(y is not None for y in ys):
            ax.plot(x[1:], ys, color=color, marker=marker, label=noise, linewidth=1.6)
    clean = grid.get(("clean", None))
    if clean is not None:
        ax.scatter(
            [x[0]], [clean], color="black", marker="*", s=110, label="clean", zorder=5
        )
    ax.set_xticks(x, SNR_TICKS)
    ax.set_ylabel("mean WER")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def chart_asr_lines(fleurs: list[dict], vivos: list[dict]) -> None:
    fleurs_rows = {
        c: [r for r in fleurs if _candidate_of(r["run"]) == c]
        for c in sorted({_candidate_of(r["run"]) for r in fleurs})
    }
    # subplot grid: one cell per (candidate, language)
    cells = [
        (c, lg)
        for c, rows in fleurs_rows.items()
        for lg in sorted({r["language"] for r in rows})
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fig.suptitle("FLEURS ASR — mean WER by condition", fontsize=13)
    for ax, (c, lg) in zip(axes.flat, cells, strict=False):
        _line_chart(ax, _grid(fleurs_rows[c], lg), f"{short_label(c)} · {lg.upper()}")
    for ax in axes.flat[len(cells) :]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "asr-wer-vs-snr-fleurs.png", dpi=150)
    plt.close(fig)

    vivos_rows = {
        c: [r for r in vivos if _candidate_of(r["run"]) == c]
        for c in sorted({_candidate_of(r["run"]) for r in vivos})
    }
    cells = sorted(vivos_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle("VIVOS ASR (vi) — mean WER by condition", fontsize=13)
    for ax, c in zip(axes, cells, strict=False):
        _line_chart(ax, _grid(vivos_rows[c], "vi"), short_label(c))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "asr-wer-vs-snr-vivos.png", dpi=150)
    plt.close(fig)


def chart_latency_wer(fleurs: list[dict], vivos: list[dict]) -> None:
    def points(rows, lang):
        agg = {}
        for r in rows:
            if r["language"] != lang:
                continue
            c = _candidate_of(r["run"])
            w, la, rm = fnum(r["wer"]), fnum(r["latency_s"]), fnum(r["peak_ram_mb"])
            if w is None or la is None:
                continue
            a = agg.setdefault(c, [[], [], []])
            a[0].append(w)
            a[1].append(la)
            a[2].append(rm)
        return {
            c: (np.mean(v[0]), np.mean(v[1]), np.mean(v[2])) for c, v in agg.items()
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ASR quality vs speed (vi; bubble = peak RAM)", fontsize=13)
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for ax, rows, label in (
        (axes[0], fleurs, "FLEURS"),
        (axes[1], vivos, "VIVOS"),
    ):
        pts = points(rows, "vi")
        for i, (c, (w, la, rm)) in enumerate(sorted(pts.items())):
            ax.scatter(
                la,
                w,
                s=max(rm / 8, 60),
                color=colors[i % 8],
                alpha=0.75,
                edgecolors="black",
                linewidths=0.8,
                label=short_label(c),
            )
        ax.set_xscale("log")
        ax.set_xlabel("mean latency (s, log)")
        ax.set_ylabel("mean WER")
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "asr-latency-vs-wer.png", dpi=150)
    plt.close(fig)


def chart_mt(fleurs_mt: list[dict]) -> None:
    runs = {}
    for r in fleurs_mt:
        b, la, rm = fnum(r["bleu"]), fnum(r["latency_s"]), fnum(r["peak_ram_mb"])
        if b is None or la is None:
            continue
        a = runs.setdefault(r["run"], [[], [], []])
        a[0].append(b)
        a[1].append(la)
        a[2].append(rm)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i, (run, (bs, las, rms)) in enumerate(sorted(runs.items())):
        ax.scatter(
            np.mean(las),
            np.mean(bs),
            s=max(np.mean(rms) / 8, 60),
            color=colors[i % 8],
            alpha=0.75,
            edgecolors="black",
            linewidths=0.8,
        )
        ax.annotate(
            run.replace("fleurs-mt/", "").replace("100-vi-en-ct2-cpu", "100"),
            (np.mean(las), np.mean(bs)),
            fontsize=8,
            xytext=(6, 6),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("mean latency (s, log)")
    ax.set_ylabel("mean BLEU (sacrebleu)")
    ax.set_title("FLEURS MT vi→en — BLEU vs latency (bubble = peak RAM)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "mt-bleu-vs-latency.png", dpi=150)
    plt.close(fig)


def _heatmap_rows_for_noise(rows, cand_lang_pairs, noise: str):
    """Build (row_label, [wer per SNR_TICKS column]) for one noise type
    (steady or impulsive) across candidate/language pairs. The 'clean'
    column reuses the clean baseline (which has no noise type)."""
    out = []
    for c, lg in cand_lang_pairs:
        c_rows = [r for r in rows if _candidate_of(r["run"]) == c]
        g = _grid(c_rows, lg)
        clean = g.get(("clean", None))
        row_label = short_label(c) if lg is None else f"{short_label(c)} · {lg.upper()}"
        vals = [clean] + [g.get((noise, k)) for k in SNR_KEYS[1:]]
        out.append((row_label, vals))
    return out


def chart_heatmap(rows, out_stub: str, title_prefix: str, cand_lang_pairs) -> None:
    """cand_lang_pairs: list of (candidate_id, language) tuples to include.
    Passing the language explicitly (instead of None) prevents mixing vi/en
    items for candidates like whisper-small that share one id across
    languages.

    Produces two separate heatmap images — one for steady noise, one for
    impulsive — each with its own color scale, since impulsive WER varies
    on a much narrower range than steady and gets visually flattened when
    both share one colorbar.
    """
    for noise in ("steady", "impulsive"):
        heat_rows = _heatmap_rows_for_noise(rows, cand_lang_pairs, noise)
        labels = [lbl for lbl, _ in heat_rows]
        mat = np.array(
            [[np.nan if v is None else v for v in vals] for _, vals in heat_rows]
        )

        fig, ax = plt.subplots(figsize=(9, 0.5 * len(labels) + 2))
        vmax = np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else 0.6
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(0.1, vmax))
        ax.set_xticks(range(len(SNR_TICKS)), SNR_TICKS)
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(SNR_TICKS)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{title_prefix} — {noise} noise")
        fig.colorbar(im, ax=ax, label="mean WER")
        fig.tight_layout()
        fig.savefig(OUT / f"{out_stub}-{noise}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fleurs = load_csv(RAW / "fleurs-asr.csv")
    fleurs_mt = load_csv(RAW / "fleurs-mt.csv")
    vivos = load_csv(RAW / "vivos.csv")

    chart_asr_lines(fleurs, vivos)
    chart_latency_wer(fleurs, vivos)
    chart_mt(fleurs_mt)

    fleurs_pairs = sorted({(_candidate_of(r["run"]), r["language"]) for r in fleurs})
    chart_heatmap(
        fleurs,
        "asr-wer-heatmap-fleurs",
        "FLEURS ASR — mean WER (candidate x condition)",
        fleurs_pairs,
    )

    vivos_cands = sorted({_candidate_of(r["run"]) for r in vivos})
    vivos_pairs = [(c, "vi") for c in vivos_cands]
    chart_heatmap(
        vivos,
        "asr-wer-heatmap-vivos",
        "VIVOS ASR (vi) — mean WER (candidate x condition)",
        vivos_pairs,
    )
    print("charts written to", OUT)


if __name__ == "__main__":
    main()
