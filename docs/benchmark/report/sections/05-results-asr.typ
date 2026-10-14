#let d = json("../data.json")

// 05-results-asr.typ

#let blue-header(cells) = table.header(..cells.map(c => table.cell(fill: rgb("#D6E8F7"), c)))

// Helper: SNR table from grid data (JSON keys: clean, steady_15..0, impulsive_15..0)
#let snr-table(grid, cands, noise) = table(
  columns: 6,
  align: center,
  blue-header(([Model], [Clean], [15 dB], [10 dB], [5 dB], [0 dB])),
  ..cands.map(((label, key)) => (
    [#label], [#grid.at(key).at("clean")],
    [#grid.at(key).at(noise + "_15")],
    [#grid.at(key).at(noise + "_10")],
    [#grid.at(key).at(noise + "_5")],
    [#grid.at(key).at(noise + "_0")],
  )).flatten(),
)

= Kết quả Benchmark ASR

== Chất lượng nhận dạng (WER trung bình)

FLEURS (5 candidates × 2 ngôn ngữ):

#figure(
  kind: "table",
  supplement: [Bảng],
  table(
    columns: 6,
    align: center,
    blue-header(([Candidate], [Language], [Mean WER], [Latency (s)], [RTF], [RAM (MB)])),
    [Whisper Small], [vi], [#d.fa.at("whisper-vi").wer], [#d.fa.at("whisper-vi").lat], [#d.fa.at("whisper-vi").rtf], [#d.fa.at("whisper-vi").ram],
    [Whisper Small], [en], [#d.fa.at("whisper-en").wer], [#d.fa.at("whisper-en").lat], [#d.fa.at("whisper-en").rtf], [#d.fa.at("whisper-en").ram],
    [Moonshine Tiny], [vi], [#d.fa.at("moonshine-vi").wer], [#d.fa.at("moonshine-vi").lat], [#d.fa.at("moonshine-vi").rtf], [#d.fa.at("moonshine-vi").ram],
    [Moonshine Tiny], [en], [#d.fa.at("moonshine-en").wer], [#d.fa.at("moonshine-en").lat], [#d.fa.at("moonshine-en").rtf], [#d.fa.at("moonshine-en").ram],
    [Zipformer-30M],  [vi], [#d.fa.at("zipformer-vi").wer], [#d.fa.at("zipformer-vi").lat], [#d.fa.at("zipformer-vi").rtf], [#d.fa.at("zipformer-vi").ram],
    [Zipformer-EN],   [en], [#d.fa.at("zipformer-en").wer], [#d.fa.at("zipformer-en").lat], [#d.fa.at("zipformer-en").rtf], [#d.fa.at("zipformer-en").ram],
  ),
  caption: [WER trung bình, latency, RTF và RAM trên FLEURS (vi/en).],
)

VIVOS (3 candidates, vi):

#figure(
  kind: "table",
  supplement: [Bảng],
  table(
    columns: 5,
    align: center,
    blue-header(([Candidate], [Mean WER], [Latency (s)], [RTF], [RAM (MB)])),
    [Whisper Small],  [#d.va.at("whisper-vi").wer], [#d.va.at("whisper-vi").lat], [#d.va.at("whisper-vi").rtf], [#d.va.at("whisper-vi").ram],
    [Moonshine Tiny], [#d.va.at("moonshine-vi").wer], [#d.va.at("moonshine-vi").lat], [#d.va.at("moonshine-vi").rtf], [#d.va.at("moonshine-vi").ram],
    [Zipformer-30M],  [#d.va.at("zipformer-vi").wer], [#d.va.at("zipformer-vi").lat], [#d.va.at("zipformer-vi").rtf], [#d.va.at("zipformer-vi").ram],
  ),
  caption: [WER trung bình, latency, RTF và RAM trên VIVOS (vi).],
)

*Finding:* Zipformer-vi thấp nhất cả 2 dataset (FLEURS #d.fa.at("zipformer-vi").wer, VIVOS #d.va.at("zipformer-vi").wer); Whisper-vi cao WER trên cả 2; Moonshine-vi tốt hơn Whisper-vi trên VIVOS nhưng kém hơn trên FLEURS.

#pagebreak()

== Phân tích WER theo điều kiện nhiễu (SNR × noise type)

*Finding chính:* steady noise degrade mạnh theo SNR; impulsive noise degrade ít.
Ví dụ Moonshine-vi trên FLEURS: steady #d.fa-grid.at("moonshine-vi").steady_15 → #d.fa-grid.at("moonshine-vi").steady_0 (tăng mạnh) vs impulsive #d.fa-grid.at("moonshine-vi").impulsive_15 → #d.fa-grid.at("moonshine-vi").impulsive_0 (tăng nhẹ).

=== FLEURS — steady noise

#figure(
  kind: "table",
  supplement: [Bảng],
  snr-table(d.at("fa-grid"), (
    ("Moonshine-en", "moonshine-en"),
    ("Moonshine-vi", "moonshine-vi"),
    ("Whisper-en", "whisper-en"),
    ("Whisper-vi", "whisper-vi"),
    ("Zipformer-en", "zipformer-en"),
    ("Zipformer-vi", "zipformer-vi"),
  ), "steady"),
  caption: [FLEURS — WER theo SNR dưới steady noise.],
)

=== FLEURS — impulsive noise

#figure(
  kind: "table",
  supplement: [Bảng],
  snr-table(d.at("fa-grid"), (
    ("Moonshine-en", "moonshine-en"),
    ("Moonshine-vi", "moonshine-vi"),
    ("Whisper-en", "whisper-en"),
    ("Whisper-vi", "whisper-vi"),
    ("Zipformer-en", "zipformer-en"),
    ("Zipformer-vi", "zipformer-vi"),
  ), "impulsive"),
  caption: [FLEURS — WER theo SNR dưới impulsive noise.],
)

=== VIVOS — steady & impulsive

#figure(
  kind: "table",
  supplement: [Bảng],
  snr-table(d.at("va-grid"), (
    ("Moonshine-vi", "moonshine-vi"),
    ("Whisper-vi", "whisper-vi"),
    ("Zipformer-vi", "zipformer-vi"),
  ), "steady"),
  caption: [VIVOS — WER theo SNR dưới steady noise.],
)

#figure(
  kind: "table",
  supplement: [Bảng],
  snr-table(d.at("va-grid"), (
    ("Moonshine-vi", "moonshine-vi"),
    ("Whisper-vi", "whisper-vi"),
    ("Zipformer-vi", "zipformer-vi"),
  ), "impulsive"),
  caption: [VIVOS — WER theo SNR dưới impulsive noise.],
)

== Biểu đồ

#figure(
  image("../charts/asr-wer-vs-snr-fleurs.png", width: 100%),
  caption: [FLEURS ASR — mean WER theo SNR. Đường steady degrade mạnh; impulsive gần như nằm ngang.],
) <fig:asr-wer-snr-fleurs>

#figure(
  image("../charts/asr-wer-vs-snr-vivos.png", width: 100%),
  caption: [VIVOS ASR (vi) — mean WER theo SNR.],
) <fig:asr-wer-snr-vivos>

#figure(
  image("../charts/asr-latency-vs-wer.png", width: 90%),
  caption: [ASR quality vs speed (vi). Trục X: latency (log scale), Y = WER (thấp = tốt), bubble = peak RAM.],
) <fig:asr-latency-wer>

#figure(
  image("../charts/asr-wer-heatmap-fleurs-steady.png", width: 110%),
  caption: [FLEURS ASR — heatmap WER (candidate × condition) - steady noise.],
)

#figure(
  image("../charts/asr-wer-heatmap-fleurs-impulsive.png", width: 110%),
  caption: [FLEURS ASR - heatmap WER (candidate × condition) - impulsive noise.],
)

#figure(
  image("../charts/asr-wer-heatmap-vivos-steady.png", width: 110%),
  caption: [VIVOS ASR — heatmap WER (candidate × condition) - steady noise.],
)

#figure(
  image("../charts/asr-wer-heatmap-vivos-impulsive.png", width: 110%),
  caption: [VIVOS ASR — heatmap WER (candidate × condition) - impulsive noise.],
)