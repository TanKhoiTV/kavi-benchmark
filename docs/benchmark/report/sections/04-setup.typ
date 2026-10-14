// 04-setup.typ

#let blue-header(cells) = table.header(..cells.map(c => table.cell(fill: rgb("#D6E8F7"), c)))

= Thiết lập thử nghiệm

== Phần cứng & Môi trường

#figure(
  kind: "table",
  supplement: [Bảng],
  table(
    columns: 2,
    align: (center, left),
    blue-header(([Thông số], [Chi tiết])),
    [CPU],     [11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz (Tiger Lake, 4C/8T)],
    [RAM],     [15.7GB DDR4],
    [GPU],     [— (CPU-only evaluation)],
    [Framework], [PyTorch 2.12.1+cpu, Transformers 4.57.6, CTranslate2 4.8.0, sherpa-onnx 1.12.40],
    [Precision], [bf16 (HF models), int8 (CT2 models)],
    [Batch size], [1 (per-item inference)],
  ),
  caption: [Thông số phần cứng và môi trường chạy benchmark.],
)

*Lưu ý:* HY-MT chạy HF Transformers eager, bf16 — đo thực tế trên CPU này cho thấy fp32 nhanh hơn bf16 (xem @sec:analysis).

== Cấu hình Benchmark

=== ASR

- Ngôn ngữ chỉ định: vi hoặc en theo item (không auto-detect).
- Decode: beam search (beam tối ưu từ sweep phase, xem @sec:setup-beam).
- Audio resample về 16 kHz.
- *Text normalization:* lowercase cả ref và hyp (jiwer case-sensitive; Zipformer output ALL-CAP). Không bỏ dấu câu, không bỏ số.

=== NMT

- Cặp ngôn ngữ: vi → en. Beam size: opus-mt - 5, m2m100 - 4, hy-mt - 5.
- Max length: 256 tokens.
- BLEU: sacrebleu default (case-sensitive, lowercase=False).

== Beam-sweep summary <sec:setup-beam>

Beam tối ưu từ sweep phase (archive `bench-results/archive/beam-sweep/`):

#figure(
  kind: "table",
  supplement: [Bảng],
  table(
    columns: 3,
    align: center,
    blue-header(([Model], [Beam tối ưu], [Ghi chú])),
    [Whisper Small], [2], [cân bằng WER/latency],
    [Moonshine-vi], [1], [greedy tốt hơn beam],
    [Moonshine-en], [4], [],
    [Zipformer-vi], [5], [modified_beam_search],
    [Zipformer-en], [4], [],
    [Opus-MT], [5], [chống repetition loop],
    [M2M-100], [4], [],
    [HY-MT], [5], [deterministic beam, so sánh BLEU],
  ),
  caption: [Beam size tối ưu từ sweep phase cho từng model.],
)