#let d = json("../data.json")

// 02-models.typ

#let blue-header(cells) = table.header(..cells.map(c => table.cell(fill: rgb("#D6E8F7"), c)))

= Tổng quan các Model

== Neural Machine Translation

Tổng hợp 3 mô hình NMT tham gia benchmark:

#figure(
  kind: "table",
  supplement: [Bảng],
  table(
    columns: 5,
    align: center,
    blue-header(([Model], [Kiến trúc], [Params], [Checkpoint], [Runtime])),
    [Opus-MT], [encoder-decoder], [~74M], [Helsinki-NLP/opus-mt-vi-en], [CT2 int8],
    [M2M-100], [encoder-decoder], [418M], [facebook/m2m100_418M], [CT2 int8],
    [HY-MT1.5], [decoder-only], [1.8B], [tencent/HY-MT1.5-1.8B], [HF Transformers bf16],
  ),
  caption: [Tổng hợp các mô hình NMT tham gia benchmark.],
)

=== Opus-MT (Helsinki-NLP)

- Cặp ngôn ngữ cố định vi→en (v0, weights en→vi chưa có trong repo).
- Inference offline qua CTranslate2 int8, SentencePiece tokenizer.
- Beam size 5 + repetition_penalty 1.1 + no_repeat_ngram 3 (sweep phase chọn).
- Ưu điểm: nhẹ (~74M params), latency thấp, RAM $~$400 MB.
- Nhược điểm: BLEU thấp nhất trong 3 mô hình (xem @sec:results-mt).

=== M2M-100 (facebook/m2m100_418M)

- Encoder-decoder đa ngôn ngữ; cần target_prefix `en` để ép đầu ra tiếng Anh.
- Inference offline qua CTranslate2 int8, CT2 translator + SentencePiece.
- Beam size 4.
- Ưu điểm: chất lượng BLEU cao nhất full-pool; hỗ trợ đa ngôn ngữ.
- Nhược điểm: RAM ~1.1 GB (cao nhất nhóm CT2).

=== HY-MT1.5-1.8B (Tencent HunYuan) <sec:models-hymt>

- Kiến trúc: HunYuanDenseV1 (decoder-only CausalLM, không phải encoder-decoder).
- Issue \#91: CT2 không convert được (custom arch + dynamic RoPE + QK-norm).
- Runtime: HF Transformers CPU, bf16 — latency ~50 s/inference, RAM ~3.8 GB.
- Latency HY-MT không thuận lợi so với CT2-int8 (opus/m2m) — chỉ so sánh BLEU.
- Hiện chỉ có gold-set (42 items, BLEU #d.mt.at("hymt-gold").bleu); full-pool chưa chạy.

== Automatic Speech Recognition

Tổng hợp 5 mô hình ASR tham gia benchmark:

#figure(
  kind: "table",
  supplement: [Bảng],
  table(
    columns: 5,
    align: center,
    blue-header(([Model], [Kiến trúc], [Params], [Ngôn ngữ], [Runtime])),
    [Whisper Small], [encoder-decoder], [244M], [vi/en], [faster-whisper],
    [Moonshine Tiny vi], [encoder-decoder], [~27M], [vi], [HF Transformers],
    [Moonshine Tiny en], [encoder-decoder], [~27M], [en], [HF Transformers],
    [Zipformer-30M], [transducer], [~30M], [vi], [sherpa-onnx int8],
    [Zipformer-EN], [transducer], [~90M], [en], [sherpa-onnx int8],
  ),
  caption: [Tổng hợp các mô hình ASR tham gia benchmark.],
)

=== Whisper Small (faster-whisper)

- Multilingual; ngôn ngữ chỉ định per-item (vi/en), beam 2.
- Chạy qua faster-whisper (CTranslate2-backed), model_size "small".

=== Moonshine Tiny (vi / en)

- Single-language: moonshine-tiny-vi và moonshine-tiny (en), chạy qua HF Transformers.
- Beam 1 (vi) / beam 4 (en), no_repeat_ngram 3, max_length 448 (đủ cho utterance 15s).

=== Zipformer-30M (vi / en, sherpa-onnx)

- Transducer (encoder-decoder-joiner), sherpa-onnx OfflineRecognizer.
- Beam >= 2 → modified_beam_search (max_active_paths=beam); beam 1 → greedy.
- Zipformer-vi: beam 5, int8 encoder. Zipformer-en: beam 4.
- *Lưu ý:* Zipformer-vi được train trên VSS (xem @sec:datasets-vss) — tránh dùng VSS để eval model này.
