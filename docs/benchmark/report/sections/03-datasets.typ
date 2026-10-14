// 03-datasets.typ

#let blue-header(cells) = table.header(..cells.map(c => table.cell(fill: rgb("#D6E8F7"), c)))

= Dataset Benchmark <sec:datasets>

#figure(
  kind: "table",
  supplement: [Bảng],
  table(
    columns: 4,
    align: center,
    blue-header(([Dataset], [Ngôn ngữ], [Số item], [Cấu trúc])),
    [FLEURS], [vi, en], [540 (ASR) / 347 (MT)], [30 + 30 base × 9 conditions],
    [VIVOS], [vi], [900], [100 base × 9 conditions],
    [VSS], [vi], [reference only], [chỉ dùng làm reference text],
  ),
  caption: [Tổng quan các dataset benchmark.],
)

== FLEURS

- Nguồn: google/fleurs (HuggingFace), tập test tiếng Việt + tiếng Anh.
- Split: test. 30 base utterances mỗi ngôn ngữ × 9 điều kiện SNR = 270 items/language.
- 9 conditions: clean + steady noise + impulsive noise @ 15/10/5/0 dB.
- *Đặc điểm tham chiếu:* transcript FLEURS là informal spoken-language → BLEU thường thấp hơn so với text formal (ảnh hưởng interpretation).

== VIVOS

- Nguồn: VIVOS corpus.
- Ngôn ngữ: vi. Split: test.
- 100 utterances × 9 conditions = 900 items; cùng SNR grid như FLEURS.
- Giọng đọc sạch, đọc chuẩn → dễ nhận dạng hơn FLEURS (WER thấp hơn rõ rệt ở mọi model).

== VietSuperSpeech (VSS) <sec:datasets-vss>

- Zipformer được train trên VSS.
- VSS chỉ được dùng làm reference text cho các model ASR còn lại (Moonshine, Whisper), KHÔNG dùng để eval Zipformer (tránh train/test overlap).
- *Hệ quả:* Zipformer-vi có lợi thế trên VIVOS (cùng miền giọng đọc chuẩn với VSS) — kết quả cần đọc với lưu ý này (xem @sec:appendix-limits).
