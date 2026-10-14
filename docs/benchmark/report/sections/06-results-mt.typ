#pagebreak()

#let d = json("../data.json")

// 06-results-mt.typ

#let blue-header(cells) = table.header(..cells.map(c => table.cell(fill: rgb("#D6E8F7"), c)))

= Kết quả Benchmark NMT <sec:results-mt>

== Chất lượng dịch (BLEU)

*Lưu ý:* HY-MT chỉ có gold-set (42 items, HF CPU ~55 s/inference); opus/m2m có full-pool (347 items, CT2-int8 ~1 s). Chỉ so sánh BLEU; latency không comparable.

=== Full-pool (347 items)

#figure(
  kind: "table",
  supplement: [Bảng],
  table(
    columns: 5,
    align: center,
    blue-header(([Candidate], [Mean BLEU], [Median], [Min], [Max])),
    [Opus-MT],     [#d.mt.at("opus-full").bleu], [#d.mt.at("opus-full").med], [#d.mt.at("opus-full").min], [#d.mt.at("opus-full").max],
    [M2M-100],     [#d.mt.at("m2m-full").bleu],  [#d.mt.at("m2m-full").med],  [#d.mt.at("m2m-full").min],  [#d.mt.at("m2m-full").max],
  ),
  caption: [BLEU trên full-pool (347 items).],
)

=== Gold-set (42 items, gồm 12 mục gold-mt)

#figure(
  kind: "table",
  supplement: [Bảng],
  table(
    columns: 5,
    align: center,
    blue-header(([Candidate], [Mean BLEU], [Median], [Min], [Max])),
    [Opus-MT],   [#d.mt.at("opus-gold").bleu], [#d.mt.at("opus-gold").med], [#d.mt.at("opus-gold").min], [#d.mt.at("opus-gold").max],
    [M2M-100],   [#d.mt.at("m2m-gold").bleu],  [#d.mt.at("m2m-gold").med],  [#d.mt.at("m2m-gold").min],  [#d.mt.at("m2m-gold").max],
    [HY-MT1.5],  [#d.mt.at("hymt-gold").bleu], [#d.mt.at("hymt-gold").med], [#d.mt.at("hymt-gold").min], [#d.mt.at("hymt-gold").max],
  ),
  caption: [BLEU trên gold-set (42 items, gồm 12 mục gold-mt).],
)

Trên 12 mục gold-only (không tính 30 mục vi-en-mt còn lại): HY-MT #d.hymt-gold-only-bleu · M2M-100 #d.m2m-gold-only-bleu · Opus-MT #d.opus-gold-only-bleu — HY-MT dẫn đầu.

== Hiệu suất tính toán

#figure(
  kind: "table",
  supplement: [Bảng],
  table(
    columns: 4,
    align: center,
    blue-header(([Candidate], [Mean latency (s)], [Peak RAM (MB)], [Runtime])),
    [Opus-MT],  [#d.mt.at("opus-full").lat], [#d.mt.at("opus-full").ram], [CT2 int8],
    [M2M-100],  [#d.mt.at("m2m-full").lat],  [#d.mt.at("m2m-full").ram],  [CT2 int8],
    [HY-MT1.5], [#d.mt.at("hymt-gold").lat], [#d.mt.at("hymt-gold").ram], [HF bf16],
  ),
  caption: [Hiệu suất tính toán của các mô hình NMT.],
)

*Cảnh báo:* HY-MT chạy HF Transformers eager, bf16 — latency ~55 s/inference gấp ~50 lần CT2 (một phần do eager inference; bf16 đo chậm hơn fp32 trên CPU này, xem @sec:analysis). Con số chỉ mang tính tham chiếu; BLEU mới là metric so sánh được (xem @sec:models-hymt).

== Biểu đồ

#figure(
  image("../charts/mt-bleu-vs-latency.png", width: 100%),
  caption: [FLEURS MT vi→en — BLEU vs latency (log scale), bubble = peak RAM.],
) <fig:mt-bleu-latency>
