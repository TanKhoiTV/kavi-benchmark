#let d = json("../data.json")

// 07-analysis.typ

= Phân tích & Thảo luận <sec:analysis>

== Phân tích theo Chất lượng

=== ASR: Zipformer-vi dẫn đầu, nhất quán trên cả 2 dataset

- FLEURS: Zipformer-vi #d.fa.at("zipformer-vi").wer; VIVOS: #d.va.at("zipformer-vi").wer — thấp nhất ở mọi điều kiện SNR.
- VIVOS dễ hơn FLEURS rõ rệt cho mọi model (giọng đọc chuẩn, ít informal speech).
- *Lưu ý contamination:* Zipformer-vi train trên VSS — VIVOS cùng miền giọng đọc nên kết quả VIVOS có lợi thế không công bằng (xem @sec:appendix-limits).

=== Whisper-vi yếu hơn Whisper-en

- Whisper-vi WER cao hơn Whisper-en trên FLEURS (#d.fa.at("whisper-vi").wer vs #d.fa.at("whisper-en").wer) — tiếng Việt khó hơn (tone, informal).
- 3 item FLEURS whisper vượt WER 1.0 (repetition loops ở SNR thấp — hành vi thật của model, không phải lỗi pipeline).

=== Moonshine-vi: trung bình, nhạy cảm steady noise

- VIVOS: Moonshine-vi (#d.va.at("moonshine-vi").wer) tốt hơn Whisper-vi (#d.va.at("whisper-vi").wer) — nhưng FLEURS thì ngược lại.
- Steady noise 0 dB làm Moonshine-vi tăng WER mạnh nhất (FLEURS #d.fa-grid.at("moonshine-vi").steady_0).

=== NMT: M2M-100 vs HY-MT — "lexical match" vs "fluent paraphrase"

- Full-pool: M2M-100 (#d.mt.at("m2m-full").bleu) > Opus-MT (#d.mt.at("opus-full").bleu).
- Gold-set: M2M-100 (#d.mt.at("m2m-gold").bleu) > HY-MT (#d.mt.at("hymt-gold").bleu) > Opus-MT (#d.mt.at("opus-gold").bleu) — nhưng HY-MT thắng trên 12 mục gold-only (25.87 vs 23.49).
- Giải thích: FLEURS reference là informal spoken transcripts; HY-MT sinh fluent paraphrase lệch từ vựng → BLEU bị phạt. M2M-100 bám sát từ vựng hơn → BLEU cao hơn dù câu kém tự nhiên hơn.

== Phân tích theo Hiệu suất

=== Edge device (RAM < 1 GB, latency thấp)

- Zipformer-vi: VIVOS latency #d.va.at("zipformer-vi").lat s, RAM #d.va.at("zipformer-vi").ram MB — ứng viên số 1.
- Moonshine-vi: RAM #d.va.at("moonshine-vi").ram MB, latency #d.va.at("moonshine-vi").lat s (VIVOS) — chấp nhận được, WER tốt hơn Whisper.
- Whisper: RAM 923 MB + latency ~3 s — không phù hợp edge, phù hợp server/desktop.

=== Server / chất lượng tối đa

- ASR: Whisper Small (đa ngôn ngữ, WER en thấp nhất #d.fa.at("whisper-en").wer) — linh hoạt nhất.
- MT: M2M-100 (BLEU cao nhất, RAM #d.mt.at("m2m-full").ram MB) — chọn cho server.

=== Trade-off chất lượng vs tốc độ

- RTF Zipformer-vi #d.va.at("zipformer-vi").rtf (VIVOS) — $~$25× nhanh hơn real-time; Whisper-vi #d.va.at("whisper-vi").rtf — sát ngưỡng real-time, biên an toàn rất mỏng, dễ tụt xuống > 1 nếu hardware yếu hơn.
- *Lưu ý RTF:* không so sánh chéo dataset (Whisper có fixed overhead $~$3.3 s/item bất kể độ dài clip — xem benchmark-results-fleurs-vivos.md §4).

=== Giới hạn phần cứng

- CPU-only Core i7-1165G7 (Tiger Lake; có AVX-512/BF16 nhưng không có AMX) — HY-MT chạy HF eager, bf16 ~55 s/inference (#d.mt.at("hymt-gold").lat); probe riêng cho thấy fp32 nhanh hơn (~1.21 vs 1.51 s/token), dẫn đến ước tính ~44 s/item — nhưng bf16 vẫn chậm do eager kernels HF chưa tối ưu.
- Nếu chạy GPU hoặc float8/quantized kernels, HY-MT latency kỳ vọng giảm đáng kể — con số hiện tại không phản ánh chất lượng runtime thật.
