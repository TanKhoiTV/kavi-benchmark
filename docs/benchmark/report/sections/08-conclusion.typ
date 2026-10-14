#pagebreak()

#let d = json("../data.json")

// 08-conclusion.typ

= Kết luận & Khuyến nghị

== Khuyến nghị theo use-case

=== Edge device (RAM < 1 GB, latency < 1 s)

- *ASR vi:* Zipformer-vi — WER #d.va.at("zipformer-vi").wer (VIVOS) / #d.fa.at("zipformer-vi").wer (FLEURS), latency #d.va.at("zipformer-vi").lat s, RAM #d.va.at("zipformer-vi").ram MB.
- *MT:* *không ứng viên nào đạt cả hai ngưỡng* — M2M-100 đạt độ trễ #d.mt.at("m2m-full").lat s nhưng RAM #d.mt.at("m2m-full").ram MB vượt ngưỡng 1 GB; Opus-MT đạt RAM #d.mt.at("opus-full").ram MB nhưng độ trễ #d.mt.at("opus-full").lat s *vượt ngưỡng 1 s*, đổi lại BLEU #d.mt.at("opus-full").bleu thấp hơn M2M. Chọn Opus-MT khi bộ nhớ là ràng buộc cứng; nếu ưu tiên độ trễ thì chọn M2M-100 và chấp nhận bộ nhớ.

=== Server (chất lượng tối đa)

- *ASR:* Whisper Small — đa ngôn ngữ, WER en tốt nhất (#d.fa.at("whisper-en").wer).
- *MT:* M2M-100 — BLEU #d.mt.at("m2m-full").bleu full-pool; RAM #d.mt.at("m2m-full").ram MB chấp nhận được cho server.

=== Hybrid edge-server (đề xuất pipeline)

ASR vi edge (Zipformer-vi) → MT cloud/server (M2M-100). Nếu MT phải chạy on-device → Opus-MT (đủ tốt, nhẹ).

== Tổng kết

- *ASR vi:* Zipformer-vi thắng toàn diện (nhỏ, nhanh, chính xác) — nhưng cần ghi nhận contamination VSS/VIVOS. Whisper-vi là lựa chọn thay thế đa ngôn ngữ.
- *NMT vi→en:* M2M-100 tốt nhất theo BLEU; Opus-MT nhẹ nhất; HY-MT1.5 chất lượng promising (gold-only BLEU #d.hymt-gold-only-bleu) nhưng chưa deployable (issue #91, latency HF CPU).
- *Pipeline đề xuất:* Zipformer-vi (edge) → M2M-100 (server).

== Hướng phát triển

- Chạy full-pool (347 items) cho HY-MT — ước tính ~4.8–5.3 giờ (CPU-only, bf16); hoặc fp32 + 8 threads (~4 h).
- Hỗ trợ QNN/HTP on-device (cần Qualcomm hardware).
- Nếu CTranslate2 upstream hỗ trợ dynamic RoPE → HY-MT CT2 → latency giảm ~50×.
- Error analysis có hệ thống (hiện là manual sampling, tùy chọn).
