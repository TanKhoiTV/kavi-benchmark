// 01-overview.typ

= Tổng quan & Mục tiêu

== Mục tiêu

Báo cáo này đánh giá và so sánh hiệu năng của các mô hình ASR (nhận dạng tiếng nói) và NMT (dịch máy) trên nền tảng prototype KAVI — một hệ thống CPU-only chạy cục bộ trên máy tính cá nhân. Mục đích là xác định tổ hợp mô hình phù hợp nhất cho pipeline *ASR tiếng Việt → dịch sang tiếng Anh*, với tiêu chí cân bằng giữa chất lượng nhận dạng/dịch (WER, BLEU) và chi phí tính toán (latency, RAM).

Benchmark được thiết kế theo hai trục:

- *Chất lượng:* WER trên 2 dataset ASR tiếng Việt (FLEURS + VIVOS) với lưới điều kiện nhiễu 9 mức (clean + steady/impulsive noise @ 15/10/5/0 dB), và BLEU trên tập MT vi→en.
- *Hiệu suất:* latency trung bình (s), peak RAM (MB), và RTF (real-time factor) cho từng mô hình trên cùng một phần cứng.

== Phạm vi benchmark

- *NMT:* 3 candidates
  - Opus-MT
  - M2M-100
  - HY-MT1.5-1.8B (HF Transformers, latency không so sánh với CT2, xem @sec:models-hymt).
- *ASR:* 5 candidates
  - Whisper Small
  - Moonshine-vi
  - Moonshine-en
  - Zipformer-vi
  - Zipformer-en

== Tóm tắt dataset & metrics

- Dataset: FLEURS (vi/en), VIVOS (vi) — xem @sec:datasets.
- Metrics:
  - ASR: WER, CER (jiwer, lowercase normalized).
  - Metrics NMT: BLEU (mean per-item sacrebleu, case-sensitive; không phải corpus BLEU).
- Metrics hiệu suất: latency (s), peak RAM (MB), RTF.

== Câu hỏi nghiên cứu

1. Mô hình ASR nào đạt WER thấp nhất cho tiếng Việt trên FLEURS và VIVOS? Kết quả có nhất quán giữa hai dataset không?
2. Các mô hình suy giảm WER thế nào khi nhiễu nền tăng (steady vs impulsive, 15 → 0 dB)? Mô hình nào robust nhất với nhiễu?
3. Mô hình NMT nào dịch vi→en tốt nhất theo BLEU? HY-MT1.5 (1.8B) có vượt trội M2M-100 (418M) và Opus-MT không?
4. Đánh đổi giữa chất lượng và hiệu suất: mô hình nào phù hợp cho edge device (RAM < 1 GB, latency thấp), mô hình nào cho server?
5. Pipeline ASR vi → MT có khả thi chạy CPU-only trên phần cứng Core i7-1165G7 4 nhân không?
