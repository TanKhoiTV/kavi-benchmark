# Dàn ý báo cáo Benchmark

## 1. Tổng quan & Mục tiêu

- Mục tiêu: đánh giá và so sánh hiệu năng các model NMT và ASR trên dữ liệu đã chọn
- Phạm vi benchmark:
    + NMT: Opus-MT, M2M-100, HY-MT1.5-1.8B *(HF Transformers — latency không so sánh được với CT2 candidates; xem issue #91)*
    + ASR: Whisper Small, Moonshine-vi, Moonshine-en, Zipformer-vi, Zipformer-en *(5 candidates)*
- Tóm tắt nhanh: dataset sử dụng, metrics đánh giá
- Câu hỏi nghiên cứu / tiêu chí ưu tiên mà nhóm đã lựa chọn

## 2. Tổng quan các Model

### 2.1 Neural Machine Translation

(bảng các model NMT thêm ở đây)

Nội dung cần viết cho mỗi model:

- Kiến trúc, số tham số (parameters).
- Phiên bản / checkpoint cụ thể dùng trong benchmark.
- Cặp ngôn ngữ benchmark (vd: vi → en, en → vi, hay vi → fr…).
- Có fine-tune không, hay chỉ inference trực tiếp (zero-shot)?

#### 2.1.1 Opus-MT (Helsinki-NLP)

#### 2.1.2 M2M-100 (facebook/m2m100_418M)

#### 2.1.3 HY-MT1.5-1.8B (Tencent HunYuan)

- Kiến trúc HunYuanDenseV1 (decoder-only CausalLM, không phải encoder-decoder).
- Issue #91: CT2 không convert được (custom arch + dynamic RoPE + QK-norm).
- Runtime: HF Transformers CPU, bf16 → latency ~50s/inference, RAM ~3.8 GB.
- **Cảnh báo so sánh:** latency HY-MT không thuận lợi vs CT2-int8 (opus/m2m) — chỉ so sánh BLEU (chất lượng).
- Hiện chỉ có gold-set (42 items); full-pool chưa chạy.

### 2.2 Automatic Speech Recognition

(bảng các model ASR thêm ở đây)

Nội dung cần viết cho mỗi model:

- Kiến trúc, phương pháp decode (greedy, beam search, CTC…).
- Ngôn ngữ đầu vào (tiếng Việt, đa ngôn ngữ).
- Chế độ inference: batch / streaming, ngôn ngữ chỉ định (language="vi" hay auto-detect).
- Phiên bản checkpoint, thư viện sử dụng (transformers, sherpa-onnx, faster-whisper…).

## 3. Dataset Benchmark

### 3.1 FLEURS

- Nguồn: google/fleurs (HuggingFace), tập test tiếng Việt + tiếng Anh.
- Ngôn ngữ: vi, en.
- Split: test.
- Đặc điểm: 30 base utterances mỗi ngôn ngữ × 9 điều kiện SNR = 540 items tổng cộng (60 items/condition).
    + 9 conditions: clean + steady noise + impulsive noise @ 15/10/5/0 dB.
    + Con số 594 là tổng số item của `eval_manifest_v1.json` (540 ASR + 42 MT + 12 TTS), không phải items/condition.
- Đặc điểm tham chiếu: transcript FLEURS là informal spoken-language (ảnh hưởng BLEU interpretation).

### 3.2 VIVOS

- Nguồn: VIVOS corpus.
- Ngôn ngữ: vi.
- Split: test.
- Đặc điểm: 100 utterances × 9 conditions = 900 items.
    + Cùng SNR grid như FLEURS (clean + steady/impulsive @ 15/10/5/0 dB).
- Giọng đọc sạch, đọc chuẩn → khác biệt với FLEURS.

### 3.3 VietSuperSpeech (VSS) — dùng làm reference

- Zipformer được train trên VSS.
- VSS chỉ được dùng làm reference text cho các model ASR còn lại (Moonshine, Whisper), KHÔNG dùng để eval Zipformer (tránh train/test overlap).

(Bảng tổng hợp dataset)

## 4. Thiết lập thử nghiệm (Experimental Setup)

### 4.1. Phần cứng & Môi trường

| Thông số | Chi tiết |
| --- | --- |
| CPU | 11th Gen Intel Core i7-1165G7 (Tiger Lake), 4 cores / 8 threads |
| RAM | 16 GB |
| GPU | — (CPU-only evaluation) |
| Framework | PyTorch 2.12.1+cpu, Transformers 4.57.6, CTranslate2 4.8.0 |
| Precision | bf16 (HF models), int8 (CT2 models) |
| Batch size | 1 (per-item inference) |

### 4.2. Cấu hình Benchmark

Cho ASR:
- Ngôn ngữ chỉ định: vi hay auto-detect.
- Decode strategy: beam search (beam tối ưu từ sweep phase).
- Chunk size (nếu streaming).
- Audio đã resample về 16kHz chưa.
- **Text normalization trước khi tính WER/CER:** lowercase cả ref và hyp (jiwer là case-sensitive; Zipformer output ALL-CAP nên cần lowercase). **Không bỏ dấu câu, không bỏ số.**

Cho NMT:
- Cặp ngôn ngữ dịch: vi → en (MT candidates hiện chỉ hỗ trợ vi→en).
- Beam size.
- Max length (max_new_tokens = 256).
- Tokenizer sử dụng.
- **BLEU:** sacrebleu default (case-sensitive, lowercase=False).

## 5. Kết quả Benchmark

*(Sau đây có bảng tóm tắt beam-sweep phase — beam tối ưu per model)*

### 5.1. Kết quả ASR

#### 5.1.1. Chất lượng nhận dạng (WER/CER) — trung bình

(bảng WER trung bình trên từng dataset × candidate × language)

(bảng CER trung bình)

#### 5.1.2. Phân tích WER theo điều kiện nhiễu *(SNR × noise type)*

> Finding chính: **steady noise degrade mạnh theo SNR, impulsive noise degrade ít** (VD: moonshine-vi steady 0.31 → 0.77 vs impulsive 0.32 → 0.38).

(bảng WER × noise_type × SNR cho mỗi candidate × language × dataset)

(phân tích: model nào robust với impulsive? Model nào collapse ở steady 0 dB?)

#### 5.1.3. Hiệu suất tính toán (RTF, Latency, Peak RAM)

(bảng hiệu suất suy luận — latency, peak RAM, RTF)

#### 5.1.4. Biểu đồ gợi ý

- **Line chart:** WER theo SNR (15 → 0 dB) cho mỗi model × language × dataset — steady line + impulsive line + clean marker. *(đã generate: `asr-wer-vs-snr-fleurs.png`, `asr-wer-vs-snr-vivos.png`)*
- **Bubble scatter:** X = latency (log scale), Y = WER (thấp = tốt nhất), bubble size = peak RAM. Chỉ vi, 2 panels FLEURS | VIVOS. *(đã generate: `asr-latency-vs-wer.png`)*
- **Heatmap:** candidate × 9 conditions, cell = mean WER. *(đã generate: `asr-wer-heatmap-fleurs.png`, `asr-wer-heatmap-vivos.png`)*
- **Grouped bar chart:** so sánh 5 ASR candidates (× language) trên từng dataset.
- *(có thể thêm radar chart tổng hợp 5 metrics: WER, CER, RTF, latency, RAM — đang cân nhắc)*

### 5.2. Kết quả NMT

(bảng chất lượng dịch BLEU — full-pool 347 items cho opus/m2m; gold-set 42 items cho hy-mt)

(bảng hiệu suất tính toán: latency, peak RAM)

**Lưu ý:** HY-MT chỉ có gold-set (42 items), latency ~50s (HF CPU) không so sánh được với opus/m2m CT2-int8 (~1s). Chỉ so sánh BLEU.

(biểu đồ BLEU bar chart + bubble scatter BLEU vs latency — `mt-bleu-vs-latency.png`)

## 6. Phân tích & Thảo luận

### 6.1 Phân tích theo Chất lượng

- Model ASR nào có WER/CER thấp nhất trên từng dataset? Có nhất quán không?
- Vì sao Zipformer-vi tốt nhất trên VIVOS (giọng sạch) và FLEURS? Moonshine-vi vs Whisper-vi khác biệt ra sao?
- NMT: Opus-MT vs M2M-100 — model nào dịch tốt hơn cho cặp vi→en? Vì sao?
- HY-MT (1.8B) vs M2M-100 (418M) — chất lượng BLEU so sánh ra sao trên gold-set? Giải thích khác biệt (fluent paraphrase vs lexical match với FLEURS spoken references).

### 6.2 Phân tích theo Hiệu suất

- Model nào phù hợp cho edge device (RTF thấp, RAM nhỏ)?
- Model nào phù hợp cho server (chất lượng cao, không quan trọng RAM)?
- Đánh đổi (trade-off) giữa chất lượng và tốc độ.
- **Giới hạn phần cứng:** CPU-only, Intel i7-1165G7 (Tiger Lake) 4C/8T, bf16 không có native support → HY-MT latency cao hơn nếu chạy trên GPU/AMX.

### 6.3 Phân tích lỗi (Error Analysis) — *(tùy chọn, có thể bỏ; manual sampling)*

Lấy 3–5 mẫu lỗi tiêu biểu của ASR: từ bị nhận sai, mất dấu, nhầm âm tiết.
Lỗi của NMT: dịch sai ngữ cảnh, sai từ vựng chuyên ngành, trật tự từ.
Phân loại lỗi: do giọng nói, do nhiễu, do từ hiếm (OOV), do cấu trúc câu.
*(Lưu yêu cầu đọc per-item output thủ công — không có tooling tự động)*

## 7. Kết luận & Khuyến nghị

### 7.1 Khuyến nghị theo use-case

- **Edge device (RAM < 1 GB, latency < 1s):** model ASR nào? Model MT nào?
- **Server (chất lượng tối đa):** model ASR nào? Model MT nào?
- **Hybrid edge-server:** pipeline đề xuất (ASR vi edge → MT cloud?).

### 7.2 Tổng kết

- ASR vi: Zipformer-vi (nhỏ, nhanh, chínhác) vs Whisper-vi (đa ngôn ngữ, linh hoạt).
- NMT vi→en: M2M-100 vs Opus-MT — trade-off chất lượng vs tài nguyên.
- HY-MT: chất lượng promising (gold-set BLEU 25.87) nhưng chưa deployable (issue #91).

### 7.3 Hướng phát triển

- Chạy full-pool (347 items) cho HY-MT — ước tính ~4.8–5.3 giờ (CPU-only).
- Hỗ trợ QNN/HTP on-device (cần Qualcomm hardware).
- Nếu CTranslate2 upstream hỗ trợ dynamic RoPE → HY-MT CT2 → latency giảm ~50x.

## 8. Reproducibility & Limitations

### 8.1 Reproducibility

- Lệnh chạy benchmark: `uv run python bench/run.py --manifest eval_data/mt_vi_en_eval_manifest.json --candidate <id>`
- Manifest paths: `eval_data/eval_manifest_v1.json` (FLEURS), `eval_data/vivos_vi_eval_manifest.json` (VIVOS), `eval_data/mt_vi_en_eval_manifest.json` (MT).
- Model paths: `models/` (CT2 weights, sherpa-onnx), HF cache (transformers).
- Beam config: xem `bench/registry.py` REGISTRY dict.

### 8.2 Limitations

- **FLEURS references là informal spoken transcripts** → BLEU penalty cho output fluent/literary (ảnh hưởng HY-MT nhiều hơn m2m/opus).
- **Single test environment** — chỉ CPU Intel i7-1165G7 (Tiger Lake) 4C/8T, không có GPU hay NPU.
- **Chưa có on-device (QNN/HTP)** numbers — cần Qualcomm HTP hardware.
- **HY-MT chỉ gold-set** — full-pool chưa chạy; latency không comparable với CT2.
- **Text normalization hạn chế** — chỉ lowercase, không strip punctuation/số → WER/CER có thể cao hơn báo cáo khác dùng aggressive normalization.
