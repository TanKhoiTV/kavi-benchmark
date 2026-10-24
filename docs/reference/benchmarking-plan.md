# Benchmarking & Test-Suite Plan

> **Status:** Draft (planning). This document captures *what we will measure and on what data* — it deliberately does **not** pick the ASR / MT / TTS tech stack. Those choices are deferred to the **[open-parameters register](../decisions/README.md#open-parameters)**, which this plan feeds.
>
> **Sources:** synthesized from an external analysis pass (Claude) on datasets, the RTranslator baseline, and harness design, cross-checked with Context7 for the `datasets` / `torchaudio` / `evaluate` library APIs. Corrections to earlier assumptions are flagged inline.

---

## 1. Purpose & scope

We run experiments **before** settling the tech stack (per the agreed sequencing). This plan is the first deliverable: a **model-agnostic testing & benchmarking suite** plus a **dataset decision**. The numbers it produces are what the register will record as the architecture choice.

The suite must be **pluggable**: we swap ASR / MT / TTS candidates in and out and score them against identical data and the contest metrics. No candidate is wired into harness code.

### Relationship to existing decisions

| Doc | What it locks | What this plan adds |
| --- | --- | --- |
| **ADR-001** Offline-first, on-device | Zero cloud dependency | No-internet DQ is a scored gate in §7 |
| **ADR-002** Target platform | Snapdragon 8 Gen 2 / Android 16, Hexagon HTP v73, GPU→CPU fallback | Datasets + noise must reflect the *noisy* deployment, not clean-room only |
| **ADR-003** Hexagon runtime (Proposed) | QAIRT/QNN primary, ORT-XNNPACK CPU fallback, w8a16, fixed shapes | Determination method = **CPU baseline first, then on-device benchmark** (§6, §8) |
| `docs/specifications.md` | The six objective metrics + hard thresholds | §7 restates them as harness outputs |

> **Out of scope here:** which models/frameworks we ship. That is the [open-parameters register](../decisions/README.md#open-parameters), fed by the v0 results in §8.

---

## 2. Evaluation axes

The pipeline is **ASR → MT → TTS** (VI↔EN). We need data + a noise bank covering four axes:

1. **VI ASR** — Vietnamese speech recognition.
2. **EN ASR** — English speech recognition (for the EN→VI direction).
3. **VI↔EN speech translation** — the real end-to-end test.
4. **TTS reference audio** — targets / MOS references.

Plus a **noise bank** for the contest's factory / construction / logistics settings.

---

## 3. Datasets (shortlist)

### 3.1 Vietnamese ASR

| Dataset | Size | Style | Transcripts | License | Obtain |
| --- | --- | --- | --- | --- | --- |
| **VIVOS** | 15 h, 11,660 train / 760 test | Read speech, 65 speakers, quiet studio | Human-annotated (gold) | **CC BY-NC-SA 4.0** ("academic purposes only") | `AILAB-VNUHCM/vivos` |
| **VietSuperSpeech (VSS)** | 267.4 h, 52,023 utterances (240.7 h train / 26.7 h dev-test) | Casual conversational — YouTube vlogs, informal, diaspora dialogue | **Pseudo-labeled** via Zipformer-30M-RNNT (not human-verified) | **MIT** (confirmed) — commercial-safe; pseudo-labeled refs are a quality, not license, caveat | `thanhnew2001/VietSuperSpeech` |
| **FOSD / FPTVIET** | clean studio | clean anchor | — | verify | — |

**Use them as complements, not substitutes:**

- **VIVOS** = clean-room, gold-labeled read speech → a controlled WER baseline, but near-zero acoustic/register match to a noisy factory floor.
- **VietSuperSpeech** = far closer to deployment register (spontaneous, informal) and gives real acoustic diversity, but its labels are **model-generated**. Use it for robustness/stress testing, and hand-verify a small subset (50–100 utts) if you want a trustworthy WER figure. Do **not** treat its labels as ground truth for a final WER number.

### 3.2 English ASR / TTS reference

- **Common Voice (Mozilla)** — **CC0** (safest license in this list). EN + VI splits from the same crowdsourced pipeline; accent-diverse. Good for EN ASR eval and as a second VI source beyond VIVOS's 65 speakers.
- **LibriSpeech** — CC BY 4.0, the standard EN ASR benchmark (clean/other splits give a free "quiet room" vs "harder acoustic" split).
- **LibriTTS** — CC BY 4.0, 585 h, 2,456 speakers, 24 kHz — best bet for **EN TTS reference audio** (multi-speaker, studio, designed for TTS).
- **InfoRe** (VI TTS ref) — single female speaker, ~14,935 clips, reused in open VI TTS projects (e.g. `vietTTS`); closest VI analog to LJSpeech. **Verify donation terms** before bundling raw clips in a public repo.
- **viVoice** (VI TTS ref) — 1,017 h, multi-speaker, YouTube-sourced; **CC BY-NC-SA 4.0, research-only, institutional-email gate** → internal-eval-only, do not redistribute.

### 3.3 VI↔EN speech translation — correction

> **Correction (important):** earlier we assumed **CoVoST-2** / **MuST-C** covered Vietnamese. They do **not**:
>
> - **CoVoST-2** translates 21 langs → EN and EN → 15 of those. **Vietnamese is in
> neither list.**
> - **MuST-C** is EN-source only → 8 European langs. No Vietnamese, no non-EN audio
> input either way.
>
> Do **not** build around them for VI.

What exists and is usable:

- **PhoST** (VinAI, Interspeech 2022) — 508 audio h, 331K triplets of (EN audio, EN transcript, VI subtitle text). `github.com/VinAIResearch/PhoST`. A real, purpose-built **EN audio → VI text** corpus → tests the **EN→VI** direction. **Research/educational use only, no redistribution** (VinAI terms) — eval-only; do not bundle in the shipped app (see license decision record).
- **VI→EN direction has no equivalent large purpose-built corpus.** Two workarounds:
  1. **FLEURS-vi** is built from FLoRes-101, so each VI recording's transcript is a translation of the *same underlying sentence* that exists in **FLEURS-en** (and 100+ langs). That gives a legitimate, if indirect, **VI-audio → EN-text** pairing by matching sentence IDs across the two language splits — **CC BY 4.0**, directly downloadable (`google/fleurs`). Validate the exact ID-alignment mechanics before leaning on it heavily.
  2. **Bespoke gold set** — take 150–300 sentences from a verified VI subset (VIVOS / Common Voice-vi / VSS), get careful human translations into EN. For a *lean* eval set this is a half-day task; write factory/logistics-flavored sentences (equipment names, safety instructions, numbers, short imperatives) to match the contest domain.

### 3.4 Noise bank

| Corpus | Content | License | Fit |
| --- | --- | --- | --- |
| **MUSAN** | ~109 h music/speech/technical & non-technical noise | CC / US public domain, built legally clean | General additive noise + babble (speech subset) |
| **DEMAND** | 18 multichannel real-world recordings, 6 env categories | CC BY-SA 3.0 | Reverberant real-room character; noisier cats = logistics proxy |
| **NOISEX-92** | factory1/factory2, babble, machine-gun, military | Murky redistribution terms | **Internal eval only** — closest to literal "factory floor" |
| **OpenSLR RIRS_NOISES** | Simulated + real room impulse responses | CC BY 4.0 | Convolutional reverb (warehouse/factory-scale rooms) |
| **Freesound** (curated industrial/machinery/forklift packs) | Real recordings, per-clip licensing | Per-clip CC — check each | Best for construction/logistics sounds not elsewhere represented |

### 3.5 Noise-mixing recipe (eval time, not baked in)

Applied at **eval time** so every candidate is scored on **byte-identical inputs**:

1. **Reverb** — convolve clean utterance with a large/medium-room RIR from RIRS_NOISES (warehouse/factory reflections).
2. **Additive noise** — select a noise clip (rotate: mechanical/steady vs babble/impulsive), trim/loop to utterance length, scale to target SNR via RMS-based scaling, mix, peak-normalize. (`torchaudio.functional.add_noise` + `fftconvolve` + `resample` cover this directly.)
3. **SNR sweep** — five conditions is enough for a real curve:

| Condition | SNR | Purpose |
| --- | --- | --- |
| Clean | ∞ | Reference / ceiling |
| Mild | +15 dB | Light background |
| Moderate | +10 dB | Noisy but workable |
| Hard | +5 dB | Loud warehouse / near-forklift |
| Severe | 0 dB | Stress test, near WER breakdown |
| (Stretch) Worst-case | −5 dB | Stability check only, not a pass/fail bar |

Run each SNR against **≥2 noise types** (steady vs impulsive) — ASR degrades differently under the two, and a single averaged "noisy" number hides which failure mode you have.

### 3.6 License watch-outs (for a possibly-public contest repo)

- **Safe to bundle/reference publicly:** Common Voice (CC0), LibriSpeech / FLEURS / LibriTTS (CC BY 4.0, credit), MUSAN (public domain), RIRS_NOISES (CC BY 4.0).
- **Attribution + share-alike:** DEMAND (CC BY-SA 3.0).
- **Research/eval-only, don't ship:** VIVOS (CC BY-NC-SA), viVoice (CC BY-NC-SA), NOISEx-92 (ambiguous redistribution).
- **Verify before relying on:** InfoRe donation terms (VSS = MIT and PhoST = research-only / no-redistribution are now resolved — see the license decision record).
- **NLLB** (RTranslator ships it; we already avoid it): CC BY-NC 4.0 — confirmed non-commercial. Do not let RTranslator's NLLB weights leak into anything we distribute, even for comparison.

---

## 4. Model candidate landscape (ASR, MT, TTS)

> **Input to the [open-parameters register](../decisions/README.md#open-parameters), not a decision here.** Candidate-model comparisons for the ASR, MT, and TTS stages. ASR (§4.1–4.3) is sourced from one external analysis pass; MT and TTS (§4.4–4.5) were synthesized from the repo's archived docs (architecture.md, ADR-001) + Context7 library docs (CTranslate2, Piper, Transformers), then **reconciled with a second external MT/TTS analysis pass** (Claude). It refines the v0 candidate set (§8) and the per-stage list (§9); the final picks stay with the register.

Four VI/EN ASR candidates were compared on accuracy, latency (Snapdragon 8 Gen 2 estimates), memory, licensing, and Qualcomm AI Hub availability.

### 4.1 Accuracy — Vietnamese WER (lower is better)

| Model | CMV-Vi | VIVOS | VLSP T2 | EN support | License | AI Hub (SD8G2) |
| --- | --- | --- | --- | --- | --- | --- |
| **Whisper Small** (244M) | ~26–30% | ~20–25% | ~55–65% | Full multilingual | MIT | Yes (w8a16) |
| **PhoWhisper Small** (244M) | **11.08** | **6.33** | **32.96** | Inherited (unbench) | BSD-3 | No (DIY export) |
| **Zipformer 30M VI** (~30M) | no public # | — | — | **None — VI only** | Apache-2.0 | No (untested) |
| **Zipformer small EN** (csukuangfj, ~27M) | — | — | — | **EN only** | Apache-2.0 | No (untested) |
| **Moonshine Tiny VI** (27M) | 18.8 (CV17) | — | — | **None — VI only** | Apache-2.0 | No (DIY) |

- **PhoWhisper Small** is the VI-accuracy winner (~2.4× better than Whisper Small on CMV-Vi) and is the same architecture as Whisper Small, so the AI Hub `SHA+conv` → w8a16 trick applies — a QNN/HTP path is feasible, just not prebuilt.
- **Whisper Small** is the only candidate with **confirmed EN + VI** support and the only one with **published SD8G2 latency** (via Qualcomm AI Hub).
- **Zipformer / Moonshine are single-language** — neither serves both directions alone.

> **Superseded (ADR-008, 2026-07-30):** the v1 ASR decision pairs the VI Zipformer-30M with the **EN Zipformer small** as a **Dual-Zipformer CPU configuration** via sherpa-onnx, serving both directions. Whisper/PhoWhisper QNN re-sourcing is out of scope; this table is the pre-decision landscape.

### 4.2 Latency & memory (SD8G2 estimates)

| Model | Encoder TTFT | Decoder/token | Params | INT8 size | Window |
| --- | --- | --- | --- | --- | --- |
| Whisper Small | ~400–800 ms | ~15–30 ms | 244M | ~244 MB | **30 s fixed** (overhead on short utts) |
| PhoWhisper Small | ~400–800 ms | ~15–30 ms | 244M | ~244 MB | 30 s fixed |
| Zipformer 30M VI | ~50–150 ms | ~5–15 ms | ~30M | ~32 MB | scales w/ duration (streaming) |
| Moonshine Tiny VI | ~50 ms | ~10–20 ms | 27M | ~27–34 MB | scales w/ duration (streaming) |

- Whisper/PhoWhisper's **fixed 30 s window** is pure overhead on short utterances and directly pressures the **2.0 s turnaround budget** (§7).
- Zipformer/Moonshine **stream and scale with actual duration** (~50 ms TTFT) — a better structural fit for that budget, *if* we accept VI-only + a CPU/DIY-NPU path.

### 4.3 Implications for the candidate matrix

- **Kavi is bidirectional VI↔EN**, so the ASR candidate set is constrained *per direction*: EN→VI needs an EN-capable ASR (Whisper Small, or PhoWhisper by inheritance), while VI→EN may also evaluate PhoWhisper / Zipformer / Moonshine.
- **Licensing is clean** across all four (Apache-2.0 / BSD-3, commercial OK) — no NLLB-style blocker on the ASR side.
- Only Whisper Small has published on-device numbers; the rest need the manual profiling our v0 harness exists to produce — so this comparison *broadens* the candidate set without answering the on-device question.

> **Caveats:** FLEURS is read/quiet speech; real-world VI WER is 2–5× higher. PhoWhisper/Zipformer numbers are on more conversational data (VLSP, VIVOS) and are more representative. No public head-to-head of Moonshine Tiny VI vs PhoWhisper Small exists (different evaluation sets). Latencies are paper/architecture estimates, not measured on our unit.

### 4.4 MT candidate landscape (VI↔EN)

> Sources: archived `architecture.md` + `docs/adr/001-prioritize-hy-mt-over-nllb.md` (ADR-001) + Context7 (CTranslate2, Transformers) + a second external MT analysis pass (Claude). The prototype (`aivoice-2026/archive/pipeline.py`, `models/opus-mt-vi-en-ct2`) runs **Opus-MT vi-en via CTranslate2 int8**; ADR-001 selected **Hy-MT1.5** (NLLB only as reference). MT is **bidirectional** — VI→EN and EN→VI.

| Model | VI↔EN quality | Latency (SD8G2 est) | Size | License (commercial?) | AI Hub / QNN status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **Opus-MT** (Helsinki-NLP/opus-mt-vi-en + opus-mt-en-vi) | vi→en **BLEU 42.8 / chrF 0.608**; en→vi **BLEU 37.2 / chrF 0.542** (Tatoeba — *not* FLORES-200; no COMET published) | Sub-100 ms class even at fp32 | ~75M params; ~300 MB fp32 → int8 well under 100 MB | **Apache-2.0** → OK | Not prebuilt (only en-es). HF Marian → ONNX/QNN-convertible, but re-export from original HF PyTorch, not the CT2 build (CT2 not QNN-convertible) | **Current prototype; genuinely bidirectional.** Only model with published pair-specific BLEU both ways. |
| **NLLB-200-distilled-600M** (Meta) | High (FLORES-200 BLEU 30+ "accurate/fluent"; vi-en not separately confirmed) | RTranslator optimized int8+kv-cache: **1.3 GB RAM, ~2 s / 75 tok** | ~2.5 GB fp32 | **CC-BY-NC-4.0** → avoid | Not prebuilt. Standard HF seq2seq → convertible in principle | **Reference only** (non-commercial). Real measured on-device NLLB-family numbers. |
| **Hy-MT1.5-1.8B** (Tencent) | High (surpasses 72B-class on FLORES-200, per ADR-001) | ~400–800 ms CPU (SD888 tested; SD8G2 faster est.) | 440 MB (1.25-bit STQ) | HY Community — commercial OK *per ADR-001*, but **regional carve-out flagged by current stance → verify/avoid** | No (CPU-only STQ kernel, no NPU path) | Old ADR-001 sole pick; covers all pairs in one model. CPU-only, no NPU. |
| **M2M-100** (Meta, 418M / 1.2B) | No direct vi↔en number; one en↔ga proxy showed Opus-MT beating M2M-100 by ~5–7 BLEU (suggestive, different pair) | Heavier than Opus-MT (~5–8× params); unverified on SD8G2 | 418M ≈ 1.9 GB fp32 | **MIT** → OK | Not prebuilt. HF seq2seq → ONNX/QNN-convertible | Bidirectional by design (9,900 directions). **Realistic finalist alongside Opus-MT** if Opus underperforms on domain data — budget an eval to confirm. |
| **MADLAD-400** (Google, 3B/7B/10B) | Strong aggregate FLORES-200 (within ~3.8 chrF of NLLB-54B at 5× smaller); no vi-en pair-specific number | Almost certainly blows RTF/turnaround budget even quantized | 3B ≈ 12 GB fp16 | **Apache-2.0** → OK | Not prebuilt. T5 → ONNX-exportable | License clean, quality real, but **size is the blocker**, not license. |
| **SeamlessM4T v2** (Meta) | High (unified S2ST; VI in 101-lang coverage, no vi-en BLEU found) | Too slow (2.3B, multi-component) | 2.3B | **CC-BY-NC-4.0** → avoid | Not prebuilt. Complex multi-component export | Doubly disqualified: non-commercial + hardest export. Could fold ASR+MT but ruled out. |

- **Realistic finalist set:** **Opus-MT (current) vs M2M-100** — the two commercial-safe, on-device-feasible dedicated MT models. Everything else is license-blocked (NLLB, SeamlessM4T), CPU-only (Hy-MT), or too large (MADLAD-400).

> **Resolved (ADR-007):** Opus-MT vi↔en is the v1 MT choice; M2M-100 remains a contingency candidate.

- **Bidirectional coverage:** Opus-MT needs two models (vi-en + en-vi, both published). NLLB / Hy-MT / MADLAD / M2M / Seamless are multilingual (one model, both directions).
- **NPU path (ADR-003):** CTranslate2 (current MT runtime) is **not** QNN-convertible — re-export Opus-MT from the original HF PyTorch checkpoint to ONNX → QAIRT. NLLB / M2M / MADLAD are also standard PyTorch → ONNX → QAIRT.
- **Licensing gate:** clean = Opus-MT, M2M-100, MADLAD-400, Hy-MT (pending carve-out check). **Avoid** = NLLB (NC), SeamlessM4T (NC).
- **LLM-prompting (Gemma/Phi) and Bergamot/Marian are out-of-scope primaries:** Gemma/Phi (MIT / Gemma-ToS-with-conditions) are latency-risky exploratory options; Bergamot/Marian Firefox student models are attractive (MPL-2.0) but **vi↔en coverage is unconfirmed** — verify before counting on them.

### 4.5 TTS candidate landscape (VI + EN voices)

> Sources: archived `architecture.md` (MeloTTS-VI primary, MMS-TTS-vie fallback) + Context7 (Piper) + a second external TTS analysis pass (Claude). The prototype (`archive/pipeline.py`, `voices/`) uses **Piper** (`en_US-lessac-medium`). TTS must supply **both** a Vietnamese voice (EN→VI output) and an English voice (VI→EN output).

| Model | VI + EN voices | Quality (MOS) | Latency (SD8G2 est) | Size | License | AI Hub / QNN status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Piper** (rhasspy, then OHF-Voice) | EN: extensive, high tiers. **VI: `vi_VN` exists** — `25hours_single` (low), `vais1000` (medium), `vivos` (x_low); quality sits at low/x_low tiers | No formal MOS published; "clear but synthetic" at medium/high, rougher at low/x_low | Very fast (real-time on Raspberry Pi 4/5); < RTF 1, CPU-only | ~tens MB/voice (ONNX) | **Split / time-sensitive:** old `rhasspy/piper` = **MIT** (frozen, no fixes); active `OHF-Voice/piper1-gpl` = **GPL-3.0** (copyleft — problematic for commercial). `espeak-ng` phonemizer also GPL. **Decision needed now.** | **PiperTTS-EN prebuilt on AI Hub**; no PiperTTS-VI prebuilt, but Piper is ONNX-native → easier DIY | **Current prototype.** VI coverage already exists (corrects "VI is the hard gap"). Open items: which VI voice is acceptable + the GPL question. |
| **MeloTTS** (myshell-ai) | EN/ZH/ES on AI Hub yes; VI absent (needs fine-tune from VIVOS/CommonVoice-VI) | Good (VITS, BERT-conditioned) | ~150–250 ms (target) | ~tens–100 MB | **MIT** → OK | EN/ZH/ES on Hub (w8a16, SD8G2); VI needs self-export | Archived primary TTS. VITS-based, non-AR. VI is the known gap. |
| **vietTTS / VITS-VI** | VI yes (single female, InfoRe corpus + HiFi-GAN) | Not independently benchmarked | Low (CPU, VITS-class) | ~tens MB | VITS ref impl **MIT** (verify); **vietTTS repo + InfoRe terms need direct verification** | No (DIY ONNX) | VI-native; same architecture family as Piper — question is whether it scores better than Piper's VI voices in a side-by-side. |
| **MMS-TTS-vie** (Meta) | VI yes (purpose-trained, single checkpoint — most out-of-box VI voice) | Not benchmarked | Very low (~15 ms, VITS-class) | small | **CC-BY-NC-4.0** → avoid | No | **Avoid for production** (non-commercial). Re-check if Meta relicenses. |
| **Coqui XTTS-v2** | EN + 16 langs (verify VI) | High (~94% ElevenLabs per one benchmark) | Higher (AR, GPU-oriented 4–6 GB VRAM) | ~1 GB+ | **CPML → avoid**; Coqui Inc. shut down Jan 2024 (no one to sell a commercial license) | No | Doubly disqualified: non-commercial weights + not latency-suited. |
| **Kokoro** (hexgrad) | EN yes; VI **not confirmed** (community langs ja/zh/fr/hi/it/pt/es, not VI) | Good (lightweight, 82M) | Very fast | ~80 MB | **Apache-2.0** → OK | No | Strong EN-leg-only candidate; bet against VI. |
| **SpeechT5** (Microsoft) | EN only (LibriTTS); no native VI | Moderate | Higher (larger) | ~100s MB | **MIT** → OK | No | Clean license, no VI (would need fine-tuning from scratch). |

- **Commercial-safe VI options are narrow:** Piper `vais1000` / `25hours_single` (pending the GPL question) and vietTTS/InfoRe (pending InfoRe terms). MMS-TTS-vie is the only other purpose-built VI voice and it's license-blocked. Kokoro / Parler / SpeechT5 / Bark have **no confirmed Vietnamese** — EN-leg only.
- **Voice-model licenses are independent of the engine.** Piper's `vivos`-derived VI voice inherits **VIVOS CC-BY-NC-SA-4.0 (research-only)** — do not ship it. `vais1000` and `25hours_single` need their own license checks.
- **Offline guarantee:** Piper and MeloTTS both run fully offline (unlike RTranslator's system-TTS dependency — see §5). Verifiable Kavi advantage.
- **NPU path (ADR-003):** Piper/MeloTTS are ONNX → ORT QNN EP / QAIRT feasible (Piper already ONNX). vietTTS/VITS also ONNX-exportable.
- **Licensing gate:** clean = MeloTTS, vietTTS (verify), Kokoro, SpeechT5, Piper (if pinned to MIT-era). **Avoid** = MMS-TTS-vie (NC), Coqui XTTS (CPML). **Piper GPL fork is the live decision** (see caveats).

> **Caveats (MT/TTS) — reconciled from internal research + external analysis:**
>
> - **Quality numbers are mostly not vi↔en-specific.** Only Opus-MT has published
> pair-specific BLEU (Tatoeba test set — *not* FLORES-200; be precise in writeups). No model has a published **COMET** for vi↔en — you must run it (e.g. `Unbabel/wmt22-comet-da`) against your own reference set.
> - **Cascaded error propagation is untested.** Every MT number above is text-to-text
> on clean reference text, not MT-on-ASR-output. The harness must close this gap; don't treat these as real pipeline quality.
> - **Read/clean-text bias:** Tatoeba/FLORES-200 are formal/clean; expect lower
> real-domain BLEU for all candidates — use as relative rankings.
> - **No public head-to-head** of Opus-MT vs M2M-100 vs MADLAD-400 on vi↔en exists.
> - **TTS has less published data than MT**, especially Vietnamese: no MOS found for
> any VI-capable candidate — the human MOS panel (§7) exists for exactly this.

- **Piper license — resolved for v1 (ADR-009):** TTS is Supertonic Phase 1 → VieNeu-TTS Phase 2; Piper VITS is fallback only. The MIT/GPL fork question matters only if the fallback is exercised — pin the MIT-era `rhasspy/piper` snapshot then.

>

> - Sizes / latencies above are estimates or training-knowledge unless a measured
> figure is cited (e.g. RTranslator's NLLB numbers, Piper on Pi 4/5). The v0 harness must produce the real on-device numbers.

## 5. RTranslator as product baseline

Kavi must not accept worse benchmark numbers than RTranslator (`niedev/RTranslator`, app code **Apache-2.0**) — the closest reference product — while it narrows to VI↔EN on a fixed platform. Claims below are repo-confirmed (README / releases / LICENSE) unless tagged **[INFER]**.

> **Correction:** RTranslator does **not** bundle Piper or any embedded TTS — it uses the **Android system TTS** the user has installed (defaults to Google TTS). That *strengthens* our "bundled Piper = verifiable 100% offline" advantage: RTranslator's offline guarantee for the output leg depends on the user's phone having an offline voice pack, which the app doesn't control or verify — and a fresh/flashed device may lazy-download that voice pack on first use, which could trip a no-network grading rule.

### 5.1 Two different pipelines (not one)

"RTranslator's pipeline" is not singular, so pick the right comparison target:

- **Conversation mode (flagship, two phones):** each phone runs its *own* full ASR→MT→TTS on its *own* speech and ships only **translated text over Bluetooth LE** (no audio in transit, no runtime language detection — each user sets their own language). This is a split-brain two-phone design and a **different, easier** problem than Kavi's single-phone speech-to-speech. **Do not benchmark Conversation mode as Kavi's counterpart.**
- **WalkieTalkie mode (one phone):** listens continuously; **ML Kit** (closed-source, Google) performs live language-ID to steer Whisper/NLLB. The *only* place a proprietary ML component sits in RTranslator's critical path.
- **Text mode:** NLLB only — not relevant to speech-to-speech.

Kavi's architecture (single phone, full on-device ASR→MT→TTS) should be compared against RTranslator's **WalkieTalkie** path, not Conversation.

### 5.2 Latency / RAM table (author-reported, published in the README)

| Model | Variant | RAM | Latency |
| --- | --- | --- | --- |
| NLLB-Distilled-600M | Full int8, no KV-cache | 2.5 GB | 8 s / 75 tokens |
| NLLB-Distilled-600M | RTranslator optimized (partial int8, KV-cache) | 1.3 GB | 2 s / 75 tokens |
| Whisper-Small-244M | Olive-optimized (full int8, KV-cache) | 1.4 GB | 1.9 s / 11 s audio |
| Whisper-Small-244M | RTranslator optimized | 0.9 GB | 1.6 s / 11 s audio |
| Whisper-Small-244M | Low-RAM mode (<8 GB phones) | 0.5 GB | 2.1 s / 11 s audio |

Use this as the **latency/RAM anchor** — order-of-magnitude reference; re-measure on our actual 8 Gen 2 unit (author's test device is never named in the docs).

> **Gap:** RTranslator publishes **no end-to-end (mic→speaker) or EOS→speech latency figure** — only the isolated per-model numbers above. There is therefore **no RTranslator number to literally "beat"**; we must construct our own full-pipeline figure, and we get to be the **first to publish a defensible mic→speaker number** for this device class. Treat RTF < 1.0 and turnaround < 2.0 s as **Kavi's own internal bar**, not a value RTranslator has published.

### 5.3 Quality + offline behavior

**Quality (BLEU/COMET/MOS):** no published number. Budget an explicit early task to **install the APK** (latest v2.1.5 on 8 Gen 2) and run our lean eval set through it manually, capturing transcribed/translated/synthesized output for scoring against the same references. Semi-manual (feed audio via UI, capture output), not scriptable. **RTranslator publishes no per-language WER/BLEU either**, so any Vietnamese accuracy claim we make must come from our own benchmark.

**Offline behavior (verified):** core ASR+MT are on-device once the ~1.2 GB model bundle is downloaded on first launch; no telemetry backend. The **two real offline gotchas** to verify on a grading rig:

- system-TTS voice-pack **first-use download** on a fresh device (output leg only);
- **ML Kit** language-ID may use an *unbundled* model fetched via Play Services on first WalkieTalkie use (only relevant to that mode). **[INFER — verify `build.gradle`**]

### 5.4 Structural gaps Kavi can exploit

- **Commercial license:** NLLB-600M is **CC-BY-NC-4.0** — RTranslator's own README hedges it as "(almost) open-source," so it is **not commercially shippable as-is**. This is an *admitted* gap; Kavi's cleared MT stack (Opus-MT / Hy-MT1.5) is a genuine structural advantage, not just a technical one.
- **TTS ownership:** RTranslator's VI voice quality is whatever the grading device's system TTS provides (uncontrolled, untested by them). Kavi's bundled, license-cleared Piper voice is deterministic and reproducible on any device.
- **No confirmed NPU/Hexagon acceleration:** the README credits "OnnxRuntime" with **no execution provider named** (no NNAPI/QNN/GPU). **[INFER]** If RTranslator is CPU-only ORT, a real QAIRT/Hexagon-HTP pipeline on our 8 Gen 2 is a legitimate, testable latency/RAM edge — verify by profiling their APK (`build.gradle` EP selection), don't assume from docs silence.

### 5.5 Caveats to record when benchmarking

- **VI is in RTranslator's full-quality tier** — a fair, non-degraded VI↔EN comparison (translation quality, at least; TTS voice still depends on the system engine).
- **RTranslator 3.0 is imminent** (NGI Mobifree-funded; first beta Jun–Aug 2026): drops NLLB for Bergamot / Madlad-400-3B / **HY-MT-1.5-1.8B** (the HY-MT we avoid for its regional license carve-out). **Snapshot the exact version/commit + date** you test; if later citations use "3.0 numbers," check which backend variant was used.
- **What to record (checklist):** exact APK **version + commit hash** + date · model-bundle hash · device / chipset / RAM / Android version / thermal state · mode tested (Conversation / WalkieTalkie / Text) · system-TTS engine + version · RAM-mode switch state (0.9 GB vs 0.5 GB Whisper) · network state (airplane mode on, verify no calls) · ML Kit model pre-warmed? (WalkieTalkie only) · audio input source (built-in vs BT) · beam-search on/off · backend generation: **2.x (NLLB)** or **3.0 (HY-MT / Bergamot / Madlad)**.
- App code is **Apache-2.0** — its ONNX conversion/optimization scripts are fair game to study/adapt (not the NLLB weights).

---

## 6. Harness design (pluggable by construction)

One thin adapter interface per stage so candidates are swappable without harness changes:

```python
ASRCandidate.transcribe(audio)   -> (text, latency, peak_ram)
MTCandidate.translate(text)       -> (text, latency, peak_ram)
TTSCandidate.synthesize(text)     -> (audio, latency, peak_ram)
```

Each real candidate (whisper.cpp-ggml-int8, Whisper-Small-Quantized-QNN-DLC, CTranslate2-int8-OpusMT, ORT+QNN-OpusMT, Piper-CPU, Piper-QNN once compiled) is just a class implementing that interface — register in a manifest, fan out across every registered candidate per stage with zero harness changes.

- **On-device runner:** an Android instrumented-test / thin service app reads a run manifest (`{stage, candidate_id, model_path, config}`), executes the eval set through each candidate, logs per-utterance latency, RTF, peak RSS, and dumps raw outputs to disk.
- **Off-device scorer:** a host-side Python script ingests those dumps + reference labels and computes WER/CER, BLEU/COMET, and objective TTS-quality proxies — keep scoring off the phone. (`datasets` + `evaluate` cover loading/corpus + WER/BLEU/COMET; MOS needs a separate path — DNSMOS model or a subjective panel.)
- **Fixed, versioned eval manifest** (`eval_manifest_v1.json`): clean + noisy variants per SNR level, generated once via §3.5, so every candidate is scored on byte-identical inputs — what makes a whisper.cpp-vs-QNN-Whisper A/B fair.

---

## 7. Metrics (from `specifications.md` §3)

| Metric | Requirement | Hard? |
| --- | --- | --- |
| RTF (per stage + E2E) | **< 1.0** | **Yes** |
| Turnaround (EOS→SA) | **< 2.0 s** (instrumented from VAD EOS to first SA, not raw compute) | **Yes** |
| No internet at runtime | **Zero network calls (else DQ)** | **Yes** |
| MT BLEU + COMET | Maximize; both directions; on clean ref **and** actual ASR output (cascaded error) | Target |
| TTS MOS | Maximize via high-quality offline TTS + clean input | Target |
| Stability | Crash / silent-failure rate over N consecutive runs (named contest metric) | Named |

---

## 8. Lean eval set & v0 minimal

> **Implementation status (2026-07-18):** the v0 host-side harness described in §6 is **implemented** in `bench/` (PR #28, #36). It runs today on CPU-default candidates — **faster-whisper** Small int8 (ASR), CTranslate2 Opus-MT vi→en int8 (MT), Piper EN (TTS) — with an off-device scorer (jiwer WER/CER, sacrebleu BLEU; COMET/MOS deferred to v1). The on-device QNN runner (Phase 4) is ready to begin now that the QAIRT gate is resolved. The initial v0 ASR candidate is **faster-whisper**, not whisper.cpp (whisper.cpp / QNN-Whisper remain later candidates).

**Lean eval set:** ~40–60 utterances per direction (VI ASR, EN ASR, VI→EN, EN→VI) from VIVOS (clean anchor) + a hand-verified VSS/Common Voice-vi slice (conversational)

- Common Voice-en/LibriSpeech (EN) + the bespoke 150–300-sentence gold VI↔EN set skewed to factory/logistics vocabulary. Each run through the §3.5 SNR sweep (clean + 4 noisy × 2 noise types) ≈ 400–500 scored clips — small enough to re-run on every candidate swap, large enough to catch real trend breaks. Scale to full VSS/PhoST/FLEURS only after narrowing to 1–2 finalists per stage.

**v0 (smallest harness that gates the decision):**

1. ~30–50 utterance slice — clean + one moderate (+5 dB) + one hard (0 dB) condition, both languages, both directions.
2. **Two candidates per stage** to start: faster-whisper-int8 vs Whisper-Small-Quantized-QNN (the only confirmed EN+VI pair; see §4 for the broader ASR landscape — the v0 harness implements **faster-whisper Small int8 (CPU)** as the initial ASR candidate; whisper.cpp / QNN-Whisper remain later candidates); CTranslate2-int8-OpusMT vs ORT+QNN-OpusMT (once a vi↔en compile works); Piper-CPU vs Piper-QNN if/when compiled.
3. **Automated metrics only:** RTF, turnaround, WER, BLEU, RAM. Defer COMET + human MOS to v1 (slower to stand up; not needed to answer "does QNN beat CPU here").
4. **Single device, single run per config** during exploration; add averaging only when confirming final numbers.

**Job of v0 (narrowly):** answer *does QNN meaningfully beat CPU on this chip, for these specific models, before we sink more days into harder QNN engineering* (dynamic-shape rework, AIMET quantization, DLC packaging)? If the gap on this rough slice is marginal, that's a cheap, legitimate signal to reconsider effort allocation — exactly what the experiments were meant to gate.

### Pre-ASR denoising — status & open gate

> Sourced from a scouting pass over `aivoice-2026/archive/` (the old implementation). **Not a decision here** — it is an open pipeline question the register should record, and this harness (§6/§8) is the vehicle to close it.

- **Current state:** `pipeline.py` has **Silero VAD but no denoising stage**. Denoising is a CPU/numpy stage (`noisereduce` Wiener / RNNoise), **not** NPU-accelerated — it competes with ASR int8 for the CPU budget that ADR-003 / RTF<1 guards.
- **What we had (archived):** the only run with results is a 760-utt VIVOS run on **Whisper Medium**, ESC-50 industrial mix @ **SNR 5 dB**: clean **15.53%**, raw noisy **20.23%**, RNNoise (`stationary=True`) **33.10%**, DeepFilterNet **27.05%** WER. Both denoisers *hurt* WER — but RNNoise used the wrong `stationary` setting and the wrong ASR model (Medium, not Small int8).
- **Decision on record** (archived denoising-scope ADR, `aivoice-2026/archive/docs/adr/002-denoising-experiment-scope.md`): Wiener (`noisereduce`, `prop_decrease=0.5`) = primary, RNNoise `stationary=False` = secondary. **Binary gate:** if denoisers beat raw-noisy WER → tune `prop_decrease` on 50–100 files; else **VAD-only pipeline**. Phase-1 was scoped to 10 VIVOS files × 4 conditions on Whisper Small int8.
- **RESOLVED (2026-07-20, Phase-6 gate):** executed — see `docs/reference/denoising-gate-results.md`. **Wiener ADOPTED** (`noisereduce`, `prop_decrease=0.5`): weighted avg WER **49.82%** vs raw **51.23%**; RNNoise REJECTED (64.36%). Smoke test only (2 utts × 5 conditions); adaptive threshold on clean/high-SNR input is an open tune. Feeds ADR-018's GTCRN-vs-Wiener choice (android plan Risk R1).
- **DeepFilterNet dropped** (unresolved `torchaudio 2.x` PyPI bug; fix only on GitHub main). The old `architecture.md` "tonal-preservation" Anchor-2 rested on that now-dropped model + the flawed run → **treat as obsolete** unless re-validated.
- **Recommendation:** add a denoising toggle (raw vs Wiener vs RNNoise) as a fixed-factors comparison in the §8 v0 lean slice (clean / +5 dB / 0 dB, both languages). Keep the mixing method consistent with §3.5 (`torchaudio.add_noise`), *not* the archived script's custom RMS mix, so the historical "noisy 20.23%" ceiling stays comparable. Outcome feeds the register directly.

### VAD noise benchmark (ADR-022) — status

- **Scope:** host-side scoring of an energy VAD on constructed audio (two FLEURS utterances per item, MUSAN noise, real RIRS impulse responses); method and assumptions in [`docs/vad-benchmark.md`](../vad-benchmark.md), results in [`docs/vad-results.md`](../vad-results.md).
- **Make targets:** `vad-bounds`, `vad-manifest`, `vad-audit`, `vad-gold-audio`, `vad-collar`, `vad-run`, `vad-smoke`, `vad-sweep`, `vad-sweep-smoke`, `vad-manifest-40`, `vad-collar-sensitivity`, `vad-stage5`, `vad-decision`.
- **Result (Case B):** no energy threshold and speech timeout in the grid passes in every cell; only the quiet/clean cells pass. Table and command record: `bench/results/vad/sweep_table.csv`, `bench/results/vad/sweep_record.md`.
- **Open:** hand-labelled collar (Stage 3B) not performed; on-device confirmation pending; model-based VAD follow-up. Outcome recorded in ADR-022.

---

## 9. Open questions / deferred decisions (→ [open-parameters register](../decisions/README.md#open-parameters))

- **Final dataset set** — VSS (MIT), PhoST (research-only), and **InfoRe** licenses now resolved: VSS/PhoST as noted; **InfoRe confirmed AVOID** (no published terms, 401-gated) and `25hours_single` also **AVOID** (license unknown, InfoRe-derived) — see `aivoice-2026/docs/decisions/ADR-029-license-gate.md` + PR #35.
- **VI→EN ST corpus** — adopt FLEURS ID-alignment, the bespoke gold set, or both?
- **RTranslator snapshot** — which version/commit tested + dated in writeup; APK-eval task owner & timeline.
- **MOS proxy** — DNSMOS vs small human panel (5–10 raters, 20–30 clips) for v1.
- **Per-stage candidate list** — which two (and later finalist) models per stage, contingent on ADR-003 runtime availability (QAIRT Community Edition access).
- **Pre-ASR denoising gate — gate decision made (Wiener ADOPTED); full lean-slice eval still pending** — see `denoising-gate-results.md`; the on-device GTCRN-vs-Wiener pick (ADR-018) remains → android plan R1.
- **VAD thresholds — closed by the host-side noise benchmark (Case B: model-based VAD follow-up); collar and on-device confirmation pending** — see ADR-022.
- **Architecture — decided in ADR-007/008/009/010** (Dual Zipformer ASR, Opus-MT MT, Supertonic Phase-1 TTS, sherpa-onnx + ORT, coroutines pipeline). **The register remains Draft** pending the Phase-4/5/7 gates.
