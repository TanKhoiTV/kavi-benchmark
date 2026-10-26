# Competitive and literature context for the latency design

Why this page exists: the pipeline design here was derived from measurement, and the measurements disagree with several defaults that other projects and papers use. This records what the field does, so that a choice of ours which looks unusual is recognisable as either a deliberate departure or a mistake nobody has made before.

## 1. RTranslator — the closest mature reference

Apache-2.0, offline, Android, ~11k stars. Its v3.0 betas replace NLLB with a user-selected MT backend (Bergamot / MADLAD-400 3B / HY-MT 1.5 1.8B), replace ML Kit with FastText + Lingua, and add Silero VAD alongside the old amplitude threshold.

The numbers worth knowing next to ours:

| RTranslator | This project | Note |
|---|---|---|
| `SPEECH_TIMEOUT_MILLIS` ≈ **1300 ms** | 500 ms | The mature app waits 2.6x longer before endpointing |
| pre-roll buffer before the VAD trigger | not implemented | Keeps the first phoneme that triggered the VAD |
| mic muted only during TTS playback | not implemented | Overlaps *capture* with synthesis; we overlap MT with synthesis |
| ONNX graphs split into initializer / encoder / decoder | single graphs | Their RAM tactic for the same 8 GB class of phone |
| ~1.2 GB of models, 6 GB RAM floor | ~0.6-0.9 GB resident | Comparable order of magnitude |

**The 1300 ms timeout is the single most useful data point.** It says our 500 ms is aggressive rather than conservative, which is exactly what the pause measurement found independently (`pause-distribution.md`: 500 ms already truncates 3 of 100 Vietnamese utterances). Two independent lines converge: this timeout is not free latency, it is a correctness parameter.

The two ideas worth copying are cheap and orthogonal to our model work: a pre-roll buffer, and muting the mic only for the duration of playback rather than for the whole synthesis.

## 2. Where our streaming design sits in the literature

Reading target tokens as they are decoded and cutting the spoken chunk at the first clause boundary is, in the literature's terms, **Local Agreement** (Liu et al. 2020; Polak et al.'s IWSLT 2022 submission popularised it for speech translation) with n ≈ 5 words. That is the training-free baseline: re-decode a growing prefix, emit the longest common prefix that two consecutive hypotheses agree on.

Two consequences:

- **We are not inventing an unstudied mechanism**, which is good — the quality/latency curve for Local Agreement is documented, including its failure modes.
- **We are cheaper than the textbook version.** Local Agreement's real cost is re-decoding the whole prefix per chunk (papers attribute ~1.3 s of compute overhead to it, against ~0.9 s for attention-based policies). We decode the sentence **once** and read the stream, so we pay one decode and get chunk boundaries for free. That is a genuine advantage of the streaming callback over the prefix-re-decode formulation, and it is worth stating explicitly rather than rediscovering.

The literature's alternative is a *learned* monotonic policy — EMMA in Meta's SeamlessStreaming, or wait-k (Ma et al. 2019). Those require fine-tuning the model with a latency loss, which is not available to us with a frozen Opus-MT.

### Comparability of our numbers

The field measures streaming latency with **Average Lagging** and, for speech output, **Ending Offset** (time from source speech ending to the last translated audio emitted). For scale:

| system | metric | value |
|---|---|---|
| SeamlessStreaming S2TT, X→eng | AL / LAAL | 1.68 s / 2.20 s |
| SeamlessStreaming S2ST, X→eng | Ending Offset | **2.79 s** |
| SeamlessStreaming S2ST, eng→X | Ending Offset | **4.64 s** |
| This project (target) | time to first audio after speech end | **1.85 s median** |

Our gate is a first-audio metric, closest to the low end of their ending offset. So a contest submission that meets 2.0 s would be, on the only axis the literature reports for speech output, better than Meta's 2.5B-parameter streaming system — which is credible given theirs streams long-form audio while we optimise a single short turn, but it is worth knowing before anyone claims novelty either way. **We should report Ending Offset in any write-up**, because first-audio alone hides the tail, and the same paper shows ending offset accumulating over long-form audio.

## 3. Endpointing: the literature is emphatic, and it agrees with our measurement

`pause-distribution.md` concluded the timeout is boxed in: lowering truncates, raising breaks the gate. The speech-dialogue literature has been here for years.

- **Within-turn pauses are frequently longer than between-turn gaps** (Heldner & Edlund 2010). The 200-500 ms band contains both classes almost completely, so *silence duration carries little information about whether the speaker is finished*. This is the theoretical statement of what we measured empirically on Vietnamese.
- **Pause durations are exponential**, so false-cut-off rate falls exponentially with the timeout: `FA = e^(β + α·L)` (Raux & Eskenazi 2008, 18k turns). Practical reading: halving the cut-off rate costs a roughly fixed number of milliseconds, which is why the trade never becomes comfortable.
- **Shipped defaults cluster at 500 ms, not 300 ms**: OpenAI `server_vad` 500 ms, LiveKit `min_endpointing_delay` 500 ms, Deepgram `utterance_end_ms` >= 1000 ms. Our 500 ms is the industry default, not an outlier.
- **A bare silence timer is the known-bad baseline.** In LiveKit's eot-bench, silence-only VAD cuts off **55.6%** of turns at a 300 ms budget; the best learned turn detector reaches 9.9%. At 600 ms the same baseline still cuts 21.7%.

That last row is the useful one for us: a **text model gating the silence timer** recovers most of the gap — the architecture is "VAD detects silence, a small text model decides whether to commit", with reported ~50 ms CPU inference for a 135M model. Applied here, that is precisely the proposal in kavi-android#56: let the transducer hypothesis decide, not the clock. The literature predicts it can make a ~300 ms timeout safe, which would return the 200 ms the pause measurement had to decline.

Two further transferable rules:

- **Modulate by question type.** Lowering the threshold after a yes/no prompt cut median latency from 965 to 324 ms in Raux & Eskenazi. Our corpus is full of questions ("Anh giải thích lại chậm hơn được không?"), so a question-mark-gated timeout is a cheap, evidence-backed win.
- **Keep a hard ceiling.** A 1.5-2.5 s backstop prevents dead air; one team found 3 s made users re-start out of confusion.
- **Buy latency with speculation, not aggression.** Eager generation buys 150-250 ms at 50-70% extra model calls. Our first-clause commit is a speculative bet with no retraction — if a first chunk turns out wrong, we cannot un-say it. That risk should be measured against the quality criterion, not just the latency one.

## 4. The competitive field in this contest

Public entries are scarce and attrition is high:

| entry | state | stack |
|---|---|---|
| Khanhhh239/OneVoice | inactive ~2 months before this check | — |
| technoob05/VoiceAI_SaiGonHub | author "retired" it (2026-10-14 snapshot) | PhoWhisper + NLLB-600M + MMS-TTS; T4 GPU medians 454 ms VI→EN; tone-aware fingerprint cache |
| voicebridge-oss | active | sherpa-onnx Whisper + Supertonic/MeloTTS, MT via ML Kit, "GPU-first, HTP as stretch" |
| ToneBridge | laptop only, never ran on device | documented per-stage laptop latencies |

No organizer-published repo index or leaderboard exists; submissions go through forms and most repos stay private until the November finals. The practical read: **the field is thin, and the public entrants mostly did not solve on-device latency** — SaiGonHub's 454 ms VI→EN is on a T4 GPU, ToneBridge never ran on hardware, voicebridge ships ML Kit for MT (closed-source, and its README concedes the NPU is unbuilt).

That makes the differentiator the thing this project spent its measurement budget on: **latency engineering on a real phone**, not model choice. Which is also what the contest's own framing emphasises — field testing in October, finals in November, Qualcomm hardware and AI Hub validation.

## 5. Gaps this research exposes in our own plan

1. **No pre-roll buffer.** RTranslator keeps audio from before the VAD fires. Without it the first phoneme is lost, which for Vietnamese (heavy initial consonants, tone-bearing) matters more than for English.
2. **No Vietnamese text normalisation before TTS.** MT output will contain digits, units and dates that a Vietnamese TTS will mispronounce; `vinorm` exists for exactly this and is not in our pipeline.
3. **We report first-audio only.** Ending Offset is the metric the literature uses for speech output, and it is the one that catches a slow tail. Cheap to add to the harness.
4. **The first-clause commit is unretractable.** Local Agreement normally re-emits corrected prefixes; we cannot, because we have already synthesised audio. Worth a quality check rather than an assumption.

## Sources

- RTranslator releases and repo: <https://github.com/niedev/RTranslator>
- Seamless (streaming S2ST, EMMA, AL/LAAL/ending offset): <https://arxiv.org/abs/2312.05187>
- Local Agreement policy (Liu et al. 2020); Polak et al., IWSLT 2022
- Heldner & Edlund 2010, *Pauses, gaps and overlaps in conversations*
- Raux & Eskenazi 2008, *The INPROTK 2012 release* lineage — endpointing trade-off curve
- LiveKit end-of-turn benchmark: <https://livekit.com/benchmarks/eot-bench>
- Contest context: Saigon AI Hub / VNG press release, <https://vng.com.vn/news/press-release/>
