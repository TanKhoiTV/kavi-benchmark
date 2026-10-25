# AI Hub: what it actually offers this project

Earlier doctrine (settled after the encoder measurement) was: **AI Hub is a feasibility check, never a budget line**, because AI Hub reported the encoder at 2.03 ms where the device measured 6.43 ms and the decoder step at 2.667 against 11.97 ms — consistency about the wrong quantity, since the harness omits the per-launch host-side descriptor setup our real path pays.

That conclusion stands for *latency*. It was too broad as a verdict on the platform, and this page records what a second look found.

## What is actually there

| capability | what it gives us |
|---|---|
| **Bring-your-own-model** (PyTorch/ONNX → quantize → compile → profile) | compile and profile **our** graphs, not just catalog models |
| **`qnn_context_binary` per chipset** | binaries for silicon we do not own |
| **Proxy devices** (e.g. `qualcomm-qcs9075-proxy`) | compile-only targeting for a chipset absent from our desk |
| **Cloud physical devices** | profile/inference on 8 Gen 1/2/3, 8 Elite, X Elite, RB3 Gen 2, IQ-9075… |
| **Numerics jobs** | quantization accuracy delta against float, per model |
| **Catalog recipes** | Zipformer mixed precision (encoder w8a16, decoder+joiner w16a16), Whisper's fixed-shape KV-cache decoder |

Catalog coverage relevant to us, and the gaps:

- **ASR:** Whisper tiny/base/small/large-v3-turbo in float and w8a16, with per-chipset decoder latencies. **Zipformer** is present, and its precision split is exactly the pattern we would want.
- **TTS:** MeloTTS **en/es/zh only**. No Vietnamese, no Piper, no Supertonic, no ZipVoice.
- **MT:** nothing dedicated; the LLM section (Qwen3, Llama, Phi, Mistral at w4a16) is the only translation-adjacent family.

So the catalog itself offers this project nothing we can ship. The **platform** is a different matter.

## The revision

> **AI Hub is still not a latency budget line. It is a portability and quantization-QA lab.**

Three things it can do that we currently cannot do at all:

1. **Prove our binaries load on other silicon.** `qnn_context_binary` is SoC-specific — the classic failure is `Failed to create context from binary … 0x138d` on an architecture mismatch. Every binary this project has built is **v73 / SM8550**. The brief's supported-hardware list includes RB3 Gen 2 (QCS6490, **v68-class**), IQ-9075 (v73-class), 8 Gen 3 (**v75**) and X Elite. Compiling per target turns the submission's minimum-requirements table from *estimated* into *tested*, which the template explicitly demands ("do not imply untested compatibility").
2. **Check quantization damage on our own graphs.** A `numerics` job compares w8a16 against float for a given model. We have never verified that our int8 MT encoder/decoder lost nothing measurable — the MT quality numbers came from the int8 model, but the *delta* against float is unknown.
3. **Detect silent CPU fallback.** A profile job reports offload percentage and subgraph counts. Ops the HTP partitioner cannot take fall back **silently**. Our 6.43 ms encoder and 11.97 ms decoder step have never been checked for partial fallback, and a partial fallback would be hidden inside those numbers.

## Gotchas worth writing down before we use it

- **One binary per chipset, always.** Architecture mismatch is the number-one cause of load failure.
- **Pin `--qairt_version`** on compile *and* profile; a runtime older than the compile QAIRT fails.
- **W8A16 does not convert on v68** (RB3 Gen 2). That class needs full INT8 (`--quantize_full_type int8 --quantize_io`), and float graphs are rejected outright.
- **Never import externally quantized QDQ.** The HTP importer ignores foreign scales; measured consequence in the field is top-1 collapsing from 0.75 to 0.005 while compile and profile both look healthy. Use the hosted quantize job.
- **Multi-graph binaries need `context_enable_graphs`** at runtime.
- **Compile success is not correctness.** Always follow with a profile and an inference job compared against float output.

## What to do inside the Phase-3 window

Ordered by submission value, all of it cheap:

| # | action | why it matters | effort |
|---|---|---|---|
| 1 | Compile the ASR encoder/decoder/joiner, MT encoder/decoder for **8 Gen 3 (v75)** | makes the minimum-requirements row *tested* for the most likely judge hardware | hours |
| 2 | Profile those binaries, read the **offload %** | first check that our measured latencies are not hiding CPU fallback | hours |
| 3 | `numerics` on the MT decoder (w8a16 vs float) | the int8 quality delta is currently unverified | hours |
| 4 | Compile for RB3 Gen 2 (v68) with full INT8 | covers the brief's IoT board; the W8A16 failure is a known trap | ~1 day |

Not worth doing: profiling on AI Hub to predict device latency (the k≈3.2 / 4.5 calibration factors already show it measures a different quantity), and any catalog model — none of them is Vietnamese.

## Sources

- AI Hub model catalog and per-model device/latency tables: <https://aihub.qualcomm.com/models>
- Workbench API, `qnn_options`, device list: <https://workbench.aihub.qualcomm.com/docs/hub/api.html>
- `qualcomm/ai-hub-models` (authoritative model inventory, pinned by tag)
