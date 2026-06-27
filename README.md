# Self-Hosted Streaming TTS & Voice Cloning on RTX 5090 / RTX PRO 6000 — Fish Audio S2-Pro + vLLM

> **Low-latency (~100 ms time-to-first-audio) text-to-speech and instant voice cloning** powered by [Fish Audio's **OpenAudio S2-Pro**](https://huggingface.co/fishaudio/s2-pro) and served with **vLLM-Omni** — a private, self-hosted alternative to ElevenLabs and OpenAI TTS that runs on new **NVIDIA Blackwell** GPUs (`sm_120`, RTX 5090 / RTX PRO 6000) the upstream recipe doesn't cover.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)
![vLLM-Omni](https://img.shields.io/badge/vLLM--Omni-0.22-FF6F00)
![GPU](https://img.shields.io/badge/GPU-RTX%205090%20%7C%20RTX%20PRO%206000%20(Blackwell%20sm__120)-76B900?logo=nvidia&logoColor=white)
![OpenAI API](https://img.shields.io/badge/API-OpenAI--compatible-412991?logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-Fish%20Audio%20Research%20(non--commercial)-blue)
![Stars](https://img.shields.io/github/stars/Genesis1231/fish-audo-s2-vllm-RTX?style=flat&logo=github)

**Keywords:** `text-to-speech` · `tts` · `streaming-tts` · `voice-cloning` · `self-hosted` · `openaudio-s2` · `fish-speech` · `vllm` · `vllm-omni` · `rtx-5090` · `rtx-pro-6000` · `blackwell` · `sm120` · `real-time` · `low-latency` · `openai-compatible` · `elevenlabs-alternative`

---

## Table of Contents

- [Why this exists](#-why-this-exists)
- [Features](#-features)
- [Supported GPUs](#-supported-gpus)
- [Benchmarks](#-benchmarks)
- [Quick start](#-quick-start)
- [Usage (curl · Python · OpenAI SDK)](#-usage)
- [API reference](#-api-reference)
- [Voice cloning](#-voice-cloning)
- [Configuration](#-configuration)
- [How it works (architecture)](#-how-it-works)
- [Comparison vs alternatives](#-comparison-vs-alternatives)
- [FAQ](#-faq)
- [License & credits](#-license--credits)

---

## 🛠 Why this exists

vLLM-Omni can already serve S2-Pro (`vllm serve fishaudio/s2-pro --omni`), but that's a **bare inference endpoint**, and the standard way to add S2-Pro's DAC codec to it **breaks on new RTX 5090 / RTX PRO 6000 Blackwell** cards (FlashAttention-3 ships kernels only for Hopper `sm90a` / Blackwell-Ultra `sm120a`, and SGLang is blocked on `sm_120`). This project ships a working recipe **plus** a production-grade serving layer on top:

- a **container build** that pins the DAC codec deps against vLLM-Omni 0.22's `torch 2.11 + cu130 + sm_120` kernels, and
- a **lightweight FastAPI proxy** that adds named voices, "breathing", live audio encoding, and an OpenAI-compatible API.

The result: **your own private ElevenLabs / OpenAI-TTS-style endpoint** on hardware you control — no per-character billing, no data leaving the box.

---

## ✨ Features

- **🟩 Runs on RTX 5090 / RTX PRO 6000 (Blackwell, `sm_120`)** — vLLM-Omni 0.22 ships `torch 2.11+cu130` with the right kernels; this repo's container build keeps the codec compatible.
- **🎙️ Streaming & full-clip synthesis** — `/stream` delivers audio as it renders (**~100 ms TTFA**); `/generate` returns a complete clip.
- **🧬 Instant voice cloning** — clone any voice from a few seconds of reference audio: persistent named profiles **or** one-off ad-hoc cloning per request.
- **🎵 Seven output formats, all live** — `wav` · `pcm` · `mp3` · `opus` · `flac` · `ogg` · `aac` through a live ffmpeg pipe (the bare engine streams only `pcm`/`wav` and can't encode `opus`/`aac`/`ogg` at all).
- **🤖 OpenAI-compatible** — point any OpenAI SDK at `/v1/audio/speech` and it just works (`input`, `voice`, `response_format`, `speed`, `stream`).
- **🌬️ Breathing** — detects sentence boundaries on the live stream and pads each pause to an even, natural length (the raw engine's inter-sentence pauses are short and erratic).
- **🎭 Emotion & style tags** — inline `[whisper]`, `[excited]`, `[angry]`, … plus free-form style descriptors.
- **🧯 Production hardening** — truthful HTTP status codes (400 / 404 / 502 / 503), a real readiness probe, failure-driven backend liveness recovery, and concurrency that scales.

---

## 🖥 Supported GPUs

Tested on the RTX PRO 6000 Blackwell; any recent NVIDIA data-center or RTX card that vLLM-Omni 0.22 supports should work.

| GPU | Arch | Compute | Status |
|-----|------|---------|--------|
| **RTX PRO 6000 Blackwell** | Blackwell | `sm_120` | ✅ Verified (dev box) |
| **RTX 5090 / 5080** | Blackwell | `sm_120` | ✅ Expected (same kernel path) |
| RTX 4090 / 4080 / 6000 Ada | Ada | `sm_89` | ✅ Supported by vLLM-Omni |
| A100 / H100 | Ampere / Hopper | `sm_80` / `sm_90` | ✅ Supported by vLLM-Omni |

> Why Blackwell is special: FlashAttention-3 has no `sm_120` kernel and SGLang is blocked there, but **vLLM-Omni's Triton "fish kvcache" decode path runs on `sm_120`** — which is exactly what makes this work on RTX 5090 / RTX PRO 6000.

---

## ⚡ Benchmarks

Measured **warm**, 44.1 kHz mono, **NVIDIA RTX PRO 6000 Blackwell**:

| Metric | Value |
|--------|-------|
| Time to first audio — **zero-shot** | **~100 ms** |
| Time to first audio — **voice clone** | **~145 ms** |
| Real-time factor — 1 stream | ~2.8× |
| Real-time factor — 4 concurrent streams | ~7× aggregate |

> Zero-shot is faster than cloning because a named/ad-hoc clone first conditions on the reference voice tokens; the default speaker skips that step.
> Prefix caching is intentionally **off** — it intermittently produces empty audio in vLLM-Omni 0.22.

---

## 🚀 Quick start

**Requirements:** Linux · NVIDIA GPU + recent driver · Docker (NVIDIA Container Toolkit) · ~12 GB disk for the S2-Pro weights.

```bash
# 1. System deps (proxy audio encoding + test playback)
sudo apt install -y ffmpeg libportaudio2

# 2. Download the model weights (public, ungated)
#    https://huggingface.co/fishaudio/s2-pro  ->  ../models/s2-pro

# 3. Build the backend image: vllm/vllm-omni:v0.22.0 + DAC codec deps,
#    tagged vllm-omni-fish:local  (see run_vllm_omni.sh for the exact setup)

# 4. Start everything (idempotent — reuses a running backend container)
./run.sh
```

The proxy binds `0.0.0.0:8765`; the vLLM-Omni backend listens on `:8091` (local only).

---

## 🔧 Usage

### curl

```bash
# Full clip
curl -s -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello from my own GPU","voice":"samantha","format":"wav"}' --output out.wav

# Live stream — pcm is lowest latency; first byte is first audio
curl -s -N -X POST http://127.0.0.1:8765/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"streaming hello","voice":"samantha","format":"pcm"}' --output out.pcm

# OpenAI-compatible endpoint
curl -s -X POST http://127.0.0.1:8765/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hi there","voice":"samantha","response_format":"mp3"}' --output out.mp3
```

### Python (in-process — no proxy)

Drive the engine straight from your own Python process — no FastAPI proxy needed. You still need the vLLM-Omni **backend** container running (`./run_vllm_omni.sh`, that's where the GPU model lives), but `server.py` doesn't have to be up.

```python
import soundfile as sf
from vllm_backend import engine          # run from the repo root (or put it on PYTHONPATH)

engine.load()                            # connect to the backend + upload voices/ (idempotent)

# Full clip -> float32 numpy array @ 44.1 kHz
audio, sr = engine.generate("hello from my own GPU", voice=None, params={})
sf.write("out.wav", audio, sr)           # voice=None -> built-in default (zero-shot)

# Stream as it renders, with a cloned voice (needs voices/samantha.{wav,txt})
# (engine.generate_stream yields float32 PCM; write wav — soundfile mp3 support is
#  build-dependent. For compressed output, go through the proxy's ffmpeg pipe.)
with sf.SoundFile("stream.wav", "w", samplerate=engine.sample_rate, channels=1) as f:
    for chunk in engine.generate_stream("a streamed, cloned hello", voice="samantha", params={}):
        f.write(chunk)
```

### OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host-ip>:8765/v1", api_key="x")

# Full clip
client.audio.speech.create(
    model="s2-pro", voice="samantha", input="hello",
).write_to_file("out.mp3")

# Low-latency streaming
with client.audio.speech.with_streaming_response.create(
    model="s2-pro", voice="samantha", input="streamed hello",
    response_format="pcm", extra_body={"stream": True},
) as resp:
    resp.stream_to_file("out.pcm")
```

> Any `response_format` streams (`mp3`/`opus` are encoded live via ffmpeg); `pcm` or `wav` give the lowest first-audio latency.

---

## 📡 API reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Readiness probe — 503 until backend is up; reports model/GPU/voices |
| `GET` | `/voices` | List available voice profiles |
| `POST` | `/voices/reload` | Re-scan `voices/` without a restart |
| `POST` | `/generate` | Render a full audio clip (any format) |
| `POST` | `/stream` | Live audio stream as it renders (`pcm` = lowest latency) |
| `POST` | `/v1/audio/speech` | OpenAI-compatible TTS (`input`, `voice`, `response_format`, `speed`, `stream`) |

### Request body (`/generate` and `/stream`)

| Field | Default | Notes |
|-------|---------|-------|
| `text` | — | Required |
| `voice` | `default_voice` | Named profile in `voices/`; `""` = zero-shot built-in |
| `format` | `wav` / `pcm` | `wav` `pcm` `mp3` `opus` `flac` `ogg` `aac`; compressed formats require ffmpeg |
| `speed` | `1.0` | Speech rate multiplier: 0.25–4× |
| `stream_sentence_gap_ms` | `600` | Breathing: pad inter-sentence pauses *up to* this ms (only lengthens, never shortens; values below ~240 ms have little effect); `0` = off |
| `initial_codec_chunk_frames` | backend default | TTFA tuning knob (smaller = lower first-audio latency) |
| `max_new_tokens` | `4096` | Caps output length (~200 s of audio at 4096). Longer inputs are silently truncated at the cap — raise for very long text |
| `seed` | — | Reproducible generation |
| `reference_audio` | — | Base64 WAV for ad-hoc voice cloning (≥1 s of clear speech; clips longer than ~28 s are auto-trimmed) |
| `reference_text` | — | Transcript of `reference_audio`; the two must be sent together (both or neither) |

### Streaming notes

- `/stream` sets `X-Accel-Buffering: no` and `Cache-Control: no-cache` — nginx/Caddy won't buffer frames.
- `pcm` = raw s16le mono @ 44.1 kHz; first byte is first audio (no header overhead).
- **Jitter-buffer** ~2–3 chunks before starting playback — `tests/play.py` shows the pattern (0.5 s lead buffer, silence on underrun).
- Compressed formats (`mp3`, `opus`, `flac`, …) stream via a live ffmpeg pipe; no full-clip buffering.

### Emotion & style tags

Embed inline in `text`:

```
[whisper] this is a secret [excited] and this part is loud!
[professional broadcast tone] Welcome to the evening news.
```

Supports `[whisper]`, `[excited]`, `[laughing]`, `[sigh]`, `[angry]`, plus 15,000+ free-form style descriptors.

---

## 🎙️ Voice cloning

### Three voice modes

| Mode | Request | Notes |
|------|---------|-------|
| **Default** | – | Fish Audio's built-in default; no reference needed; ~100 ms TTFA |
| **Named clone** | `voice: "samantha"` | Persistent identity loaded from `voices/samantha.{wav,txt}` |
| **Ad-hoc clone** | `reference_audio` + `reference_text` | One-off; base64 WAV + exact transcript (≥1 s of clear speech; auto-trimmed to ~28 s) |

> When `voice` is **omitted**, the service uses `default_voice` from `config.json`.
> Send `voice: ""` explicitly to select zero-shot.

### Add a named voice

1. Add `voices/<name>.wav` — a clean 10–30 s mono recording.
2. Add `voices/<name>.txt` — its exact transcript.
3. Reload without restart: `curl -X POST http://127.0.0.1:8765/voices/reload`

---

## ⚙️ Configuration

`config.json` (every key is also overridable by an env var, e.g. `FISH_PORT`):

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `"0.0.0.0"` | Proxy bind address |
| `port` | `8765` | Proxy port |
| `vllm_url` | `"http://127.0.0.1:8091"` | vLLM-Omni backend URL |
| `ref_max_seconds` | `28` | Voice reference trimmed to this before upload |
| `default_voice` | `"samantha"` | Voice used when `voice` is omitted |
| `api_key` | `null` | Set to require bearer / `X-Api-Key` auth |
| `defaults.format` | `"wav"` | Default format for `/generate` |
| `defaults.max_new_tokens` | `4096` | Default generation token limit (~200 s of audio) |
| `defaults.stream_sentence_gap_ms` | `600` | Default breathing pause (ms) |

---

## 🧩 How it works

A **two-part proxy** architecture:

```
client ──HTTP──> server.py (FastAPI proxy, no GPU)  ──HTTP──> vLLM-Omni container (GPU)
                 • named voices + cloning                     • OpenAudio S2-Pro (4B dual-AR)
                 • breathing (pause padding)                  • DAC codec @ 44.1 kHz
                 • live ffmpeg encoding                        • Triton "fish kvcache" attention
                 • OpenAI-compatible API                       • streams raw PCM
```

- **OpenAudio S2-Pro** is a dual autoregressive model (a 4B "slow" AR + a small "fast" AR over RVQ codebooks) decoded by a **DAC** neural codec to 44.1 kHz audio.
- **vLLM-Omni** runs the model with continuous batching and a Triton decode-attention fast path that works on Blackwell `sm_120`.
- **`server.py`** is a thin, GPU-less FastAPI layer (`vllm_backend.py` adapts it to the container over HTTP) — it owns the voice-name API, breathing, live audio encoding, and OpenAI compatibility, so it stays cheap and restartable.

---

## 🆚 Comparison vs alternatives

| | **This project** | ElevenLabs | OpenAI TTS | Bare `vllm serve --omni` |
|---|:--:|:--:|:--:|:--:|
| Self-hosted / private | ✅ | ❌ cloud | ❌ cloud | ✅ |
| Per-use cost | free (your GPU) | $$ / char | $$ / char | free |
| Instant voice cloning | ✅ | ✅ | ❌ | ✅ (raw) |
| Streaming TTFA | ~100 ms | low | low | ~100 ms |
| OpenAI-compatible API | ✅ | ❌ | n/a | partial |
| `mp3`/`opus`/`aac` live encode | ✅ | ✅ | ✅ | ❌ pcm/wav only |
| Named voices + hot-reload | ✅ | ✅ | preset only | ❌ |
| Runs on RTX 5090 / PRO 6000 | ✅ | n/a | n/a | ⚠️ codec breaks |

---

## ❓ FAQ

**Does Fish Audio S2 / OpenAudio S2-Pro run on the RTX 5090 or RTX PRO 6000 (Blackwell)?**
Yes. The upstream codec recipe breaks on `sm_120`, but vLLM-Omni 0.22's Triton kvcache attention path works — this repo packages the compatible build.

**Is this a self-hosted ElevenLabs / OpenAI TTS alternative?**
Yes — same shape (streaming TTS + voice cloning + an OpenAI-compatible `/v1/audio/speech`), but it runs entirely on your own GPU, so audio and voice profiles never leave the machine and there's no per-character billing.

**What's the latency?**
~100 ms time-to-first-audio for the built-in voice and ~145 ms for a clone, warm, on an RTX PRO 6000. Streaming real-time factor is ~2.8× single / ~7× aggregate at 4 concurrent streams.

**Can I use the OpenAI Python/Node SDK?**
Yes. Point `base_url` at `http://<host>:8765/v1` and call `audio.speech.create(...)`; set `extra_body={"stream": True}` for low-latency streaming.

**Why vLLM-Omni and not SGLang or FlashAttention-3?**
FA3 has no `sm_120` kernel and SGLang is blocked on Blackwell; vLLM-Omni's Triton decode path is the route that actually runs there.

**Can it clone a voice from a short sample?**
Yes — ≥1 s of clean speech plus its transcript, either as a persistent named profile or one-off per request.

**Which audio formats can it stream?** `pcm`, `wav`, `mp3`, `opus`, `flac`, `ogg`, `aac` — compressed formats are encoded live through ffmpeg.

---

## 📦 Tech stack

`OpenAudio S2-Pro` · `Fish Speech` · `vLLM-Omni 0.22` · `DAC neural codec` · `FastAPI` · `Triton` · `PyTorch 2.11 (cu130)` · `Docker` · `ffmpeg` · `NVIDIA Blackwell sm_120`

## 📄 License & credits

- **Code in this repo:** see [`LICENSE`](LICENSE) — the **Fish Audio Research License Agreement**. Research and **non-commercial** use is free; **commercial use requires a separate license from Fish Audio**.
- **Model:** [OpenAudio S2-Pro](https://huggingface.co/fishaudio/s2-pro) by [Fish Audio](https://fish.audio/).
- **Serving:** [vLLM](https://github.com/vllm-project/vllm) / vLLM-Omni.
- The vendored `fish_speech/` is included only for the DAC codec the backend imports; model weights and `.venv` live outside the tree and are gitignored.

---

<sub>Self-hosted streaming text-to-speech (TTS) and voice cloning with Fish Audio OpenAudio S2-Pro on vLLM-Omni — low-latency, OpenAI-compatible, and running on NVIDIA Blackwell RTX 5090 / RTX PRO 6000 (sm_120). A private, open ElevenLabs / OpenAI-TTS alternative.</sub>
