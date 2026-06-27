# Self-Hosted Streaming TTS & Voice Cloning on RTX 5090 / RTX PRO 6000 — Fish Audio S2-Pro + vLLM

> **Low-latency (~100 ms time-to-first-audio) text-to-speech and instant voice cloning** powered by [Fish Audio's **OpenAudio S2-Pro**](https://huggingface.co/fishaudio/s2-pro) and served with [**vLLM-Omni**](https://github.com/vllm-project/vllm-omni) — a private, self-hosted alternative to ElevenLabs and OpenAI TTS that runs on new **NVIDIA Blackwell** GPUs (`sm_120`, RTX 5090 / RTX PRO 6000) the upstream recipe doesn't cover.


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
| **RTX PRO 6000** | Blackwell | `sm_120` | ✅ Verified (dev box) |
| **RTX 5090 / 5080** | Blackwell | `sm_120` | ✅ Verified |
| RTX 4090 / 4080 / 6000 Ada | Ada | `sm_89` | ✅ Supported by vLLM-Omni |

> Why Blackwell is special: FlashAttention-3 has no `sm_120` kernel and SGLang is blocked there, but **vLLM-Omni's Triton "fish kvcache" decode path runs on `sm_120`** — which is exactly what makes this work on RTX 5090 / RTX PRO 6000.

---

## ⚡ Benchmarks

Measured **warm**, 44.1 kHz mono, **NVIDIA RTX PRO 6000**:

| Metric | Value |
|--------|-------|
| Time to first audio — **zero-shot** | **~100 ms** |
| Time to first audio — **voice clone** | **~135 ms** |
| Real-time factor — 1 stream | ~2.8× |
| Real-time factor — 4 concurrent streams | ~7× aggregate |

> Prefix caching is intentionally **off** — it intermittently produces empty audio in vLLM-Omni 0.22.

---

## 🚀 Quick start

**Requirements:** Linux · NVIDIA GPU + recent driver · Docker (NVIDIA Container Toolkit) · ~12 GB disk for the S2-Pro weights.

```bash
# 1. System deps and the lightweight proxy environment
sudo apt update
sudo apt install -y curl ffmpeg libportaudio2 python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install fastapi uvicorn numpy requests \
  huggingface-hub sounddevice

# 2. Download the public, ungated model into the path the launcher expects
mkdir -p ../models/s2-pro
.venv/bin/hf download fishaudio/s2-pro --local-dir ../models/s2-pro

# 3. Extend the official image with Fish Audio's DAC codec dependencies.
#    The constraints preserve vLLM-Omni's Blackwell-compatible core packages.
docker pull vllm/vllm-omni:v0.22.0
docker build -f Dockerfile.vllm -t vllm-omni-fish:local .

# 4. Start the backend and proxy (idempotent; reuses a running container)
./run.sh
```

The proxy binds `0.0.0.0:8765`; the vLLM-Omni backend listens on `:8091` (local only).

---

## 🔧 Usage

### curl

```bash
# Full clip
curl -s -X POST http://localhost:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello from my own GPU","voice":"samantha","format":"wav"}' --output out.wav

# Live stream — pcm is lowest latency; first byte is first audio
curl -s -N -X POST http://localhost:8765/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"streaming hello","voice":"samantha","format":"pcm"}' --output out.pcm

# OpenAI-compatible endpoint
curl -s -X POST http://localhost:8765/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hi there","voice":"samantha","response_format":"mp3"}' --output out.mp3
```

### Python (in-process, no FastAPI)

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
client = OpenAI(base_url="http://localhost:8765/v1", api_key="x")

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
3. Reload without restart: `curl -X POST http://localhost:8765/voices/reload`

---

## ⚙️ Configuration

`config.json` (every key is also overridable by an env var, e.g. `FISH_PORT`):

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `"0.0.0.0"` | Proxy bind address |
| `port` | `8765` | Proxy port |
| `vllm_url` | `"http://localhost:8091"` | vLLM-Omni backend URL |
| `ref_max_seconds` | `28` | Voice reference trimmed to this before upload |
| `default_voice` | `"samantha"` | Voice used when `voice` is omitted |
| `api_key` | `null` | Set to require bearer / `X-Api-Key` auth |
| `defaults.format` | `"wav"` | Default format for `/generate` |
| `defaults.max_new_tokens` | `4096` | Default generation token limit (~200 s of audio) |
| `defaults.stream_sentence_gap_ms` | `600` | Default breathing pause (ms) |

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

## 📦 Tech stack

`OpenAudio S2-Pro` · `Fish Speech` · `vLLM-Omni 0.22` · `DAC neural codec` · `FastAPI` · `Triton` · `PyTorch 2.11 (cu130)` · `Docker` · `ffmpeg` · `NVIDIA Blackwell sm_120`

## 📄 Credits

- **Model:** [OpenAudio S2-Pro](https://huggingface.co/fishaudio/s2-pro) by [Fish Audio](https://fish.audio/).
- **Serving:** [vLLM](https://github.com/vllm-project/vllm) / [vLLM-Omni](https://github.com/vllm-project/vllm-omni).
- The vendored `fish_speech/` is included only for the DAC codec the backend imports; model weights and `.venv` live outside the tree and are gitignored.

---

<sub>Self-hosted streaming text-to-speech (TTS) and voice cloning with Fish Audio OpenAudio S2-Pro on vLLM-Omni — low-latency, OpenAI-compatible, and running on NVIDIA Blackwell RTX 5090 / RTX PRO 6000 (sm_120). A private, open ElevenLabs / OpenAI-TTS alternative.</sub>
