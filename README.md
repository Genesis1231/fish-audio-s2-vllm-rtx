# Self-Hosted Streaming TTS with Fish S2 Pro - 100ms TTFA on RTX 5090/6000

A self-hosted streaming text-to-speech service wrapping **Fish Audio's OpenAudio S2-Pro** (4B dual-AR) on a local NVIDIA GPU — including new **Blackwell cards (RTX 5090, RTX PRO 6000)** that the upstream recipe doesn't cover. POST text and a voice name, get back a full audio clip or a live stream — first audio in ~100 ms. A self-hosted alternative to ElevenLabs or OpenAI TTS that keeps your data and voice profiles private.

---

## 🛠️ What is this?

vLLM-Omni can already serve S2-Pro (`vllm serve fishaudio/s2-pro --omni`), but that's a bare inference endpoint and the standard way to add S2-Pro's codec to it breaks on new **RTX 5090 / RTX PRO 6000 Blackwell** cards. This repo includes:

- **🟩 Runs smoothly on RTX 5090 / RTX PRO 6000** — vLLM-Omni 0.22 ships `torch 2.11+cu130` with the `sm_120` kernels. 
- 🎙️ **Streaming & full-clip synthesis** — `/stream` delivers audio as it renders (~100 ms TTFA); `/generate` returns a complete clip
- 🎵 **Seven output formats, all live** — `wav` `pcm` `mp3` `opus` `flac` `ogg` `aac` through a live ffmpeg pipe; the bare engine streams only `pcm`/`wav` and can't encode `opus`/`aac`/`ogg` at all
- **🌬️ Breathing** — detects sentence boundaries on the live stream and pads each pause to an even, natural length (the raw engine's inter-sentence pauses are short and erratic)
- **🗂️ Voice management** — name-based profiles with automatic upload, reference trimming, and hot-reload, instead of a base64 reference attached to every request
- **🧯 Production hardening** — OpenAI-compatible front, truthful HTTP status codes (400 / 404 / 503 / 502), a real readiness probe, and cold-start recovery

---

## ⚡ Benchmarks

Measured **warm**, 44.1 kHz mono, **NVIDIA RTX PRO 6000**:

| Metric | Value |
|--------|-------|
| Time to first audio — zero-shot | ~100 ms |
| Time to first audio — voice clone | ~145 ms |
| Real-time factor — 1 stream | ~2.8× |
| Real-time factor — 4 concurrent streams | ~7× aggregate |

> Prefix caching is intentionally **off** — it intermittently produces empty audio in vLLM-Omni 0.22.  

---

## 🚀 Quick Start

```bash
# 1. System deps (proxy audio encoding + test playback)
sudo apt install -y ffmpeg libportaudio2

# 2. Download model weights (public, ungated)
#    https://huggingface.co/fishaudio/s2-pro  ->  ../models/s2-pro

# 3. Build the backend image: vllm/vllm-omni:v0.22.0 + DAC codec deps, tagged vllm-omni-fish:local
#    (see run_vllm_omni.sh for the exact container setup)

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

## 📡 API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Readiness probe — 503 until backend is up; reports model/GPU/voices |
| `GET` | `/voices` | List available voice profiles |
| `POST` | `/voices/reload` | Re-scan `voices/` without a restart |
| `POST` | `/generate` | Render full audio clip (any format) |
| `POST` | `/stream` | Live audio stream as it renders (`pcm` = lowest latency) |
| `POST` | `/v1/audio/speech` | OpenAI-compatible TTS (`input`, `voice`, `response_format`, `speed`, `stream`) |

### Request body (`/generate` and `/stream`)

| Field | Default | Notes |
|-------|---------|-------|
| `text` | — | Required |
| `voice` | `default_voice` | Named profile in `voices/`; `""` = zero-shot built-in |
| `format` | `wav` / `pcm` | `wav` `pcm` `mp3` `opus` `flac` `ogg` `aac`; compressed formats require ffmpeg |
| `speed` | `1.0` | Speech rate multiplier: 0.25–4× |
| `stream_sentence_gap_ms` | `600` | Breathing: pad inter-sentence pauses *up to* this ms (only lengthens, never shortens; values below ~240 ms have little effect); `0` = off, model's natural pacing |
| `initial_codec_chunk_frames` | backend default | TTFA tuning knob (smaller = lower first-audio latency) |
| `max_new_tokens` | `4096` | Caps output length (~200 s of audio at 4096). Longer inputs are silently truncated at the cap — raise it for very long text |
| `seed` | — | Reproducible generation |
| `reference_audio` | — | Base64 WAV for ad-hoc voice cloning (≥1 s of clear speech; clips longer than ~28 s are auto-trimmed) |
| `reference_text` | — | Transcript of `reference_audio`; the two must be sent together (both or neither) |

### Streaming notes

- `/stream` sets `X-Accel-Buffering: no` and `Cache-Control: no-cache` — nginx/Caddy won't buffer frames
- `pcm` = raw s16le mono @ 44.1 kHz; first byte is first audio (no header overhead)
- **Jitter-buffer** ~2–3 chunks before starting playback — `tests/play.py` shows the pattern (0.5 s lead buffer, silence on underrun)
- Compressed formats (`mp3`, `opus`, `flac`, …) stream via a live ffmpeg pipe; no full-clip buffering

### Emotion & style tags

Embed inline in `text`:

```
[whisper] this is a secret [excited] and this part is loud!
[professional broadcast tone] Welcome to the evening news.
```

Supports `[whisper]`, `[excited]`, `[laughing]`, `[sigh]`, `[angry]`, plus 15,000+ free-form style descriptors.

---

## 🎙️ Voices

### Three voice modes

| Mode | Request | Notes |
|------|---------|-------|
| **Default** | - | Fish Audio's built-in default; no reference needed; ~100 ms TTFA |
| **Named clone** | `voice: "samantha"` | Persistent identity loaded from `voices/samantha.{wav,txt}` |
| **Ad-hoc clone** | `reference_audio` + `reference_text` | One-off; base64 WAV + exact transcript (≥1 s of clear speech; auto-trimmed to ~28 s) |

> When `voice` is **omitted**, the service uses `default_voice` from `config.json`.  
> Send `voice: ""` explicitly to select zero-shot.

### Add a voice

1. Add `voices/<name>.wav` — clean 10–30 s mono recording
2. Add `voices/<name>.txt` — its exact transcript
3. Reload without restart: `curl -X POST http://127.0.0.1:8765/voices/reload`

---

## ⚙️ Configuration

`config.json`:

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


