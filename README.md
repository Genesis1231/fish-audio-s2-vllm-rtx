# Fish Studio — LAN TTS service (Fish Audio S2-Pro, GPU)

A resident text-to-speech service for the local network, powered by the
**Fish Audio S2-Pro** (4B Dual-AR) model on an NVIDIA GPU. Your programs POST
text + a voice name and get audio back — a full clip or streamed as it renders.

**Architecture (two parts):**
1. **vLLM-Omni backend** — a Docker container (`vllm serve /models/s2-pro --omni`)
   does the GPU heavy-lifting. On this RTX PRO 6000 Blackwell (sm_120) it streams
   first audio in **~150 ms** (vs ~1.1 s for the old in-process engine) and
   handles real concurrency, using a Triton kvcache attention path that works on
   sm_120 where SGLang's FA3 kernel could not.
2. **Proxy API** (`server.py`) — a lightweight HTTP front door (no GPU) that keeps
   our voice-name + `/generate` + `/stream` API, uploads voice profiles to the
   backend (trimmed to ≤30 s), and re-adds **breathing** (inter-sentence pauses)
   on the stream.

## What's here

| file | role |
|------|------|
| `server.py`        | FastAPI proxy API (the HTTP front door) |
| `vllm_backend.py`  | adapter to the vLLM-Omni container + voice upload + breathing |
| `run_vllm_omni.sh` | launch/resume the vLLM-Omni backend container |
| `run.sh`           | start everything (backend container, then proxy) |
| `config.py` / `config.json` | settings (host, port, vllm_url, default voice, breathing) |
| `voices/`          | `<name>.wav` + `<name>.txt` cloning profiles (`samantha`, `sample`) |
| `tests/`           | `play.py`, `vllm_bench.py`, `tune_stream.py`, `per_sentence_demo.py` |
| `fish_speech/`     | vendored upstream lib — the backend container mounts it and imports only its **DAC codec** (`models/dac`) |

## Setup

```bash
# 1. system libs (sudo) — for the proxy's audio encoding + test playback
sudo apt install -y ffmpeg libportaudio2

# 2. model weights -> ../models/s2-pro   (public, ungated)
#    https://huggingface.co/fishaudio/s2-pro

# 3. backend image: official vllm/vllm-omni:v0.22.0 + the DAC codec's deps
#    (descript-audio-codec stack) installed under a constraints file so vLLM's
#    torch 2.11 / transformers 5.8 / sm_120 kernels stay intact. Tagged
#    `vllm-omni-fish:local`. (See run_vllm_omni.sh for the exact mounts.)

# 4. run (starts the backend container, then the proxy)
./run.sh
```

The proxy binds `0.0.0.0:8765` (set in `config.json`), so any machine on the LAN
can reach it at `http://<this-host-ip>:8765`. The backend listens on `:8091`
(local only). `run.sh` is idempotent — it reuses a running backend container.

## Use it

```bash
# full clip
curl -s -X POST http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello from the GPU","voice":"samantha","format":"wav"}' --output out.wav

# streaming (low latency) — frames arrive as they render; pcm => first byte is first audio
curl -s -N -X POST http://127.0.0.1:8765/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"streaming hello","voice":"samantha","format":"pcm"}' --output out.pcm

# OpenAI-compatible (point any OpenAI client's base_url here)
curl -s -X POST http://127.0.0.1:8765/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hi there","voice":"samantha","response_format":"mp3"}' --output out.mp3
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.3.56:8765/v1", api_key="x")
client.audio.speech.create(model="s2-pro", voice="samantha",
                           input="hello").stream_to_file("out.mp3")
```

## API

`POST /generate` (full clip) and `POST /stream` (live) share this JSON body:

| field | default | notes |
|-------|---------|-------|
| `text` | — | required |
| `voice` | `default_voice` | a profile in `voices/` (uploaded to the backend by name) |
| `format` | `/generate`→`wav`, `/stream`→`pcm` | `wav` `pcm` `mp3` `opus` `flac` `ogg` `aac` (compressed need ffmpeg; `/stream` serves any — `pcm`/`wav` raw, compressed via a live ffmpeg pipe) |
| `stream_sentence_gap_ms` | `600` | breathing: inter-sentence pause target (0 = vLLM's natural pacing) |
| `initial_codec_chunk_frames` | from backend | vLLM first-chunk size (TTFA knob) |
| `speed` | `1.0` | 0.25–4× speech rate |
| `max_new_tokens` `seed` | from config | generation |
| `reference_audio` + `reference_text` | — | base64 wav + transcript for ad-hoc cloning (1–30 s of clear speech) |

**Low-latency streaming:** `/stream` makes one streaming request to the vLLM-Omni
backend (response_format `pcm`) and relays the audio as it arrives — first audio
in **~150 ms**. The backend keeps the cloned-voice reference cached and does its
own causal-overlap decode at chunk boundaries (`codec_left_context_frames`), so
the waveform is continuous (no per-segment onset).

**Breathing.** vLLM emits a short, *variable* pause between sentences. The proxy
detects those boundaries on the stream and pads each up to `stream_sentence_gap_ms`
(default 600 ms), with blip-bridging so a soft breath inside a pause doesn't defeat
detection. Set it lower for tighter pacing, or `0` to use vLLM's pacing as-is.
(Detection is heuristic — vLLM doesn't expose sentence markers — so pause lengths
still vary a little, which reads as natural.)

> **Consumers should jitter-buffer.** Best practice (per Fish Audio's own guide)
> is to buffer ~2–3 chunks before starting playback so generation jitter never
> causes a gap. `tests/play.py` shows the pattern: a 0.5 s lead buffer feeding a
> callback that outputs silence on underrun rather than stalling.

`/stream` sets `X-Accel-Buffering: no` + `Cache-Control: no-cache` so proxies/clients
don't buffer frames. Inline emotion tags work in `text`: `[whisper]`, `[excited]`,
`[laughing]`, `[sigh]`, `[angry]`, and 15,000+ free-form tags like `[professional broadcast tone]`.

Live playback demo: `python tests/play.py --mode both`. Other routes: `GET /health`,
`GET /voices`, `POST /voices/reload`, OpenAI-compatible `POST /v1/audio/speech`.

## Add a voice

Two files in `voices/`: `name.wav` (clean 10–30s mono clip) + `name.txt`
(its exact transcript). Then `curl -XPOST .../voices/reload` (or restart).

## Config

`config.json`: `host`, `port`, `vllm_url` (backend, default `http://127.0.0.1:8091`),
`ref_max_seconds` (voice trim cap, 28), `default_voice`, `api_key` (set to require
auth on the LAN), and `defaults` (`stream_sentence_gap_ms`, `max_new_tokens`, `format`).

> **Prefix caching** (a vLLM TTFA knob) is intentionally left **off**: enabling it
> on the Fish model in vLLM-Omni 0.22 intermittently produces empty audio. TTFA is
> already at its ~150 ms floor; `initial_codec_chunk_frames` does not move it.
