"""Fish Studio — a LAN TTS service powered by Fish Audio S2-Pro on the GPU.

A resident HTTP service your programs call over the local network: POST text +
a voice name, get audio back (full clip or streamed as it generates).

Endpoints
  GET  /                 service info
  GET  /health           readiness + model/gpu/voices
  GET  /voices           available voice profiles
  POST /voices/reload    re-scan voices/ without a restart
  POST /generate         JSON -> full audio clip (any format)
  POST /stream           JSON -> audio streamed as it renders (pcm lowest-latency; wav/mp3/… too)
  POST /v1/audio/speech  OpenAI-compatible TTS (so OpenAI SDKs work unchanged)

Run:  ./run.sh           (or: uvicorn server:app --host 0.0.0.0 --port 8765)
"""

import functools
import io
import shutil
import struct
import subprocess
import threading
import wave
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import config
from config import API_KEY, DEFAULTS, DEFAULT_VOICE, logger
from vllm_backend import BackendError, UnknownVoiceError, engine   # vLLM-Omni proxy backend

# ---- audio encoding ------------------------------------------------------
FFMPEG = shutil.which("ffmpeg")

# ffmpeg output args per compressed format (fed s16le mono on stdin).
_FFMPEG_ARGS = {
    "mp3":  ["-f", "mp3", "-b:a", "192k"],
    "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "96k"],
    "flac": ["-f", "flac"],
    "ogg":  ["-f", "ogg", "-c:a", "libvorbis"],
    "aac":  ["-f", "adts", "-c:a", "aac", "-b:a", "192k"],
}
MEDIA_TYPES = {
    "wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg",
    "opus": "audio/ogg", "flac": "audio/flac", "ogg": "audio/ogg", "aac": "audio/aac",
}
# Tell reverse proxies (nginx/Caddy) and clients not to buffer the stream, so
# audio frames are delivered the instant they're produced (lower perceived latency).
STREAM_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}


def _pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1,1] mono -> little-endian int16 PCM bytes."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _wav_header(sample_rate: int) -> bytes:
    """A 44-byte WAV header for a stream of unknown length (mono, 16-bit).

    The RIFF and data chunk sizes are the sentinel 0xFFFFFFFF ("until end of
    stream"), NOT 0 — a 0 tells strict players the file holds no audio, so they
    play nothing. The sentinel makes the streamed clip a valid, playable WAV.
    """
    byte_rate = sample_rate * 1 * 2          # channels * bytes-per-sample
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, byte_rate, 2, 16,
        b"data", 0xFFFFFFFF,
    )


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(_pcm16(audio))
    return buf.getvalue()


def _ffmpeg_cmd(sample_rate: int, fmt: str) -> list[str]:
    """ffmpeg argv to transcode s16le mono PCM (on stdin) to `fmt` (on stdout)."""
    return [FFMPEG, "-loglevel", "error", "-f", "s16le", "-ar", str(sample_rate),
            "-ac", "1", "-i", "pipe:0", *_FFMPEG_ARGS[fmt], "pipe:1"]


def _ffmpeg_encode(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    # Callers reach this only via encode_audio after _validate_format has confirmed
    # ffmpeg is present, so no FFMPEG guard here (it owns that check).
    proc = subprocess.run(
        _ffmpeg_cmd(sample_rate, fmt),
        input=_pcm16(audio), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise HTTPException(500, f"ffmpeg failed: {proc.stderr.decode()[:200]}")
    return proc.stdout


def encode_audio(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    fmt = fmt.lower()
    if fmt == "wav":
        return _wav_bytes(audio, sample_rate)
    if fmt == "pcm":
        return _pcm16(audio)
    if fmt in _FFMPEG_ARGS:
        return _ffmpeg_encode(audio, sample_rate, fmt)
    raise HTTPException(400, f"unsupported format '{fmt}'")


def _validate_format(fmt: str) -> str:
    """Reject a bad/unsupported format BEFORE spending GPU time generating
    (encode_audio would otherwise only fail after the whole clip is rendered)."""
    fmt = fmt.lower()
    if fmt not in MEDIA_TYPES:
        raise HTTPException(400, f"unsupported format '{fmt}'; use {sorted(MEDIA_TYPES)}")
    if fmt in _FFMPEG_ARGS and not FFMPEG:
        raise HTTPException(503, f"ffmpeg not installed; cannot encode '{fmt}' (use wav/pcm)")
    return fmt


# ---- request models ------------------------------------------------------
class SpeakBody(BaseModel):
    text: str
    voice: Optional[str] = None             # named profile; None -> default; "" -> zero-shot
    format: Optional[str] = None            # wav|pcm|mp3|opus|flac|ogg|aac
    max_new_tokens: Optional[int] = None
    stream_sentence_gap_ms: Optional[int] = None       # breathing: pad inter-sentence pauses to this (ms)
    initial_codec_chunk_frames: Optional[int] = None   # vLLM TTFA knob (smaller = lower first-audio)
    speed: Optional[float] = Field(default=None, ge=0.25, le=4.0)   # speech rate
    seed: Optional[int] = None
    reference_audio: Optional[str] = None   # base64 wav for ad-hoc cloning
    reference_text: Optional[str] = None


class OpenAISpeechBody(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "mp3"  # OpenAI default
    speed: float = Field(default=1.0, ge=0.25, le=4.0)   # forwarded to the backend
    stream: bool = False


# ---- auth ----------------------------------------------------------------
def require_key(authorization: str = Header(None), x_api_key: str = Header(None)):
    """No-op when api_key is unset (open LAN service). Otherwise require a match."""
    if not API_KEY:
        return
    token = x_api_key or ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    if token != API_KEY:
        raise HTTPException(401, "invalid or missing API key")


async def require_backend():
    """Ensure the vLLM-Omni backend is reachable before a generation request.

    Runs as a dependency (before the route body) so load() — which retries until
    the backend is up and refreshes the voice list — happens before voice
    resolution, and a down backend is a clean 503 for every generation route,
    rather than each one hand-inlining the same check. Near-free once ready."""
    try:
        await run_in_threadpool(engine.load)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


# ---- helpers -------------------------------------------------------------
def _params(body: SpeakBody) -> dict:
    """Merge request params over the config defaults (request wins)."""
    p = dict(DEFAULTS)
    for k in ("max_new_tokens", "stream_sentence_gap_ms",
              "initial_codec_chunk_frames", "speed", "seed"):
        v = getattr(body, k)
        if v is not None:
            p[k] = v
    return p


def _resolve_voice(voice: Optional[str]) -> Optional[str]:
    """None -> configured default voice; "" -> zero-shot (no reference)."""
    if voice is None:
        return DEFAULT_VOICE or None
    return voice or None


def _decode_ref(body: SpeakBody) -> tuple[Optional[bytes], Optional[str]]:
    """Decode an ad-hoc cloning reference, confirming up front that it's real,
    readable audio — so a bad clip is a clean 400, not a 500 (or a broken stream
    after 200) when soundfile later chokes on it deep inside synthesis."""
    ra, rt = body.reference_audio, (body.reference_text or "").strip()
    if bool(ra) != bool(rt):                       # exactly one supplied -> caller error
        raise HTTPException(400, "reference_audio and reference_text must be provided together")
    if not (ra and rt):
        return None, None
    import base64
    import soundfile as sf
    try:
        decoded = base64.b64decode(ra, validate=True)
    except Exception:
        raise HTTPException(400, "reference_audio must be valid base64-encoded audio")
    try:
        info = sf.info(io.BytesIO(decoded))        # real audio, not random/text bytes?
    except Exception:
        raise HTTPException(400, "reference_audio is not readable audio (expected wav/flac/ogg/…)")
    if info.duration < 1.0:                         # the backend needs >=1s of clear speech
        raise HTTPException(400, f"reference_audio too short ({info.duration:.1f}s); need >=1s of clear speech")
    return decoded, rt


def _synth_full(text, voice, params, ref_audio, ref_text, fmt) -> bytes:
    audio, sr = engine.generate(text, voice, params, ref_audio, ref_text)
    return encode_audio(audio, sr, fmt)


def _ffmpeg_stream(pcm_chunks, sample_rate: int, fmt: str):
    """Stream-encode live PCM to a compressed format through a running ffmpeg.

    A feeder thread writes PCM to ffmpeg's stdin while we yield encoded bytes
    from its stdout, so callers get mp3/opus/… frames as they render instead of
    a downgraded format or a buffer-the-whole-clip stall. Encoder latency is a
    few ms. On client disconnect (GeneratorExit) the finally tears ffmpeg down."""
    proc = subprocess.Popen(
        _ffmpeg_cmd(sample_rate, fmt),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    def _feed():
        try:
            for chunk in pcm_chunks:
                proc.stdin.write(chunk)
        except (BrokenPipeError, ValueError, OSError):
            pass                                   # ffmpeg gone / client disconnected
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
            pcm_chunks.close()                     # propagate close to engine.generate_stream

    feeder = threading.Thread(target=_feed, daemon=True)
    feeder.start()
    try:
        while True:
            buf = proc.stdout.read1(8192)          # read1: yield as soon as bytes are available
            if not buf:
                break
            yield buf
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        feeder.join(timeout=1)


def _stream(text, voice, params, ref_audio, ref_text, fmt):
    """Sync generator of encoded audio as it renders: raw PCM, a sentinel-header
    WAV stream, or a live ffmpeg-encoded compressed stream (mp3/opus/flac/…)."""
    pcm = (_pcm16(seg) for seg in
           engine.generate_stream(text, voice, params, ref_audio, ref_text))
    if fmt == "pcm":
        yield from pcm
    elif fmt == "wav":
        yield _wav_header(engine.sample_rate)
        yield from pcm
    else:
        yield from _ffmpeg_stream(pcm, engine.sample_rate, fmt)


# ---- app -----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading s2-pro (this can take a bit on first run)...")
    await run_in_threadpool(engine.warm_up)
    if engine.ready:
        logger.info("Fish Studio ready on %s:%s", config.HOST, config.PORT)
    else:
        logger.warning("Fish Studio started but backend NOT ready — retrying per request "
                       "(check %s)", config.VLLM_URL)
    yield


app = FastAPI(title="Fish Studio TTS", version="1.0", lifespan=lifespan)


@functools.lru_cache(maxsize=1)
def _gpu_name() -> Optional[str]:
    """GPU name via nvidia-smi (cached). Avoids importing torch into this
    lightweight, GPU-less proxy just to label /health."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


@app.get("/")
def root():
    return {
        "service": "Fish Studio TTS (Fish Audio S2-Pro)",
        "ready": engine.ready,
        "default_voice": DEFAULT_VOICE,
        "voices": engine.list_voices(),
        "endpoints": ["/health", "/voices", "/generate", "/stream", "/v1/audio/speech"],
        "example": {
            "url": "POST /generate",
            "body": {"text": "hello from the GPU", "voice": DEFAULT_VOICE, "format": "wav"},
        },
    }


@app.get("/health")
def health():
    # Readiness probe: 503 until the backend is reachable + voices synced, so
    # load balancers / monitors don't route to an instance that can't generate.
    ready = engine.ready
    payload = {
        "ok": ready,
        "ready": ready,
        "model": "fishaudio/s2-pro",
        "backend": f"vllm-omni @ {config.VLLM_URL}",
        "gpu": _gpu_name(),
        "sample_rate": engine.sample_rate,
        "voices": engine.list_voices(),
        "default_voice": DEFAULT_VOICE,
        "load_seconds": round(engine.load_seconds, 1) if engine.load_seconds else None,
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/voices")
def voices():
    return {"voices": engine.list_voices(), "default": DEFAULT_VOICE}


@app.post("/voices/reload")
def voices_reload(_=Depends(require_key)):
    return {"voices": engine.reload_voices(), "default": DEFAULT_VOICE}


@app.post("/generate")
async def generate(body: SpeakBody, _=Depends(require_key), __=Depends(require_backend)):
    """Full clip: render the whole utterance, return it as one response."""
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    fmt = _validate_format(body.format or DEFAULTS.get("format", "wav"))
    voice = _resolve_voice(body.voice)
    params = _params(body)
    ref_audio, ref_text = _decode_ref(body)
    data = await run_in_threadpool(_synth_full, text, voice, params, ref_audio, ref_text, fmt)
    return Response(content=data, media_type=MEDIA_TYPES[fmt])


@app.post("/stream")
async def stream(body: SpeakBody, _=Depends(require_key), __=Depends(require_backend)):
    """Live stream: emit audio as it renders. Defaults to pcm — lowest latency,
    no header, so first byte == first audio. wav and the compressed formats
    (mp3/opus/…) stream too (compressed via a live ffmpeg pipe)."""
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    fmt = _validate_format(body.format or "pcm")
    voice = _resolve_voice(body.voice)
    # require_backend already loaded the backend, so the voice list is fresh. Check
    # it here (not deep in the generator) because a lazy StreamingResponse has
    # already sent its 200 before the generator runs — too late for a 404.
    if voice and voice not in engine.list_voices():
        raise HTTPException(404, f"voice '{voice}' not found")
    params = _params(body)
    ref_audio, ref_text = _decode_ref(body)
    gen = _stream(text, voice, params, ref_audio, ref_text, fmt)
    return StreamingResponse(gen, media_type=MEDIA_TYPES[fmt], headers=STREAM_HEADERS)


@app.post("/v1/audio/speech")
async def openai_speech(body: OpenAISpeechBody, _=Depends(require_key), __=Depends(require_backend)):
    """OpenAI-compatible TTS. Point any OpenAI client's base_url here."""
    text = (body.input or "").strip()
    if not text:
        raise HTTPException(400, "empty input")
    fmt = _validate_format(body.response_format or "mp3")
    # require_backend already refreshed the voice list. Map the voice: use it if we
    # have it, else fall back to the default so clients sending OpenAI preset names
    # (alloy, etc.) still get audio.
    voice = body.voice if body.voice in engine.list_voices() else (DEFAULT_VOICE or None)
    params = dict(DEFAULTS)
    if body.speed is not None:
        params["speed"] = body.speed

    if body.stream:
        gen = _stream(text, voice, params, None, None, fmt)
        return StreamingResponse(gen, media_type=MEDIA_TYPES[fmt], headers=STREAM_HEADERS)

    data = await run_in_threadpool(_synth_full, text, voice, params, None, None, fmt)
    return Response(content=data, media_type=MEDIA_TYPES[fmt])


@app.exception_handler(UnknownVoiceError)
async def _unknown_voice(request, exc):
    return JSONResponse(status_code=404, content={"error": str(exc)})


@app.exception_handler(BackendError)
async def _backend_error(request, exc):
    # Mirror the backend's status (4xx client error stays 4xx; 5xx -> 502).
    # Only effective for non-streaming routes — a streaming error after the 200
    # is already sent just ends the stream.
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RuntimeError)
async def _runtime_error(request, exc):
    logger.exception("generation error")
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=config.HOST, port=config.PORT, log_level="info")
