"""vLLM-Omni proxy backend for the Fish Studio TTS service.

This replaces the in-process classic fish-speech engine. The heavy lifting now
runs in a separate vLLM-Omni container (``vllm serve /models/s2-pro --omni``),
which gives much lower time-to-first-audio (~150 ms vs ~1.1 s) and real
concurrency on this Blackwell (sm_120) GPU. This module is a thin HTTP adapter
that exposes the SAME interface ``server.py`` expects from the old ``engine``
object, so the rest of the service is unchanged.

Two things we add on top of vLLM-Omni's ``/v1/audio/speech``:
  * **Voice names** — our ``voices/<name>.{wav,txt}`` profiles are uploaded to
    vLLM once (trimmed to ``REF_MAX_SECONDS``, under vLLM's 30 s reference cap)
    and then referenced by name, so callers keep using ``voice:"samantha"``.
  * **Breathing** — vLLM emits a short, *variable* pause between sentences; we
    detect those boundaries and pad each to ``stream_sentence_gap_ms`` (default
    600 ms, per-request overridable), restoring the classic engine's breathing.
"""

import base64
import io
import re
import threading
import time
from typing import Iterator, Optional

import numpy as np
import requests

from config import DEFAULT_VOICE, REF_MAX_SECONDS, VLLM_URL, VOICES_DIR, logger

SAMPLE_RATE = 44100          # s2-pro DAC output
_DEFAULT_GAP_MS = 600        # fallback breathing target if not in params/config


class BackendError(Exception):
    """A non-2xx response from the vLLM-Omni backend, carrying a status code so
    the proxy can mirror it: client errors (4xx) stay 4xx, upstream failures
    become 502. server.py maps this to the HTTP response."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(f"vLLM-Omni error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class UnknownVoiceError(ValueError):
    """A named voice the backend doesn't have. server.py maps this to 404 (a
    dedicated type so it isn't confused with other ValueErrors)."""


def _pcm_to_f32(buf: bytes) -> np.ndarray:
    """little-endian int16 PCM bytes -> float32 [-1,1] mono (inverse of server._pcm16)."""
    return np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0


class BreathingExtender:
    """Pad vLLM's natural sentence-boundary pauses up to ``target_ms``.

    Streaming + stateful: ``feed()`` chunks of float32 audio in order; when speech
    *truly* resumes after a silence long enough to be a sentence break (>=
    ``min_pause_ms``) but shorter than the target, we inject the missing silence
    just before the speech. Comma-length pauses are left alone, and no real audio
    is dropped (we only ever *insert* silence).

    Robustness: vLLM's pauses often contain a brief soft blip (a breath/click)
    that would otherwise split one pause into two short runs and defeat detection.
    We **debounce** — a silence run only ends once speech persists for
    ``debounce_ms``; blips shorter than that are absorbed into the pause. This is
    what makes the breathing *consistent* across vLLM's stochastic output.

    Feeding a whole clip then ``flush()`` gives the same result (non-streaming).
    """

    def __init__(self, sr, target_ms, min_pause_ms=240, thresh_rms=0.015,
                 debounce_ms=70, frame_ms=10):
        self.sr = sr
        self.frame = max(1, int(sr * frame_ms / 1000))
        self.target = int(sr * target_ms / 1000)
        self.min_pause = int(sr * min_pause_ms / 1000)
        self.debounce = int(sr * debounce_ms / 1000)
        self.thresh = thresh_rms
        self.enabled = bool(target_ms and target_ms > 0)
        self._buf = np.zeros(0, dtype=np.float32)
        self._sil = 0              # samples since last *confirmed* speech (incl. absorbed blips)
        self._pending: list = []   # candidate speech frames not yet confirmed real
        self._pending_n = 0
        self._started = False      # seen confirmed speech yet? (don't pad leading silence)

    def _emit_silence(self, out, fr):
        self._sil += len(fr)
        out.append(fr)

    def feed(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return audio
        self._buf = np.concatenate([self._buf, audio]) if self._buf.size else audio
        n = (self._buf.size // self.frame) * self.frame
        frames, self._buf = self._buf[:n], self._buf[n:]
        out: list = []
        for i in range(0, n, self.frame):
            fr = frames[i:i + self.frame]
            silent = float(np.sqrt(np.mean(fr * fr) + 1e-12)) < self.thresh
            if silent:
                # a short speech blip inside a pause: absorb it as part of the silence
                for pf in self._pending:
                    self._emit_silence(out, pf)
                self._pending, self._pending_n = [], 0
                self._emit_silence(out, fr)
            else:
                self._pending.append(fr)
                self._pending_n += len(fr)
                if self._pending_n >= self.debounce:        # confirmed real speech
                    if self._started and self.min_pause <= self._sil < self.target:
                        out.append(np.zeros(self.target - self._sil, dtype=np.float32))
                    self._sil = 0
                    self._started = True
                    out.extend(self._pending)
                    self._pending, self._pending_n = [], 0
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        out = list(self._pending) + ([self._buf] if self._buf.size else [])
        self._pending, self._pending_n = [], 0
        self._buf = np.zeros(0, dtype=np.float32)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _trim_ref(wav_bytes: bytes, text: str, max_s: float) -> tuple[bytes, str]:
    """Trim a reference clip to <= max_s (vLLM's cap), with a sentence-bounded
    transcript prefix so ref_text still matches the (shorter) ref_audio."""
    import soundfile as sf
    if sf.info(io.BytesIO(wav_bytes)).duration <= max_s:
        return wav_bytes, text                        # already within cap — skip the full decode
    a, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if a.ndim > 1:
        a = a[:, 0]
    dur = len(a) / sr
    a2 = a[: int(max_s * sr)]
    buf = io.BytesIO()
    sf.write(buf, a2, sr, format="WAV")
    words = text.split()
    budget = max(1, int(len(words) * (max_s / dur) * 0.92))   # keep text slightly < audio
    prefix = " ".join(words[:budget])
    m = list(re.finditer(r"[.!?]", prefix))
    rtext = prefix[: m[-1].end()] if m else prefix
    return buf.getvalue(), rtext


class VLLMEngine:
    def __init__(self) -> None:
        self._base = VLLM_URL.rstrip("/")
        self._sr = SAMPLE_RATE
        self._voices: dict[str, tuple[bytes, str]] = {}   # name -> (wav_bytes, transcript)
        self._available: set[str] = set()                 # names confirmed usable on the backend
        self._lock = threading.Lock()                     # guards one-time load/voice-sync
        self._ready = False
        self._load_time: Optional[float] = None

    # ---- lifecycle -------------------------------------------------------
    def load(self) -> float:
        """Connect to vLLM-Omni and make sure our voices are uploaded. Idempotent;
        retried by every request until it succeeds (so the proxy survives the
        backend starting later)."""
        if self._ready:
            return 0.0
        with self._lock:
            if self._ready:
                return 0.0
            t0 = time.time()
            try:
                requests.get(f"{self._base}/v1/models", timeout=5).raise_for_status()
            except Exception as e:
                raise RuntimeError(
                    f"vLLM-Omni backend not reachable at {self._base} ({e}). "
                    f"Start it with ./run_vllm_omni.sh"
                )
            self._load_local_voices()
            self._sync_voices()
            self._ready = True
            self._load_time = time.time() - t0
            logger.info("vLLM backend ready in %.1fs (voices: %s)",
                        self._load_time, ", ".join(self.list_voices()) or "none")
            return self._load_time

    def warm_up(self) -> None:
        """Prime the backend (clears its one-time Triton JIT spike) using the
        default voice, so the first real request is fast."""
        try:
            self.load()
        except Exception:
            logger.exception("vLLM backend not ready at startup (will retry per request)")
            return
        voice = DEFAULT_VOICE if DEFAULT_VOICE in self._available else next(iter(self._available), None)
        if not voice:
            logger.warning("no usable named voices in %s — warming zero-shot; "
                           "upload a voice and POST /voices/reload", VOICES_DIR)
        try:                                          # voice=None -> zero-shot still primes the JIT
            for _ in self.generate_stream("Hello there.", voice, {"stream_sentence_gap_ms": 0}):
                pass
            logger.info("warm-up complete (JIT primed, voice=%s)", voice or "zero-shot")
        except Exception:
            logger.exception("warm-up failed (continuing anyway)")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def load_seconds(self) -> Optional[float]:
        return self._load_time

    # ---- voices ----------------------------------------------------------
    def _load_local_voices(self) -> None:
        self._voices.clear()
        if not VOICES_DIR.is_dir():
            return
        for wav in sorted(VOICES_DIR.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if txt.exists():
                self._voices[wav.stem] = (wav.read_bytes(), txt.read_text().strip())

    def _vllm_voice_names(self) -> set:
        try:
            r = requests.get(f"{self._base}/v1/audio/voices", timeout=10)
            r.raise_for_status()
            return set(r.json().get("voices", []))
        except Exception:
            return set()

    def _delete_voice(self, name: str) -> None:
        try:
            requests.delete(f"{self._base}/v1/audio/voices/{name}", timeout=30)
        except Exception:
            logger.exception("voice '%s' delete failed", name)

    def _sync_voices(self, replace: bool = False) -> None:
        """Make the backend's voices match our local profiles, and record which
        names are *actually usable* in ``self._available`` — so list_voices()
        never advertises a voice whose upload failed.

        ``replace=True`` (used by /voices/reload) re-uploads even voices the
        backend already has, so an edited local clip actually takes effect
        (otherwise a same-named voice would keep the stale reference)."""
        have = self._vllm_voice_names()
        available: set[str] = set()
        for name, (wav_bytes, text) in self._voices.items():
            if name in have and not replace:
                available.add(name)                    # already uploaded; trust it
                continue
            if name in have:                           # replace: drop the stale one first
                self._delete_voice(name)
            wav, rtext = _trim_ref(wav_bytes, text, REF_MAX_SECONDS)
            try:
                r = requests.post(
                    f"{self._base}/v1/audio/voices",
                    files={"audio_sample": (f"{name}.wav", wav, "audio/wav")},
                    data={"name": name, "ref_text": rtext,
                          "consent": "Owner-provided reference for local TTS."},
                    timeout=60,
                )
                if r.status_code == 200:
                    available.add(name)
                    logger.info("uploaded voice '%s' to vLLM (%.0fs ref)", name, REF_MAX_SECONDS)
                else:
                    logger.error("voice '%s' upload failed: %s %s — excluded from /voices",
                                 name, r.status_code, r.text[:200])
            except Exception:
                logger.exception("voice '%s' upload error — excluded from /voices", name)
        self._available = available

    def list_voices(self) -> list[str]:
        """Only voices confirmed usable on the backend (failed uploads excluded)."""
        return sorted(self._available)

    def reload_voices(self) -> list[str]:
        with self._lock:
            self._load_local_voices()
            self._sync_voices(replace=True)            # force re-upload so edited clips take effect
        return self.list_voices()

    # ---- request building ------------------------------------------------
    def _gap_ms(self, params: dict) -> int:
        v = params.get("stream_sentence_gap_ms")
        return int(v) if v is not None else _DEFAULT_GAP_MS

    def _build_body(self, text, voice, params, ref_audio, ref_text, stream) -> dict:
        body = {"input": text, "response_format": "pcm", "stream": stream}
        if ref_audio:                                  # ad-hoc cloning
            wav, rtext = _trim_ref(ref_audio, ref_text or "", REF_MAX_SECONDS)
            body["ref_audio"] = "data:audio/wav;base64," + base64.b64encode(wav).decode()
            body["ref_text"] = rtext
        elif voice:                                    # a named profile was requested
            if voice not in self._available:
                raise UnknownVoiceError(
                    f"voice '{voice}' not found (have: {', '.join(self.list_voices()) or 'none'})"
                )
            body["voice"] = voice
        # else: voice is None -> zero-shot. Send neither voice nor ref_audio; the
        # backend generates with its built-in default speaker.
        if params.get("seed") is not None:
            body["seed"] = int(params["seed"])
        if params.get("max_new_tokens"):
            body["max_new_tokens"] = int(params["max_new_tokens"])
        if params.get("initial_codec_chunk_frames") is not None:
            body["initial_codec_chunk_frames"] = int(params["initial_codec_chunk_frames"])
        if params.get("speed"):
            body["speed"] = float(params["speed"])
        return body

    def _post(self, body, stream):
        r = requests.post(f"{self._base}/v1/audio/speech", json=body, stream=stream, timeout=300)
        if r.status_code == 200:
            return r
        detail = (r.text or "")[:300]
        # Mirror the backend: a bad request (4xx, e.g. ref too short) stays 4xx;
        # an upstream failure (5xx) becomes 502 Bad Gateway.
        raise BackendError(r.status_code if 400 <= r.status_code < 500 else 502, detail)

    # ---- generation ------------------------------------------------------
    def generate(self, text: str, voice: Optional[str], params: dict,
                 ref_audio: Optional[bytes] = None,
                 ref_text: Optional[str] = None) -> tuple[np.ndarray, int]:
        """Full clip: render the whole utterance, apply breathing, return float32."""
        self.load()
        body = self._build_body(text, voice, params, ref_audio, ref_text, stream=False)
        r = self._post(body, stream=False)
        audio = _pcm_to_f32(r.content)
        ext = BreathingExtender(self._sr, self._gap_ms(params))
        out = ext.feed(audio)
        tail = ext.flush()
        return np.concatenate([out, tail]) if tail.size else out, self._sr

    def generate_stream(self, text: str, voice: Optional[str], params: dict,
                        ref_audio: Optional[bytes] = None,
                        ref_text: Optional[str] = None) -> Iterator[np.ndarray]:
        """Stream vLLM-Omni's PCM as it renders, extending sentence pauses to the
        configured breathing target on the fly. No GPU lock — vLLM handles
        concurrency, so multiple streams run in parallel."""
        self.load()
        body = self._build_body(text, voice, params, ref_audio, ref_text, stream=True)
        ext = BreathingExtender(self._sr, self._gap_ms(params))
        r = self._post(body, stream=True)
        try:
            rem = b""
            for chunk in r.iter_content(4096):
                if not chunk:
                    continue
                rem += chunk
                n = len(rem) - (len(rem) % 2)          # whole int16 samples only
                frames, rem = rem[:n], rem[n:]
                if not frames:
                    continue
                out = ext.feed(_pcm_to_f32(frames))
                if out.size:
                    yield out
            tail = ext.flush()
            if tail.size:
                yield tail
        finally:
            # Close the backend connection on completion OR client disconnect
            # (FastAPI throws GeneratorExit here), so vLLM can stop generating.
            r.close()


# Module-level singleton (drop-in replacement for the old `engine`).
engine = VLLMEngine()
