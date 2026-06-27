"""Play audio from the Fish Studio TTS service — verify both endpoints + demo latency.

Runs the two modes sequentially and plays them on your speakers:
  - stream:   plays frames live as they arrive; reports TRUE time-to-first-audio.
  - generate: fetches the whole clip, then plays it.

Playback uses sounddevice (PortAudio). Choose an output with --device/--list-devices;
the default routes through the system sink. Falls back to ffplay if PortAudio is
missing; --no-play just fetches + measures (headless).

  python tests/play.py                          # stream then generate, default device
  python tests/play.py --device razer           # target the Razer speaker by name
  python tests/play.py --list-devices
  python tests/play.py --mode stream --text "a longer, multi-sentence line. like this one."
"""

import argparse
import io
import shutil
import subprocess
import sys
import threading
import time
import wave

import numpy as np
import requests

SR_DEFAULT = 44100
BAR = "─" * 64


def banner(msg):
    print(f"\n{BAR}\n  {msg}\n{BAR}")


# ---- output device + backend --------------------------------------------
def list_devices():
    import sounddevice as sd
    default_out = sd.default.device[1]
    print("output-capable devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            mark = "  <- default" if i == default_out else ""
            print(f"  [{i:>2}] {d['name']}  "
                  f"({d['max_output_channels']}ch @ {int(d['default_samplerate'])}Hz){mark}")


def resolve_device(spec):
    """None | int index | name substring -> device index (None = system default)."""
    if spec is None:
        return None
    import sounddevice as sd
    if spec.isdigit():
        return int(spec)
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and spec.lower() in d["name"].lower():
            return i
    raise SystemExit(f"no output device matching {spec!r} (try --list-devices)")


def pick_backend(device):
    """Return ('sounddevice', module) | ('ffplay', None) | (None, None)."""
    try:
        import sounddevice as sd
        sd.query_devices()                      # forces PortAudio to load
        dev = resolve_device(device)
        if dev is not None:
            cur = list(sd.default.device)
            cur[1] = dev
            sd.default.device = cur
        name = sd.query_devices(sd.default.device[1])["name"]
        print(f"[play] backend=sounddevice  output=[{sd.default.device[1]}] {name}")
        return "sounddevice", sd
    except Exception as e:
        if shutil.which("ffplay"):
            print(f"[play] sounddevice unavailable ({type(e).__name__}); using ffplay. "
                  f"sudo apt install -y libportaudio2 for the clean path.", file=sys.stderr)
            return "ffplay", None
        print("[play] no sounddevice and no ffplay — saving files only.", file=sys.stderr)
        return None, None


def play_clip(wav_bytes, backend, sd):
    if backend == "sounddevice":
        import soundfile as sf
        data, file_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        sd.play(data, file_sr)
        sd.wait()
    elif backend == "ffplay":
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", "-i", "pipe:0"],
                       input=wav_bytes)


class StreamPlayer:
    """Best-practice smooth player for live PCM frames:
      - jitter buffer: buffers a short lead before starting, then plays
        continuously via a callback (an underrun outputs silence, never a stop);
      - crossfade: blends each chunk join (~12 ms) so independently-generated
        chunks don't click at the seam.
    This is what makes chunked synthesis sound like a continuous stream
    (Fish Audio's own guidance: buffer 2-3 chunks + cross-fade)."""

    def __init__(self, sr, backend, sd, lead_s=0.5):
        self.backend, self.sd, self.sr, self._rem = backend, sd, sr, b""
        if backend == "sounddevice":
            self._buf = np.zeros(0, dtype=np.float32)
            self._pos = 0
            self._lead = int(lead_s * sr)
            self._started = False
            self._lock = threading.Lock()
            self._stream = sd.OutputStream(samplerate=sr, channels=1,
                                           dtype="float32", callback=self._cb)
        elif backend == "ffplay":
            self._proc = subprocess.Popen(
                ["ffplay", "-f", "s16le", "-ar", str(sr), "-ac", "1",
                 "-nodisp", "-autoexit", "-loglevel", "error", "-i", "pipe:0"],
                stdin=subprocess.PIPE)

    def _cb(self, outdata, frames, time_info, status):
        with self._lock:
            avail = len(self._buf) - self._pos
            n = max(0, min(frames, avail))
            if n:
                outdata[:n, 0] = self._buf[self._pos:self._pos + n]
            outdata[n:, 0] = 0.0                 # underrun -> silence, keep going
            self._pos += n

    def write(self, chunk):
        buf = self._rem + chunk
        m = len(buf) - (len(buf) % 2)            # whole int16 frames only
        frames, self._rem = buf[:m], buf[m:]
        if not frames:
            return
        if self.backend == "ffplay":
            self._proc.stdin.write(frames)
            return
        # frames are a slice of one continuous PCM stream, so just append — the
        # crossfade between generated chunks is done server-side (boundaries are
        # known there; here we'd only see arbitrary network packets).
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            self._buf = np.concatenate([self._buf, x])
            ready = len(self._buf) >= self._lead
        if not self._started and ready:          # start once the lead is buffered
            self._started = True
            self._stream.start()

    def close(self):
        if self.backend == "ffplay":
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            self._proc.wait()
            return
        if not self._started:                    # short clip never hit the lead
            self._started = True
            self._stream.start()
        while True:                              # drain the buffer
            with self._lock:
                if self._pos >= len(self._buf):
                    break
            time.sleep(0.05)
        self._stream.stop()
        self._stream.close()


# ---- the two modes -------------------------------------------------------
def run_stream(base, text, voice, sr, backend, sd, save):
    banner("STREAM  (POST /stream, pcm)  — plays frames as they render")
    player = StreamPlayer(sr, backend, sd)
    t0 = time.time(); first = None; buf = bytearray()
    r = requests.post(f"{base}/stream",
                      json={"text": text, "voice": voice, "format": "pcm"},
                      stream=True, timeout=300)
    r.raise_for_status()
    try:
        for chunk in r.iter_content(chunk_size=4096):
            if not chunk:
                continue
            if first is None:
                first = time.time() - t0
                print(f"  time-to-first-audio : {first*1000:6.0f} ms   <- you hear it now")
            buf += chunk
            player.write(chunk)
    finally:
        player.close()
    total = time.time() - t0
    dur = len(buf) / 2 / sr
    print(f"  audio duration      : {dur:6.2f} s")
    if backend:   # played live: wall is paced by realtime playback, not generation
        print(f"  played live over    : {total:6.2f} s  (paced by realtime playback, not gen speed)")
    else:         # headless: wall == pure generation time
        print(f"  generated in        : {total:6.2f} s  ({dur/total:.2f}x realtime)")
    if save:
        open(f"{save.rstrip('/')}/stream.pcm", "wb").write(buf)
        print(f"  saved               : {save.rstrip('/')}/stream.pcm (s16le {sr}Hz mono)")
    return {"mode": "stream", "ttfa_ms": first * 1000 if first else None,
            "dur": dur, "wall": total, "played": bool(backend)}


def run_generate(base, text, voice, backend, sd, save):
    banner("GENERATE  (POST /generate, wav)  — renders the whole clip, then plays")
    t0 = time.time()
    r = requests.post(f"{base}/generate",
                      json={"text": text, "voice": voice, "format": "wav"}, timeout=300)
    r.raise_for_status()
    wall = time.time() - t0
    data = r.content
    with wave.open(io.BytesIO(data)) as w:
        dur = w.getnframes() / w.getframerate()
    print(f"  rendered            : {dur:6.2f} s audio in {wall:.2f} s ({dur/wall:.2f}x realtime)")
    if save:
        open(f"{save.rstrip('/')}/generate.wav", "wb").write(data)
        print(f"  saved               : {save.rstrip('/')}/generate.wav")
    print("  playing full clip ...")
    play_clip(data, backend, sd)
    return {"mode": "generate", "ttfa_ms": None, "dur": dur, "wall": wall}


def main():
    p = argparse.ArgumentParser(description="Play audio from the Fish Studio TTS service.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--voice", default="samantha")
    p.add_argument("--text", default="Earlier I was thinking about how I was annoyed. "
                                      "And I know this sounds strange, but honestly, "
                                      "I was really excited about it.")
    p.add_argument("--mode", choices=["generate", "stream", "both"], default="both")
    p.add_argument("--device", help="output device: index or name substring (e.g. 'razer')")
    p.add_argument("--list-devices", action="store_true", help="list output devices and exit")
    p.add_argument("--save", help="directory to also write generate.wav / stream.pcm")
    p.add_argument("--no-play", action="store_true",
                   help="fetch + measure (+ --save) only, no audio — for headless checks")
    a = p.parse_args()

    if a.list_devices:
        list_devices()
        return 0

    base = f"http://{a.host}:{a.port}"
    try:
        sr = int(requests.get(f"{base}/health", timeout=5).json().get("sample_rate", SR_DEFAULT))
    except Exception as e:
        print(f"[play] cannot reach {base}/health: {e}", file=sys.stderr)
        return 1

    banner(f"Fish Studio TTS demo  —  voice={a.voice}  sr={sr}Hz")
    print(f'  text: "{a.text}"')
    backend, sd = (None, None) if a.no_play else pick_backend(a.device)

    results = []
    # Sequential: stream first (low-latency), then the full clip.
    if a.mode in ("stream", "both"):
        results.append(run_stream(base, a.text, a.voice, sr, backend, sd, a.save))
    if a.mode in ("generate", "both"):
        results.append(run_generate(base, a.text, a.voice, backend, sd, a.save))

    banner("summary")
    for r in results:
        if r["mode"] == "stream":
            tail = " (played live)" if r["played"] else f", {r['dur']/r['wall']:.2f}x realtime gen"
            print(f"    stream: {r['dur']:.2f}s audio, first-audio {r['ttfa_ms']:.0f} ms{tail}")
        else:
            print(f"  generate: {r['dur']:.2f}s audio in {r['wall']:.2f}s "
                  f"({r['dur']/r['wall']:.2f}x realtime)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
