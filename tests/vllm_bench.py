"""Benchmark vLLM-Omni vs the classic fish-speech engine for streaming TTS.

Measures, per backend, with identical text + reference voice:
  - TTFA: time to first audio byte (true first-audio, PCM stream, not HTTP headers)
  - RTF : generated audio seconds / wall seconds (>1 = faster than realtime)
over N warm runs (a couple of warm-ups first, because vLLM's first request pays a
one-time Triton JIT-compile spike that is NOT representative of steady state).

Both backends stream raw PCM (s16le, 44.1 kHz mono) so first byte == first audio.

  vLLM-Omni : POST /v1/audio/speech  {stream:true, response_format:"pcm",
              ref_audio:"data:audio/wav;base64,...", ref_text:...}
  classic   : POST /stream           {format:"pcm", voice:<name>}

Voice parity: both use the SAME reference voice (default: samantha). The proxy
uploads it to vLLM trimmed to <=30s (vLLM-Omni's reference cap).

  python tests/vllm_bench.py                          # both backends, default text
  python tests/vllm_bench.py --runs 5 --save /tmp/bench
  python tests/vllm_bench.py --only vllm --trace      # one backend + arrival trace
  python tests/vllm_bench.py --concurrency 4          # quick concurrent-throughput probe
"""

import argparse
import base64
import concurrent.futures as cf
import os
import statistics
import sys
import time

import requests

SR = 44100
BAR = "─" * 70
HERE = os.path.dirname(os.path.abspath(__file__))
VOICES = os.path.join(HERE, "..", "voices")

DEFAULT_TEXT = ("Earlier I was thinking about how I was annoyed. And I know this sounds "
                "strange, but honestly, I was really excited about it. So here we are.")


def load_ref_data_url(name):
    wav = os.path.join(VOICES, f"{name}.wav")
    txt = os.path.join(VOICES, f"{name}.txt")
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    return "data:audio/wav;base64," + b64, open(txt).read().strip()


def stream_pcm(endpoint, body, trace=False):
    """POST a streaming PCM request; return timing + the audio bytes."""
    t0 = time.time()
    first = None
    buf = bytearray()
    marks = []
    last = 0.0
    r = requests.post(endpoint, json=body, stream=True, timeout=300)
    r.raise_for_status()
    for ch in r.iter_content(4096):
        if not ch:
            continue
        now = time.time() - t0
        if first is None:
            first = now
        buf += ch
        if trace:
            cum = len(buf) / 2 / SR
            if cum - last >= 0.5:
                marks.append((now, cum))
                last = cum
    wall = time.time() - t0
    dur = len(buf) / 2 / SR
    return {"ttfa": first, "wall": wall, "dur": dur,
            "rtf": (dur / wall) if wall else 0.0, "pcm": bytes(buf), "trace": marks}


def make_vllm_call(url, text, ref_audio, ref_text, seed, icf=None):
    body = {"input": text, "ref_audio": ref_audio, "ref_text": ref_text,
            "response_format": "pcm", "stream": True, "seed": seed}
    if icf is not None:
        body["initial_codec_chunk_frames"] = icf
    return lambda trace=False: stream_pcm(f"{url}/v1/audio/speech", body, trace)


def make_classic_call(url, text, voice, seed):
    body = {"text": text, "voice": voice, "format": "pcm", "seed": seed}
    return lambda trace=False: stream_pcm(f"{url}/stream", body, trace)


def reachable(url, path="/health"):
    try:
        requests.get(f"{url}{path}", timeout=3)
        return True
    except Exception:
        # vLLM-Omni has no /health; a 404 from the base still means it's up
        try:
            requests.get(url, timeout=3)
            return True
        except Exception:
            return False


def bench(name, call, runs, warmup):
    print(f"  [{name}] warming up ({warmup}x)...", flush=True)
    for _ in range(warmup):
        call()
    ttfas, rtfs, durs = [], [], []
    last = None
    for i in range(runs):
        r = call()
        ttfas.append(r["ttfa"] * 1000)
        rtfs.append(r["rtf"])
        durs.append(r["dur"])
        last = r
        print(f"    run {i}: TTFA={r['ttfa']*1000:6.0f} ms  RTF={r['rtf']:.2f}x  "
              f"audio={r['dur']:.2f}s wall={r['wall']:.2f}s", flush=True)
    return {"name": name, "ttfa_med": statistics.median(ttfas), "ttfa_min": min(ttfas),
            "rtf_med": statistics.median(rtfs), "dur": statistics.median(durs),
            "pcm": last["pcm"]}


def concurrency_probe(name, call, n):
    print(f"  [{name}] concurrency probe: {n} simultaneous...", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(lambda _: call(), range(n)))
    wall = time.time() - t0
    total_audio = sum(r["dur"] for r in res)
    ttfas = sorted(r["ttfa"] * 1000 for r in res)
    print(f"    {n} reqs in {wall:.2f}s  aggregate RTF={total_audio/wall:.2f}x  "
          f"TTFA min/med/max = {ttfas[0]:.0f}/{statistics.median(ttfas):.0f}/{ttfas[-1]:.0f} ms")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm-url", default="http://127.0.0.1:8091")
    p.add_argument("--classic-url", default="http://127.0.0.1:8765")
    p.add_argument("--voice", default="samantha", help="reference voice name in voices/")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--only", choices=["vllm", "classic"], help="benchmark just one backend")
    p.add_argument("--initial-codec-chunk-frames", type=int, default=None,
                   help="vLLM TTFA knob (smaller = lower first-audio)")
    p.add_argument("--trace", action="store_true", help="print arrival trace for one run each")
    p.add_argument("--concurrency", type=int, default=0, help="also run an N-way concurrent probe")
    p.add_argument("--save", help="dir to write <backend>.pcm for A/B listening")
    a = p.parse_args()

    ref_audio, ref_text = load_ref_data_url(a.voice)
    print(BAR)
    print(f"  TTS bench  voice={a.voice}  runs={a.runs} (warmup {a.warmup})  seed={a.seed}")
    print(f'  text: "{a.text[:70]}..."')
    print(BAR)

    backends = []
    if a.only != "classic":
        if reachable(a.vllm_url, "/v1/models"):
            backends.append(("vLLM-Omni", make_vllm_call(
                a.vllm_url, a.text, ref_audio, ref_text, a.seed, a.initial_codec_chunk_frames)))
        else:
            print(f"  ! vLLM-Omni unreachable at {a.vllm_url} — skipping")
    if a.only != "vllm":
        if reachable(a.classic_url, "/health"):
            backends.append(("classic", make_classic_call(
                a.classic_url, a.text, a.voice, a.seed)))
        else:
            print(f"  ! classic unreachable at {a.classic_url} — skipping")

    if not backends:
        print("no backends reachable"); return 1

    results = []
    for name, call in backends:
        results.append(bench(name, call, a.runs, a.warmup))
        if a.trace:
            tr = call(trace=True)["trace"]
            print(f"    arrival trace: " + ", ".join(f"{w:.2f}s→{c:.2f}s" for w, c in tr))
        if a.concurrency:
            concurrency_probe(name, call, a.concurrency)
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            path = os.path.join(a.save, f"{name}.pcm")
            open(path, "wb").write(results[-1]["pcm"])
            print(f"    saved {path}  (play: ffplay -f s16le -ar {SR} -ac 1 {path})")

    print("\n" + BAR)
    print(f"  {'backend':<12} {'TTFA median':>12} {'TTFA best':>11} {'RTF median':>11}")
    print(BAR)
    for r in results:
        print(f"  {r['name']:<12} {r['ttfa_med']:>9.0f} ms {r['ttfa_min']:>8.0f} ms "
              f"{r['rtf_med']:>9.2f}x")
    if len(results) == 2:
        a0, b0 = results
        if b0["ttfa_med"] and a0["ttfa_med"]:
            faster = b0["ttfa_med"] / a0["ttfa_med"]
            print(f"\n  {a0['name']} TTFA is {faster:.1f}x {'lower' if faster>1 else 'higher'} "
                  f"than {b0['name']} ({a0['ttfa_med']:.0f} vs {b0['ttfa_med']:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
