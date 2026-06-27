"""Stream from the proxy with different settings, PLAY LIVE, report TTFA.

Sweeps a knob (initial_codec_chunk_frames and/or breathing gap), and for each
value streams audio from POST /stream and plays it on your speakers as it
arrives, printing true time-to-first-audio. Listen and judge the quality/latency
tradeoff per setting.

  python tests/tune_stream.py --icf 1,2,4,8        # TTFA knob sweep, play each
  python tests/tune_stream.py --gap 0,400,600,800  # breathing sweep
  python tests/tune_stream.py --icf 2 --device razer
"""
import argparse
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play import StreamPlayer, banner, pick_backend   # reuse the live jitter-buffered player

SR = 44100


def play_stream(base, text, voice, backend, sd, icf=None, gap=None, seed=None, label=""):
    banner(label)
    body = {"text": text, "voice": voice, "format": "pcm"}
    if icf is not None:
        body["initial_codec_chunk_frames"] = icf
    if gap is not None:
        body["stream_sentence_gap_ms"] = gap
    if seed is not None:
        body["seed"] = seed
    player = StreamPlayer(SR, backend, sd)
    t0 = time.time(); first = None; nb = 0
    r = requests.post(f"{base}/stream", json=body, stream=True, timeout=120)
    r.raise_for_status()
    for ch in r.iter_content(4096):
        if not ch:
            continue
        if first is None:
            first = time.time() - t0
            print(f"  TTFA = {first*1000:5.0f} ms   <- audio starts now")
        nb += len(ch)
        player.write(ch)
    player.close()
    print(f"  audio {nb/2/SR:.2f}s, streamed+played over {time.time()-t0:.2f}s")
    return first


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="http://127.0.0.1:8765")
    p.add_argument("--voice", default="samantha")
    p.add_argument("--text", default="Earlier I was thinking about how I was annoyed. "
                                     "And I know this sounds strange, but honestly, I was "
                                     "really excited about it. So here we are.")
    p.add_argument("--icf", help="comma list of initial_codec_chunk_frames to sweep (e.g. 1,2,4,8)")
    p.add_argument("--gap", help="comma list of breathing gap ms to sweep (e.g. 0,400,600,800)")
    p.add_argument("--device", help="output device: index or name substring (e.g. 'razer')")
    p.add_argument("--repeat", type=int, default=1, help="play each setting N times")
    p.add_argument("--pause", type=float, default=1.0, help="seconds of silence between settings")
    p.add_argument("--seed", type=int, default=7, help="fix seed so only the swept knob varies (-1 = random)")
    a = p.parse_args()
    seed = None if a.seed == -1 else a.seed

    backend, sd = pick_backend(a.device)
    if not backend:
        print("no audio backend — install libportaudio2 or ffplay"); return 1

    icfs = [int(x) for x in a.icf.split(",")] if a.icf else [None]
    gaps = [int(x) for x in a.gap.split(",")] if a.gap else [None]

    results = []
    for icf in icfs:
        for gap in gaps:
            parts = []
            if icf is not None: parts.append(f"initial_codec_chunk_frames={icf}")
            if gap is not None: parts.append(f"gap={gap}ms")
            label = "STREAM  " + (", ".join(parts) or "defaults")
            for r in range(a.repeat):
                ttfa = play_stream(a.base, a.text, a.voice, backend, sd, icf, gap, seed, label)
                results.append((label, ttfa))
                time.sleep(a.pause)

    banner("summary")
    for label, ttfa in results:
        print(f"  {label:<48} TTFA {ttfa*1000:5.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
