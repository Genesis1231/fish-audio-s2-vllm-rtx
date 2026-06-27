"""Prototype: DETERMINISTIC breathing via per-sentence synthesis.

Splits the text into sentences, asks vLLM-Omni for each one separately (same
cloned voice), and concatenates them with an exact fixed gap between — so the
inter-sentence pause is always exactly what you set, regardless of vLLM's
stochastic prosody. The open question this lets you judge by ear: does
generating each sentence independently hurt onset/continuity?

Plays the result live. Compares against the single-request stream for reference.

  python tests/per_sentence_demo.py --gap 600
  python tests/per_sentence_demo.py --gap 600 --compare
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play import StreamPlayer, banner, pick_backend

SR = 44100
SENT = re.compile(r"(?<=[.!?…])\s+")


def synth_sentence(vllm, voice, text, seed=None):
    body = {"input": text, "voice": voice, "response_format": "pcm", "stream": True}
    if seed is not None:
        body["seed"] = seed
    r = requests.post(f"{vllm}/v1/audio/speech", json=body, stream=True, timeout=120)
    r.raise_for_status()
    return b"".join(c for c in r.iter_content(4096) if c)


def play_per_sentence(vllm, voice, text, gap_ms, backend, sd):
    banner(f"PER-SENTENCE  exact {gap_ms} ms gaps")
    sents = [s for s in SENT.split(" ".join(text.split())) if s]
    print(f"  {len(sents)} sentences")
    gap = b"\x00\x00" * int(SR * gap_ms / 1000)
    player = StreamPlayer(SR, backend, sd)
    t0 = time.time(); first = None
    for i, s in enumerate(sents):
        pcm = synth_sentence(vllm, voice, s)        # generated independently
        if first is None:
            first = time.time() - t0
            print(f"  TTFA = {first*1000:.0f} ms")
        if i > 0:
            player.write(gap)                        # exact, deterministic breath
        player.write(pcm)
        print(f"    sentence {i}: {len(pcm)/2/SR:.2f}s  \"{s[:40]}\"")
    player.close()
    print(f"  done over {time.time()-t0:.2f}s")


def play_single(base, voice, text, backend, sd):
    banner("SINGLE REQUEST  (current proxy /stream, detect+extend breathing)")
    player = StreamPlayer(SR, backend, sd)
    r = requests.post(f"{base}/stream",
                      json={"text": text, "voice": voice, "format": "pcm"},
                      stream=True, timeout=120)
    r.raise_for_status()
    t0 = time.time(); first = None
    for ch in r.iter_content(4096):
        if ch:
            if first is None:
                first = time.time() - t0; print(f"  TTFA = {first*1000:.0f} ms")
            player.write(ch)
    player.close()
    print(f"  done over {time.time()-t0:.2f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm", default="http://127.0.0.1:8091")
    p.add_argument("--proxy", default="http://127.0.0.1:8765")
    p.add_argument("--voice", default="samantha")
    p.add_argument("--text", default="Earlier I was thinking about how I was annoyed. "
                                     "And I know this sounds strange, but honestly, I was "
                                     "really excited about it. So here we are.")
    p.add_argument("--gap", type=int, default=600)
    p.add_argument("--device")
    p.add_argument("--compare", action="store_true", help="also play the single-request version")
    a = p.parse_args()
    backend, sd = pick_backend(a.device)
    if not backend:
        print("no audio backend"); return 1
    play_per_sentence(a.vllm, a.voice, a.text, a.gap, backend, sd)
    if a.compare:
        time.sleep(1.0)
        play_single(a.proxy, a.voice, a.text, backend, sd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
