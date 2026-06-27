"""Example client for the Fish Studio TTS service — copy this into your program.

  python client_example.py "hello from the GPU"            # POST /generate -> out.wav
  python client_example.py --voice samantha --stream "hi"  # POST /stream (low latency)
  python client_example.py --host 192.168.3.56 "from another machine on the LAN"

The mode is the endpoint: /generate returns the full clip, /stream emits audio
as it renders. (For live playback of a stream, see tests/play.py.)
"""

import argparse
import sys
import urllib.request
import json


def speak(host, port, text, voice, fmt, stream, out):
    path = "/stream" if stream else "/generate"
    url = f"http://{host}:{port}{path}"
    payload = {"text": text, "voice": voice, "format": fmt}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req) as r, open(out, "wb") as f:
        # Streaming responses arrive in chunks; write them as they come.
        while True:
            chunk = r.read(8192)
            if not chunk:
                break
            f.write(chunk)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--voice", default="samantha")
    p.add_argument("--format", default="wav")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--out", default="out.wav")
    a = p.parse_args()
    speak(a.host, a.port, a.text, a.voice, a.format, a.stream, a.out)


if __name__ == "__main__":
    sys.exit(main())
