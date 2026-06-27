#!/usr/bin/env bash
# Launch (or resume) the vLLM-Omni S2-pro backend container that server.py proxies.
# The image `vllm-omni-fish:local` is the official vllm/vllm-omni:v0.22.0 plus the
# DAC codec's deps (descript-audio-codec stack) installed under a constraints file
# so vLLM's torch 2.11 / transformers 5.8 / sm_120 kernels stay intact. fish_speech
# source is mounted (the codec imports it) via PYTHONPATH.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME=vllm_omni_s2
IMAGE=vllm-omni-fish:local
PORT=8091

if [ ! -d "$DIR/../models/s2-pro" ]; then
  echo "[vllm-omni] error: model weights not found at $DIR/../models/s2-pro" >&2
  echo "            download them (see README, huggingface.co/fishaudio/s2-pro) and retry." >&2
  exit 1
fi
MODELS="$(cd "$DIR/../models" && pwd)"

# If a container already exists but was built from a DIFFERENT image than the
# current $IMAGE (i.e. you rebuilt vllm-omni-fish:local), recreate it — otherwise
# `docker start` below would silently resurrect the old image and old serve args.
if docker inspect "$NAME" >/dev/null 2>&1; then
  want="$(docker inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
  have="$(docker inspect -f '{{.Image}}' "$NAME" 2>/dev/null || true)"
  if [ -n "$want" ] && [ "$want" != "$have" ]; then
    echo "[vllm-omni] image changed since this container was created — recreating..."
    docker rm -f "$NAME" >/dev/null
  fi
fi

if docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "[vllm-omni] already running"
elif docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "[vllm-omni] starting existing container..."
  docker start "$NAME" >/dev/null
else
  echo "[vllm-omni] creating container from $IMAGE ..."
  docker run -d --name "$NAME" --gpus all -p ${PORT}:${PORT} \
    -v "$MODELS":/models:ro \
    -v "$DIR":/fishaudio:ro \
    -e PYTHONPATH=/fishaudio \
    --entrypoint vllm \
    "$IMAGE" \
    serve /models/s2-pro --omni --port ${PORT} >/dev/null
fi

echo -n "[vllm-omni] waiting for API on :${PORT}"
until curl -s -m3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo " — container exited:"; docker logs --tail 30 "$NAME"; exit 1
  fi
  echo -n "."; sleep 3
done
echo " ready."
