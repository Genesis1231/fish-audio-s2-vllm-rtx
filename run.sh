#!/bin/bash
# Start the Fish Studio TTS service.
#   ./run.sh
# Two parts: (1) the vLLM-Omni S2-pro backend container (GPU heavy-lifting), and
# (2) our lightweight proxy API (server.py) that fronts it with voice names +
# breathing. The proxy needs no GPU/torch.compile of its own.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 1. ensure the backend container is up (idempotent; ~20s JIT warm on first start)
"$DIR/run_vllm_omni.sh"

# 2. start the proxy API (binds the LAN per config.json host/port)
exec "$DIR/.venv/bin/python" "$DIR/server.py" "$@"
