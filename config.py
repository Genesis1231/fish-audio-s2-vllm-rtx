"""Configuration for the Fish Studio TTS service (vLLM-Omni proxy).

Values come from config.json next to this file; any can be overridden by an
environment variable (handy for systemd / docker) without editing the file.
"""

import logging
import os
import json
from pathlib import Path

HERE = Path(__file__).parent

_cfg = json.loads((HERE / "config.json").read_text())


def _env(name: str, default):
    """Read an override from the environment, coercing to the default's type."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    return raw


HOST            = _env("FISH_HOST", _cfg.get("host", "0.0.0.0"))
PORT            = _env("FISH_PORT", _cfg.get("port", 8765))
DEFAULT_VOICE   = _env("FISH_DEFAULT_VOICE", _cfg.get("default_voice", ""))
API_KEY         = _env("FISH_API_KEY", _cfg.get("api_key") or "")   # "" = open (no auth)
DEFAULTS        = _cfg.get("defaults", {})

# vLLM-Omni backend (the GPU model; this service is a thin proxy in front of it).
VLLM_URL        = _env("FISH_VLLM_URL", _cfg.get("vllm_url", "http://127.0.0.1:8091"))
REF_MAX_SECONDS = _env("FISH_REF_MAX_SECONDS", _cfg.get("ref_max_seconds", 28))

VOICES_DIR = HERE / "voices"

logger = logging.getLogger("fish-studio")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
