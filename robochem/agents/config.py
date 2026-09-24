"""
Model tiers and key handling for the LLM agents.

Merged from robomail_Aliyah/config/{env,models}.py. Two things change here:

  * This repo's ``.env`` writes the key as lowercase ``openai_api_key`` (that is
    what ``scripts/env.sh`` normalises), so the loader accepts either spelling.
    A run out of ``perception_env`` never sources env.sh -- that is the whole
    point of the simulated cell -- so the loader has to do it itself.
  * The model ids are not baked in anywhere: both tiers are environment
    variables with current defaults, and selecting a retired model warns rather
    than failing, so an old model can still be pinned for a reproduction run.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

ENV_API_KEY = "OPENAI_API_KEY"
ENV_BASE_URL = "ROBOCHEM_LLM_BASE_URL"
ENV_CHEAP = "ROBOCHEM_MODEL_CHEAP"
ENV_STRONG = "ROBOCHEM_MODEL_STRONG"

#: Keys taken from a .env file. An allowlist, so a stray line cannot redefine
#: PATH or anything else that matters.
ALLOWED_KEYS = frozenset({ENV_API_KEY, ENV_BASE_URL, ENV_CHEAP, ENV_STRONG})

#: .env spellings that map onto an allowed key. This repo's file predates the
#: agents and uses the lowercase form.
KEY_ALIASES = {"openai_api_key": ENV_API_KEY}

#: Chosen 2026-09. Both support vision and strict JSON-schema structured output,
#: which every agent in this package depends on.
DEFAULT_CHEAP_MODEL = "gpt-4.1-mini"
DEFAULT_STRONG_MODEL = "gpt-4.1"

#: Models with an announced API shutdown. Warned about, not blocked.
RETIRED_MODELS = {
    "gpt-4o": "API shutdown 2026-10-23",
    "gpt-4o-2024-05-13": "API shutdown 2026-10-23",
    "gpt-4o-2024-08-06": "API shutdown 2026-10-23",
    "gpt-4o-2024-11-20": "API shutdown 2026-10-23",
}

_loaded = False


def project_root() -> Path:
    """The repository root (this file lives in <root>/robochem/agents/)."""
    return Path(__file__).resolve().parents[2]


def parse_env_text(text: str) -> dict:
    """
    Parse ``.env`` content into a mapping of allowed keys.

    Supports ``KEY=value``, ``export KEY=value``, ``#`` comments, blank lines
    and quoted values. No interpolation and no multi-line values: an API key
    needs neither, and each extra feature is another way to parse it wrong.
    """
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        key = KEY_ALIASES.get(key, KEY_ALIASES.get(key.lower(), key))
        if key in ALLOWED_KEYS:
            values[key] = value
    return values


def load_env(path=None, *, override: bool = False) -> list:
    """
    Load ``.env`` into ``os.environ``. Returns the NAMES that were set.

    Never returns or logs a value. A real environment variable always wins
    unless ``override``, so a lab machine's environment or a CI secret is not
    quietly replaced by a stale checkout.
    """
    global _loaded
    # utf-8-sig, not utf-8: a Windows editor's BOM would otherwise become part
    # of the first key's NAME and that key would be silently dropped.
    env_path = Path(path) if path is not None else project_root() / ".env"
    if not env_path.is_file():
        _loaded = True
        return []

    applied = []
    for key, value in parse_env_text(env_path.read_text(encoding="utf-8-sig")).items():
        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
        applied.append(key)
    _loaded = True
    return applied


def ensure_loaded() -> None:
    """Load ``.env`` once per process. Called by :func:`api_key`."""
    if not _loaded:
        load_env()


def _resolve(env_var: str, default: str) -> str:
    model = os.environ.get(env_var, "").strip() or default
    if model in RETIRED_MODELS:
        warnings.warn(
            f"{env_var}={model} is retired ({RETIRED_MODELS[model]}). "
            f"Set {env_var} to a current model.",
            RuntimeWarning,
            stacklevel=3,
        )
    return model


def cheap_model() -> str:
    """Vision-heavy, low-reasoning tier: scene understanding, goal extraction."""
    return _resolve(ENV_CHEAP, DEFAULT_CHEAP_MODEL)


def strong_model() -> str:
    """Reasoning tier: the planner and the skill-call planner."""
    return _resolve(ENV_STRONG, DEFAULT_STRONG_MODEL)


def base_url():
    """Optional OpenAI-compatible base URL (Azure, vLLM, OpenRouter, ...)."""
    ensure_loaded()
    return os.environ.get(ENV_BASE_URL, "").strip() or None


def api_key():
    """The API key, or None. Never hardcoded, never logged."""
    ensure_loaded()
    return os.environ.get(ENV_API_KEY, "").strip() or None


def describe() -> dict:
    """Non-secret snapshot of the model configuration, safe to write to a log."""
    return {
        "cheap_model": cheap_model(),
        "strong_model": strong_model(),
        "base_url": base_url(),
        "api_key_present": api_key() is not None,
    }
