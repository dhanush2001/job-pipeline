"""
Auto-load a project-local .env so secrets (DEEPSEEK_API_KEY, ADZUNA_*, ...) can
live with the project instead of your global shell profile. Runs on first
import of the `src` package -- which every entry point does -- so all commands
pick it up.

Zero-dependency and deliberately simple: `KEY=value` lines, `#` comments, an
optional leading `export`, and surrounding quotes. A variable already set in
the real environment always wins (an explicit `export` is never overridden).
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:  # real env wins over the file
            os.environ[key] = val


_load_dotenv()
