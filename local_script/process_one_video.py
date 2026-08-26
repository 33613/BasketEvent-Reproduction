"""Compatibility CLI for the modular single-clip application."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.process_clip import *  # noqa: E402,F401,F403


if __name__ == "__main__":
    main()
