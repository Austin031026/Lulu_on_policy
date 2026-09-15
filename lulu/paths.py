"""Project locations; importing these helpers never loads ML dependencies."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT_ROOT = Path(os.environ.get('LULU_OUTPUT_ROOT', str(WORKSPACE_ROOT / 'LuLu_outputs'))).expanduser().resolve()


def soraka_root(value: str | Path | None = None) -> Path:
    """Locate the evaluation framework, which is bundled with Lulu by default.

    An external compatible checkout can still be selected by argument/env.
    """
    location = value if value is not None else os.environ.get('LULU_SORAKA_ROOT')
    return Path(location).expanduser().resolve() if location else PROJECT_ROOT
