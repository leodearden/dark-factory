"""Neutral CLI cwd for pure prompt-contained LLM classifiers.

The ``claude`` CLI auto-loads ``<cwd>/CLAUDE.md`` AND the auto-memory
``MEMORY.md`` (keyed to the cwd's project slug) into every call's
cache-creation. For classifiers that decide purely from prompt-contained
input (``disallowed_tools=['*']`` — no file access, no codebase signal
needed), the filing project's docs contribute zero decision signal but are
still paid on every call. Running such classifiers with an empty, neutral
cwd instead of the filing project's root removes both cost vectors.

``neutral_cli_cwd()`` lazily creates ONE empty scratch directory via
``tempfile.mkdtemp`` and caches it in a module-global so repeated calls
reuse the same directory instead of leaking a temp dir per classifier call.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

_neutral_cwd: Path | None = None


def neutral_cli_cwd() -> Path:
    """Return a process-wide cached empty directory for neutral CLI invocations.

    The directory contains no ``CLAUDE.md`` and is not any project's root, so
    the CLI auto-loads neither a filing project's ``CLAUDE.md`` nor its
    cwd-keyed ``MEMORY.md``. Created once per process and reused on every
    call.
    """
    global _neutral_cwd
    if _neutral_cwd is None:
        _neutral_cwd = Path(tempfile.mkdtemp(prefix='fm-neutral-classifier-cwd-'))
    return _neutral_cwd
