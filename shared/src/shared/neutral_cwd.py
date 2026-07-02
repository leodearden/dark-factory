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
Initialization is guarded by a lock (double-checked locking) so concurrent
callers from multiple threads still converge on a single directory, and the
directory is removed automatically via ``atexit`` on clean process shutdown.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import threading
from pathlib import Path

_neutral_cwd: Path | None = None
_lock = threading.Lock()


def neutral_cli_cwd() -> Path:
    """Return a process-wide cached empty directory for neutral CLI invocations.

    The directory contains no ``CLAUDE.md`` and is not any project's root, so
    the CLI auto-loads neither a filing project's ``CLAUDE.md`` nor its
    cwd-keyed ``MEMORY.md``. Created once per process and reused on every
    call — thread-safe via double-checked locking, so concurrent callers
    never race into creating two scratch dirs. Registered for best-effort
    removal via ``atexit`` on clean shutdown (relies on OS tmp-reaping
    otherwise, e.g. after a hard kill).
    """
    global _neutral_cwd
    if _neutral_cwd is None:
        with _lock:
            if _neutral_cwd is None:
                path = Path(tempfile.mkdtemp(prefix='fm-neutral-classifier-cwd-'))
                atexit.register(shutil.rmtree, path, ignore_errors=True)
                _neutral_cwd = path
    return _neutral_cwd
