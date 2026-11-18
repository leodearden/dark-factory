"""orchestrator.fleet_files — the filesystem primitives the fleet-global registries under the fleet root share.

Fleet-root resolution, the atomic tmp+rename text writer and the stable-sidecar
exclusive lock, used by ``session_registry.py`` and ``decision_registry.py``.

It is stdlib-only and imports no orchestrator sibling and no ``shared``:
``session_registry.py`` imports it as a bare sibling when it runs by absolute
path with no venv or PYTHONPATH, and task 9778 can move it into
``shared.session`` unchanged. The bare-shell contract is pinned by
``orchestrator/tests/test_session_registry.py::TestStdlibOnlySelfContainment``.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path


def fleet_root(root: Path | str | None = None) -> Path:
    """Resolve the fleet root: explicit *root* > $CLAUDE_FLEET_ROOT > ~/.claude/fleet.

    Every registry function takes this same *root* parameter (or derives it
    via this function) so tests never touch the real ``~/.claude`` tree.
    """
    if root is not None:
        return Path(root)
    env_root = os.environ.get('CLAUDE_FLEET_ROOT')
    if env_root:
        return Path(env_root)
    return Path.home() / '.claude' / 'fleet'


ATOMIC_WRITE_TEMP_SUFFIX = '.tmp'
"""Suffix of ``atomic_write_text``'s in-flight temp file. A sweep that visits
every entry of a dir the registries write into
(``session_reaper.py::reap_stale_session_pointers``) keys on it to leave a
concurrent writer's temp file alone."""


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write *text* to *path* (tmp file in the same dir, then os.replace).

    THIS IS THE DELIBERATE EXCEPTION to task 3223's consolidation, which moved
    this repo's copies of the tmp+rename writer onto
    :func:`shared.safe_io.atomic_write_text`. This module does NOT delegate.

    Why it was kept: ``session_registry.py``, which imports this module, is
    invoked by absolute path from
    ``skills/spawn/spawn-claude.sh`` under an interpreter with no venv, no
    install and no workspace packages on ``sys.path``. Before session_registry's
    module-scope ``shared/src`` bootstrap existed, a ``from shared import safe_io``
    made the whole module unimportable there — measured as ``ModuleNotFoundError: No
    module named 'shared'`` — and the only visible symptom was a hook
    subprocess that silently never wrote ``record.json``. The bootstrap now
    reaches ``shared``'s stdlib-only leaves, ``safe_io`` among them, so whether
    to delegate is open again (ticket tkt_0RW29CREM5XVXNE710FGF4W5MA). Until
    that is decided, this function stays recorded in ``_ALLOWED_RENAMERS`` in
    ``tests/scripts/test_atomic_write_regrowth.py`` so the anti-regrowth guard
    reads it as a documented exception rather than a fresh copy.

    ``os.fdopen(fd, 'w')`` with no explicit encoding defaults to the ambient
    locale's encoding, which is host-dependent. Task 3387 fixed the resulting
    latent bug: the payload is always JSON (``record.to_json()``), and RFC 8259
    requires JSON on disk to be UTF-8 — under a non-UTF-8 locale this wrote
    bytes that the registries' OWN READERS then reject. Name them precisely, so
    the next investigator does not go looking in the wrong module:
    ``session_registry.py::read_record`` raises ``CorruptSessionRecord`` (the
    decode error is a ``UnicodeDecodeError``, a ``ValueError`` subclass, so it
    lands in that function's ``except`` clause),
    ``decision_registry.py::list_decisions`` and
    ``session_leases.py::reap_stale_leases`` log and SKIP the file, and
    ``decision_registry.py::_mutate_decision`` returns ``None``. (An
    earlier version of this paragraph cited
    ``shared.safe_io.load_json_or_warn``. That helper never reads session,
    decision or lease records — its callers are ``b3_gate``, ``chronic_flake``,
    ``landed_outbox`` and ``merge_queue_store``.)

    The encoding is now pinned ``'utf-8'`` explicitly at both halves of the
    round-trip — here on write, and on every ``read_text()`` of the registries —
    so neither half follows the ambient locale. The literal is preferred over
    ``locale.getpreferredencoding(False)`` because it is locale-INDEPENDENT,
    which is the entire point; ``locale`` is stdlib and importing it would not
    have violated this module's stdlib-only constraint above.

    Error policy stays with the callers: session_registry.py::write_record lets
    a failure propagate (its sole caller, the CLI main(), provides the outer
    fail-soft boundary); decision_registry.py::write_decision swallows a
    failure itself (it is called directly by
    watchers/cockpit code with no such boundary).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        suffix=ATOMIC_WRITE_TEMP_SUFFIX,
        prefix=path.stem,
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp_path_str, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path_str)
        raise


@contextlib.contextmanager
def exclusive_sidecar_lock(lock_path: Path) -> Iterator[None]:
    """Blocking exclusive flock on the stable sidecar *lock_path*, creating it and its parent."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
