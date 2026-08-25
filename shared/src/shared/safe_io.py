"""Safe JSON I/O with explicit benign-absent vs corrupt-present handling.

Provides :func:`load_json_or_warn` which splits the common "read optional
JSON state file" pattern into two clearly-separated branches:

* **FileNotFoundError** — first-run / file absent; silently returns the
  caller-supplied ``default``.
* **JSONDecodeError / ValueError** — file present but corrupt (includes
  non-UTF-8 / binary garbage); emits a deduped WARNING and dispatches on
  *on_corrupt*.

Other OSErrors (PermissionError, IsADirectoryError, …) propagate uncaught
so the caller sees genuinely unexpected environmental failures.

Also provides :func:`atomic_write_text`, the write-side counterpart: the
consolidated tmp+rename atomic writer that every "write optional JSON state
file" site in the repo delegates to (task 3223).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Per-process dedup set for corrupt-path warnings.
# Bounded by the number of distinct paths that go corrupt in a process lifetime.
# A restart re-enables the warning (mirrors sqlite_task_backend._warned_malformed_task_ids:51).
_warned_corrupt_paths: set[str] = set()

# Retry bound for the exclusive temp-file create in _create_temp.  A uuid4
# collision is astronomically unlikely; this exists so a pathological
# environment fails loudly with an OSError rather than spinning forever.
_TEMP_CREATE_ATTEMPTS = 10

__all__ = ['atomic_write_text', 'load_json_or_warn']


def load_json_or_warn(
    path: str | os.PathLike[str],
    *,
    default: Any,
    on_corrupt: str = 'warn',
) -> tuple[Any, bool]:
    """Load *path* as JSON and return ``(parsed, True)`` on success.

    Parameters
    ----------
    path:
        File to read.  Accepts any ``os.PathLike`` or ``str``.
    default:
        Required keyword-only.  Returned as the value component whenever the
        file is absent or corrupt — forces callers to state their benign
        fallback explicitly.
    on_corrupt:
        One of ``'warn'`` (default), ``'fail_closed'``, or ``'quarantine'``.
        Controls what happens when a present-but-corrupt file is found;
        see return values below.  Validated eagerly — an unrecognised value
        raises ``ValueError`` before any I/O is attempted.

        .. note::
            The per-process dedup set gates the WARNING only (once per path
            per process); the corrective action (quarantine rename) always
            runs.  A path that corrupts, gets quarantined, is re-created, and
            corrupts again will be silently quarantined without a second
            WARNING.  On process restart the WARNING re-fires.

    Returns
    -------
    tuple[Any, bool]
        ``(parsed, True)`` — valid JSON was read.

        ``(default, True)`` — file did not exist (benign, no log emitted).

        ``(default, False)`` — file was corrupt; a WARNING was emitted and
        the corrective action was taken.

    Raises
    ------
    json.JSONDecodeError / ValueError
        When *on_corrupt='fail_closed'*.
    ValueError
        When *on_corrupt* is an unrecognised value.
    OSError
        When any OS error other than ``FileNotFoundError`` occurs.
    """
    if on_corrupt not in ('warn', 'fail_closed', 'quarantine'):
        raise ValueError(f'unknown on_corrupt={on_corrupt!r}')

    p = Path(path)

    # --- read phase ---
    # Read raw bytes — no decode here (only FileNotFoundError is benign;
    # other OSErrors such as PermissionError/IsADirectoryError propagate).
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        # Benign first-run absence — caller's default, no log.
        return (default, True)

    # --- parse phase ---
    # json.loads accepts bytes (since 3.6) and auto-detects the encoding.
    # Both malformed JSON and non-UTF-8/binary garbage surface here as a
    # ValueError (JSONDecodeError/UnicodeDecodeError respectively).
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        # Corrupt branch: dispatch on on_corrupt.
        #
        # fail_closed: the exception IS the loud channel — re-raise BEFORE
        # any WARNING to avoid double-signalling.
        if on_corrupt == 'fail_closed':
            raise

        # warn / quarantine both emit the deduped WARNING — once per path per
        # process (mirrors sqlite_task_backend._warned_malformed_task_ids:51).
        # The corrective action (return / quarantine rename) always runs.
        key = str(p)
        if key not in _warned_corrupt_paths:
            _warned_corrupt_paths.add(key)
            logger.warning('safe_io: corrupt JSON at %s: %s', p, exc)

        if on_corrupt == 'warn':
            return (default, False)

        # quarantine: atomic rename to <name>.corrupt (os.replace overwrites any
        # prior quarantine slot, keeping disk growth bounded to one file per path).
        dest = p.with_name(p.name + '.corrupt')
        os.replace(p, dest)
        return (default, False)

    return (parsed, True)


def _create_temp(p: Path, mode: int | None) -> tuple[int, str]:
    """Exclusively create a temp file next to *p*; return ``(fd, path_str)``.

    The temp lives in ``p.parent`` — the same directory, hence the same
    filesystem — so the ``os.replace`` that follows is atomic per rename(2).

    ``O_CREAT | O_EXCL`` gives the two properties this helper is built on:

    * **Uniqueness per writer.**  The name carries a ``uuid4``, and O_EXCL makes
      the kernel reject a collision outright, so two concurrent writers to the
      same destination can never share a temp.  This is what a fixed
      ``<dest>.json.tmp`` name cannot provide, and it is the race the
      consolidated sites had.
    * **Umask semantics for free.**  Creating with 0o666 lets the kernel apply
      the process umask exactly as ``open()``/``Path.write_text`` do, avoiding
      the non-thread-safe ``os.umask(os.umask(0))`` read-back.  When *mode* is
      an explicit int, an ``fchmod`` then sets it exactly — on the still-open
      fd, so the bits are already correct at the instant ``os.replace`` makes
      the file visible, leaving no window where a reader sees the wrong mode.
    """
    for _ in range(_TEMP_CREATE_ATTEMPTS):
        tmp_str = str(p.parent / f'.{p.name}.{uuid.uuid4().hex}.tmp')
        try:
            fd = os.open(tmp_str, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            continue  # uuid4 collision — astronomically unlikely; retry anyway
        if mode is not None:
            try:
                os.fchmod(fd, mode)
            except BaseException:
                os.close(fd)
                with contextlib.suppress(OSError):
                    os.unlink(tmp_str)
                raise
        return (fd, tmp_str)
    raise OSError(f'could not create a unique temp file next to {p}')


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = 'utf-8',
    mode: int | None = None,
    fsync: bool = False,
    mkdir: bool = False,
) -> None:
    """Write *text* to *path* atomically via the tmp+rename pattern.

    The guarantee a concurrent reader may rely on: it observes either the
    complete OLD contents of *path* or the complete NEW contents — never a
    truncated or interleaved partial write, and never a missing file.  The
    text is written to a temp file that is exclusively created in *path*'s own
    directory and only then ``os.replace``-d into place, which rename(2)
    defines as atomic within a filesystem.

    The temp name is unique per writer (see :func:`_create_temp`), so two
    concurrent writers to the same destination cannot collide on one temp and
    tear each other's content — the specific latent race in the fixed
    ``<dest>.json.tmp`` sites this helper consolidates.

    No temp residue survives either outcome: the temp is renamed away on
    success and unlinked on any failure, including ``KeyboardInterrupt``.

    Parameters
    ----------
    path:
        Destination file.  Accepts any ``os.PathLike`` or ``str``.

        The argument is coerced with ``Path(path)``, so ANYTHING satisfying
        ``os.PathLike[str]`` becomes a real filesystem path — and under
        ``mkdir=True`` its parent chain is really created.  Callers must pass a
        genuine path; tests must never pass a mock.  This helper cannot detect
        one: ``unittest.mock.MagicMock`` supports ``__fspath__`` with a genuine
        ``str`` default return, so a MagicMock is a *valid* ``PathLike`` whose
        ``os.fspath()`` is a repr-derived string (``'MagicMock/mock.foo()/…'``)
        indistinguishable at the type level from a legitimate relative path.
        Passing one materialises that repr as a directory tree (task 3223).
    text:
        Complete contents to write.  The destination is replaced wholesale.
    encoding:
        Text encoding for the write.  Defaults to ``'utf-8'``.
    mode:
        Exact permission bits for the destination.  ``None`` (default) applies
        the process umask exactly as ``open()``/``Path.write_text`` would.

        This is explicit per call rather than a single hard-coded default
        because the sites consolidated here disagree: the ``mkstemp``-based
        ones produce 0600, the ``write_text``-based ones produce 0o666 & ~umask
        (typically 0664).  Silently narrowing the latter to 0600 would be a
        latent cross-uid breakage for the state files other processes read.
    fsync:
        When True, fsync the temp file before the rename AND the containing
        directory after it.  Default False, because plain tmp+rename already
        guarantees the rename is *atomic*; the fsyncs additionally make it
        *durable* across a crash immediately after this call returns, which
        only ``landed_outbox`` requires.
    mkdir:
        When True, create ``path.parent`` (and any missing ancestors) before
        writing.  Default False, so a missing parent surfaces as
        ``FileNotFoundError`` rather than being silently conjured.

        Creation is relative to the process CWD when *path* is relative, and
        reaches exactly the destination's ancestor chain — no further.  Pair
        this with the coercion note under *path*: ``mkdir=True`` is what turns
        a bogus duck-typed path from a harmless no-op into a real tree on disk.

    Raises
    ------
    OSError
        Any failure to create, write, or rename propagates unchanged; this
        helper never swallows.  Error policy belongs to the caller.
    """
    p = Path(path)

    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_str = _create_temp(p, mode)
    try:
        with os.fdopen(fd, 'w', encoding=encoding) as f:
            f.write(text)
            if fsync:
                # Inside the `with`, so the buffered text reaches the kernel
                # (flush) and then the platter (fsync) BEFORE the rename makes
                # it reachable — otherwise a crash could expose a renamed but
                # empty file.
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp_str, str(p))
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt landing mid-write
        # must still remove the temp, or an interrupted process leaves residue
        # next to the destination forever.  Mirrors the prompt_artifact /
        # memory_eval helpers this consolidates.  The unlink is best-effort;
        # the ORIGINAL exception is what the caller must see.
        with contextlib.suppress(OSError):
            os.unlink(tmp_str)
        raise

    if fsync:
        # The directory entry created by the rename is a separate durability
        # domain from the file's contents on most filesystems: without this,
        # a crash can leave the fully-synced data unreachable because the new
        # name was never persisted.  Must run AFTER the replace — syncing the
        # directory beforehand would not cover the rename at all.
        dir_fd = os.open(str(p.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
