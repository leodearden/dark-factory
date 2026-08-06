"""Fail-closed preflight for in-process mutation of the shared Mem0 store (task 3686).

THE POLICY THIS ENFORCES
------------------------
**Mutating memory operations belong to the fused-memory MCP server process --
the single, unsandboxed owner of the store. An in-sandbox script must never
mutate the shared store directly.**

``.mcp.json`` declares fused-memory as
``{"type": "http", "url": "http://127.0.0.1:8002/mcp"}`` -- a separate,
long-running process. A mutation routed through an MCP tool therefore executes
OUTSIDE the calling agent's landlock confinement and is unaffected by it. A
script that constructs its own ``MemoryService`` inside an agent's process tree
gets no such protection, and that is the hole this module closes.

THE MEASURED INCIDENT
---------------------
``scripts/sweep_toolcall_xml_leak.py --apply`` was run from a sandboxed agent
session. Its repair is delete-then-re-add, and the two halves sit on different
substrates:

  * the Qdrant delete is a NETWORK call to ``localhost:6333``. Landlock governs
    the FILESYSTEM only (see ``reconciliation.sandbox_guard``, which documents
    this same asymmetry from the other direction), so the delete SUCCEEDED;
  * mem0's history write is a SQLite write under ``~/.mem0`` -- a filesystem
    operation, so it was DENIED.

The mutation split in half. Memory ``7d073281-4c5d-4ba3-a01c-3a167f4460f4`` was
removed from the store and the repaired text was never written back; it
survived only in the sweep's printed report. Because landlock cannot block the
network delete, no widening of the sandbox write-set could have made that
two-phase mutation atomic -- it would only have lowered the probability. Only
an application-level refusal BEFORE the first mutation bounds the blast radius,
which is what :func:`assert_store_mutation_allowed` does.

WHY A CAPABILITY PROBE, NOT AN "AM I SANDBOXED?" INFERENCE
----------------------------------------------------------
A process cannot reliably introspect its own confinement from the inside, and
there is no runtime sandbox detector in this repo. A real create-and-remove
probe is honest in both worlds: it passes wherever the mutation would actually
work (an operator shell, an unsandboxed role, the MCP server itself) and fails
wherever it would not -- with no policy table to drift out of date.

``os.access(W_OK)`` is provably insufficient here. In the measured failure
``history.db`` was mode 0644 / uid 1000, ``os.access(W_OK)`` returned True, and
SQLite even accepted ``BEGIN IMMEDIATE`` -- while creating a NEW file in the
same directory raised EACCES, because a syscall filter denied the write the
directory's own mode permitted. The repo's only other write-probe idiom
(``_var_tmp_writable`` in ``tests/reconciliation/test_recon_sandbox_guard.py``)
documents exactly this and mandates a real write attempt for the same reason.

Pure stdlib plus a lazy read of mem0's own configuration. No Qdrant, no
network, no ``MemoryService``.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

__all__ = [
    'StoreMutationUnavailable',
    'assert_store_mutation_allowed',
    'resolve_history_dir',
]


class StoreMutationUnavailable(RuntimeError):
    """Raised when this process cannot safely mutate the shared Mem0 store.

    CALLER CONTRACT: on this exception the caller MUST NOT begin the mutation.
    Refusing to start is the entire point -- the failure mode being prevented
    is a HALF-completed mutation (Qdrant point already deleted, history write
    denied), which destroys content that then exists nowhere but a log.

    Modelled on ``reconciliation.sandbox_guard.RemediationSandboxUnavailable``:
    same ``RuntimeError`` base, same fail-closed posture, same convention of
    stating the caller contract in the docstring rather than leaving it to the
    call site.

    The remedy is never to weaken the guard. Route the mutation through the
    fused-memory MCP server (the unsandboxed owner of the store), or re-run the
    operation from an unsandboxed operator shell.
    """


def _mem0_configured_history_db_path() -> str:
    """Return mem0's own configured ``history_db_path``.

    Imported LAZILY and isolated behind this seam for two reasons: importing
    ``mem0.configs.base`` drags in the embeddings / vector-store / LLM config
    stack and costs seconds, and the seam lets the tests pin the resolution
    contract hermetically without paying that cost or touching a real
    ``~/.mem0``.
    """
    from mem0.configs.base import MemoryConfig  # noqa: PLC0415

    return MemoryConfig().history_db_path


def resolve_history_dir() -> Path:
    """Resolve the directory holding mem0's history database.

    Derived from mem0's own configuration, never hardcoded. ``mem0.configs.base``
    computes ``mem0_dir = os.environ.get('MEM0_DIR') or ~/.mem0`` and derives
    ``history_db_path`` from it -- but it does so at IMPORT time, so a
    ``MEM0_DIR`` set later in the process would be silently ignored by a plain
    re-read of the configured default. ``MEM0_DIR`` is therefore consulted here
    at CALL time and takes precedence, with the configured ``history_db_path``
    as the fallback.

    Hardcoding ``~/.mem0`` would be worse than useless: under a redirected
    ``MEM0_DIR`` the guard would probe a directory mem0 is not using and report
    "mutation is safe" about the wrong filesystem location.
    """
    env_dir = os.environ.get('MEM0_DIR')
    if env_dir:
        return Path(env_dir).expanduser()
    return Path(_mem0_configured_history_db_path()).expanduser().parent


def _create_probe_file(path: Path) -> None:
    """Create *path* exclusively, then leave it for the caller to unlink.

    ``O_CREAT | O_EXCL`` so the probe can never truncate a pre-existing file --
    the target directory holds live production state. Isolated as a seam so a
    test can simulate the measured EACCES without needing a genuinely
    write-denied directory (which is not reproducible portably, and is a no-op
    when the suite runs as root).
    """
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)


def assert_store_mutation_allowed(*, operation: str) -> None:
    """Refuse *operation* unless this process can actually write mem0's history dir.

    Probes real capability by creating a uniquely-named file in the resolved
    history directory and removing it again. Catches ANY OS-level refusal, not
    just ``PermissionError``: a syscall filter surfaces as EACCES, a missing
    directory as ENOENT, a read-only bind mount as EROFS. All three mean the
    mutation must not begin.

    A missing directory raises rather than passing -- an environment where
    mem0's history directory does not exist is an unknown one, not a permissive
    one, and silently proceeding is the failure mode this module exists to
    prevent.

    Args:
        operation: Human-readable name of the mutation being gated (e.g.
            ``'sweep_toolcall_xml_leak --apply'``). Echoed into the refusal
            message so an operator can tell WHICH operation was refused.

    Raises:
        StoreMutationUnavailable: When the probe cannot be created. The caller
            MUST NOT begin the mutation.
    """
    history_dir = resolve_history_dir()
    probe = history_dir / f'.store-mutation-probe-{uuid.uuid4().hex}'
    try:
        _create_probe_file(probe)
    except OSError as exc:
        raise StoreMutationUnavailable(
            f'Refusing to begin {operation!r}: this process cannot create files in '
            f"mem0's history directory {str(history_dir)!r} ({exc!r}). "
            f'SQLite must create a rollback journal beside history.db, so a '
            f"directory that denies file creation makes mem0's history write "
            f'fail with "attempt to write a readonly database" -- while a Qdrant '
            f'delete, being a NETWORK call, still succeeds. That splits a '
            f'delete-then-re-add in half and destroys content. Mutating memory '
            f'operations must go through the fused-memory MCP server (the '
            f'unsandboxed owner of the store); an in-sandbox script must never '
            f'mutate the shared store directly.'
        ) from exc
    finally:
        # Unconditional cleanup: the probe is scratch, never residue. A failed
        # create leaves nothing to remove, hence the swallowed OSError.
        try:
            probe.unlink()
        except OSError:
            pass
