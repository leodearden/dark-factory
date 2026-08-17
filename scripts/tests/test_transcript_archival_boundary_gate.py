"""ε B+H boundary gate — scripts-consumer half (task 2732).

The end-to-end transcript-archival boundary gate over the ALREADY-INTEGRATED
code paths landed by α (producer hook + archiver primitive), β (teardown
backstop), γ (legibility archive mining) and δ (retention GC). See
``plans/agent-transcript-archival-prd.md`` Appendix B for the matrix::

    E1  a completed session's transcript is archived at completion
    E2  the archive survives worktree teardown
    E3  the teardown backstop is idempotent w.r.t. the producer
    E4  the archive is credential-safe (only projects/**.jsonl is ever copied)
    E5  legibility mining enumerates the archived transcript
    E6  a resumed session re-archives its grown transcript (last-write-wins)
    E7  an archive failure is SOFT (task still succeeds) and LOUD (counted+logged)
    E8  the retention GC prunes by cap, loudly; default caps are a no-op

**This file owns E5 and E8** — the two rows that read the archive from OUTSIDE
the orchestrator (the legibility miner and the retention GC). Its two siblings
own the rest:

* ``orchestrator/tests/test_transcript_archival_boundary_gate.py`` — E1, E6,
  E2, E3, E7 (the producer-hook and teardown-backstop rows).
* ``shared/tests/test_transcript_archival_boundary_gate.py`` — E4 (the
  credential-safety row, kept orchestrator-free so ``shared`` stays a leaf).

The gate is three files rather than one because ``verify`` is directory-scoped:
each package's ``orchestrator.yaml`` declares its own ``test_command``, so a
single cross-package module would run in exactly one lane and a shared-only or
scripts-only diff would never exercise its rows.

ARCHIVE FORMAT: plain ``.jsonl``, byte-verbatim, NO added suffix. Task 3618
(leaf α of ``plans/transcript-preservation-seam-prd.md``) dropped gzip from the
archive AFTER the PRD was written, so Appendix B's "gz round-trips" wording for
E1/E5 is stale — do not read it as a gap. The residual-``.jsonl.gz`` contract
that SURVIVED 3618 — such a file is not enumerated, but IS counted by
``inventory.count_residual_gz`` and announced as one WARNING — is pinned by E5
here; that sub-row is where the gz dimension of the matrix now lives.

No row mocks the component under test. Both rows build their archive by
calling the REAL :func:`shared.transcript_archive.archive_task_transcripts`
rather than hand-writing the archive layout, so E5 composes α's writer with γ's
reader and E8 composes α's source-mtime mirror with δ's ``scan_task_dirs`` age
derivation — instead of assuming the two agree.

Fixtures are kept module-local; ``scripts/tests/conftest.py``'s existing
sys.path wiring is what makes ``import gc_agent_transcripts`` and
``from legibility import ...`` resolve, so this file adds no conftest changes.
"""

from __future__ import annotations

import os
from pathlib import Path

# scripts/tests/../../ = the repo root, where docs/legibility/legibility.yaml
# (the SHIPPED config E5 reads its roots/prefixes from) lives.
REPO_ROOT = Path(__file__).resolve().parents[2]

DAY = 86_400
NOW = 1_000_000_000.0


def _touch(path: Path, mtime: float) -> None:
    """Create *path* (and parents) as a small file with the given mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x')
    os.utime(path, (mtime, mtime))
