"""ε B+H boundary gate — shared-primitive half (task 2732).

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

**This file owns E4.** Its two siblings own the rest:

* ``orchestrator/tests/test_transcript_archival_boundary_gate.py`` — E1, E6,
  E2, E3, E7 (the producer-hook and teardown-backstop rows).
* ``scripts/tests/test_transcript_archival_boundary_gate.py`` — E5, E8 (the
  legibility-mining and retention-GC rows).

The gate is three files rather than one because ``verify`` is directory-scoped:
each package's ``orchestrator.yaml`` declares its own ``test_command``, so a
single cross-package module would run in exactly one lane and a shared-only or
scripts-only diff would never exercise its rows. This file additionally imports
ONLY :mod:`shared.transcript_archive`, so the package dependency direction
holds and ``test_pure_stdlib_leaves.py``'s pure-stdlib-leaf guarantee for that
module is not disturbed.

ARCHIVE FORMAT: plain ``.jsonl``, byte-verbatim, NO added suffix. Task 3618
(leaf α of ``plans/transcript-preservation-seam-prd.md``) dropped gzip from the
archive AFTER the PRD was written, so Appendix B's "gz round-trips" wording for
E1/E5 is stale — do not read it as a gap. The residual-``.jsonl.gz`` contract
that survived 3618 (not enumerated, but counted + warned) is pinned by E5 in
the scripts file.

No row mocks the component under test: every row here drives the REAL
:func:`shared.transcript_archive.archive_task_transcripts`.

Fixtures are kept module-local (no conftest.py additions), matching
``orchestrator/tests/test_transcript_archive_backstop.py``'s documented choice.
"""

from __future__ import annotations

from pathlib import Path

# The encoded-project dir the fake transcripts are laid down under.
ENC = '-home-leo-projX'


def _config_dir(root: Path, task_id: str) -> Path:
    """The per-task Claude config dir shape the archiver is pointed at."""
    return root / '.task' / f'claude-config-{task_id}'


def _write_transcript(config_dir: Path, rel: str, data: bytes) -> Path:
    """Lay down a transcript at ``<config_dir>/projects/<rel>`` and return it."""
    p = config_dir / 'projects' / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p
