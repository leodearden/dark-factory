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
from datetime import date
from pathlib import Path

from legibility import config as legibility_config
from legibility import inventory, sampling
from shared.transcript_archive import archive_task_transcripts

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


# ---------------------------------------------------------------------------
# E5 — legibility mining enumerates the archived transcript
# ---------------------------------------------------------------------------

# The SHIPPED per-project config the miner actually reads. E5 loads THIS file,
# not a fixture: γ's stated signal has two halves — the enumerate change works,
# AND the shipped config turns it on with no operator flip — and only reading
# the real YAML can assert the second.
SHIPPED_LEGIBILITY_YAML = REPO_ROOT / 'docs' / 'legibility' / 'legibility.yaml'

# A real orchestrated-task worktree cwd. Its encoding contains '--worktrees-',
# which is what makes sampling.classify_agent_class return 'orchestrated-task'
# and what inventory._enumerate's cheap <enc> pre-filter matches on.
WORKTREE_CWD = '/home/leo/src/dark-factory/.worktrees/2732'
SESSION_DATE = date(2026, 8, 17)
SESSION_SID = 'a1b2c3d4-0000-4000-8000-abcdefabcdef'


def _archive_a_real_session(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Produce an archive by running the REAL archiver, never by hand.

    Returns ``(project_root, archive_root, archived_transcript)``. The source
    is a per-task config dir whose encoded dir is ``encode_cwd(WORKTREE_CWD)``
    and whose transcript carries that same absolute path as its ``cwd`` line
    plus an explicit ISO-8601 ``timestamp`` — so the row pins a FIXED session
    date rather than depending on what "today" happens to be.

    Hand-writing the archive layout would make E5 assert only that the miner
    reads a shape the TEST invented; running α's writer is what makes it assert
    the miner reads the shape α actually produces.
    """
    project_root = tmp_path / 'fake-project-root'
    config_dir = project_root / '.task' / 'claude-config-2732'
    enc = inventory.encode_cwd(WORKTREE_CWD)
    src = config_dir / 'projects' / enc / f'{SESSION_SID}.jsonl'
    src.parent.mkdir(parents=True)
    src.write_text(
        f'{{"type":"user","cwd":"{WORKTREE_CWD}",'
        f'"timestamp":"{SESSION_DATE}T09:15:00+00:00",'
        f'"message":{{"role":"user","content":"run the gate"}}}}\n'
        f'{{"type":"assistant","cwd":"{WORKTREE_CWD}",'
        f'"timestamp":"{SESSION_DATE}T09:15:30+00:00"}}\n',
        encoding='utf-8',
    )

    archive_root = project_root / 'data' / 'orchestrator' / 'agent-transcripts'
    written = archive_task_transcripts(
        config_dir, '2732', SESSION_SID, archive_root=archive_root
    )
    assert written == 1
    archived = archive_root / '2732' / enc / f'{SESSION_SID}.jsonl'
    assert archived.exists()
    return project_root, archive_root, archived


class TestE5MiningEnumeratesTheArchive:
    """E5 — the orchestrator→legibility cross-tool seam, integrated.

    The archive is written by the REAL ``archive_task_transcripts`` and read by
    the REAL ``enumerate_sessions``, through the roots and prefixes the SHIPPED
    ``docs/legibility/legibility.yaml`` actually carries. The relative root is
    resolved against a TMP project root via the production
    ``resolve_agent_transcript_roots``, so the live (git-ignored,
    concurrently-written) ``data/orchestrator/agent-transcripts`` tree is never
    touched.
    """

    def test_shipped_config_turns_archive_mining_on_with_no_operator_flip(self):
        cfg = legibility_config.load_config(SHIPPED_LEGIBILITY_YAML)
        assert cfg.agent_transcript_roots == ['data/orchestrator/agent-transcripts']
        assert cfg.cwd_prefixes == ['/home/leo/src/dark-factory']

    def test_real_archive_is_enumerated_and_classified(self, tmp_path):
        project_root, _archive_root, archived = _archive_a_real_session(tmp_path)
        cfg = legibility_config.load_config(SHIPPED_LEGIBILITY_YAML)

        # Production resolution of the project_root-relative root, against the
        # TMP root — never the live tree.
        resolved = inventory.resolve_agent_transcript_roots(
            project_root, cfg.agent_transcript_roots
        )
        assert resolved == [project_root / 'data/orchestrator/agent-transcripts']

        # An empty stand-in for ~/.claude/projects, so everything enumerated
        # below came from the ARCHIVE root.
        fake_projects_root = tmp_path / 'claude-projects'
        fake_projects_root.mkdir()

        records = inventory.enumerate_sessions(
            fake_projects_root,
            cfg.cwd_prefixes,
            SESSION_DATE,
            agent_transcript_roots=resolved,
        )

        assert len(records) == 1
        record = records[0]
        assert record.path == archived
        assert record.cwd == WORKTREE_CWD
        assert record.encoded_dir == inventory.encode_cwd(WORKTREE_CWD)
        assert record.date == SESSION_DATE

        # Membership is decided on the REAL decoded cwd, not the encoded dir.
        assert inventory.is_member(record.cwd, cfg.cwd_prefixes)

        # ...and the session lands in the orchestrated-task stratum, which is
        # the whole point of mining the fleet archive.
        first_turn = sampling._find_first_user_turn(record.path)
        assert sampling.classify_agent_class(first_turn, record.path) == 'orchestrated-task'

    def test_empty_roots_default_is_byte_parity_with_today(self, tmp_path):
        """PARITY — the archive is opt-in. With the empty code default the walk
        is byte-identical to the projects-only path it always was, so nothing
        that does not pass roots can start seeing fleet transcripts."""
        project_root, _archive_root, _archived = _archive_a_real_session(tmp_path)
        cfg = legibility_config.load_config(SHIPPED_LEGIBILITY_YAML)
        fake_projects_root = tmp_path / 'claude-projects'
        fake_projects_root.mkdir()

        assert inventory.enumerate_sessions(
            fake_projects_root, cfg.cwd_prefixes, SESSION_DATE
        ) == []
        assert inventory.enumerate_sessions(
            fake_projects_root,
            cfg.cwd_prefixes,
            SESSION_DATE,
            agent_transcript_roots=(),
        ) == []
        # The archive really is non-empty — the parity above is not vacuous.
        assert inventory.enumerate_sessions(
            fake_projects_root,
            cfg.cwd_prefixes,
            SESSION_DATE,
            agent_transcript_roots=inventory.resolve_agent_transcript_roots(
                project_root, cfg.agent_transcript_roots
            ),
        )
