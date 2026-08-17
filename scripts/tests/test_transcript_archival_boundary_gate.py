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

VERDICT — ALL EIGHT ROWS GREEN over the integrated α/β/γ/δ paths, and NO
production change was required by any of them. That is the gate's finding, not
a formality: ε exists to answer whether four independently-merged leaves
actually compose end to end, and the answer measured here is yes. Because the
matrix is split three ways, NO SINGLE FILE CAN REPORT THAT VERDICT — each of
the three states it and names the other two, so a reader who lands on any one
of them can reach the whole result:

* this file — E5, E8 (7 rows incl. controls)
* ``orchestrator/tests/...`` — E1, E6, E2, E3, E7 (7 rows incl. controls)
* ``shared/tests/...`` — E4 (1 row)

Provenance (a point-in-time measurement, NOT an invariant to keep updated):
branch ``task/2732`` on base main ``7ef5ccdcf6``, each package's VERBATIM
``test_command`` — shared 4212 passed / orchestrator 17104 passed / scripts
3623 passed, all rc=0. Rows were additionally checked non-tautological by
mutation (each deliberate production mutant fails its row; all reverted).

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

import json
import logging
import os
from datetime import date
from pathlib import Path

import gc_agent_transcripts as gct
import pytest
from legibility import config as legibility_config
from legibility import inventory, sampling
from shared.transcript_archive import archive_task_transcripts

# scripts/tests/../../ = the repo root, where docs/legibility/legibility.yaml
# (the SHIPPED config E5 reads its roots/prefixes from) lives.
REPO_ROOT = Path(__file__).resolve().parents[2]

DAY = 86_400
NOW = 1_000_000_000.0


def _touch(path: Path, mtime: float, content: bytes = b'x') -> None:
    """Create *path* (and parents) with *content* and the given mtime.

    The ``os.utime`` stamp is the load-bearing half for E8: α mirrors a SOURCE
    transcript's mtime onto its archived copy, and that mirrored mtime is what
    δ's ``scan_task_dirs`` reads back as the task dir's retention age.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
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

    def test_residual_gz_is_not_enumerated_but_is_counted_and_announced(
        self, tmp_path, caplog
    ):
        """RESIDUAL-GZ CONTRACT — the honest successor to Appendix B's "the
        archived gz transcript" wording for E5.

        Task 3618 dropped gzip from the archive, but the destructive migration
        sweep that converts the EXISTING corpus
        (``scripts/migrate_transcript_archive_gunzip.py --apply``) is a
        human-operated step. Between that merge and the operator's run, every
        pre-3618 ``.jsonl.gz`` is simply not enumerated — silently absent from
        the mined corpus. "Accepted" must not mean "invisible" (INV-4), so the
        gap announces itself: counted by :func:`count_residual_gz` (the
        machine-readable half) and one greppable WARNING per walk naming the
        migration script (the human half). Nothing here pins a gunzip
        round-trip, because there is no longer one to pin; this is where the gz
        dimension of the matrix now lives.
        """
        project_root, archive_root, archived = _archive_a_real_session(tmp_path)
        cfg = legibility_config.load_config(SHIPPED_LEGIBILITY_YAML)
        resolved = inventory.resolve_agent_transcript_roots(
            project_root, cfg.agent_transcript_roots
        )
        fake_projects_root = tmp_path / 'claude-projects'
        fake_projects_root.mkdir()

        # A pre-3618 archive sitting beside its post-3618 successor.
        residual = archived.with_name(f'{SESSION_SID}-older.jsonl.gz')
        residual.write_bytes(b'\x1f\x8b\x08\x00 not actually read by anyone ')

        with caplog.at_level(logging.WARNING, logger='legibility.inventory'):
            records = inventory.enumerate_sessions(
                fake_projects_root,
                cfg.cwd_prefixes,
                SESSION_DATE,
                agent_transcript_roots=resolved,
            )

        # NOT enumerated — the walk is a bare rglob('*.jsonl') and stays one.
        assert [r.path for r in records] == [archived]

        # ...but COUNTED (machine-readable half).
        assert inventory.count_residual_gz(archive_root) == 1

        # ...and ANNOUNCED exactly once per walk, naming the migration script
        # so the line is actionable rather than merely alarming (human half).
        warnings = [
            r for r in caplog.records
            if r.name == 'legibility.inventory' and r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert 'scripts/migrate_transcript_archive_gunzip.py' in message
        assert str(archive_root) in message

        # The signal disappears on its own once the migration has run.
        residual.unlink()
        assert inventory.count_residual_gz(archive_root) == 0


# ---------------------------------------------------------------------------
# E8 — the retention GC prunes by cap, loudly; default caps are a no-op
# ---------------------------------------------------------------------------

# Five per-task archive dirs, NEWEST FIRST: GC_TASK_IDS[i] carries mtime
# NOW - i*DAY, so [0] is the newest and [4] the oldest. All five sit far inside
# the default 90-day age cap, so the age arm never fires and the count arm is
# the only thing the cap row measures.
GC_TASK_IDS = ('8100', '8101', '8102', '8103', '8104')
GC_CAP = 2
GC_KEPT_IDS = GC_TASK_IDS[:GC_CAP]
GC_PRUNED_IDS = GC_TASK_IDS[GC_CAP:]

GC_LOG_PREFIX = 'gc_agent_transcripts:'


def _gc_transcript_bytes(task_id: str) -> bytes:
    """Per-task transcript content, so "survived intact" is content-checkable."""
    return (
        f'{{"type":"user","cwd":"/home/leo/src/dark-factory/.worktrees/{task_id}",'
        f'"message":{{"role":"user","content":"task {task_id}"}}}}\n'
    ).encode()


def _archive_five_real_task_dirs(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """Build a five-task archive by running the REAL archiver once per task.

    Returns ``(archive_root, {task_id: archived_transcript})``.

    Each source transcript is stamped to a distinct fixed epoch with
    ``os.utime`` BEFORE archival, and α's :func:`_archive_one` mirrors that
    mtime onto the published copy — which is precisely what δ's
    :func:`gc_agent_transcripts.scan_task_dirs` reads back as the dir's
    retention age. Hand-building ``<root>/<id>/enc/x.jsonl`` with a stamped
    mtime (what ``test_gc_agent_transcripts.py`` does, correctly, for a δ-only
    unit) would assert only that δ can read a shape the TEST invented; driving
    α's writer is what makes E8 a cross-leaf composition and lets a drift
    between the mirror and the scan actually fail this row.
    """
    project_root = tmp_path / 'fake-project-root'
    archive_root = project_root / 'data' / 'orchestrator' / 'agent-transcripts'

    archived: dict[str, Path] = {}
    for index, task_id in enumerate(GC_TASK_IDS):
        # Each task really does run in its own worktree, hence its own <enc>.
        enc = inventory.encode_cwd(f'/home/leo/src/dark-factory/.worktrees/{task_id}')
        sid = f'{task_id}0000-0000-4000-8000-abcdefabcdef'
        config_dir = project_root / '.task' / f'claude-config-{task_id}'
        src = config_dir / 'projects' / enc / f'{sid}.jsonl'
        _touch(src, NOW - index * DAY, content=_gc_transcript_bytes(task_id))

        assert archive_task_transcripts(
            config_dir, task_id, sid, archive_root=archive_root
        ) == 1
        dest = archive_root / task_id / enc / f'{sid}.jsonl'
        assert dest.exists()
        archived[task_id] = dest

    return archive_root, archived


def _gc_log_messages(caplog) -> list[str]:
    """Formatted INFO+ messages emitted by the GC's own logger."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == 'gc_agent_transcripts'
    ]


class TestE8RetentionGcPrunesByCap:
    """E8 — δ's retention sweep run for real over an archive α actually wrote.

    Both halves drive ``gc_agent_transcripts.main`` in-process (so ``caplog``
    can see the LOUD lines that are half the contract) with an explicit
    ``--now``, against a tmp archive root — the live, concurrently-written
    ``data/orchestrator/agent-transcripts`` tree is never swept.
    """

    def test_alpha_mtime_mirror_is_what_delta_reads_as_retention_age(self, tmp_path):
        """The cross-leaf seam E8 rests on, asserted rather than assumed.

        If α ever stamps the archived copy with ``now`` instead of mirroring the
        source mtime, every task dir would read to the GC as freshly active and
        retention would silently stop expiring anything — this row is where that
        drift surfaces.
        """
        archive_root, _archived = _archive_five_real_task_dirs(tmp_path)

        scanned = dict(gct.scan_task_dirs(archive_root))

        assert set(scanned) == {archive_root / task_id for task_id in GC_TASK_IDS}
        for index, task_id in enumerate(GC_TASK_IDS):
            assert scanned[archive_root / task_id] == pytest.approx(
                NOW - index * DAY, abs=1
            )

    def test_count_cap_prunes_the_oldest_dirs_loudly(self, tmp_path, caplog, capsys):
        """(a) CAP — ``--max-task-dirs 2`` over a real 5-task archive removes
        the 3 oldest dirs, keeps the 2 newest INTACT, and says so both ways:
        one greppable INFO line per removal plus the summary counts, and a
        machine-readable JSON report on stdout."""
        archive_root, archived = _archive_five_real_task_dirs(tmp_path)
        caplog.set_level(logging.INFO, logger='gc_agent_transcripts')

        # δ's OWN pre-sweep view of the tree, so "the newest two" below is
        # MEASURED off the archive's real mtimes rather than assumed from the
        # order the fixture happened to build them in. Newest-first with
        # select_prunable's exact tiebreak (-mtime, path); the five mtimes are
        # a day apart, so the tiebreak never actually engages.
        before = dict(gct.scan_task_dirs(archive_root))
        newest_first = sorted(before, key=lambda path: (-before[path], str(path)))
        expected_kept = set(newest_first[:GC_CAP])
        expected_pruned = set(newest_first[GC_CAP:])
        # The fixture's intent and the measured order agree — if they ever stop
        # agreeing, that is the fixture lying, and it fails here rather than
        # silently redefining what the rest of the row asserts.
        assert expected_kept == {archive_root / t for t in GC_KEPT_IDS}
        assert expected_pruned == {archive_root / t for t in GC_PRUNED_IDS}

        exit_code = gct.main([
            '--root', str(archive_root),
            '--max-task-dirs', str(GC_CAP),
            '--now', str(NOW),
        ])

        assert exit_code == 0

        # The survivors are exactly the NEWEST GC_CAP dirs by mtime — not
        # merely "two of them". A GC that kept the right COUNT but the wrong
        # dirs (an inverted sort) would satisfy every count in the report and
        # is caught only here.
        surviving = {child for child in archive_root.iterdir() if child.is_dir()}
        assert surviving == expected_kept
        # ...and nothing was both kept and pruned: the two sets partition the
        # scan, so a dir cannot be reported removed while still on disk.
        assert surviving & expected_pruned == set()
        assert surviving | expected_pruned == set(before)

        # The 2 newest survive INTACT — the dir is still there AND its archived
        # transcript is still readable and byte-correct, not an emptied husk.
        for task_id in GC_KEPT_IDS:
            assert (archive_root / task_id).is_dir()
            assert archived[task_id].read_bytes() == _gc_transcript_bytes(task_id)
        # ...and the 3 oldest are gone from disk, not merely classified.
        for task_id in GC_PRUNED_IDS:
            assert not (archive_root / task_id).exists()

        report = json.loads(capsys.readouterr().out)
        assert report['scanned'] == 5
        assert report['kept'] == 2
        assert report['pruned'] == 3
        assert report['removed'] == 3
        assert report['failed'] == 0
        assert report['check'] is False
        # The age arm never fired: every drop is attributed to the count cap.
        assert report['reason_counts'] == {'count': 3}
        # The report names the dirs that actually went, measured-newest-first.
        assert {Path(p) for p in report['removed_paths']} == expected_pruned

        messages = _gc_log_messages(caplog)

        # LOUD, per removal: prefix + the exact dir + the reason.
        removal_lines = [m for m in messages if 'pruned task dir' in m]
        assert len(removal_lines) == 3
        for task_id in GC_PRUNED_IDS:
            assert any(
                m.startswith(GC_LOG_PREFIX)
                and str(archive_root / task_id) in m
                and 'reason=count' in m
                for m in removal_lines
            ), f'no removal line for {task_id} in {removal_lines!r}'

        # ...and once for the sweep as a whole.
        summary_lines = [m for m in messages if 'sweep complete' in m]
        assert len(summary_lines) == 1
        assert summary_lines[0].startswith(GC_LOG_PREFIX)
        assert 'scanned=5 kept=2 pruned=3 removed=3 failed=0' in summary_lines[0]

    def test_default_caps_are_a_no_op(self, tmp_path, caplog, capsys):
        """(b) DEFAULT CAPS — the same real archive under the shipped defaults
        (90 days / 5000 dirs) is untouched: nothing removed, nothing even
        classified as prunable, and not one prune line emitted."""
        archive_root, archived = _archive_five_real_task_dirs(tmp_path)
        caplog.set_level(logging.INFO, logger='gc_agent_transcripts')

        exit_code = gct.main(['--root', str(archive_root), '--now', str(NOW)])

        assert exit_code == 0

        for task_id in GC_TASK_IDS:
            assert (archive_root / task_id).is_dir()
            assert archived[task_id].read_bytes() == _gc_transcript_bytes(task_id)

        report = json.loads(capsys.readouterr().out)
        # The defaults really are the shipped ones, not a fixture's invention.
        assert report['caps'] == {
            'max_age_days': gct.DEFAULT_MAX_AGE_DAYS,
            'max_task_dirs': gct.DEFAULT_MAX_TASK_DIRS,
        }
        assert (gct.DEFAULT_MAX_AGE_DAYS, gct.DEFAULT_MAX_TASK_DIRS) == (90, 5000)
        assert report['scanned'] == 5
        assert report['kept'] == 5
        assert report['pruned'] == 0
        assert report['removed'] == 0
        assert report['reason_counts'] == {}

        messages = _gc_log_messages(caplog)
        assert [m for m in messages if 'task dir' in m] == []
        assert any('sweep complete' in m for m in messages)
