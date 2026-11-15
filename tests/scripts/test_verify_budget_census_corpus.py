"""parse_record_path and load_records: what a file's location says about the
record, and a walk that loads every selected file or counts why not.

THE PATH IS THE ONLY SOURCE OF TWO FACTS, AND THE FALLBACK FOR A THIRD. A
summary.json carries no task id and no module prefix, so both come from where
the file sits. Since task 5671 it carries its role; a record written before
that falls back to the role its path implies. That makes path parsing
load-bearing rather than incidental, and is why it gets its own section here.

The synthetic-corpus norm lives in tests/scripts/verify_budget_census_fixture.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from verify_budget_census import parse_record_path
from verify_budget_census_fixture import (
    FULL_SUITE,
    STAMP_SEC,
    STAMP_US,
    command_entry,
    run_main,
    summary_payload,
    write_archive_record,
    write_module_yaml,
    write_worktree_record,
)


class TestFixtureBuildersEmitLiveShapedPaths:
    """Self-check: the helper's record builders reproduce the shapes measured on the tree.

    Those builders are `verify_budget_census_fixture.write_worktree_record` /
    `write_archive_record`. Without this the whole suite could pass against a
    corpus shape that does not exist. The literals are the two real spellings,
    recorded here so a later reader can see what was measured rather than
    trusting the parser.
    """

    def test_the_worktree_shape(self, tmp_path):
        path = write_worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json')

        assert path.relative_to(tmp_path).as_posix() == (
            '.worktrees/3353/.task/verify/attempt-1.orchestrator.summary.json'
        )

    def test_the_archive_shape(self, tmp_path):
        path = write_archive_record(
            tmp_path, '3353', f'attempt-1.orchestrator.summary-{STAMP_US}.json',
        )

        assert path.relative_to(tmp_path).as_posix() == (
            f'data/verify-logs/3353/attempt-1.orchestrator.summary-{STAMP_US}.json'
        )


class TestParseRecordPathWorktreeCorpus:
    def test_the_five_facts_come_off_the_path(self, tmp_path):
        record = parse_record_path(
            write_worktree_record(tmp_path, '3353', 'attempt-2.orchestrator.summary.json'),
        )

        assert record is not None
        assert record.task_id == '3353'
        assert record.attempt == 2
        assert record.module_prefix == 'orchestrator'
        assert record.role == 'task'
        assert record.corpus == 'worktree'

    def test_a_worktree_record_has_no_archive_stamp(self, tmp_path):
        record = parse_record_path(
            write_worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json'),
        )

        assert record is not None
        assert record.archived_at is None

    def test_a_prefixless_record_yields_a_null_prefix(self, tmp_path):
        """`_make_infix(None)` is `''`, so the module is genuinely unknown here."""
        record = parse_record_path(
            write_worktree_record(tmp_path, '3353', 'attempt-1.summary.json'),
        )

        assert record is not None
        assert record.module_prefix is None

    @pytest.mark.parametrize(
        'infix',
        ['orchestrator', 'fused-memory', 'scripts', 'tests_scripts', '__fallback__',
         'remote-buildbox', 'cockpit'],
    )
    def test_every_infix_seen_in_the_wild_parses_as_a_prefix(self, tmp_path, infix):
        """Including `__fallback__` and a `remote-<host>` leg — parsed, not crashed."""
        record = parse_record_path(
            write_worktree_record(tmp_path, '3353', f'attempt-1.{infix}.summary.json'),
        )

        assert record is not None
        assert record.module_prefix == infix


class TestParseRecordPathArchiveCorpus:
    """The D17 premise correction, pinned.

    The ruling's text globs `**/*.summary.json`. Measured, that selects ZERO
    archive records: the stamp sits AFTER the word `summary`, so the archive
    spelling is `*.summary-<STAMP>.json`. A census built on the ruling's glob
    would have reported the worktree corpus alone and called it everything.
    """

    def test_the_timestamped_form_is_matched(self, tmp_path):
        record = parse_record_path(
            write_archive_record(
                tmp_path, '5422', f'attempt-1.orchestrator.summary-{STAMP_US}.json',
            ),
        )

        assert record is not None, (
            'the archive spelling is *.summary-<STAMP>.json — a *.summary.json '
            'glob selects zero archive records (D17 premise correction)'
        )
        assert record.corpus == 'archive'
        assert record.task_id == '5422'
        assert record.attempt == 1
        assert record.module_prefix == 'orchestrator'
        assert record.archived_at == STAMP_US

    def test_the_second_resolution_stamp_also_parses(self, tmp_path):
        """Older records carry no microseconds; both spellings are live."""
        record = parse_record_path(
            write_archive_record(
                tmp_path, '5422', f'attempt-1.dashboard.summary-{STAMP_SEC}.json',
            ),
        )

        assert record is not None
        assert record.archived_at == STAMP_SEC

    def test_a_prefixless_archive_record_parses(self, tmp_path):
        """Measured: the live archive carries these (empty infix)."""
        record = parse_record_path(
            write_archive_record(tmp_path, '5422', f'attempt-3.summary-{STAMP_US}.json'),
        )

        assert record is not None
        assert record.module_prefix is None
        assert record.attempt == 3

    def test_the_archive_role_is_unknown_not_guessed(self, tmp_path):
        """The archive carries BOTH task-path and merge-path legs.

        A summary.json has no role field and the archive path does not say
        which lane wrote it, so the role is genuinely not knowable here. `None`
        says that; defaulting it to 'task' would fold merge legs — which run a
        different breadth on a different budget — into a task distribution and
        call the result measured.
        """
        record = parse_record_path(
            write_archive_record(
                tmp_path, '5422', f'attempt-1.orchestrator.summary-{STAMP_US}.json',
            ),
        )

        assert record is not None
        assert record.role is None


class TestParseRecordPathRoleClassification:
    @pytest.mark.parametrize(
        ('lane', 'role'),
        [
            ('3353', 'task'),
            ('774', 'task'),
            ('_merge-a1b2c3', 'merge'),
            ('_mainprobe-c4ece', 'probe'),
        ],
    )
    def test_the_named_lanes_classify_as_ruled(self, tmp_path, lane, role):
        record = parse_record_path(
            write_worktree_record(tmp_path, lane, 'attempt-1.orchestrator.summary.json'),
        )

        assert record is not None
        assert record.role == role

    @pytest.mark.parametrize('lane', ['_mainsweep-9ac2bd3', '_offline-7f1a'])
    def test_an_unruled_underscore_lane_is_not_called_a_task(self, tmp_path, lane):
        """MEASURED on the tree (2026-09-14): `.worktrees/` holds FIVE lane
        classes, not three — 774 bare task ids, and 5 `_mainsweep-*`, 5
        `_mainprobe-*`, 3 `_merge-*`, 1 `_offline-*`.

        A fall-through that called every non-merge, non-probe lane a `task`
        would silently file a sweep or offline-lane verify into the task
        distribution the budget is derived from. Those lanes carry zero summary
        records TODAY, so this is latent rather than live — which is exactly
        when to fix it, since the day one appears there would be no signal.
        """
        record = parse_record_path(
            write_worktree_record(tmp_path, lane, 'attempt-1.orchestrator.summary.json'),
        )

        assert record is not None
        assert record.role != 'task'
        assert record.role == lane.split('-', 1)[0].lstrip('_')

    def test_a_lane_name_that_is_neither_shape_is_unknown(self, tmp_path):
        """Total, and never a guess: an unrecognised lane says so."""
        record = parse_record_path(
            write_worktree_record(tmp_path, 'handmade', 'attempt-1.summary.json'),
        )

        assert record is not None
        assert record.role is None


class TestParseRecordPathRejections:
    """Not every file under these trees is a summary record."""

    @pytest.mark.parametrize(
        'name',
        [
            'attempt-1.orchestrator.test.log',
            'attempt-1.orchestrator.summary.json.tmp',
            'verify_warmed',
            'attempt-X.orchestrator.summary.json',
            'summary.json',
        ],
    )
    def test_a_non_record_returns_none(self, tmp_path, name):
        assert parse_record_path(
            write_worktree_record(tmp_path, '3353', name),
        ) is None

    def test_a_summary_outside_either_corpus_returns_none(self, tmp_path):
        stray = tmp_path / 'attempt-1.orchestrator.summary.json'
        stray.write_text('{}', encoding='utf-8')

        assert parse_record_path(stray) is None

    def test_the_archive_log_files_are_not_records(self, tmp_path):
        """`*.summary-*.json` must not admit the sibling `.log` legs."""
        assert parse_record_path(
            write_archive_record(
                tmp_path, '5422', f'attempt-1.orchestrator.test-{STAMP_US}.log',
            ),
        ) is None


class TestLoadRecordsIsTotal:
    """The walker must never quietly shrink its own corpus.

    That is the exact failure this task exists to end: a budget derived from a
    distribution that silently dropped the records it could not read is not a
    measurement, and nothing in its output would say so. Every file the globs
    select is either a loaded record or a COUNTED skip carrying its reason, and
    `len(records) + len(skipped)` reconciles against the corpus size.
    """

    def test_both_corpora_are_walked(self, tmp_path):
        from verify_budget_census import load_records  # noqa: PLC0415

        write_worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json')
        write_archive_record(
            tmp_path, '5422', f'attempt-1.orchestrator.summary-{STAMP_US}.json',
        )

        corpus = load_records([tmp_path])

        assert {r.where.corpus for r in corpus.records} == {'worktree', 'archive'}
        assert corpus.skipped == ()

    def test_several_roots_are_walked(self, tmp_path):
        """`--project-root` is repeatable, like the sibling report's."""
        from verify_budget_census import load_records  # noqa: PLC0415

        one, two = tmp_path / 'one', tmp_path / 'two'
        write_worktree_record(one, '1', 'attempt-1.orchestrator.summary.json')
        write_worktree_record(two, '2', 'attempt-1.orchestrator.summary.json')

        corpus = load_records([one, two])

        assert sorted(r.where.task_id for r in corpus.records) == ['1', '2']

    def test_a_missing_root_is_not_a_crash(self, tmp_path):
        from verify_budget_census import load_records  # noqa: PLC0415

        corpus = load_records([tmp_path / 'absent'])

        assert corpus.records == ()
        assert corpus.skipped == ()

    def test_the_payload_is_carried_verbatim(self, tmp_path):
        from verify_budget_census import load_records  # noqa: PLC0415

        path = write_worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json')
        payload = {'category': 'clean', 'rc': 0, 'commands': [{'label': 'test'}]}
        path.write_text(json.dumps(payload), encoding='utf-8')

        corpus = load_records([tmp_path])

        assert [r.payload for r in corpus.records] == [payload]

    @pytest.mark.parametrize(
        ('content', 'reason'),
        [
            ('not json at all', 'not_json'),
            ('', 'not_json'),
            ('[1, 2, 3]', 'not_an_object'),
            ('"a string"', 'not_an_object'),
        ],
    )
    def test_an_unusable_file_is_counted_with_its_reason(
        self, tmp_path, content, reason,
    ):
        from verify_budget_census import load_records  # noqa: PLC0415

        path = write_worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json')
        path.write_text(content, encoding='utf-8')

        corpus = load_records([tmp_path])

        assert corpus.records == ()
        assert [s.reason for s in corpus.skipped] == [reason]
        assert corpus.skipped[0].path == path

    def test_an_unreadable_file_is_counted_not_raised(self, tmp_path):
        """A directory standing where a summary should be — the never-raise arm."""
        from verify_budget_census import load_records  # noqa: PLC0415

        stub = tmp_path / '.worktrees' / '3353' / '.task' / 'verify'
        (stub / 'attempt-1.orchestrator.summary.json').mkdir(parents=True)

        corpus = load_records([tmp_path])

        assert corpus.records == ()
        assert [s.reason for s in corpus.skipped] == ['unreadable']

    def test_a_read_torn_mid_multibyte_character_is_unreadable(self, tmp_path):
        """Summaries are written `ensure_ascii=False`, so this is reachable."""
        from verify_budget_census import _read_payload  # noqa: PLC0415

        raw = json.dumps({'cmd': 'pytest — dash'}, ensure_ascii=False).encode('utf-8')
        torn = tmp_path / 'attempt-1.orchestrator.summary.json'
        torn.write_bytes(raw[:raw.find('—'.encode()) + 1])

        assert _read_payload(torn) == (None, 'unreadable')

    def test_every_selected_file_is_accounted_for(self, tmp_path):
        """The reconciliation property, asserted directly."""
        from verify_budget_census import load_records  # noqa: PLC0415

        good = write_worktree_record(tmp_path, '1', 'attempt-1.orchestrator.summary.json')
        bad = write_worktree_record(tmp_path, '2', 'attempt-1.orchestrator.summary.json')
        bad.write_text('{oops', encoding='utf-8')
        write_worktree_record(tmp_path, '3', 'attempt-1.orchestrator.test.log')

        corpus = load_records([tmp_path])

        assert len(corpus.records) + len(corpus.skipped) == 2
        assert [r.where.path for r in corpus.records] == [good]

    def test_the_log_legs_beside_a_summary_are_not_selected(self, tmp_path):
        """Neither glob admits a `.log`, so they are not skips either."""
        from verify_budget_census import load_records  # noqa: PLC0415

        write_worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.test.log')
        write_archive_record(
            tmp_path, '5422', f'attempt-1.orchestrator.test-{STAMP_US}.log',
        )

        corpus = load_records([tmp_path])

        assert corpus.records == ()
        assert corpus.skipped == ()


def _live_and_archived(
    root: Path,
    payload: dict[str, Any],
    *,
    task_id: str = '5199',
    attempt: int = 1,
    prefix: str = 'orchestrator',
    stamp: str = STAMP_US,
) -> tuple[Path, Path]:
    """One attempt as a live task leaves it: a worktree copy and its archive twin."""
    worktree = write_worktree_record(root, task_id, f'attempt-{attempt}.{prefix}.summary.json')
    archive = write_archive_record(
        root, task_id, f'attempt-{attempt}.{prefix}.summary-{stamp}.json',
    )
    for path in (worktree, archive):
        path.write_text(json.dumps(payload), encoding='utf-8')
    return worktree, archive


class TestALiveAttemptIsCountedOnce:
    """A live task's attempt is one record, even though it sits in two corpora.

    Task 5199 archives every task-path summary, so while a task's worktree
    lives each attempt sits in BOTH corpora with identical content. The census
    must count it once and account for the other copy as a counted skip.
    """

    def test_the_archived_twin_is_a_counted_skip_not_a_second_record(self, tmp_path):
        from verify_budget_census import load_records  # noqa: PLC0415

        wt, ar = _live_and_archived(tmp_path, summary_payload(command_entry()))

        corpus = load_records([tmp_path])

        assert [r.where.path for r in corpus.records] == [wt]
        assert [(s.path, s.reason) for s in corpus.skipped] == [
            (ar, 'duplicate_of_worktree_record'),
        ]
        assert len(corpus.records) + len(corpus.skipped) == 2

    def test_a_role_filtered_census_still_sees_the_live_attempt(self, tmp_path):
        from verify_budget_census import (  # noqa: PLC0415
            load_records,
            select_full_suite_legs,
        )

        _live_and_archived(tmp_path, summary_payload(command_entry()))

        selection = select_full_suite_legs(
            load_records([tmp_path]),
            expected=FULL_SUITE,
            prefix='orchestrator',
            role='task',
        )

        assert len(selection.legs) == 1
        assert selection.rejected_records == {}

    def test_a_default_census_reports_n_once_and_names_the_skip(
        self, tmp_path, capsys,
    ):
        write_module_yaml(tmp_path)
        _live_and_archived(tmp_path, summary_payload(command_entry(duration_secs=3300.0)))

        rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )
        report = json.loads(out)

        assert rc == 0
        assert report['overall']['durations']['n'] == 1
        assert report['corpus']['records'] == 1
        assert report['corpus']['skipped'] == {'duplicate_of_worktree_record': 1}

    def test_an_earlier_run_of_the_same_attempt_still_counts(self, tmp_path):
        """An infra retry reuses attempt_id and overwrites the worktree copy.

        That is workflow.py::_run_scoped_verification_with_infra_retry. The
        archive keeps both runs; only the rerun has a worktree twin, so twin
        identity has to be content and not name.
        """
        from verify_budget_census import load_records  # noqa: PLC0415

        first = summary_payload(
            command_entry(started_at='2026-09-13T04:00:00+00:00', duration_secs=3300.0),
        )
        rerun = summary_payload(
            command_entry(started_at='2026-09-13T05:00:00+00:00', duration_secs=3400.0),
        )
        wt, rerun_archived = _live_and_archived(
            tmp_path, rerun, stamp='20260913T050000_000001Z',
        )
        first_archived = write_archive_record(
            tmp_path, '5199', f'attempt-1.orchestrator.summary-{STAMP_US}.json',
        )
        first_archived.write_text(json.dumps(first), encoding='utf-8')

        corpus = load_records([tmp_path])

        assert {r.where.path for r in corpus.records} == {wt, first_archived}
        assert [(s.path, s.reason) for s in corpus.skipped] == [
            (rerun_archived, 'duplicate_of_worktree_record'),
        ]

    @pytest.mark.parametrize(
        ('task_id', 'attempt', 'prefix'),
        [('5422', 1, 'orchestrator'), ('5199', 2, 'orchestrator'), ('5199', 1, 'shared')],
    )
    def test_identical_content_under_another_task_attempt_or_module_is_not_a_twin(
        self, tmp_path, task_id, attempt, prefix,
    ):
        from verify_budget_census import load_records  # noqa: PLC0415

        payload = summary_payload(command_entry())
        worktree = write_worktree_record(
            tmp_path, '5199', 'attempt-1.orchestrator.summary.json',
        )
        worktree.write_text(json.dumps(payload), encoding='utf-8')
        archive = write_archive_record(
            tmp_path, task_id, f'attempt-{attempt}.{prefix}.summary-{STAMP_US}.json',
        )
        archive.write_text(json.dumps(payload), encoding='utf-8')

        corpus = load_records([tmp_path])

        assert len(corpus.records) == 2
        assert corpus.skipped == ()
