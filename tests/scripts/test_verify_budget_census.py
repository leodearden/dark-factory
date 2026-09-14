"""Tests for scripts/verify_budget_census.py — the READ-ONLY verify-budget census.

Ruling D17 (task 3353) deliverable 2. Deliverable 1 stamps host load on every
verify command; this censuses the resulting corpus so a module's verify budget
is DERIVED from a dated distribution instead of estimated.

NO TEST HERE ASSERTS A NUMBER READ FROM THE LIVE SUMMARY CORPUS.
This norm is inherited verbatim from the sibling suites in this directory —
tests/scripts/test_census_tagger_debris.py's module docstring records a
candidate count that moved 40 -> 43 -> 45 across a single task's planning
sessions — and scripts/tests/test_merge_lane_throughput.py runs entirely on
synthetic roots for the same reason. The corpus here is mutated continuously by
a running fleet: worktrees are created and pruned, and each attempt OVERWRITES
the previous summary, so it is both non-deterministic and survivorship-biased.
A test pinning "the corpus yields n=14, max 3753" is a guessed threshold that
goes red the moment any task verifies.

D17 does name numbers the census "must reproduce". Those are discharged as a
recorded reproduction run in the iteration log plus a provenance-stamped
constant beside `MEASURED_MODULE_SUITE_WORST_SECS` — which is what that
constant's own provenance block already demands ("RE-MEASURE BY REPEATING THAT
CENSUS, so the next figure is a repeat rather than a re-derivation") — never as
an assertion here. Every assertion below runs against a synthetic `tmp_path`
corpus whose contents the test controls exactly and whose expected numbers are
computed by hand.

THE PATH IS THE ONLY SOURCE OF THREE FACTS. A summary.json carries no task id,
no module prefix and no role, so all three come from where the file sits. That
makes path parsing load-bearing rather than incidental, and is why it gets its
own section here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from verify_budget_census import parse_record_path

# Both archive stamp spellings seen in the wild. The microsecond-bearing form
# dominates; the second-resolution form is what older records carry.
_STAMP_US = '20260914T123016_283575Z'
_STAMP_SEC = '20260816T080255Z'


def _worktree_record(root: Path, lane: str, name: str) -> Path:
    path = root / '.worktrees' / lane / '.task' / 'verify' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'commands': []}), encoding='utf-8')
    return path


def _archive_record(root: Path, task_id: str, name: str) -> Path:
    path = root / 'data' / 'verify-logs' / task_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'commands': []}), encoding='utf-8')
    return path


class TestFixtureBuildersEmitLiveShapedPaths:
    """Self-check: the builders above reproduce the shapes measured on the tree.

    Without this the whole suite could pass against a corpus shape that does
    not exist. The literals are the two real spellings, recorded here so a
    later reader can see what was measured rather than trusting the parser.
    """

    def test_the_worktree_shape(self, tmp_path):
        path = _worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json')

        assert path.relative_to(tmp_path).as_posix() == (
            '.worktrees/3353/.task/verify/attempt-1.orchestrator.summary.json'
        )

    def test_the_archive_shape(self, tmp_path):
        path = _archive_record(
            tmp_path, '3353', f'attempt-1.orchestrator.summary-{_STAMP_US}.json',
        )

        assert path.relative_to(tmp_path).as_posix() == (
            f'data/verify-logs/3353/attempt-1.orchestrator.summary-{_STAMP_US}.json'
        )


class TestParseRecordPathWorktreeCorpus:
    def test_the_five_facts_come_off_the_path(self, tmp_path):
        record = parse_record_path(
            _worktree_record(tmp_path, '3353', 'attempt-2.orchestrator.summary.json'),
        )

        assert record is not None
        assert record.task_id == '3353'
        assert record.attempt == 2
        assert record.module_prefix == 'orchestrator'
        assert record.role == 'task'
        assert record.corpus == 'worktree'

    def test_a_worktree_record_has_no_archive_stamp(self, tmp_path):
        record = parse_record_path(
            _worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json'),
        )

        assert record is not None
        assert record.archived_at is None

    def test_a_prefixless_record_yields_a_null_prefix(self, tmp_path):
        """`_make_infix(None)` is `''`, so the module is genuinely unknown here."""
        record = parse_record_path(
            _worktree_record(tmp_path, '3353', 'attempt-1.summary.json'),
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
            _worktree_record(tmp_path, '3353', f'attempt-1.{infix}.summary.json'),
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
            _archive_record(
                tmp_path, '5422', f'attempt-1.orchestrator.summary-{_STAMP_US}.json',
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
        assert record.archived_at == _STAMP_US

    def test_the_second_resolution_stamp_also_parses(self, tmp_path):
        """Older records carry no microseconds; both spellings are live."""
        record = parse_record_path(
            _archive_record(
                tmp_path, '5422', f'attempt-1.dashboard.summary-{_STAMP_SEC}.json',
            ),
        )

        assert record is not None
        assert record.archived_at == _STAMP_SEC

    def test_a_prefixless_archive_record_parses(self, tmp_path):
        """Measured: the live archive carries these (empty infix)."""
        record = parse_record_path(
            _archive_record(tmp_path, '5422', f'attempt-3.summary-{_STAMP_US}.json'),
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
            _archive_record(
                tmp_path, '5422', f'attempt-1.orchestrator.summary-{_STAMP_US}.json',
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
            _worktree_record(tmp_path, lane, 'attempt-1.orchestrator.summary.json'),
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
            _worktree_record(tmp_path, lane, 'attempt-1.orchestrator.summary.json'),
        )

        assert record is not None
        assert record.role != 'task'
        assert record.role == lane.split('-', 1)[0].lstrip('_')

    def test_a_lane_name_that_is_neither_shape_is_unknown(self, tmp_path):
        """Total, and never a guess: an unrecognised lane says so."""
        record = parse_record_path(
            _worktree_record(tmp_path, 'handmade', 'attempt-1.summary.json'),
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
            _worktree_record(tmp_path, '3353', name),
        ) is None

    def test_a_summary_outside_either_corpus_returns_none(self, tmp_path):
        stray = tmp_path / 'attempt-1.orchestrator.summary.json'
        stray.write_text('{}', encoding='utf-8')

        assert parse_record_path(stray) is None

    def test_the_archive_log_files_are_not_records(self, tmp_path):
        """`*.summary-*.json` must not admit the sibling `.log` legs."""
        assert parse_record_path(
            _archive_record(
                tmp_path, '5422', f'attempt-1.orchestrator.test-{_STAMP_US}.log',
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

        _worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json')
        _archive_record(
            tmp_path, '5422', f'attempt-1.orchestrator.summary-{_STAMP_US}.json',
        )

        corpus = load_records([tmp_path])

        assert {r.where.corpus for r in corpus.records} == {'worktree', 'archive'}
        assert corpus.skipped == ()

    def test_several_roots_are_walked(self, tmp_path):
        """`--project-root` is repeatable, like the sibling report's."""
        from verify_budget_census import load_records  # noqa: PLC0415

        one, two = tmp_path / 'one', tmp_path / 'two'
        _worktree_record(one, '1', 'attempt-1.orchestrator.summary.json')
        _worktree_record(two, '2', 'attempt-1.orchestrator.summary.json')

        corpus = load_records([one, two])

        assert sorted(r.where.task_id for r in corpus.records) == ['1', '2']

    def test_a_missing_root_is_not_a_crash(self, tmp_path):
        from verify_budget_census import load_records  # noqa: PLC0415

        corpus = load_records([tmp_path / 'absent'])

        assert corpus.records == ()
        assert corpus.skipped == ()

    def test_the_payload_is_carried_verbatim(self, tmp_path):
        from verify_budget_census import load_records  # noqa: PLC0415

        path = _worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json')
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

        path = _worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json')
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

    def test_every_selected_file_is_accounted_for(self, tmp_path):
        """The reconciliation property, asserted directly."""
        from verify_budget_census import load_records  # noqa: PLC0415

        good = _worktree_record(tmp_path, '1', 'attempt-1.orchestrator.summary.json')
        bad = _worktree_record(tmp_path, '2', 'attempt-1.orchestrator.summary.json')
        bad.write_text('{oops', encoding='utf-8')
        _worktree_record(tmp_path, '3', 'attempt-1.orchestrator.test.log')

        corpus = load_records([tmp_path])

        assert len(corpus.records) + len(corpus.skipped) == 2
        assert [r.where.path for r in corpus.records] == [good]

    def test_the_log_legs_beside_a_summary_are_not_selected(self, tmp_path):
        """Neither glob admits a `.log`, so they are not skips either."""
        from verify_budget_census import load_records  # noqa: PLC0415

        _worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.test.log')
        _archive_record(
            tmp_path, '5422', f'attempt-1.orchestrator.test-{_STAMP_US}.log',
        )

        corpus = load_records([tmp_path])

        assert corpus.records == ()
        assert corpus.skipped == ()


# The orchestrator module's declared test_command, as of this writing. Used
# ONLY to shape the synthetic fixtures below — never as the value under test.
# The census reads the real one out of `<root>/<prefix>/orchestrator.yaml`,
# which is what `TestTheExpectedCommandHasOneHome` pins.
_FULL_SUITE = (
    'uv run --directory orchestrator pytest tests/ --tb=short -q --timeout=300'
)


def _summary(*commands, **top):
    """A summary.json payload. Top-level fields default to a LOUD lint leg.

    Defaulting the top level to something LOUD and SHORT is deliberate: it is
    the shape that catches a census reading the wrong place. See
    `TestTheSelectorReadsTheCommandsArray`.
    """
    payload = {
        'category': 'lint_error',
        'cause_hint': '',
        'rc': -9,
        'timed_out': False,
        'cmd': 'uv run --directory orchestrator ruff check src/',
        'started_at': '2026-09-13T04:00:00+00:00',
        'duration_secs': 4.5,
        'commands': list(commands),
    }
    payload.update(top)
    return payload


def _leg(label='test', cmd=_FULL_SUITE, **overrides):
    entry = {
        'label': label,
        'cmd': cmd,
        'rc': 0,
        'timed_out': False,
        'started_at': '2026-09-13T04:00:00+00:00',
        'duration_secs': 3300.0,
        'segments': None,
        'load': None,
    }
    entry.update(overrides)
    return entry


def _corpus_with(tmp_path, *commands, prefix='orchestrator', lane='3353', **top):
    """A one-record synthetic root, plus the module yaml the selector reads."""
    module_yaml = tmp_path / prefix / 'orchestrator.yaml'
    module_yaml.parent.mkdir(parents=True, exist_ok=True)
    module_yaml.write_text(f'test_command: "{_FULL_SUITE}"\n', encoding='utf-8')

    path = _worktree_record(tmp_path, lane, f'attempt-1.{prefix}.summary.json')
    path.write_text(json.dumps(_summary(*commands, **top)), encoding='utf-8')
    return path


class TestTheSelectorReadsTheCommandsArray:
    """`commands[]`, never the top level — and the difference is not cosmetic.

    `_build_summary_payload`'s own docstring states the rule: the top-level
    rc/cmd/started_at/duration_secs come from "the loudest raw exit code", the
    run with the highest rc with a NEGATIVE rc sorting above every non-negative
    one. So on an attempt whose lint leg was killed, the top level describes
    the LINT leg — a 4-second command — while the test leg that actually ran
    the full suite for 55 minutes is only reachable inside `commands[]`.

    Real evidence in the corpus, not a hypothetical: an attempt whose `test`
    leg ran the verbatim full suite while its `lint`/`type` legs were
    FILE-SCOPED. A per-module duration census reading the top level would have
    censused the wrong command, at the wrong scope, and reported it as the
    suite.
    """

    def test_the_test_legs_duration_is_reported_not_the_top_levels(self, tmp_path):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, _leg(duration_secs=3300.0))

        selection = select_full_suite_legs(
            load_records([tmp_path]), root=tmp_path, prefix='orchestrator',
        )

        assert [leg.duration_secs for leg in selection.legs] == [3300.0]

    def test_a_loud_top_level_does_not_mask_a_green_test_leg(self, tmp_path):
        """The top level says rc=-9; the selected leg says rc=0. Both are true."""
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, _leg(rc=0))

        selection = select_full_suite_legs(
            load_records([tmp_path]), root=tmp_path, prefix='orchestrator',
        )

        assert [leg.rc for leg in selection.legs] == [0]


class TestTheFullSuiteShapeFilter:
    """Selection is by label AND by command SHAPE, each rejection counted."""

    def _select(self, tmp_path, *commands):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, *commands)
        return select_full_suite_legs(
            load_records([tmp_path]), root=tmp_path, prefix='orchestrator',
        )

    def test_the_verbatim_declared_command_is_selected(self, tmp_path):
        selection = self._select(tmp_path, _leg())

        assert len(selection.legs) == 1
        assert selection.rejected == {}

    def test_whitespace_only_differences_are_tolerated(self, tmp_path):
        """A re-rendered command may differ in spacing and still be the suite."""
        spaced = _FULL_SUITE.replace(' pytest ', '  pytest   ')
        selection = self._select(tmp_path, _leg(cmd=spaced))

        assert len(selection.legs) == 1

    @pytest.mark.parametrize(
        ('cmd', 'note'),
        [
            ('uv run --directory orchestrator pytest tests/test_foo.py -q', 'file-scoped'),
            (
                'uv run --directory orchestrator pytest tests/ --tb=short -q '
                '--timeout=300 -k "psi"',
                'k-narrowed',
            ),
            (
                'uv run --directory orchestrator pytest tests/ --tb=short -q '
                '--timeout=300 --lf',
                'lf-narrowed',
            ),
        ],
    )
    def test_a_narrowed_run_is_rejected_with_a_reason(self, tmp_path, cmd, note):
        """Their durations are not comparable to a full-suite run's — that is the
        entire reason this filter exists."""
        selection = self._select(tmp_path, _leg(cmd=cmd))

        assert selection.legs == (), f'{note} run was admitted'
        assert selection.rejected == {'command_mismatch': 1}

    def test_a_segmented_entry_is_rejected_with_its_own_reason(self, tmp_path):
        """A segmented leg's duration covers a DIFFERENT topology of the same
        chain, so it is excluded — and counted separately from a plain command
        mismatch, because it is a different fact about the corpus."""
        selection = self._select(
            tmp_path, _leg(segments=[{'index': 1, 'label': 'a'}]),
        )

        assert selection.legs == ()
        assert selection.rejected == {'segmented': 1}

    def test_a_non_test_leg_is_rejected(self, tmp_path):
        selection = self._select(
            tmp_path,
            _leg(label='lint', cmd='uv run --directory orchestrator ruff check src/'),
            _leg(label='type', cmd='uv run --directory orchestrator pyright src/'),
        )

        assert selection.legs == ()
        assert selection.rejected == {'label_mismatch': 2}

    def test_a_null_cmd_is_rejected_rather_than_compared(self, tmp_path):
        """A skipped leg ran nothing, so there is no duration to census."""
        selection = self._select(tmp_path, _leg(cmd=None, duration_secs=0.0))

        assert selection.legs == ()
        assert selection.rejected == {'no_cmd': 1}

    def test_every_entry_is_either_selected_or_counted(self, tmp_path):
        """The reconciliation property again, at the selector."""
        selection = self._select(
            tmp_path,
            _leg(),
            _leg(label='lint', cmd='uv run ruff check src/'),
            _leg(cmd='uv run --directory orchestrator pytest tests/test_foo.py'),
        )

        assert len(selection.legs) + sum(selection.rejected.values()) == 3

    def test_another_modules_record_is_rejected_on_prefix(self, tmp_path):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, _leg())
        other = _worktree_record(tmp_path, '4242', 'attempt-1.shared.summary.json')
        other.write_text(json.dumps(_summary(_leg())), encoding='utf-8')

        selection = select_full_suite_legs(
            load_records([tmp_path]), root=tmp_path, prefix='orchestrator',
        )

        assert len(selection.legs) == 1
        assert selection.rejected == {'prefix_mismatch': 1}

    def test_a_role_filter_excludes_other_lanes(self, tmp_path):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, _leg())
        merge = _worktree_record(
            tmp_path, '_merge-abc123', 'attempt-1.orchestrator.summary.json',
        )
        merge.write_text(json.dumps(_summary(_leg())), encoding='utf-8')

        selection = select_full_suite_legs(
            load_records([tmp_path]), root=tmp_path, prefix='orchestrator', role='task',
        )

        assert len(selection.legs) == 1
        assert selection.rejected == {'role_mismatch': 1}


class TestTheExpectedCommandHasOneHome:
    """The comparison target is READ from the module's yaml, never pasted here.

    A literal copy in the script would be a second home for a value that
    already has one, and would rot silently the moment the yaml changed: the
    census would reject every real record as a command mismatch and report n=0
    — which reads exactly like "this module never ran the full suite".
    """

    def test_the_command_comes_from_the_modules_own_yaml(self, tmp_path):
        from verify_budget_census import read_module_test_command  # noqa: PLC0415

        module_yaml = tmp_path / 'orchestrator' / 'orchestrator.yaml'
        module_yaml.parent.mkdir(parents=True)
        module_yaml.write_text('test_command: "pytest tests/ -q"\n', encoding='utf-8')

        assert read_module_test_command(tmp_path, 'orchestrator') == 'pytest tests/ -q'

    def test_a_changed_yaml_changes_what_is_selected(self, tmp_path):
        """The property a pasted literal would break — asserted, not asserted about."""
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, _leg(cmd='pytest tests/ --brand-new-flag'))
        (tmp_path / 'orchestrator' / 'orchestrator.yaml').write_text(
            'test_command: "pytest tests/ --brand-new-flag"\n', encoding='utf-8',
        )

        selection = select_full_suite_legs(
            load_records([tmp_path]), root=tmp_path, prefix='orchestrator',
        )

        assert len(selection.legs) == 1, (
            'the selector must track the yaml, not a literal pasted into the script'
        )

    def test_an_absent_module_yaml_is_reported_not_guessed(self, tmp_path):
        from verify_budget_census import read_module_test_command  # noqa: PLC0415

        assert read_module_test_command(tmp_path, 'nosuchmodule') is None

    def test_a_yaml_without_a_test_command_is_reported_not_guessed(self, tmp_path):
        from verify_budget_census import read_module_test_command  # noqa: PLC0415

        module_yaml = tmp_path / 'docs' / 'orchestrator.yaml'
        module_yaml.parent.mkdir(parents=True)
        module_yaml.write_text('lint_command: "ruff check ."\n', encoding='utf-8')

        assert read_module_test_command(tmp_path, 'docs') is None

    def test_an_unparseable_yaml_is_reported_not_guessed(self, tmp_path):
        """Read-only and total: a broken config is a None, never a traceback."""
        from verify_budget_census import read_module_test_command  # noqa: PLC0415

        module_yaml = tmp_path / 'broken' / 'orchestrator.yaml'
        module_yaml.parent.mkdir(parents=True)
        module_yaml.write_text('test_command: "unclosed\n  - [oops\n', encoding='utf-8')

        assert read_module_test_command(tmp_path, 'broken') is None
