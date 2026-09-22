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
from typing import Any

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


def _leg(label: str = 'test', cmd: str | None = _FULL_SUITE, **overrides):
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


def _module_yaml(root: Path, prefix: str = 'orchestrator') -> Path:
    """The `<root>/<prefix>/orchestrator.yaml` the selector reads its command from."""
    path = root / prefix / 'orchestrator.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'test_command: "{_FULL_SUITE}"\n', encoding='utf-8')
    return path


def _corpus_with(tmp_path, *commands, prefix='orchestrator', lane='3353', **top):
    """A one-record synthetic root, plus the module yaml the selector reads."""
    _module_yaml(tmp_path, prefix)

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
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        )

        assert [leg.duration_secs for leg in selection.legs] == [3300.0]

    def test_a_loud_top_level_does_not_mask_a_green_test_leg(self, tmp_path):
        """The top level says rc=-9; the selected leg says rc=0. Both are true."""
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, _leg(rc=0))

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        )

        assert [leg.rc for leg in selection.legs] == [0]


class TestTheFullSuiteShapeFilter:
    """Selection is by label AND by command SHAPE, each rejection counted."""

    def _select(self, tmp_path, *commands):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, *commands)
        return select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        )

    def test_the_verbatim_declared_command_is_selected(self, tmp_path):
        selection = self._select(tmp_path, _leg())

        assert len(selection.legs) == 1
        assert selection.rejected_entries == {}
        assert selection.rejected_records == {}

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
        assert selection.rejected_entries == {'command_mismatch': 1}

    def test_a_segmented_entry_is_rejected_with_its_own_reason(self, tmp_path):
        """A segmented leg's duration covers a DIFFERENT topology of the same
        chain, so it is excluded — and counted separately from a plain command
        mismatch, because it is a different fact about the corpus."""
        selection = self._select(
            tmp_path, _leg(segments=[{'index': 1, 'label': 'a'}]),
        )

        assert selection.legs == ()
        assert selection.rejected_entries == {'segmented': 1}

    def test_a_non_test_leg_is_rejected(self, tmp_path):
        selection = self._select(
            tmp_path,
            _leg(label='lint', cmd='uv run --directory orchestrator ruff check src/'),
            _leg(label='type', cmd='uv run --directory orchestrator pyright src/'),
        )

        assert selection.legs == ()
        assert selection.rejected_entries == {'label_mismatch': 2}

    def test_a_null_cmd_is_rejected_rather_than_compared(self, tmp_path):
        """A skipped leg ran nothing, so there is no duration to census."""
        selection = self._select(tmp_path, _leg(cmd=None, duration_secs=0.0))

        assert selection.legs == ()
        assert selection.rejected_entries == {'no_cmd': 1}

    def test_every_entry_is_either_selected_or_counted(self, tmp_path):
        """The reconciliation property again, at the selector — in BOTH units.

        The second record is what makes this falsifiable. Built from a single
        record, the property cannot distinguish a counter that counts ENTRIES
        from one that counts RECORDS — every record-level reason stays at zero,
        so both unit systems produce the same total. A record skipped WHOLE and
        carrying more than one entry is the only shape that separates them:
        three entries leave the corpus and a per-record counter books one.

        So the two counters are asserted against each other, not just summed.
        Folding `prefix_mismatch` back into the entry counter — the mixed-unit
        shape this replaced — makes the entry identity read 1 + 3 == 3 and
        fails here.
        """
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(
            tmp_path,
            _leg(),
            _leg(label='lint', cmd='uv run ruff check src/'),
            _leg(cmd='uv run --directory orchestrator pytest tests/test_foo.py'),
        )
        other = _worktree_record(tmp_path, '4242', 'attempt-1.shared.summary.json')
        other.write_text(
            json.dumps(_summary(_leg(), _leg(), _leg())), encoding='utf-8',
        )

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        )

        # Entries: only the matching record's three were ever READ.
        assert len(selection.legs) + sum(selection.rejected_entries.values()) == 3
        # Records: the other module's is accounted for exactly once, and its
        # three entries are nowhere in the entry counter.
        assert selection.rejected_records == {'prefix_mismatch': 1}

    def test_another_modules_record_is_rejected_on_prefix(self, tmp_path):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, _leg())
        other = _worktree_record(tmp_path, '4242', 'attempt-1.shared.summary.json')
        other.write_text(json.dumps(_summary(_leg())), encoding='utf-8')

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        )

        assert len(selection.legs) == 1
        assert selection.rejected_records == {'prefix_mismatch': 1}
        assert selection.rejected_entries == {}

    def test_a_role_filter_excludes_other_lanes(self, tmp_path):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, _leg())
        merge = _worktree_record(
            tmp_path, '_merge-abc123', 'attempt-1.orchestrator.summary.json',
        )
        merge.write_text(json.dumps(_summary(_leg())), encoding='utf-8')

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator', role='task',
        )

        assert len(selection.legs) == 1
        assert selection.rejected_records == {'role_mismatch': 1}
        assert selection.rejected_entries == {}


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

    def test_a_changed_yaml_changes_what_is_selected(self, tmp_path, capsys):
        """The property a pasted literal would break — asserted, not asserted about.

        Driven through the CLI rather than through ``select_full_suite_legs``,
        because the selector no longer reads the yaml: ``build_report`` resolves
        the command once and passes it down. The property under test is the
        whole chain's (yaml -> resolve -> select), so exercising the chain is
        what asserts it; calling the selector with a hand-passed ``expected``
        would assert only that the selector compares strings.
        """
        _corpus_with(tmp_path, _leg(cmd='pytest tests/ --brand-new-flag'))
        (tmp_path / 'orchestrator' / 'orchestrator.yaml').write_text(
            'test_command: "pytest tests/ --brand-new-flag"\n', encoding='utf-8',
        )

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )
        report = json.loads(out)

        assert report['expected_command'] == 'pytest tests/ --brand-new-flag'
        assert report['overall']['durations']['n'] == 1, (
            'the census must track the yaml, not a literal pasted into the script'
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


class TestSeriesAggregation:
    """p50/p90/max over a duration series, with every number computed by hand.

    Linear interpolation between the two nearest order statistics — the
    `numpy.percentile` default, and the same one
    `merge_lane_throughput._percentile` uses. Shape copied from that sibling,
    not imported: `scripts/` modules do not import one another here.
    """

    def test_an_empty_series_is_null_never_zero(self):
        """The single most consequential misreading a budget report can make.

        `_percentile`'s own docstring states it for the sibling: a `0.0` p50
        would render "no run in this window" as an instantaneous suite. In a
        BUDGET report that is worse than wrong — a p50 of zero invites a reader
        to conclude the suite got fast, on evidence that says nothing at all.
        """
        from verify_budget_census import _series  # noqa: PLC0415

        assert _series([]) == {'n': 0, 'p50': None, 'p90': None, 'max': None}

    def test_a_single_value_series(self):
        from verify_budget_census import _series  # noqa: PLC0415

        assert _series([3300.0]) == {
            'n': 1, 'p50': 3300.0, 'p90': 3300.0, 'max': 3300.0,
        }

    def test_percentiles_are_interpolated_by_hand_checked_arithmetic(self):
        """values = [100, 200, 300, 400, 500], n=5.

        p50: k = (5-1) * 0.50 = 2.0      -> exactly s[2]           = 300
        p90: k = (5-1) * 0.90 = 3.6      -> s[3] + 0.6*(s[4]-s[3]) = 460
        """
        from verify_budget_census import _series  # noqa: PLC0415

        assert _series([100.0, 200.0, 300.0, 400.0, 500.0]) == {
            'n': 5, 'p50': 300.0, 'p90': 460.0, 'max': 500.0,
        }

    def test_the_input_order_does_not_matter(self):
        from verify_budget_census import _series  # noqa: PLC0415

        assert _series([500.0, 100.0, 400.0, 200.0, 300.0])['p50'] == 300.0

    def test_an_even_length_series_interpolates_the_median(self):
        """values = [10, 20, 30, 40], n=4. p50: k = 3 * 0.5 = 1.5 -> 20 + 0.5*10 = 25."""
        from verify_budget_census import _series  # noqa: PLC0415

        assert _series([10.0, 20.0, 30.0, 40.0])['p50'] == 25.0


class TestTimedOutAndFailedAreCountedApart:
    """A timed-out run's duration is the BUDGET, not the suite.

    Folding it into the percentiles measures the ceiling and calls it the
    workload — and then a budget derived from that distribution is derived from
    itself, which is the one circularity a census must not have. Counted, and
    excluded.
    """

    def _legs(self, tmp_path, *entries):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, *entries)
        return select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        ).legs

    def test_a_timed_out_leg_is_counted_and_excluded(self, tmp_path):
        from verify_budget_census import summarise_legs  # noqa: PLC0415

        legs = self._legs(
            tmp_path,
            _leg(duration_secs=3300.0),
            _leg(duration_secs=7200.0, timed_out=True, rc=-9),
        )

        summary = summarise_legs(legs)

        assert summary['timed_out'] == 1
        assert summary['durations'] == {
            'n': 1, 'p50': 3300.0, 'p90': 3300.0, 'max': 3300.0,
        }

    def test_a_failing_leg_is_counted_and_excluded(self, tmp_path):
        """A red run stopped at the first failure, so its duration is not the
        suite's either — a different fact from a timeout, counted separately."""
        from verify_budget_census import summarise_legs  # noqa: PLC0415

        legs = self._legs(
            tmp_path, _leg(duration_secs=3300.0), _leg(duration_secs=120.0, rc=1),
        )

        summary = summarise_legs(legs)

        assert summary['failed'] == 1
        assert summary['timed_out'] == 0
        assert summary['durations']['max'] == 3300.0

    def test_a_timed_out_leg_is_not_double_counted_as_failed(self, tmp_path):
        """A timeout carries a non-zero rc too, so the two buckets must not
        both claim it — n would stop reconciling."""
        from verify_budget_census import summarise_legs  # noqa: PLC0415

        legs = self._legs(tmp_path, _leg(duration_secs=7200.0, timed_out=True, rc=-9))

        summary = summarise_legs(legs)

        assert (summary['timed_out'], summary['failed']) == (1, 0)

    def test_every_leg_is_accounted_for(self, tmp_path):
        from verify_budget_census import summarise_legs  # noqa: PLC0415

        legs = self._legs(
            tmp_path,
            _leg(duration_secs=3300.0),
            _leg(duration_secs=3400.0),
            _leg(duration_secs=7200.0, timed_out=True, rc=-9),
            _leg(duration_secs=120.0, rc=1),
        )

        summary = summarise_legs(legs)

        assert (
            summary['durations']['n'] + summary['timed_out'] + summary['failed']
        ) == len(legs) == 4


class TestDayBucketing:
    """Buckets are the entry's `started_at`, normalised to UTC, then `.date()`."""

    def test_a_non_utc_offset_is_normalised_before_bucketing(self):
        """23:30-04:00 is 03:30Z the NEXT day — the bucket must say so.

        Bucketing on the raw string's leading 10 characters would file it under
        the local date, silently sliding runs across a day boundary and
        smearing the trailing-window edge this census's floor depends on.
        """
        from verify_budget_census import day_bucket  # noqa: PLC0415

        assert str(day_bucket('2026-09-13T23:30:00-04:00')) == '2026-09-14'

    def test_a_utc_timestamp_buckets_on_its_own_date(self):
        from verify_budget_census import day_bucket  # noqa: PLC0415

        assert str(day_bucket('2026-09-14T03:30:00+00:00')) == '2026-09-14'

    def test_a_naive_timestamp_is_read_as_utc(self):
        from verify_budget_census import day_bucket  # noqa: PLC0415

        assert str(day_bucket('2026-09-14T03:30:00')) == '2026-09-14'

    @pytest.mark.parametrize('stamp', ['', 'not a date', '2026-13-45T99:99:99+00:00'])
    def test_an_unparseable_timestamp_is_none_not_today(self, stamp):
        """Defaulting to the current date would invent a run inside the window."""
        from verify_budget_census import day_bucket  # noqa: PLC0415

        assert day_bucket(stamp) is None


class TestParseWindow:
    """Both `--window` forms, resolved against an INJECTED clock."""

    def test_the_relative_form(self):
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import parse_window  # noqa: PLC0415

        now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
        lo, hi = parse_window('14d', now)

        assert hi == now
        assert lo == datetime(2026, 8, 31, 12, 0, tzinfo=UTC)

    def test_the_dated_form_is_exactly_those_instants(self):
        """The mechanism for a report whose header carries a fixed date."""
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import parse_window  # noqa: PLC0415

        lo, hi = parse_window(
            '2026-09-12T08:00:00+00:00..2026-09-14T00:00:00+00:00',
            datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
        )

        assert lo == datetime(2026, 9, 12, 8, 0, tzinfo=UTC)
        assert hi == datetime(2026, 9, 14, 0, 0, tzinfo=UTC)

    def test_a_naive_endpoint_is_read_as_utc(self):
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import parse_window  # noqa: PLC0415

        lo, _hi = parse_window(
            '2026-09-12T08:00:00..2026-09-14T00:00:00',
            datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
        )

        assert lo == datetime(2026, 9, 12, 8, 0, tzinfo=UTC)

    @pytest.mark.parametrize(
        'spec',
        ['', '0d', '-3d', 'fortnight', '..', '2026-09-14T00:00:00+00:00..',
         'nonsense..alsononsense'],
    )
    def test_a_malformed_spec_is_rejected_echoing_the_spec(self, spec):
        import argparse  # noqa: PLC0415
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import parse_window  # noqa: PLC0415

        with pytest.raises(argparse.ArgumentTypeError) as caught:
            parse_window(spec, datetime(2026, 9, 14, 12, 0, tzinfo=UTC))

        assert repr(spec) in str(caught.value)

    def test_a_reversed_range_is_rejected_not_swapped(self):
        """Far more often a pasted-backwards pair than a request for one instant."""
        import argparse  # noqa: PLC0415
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import parse_window  # noqa: PLC0415

        with pytest.raises(argparse.ArgumentTypeError):
            parse_window(
                '2026-09-14T00:00:00+00:00..2026-09-12T08:00:00+00:00',
                datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
            )


class TestLegsAreFilteredToTheResolvedWindow:
    def test_only_legs_inside_the_window_are_summarised(self, tmp_path):
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import (  # noqa: PLC0415
            load_records,
            select_full_suite_legs,
            within_window,
        )

        _corpus_with(
            tmp_path,
            _leg(started_at='2026-09-13T04:00:00+00:00', duration_secs=3300.0),
            _leg(started_at='2026-08-25T04:00:00+00:00', duration_secs=4991.0),
        )
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        ).legs

        window = (
            datetime(2026, 9, 12, 8, 0, tzinfo=UTC),
            datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
        )
        kept = within_window(legs, window)

        assert [leg.duration_secs for leg in kept] == [3300.0]

    def test_a_leg_with_an_unparseable_timestamp_is_excluded(self, tmp_path):
        """It cannot be placed in the window, so it cannot be counted in it."""
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import (  # noqa: PLC0415
            load_records,
            select_full_suite_legs,
            within_window,
        )

        _corpus_with(tmp_path, _leg(started_at='', duration_secs=3300.0))
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        ).legs

        window = (
            datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC),
        )

        assert within_window(legs, window) == ()


# "Whatever the start instant reads" — a sentinel, because `None` is itself a
# MEANINGFUL end reading here (a degraded component) and cannot double as the
# default.
_MIRROR_START: Any = object()


def _load_record(
    cpu: float | None = 1.0,
    cpu60: float | None = None,
    runqueue: float = 0.5,
    end_cpu: float | None = _MIRROR_START,
    end_cpu60: float | None = _MIRROR_START,
):
    """One `load` record as `verify._load_sample`/`_xdist_workers` emit it.

    The end instant MIRRORS the start unless a case says otherwise, which is
    the shape a stable host leaves behind.
    """
    end = cpu if end_cpu is _MIRROR_START else end_cpu
    return {
        'start': {
            'cpu_some10': cpu,
            'cpu_some60': cpu if cpu60 is None else cpu60,
            'runqueue_ratio': runqueue,
        },
        'end': {
            'cpu_some10': end,
            'cpu_some60': end if end_cpu60 is _MIRROR_START else end_cpu60,
            'runqueue_ratio': runqueue,
        },
        'xdist': {'n_flag': None, 'auto_num_workers': None},
    }


class TestPsiBanding:
    """Durations bucketed by the host pressure their command actually ran under.

    This is what deliverable 1 exists for: "measure on a quiet host" stops
    being a precondition, because every record says how quiet its host was.
    """

    def _banded(self, tmp_path, *entries):
        from verify_budget_census import (  # noqa: PLC0415
            by_load_band,
            load_records,
            select_full_suite_legs,
        )

        _corpus_with(tmp_path, *entries)
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        ).legs
        return by_load_band(legs)

    def test_a_quiet_and_a_busy_run_land_in_different_bands(self, tmp_path):
        bands = self._banded(
            tmp_path,
            _leg(duration_secs=3000.0, load=_load_record(cpu=1.0)),
            _leg(duration_secs=4600.0, load=_load_record(cpu=75.0)),
        )

        assert bands['unstamped']['durations']['n'] == 0
        banded = {
            name: row['durations']['max']
            for name, row in bands.items()
            if name != 'unstamped' and row['durations']['n']
        }
        assert sorted(banded.values()) == [3000.0, 4600.0]
        assert len(banded) == 2, 'a quiet and a heavily-loaded run must not share a band'

    def test_the_bands_are_named_constants_not_inline_numbers(self):
        """Band edges carry a rationale, so a later reader can argue with them."""
        from verify_budget_census import PSI_BANDS  # noqa: PLC0415

        assert PSI_BANDS
        for band in PSI_BANDS:
            assert band.name
            assert band.why, f'band {band.name} has no stated rationale'

    def test_the_bands_tile_the_range_without_gaps_or_overlap(self):
        """Every possible pressure reading lands in exactly one band."""
        from verify_budget_census import PSI_BANDS, band_for  # noqa: PLC0415

        for value in (0.0, 0.01, 4.9, 5.0, 24.9, 25.0, 49.9, 50.0, 99.9, 100.0):
            matches = [b.name for b in PSI_BANDS if b.lo <= value < b.hi]
            assert len(matches) == 1, f'{value} matched {matches}'
            assert band_for(value) == matches[0]


class TestUnstampedIsItsOwnRow:
    """The honesty property that matters most for the HISTORICAL corpus.

    Measured while building this: ZERO records in the live corpus carry a load
    stamp, because deliverable 1 has not merged. So `unstamped` is not an edge
    case — today it is the entire corpus, and every figure the budget is
    currently derived from sits in it.

    Putting those in a zero-pressure band would be the worst available lie: it
    would file the busiest historical runs in the IDLE band and then invite the
    conclusion that the suite is slow even on a quiet host.
    """

    def _banded(self, tmp_path, *entries):
        from verify_budget_census import (  # noqa: PLC0415
            by_load_band,
            load_records,
            select_full_suite_legs,
        )

        _corpus_with(tmp_path, *entries)
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        ).legs
        return by_load_band(legs)

    def test_a_record_predating_the_stamp_is_unstamped(self, tmp_path):
        bands = self._banded(tmp_path, _leg(duration_secs=4991.0, load=None))

        assert bands['unstamped']['durations'] == {
            'n': 1, 'p50': 4991.0, 'p90': 4991.0, 'max': 4991.0,
        }

    def test_a_degraded_psi_read_is_unstamped_not_zero_pressure(self, tmp_path):
        """A null cpu_some10 means "not knowable", never "the host was idle"."""
        bands = self._banded(
            tmp_path, _leg(duration_secs=4991.0, load=_load_record(cpu=None)),
        )

        assert bands['unstamped']['durations']['n'] == 1
        assert bands[_idle_band_name()]['durations']['n'] == 0

    def test_a_genuinely_idle_host_is_banded_not_unstamped(self, tmp_path):
        """The complement — 0.0 is a real reading and must not read as missing."""
        bands = self._banded(
            tmp_path, _leg(duration_secs=3000.0, load=_load_record(cpu=0.0)),
        )

        assert bands['unstamped']['durations']['n'] == 0
        assert bands[_idle_band_name()]['durations']['n'] == 1

    def test_the_bands_reconcile_against_the_selected_n(self, tmp_path):
        """Never silently dropped: banded + unstamped accounts for every leg."""
        bands = self._banded(
            tmp_path,
            _leg(duration_secs=3000.0, load=_load_record(cpu=0.0)),
            _leg(duration_secs=3100.0, load=_load_record(cpu=40.0)),
            _leg(duration_secs=4991.0, load=None),
            _leg(duration_secs=7200.0, load=None, timed_out=True, rc=-9),
        )

        accounted = sum(
            row['durations']['n'] + row['timed_out'] + row['failed']
            for row in bands.values()
        )
        assert accounted == 4


def _idle_band_name():
    from verify_budget_census import PSI_BANDS  # noqa: PLC0415

    return PSI_BANDS[0].name


def _heavy_band_name():
    from verify_budget_census import band_for  # noqa: PLC0415

    return band_for(75.0)


class TestTheBandStatisticSpansTheRun:
    """A leg is banded by the HIGHER of its two readings, not by the start alone.

    The start sample is taken immediately BEFORE the command. For the population
    being censused — full-suite runs with a p50 near 3000s — that 10-second
    window describes the host during the QUEUE WAIT, not during the run. The end
    sample is stamped precisely to cover the other end of the interval, so
    banding that read nothing from it rested the report's central claim ("a
    duration figure, banded by the load it ran under") on its weakest available
    reading.
    """

    def _banded(self, tmp_path, *entries):
        from verify_budget_census import (  # noqa: PLC0415
            by_load_band,
            load_records,
            select_full_suite_legs,
        )

        _corpus_with(tmp_path, *entries)
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        ).legs
        return by_load_band(legs)

    def test_a_run_that_started_quiet_and_ended_busy_bands_busy(self, tmp_path):
        """The case start-banding got wrong: an hour of contention filed as idle."""
        bands = self._banded(
            tmp_path,
            _leg(duration_secs=4600.0, load=_load_record(cpu=1.0, end_cpu=75.0)),
        )

        assert bands[_idle_band_name()]['durations']['n'] == 0
        assert bands[_heavy_band_name()]['durations']['max'] == 4600.0

    def test_the_peak_counts_whichever_end_it_falls_on(self, tmp_path):
        """A run that started busy and ended quiet is not a quiet run either."""
        bands = self._banded(
            tmp_path,
            _leg(duration_secs=4600.0, load=_load_record(cpu=75.0, end_cpu=1.0)),
        )

        assert bands[_idle_band_name()]['durations']['n'] == 0
        assert bands[_heavy_band_name()]['durations']['max'] == 4600.0

    @pytest.mark.parametrize(
        ('start', 'end'), [(75.0, None), (None, 75.0)], ids=['end-null', 'start-null'],
    )
    def test_one_knowable_reading_still_bands_the_leg(self, tmp_path, start, end):
        """A half-degraded record carries a real reading; discarding it would be
        the silent-fail-soft shape `unstamped` exists to avoid in reverse."""
        bands = self._banded(
            tmp_path,
            _leg(duration_secs=4600.0, load=_load_record(cpu=start, end_cpu=end)),
        )

        assert bands['unstamped']['durations']['n'] == 0
        assert bands[_heavy_band_name()]['durations']['max'] == 4600.0

    def test_the_statistic_is_the_max_of_the_two_instants(self):
        """Stated directly, since the whole report rests on this one choice.

        The MAX, not the mean: a band says "measured under AT LEAST this much
        contention", and a mean would let a run that started quiet and ended
        saturated file as merely moderate.
        """
        from verify_budget_census import band_pressure  # noqa: PLC0415

        assert band_pressure(_load_record(cpu=1.0, end_cpu=75.0)) == 75.0
        assert band_pressure(_load_record(cpu=75.0, end_cpu=1.0)) == 75.0
        assert band_pressure(_load_record(cpu=None, end_cpu=None)) is None
        assert band_pressure(None) is None

    def test_the_smoothed_window_is_not_folded_into_the_band(self):
        """`cpu_some60` is a DIFFERENT averaging window, and the band edges were
        argued for avg10 — mixing them would make the edges mean something
        nobody stated. It is reported by `load_movement` instead."""
        from verify_budget_census import band_pressure  # noqa: PLC0415

        load = _load_record(cpu=1.0, cpu60=90.0, end_cpu=2.0, end_cpu60=95.0)

        assert band_pressure(load) == 2.0


class TestHostMovementIsReported:
    """The end reading reaches a reader, not just the banding statistic.

    A band alone cannot show that a run began idle and ended saturated, and that
    movement is what tells an operator whether a duration is a statement about
    the suite or about the fleet's occupancy while it ran.
    """

    def _movement(self, tmp_path, *entries):
        from verify_budget_census import (  # noqa: PLC0415
            load_movement,
            load_records,
            select_full_suite_legs,
        )

        _corpus_with(tmp_path, *entries)
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        ).legs
        return load_movement(legs)

    def test_a_rise_across_bands_is_counted_as_a_rise(self, tmp_path):
        movement = self._movement(
            tmp_path, _leg(load=_load_record(cpu=1.0, end_cpu=75.0)),
        )

        assert movement['rose'] == 1
        assert movement['held'] == movement['fell'] == 0

    def test_a_fall_across_bands_is_counted_as_a_fall(self, tmp_path):
        movement = self._movement(
            tmp_path, _leg(load=_load_record(cpu=75.0, end_cpu=1.0)),
        )

        assert movement['fell'] == 1
        assert movement['held'] == movement['rose'] == 0

    def test_a_move_inside_one_band_is_held(self, tmp_path):
        """Counted in BANDS, because a 3-point rise inside one band is not a fact
        anyone would act on."""
        movement = self._movement(
            tmp_path, _leg(load=_load_record(cpu=60.0, end_cpu=63.0)),
        )

        assert movement['held'] == 1
        assert movement['rose'] == movement['fell'] == 0

    def test_a_half_degraded_record_is_not_knowable_yet_still_banded(self, tmp_path):
        """The two counts differ DELIBERATELY: a movement needs both instants, a
        band needs only one. Asserted together so neither drifts into the other."""
        from verify_budget_census import (  # noqa: PLC0415
            by_load_band,
            load_movement,
            load_records,
            select_full_suite_legs,
        )

        _corpus_with(
            tmp_path,
            _leg(duration_secs=4600.0, load=_load_record(cpu=75.0, end_cpu=None)),
        )
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        ).legs

        assert load_movement(legs)['not_knowable'] == 1
        assert by_load_band(legs)['unstamped']['durations']['n'] == 0

    def test_the_smoothed_end_of_run_reading_is_surfaced(self, tmp_path):
        """`cpu_some60` at the END is the one reading describing an INTERVAL of
        the run itself rather than an instant before or after it."""
        movement = self._movement(
            tmp_path,
            _leg(load=_load_record(cpu=1.0, end_cpu=70.0, end_cpu60=40.0)),
            _leg(load=_load_record(cpu=1.0, end_cpu=70.0, end_cpu60=60.0)),
        )

        assert movement['end_cpu_some60']['n'] == 2
        assert movement['end_cpu_some60']['max'] == 60.0

    def test_a_degraded_smoothed_reading_is_absent_not_zero(self, tmp_path):
        movement = self._movement(
            tmp_path, _leg(load=_load_record(cpu=1.0, end_cpu=70.0, end_cpu60=None)),
        )

        assert movement['end_cpu_some60'] == {
            'n': 0, 'p50': None, 'p90': None, 'max': None,
        }

    def test_it_rides_in_the_report_and_prints(self, tmp_path, capsys):
        _corpus_with(tmp_path, _leg(load=_load_record(cpu=1.0, end_cpu=75.0)))

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )
        assert json.loads(out)['load_movement']['rose'] == 1

        _rc, text, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )
        assert 'HOST MOVEMENT DURING THE RUN' in text
        assert 'rose 1' in text


class TestTheProducerAndTheConsumerAgreeOnTheLoadShape:
    """The only seam in this deliverable that crosses a module boundary.

    Everything else on both sides hand-builds the `load` dict — `_load_record`
    here, `TestCheckRunLoad` and `TestLoadReachesTheSummaryPayload._load` in
    `orchestrator/tests/test_verify_load_stamp.py` — so neither suite can see a
    RENAME. Renesting a key in `verify._load_sample`, or in the
    `{'start','end','xdist'}` composition at its single call site, would leave
    all of them green while the census silently booked every run under
    `unstamped` — the bucket whose whole purpose is to never be confused with a
    reading. That is the same whitelist-drift failure `segments` actually
    suffered, at the one place no single-module test can reach.

    So this drives the REAL `run_verification` into a corpus-shaped tmp root and
    reads the summary.json it WRITES with the REAL census.
    """

    _START_CPU = 1.0
    _END_CPU = 75.0
    _END_CPU60 = 40.0

    def _legs_from_a_real_verify(self, tmp_path, monkeypatch):
        import asyncio  # noqa: PLC0415
        from unittest.mock import patch  # noqa: PLC0415

        from orchestrator.config import ModuleConfig, OrchestratorConfig  # noqa: PLC0415
        from shared.psi import PsiSample  # noqa: PLC0415
        from verify_budget_census import (  # noqa: PLC0415
            load_records,
            select_full_suite_legs,
        )

        from orchestrator import verify  # noqa: PLC0415

        # `OrchestratorConfig(project_root=...)` reads an ambient
        # ORCH_CONFIG_PATH, so without this the "default" config is whichever
        # project the runner happens to point at — the same leak
        # `_target_subprocess_env` scrubs for child processes (task 2957).
        # Pointed at a guaranteed-ABSENT file rather than deleted, which is the
        # spelling orchestrator/tests/conftest.py's `code_default_config` uses
        # (that fixture is not visible from this directory): an absent path makes
        # the project layer skip itself and leaves `defaults.yaml` loading, while
        # an unset var falls back to the RELATIVE `config.yaml` and so depends on
        # the runner's cwd.
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'no-such-config.yaml'))

        def sample(cpu, cpu60):
            return PsiSample(
                cpu_some10=cpu, cpu_some60=cpu60, mem_some10=0.0, mem_full10=0.0,
                io_some10=0.0, read_ok=True, runqueue_ratio=0.25,
                runqueue_read_ok=True,
            )

        # Exactly two readings, start then end. A third call exhausts the
        # iterator, which `_load_sample`'s never-raise wrapper turns into an
        # all-null record — and the assertions below then fail loudly rather
        # than passing on a fabricated reading.
        readings = iter((
            sample(self._START_CPU, 0.5),
            sample(self._END_CPU, self._END_CPU60),
        ))
        real_load_sample = verify._load_sample

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **_kw):
            return 0, 'ok', False

        # The worktree IS the corpus path: `<root>/.worktrees/<lane>/` is where
        # `load_records` looks, and `run_verification` persists into
        # `<worktree>/.task/verify/` on its own.
        worktree = tmp_path / '.worktrees' / '3353'
        (worktree / '.task').mkdir(parents=True, exist_ok=True)
        _module_yaml(tmp_path)

        with patch.object(verify, '_run_cmd', side_effect=fake_run_cmd), \
             patch.object(
                 verify, '_load_sample',
                 side_effect=lambda: real_load_sample(read=lambda: next(readings)),
             ):
            asyncio.run(
                verify.run_verification(
                    worktree,
                    OrchestratorConfig(
                        project_root=tmp_path, verify_admission_enabled=False,
                    ),
                    ModuleConfig(prefix='orchestrator', test_command=_FULL_SUITE),
                    attempt_id=1,
                    task_id='3353',
                    max_retries=0,
                ),
            )

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        )
        assert len(selection.legs) == 1, (
            f'the real producer wrote nothing the census selects: '
            f'{selection.rejected_entries} / {selection.rejected_records}'
        )
        return selection.legs

    def test_the_census_reads_a_pressure_out_of_a_real_record(
        self, tmp_path, monkeypatch,
    ):
        """`band_pressure` indexes the producer's own keys, not a literal."""
        from verify_budget_census import band_pressure  # noqa: PLC0415

        legs = self._legs_from_a_real_verify(tmp_path, monkeypatch)

        assert band_pressure(legs[0].load) == self._END_CPU

    def test_a_real_record_bands_instead_of_landing_in_unstamped(
        self, tmp_path, monkeypatch,
    ):
        """The failure this guard exists for: a renamed key books every run as
        "load not knowable" while every single-module test stays green."""
        from verify_budget_census import by_load_band  # noqa: PLC0415

        bands = by_load_band(self._legs_from_a_real_verify(tmp_path, monkeypatch))

        assert bands['unstamped']['durations']['n'] == 0
        assert bands[_heavy_band_name()]['durations']['n'] == 1

    def test_the_movement_section_reads_both_instants_of_a_real_record(
        self, tmp_path, monkeypatch,
    ):
        from verify_budget_census import load_movement  # noqa: PLC0415

        movement = load_movement(self._legs_from_a_real_verify(tmp_path, monkeypatch))

        assert movement['rose'] == 1
        assert movement['not_knowable'] == 0
        assert movement['end_cpu_some60']['max'] == self._END_CPU60


class TestColdSeparabilityIsReportedNotInferred:
    """D17's ruled fallback, asserted as a property of the report.

    A summary.json carries no is-cold flag. The only available inference —
    "attempt-1 in a worktree with no prior verify dir is cold" — would mix warm
    reruns into a cold distribution and then get frozen into a budget as if
    measured. D17 anticipates this and rules the fallback: label the cold value
    INTERIM with its basis. So the census says it CANNOT separate them, counts
    the records it could only have guessed at, and labels no series "cold".
    """

    def _finding(self, tmp_path, *entries):
        from verify_budget_census import (  # noqa: PLC0415
            cold_separability,
            load_records,
            select_full_suite_legs,
        )

        _corpus_with(tmp_path, *entries)
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        ).legs
        return cold_separability(legs)

    def test_the_finding_says_not_separable(self, tmp_path):
        finding = self._finding(tmp_path, _leg())

        assert finding['cold_separable'] is False
        assert finding['basis'], 'the finding must state WHY it cannot separate'

    def test_the_guessable_records_are_counted(self, tmp_path):
        """attempt-1 records are the ones a cold inference would have claimed."""
        from verify_budget_census import (  # noqa: PLC0415
            cold_separability,
            load_records,
            select_full_suite_legs,
        )

        _corpus_with(tmp_path, _leg(), lane='100')
        second = _worktree_record(tmp_path, '200', 'attempt-3.orchestrator.summary.json')
        second.write_text(json.dumps(_summary(_leg())), encoding='utf-8')

        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        ).legs
        finding = cold_separability(legs)

        assert finding['first_attempt_records'] == 1
        assert finding['later_attempt_records'] == 1

    def test_no_series_is_labelled_cold(self, tmp_path):
        """The assertion that stops a guess becoming a measurement."""
        finding = self._finding(tmp_path, _leg())

        assert 'cold' not in {k.lower() for k in finding} - {'cold_separable'}
        assert not any(
            isinstance(v, dict) and 'p50' in v for v in finding.values()
        ), 'a duration series here would be a cold distribution built on a guess'


class TestTheMergeGateBudgetIsReported:
    """Reported, never changed. The merge gate is ALWAYS cold.

    It verifies a freshly-created worktree every time, so its budget is a
    different question from the task lane's — and this census does not answer
    it. Saying so in the report is what stops a reader applying a warm-derived
    figure to a strictly costlier path.
    """

    def test_the_project_yaml_value_is_read(self, tmp_path):
        from verify_budget_census import merge_gate_budget  # noqa: PLC0415

        (tmp_path / 'dark-factory-orchestrator.yaml').write_text(
            'merge_verify_cold_command_timeout_secs: 7200\n', encoding='utf-8',
        )

        readout = merge_gate_budget(tmp_path)

        assert readout['merge_verify_cold_command_timeout_secs'] == 7200
        assert readout['source'].endswith('dark-factory-orchestrator.yaml')

    def test_the_defaults_yaml_is_the_fallback(self, tmp_path):
        """Unset in the project yaml is the LIVE state — the default governs."""
        from verify_budget_census import merge_gate_budget  # noqa: PLC0415

        (tmp_path / 'dark-factory-orchestrator.yaml').write_text(
            'max_concurrent_tasks: 4\n', encoding='utf-8',
        )
        defaults = tmp_path / 'orchestrator' / 'src' / 'orchestrator' / 'defaults.yaml'
        defaults.parent.mkdir(parents=True)
        defaults.write_text(
            'merge_verify_cold_command_timeout_secs: 7200\n', encoding='utf-8',
        )

        readout = merge_gate_budget(tmp_path)

        assert readout['merge_verify_cold_command_timeout_secs'] == 7200
        assert readout['source'].endswith('defaults.yaml')

    def test_an_unresolvable_budget_is_null_not_guessed(self, tmp_path):
        from verify_budget_census import merge_gate_budget  # noqa: PLC0415

        readout = merge_gate_budget(tmp_path)

        assert readout['merge_verify_cold_command_timeout_secs'] is None
        assert readout['source'] is None

    def test_the_readout_states_the_merge_gate_is_always_cold(self, tmp_path):
        from verify_budget_census import merge_gate_budget  # noqa: PLC0415

        readout = merge_gate_budget(tmp_path)

        assert readout['always_cold'] is True
        assert 'census changes nothing' in readout['note'].lower()


# A frozen clock, injected into every main() call below, so the trailing-window
# boundary is deterministic. The sibling suites do the same; a window resolved
# against the real clock makes a test's own fixtures age out of it.
_NOW = '2026-09-14T12:00:00+00:00'


def _main(capsys, *argv):
    """Drive `main()` and return `(rc, stdout, stderr)` — the sibling's idiom."""
    from datetime import datetime  # noqa: PLC0415

    import verify_budget_census as mod  # noqa: PLC0415

    code = mod.main(list(argv), now=datetime.fromisoformat(_NOW))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestTheCliContract:
    """`main(argv, now=None)` over the same report dict the text renderer reads.

    `--root` is the injection point that makes the whole script testable
    against a synthetic corpus. Without it every test here would have to run
    against the live tree, which is the thing this suite's norm forbids.
    """

    def test_a_synthetic_corpus_reports_its_own_numbers(self, tmp_path, capsys):
        _corpus_with(tmp_path, _leg(duration_secs=3300.0))

        rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert rc == 0
        assert '3300' in out

    def test_json_switches_the_whole_output_to_one_document(self, tmp_path, capsys):
        _corpus_with(tmp_path, _leg(duration_secs=3300.0))

        rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )

        assert rc == 0
        report = json.loads(out)
        assert report['window']
        assert report['module'] == 'orchestrator'

    def test_the_text_renderer_reads_the_same_dict_json_emits(self, tmp_path, capsys):
        """One structure, two renderings — the sibling's build_report/render split.

        Asserted by driving both and checking the text carries the figure the
        JSON reports, so the two cannot describe different runs.
        """
        _corpus_with(tmp_path, _leg(duration_secs=3300.0))

        _rc, text, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )
        _rc, raw, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )

        reported_max = json.loads(raw)['overall']['durations']['max']
        assert str(int(reported_max)) in text

    def test_the_window_is_resolved_against_the_injected_clock(self, tmp_path, capsys):
        _corpus_with(tmp_path, _leg(duration_secs=3300.0))

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
            '--window', '14d', '--json',
        )

        lo, hi = json.loads(out)['window']
        assert hi.startswith('2026-09-14T12:00:00')
        assert lo.startswith('2026-08-31T12:00:00')

    def test_the_dated_window_is_honoured(self, tmp_path, capsys):
        _corpus_with(
            tmp_path,
            _leg(started_at='2026-09-13T04:00:00+00:00', duration_secs=3300.0),
            _leg(started_at='2026-08-25T04:00:00+00:00', duration_secs=4991.0),
        )

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
            '--window', '2026-09-12T08:00:00+00:00..2026-09-14T12:00:00+00:00',
        )

        assert json.loads(out)['overall']['durations']['max'] == 3300.0

    def test_the_role_filter_reaches_the_selector(self, tmp_path, capsys):
        _corpus_with(tmp_path, _leg(duration_secs=3300.0))
        merge = _worktree_record(
            tmp_path, '_merge-abc123', 'attempt-1.orchestrator.summary.json',
        )
        merge.write_text(
            json.dumps(_summary(_leg(duration_secs=9999.0))), encoding='utf-8',
        )

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
            '--role', 'task', '--json',
        )

        assert json.loads(out)['overall']['durations']['max'] == 3300.0

    def test_the_label_filter_reaches_the_selector(self, tmp_path, capsys):
        _corpus_with(tmp_path, _leg(label='lint', cmd=_FULL_SUITE))

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
            '--label', 'lint', '--json',
        )

        assert json.loads(out)['overall']['durations']['n'] == 1

    def test_roots_are_repeatable(self, tmp_path, capsys):
        one, two = tmp_path / 'one', tmp_path / 'two'
        _corpus_with(one, _leg(duration_secs=3300.0), lane='1')
        _corpus_with(two, _leg(duration_secs=3400.0), lane='2')

        _rc, out, _err = _main(
            capsys, '--root', str(one), '--root', str(two),
            '--module', 'orchestrator', '--json',
        )

        assert json.loads(out)['overall']['durations']['n'] == 2

    def test_a_first_root_without_the_module_yaml_still_selects(
        self, tmp_path, capsys,
    ):
        """The command is resolved across ALL roots, and the selection uses it.

        `_corpus_with` writes the module yaml into EVERY root it builds, which
        is why `test_roots_are_repeatable` above cannot reach this: the two
        resolutions agree whenever every root declares. Here only the SECOND
        root declares, which is an ordinary shape — `--root` is repeatable
        precisely so an archive path can be censused alongside a checkout, and
        an archive carries records without carrying a module yaml.

        The regression: `build_report` resolved `expected_command` from the
        first root that DECLARES one while the selector re-resolved it from
        `roots[0]` alone. With `roots[0]` not declaring, the selector compared
        against None, rejected every entry as `no_declared_command`, and the
        report printed the resolved command above `n=0` and the NOTE "no
        full-suite run matched in this window" — telling a reader it had
        compared against a command it had never used.
        """
        bare, declaring = tmp_path / 'bare', tmp_path / 'declaring'
        _corpus_with(declaring, _leg(duration_secs=3300.0), lane='2')
        _corpus_with(bare, _leg(duration_secs=3400.0), lane='1')
        (bare / 'orchestrator' / 'orchestrator.yaml').unlink()

        _rc, out, _err = _main(
            capsys, '--root', str(bare), '--root', str(declaring),
            '--module', 'orchestrator', '--json',
        )
        report = json.loads(out)

        assert report['expected_command'] == _FULL_SUITE
        assert 'no_declared_command' not in report['rejected_entries']
        assert report['overall']['durations']['n'] == 2, (
            'the legs were rejected against a command the report still printed'
        )

    def test_no_root_declaring_reports_none_rather_than_a_silent_zero(
        self, tmp_path, capsys,
    ):
        """`expected=None` is honest, not a bug — and must render as such.

        The sibling above fixes the case where a command WAS resolvable. This
        pins the case where none was: the count is still 0, but the report says
        `<none declared>` rather than naming a command, so "nothing to compare
        against" stays distinguishable from "compared and found nothing". That
        distinction is the whole reason `read_module_test_command` returns None
        instead of guessing a default.
        """
        _corpus_with(tmp_path, _leg(duration_secs=3300.0))
        (tmp_path / 'orchestrator' / 'orchestrator.yaml').unlink()

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )
        report = json.loads(out)

        assert report['expected_command'] is None
        assert report['rejected_entries']['no_declared_command'] == 1
        assert report['overall']['durations']['n'] == 0

        _rc, text, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert '<none declared>' in text


class TestTheExitCodeVocabulary:
    """0 / 1 / 2, as `merge_lane_throughput.main` documents."""

    def test_success_is_zero(self, tmp_path, capsys):
        _corpus_with(tmp_path, _leg())

        rc, _out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert rc == 0

    def test_an_unreadable_named_root_is_one_and_the_others_still_report(
        self, tmp_path, capsys,
    ):
        """Partial failure is reported on stderr, not turned into a crash."""
        good = tmp_path / 'good'
        _corpus_with(good, _leg(duration_secs=3300.0))

        rc, out, err = _main(
            capsys, '--root', str(good), '--root', str(tmp_path / 'absent'),
            '--module', 'orchestrator', '--json',
        )

        assert rc == 1
        assert 'absent' in err
        assert json.loads(out)['overall']['durations']['max'] == 3300.0

    def test_a_malformed_window_is_two_with_nothing_on_stdout(self, tmp_path, capsys):
        rc, out, err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
            '--window', 'fortnight',
        )

        assert rc == 2
        assert out == ''
        assert 'fortnight' in err


class TestAnEmptyCorpusIsReportedNotFabricated:
    """The case that separates an honest census from a reassuring one.

    A root with no corpus must not yield a p50 of 0.0, and must not traceback.
    It must say, in as many words, that it found nothing — which is the only
    output that cannot be mistaken for a fast suite.
    """

    def test_a_root_with_neither_corpus_exits_zero(self, tmp_path, capsys):
        rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert rc == 0
        assert out

    def test_the_empty_report_carries_nulls_not_zeros(self, tmp_path, capsys):
        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )

        durations = json.loads(out)['overall']['durations']
        assert durations == {'n': 0, 'p50': None, 'p90': None, 'max': None}

    def test_the_text_report_says_it_found_nothing(self, tmp_path, capsys):
        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert 'no ' in out.lower() or 'none' in out.lower() or '0 ' in out

    def test_a_root_with_records_but_no_matching_module_is_also_empty(
        self, tmp_path, capsys,
    ):
        """Zero SELECTED is a different fact from zero RECORDS, and both are
        reported — the rejection counts are what tell them apart."""
        _corpus_with(tmp_path, _leg())

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'shared', '--json',
        )

        report = json.loads(out)
        assert report['overall']['durations']['n'] == 0
        assert report['corpus']['records'] == 1
        assert report['rejected_records']['prefix_mismatch'] == 1

    def test_the_two_rejection_units_stay_apart_in_the_text(self, tmp_path, capsys):
        """The operator-facing half: a reader must not sum the two counters.

        They answer different questions — "the shape filter stopped matching"
        (entries) versus "most of this corpus is other modules" (records) — and
        a single undifferentiated `rejected {...}` blob invites adding them,
        which is how the mixed-unit counter this replaced read as reconcilable
        when it was not. So each line names its own unit, and the reasons are
        asserted to land on the right one rather than merely to appear.
        """
        _corpus_with(
            tmp_path,
            _leg(),
            _leg(label='lint', cmd='uv run ruff check src/'),
        )
        other = _worktree_record(tmp_path, '4242', 'attempt-1.shared.summary.json')
        other.write_text(json.dumps(_summary(_leg(), _leg())), encoding='utf-8')

        _rc, text, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        lines = {
            word: line
            for line in text.splitlines()
            for word in ('entries', 'records')
            if line.strip().startswith(word)
        }
        assert set(lines) == {'entries', 'records'}, text
        assert 'prefix_mismatch' in lines['records']
        assert 'prefix_mismatch' not in lines['entries']
        assert 'label_mismatch' in lines['entries']
        assert 'label_mismatch' not in lines['records']


class TestTheReportCarriesItsOwnProvenance:
    """A figure without its window and its filters is not reproducible."""

    def test_the_report_names_its_window_module_label_and_role(
        self, tmp_path, capsys,
    ):
        _corpus_with(tmp_path, _leg())

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
            '--role', 'task', '--json',
        )

        report = json.loads(out)
        assert report['module'] == 'orchestrator'
        assert report['label'] == 'test'
        assert report['role'] == 'task'
        assert len(report['window']) == 2

    def test_the_report_names_the_command_it_compared_against(self, tmp_path, capsys):
        """So a reader can see WHICH command's distribution this is."""
        _corpus_with(tmp_path, _leg())

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )

        assert json.loads(out)['expected_command'] == _FULL_SUITE

    def test_the_report_carries_the_skip_and_rejection_counts(self, tmp_path, capsys):
        """The reconciliation trail, in the artifact itself."""
        _corpus_with(tmp_path, _leg())

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )

        report = json.loads(out)
        assert 'skipped' in report['corpus']
        assert 'rejected_entries' in report
        assert 'rejected_records' in report

    def test_the_findings_ride_in_the_report(self, tmp_path, capsys):
        _corpus_with(tmp_path, _leg())

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )

        report = json.loads(out)
        assert report['cold_separability']['cold_separable'] is False
        assert report['merge_gate']['always_cold'] is True
        assert 'unstamped' in report['by_load_band']


# An unmistakable stand-in for the family's answer, used by the delegation
# guards below. Deliberately BELOW `REFUSAL_CEILING` (7200) so the section under
# test stays in its ordinary arm and the guards do not accidentally assert the
# escalation branch, and deliberately not a value `census_budget_floor` would
# ever return for the fixtures here (those derive 5700-7500), so agreement with
# it cannot be a coincidence.
_SENTINEL_FLOOR = 4242


class TestTheCensusAndTheGuardShareOneDerivation:
    """INV-10, closing guard: the report and the gate cannot drift apart.

    The census prints a derived floor and
    `test_module_verify_budgets.py`'s excepted branch asserts one. If each
    spelled `1.5 * max` for itself they could disagree silently — and the
    disagreement would surface as an operator running the census, reading a
    floor, setting the budget to it, and watching the gate go red anyway.

    This is the same property
    `test_the_budget_family_derives_every_floor_from_one_canonical_expression`
    establishes for `min_budget`, applied to the new expression, and it is what
    makes step-22's recorded run a genuine REPEAT of what the guard asserts
    rather than a second opinion.
    """

    def _report(self, tmp_path, capsys, *entries):
        _corpus_with(tmp_path, *entries)
        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )
        return json.loads(out)

    def test_the_printed_floor_equals_the_familys_expression(self, tmp_path, capsys):
        from module_budget_family import census_budget_floor  # noqa: PLC0415

        report = self._report(tmp_path, capsys, _leg(duration_secs=4626.166946739017))

        assert report['budget_check']['derived_floor'] == census_budget_floor(
            4626.166946739017,
        )

    @pytest.mark.parametrize('worst', [3753.0, 4626.166946739017, 4991.13326132996])
    def test_it_agrees_at_every_measured_figure(self, tmp_path, capsys, worst):
        """Agreement at one value could be coincidence; at the three that
        matter it is the shared expression."""
        from module_budget_family import census_budget_floor  # noqa: PLC0415

        report = self._report(tmp_path, capsys, _leg(duration_secs=worst))

        assert report['budget_check']['derived_floor'] == census_budget_floor(worst)

    def test_the_section_names_the_constant_it_feeds(self, tmp_path, capsys):
        """So the operator running it knows where the number lands."""
        report = self._report(tmp_path, capsys, _leg(duration_secs=4626.0))

        assert 'ORCHESTRATOR_BUDGET_CENSUS' in report['budget_check']['feeds']

    def test_the_section_names_the_refusal_ceiling_and_its_meaning(
        self, tmp_path, capsys,
    ):
        """A derived floor above the ceiling is a FINDING, and the report must
        not let an operator discover that only from a red gate."""
        report = self._report(tmp_path, capsys, _leg(duration_secs=4991.13326132996))

        check = report['budget_check']
        assert check['refusal_ceiling'] == 7200
        assert check['exceeds_refusal_ceiling'] is True
        assert 'escalate' in check['note'].lower()

    def test_a_floor_under_the_ceiling_says_so(self, tmp_path, capsys):
        report = self._report(tmp_path, capsys, _leg(duration_secs=4626.166946739017))

        assert report['budget_check']['exceeds_refusal_ceiling'] is False

    def test_an_empty_selection_derives_no_floor(self, tmp_path, capsys):
        """No max means no floor — never a 0 that reads as a derived answer."""
        report = self._report(tmp_path, capsys, _leg(cmd='pytest tests/test_foo.py'))

        assert report['budget_check']['derived_floor'] is None
        assert report['budget_check']['exceeds_refusal_ceiling'] is False

    def test_the_text_report_prints_the_floor(self, tmp_path, capsys):
        _corpus_with(tmp_path, _leg(duration_secs=4626.166946739017))

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert '7000' in out
        assert 'ORCHESTRATOR_BUDGET_CENSUS' in out

    def test_the_floor_is_whatever_the_family_returns(
        self, tmp_path, capsys, monkeypatch,
    ):
        """Delegation asserted by BEHAVIOUR: patch the family, and the report
        follows it anywhere.

        The import is the mechanism, so its absence is the failure mode — but
        the absence is only observable by running the thing. `_census_budget_
        floor` performs its `from module_budget_family import
        census_budget_floor` INSIDE the function body, so the name is resolved
        at CALL time and this patch is seen. A census that re-spelled the
        multiple for itself — `1.5 * worst`, `worst * 1.5`, `3 * worst / 2`,
        `Fraction(3, 2) * worst`, any of them — would keep returning the real
        floor here and go red, which no source-text pattern over this file
        reliably distinguishes.
        """
        import module_budget_family  # noqa: PLC0415

        monkeypatch.setattr(
            module_budget_family,
            'census_budget_floor',
            lambda _worst: _SENTINEL_FLOOR,
        )

        report = self._report(tmp_path, capsys, _leg(duration_secs=4626.166946739017))

        assert report['budget_check']['derived_floor'] == _SENTINEL_FLOOR

    def test_the_text_rendering_follows_the_family_too(
        self, tmp_path, capsys, monkeypatch,
    ):
        """`format_report` is a second consumer of the same key, and the
        operator-facing one — the floor an operator copies out of the text
        report has to be the family's number, not a second opinion."""
        import module_budget_family  # noqa: PLC0415

        monkeypatch.setattr(
            module_budget_family,
            'census_budget_floor',
            lambda _worst: _SENTINEL_FLOOR,
        )
        _corpus_with(tmp_path, _leg(duration_secs=4626.166946739017))

        _rc, out, _err = _main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert str(_SENTINEL_FLOOR) in out

    def test_an_unreachable_family_derives_nothing_rather_than_guessing(
        self, tmp_path, capsys, monkeypatch,
    ):
        """The `except ImportError` arm: no floor, and a note that says so.

        A locally-recomputed fallback is the one thing that arm must not do,
        so the ABSENCE of a number is the contract — a census run from a
        partial checkout must report that it cannot derive the floor rather
        than print one the gate never agreed to.

        Evicting the module from `sys.modules` is not enough on its own:
        `tests/scripts` sits on `sys.path` (conftest puts it there) and
        `_census_budget_floor` re-inserts `_FAMILY_DIR` itself, so both have to
        go for the import to actually fail.
        """
        import sys  # noqa: PLC0415

        import verify_budget_census as mod  # noqa: PLC0415

        no_family = tmp_path / 'no-family'
        no_family.mkdir()
        monkeypatch.setattr(mod, '_FAMILY_DIR', no_family)
        monkeypatch.setattr(sys, 'path', [
            entry for entry in sys.path
            if not (Path(entry or '.') / 'module_budget_family.py').exists()
        ])
        monkeypatch.delitem(sys.modules, 'module_budget_family', raising=False)

        check = self._report(
            tmp_path, capsys, _leg(duration_secs=4626.166946739017),
        )['budget_check']

        assert check['derived_floor'] is None
        assert 'unavailable' in check['note'].lower()
        assert check['exceeds_refusal_ceiling'] is False


class TestATornOrMalformedRecordIsCountedNotFatal:
    """The walk stays TOTAL: a bad record is counted, never a traceback.

    Both shapes here aborted the whole census before esc-3353-20, discarding
    every record already parsed — the one failure a read-only walk over a LIVE
    tree must not have, and the exact case the module docstring advertises
    safety for.
    """

    def test_a_read_torn_mid_multibyte_character_is_unreadable(self, tmp_path):
        """Summaries are written `ensure_ascii=False`, so this is reachable."""
        from verify_budget_census import _read_payload  # noqa: PLC0415

        raw = json.dumps({'cmd': 'pytest — dash'}, ensure_ascii=False).encode('utf-8')
        torn = tmp_path / 'attempt-1.orchestrator.summary.json'
        torn.write_bytes(raw[:raw.find('—'.encode()) + 1])

        assert _read_payload(torn) == (None, 'unreadable')

    def test_a_torn_module_yaml_is_none_not_a_traceback(self, tmp_path):
        from verify_budget_census import read_module_test_command  # noqa: PLC0415

        module_yaml = tmp_path / 'orchestrator' / 'orchestrator.yaml'
        module_yaml.parent.mkdir(parents=True)
        module_yaml.write_bytes('test_command: "pytest —'.encode()[:-1])

        assert read_module_test_command(tmp_path, 'orchestrator') is None

    def test_an_entry_without_rc_is_rejected_not_indexed(self, tmp_path):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        entry = _leg()
        del entry['rc']
        _corpus_with(tmp_path, entry)

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        )

        assert selection.legs == ()
        assert selection.rejected_entries['no_rc'] == 1

    def test_a_null_rc_is_rejected_rather_than_counted_as_a_failure(self, tmp_path):
        """The worse half: `rc: null` was ACCEPTED and filed under `failed`.

        `summarise_legs` partitions on `leg.rc != 0`, so None — an unusable
        record — became evidence of a failing run in the very distribution a
        budget is derived from.
        """
        from verify_budget_census import (  # noqa: PLC0415
            load_records,
            select_full_suite_legs,
            summarise_legs,
        )

        _corpus_with(tmp_path, _leg(rc=None))

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        )

        assert selection.rejected_entries['no_rc'] == 1
        assert summarise_legs(selection.legs)['failed'] == 0

    def test_a_bool_is_not_a_duration(self, tmp_path):
        """`bool` subclasses `int`, so `True` was admitted as a 1.0s suite."""
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        _corpus_with(tmp_path, _leg(duration_secs=True))

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=_FULL_SUITE, prefix='orchestrator',
        )

        assert selection.legs == ()
        assert selection.rejected_entries['no_duration'] == 1
