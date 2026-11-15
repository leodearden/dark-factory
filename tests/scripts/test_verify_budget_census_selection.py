"""select_full_suite_legs and read_module_test_command: which commands[] entries
count as full-suite legs, with every rejection counted.

The synthetic-corpus norm lives in tests/scripts/verify_budget_census_fixture.py.
"""
from __future__ import annotations

import json

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from verify_budget_census_fixture import (
    FULL_SUITE,
    STAMP_US,
    command_entry,
    run_main,
    summary_payload,
    write_archive_record,
    write_corpus,
    write_worktree_record,
)

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


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

        write_corpus(tmp_path, command_entry(duration_secs=3300.0))

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        )

        assert [leg.duration_secs for leg in selection.legs] == [3300.0]

    def test_a_loud_top_level_does_not_mask_a_green_test_leg(self, tmp_path):
        """The top level says rc=-9; the selected leg says rc=0. Both are true."""
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        write_corpus(tmp_path, command_entry(rc=0))

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        )

        assert [leg.rc for leg in selection.legs] == [0]


class TestTheFullSuiteShapeFilter:
    """Selection is by label AND by command SHAPE, each rejection counted."""

    def _select(self, tmp_path, *commands):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        write_corpus(tmp_path, *commands)
        return select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        )

    def test_the_verbatim_declared_command_is_selected(self, tmp_path):
        selection = self._select(tmp_path, command_entry())

        assert len(selection.legs) == 1
        assert selection.rejected_entries == {}
        assert selection.rejected_records == {}

    def test_whitespace_only_differences_are_tolerated(self, tmp_path):
        """A re-rendered command may differ in spacing and still be the suite."""
        spaced = FULL_SUITE.replace(' pytest ', '  pytest   ')
        selection = self._select(tmp_path, command_entry(cmd=spaced))

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
        selection = self._select(tmp_path, command_entry(cmd=cmd))

        assert selection.legs == (), f'{note} run was admitted'
        assert selection.rejected_entries == {'command_mismatch': 1}

    def test_a_segmented_entry_is_rejected_with_its_own_reason(self, tmp_path):
        """A segmented leg's duration covers a DIFFERENT topology of the same
        chain, so it is excluded — and counted separately from a plain command
        mismatch, because it is a different fact about the corpus."""
        selection = self._select(
            tmp_path, command_entry(segments=[{'index': 1, 'label': 'a'}]),
        )

        assert selection.legs == ()
        assert selection.rejected_entries == {'segmented': 1}

    def test_a_non_test_leg_is_rejected(self, tmp_path):
        selection = self._select(
            tmp_path,
            command_entry(label='lint', cmd='uv run --directory orchestrator ruff check src/'),
            command_entry(label='type', cmd='uv run --directory orchestrator pyright src/'),
        )

        assert selection.legs == ()
        assert selection.rejected_entries == {'label_mismatch': 2}

    def test_a_null_cmd_is_rejected_rather_than_compared(self, tmp_path):
        """A skipped leg ran nothing, so there is no duration to census."""
        selection = self._select(tmp_path, command_entry(cmd=None, duration_secs=0.0))

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

        write_corpus(
            tmp_path,
            command_entry(),
            command_entry(label='lint', cmd='uv run ruff check src/'),
            command_entry(cmd='uv run --directory orchestrator pytest tests/test_foo.py'),
        )
        other = write_worktree_record(tmp_path, '4242', 'attempt-1.shared.summary.json')
        other.write_text(
            json.dumps(summary_payload(command_entry(), command_entry(), command_entry())), encoding='utf-8',
        )

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        )

        # Entries: only the matching record's three were ever READ.
        assert len(selection.legs) + sum(selection.rejected_entries.values()) == 3
        # Records: the other module's is accounted for exactly once, and its
        # three entries are nowhere in the entry counter.
        assert selection.rejected_records == {'prefix_mismatch': 1}

    def test_another_modules_record_is_rejected_on_prefix(self, tmp_path):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        write_corpus(tmp_path, command_entry())
        other = write_worktree_record(tmp_path, '4242', 'attempt-1.shared.summary.json')
        other.write_text(json.dumps(summary_payload(command_entry())), encoding='utf-8')

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        )

        assert len(selection.legs) == 1
        assert selection.rejected_records == {'prefix_mismatch': 1}
        assert selection.rejected_entries == {}

    def test_a_role_filter_excludes_other_lanes(self, tmp_path):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        write_corpus(tmp_path, command_entry())
        merge = write_worktree_record(
            tmp_path, '_merge-abc123', 'attempt-1.orchestrator.summary.json',
        )
        merge.write_text(json.dumps(summary_payload(command_entry())), encoding='utf-8')

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator', role='task',
        )

        assert len(selection.legs) == 1
        assert selection.rejected_records == {'role_mismatch': 1}
        assert selection.rejected_entries == {}


class TestTheStampedRoleWinsOverThePath:
    """Since task 5671 a summary names its own role; the path is the fallback.

    The archive path says nothing about which lane wrote a record, so before
    the stamp every archived record was role-less and a ``--role merge``
    census selected none of them. Records written before the stamp keep the
    path inference that
    ``tests/scripts/test_verify_budget_census_corpus.py::TestParseRecordPathRoleClassification``
    describes.
    """

    @staticmethod
    def _only_record(root):
        from verify_budget_census import load_records  # noqa: PLC0415

        corpus = load_records([root])
        assert len(corpus.records) == 1, corpus
        return corpus.records[0]

    @staticmethod
    def _write(path, **top):
        path.write_text(json.dumps(summary_payload(command_entry(), **top)), encoding='utf-8')
        return path

    def test_an_archive_record_stamped_merge_is_a_merge_record(self, tmp_path):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        self._write(
            write_archive_record(
                tmp_path, '5422', f'attempt-1.orchestrator.summary-{STAMP_US}.json',
            ),
            role='merge',
        )

        assert self._only_record(tmp_path).role == 'merge'
        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
            role='merge',
        )
        assert len(selection.legs) == 1
        assert selection.rejected_records == {}

    def test_the_stamp_wins_over_the_lane_the_record_sits_in(self, tmp_path):
        self._write(
            write_worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json'),
            role='background',
        )

        assert self._only_record(tmp_path).role == 'background'

    def test_an_unstamped_worktree_record_keeps_its_lane_role(self, tmp_path):
        self._write(
            write_worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json'),
        )

        assert self._only_record(tmp_path).role == 'task'

    def test_an_unstamped_archive_record_stays_unknown(self, tmp_path):
        self._write(
            write_archive_record(
                tmp_path, '5422', f'attempt-1.orchestrator.summary-{STAMP_US}.json',
            ),
        )

        assert self._only_record(tmp_path).role is None

    @pytest.mark.parametrize('stamp', [7, None], ids=['int', 'null'])
    def test_a_malformed_stamp_falls_back_to_the_path(self, tmp_path, stamp):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        self._write(
            write_worktree_record(tmp_path, '3353', 'attempt-1.orchestrator.summary.json'),
            role=stamp,
        )

        assert self._only_record(tmp_path).role == 'task'
        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
            role='task',
        )
        assert len(selection.legs) == 1


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
        write_corpus(tmp_path, command_entry(cmd='pytest tests/ --brand-new-flag'))
        (tmp_path / 'orchestrator' / 'orchestrator.yaml').write_text(
            'test_command: "pytest tests/ --brand-new-flag"\n', encoding='utf-8',
        )

        _rc, out, _err = run_main(
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


class TestATornOrMalformedRecordIsCountedNotFatal:
    """The walk stays TOTAL: a bad record is counted, never a traceback.

    Both shapes here aborted the whole census before esc-3353-20, discarding
    every record already parsed — the one failure a read-only walk over a LIVE
    tree must not have, and the exact case the module docstring advertises
    safety for.
    """

    def test_a_torn_module_yaml_is_none_not_a_traceback(self, tmp_path):
        from verify_budget_census import read_module_test_command  # noqa: PLC0415

        module_yaml = tmp_path / 'orchestrator' / 'orchestrator.yaml'
        module_yaml.parent.mkdir(parents=True)
        module_yaml.write_bytes('test_command: "pytest —'.encode()[:-1])

        assert read_module_test_command(tmp_path, 'orchestrator') is None

    def test_an_entry_without_rc_is_rejected_not_indexed(self, tmp_path):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        entry = command_entry()
        del entry['rc']
        write_corpus(tmp_path, entry)

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
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

        write_corpus(tmp_path, command_entry(rc=None))

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        )

        assert selection.rejected_entries['no_rc'] == 1
        assert summarise_legs(selection.legs)['failed'] == 0

    def test_a_bool_is_not_a_duration(self, tmp_path):
        """`bool` subclasses `int`, so `True` was admitted as a 1.0s suite."""
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        write_corpus(tmp_path, command_entry(duration_secs=True))

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        )

        assert selection.legs == ()
        assert selection.rejected_entries['no_duration'] == 1
