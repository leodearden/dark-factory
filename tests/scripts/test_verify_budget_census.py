"""scripts/verify_budget_census.py's main(argv, now) and the one report dict both
renderings read, including its provenance and a budget floor that follows the
rule the budget gate applies to the censused module.

The synthetic-corpus norm lives in tests/scripts/verify_budget_census_fixture.py.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import module_budget_family
import pytest
import verify_budget_census
import verify_budget_census_records
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from verify_budget_census_fixture import (
    FULL_SUITE,
    command_entry,
    run_main,
    summary_payload,
    write_corpus,
    write_module_yaml,
    write_worktree_record,
)

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _budget_section(text: str) -> str:
    """The text report's floor section: its header line up to the next blank line."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('DERIVED BUDGET FLOOR'))
    end = next((i for i in range(start, len(lines)) if not lines[i].strip()), len(lines))
    return '\n'.join(lines[start:end])


def _module_level_literal(path: Path, symbol: str) -> Any:
    """The literal bound to module-level *symbol* in *path*, or None if unbound."""
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == symbol for target in targets):
            return ast.literal_eval(value)
    return None


class TestTheCliContract:
    """`main(argv, now=None)` over the same report dict the text renderer reads.

    `--root` is the injection point that makes the whole script testable
    against a synthetic corpus. Without it every test here would have to run
    against the live tree, which is the thing this suite's norm forbids.
    """

    def test_a_synthetic_corpus_reports_its_own_numbers(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry(duration_secs=3300.0))

        rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert rc == 0
        assert '3300' in out

    def test_json_switches_the_whole_output_to_one_document(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry(duration_secs=3300.0))

        rc, out, _err = run_main(
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
        write_corpus(tmp_path, command_entry(duration_secs=3300.0))

        _rc, text, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )
        _rc, raw, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )

        reported_max = json.loads(raw)['overall']['durations']['max']
        assert str(int(reported_max)) in text

    def test_the_window_is_resolved_against_the_injected_clock(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry(duration_secs=3300.0))

        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
            '--window', '14d', '--json',
        )

        lo, hi = json.loads(out)['window']
        assert hi.startswith('2026-09-14T12:00:00')
        assert lo.startswith('2026-08-31T12:00:00')

    def test_the_dated_window_is_honoured(self, tmp_path, capsys):
        write_corpus(
            tmp_path,
            command_entry(started_at='2026-09-13T04:00:00+00:00', duration_secs=3300.0),
            command_entry(started_at='2026-08-25T04:00:00+00:00', duration_secs=4991.0),
        )

        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
            '--window', '2026-09-12T08:00:00+00:00..2026-09-14T12:00:00+00:00',
        )

        assert json.loads(out)['overall']['durations']['max'] == 3300.0

    def test_the_role_filter_reaches_the_selector(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry(duration_secs=3300.0))
        merge = write_worktree_record(
            tmp_path, '_merge-abc123', 'attempt-1.orchestrator.summary.json',
        )
        merge.write_text(
            json.dumps(summary_payload(command_entry(duration_secs=9999.0))), encoding='utf-8',
        )

        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
            '--role', 'task', '--json',
        )

        assert json.loads(out)['overall']['durations']['max'] == 3300.0

    def test_the_label_filter_reaches_the_selector(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry(label='lint', cmd=FULL_SUITE))

        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
            '--label', 'lint', '--json',
        )

        assert json.loads(out)['overall']['durations']['n'] == 1

    def test_roots_are_repeatable(self, tmp_path, capsys):
        one, two = tmp_path / 'one', tmp_path / 'two'
        write_corpus(one, command_entry(duration_secs=3300.0), lane='1')
        write_corpus(two, command_entry(duration_secs=3400.0), lane='2')

        _rc, out, _err = run_main(
            capsys, '--root', str(one), '--root', str(two),
            '--module', 'orchestrator', '--json',
        )

        assert json.loads(out)['overall']['durations']['n'] == 2

    def test_a_first_root_without_the_module_yaml_still_selects(
        self, tmp_path, capsys,
    ):
        """The command is resolved across ALL roots, and the selection uses it.

        `write_corpus` writes the module yaml into EVERY root it builds, which
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
        write_corpus(declaring, command_entry(duration_secs=3300.0), lane='2')
        write_corpus(bare, command_entry(duration_secs=3400.0), lane='1')
        (bare / 'orchestrator' / 'orchestrator.yaml').unlink()

        _rc, out, _err = run_main(
            capsys, '--root', str(bare), '--root', str(declaring),
            '--module', 'orchestrator', '--json',
        )
        report = json.loads(out)

        assert report['expected_command'] == FULL_SUITE
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
        write_corpus(tmp_path, command_entry(duration_secs=3300.0))
        (tmp_path / 'orchestrator' / 'orchestrator.yaml').unlink()

        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )
        report = json.loads(out)

        assert report['expected_command'] is None
        assert report['rejected_entries']['no_declared_command'] == 1
        assert report['overall']['durations']['n'] == 0

        _rc, text, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert '<none declared>' in text


class TestTheExitCodeVocabulary:
    """0 / 1 / 2, as `merge_lane_throughput.main` documents."""

    def test_success_is_zero(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry())

        rc, _out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert rc == 0

    def test_an_unreadable_named_root_is_one_and_the_others_still_report(
        self, tmp_path, capsys,
    ):
        """Partial failure is reported on stderr, not turned into a crash."""
        good = tmp_path / 'good'
        write_corpus(good, command_entry(duration_secs=3300.0))

        rc, out, err = run_main(
            capsys, '--root', str(good), '--root', str(tmp_path / 'absent'),
            '--module', 'orchestrator', '--json',
        )

        assert rc == 1
        assert 'absent' in err
        assert json.loads(out)['overall']['durations']['max'] == 3300.0

    def test_a_malformed_window_is_two_with_nothing_on_stdout(self, tmp_path, capsys):
        rc, out, err = run_main(
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
        rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert rc == 0
        assert out

    def test_the_empty_report_carries_nulls_not_zeros(self, tmp_path, capsys):
        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )

        durations = json.loads(out)['overall']['durations']
        assert durations == {'n': 0, 'p50': None, 'p90': None, 'max': None}

    def test_the_text_report_says_it_found_nothing(self, tmp_path, capsys):
        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert 'no ' in out.lower() or 'none' in out.lower() or '0 ' in out

    def test_a_root_with_records_but_no_matching_module_is_also_empty(
        self, tmp_path, capsys,
    ):
        """Zero SELECTED is a different fact from zero RECORDS, and both are
        reported — the rejection counts are what tell them apart."""
        write_corpus(tmp_path, command_entry())

        _rc, out, _err = run_main(
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
        write_corpus(
            tmp_path,
            command_entry(),
            command_entry(label='lint', cmd='uv run ruff check src/'),
        )
        other = write_worktree_record(tmp_path, '4242', 'attempt-1.shared.summary.json')
        other.write_text(json.dumps(summary_payload(command_entry(), command_entry())), encoding='utf-8')

        _rc, text, _err = run_main(
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
        write_corpus(tmp_path, command_entry())

        _rc, out, _err = run_main(
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
        write_corpus(tmp_path, command_entry())

        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )

        assert json.loads(out)['expected_command'] == FULL_SUITE

    def test_the_report_carries_the_skip_and_rejection_counts(self, tmp_path, capsys):
        """The reconciliation trail, in the artifact itself."""
        write_corpus(tmp_path, command_entry())

        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )

        report = json.loads(out)
        assert 'skipped' in report['corpus']
        assert 'rejected_entries' in report
        assert 'rejected_records' in report

    def test_the_findings_ride_in_the_report(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry())

        _rc, out, _err = run_main(
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
        write_corpus(tmp_path, *entries)
        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )
        return json.loads(out)

    def test_the_printed_floor_equals_the_familys_expression(self, tmp_path, capsys):
        from module_budget_family import census_budget_floor  # noqa: PLC0415

        report = self._report(tmp_path, capsys, command_entry(duration_secs=4626.166946739017))

        assert report['budget_check']['derived_floor'] == census_budget_floor(
            4626.166946739017,
        )

    @pytest.mark.parametrize('worst', [3753.0, 4626.166946739017, 4991.13326132996])
    def test_it_agrees_at_every_measured_figure(self, tmp_path, capsys, worst):
        """Agreement at one value could be coincidence; at the three that
        matter it is the shared expression."""
        from module_budget_family import census_budget_floor  # noqa: PLC0415

        report = self._report(tmp_path, capsys, command_entry(duration_secs=worst))

        assert report['budget_check']['derived_floor'] == census_budget_floor(worst)

    def test_the_section_names_the_constant_it_feeds(self, tmp_path, capsys):
        """So the operator running it knows where the number lands.

        The pointer must RESOLVE to the record's live home, not merely mention
        its name: a moved record would leave a name-only check passing on a
        stale path (task 6172).
        """
        check = self._report(tmp_path, capsys, command_entry(duration_secs=4626.0))['budget_check']
        feeds = check['feeds']
        resolver = 'scripts/verify_budget_census.py::gate_floor_rule'

        assert check['rule'] == 'd17_census'
        assert (REPO_ROOT / feeds['path']).resolve() == Path(
            verify_budget_census_records.__file__,
        ).resolve(), (
            f"feeds names {feeds['path']!r}, not the census record home "
            f'{verify_budget_census_records.__file__}; repoint {resolver}'
        )
        cited = getattr(verify_budget_census_records, feeds['symbol'], {})
        assert cited.get(feeds['key']) is verify_budget_census_records.ORCHESTRATOR_BUDGET_CENSUS, (
            f"feeds names {feeds['symbol']}[{feeds['key']!r}], which is not "
            f'verify_budget_census_records.ORCHESTRATOR_BUDGET_CENSUS; repoint {resolver}'
        )

    def test_the_section_names_the_refusal_ceiling_and_its_meaning(
        self, tmp_path, capsys,
    ):
        """A derived floor above the ceiling is a FINDING, and the report must
        not let an operator discover that only from a red gate."""
        report = self._report(tmp_path, capsys, command_entry(duration_secs=4991.13326132996))

        check = report['budget_check']
        assert check['refusal_ceiling'] == 7200
        assert check['exceeds_refusal_ceiling'] is True
        assert 'escalate' in check['note'].lower()

    def test_a_floor_under_the_ceiling_says_so(self, tmp_path, capsys):
        report = self._report(tmp_path, capsys, command_entry(duration_secs=4626.166946739017))

        assert report['budget_check']['exceeds_refusal_ceiling'] is False

    def test_an_empty_selection_derives_no_floor(self, tmp_path, capsys):
        """No max means no floor — never a 0 that reads as a derived answer."""
        report = self._report(tmp_path, capsys, command_entry(cmd='pytest tests/test_foo.py'))

        assert report['budget_check']['derived_floor'] is None
        assert report['budget_check']['exceeds_refusal_ceiling'] is False

    def test_the_text_report_prints_the_floor(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry(duration_secs=4626.166946739017))

        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )

        assert '7000' in out
        assert "RULED_BUDGET_CENSUS['orchestrator']" in out

    def test_the_floor_is_whatever_the_family_returns(
        self, tmp_path, capsys, monkeypatch,
    ):
        """Delegation asserted by BEHAVIOUR: patch the family, and the report
        follows it anywhere.

        The import is the mechanism, so its absence is the failure mode — but
        the absence is only observable by running the thing. `gate_floor_rule`
        imports `module_budget_family` at CALL time and reads
        `census_budget_floor` off the module object, so this patch is seen. A
        census that re-spelled the multiple for itself — `1.5 * worst`,
        `worst * 1.5`, `3 * worst / 2`, `Fraction(3, 2) * worst`, any of them —
        would keep returning the real floor here and go red, which no
        source-text pattern over this file reliably distinguishes.
        """
        import module_budget_family  # noqa: PLC0415

        monkeypatch.setattr(
            module_budget_family,
            'census_budget_floor',
            lambda _worst: _SENTINEL_FLOOR,
        )

        report = self._report(tmp_path, capsys, command_entry(duration_secs=4626.166946739017))

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
        write_corpus(tmp_path, command_entry(duration_secs=4626.166946739017))

        _rc, out, _err = run_main(
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
        `_import_from_family_dir` re-inserts `_FAMILY_DIR` itself, so both have to
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
            tmp_path, capsys, command_entry(duration_secs=4626.166946739017),
        )['budget_check']

        assert check['derived_floor'] is None
        assert 'unavailable' in check['note'].lower()
        assert check['exceeds_refusal_ceiling'] is False


class TestTheFloorSectionFollowsTheRuleTheGateApplies:
    """Task 7929: the section states the floor the budget gate enforces for --module.

    Only the modules in ``verify_budget_census_records.RULED_BUDGET_CENSUS`` are
    held to D17's census floor. The budget guards hold every other module to
    ``min_budget(W)``, so the section must derive that floor and cite where W lives.
    """

    def _check(self, tmp_path, capsys, module):
        _rc, out, _err = run_main(capsys, '--root', str(tmp_path), '--module', module, '--json')
        return json.loads(out)['budget_check']

    def _section(self, tmp_path, capsys, module):
        _rc, out, _err = run_main(capsys, '--root', str(tmp_path), '--module', module)
        return _budget_section(out)

    def test_a_gate_table_module_gets_min_budget_and_cites_its_row(self, tmp_path, capsys):
        worst = 1695.0841336008161
        write_corpus(tmp_path, command_entry(duration_secs=worst), prefix='fused-memory')

        check = self._check(tmp_path, capsys, 'fused-memory')

        assert check['rule'] == 'min_budget'
        assert check['derived_floor'] == module_budget_family.min_budget(worst)
        assert check['refusal_ceiling'] is None
        assert check['exceeds_refusal_ceiling'] is False
        assert check['provenance'] == 'fused-memory/orchestrator.yaml'
        feeds = check['feeds']
        assert feeds['key'] == 'fused-memory'
        cited_path = (REPO_ROOT / feeds['path']).resolve()
        assert cited_path == module_budget_family.FAMILY_READER_PATH.resolve()
        assert 'fused-memory' in (_module_level_literal(cited_path, feeds['symbol']) or {}), (
            f"feeds cites {feeds['path']}::{feeds['symbol']}[{feeds['key']!r}], "
            'which does not resolve'
        )

    def test_the_text_section_states_the_min_budget_rule_and_nothing_of_d17(
        self, tmp_path, capsys,
    ):
        write_corpus(
            tmp_path, command_entry(duration_secs=1695.0841336008161), prefix='fused-memory',
        )

        section = self._section(tmp_path, capsys, 'fused-memory')

        for expected in (
            '3300',
            'min_budget',
            "MEASURED_MODULE_SUITE_WORST_SECS['fused-memory']",
            'fused-memory/orchestrator.yaml',
        ):
            assert expected in section, section
        for absent in (
            'D17', '1.5x', 'refusal ceiling', 'RULED_BUDGET_CENSUS', 'ORCHESTRATOR_BUDGET_CENSUS',
        ):
            assert absent not in section, section

    def test_the_min_budget_floor_is_whatever_the_family_returns(
        self, tmp_path, capsys, monkeypatch,
    ):
        monkeypatch.setattr(module_budget_family, 'min_budget', lambda _worst: 4343)
        write_corpus(
            tmp_path, command_entry(duration_secs=1695.0841336008161), prefix='fused-memory',
        )

        assert self._check(tmp_path, capsys, 'fused-memory')['derived_floor'] == 4343
        assert '4343' in self._section(tmp_path, capsys, 'fused-memory')

    def test_a_degenerate_floor_of_zero_prints_as_zero(self, tmp_path, capsys):
        """min_budget legitimately returns 0 for a suite under ~50s."""
        write_corpus(tmp_path, command_entry(duration_secs=22.49), prefix='fused-memory')

        assert self._check(tmp_path, capsys, 'fused-memory')['derived_floor'] == 0
        section = self._section(tmp_path, capsys, 'fused-memory')
        floor_line = next(
            line for line in section.splitlines() if line.strip().startswith('derived floor')
        )
        assert floor_line.split()[2] == '0', floor_line
        assert '<none>' not in floor_line

    @staticmethod
    def _write_sibling_corpus(tmp_path, module, worst):
        """Built without write_corpus, whose record name would nest on the slash
        in 'tests/scripts'."""
        write_module_yaml(tmp_path, module)
        record = write_worktree_record(
            tmp_path, '3353', f'attempt-1.{verify_budget_census.sanitise_prefix(module)}.summary.json',
        )
        record.write_text(
            json.dumps(summary_payload(command_entry(duration_secs=worst))), encoding='utf-8',
        )

    @pytest.mark.parametrize('module', sorted(module_budget_family.published_pairs()))
    def test_a_sibling_guarded_module_cites_its_own_publisher(self, tmp_path, capsys, module):
        """These modules' W lives in their own sibling guard, not in the gate's table."""
        worst = 500.0
        self._write_sibling_corpus(tmp_path, module, worst)

        _rc, out, _err = run_main(capsys, '--root', str(tmp_path), '--module', module, '--json')
        report = json.loads(out)

        assert report['overall']['durations']['n'] == 1
        check = report['budget_check']
        feeds = check['feeds']
        assert check['rule'] == 'min_budget'
        assert check['derived_floor'] == module_budget_family.min_budget(worst)
        assert check['derived_floor'] != module_budget_family.census_budget_floor(worst)
        assert check['refusal_ceiling'] is None
        publisher = module_budget_family.published_pairs()[module]
        assert (REPO_ROOT / feeds['path']).resolve() == publisher.path.resolve()
        assert feeds['symbol'] == module_budget_family.WORST_NAME
        assert feeds['key'] is None
        assert 'MEASURED_MODULE_SUITE_WORST_SECS' not in self._section(tmp_path, capsys, module)

    @pytest.mark.parametrize('module', sorted(module_budget_family.published_pairs()))
    def test_a_sibling_guarded_floor_is_whatever_min_budget_returns(
        self, tmp_path, capsys, monkeypatch, module,
    ):
        monkeypatch.setattr(module_budget_family, 'min_budget', lambda _worst: 4343)
        self._write_sibling_corpus(tmp_path, module, 500.0)

        assert self._check(tmp_path, capsys, module)['derived_floor'] == 4343
        assert '4343' in self._section(tmp_path, capsys, module)

    def test_only_the_unverified_gate_row_citation_says_it_is_unchecked(self, tmp_path, capsys):
        """The ruled and sibling citations are resolved from what the census
        imports; the gate-table row is not, so a module outside every table
        (here an unregistered one with measured runs) cites a row that does not exist."""
        write_corpus(tmp_path, command_entry(duration_secs=900.0), prefix='unregistered')
        write_corpus(tmp_path, command_entry(duration_secs=4626.166946739017))
        self._write_sibling_corpus(tmp_path, 'scripts', 500.0)

        unregistered = self._check(tmp_path, capsys, 'unregistered')
        assert unregistered['feeds']['key'] == 'unregistered'
        assert 'not check' in unregistered['note']
        for module in ('orchestrator', 'scripts'):
            assert 'not check' not in self._check(tmp_path, capsys, module)['note'], module

    def test_an_unreadable_ruled_table_derives_nothing_rather_than_guessing(
        self, tmp_path, capsys, monkeypatch,
    ):
        """Without the ruled table the rule is unknowable, so no rule is assumed.

        A min_budget guess would print 9200 for orchestrator; a D17 guess would
        reproduce task 7929's defect for every other module.
        """
        import sys

        import verify_budget_census as mod

        no_records = tmp_path / 'no-records'
        no_records.mkdir()
        monkeypatch.setattr(mod, '_FAMILY_DIR', no_records)
        monkeypatch.setattr(sys, 'path', [
            entry for entry in sys.path
            if not (Path(entry or '.') / 'verify_budget_census_records.py').exists()
        ])
        monkeypatch.delitem(sys.modules, 'verify_budget_census_records', raising=False)
        write_corpus(tmp_path, command_entry(duration_secs=4626.166946739017))

        check = self._check(tmp_path, capsys, 'orchestrator')

        assert check['rule'] is None
        assert check['derived_floor'] is None
        assert check['feeds'] is None
        assert check['exceeds_refusal_ceiling'] is False
        assert 'unavailable' in check['note'].lower()
