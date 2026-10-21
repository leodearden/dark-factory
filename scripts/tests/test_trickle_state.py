"""Tests for scripts/legibility/trickle_state.py — the nightly trickle
run-state recorder that lets a probe answer WHY a night produced nothing.

The classifier under test is not a tuned heuristic; it is a derivation from
:class:`legibility.sampling.SampleResult`'s CONSERVATION INVARIANT::

    total_records == zero_signal_dropped + dedupe_collapsed
                     + below_sampling_cut + budget_skipped + len(selected)

Every counter fixture below therefore SATISFIES that identity — an
inconsistent fixture would prove nothing about a classifier derived from it.

The outcomes of a night that exited 0 and was not deferred by the usage cap:

- ``productive`` — ``selected > 0``; digests were built.
- ``barren``     — ``selected == 0`` and real, distinct, non-duplicate signal
  reached the sampling/budget stage and NOTHING was digested.
- ``quiet``      — everything else, which BY THE INVARIANT means every
  enumerated record left by the zero-signal or dedupe door (or nothing was
  enumerated at all). That is exactly the "genuinely quiet night" PRD
  decision 7 protects, so a quiet or dormant project can never be
  classified barren — the no-false-alarm guarantee as a PROOF from the
  invariant rather than a threshold someone picked.

A night that did not exit 0 is ``failed``; one the usage cap deferred is
``deferred``.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import os
import pwd
import tempfile
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import pytest
from legibility import trickle_state
from legibility.state_docs import Malformed, Missing, Ok
from legibility.trickle_state import (
    CensusFireStreak,
    RunCounters,
    RunOutcome,
    TrickleState,
    classify_run,
)


def _counters(
    *,
    zero_signal_dropped=0,
    dedupe_collapsed=0,
    below_sampling_cut=0,
    budget_skipped=0,
    selected_count=0,
) -> RunCounters:
    """A RunCounters whose ``total_records`` is DERIVED from the other five
    counters, so every fixture satisfies SampleResult's conservation
    identity by construction."""
    return RunCounters(
        total_records=(
            zero_signal_dropped
            + dedupe_collapsed
            + below_sampling_cut
            + budget_skipped
            + selected_count
        ),
        zero_signal_dropped=zero_signal_dropped,
        dedupe_collapsed=dedupe_collapsed,
        below_sampling_cut=below_sampling_cut,
        budget_skipped=budget_skipped,
        selected_count=selected_count,
    )


def _classify(exit_code=0, *, capped=False, **counters):
    """classify_run over a derived-total counter set, with the exit code and
    cap verdict stated explicitly (both are required keywords)."""
    return classify_run(
        exit_code=exit_code, counters=_counters(**counters), capped=capped
    )


class TestClassifyRun:
    """The absence classifier over a night that exited 0 uncapped."""

    def test_selected_is_productive(self):
        assert _classify(selected_count=3) == RunOutcome.PRODUCTIVE

    def test_partially_truncated_night_is_still_productive(self):
        """A night that digested SOMETHING and skipped the rest on budget is
        the byte budget working as designed — never barren."""
        result = _classify(selected_count=2, budget_skipped=9)
        assert result == RunOutcome.PRODUCTIVE

    def test_selected_with_every_other_door_open_is_productive(self):
        result = _classify(
            selected_count=1,
            budget_skipped=4,
            below_sampling_cut=7,
            dedupe_collapsed=2,
            zero_signal_dropped=5,
        )
        assert result == RunOutcome.PRODUCTIVE

    def test_budget_door_is_barren(self):
        """Reproduction of the real 2026-07-16..29 incident: candidates
        existed, competed, and were ALL discarded on the byte budget."""
        result = _classify(selected_count=0, budget_skipped=4)
        assert result == RunOutcome.BARREN

    def test_sampling_cut_door_is_barren(self):
        """Real, distinct signal held back by the sampling cut, nothing
        digested. Different remedy (sampling.top_fraction/per_stratum_min,
        never budgets.max_daily_digest_bytes) — see SampleResult's
        docstring."""
        result = _classify(
            selected_count=0, below_sampling_cut=3, budget_skipped=0
        )
        assert result == RunOutcome.BARREN

    def test_both_doors_open_is_barren(self):
        result = _classify(
            selected_count=0, below_sampling_cut=3, budget_skipped=4
        )
        assert result == RunOutcome.BARREN

    def test_dormant_project_is_quiet(self):
        """All counters zero — nothing was even enumerated. A dormant
        project is a legitimate state, not a degradation."""
        assert _classify() == RunOutcome.QUIET

    def test_all_zero_signal_is_quiet(self):
        result = _classify(zero_signal_dropped=17)
        assert result == RunOutcome.QUIET

    def test_zero_signal_plus_dedupe_only_is_quiet(self):
        result = _classify(
            zero_signal_dropped=6, dedupe_collapsed=3
        )
        assert result == RunOutcome.QUIET

    def test_the_outcome_vocabulary_is_a_str_enum_of_five_values(self):
        assert issubclass(RunOutcome, StrEnum)
        assert {member.value for member in RunOutcome} == {
            'productive', 'quiet', 'barren', 'failed', 'deferred',
        }


class TestQuietNightNeverBarren:
    """PRD decision 7's no-false-alarm guarantee, in executable form.

    Given its own named test rather than hiding inside a parametrize list:
    this is THE property that lets a progress probe exist at all without
    re-opening the false-alarm objection decision 7 raised against
    git-history probes.
    """

    @pytest.mark.parametrize("zero_signal_dropped", range(0, 6))
    @pytest.mark.parametrize("dedupe_collapsed", range(0, 6))
    def test_a_quiet_night_is_never_barren(
        self, zero_signal_dropped, dedupe_collapsed
    ):
        """With BOTH cut counters at 0 and nothing selected, no combination
        of zero-signal/dedupe volume may ever classify barren."""
        result = _classify(
            exit_code=0,
            zero_signal_dropped=zero_signal_dropped,
            dedupe_collapsed=dedupe_collapsed,
            below_sampling_cut=0,
            budget_skipped=0,
            selected_count=0,
        )
        assert result == RunOutcome.QUIET, (
            f"zero_signal_dropped={zero_signal_dropped} "
            f"dedupe_collapsed={dedupe_collapsed} classified {result!r}; a "
            f"night where every record left by the zero-signal or dedupe "
            f"door is exactly the 'genuinely quiet night' decision 7 "
            f"protects and must never alarm."
        )


class TestFailedOutcome:
    """A run that did not complete cannot be described by the sampler
    counters: ``selected_count > 0`` proves signal reached the digest
    stage, never that the night FINISHED. So ``exit_code != 0`` is read
    first and wins outright.
    """

    def test_failed_member_is_the_expected_string(self):
        assert RunOutcome.FAILED == 'failed'

    def test_failure_beats_productive(self):
        """The 2026-08-18 reify run selected six digests, exited 1, applied
        0 and made no commit — and recorded ``productive``."""
        assert _classify(exit_code=1, selected_count=6) == RunOutcome.FAILED

    def test_failure_beats_barren_at_the_budget_door(self):
        assert _classify(exit_code=1, budget_skipped=4) == RunOutcome.FAILED

    def test_failure_beats_barren_at_the_sampling_cut(self):
        assert _classify(exit_code=1, below_sampling_cut=3) == RunOutcome.FAILED

    def test_failure_beats_quiet(self):
        assert _classify(exit_code=1) == RunOutcome.FAILED
        assert _classify(exit_code=1, zero_signal_dropped=17) == RunOutcome.FAILED

    @pytest.mark.parametrize('exit_code', [1, 2, 127, 137, -1])
    def test_any_non_zero_code_is_failed(self, exit_code):
        """Not just 1. 127 is a missing interpreter, 137 a SIGKILL/OOM, and
        a negative code is how ``subprocess`` reports a signal — every one
        of them is a night that did not finish."""
        assert _classify(exit_code=exit_code, selected_count=6) == RunOutcome.FAILED

    def test_a_clean_exit_still_classifies_by_the_counters(self):
        assert _classify(exit_code=0) == RunOutcome.QUIET
        assert _classify(exit_code=0, selected_count=3) == RunOutcome.PRODUCTIVE
        assert _classify(exit_code=0, budget_skipped=4) == RunOutcome.BARREN

    @pytest.mark.parametrize('zero_signal_dropped', range(0, 6))
    @pytest.mark.parametrize('dedupe_collapsed', range(0, 6))
    def test_a_quiet_night_that_exited_zero_is_still_never_barren(
        self, zero_signal_dropped, dedupe_collapsed
    ):
        """``TestQuietNightNeverBarren``'s property RE-STATED with an
        explicit ``exit_code=0``, so the guarantee stays measured across the
        failed branch."""
        result = _classify(
            exit_code=0,
            zero_signal_dropped=zero_signal_dropped,
            dedupe_collapsed=dedupe_collapsed,
            below_sampling_cut=0,
            budget_skipped=0,
            selected_count=0,
        )
        assert result == RunOutcome.QUIET


class TestDeferredOutcome:
    """A night the usage cap deferred coded nothing, so its counters say
    nothing about whether signal flows or the pipeline finishes."""

    @pytest.mark.parametrize('counters', [
        pytest.param(dict(selected_count=6), id='digests-selected'),
        pytest.param(dict(budget_skipped=4), id='budget-door-open'),
        pytest.param(dict(below_sampling_cut=3), id='sampling-door-open'),
        pytest.param(dict(), id='all-zero'),
    ])
    def test_a_capped_clean_exit_is_deferred_whatever_the_counters(
        self, counters
    ):
        assert _classify(exit_code=0, capped=True, **counters) == (
            RunOutcome.DEFERRED
        )

    @pytest.mark.parametrize('exit_code', [1, 137])
    def test_a_capped_night_that_crashed_is_failed(self, exit_code):
        assert _classify(
            exit_code=exit_code, capped=True, selected_count=6
        ) == RunOutcome.FAILED

    def test_capped_is_a_required_keyword(self):
        with pytest.raises(TypeError):
            classify_run(exit_code=0, counters=_counters())  # type: ignore[call-arg]


class TestRunCounters:
    @pytest.mark.parametrize('field', [
        f.name for f in dataclasses.fields(RunCounters)
    ])
    @pytest.mark.parametrize('bad', [-1, '3', True, 2.0])
    def test_a_negative_or_non_int_counter_is_rejected_by_name(self, field, bad):
        values = dataclasses.asdict(_counters())
        values[field] = bad
        with pytest.raises(ValueError, match=field):
            RunCounters(**values)


_CENSUS_STATE = '/p/docs/legibility/census-state.json'

_T0 = datetime(2026, 7, 1, 3, 0, 0, tzinfo=UTC)


def _state(**overrides) -> TrickleState:
    """A fully populated TrickleState: every optional field set."""
    fields = dict(
        project_id='dark_factory',
        target_date=date(2026, 7, 1),
        recorded_at=_T0,
        exit_code=0,
        outcome=RunOutcome.PRODUCTIVE,
        applied=2,
        commit_made=True,
        budget_suppressed=False,
        capped=False,
        consecutive_barren_runs=0,
        consecutive_failed_runs=0,
        last_productive_at=_T0,
        last_scheduled_target_date=date(2026, 7, 1),
        census_fired=True,
        census_fire_streak=CensusFireStreak(
            fires=2,
            last_census_at='2026-06-01',
            census_state_path=_CENSUS_STATE,
            since='2026-06-30',
        ),
        counters=_counters(selected_count=2, zero_signal_dropped=4),
        retry_backlog={date(2026, 6, 29): 1},
        barren_escalated=True,
    )
    fields.update(overrides)
    return TrickleState(**fields)


def _pre_gamma_doc(**overrides) -> dict:
    """A document exactly as the first record_run wrote it: no
    consecutive_failed_runs, capped, retry_backlog,
    last_scheduled_target_date, census fields or barren_escalated."""
    doc = {
        'schema_version': 1,
        'project_id': 'dark_factory',
        'target_date': '2026-07-01',
        'recorded_at': '2026-07-01T03:00:00+00:00',
        'exit_code': 0,
        'outcome': 'barren',
        'applied': 0,
        'commit_made': False,
        'budget_suppressed': False,
        'consecutive_barren_runs': 2,
        'last_productive_at': None,
        'counters': {
            'total_records': 4,
            'zero_signal_dropped': 0,
            'dedupe_collapsed': 0,
            'below_sampling_cut': 0,
            'budget_skipped': 4,
            'selected_count': 0,
        },
    }
    doc.update(overrides)
    return doc


def _write(tmp_path: Path, doc) -> Path:
    path = tmp_path / 'trickle-state.json'
    path.write_text(json.dumps(doc))
    return path


def _warnings(caplog):
    return [r for r in caplog.records if r.levelname == 'WARNING']


class TestTrickleStateCodec:
    def test_a_fully_populated_state_round_trips(self):
        state = _state()
        assert TrickleState.from_json(state.to_json()) == state

    def test_the_round_trip_survives_json_text(self):
        state = _state(outcome=RunOutcome.DEFERRED, capped=True)
        raw = json.loads(json.dumps(state.to_json()))
        assert TrickleState.from_json(raw) == state

    def test_to_json_writes_the_recorded_key_set(self):
        assert set(_state().to_json()) == {
            'schema_version', 'project_id', 'target_date', 'recorded_at',
            'exit_code', 'outcome', 'applied', 'commit_made',
            'budget_suppressed', 'capped', 'consecutive_barren_runs',
            'consecutive_failed_runs', 'last_productive_at',
            'last_scheduled_target_date', 'census_fired',
            'census_fire_streak', 'counters', 'retry_backlog',
            'barren_escalated',
        }

    def test_to_json_uses_iso_text_and_plain_values(self):
        doc = _state().to_json()
        assert doc['schema_version'] == trickle_state.STATE_SCHEMA_VERSION
        assert doc['target_date'] == '2026-07-01'
        assert doc['recorded_at'] == '2026-07-01T03:00:00+00:00'
        assert doc['last_productive_at'] == '2026-07-01T03:00:00+00:00'
        assert doc['outcome'] == 'productive'
        assert doc['retry_backlog'] == {'2026-06-29': 1}
        assert doc['census_fire_streak'] == {
            'fires': 2,
            'last_census_at': '2026-06-01',
            'census_state_path': _CENSUS_STATE,
            'since': '2026-06-30',
        }
        assert doc['counters']['selected_count'] == 2
        assert doc['barren_escalated'] is True

    def test_the_backlog_is_read_only(self):
        state = _state()
        with pytest.raises(TypeError):
            state.retry_backlog[date(2026, 7, 2)] = 1  # type: ignore[index]

    def test_the_state_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _state().consecutive_barren_runs = 9  # type: ignore[misc]


class TestLoadState:
    """``load_state(path)`` answers Missing | Malformed(reason) | Ok(state)."""

    def test_missing_file_is_missing_and_silent(self, tmp_path, caplog):
        """A project that has never recorded a run is a NORMAL state, not a
        degradation — so it logs nothing."""
        path = tmp_path / 'nope.json'
        with caplog.at_level('WARNING'):
            read = trickle_state.load_state(path)

        assert read == Missing(path)
        assert not caplog.records

    def test_invalid_json_is_malformed_with_one_warning(self, tmp_path, caplog):
        path = tmp_path / 'trickle-state.json'
        path.write_text('{not json at all')

        with caplog.at_level('WARNING'):
            read = trickle_state.load_state(path)

        assert isinstance(read, Malformed)
        assert 'JSON' in read.reason
        assert len(_warnings(caplog)) == 1

    @pytest.mark.parametrize('doc, field', [
        pytest.param(['a', 'list'], 'object', id='non-object'),
        pytest.param(_pre_gamma_doc(schema_version=99999), 'schema_version',
                     id='unknown-schema-version'),
        pytest.param(_pre_gamma_doc(outcome='sideways'), 'outcome',
                     id='unknown-outcome'),
        pytest.param(_pre_gamma_doc(recorded_at='not-a-timestamp'),
                     'recorded_at', id='unparseable-recorded_at'),
        pytest.param(_pre_gamma_doc(consecutive_barren_runs='2'),
                     'consecutive_barren_runs', id='non-int-streak'),
        pytest.param(_pre_gamma_doc(barren_escalated='yes'),
                     'barren_escalated', id='non-bool-escalated'),
        pytest.param(_pre_gamma_doc(counters={'total_records': 1}),
                     'zero_signal_dropped', id='incomplete-counters'),
    ])
    def test_an_invalid_document_is_malformed_naming_the_field(
        self, tmp_path, caplog, doc, field
    ):
        """A reader that silently accepts a shape it does not understand is
        the silent-degradation mode this module exists to close."""
        path = _write(tmp_path, doc)

        with caplog.at_level('WARNING'):
            read = trickle_state.load_state(path)

        assert isinstance(read, Malformed)
        assert field in read.reason
        assert len(_warnings(caplog)) == 1

    def test_unreadable_path_is_malformed_not_raised(self, tmp_path, caplog):
        """A directory where a file is expected raises OSError on read; the
        reader must absorb it, never propagate."""
        path = tmp_path / 'trickle-state.json'
        path.mkdir()

        with caplog.at_level('WARNING'):
            read = trickle_state.load_state(path)

        assert isinstance(read, Malformed)

    def test_a_valid_document_is_ok(self, tmp_path):
        state = _state()
        path = _write(tmp_path, state.to_json())

        assert trickle_state.load_state(path) == Ok(state)


class TestPreGammaDocuments:
    """Every live state file predates some additive field. Such a document
    must load Ok with documented defaults: reading it malformed would be a
    guaranteed false alarm per project on the first post-deploy probe."""

    def test_the_original_writer_shape_loads_ok(self, tmp_path, caplog):
        path = _write(tmp_path, _pre_gamma_doc())

        with caplog.at_level('WARNING'):
            read = trickle_state.load_state(path)

        assert isinstance(read, Ok)
        state = read.value
        assert state.outcome == RunOutcome.BARREN
        assert state.consecutive_barren_runs == 2
        assert state.consecutive_failed_runs == 0
        assert state.capped is False
        assert dict(state.retry_backlog) == {}
        assert state.last_scheduled_target_date is None
        assert state.census_fired is None
        assert state.census_fire_streak is None
        assert state.last_productive_at is None
        assert not _warnings(caplog)

    def test_an_absent_barren_escalated_is_unrecorded_not_false(self, tmp_path):
        read = trickle_state.load_state(_write(tmp_path, _pre_gamma_doc()))
        assert isinstance(read, Ok)
        assert read.value.barren_escalated is None

    @pytest.mark.parametrize('recorded', [True, False])
    def test_a_recorded_barren_escalated_reads_as_written(
        self, tmp_path, recorded
    ):
        doc = _pre_gamma_doc(barren_escalated=recorded)
        read = trickle_state.load_state(_write(tmp_path, doc))
        assert isinstance(read, Ok)
        assert read.value.barren_escalated is recorded

    @pytest.mark.parametrize('field', [
        'consecutive_failed_runs', 'capped', 'retry_backlog',
        'last_scheduled_target_date', 'census_fired', 'census_fire_streak',
    ])
    def test_each_later_additive_field_may_be_absent_alone(
        self, tmp_path, field
    ):
        doc = _state().to_json()
        del doc[field]
        assert isinstance(trickle_state.load_state(_write(tmp_path, doc)), Ok)

    def test_schema_version_is_not_bumped_for_an_additive_field(self):
        """Pins the no-bump decision so a later tidy-up bump has to argue
        with a test."""
        assert trickle_state.STATE_SCHEMA_VERSION == 1


class TestRecordedCensusFireStreak:
    """The recorded ``census_fire_streak`` keeps its under-reporting parse:
    an unusable streak object reads as no streak, never as malformed."""

    _WELL_FORMED = {
        'fires': 2,
        'last_census_at': '2026-08-02',
        'census_state_path': _CENSUS_STATE,
        'since': '2026-07-01',
    }

    def _streak(self, raw):
        return TrickleState.from_json(
            _pre_gamma_doc(census_fire_streak=raw)
        ).census_fire_streak

    def test_a_well_formed_streak_parses_to_a_frozen_value(self):
        streak = self._streak(dict(self._WELL_FORMED))

        assert streak == CensusFireStreak(
            fires=2,
            last_census_at='2026-08-02',
            census_state_path=_CENSUS_STATE,
            since='2026-07-01',
        )
        assert streak is not None
        for field in dataclasses.fields(streak):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(streak, field.name, None)

    def test_a_never_censused_anchor_parses(self):
        streak = self._streak(dict(self._WELL_FORMED, last_census_at=None))
        assert streak is not None
        assert streak.last_census_at is None

    @pytest.mark.parametrize('raw', [
        pytest.param(None, id='null'),
        pytest.param([1, 2], id='not-a-dict'),
        pytest.param('streak', id='a-string'),
    ])
    def test_no_streak_object_parses_to_none(self, raw):
        assert self._streak(raw) is None

    @pytest.mark.parametrize('field, value', [
        pytest.param('fires', '3', id='fires-str'),
        pytest.param('fires', True, id='fires-bool'),
        pytest.param('fires', 0, id='fires-zero'),
        pytest.param('fires', -1, id='fires-negative'),
        pytest.param('census_state_path', None, id='path-null'),
        pytest.param('since', 20260701, id='since-int'),
        pytest.param('last_census_at', 20260802, id='anchor-int'),
    ])
    def test_an_unusable_field_parses_to_none(self, field, value):
        assert self._streak(dict(self._WELL_FORMED, **{field: value})) is None


class TestRecordedRetryBacklog:
    """The recorded ``retry_backlog`` keeps :func:`parse_retry_backlog`'s
    under-reporting parse inside the codec."""

    def test_invalid_entries_are_dropped_and_valid_ones_kept(self):
        raw = {
            '2026-07-01': 2,
            'not-a-date': 1,
            '2026-07-02': 0,
            '2026-07-03': -1,
            '2026-07-04': True,
        }
        state = TrickleState.from_json(_pre_gamma_doc(retry_backlog=raw))
        assert dict(state.retry_backlog) == {
            date(2026, 7, 1): 2, date(2026, 7, 2): 0,
        }

    @pytest.mark.parametrize('raw', [['2026-07-01'], '2026-07-01', 3])
    def test_a_non_mapping_backlog_reads_empty(self, raw):
        state = TrickleState.from_json(_pre_gamma_doc(retry_backlog=raw))
        assert dict(state.retry_backlog) == {}


class TestTrickleStatePath:
    """Where the state file lives — and, just as load-bearing, WHY it is
    resolved from the account rather than from the process environment.

    Rooted at ``<passwd home>/.local/state/dark-factory/legibility/
    <project_id>/trickle-state.json``. Never under ``docs/legibility/``:
    that path is git-TRACKED, and a file rewritten EVERY night on a
    tracked path would either leave the machine-operated project_root
    checkout permanently dirty or force a nightly commit — which would
    make "the repo has a commit today" a valid liveness signal and
    CONTRADICT PRD decision 7 outright.

    ``test_uses_xdg_state_home_when_set``,
    ``test_falls_back_under_home_local_state`` and
    ``test_empty_xdg_state_home_is_treated_as_unset`` were REMOVED in task
    4514. They pinned the behaviour that IS the defect: honouring two
    ambient, general-purpose variables in a file that two independently
    launched processes must agree on. The writer is
    ``legibility-trickle@<project>.service`` under the ``systemd --user``
    manager; the reader is the health timer, an orchestrator-EXEC'd
    ``before_done`` predicate, or a dev shell. Nothing pinned their
    environments to agree, so they silently read and wrote different
    files. The surviving idea from the third — set-but-empty must be
    treated as unset — is kept as
    ``test_empty_override_is_treated_as_unset``.
    """

    def test_resolves_identically_under_divergent_ambient_environments(
        self, tmp_path, monkeypatch
    ):
        """THE property the task exists for: two processes whose ambient
        environments disagree on BOTH levers still resolve one file."""
        monkeypatch.delenv(trickle_state.STATE_ROOT_ENV, raising=False)

        monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'a'))
        monkeypatch.setenv('HOME', str(tmp_path / 'home-a'))
        writer_path = trickle_state.trickle_state_path('dark_factory')

        monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'b'))
        monkeypatch.setenv('HOME', str(tmp_path / 'home-b'))
        reader_path = trickle_state.trickle_state_path('dark_factory')

        assert writer_path == reader_path
        for root in ('a', 'b', 'home-a', 'home-b'):
            assert tmp_path / root not in writer_path.parents, (
                f'{writer_path} followed the ambient {root!r} root; the two '
                f'processes would diverge again'
            )

    def test_ambient_home_alone_cannot_move_the_path(
        self, tmp_path, monkeypatch
    ):
        """``HOME`` was the second, subtler lever: with ``XDG_STATE_HOME``
        unset the old code called ``Path.home()``, which follows ``HOME``
        while the passwd entry for the same uid does not."""
        monkeypatch.delenv(trickle_state.STATE_ROOT_ENV, raising=False)
        monkeypatch.delenv('XDG_STATE_HOME', raising=False)

        monkeypatch.setenv('HOME', str(tmp_path / 'home-a'))
        first = trickle_state.trickle_state_path('dark_factory')

        monkeypatch.setenv('HOME', str(tmp_path / 'home-b'))
        second = trickle_state.trickle_state_path('dark_factory')

        assert first == second
        assert tmp_path not in first.parents

    def test_default_root_is_anchored_to_the_passwd_home_not_the_environment(
        self, monkeypatch
    ):
        """The anchor is a property of the ACCOUNT that owns the pipeline,
        read from NSS, and so is identical in every process environment for
        the same uid."""
        monkeypatch.delenv(trickle_state.STATE_ROOT_ENV, raising=False)

        result = trickle_state.trickle_state_path('dark_factory')

        expected_root = (
            Path(pwd.getpwuid(os.getuid()).pw_dir) / '.local' / 'state'
        )
        assert result == (
            expected_root / 'dark-factory' / 'legibility' / 'dark_factory'
            / 'trickle-state.json'
        )

    def test_dedicated_override_relocates_the_root(self, tmp_path, monkeypatch):
        """The ONE deliberate seam. It is safe precisely because nothing
        sets it incidentally: a shell, a systemd user manager, a container
        and CI all set ``XDG_STATE_HOME``/``HOME`` for reasons unrelated to
        legibility, but this name is only ever set by someone who means
        this file."""
        assert trickle_state.STATE_ROOT_ENV == 'DARK_FACTORY_LEGIBILITY_STATE_ROOT'
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

        result = trickle_state.trickle_state_path('dark_factory')

        assert result == (
            tmp_path / 'dark-factory' / 'legibility' / 'dark_factory'
            / 'trickle-state.json'
        )

    def test_empty_override_is_treated_as_unset(self, monkeypatch):
        """The classic systemd ``Environment=`` foot-gun: set-but-empty must
        fall back to the anchored default, never resolve to the filesystem
        root."""
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, '')

        result = trickle_state.trickle_state_path('dark_factory')

        expected_root = (
            Path(pwd.getpwuid(os.getuid()).pw_dir) / '.local' / 'state'
        )
        assert result == (
            expected_root / 'dark-factory' / 'legibility' / 'dark_factory'
            / 'trickle-state.json'
        )

    def test_distinct_projects_never_collide(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))
        a = trickle_state.trickle_state_path('dark_factory')
        b = trickle_state.trickle_state_path('reify')
        assert a != b
        assert a.parent != b.parent

    def test_no_passwd_entry_degrades_to_tempdir_instead_of_raising(
        self, monkeypatch, caplog
    ):
        """The trigger is no longer ``Path.home()`` raising — it is the
        passwd lookup failing, which now means a genuinely absent passwd
        entry (an arbitrary-uid container), not merely an unset ``HOME``.

        This is the REPAIR of the finding's "worse variant". A stripped
        systemd environment with no ``HOME`` used to land here: the old
        code wrote to ``tempfile.gettempdir()`` — a path the nightly never
        reads — while logging to a logger the bare-``python3`` predicate
        never configures, so the divergence was invisible. That case now
        resolves correctly. The degradation survives only so this helper
        keeps its never-raise property on the nightly's unconditional
        path."""
        monkeypatch.delenv(trickle_state.STATE_ROOT_ENV, raising=False)

        def _no_passwd_entry(uid):
            raise KeyError(f'getpwuid(): uid not found: {uid}')

        monkeypatch.setattr(pwd, 'getpwuid', _no_passwd_entry)

        with caplog.at_level('WARNING'):
            result = trickle_state.trickle_state_path('dark_factory')

        assert result == (
            Path(tempfile.gettempdir()) / 'dark-factory' / 'legibility'
            / 'dark_factory' / 'trickle-state.json'
        )
        assert len([r for r in caplog.records if r.levelname == 'WARNING']) == 1, (
            'the temp-dir degradation must be announced exactly once, not '
            'silent and not repeated'
        )

    def test_state_path_is_outside_any_repo_checkout(self, monkeypatch):
        """The location property that motivated the whole choice: the state
        file must never dirty a machine-operated checkout, and must never
        become a git signal."""
        monkeypatch.delenv(trickle_state.STATE_ROOT_ENV, raising=False)

        result = trickle_state.trickle_state_path('dark_factory')

        worktree_root = Path(__file__).resolve().parent.parent.parent
        parents = list(result.parents)
        assert worktree_root not in parents, (
            f'{result} lives inside the checkout at {worktree_root}; a file '
            f'rewritten every night must not dirty a machine-operated tree'
        )
        parts = result.parts
        assert not any(
            parts[i] == 'docs' and parts[i + 1] == 'legibility'
            for i in range(len(parts) - 1)
        ), f'{result} must not live under the git-tracked docs/legibility/'


def _night(*, day=1, hour=3, exit_code=0, applied=0, commit_made=False,
           budget_suppressed=False, capped=False, scheduled=False,
           census: trickle_state.CensusObservation | None = None,
           **counters) -> trickle_state.NightRun:
    """One night's facts with a derived-total counter set and an injected
    clock (the module is deliberately clock-injectable — no time faking)."""
    return trickle_state.NightRun(
        target_date=date(2026, 7, day),
        recorded_at=datetime(2026, 7, day, hour, 0, 0, tzinfo=UTC),
        exit_code=exit_code,
        counters=_counters(**counters),
        census=census,
        applied=applied,
        commit_made=commit_made,
        budget_suppressed=budget_suppressed,
        capped=capped,
        scheduled=scheduled,
    )


def _record(project_id='dark_factory', *, barren_escalated=False,
            **night) -> TrickleState:
    return trickle_state.record_run(
        project_id, _night(**night), barren_escalated=barren_escalated
    )


def _path(project_id='dark_factory') -> Path:
    return trickle_state.trickle_state_path(project_id)


class TestNightRun:
    def test_the_outcome_is_classify_runs_verdict(self):
        assert _night(selected_count=1).outcome == RunOutcome.PRODUCTIVE
        assert _night(budget_skipped=4).outcome == RunOutcome.BARREN
        assert _night(exit_code=1, selected_count=6).outcome == RunOutcome.FAILED
        assert _night(capped=True, selected_count=6).outcome == RunOutcome.DEFERRED
        assert _night(zero_signal_dropped=3).outcome == RunOutcome.QUIET

    def test_census_is_a_required_keyword(self):
        """A defaulted ``census=None`` is how a future caller would silently
        blind the census progress probe into a permanent OK."""
        census = inspect.signature(trickle_state.NightRun).parameters['census']
        assert census.kind is inspect.Parameter.KEYWORD_ONLY
        assert census.default is inspect.Parameter.empty


class TestRecordRun:
    """The writer: streak arithmetic, carried-forward fields, atomicity."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_first_call_creates_parents_and_writes_every_field(self):
        state = _record(day=1, selected_count=2, zero_signal_dropped=4,
                        applied=2, commit_made=True)

        path = _path()
        assert path.is_file(), 'record_run must create the parent directories'
        assert json.loads(path.read_text()) == state.to_json()

        assert state.project_id == 'dark_factory'
        assert state.target_date == date(2026, 7, 1)
        assert state.recorded_at == datetime(2026, 7, 1, 3, tzinfo=UTC)
        assert state.exit_code == 0
        assert state.outcome == RunOutcome.PRODUCTIVE
        assert state.applied == 2
        assert state.commit_made is True
        assert state.budget_suppressed is False
        assert state.consecutive_barren_runs == 0
        assert state.consecutive_failed_runs == 0
        assert state.last_scheduled_target_date is None
        assert state.counters == RunCounters(
            total_records=6,
            zero_signal_dropped=4,
            dedupe_collapsed=0,
            below_sampling_cut=0,
            budget_skipped=0,
            selected_count=2,
        )

    def test_recorded_at_is_timezone_aware_utc_iso8601(self):
        _record(day=2, selected_count=1)
        parsed = datetime.fromisoformat(
            json.loads(_path().read_text())['recorded_at']
        )
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)

    def test_barren_streak_increments(self):
        assert _record(day=1, budget_skipped=4).consecutive_barren_runs == 1
        assert _record(day=2, budget_skipped=4).consecutive_barren_runs == 2
        assert _record(day=3, below_sampling_cut=3).consecutive_barren_runs == 3

    def test_productive_run_resets_the_streak(self):
        _record(day=1, budget_skipped=4)
        _record(day=2, budget_skipped=4)
        state = _record(day=3, selected_count=1)
        assert state.outcome == RunOutcome.PRODUCTIVE
        assert state.consecutive_barren_runs == 0

    def test_quiet_run_also_resets_the_streak(self):
        """A legitimately quiet night is NOT evidence of breakage — PRD
        decision 7's guarantee expressed in the streak itself."""
        _record(day=1, budget_skipped=4)
        _record(day=2, budget_skipped=4)
        state = _record(day=3, zero_signal_dropped=5)
        assert state.outcome == RunOutcome.QUIET
        assert state.consecutive_barren_runs == 0

    def test_streak_rearms_after_a_reset(self):
        _record(day=1, budget_skipped=4)
        _record(day=2, selected_count=1)
        assert _record(day=3, budget_skipped=4).consecutive_barren_runs == 1

    def test_last_productive_at_is_stamped_and_carried_forward(self):
        """The field an operator reads to answer 'when did this pipeline
        last actually do anything'."""
        first = _record(day=1, zero_signal_dropped=3)
        assert first.last_productive_at is None

        prod = _record(day=2, selected_count=1)
        assert prod.last_productive_at == prod.recorded_at
        stamp = prod.last_productive_at

        assert _record(day=3, budget_skipped=4).last_productive_at == stamp
        assert _record(day=4, zero_signal_dropped=1).last_productive_at == stamp
        assert _record(day=5, budget_skipped=4).last_productive_at == stamp

        newer = _record(day=6, selected_count=2)
        assert newer.last_productive_at == newer.recorded_at != stamp

    def test_the_write_leaves_no_temp_residue(self):
        _record(day=1, selected_count=1)
        _record(day=2, selected_count=1)
        assert [p.name for p in _path().parent.iterdir()] == [
            trickle_state.STATE_FILENAME
        ]

    def test_a_fixed_name_tmp_sibling_neither_blocks_nor_is_consumed(self):
        """The old writer staged every write through one fixed
        ``<name>.tmp``, so two writers raced on it. The write now never
        touches that name."""
        _record(day=1, selected_count=1)
        sentinel = _path().parent / (trickle_state.STATE_FILENAME + '.tmp')
        sentinel.write_text('another writer is mid-write')

        state = _record(day=2, budget_skipped=4)

        assert sentinel.read_text() == 'another writer is mid-write'
        assert trickle_state.load_state(_path()) == Ok(state)

    def test_corrupt_predecessor_degrades_to_under_reporting(self, caplog):
        """Losing streak history must degrade to UNDER-reporting, never to a
        crash inside the nightly run."""
        _record(day=1, budget_skipped=4)
        _record(day=2, budget_skipped=4)
        _path().write_text('{corrupt')

        with caplog.at_level('WARNING'):
            state = _record(day=3, budget_skipped=4)

        assert state.outcome == RunOutcome.BARREN
        assert state.consecutive_barren_runs == 1, (
            'a lost predecessor restarts the streak from this run alone'
        )
        assert state.last_productive_at is None
        assert any(r.levelname == 'WARNING' for r in caplog.records)

    def test_round_trips_through_load_state(self):
        state = _record(day=1, budget_skipped=4, budget_suppressed=True)
        assert trickle_state.load_state(_path()) == Ok(state)

    def test_distinct_projects_keep_independent_streaks(self):
        _record('dark_factory', day=1, budget_skipped=4)
        _record('dark_factory', day=2, budget_skipped=4)
        other = _record('reify', day=2, budget_skipped=4)
        assert other.consecutive_barren_runs == 1


class TestRecordRunBarrenEscalated:
    """Whether the nightly escalated this night is recorded as a fact."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    @pytest.mark.parametrize('escalated', [True, False])
    def test_the_flag_is_recorded_as_passed_and_round_trips(self, escalated):
        state = _record(day=1, budget_skipped=4, barren_escalated=escalated)

        assert state.barren_escalated is escalated
        read = trickle_state.load_state(_path())
        assert isinstance(read, Ok)
        assert read.value.barren_escalated is escalated

    def test_the_flag_has_no_default(self):
        parameter = inspect.signature(trickle_state.record_run).parameters[
            'barren_escalated'
        ]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
        with pytest.raises(TypeError):
            trickle_state.record_run('dark_factory', _night())  # type: ignore[call-arg]


class TestNextState:
    """next_state(prior, project_id, run) is the pure fold record_run
    writes, so the nightly can read tonight's state before recording it."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_it_is_the_state_record_run_then_writes(self):
        _record(day=1, budget_skipped=4)
        _record(day=2, budget_skipped=4)
        run = _night(day=3, budget_skipped=4, scheduled=True)

        projected = trickle_state.next_state(
            trickle_state.read_prior('dark_factory'), 'dark_factory', run
        )
        recorded = trickle_state.record_run(
            'dark_factory', run, barren_escalated=True
        )

        assert recorded == dataclasses.replace(projected, barren_escalated=True)
        assert projected.consecutive_barren_runs == 3

    def test_it_touches_no_file(self, tmp_path):
        _record(day=1, budget_skipped=4)
        before = {p.name: p.read_bytes() for p in _path().parent.iterdir()}

        trickle_state.next_state(
            trickle_state.read_prior('dark_factory'),
            'dark_factory',
            _night(day=2, exit_code=1),
        )

        assert {p.name: p.read_bytes() for p in _path().parent.iterdir()} == before

    def test_from_no_prior_every_streak_starts_at_this_run(self):
        barren = trickle_state.next_state(None, 'p', _night(budget_skipped=4))
        failed = trickle_state.next_state(None, 'p', _night(exit_code=1))

        assert (barren.consecutive_barren_runs, barren.consecutive_failed_runs) == (1, 0)
        assert (failed.consecutive_barren_runs, failed.consecutive_failed_runs) == (0, 1)
        assert barren.project_id == 'p'
        assert barren.last_productive_at is None
        assert barren.census_fire_streak is None
        assert not _path('p').exists()


class TestReadPrior:
    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_a_readable_document_is_the_prior(self):
        state = _record(day=1, selected_count=1)
        assert trickle_state.read_prior('dark_factory') == state

    def test_a_missing_document_is_no_prior(self):
        assert trickle_state.read_prior('dark_factory') is None

    def test_a_malformed_document_is_no_prior(self):
        _record(day=1, selected_count=1)
        _path().write_text('{corrupt')
        assert trickle_state.read_prior('dark_factory') is None


class TestBarrenStreakDue:
    """The one barren-threshold predicate the nightly and the progress probe
    share."""

    @pytest.mark.parametrize('streak, threshold, due', [
        (2, 3, False), (3, 3, True), (4, 3, True), (4, 5, False), (5, 5, True),
    ])
    def test_due_at_or_past_the_threshold(self, streak, threshold, due):
        state = _state(consecutive_barren_runs=streak)
        assert trickle_state.barren_streak_due(state, threshold) is due


class TestBarrenRemedy:
    """Each barren door has ONE remedy, spelled once."""

    def test_every_door_has_exactly_one_remedy(self):
        assert set(trickle_state.BARREN_REMEDIES) == set(trickle_state.BarrenDoor)
        assert {door.value for door in trickle_state.BarrenDoor} == {
            'budget_skipped', 'below_sampling_cut',
        }

    def test_the_remedies_are_read_only(self):
        with pytest.raises(TypeError):
            trickle_state.BARREN_REMEDIES[trickle_state.BarrenDoor.BUDGET] = 'x'  # type: ignore[index]

    def test_the_budget_door_names_the_budget_key_only(self):
        text = trickle_state.barren_remedy(_counters(budget_skipped=4))
        assert text == trickle_state.BARREN_REMEDIES[trickle_state.BarrenDoor.BUDGET]
        assert 'budgets.max_daily_digest_bytes' in text
        assert 'sampling.top_fraction' not in text

    def test_the_sampling_cut_door_names_the_sampling_keys_only(self):
        text = trickle_state.barren_remedy(_counters(below_sampling_cut=3))
        assert text == trickle_state.BARREN_REMEDIES[trickle_state.BarrenDoor.SAMPLING_CUT]
        assert 'sampling.top_fraction' in text
        assert 'sampling.per_stratum_min' in text
        assert 'budgets.max_daily_digest_bytes' not in text

    def test_both_doors_name_both_budget_first(self):
        text = trickle_state.barren_remedy(
            _counters(budget_skipped=4, below_sampling_cut=3)
        )
        assert text.index('budgets.max_daily_digest_bytes') < text.index(
            'sampling.top_fraction'
        )

    def test_no_open_door_falls_back_to_the_counters(self):
        assert 'inspect the recorded counters' in trickle_state.barren_remedy(
            _counters(zero_signal_dropped=2)
        )


class TestRecordRunFailedStreak:
    """The parallel ``failed`` streak, and what a failed run must NOT do to
    the barren one."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_a_failed_run_records_the_real_counters(self):
        """Nothing is hidden by classifying ``failed``: the counters still
        say signal reached the digest stage, which is the question they
        answer."""
        state = _record(day=1, exit_code=1, selected_count=6)

        assert state.outcome == RunOutcome.FAILED
        assert state.exit_code == 1
        assert state.counters.selected_count == 6

    def test_a_failed_run_never_restamps_last_productive_at(self):
        """An operator reading ``last_productive_at`` must see the last
        night that actually did something, not the last night that merely
        started."""
        prod = _record(day=1, selected_count=2)
        stamp = prod.last_productive_at
        assert stamp == prod.recorded_at

        assert _record(day=2, exit_code=1, selected_count=6).last_productive_at == stamp
        assert _record(day=3, exit_code=1, selected_count=6).last_productive_at == stamp

    def test_last_productive_at_stays_none_when_the_first_run_fails(self):
        assert _record(day=1, exit_code=1, selected_count=6).last_productive_at is None
        assert _record(day=2, exit_code=1, selected_count=6).last_productive_at is None

    def test_failed_streak_increments(self):
        assert _record(day=1, exit_code=1).consecutive_failed_runs == 1
        assert _record(day=2, exit_code=2).consecutive_failed_runs == 2
        assert _record(day=3, exit_code=137).consecutive_failed_runs == 3

    @pytest.mark.parametrize('resetter', [
        pytest.param(dict(selected_count=1), id='productive'),
        pytest.param(dict(zero_signal_dropped=5), id='quiet'),
        pytest.param(dict(budget_skipped=4), id='barren'),
    ])
    def test_any_completed_run_resets_the_failed_streak(self, resetter):
        _record(day=1, exit_code=1)
        _record(day=2, exit_code=1)

        state = _record(day=3, **resetter)
        assert state.outcome != RunOutcome.FAILED
        assert state.consecutive_failed_runs == 0

    def test_failed_streak_rearms_after_a_reset(self):
        _record(day=1, exit_code=1)
        _record(day=2, selected_count=1)
        assert _record(day=3, exit_code=1).consecutive_failed_runs == 1

    def test_a_failed_run_carries_the_barren_streak_forward(self):
        """CARRY FORWARD, never reset and never increment: a crash is no
        evidence that signal started flowing again, and its counters
        describe an unfinished night."""
        assert _record(day=1, budget_skipped=4).consecutive_barren_runs == 1
        assert _record(day=2, budget_skipped=4).consecutive_barren_runs == 2

        crashed = _record(day=3, exit_code=1, selected_count=6)
        assert crashed.outcome == RunOutcome.FAILED
        assert crashed.consecutive_barren_runs == 2

        assert _record(day=4, budget_skipped=4).consecutive_barren_runs == 3

    def test_a_pre_gamma_document_still_folds(self, caplog):
        """Backward compatibility with the document every live state file
        held on the first post-deploy run."""
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_pre_gamma_doc(
            outcome='productive',
            consecutive_barren_runs=0,
            last_productive_at='2026-07-01T03:00:00+00:00',
        )))

        with caplog.at_level('WARNING'):
            state = _record(day=2, exit_code=1, selected_count=6)

        assert not _warnings(caplog)
        assert state.consecutive_failed_runs == 1
        assert state.last_productive_at == _T0

    def test_failed_threshold_is_tighter_than_the_barren_one(self):
        """ONE failed night is already owned twice over — by the nightly's
        own per-run escalation and by ``check_trickle_liveness.sh``'s
        ``Result != success`` gate. TWO consecutive is the PERSISTENT shape
        neither per-run signal can express."""
        assert trickle_state.DEFAULT_MAX_FAILED_RUNS == 2
        assert (
            trickle_state.DEFAULT_MAX_FAILED_RUNS
            < trickle_state.DEFAULT_MAX_BARREN_RUNS
        )


class TestRecordRunDeferred:
    """A night the usage cap deferred observed nothing, so it moves neither
    streak and owes its date a retry."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_productive_barren_then_capped_records_deferred(self):
        """Boundary row 5, at the writer."""
        prod = _record(day=1, selected_count=2, census=_fire())
        _record(day=2, budget_skipped=4, census=_fire())

        capped = _record(day=3, capped=True, selected_count=6, census=None)

        assert capped.exit_code == 0
        assert capped.outcome == RunOutcome.DEFERRED
        assert capped.last_productive_at == prod.last_productive_at
        assert capped.consecutive_barren_runs == 1
        assert capped.consecutive_failed_runs == 0
        assert capped.census_fired is None
        assert capped.census_fire_streak is not None
        assert capped.census_fire_streak.fires == 2
        assert dict(capped.retry_backlog) == {_d(3): 1}

    def test_a_deferred_night_carries_the_failed_streak_too(self):
        _record(day=1, exit_code=1)
        assert _record(day=2, capped=True).consecutive_failed_runs == 1
        assert _record(day=3, exit_code=1).consecutive_failed_runs == 2

    def test_a_deferred_night_is_never_barren_however_its_doors_read(self):
        _record(day=1, budget_skipped=4)
        _record(day=2, budget_skipped=4)
        deferred = _record(day=3, capped=True, budget_skipped=4)
        assert deferred.outcome == RunOutcome.DEFERRED
        assert deferred.consecutive_barren_runs == 2
        assert _record(day=4, budget_skipped=4).consecutive_barren_runs == 3


def _fire(anchor: str | None = '2026-08-02', path=_CENSUS_STATE):
    return trickle_state.CensusObservation(
        fired=True, last_census_at=anchor, census_state_path=path,
    )


def _no_fire(anchor: str | None = '2026-08-02', path=_CENSUS_STATE):
    return trickle_state.CensusObservation(
        fired=False, last_census_at=anchor, census_state_path=path,
    )


class TestRecordRunCensusFireStreak:
    """The census-fire streak: how many consecutive recorded nights the
    census trigger FIRED against one unchanged ``last_census_at``.

    ``census=None`` means the trigger was not evaluated that night (the
    trickle failed or was capped before the census step)."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_consecutive_fires_against_one_anchor_increment(self):
        for day, fires in ((1, 1), (2, 2), (3, 3)):
            state = _record(day=day, zero_signal_dropped=1, census=_fire())
            assert state.census_fired is True
            assert state.census_fire_streak == CensusFireStreak(
                fires=fires,
                last_census_at='2026-08-02',
                census_state_path=_CENSUS_STATE,
                since='2026-07-01',
            )

    def test_a_no_fire_night_clears_the_streak(self):
        _record(day=1, census=_fire())
        _record(day=2, census=_fire())

        state = _record(day=3, census=_no_fire())

        assert state.census_fired is False
        assert state.census_fire_streak is None

    def test_quiet_no_fire_nights_never_build_a_streak(self):
        for day in range(1, 7):
            state = _record(day=day, census=_no_fire())
        assert state.census_fire_streak is None

    def test_a_night_that_did_not_evaluate_the_trigger_carries_the_streak(self):
        _record(day=1, census=_fire())
        before = _record(day=2, census=_fire())

        skipped = _record(day=3, exit_code=1, census=None)

        assert skipped.census_fired is None
        assert skipped.census_fire_streak == before.census_fire_streak
        resumed = _record(day=4, census=_fire())
        assert resumed.census_fire_streak is not None
        assert resumed.census_fire_streak.fires == 3
        assert resumed.census_fire_streak.since == '2026-07-01'

    def test_a_first_night_without_evaluation_records_no_streak(self):
        state = _record(day=1, census=None)
        assert state.census_fired is None
        assert state.census_fire_streak is None

    def test_a_fire_against_a_moved_anchor_restarts_the_streak(self):
        _record(day=1, census=_fire(anchor='2026-08-02'))
        _record(day=2, census=_fire(anchor='2026-08-02'))

        state = _record(day=3, census=_fire(anchor='2026-09-24'))

        assert state.census_fire_streak == CensusFireStreak(
            fires=1,
            last_census_at='2026-09-24',
            census_state_path=_CENSUS_STATE,
            since='2026-07-03',
        )

    def test_a_fire_against_another_census_state_path_restarts_the_streak(self):
        _record(day=1, census=_fire())
        _record(day=2, census=_fire())

        state = _record(day=3, census=_fire(path='/q/census-state.json'))

        assert state.census_fire_streak == CensusFireStreak(
            fires=1,
            last_census_at='2026-08-02',
            census_state_path='/q/census-state.json',
            since='2026-07-03',
        )

    def test_a_never_censused_anchor_still_builds_a_streak(self):
        _record(day=1, census=_fire(anchor=None))
        state = _record(day=2, census=_fire(anchor=None))

        assert state.census_fire_streak is not None
        assert state.census_fire_streak.fires == 2
        assert state.census_fire_streak.last_census_at is None

    def test_a_predecessor_without_census_fields_starts_the_streak_at_one(
        self, caplog,
    ):
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_pre_gamma_doc()))

        with caplog.at_level('WARNING'):
            state = _record(day=2, census=_fire())

        assert not _warnings(caplog)
        assert state.census_fire_streak is not None
        assert state.census_fire_streak.fires == 1

    def test_the_census_fields_round_trip_through_load_state(self):
        state = _record(day=1, census=_fire())
        assert trickle_state.load_state(_path()) == Ok(state)


class TestReadLastCensusAt:
    """``read_last_census_at(path)``: the ONE quiet read of census-state's
    raw ``last_census_at``, shared by the nightly's pre-launch anchor and
    the census progress probe's live comparison."""

    def test_a_missing_file_is_missing(self, tmp_path):
        assert trickle_state.read_last_census_at(
            tmp_path / 'census-state.json'
        ) == ('missing', None)

    @pytest.mark.parametrize('content, expected', [
        pytest.param('{"last_census_at": "2026-08-02"}', '2026-08-02', id='dated'),
        pytest.param('{}', None, id='never-censused'),
        pytest.param('{"last_census_at": 20260802}', 20260802, id='raw-not-validated'),
    ])
    def test_a_json_object_yields_its_raw_value(self, tmp_path, content, expected):
        path = tmp_path / 'census-state.json'
        path.write_text(content, encoding='utf-8')
        assert trickle_state.read_last_census_at(path) == ('ok', expected)

    @pytest.mark.parametrize('content', [
        pytest.param('{not json', id='invalid-json'),
        pytest.param('[1, 2]', id='not-an-object'),
    ])
    def test_an_unusable_file_is_malformed_and_quiet(self, tmp_path, caplog, content):
        """Quiet because census_trigger.load_census_state already logs the
        malformed WARNING for the same file on the same night."""
        path = tmp_path / 'census-state.json'
        path.write_text(content, encoding='utf-8')
        with caplog.at_level('DEBUG'):
            assert trickle_state.read_last_census_at(path) == ('malformed', None)
        assert caplog.records == []


class TestRecordedAgeHours:
    """``recorded_age_hours(state)`` — the freshness reading both probes
    share, so they cannot drift into two readings of "fresh"."""

    def test_a_state_stamped_n_hours_ago_reports_approximately_n(self):
        stamp = datetime.now(UTC) - timedelta(hours=30)
        age = trickle_state.recorded_age_hours(_state(recorded_at=stamp))

        # A tolerance, not equality: the wall clock advances between the
        # stamp above and the read inside the helper.
        assert abs(age - 30) < 0.5

    def test_an_injected_clock_is_used_exactly(self):
        now = _T0 + timedelta(hours=7)
        assert trickle_state.recorded_age_hours(_state(), now=now) == 7.0

    def test_a_naive_recorded_at_on_disk_is_read_as_utc(self, tmp_path):
        naive = (datetime.now(UTC) - timedelta(hours=10)).replace(tzinfo=None)
        doc = _pre_gamma_doc(recorded_at=naive.isoformat())

        read = trickle_state.load_state(_write(tmp_path, doc))

        assert isinstance(read, Ok)
        assert abs(trickle_state.recorded_age_hours(read.value) - 10) < 0.5


def _d(day):
    return date(2026, 7, day)


def _fold(backlog, *, target_date, owes_retry, today, never_run=()):
    return trickle_state.fold_retry_backlog(
        backlog, target_date=target_date, owes_retry=owes_retry, today=today,
        never_run=never_run,
    )


def _simulate_outage(outage_nights, *, healthy_nights):
    """Drive the backlog policy over consecutive nights with no I/O.

    Night ``n`` covers target date ``first + n`` and is recorded the next
    day. The first ``outage_nights`` scheduled passes are unfinished; each
    healthy night folds its finished scheduled pass, selects what is due,
    and folds a finished retry of each selected date. Returns the outage
    dates, the dates retried in order, every abandoned entry, and the
    final backlog."""
    first = _d(1)
    backlog = {}
    retried = []
    abandoned = {}
    for night in range(outage_nights + healthy_nights):
        target = first + timedelta(days=night)
        today = target + timedelta(days=1)
        healthy = night >= outage_nights
        fold = _fold(backlog, target_date=target, owes_retry=not healthy,
                     today=today)
        backlog = fold.backlog
        abandoned.update(fold.abandoned)
        if not healthy:
            continue
        for due in trickle_state.select_due_retries(backlog, today=today):
            fold = _fold(backlog, target_date=due, owes_retry=False,
                         today=today)
            backlog = fold.backlog
            abandoned.update(fold.abandoned)
            retried.append(due)
    outage = [first + timedelta(days=n) for n in range(outage_nights)]
    return outage, retried, abandoned, backlog


def _simulate_missed_elapses(missed, *, nights, failed_nights=()):
    """Drive the backlog policy over a systemd outage with no I/O.

    The last scheduled pass covered ``_d(1)``; the timer then misses
    *missed* elapses and its catch-up covers ``_d(1 + missed)``. Each of
    *nights* scheduled passes, from the catch-up on, covers the next date,
    is recorded the next day, and owes the dates since the previous
    scheduled pass. A night whose index is in *failed_nights* is unfinished
    and retries nothing; every other night folds a finished retry of each
    date due. Returns the dates retried in order, every abandoned entry,
    and the final backlog."""
    cursor = _d(1)
    catch_up = cursor + timedelta(days=missed)
    backlog = {}
    retried = []
    abandoned = {}
    for night in range(nights):
        target = catch_up + timedelta(days=night)
        today = target + timedelta(days=1)
        failed = night in failed_nights
        fold = _fold(
            backlog, target_date=target, owes_retry=failed, today=today,
            never_run=trickle_state.never_run_dates(
                cursor, target_date=target, today=today).owed,
        )
        cursor = target
        backlog = fold.backlog
        abandoned.update(fold.abandoned)
        if failed:
            continue
        for due in trickle_state.select_due_retries(backlog, today=today):
            fold = _fold(backlog, target_date=due, owes_retry=False,
                         today=today)
            backlog = fold.backlog
            abandoned.update(fold.abandoned)
            retried.append(due)
    return retried, abandoned, backlog


class TestRetryBacklogPolicy:
    """The retry-backlog policy as pure functions: no filesystem, no
    clock. A date is owed a retry when its pass did not finish and
    committed nothing; the backlog is bounded by age and by attempts."""

    @pytest.mark.parametrize('exit_code, capped, expected', [
        pytest.param(0, False, True, id='clean'),
        pytest.param(1, False, False, id='failed'),
        pytest.param(137, False, False, id='killed'),
        pytest.param(-1, False, False, id='signalled'),
        pytest.param(0, True, False, id='capped'),
        pytest.param(1, True, False, id='failed-and-capped'),
    ])
    def test_finished_means_a_clean_exit_that_was_not_capped(
        self, exit_code, capped, expected
    ):
        assert trickle_state.finished(
            exit_code=exit_code, capped=capped) is expected

    @pytest.mark.parametrize('exit_code, capped, expected', [
        pytest.param(0, False, False, id='finished'),
        pytest.param(1, False, True, id='failed'),
        pytest.param(0, True, True, id='capped'),
    ])
    def test_an_unfinished_pass_that_committed_nothing_owes_a_retry(
        self, exit_code, capped, expected
    ):
        assert trickle_state.owes_retry(
            exit_code=exit_code, capped=capped, commit_made=False
        ) is expected

    @pytest.mark.parametrize('exit_code, capped', [
        pytest.param(1, False, id='failed'),
        pytest.param(0, True, id='capped'),
        pytest.param(0, False, id='finished'),
    ])
    def test_a_pass_that_committed_never_owes_a_retry(self, exit_code, capped):
        assert trickle_state.owes_retry(
            exit_code=exit_code, capped=capped, commit_made=True
        ) is False

    def test_an_unfinished_pass_adds_its_date_with_one_attempt(self):
        fold = _fold({}, target_date=_d(1), owes_retry=True, today=_d(2))

        assert fold.backlog == {_d(1): 1}
        assert fold.abandoned == {}

    def test_an_unfinished_pass_on_an_owed_date_increments_it(self):
        fold = _fold({_d(1): 1}, target_date=_d(1), owes_retry=True,
                     today=_d(3))

        assert fold.backlog == {_d(1): 2}

    def test_a_finished_pass_removes_its_own_date_and_carries_the_rest(self):
        backlog = {_d(1): 1, _d(2): 2}

        own = _fold(backlog, target_date=_d(1), owes_retry=False, today=_d(4))
        other = _fold(backlog, target_date=_d(3), owes_retry=False,
                      today=_d(4))

        assert own.backlog == {_d(2): 2}
        assert other.backlog == backlog
        assert own.abandoned == other.abandoned == {}

    def test_a_fold_never_mutates_its_input(self):
        backlog = {_d(1): 1, _d(2): 1}
        snapshot = dict(backlog)

        _fold(backlog, target_date=_d(1), owes_retry=True, today=_d(3))
        _fold(backlog, target_date=_d(2), owes_retry=False, today=_d(3))
        _fold(backlog, target_date=_d(1), owes_retry=False, today=_d(30))

        assert backlog == snapshot

    @pytest.mark.parametrize('last_scheduled, target_date', [
        pytest.param(None, _d(4), id='no-cursor'),
        pytest.param(_d(1), _d(2), id='consecutive'),
        pytest.param(_d(3), _d(3), id='same-date'),
        pytest.param(_d(5), _d(3), id='backwards'),
    ])
    def test_no_date_is_never_run_without_a_gap(
        self, last_scheduled, target_date
    ):
        assert trickle_state.never_run_dates(
            last_scheduled, target_date=target_date, today=_d(6),
        ) == trickle_state.NeverRun(owed=(), expired=0)

    def test_the_never_run_dates_are_strictly_inside_the_gap_oldest_first(
        self,
    ):
        assert trickle_state.never_run_dates(
            _d(1), target_date=_d(4), today=_d(5),
        ) == trickle_state.NeverRun(owed=(_d(2), _d(3)), expired=0)

    def test_a_gap_far_past_the_window_lists_only_its_in_window_dates(self):
        today = _d(21)
        oldest = today - timedelta(days=trickle_state.RETRY_MAX_AGE_DAYS)
        hand_edited = date(2020, 1, 1)

        gap = trickle_state.never_run_dates(
            hand_edited, target_date=_d(20), today=today)

        assert gap.owed == tuple(
            oldest + timedelta(days=n) for n in range((_d(20) - oldest).days))
        assert gap.expired == (oldest - date(2020, 1, 2)).days

    def test_a_gap_wholly_past_the_window_lists_nothing(self):
        gap = trickle_state.never_run_dates(
            _d(1), target_date=_d(5), today=_d(30))

        assert gap == trickle_state.NeverRun(owed=(), expired=3)

    def test_a_never_run_date_enters_with_zero_attempts(self):
        fold = _fold({}, target_date=_d(4), owes_retry=False, today=_d(5),
                     never_run=(_d(2), _d(3)))

        assert fold.backlog == {_d(2): 0, _d(3): 0}
        assert fold.abandoned == {}

    def test_an_owed_never_run_date_keeps_its_attempt_count(self):
        fold = _fold({_d(2): 1}, target_date=_d(3), owes_retry=False,
                     today=_d(4), never_run=(_d(2),))

        assert fold.backlog == {_d(2): 1}

    def test_a_never_run_date_older_than_the_window_is_abandoned(self):
        today = _d(20)
        max_age = trickle_state.RETRY_MAX_AGE_DAYS
        too_old = today - timedelta(days=max_age + 1)
        at_the_edge = today - timedelta(days=max_age)

        backlog, abandoned = _fold(
            {}, target_date=_d(19), owes_retry=False, today=today,
            never_run=(too_old, at_the_edge),
        )

        assert backlog == {at_the_edge: 0}
        assert abandoned == {too_old: 0}

    def test_a_fold_never_mutates_its_never_run_dates(self):
        backlog = {_d(2): 1}
        never_run = [_d(2), _d(3)]

        _fold(backlog, target_date=_d(4), owes_retry=True, today=_d(5),
              never_run=never_run)

        assert backlog == {_d(2): 1}
        assert never_run == [_d(2), _d(3)]

    def test_a_finished_retry_of_a_never_run_date_removes_it(self):
        fold = _fold({_d(2): 0}, target_date=_d(2), owes_retry=False,
                     today=_d(5))

        assert fold.backlog == {}

    def test_an_unfinished_retry_of_a_never_run_date_counts_one_attempt(self):
        fold = _fold({_d(2): 0}, target_date=_d(2), owes_retry=True,
                     today=_d(5))

        assert fold.backlog == {_d(2): 1}

    def test_an_entry_older_than_the_window_is_abandoned(self):
        today = _d(20)
        max_age = trickle_state.RETRY_MAX_AGE_DAYS
        too_old = today - timedelta(days=max_age + 1)
        at_the_edge = today - timedelta(days=max_age)

        backlog, abandoned = _fold(
            {too_old: 2, at_the_edge: 1},
            target_date=_d(19), owes_retry=False, today=today,
        )

        assert backlog == {at_the_edge: 1}
        assert abandoned == {too_old: 2}

    def test_an_entry_reaching_the_attempt_bound_is_abandoned(self):
        max_attempts = trickle_state.RETRY_MAX_ATTEMPTS

        fold = _fold({_d(1): max_attempts - 1}, target_date=_d(1),
                     owes_retry=True, today=_d(3))

        assert fold.backlog == {}
        assert fold.abandoned == {_d(1): max_attempts}

    def test_due_retries_are_oldest_first_and_at_most_the_limit(self):
        backlog = {_d(5): 1, _d(3): 2, _d(4): 1}
        select = trickle_state.select_due_retries

        assert select(backlog, today=_d(6)) == (_d(3),)
        assert select(backlog, today=_d(6), limit=2) == (_d(3), _d(4))
        assert select(backlog, today=_d(6), limit=9) == (_d(3), _d(4), _d(5))

    def test_an_entry_outside_the_window_is_never_due(self):
        today = _d(20)
        max_age = trickle_state.RETRY_MAX_AGE_DAYS
        too_old = today - timedelta(days=max_age + 1)
        at_the_edge = today - timedelta(days=max_age)

        due = trickle_state.select_due_retries(
            {too_old: 1, at_the_edge: 1}, today=today, limit=5)

        assert due == (at_the_edge,)

    def test_an_empty_backlog_has_nothing_due(self):
        assert trickle_state.select_due_retries({}, today=_d(2)) == ()

    def test_an_outage_one_night_shorter_than_the_window_is_fully_recovered(
        self,
    ):
        nights = trickle_state.RETRY_MAX_AGE_DAYS - 1

        outage, retried, abandoned, backlog = _simulate_outage(
            nights, healthy_nights=nights + 2)

        assert retried == outage
        assert abandoned == {}
        assert backlog == {}

    def test_a_longer_outage_loses_only_the_dates_the_window_cannot_reach(
        self,
    ):
        nights = trickle_state.RETRY_MAX_AGE_DAYS + 2
        lost = nights - (trickle_state.RETRY_MAX_AGE_DAYS - 1)

        outage, retried, abandoned, backlog = _simulate_outage(
            nights, healthy_nights=nights + 2)

        assert abandoned == {d: 1 for d in outage[:lost]}
        assert retried == outage[lost:]
        assert backlog == {}

    def test_a_systemd_outage_of_the_window_in_missed_elapses_is_recovered(
        self,
    ):
        missed = trickle_state.RETRY_MAX_AGE_DAYS

        retried, abandoned, backlog = _simulate_missed_elapses(
            missed, nights=missed + 2)

        assert retried == [_d(day) for day in range(2, missed + 1)]
        assert abandoned == {}
        assert backlog == {}

    def test_a_failed_night_after_the_catch_up_costs_the_oldest_owed_date(
        self,
    ):
        """The full-recovery bound holds only when every later night
        finishes: a failed night retries nothing, so the date it would have
        retried ages out."""
        missed = trickle_state.RETRY_MAX_AGE_DAYS
        failed_target = _d(2 + missed)

        retried, abandoned, backlog = _simulate_missed_elapses(
            missed, nights=missed + 3, failed_nights={1})

        assert abandoned == {_d(3): 0}
        assert retried == [
            _d(day) for day in range(2, missed + 1) if day != 3
        ] + [failed_target]
        assert backlog == {}

    def test_a_poison_date_is_abandoned_after_the_attempt_bound(self):
        poison = _d(1)
        backlog = _fold({}, target_date=poison, owes_retry=True,
                        today=_d(2)).backlog
        unfinished_passes = 1
        abandoned = {}
        for night in range(2, 12):
            today = _d(night + 1)
            backlog = _fold(backlog, target_date=_d(night), owes_retry=False,
                            today=today).backlog
            if not trickle_state.select_due_retries(backlog, today=today):
                break
            backlog, abandoned = _fold(backlog, target_date=poison,
                                       owes_retry=True, today=today)
            unfinished_passes += 1

        assert abandoned == {poison: trickle_state.RETRY_MAX_ATTEMPTS}
        assert unfinished_passes == trickle_state.RETRY_MAX_ATTEMPTS
        assert backlog == {}

    @pytest.mark.parametrize('raw', [
        pytest.param(None, id='None'),
        pytest.param([['2026-07-01', 1]], id='a-list'),
        pytest.param('2026-07-01', id='a-string'),
        pytest.param(3, id='an-int'),
    ])
    def test_parsing_a_non_mapping_gives_an_empty_backlog(self, raw):
        assert trickle_state.parse_retry_backlog(raw) == {}

    def test_parsing_drops_every_invalid_entry_and_keeps_the_rest(self):
        raw = {
            '2026-07-01': 2,
            'not-a-date': 1,
            '2026-02-30': 1,
            '2026-07-02': 0,
            '2026-07-03': -1,
            '2026-07-04': True,
            '2026-07-05': '1',
            '2026-07-06': 1.0,
        }

        assert trickle_state.parse_retry_backlog(raw) == {_d(1): 2, _d(2): 0}

    def test_a_serialized_fold_round_trips_through_the_parser(self):
        fold = _fold({_d(1): 2}, target_date=_d(3), owes_retry=True,
                     today=_d(4), never_run=(_d(2),))
        serialized = {d.isoformat(): n for d, n in fold.backlog.items()}

        parsed = trickle_state.parse_retry_backlog(
            json.loads(json.dumps(serialized)))

        assert parsed == fold.backlog == {_d(1): 2, _d(2): 0, _d(3): 1}


def _warnings_naming(caplog, text):
    return [
        r for r in caplog.records
        if r.levelname == 'WARNING' and text in r.getMessage()
    ]


def _without_backlog(state: TrickleState) -> TrickleState:
    return dataclasses.replace(state, retry_backlog={})


class TestRecordRunRetryBacklog:
    """record_run folds each scheduled pass into the document's additive
    ``retry_backlog`` ({date: unfinished attempts})."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_a_failed_run_owes_its_own_date_and_a_repeat_counts_again(self):
        first = _record(day=1, exit_code=1, selected_count=6)
        again = _record(day=1, exit_code=1, selected_count=6)

        assert dict(first.retry_backlog) == {_d(1): 1}
        assert dict(again.retry_backlog) == {_d(1): 2}

    def test_a_capped_run_owes_its_date_and_records_capped(self):
        capped = _record(day=1, capped=True, zero_signal_dropped=2)

        assert capped.exit_code == 0
        assert capped.capped is True
        assert dict(capped.retry_backlog) == {_d(1): 1}

    def test_a_normal_run_records_capped_false_and_owes_nothing(self):
        state = _record(day=1, selected_count=1)

        assert state.capped is False
        assert dict(state.retry_backlog) == {}

    def test_a_run_that_committed_never_owes_its_date(self):
        state = _record(day=1, exit_code=1, selected_count=6, commit_made=True)

        assert dict(state.retry_backlog) == {}

    def test_a_finished_run_clears_its_own_date_and_carries_the_rest(self):
        _record(day=1, exit_code=1)
        _record(day=2, exit_code=1)

        cleared = _record(day=1, selected_count=1)
        carried = _record(day=3, selected_count=1)

        assert dict(cleared.retry_backlog) == {_d(2): 1}
        assert dict(carried.retry_backlog) == {_d(2): 1}

    def test_an_entry_past_the_window_is_pruned_with_one_warning(self, caplog):
        _record(day=1, exit_code=1)

        with caplog.at_level('WARNING'):
            state = _record(day=2 + trickle_state.RETRY_MAX_AGE_DAYS,
                            selected_count=1)

        assert dict(state.retry_backlog) == {}
        assert len(_warnings_naming(caplog, '2026-07-01')) == 1

    def test_next_state_folds_the_backlog_without_warning(self, caplog):
        """The abandonment WARNING belongs to the write, so a nightly that
        folds tonight's state before recording it logs it once."""
        _record(day=1, exit_code=1)
        run = _night(day=2 + trickle_state.RETRY_MAX_AGE_DAYS, selected_count=1)

        with caplog.at_level('WARNING'):
            projected = trickle_state.next_state(
                trickle_state.read_prior('dark_factory'), 'dark_factory', run
            )

        assert dict(projected.retry_backlog) == {}
        assert not _warnings(caplog)

    def test_a_document_without_a_backlog_reads_ok_and_starts_one(
        self, caplog
    ):
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_pre_gamma_doc()))

        with caplog.at_level('WARNING'):
            state = _record(day=2, exit_code=1)

        assert not _warnings(caplog)
        assert dict(state.retry_backlog) == {_d(2): 1}

    @pytest.mark.parametrize('garbage', [
        pytest.param(['2026-07-01'], id='a-list'),
        pytest.param({'yesterday': 1, '2026-07-01': 'x'}, id='bad-entries'),
        pytest.param('2026-07-01', id='a-string'),
    ])
    def test_a_garbage_backlog_is_dropped_never_raised(self, garbage):
        state = _record(day=1, selected_count=1)
        _path().write_text(json.dumps({**state.to_json(), 'retry_backlog': garbage}))

        after = _record(day=2, exit_code=1)

        assert dict(after.retry_backlog) == {_d(2): 1}
        assert after.consecutive_failed_runs == 1


def _retry(project_id='dark_factory', *, day, recorded_day, exit_code=0,
           capped=False, commit_made=False) -> TrickleState | None:
    return trickle_state.record_retry(
        project_id,
        target_date=date(2026, 7, day),
        recorded_at=datetime(2026, 7, recorded_day, 3, 30, 0, tzinfo=UTC),
        exit_code=exit_code,
        capped=capped,
        commit_made=commit_made,
    )


def _retried(**kwargs) -> TrickleState:
    """:func:`_retry` over a readable predecessor, so a document is written."""
    state = _retry(**kwargs)
    assert state is not None, 'record_retry wrote nothing'
    return state


class TestRecordRetry:
    """record_retry rewrites ONLY ``retry_backlog``: the rest of the
    document keeps describing the scheduled night."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_a_finished_retry_clears_its_date_and_changes_nothing_else(self):
        _record(day=1, exit_code=1, selected_count=6)
        scheduled = _record(day=2, selected_count=1, barren_escalated=True)

        retried = _retried(day=1, recorded_day=2)

        assert trickle_state.load_state(_path()) == Ok(retried)
        assert dict(scheduled.retry_backlog) == {_d(1): 1}
        assert dict(retried.retry_backlog) == {}
        assert _without_backlog(retried) == _without_backlog(scheduled)

    def test_an_unfinished_retry_counts_an_attempt(self):
        _record(day=1, exit_code=1)
        scheduled = _record(day=2, selected_count=1)

        retried = _retried(day=1, recorded_day=2, exit_code=1)

        assert dict(retried.retry_backlog) == {_d(1): 2}
        assert _without_backlog(retried) == _without_backlog(scheduled)

    def test_a_capped_retry_counts_an_attempt(self):
        _record(day=1, exit_code=1)
        _record(day=2, selected_count=1)

        retried = _retried(day=1, recorded_day=2, capped=True)

        assert dict(retried.retry_backlog) == {_d(1): 2}

    def test_an_unrecorded_barren_escalated_stays_unrecorded(self):
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_pre_gamma_doc(
            retry_backlog={'2026-07-01': 1}
        )))

        retried = _retried(day=1, recorded_day=2)

        assert retried.barren_escalated is None
        read = trickle_state.load_state(path)
        assert isinstance(read, Ok)
        assert read.value.barren_escalated is None

    def test_the_attempt_bound_abandons_the_date_with_one_warning(
        self, caplog
    ):
        _record(day=1, exit_code=1)
        _record(day=2, selected_count=1)
        unfinished = trickle_state.RETRY_MAX_ATTEMPTS - 1

        with caplog.at_level('WARNING'):
            for night in range(unfinished):
                state = _retried(day=1, recorded_day=2 + night, exit_code=1)

        assert dict(state.retry_backlog) == {}
        assert len(_warnings_naming(caplog, '2026-07-01')) == 1

    def test_with_no_predecessor_nothing_is_written(self, caplog):
        with caplog.at_level('WARNING'):
            state = _retry(day=1, recorded_day=2)

        assert state is None
        assert not _path().exists()
        assert len(_warnings_naming(caplog, '2026-07-01')) == 1

    def test_with_a_corrupt_predecessor_nothing_is_written(self, caplog):
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{corrupt')

        with caplog.at_level('WARNING'):
            state = _retry(day=1, recorded_day=2)

        assert state is None
        assert path.read_bytes() == b'{corrupt'
        assert len(_warnings_naming(caplog, '2026-07-01')) == 1

    def test_the_write_leaves_no_temp_residue(self):
        _record(day=1, exit_code=1)
        _record(day=2, selected_count=1)

        _retry(day=1, recorded_day=2)

        assert [p.name for p in _path().parent.iterdir()] == [
            trickle_state.STATE_FILENAME
        ]


def _rewrite_state(doc: dict) -> None:
    _path().write_text(json.dumps(doc))


class TestRecordRunScheduleCursor:
    """record_run's ``last_scheduled_target_date``: only a scheduled pass
    moves it, and a scheduled pass owes, with 0 attempts, every date it
    jumped over since the previous scheduled pass."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_a_first_scheduled_run_sets_the_cursor_and_owes_nothing(self):
        state = _record(day=1, scheduled=True)

        assert state.last_scheduled_target_date == _d(1)
        assert dict(state.retry_backlog) == {}

    def test_a_first_manual_run_leaves_no_cursor(self):
        assert _record(day=1).last_scheduled_target_date is None

    def test_a_scheduled_run_owes_the_dates_it_jumped_over(self):
        _record(day=1, scheduled=True)

        state = _record(day=4, scheduled=True, selected_count=1)

        assert dict(state.retry_backlog) == {_d(2): 0, _d(3): 0}
        assert state.last_scheduled_target_date == _d(4)

    def test_a_post_midnight_catch_up_owes_the_skipped_date_once(self):
        _record(day=1, scheduled=True)

        catch_up = _record(day=3, scheduled=True)
        again = _record(day=3, scheduled=True)

        assert dict(catch_up.retry_backlog) == {_d(2): 0}
        assert dict(again.retry_backlog) == {_d(2): 0}

    def test_a_manual_backfill_never_reads_as_a_gap(self):
        _record(day=5, scheduled=True)

        backfill = _record(day=1, selected_count=1)
        next_night = _record(day=6, scheduled=True)

        assert backfill.last_scheduled_target_date == _d(5)
        assert backfill.target_date == _d(1)
        assert dict(next_night.retry_backlog) == {}
        assert next_night.last_scheduled_target_date == _d(6)

    def test_a_later_manual_run_never_moves_the_cursor(self):
        _record(day=1, scheduled=True)
        _record(day=9)

        assert dict(_record(day=2, scheduled=True).retry_backlog) == {}

    def test_an_owed_never_run_date_keeps_its_attempt_count(self):
        _record(day=1, scheduled=True)
        _record(day=2, exit_code=1)

        state = _record(day=3, scheduled=True)

        assert dict(state.retry_backlog) == {_d(2): 1}

    def test_a_predecessor_without_a_cursor_owes_no_never_run_dates(self):
        pre_6548 = _record(day=1, scheduled=True).to_json()
        del pre_6548['last_scheduled_target_date']
        _rewrite_state(pre_6548)

        state = _record(day=4, scheduled=True)

        assert dict(state.retry_backlog) == {}
        assert state.last_scheduled_target_date == _d(4)

    @pytest.mark.parametrize('garbage', [
        pytest.param('yesterday', id='not-a-date'),
        pytest.param(7, id='an-int'),
        pytest.param(None, id='null'),
    ])
    def test_a_garbage_cursor_owes_no_never_run_dates(self, garbage):
        _rewrite_state({
            **_record(day=1, scheduled=True).to_json(),
            'last_scheduled_target_date': garbage,
        })

        state = _record(day=4, scheduled=True)

        assert dict(state.retry_backlog) == {}
        assert state.last_scheduled_target_date == _d(4)

    def test_a_gap_past_the_window_owes_its_in_window_dates_only(self):
        max_age = trickle_state.RETRY_MAX_AGE_DAYS
        last = 1 + max_age + 2
        _record(day=1, scheduled=True)

        state = _record(day=last, scheduled=True)

        assert dict(state.retry_backlog) == {
            _d(day): 0 for day in range(last - max_age, last)
        }

    def test_a_gap_of_years_is_abandoned_with_one_summary_warning(
        self, caplog
    ):
        _rewrite_state({
            **_record(day=1, scheduled=True).to_json(),
            'last_scheduled_target_date': '2020-01-01',
        })
        oldest_owed = _d(20) - timedelta(days=trickle_state.RETRY_MAX_AGE_DAYS)
        expired = (oldest_owed - date(2020, 1, 2)).days

        with caplog.at_level('WARNING'):
            state = _record(day=20, scheduled=True)

        warnings = _warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert f'abandoning {expired} never-run date(s)' in message
        assert '2020-01-02..2026-07-12' in message
        assert min(state.retry_backlog) == oldest_owed
        assert len(state.retry_backlog) == trickle_state.RETRY_MAX_AGE_DAYS

    def test_a_scheduled_run_behind_the_cursor_never_moves_it_back(self):
        _record(day=5, scheduled=True)

        skewed = _record(day=3, scheduled=True)
        next_night = _record(day=6, scheduled=True)

        assert skewed.last_scheduled_target_date == _d(5)
        assert dict(next_night.retry_backlog) == {}
        assert next_night.last_scheduled_target_date == _d(6)

    def test_a_retry_leaves_the_cursor_untouched(self):
        _record(day=1, scheduled=True)
        _record(day=3, scheduled=True)

        retried = _retried(day=2, recorded_day=3)

        assert retried.last_scheduled_target_date == _d(3)
        assert dict(retried.retry_backlog) == {}


class TestDueRetries:
    """``due_retries(state, *, today)`` — the reader the nightly calls, so
    the backlog's policy stays inside trickle_state."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_a_recorded_failed_night_is_due_after_a_productive_one(self):
        _record(day=1, exit_code=1, selected_count=6)
        state = _record(day=2, selected_count=1)

        assert trickle_state.due_retries(state, today=_d(3)) == (_d(1),)

    def test_several_owed_dates_give_the_oldest_up_to_the_nightly_limit(self):
        _record(day=1, exit_code=1)
        _record(day=2, exit_code=1)
        state = _record(day=3, selected_count=1)

        due = trickle_state.due_retries(state, today=_d(4))

        assert due == (_d(1), _d(2))[:trickle_state.RETRIES_PER_NIGHT]

    @pytest.mark.parametrize('backlog', [
        pytest.param({}, id='empty'),
        pytest.param({_d(1): 1}, id='outside-the-window'),
    ])
    def test_nothing_is_due_from_what_owes_nothing_today(self, backlog):
        today = _d(1) + timedelta(days=trickle_state.RETRY_MAX_AGE_DAYS + 1)

        state = _state(retry_backlog=backlog)

        assert trickle_state.due_retries(state, today=today) == ()
