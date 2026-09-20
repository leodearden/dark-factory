"""Tests for scripts/legibility/trickle_state.py — the nightly trickle
run-state recorder that lets a probe answer WHY a night produced nothing.

The classifier under test is not a tuned heuristic; it is a derivation from
:class:`legibility.sampling.SampleResult`'s CONSERVATION INVARIANT::

    total_records == zero_signal_dropped + dedupe_collapsed
                     + below_sampling_cut + budget_skipped + len(selected)

Every counter fixture below therefore SATISFIES that identity — an
inconsistent fixture would prove nothing about a classifier derived from it.

The three outcomes:

- ``productive`` — ``selected > 0``; digests were built.
- ``barren``     — ``selected == 0`` and real, distinct, non-duplicate signal
  reached the sampling/budget stage and NOTHING was digested.
- ``quiet``      — everything else, which BY THE INVARIANT means every
  enumerated record left by the zero-signal or dedupe door (or nothing was
  enumerated at all). That is exactly the "genuinely quiet night" PRD
  decision 7 protects, so a quiet or dormant project can never be
  classified barren — the no-false-alarm guarantee as a PROOF from the
  invariant rather than a threshold someone picked.
"""
from __future__ import annotations

import json
import os
import pwd
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from legibility import trickle_state
from legibility.trickle_state import (
    OUTCOME_BARREN,
    OUTCOME_PRODUCTIVE,
    OUTCOME_QUIET,
    classify_run,
)


def _counters(
    *,
    zero_signal_dropped=0,
    dedupe_collapsed=0,
    below_sampling_cut=0,
    budget_skipped=0,
    selected_count=0,
):
    """Build a classify_run kwargs dict whose ``total_records`` is DERIVED
    from the other five counters, so every fixture satisfies SampleResult's
    conservation identity by construction."""
    return dict(
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


class TestClassifyRun:
    """The three-valued absence classifier."""

    def test_selected_is_productive(self):
        assert classify_run(**_counters(selected_count=3)) == OUTCOME_PRODUCTIVE

    def test_partially_truncated_night_is_still_productive(self):
        """A night that digested SOMETHING and skipped the rest on budget is
        the byte budget working as designed — never barren."""
        result = classify_run(**_counters(selected_count=2, budget_skipped=9))
        assert result == OUTCOME_PRODUCTIVE

    def test_selected_with_every_other_door_open_is_productive(self):
        result = classify_run(
            **_counters(
                selected_count=1,
                budget_skipped=4,
                below_sampling_cut=7,
                dedupe_collapsed=2,
                zero_signal_dropped=5,
            )
        )
        assert result == OUTCOME_PRODUCTIVE

    def test_budget_door_is_barren(self):
        """Reproduction of the real 2026-07-16..29 incident: candidates
        existed, competed, and were ALL discarded on the byte budget."""
        result = classify_run(**_counters(selected_count=0, budget_skipped=4))
        assert result == OUTCOME_BARREN

    def test_sampling_cut_door_is_barren(self):
        """The sibling absence mode task 3270 does NOT cover: real, distinct
        signal held back by the sampling cut, nothing digested. Different
        remedy (sampling.top_fraction/per_stratum_min, never
        budgets.max_daily_digest_bytes) — see SampleResult's docstring."""
        result = classify_run(
            **_counters(selected_count=0, below_sampling_cut=3, budget_skipped=0)
        )
        assert result == OUTCOME_BARREN

    def test_both_doors_open_is_barren(self):
        result = classify_run(
            **_counters(selected_count=0, below_sampling_cut=3, budget_skipped=4)
        )
        assert result == OUTCOME_BARREN

    def test_dormant_project_is_quiet(self):
        """All counters zero — nothing was even enumerated. A dormant
        project is a legitimate state, not a degradation."""
        assert classify_run(**_counters()) == OUTCOME_QUIET

    def test_all_zero_signal_is_quiet(self):
        result = classify_run(**_counters(zero_signal_dropped=17))
        assert result == OUTCOME_QUIET

    def test_zero_signal_plus_dedupe_only_is_quiet(self):
        result = classify_run(
            **_counters(zero_signal_dropped=6, dedupe_collapsed=3)
        )
        assert result == OUTCOME_QUIET

    def test_outcome_constants_are_distinct_strings(self):
        outcomes = {OUTCOME_PRODUCTIVE, OUTCOME_QUIET, OUTCOME_BARREN}
        assert len(outcomes) == 3
        assert all(isinstance(o, str) and o for o in outcomes)


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
        result = classify_run(
            **_counters(
                zero_signal_dropped=zero_signal_dropped,
                dedupe_collapsed=dedupe_collapsed,
                below_sampling_cut=0,
                budget_skipped=0,
                selected_count=0,
            )
        )
        assert result == OUTCOME_QUIET, (
            f"zero_signal_dropped={zero_signal_dropped} "
            f"dedupe_collapsed={dedupe_collapsed} classified {result!r}; a "
            f"night where every record left by the zero-signal or dedupe "
            f"door is exactly the 'genuinely quiet night' decision 7 "
            f"protects and must never alarm."
        )


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


class TestLoadState:
    """The three-valued reader.

    Shape AND vocabulary are copied VERBATIM from the sibling reader in
    this same package, ``census_trigger.load_census_state``
    (census_trigger.py:239) — same tuple shape, same literal status
    strings ``'missing' | 'malformed' | 'ok'`` (NOT ``'invalid'``), same
    fail-safe posture — so the two state readers in scripts/legibility/
    cannot drift into two vocabularies for one concept.
    """

    def test_missing_file_is_missing_and_silent(self, tmp_path, caplog):
        """A project that has never recorded a run is a NORMAL state, not a
        degradation — so it logs nothing."""
        with caplog.at_level('WARNING'):
            status, data = trickle_state.load_state(tmp_path / 'nope.json')

        assert (status, data) == ('missing', None)
        assert not caplog.records

    def test_invalid_json_is_malformed_with_one_warning(self, tmp_path, caplog):
        path = tmp_path / 'trickle-state.json'
        path.write_text('{not json at all')

        with caplog.at_level('WARNING'):
            status, data = trickle_state.load_state(path)

        assert (status, data) == ('malformed', None)
        assert len([r for r in caplog.records if r.levelname == 'WARNING']) == 1

    def test_non_dict_top_level_is_malformed(self, tmp_path, caplog):
        path = tmp_path / 'trickle-state.json'
        path.write_text('["a", "list"]')

        with caplog.at_level('WARNING'):
            status, data = trickle_state.load_state(path)

        assert (status, data) == ('malformed', None)
        assert len([r for r in caplog.records if r.levelname == 'WARNING']) == 1

    def test_unknown_schema_version_is_malformed(self, tmp_path, caplog):
        """A reader that silently accepts a shape it does not understand is
        the silent-degradation mode this module exists to close."""
        path = tmp_path / 'trickle-state.json'
        path.write_text(
            '{"schema_version": 99999, "project_id": "dark_factory"}'
        )

        with caplog.at_level('WARNING'):
            status, data = trickle_state.load_state(path)

        assert (status, data) == ('malformed', None)
        assert len([r for r in caplog.records if r.levelname == 'WARNING']) == 1

    def test_unreadable_path_is_malformed_not_raised(self, tmp_path, caplog):
        """A directory where a file is expected raises OSError on read; the
        reader must absorb it, never propagate."""
        path = tmp_path / 'trickle-state.json'
        path.mkdir()

        with caplog.at_level('WARNING'):
            status, data = trickle_state.load_state(path)

        assert (status, data) == ('malformed', None)

    def test_well_formed_is_ok(self, tmp_path):
        path = tmp_path / 'trickle-state.json'
        path.write_text(
            f'{{"schema_version": {trickle_state.STATE_SCHEMA_VERSION}, '
            f'"project_id": "dark_factory", "outcome": "quiet"}}'
        )

        status, data = trickle_state.load_state(path)

        assert status == 'ok'
        assert isinstance(data, dict)
        assert data['project_id'] == 'dark_factory'

    def test_uses_the_same_vocabulary_as_the_sibling_reader(self, tmp_path):
        """Guards against drift into a second vocabulary for one concept."""
        from legibility import census_trigger

        census_path = tmp_path / 'census-state.json'
        assert census_trigger.load_census_state(census_path)[0] == 'missing'
        assert trickle_state.load_state(tmp_path / 'nope.json')[0] == 'missing'

        census_path.write_text('{oops')
        (tmp_path / 'bad.json').write_text('{oops')
        assert census_trigger.load_census_state(census_path)[0] == 'malformed'
        assert trickle_state.load_state(tmp_path / 'bad.json')[0] == 'malformed'


def _record(project_id='dark_factory', *, day=1, hour=3, exit_code=0,
            applied=0, commit_made=False, budget_suppressed=False, **counters):
    """Call record_run with a derived-total counter set and an injected
    clock (the module is deliberately clock-injectable — no time faking)."""
    kw = _counters(**counters)
    recorded_at = datetime(2026, 7, day, hour, 0, 0, tzinfo=UTC)
    return trickle_state.record_run(
        project_id,
        target_date=date(2026, 7, day),
        recorded_at=recorded_at,
        exit_code=exit_code,
        applied=applied,
        commit_made=commit_made,
        budget_suppressed=budget_suppressed,
        **kw,
    )


class TestRecordRun:
    """The writer: streak arithmetic, carried-forward fields, atomicity."""

    @pytest.fixture(autouse=True)
    def _xdg(self, tmp_path, monkeypatch):
        monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))

    def test_first_call_creates_parents_and_writes_every_field(self, tmp_path):
        doc = _record(day=1, selected_count=2, zero_signal_dropped=4,
                      applied=2, commit_made=True)

        path = trickle_state.trickle_state_path('dark_factory')
        assert path.is_file(), 'record_run must mkdir(parents=True)'

        on_disk = json.loads(path.read_text())
        assert on_disk == doc

        assert doc['schema_version'] == trickle_state.STATE_SCHEMA_VERSION
        assert doc['project_id'] == 'dark_factory'
        assert doc['target_date'] == '2026-07-01'
        assert doc['recorded_at'] == '2026-07-01T03:00:00+00:00'
        assert doc['exit_code'] == 0
        assert doc['outcome'] == OUTCOME_PRODUCTIVE
        assert doc['applied'] == 2
        assert doc['commit_made'] is True
        assert doc['budget_suppressed'] is False
        assert doc['consecutive_barren_runs'] == 0
        assert doc['counters'] == {
            'total_records': 6,
            'zero_signal_dropped': 4,
            'dedupe_collapsed': 0,
            'below_sampling_cut': 0,
            'budget_skipped': 0,
            'selected_count': 2,
        }

    def test_recorded_at_is_timezone_aware_utc_iso8601(self):
        doc = _record(day=2, selected_count=1)
        parsed = datetime.fromisoformat(doc['recorded_at'])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)

    def test_barren_streak_increments(self):
        assert _record(day=1, budget_skipped=4)['consecutive_barren_runs'] == 1
        assert _record(day=2, budget_skipped=4)['consecutive_barren_runs'] == 2
        assert _record(day=3, below_sampling_cut=3)['consecutive_barren_runs'] == 3

    def test_productive_run_resets_the_streak(self):
        _record(day=1, budget_skipped=4)
        _record(day=2, budget_skipped=4)
        doc = _record(day=3, selected_count=1)
        assert doc['outcome'] == OUTCOME_PRODUCTIVE
        assert doc['consecutive_barren_runs'] == 0

    def test_quiet_run_also_resets_the_streak(self):
        """A legitimately quiet night is NOT evidence of breakage — PRD
        decision 7's guarantee expressed in the streak itself."""
        _record(day=1, budget_skipped=4)
        _record(day=2, budget_skipped=4)
        doc = _record(day=3, zero_signal_dropped=5)
        assert doc['outcome'] == OUTCOME_QUIET
        assert doc['consecutive_barren_runs'] == 0

    def test_streak_rearms_after_a_reset(self):
        _record(day=1, budget_skipped=4)
        _record(day=2, selected_count=1)
        assert _record(day=3, budget_skipped=4)['consecutive_barren_runs'] == 1

    def test_last_productive_at_is_stamped_and_carried_forward(self):
        """The field an operator reads to answer 'when did this pipeline
        last actually do anything'."""
        first = _record(day=1, zero_signal_dropped=3)
        assert first['last_productive_at'] is None

        prod = _record(day=2, selected_count=1)
        assert prod['last_productive_at'] == prod['recorded_at']
        stamp = prod['last_productive_at']

        assert _record(day=3, budget_skipped=4)['last_productive_at'] == stamp
        assert _record(day=4, zero_signal_dropped=1)['last_productive_at'] == stamp
        assert _record(day=5, budget_skipped=4)['last_productive_at'] == stamp

        newer = _record(day=6, selected_count=2)
        assert newer['last_productive_at'] == newer['recorded_at'] != stamp

    def test_write_is_atomic_and_leaves_no_tmp_file(self):
        _record(day=1, selected_count=1)
        state_dir = trickle_state.trickle_state_path('dark_factory').parent
        assert [p.name for p in state_dir.iterdir()] == [
            trickle_state.STATE_FILENAME
        ]

    def test_a_pre_seeded_tmp_sentinel_does_not_survive(self):
        _record(day=1, selected_count=1)
        state_dir = trickle_state.trickle_state_path('dark_factory').parent
        sentinel = state_dir / (trickle_state.STATE_FILENAME + '.tmp')
        sentinel.write_text('leftover from a crashed write')

        _record(day=2, selected_count=1)

        assert not sentinel.exists(), (
            'the tmp sibling must be os.replace()d onto the final path, not '
            'left behind'
        )
        assert [p.name for p in state_dir.iterdir()] == [
            trickle_state.STATE_FILENAME
        ]

    def test_corrupt_predecessor_degrades_to_under_reporting(self, caplog):
        """Losing streak history must degrade to UNDER-reporting, never to a
        crash inside the nightly run."""
        _record(day=1, budget_skipped=4)
        _record(day=2, budget_skipped=4)

        path = trickle_state.trickle_state_path('dark_factory')
        path.write_text('{corrupt')

        with caplog.at_level('WARNING'):
            doc = _record(day=3, budget_skipped=4)

        assert doc['outcome'] == OUTCOME_BARREN
        assert doc['consecutive_barren_runs'] == 1, (
            'a lost predecessor restarts the streak from this run alone'
        )
        assert doc['last_productive_at'] is None
        assert any(r.levelname == 'WARNING' for r in caplog.records)

    def test_round_trips_through_load_state(self):
        doc = _record(day=1, budget_skipped=4, budget_suppressed=True)
        status, loaded = trickle_state.load_state(
            trickle_state.trickle_state_path('dark_factory')
        )
        assert status == 'ok'
        assert loaded == doc

    def test_distinct_projects_keep_independent_streaks(self):
        _record('dark_factory', day=1, budget_skipped=4)
        _record('dark_factory', day=2, budget_skipped=4)
        other = _record('reify', day=2, budget_skipped=4)
        assert other['consecutive_barren_runs'] == 1
