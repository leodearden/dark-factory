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
    OUTCOME_FAILED,
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


def _classify(exit_code=0, **counters):
    """Call classify_run with a derived-total counter set and an explicit
    exit code.

    ``exit_code`` is a REQUIRED keyword-only parameter of ``classify_run``
    (task 4514), so every call site must state it. It lives here rather
    than in ``_counters`` because it is not a counter and must not be
    folded into the conservation identity that helper exists to satisfy."""
    return classify_run(exit_code=exit_code, **_counters(**counters))


class TestClassifyRun:
    """The three-valued absence classifier."""

    def test_selected_is_productive(self):
        assert _classify(selected_count=3) == OUTCOME_PRODUCTIVE

    def test_partially_truncated_night_is_still_productive(self):
        """A night that digested SOMETHING and skipped the rest on budget is
        the byte budget working as designed — never barren."""
        result = _classify(selected_count=2, budget_skipped=9)
        assert result == OUTCOME_PRODUCTIVE

    def test_selected_with_every_other_door_open_is_productive(self):
        result = _classify(
            selected_count=1,
            budget_skipped=4,
            below_sampling_cut=7,
            dedupe_collapsed=2,
            zero_signal_dropped=5,
        )
        assert result == OUTCOME_PRODUCTIVE

    def test_budget_door_is_barren(self):
        """Reproduction of the real 2026-07-16..29 incident: candidates
        existed, competed, and were ALL discarded on the byte budget."""
        result = _classify(selected_count=0, budget_skipped=4)
        assert result == OUTCOME_BARREN

    def test_sampling_cut_door_is_barren(self):
        """The sibling absence mode task 3270 does NOT cover: real, distinct
        signal held back by the sampling cut, nothing digested. Different
        remedy (sampling.top_fraction/per_stratum_min, never
        budgets.max_daily_digest_bytes) — see SampleResult's docstring."""
        result = _classify(
            selected_count=0, below_sampling_cut=3, budget_skipped=0
        )
        assert result == OUTCOME_BARREN

    def test_both_doors_open_is_barren(self):
        result = _classify(
            selected_count=0, below_sampling_cut=3, budget_skipped=4
        )
        assert result == OUTCOME_BARREN

    def test_dormant_project_is_quiet(self):
        """All counters zero — nothing was even enumerated. A dormant
        project is a legitimate state, not a degradation."""
        assert _classify() == OUTCOME_QUIET

    def test_all_zero_signal_is_quiet(self):
        result = _classify(zero_signal_dropped=17)
        assert result == OUTCOME_QUIET

    def test_zero_signal_plus_dedupe_only_is_quiet(self):
        result = _classify(
            zero_signal_dropped=6, dedupe_collapsed=3
        )
        assert result == OUTCOME_QUIET

    def test_outcome_constants_are_distinct_strings(self):
        outcomes = {
            OUTCOME_PRODUCTIVE, OUTCOME_QUIET, OUTCOME_BARREN, OUTCOME_FAILED,
        }
        assert len(outcomes) == 4
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
        result = _classify(
            exit_code=0,
            zero_signal_dropped=zero_signal_dropped,
            dedupe_collapsed=dedupe_collapsed,
            below_sampling_cut=0,
            budget_skipped=0,
            selected_count=0,
        )
        assert result == OUTCOME_QUIET, (
            f"zero_signal_dropped={zero_signal_dropped} "
            f"dedupe_collapsed={dedupe_collapsed} classified {result!r}; a "
            f"night where every record left by the zero-signal or dedupe "
            f"door is exactly the 'genuinely quiet night' decision 7 "
            f"protects and must never alarm."
        )


class TestFailedOutcome:
    """The fourth outcome and its PRIORITY over the other three.

    A run that did not complete cannot be described by the sampler
    counters: ``selected_count > 0`` proves signal reached the digest
    stage, never that the night FINISHED. So ``exit_code != 0`` is read
    first and wins outright.
    """

    def test_failed_constant_is_the_expected_string(self):
        assert OUTCOME_FAILED == 'failed'

    def test_failure_beats_productive(self):
        """THE regression this task exists for. The 2026-08-18 reify run
        selected six digests, exited 1, applied 0 and made no commit — and
        recorded ``productive``. Generalised: a permanently broken coder
        samples > 0, storms, exits 1, and records ``productive`` / streak
        0 / a fresh ``last_productive_at`` every night forever."""
        assert _classify(exit_code=1, selected_count=6) == OUTCOME_FAILED

    def test_failure_beats_barren_at_the_budget_door(self):
        assert _classify(exit_code=1, budget_skipped=4) == OUTCOME_FAILED

    def test_failure_beats_barren_at_the_sampling_cut(self):
        assert _classify(exit_code=1, below_sampling_cut=3) == OUTCOME_FAILED

    def test_failure_beats_quiet(self):
        assert _classify(exit_code=1) == OUTCOME_FAILED
        assert _classify(exit_code=1, zero_signal_dropped=17) == OUTCOME_FAILED

    @pytest.mark.parametrize('exit_code', [1, 2, 127, 137, -1])
    def test_any_non_zero_code_is_failed(self, exit_code):
        """Not just 1. 127 is a missing interpreter, 137 a SIGKILL/OOM, and
        a negative code is how ``subprocess`` reports a signal — every one
        of them is a night that did not finish."""
        assert _classify(exit_code=exit_code, selected_count=6) == OUTCOME_FAILED

    def test_a_clean_exit_still_classifies_by_the_counters(self):
        """The no-false-alarm guarantee is unweakened BY CONSTRUCTION: the
        new branch is gated purely on ``exit_code != 0``, so an
        ``exit_code == 0`` night falls through to exactly the branches
        that classified it before."""
        assert _classify(exit_code=0) == OUTCOME_QUIET
        assert _classify(exit_code=0, selected_count=3) == OUTCOME_PRODUCTIVE
        assert _classify(exit_code=0, budget_skipped=4) == OUTCOME_BARREN

    @pytest.mark.parametrize('zero_signal_dropped', range(0, 6))
    @pytest.mark.parametrize('dedupe_collapsed', range(0, 6))
    def test_a_quiet_night_that_exited_zero_is_still_never_barren(
        self, zero_signal_dropped, dedupe_collapsed
    ):
        """``TestQuietNightNeverBarren``'s 36-case property RE-STATED with
        an explicit ``exit_code=0`` rather than replaced, so PRD decision
        7's guarantee stays MEASURED across the contract change. The proof
        it rests on is ``trickle_state.py::classify_run``'s "WHY BRANCH 3
        IS PROVABLY SAFE" derivation from the conservation identity — not
        restated here."""
        result = _classify(
            exit_code=0,
            zero_signal_dropped=zero_signal_dropped,
            dedupe_collapsed=dedupe_collapsed,
            below_sampling_cut=0,
            budget_skipped=0,
            selected_count=0,
        )
        assert result == OUTCOME_QUIET

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
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

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
        assert doc['consecutive_failed_runs'] == 0
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
        doc = _record(day=1, exit_code=1, selected_count=6)

        assert doc['outcome'] == OUTCOME_FAILED
        assert doc['exit_code'] == 1
        assert doc['counters']['selected_count'] == 6

    def test_a_failed_run_never_restamps_last_productive_at(self):
        """The "forever green" scenario, at the writer. An operator reading
        ``last_productive_at`` must see the last night that actually did
        something, not the last night that merely started."""
        prod = _record(day=1, selected_count=2)
        stamp = prod['last_productive_at']
        assert stamp == prod['recorded_at']

        assert _record(day=2, exit_code=1, selected_count=6)[
            'last_productive_at'] == stamp
        assert _record(day=3, exit_code=1, selected_count=6)[
            'last_productive_at'] == stamp

    def test_last_productive_at_stays_none_when_the_first_run_fails(self):
        doc = _record(day=1, exit_code=1, selected_count=6)
        assert doc['last_productive_at'] is None
        assert _record(day=2, exit_code=1, selected_count=6)[
            'last_productive_at'] is None

    def test_failed_streak_increments(self):
        assert _record(day=1, exit_code=1)['consecutive_failed_runs'] == 1
        assert _record(day=2, exit_code=2)['consecutive_failed_runs'] == 2
        assert _record(day=3, exit_code=137)['consecutive_failed_runs'] == 3

    @pytest.mark.parametrize('resetter', [
        pytest.param(dict(selected_count=1), id='productive'),
        pytest.param(dict(zero_signal_dropped=5), id='quiet'),
        pytest.param(dict(budget_skipped=4), id='barren'),
    ])
    def test_any_completed_run_resets_the_failed_streak(self, resetter):
        _record(day=1, exit_code=1)
        _record(day=2, exit_code=1)

        doc = _record(day=3, **resetter)
        assert doc['outcome'] != OUTCOME_FAILED
        assert doc['consecutive_failed_runs'] == 0

    def test_failed_streak_rearms_after_a_reset(self):
        _record(day=1, exit_code=1)
        _record(day=2, selected_count=1)
        assert _record(day=3, exit_code=1)['consecutive_failed_runs'] == 1

    def test_a_failed_run_carries_the_barren_streak_forward(self):
        """CARRY FORWARD, never reset and never increment. A run that
        crashed is no evidence that signal started flowing again, so it
        must not erase a real barren streak; and its counters describe an
        unfinished night, so it must not attribute an absence the sampler
        never observed either."""
        assert _record(day=1, budget_skipped=4)['consecutive_barren_runs'] == 1
        assert _record(day=2, budget_skipped=4)['consecutive_barren_runs'] == 2

        crashed = _record(day=3, exit_code=1, selected_count=6)
        assert crashed['outcome'] == OUTCOME_FAILED
        assert crashed['consecutive_barren_runs'] == 2, (
            'a crashed run is evidence about the RUN, not about whether '
            'signal is flowing'
        )

        assert _record(day=4, budget_skipped=4)['consecutive_barren_runs'] == 3

    def test_a_document_without_the_new_field_still_reads_ok(self, caplog):
        """Backward compatibility with a pre-change document, which is what
        every live state file is on the first post-deploy run. Mirrors the
        existing degrade-to-under-reporting posture."""
        path = trickle_state.trickle_state_path('dark_factory')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'schema_version': trickle_state.STATE_SCHEMA_VERSION,
            'project_id': 'dark_factory',
            'outcome': OUTCOME_PRODUCTIVE,
            'consecutive_barren_runs': 0,
            'last_productive_at': '2026-07-01T03:00:00+00:00',
        }))

        with caplog.at_level('WARNING'):
            status, _ = trickle_state.load_state(path)
            doc = _record(day=2, exit_code=1, selected_count=6)

        assert status == 'ok', 'a missing additive field is not malformed'
        assert not [r for r in caplog.records if r.levelname == 'WARNING']
        assert doc['consecutive_failed_runs'] == 1

    def test_schema_version_is_not_bumped_for_an_additive_field(self):
        """Pins the no-bump decision so a later tidy-up bump has to argue
        with a test. Bumping would make every live state file read
        ``malformed`` on the first post-deploy probe — one guaranteed false
        alarm per project, plus a reset streak — which is the degradation
        this module exists to close."""
        assert trickle_state.STATE_SCHEMA_VERSION == 1

    def test_a_failed_document_round_trips_through_load_state(self):
        doc = _record(day=1, exit_code=1, selected_count=6)
        status, loaded = trickle_state.load_state(
            trickle_state.trickle_state_path('dark_factory')
        )
        assert (status, loaded) == ('ok', doc)

    def test_failed_threshold_is_tighter_than_the_barren_one(self):
        """ONE failed night is already owned twice over — by the nightly's
        own per-run escalation and by ``check_trickle_liveness.sh``'s
        ``Result != success`` gate — so firing at 1 would only duplicate
        them. TWO consecutive is the PERSISTENT shape neither per-run
        signal can express."""
        assert trickle_state.DEFAULT_MAX_FAILED_RUNS == 2
        assert (
            trickle_state.DEFAULT_MAX_FAILED_RUNS
            < trickle_state.DEFAULT_MAX_BARREN_RUNS
        )


class TestRecordedAgeHours:
    """``recorded_age_hours(doc)`` — the freshness helper SHARED by
    ``check_trickle_progress.py``'s staleness branch and
    ``check_trickle_health.py``'s barren-edge suppression.

    It exists so those two cannot drift into two readings of "fresh"; the
    three-valued contract below is what lets each caller keep its own
    distinct wording for an input it cannot assess."""

    def test_a_document_stamped_n_hours_ago_reports_approximately_n(self):
        stamp = datetime.now(UTC) - timedelta(hours=30)
        age = trickle_state.recorded_age_hours({'recorded_at': stamp.isoformat()})

        assert age is not None
        # A tolerance, not equality: the wall clock advances between the
        # write above and the read inside the helper.
        assert abs(age - 30) < 0.5

    def test_a_naive_recorded_at_is_read_as_utc(self):
        """Matches the normalization ``check_trickle_progress.py``'s
        staleness branch already did, so the refactor onto this helper is
        provably behaviour-preserving on the one input shape most likely
        to differ."""
        naive = (datetime.now(UTC) - timedelta(hours=10)).replace(tzinfo=None)
        age = trickle_state.recorded_age_hours({'recorded_at': naive.isoformat()})

        assert age is not None
        assert abs(age - 10) < 0.5

    @pytest.mark.parametrize('doc', [
        pytest.param({}, id='recorded_at-absent'),
        pytest.param({'recorded_at': 'not-a-timestamp'}, id='unparseable'),
        pytest.param({'recorded_at': None}, id='recorded_at-None'),
        pytest.param(None, id='not-a-dict'),
        pytest.param('a string', id='a-string'),
    ])
    def test_what_cannot_be_assessed_is_none(self, doc):
        """``None`` means "freshness cannot be assessed" — deliberately NOT
        "old" and NOT "fresh". Each caller decides what that means, which
        is exactly what lets the progress probe keep its own unparseable
        verdict while the health probe treats the same input as
        post-worthy."""
        assert trickle_state.recorded_age_hours(doc) is None
