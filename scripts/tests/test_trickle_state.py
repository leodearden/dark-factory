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
            applied=0, commit_made=False, budget_suppressed=False,
            capped=False, **counters):
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
        capped=capped,
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

    def test_a_capped_run_never_restamps_last_productive_at(self):
        """A usage-cap DEFERRED night built digests and coded none of them,
        so it reads productive while doing nothing."""
        prod = _record(day=1, selected_count=2)
        stamp = prod['last_productive_at']

        capped = _record(day=2, capped=True, selected_count=6)

        assert capped['outcome'] == OUTCOME_PRODUCTIVE
        assert capped['last_productive_at'] == stamp

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


def _simulate_missed_elapses(missed, *, nights):
    """Drive the backlog policy over a systemd outage with no I/O.

    The last scheduled pass covered ``_d(1)``; the timer then misses
    *missed* elapses and its catch-up covers ``_d(1 + missed)``. Each of
    *nights* finished scheduled passes, from the catch-up on, covers the
    next date, is recorded the next day, owes the dates since the previous
    scheduled pass, and folds a finished retry of each date due. Returns
    the dates retried in order, every abandoned entry, and the final
    backlog."""
    cursor = _d(1)
    catch_up = cursor + timedelta(days=missed)
    backlog = {}
    retried = []
    abandoned = {}
    for night in range(nights):
        target = catch_up + timedelta(days=night)
        today = target + timedelta(days=1)
        fold = _fold(
            backlog, target_date=target, owes_retry=False, today=today,
            never_run=trickle_state.never_run_dates(cursor, target_date=target),
        )
        cursor = target
        backlog = fold.backlog
        abandoned.update(fold.abandoned)
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
            last_scheduled, target_date=target_date) == ()

    def test_the_never_run_dates_are_strictly_inside_the_gap_oldest_first(
        self,
    ):
        assert trickle_state.never_run_dates(
            _d(1), target_date=_d(4)) == (_d(2), _d(3))

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


def _without_backlog(doc):
    return {k: v for k, v in doc.items() if k != 'retry_backlog'}


class TestRecordRunRetryBacklog:
    """record_run folds each scheduled pass into the document's additive
    ``retry_backlog`` ({ISO date: unfinished attempts})."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_a_failed_run_owes_its_own_date_and_a_repeat_counts_again(self):
        first = _record(day=1, exit_code=1, selected_count=6)
        again = _record(day=1, exit_code=1, selected_count=6)

        assert first['retry_backlog'] == {'2026-07-01': 1}
        assert again['retry_backlog'] == {'2026-07-01': 2}

    def test_a_capped_run_owes_its_date_and_records_capped(self):
        capped = _record(day=1, capped=True, zero_signal_dropped=2)

        assert capped['exit_code'] == 0
        assert capped['capped'] is True
        assert capped['retry_backlog'] == {'2026-07-01': 1}

    def test_a_normal_run_records_capped_false_and_owes_nothing(self):
        doc = _record(day=1, selected_count=1)

        assert doc['capped'] is False
        assert doc['retry_backlog'] == {}

    def test_a_run_that_committed_never_owes_its_date(self):
        doc = _record(day=1, exit_code=1, selected_count=6, commit_made=True)

        assert doc['retry_backlog'] == {}

    def test_a_finished_run_clears_its_own_date_and_carries_the_rest(self):
        _record(day=1, exit_code=1)
        _record(day=2, exit_code=1)

        cleared = _record(day=1, selected_count=1)
        carried = _record(day=3, selected_count=1)

        assert cleared['retry_backlog'] == {'2026-07-02': 1}
        assert carried['retry_backlog'] == {'2026-07-02': 1}

    def test_an_entry_past_the_window_is_pruned_with_one_warning(self, caplog):
        _record(day=1, exit_code=1)

        with caplog.at_level('WARNING'):
            doc = _record(day=2 + trickle_state.RETRY_MAX_AGE_DAYS,
                          selected_count=1)

        assert doc['retry_backlog'] == {}
        assert len(_warnings_naming(caplog, '2026-07-01')) == 1

    def test_a_document_without_a_backlog_reads_ok_and_starts_one(
        self, caplog
    ):
        path = trickle_state.trickle_state_path('dark_factory')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            'schema_version': trickle_state.STATE_SCHEMA_VERSION,
            'project_id': 'dark_factory',
            'outcome': OUTCOME_PRODUCTIVE,
            'consecutive_barren_runs': 0,
            'consecutive_failed_runs': 0,
        }))

        with caplog.at_level('WARNING'):
            status, _ = trickle_state.load_state(path)
            doc = _record(day=2, exit_code=1)

        assert status == 'ok'
        assert not [r for r in caplog.records if r.levelname == 'WARNING']
        assert doc['retry_backlog'] == {'2026-07-02': 1}

    @pytest.mark.parametrize('garbage', [
        pytest.param(['2026-07-01'], id='a-list'),
        pytest.param({'yesterday': 1, '2026-07-01': 'x'}, id='bad-entries'),
        pytest.param('2026-07-01', id='a-string'),
    ])
    def test_a_garbage_backlog_is_dropped_never_raised(self, garbage):
        doc = _record(day=1, selected_count=1)
        path = trickle_state.trickle_state_path('dark_factory')
        path.write_text(json.dumps({**doc, 'retry_backlog': garbage}))

        after = _record(day=2, exit_code=1)

        assert after['retry_backlog'] == {'2026-07-02': 1}


def _retry(project_id='dark_factory', *, day, recorded_day, exit_code=0,
           capped=False, commit_made=False):
    return trickle_state.record_retry(
        project_id,
        target_date=date(2026, 7, day),
        recorded_at=datetime(2026, 7, recorded_day, 3, 30, 0, tzinfo=UTC),
        exit_code=exit_code,
        capped=capped,
        commit_made=commit_made,
    )


def _retried(**kwargs) -> dict:
    """:func:`_retry` over a readable predecessor, so a document is written."""
    doc = _retry(**kwargs)
    assert doc is not None, 'record_retry wrote nothing'
    return doc


class TestRecordRetry:
    """record_retry rewrites ONLY ``retry_backlog``: the rest of the
    document keeps describing the scheduled night."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_a_finished_retry_clears_its_date_and_changes_nothing_else(self):
        _record(day=1, exit_code=1, selected_count=6)
        scheduled = _record(day=2, selected_count=1)

        retried = _retried(day=1, recorded_day=2)

        path = trickle_state.trickle_state_path('dark_factory')
        assert json.loads(path.read_text()) == retried
        assert scheduled['retry_backlog'] == {'2026-07-01': 1}
        assert retried['retry_backlog'] == {}
        assert _without_backlog(retried) == _without_backlog(scheduled)

    def test_an_unfinished_retry_counts_an_attempt(self):
        _record(day=1, exit_code=1)
        scheduled = _record(day=2, selected_count=1)

        retried = _retried(day=1, recorded_day=2, exit_code=1)

        assert retried['retry_backlog'] == {'2026-07-01': 2}
        assert _without_backlog(retried) == _without_backlog(scheduled)

    def test_a_capped_retry_counts_an_attempt(self):
        _record(day=1, exit_code=1)
        _record(day=2, selected_count=1)

        retried = _retried(day=1, recorded_day=2, capped=True)

        assert retried['retry_backlog'] == {'2026-07-01': 2}

    def test_the_attempt_bound_abandons_the_date_with_one_warning(
        self, caplog
    ):
        _record(day=1, exit_code=1)
        _record(day=2, selected_count=1)
        unfinished = trickle_state.RETRY_MAX_ATTEMPTS - 1

        with caplog.at_level('WARNING'):
            for night in range(unfinished):
                doc = _retried(day=1, recorded_day=2 + night, exit_code=1)

        assert doc['retry_backlog'] == {}
        assert len(_warnings_naming(caplog, '2026-07-01')) == 1

    def test_with_no_predecessor_nothing_is_written(self, caplog):
        path = trickle_state.trickle_state_path('dark_factory')

        with caplog.at_level('WARNING'):
            doc = _retry(day=1, recorded_day=2)

        assert doc is None
        assert not path.exists()
        assert len(_warnings_naming(caplog, '2026-07-01')) == 1

    def test_with_a_corrupt_predecessor_nothing_is_written(self, caplog):
        path = trickle_state.trickle_state_path('dark_factory')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{corrupt')

        with caplog.at_level('WARNING'):
            doc = _retry(day=1, recorded_day=2)

        assert doc is None
        assert path.read_bytes() == b'{corrupt'
        assert len(_warnings_naming(caplog, '2026-07-01')) == 1

    def test_the_write_is_atomic_and_leaves_no_tmp_file(self):
        _record(day=1, exit_code=1)
        _record(day=2, selected_count=1)

        _retry(day=1, recorded_day=2)

        state_dir = trickle_state.trickle_state_path('dark_factory').parent
        assert [p.name for p in state_dir.iterdir()] == [
            trickle_state.STATE_FILENAME
        ]


class TestDueRetries:
    """``due_retries(doc, *, today)`` — the reader the nightly calls, so the
    backlog's field name and encoding stay inside trickle_state."""

    @pytest.fixture(autouse=True)
    def _state_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    def test_a_recorded_failed_night_is_due_after_a_productive_one(self):
        _record(day=1, exit_code=1, selected_count=6)
        doc = _record(day=2, selected_count=1)

        assert trickle_state.due_retries(doc, today=_d(3)) == (_d(1),)

    def test_several_owed_dates_give_the_oldest_up_to_the_nightly_limit(self):
        _record(day=1, exit_code=1)
        _record(day=2, exit_code=1)
        doc = _record(day=3, selected_count=1)

        due = trickle_state.due_retries(doc, today=_d(4))

        assert due == (_d(1), _d(2))[:trickle_state.RETRIES_PER_NIGHT]

    @pytest.mark.parametrize('doc', [
        pytest.param(None, id='None'),
        pytest.param(['2026-07-01'], id='not-a-dict'),
        pytest.param({'outcome': OUTCOME_PRODUCTIVE}, id='no-backlog-key'),
        pytest.param({'retry_backlog': 'garbage'}, id='garbage-backlog'),
        pytest.param(
            {'retry_backlog': {'2026-07-01': 1}}, id='outside-the-window'
        ),
    ])
    def test_nothing_is_due_from_what_owes_nothing_today(self, doc):
        today = _d(1) + timedelta(days=trickle_state.RETRY_MAX_AGE_DAYS + 1)

        assert trickle_state.due_retries(doc, today=today) == ()
