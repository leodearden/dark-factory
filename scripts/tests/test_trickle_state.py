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

import tempfile
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
    """Where the state file lives — and, just as load-bearing, where it
    does NOT.

    Rooted at ``${XDG_STATE_HOME:-~/.local/state}/dark-factory/legibility/
    <project_id>/trickle-state.json``, re-deriving
    ``orchestrator.mcp_lifecycle.managed_runtime_data_dirs``'s scheme
    (task 2439). Never under ``docs/legibility/``: that path is
    git-TRACKED, and a file rewritten EVERY night on a tracked path would
    either leave the machine-operated project_root checkout permanently
    dirty or force a nightly commit — which would make "the repo has a
    commit today" a valid liveness signal and CONTRADICT PRD decision 7
    outright.
    """

    def test_uses_xdg_state_home_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))
        result = trickle_state.trickle_state_path('dark_factory')
        assert result == (
            tmp_path / 'dark-factory' / 'legibility' / 'dark_factory'
            / 'trickle-state.json'
        )

    def test_falls_back_under_home_local_state(self, tmp_path, monkeypatch):
        monkeypatch.delenv('XDG_STATE_HOME', raising=False)
        fake_home = tmp_path / 'home' / 'someone'
        monkeypatch.setattr(Path, 'home', classmethod(lambda cls: fake_home))

        result = trickle_state.trickle_state_path('dark_factory')

        assert result == (
            fake_home / '.local' / 'state' / 'dark-factory' / 'legibility'
            / 'dark_factory' / 'trickle-state.json'
        )

    def test_empty_xdg_state_home_is_treated_as_unset(self, tmp_path, monkeypatch):
        """An empty env var is the classic systemd `Environment=` foot-gun:
        `XDG_STATE_HOME=` set-but-empty must fall back, not resolve to the
        filesystem root."""
        monkeypatch.setenv('XDG_STATE_HOME', '')
        fake_home = tmp_path / 'home' / 'someone'
        monkeypatch.setattr(Path, 'home', classmethod(lambda cls: fake_home))

        result = trickle_state.trickle_state_path('dark_factory')

        assert result == (
            fake_home / '.local' / 'state' / 'dark-factory' / 'legibility'
            / 'dark_factory' / 'trickle-state.json'
        )

    def test_distinct_projects_never_collide(self, tmp_path, monkeypatch):
        monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))
        a = trickle_state.trickle_state_path('dark_factory')
        b = trickle_state.trickle_state_path('reify')
        assert a != b
        assert a.parent != b.parent

    def test_no_home_degrades_to_tempdir_instead_of_raising(
        self, tmp_path, monkeypatch, caplog
    ):
        """A stripped daemon/CI environment with no HOME (and no pwd entry)
        makes Path.home() raise RuntimeError. The probe must still produce
        a verdict, never a traceback — mirroring
        managed_runtime_data_dirs' identical degradation (task 2439
        amendment)."""
        monkeypatch.delenv('XDG_STATE_HOME', raising=False)

        def _no_home(cls):
            raise RuntimeError('Could not determine home directory.')

        monkeypatch.setattr(Path, 'home', classmethod(_no_home))

        with caplog.at_level('WARNING'):
            result = trickle_state.trickle_state_path('dark_factory')

        assert result == (
            Path(tempfile.gettempdir()) / 'dark-factory' / 'legibility'
            / 'dark_factory' / 'trickle-state.json'
        )
        assert any(r.levelname == 'WARNING' for r in caplog.records), (
            'the temp-dir degradation must be announced, not silent'
        )

    def test_state_path_is_outside_any_repo_checkout(self, tmp_path, monkeypatch):
        """The location property that motivated the whole choice: the state
        file must never dirty a machine-operated checkout, and must never
        become a git signal."""
        monkeypatch.delenv('XDG_STATE_HOME', raising=False)
        fake_home = tmp_path / 'home' / 'someone'
        monkeypatch.setattr(Path, 'home', classmethod(lambda cls: fake_home))

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
