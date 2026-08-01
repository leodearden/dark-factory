"""Tests for CheckRun/VerifyAttempt — task 2133 verify ε.

``run_verification`` (verify.py) tracked each of the three checks
(test/lint/type) as five parallel scalar locals — 15 locals total — and
wrote the 6-clause pure-timeout-consistency formula (passed / any_timed_out /
pure_timeout_failure) TWICE: once in the first-pass retry loop and once in
the env-recovery branch (task 2048's fix duplicated it there). These two
dataclasses collapse the 15 locals into one ``VerifyAttempt`` and compute the
derived formula in exactly ONE place, so the two call sites can never drift
apart again.

Test coverage:
  step-1: CheckRun — field exposure, .skipped() classmethod, .to_dict() schema
  step-3: VerifyAttempt — derived-property truth table (passed/any_timed_out/
          pure_timeout_failure) + .test/.lint/.type label accessors
"""

from __future__ import annotations

import pytest


class TestCheckRun:
    """step-1: CheckRun exposes its 7 fields, a .skipped() constructor, and
    a .to_dict() that reproduces the pre-refactor runs-dict schema exactly
    (verify.py:3182-3210), including the ``started_at or ''`` normalisation.

    RED today: ``from orchestrator.verify import CheckRun`` raises ImportError.
    """

    def test_fields_exposed(self):
        from orchestrator.verify import CheckRun  # noqa: PLC0415

        run = CheckRun(
            label='test',
            cmd='uv run pytest',
            rc=1,
            output='boom',
            timed_out=True,
            started_at='2026-07-14T00:00:00+00:00',
            duration_secs=1.5,
        )
        assert run.label == 'test'
        assert run.cmd == 'uv run pytest'
        assert run.rc == 1
        assert run.output == 'boom'
        assert run.timed_out is True
        assert run.started_at == '2026-07-14T00:00:00+00:00'
        assert run.duration_secs == 1.5

    def test_skipped_classmethod(self):
        from orchestrator.verify import CheckRun  # noqa: PLC0415

        run = CheckRun.skipped('lint')
        assert run.label == 'lint'
        assert run.cmd is None
        assert run.rc == 0
        assert run.output == ''
        assert run.timed_out is False
        assert run.started_at is None
        assert run.duration_secs == 0.0

    def test_to_dict_schema_keys_exact(self):
        from orchestrator.verify import CheckRun  # noqa: PLC0415

        run = CheckRun(
            label='test',
            cmd='uv run pytest',
            rc=1,
            output='boom',
            timed_out=True,
            started_at='2026-07-14T00:00:00+00:00',
            duration_secs=1.5,
        )
        d = run.to_dict()
        assert set(d) == {
            'label', 'cmd', 'rc', 'output', 'timed_out', 'started_at', 'duration_secs',
            'segments',
        }

    def test_to_dict_started_at_passthrough_when_not_none(self):
        from orchestrator.verify import CheckRun  # noqa: PLC0415

        run = CheckRun(
            label='test', cmd='uv run pytest', rc=0, output='',
            timed_out=False, started_at='2026-07-14T00:00:00+00:00', duration_secs=1.5,
        )
        assert run.to_dict()['started_at'] == '2026-07-14T00:00:00+00:00'

    def test_to_dict_started_at_normalised_to_empty_string_when_skipped(self):
        """Pins byte-parity with the pre-refactor runs-dict entry, which
        stored ``test_started_at or ''`` (verify.py:3189/3198/3207) — a
        skipped check's ``started_at=None`` must serialise as ``''``, not
        ``None``, so ``_persist_attempt_logs``/``_build_summary_payload``
        (which write it straight into JSON) see the same shape as before.
        """
        from orchestrator.verify import CheckRun  # noqa: PLC0415

        run = CheckRun.skipped('lint')
        assert run.to_dict()['started_at'] == ''

    def test_to_dict_all_values_roundtrip(self):
        from orchestrator.verify import CheckRun  # noqa: PLC0415

        run = CheckRun(
            label='type', cmd='uv run pyright', rc=2, output='err',
            timed_out=False, started_at='ts', duration_secs=3.25,
        )
        assert run.to_dict() == {
            'label': 'type',
            'cmd': 'uv run pyright',
            'rc': 2,
            'output': 'err',
            'timed_out': False,
            'started_at': 'ts',
            'duration_secs': 3.25,
            'segments': None,
        }


class TestCheckRunSegments:
    """task 3338: CheckRun carries per-segment execution facts for a segmented run.

    The fallback verify runs `dark-factory-orchestrator.yaml`'s `test_command`
    — an `&&` chain across every subproject — so an unrelated earlier
    subproject's red short-circuits the shell and a task's OWN tests never run
    (esc-3062-2). The fix runs each segment separately, which produces per-
    segment facts that have to reach the triaging agent. They ride on the
    existing `CheckRun`, so `runs = [c.to_dict() for c in attempt.checks]`
    already persists them to `.task/verify/attempt-N.json` with no change to
    `_persist_attempt_logs`/`_build_summary_payload`.

    `segments` is emitted UNCONDITIONALLY (as `None` when the check was not
    run segmented). A conditionally-present key would leave a consumer unable
    to tell "this build predates segments" from "this run was not segmented"
    — an absent-vs-null ambiguity in the very schema whose job is to make
    skipped-vs-passed unambiguous.
    """

    def test_segments_defaults_to_none(self):
        """Every pre-3338 construction site stays valid — it is the LAST field."""
        from orchestrator.verify import CheckRun  # noqa: PLC0415

        run = CheckRun(
            label='test',
            cmd='uv run pytest',
            rc=0,
            output='',
            timed_out=False,
            started_at='ts',
            duration_secs=1.5,
        )
        assert run.segments is None
        assert run.to_dict()['segments'] is None

    def test_skipped_leaves_segments_none(self):
        """A module_config skip ran no command at all, so it ran no segments."""
        from orchestrator.verify import CheckRun  # noqa: PLC0415

        assert CheckRun.skipped('test').segments is None

    def test_segments_round_trip_through_to_dict_unchanged(self):
        """Plain JSON-native `list[dict]`, passed through verbatim.

        Flat dicts rather than a nested dataclass for the same reason the rest
        of this schema is flat: `to_dict()` output is written straight into
        JSON, so anything that needs its own serialisation step is a second
        place for the shape to drift.
        """
        from orchestrator.verify import CheckRun  # noqa: PLC0415

        segments = [
            {
                'index': 1,
                'label': 'shared-1',
                'cwd': 'shared',
                'cmd': 'uv run pytest tests/ --timeout=300',
                'status': 'passed',
                'rc': 0,
                'timed_out': False,
                'duration_secs': 12.5,
                'skip_reason': None,
            },
            {
                'index': 2,
                'label': 'root-2',
                'cwd': '.',
                'cmd': 'uv run --project shared pytest tests/scripts/ --timeout=300',
                'status': 'not_run',
                'rc': None,
                'timed_out': False,
                'duration_secs': 0.0,
                'skip_reason': 'chain wall-clock budget exhausted',
            },
        ]
        run = CheckRun(
            label='test',
            cmd='cd shared && uv run pytest tests/ --timeout=300 && uv run pytest tests/scripts/',
            rc=1,
            output='...',
            timed_out=True,
            started_at='ts',
            duration_secs=13.0,
            segments=segments,
        )
        assert run.to_dict()['segments'] == segments
        # Passed through, not copied-and-rebuilt: a rebuild would be a second
        # definition of the segment shape, free to drift from _run_segmented's.
        assert run.to_dict()['segments'] is segments


def _run(label, rc=0, timed_out=False, cmd='cmd', output='', started_at='ts', duration_secs=1.0):
    """Build a CheckRun with sane defaults, overriding only what a case cares about."""
    from orchestrator.verify import CheckRun  # noqa: PLC0415

    return CheckRun(
        label=label, cmd=cmd, rc=rc, output=output, timed_out=timed_out,
        started_at=started_at, duration_secs=duration_secs,
    )


class TestVerifyAttempt:
    """step-3: VerifyAttempt derives passed/any_timed_out/pure_timeout_failure
    ONCE, generically over `checks` — the single source of truth that
    replaces the two hand-duplicated 6-clause formula copies in
    run_verification (first-pass retry loop verify.py:3009-3020 and the
    env-recovery branch verify.py:3105-3113, added by task 2048's drift fix).

    RED today: ``from orchestrator.verify import VerifyAttempt`` raises ImportError.
    """

    def test_all_passing_including_a_skipped_check(self):
        """(a) Three rc=0 legs, one of them skipped -> clean pass, no timeout signal."""
        from orchestrator.verify import CheckRun, VerifyAttempt  # noqa: PLC0415

        attempt = VerifyAttempt(checks=[
            _run('test', rc=0),
            _run('lint', rc=0),
            CheckRun.skipped('type'),
        ])
        assert attempt.passed is True
        assert attempt.any_timed_out is False
        assert attempt.pure_timeout_failure is False

    def test_single_timed_out_leg_is_a_pure_timeout_failure(self):
        """(b) test times out; lint/type clean -> the env-recovery-wall-clock
        invariant at the single-source seam: a pure timeout computed ONCE
        means `timed_out=(not passed and pure_timeout_failure)` can never be
        False while this is classified as a pure timeout (the 2048 drift
        this task removes structurally)."""
        from orchestrator.verify import VerifyAttempt  # noqa: PLC0415

        attempt = VerifyAttempt(checks=[
            _run('test', rc=1, timed_out=True),
            _run('lint', rc=0),
            _run('type', rc=0),
        ])
        assert attempt.passed is False
        assert attempt.any_timed_out is True
        assert attempt.pure_timeout_failure is True

    def test_real_failure_alongside_a_timeout_poisons_pure_timeout(self):
        """(c) test has a REAL (non-timeout) failure; type times out ->
        pure_timeout_failure must be False, since not every non-zero rc is
        attributable to a timeout."""
        from orchestrator.verify import VerifyAttempt  # noqa: PLC0415

        attempt = VerifyAttempt(checks=[
            _run('test', rc=1, timed_out=False),
            _run('type', rc=1, timed_out=True),
        ])
        assert attempt.passed is False
        assert attempt.any_timed_out is True
        assert attempt.pure_timeout_failure is False

    def test_lint_only_pure_timeout(self):
        """(d) A single-check attempt where that lone check timed out is
        still a pure timeout."""
        from orchestrator.verify import VerifyAttempt  # noqa: PLC0415

        attempt = VerifyAttempt(checks=[_run('lint', rc=1, timed_out=True)])
        assert attempt.pure_timeout_failure is True

    def test_two_timed_out_legs_is_a_pure_timeout_failure(self):
        """(e) Two of three checks time out (test AND type), lint clean ->
        still a pure timeout, and the derived `timed_out_names` list (built
        by run_verification as `[c.label for c in attempt.checks if
        c.timed_out]`) preserves `checks` order rather than e.g. sorting or
        dropping duplicates."""
        from orchestrator.verify import VerifyAttempt  # noqa: PLC0415

        attempt = VerifyAttempt(checks=[
            _run('test', rc=1, timed_out=True),
            _run('lint', rc=0),
            _run('type', rc=1, timed_out=True),
        ])
        assert attempt.passed is False
        assert attempt.any_timed_out is True
        assert attempt.pure_timeout_failure is True
        assert [c.label for c in attempt.checks if c.timed_out] == ['test', 'type']

    def test_by_label_raises_keyerror_for_missing_label(self):
        """A VerifyAttempt built without a 'test' check (a mislabeled/partial
        construction a future caller could make) must fail loudly on
        `.test` rather than raising a bare StopIteration from the
        underlying `next()` call."""
        from orchestrator.verify import VerifyAttempt  # noqa: PLC0415

        attempt = VerifyAttempt(checks=[_run('lint', rc=0)])
        with pytest.raises(KeyError, match="no check labeled 'test'"):
            _ = attempt.test

    def test_label_accessors_return_matching_check(self):
        """.test/.lint/.type return the CheckRun with the matching label —
        the per-leg consumer sites (env-recovery, _summarize_checks call
        sites, xdist gate, VerifyResult construction) read through these
        instead of the old parallel scalar locals."""
        from orchestrator.verify import VerifyAttempt  # noqa: PLC0415

        test_run = _run('test', cmd='uv run pytest')
        lint_run = _run('lint', cmd='uv run ruff')
        type_run = _run('type', cmd='uv run pyright')
        attempt = VerifyAttempt(checks=[test_run, lint_run, type_run])
        assert attempt.test is test_run
        assert attempt.lint is lint_run
        assert attempt.type is type_run
