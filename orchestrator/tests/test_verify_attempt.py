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
        }
