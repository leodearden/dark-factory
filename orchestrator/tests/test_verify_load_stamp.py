"""Tests for the verify-summary LOAD STAMP — task 3353, ruling D17.

D17's first deliverable: every per-command entry of a verify ``summary.json``
records the host load the command actually ran under, at its START and at its
END. That is what turns the production corpus itself into the measurement —
"wait for a quiet host" stops being a precondition for sizing a budget,
because every run says how quiet its host was.

Two invariants run through every case here, and neither is new: both are
copied from ``verify._psi_cpu_some10_or_none``, the module's existing PSI
consumer, so the stamp cannot drift from its neighbour.

- A DEGRADED component reads ``None``, never ``0.0``. ``0.0`` is a real and
  common reading — it means an idle host — so fabricating it for a failed read
  would make "we could not tell" indistinguishable from "the host was quiet",
  in the one record whose purpose is to tell those apart. The same
  null-on-degraded convention is already in the flake ledger's SQL.
- A telemetry read may never change a gate's verdict (INV-1). The reader
  already fails open by value, so the wrapper is belt to that braces.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest


def _sample(**overrides):
    """A healthy host sample; overrides degrade exactly the component under test."""
    from shared.psi import PsiSample  # noqa: PLC0415

    # dict[str, Any], not the inferred join: a dict mixing floats with bools
    # widens to dict[str, float], and PsiSample's bool/str fields reject that
    # under **-unpacking. The helper is override-driven anyway.
    fields: dict[str, Any] = dict(
        cpu_some10=2.50,
        cpu_some60=1.80,
        mem_some10=1.23,
        mem_full10=0.0,
        io_some10=0.75,
        read_ok=True,
        runqueue_ratio=0.75,
        runqueue_read_ok=True,
    )
    fields.update(overrides)
    return PsiSample(**fields)


class TestLoadSample:
    """``verify._load_sample()`` — ONE dict describing host load at an instant."""

    def test_the_exact_key_set(self):
        """The record is flat and closed: three keys, no nesting.

        Flat for the same reason the rest of this schema is flat — the value is
        written straight into JSON, so anything needing its own serialisation
        step is a second place for the shape to drift.
        """
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        assert set(_load_sample(read=_sample)) == {
            'cpu_some10',
            'cpu_some60',
            'runqueue_ratio',
        }

    def test_a_healthy_sample_carries_floats(self):
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        record = _load_sample(read=_sample)

        assert record['cpu_some10'] == pytest.approx(2.50)
        assert record['cpu_some60'] == pytest.approx(1.80)
        assert record['runqueue_ratio'] == pytest.approx(0.75)
        assert all(isinstance(v, float) for v in record.values())

    def test_an_idle_host_reads_zero_not_null(self):
        """The complement of the convention, and the one a truthiness bug breaks.

        A genuinely quiet host reads 0.0 across the board with every read_ok
        True. Nulling that would discard a successful measurement — and 0.0 is
        the single most interesting reading for a budget census, since it is
        what the "measure on a quiet host" advice was asking for.
        """
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        record = _load_sample(
            read=lambda: _sample(cpu_some10=0.0, cpu_some60=0.0, runqueue_ratio=0.0),
        )

        assert record == {'cpu_some10': 0.0, 'cpu_some60': 0.0, 'runqueue_ratio': 0.0}

    def test_a_degraded_host_reads_null_not_zero(self):
        """``read_ok=False`` governs the HOST component — both CPU windows."""
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        record = _load_sample(read=lambda: _sample(read_ok=False))

        assert record['cpu_some10'] is None
        assert record['cpu_some60'] is None

    def test_a_degraded_host_nulls_only_the_host_component(self):
        """Per-component degradation, mirroring ``shared.psi``'s own contract.

        A host-PSI failure must not discard a runqueue reading that actually
        succeeded — the two are read independently, and collapsing them would
        be the silent-fail-soft shape INV-11 forbids.
        """
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        record = _load_sample(read=lambda: _sample(read_ok=False))

        assert record['runqueue_ratio'] == pytest.approx(0.75)

    def test_a_degraded_runqueue_nulls_only_itself(self):
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        record = _load_sample(read=lambda: _sample(runqueue_read_ok=False))

        assert record['runqueue_ratio'] is None
        assert record['cpu_some10'] == pytest.approx(2.50)
        assert record['cpu_some60'] == pytest.approx(1.80)

    def test_a_degraded_runqueue_reading_of_zero_is_still_null(self):
        """``read_ok=False`` wins over the value, which is the fail-open zero."""
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        record = _load_sample(
            read=lambda: _sample(runqueue_ratio=0.0, runqueue_read_ok=False),
        )

        assert record['runqueue_ratio'] is None

    def test_a_raising_reader_returns_the_all_none_record(self, caplog):
        """INV-1: a telemetry failure may not raise into a verify verdict.

        WARNING, not DEBUG: the reader already fails open BY VALUE via its
        read_ok flags, so reaching this handler at all means the telemetry path
        broke in a way it does not itself model. Swallowing that quietly is the
        silent-degradation shape the tree-wide gate exists to catch.
        """
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        def boom():
            raise RuntimeError('/proc went away')

        with caplog.at_level(logging.WARNING, logger='orchestrator.verify'):
            record = _load_sample(read=boom)

        assert record == {
            'cpu_some10': None,
            'cpu_some60': None,
            'runqueue_ratio': None,
        }
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert '_load_sample' in caplog.text

    def test_a_raising_reader_is_not_a_bare_except(self, caplog):
        """A BaseException — KeyboardInterrupt, SystemExit — must propagate.

        Swallowing those would make a verify unkillable, which is a strictly
        worse failure than a missing telemetry field.
        """
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        def interrupt():
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            _load_sample(read=interrupt)

    @pytest.mark.parametrize(
        'degrade',
        [{}, {'read_ok': False}, {'runqueue_read_ok': False}],
        ids=['healthy', 'host-degraded', 'runqueue-degraded'],
    )
    def test_the_record_is_json_native(self, degrade):
        """It lands in summary.json verbatim, so it must serialise with no step."""
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        record = _load_sample(read=lambda: _sample(**degrade))

        assert json.loads(json.dumps(record)) == record

    def test_the_zero_argument_call_reads_the_live_host(self):
        """The shipped call shape — the default reader is the module's own.

        Values are NOT asserted: this runs on a live host under a verify leg,
        so any threshold would be a guessed number. What is asserted is the
        shape and the never-raise guarantee.
        """
        from orchestrator.verify import _load_sample  # noqa: PLC0415

        record = _load_sample()

        assert set(record) == {'cpu_some10', 'cpu_some60', 'runqueue_ratio'}
        assert all(v is None or isinstance(v, float) for v in record.values())

    def test_there_is_exactly_one_psi_reader_in_the_module(self):
        """INV-5, and D17's explicit "do not write a second PSI reader".

        The stamp reuses the import verify.py already had. Asserted by object
        identity rather than by reading source text, which is the same shape
        the sampler's own re-home guard uses.
        """
        import shared.psi  # noqa: PLC0415

        import orchestrator.verify  # noqa: PLC0415

        assert orchestrator.verify.read_psi_sample is shared.psi.read_psi_sample


# The live orchestrator `test_command` (orchestrator/orchestrator.yaml:5) —
# the dominant case, and the one that makes a single resolved integer a lie:
# it carries NO `-n`, so the worker count is decided by pyproject `addopts`,
# which the verify path never reads.
LIVE_TEST_COMMAND = (
    'uv run --directory orchestrator pytest tests/ --tb=short -q --timeout=300'
)


class TestXdistWorkers:
    """``verify._xdist_workers(cmd, verify_env)`` — two facts, neither guessed.

    D17 asks for "the xdist worker count actually in effect". Measured, the
    dominant live case admits no single answer: the command above carries no
    ``-n``, so the count comes from pyproject ``addopts``, which verify never
    reads. Collapsing that to one integer would mean either guessing (report
    the env value even when nothing says ``auto``) or fabricating (report a
    default) — and a guessed worker count inside a measurement corpus is
    indistinguishable from a measured one a month later, which is the exact
    class of fabricated datum this deliverable exists to remove.

    So: two orthogonal, separately-sourced, independently-nullable facts. The
    census derives the count where it is derivable and says "unknown" where it
    is not.
    """

    def test_the_exact_key_set(self):
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        assert set(_xdist_workers(LIVE_TEST_COMMAND, {})) == {
            'n_flag',
            'auto_num_workers',
        }

    def test_an_explicit_worker_count_is_read_off_the_flag(self):
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        record = _xdist_workers('uv run pytest tests/ -n 8 --timeout=300', {})

        assert record['n_flag'] == '8'

    def test_n_auto_is_reported_verbatim_not_resolved(self):
        """``'auto'`` is what the command SAYS; resolving it here would guess."""
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        assert _xdist_workers('uv run pytest tests/ -n auto', {})['n_flag'] == 'auto'

    def test_the_live_command_carries_no_flag_and_says_so(self):
        """The dominant case: addopts decides, and verify cannot see addopts."""
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        assert _xdist_workers(LIVE_TEST_COMMAND, {})['n_flag'] is None

    def test_a_non_pytest_command_yields_no_flag(self):
        """Mirrors the no-op guard ``apply_pytest_numprocesses`` documents."""
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        assert _xdist_workers('uv run ruff check src/ tests/', {})['n_flag'] is None
        assert _xdist_workers('uv run pyright src/', {})['n_flag'] is None

    def test_a_governed_wrapper_yields_no_flag_rather_than_a_misparse(self):
        """The reason there is no regex here.

        Once cpu-governed, the command is an opaque outer
        ``<exec> -- /bin/bash -c '...'`` string. A regex would happily find the
        inner ``-n 8`` and report it as this command's flag; the structured
        parser reports OPAQUE, which is the truth about what verify can see.
        The stamp is taken BEFORE the wrap for exactly this reason, so this
        case is defence in depth rather than the live path.
        """
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        governed = (
            "/opt/df/cpu-governed-exec.sh -- /bin/bash -c "
            "'uv run pytest tests/ -n 8'"
        )
        assert _xdist_workers(governed, {})['n_flag'] is None

    def test_a_raw_retained_chain_yields_no_flag(self):
        """An `&&` chain has no ONE invocation's flag to report."""
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        chain = 'cd shared && uv run pytest tests/ -n 4 && uv run pytest tests/scripts/'
        assert _xdist_workers(chain, {})['n_flag'] is None

    def test_auto_num_workers_comes_off_the_passed_env(self):
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        record = _xdist_workers(
            'uv run pytest tests/ -n auto', {'PYTEST_XDIST_AUTO_NUM_WORKERS': '6'},
        )

        assert record['auto_num_workers'] == '6'

    def test_auto_num_workers_is_null_when_the_env_does_not_set_it(self):
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        assert _xdist_workers(LIVE_TEST_COMMAND, {})['auto_num_workers'] is None
        assert _xdist_workers(LIVE_TEST_COMMAND, None)['auto_num_workers'] is None

    def test_the_two_facts_are_independent(self):
        """Neither field is derived from the other — that is the whole point.

        The env value is reported even with no ``-n auto`` to consume it, and a
        literal ``-n`` is reported with no env value present. A reader that
        wants the effective count joins them itself and can see when it cannot.
        """
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        env = {'PYTEST_XDIST_AUTO_NUM_WORKERS': '6'}

        assert _xdist_workers(LIVE_TEST_COMMAND, env) == {
            'n_flag': None,
            'auto_num_workers': '6',
        }
        assert _xdist_workers('uv run pytest tests/ -n 8', {}) == {
            'n_flag': '8',
            'auto_num_workers': None,
        }

    @pytest.mark.parametrize(
        'cmd',
        ['', '   ', 'uv run pytest tests/ -n', "pytest 'unclosed", '&&', '-n 8'],
        ids=['empty', 'blank', 'dangling-n', 'unbalanced-quote', 'bare-op', 'bare-flag'],
    )
    def test_a_malformed_command_never_raises(self, cmd):
        """Same never-raise contract as ``_load_sample``: nulls, not an exception.

        ``-n`` with no following token is the interesting one — the structured
        splitter falls back to bare-flag classification rather than indexing
        past the end, so there is no value to report and none is invented.
        """
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        record = _xdist_workers(cmd, {})

        assert set(record) == {'n_flag', 'auto_num_workers'}
        assert record['n_flag'] is None

    def test_the_record_is_json_native(self):
        from orchestrator.verify import _xdist_workers  # noqa: PLC0415

        record = _xdist_workers(
            'uv run pytest tests/ -n 8', {'PYTEST_XDIST_AUTO_NUM_WORKERS': '6'},
        )

        assert json.loads(json.dumps(record)) == record


class TestLoadReachesTheSummaryPayload:
    """``_build_summary_payload`` rebuilds each entry from an explicit WHITELIST.

    Not a passthrough — so a field absent from that list is silently dropped.
    Its own docstring records the time that happened to ``segments``, leaving
    the one structured record of which segments never ran available only as
    free text inside the aggregated ``output`` blob. ``load`` is the same
    shape of field landing in the same place, so it gets the same guard.
    """

    def _load(self, cpu=2.5):
        return {
            'start': {'cpu_some10': cpu, 'cpu_some60': 1.8, 'runqueue_ratio': 0.75},
            'end': {'cpu_some10': 9.5, 'cpu_some60': 4.0, 'runqueue_ratio': 2.25},
            'xdist': {'n_flag': None, 'auto_num_workers': '6'},
        }

    def _run(self, label='test', **overrides):
        run = {
            'label': label,
            'cmd': f'uv run {label}',
            'rc': 0,
            'output': '',
            'timed_out': False,
            'started_at': 'ts',
            'duration_secs': 1.5,
            'segments': None,
            'load': self._load(),
        }
        run.update(overrides)
        return run

    def test_every_command_entry_carries_its_load(self):
        from orchestrator.verify import _build_summary_payload  # noqa: PLC0415

        payload = _build_summary_payload(
            [self._run('test'), self._run('lint', load=self._load(cpu=7.5))],
            'clean',
            '',
        )

        assert [e['load'] for e in payload['commands']] == [
            self._load(),
            self._load(cpu=7.5),
        ]

    def test_a_run_dict_with_no_load_key_serialises_as_null(self):
        """`.get`, not `[]` — the remote merge-verify path hand-builds run dicts.

        Those predate the key, so indexing would raise a KeyError on the merge
        path. This is the exact accommodation `segments` already documents for
        the same callers.
        """
        from orchestrator.verify import _build_summary_payload  # noqa: PLC0415

        legacy = self._run()
        del legacy['load']

        payload = _build_summary_payload([legacy], 'clean', '')

        assert payload['commands'][0]['load'] is None

    def test_load_is_present_on_every_entry_even_when_null(self):
        """Unconditional: absent-vs-null is the ambiguity the key exists to kill."""
        from orchestrator.verify import _build_summary_payload  # noqa: PLC0415

        legacy = self._run()
        del legacy['load']

        payload = _build_summary_payload([legacy, self._run('lint')], 'clean', '')

        assert all('load' in entry for entry in payload['commands'])

    def test_a_skipped_leg_contributes_no_entry_at_all(self):
        """Unchanged behaviour — `cmd is None` is filtered before the rebuild."""
        from orchestrator.verify import _build_summary_payload  # noqa: PLC0415

        payload = _build_summary_payload(
            [self._run('test'), self._run('type', cmd=None, load=None)],
            'clean',
            '',
        )

        assert [e['label'] for e in payload['commands']] == ['test']

    def test_the_payload_is_json_native(self):
        from orchestrator.verify import _build_summary_payload  # noqa: PLC0415

        payload = _build_summary_payload([self._run()], 'clean', '')

        assert json.loads(json.dumps(payload)) == payload


# The stamp is only worth anything if it is TAKEN. Everything above proves the
# record serialises; these drive `run_verification` for real against a tmp_path
# worktree and read the summary.json it WRITES.
#
# That distinction is not hypothetical here. Every test in
# test_verify_segmented_fallback.py once spied on `_persist_attempt_logs`'
# ARGUMENT, and that seam cannot see `_build_summary_payload`'s key whitelist
# dropping a field on the way to disk — which is exactly what happened to
# `segments`: the docstrings claimed the JSON and the tests asserting the JSON
# were asserting the argument.

_CHAIN_TEST_COMMAND = (
    'cd shared && uv run pytest tests/ -q && uv run --project shared pytest tests/scripts/ -q'
)


def _module_config(*, test_command, type_check_command=None):
    """A `__fallback__` module config; type is left SKIPPED by default."""
    from orchestrator.config import ModuleConfig  # noqa: PLC0415

    return ModuleConfig(
        prefix='__fallback__',
        test_command=test_command,
        lint_command='uv run ruff check src/',
        type_check_command=type_check_command,
    )


async def _run_and_read_summary(tmp_path, *, segmented, reader):
    """Drive the REAL `run_verification` and return (result, parsed summary.json).

    *reader* is handed to the REAL `_load_sample`, so its never-raise wrapper
    and its read_ok -> None mapping are the ones under test here — only the
    /proc read itself is replaced. Patching `verify.read_psi_sample` would NOT
    work: `_load_sample`'s `read=` default is bound at def time, so a later
    patch of the module attribute never reaches it.
    """
    from unittest.mock import patch  # noqa: PLC0415

    from orchestrator import verify as verify_mod  # noqa: PLC0415
    from orchestrator.config import OrchestratorConfig  # noqa: PLC0415

    (tmp_path / '.task').mkdir(parents=True, exist_ok=True)
    real_load_sample = verify_mod._load_sample

    def scripted_load_sample():
        return real_load_sample(read=reader)

    async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **_kw):
        return 0, 'ok', False

    config = OrchestratorConfig(
        project_root=tmp_path, verify_admission_enabled=False,
    )
    with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd), \
         patch('orchestrator.verify._load_sample', side_effect=scripted_load_sample):
        result = await verify_mod.run_verification(
            tmp_path,
            config,
            _module_config(
                test_command=_CHAIN_TEST_COMMAND if segmented else 'uv run pytest tests/ -n 8',
            ),
            attempt_id=1,
            # Load-bearing, not decoration: run_verification persists only when
            # attempt_id AND task_id are both set.
            task_id='3353',
            max_retries=0,
            segment_chained_test=segmented,
        )

    summary_path = tmp_path / '.task' / 'verify' / 'attempt-1.__fallback__.summary.json'
    assert summary_path.exists(), (
        f'run_verification persisted no summary JSON at {summary_path}; wrote '
        f'{sorted(p.name for p in (tmp_path / ".task" / "verify").glob("*"))}'
    )
    return result, json.loads(summary_path.read_text(encoding='utf-8'))


def _rising_reader():
    """A reader whose every call reads a BUSIER host than the last.

    Monotonic rather than a fixed pair, which is what makes the start/end
    assertions safe under the concurrent test/lint/type gather: the legs
    interleave their samples arbitrarily, but a single leg's start is always
    drawn before its own end.
    """
    import itertools  # noqa: PLC0415

    counter = itertools.count(1)

    def read():
        n = float(next(counter))
        return _sample(cpu_some10=n, cpu_some60=n / 2, runqueue_ratio=n / 4)

    return read


def _raising_reader():
    def read():
        raise OSError('/proc/pressure/cpu went away mid-verify')

    return read


@pytest.mark.parametrize('segmented', [False, True], ids=['unsegmented', 'segmented'])
class TestTheStampIsTakenOnTheRealPath:
    """INV-10, and the guard D17 names.

    Parametrized over BOTH execution branches of `_run_or_skip_timed` — the
    plain one and the `&&`-chain segmented one — so the stamp cannot land on
    one path only. That asymmetry is not imagined: the `-n` cap sitting three
    lines away WAS silently dropped on the segmented path, because the rewrite
    landed on `cmd` while segments were built from `config_cmd`, and task 3478
    existed to remove it.
    """

    @staticmethod
    def _stamped(summary):
        return [c for c in summary['commands'] if c['cmd'] is not None]

    @pytest.mark.asyncio
    async def test_every_command_that_ran_carries_a_start_and_an_end(
        self, tmp_path, segmented,
    ):
        _result, summary = await _run_and_read_summary(
            tmp_path, segmented=segmented, reader=_rising_reader(),
        )

        entries = self._stamped(summary)
        assert entries, 'no command entry reached the written summary at all'
        for entry in entries:
            assert entry['load'] is not None, (
                f"{entry['label']}: load never reached disk — "
                '_build_summary_payload rebuilds from a key whitelist'
            )
            assert set(entry['load']) == {'start', 'end', 'xdist'}

    @pytest.mark.asyncio
    async def test_the_start_and_end_are_two_distinct_samples(
        self, tmp_path, segmented,
    ):
        """The point of the pair: a budget reader wants the load DURING the run.

        A single sample reused for both would serialise identically and satisfy
        every shape assertion above, so this is the one that catches it.
        """
        _result, summary = await _run_and_read_summary(
            tmp_path, segmented=segmented, reader=_rising_reader(),
        )

        for entry in self._stamped(summary):
            start, end = entry['load']['start'], entry['load']['end']
            assert start != end, f"{entry['label']}: one sample reused for both ends"
            assert start['cpu_some10'] < end['cpu_some10'], (
                f"{entry['label']}: end sample was drawn before the start sample"
            )

    @pytest.mark.asyncio
    async def test_the_xdist_facts_ride_along(self, tmp_path, segmented):
        _result, summary = await _run_and_read_summary(
            tmp_path, segmented=segmented, reader=_rising_reader(),
        )

        for entry in self._stamped(summary):
            assert set(entry['load']['xdist']) == {'n_flag', 'auto_num_workers'}

    @pytest.mark.asyncio
    async def test_a_skipped_leg_contributes_no_entry(self, tmp_path, segmented):
        """Unchanged behaviour: the type leg has no command, so there is nothing
        to stamp and no entry to stamp it on."""
        _result, summary = await _run_and_read_summary(
            tmp_path, segmented=segmented, reader=_rising_reader(),
        )

        assert 'type' not in {c['label'] for c in summary['commands']}

    @pytest.mark.asyncio
    async def test_the_written_summary_is_still_json(self, tmp_path, segmented):
        _result, summary = await _run_and_read_summary(
            tmp_path, segmented=segmented, reader=_rising_reader(),
        )

        assert json.loads(json.dumps(summary)) == summary


@pytest.mark.parametrize('segmented', [False, True], ids=['unsegmented', 'segmented'])
class TestTelemetryFailureDoesNotChangeTheVerdict:
    """INV-1 / INV-11 on the REAL path: a broken /proc may not fail a verify.

    The verdict is compared against the healthy run's verdict rather than
    against a hardcoded expectation, so this stays a statement about EQUALITY
    under degradation — it cannot pass by both runs being broken in the same
    new way.
    """

    @pytest.mark.asyncio
    async def test_the_verdict_is_identical_with_a_broken_reader(
        self, tmp_path, segmented,
    ):
        healthy, healthy_summary = await _run_and_read_summary(
            tmp_path / 'healthy', segmented=segmented, reader=_rising_reader(),
        )
        degraded, degraded_summary = await _run_and_read_summary(
            tmp_path / 'degraded', segmented=segmented, reader=_raising_reader(),
        )

        assert (degraded.passed, degraded.timed_out, degraded.category) == (
            healthy.passed, healthy.timed_out, healthy.category,
        )
        assert degraded_summary['rc'] == healthy_summary['rc']
        assert degraded_summary['category'] == healthy_summary['category']
        assert degraded_summary['timed_out'] == healthy_summary['timed_out']

    @pytest.mark.asyncio
    async def test_the_load_fields_are_null_throughout_rather_than_zero(
        self, tmp_path, segmented,
    ):
        """Null, not 0.0 — a broken read must not render as an idle host."""
        _result, summary = await _run_and_read_summary(
            tmp_path, segmented=segmented, reader=_raising_reader(),
        )

        entries = [c for c in summary['commands'] if c['cmd'] is not None]
        assert entries
        for entry in entries:
            for end in ('start', 'end'):
                assert entry['load'][end] == {
                    'cpu_some10': None,
                    'cpu_some60': None,
                    'runqueue_ratio': None,
                }

    @pytest.mark.asyncio
    async def test_the_record_is_still_structurally_complete(
        self, tmp_path, segmented,
    ):
        """Degraded means "nulls inside", never "the key went missing"."""
        _result, summary = await _run_and_read_summary(
            tmp_path, segmented=segmented, reader=_raising_reader(),
        )

        for entry in [c for c in summary['commands'] if c['cmd'] is not None]:
            assert set(entry['load']) == {'start', 'end', 'xdist'}
