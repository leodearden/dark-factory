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

import pytest


def _sample(**overrides):
    """A healthy host sample; overrides degrade exactly the component under test."""
    from shared.psi import PsiSample  # noqa: PLC0415

    fields = dict(
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
