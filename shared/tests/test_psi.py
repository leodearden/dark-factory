"""Tests for shared.psi — the re-homed PSI parser/reader (PRD DA1, DA-D9 reuse).

parse_pressure_file is re-homed VERBATIM from sampler/src/sampler/metrics.py
(behavior-identical) — see that module's sibling test suite
(sampler/tests/test_load_metrics.py) for the original regression guard. The
fixture text literals below reuse the same PSI_CPU_TEXT/PSI_MEM_TEXT/PSI_IO_TEXT
shapes so both suites pin the same parsing contract.
"""

from __future__ import annotations

import types

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures (same shapes as sampler/tests/test_load_metrics.py)
# ---------------------------------------------------------------------------

# Realistic /proc/pressure/* text with both some and full lines
PSI_CPU_TEXT = (
    'some avg10=2.50 avg60=1.80 avg300=1.20 total=123456\n'
    'full avg10=0.30 avg60=0.20 avg300=0.10 total=9876\n'
)

# Memory pressure — only 'some' line present (full missing)
PSI_MEM_TEXT = 'some avg10=1.23 avg60=0.90 avg300=0.50 total=654321\n'

# IO pressure — both lines
PSI_IO_TEXT = (
    'some avg10=0.75 avg60=0.60 avg300=0.40 total=11111\n'
    'full avg10=0.45 avg60=0.30 avg300=0.20 total=2222\n'
)


class TestParsePressureFileRehome:
    def test_both_lines_extracted(self):
        from shared.psi import parse_pressure_file

        result = parse_pressure_file(PSI_CPU_TEXT)
        assert result == {'some_avg10': 2.50, 'full_avg10': 0.30}

    def test_missing_full_defaults_to_zero(self):
        from shared.psi import parse_pressure_file

        result = parse_pressure_file(PSI_MEM_TEXT)
        assert result == {'some_avg10': 1.23, 'full_avg10': 0.0}

    def test_io_both_lines(self):
        from shared.psi import parse_pressure_file

        result = parse_pressure_file(PSI_IO_TEXT)
        assert result == {'some_avg10': 0.75, 'full_avg10': 0.45}

    def test_float_precision(self):
        from shared.psi import parse_pressure_file

        text = 'some avg10=99.99 avg60=0.00 avg300=0.00 total=0\n'
        result = parse_pressure_file(text)
        assert result is not None
        assert result['some_avg10'] == pytest.approx(99.99)
        assert result['full_avg10'] == 0.0

    def test_total_miss_returns_none(self):
        """A total parse miss (no some/full avg10 line) must return None sentinel."""
        from shared.psi import parse_pressure_file

        # Total miss — garbage text with no recognisable avg10 fields
        assert parse_pressure_file('garbage line with no avg fields\n') is None
        # Empty string — also a total miss
        assert parse_pressure_file('') is None

    def test_partial_miss_is_not_none(self):
        """Partial miss (some present, full absent) is a legitimate kernel state, not None."""
        from shared.psi import parse_pressure_file

        assert parse_pressure_file(PSI_MEM_TEXT) is not None


class TestPsiSampleV2Fields:
    """PRD `plans/load-throttle-harmonisation-prd.md` §6.1 — the v2 field surface.

    The five v2 fields are appended WITH defaults so every shipped keyword
    construction is unaffected (census: all 17 repo-wide are keyword).
    """

    V2_FIELD_ORDER = (
        'cpu_some10',
        'mem_some10',
        'mem_full10',
        'io_some10',
        'read_ok',
        'runqueue_ratio',
        'runqueue_read_ok',
        'own_cpu_some10',
        'own_cgroup',
        'own_read_ok',
    )

    def test_v1_construction_still_works_and_v2_fields_default(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=10.0,
            mem_some10=5.0,
            mem_full10=1.0,
            io_some10=5.0,
            read_ok=True,
        )
        assert sample.runqueue_ratio == 0.0
        assert sample.runqueue_read_ok is False
        assert sample.own_cpu_some10 == 0.0
        assert sample.own_cgroup == ''
        assert sample.own_read_ok is False

    def test_v2_fields_settable_by_keyword(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=10.0,
            mem_some10=5.0,
            mem_full10=1.0,
            io_some10=5.0,
            read_ok=True,
            runqueue_ratio=4.3,
            runqueue_read_ok=True,
            own_cpu_some10=55.0,
            own_cgroup='/df.slice/df-x.slice',
            own_read_ok=True,
        )
        assert sample.runqueue_ratio == pytest.approx(4.3)
        assert sample.runqueue_read_ok is True
        assert sample.own_cpu_some10 == pytest.approx(55.0)
        assert sample.own_cgroup == '/df.slice/df-x.slice'
        assert sample.own_read_ok is True

    def test_v2_instance_stays_frozen(self):
        import dataclasses

        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=10.0,
            mem_some10=5.0,
            mem_full10=1.0,
            io_some10=5.0,
            read_ok=True,
            runqueue_ratio=4.3,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample.runqueue_ratio = 9.0  # type: ignore[misc]

    def test_field_order_appends_v2_fields_last(self):
        """The five v1 fields keep their positional slots (§6.1)."""
        import dataclasses

        from shared.psi import PsiSample

        names = tuple(f.name for f in dataclasses.fields(PsiSample))
        assert names == self.V2_FIELD_ORDER


def _saturation_cfg():
    """Duck-typed cfg stub in the v2 CODE-DEFAULT shape (PRD §6.2).

    ``cpu_some_avg10``, ``runqueue_ratio`` and ``own_cpu_some_avg10`` are
    ``None`` — off by code default (D1, D2). A test that needs an arm switched
    on says so explicitly via ``_configured_cfg``, which is why the 85.0 that
    used to live here now appears only where an operator-set cpu arm is the
    thing under test.
    """
    return types.SimpleNamespace(
        cpu_some_avg10=None,
        mem_some_avg10=15.0,
        mem_full_avg10=3.0,
        io_some_avg10=40.0,
        runqueue_ratio=None,
        own_cpu_some_avg10=None,
    )


def _configured_cfg(**overrides):
    """The v2 default stub with the named arms switched on."""
    cfg = _saturation_cfg()
    for name, value in overrides.items():
        assert hasattr(cfg, name), name
        setattr(cfg, name, value)
    return cfg


def _healthy_sample(**overrides):
    """A host-readable all-quiet sample; overrides raise the arm under test."""
    from shared.psi import PsiSample

    fields = dict(
        cpu_some10=0.0,
        mem_some10=0.0,
        mem_full10=0.0,
        io_some10=0.0,
        read_ok=True,
        runqueue_ratio=0.0,
        runqueue_read_ok=True,
        own_cpu_some10=0.0,
        own_cgroup='/df.slice/df-x.slice',
        own_read_ok=True,
    )
    fields.update(overrides)
    return PsiSample(**fields)


def vars_of(sample):
    """The sample's fields as a plain dict, for building a variant of it."""
    import dataclasses

    return {f.name: getattr(sample, f.name) for f in dataclasses.fields(sample)}


class TestPsiSampleSaturated:
    def test_frozen(self):
        import dataclasses

        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=10.0,
            mem_some10=5.0,
            mem_full10=1.0,
            io_some10=5.0,
            read_ok=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample.cpu_some10 = 99.0  # type: ignore[misc]

    def test_cpu_some_trips_when_an_operator_configures_it(self):
        """D1 turns the cpu arm off by default; it does not remove it."""
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=85.0,
            mem_some10=10.0,
            mem_full10=2.0,
            io_some10=30.0,
            read_ok=True,
        )
        assert sample.saturated(_configured_cfg(cpu_some_avg10=85.0)) is True

    def test_mem_some_only_trips_saturation(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=50.0,
            mem_some10=15.0,
            mem_full10=2.0,
            io_some10=30.0,
            read_ok=True,
        )
        assert sample.saturated(_saturation_cfg()) is True

    def test_mem_full_only_trips_saturation(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=50.0,
            mem_some10=10.0,
            mem_full10=3.0,
            io_some10=30.0,
            read_ok=True,
        )
        assert sample.saturated(_saturation_cfg()) is True

    def test_io_some_only_trips_saturation(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=50.0,
            mem_some10=10.0,
            mem_full10=2.0,
            io_some10=40.0,
            read_ok=True,
        )
        assert sample.saturated(_saturation_cfg()) is True

    def test_all_under_threshold_not_saturated(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=50.0,
            mem_some10=10.0,
            mem_full10=2.0,
            io_some10=30.0,
            read_ok=True,
        )
        assert sample.saturated(_saturation_cfg()) is False

    def test_read_ok_false_gates_saturation_even_with_high_metrics(self):
        """The read_ok guard -- not the metric values -- must suppress
        saturation. Every other fail-open assertion in this suite uses the
        all-zero sentinel, which would read as not-saturated even if the
        read_ok guard were accidentally removed; this test pegs every metric
        at 99.0 against an all-zero-threshold (maximally saturating) cfg, so
        only the read_ok gate itself can explain a False result.
        """
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=99.0,
            mem_some10=99.0,
            mem_full10=99.0,
            io_some10=99.0,
            read_ok=False,
        )
        zero_threshold_cfg = types.SimpleNamespace(
            cpu_some_avg10=0.0,
            mem_some_avg10=0.0,
            mem_full_avg10=0.0,
            io_some_avg10=0.0,
        )
        assert sample.saturated(zero_threshold_cfg) is False


class TestSaturatedV2Arms:
    """PRD §6.1 saturated() semantics: an arm never trips when its cfg
    threshold is None or its component read_ok is False; host read_ok=False
    still suppresses the WHOLE sample including the new arms (DA-D6).
    """

    def test_none_threshold_never_trips_cpu_arm(self):
        """§7 row 4 — cpu_some10=99 against the v2 default stub."""
        sample = _healthy_sample(cpu_some10=99.0)
        assert sample.saturated(_saturation_cfg()) is False

    def test_none_threshold_never_trips_runqueue_arm(self):
        """§7 row 4 — runqueue_ratio=9 with cfg runqueue_ratio=None."""
        sample = _healthy_sample(runqueue_ratio=9.0)
        assert sample.saturated(_saturation_cfg()) is False

    def test_none_threshold_never_trips_own_arm(self):
        sample = _healthy_sample(own_cpu_some10=99.0)
        assert sample.saturated(_saturation_cfg()) is False

    def test_runqueue_arm_trips_when_configured_and_component_ok(self):
        sample = _healthy_sample(runqueue_ratio=4.3, runqueue_read_ok=True)
        assert sample.saturated(_configured_cfg(runqueue_ratio=4.0)) is True

    def test_runqueue_arm_inert_when_component_read_failed(self):
        """Same values, runqueue_read_ok=False — the arm must stay inert."""
        sample = _healthy_sample(runqueue_ratio=4.3, runqueue_read_ok=False)
        assert sample.saturated(_configured_cfg(runqueue_ratio=4.0)) is False

    def test_own_arm_trips_when_configured_and_component_ok(self):
        """§7 row 7 — own arm trips while every host arm is idle."""
        sample = _healthy_sample(own_cpu_some10=55.0, own_read_ok=True)
        assert sample.saturated(_configured_cfg(own_cpu_some_avg10=50.0)) is True

    def test_own_arm_inert_when_component_read_failed(self):
        """§7 row 3 — an unreadable slice file cannot trip the own arm."""
        sample = _healthy_sample(own_cpu_some10=55.0, own_read_ok=False)
        assert sample.saturated(_configured_cfg(own_cpu_some_avg10=50.0)) is False

    def test_host_read_ok_false_suppresses_both_new_arms(self):
        """DA-D6 — the outer host gate outranks every component flag."""
        sample = _healthy_sample(
            read_ok=False,
            runqueue_ratio=9.0,
            runqueue_read_ok=True,
            own_cpu_some10=99.0,
            own_read_ok=True,
        )
        cfg = _configured_cfg(runqueue_ratio=4.0, own_cpu_some_avg10=50.0)
        assert sample.saturated(cfg) is False

    def test_v1_cfg_without_new_attributes_is_tolerated(self):
        """The alpha-before-beta ordering guard.

        `scheduler.py::_phase_psi_gate` calls saturated() in live dispatch, and
        between this change and beta's `PsiAdmissionConfig` v2 the cfg object
        still carries only the four v1 thresholds. An absent attribute must
        read as "arm off", not raise AttributeError inside the gate.
        """
        v1_cfg = types.SimpleNamespace(
            cpu_some_avg10=85.0,
            mem_some_avg10=15.0,
            mem_full_avg10=3.0,
            io_some_avg10=40.0,
        )
        sample = _healthy_sample(runqueue_ratio=9.0, own_cpu_some10=99.0)
        assert sample.saturated(v1_cfg) is False

    def test_v1_cfg_still_evaluates_the_host_arms(self):
        v1_cfg = types.SimpleNamespace(
            cpu_some_avg10=85.0,
            mem_some_avg10=15.0,
            mem_full_avg10=3.0,
            io_some_avg10=40.0,
        )
        assert _healthy_sample(mem_full10=3.0).saturated(v1_cfg) is True


class TestTrippingMetric:
    """PRD D10 rank + the settled suffixed vocabulary.

    tripping_metric() returns CONFIG FIELD NAMES — the domain the shipped
    dispatch_deferred payload already uses — so the gate, the event consumers
    and the operator vocabulary stay one set.
    """

    D10_RANK = (
        'mem_full_avg10',
        'runqueue_ratio',
        'own_cpu_some_avg10',
        'mem_some_avg10',
        'io_some_avg10',
        'cpu_some_avg10',
    )

    def _all_arms_cfg(self):
        return _configured_cfg(
            cpu_some_avg10=85.0,
            runqueue_ratio=4.0,
            own_cpu_some_avg10=50.0,
        )

    def _all_arms_over(self):
        return _healthy_sample(
            cpu_some10=99.0,
            mem_some10=99.0,
            mem_full10=99.0,
            io_some10=99.0,
            runqueue_ratio=9.0,
            own_cpu_some10=99.0,
        )

    def test_runqueue_outranks_mem_some(self):
        """§7 row 5 — runqueue_ratio=4.3 over cfg 4.0, mem_some also over."""
        sample = _healthy_sample(runqueue_ratio=4.3, mem_some10=20.0)
        cfg = _configured_cfg(runqueue_ratio=4.0)
        assert sample.saturated(cfg) is True
        assert sample.tripping_metric(cfg) == 'runqueue_ratio'

    def test_mem_full_outranks_runqueue(self):
        """§7 row 5 continued — add mem_full over too."""
        sample = _healthy_sample(runqueue_ratio=4.3, mem_some10=20.0, mem_full10=5.0)
        cfg = _configured_cfg(runqueue_ratio=4.0)
        assert sample.tripping_metric(cfg) == 'mem_full_avg10'

    @pytest.mark.parametrize('dropped', range(len(D10_RANK)))
    def test_walks_the_whole_rank_from_the_top(self, dropped):
        """Every arm over its threshold, then removed one at a time from the
        top: the reported metric walks the exact D10 sequence."""
        removals = {
            'mem_full_avg10': {'mem_full10': 0.0},
            'runqueue_ratio': {'runqueue_ratio': 0.0},
            'own_cpu_some_avg10': {'own_cpu_some10': 0.0},
            'mem_some_avg10': {'mem_some10': 0.0},
            'io_some_avg10': {'io_some10': 0.0},
        }
        overrides = {}
        for name in self.D10_RANK[:dropped]:
            overrides.update(removals[name])
        sample = _healthy_sample(**{**vars_of(self._all_arms_over()), **overrides})
        cfg = self._all_arms_cfg()
        assert sample.tripping_metric(cfg) == self.D10_RANK[dropped]

    def test_none_threshold_arm_is_skipped_in_the_ranking(self):
        """mem_full is the highest-value arm but its threshold is None."""
        sample = _healthy_sample(mem_full10=99.0, runqueue_ratio=4.3)
        cfg = _configured_cfg(mem_full_avg10=None, runqueue_ratio=4.0)
        assert sample.tripping_metric(cfg) == 'runqueue_ratio'

    def test_failed_component_arm_is_skipped_in_the_ranking(self):
        """The own arm's value is highest but its component read failed."""
        sample = _healthy_sample(
            own_cpu_some10=99.0,
            own_read_ok=False,
            mem_some10=20.0,
        )
        cfg = _configured_cfg(own_cpu_some_avg10=50.0)
        assert sample.tripping_metric(cfg) == 'mem_some_avg10'

    def test_raises_on_non_saturated_sample(self):
        """Precondition violation is a programming error, not a fail-open case."""
        sample = _healthy_sample()
        with pytest.raises(ValueError, match='saturated'):
            sample.tripping_metric(self._all_arms_cfg())

    def test_raises_when_host_read_failed_even_with_everything_over(self):
        sample = _healthy_sample(**{**vars_of(self._all_arms_over()), 'read_ok': False})
        with pytest.raises(ValueError, match='saturated'):
            sample.tripping_metric(self._all_arms_cfg())


class TestReadPsiSampleHappyPath:
    def _fake_read(self):
        # Note: the memory pressure file is read under the name 'memory', not 'mem'.
        sources = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}

        def read(name):
            return sources[name]

        return read

    def test_maps_per_metric_fields(self):
        from shared.psi import read_psi_sample

        sample = read_psi_sample(read=self._fake_read())

        assert sample.read_ok is True
        assert sample.cpu_some10 == pytest.approx(2.50)
        assert sample.mem_some10 == pytest.approx(1.23)
        assert sample.mem_full10 == 0.0
        assert sample.io_some10 == pytest.approx(0.75)

    def test_saturated_reflects_injected_values(self):
        from shared.psi import read_psi_sample

        sample = read_psi_sample(read=self._fake_read())

        low_cpu_cfg = types.SimpleNamespace(
            cpu_some_avg10=1.0,
            mem_some_avg10=999.0,
            mem_full_avg10=999.0,
            io_some_avg10=999.0,
        )
        assert sample.saturated(low_cpu_cfg) is True

        all_high_cfg = types.SimpleNamespace(
            cpu_some_avg10=999.0,
            mem_some_avg10=999.0,
            mem_full_avg10=999.0,
            io_some_avg10=999.0,
        )
        assert sample.saturated(all_high_cfg) is False


class TestReadPsiSampleFailOpen:
    def _zero_threshold_cfg(self):
        """A maximally saturating cfg (all thresholds 0.0) — proves fail-open still wins."""
        return types.SimpleNamespace(
            cpu_some_avg10=0.0,
            mem_some_avg10=0.0,
            mem_full_avg10=0.0,
            io_some_avg10=0.0,
        )

    def _assert_sentinel(self, sample):
        assert sample.read_ok is False
        assert sample.cpu_some10 == 0.0
        assert sample.mem_some10 == 0.0
        assert sample.mem_full10 == 0.0
        assert sample.io_some10 == 0.0
        assert sample.saturated(self._zero_threshold_cfg()) is False

    def test_reader_raises_file_not_found_fails_open(self):
        from shared.psi import read_psi_sample

        def read(name):
            if name == 'cpu':
                raise FileNotFoundError('/proc/pressure/cpu')
            return PSI_MEM_TEXT if name == 'memory' else PSI_IO_TEXT

        self._assert_sentinel(read_psi_sample(read=read))

    def test_reader_raises_generic_oserror_fails_open(self):
        from shared.psi import read_psi_sample

        def read(name):
            if name == 'memory':
                raise PermissionError('/proc/pressure/memory')
            return PSI_CPU_TEXT if name == 'cpu' else PSI_IO_TEXT

        self._assert_sentinel(read_psi_sample(read=read))

    def test_unparseable_source_fails_open(self):
        from shared.psi import read_psi_sample

        def read(name):
            if name == 'io':
                return 'garbage line with no avg fields\n'
            return PSI_CPU_TEXT if name == 'cpu' else PSI_MEM_TEXT

        self._assert_sentinel(read_psi_sample(read=read))

    def test_reader_raises_non_os_error_fails_open(self):
        """A reader can fail with a non-OSError exception too -- e.g. a real
        Path.read_text() raises UnicodeDecodeError (a ValueError subclass,
        not an OSError) on non-UTF-8 content, and a custom injected reader
        could raise anything. DA-D6's "never wedge dispatch" guarantee is
        absolute, so this must fail open exactly like an OSError.
        """
        from shared.psi import read_psi_sample

        def read(name):
            if name == 'cpu':
                raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')
            return PSI_MEM_TEXT if name == 'memory' else PSI_IO_TEXT

        self._assert_sentinel(read_psi_sample(read=read))
