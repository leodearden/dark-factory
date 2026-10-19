"""L3b-DA2: PsiAdmissionConfig submodel + green-tier reload registration
(task 2327; PRD docs/prds/dispatch-admission-load-cap.md), and its v2
vocabulary (task 3590; PRD plans/load-throttle-harmonisation-prd.md §6.2).

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_reload_integration_gate.py's stated rationale).
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError
from shared.psi import PsiSample

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    PsiAdmissionConfig,
    apply_reload,
    diff_config,
    load_config,
)

_THRESHOLD_FIELDS = (
    'cpu_some_avg10',
    'mem_some_avg10',
    'mem_full_avg10',
    'io_some_avg10',
    'runqueue_ratio',
    'own_cpu_some_avg10',
)
_PERCENT_THRESHOLDS = tuple(f for f in _THRESHOLD_FIELDS if f != 'runqueue_ratio')


class TestPsiAdmissionConfigDefaults:
    """Defaults: enabled, memory ranked tighter than io (DA-D1), and the
    legacy CPU arm plus both v2 arms off (PRD load-throttle-harmonisation
    D1/D2).
    """

    def test_defaults_attached_on_orchestrator_config(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        assert isinstance(cfg.psi_admission, PsiAdmissionConfig)
        assert cfg.psi_admission.enabled is True
        assert cfg.psi_admission.cpu_some_avg10 is None
        assert cfg.psi_admission.mem_some_avg10 == 15.0
        assert cfg.psi_admission.mem_full_avg10 == 3.0
        assert cfg.psi_admission.io_some_avg10 == 40.0
        assert cfg.psi_admission.runqueue_ratio is None
        assert cfg.psi_admission.own_cpu_some_avg10 is None
        assert cfg.psi_admission.min_inflight_floor == 1

    def test_memory_threshold_ranked_tighter_than_io(self):
        """DA-D1: memory ranks above io, so its default threshold trips first."""
        cfg = PsiAdmissionConfig()
        assert cfg.mem_some_avg10 is not None and cfg.io_some_avg10 is not None
        assert cfg.mem_some_avg10 < cfg.io_some_avg10


class TestPsiAdmissionArmVocabulary:
    """The config's threshold leaves and shared.psi's saturation arms are one
    set. Absence in cfg means "arm off" to shared.psi, so a rename on either
    side would silently disable an arm rather than fail.
    """

    def test_threshold_fields_are_every_leaf_but_the_switch_and_floor(self):
        assert set(PsiAdmissionConfig.model_fields) - {'enabled', 'min_inflight_floor'} == set(
            _THRESHOLD_FIELDS
        )

    def test_saturation_reads_exactly_the_threshold_fields(self):
        """``saturated()`` reads its duck-typed cfg only by arm field name, so
        the names it asks for are the arm vocabulary."""
        asked: set[str] = set()

        class _EveryArmOff:
            def __getattr__(self, name: str) -> None:
                asked.add(name)
                return None

        readable = PsiSample(
            cpu_some10=0.0, mem_some10=0.0, mem_full10=0.0, io_some10=0.0, read_ok=True,
        )

        assert not readable.saturated(_EveryArmOff())
        assert asked == set(_THRESHOLD_FIELDS)

    @pytest.mark.parametrize('field', _THRESHOLD_FIELDS)
    def test_threshold_accepts_none(self, field):
        assert getattr(PsiAdmissionConfig.model_validate({field: None}), field) is None

    @pytest.mark.parametrize('field', _THRESHOLD_FIELDS)
    def test_each_threshold_alone_trips_its_own_arm(self, field):
        cfg = PsiAdmissionConfig.model_validate(
            {f: (1.0 if f == field else None) for f in _THRESHOLD_FIELDS}
        )
        sample = PsiSample(
            cpu_some10=50.0,
            mem_some10=50.0,
            mem_full10=50.0,
            io_some10=50.0,
            read_ok=True,
            runqueue_ratio=50.0,
            runqueue_read_ok=True,
            own_cpu_some10=50.0,
            own_cgroup='/df.slice/df-x.slice',
            own_read_ok=True,
        )

        assert sample.saturated(cfg)
        assert sample.tripping_metric(cfg) == field


class TestMinInflightFloorValidation:
    """DA-D3 anti-deadlock floor: a min_inflight_floor < 1 would let the gate
    hold with nothing in flight and wedge the queue, so it must be rejected
    at construction/load.
    """

    @pytest.mark.parametrize('bad_value', [0, -1])
    def test_sub_minimum_rejected(self, bad_value):
        with pytest.raises(ValidationError):
            PsiAdmissionConfig(min_inflight_floor=bad_value)

    @pytest.mark.parametrize('good_value', [1, 2])
    def test_valid_values_accepted(self, good_value):
        cfg = PsiAdmissionConfig(min_inflight_floor=good_value)
        assert cfg.min_inflight_floor == good_value

    def test_rejected_at_load_via_orchestrator_config(self, monkeypatch, tmp_path):
        """Covers the 'rejected at load' signal: load_config constructs
        OrchestratorConfig, which builds the psi_admission submodel from a
        nested dict (as YAML would deserialize into) and must raise.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError):
            OrchestratorConfig(psi_admission={'min_inflight_floor': 0})  # type: ignore[arg-type]


class TestThresholdRangeValidation:
    """An out-of-range threshold is refused at construction, at load and at
    hot reload (task 4063, absorbed by 3590; PRD
    plans/load-throttle-harmonisation-prd.md boundary row 8). A negative
    percentage would silently disable an arm and one over 100 would pin it.
    """

    @pytest.mark.parametrize('field', _PERCENT_THRESHOLDS)
    @pytest.mark.parametrize('bad_value', [-0.1, 100.1])
    def test_percentage_outside_0_to_100_rejected(self, field, bad_value):
        with pytest.raises(ValidationError, match=field):
            PsiAdmissionConfig(**{field: bad_value})

    @pytest.mark.parametrize('field', _PERCENT_THRESHOLDS)
    @pytest.mark.parametrize('good_value', [0.0, 100.0])
    def test_percentage_bounds_accepted(self, field, good_value):
        assert getattr(PsiAdmissionConfig(**{field: good_value}), field) == good_value

    @pytest.mark.parametrize('bad_value', [0.0, -1.0])
    def test_non_positive_runqueue_ratio_rejected(self, bad_value):
        with pytest.raises(ValidationError, match='runqueue_ratio'):
            PsiAdmissionConfig(runqueue_ratio=bad_value)

    @pytest.mark.parametrize('good_value', [0.1, 50.0])
    def test_positive_runqueue_ratio_accepted_with_no_upper_bound(self, good_value):
        assert PsiAdmissionConfig(runqueue_ratio=good_value).runqueue_ratio == good_value

    @pytest.mark.parametrize(
        ('field', 'bad_value'), [('mem_full_avg10', 101), ('runqueue_ratio', 0)]
    )
    def test_rejected_by_load_config(self, monkeypatch, tmp_path, field, bad_value):
        """load_config is the call Harness.reload_config makes, so a bad yaml
        edit is refused before it reaches the live config."""
        cfg_path = tmp_path / 'orchestrator.yaml'
        cfg_path.write_text(yaml.safe_dump({'psi_admission': {field: bad_value}}))
        # load_config writes os.environ; monkeypatch owns the restore.
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(cfg_path))

        with pytest.raises(ValidationError, match=field):
            load_config(cfg_path)

    @pytest.mark.parametrize(
        ('field', 'bad_value'), [('mem_full_avg10', 101.0), ('runqueue_ratio', 0.0)]
    )
    def test_hot_reload_rejected_and_live_kept(self, monkeypatch, tmp_path, field, bad_value):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig()
        live_value = getattr(live.psi_admission, field)
        fresh = OrchestratorConfig()
        object.__setattr__(fresh.psi_admission, field, bad_value)

        report = apply_reload(live, fresh)

        assert report['reloaded'] is False
        assert field in report['error']
        assert getattr(live.psi_admission, field) == live_value


class TestPsiAdmissionReloadDisposition:
    """Every psi_admission threshold is green-tier: hot-reloadable without a
    process restart, mirroring the fairness/starvation_watchdog groups.
    """

    @pytest.mark.parametrize('leaf', list(PsiAdmissionConfig.model_fields))
    def test_every_leaf_is_reloadable(self, leaf):
        assert f'psi_admission.{leaf}' in RELOADABLE_FIELDS, (
            f'psi_admission.{leaf!r} is expected to be green-tier reloadable '
            f'but is missing from RELOADABLE_FIELDS'
        )

    def test_switching_an_arm_on_lands_in_applied_candidates_not_restart_required(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(psi_admission=PsiAdmissionConfig())
        fresh = OrchestratorConfig(psi_admission=PsiAdmissionConfig(runqueue_ratio=4.0))
        diff = diff_config(live, fresh)
        assert 'psi_admission.runqueue_ratio' in diff.applied_candidates
        assert 'psi_admission.runqueue_ratio' not in diff.restart_required

    def test_apply_reload_switches_an_arm_on_in_place(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(psi_admission=PsiAdmissionConfig())
        fresh = OrchestratorConfig(psi_admission=PsiAdmissionConfig(runqueue_ratio=4.0))
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert report['applied']['psi_admission.runqueue_ratio'] == {'old': None, 'new': 4.0}
        assert 'psi_admission.runqueue_ratio' not in report['restart_required']
        assert live.psi_admission.runqueue_ratio == 4.0

    def test_apply_reload_switches_an_arm_off_in_place(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(psi_admission=PsiAdmissionConfig(runqueue_ratio=4.0))
        fresh = OrchestratorConfig(psi_admission=PsiAdmissionConfig())
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert report['applied']['psi_admission.runqueue_ratio'] == {'old': 4.0, 'new': None}
        assert live.psi_admission.runqueue_ratio is None
