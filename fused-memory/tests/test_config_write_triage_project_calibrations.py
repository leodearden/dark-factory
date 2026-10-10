"""The `write_triage_project_calibrations` config leaf (task 5547).

Per-PROJECT ANN cutoffs, each measured on that project's own curator-labelled
fixture by `scripts/calibrate_write_triage.py --write-project-config`. They live
in their own top-level section, never inside `write_triage:`, so the reify-derived
pooled bands and a project's cutoffs each have exactly one writer.

This module pins the leaf's shape, its validation, and its reload tier.
`FusedMemoryConfig()` loads the shipped config.yaml, which by design carries a
calibration run's output, so defaults are read off the declared field and every
reload test uses SYNTHETIC values that cannot equal the shipped ones.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from fused_memory.config.reload import (
    RELOADABLE_FIELDS,
    _iter_leaves,
    apply_reload,
    diff_config,
)
from fused_memory.config.schema import (
    FusedMemoryConfig,
    WriteTriageConfig,
    WriteTriageProjectCalibration,
)

FIELD = 'write_triage_project_calibrations'
# Never a shipped key: pydantic-settings deep-merges init kwargs OVER the YAML
# source, so an entry for a project config.yaml also carries would MERGE with it.
PROJECT = 'synthetic_project'
REPORT = 'calibration/x.json'


def _entry(t_high_by_category: dict) -> dict:
    return {'calibration_report_path': REPORT, 't_high_by_category': t_high_by_category}


def _config(entries: dict) -> FusedMemoryConfig:
    # Raw YAML-shaped entries, deliberately: what is pinned is how they load.
    return FusedMemoryConfig(write_triage_project_calibrations=entries)  # pyright: ignore[reportArgumentType]


class TestShape:
    def test_absent_project_is_the_only_spelling_of_no_calibration(self) -> None:
        default = FusedMemoryConfig.model_fields[FIELD].get_default(call_default_factory=True)
        assert default == {}

    def test_a_valid_entry_validates_into_the_entry_model(self) -> None:
        got = _config({PROJECT: _entry({'preferences_and_norms': 0.9})})
        entry = getattr(got, FIELD)[PROJECT]
        assert isinstance(entry, WriteTriageProjectCalibration)
        assert entry.calibration_report_path == REPORT
        assert entry.t_high_by_category == {'preferences_and_norms': 0.9}

    def test_the_entry_model_is_frozen(self) -> None:
        entry = WriteTriageProjectCalibration(**_entry({'preferences_and_norms': 0.9}))
        with pytest.raises(ValidationError):
            entry.calibration_report_path = 'calibration/other.json'

    def test_an_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WriteTriageProjectCalibration(**_entry({}), t_low=0.5)  # pyright: ignore[reportCallIssue]

    @pytest.mark.parametrize('missing', ['calibration_report_path', 't_high_by_category'])
    def test_both_keys_are_required(self, missing: str) -> None:
        raw = _entry({'preferences_and_norms': 0.9})
        del raw[missing]
        with pytest.raises(ValidationError):
            WriteTriageProjectCalibration.model_validate(raw)


class TestCutoffRange:
    @pytest.mark.parametrize('value', [-0.1, 1.0001, 42.0])
    def test_an_out_of_range_cutoff_names_the_project_category_and_value(
        self, value: float,
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _config({PROJECT: _entry({'observations_and_summaries': value})})
        message = str(excinfo.value)
        assert PROJECT in message
        assert 'observations_and_summaries' in message
        assert repr(value) in message

    @pytest.mark.parametrize('value', [0.0, 1.0])
    def test_the_unit_bounds_are_inclusive(self, value: float) -> None:
        entry = WriteTriageProjectCalibration(**_entry({'preferences_and_norms': value}))
        assert entry.t_high_by_category == {'preferences_and_norms': value}

    def test_one_bad_cutoff_rejects_the_whole_map(self) -> None:
        with pytest.raises(ValidationError):
            WriteTriageProjectCalibration(**_entry({
                'preferences_and_norms': 0.9, 'observations_and_summaries': 9.0,
            }))

    def test_a_numeric_string_is_coerced_as_the_global_sibling_coerces_it(self) -> None:
        entry = WriteTriageProjectCalibration(**_entry({'preferences_and_norms': '0.9'}))
        assert entry.t_high_by_category == {'preferences_and_norms': 0.9}

    @pytest.mark.parametrize('value', [True, False])
    def test_a_bool_is_coerced_to_float_never_kept_as_bool(self, value: bool) -> None:
        entry = WriteTriageProjectCalibration(**_entry({'preferences_and_norms': value}))
        coerced = entry.t_high_by_category['preferences_and_norms']
        assert coerced == float(value)
        assert not isinstance(coerced, bool)

    def test_the_global_map_keeps_its_own_message_through_the_shared_check(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            WriteTriageConfig(t_high_by_category={'observations_and_summaries': 42.0})
        message = str(excinfo.value)
        assert 'write_triage.t_high_by_category' in message
        assert 'observations_and_summaries' in message
        assert '42.0' in message


class TestReloadTier:
    """Green tier, ONE atomic leaf, replaced wholesale.

    Its only reader is scripts/audit_duplicate_memories.py, which builds a fresh
    FusedMemoryConfig per run, so an in-place reload is safe.
    """

    SYNTHETIC = {
        'synthetic_project': WriteTriageProjectCalibration(
            calibration_report_path='calibration/synthetic.json',
            t_high_by_category={'preferences_and_norms': 0.5},
        ),
    }

    def test_the_leaf_is_allowlisted(self) -> None:
        assert FIELD in RELOADABLE_FIELDS

    def test_iter_leaves_yields_the_map_as_exactly_one_whole_leaf(self) -> None:
        config = FusedMemoryConfig()
        object.__setattr__(config, FIELD, dict(self.SYNTHETIC))
        paths = [path for path, _ in _iter_leaves(config)]

        assert paths.count(FIELD) == 1
        assert not [p for p in paths if p.startswith(f'{FIELD}.')]
        assert dict(_iter_leaves(config))[FIELD] == self.SYNTHETIC

    def test_a_changed_map_lands_in_applied_candidates(self) -> None:
        live, fresh = FusedMemoryConfig(), FusedMemoryConfig()
        old = getattr(live, FIELD)
        assert old != self.SYNTHETIC
        object.__setattr__(fresh, FIELD, dict(self.SYNTHETIC))

        d = diff_config(live, fresh)

        assert d.applied_candidates[FIELD] == {'old': old, 'new': self.SYNTHETIC}
        assert FIELD not in d.restart_required

    def test_apply_reload_replaces_the_map_wholesale(self) -> None:
        live, fresh = FusedMemoryConfig(), FusedMemoryConfig()
        object.__setattr__(live, FIELD, {
            'other_project': WriteTriageProjectCalibration(
                calibration_report_path='calibration/other.json',
                t_high_by_category={'observations_and_summaries': 0.6},
            ),
        })
        object.__setattr__(fresh, FIELD, dict(self.SYNTHETIC))

        report = apply_reload(live, fresh)

        assert report['reloaded'] is True
        assert getattr(live, FIELD) == self.SYNTHETIC
