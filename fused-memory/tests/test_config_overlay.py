"""The config overlay (D14): one untracked YAML merged over the tracked base.

Every ``FusedMemoryConfig()`` reads the overlay named by ``OVERLAY_ENV_VAR``
through ``fused_memory.config.schema::YamlSettingsSource.__call__``, so the
server, the reload tool and the scripts all see one effective config.

Every config here is built from temp YAML files, never from init kwargs:
pydantic-settings deep-merges init kwargs over the YAML source, which would
hide the merge under test.
"""

import copy
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from fused_memory.config.overlay import OVERLAY_ENV_VAR, merge_overlay
from fused_memory.config.schema import FusedMemoryConfig


def _write_yaml(path: Path, data: Any) -> Path:
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


def _load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    base: dict[str, Any] | None,
    overlay: dict[str, Any] | None,
) -> FusedMemoryConfig:
    """Build the effective config from a temp base and a temp overlay.

    ``base=None`` points ``CONFIG_PATH`` at a missing file; ``overlay=None``
    leaves the overlay variable unset.
    """
    base_path = tmp_path / 'base.yaml'
    if base is not None:
        _write_yaml(base_path, base)
    monkeypatch.setenv('CONFIG_PATH', str(base_path))
    if overlay is None:
        monkeypatch.delenv(OVERLAY_ENV_VAR, raising=False)
    else:
        overlay_path = _write_yaml(tmp_path / 'overlay.yaml', overlay)
        monkeypatch.setenv(OVERLAY_ENV_VAR, str(overlay_path))
    return FusedMemoryConfig()


BASE_EMBEDDER = {
    'provider': 'openai',
    'model': 'text-embedding-3-small',
    'dimensions': 1536,
    'providers': {
        'openai': {'api_key': 'base-key', 'api_url': 'https://api.openai.com/v1'},
    },
}

BASE_WRITE_TRIAGE = {
    'enabled': True,
    'candidate_k': 7,
    't_high': 0.9,
    't_low': 0.5,
    'calibration_report_path': 'reports/base.json',
    't_high_by_category': {'observations': 0.91, 'procedural_knowledge': 0.88},
}


def _calibration(report: str, **cutoffs: float) -> dict[str, Any]:
    return {'calibration_report_path': report, 't_high_by_category': dict(cutoffs)}


class TestEmbedderMergesPerKey:
    def test_overlaid_leaves_land_and_unset_leaves_survive(self, monkeypatch, tmp_path):
        config = _load(
            monkeypatch,
            tmp_path,
            base={'embedder': BASE_EMBEDDER},
            overlay={
                'embedder': {
                    'model': 'granite-embedding',
                    'dimensions': 768,
                    'providers': {'openai': {'api_url': 'http://embedder.local/v1'}},
                },
            },
        )

        assert config.embedder.model == 'granite-embedding'
        assert config.embedder.dimensions == 768
        assert config.embedder.providers.openai is not None
        assert config.embedder.providers.openai.api_url == 'http://embedder.local/v1'
        assert config.embedder.providers.openai.api_key == 'base-key'
        assert config.embedder.provider == 'openai'


class TestWriteTriageBandReplacesTheUnit:
    def test_unset_band_keys_drop_and_hand_set_knobs_survive(self, monkeypatch, tmp_path):
        config = _load(
            monkeypatch,
            tmp_path,
            base={'write_triage': BASE_WRITE_TRIAGE},
            overlay={
                'write_triage': {
                    't_high': 0.8,
                    't_high_by_category': {'observations': 0.79},
                },
            },
        )

        triage = config.write_triage
        assert triage.t_high == 0.8
        assert triage.t_low is None
        assert triage.calibration_report_path is None
        assert triage.t_high_by_category == {'observations': 0.79}
        assert triage.enabled is True
        assert triage.candidate_k == 7


class TestProjectCalibrationsReplaceEachEntry:
    def test_an_overlaid_entry_replaces_its_base_entry_and_others_survive(
        self, monkeypatch, tmp_path,
    ):
        base_b = _calibration('reports/b.json', observations=0.85)
        overlay_a = _calibration('reports/a-granite.json', observations=0.7)
        config = _load(
            monkeypatch,
            tmp_path,
            base={
                'write_triage_project_calibrations': {
                    'A': _calibration(
                        'reports/a.json', observations=0.9, procedural_knowledge=0.8,
                    ),
                    'B': base_b,
                },
            },
            overlay={'write_triage_project_calibrations': {'A': overlay_a}},
        )

        calibrations = config.write_triage_project_calibrations
        assert calibrations['A'].model_dump() == overlay_a
        assert calibrations['B'].model_dump() == base_b


class TestLayering:
    def test_env_still_outranks_the_overlay(self, monkeypatch, tmp_path):
        monkeypatch.setenv('EMBEDDER__MODEL', 'model-from-env')

        config = _load(
            monkeypatch,
            tmp_path,
            base={'embedder': BASE_EMBEDDER},
            overlay={'embedder': {'model': 'model-from-overlay'}},
        )

        assert config.embedder.model == 'model-from-env'

    def test_the_overlay_applies_over_an_absent_base(self, monkeypatch, tmp_path):
        config = _load(
            monkeypatch,
            tmp_path,
            base=None,
            overlay={'embedder': {'model': 'granite-embedding', 'dimensions': 768}},
        )

        assert config.embedder.model == 'granite-embedding'
        assert config.embedder.dimensions == 768

    def test_overlay_values_interpolate_like_the_base(self, monkeypatch, tmp_path):
        monkeypatch.setenv('FM_OVERLAY_TEST_EMBEDDER_URL', 'http://from-env.local/v1')

        config = _load(
            monkeypatch,
            tmp_path,
            base={'embedder': BASE_EMBEDDER},
            overlay={
                'embedder': {
                    'model': '${FM_OVERLAY_TEST_UNSET_MODEL:granite-default}',
                    'providers': {
                        'openai': {'api_url': '${FM_OVERLAY_TEST_EMBEDDER_URL:unused}'},
                    },
                },
            },
        )

        assert config.embedder.model == 'granite-default'
        assert config.embedder.providers.openai is not None
        assert config.embedder.providers.openai.api_url == 'http://from-env.local/v1'


class TestProvenance:
    def test_provenance_names_the_file_its_merged_keys_and_mtime(
        self, monkeypatch, tmp_path,
    ):
        config = _load(
            monkeypatch,
            tmp_path,
            base={'embedder': BASE_EMBEDDER, 'write_triage': BASE_WRITE_TRIAGE},
            overlay={
                'embedder': {
                    'model': 'granite-embedding',
                    'providers': {'openai': {'api_url': 'http://embedder.local/v1'}},
                },
                'write_triage': {'t_high': 0.8},
                'write_triage_project_calibrations': {
                    'A': _calibration('reports/a.json', observations=0.7),
                },
            },
        )

        overlay_path = tmp_path / 'overlay.yaml'
        provenance = config.config_overlay
        assert provenance is not None
        assert provenance.path == str(overlay_path)
        assert provenance.merged_keys == (
            'embedder.model',
            'embedder.providers.openai.api_url',
            'write_triage.t_high',
            'write_triage_project_calibrations.A',
        )
        expected_mtime = datetime.fromtimestamp(os.stat(overlay_path).st_mtime, UTC)
        assert provenance.mtime == expected_mtime.isoformat()

    def test_no_overlay_means_no_provenance(self, monkeypatch, tmp_path):
        config = _load(
            monkeypatch, tmp_path, base={'embedder': BASE_EMBEDDER}, overlay=None,
        )

        assert config.config_overlay is None


class TestMergeOverlayReplacesTheThresholdGroup:
    def test_any_threshold_key_replaces_the_whole_group_without_mutating_base(self):
        base = {
            'embedder_thresholds': {'near_dup': 0.92, 'restatement': 0.85},
            'server': {'port': 8002},
        }
        pristine = copy.deepcopy(base)

        merged = merge_overlay(base, {'embedder_thresholds': {'near_dup': 0.81}})

        assert merged == {
            'embedder_thresholds': {'near_dup': 0.81},
            'server': {'port': 8002},
        }
        assert base == pristine
