"""Tests for the unknown-config-key census (PRD warm-lane-exhaustion-hardening leaf ζ).

Root cause of the 2026-07-22 reify incident: a top-level ``spare_warm_lanes: 8``
in the project YAML silently dropped for 3+ weeks because OrchestratorConfig uses
pydantic ``extra='ignore'`` — the field actually lives on GitConfig.  Pydantic
discards extras BEFORE validation, so unknown keys must be detected by a SEPARATE
raw-YAML-vs-model pass.  These tests pin the pure census engine.
"""

from pathlib import Path

import yaml

from orchestrator.config import (
    ConfigUnknownKey,
    census_unknown_config_keys,
    config_unknown_keys_signature,
)


def _write_yaml(tmp_path: Path, data, name: str = 'orchestrator.yaml') -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data))
    return p


# --- (a) clean config → empty census -----------------------------------------


def test_clean_config_yields_empty_census(tmp_path):
    """Valid top-level keys + valid nested git: keys produce no unknown keys."""
    p = _write_yaml(
        tmp_path,
        {
            'max_concurrent_tasks': 4,
            'max_per_module': 2,
            'git': {'remote': 'origin', 'main_branch': 'main'},
        },
    )
    assert census_unknown_config_keys(p) == []


# --- (b) unknown top-level key → recorded with no shadow hint -----------------


def test_unknown_top_level_key_recorded(tmp_path):
    p = _write_yaml(tmp_path, {'bogus_key': 1})
    census = census_unknown_config_keys(p)
    assert ConfigUnknownKey('bogus_key', None) in census
    assert [uk.path for uk in census] == ['bogus_key']


# --- (c) the incident shape: top-level spare_warm_lanes → git.spare_warm_lanes -


def test_spare_warm_lanes_incident_shape(tmp_path):
    """A top-level spare_warm_lanes (the incident key) is flagged and its shadow
    hint names the real home git.spare_warm_lanes."""
    p = _write_yaml(tmp_path, {'spare_warm_lanes': 8})
    census = census_unknown_config_keys(p)
    by_path = {uk.path: uk for uk in census}
    assert 'spare_warm_lanes' in by_path
    hint = by_path['spare_warm_lanes'].shadow_hint
    assert hint is not None
    assert 'git.spare_warm_lanes' in hint


# --- (d) unknown nested key under git: → descend works ------------------------


def test_unknown_nested_key_under_git(tmp_path):
    """Descent into the git: submodel catches a typo but leaves valid nested keys."""
    p = _write_yaml(tmp_path, {'git': {'bogus_nested': 1, 'remote': 'origin'}})
    census = census_unknown_config_keys(p)
    paths = {uk.path for uk in census}
    assert 'git.bogus_nested' in paths
    # A valid nested key must NOT be flagged — proves the walk descended.
    assert 'git.remote' not in paths
    # bogus_nested exists nowhere in the model tree → no shadow hint.
    hint = next(uk.shadow_hint for uk in census if uk.path == 'git.bogus_nested')
    assert hint is None


# --- (e) dict-data fields carry arbitrary keys → never flagged ----------------


def test_dict_data_fields_not_descended(tmp_path):
    """verify_env / module_overrides are dict[...] DATA fields; their arbitrary
    keys are operator data, never unknown config keys."""
    p = _write_yaml(
        tmp_path,
        {
            'verify_env': {'ARBITRARY_VAR': 'x', 'ANOTHER_ONE': 'y'},
            'module_overrides': {'some/module': 3},
        },
    )
    assert census_unknown_config_keys(p) == []


# --- (f) non-dict / empty YAML → empty census --------------------------------


def test_empty_yaml_yields_empty_census(tmp_path):
    empty = tmp_path / 'empty.yaml'
    empty.write_text('')
    assert census_unknown_config_keys(empty) == []


def test_non_dict_yaml_yields_empty_census(tmp_path):
    listy = _write_yaml(tmp_path, [1, 2, 3], name='list.yaml')
    assert census_unknown_config_keys(listy) == []


# --- (g) signature is set-stable and set-sensitive ---------------------------


def test_signature_stable_across_ordering(tmp_path):
    c1 = [ConfigUnknownKey('a', None), ConfigUnknownKey('spare_warm_lanes', 'git.spare_warm_lanes')]
    c2 = [ConfigUnknownKey('spare_warm_lanes', 'git.spare_warm_lanes'), ConfigUnknownKey('a', None)]
    assert config_unknown_keys_signature(c1) == config_unknown_keys_signature(c2)


def test_signature_changes_when_key_set_changes(tmp_path):
    c1 = [ConfigUnknownKey('a', None), ConfigUnknownKey('b', None)]
    c3 = [ConfigUnknownKey('a', None), ConfigUnknownKey('b', None), ConfigUnknownKey('c', None)]
    assert config_unknown_keys_signature(c1) != config_unknown_keys_signature(c3)
