"""Tests for the unknown-config-key census (PRD warm-lane-exhaustion-hardening leaf ζ).

Root cause of the 2026-07-22 reify incident: a top-level ``spare_warm_lanes: 8``
in the project YAML silently dropped for 3+ weeks because OrchestratorConfig uses
pydantic ``extra='ignore'`` — the field actually lives on GitConfig.  Pydantic
discards extras BEFORE validation, so unknown keys must be detected by a SEPARATE
raw-YAML-vs-model pass.  These tests pin the pure census engine.
"""

import logging
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from orchestrator.cli import main
from orchestrator.config import (
    RELOADABLE_FIELDS,
    ConfigIgnoredKey,
    ConfigKeyCensusConfig,
    ConfigUnknownKey,
    OrchestratorConfig,
    census_config_keys,
    census_unknown_config_keys,
    config_unknown_keys_signature,
    load_config,
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


# --- load_config stash + unknown_key_census property -------------------------


def test_load_config_stashes_census_for_dirty_config(tmp_path, monkeypatch):
    """load_config on a project YAML with a top-level spare_warm_lanes stashes a
    census carrying that key and its git.spare_warm_lanes shadow hint."""
    p = _write_yaml(
        tmp_path,
        {'project_root': str(tmp_path), 'spare_warm_lanes': 8},
        name='config.yaml',
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))
    config = load_config(p)
    census = config.unknown_key_census
    assert census, 'expected a non-empty census for the dirty config'
    by_path = {uk.path: uk for uk in census}
    assert 'spare_warm_lanes' in by_path
    hint = by_path['spare_warm_lanes'].shadow_hint
    assert hint is not None and 'git.spare_warm_lanes' in hint


def test_load_config_clean_config_empty_census(tmp_path, monkeypatch):
    p = _write_yaml(
        tmp_path,
        {'project_root': str(tmp_path), 'max_concurrent_tasks': 3},
        name='config.yaml',
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))
    config = load_config(p)
    assert config.unknown_key_census == []


def test_direct_construction_yields_empty_census(tmp_path, monkeypatch):
    """A directly-constructed OrchestratorConfig() never ran load_config, so the
    census sentinel stays None and the property normalizes to []."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    config = OrchestratorConfig()
    assert config.unknown_key_census == []


def test_load_config_logs_warning_naming_unknown_key(tmp_path, monkeypatch, caplog):
    """load_config emits a WARNING naming the unknown key (loud-over-silent)."""
    p = _write_yaml(
        tmp_path,
        {'project_root': str(tmp_path), 'spare_warm_lanes': 8},
        name='config.yaml',
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))
    with caplog.at_level(logging.WARNING, logger='orchestrator.config'):
        load_config(p)
    assert any(
        'spare_warm_lanes' in rec.getMessage() for rec in caplog.records
    ), 'expected a WARNING naming the unknown key spare_warm_lanes'


# --- check-config CLI subcommand ---------------------------------------------


def test_check_config_clean_exits_zero(tmp_path):
    p = _write_yaml(
        tmp_path,
        {'max_concurrent_tasks': 3, 'git': {'remote': 'origin'}},
        name='config.yaml',
    )
    result = CliRunner().invoke(main, ['check-config', '--config', str(p)])
    assert result.exit_code == 0, result.output
    # A clean/OK line is printed, and no unknown-key rows appear.
    assert result.output.strip(), 'expected a clean/OK line'
    assert 'spare_warm_lanes' not in result.output
    assert 'bogus_key' not in result.output


def test_check_config_reports_unknown_keys_and_exits_one(tmp_path):
    p = _write_yaml(
        tmp_path,
        {'bogus_key': 1, 'spare_warm_lanes': 8},
        name='config.yaml',
    )
    result = CliRunner().invoke(main, ['check-config', '--config', str(p)])
    assert result.exit_code == 1, result.output
    assert 'bogus_key' in result.output
    assert 'spare_warm_lanes' in result.output
    # The placement hint points at the real home of the incident key.
    assert 'git.spare_warm_lanes' in result.output


# =============================================================================
# Census escape hatches (review round 1 — steps 11/12)
#
# Without an escape hatch the always-on born-at-L2 files a PERMANENT,
# UNRESOLVABLE critical on any project whose YAML deliberately carries keys for
# NON-OrchestratorConfig consumers (measured on this machine: /home/leo/src/reify
# and /home/leo/src/reify-unblock-5196 each carry 6 such keys, read verbatim by
# reify's own scripts).  Two hatches classify such a key as `ignored` instead of
# `unknown`: a reserved `x_`/`x-` name prefix at any depth, and an operator
# `config_key_census.ignore` dotted-glob allowlist read from the same raw YAML.
# =============================================================================


# --- (a) reserved-prefix hatch ------------------------------------------------


def test_reserved_prefix_keys_are_ignored_not_unknown(tmp_path):
    """`x_`/`x-` prefixed keys are classified ignored (reason='reserved_prefix').

    Case-insensitive, mirroring OrchestratorConfig's case_sensitive=False.
    """
    p = _write_yaml(tmp_path, {'x_foo': 1, 'x-bar': 1, 'X_Baz': 1})
    census = census_config_keys(p)
    assert census.unknown == []
    ignored = {ik.path: ik.reason for ik in census.ignored}
    assert ignored == {
        'x_foo': 'reserved_prefix',
        'x-bar': 'reserved_prefix',
        'X_Baz': 'reserved_prefix',
    }


def test_reserved_prefix_applies_at_nested_depth(tmp_path):
    """The prefix hatch works at ANY depth, and does not disturb real nested keys."""
    p = _write_yaml(tmp_path, {'git': {'x_custom': 1, 'remote': 'origin'}})
    census = census_config_keys(p)
    assert census.unknown == []
    assert ConfigIgnoredKey('git.x_custom', 'reserved_prefix') in census.ignored
    assert 'git.remote' not in {ik.path for ik in census.ignored}


# --- (b) operator allowlist (the real reify shape) ----------------------------


def _reify_shape(extra: dict | None = None) -> dict:
    """The measured reify mixed-consumer shape: reify-owned knobs living in the
    SAME yaml blocks as real OrchestratorConfig fields."""
    data = {
        'config_key_census': {'ignore': ['cpu_governance.*', 'fairness.scheduler_v2']},
        'cpu_governance': {
            'enabled': True,                 # real CpuGovernConfig field
            'weights': {'task': 100},        # reify-owned
            'agent_admit': {'enabled': True},  # reify-owned
        },
        'fairness': {
            'skip_threshold': 3,   # real FairnessConfig field
            'scheduler_v2': True,  # reify-owned
        },
    }
    if extra:
        data.update(extra)
    return data


def test_allowlist_opts_out_mixed_consumer_namespace(tmp_path):
    p = _write_yaml(tmp_path, _reify_shape())
    census = census_config_keys(p)
    assert census.unknown == [], 'allowlisted keys must not be unknown'
    ignored = {ik.path: ik.reason for ik in census.ignored}
    assert ignored.get('cpu_governance.weights') == 'allowlist'
    assert ignored.get('cpu_governance.agent_admit') == 'allowlist'
    assert ignored.get('fairness.scheduler_v2') == 'allowlist'
    # Sibling REAL model keys are never flagged by either list.
    assert 'cpu_governance.enabled' not in ignored
    assert 'fairness.skip_threshold' not in ignored


# --- (c) self-flag regression guard -------------------------------------------


def test_config_key_census_block_does_not_self_flag(tmp_path):
    """The key that CONFIGURES the census must itself be a real model field.

    Otherwise an operator applying the documented remediation would trade one
    born-at-L2 for another.
    """
    p = _write_yaml(tmp_path, {'config_key_census': {'ignore': ['some.path']}})
    census = census_config_keys(p)
    assert census.unknown == []
    assert 'config_key_census' not in {ik.path for ik in census.ignored}
    # And it is a declared model field with the documented shape.
    assert 'config_key_census' in OrchestratorConfig.model_fields
    assert ConfigKeyCensusConfig().ignore == []


# --- (d) genuine signal is preserved through the hatch ------------------------


def test_allowlist_does_not_suppress_genuine_unknown_keys(tmp_path):
    """SYNTHETIC fixture: an unallowlisted key stays unknown, and a name that
    shadows a nested model field still produces its advisory hint.

    This is a claim about WALK BEHAVIOUR only — deliberately NOT a claim that
    reify's real top-level `warm_lane_pool:` is misplaced (it is a legitimate
    reify-owned dict; shadow hints are advisory and may be coincidental name
    collisions).
    """
    p = _write_yaml(tmp_path, _reify_shape({'warm_lane_pool': 6, 'bogus_key': 1}))
    census = census_config_keys(p)
    by_path = {uk.path: uk for uk in census.unknown}
    assert set(by_path) == {'warm_lane_pool', 'bogus_key'}
    hint = by_path['warm_lane_pool'].shadow_hint
    assert hint is not None and 'git.warm_lane_pool' in hint
    assert by_path['bogus_key'].shadow_hint is None


# --- (e) exact-path vs glob matching, and the bare-parent fnmatch trap ---------


def test_exact_path_allowlist_entry_matches(tmp_path):
    p = _write_yaml(
        tmp_path,
        {
            'config_key_census': {'ignore': ['fairness.scheduler_v2']},
            'fairness': {'scheduler_v2': True, 'other_bogus': 1},
        },
    )
    census = census_config_keys(p)
    assert ConfigIgnoredKey('fairness.scheduler_v2', 'allowlist') in census.ignored
    # An exact entry opts out ONLY that path — its sibling stays unknown.
    assert [uk.path for uk in census.unknown] == ['fairness.other_bogus']


def test_glob_does_not_match_bare_parent_key(tmp_path):
    """fnmatch trap: `<name>.*` does NOT match the bare top-level key `<name>`.

    Opting out a top-level dict key therefore requires listing it EXACTLY.  This
    is load-bearing for the reify follow-up: a `warm_lane_pool.*` entry alone
    would leave the born-at-L2 firing permanently.
    """
    p = _write_yaml(
        tmp_path,
        {
            'config_key_census': {'ignore': ['warm_lane_pool.*']},
            'warm_lane_pool': {'sizing': {'safety_divisor': 2}},
        },
    )
    census = census_config_keys(p)
    assert [uk.path for uk in census.unknown] == ['warm_lane_pool']

    # Listing it exactly DOES opt it out (and the walk still does not descend).
    p2 = _write_yaml(
        tmp_path,
        {
            'config_key_census': {'ignore': ['warm_lane_pool']},
            'warm_lane_pool': {'sizing': {'safety_divisor': 2}},
        },
        name='exact.yaml',
    )
    census2 = census_config_keys(p2)
    assert census2.unknown == []
    assert ConfigIgnoredKey('warm_lane_pool', 'allowlist') in census2.ignored


# --- (f) signature isolation + unchanged public wrapper -----------------------


def test_signature_ignores_escape_hatched_keys(tmp_path):
    """The L2 dedup discriminator is computed over UNKNOWN keys only, so adding
    an escape-hatched key never re-files a distinct L2."""
    plain = _write_yaml(tmp_path, {'bogus_key': 1}, name='plain.yaml')
    hatched = _write_yaml(
        tmp_path,
        {
            'bogus_key': 1,
            'x_extra': 1,
            'config_key_census': {'ignore': ['cpu_governance.*']},
            'cpu_governance': {'weights': {'task': 100}},
        },
        name='hatched.yaml',
    )
    sig_plain = config_unknown_keys_signature(census_config_keys(plain).unknown)
    sig_hatched = config_unknown_keys_signature(census_config_keys(hatched).unknown)
    assert sig_plain == sig_hatched


def test_public_wrapper_returns_unknown_view(tmp_path):
    """census_unknown_config_keys keeps its exact signature/semantics — it is a
    thin wrapper over the one walk (INV-5: one implementation, two views)."""
    p = _write_yaml(tmp_path, _reify_shape({'bogus_key': 1}))
    assert census_unknown_config_keys(p) == census_config_keys(p).unknown


# --- (g) fail-open on a malformed hatch ---------------------------------------


@pytest.mark.parametrize(
    'hatch',
    [
        5,                          # non-dict config_key_census
        {'ignore': 'not-a-list'},   # non-list ignore
        {'ignore': [123]},          # non-str entries
        {'ignore': None},
        [],
    ],
)
def test_malformed_hatch_is_treated_as_empty_allowlist(tmp_path, hatch):
    """A malformed hatch must never raise — it degrades to "no allowlist"."""
    p = _write_yaml(
        tmp_path,
        {'config_key_census': hatch, 'bogus_key': 1},
        name='malformed.yaml',
    )
    census = census_config_keys(p)
    assert 'bogus_key' in {uk.path for uk in census.unknown}
    assert not any(ik.reason == 'allowlist' for ik in census.ignored)


# --- (h) load-time stash of BOTH views ----------------------------------------


def test_load_config_stashes_both_censuses(tmp_path, monkeypatch):
    p = _write_yaml(
        tmp_path,
        {
            'project_root': str(tmp_path),
            'config_key_census': {'ignore': ['cpu_governance.*']},
            'cpu_governance': {'weights': {'task': 100}},
            'bogus_key': 1,
        },
        name='config.yaml',
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))
    config = load_config(p)
    assert [uk.path for uk in config.unknown_key_census] == ['bogus_key']
    assert ConfigIgnoredKey('cpu_governance.weights', 'allowlist') in (
        config.ignored_key_census
    )


def test_direct_construction_yields_empty_ignored_census(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    assert OrchestratorConfig().ignored_key_census == []


# --- (i) hot-reload promise ----------------------------------------------------


def test_config_key_census_ignore_is_green_tier():
    """Pins the operator-facing promise the L2 remediation text makes: the
    allowlist can be applied to a LIVE unit via hot-reload.  If this leaf were
    not in RELOADABLE_FIELDS, apply_reload would report restart_required and
    that remediation line would be a lie."""
    assert 'config_key_census.ignore' in RELOADABLE_FIELDS
