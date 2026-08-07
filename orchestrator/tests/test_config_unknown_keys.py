"""Tests for the unknown-config-key census (PRD warm-lane-exhaustion-hardening leaf ζ).

Root cause of the 2026-07-22 reify incident: a top-level ``spare_warm_lanes: 8``
in the project YAML silently dropped for 3+ weeks because OrchestratorConfig uses
pydantic ``extra='ignore'`` — the field actually lives on GitConfig.  Pydantic
discards extras BEFORE validation, so unknown keys must be detected by a SEPARATE
raw-YAML-vs-model pass.  These tests pin the pure census engine.
"""

import logging
import os
import sqlite3
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import orchestrator.config
from orchestrator.cli import main
from orchestrator.config import (
    RELOADABLE_FIELDS,
    CensusIgnoreEntry,
    ConfigIgnoredKey,
    ConfigKeyCensus,
    ConfigKeyCensusConfig,
    ConfigUnknownKey,
    OrchestratorConfig,
    census_config_keys,
    census_unknown_config_keys,
    config_unknown_keys_signature,
    load_config,
)
from orchestrator.config_census_ignore import CENSUS_IGNORE_ENTRY_KEYS


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


def test_check_config_escape_hatched_only_exits_zero(tmp_path):
    """The reviewer's explicit requirement: a config whose only non-model keys
    are escape-hatched is NOT a failure — exit 0 — but the hatched paths are
    still LISTED informationally with their reason, so an over-broad glob stays
    auditable rather than becoming an invisible blind spot."""
    p = _write_yaml(
        tmp_path,
        {
            'x_custom': 1,
            'config_key_census': {'ignore': ['cpu_governance.*']},
            'cpu_governance': {'enabled': True, 'weights': {'task': 100}},
        },
        name='config.yaml',
    )
    result = CliRunner().invoke(main, ['check-config', '--config', str(p)])
    assert result.exit_code == 0, result.output
    assert 'x_custom' in result.output
    assert 'cpu_governance.weights' in result.output
    # Each row names WHY it was excused.
    assert 'reserved prefix' in result.output
    assert 'config_key_census.ignore' in result.output
    # ...and it never reads as a failure.
    assert 'ignored' in result.output.lower()


def test_check_config_mixed_lists_both_and_exits_one(tmp_path):
    """A real unknown key still fails the gate, the ignored section is still
    shown, and hatched paths are NOT counted in the unknown total."""
    p = _write_yaml(
        tmp_path,
        {
            'x_custom': 1,
            'config_key_census': {'ignore': ['cpu_governance.*']},
            'cpu_governance': {'weights': {'task': 100}},
            'warm_lane_pool': 6,
        },
        name='config.yaml',
    )
    result = CliRunner().invoke(main, ['check-config', '--config', str(p)])
    assert result.exit_code == 1, result.output
    assert 'warm_lane_pool' in result.output
    assert 'git.warm_lane_pool' in result.output  # advisory placement hint
    # The informational section is still present alongside the failure.
    assert 'x_custom' in result.output
    assert 'cpu_governance.weights' in result.output
    # Exactly ONE unknown key — the two hatched paths are not in the total.
    assert '1 unknown config key(s)' in result.output


def test_check_config_clean_shows_no_ignored_section(tmp_path):
    p = _write_yaml(
        tmp_path,
        {'max_concurrent_tasks': 3, 'git': {'remote': 'origin'}},
        name='config.yaml',
    )
    result = CliRunner().invoke(main, ['check-config', '--config', str(p)])
    assert result.exit_code == 0, result.output
    assert 'OK' in result.output
    assert 'ignored' not in result.output.lower()


def test_config_key_census_ignore_is_green_tier():
    """Pins the operator-facing promise the L2 remediation text makes: the
    allowlist can be applied to a LIVE unit via hot-reload.  If this leaf were
    not in RELOADABLE_FIELDS, apply_reload would report restart_required and
    that remediation line would be a lie."""
    assert 'config_key_census.ignore' in RELOADABLE_FIELDS


# --- (j) unparseable / unreadable config → parse_error sentinel ----------------
#
# The census is fail-open BY DESIGN for its non-CLI consumers: a file it cannot
# read or parse yields empty key lists rather than raising, because load_config
# surfaces parse errors loudly on its own path.  But an empty census then has two
# utterly different causes — "parsed, nothing unknown" and "nothing was parsed at
# all" — and check-config, which deliberately bypasses load_config, could not tell
# them apart and printed the affirmative `OK:` for a config it never inspected.
# `parse_error` is the third view that separates the two, WITHOUT changing what
# `unknown`/`ignored` mean for anyone already reading them.

_MALFORMED_YAML = 'git:\n  remote: origin\n bad_indent: [1,\n'
# Bytes no UTF-8 decoder will accept (0xff is never a valid start byte).
_NON_UTF8_BYTES = b'git:\n  remote: \xff\xfe origin\n'


def test_malformed_yaml_sets_parse_error(tmp_path):
    """Unparseable YAML must not read as a clean census — but must not raise."""
    p = tmp_path / 'broken.yaml'
    p.write_text(_MALFORMED_YAML)

    census = census_config_keys(p)  # must NOT raise — fail-open is preserved

    assert isinstance(census.parse_error, str) and census.parse_error, (
        'expected a non-empty parse_error sentinel for malformed YAML, '
        f'got {census.parse_error!r}'
    )
    # INV-2 structured-facts-at-failure: the operator gets the facts, not just a
    # refusal — which file, and the underlying yaml diagnostic (MarkedYAMLError
    # renders file/line/column).
    assert str(p) in census.parse_error
    assert 'line' in census.parse_error
    # ...and the fail-open half is bit-for-bit intact.
    assert census.unknown == []
    assert census.ignored == []


def test_directory_path_sets_parse_error(tmp_path):
    """A directory is reachable through --config (click.Path defaults to
    dir_okay=True), and open() raises IsADirectoryError — an OSError."""
    census = census_config_keys(tmp_path)
    assert census.parse_error is not None
    assert str(tmp_path) in census.parse_error
    assert census.unknown == []


@pytest.mark.skipif(os.geteuid() == 0, reason='root bypasses file permissions')
def test_unreadable_file_sets_parse_error(tmp_path):
    """An existing-but-unreadable file (the ORCH_CONFIG_PATH route's only guard
    is exists()) raises PermissionError — also an OSError."""
    p = _write_yaml(tmp_path, {'max_concurrent_tasks': 3}, name='locked.yaml')
    p.chmod(0o000)
    try:
        census = census_config_keys(p)
        assert census.parse_error is not None
        assert str(p) in census.parse_error
        assert census.unknown == []
    finally:
        p.chmod(0o644)  # so tmp_path teardown cannot fail


def test_non_utf8_file_sets_parse_error(tmp_path):
    """The subtlest "cannot be read at all" shape: the decode happens LAZILY
    inside ``yaml.safe_load(f)``'s read, not in ``open()``, and the resulting
    UnicodeDecodeError is a ValueError — neither an OSError nor a yaml.YAMLError.
    Unless the read guard names it, it is the one unreadable file that escapes
    both handlers and propagates out of a function contracted never to raise."""
    p = tmp_path / 'latin1.yaml'
    p.write_bytes(_NON_UTF8_BYTES)

    census = census_config_keys(p)  # must NOT raise

    assert census.parse_error is not None
    assert str(p) in census.parse_error
    assert census.unknown == []
    assert census.ignored == []
    # ...and the fail-open wrapper stays fail-open for this shape too.
    assert census_unknown_config_keys(p) == []


@pytest.mark.parametrize(
    'body, kind',
    [('- a\n- b\n', 'list'), ('hello\n', 'str')],
    ids=['top-level-list', 'bare-scalar'],
)
def test_non_mapping_document_sets_parse_error(tmp_path, body, kind):
    """A document that PARSES but is not a mapping cannot be a config at all, so
    it must not read as clean either — the census walked nothing."""
    p = tmp_path / 'notamapping.yaml'
    p.write_text(body)

    census = census_config_keys(p)

    assert census.parse_error is not None
    assert 'mapping' in census.parse_error
    assert kind in census.parse_error
    assert census.unknown == []
    assert census.ignored == []


@pytest.mark.parametrize(
    'body', ['', '# nothing here\n'], ids=['empty', 'comments-only']
)
def test_empty_document_is_clean_not_a_parse_error(tmp_path, body):
    """DELIBERATE non-regression boundary: an EMPTY (or comments-only) project
    YAML legitimately means "use all defaults" — pydantic-settings loads it
    without complaint — so it is a genuinely CLEAN census, not a parse failure."""
    p = tmp_path / 'empty.yaml'
    p.write_text(body)

    census = census_config_keys(p)

    assert census.parse_error is None
    assert census.unknown == []
    assert census.ignored == []


def test_clean_config_has_no_parse_error(tmp_path):
    p = _write_yaml(
        tmp_path,
        {'max_concurrent_tasks': 3, 'git': {'remote': 'origin'}},
        name='config.yaml',
    )
    census = census_config_keys(p)
    assert census.parse_error is None
    assert census.unknown == []


def test_census_stays_fail_open_for_non_cli_consumers(tmp_path):
    """INV-5: the public wrapper's signature and fail-open semantics are
    UNCHANGED.  Its non-CLI callers (load_config's stash, the born-at-L2) must
    keep seeing an empty list — not an exception — for a file that cannot be
    parsed, because load_config raises its own loud, marked parse error."""
    p = tmp_path / 'broken.yaml'
    p.write_text(_MALFORMED_YAML)
    assert census_unknown_config_keys(p) == []


# --- (k) check-config fails CLOSED on a config it could not inspect -----------
#
# The single load-bearing regression claim in every one of these is
# `'OK:' not in result.output`.  That affirmative string is what an operator
# reads before a restart (OPERATIONS.md §6a: "Verify with check-config first"),
# and it must never appear for a file that was not parsed at all.


def test_check_config_malformed_yaml_exits_nonzero(tmp_path):
    p = tmp_path / 'broken.yaml'
    p.write_text(_MALFORMED_YAML)

    result = CliRunner().invoke(main, ['check-config', '--config', str(p)])

    assert result.exit_code == 1, result.output
    assert 'OK:' not in result.output
    assert str(p) in result.output
    assert 'Error' in result.output
    assert 'YAML' in result.output
    # Errors go to stderr so a script's stdout capture cannot mistake the
    # diagnostic for a report.  `result.output` is the COMBINED stream under
    # click 8.3, so the positive on stderr alone would still pass if the
    # diagnostic were ALSO echoed to stdout — the negative on the stdout-only
    # view is what actually pins the split.
    assert 'Error' in result.stderr
    assert result.stdout == '', f'diagnostic leaked to stdout: {result.stdout!r}'


def test_check_config_directory_path_exits_nonzero(tmp_path):
    """click.Path(exists=True) defaults to dir_okay=True, so a directory sails
    past the option's own guard and only open() rejects it."""
    result = CliRunner().invoke(main, ['check-config', '--config', str(tmp_path)])

    assert result.exit_code == 1, result.output
    assert 'OK:' not in result.output
    assert str(tmp_path) in result.output


@pytest.mark.skipif(os.geteuid() == 0, reason='root bypasses file permissions')
def test_check_config_unreadable_via_env_path_exits_nonzero(tmp_path, monkeypatch):
    """The ORCH_CONFIG_PATH route's only guard is exists(), so an unreadable
    file reaches the census with nothing in between."""
    p = _write_yaml(tmp_path, {'max_concurrent_tasks': 3}, name='locked.yaml')
    p.chmod(0o000)
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))
    try:
        result = CliRunner().invoke(main, ['check-config'])

        assert result.exit_code == 1, result.output
        assert 'OK:' not in result.output
        assert str(p) in result.output
    finally:
        p.chmod(0o644)  # so tmp_path teardown cannot fail


def test_check_config_non_mapping_document_exits_nonzero(tmp_path):
    p = tmp_path / 'list.yaml'
    p.write_text('- a\n- b\n')

    result = CliRunner().invoke(main, ['check-config', '--config', str(p)])

    assert result.exit_code == 1, result.output
    assert 'OK:' not in result.output
    assert 'mapping' in result.output


def test_check_config_empty_file_still_exits_zero(tmp_path):
    """DELIBERATE non-regression boundary: an empty project YAML means "all
    defaults" and is a real, valid operator configuration — it must stay clean."""
    p = tmp_path / 'empty.yaml'
    p.write_text('')

    result = CliRunner().invoke(main, ['check-config', '--config', str(p)])

    assert result.exit_code == 0, result.output
    assert 'OK:' in result.output


def test_check_config_non_utf8_file_exits_nonzero(tmp_path):
    """A file whose bytes are not valid UTF-8 is unreadable in the same
    operator-visible sense as a permission-denied one, and must produce the same
    structured diagnostic — not a Python traceback."""
    p = tmp_path / 'latin1.yaml'
    p.write_bytes(_NON_UTF8_BYTES)

    result = CliRunner().invoke(main, ['check-config', '--config', str(p)])

    assert result.exit_code == 1, result.output
    assert 'OK:' not in result.output
    assert str(p) in result.output
    assert 'Error' in result.output
    assert 'Traceback' not in result.output


def test_check_config_parse_failure_suppresses_the_ignored_section(tmp_path, monkeypatch):
    """The parse_error guard must return BEFORE the informational block — an
    ORDERING claim, which needs a fixture that can actually violate it.

    A real unparseable file returns ``ignored=[]`` by construction, so the
    existing ``if census.ignored:`` guard would skip the block even with the
    early exit deleted: such a fixture pins nothing beyond what the sibling
    malformed-YAML test already covers.  Forcing the otherwise-unreachable
    combination of a parse_error AND a populated ``ignored`` list is what makes
    the ordering observable.  With nothing parsed, listing keys as "excused from
    the census" would tell an operator the file WAS inspected and found to hold
    deliberately-excused keys — the same false reassurance as the `OK:`.
    """
    p = tmp_path / 'broken_with_x.yaml'
    p.write_text('x_custom: 1\ngit:\n  remote: origin\n bad_indent: [1,\n')

    # check_config imports census_config_keys from orchestrator.config INSIDE
    # the function body, so the lookup happens at call time and patching the
    # module attribute takes effect.
    monkeypatch.setattr(
        orchestrator.config,
        'census_config_keys',
        lambda _path: ConfigKeyCensus(
            [],
            [ConfigIgnoredKey('x_custom', 'reserved_prefix')],
            f'invalid YAML in {p}: mapping values are not allowed here',
        ),
    )

    result = CliRunner().invoke(main, ['check-config', '--config', str(p)])

    assert result.exit_code == 1, result.output
    assert 'OK:' not in result.output
    assert 'excused from the census' not in result.output
    assert 'x_custom' not in result.output


# ===== Reasoned ignore entries (task 3395) ===================================
#
# A bare-string ignore entry is an unfalsifiable assertion about a
# non-OrchestratorConfig consumer that is never re-checked.  These pin the
# widened {path, reason} grammar, the reason reaching the classified key, and
# the check-config / load_config surfacing.


# --- (a) the model accepts the reasoned form ---------------------------------


def test_census_config_accepts_reasoned_entry():
    cfg = ConfigKeyCensusConfig(
        ignore=['a.b', {'path': 'c.d', 'reason': 'read by scripts/x.sh'}]
    )
    assert cfg.ignore[0] == 'a.b'
    assert isinstance(cfg.ignore[0], str)
    entry = cfg.ignore[1]
    assert isinstance(entry, CensusIgnoreEntry)
    assert entry.path == 'c.d'
    assert entry.reason == 'read by scripts/x.sh'


def test_load_config_accepts_reasoned_entry_end_to_end(tmp_path, monkeypatch):
    """BACK-COMPAT HAZARD: if the model field stayed ``list[str]``, the first
    operator to adopt the documented reasoned form would hard-fail pydantic
    validation and take a LIVE unit down at load.  A widened field is not
    optional — accepting the dict form is the whole point."""
    p = _write_yaml(
        tmp_path,
        {
            'project_root': str(tmp_path),
            'config_key_census': {
                'ignore': [{'path': 'cpu_governance.weights', 'reason': 'r'}]
            },
        },
        name='config.yaml',
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))
    config = load_config(p)
    assert config.config_key_census.ignore


def test_census_ignore_entry_keys_match_the_raw_parser():
    """DRIFT GUARD (INV-5).  The entry shape necessarily lives in TWO places —
    the pydantic model that validates the config, and the raw-tree parser the
    census actually runs on (which cannot use the validated model, because it
    must keep working when the config has an unrelated value-level validation
    error).  They MUST agree byte-for-byte on the key names, so the expectation
    is DERIVED from the model rather than hardcoded: adding a field to
    CensusIgnoreEntry without teaching the parser fails right here."""
    assert set(CensusIgnoreEntry.model_fields) == set(CENSUS_IGNORE_ENTRY_KEYS)


def test_widening_ignore_adds_no_new_green_tier_leaf():
    """Widening the element type must not change the RELOADABLE_FIELDS surface:
    the field is still named ``ignore``, so _submodel_leaf_paths still yields
    exactly one leaf and the hot-reload promise above is untouched."""
    assert 'config_key_census.ignore' in RELOADABLE_FIELDS
    census_leaves = {f for f in RELOADABLE_FIELDS if f.startswith('config_key_census.')}
    assert census_leaves == {'config_key_census.ignore'}


# --- (b) the reason reaches the classified key --------------------------------


def test_reasoned_entry_threads_its_note_onto_the_ignored_key(tmp_path):
    p = _write_yaml(
        tmp_path,
        {
            'config_key_census': {
                'ignore': [
                    {
                        'path': 'cpu_governance.weights',
                        'reason': 'read verbatim by scripts/cpu-governed-exec.sh',
                    }
                ]
            },
            'cpu_governance': {'enabled': True, 'weights': {'task': 100}},
        },
    )
    census = census_config_keys(p)
    assert census.unknown == []
    assert ConfigIgnoredKey(
        'cpu_governance.weights',
        'allowlist',
        'read verbatim by scripts/cpu-governed-exec.sh',
    ) in census.ignored


def test_bare_entry_yields_no_note(tmp_path):
    p = _write_yaml(
        tmp_path,
        {
            'config_key_census': {'ignore': ['cpu_governance.weights']},
            'cpu_governance': {'enabled': True, 'weights': {'task': 100}},
        },
    )
    (ik,) = census_config_keys(p).ignored
    assert ik.path == 'cpu_governance.weights'
    assert ik.note is None


def test_reserved_prefix_key_yields_no_note(tmp_path):
    """A reserved-prefix key was never claimed by an operator assertion, so it
    has nothing to justify — note stays None rather than being invented."""
    p = _write_yaml(tmp_path, {'git': {'x_custom': 1}})
    (ik,) = census_config_keys(p).ignored
    assert (ik.path, ik.reason, ik.note) == ('git.x_custom', 'reserved_prefix', None)


def test_positional_ignored_key_equality_still_holds(tmp_path):
    """BACK-COMPAT: ``note`` is appended LAST and DEFAULTED, so every existing
    two-argument construction and tuple-equality assertion in this suite, in
    test_harness_config_unknown_keys.py, and in config.py itself stays valid."""
    p = _write_yaml(
        tmp_path,
        {'config_key_census': {'ignore': ['warm_lane_pool']}, 'warm_lane_pool': 8},
    )
    assert ConfigIgnoredKey('warm_lane_pool', 'allowlist') in census_config_keys(p).ignored


def test_first_matching_entry_supplies_the_note(tmp_path):
    """fnmatch classification is first-match-wins, so the note must come from
    the FIRST matching entry — otherwise a broad later glob could silently
    overwrite a specific entry's justification."""
    p = _write_yaml(
        tmp_path,
        {
            'config_key_census': {
                'ignore': [
                    {'path': 'cpu_governance.weights', 'reason': 'specific first'},
                    {'path': 'cpu_governance.*', 'reason': 'broad second'},
                ]
            },
            'cpu_governance': {'enabled': True, 'weights': {'task': 100}},
        },
    )
    (ik,) = census_config_keys(p).ignored
    assert ik.note == 'specific first'


def test_unknown_classification_is_identical_with_and_without_reasons(tmp_path):
    """The .unknown half — which drives the WARNING, the born-at-L2 and the
    exit-1 gate — must be byte-identical whether or not entries carry reasons.
    Adding justifications is a documentation change, not a behaviour change."""
    bare = _write_yaml(
        tmp_path,
        {
            'config_key_census': {'ignore': ['cpu_governance.weights']},
            'cpu_governance': {'weights': {'task': 100}},
            'bogus_key': 1,
        },
        name='bare.yaml',
    )
    reasoned = _write_yaml(
        tmp_path,
        {
            'config_key_census': {
                'ignore': [{'path': 'cpu_governance.weights', 'reason': 'r'}]
            },
            'cpu_governance': {'weights': {'task': 100}},
            'bogus_key': 1,
        },
        name='reasoned.yaml',
    )
    assert census_config_keys(bare).unknown == census_config_keys(reasoned).unknown
    assert [uk.path for uk in census_config_keys(reasoned).unknown] == ['bogus_key']


# --- (c) check-config surfaces reasons, findings, and a three-way exit code ---


def _seed_done_task(project_root: Path, task_id: int = 111) -> None:
    """A real task store whose only task is DONE — the orphaned-cite fixture."""
    db = project_root / '.taskmaster' / 'tasks' / 'tasks.db'
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(
        'CREATE TABLE tasks (tag TEXT, id INTEGER, status TEXT, metadata TEXT, '
        'PRIMARY KEY (tag, id))'
    )
    conn.execute(
        'INSERT INTO tasks (tag, id, status, metadata) VALUES (?,?,?,?)',
        ('master', task_id, 'done', None),
    )
    conn.commit()
    conn.close()


def _check_config(p: Path):
    return CliRunner().invoke(main, ['check-config', '--config', str(p)])


def test_check_config_prints_reason_and_debt_marker(tmp_path):
    """ACCEPTANCE (a): a new un-reasoned entry is VISIBLE as debt, and a
    reasoned one shows the justification the operator actually wrote."""
    p = _write_yaml(
        tmp_path,
        {
            'config_key_census': {'ignore': [
                {'path': 'cpu_governance.weights',
                 'reason': 'read verbatim by scripts/cpu-governed-exec.sh'},
                'fairness.scheduler_v2',
            ]},
            'cpu_governance': {'weights': {'task': 100}},
            'fairness': {'scheduler_v2': True},
        },
        name='config.yaml',
    )
    result = _check_config(p)
    assert 'read verbatim by scripts/cpu-governed-exec.sh' in result.output
    assert 'no reason' in result.output.lower()


def test_check_config_prints_a_findings_section(tmp_path):
    p = _write_yaml(
        tmp_path,
        {'config_key_census': {'ignore': ['warm_lane_pool']}, 'warm_lane_pool': 8},
        name='config.yaml',
    )
    result = _check_config(p)
    assert 'unreasoned' in result.output
    assert 'warm_lane_pool' in result.output


def test_check_config_advisory_only_still_exits_zero(tmp_path):
    """The five grandfathered bare entries must NOT turn a currently-green
    config red on upgrade — advisory findings are exit-neutral and the existing
    OK: contract still holds."""
    p = _write_yaml(tmp_path, _reify_shape(), name='config.yaml')
    result = _check_config(p)
    assert result.exit_code == 0, result.output
    assert 'OK' in result.output


def test_check_config_hard_finding_exits_two(tmp_path):
    """ACCEPTANCE (b): an entry whose cited task has CLOSED is reported loudly
    with its own exit code, naming the id and the status."""
    project = tmp_path / 'proj'
    project.mkdir()
    _seed_done_task(project)
    p = _write_yaml(
        tmp_path,
        {
            'project_root': str(project),
            'config_key_census': {'ignore': [
                {'path': 'warm_lane_pool', 'reason': 'temporary — pending #111'}
            ]},
            'warm_lane_pool': 8,
        },
        name='config.yaml',
    )
    result = _check_config(p)
    assert result.exit_code == 2, result.output
    assert '#111' in result.output
    assert 'done' in result.output
    assert 'orphaned' in result.output


def test_check_config_unknown_key_still_exits_one(tmp_path):
    p = _write_yaml(tmp_path, {'bogus_key': 1}, name='config.yaml')
    assert _check_config(p).exit_code == 1


def test_unknown_keys_dominate_a_hard_finding(tmp_path):
    """Exit 1 keeps its documented meaning — 'at least one genuinely-unknown
    key' — so a caller can never mistake the two signals for each other."""
    project = tmp_path / 'proj'
    project.mkdir()
    _seed_done_task(project)
    p = _write_yaml(
        tmp_path,
        {
            'project_root': str(project),
            'config_key_census': {'ignore': [
                {'path': 'warm_lane_pool', 'reason': 'temporary — pending #111'}
            ]},
            'warm_lane_pool': 8,
            'bogus_key': 1,
        },
        name='config.yaml',
    )
    result = _check_config(p)
    assert result.exit_code == 1, result.output


def test_check_config_survives_a_raising_audit(tmp_path, monkeypatch):
    """A broken lint must never turn a working gate into a crash."""
    def _boom(_path):
        raise RuntimeError('audit exploded')

    monkeypatch.setattr('orchestrator.cli.audit_census_ignore_entries', _boom)
    p = _write_yaml(
        tmp_path, {'max_concurrent_tasks': 3}, name='config.yaml'
    )
    result = _check_config(p)
    assert result.exit_code == 0, result.output
    assert 'OK' in result.output


# --- (d) load_config warns on hard findings (startup AND hot-reload) ----------


def _load_and_capture(tmp_path, monkeypatch, caplog, tree) -> list[str]:
    p = _write_yaml(tmp_path, tree, name='config.yaml')
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))
    with caplog.at_level(logging.WARNING, logger='orchestrator.config'):
        load_config(p)
    return [rec.getMessage() for rec in caplog.records]


def test_load_config_warns_on_a_hard_ignore_finding(tmp_path, monkeypatch, caplog):
    """Loud-over-silent. This ONE call site covers startup AND hot-reload,
    because Harness.reload_config obtains its fresh config from load_config."""
    messages = _load_and_capture(tmp_path, monkeypatch, caplog, {
        'project_root': str(tmp_path),
        'config_key_census': {'ignore': [
            {'path': 'warm_lane_pool', 'reason': 'the orchestrator reads this'}
        ]},
        'warm_lane_pool': 8,
    })
    assert any('warm_lane_pool' in m and 'self-refuting' in m for m in messages), messages


def test_load_config_silent_for_a_clean_config(tmp_path, monkeypatch, caplog):
    messages = _load_and_capture(tmp_path, monkeypatch, caplog, {
        'project_root': str(tmp_path),
        'config_key_census': {'ignore': [
            {'path': 'warm_lane_pool', 'reason': 'read by scripts/deploy.sh'}
        ]},
        'warm_lane_pool': 8,
    })
    assert not any('ignore entr' in m for m in messages), messages


def test_load_config_silent_for_advisory_only_findings(tmp_path, monkeypatch, caplog):
    """The five grandfathered bare entries must not make EVERY startup noisy —
    a warning that always fires is one operators learn to ignore."""
    messages = _load_and_capture(tmp_path, monkeypatch, caplog, {
        'project_root': str(tmp_path), **_reify_shape(),
    })
    assert not any('ignore entr' in m for m in messages), messages


def test_unknown_key_warning_still_fires_independently(tmp_path, monkeypatch, caplog):
    """The two warnings are independent signals: a config can have a hard
    finding, an unknown key, or both, and each must still be named."""
    messages = _load_and_capture(tmp_path, monkeypatch, caplog, {
        'project_root': str(tmp_path),
        'config_key_census': {'ignore': [
            {'path': 'warm_lane_pool', 'reason': 'the orchestrator reads this'}
        ]},
        'warm_lane_pool': 8,
        'spare_warm_lanes': 8,
    })
    assert any('unknown key' in m and 'spare_warm_lanes' in m for m in messages), messages
    assert any('self-refuting' in m for m in messages), messages


def test_load_config_survives_a_raising_audit(tmp_path, monkeypatch):
    """A broken audit must never take down startup OR a hot-reload."""
    def _boom(_path):
        raise RuntimeError('audit exploded')

    monkeypatch.setattr('orchestrator.config.audit_census_ignore_entries', _boom)
    p = _write_yaml(
        tmp_path, {'project_root': str(tmp_path), 'max_concurrent_tasks': 3},
        name='config.yaml',
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))
    assert load_config(p).max_concurrent_tasks == 3
