"""``orchestrator.screen_manifest``: the merge screen manifest parser, through its public seam.

INV-1: a malformed entry is an error for THAT check only, and its siblings still
parse; only a whole-document fault yields no checks at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.config import load_config
from orchestrator.screen_manifest import (
    CHECK_KEYS,
    MANIFEST_SENTINEL,
    MANIFEST_TOP_KEYS,
    ManifestCheck,
    ScreenManifest,
    load_screen_manifest,
    parse_screen_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTERED = frozenset({'orchestrator', 'scripts', 'tests/scripts'})
SOURCE = 'fixture/merge_screen.yaml'


def _entry(check_id: object = 'alpha', module: object = 'orchestrator', node_ids: object = None) -> dict:
    return {'id': check_id, 'module': module, 'node_ids': ['tests/test_a.py'] if node_ids is None else node_ids}


def _parse(*entries: object) -> ScreenManifest:
    return parse_screen_manifest({'checks': list(entries)}, REGISTERED, source=SOURCE)


def _errors_by_check(manifest: ScreenManifest) -> dict[str, str]:
    return {error.check: error.message for error in manifest.errors}


def test_the_key_sets_are_the_documented_shape():
    assert MANIFEST_TOP_KEYS == frozenset({'checks'})
    assert CHECK_KEYS == frozenset({'id', 'module', 'node_ids'})


def test_a_well_formed_doc_parses_every_check_in_file_order():
    manifest = _parse(
        _entry('alpha', 'orchestrator', ['tests/test_a.py::test_one', 'tests/test_a.py::test_two']),
        _entry('beta', 'scripts', ['scripts/tests/test_b.py']),
    )
    assert manifest.errors == ()
    assert manifest.checks == (
        ManifestCheck('alpha', 'orchestrator', ('tests/test_a.py::test_one', 'tests/test_a.py::test_two')),
        ManifestCheck('beta', 'scripts', ('scripts/tests/test_b.py',)),
    )
    assert isinstance(manifest.checks, tuple)
    assert isinstance(manifest.checks[0].node_ids, tuple)


def test_parsed_records_are_frozen():
    manifest = _parse(_entry())
    with pytest.raises(AttributeError):
        manifest.checks[0].id = 'other'  # type: ignore[misc]
    with pytest.raises(AttributeError):
        manifest.checks = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ('bad_entry', 'named_check', 'offending'),
    [
        pytest.param({**_entry('bad'), 'extra': 1}, 'bad', 'extra', id='unknown-key'),
        pytest.param(_entry('bad', module='nowhere'), 'bad', "'nowhere'", id='unregistered-module'),
        pytest.param(_entry('bad', node_ids=[]), 'bad', '[]', id='empty-node-ids'),
        pytest.param(_entry('bad', node_ids=['ok.py', 7]), 'bad', '7', id='non-string-node-id'),
        pytest.param(_entry('bad', node_ids='tests/test_a.py'), 'bad', "'tests/test_a.py'", id='node-ids-not-a-list'),
        pytest.param(_entry(42), '#1', '42', id='non-string-id'),
        pytest.param(_entry('two words'), '#1', "'two words'", id='whitespace-id'),
        pytest.param({'module': 'orchestrator', 'node_ids': ['t.py']}, '#1', 'id', id='missing-id'),
        pytest.param(['not', 'a', 'mapping'], '#1', "['not', 'a', 'mapping']", id='non-mapping-entry'),
    ],
)
def test_a_bad_entry_is_an_error_for_that_check_and_its_siblings_still_parse(bad_entry, named_check, offending):
    manifest = _parse(_entry('first'), bad_entry, _entry('last'))
    assert [check.id for check in manifest.checks] == ['first', 'last']
    errors = _errors_by_check(manifest)
    assert list(errors) == [named_check]
    assert offending in errors[named_check]


def test_a_duplicate_id_errors_both_entries_and_keeps_the_rest():
    manifest = _parse(_entry('dup'), _entry('solo'), _entry('dup', module='scripts'))
    assert [check.id for check in manifest.checks] == ['solo']
    assert [error.check for error in manifest.errors] == ['dup', 'dup']
    assert all('duplicate' in error.message for error in manifest.errors)


def _assert_whole_document_fault(manifest: ScreenManifest, source: str, offending: str) -> None:
    assert manifest.checks == ()
    assert len(manifest.errors) == 1
    (error,) = manifest.errors
    assert error.check == MANIFEST_SENTINEL
    assert source in error.message
    assert offending in error.message


@pytest.mark.parametrize(
    ('doc', 'offending'),
    [
        pytest.param({'checks': [_entry()], 'extra': 1}, 'extra', id='extra-top-key'),
        pytest.param({'chex': [_entry()]}, 'chex', id='wrong-top-key'),
        pytest.param(None, 'NoneType', id='empty-document'),
        pytest.param(['checks'], 'list', id='non-mapping-document'),
        pytest.param({'checks': []}, '[]', id='empty-checks'),
        pytest.param({'checks': {'id': 'alpha'}}, "{'id': 'alpha'}", id='checks-not-a-list'),
    ],
)
def test_a_whole_document_fault_yields_no_checks_and_one_sentinel_error(doc, offending):
    manifest = parse_screen_manifest(doc, REGISTERED, source=SOURCE)
    _assert_whole_document_fault(manifest, SOURCE, offending)


def test_unreadable_yaml_is_one_sentinel_error_naming_the_path(tmp_path):
    path = tmp_path / 'merge_screen.yaml'
    path.write_text('checks: [unclosed\n', encoding='utf-8')
    _assert_whole_document_fault(load_screen_manifest(path, REGISTERED), str(path), 'unreadable')


def test_undecodable_bytes_are_one_sentinel_error_naming_the_path(tmp_path):
    path = tmp_path / 'merge_screen.yaml'
    path.write_bytes(b'checks:\n  - id: \xff\xfe\n')
    _assert_whole_document_fault(load_screen_manifest(path, REGISTERED), str(path), 'unreadable')


def test_a_missing_file_is_one_sentinel_error_naming_the_path(tmp_path):
    path = tmp_path / 'absent.yaml'
    _assert_whole_document_fault(load_screen_manifest(path, REGISTERED), str(path), 'unreadable')


def test_load_parses_a_readable_file(tmp_path):
    path = tmp_path / 'merge_screen.yaml'
    path.write_text(
        'checks:\n  - id: alpha\n    module: scripts\n    node_ids:\n      - scripts/tests/test_x.py\n',
        encoding='utf-8',
    )
    manifest = load_screen_manifest(path, REGISTERED)
    assert manifest == ScreenManifest(checks=(ManifestCheck('alpha', 'scripts', ('scripts/tests/test_x.py',)),), errors=())


def test_the_shipped_manifest_parses_clean_against_the_shipped_registry(monkeypatch):
    monkeypatch.setenv('ORCH_PROJECT_ROOT', str(REPO_ROOT))
    config = load_config(REPO_ROOT / 'dark-factory-orchestrator.yaml')
    registered = {mc.prefix for mc in config.module_configs_or_empty.values()}

    manifest = load_screen_manifest(REPO_ROOT / 'scripts' / 'merge_screen.yaml', registered)

    assert manifest.errors == ()
    assert [check.id for check in manifest.checks] == [
        'inline_suppressions',
        'remote_gate_wire_fingerprint',
        'watchdog_fake_systemctl',
        'subproject_test_collection',
        'uv_run_venv_isolation',
        'recovery_veto_predicate_collapse',
        'disjoint_skip_breadth_yaml',
    ]
