"""The static/redux asset-build identity: what it is, and where the server publishes it.

Section 1 drives ``tree_identity`` and ``AssetBuild`` over real temp trees,
with no app and no patching. Section 2 drives the real app: every response
under ``/api/v2/dashboard/`` carries both identities, and the on-disk one
follows a content change with no restart.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from _dashboard_routes import dashboard_get_paths

import dashboard
from dashboard.asset_build import (
    ASSET_BUILD_ON_DISK_HEADER,
    ASSET_BUILD_STARTED_HEADER,
    AssetBuild,
    app_asset_build,
    install_asset_build,
    tree_identity,
)

_PACKAGE_REDUX_DIR = Path(dashboard.__file__).parent / 'static' / 'redux'
_CHEAP_ROUTE = '/api/v2/dashboard/memory-graphs'


def _build_tree(root: Path) -> Path:
    (root / 'nested').mkdir(parents=True)
    (root / 'app.jsx').write_text('const App = () => null;\n')
    (root / 'data.js').write_text('window.DF_DATA = {};\n')
    (root / 'nested' / 'styles.css').write_text('body { margin: 0; }\n')
    return root


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    return _build_tree(tmp_path / 'redux')


# --- Section 1: the tree identity and AssetBuild ----------------------------


def test_identity_is_lowercase_hex_and_equal_for_identical_trees(tmp_path: Path) -> None:
    first = tree_identity(_build_tree(tmp_path / 'one'))
    second = tree_identity(_build_tree(tmp_path / 'two'))

    assert first is not None
    assert re.fullmatch(r'[0-9a-f]+', first)
    assert first == second


def test_editing_a_file_content_changes_the_identity(tree: Path) -> None:
    before = tree_identity(tree)

    (tree / 'data.js').write_text('window.DF_DATA = { edited: true };\n')

    assert tree_identity(tree) != before


def test_rewriting_identical_bytes_with_a_new_mtime_keeps_the_identity(tree: Path) -> None:
    before = tree_identity(tree)
    target = tree / 'data.js'
    stat = target.stat()

    target.write_bytes(target.read_bytes())
    os.utime(target, ns=(stat.st_atime_ns + 10**9, stat.st_mtime_ns + 10**9))

    assert target.stat().st_mtime_ns != stat.st_mtime_ns
    assert tree_identity(tree) == before


def test_adding_a_file_changes_the_identity(tree: Path) -> None:
    before = tree_identity(tree)

    (tree / 'extra.js').write_text('// new\n')

    assert tree_identity(tree) != before


def test_removing_a_file_changes_the_identity(tree: Path) -> None:
    before = tree_identity(tree)

    (tree / 'app.jsx').unlink()

    assert tree_identity(tree) != before


def test_adding_a_file_in_a_subdirectory_changes_the_identity(tree: Path) -> None:
    before = tree_identity(tree)

    (tree / 'nested' / 'deeper').mkdir()
    (tree / 'nested' / 'deeper' / 'font.woff').write_bytes(b'\x00\x01')

    assert tree_identity(tree) != before


def test_file_boundaries_cannot_alias(tmp_path: Path) -> None:
    left = tmp_path / 'left'
    left.mkdir()
    (left / 'a').write_text('bc')
    right = tmp_path / 'right'
    right.mkdir()
    (right / 'a').write_text('b')
    (right / 'ac').write_text('')

    assert tree_identity(left) != tree_identity(right)


def test_started_stays_fixed_while_on_disk_follows_each_edit(tree: Path) -> None:
    build = AssetBuild.at_start(tree)
    started = build.started
    assert started == tree_identity(tree)
    assert build.on_disk() == started

    (tree / 'data.js').write_text('window.DF_DATA = { first: 1 };\n')
    first_edit = tree_identity(tree)

    assert build.started == started
    assert build.on_disk() == first_edit != started

    (tree / 'data.js').write_text('window.DF_DATA = { second: 2 };\n')
    second_edit = tree_identity(tree)

    assert build.started == started
    assert build.on_disk() == second_edit
    assert second_edit not in (started, first_edit)


def test_a_missing_root_has_no_identity(tmp_path: Path) -> None:
    missing = tmp_path / 'absent'

    build = AssetBuild.at_start(missing)

    assert tree_identity(missing) is None
    assert build.started is None
    assert build.on_disk() is None


@pytest.mark.skipif(os.geteuid() == 0, reason='root reads a mode-000 file regardless')
def test_an_unreadable_root_has_no_identity(tree: Path) -> None:
    unreadable = tree / 'data.js'
    unreadable.chmod(0)
    try:
        build = AssetBuild.at_start(tree)

        assert tree_identity(tree) is None
        assert build.started is None
        assert build.on_disk() is None
    finally:
        unreadable.chmod(0o644)


# --- Section 2: every dashboard API response carries both identities --------


def _with_dummy_parameters(templated_path: str) -> str:
    return re.sub(r'\{[^}]+\}', '1', templated_path)


def test_every_dashboard_api_response_carries_both_identities(client) -> None:
    expected = tree_identity(_PACKAGE_REDUX_DIR)
    assert expected is not None
    routes = dashboard_get_paths()
    assert routes.literal and routes.templated

    responses = {path: client.get(path) for path in routes.literal}
    for templated in routes.templated:
        responses[templated] = client.get(_with_dummy_parameters(templated))
    unknown = client.get('/api/v2/dashboard/no-such-endpoint')
    invalid_post = client.post('/api/v2/dashboard/curator/cancel', json={})
    assert unknown.status_code == 404
    assert invalid_post.status_code == 400
    responses['unknown path'] = unknown
    responses['invalid POST'] = invalid_post

    for label, resp in responses.items():
        assert resp.headers.get(ASSET_BUILD_STARTED_HEADER) == expected, label
        assert resp.headers.get(ASSET_BUILD_ON_DISK_HEADER) == expected, label


@pytest.fixture()
def swapped_build(client, tree: Path):
    from dashboard.app import app

    original = app_asset_build(app)
    assert original is not None
    build = AssetBuild.at_start(tree)
    install_asset_build(app, build)
    yield build
    install_asset_build(app, original)


def test_on_disk_header_follows_a_content_edit_with_no_restart(client, swapped_build: AssetBuild, tree: Path) -> None:
    started = swapped_build.started
    assert started is not None
    target = tree / 'data.js'
    original_content = target.read_bytes()

    before = client.get(_CHEAP_ROUTE)
    target.write_text('window.DF_DATA = { merged: true };\n')
    edited = client.get(_CHEAP_ROUTE)
    edited_identity = tree_identity(tree)
    target.write_bytes(original_content)
    restored = client.get(_CHEAP_ROUTE)

    assert before.headers[ASSET_BUILD_STARTED_HEADER] == started
    assert before.headers[ASSET_BUILD_ON_DISK_HEADER] == started
    assert edited.headers[ASSET_BUILD_STARTED_HEADER] == started
    assert edited_identity not in (None, started)
    assert edited.headers[ASSET_BUILD_ON_DISK_HEADER] == edited_identity
    assert restored.headers[ASSET_BUILD_STARTED_HEADER] == started
    assert restored.headers[ASSET_BUILD_ON_DISK_HEADER] == started
