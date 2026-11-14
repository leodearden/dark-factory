"""The shared capability-manifest sidecar surface: where a PRD's sidecar sits, and which are tracked.

Rows B1 and B2 of contract C3 in
``plans/review-all-s20-cli-invocation-and-usage-gate-prd.md``. B1's expected
values were pinned from main's
``scripts/audit_manifest_descriptor_drift.py::_contained_sidecar_relpath``.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from polarity_fixture_repo import init_git_repo

from shared.capability_manifest import (
    SIDECAR_SUFFIX,
    ManifestDiscoveryUnavailable,
    sidecar_for,
    sidecar_path,
    tracked_sidecar_paths,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


class TestSidecarFor:
    @pytest.mark.parametrize(
        ('prd_path', 'expected'),
        [
            pytest.param('plans/x-prd.md', 'plans/x-prd.capability-manifest.yaml', id='plain'),
            pytest.param('./plans/x-prd.md', 'plans/x-prd.capability-manifest.yaml', id='dot-slash'),
            pytest.param(
                'plans/../plans/x-prd.md', 'plans/x-prd.capability-manifest.yaml', id='dotdot-in-root'
            ),
            pytest.param('plans/x.prd.md', 'plans/x.prd.capability-manifest.yaml', id='prd-dot-md-stem'),
            pytest.param('../outside-prd.md', None, id='escaping-root'),
            pytest.param('plans/a\x00b-prd.md', None, id='nul-byte'),
            pytest.param('', None, id='empty'),
            pytest.param(None, None, id='none'),
            pytest.param(7, None, id='int'),
            pytest.param(['plans/x-prd.md'], None, id='list'),
            pytest.param(b'plans/x-prd.md', None, id='bytes'),
        ],
    )
    def test_relative_and_non_str_prd_paths(self, tmp_path, prd_path, expected):
        assert sidecar_for(prd_path, tmp_path) == expected

    def test_absolute_prd_path_inside_the_root(self, tmp_path):
        prd_path = str(tmp_path / 'plans' / 'x-prd.md')
        assert sidecar_for(prd_path, tmp_path) == 'plans/x-prd.capability-manifest.yaml'

    def test_absolute_prd_path_outside_the_root(self, tmp_path):
        assert sidecar_for(str(tmp_path.parent / 'x-prd.md'), tmp_path) is None

    def test_project_root_may_be_a_str(self, tmp_path):
        assert sidecar_for('plans/x-prd.md', str(tmp_path)) == 'plans/x-prd.capability-manifest.yaml'

    def test_dot_project_root_resolves_against_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert sidecar_for('plans/../plans/x-prd.md', '.') == 'plans/x-prd.capability-manifest.yaml'

    @pytest.mark.parametrize('prd_key', ['plans/x-prd.md', 'plans/x.prd.md'])
    def test_composes_the_one_derivation_rule(self, tmp_path, prd_key):
        assert sidecar_for(prd_key, tmp_path) == sidecar_path(prd_key)

    def test_the_suffix_is_public_and_is_the_one_sidecar_path_appends(self):
        assert SIDECAR_SUFFIX == '.capability-manifest.yaml'
        assert sidecar_path('plans/x-prd.md').endswith(SIDECAR_SUFFIX)


class TestTrackedSidecarPaths:
    def test_lists_only_the_tracked_sidecars_sorted(self, tmp_path):
        root = init_git_repo(
            tmp_path / 'repo',
            {
                'plans/b-prd.capability-manifest.yaml': 'schema_version: 1\n',
                'docs/prds/a-prd.capability-manifest.yaml': 'schema_version: 1\n',
                'plans/b-prd.md': '# b\n',
            },
        )
        (root / 'plans' / 'c-prd.capability-manifest.yaml').write_text(
            'schema_version: 1\n', encoding='utf-8'
        )
        assert tracked_sidecar_paths(root) == [
            'docs/prds/a-prd.capability-manifest.yaml',
            'plans/b-prd.capability-manifest.yaml',
        ]

    def test_a_non_checkout_raises_naming_the_root(self, tmp_path):
        with pytest.raises(ManifestDiscoveryUnavailable) as excinfo:
            tracked_sidecar_paths(tmp_path)
        assert str(tmp_path) in str(excinfo.value)

    def test_a_missing_git_binary_raises(self, tmp_path, monkeypatch):
        empty_bin = tmp_path / 'empty-bin'
        empty_bin.mkdir()
        monkeypatch.setenv('PATH', str(empty_bin))
        with pytest.raises(ManifestDiscoveryUnavailable) as excinfo:
            tracked_sidecar_paths(tmp_path)
        assert isinstance(excinfo.value.__cause__, OSError)

    def test_a_wedged_git_raises_at_the_timeout(self, tmp_path, monkeypatch):
        sleep = shutil.which('sleep')
        assert sleep is not None
        fake_bin = tmp_path / 'bin'
        fake_bin.mkdir()
        fake_git = fake_bin / 'git'
        fake_git.write_text(f'#!/bin/sh\nexec {sleep} 30\n', encoding='utf-8')
        fake_git.chmod(0o755)
        monkeypatch.setenv('PATH', str(fake_bin))
        with pytest.raises(ManifestDiscoveryUnavailable) as excinfo:
            tracked_sidecar_paths(tmp_path, timeout=0.5)
        assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)
