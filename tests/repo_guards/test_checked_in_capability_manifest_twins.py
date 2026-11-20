"""Every tracked capability-manifest sidecar's .md twin mentions each of its task labels.

This reads the committed plans/ and docs/prds/ trees, so it lives in this never-cached module;
the matcher's unit tests are in shared/tests/test_capability_manifest_twins.py.
"""

from __future__ import annotations

import pytest
from capability_manifest_corpus import REPO_ROOT, discover_manifests, missing_from_worktree_message
from capability_manifest_twins import check_md_twin
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_DISCOVERED = discover_manifests()
_SIDECARS = _DISCOVERED or []
_SIDECAR_IDS = [str(p.relative_to(REPO_ROOT)) for p in _SIDECARS]


class TestCheckedInManifestTwins:
    def test_corpus_discovery_is_not_vacuous(self):
        if _DISCOVERED is None:
            pytest.skip('not a git checkout (git ls-files failed)')
        assert _SIDECARS, 'discover_manifests() returned an empty corpus'

    @pytest.mark.parametrize('sidecar', _SIDECARS, ids=_SIDECAR_IDS)
    def test_md_twin_mentions_every_sidecar_label(self, sidecar):
        try:
            message = check_md_twin(sidecar, REPO_ROOT)
        except FileNotFoundError:
            pytest.fail(missing_from_worktree_message(sidecar))
        assert message is None, message
