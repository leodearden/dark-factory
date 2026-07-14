"""Tests for shared.branch_names — canonical_queued_branch_name primitive.

Locks the "prepend branch_prefix unless branch already starts with it"
shape-normalization rule shared by orchestrator's ``recover_pending_merges``
(merge_queue_store.py), ``QueuedBranch.parse`` (merge_types.py), and
escalation's merge_status/merge_request handlers (server.py) — the single
source of truth so the "is this branch name already prefixed?" rule is not
duplicated with divergent fidelity across packages.
"""

from __future__ import annotations

import pytest

from shared.branch_names import canonical_queued_branch_name


class TestCanonicalQueuedBranchName:
    @pytest.mark.parametrize(
        'branch, branch_prefix, expected',
        [
            # Bare branch name gets the prefix prepended.
            ('x', 'task/', 'task/x'),
            # Already-prefixed: NO double prefix — this is the bug guard.
            ('task/x', 'task/', 'task/x'),
            # Acceptance-(a) pair: bare and pre-prefixed inputs converge on
            # the same canonical ref.
            ('123', 'task/', 'task/123'),
            ('task/123', 'task/', 'task/123'),
            # Empty prefix leaves the branch unchanged.
            ('x', '', 'x'),
        ],
    )
    def test_canonical_shape(
        self, branch: str, branch_prefix: str, expected: str
    ) -> None:
        """canonical_queued_branch_name prepends branch_prefix unless already present."""
        assert canonical_queued_branch_name(branch, branch_prefix) == expected
