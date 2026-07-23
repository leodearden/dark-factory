"""C3 submit-path request-identity gate tests (PRD merge-worktree-lifecycle-integrity §4 C3 / §5 D1-D3).

Covers the SHA-sensitive merge_request submit gate enforced at
``coalesce_or_enqueue_merge_request`` and surfaced through the escalation
``merge_request`` MCP tool:

- same branch + same SHA → COALESCE (never a second work item, even mid-verify — D1)
- same branch + new SHA + QUEUED (verify not started) → REPLACE iff descendant (D2)
- same branch + new SHA + IN VERIFY → structured REJECT ``duplicate_in_verify`` (D3)

Imports of ``orchestrator.merge_queue`` symbols are done LOCALLY inside each
test so a not-yet-defined symbol never breaks collection during RED.
"""

import pytest


# ---------------------------------------------------------------------------
# TestDecideC3SubmitAction — step-1 RED: pure decision function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDecideC3SubmitAction:
    """step-1 RED: decide_c3_submit_action pure mapping (PRD §4 C3 / §5 D1-D3).

    RED until step-2 impl: C3SubmitAction/decide_c3_submit_action don't exist.
    Mirrors the decide_attach_action test conventions (local imports, pure,
    no git I/O — resolved TipRelation + verifying flag → C3SubmitAction).

    Table:

    +------------------+------------------+------------------+
    | relation         | verifying=False  | verifying=True   |
    +==================+==================+==================+
    | SAME             | COALESCE         | COALESCE         |  (D1: same SHA never rejects)
    +------------------+------------------+------------------+
    | SUBSET           | COALESCE         | COALESCE         |  (stale retry / patch-id twin)
    +------------------+------------------+------------------+
    | SUPERSET         | REPLACE          | REJECT           |  (D2 queued / D3 in-verify)
    +------------------+------------------+------------------+
    | DIVERGENT        | ValueError       | ValueError       |  (resolve_divergent first)
    +------------------+------------------+------------------+
    """

    async def test_same_not_verifying_coalesces(self):
        """SAME + not verifying → COALESCE (D1)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SAME, verifying=False) is C3SubmitAction.COALESCE

    async def test_same_verifying_coalesces(self):
        """SAME + verifying → COALESCE (D1: same SHA never rejects/replaces even during verify)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SAME, verifying=True) is C3SubmitAction.COALESCE

    async def test_subset_not_verifying_coalesces(self):
        """SUBSET + not verifying → COALESCE (stale retry / patch-id-equivalent twin)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SUBSET, verifying=False) is C3SubmitAction.COALESCE

    async def test_subset_verifying_coalesces(self):
        """SUBSET + verifying → COALESCE (containment coalesces regardless of verify state)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SUBSET, verifying=True) is C3SubmitAction.COALESCE

    async def test_superset_not_verifying_replaces(self):
        """SUPERSET + not verifying (QUEUED) → REPLACE (D2: drop old queued, dispatch fresh)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SUPERSET, verifying=False) is C3SubmitAction.REPLACE

    async def test_superset_verifying_rejects(self):
        """SUPERSET + verifying (IN VERIFY) → REJECT (D3: structured duplicate_in_verify)."""
        from orchestrator.merge_queue import C3SubmitAction, TipRelation, decide_c3_submit_action
        assert decide_c3_submit_action(TipRelation.SUPERSET, verifying=True) is C3SubmitAction.REJECT

    async def test_divergent_raises_value_error(self):
        """DIVERGENT → ValueError (must resolve via resolve_divergent first)."""
        from orchestrator.merge_queue import TipRelation, decide_c3_submit_action
        with pytest.raises(ValueError, match='DIVERGENT'):
            decide_c3_submit_action(TipRelation.DIVERGENT, verifying=False)

    async def test_divergent_verifying_also_raises(self):
        """DIVERGENT + verifying → ValueError (same resolve-first constraint)."""
        from orchestrator.merge_queue import TipRelation, decide_c3_submit_action
        with pytest.raises(ValueError, match='DIVERGENT'):
            decide_c3_submit_action(TipRelation.DIVERGENT, verifying=True)
