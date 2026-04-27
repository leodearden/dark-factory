"""Role-allowlist tests for the done_provenance gate.

The 2026-04-27 incident had an `implementer` subprocess call
``set_task_status(done, ...)`` directly. The fused-memory schema-side
validator is the load-bearing control; this test pins the role allowlist as
the additional defense so a regression in the role config can be caught at
unit-test time.
"""

from orchestrator.agents.roles import (
    ARCHITECT,
    DEBUGGER,
    DEEP_REVIEWER,
    IMPLEMENTER,
    JUDGE,
    MERGER,
    REVIEWER_COMPREHENSIVE,
    STEWARD,
)

SET_TASK_STATUS = 'mcp__fused-memory__set_task_status'


class TestStewardCanWriteTaskStatus:
    def test_steward_allows_set_task_status(self):
        assert SET_TASK_STATUS in STEWARD.allowed_tools

    def test_steward_does_not_disallow_set_task_status(self):
        assert SET_TASK_STATUS not in STEWARD.disallowed_tools


class TestWorkflowRolesCannotWriteTaskStatus:
    def test_implementer_disallows_set_task_status(self):
        assert SET_TASK_STATUS in IMPLEMENTER.disallowed_tools

    def test_architect_disallows_set_task_status(self):
        assert SET_TASK_STATUS in ARCHITECT.disallowed_tools

    def test_debugger_disallows_set_task_status(self):
        assert SET_TASK_STATUS in DEBUGGER.disallowed_tools

    def test_merger_disallows_set_task_status(self):
        assert SET_TASK_STATUS in MERGER.disallowed_tools

    def test_judge_disallows_set_task_status(self):
        assert SET_TASK_STATUS in JUDGE.disallowed_tools

    def test_reviewer_comprehensive_disallows_set_task_status(self):
        assert SET_TASK_STATUS in REVIEWER_COMPREHENSIVE.disallowed_tools

    def test_deep_reviewer_disallows_set_task_status(self):
        assert SET_TASK_STATUS in DEEP_REVIEWER.disallowed_tools


class TestStewardPromptDocumentsTwoKinds:
    """The schema is enforced server-side, but the steward needs procedural
    guidance to get the call right the first time."""

    def test_prompt_mentions_kind_merged(self):
        assert 'kind="merged"' in STEWARD.system_prompt

    def test_prompt_mentions_kind_found_on_main(self):
        assert 'kind="found_on_main"' in STEWARD.system_prompt

    def test_prompt_documents_ancestor_check(self):
        assert 'merge-base' in STEWARD.system_prompt
        assert '--is-ancestor' in STEWARD.system_prompt

    def test_prompt_forbids_update_task_for_provenance(self):
        # Layer 1 of the gate is the server-side block on
        # update_task(metadata={done_provenance: ...}); the prompt should
        # match that policy so the steward never tries it.
        assert 'update_task' in STEWARD.system_prompt
