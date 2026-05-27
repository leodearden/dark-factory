"""Consistency tests: no doc/prompt claims 'L1 = human'.

Under the E1 escalation ladder the consumer-per-level contract is:
  L0  agent → steward
  L1  steward/workflow → escalation-watcher-auto  (NOT a human)
  L2  auto-watcher → human  (direct; bypasses auto-watcher)

These tests pin the key prose so future drift is caught at CI time.
They are intentionally *content-pin* tests, not behaviour tests.

Reference: escalation/src/escalation/models.py:3-9 (authoritative contract).
"""

from pathlib import Path

from orchestrator.agents.roles import ARCHITECT, STEWARD
from orchestrator.artifacts import TaskArtifacts
from orchestrator.harness import Harness
from orchestrator.workflow import TaskWorkflow, _StewardReescalated

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRIEFING_PY = (
    Path(__file__).parent.parent / 'src' / 'orchestrator' / 'agents' / 'briefing.py'
)


# ===========================================================================
# Step-1 tests: STEWARD and ARCHITECT prompts in roles.py
# ===========================================================================


class TestRolesPromptsReflectL2Ladder:
    """roles.py: stale 'L1 = human' glosses must be corrected."""

    # --- STEWARD prompt ---

    def test_steward_prompt_no_stale_steward_to_human_arrow(self):
        """The parenthetical '(steward→human)' must be replaced."""
        assert "(steward→human)" not in STEWARD.system_prompt

    def test_steward_prompt_no_steward_arrow_human_ascii(self):
        """ASCII variant '(steward->human)' must also be absent."""
        assert "(steward->human)" not in STEWARD.system_prompt

    def test_steward_prompt_no_steward_space_to_space_human(self):
        """Plain-text variant 'steward → human' must also be absent."""
        assert "steward → human" not in STEWARD.system_prompt

    def test_steward_prompt_has_new_auto_watcher_gloss(self):
        """The replacement gloss must name the actual L1 consumer."""
        assert "(steward→auto-watcher" in STEWARD.system_prompt

    def test_steward_prompt_still_targets_level_1(self):
        """The *level* the steward re-escalates to is still 1 — unchanged."""
        assert "level=1" in STEWARD.system_prompt

    # --- ARCHITECT prompt ---

    def test_architect_prompt_no_human_curator_escalation(self):
        """The architect prompt must not promise L1 → human curator."""
        assert "escalate to a human curator" not in ARCHITECT.system_prompt

    def test_architect_prompt_names_l1_auto_watcher_path(self):
        """After report_false_premise the prompt should name the L1 path."""
        assert "level-1" in ARCHITECT.system_prompt
        assert "auto-watcher" in ARCHITECT.system_prompt


# ===========================================================================
# Step-3 tests: workflow.py / briefing.py / artifacts.py / harness.py
# ===========================================================================


class TestWorkflowDocstringsReflectL2Ladder:
    """workflow.py docstrings must not claim L1 is consumed by a human."""

    def test_steward_reescalated_no_human_intervention(self):
        """_StewardReescalated docstring must not say '(human intervention)'."""
        assert "(human intervention)" not in (_StewardReescalated.__doc__ or "")

    def test_steward_reescalated_mentions_auto_watcher(self):
        """_StewardReescalated docstring must name the actual L1 consumer."""
        assert "auto-watcher" in (_StewardReescalated.__doc__ or "")

    def test_wait_for_resolution_no_level1_human(self):
        """_wait_for_resolution docstring must not say 'level-1 (human)'."""
        doc = TaskWorkflow._wait_for_resolution.__doc__ or ""
        assert "level-1 (human)" not in doc

    def test_wait_for_resolution_mentions_auto_watcher(self):
        """_wait_for_resolution docstring must name the auto-watcher consumer."""
        doc = TaskWorkflow._wait_for_resolution.__doc__ or ""
        assert "auto-watcher" in doc

    def test_await_steward_completion_no_steward_to_human(self):
        """_await_steward_completion docstring must not say '(steward→human)'."""
        doc = TaskWorkflow._await_steward_completion.__doc__ or ""
        assert "(steward→human)" not in doc


class TestArtifactsDocstringsReflectL2Ladder:
    """artifacts.py docstrings must not claim 'only a human' resolves L1 items."""

    def test_write_unactionable_task_no_only_a_human(self):
        """write_unactionable_task docstring must not say 'only a human'."""
        doc = TaskArtifacts.write_unactionable_task.__doc__ or ""
        assert "only a human" not in doc

    def test_write_false_premise_no_only_a_human_curator(self):
        """write_false_premise docstring must not say 'only a human/curator'."""
        doc = TaskArtifacts.write_false_premise.__doc__ or ""
        assert "only a human/curator" not in doc


class TestBriefingPromptReflectsL2Ladder:
    """briefing.py narrowing prompt must not promise 'escalate to a human'."""

    def test_briefing_no_will_then_escalate_to_a_human(self):
        """The substring 'will then escalate to a human' must not appear."""
        text = _BRIEFING_PY.read_text()
        assert "will then escalate to a human" not in text


class TestHarnessDocstringReflectsL2Ladder:
    """harness.py _dismiss_stale_escalations must not say '(steward→human)'."""

    def test_dismiss_stale_no_steward_to_human(self):
        """_dismiss_stale_escalations docstring must not say '(steward→human)'."""
        doc = Harness._dismiss_stale_escalations.__doc__ or ""
        assert "(steward→human)" not in doc

    def test_dismiss_stale_mentions_auto_watcher(self):
        """_dismiss_stale_escalations docstring must name the actual L1 consumer."""
        doc = Harness._dismiss_stale_escalations.__doc__ or ""
        assert "auto-watcher" in doc


# ===========================================================================
# Step-5 tests: audit-notes markdown file
# ===========================================================================

_AUDIT_DOC = _REPO_ROOT / 'plans' / 'task-1503-l1-assumptions-audit.md'

_REQUIRED_HEADINGS = [
    '## Flagged behavioral L1-assumptions',
    '## Files updated in task 1503',
    '## Deferred / out-of-scope',
]

_REQUIRED_CALL_SITES = [
    '_handle_wip_conflict',
    '_handle_wip_recovery',
    '_handle_wip_recovery_no_advance',
    '_handle_unmerged_state',
    '_handle_terminal_exit_on_block',
    '_ensure_l1_escalation_for_blocked',
    'trigger_retry_cap_exhausted',
    'escalate_to_human=',
]


class TestL1AssumptionsAuditDocExists:
    """plans/task-1503-l1-assumptions-audit.md must exist and name all flagged sites."""

    def test_audit_doc_exists(self):
        assert _AUDIT_DOC.exists(), f"Expected audit doc at {_AUDIT_DOC}"

    def test_audit_doc_has_flagged_section(self):
        text = _AUDIT_DOC.read_text()
        assert '## Flagged behavioral L1-assumptions' in text

    def test_audit_doc_has_files_updated_section(self):
        text = _AUDIT_DOC.read_text()
        assert '## Files updated in task 1503' in text

    def test_audit_doc_has_deferred_section(self):
        text = _AUDIT_DOC.read_text()
        assert '## Deferred / out-of-scope' in text

    def test_audit_doc_names_handle_wip_conflict(self):
        assert '_handle_wip_conflict' in _AUDIT_DOC.read_text()

    def test_audit_doc_names_handle_wip_recovery(self):
        assert '_handle_wip_recovery' in _AUDIT_DOC.read_text()

    def test_audit_doc_names_handle_wip_recovery_no_advance(self):
        assert '_handle_wip_recovery_no_advance' in _AUDIT_DOC.read_text()

    def test_audit_doc_names_handle_unmerged_state(self):
        assert '_handle_unmerged_state' in _AUDIT_DOC.read_text()

    def test_audit_doc_names_handle_terminal_exit_on_block(self):
        assert '_handle_terminal_exit_on_block' in _AUDIT_DOC.read_text()

    def test_audit_doc_names_ensure_l1_escalation_for_blocked(self):
        assert '_ensure_l1_escalation_for_blocked' in _AUDIT_DOC.read_text()

    def test_audit_doc_names_trigger_retry_cap_exhausted(self):
        assert 'trigger_retry_cap_exhausted' in _AUDIT_DOC.read_text()

    def test_audit_doc_names_escalate_to_human_param(self):
        assert 'escalate_to_human=' in _AUDIT_DOC.read_text()

    def test_audit_doc_links_to_l2_tiering_plan(self):
        assert 'escalation-l2-tiering.md' in _AUDIT_DOC.read_text()
