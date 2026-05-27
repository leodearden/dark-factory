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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]


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
