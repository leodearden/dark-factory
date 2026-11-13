"""Rotation-prompt pointer from the re-arm command to the auto skill's outcome contract.

`orchestrator/src/orchestrator/harness.py::Harness._run_watcher_rotation` splices
`WATCHER_REARM_OUTCOME_GUIDANCE` into the escalation-watcher-auto rotation prompt,
right after the re-arm command. The outcome policy itself lives once, under
`WATCHER_REARM_OUTCOME_SECTION` in `skills/escalation-watcher-auto/SKILL.md`, the
rotation's system prompt; `orchestrator/tests/test_watcher_outcome_branching.py`
holds that section to the real outcomes of `scripts/watcher-rearm.sh`.
"""

WATCHER_REARM_OUTCOME_SECTION = 'Waiting for the next L1'

WATCHER_REARM_OUTCOME_GUIDANCE = f"""
What to do with a re-arm's result: branch on the `WATCHER_REARM_OUTCOME` marker it prints on stderr exactly as your system prompt's "{WATCHER_REARM_OUTCOME_SECTION}" section says, by its outcome table, its identical-retry cap and its rule for STOP directives in tool results. Never loop on the identical call.
"""
