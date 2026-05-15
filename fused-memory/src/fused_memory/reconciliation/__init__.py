"""Sleep mode reconciliation system."""

# Shared cap-wait sanity bound for all reconciliation stage runners (agent_loop,
# judge, cli_stage_runner).  Each runner imports and aliases this constant under its
# own name; see the per-caller policy table in shared/src/shared/cli_invoke.py and
# plans/afk-A1-cap-wait.md for full rationale.
_RECONCILIATION_STAGE_CAP_WAIT_SECS: float = 1800.0
