"""Sleep mode reconciliation system."""

# Shared cap-wait sanity bound for all reconciliation stage runners (agent_loop,
# judge, cli_stage_runner).  Each runner imports this constant directly; see the
# per-caller policy table in shared/src/shared/cli_invoke.py for full rationale.
_RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS: float = 1800.0
