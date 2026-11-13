"""Sleep mode reconciliation system."""

# Cap-wait sanity bound shared by the reconciliation stage runners (agent_loop, judge,
# cli_stage_runner). A stage must finish inside its cycle: 1800 s lets a brief cap window
# resolve in-band, where the 14-day default would stall the queue under a sustained cap.
_RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS: float = 1800.0
