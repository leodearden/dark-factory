"""Sleep mode reconciliation system."""

from collections.abc import Mapping
from types import MappingProxyType

# Cap-wait sanity bound shared by the reconciliation stage runners (agent_loop, judge,
# cli_stage_runner). A stage must finish inside its cycle: 1800 s lets a brief cap window
# resolve in-band, where the 14-day default would stall the queue under a sustained cap.
RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS: float = 1800.0
_RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS = RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS  # alias; removed by task 9501

# Recon CLI calls load no settings file, CLAUDE.md or memory index; why each switch
# exists: fused-memory/tests/reconciliation/test_recon_cli_context.py.
RECONCILIATION_CLI_SETTING_SOURCES: tuple[str, ...] = ()
RECONCILIATION_CLI_ENV_OVERRIDES: Mapping[str, str] = MappingProxyType({
    'CLAUDE_CODE_DISABLE_CLAUDE_MDS': '1',  # every CLAUDE.md, ancestor walk included
    'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',  # MEMORY.md
    'DISABLE_AUTO_COMPACT': '1',  # compaction off
    'CLAUDE_CODE_FORCE_SESSION_PERSISTENCE': '1',  # transcripts kept
})
