"""What every reconciliation CLI call loads, and what it is denied (task 10492).

Every recon CLI call (the stages, the judge and the verify loop) reads no
settings file, loads no CLAUDE.md and no auto-memory index, keeps
auto-compaction off, and keeps its transcript. Each switch has one job:

- Empty setting sources keep out the settings files: hooks, permission rules,
  settings env and ``.mcp.json`` approvals.
- ``CLAUDE_CODE_DISABLE_CLAUDE_MDS`` keeps out every CLAUDE.md, including the
  ancestor-directory walk, which ``--setting-sources ''`` does not cover for the
  user file reached through /home/leo.
- ``CLAUDE_CODE_DISABLE_AUTO_MEMORY`` keeps out MEMORY.md, which
  ``--setting-sources ''`` was measured not to stop.
- ``DISABLE_AUTO_COMPACT`` keeps compaction off, as ``autoCompactEnabled`` in
  ~/.claude/settings.json did before the settings files were dropped.
- ``CLAUDE_CODE_FORCE_SESSION_PERSISTENCE`` keeps transcripts if a child-session
  marker is ever inherited.
"""

from __future__ import annotations

from fused_memory.reconciliation import (
    RECONCILIATION_CLI_ENV_OVERRIDES,
    RECONCILIATION_CLI_SETTING_SOURCES,
)


def test_recon_cli_context_constants_name_one_rule():
    assert list(RECONCILIATION_CLI_SETTING_SOURCES) == []
    assert dict(RECONCILIATION_CLI_ENV_OVERRIDES) == {
        'CLAUDE_CODE_DISABLE_CLAUDE_MDS': '1',
        'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
        'DISABLE_AUTO_COMPACT': '1',
        'CLAUDE_CODE_FORCE_SESSION_PERSISTENCE': '1',
    }
    assert not hasattr(RECONCILIATION_CLI_ENV_OVERRIDES, '__setitem__')
