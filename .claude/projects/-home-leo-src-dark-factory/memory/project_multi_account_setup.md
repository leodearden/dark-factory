---
name: Multi-Account Max Failover Setup
description: Two Claude Max accounts configured for orchestrator failover — max-a is dev-only (drained first), max-b is personal with claude.ai chats
type: project
---

Two Claude Max accounts configured for usage cap failover in orchestrator:
- **max-a** (`CLAUDE_OAUTH_TOKEN_A`): Dev-only account, drained first
- **max-b** (`CLAUDE_OAUTH_TOKEN_B`): Personal account with claude.ai content/chats, used only after max-a is capped

Tokens are long-lived (via `claude setup-token`), stored in `.env` at project root (gitignored), loaded by `dotenv` in `cli.py`.

**Why:** Maximizes orchestrator uptime by failing over to the second account. Dev-only account is prioritized so the personal account stays available for interactive use.

**How to apply:** Never change the account order in config.yaml — max-a must remain first. If adding more accounts, dev/disposable ones go before personal ones.
