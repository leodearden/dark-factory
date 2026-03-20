---
name: Multi-Account Max Failover Setup
description: Two Claude Max accounts configured for orchestrator failover — max-b drained first (as of 2026-03-20), max-a is fallback
type: project
---

Two Claude Max accounts configured for usage cap failover in orchestrator:
- **max-b** (`CLAUDE_OAUTH_TOKEN_B`): Drained first (as of 2026-03-20 — A was nearing weekly cap)
- **max-a** (`CLAUDE_OAUTH_TOKEN_A`): Dev-only account, used as fallback

Priority order is set by list position in `orchestrator/config.yaml` under `usage_cap.accounts`. `before_invoke()` picks the first non-capped account.

Tokens are long-lived (via `claude setup-token`), stored in `.env` at project root (gitignored), loaded by `dotenv` in `cli.py`.

**Why:** Maximizes orchestrator uptime by failing over to the second account. Order was swapped 2026-03-20 because max-a was nearing its weekly cap.

**How to apply:** To re-prioritize, swap the two entries in `orchestrator/config.yaml` under `usage_cap.accounts`. The user may change this again once caps reset — check before assuming a fixed order.
