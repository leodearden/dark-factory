# Cumulative-cost corpus (`sessions.json`)

Real Claude CLI result lines from three resumed sessions, reshaped into
full result JSON objects so tests can feed them through the real parse path.
`shared/src/shared/process_cost.py` derives per-process cost from them.

## Shape

`sessions.json` is an object keyed by full session id. Each value is the
session's invocations in the order they ran:

```json
{"logged_at": "<syslog timestamp, host-local BST>",
 "resumed": false,
 "stdout": {"type": "result", "...": "..."}}
```

The first invocation of every session has `resumed: false`. Every later one
was a `--resume` of the same session id.

## Provenance

All three came from the host syslog, from the orchestrator's
`Agent stdout ... first 500:` lines.

| session | orchestrator pid | owner |
|---|---|---|
| `59607ecd-9ecd-4f90-9ce6-bc71a7fb33e9` | 1325498 | task 4895 steward, 2026-09-22 |
| `f615388b-10b7-4dc9-b86b-4d011dcb0069` | 3246217 | unknown |
| `4bd5a89f-1992-4691-a968-191f2ccc9abf` | 1937083 | unknown |

## Verbatim and reconstructed fields

The syslog kept only the first 500 bytes of each stdout line, so only these
fields are copied verbatim:

- `session_id`;
- `total_cost_usd`, as the exact logged float (for example
  `3.406018500000001`);
- `usage.input_tokens`, `usage.cache_creation_input_tokens`,
  `usage.cache_read_input_tokens` and `usage.output_tokens`;
- `usage.cache_creation.ephemeral_1h_input_tokens` and
  `usage.cache_creation.ephemeral_5m_input_tokens`. Every logged cache write
  was 1h, so the 5m count is 0 throughout.

Everything else was cut off in the log and is reconstructed:

- `type: "result"`, `subtype: "success"`, `is_error: false`, `result: "ok"`,
  `num_turns: 1` and `duration_ms: 1000`;
- `modelUsage`, as a single `claude-opus-5` entry with
  `canonicalModel: "claude-opus-5"` and `costUSD` equal to `total_cost_usd`.
  Its token counts are the running sums of the session's usage blocks over
  the processes that ran on CLI 2.1.277 or later, for a resumed process on
  that CLI. Otherwise they are the process's own usage. That matches
  `costUSD`: a session that crossed the CLI upgrade reports only the spend
  made after it.

## Why the totals are cumulative

CLI 2.1.277 was installed on this host at 2026-09-18 17:22Z. From then on, a
`--resume`d process reports `total_cost_usd` (and `modelUsage[*].costUSD`) as
the running total of the whole session, while its `usage` block still covers
only that process. The CLI session transcript carries no cost field, so the
previous total cannot be re-read from disk.

## The price rule the deltas match

Every usage block here is priced at the `claude-opus-5` list rates, in USD
per 1M tokens:

- input: 5;
- output: 25;
- 1h cache write: 10;
- cache read: 0.5.

For every resume that ran wholly on the new CLI, the reported total minus the
previous total equals that process's own usage priced at these rates. The two
sessions that crossed the upgrade do not satisfy this rule. The expected
numbers live in the tests, not here.
