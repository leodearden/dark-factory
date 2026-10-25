# LME θ LLM-axis evidence (task 3721)

This directory holds the evidence behind θ's LLM-axis findings for PRD
`plans/local-memory-models-eval-prd.md`. θ ran no arm, because η's
`plans/local-memory-models-eval-screening/screening-verdict.json` has no survivor. So
the directory holds no arm runs. It holds the incumbent's measured production cost and
the graphiti write history that the availability assessment reads. This file is
provenance only: what was read, from where, by which command, and what came back. The
findings are in `plans/local-memory-models-eval-llm-report.md`, which carries the
delivered-check marker. This file spells that marker only as
`PRD[-]MARKER:local-memory-models-eval llm[-]report`.

`fused-memory/tests/arm_harness/test_lme_llm_axis_artifacts.py` re-derives
`incumbent-cost.json` from the committed window and the two ζ control runs, and requires
it to be byte-identical. It also pins the tracked file set and η's empty survivor list.
It carries no integration marker, so the merge lane runs it.

## Layout

| Path | What it is |
|---|---|
| `production-telemetry.jsonl` | The selected measurement window: one `LlmAttemptTelemetry` per line, in `created_at` order (`incumbent_cost.serialize_llm_attempts`) |
| `incumbent-cost.json` | `harness.py incumbent-cost` over the dump: the production cost of that window, priced at control A's rates, plus each control run's replay unit cost (`incumbent_cost.IncumbentCost`) |
| `graphiti-write-history.jsonl` | Query 2 below: graphiti `add_memory`/`add_episode` attempts per day, operation, success and error class, 2026-09-01 up to the cutoff |

The raw dump, the journal and the query logs are not committed. They sat under
`/tmp/lme-theta-3721/`.

## Source

- The journal is `/home/leo/src/dark-factory/data/reconciliation/write_journal.db`, the
  serving fused-memory instance's live write journal. Every read opened it with
  `?mode=ro` and `PRAGMA query_only=ON`. Nothing wrote to the journal, a graph, the
  queue or any store.
- All reads were taken on 2026-10-08 between 11:02:55Z and about 11:12Z, from this
  branch's worktree (base `71c9da8f8a`).
- The cutoff is `2026-10-08T11:00:00+00:00`: the top of the UTC hour before the dump
  began. Every query below ends there.

## 1. Production telemetry and `incumbent-cost.json`

The dump ran from the checkout root, 11:02:55Z to 11:03:40Z, exit 0, 132095 rows. The
earliest is `2026-09-01T00:00:10.729024+00:00`.

```bash
uv run --frozen --project fused-memory python fused-memory/scripts/telemetry_query.py \
  --since 2026-09-01T00:00:00Z --limit 1000000 \
  > /tmp/lme-theta-3721/telemetry-raw.jsonl 2> /tmp/lme-theta-3721/telemetry-raw.err
```

`telemetry_query.py` runs
`fused-memory/src/fused_memory/services/write_journal.py::OPERATOR_TELEMETRY_QUERY`, the
one copy of the token SQL. Then, offline:

```bash
uv run --frozen --project fused-memory python fused-memory/scripts/local_memory_models_eval/harness.py \
  incumbent-cost --telemetry /tmp/lme-theta-3721/telemetry-raw.jsonl \
  --until 2026-10-08T11:00:00+00:00 \
  --pricing-spec plans/local-memory-models-eval-controls/specs/incumbent-generic-a.json \
  --control-run plans/local-memory-models-eval-controls/runs/incumbent-generic-a/20261007T011000Z \
  --control-run plans/local-memory-models-eval-controls/runs/incumbent-generic-b/20261007T013800Z \
  --out-dir plans/local-memory-models-eval-llm
```

Exit 0. It wrote both committed files.

A review amendment later renamed the artifact's counted fields from `writes` to
`attempts` and added the window-end coverage check. On 2026-10-08 the same command ran
again over the same dump, with only `--out-dir /tmp/lme-theta-3721/amend-out` changed.
Exit 0. Its `production-telemetry.jsonl` is byte-identical to the committed one. Its
`incumbent-cost.json` is the committed one: every value matches the first run's, and
only the four renamed keys differ. Its stdout, verbatim:

```text
window: 2026-10-05T11:23:27.597318+00:00 to 2026-10-08T11:00:00+00:00 (2.983708364375 days)
attempts 1850 (53 failed), llm_calls 18104, tokens/attempt 19636.374054054053
usd 6.20433015 at incumbent-generic-a pricing: usd/day 2.0794023384050226, projected usd/30 days 62.38207015215068
replay incumbent-generic-a: usd/episode 0.0027364319999999996 tokens/episode 15688.47 (n 200)
replay incumbent-generic-b: usd/episode 0.002764701 tokens/episode 15836.355 (n 200)
wrote: /tmp/lme-theta-3721/amend-out/production-telemetry.jsonl
wrote: /tmp/lme-theta-3721/amend-out/incumbent-cost.json
```

- **Selection rule.** A row is an LLM attempt when its backend is `graphiti` and its
  operation is `add_memory` or `add_episode` (`incumbent_cost.select_llm_attempts`). In a
  census of the raw dump, all 1850 token-bearing rows are graphiti `add_memory` rows. No
  `add_episode` row carries tokens, and no other operation does either. Each selected
  row is one `backend_ops` row, which is one graphiti attempt. A retried write
  contributes one row per attempt, so `incumbent-cost.json` counts attempts, not writes.
- **Window rule.** The window starts at the `created_at` of the first token-bearing LLM
  attempt and ends at the cutoff, end excluded. Code applies this rule; it is not a
  choice made per run. The dump reaches back to 2026-09-01, past that start. Its latest
  row is at `2026-10-08T11:02:49.957469+00:00`, past the cutoff. So both coverage checks
  passed. Every LLM attempt in the window carried all four token columns, so the
  accounting check passed.
- **Under-count.** OPERATIONS.md §"Per-write telemetry (duration + tokens)": on the
  OpenAI-shaped clients a call records only its successful attempt. A re-prompted
  attempt is not counted, and a call that exhausted its retries records nothing. The
  columns count the LLM client's tokens only, so embedding calls are not in them. The
  measured spend is therefore a lower bound on the incumbent's LLM spend, and it
  excludes embedding spend.
- **Price.** The price is control A's spec: gpt-4o-mini at $0.15 per 1M input tokens
  and $0.60 per 1M output tokens. Its source is in
  `plans/local-memory-models-eval-controls/README.md`. Production runs the same model:
  `fused-memory/config/config.yaml` `llm.model: "gpt-4o-mini"`. Failed attempts are
  priced too, because their recorded tokens were spent.

## 2. `graphiti-write-history.jsonl`

One read-only SQL, run in a `python3` heredoc from the checkout root:

```sql
SELECT substr(bo.created_at,1,10) AS day, wo.operation, bo.success,
       CASE WHEN bo.success=1 THEN NULL
            ELSE substr(bo.error,1,instr(bo.error||':',':')-1) END AS error_class,
       count(*) AS attempts, count(DISTINCT bo.write_op_id) AS writes
FROM backend_ops bo JOIN write_ops wo ON wo.id=bo.write_op_id
WHERE bo.backend='graphiti' AND wo.operation IN ('add_memory','add_episode')
  AND bo.created_at >= '2026-09-01' AND bo.created_at < :until
GROUP BY day, wo.operation, bo.success, error_class
ORDER BY day, wo.operation, bo.success, error_class
```

The connection was
`sqlite3.connect('file:/home/leo/src/dark-factory/data/reconciliation/write_journal.db?mode=ro', uri=True, timeout=5.0)`,
then `PRAGMA query_only=ON` and `PRAGMA temp_store=MEMORY`, with
`:until = '2026-10-08T11:00:00+00:00'`. Each result row was written as
`json.dumps(dict(row), sort_keys=True, ensure_ascii=False)`, one per line. That gave 80
rows in 17.9 s.

A first attempt without `PRAGMA temp_store=MEMORY` failed while executing the query, not
at connect: `sqlite3.OperationalError: unable to open database file`. With the pragma
the same SQL ran.
*Hypothesis:* this session's sandbox refused SQLite's on-disk temp file for the
grouping.

How to read a row:
- `attempts` counts `backend_ops` rows.
- `writes` counts the distinct `write_op_id`s in that group. A write retried across
  groups counts once in each group.
- `error_class` is the error text before its first `:`, and it is null on a success.
  Pre-October FalkorDB timeouts appear as `error_class` `Query timed out`, because
  their text carries no type prefix.

## 3. Forensic observations

These are measurements. Nothing here is inferred unless a line says `Hypothesis:`.

### 3a. What became of the rate-limited writes

```sql
WITH rate_limited AS (
  SELECT bo.write_op_id AS id, max(bo.created_at) AS last_rate_limited
  FROM backend_ops bo JOIN write_ops wo ON wo.id=bo.write_op_id
  WHERE bo.backend='graphiti' AND wo.operation IN ('add_memory','add_episode')
    AND bo.success=0 AND substr(bo.error,1,14)='RateLimitError'
    AND bo.created_at >= '2026-09-01' AND bo.created_at < :until
  GROUP BY bo.write_op_id)
SELECT CASE WHEN EXISTS (
         SELECT 1 FROM backend_ops ok
         WHERE ok.write_op_id=rate_limited.id AND ok.backend='graphiti' AND ok.success=1
           AND ok.created_at > rate_limited.last_rate_limited AND ok.created_at < :until)
       THEN 'later_success' ELSE 'no_later_success' END AS fate,
       wo.terminal_status, count(*) AS writes
FROM rate_limited JOIN write_ops wo ON wo.id=rate_limited.id
GROUP BY fate, wo.terminal_status ORDER BY fate, wo.terminal_status
```

The connection and `:until` were the same as query 2. Result: one row,
`{"fate": "later_success", "terminal_status": "completed", "writes": 499}`. 499
distinct writes had at least one RateLimitError attempt. Every one of them has a later
successful graphiti attempt, and its `write_ops.terminal_status` is `completed`. No
write is in `no_later_success`.

The queue snapshot came from the fused-memory MCP tools, called unscoped at about
2026-10-08T11:11Z. `get_queue_stats` returned
`{"counts": {"completed": 19832, "dead": 1}, "oldest_pending_age_seconds": null,
"dead_by_operation": {"add_episode": 1}}`. `get_dead_letters` returned that one dead
item, which is the 2026-09-07 `add_episode` that failed with
`NodeNotFoundError: node 17f33152-654c-428e-88b9-7c18d10c6a28 not found`. It is the
history's 2026-09-07 failure row. The event dead-letter count was 0. No rate-limited
write is dead-lettered.

### 3b. The rate-limit texts, and which client raised them

```sql
SELECT substr(bo.created_at,1,10) AS day, substr(bo.error,1,90) AS error_head, count(*) AS attempts
FROM backend_ops bo JOIN write_ops wo ON wo.id=bo.write_op_id
WHERE bo.backend='graphiti' AND wo.operation IN ('add_memory','add_episode') AND bo.success=0
  AND substr(bo.error,1,14)='RateLimitError'
  AND bo.created_at >= '2026-09-01' AND bo.created_at < :until
GROUP BY day, error_head ORDER BY day, error_head
```

| Day | Text | Attempts |
|---|---|---|
| 2026-10-03 | `RateLimitError: Rate limit exceeded. Please try again later.` | 119 |
| 2026-10-03 | `RateLimitError: Error code: 429 - …no credits remaining…` | 15 |
| 2026-10-04 | `RateLimitError: Rate limit exceeded. Please try again later.` | 2043 |
| 2026-10-04 | `RateLimitError: Error code: 429 - …no credits remaining…` | 2 |

The full text of the 429 rows comes from a second query:
`SELECT substr(bo.created_at,1,10) AS day, bo.error, count(*) AS attempts, count(DISTINCT bo.write_op_id) AS writes FROM backend_ops bo WHERE bo.backend='graphiti' AND bo.error LIKE '%no credits remaining%' AND bo.created_at >= '2026-09-01' AND bo.created_at < :until GROUP BY day, bo.error ORDER BY day, bo.error`.
That text is one string, with 15 attempts over 7 writes on 2026-10-03 and 2 attempts
over 2 writes on 2026-10-04:

```text
RateLimitError: Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', 'param': None, 'code': 'credit_balance_exhausted'}}
```

The text names neither the LLM client nor the embedder, so the raising client is
unknown. In the `graphiti_core` installed in this worktree's `.venv`, the OpenAI LLM
clients (`llm_client/openai_base_client.py`, `llm_client/openai_generic_client.py`) and
the OpenAI reranker (`cross_encoder/openai_reranker_client.py`) all catch
`openai.RateLimitError` and raise graphiti's own `RateLimitError`. Its default message is
`Rate limit exceeded. Please try again later.` (`llm_client/errors.py`).
`embedder/openai.py` has no such handler.
*Hypothesis:* the 17 raw-429 attempts were raised by the OpenAI embedder, the one
OpenAI path that does not wrap the error. Some of the 2162 `Rate limit exceeded`
attempts may have been the same quota exhaustion on the LLM path, with the reason lost
to the wrapper's fixed message. The journal cannot tell these apart.

### 3c. The other October failure classes

```sql
SELECT substr(bo.error,1,instr(bo.error||':',':')-1) AS error_class,
       substr(bo.error,1,70) AS error_head, count(*) AS attempts
FROM backend_ops bo JOIN write_ops wo ON wo.id=bo.write_op_id
WHERE bo.backend='graphiti' AND wo.operation IN ('add_memory','add_episode') AND bo.success=0
  AND substr(bo.error,1,14)!='RateLimitError'
  AND bo.created_at >= '2026-10-01' AND bo.created_at < :until
GROUP BY error_class, error_head ORDER BY attempts DESC LIMIT 25
```

| Text (first 70 characters) | Attempts |
|---|---|
| `ResponseError: Query timed out` | 8530 |
| `BusyLoadingError: Redis is loading the dataset in memory` | 252 |
| `ConnectionError: Error 111 connecting to localhost:6379. Connect call ` | 50 |
| `CancelledError: ` | 42 |
| `APITimeoutError: Request timed out.` | 12 |

## Not taken

- No arm run, no scratch graph, and no FalkorDB or Qdrant connection.
- No θ-stage VRAM reading.
