# `cleanup_count_snapshots.py` — Operator Guide

One-shot script to sweep every known project's Graphiti entity summaries and
connected edges for legacy **count-snapshot pollution** (e.g.
`"1505 done / 148 cancelled"`), invalidate the matching edges, refresh affected
entity summaries, and write per-edge rollback-audit memories to Mem0.

This is a **one-shot operator action** — it is NOT scheduled, NOT wired into
reconciliation, and NOT intended to run repeatedly (see PRD §12 OQ3).

---

## Prerequisites

```bash
cd fused-memory
uv sync

# Required env vars
export OPENAI_API_KEY=...
export DASHBOARD_KNOWN_PROJECT_ROOTS=/path/to/proj1,/path/to/proj2
```

---

## Step 1 — Dry run: produce the audit report

```bash
python scripts/cleanup_count_snapshots.py
```

This is the **default mode** (`--apply` is NOT set).  No writes are made.

Output:
- **stdout** — JSON audit report with keys `dry_run`, `generated_at`, `projects`,
  `matches`, `totals`, `summaries_matched`, `failed_refreshes`, `failed_invalidations`.
- **stderr** — Human-readable per-project summary table (not part of the JSON).

Redirect the JSON to a file for review (summary table stays on terminal via stderr):

```bash
python scripts/cleanup_count_snapshots.py > /tmp/snapshot_audit.json
```

Parse the JSON file directly:

```bash
python -m json.tool /tmp/snapshot_audit.json | less
```

---

## Step 2 — Review the report

Open `/tmp/snapshot_audit.json` and check:

1. **`matches`** — list every detected count-snapshot edge. Confirm the known
   evidence entities appear, e.g.:
   - Entity `371b46ea` (reify evidence: `"1505 done / 148 cancelled"`)
   - Entity `5e48dbe6`
   - Entity `96cddd4d`

2. **`totals`** — `edges_matched` should be non-zero if pollution exists.

3. **`projects`** — per-project `entities_scanned` vs `edges_matched` summary.

4. **`matches[*].fact_excerpt`** — review each matched edge text to confirm they
   are genuine count-snapshot strings, not false positives.

If the report looks correct, proceed to Step 3.

---

## Step 3 — Apply: invalidate edges, refresh summaries, write audit memories

```bash
python scripts/cleanup_count_snapshots.py --apply
```

For each matched edge the script:

1. Sets `invalid_at` on the edge via `memory.update_edge` (preserves audit trail).
2. Writes a per-edge rollback-audit memory to Mem0 via `memory.add_memory`:
   - `category`: `observations_and_summaries`
   - `agent_id`: `cleanup-count-snapshots`
   - `metadata.kind`: `count_snapshot_cleanup_audit`
3. Refreshes endpoint entity summaries via `memory.refresh_entity_summary`
   (rebuild from remaining valid edges; refresh failures are non-fatal and
   reported in `failed_refreshes`).

---

## Step 4 — Verify rollback evidence

After `--apply`, confirm audit memories are searchable:

```python
# via MCP or in-process MemoryService:
results = await memory.search(
    query='count_snapshot_cleanup_audit',
    project_id='dark_factory',
)
```

Each invalidated edge should have a corresponding memory with
`metadata.kind == 'count_snapshot_cleanup_audit'` and the original fact text
in `metadata.fact_text_original`.

To rollback: use the `edge_uuid` from the audit memory metadata and call
`memory.update_edge(edge_uuid=..., project_id=..., invalid_at=None)` to
re-validate the edge, then `memory.refresh_entity_summary` to rebuild summaries.

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--apply` | off | Commit invalidations (default: dry-run) |
| `--project-id <id>` | all | Restrict sweep to a single project |
| `--limit-per-project <N>` | 1000 | Abort if any project has > N entities (safety cap) |
| `--yes-i-am-sure` | off | Override the entity count safety cap |

### Single-project example

```bash
python scripts/cleanup_count_snapshots.py --project-id dark_factory --apply
```

### Large project override

```bash
python scripts/cleanup_count_snapshots.py \
    --apply \
    --limit-per-project 5000 \
    --yes-i-am-sure
```

---

## Safety properties

- **Dry-run by default** — no writes unless `--apply` is passed.
- **Edge invalidation, not deletion** — `invalid_at` is set; the historical
  fact is preserved in the audit trail.
- **Non-fatal refresh failures** — a `NodeNotFoundError` on one entity does not
  abort the sweep; it is recorded in `failed_refreshes` in the report.
- **Deduplicated writes** — each edge is invalidated exactly once even if it
  appears under multiple entity endpoints (the Graphiti double-attribution
  pattern).
- **Rollback recoverable** — audit memories in Mem0 carry the original
  `fact_text_original` and `edge_uuid` for re-validation if needed.
