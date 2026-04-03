#!/usr/bin/env bash
# query-events.sh — query the orchestrator event store for efficiency and robustness insights.
#
# Usage: bash scripts/query-events.sh [runs.db path] [run_id]
#   - Default DB: data/orchestrator/runs.db
#   - If run_id omitted, uses the most recent run
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="${1:-$REPO_ROOT/data/orchestrator/runs.db}"

if [ ! -f "$DB" ]; then
    echo "Error: database not found at $DB" >&2
    exit 1
fi

# Resolve run_id: use argument or most recent
if [ -n "${2:-}" ]; then
    RUN_ID="$2"
else
    RUN_ID=$(sqlite3 "$DB" "SELECT run_id FROM events ORDER BY timestamp DESC LIMIT 1;" 2>/dev/null || true)
    if [ -z "$RUN_ID" ]; then
        echo "Error: no events found in $DB" >&2
        exit 1
    fi
fi

echo "=== Event Store Report ==="
echo "Database: $DB"
echo "Run ID:   $RUN_ID"
echo ""

# --------------------------------------------------------------------------
# 1. Run overview
# --------------------------------------------------------------------------
echo "--- Run Overview ---"
sqlite3 -header -column "$DB" <<SQL
SELECT
    count(*)                                          AS total_events,
    count(DISTINCT task_id)                           AS tasks,
    printf('%.2f', sum(CASE WHEN event_type='invocation_end' THEN cost_usd ELSE 0 END)) AS total_cost,
    count(CASE WHEN event_type='invocation_end' THEN 1 END)   AS invocations,
    count(CASE WHEN event_type='waste_detected' THEN 1 END)   AS waste_events,
    count(CASE WHEN event_type='escalation_created' THEN 1 END) AS escalations
FROM events
WHERE run_id = '$RUN_ID';
SQL
echo ""

# --------------------------------------------------------------------------
# 2. Cost by role
# --------------------------------------------------------------------------
echo "--- Cost by Role ---"
sqlite3 -header -column "$DB" <<SQL
SELECT
    role,
    count(*)                                    AS invocations,
    printf('%.2f', sum(cost_usd))               AS cost_usd,
    printf('%.1f', avg(cost_usd))               AS avg_cost,
    sum(json_extract(data, '$.input_tokens'))    AS input_tokens,
    sum(json_extract(data, '$.output_tokens'))   AS output_tokens,
    sum(json_extract(data, '$.cache_read_tokens'))   AS cache_read,
    sum(json_extract(data, '$.cache_create_tokens')) AS cache_create
FROM events
WHERE run_id = '$RUN_ID' AND event_type = 'invocation_end'
GROUP BY role
ORDER BY sum(cost_usd) DESC;
SQL
echo ""

# --------------------------------------------------------------------------
# 3. Cost by model
# --------------------------------------------------------------------------
echo "--- Cost by Model ---"
sqlite3 -header -column "$DB" <<SQL
SELECT
    json_extract(data, '$.model')   AS model,
    count(*)                        AS invocations,
    printf('%.2f', sum(cost_usd))   AS cost_usd,
    sum(duration_ms) / 1000         AS duration_s
FROM events
WHERE run_id = '$RUN_ID' AND event_type = 'invocation_end'
GROUP BY model
ORDER BY sum(cost_usd) DESC;
SQL
echo ""

# --------------------------------------------------------------------------
# 4. Phase timing per task
# --------------------------------------------------------------------------
echo "--- Phase Timing (per task) ---"
sqlite3 -header -column "$DB" <<SQL
SELECT
    task_id,
    phase,
    printf('%.2f', cost_usd)  AS phase_cost
FROM events
WHERE run_id = '$RUN_ID' AND event_type = 'phase_exit'
ORDER BY task_id, timestamp;
SQL
echo ""

# --------------------------------------------------------------------------
# 5. Waste events
# --------------------------------------------------------------------------
echo "--- Waste Events ---"
WASTE_COUNT=$(sqlite3 "$DB" "SELECT count(*) FROM events WHERE run_id='$RUN_ID' AND event_type='waste_detected';")
if [ "$WASTE_COUNT" -gt 0 ]; then
    sqlite3 -header -column "$DB" <<SQL
SELECT
    task_id,
    phase,
    json_extract(data, '$.waste_type') AS waste_type,
    substr(json_extract(data, '$.summary'), 1, 80) AS summary,
    timestamp
FROM events
WHERE run_id = '$RUN_ID' AND event_type = 'waste_detected'
ORDER BY timestamp;
SQL
else
    echo "(none)"
fi
echo ""

# --------------------------------------------------------------------------
# 6. Escalation lifecycle
# --------------------------------------------------------------------------
echo "--- Escalations ---"
ESC_COUNT=$(sqlite3 "$DB" "SELECT count(*) FROM events WHERE run_id='$RUN_ID' AND event_type LIKE 'escalation%';")
if [ "$ESC_COUNT" -gt 0 ]; then
    sqlite3 -header -column "$DB" <<SQL
SELECT
    task_id,
    event_type,
    json_extract(data, '$.category')      AS category,
    json_extract(data, '$.severity')      AS severity,
    substr(json_extract(data, '$.summary'), 1, 80) AS summary,
    json_extract(data, '$.outcome')       AS outcome,
    timestamp
FROM events
WHERE run_id = '$RUN_ID' AND event_type LIKE 'escalation%'
ORDER BY timestamp;
SQL
else
    echo "(none)"
fi
echo ""

# --------------------------------------------------------------------------
# 7. Merge attempts
# --------------------------------------------------------------------------
echo "--- Merge Attempts ---"
MERGE_COUNT=$(sqlite3 "$DB" "SELECT count(*) FROM events WHERE run_id='$RUN_ID' AND event_type='merge_attempt';")
if [ "$MERGE_COUNT" -gt 0 ]; then
    sqlite3 -header -column "$DB" <<SQL
SELECT
    task_id,
    json_extract(data, '$.outcome')  AS outcome,
    json_extract(data, '$.attempt')  AS attempt,
    timestamp
FROM events
WHERE run_id = '$RUN_ID' AND event_type = 'merge_attempt'
ORDER BY timestamp;
SQL
else
    echo "(none)"
fi
echo ""

# --------------------------------------------------------------------------
# 8. Cap hits
# --------------------------------------------------------------------------
echo "--- Cap Hits ---"
CAP_COUNT=$(sqlite3 "$DB" "SELECT count(*) FROM events WHERE run_id='$RUN_ID' AND event_type='cap_hit';")
if [ "$CAP_COUNT" -gt 0 ]; then
    sqlite3 -header -column "$DB" <<SQL
SELECT
    task_id,
    json_extract(data, '$.account_name')     AS account,
    json_extract(data, '$.consecutive_hits')  AS consecutive,
    timestamp
FROM events
WHERE run_id = '$RUN_ID' AND event_type = 'cap_hit'
ORDER BY timestamp;
SQL
else
    echo "(none)"
fi
echo ""

# --------------------------------------------------------------------------
# 9. Task outcomes
# --------------------------------------------------------------------------
echo "--- Task Outcomes ---"
sqlite3 -header -column "$DB" <<SQL
SELECT
    task_id,
    json_extract(data, '$.outcome')           AS outcome,
    printf('%.2f', cost_usd)                  AS cost_usd,
    duration_ms / 1000                        AS duration_s,
    json_extract(data, '$.agent_invocations') AS invocations,
    json_extract(data, '$.verify_attempts')   AS verifies,
    json_extract(data, '$.review_cycles')     AS reviews,
    printf('%.2f', json_extract(data, '$.steward_cost_usd')) AS steward_cost
FROM events
WHERE run_id = '$RUN_ID' AND event_type = 'task_completed'
ORDER BY cost_usd DESC;
SQL
echo ""

# --------------------------------------------------------------------------
# 10. Failed invocations (success=false)
# --------------------------------------------------------------------------
echo "--- Failed Invocations ---"
FAIL_COUNT=$(sqlite3 "$DB" "SELECT count(*) FROM events WHERE run_id='$RUN_ID' AND event_type='invocation_end' AND json_extract(data,'$.success')=0;")
if [ "$FAIL_COUNT" -gt 0 ]; then
    sqlite3 -header -column "$DB" <<SQL
SELECT
    task_id,
    role,
    phase,
    printf('%.2f', cost_usd)                AS cost_usd,
    json_extract(data, '$.subtype')         AS subtype,
    json_extract(data, '$.model')           AS model,
    timestamp
FROM events
WHERE run_id = '$RUN_ID'
  AND event_type = 'invocation_end'
  AND json_extract(data, '$.success') = 0
ORDER BY timestamp;
SQL
else
    echo "(none)"
fi
echo ""

# --------------------------------------------------------------------------
# 11. Available runs (for reference)
# --------------------------------------------------------------------------
echo "--- Available Runs ---"
sqlite3 -header -column "$DB" <<SQL
SELECT
    run_id,
    min(timestamp) AS started,
    count(*)       AS events,
    count(DISTINCT task_id) AS tasks
FROM events
GROUP BY run_id
ORDER BY started DESC
LIMIT 10;
SQL
