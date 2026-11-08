# Reading the `landed_scope_diff` series

`landed_scope_diff` rows record, at done, how a task's declared files compare
with the files its merge actually landed. The producer is
`orchestrator/src/orchestrator/workflow.py::TaskWorkflow._stamp_landed_scope`.
The payload contract is the comment on
`orchestrator/src/orchestrator/event_store.py::EventType.landed_scope_diff`.
This page covers how to read the series.

Rows live in `<project_root>/data/orchestrator/runs.db`, table `events`.

## One row per merge: keep the first

The done reconcile can run more than once for the same merge. Every run
emits a row. Only the FIRST row per `(task_id, merge_sha)` is meaningful for
`metadata_files`, because a later run compares against the `metadata.files`
the first run already replaced with the landed diff. So dedupe on the
lowest `id`, not on any row.

## Undeclared-landing rate (the census basis)

`plan_files` is the basis that reproduces the plan.json census. Rows where
no plan was loaded carry `plan_files: null`, and the filter below drops them.

```bash
sqlite3 data/orchestrator/runs.db <<'SQL'
WITH first AS (
  SELECT MIN(id) AS id FROM events
  WHERE event_type = 'landed_scope_diff'
  GROUP BY task_id, json_extract(data, '$.merge_sha')
)
SELECT COUNT(*) AS stamped,
       SUM(json_array_length(e.data, '$.plan_files.undeclared') > 0) AS undeclared
FROM events e JOIN first USING (id)
WHERE json_type(e.data, '$.plan_files') = 'object';
SQL
```

For the `metadata_files` basis, use the same query with `$.metadata_files` in
both paths. Rows where the backend read failed carry `metadata_files: null`
and are dropped the same way.

Cross-check: replaying this coverage rule over 1,912 task merges from
2026-07-13 onward gave 12.8% / 13.6% / 25.4% / 25.5% for July, August,
September and October. If a rate reads near zero, the stamp is comparing
`metadata.files` against itself.

## The dispatch-time declaration

No payload field carries it. Join on `task_id` against the `lock_acquired`
and `set_to_plan` rows in the same table. Note that a task's first
`lock_acquired` row carries the task author's scope, not the architect's
plan.
