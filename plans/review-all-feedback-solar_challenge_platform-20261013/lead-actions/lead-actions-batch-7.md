# Lead task-store actions, batch 7 (from S6 decompose). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-14.

## A. Cancellations (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled)
- 990 -> "subsumed by review-all stream S6 leaf iota (task 1182, docs/prds/review-all-settlement-store-seam.md): MemoryStore._query_settlement_hh and the code and comments this task edits are deleted; lead cancellation 2026-10-14"
- 977 -> same reason as 990
- 910 -> "subsumed by review-all stream S6 leaves alpha (task 1158: TimescaleStore relation literals and validation/waits.py's telemetry name derive from timeseries.relations), kappa (task 1184: ops settlement SQL through SettlementStore) and mu (task 1188: validation/run.py and dropped_readings read through the store); the remaining ops liveness 'FROM telemetry' read is S19's (task 1013, ruling C1); lead cancellation 2026-10-14"

## B. Edge + note
- add_dependency 628 -> 1180; details append on 628: "COORDINATION NOTE 2026-10-14 (review-all S6): depends on task 1180 (S6 theta). Read the half-hour layer through SettlementStore.current(as_of=Statement.settlement_basis) over one SettlementSnapshot, not Resolution.SETTLEMENT_HH queries. Your delivered_check 'hh-layer-through-store' (expects Resolution\\.SETTLEMENT_HH present under src/solar_challenge_platform/period_pack) conflicts with S6 iota (task 1182), whose src-wide absent check retires that symbol: rewrite the check to name the store read (e.g. SettlementStore.current / settlement_basis) before this task lands."

## C. Notes only (details append=True)
- 634: "2026-10-14 (review-all S6, lead ruling Q3): scp-settle's revised_after_billing summary count (S6 eta, task 584) is this cycle report's input for 'settlement revised after a statement used it'; P-Alert does not consume it this program."
- 1082: "2026-10-14 (review-all S6): S6 alpha (task 1158) renames timeseries/retention.py to timeseries/relations.py and S6 delta (task 1167) may move the require_utc_hh_aligned('t0'...) call into timeseries/settlement_store.py::SettlementWindow. If either lands before this task, point your delivered_checks 'retention-hh-width-from-leaf' and 'materialise-checks-before-io' at the new files; if this task lands first, 1158/1167 repath them (removed-path wave)."

## D. Release a held leaf (S23 beta3 on S6 kappa)
- add_dependency 1119 -> 1184; then commit_planning(project_root, task_ids="1119"). Report the manifest_stamping result (it should stamp the S23 sidecar's beta3 block). Do not touch 1065.

Return ONLY JSON: {"cancelled": [...], "skipped": [...], "edges_added": [...], "notes_appended": [...], "released": [{"task": "1119", "commit_planning_result": "..."}], "errors": [...]}
