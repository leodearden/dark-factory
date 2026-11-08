# Lead task-store actions, batch 6 (from S3 decompose). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-14.

## A. Cancellations (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled)
- 1001 -> "subsumed by review-all stream S3 leaf alpha (task 1137, docs/prds/review-all-composition-root-per-role.md, D5: one boot sequence configures logging through bootstrap_role; migrations/env.py skips fileConfig); lead cancellation 2026-10-14"
- 395 -> "subsumed by review-all stream S3 leaf beta1 (task 1141, which carries fk-1001a64a576c: create_app(graph) with one typed dependency module replaces the three get_telemetry_store copies); lead cancellation 2026-10-14"
- 390 -> "subsumed by review-all stream S3 leaf beta1 (task 1141: create_app(graph), runtime/asgi.py deleted, no lazy module-level app); lead cancellation 2026-10-14"
- 544 -> "subsumed by review-all stream S3 leaves beta1 (task 1141) and epsilon (task 1151: per-role builders; Wired/build_graph deleted so scp-api and scp-settle no longer compose dispatch); lead cancellation 2026-10-14"

## B. Release a held leaf (S20 delta on S3 delta/epsilon)
- add_dependency 1118 -> 1150; add_dependency 1118 -> 1151; then commit_planning(project_root, task_ids="1118"). Report the manifest_stamping result (it should stamp the S20 sidecar's delta block). Task 1118 also depends on 1116, which stays deferred (held on S4) — that is expected; do not touch 1116 or 1065.

Return ONLY JSON: {"cancelled": [...], "skipped": [...], "released": [{"task": "1118", "edges": [...], "commit_planning_result": "..."}], "errors": [...]}
