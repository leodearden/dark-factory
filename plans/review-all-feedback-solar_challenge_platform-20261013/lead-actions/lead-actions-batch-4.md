# Lead task-store actions, batch 4 (from S17 decompose). project_root /home/leo/src/solar-challenge-platform. Program doc: plans/review-all-program-solar_challenge_platform-2026-10-14.md. Run: review-all-solar_challenge_platform-20261013. Today 2026-10-14.

## A. Edge + note
- add_dependency 596 -> 1123; then update_task(596, details append=True): "COORDINATION NOTE 2026-10-14 (review-all S17, lead ruling): depends on task 1123 (S17 beta1) so the commissioned_* fields land on the one BatterySpec field list in config/contract.py and to_battery_config excludes them by derivation; do not add a second field list."

## B. Key stamp (metadata merge only)
- 446: metadata merge {"x_finding_key": ["fk-1f95e9a4c7b0"], "x_finding_run": "review-all-solar_challenge_platform-20261013"}; then details append: "2026-10-14 (review-all S17): this task carries fk-1f95e9a4c7b0 (every CRU write records the actor). After S17 delta2 (task 1134) lands, stamp audited_session on the one write mechanism api/writes.py rather than per router."

## C. Cancellations (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled)
- 673 -> "subsumed by review-all stream S17 leaf beta1 (task 1123, docs/prds/review-all-wire-schemas-derived-from-one-field-list.md): BatterySpec is the one field list from which BatterySpecIn/Out, the ORM kwargs and the view derive, so the third copy of the defaults disappears; lead cancellation 2026-10-14"
- 684 -> "subsumed by review-all stream S17 leaf beta1 (task 1123): one definition of the governed battery defaults on the BatterySpec field list; lead cancellation 2026-10-14"

## D. Release a held leaf (S21 gamma on S17 beta2)
- add_dependency 1133 -> 1126; then commit_planning(project_root, task_ids="1133") (target pending). Report the manifest_stamping result (it should stamp the S21 sidecar's gamma block with 1133). Do NOT touch task 1065.

Return ONLY JSON: {"edges_added": [...], "stamped": [...], "cancelled": [...], "skipped": [...], "released": [{"task": "1133", "commit_planning_result": "..."}], "errors": [...]}
