# Lead task-store actions, batch 11 (from S10 decompose). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-15.

## A. Edges + coordination notes (add_dependency(id=<dependent>, depends_on=<upstream>); details append on the dependent with append=True and NO metadata in that call)
- 601 -> 1205; note on 601: "COORDINATION NOTE 2026-10-15 (review-all S10): depends on task 1205 (S10 epsilon), which delivers this task's per-window finance read through BillingConfig.from_governed_finance(store, at, base=BillingConfig()) and deletes the current_finance call and the restart notes. Wrap that read in the last-good state machine; WindowGovernance.finance is retyped from OptimiserFinanceConfig to the builder's value."
- 602 -> 1203; note on 602: "COORDINATION NOTE 2026-10-15 (review-all S10): depends on task 1203 (S10 beta). The standing floor is computed over GovernedBattery and to_setpoint has no governed_* parameters (S10 PRD section 7)."
- 603 -> 1204; note on 603: "COORDINATION NOTE 2026-10-15 (review-all S10): depends on task 1204 (S10 gamma). A restore-only outcome that keeps a skip reason adds its row to the PRD section 9.3 BatchOutcome table in the same change."
- note on 899 (no edge; 1202 already depends on 899): "COORDINATION NOTE 2026-10-15 (review-all S10): task 1202 (S10 alpha) depends on this task and then replaces the two-Series Forecast with the one-index shape, folding require_shared_index / require_slot_grid into require_forecast_shape. Land this branch as is; do not widen it."

## B. Key stamp on 699 (status blocked; metadata merge only — update_task(metadata={...}, metadata_mode="merge") with the COMPLETE x_finding_key list = existing entries plus "fk-643fb20ec130", and x_finding_run "review-all-solar_challenge_platform-20261013"; then a separate details append):
- details append on 699: "2026-10-15 (review-all S10, lead ruling Q1): this task carries fk-643fb20ec130's device-encoder half — a device-neutral DispatchIntent with a control-owned encoder (optimiser emits intent; control encodes registers). Task 1020 (S10 theta) covers the prose half only (metadata.x_finding_key_partial). Open on the real-GX milestone."

## C. Release the held leaf S10 epsilon (1205, currently deferred) onto S1 epsilon (267)
- add_dependency 1205 -> 267; then commit_planning(project_root, task_ids="1205"). Report the full commit_planning result including manifest_stamping (it should stamp the S10 sidecar docs/prds/review-all-planner-boundary-contracts.capability-manifest.yaml epsilon block) and any delivered_check_warnings. If commit_planning is rejected (DeliveredCheckPolarityViolation), report the payload verbatim and stop; do not edit anything.

Never touch 1065. Never edit files. Never write memory.

Return ONLY JSON: {"edges_added": [...], "notes_appended": [...], "stamped": [...], "released": [{"task": "1205", "commit_planning_result": "..."}], "errors": [...]}
