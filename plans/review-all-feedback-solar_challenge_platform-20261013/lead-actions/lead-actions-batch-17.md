# Lead task-store actions, batch 17 (from S12 decompose). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-15.

## A. Cancellations (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled). Cancel 708 before 514 (708 depends on 514).
- 708 -> "accepted into review-all S12 leaf zeta (task 1259: core drills ingest-sigkill-mid-drain + broker-restart-mid-drain, GATING on exact drain count; lead ruling Q4); lead cancellation 2026-10-15"
- 514 -> "accepted into review-all S12 (docs/prds/review-all-validation-topology-as-a-value.md): items 1-2 -> S12 zeta task 1259; item 4 -> S12 beta task 1248; item 3 -> task 1016 (resolve_drills + CLI drill check); lead cancellation 2026-10-15"

## B. Key stamps (metadata merge only: update_task(metadata={"x_finding_key": <COMPLETE existing list plus the new key>, "x_finding_run": "review-all-solar_challenge_platform-20261013"}, metadata_mode="merge"); then a separate details append)
- 1016: add "fk-e507cf236576"; details append: "2026-10-15 (review-all S12): this task carries fk-e507cf236576 (resolve_drills + CLI drill check, item 3 of cancelled 514) beside fk-598e0f0aadce; S12 beta (1248) and gamma (1242) depend on it."
- 308: add "fk-8e67094cbf75"; details append: "2026-10-15 (review-all S12): this task carries fk-8e67094cbf75 — its hoist closes the run <-> drills import cycle; S12 leaves 1248, 1242, 1246, 809 and 1263 depend on this branch landing (PRD D1)."

## C. Edges + coordination notes (add_dependency(id=<dependent>, depends_on=<upstream>); details append with append=True and NO metadata in that call)
- 471 -> 1259; note on 471: "COORDINATION NOTE 2026-10-15 (review-all S12): depends on task 1259 (S12 zeta). P-Alert drills build on Drill.exercise(Topology) + OwnTopology + run_through (1253 / 1259); their own topology_containers + migrate + seed composition becomes OwnTopology(..., through=Step.SEED); DrillGroup.ALERT returns with this task."
- 472 -> 1259; note on 472: "COORDINATION NOTE 2026-10-15 (review-all S12): depends on task 1259 (S12 zeta). Build on Drill.exercise(Topology) + OwnTopology + run_through (1253 / 1259); the topology composition becomes OwnTopology(..., through=Step.SEED)."
- 984 -> 1263; note on 984: "COORDINATION NOTE 2026-10-15 (review-all S12, lead): depends on task 1263 (S12 kappa), which deletes every file this task names (tests/_lane_test_paths.py, .github/workflows/validation.yml, tests/validation/test_validation_workflow.py). No dormant job is needed; read lane.targets (PytestTarget) after kappa and re-scope to what remains."
- note on 411 (no edge; blocked task — details append only): "COORDINATION NOTE 2026-10-15 (review-all S12): after S12 lambda (task 1266) billing-golden-weekly targets G1's node id alone and stays HELD per esc-411-2."
- note on 416 (no edge; blocked task — details append only): "COORDINATION NOTE 2026-10-15 (review-all S12): after S12 kappa (task 1263) this task's validation.yml item is moot (the workflows are deleted; cadence = LANES only)."
- note on 213 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S12): after S12 epsilon1 (task 1253) the drain lives in src/solar_challenge_platform/validation/simulated_run.py and validation/pipeline.py."
- note on 1072 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S12, lead ruling Q1): S12 gamma (task 1242) finds the golden account through fleet_shape.home_id(slot) and does not use runtime/seed.py::golden_fleet_account_id; drop that accessor if its only consumer is tests."
- note on 1170 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S12, lead ruling Q5): S12 theta (task 1246) depends on this task and needs a public IngestSession constructor from Settings that IngestionService.from_settings also calls; add it here. theta adds it in ingestion/session.py itself only if this task lands without it."

## D. No commit_planning. Never touch 1065. Never edit files. Never write memory.

Return ONLY JSON: {"cancelled": [...], "skipped": [...], "stamped": [...], "edges_added": [...], "notes_appended": [...], "errors": [...]}
