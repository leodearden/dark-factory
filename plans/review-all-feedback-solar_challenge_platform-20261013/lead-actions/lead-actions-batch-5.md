# Lead task-store actions, batch 5 (from S14 decompose). project_root /home/leo/src/solar-challenge-platform. Program doc: plans/review-all-program-solar_challenge_platform-2026-10-14.md. Run: review-all-solar_challenge_platform-20261013. Today 2026-10-14.

## A. Cancellations (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled)
- 567 -> "items 1-2 landed in task 443; item 3 (the image pre-creates and chowns /var/lib/scp/dispatch to the image user, whose uid has one home) is review-all stream S14 leaf beta (task 1128, docs/prds/review-all-instance-render-owns-runtime-identity-and-schedules.md), which carries fk-2ac9ca1ae8db (R14); lead cancellation 2026-10-14"
- 931 -> "subsumed by review-all stream S14 leaf gamma (task 1132): one systemd_units/ calendar + unit-file module below instance/ and ci/ per the PRD §6; lead cancellation 2026-10-14"

## B. Repath delivered_checks on DONE tasks (get_task, take the whole metadata.delivered_checks list, change ONLY the named entry's paths, update_task with metadata_mode="merge" sending the complete list back under "delivered_checks"; the tasks are done — do not touch status or done_provenance)
- 443: entry name "schedule-first-fire-fn" (pattern 'def first_fire_at_or_after\(' ): paths become ["src/solar_challenge_platform/instance/topology.py", "src/solar_challenge_platform/systemd_units/schedule.py"]
- 484: entry name "rows-per-home-month-imported" (pattern 'TICK_METRICS'): paths become ["src/solar_challenge_platform/docsgen/cost_model.py", "src/solar_challenge_platform/docsgen/cost_relations.py"]
If an entry with that exact name is absent, report it and change nothing on that task.

## C. Edge + note
- add_dependency 477 -> 1143; details append on 477: "COORDINATION NOTE 2026-10-14 (review-all S14): depends on task 1143 (S14 epsilon). Publish the caddy/mosquitto host ports only through the `config: Mount | None` parameter 1143 adds to compose._mosquitto/_caddy, and read MOSQUITTO_LISTENER_PORT, MOSQUITTO_CONFIG_TARGET, CADDY_LISTENER_PORT and CADDY_CONFIG_TARGET from instance/render/compose.py (their one home)."

## D. Notes only (details append=True)
- 479: "2026-10-14 (review-all S14): the uid host-init uses for host secrets and state-directory ownership is read from the built image (docker run --rm --entrypoint id <image> -u / -g), never a Python constant: after task 1125 (S14 alpha) the Dockerfile is the uid's one home (R14, S14 D1)."
- 465, 646, 469 (same text each): "2026-10-14 (review-all S14): after task 1132 (S14 gamma) import EveryMinutes, Schedule, DailyAt/MonthlyAt and first_fire_at_or_after from solar_challenge_platform.systemd_units.schedule, not instance/topology.py (strict mypy flags a stale import)."
- 485: "2026-10-14 (review-all S14): measure bytes_per_row for the four relations task 1144 (S14 zeta) adds to docs/operations/cost-inputs.yaml — control_command_log, dispatch_tick, refresh_tick, ingest_tick — which are tagged assumed until then."
- 810: "2026-10-14 (review-all S14): control_command_log's retention is one row of docsgen/cost_relations.py::GROWING_RELATIONS after task 1144 (S14 zeta); landing this task changes that row's retention (same for refresh_tick)."

Return ONLY JSON: {"cancelled": [...], "skipped": [...], "repathed": [{"task", "entry", "ok"}], "edges_added": [...], "notes_appended": [...], "errors": [...]}
