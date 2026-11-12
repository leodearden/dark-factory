# Lead task-store actions, batch 9 (from S4 decompose). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-14.

## A. Cancellations (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled)
- 895 -> "subsumed by review-all stream S4 leaf gamma (task 1154, docs/prds/review-all-control-feedback-as-typed-values-and-provenance-by-type.md: one AppliedSetpoint echo definition, Leo's 2026-10-08 constraints carried verbatim) with the frame half in leaf epsilon (task 1162); lead cancellation 2026-10-14"
- 893 -> "subsumed by review-all stream S4 leaf epsilon (task 1162: frame reg 4 goes through 2901's row via encode_word); lead cancellation 2026-10-14"
- 853 -> "subsumed by review-all stream S4 leaf eta (task 1168: one gateway batch per window through apply_window); lead cancellation 2026-10-14"

## B. Key stamp (metadata merge)
- 734: get_task; metadata merge {"x_finding_key": <existing list if any, plus "fk-f87240eea84a">, "x_finding_run": "review-all-solar_challenge_platform-20261013"}; details append: "2026-10-14 (review-all S4): this task carries fk-f87240eea84a; leaf lambda (task 1177) completes the remainder (control-typed disposition; RefreshTick embedding RefreshPass) with metadata.x_completes naming this task."

## C. Edges + notes
- add_dependency 511 -> 1168; details append on 511: "COORDINATION NOTE 2026-10-14 (review-all S4, lead ruling on esc-511-8): re-scoped onto S4 eta (task 1168), whose lifecycle table encodes this task's failed-submit behaviour (SUBMIT_FAILED -> RETRYING with the new command); this task's ruled neutral Mode-2 write (esc-585-7 A) becomes the action of eta's RELEASED transition. Build on eta's table; do not add a second release path."
- add_dependency 601 -> 1173; details append on 601: "COORDINATION NOTE 2026-10-14 (review-all S4): depends on task 1173 (S4 iota), which makes CommandProvenance required by type on BatchOutcome and submit_intent; drop this task's duplicate 'outcomes carry provenance' / loop-submit edit and build WindowGovernance on iota's provenance."

## D. Release a held leaf (S20 gamma on S4 loop leaves)
- add_dependency 1116 -> 1168; add_dependency 1116 -> 1173; add_dependency 1116 -> 1177; add_dependency 1116 -> 1178; then commit_planning(project_root, task_ids="1116"). Report the manifest_stamping result (it should stamp the S20 sidecar's gamma block). Do not touch 1065.

Return ONLY JSON: {"cancelled": [...], "skipped": [...], "stamped": [...], "edges_added": [...], "notes_appended": [...], "released": [{"task": "1116", "commit_planning_result": "..."}], "errors": [...]}
