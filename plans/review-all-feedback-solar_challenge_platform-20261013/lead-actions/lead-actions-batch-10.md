# Lead task-store actions, batch 10 (from S13 decompose). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-15.

## A. Cancellations (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled)
- 1201 (currently deferred; S13 theta acceptance record) -> "The measured three-group partition (ports+standing / guard+cache / rate-limited reporter) of consent/enforcement.py is acyclic and its reporter state is read only through its own methods; the synthesis found no heuristic-13 symptom, so the module stays whole (review-all S13 remedy, docs/prds/review-all-consent-purpose-catalogue-and-gate-decide.md). The placement-axis half of fk-21cfdda2e0b0 is routed to S5 (task 1191). Lead cancellation 2026-10-15."
- 592 -> "subsumed by review-all stream S13 leaf beta (task 1193): POST /consent no longer accepts expiry (R13 purpose-only body, extra=forbid); lead cancellation 2026-10-15"
- 583: expected already cancelled (batch 8, naming S5-epsilon 1174). Verify with get_task and report its status; do nothing else.

## B. Edges + coordination notes (add_dependency(id=<dependent>, depends_on=<upstream>); details append on the dependent with append=True and NO metadata in that call)
- 617 -> 1192; note on 617: "COORDINATION NOTE 2026-10-15 (review-all S13): depends on task 1192 (S13 alpha, consent/purposes.py REQUIRED_SCOPE). Build terms versions on consent/purposes.py; if versions carry categories, REQUIRED_SCOPE is computed inside that module — never a second categories_for name or a consent/catalogue.py."
- 618 -> 1193; note on 618: "COORDINATION NOTE 2026-10-15 (review-all S13): depends on task 1193 (S13 beta, purpose-only GrantRequest). Add terms_version to the purpose-only body and keep extra='forbid'."
- 619 -> 1198 and 619 -> 1197; note on 619: "COORDINATION NOTE 2026-10-15 (review-all S13): depends on 1198 (S13 epsilon: the consent row lock) and 1197 (S13 delta2: stored ConsentStatus loses EXPIRED). Keep epsilon's row lock and delta2's stored vocabulary; liveness is EffectiveConsentStatus."
- 620 -> 1196; note on 620: "COORDINATION NOTE 2026-10-15 (review-all S13): depends on task 1196 (S13 delta1, EffectiveConsentStatus on the wire). renewals_due keys on effective_status: EXPIRED, or EXPIRING = ACTIVE now but EXPIRED at now+lead. Program contradiction C8 is resolved by R13 + delta1."
- 972 -> 582; note on 972: "COORDINATION NOTE 2026-10-15 (review-all S13): depends on task 582 (re-scoped 2026-10-14 to the lawful-basis branch + GET /consent/purposes). This task assumes 582's basis branch and contends with 1195 (S13 gamma3) on consent/enforcement.py; the second to land adapts."
- note on 582 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S13, lead): task 617 (P-Member delta1a) already depends on this task and largely repeats the lawful-basis catalogue + basis-branch scope. Before dispatch Leo decides which of 582 and 617 owns the basis branch (with esc-582-5's cbs_billing basis conflict). Do not build the branch twice."

## C. No commit_planning in this batch. Never touch 1065.

Return ONLY JSON: {"cancelled": [...], "skipped": [...], "edges_added": [...], "notes_appended": [...], "status_583": "...", "errors": [...]}
