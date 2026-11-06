# Lead task-store actions, batch 12 (from S1 decompose, plus one S4 leftover). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-15.

## A. Cancellations (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled)
- 192 -> "subsumed by review-all stream S1 leaf delta (task 555, docs/prds/review-all-billing-engine-evidence-module-and-typed-facts.md: PricedSpan structural register pairing); lead cancellation 2026-10-15"
- 329 -> "subsumed by review-all stream S1 leaf zeta2 (task 519: roster <=> period_to_bill agreement re-rosters a skipped CLOSED final period); lead cancellation 2026-10-15"
- 280 -> "subsumed by review-all stream S1 leaf epsilon (task 267: D6 logs the governed no-row fallback as WARNING; D7 pins the VAT overlay at the governed rate by test); lead cancellation 2026-10-15"
- 879 -> "superseded by review-all stream S1 (tasks 194 alpha, 1209 beta, 1211 theta): PRD D1 keeps the BillingService and Statement value layer in statement.py and moves only the meter-evidence and segment-register partitions out, so this task's split contradicts the measured partition; lead cancellation 2026-10-15"

## B. Edges + coordination notes (add_dependency(id=<dependent>, depends_on=<upstream>); details append on the named task with append=True and NO metadata in that call)
- 521 -> 555; note on 521: "COORDINATION NOTE 2026-10-15 (review-all S1): depends on task 555 (S1 delta). Read the rates a statement was priced at from price_plan's PricedSegment.finance (S1 PRD D7); do not add a second rate record."
- note on 221 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S1): task 1215 (S1 gamma) depends on this task and unifies the 'observed' predicate with the expected half-hour grid that assess_coverage and _band_weights share. Keep this task's frozen register-evidence hold verdict and its named predicate; gamma builds on them."
- note on 462 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S1): task 1211 (S1 theta) decomposes run_billing into named steps; this task's classify-at-_persist edit lands in the persist step. Whichever lands second adapts."
- note on 350 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S1): task 1211 (S1 theta) decomposes run_billing into named steps; this task's sub-GBP-1 carry edits the period step. Whichever lands second adapts."
- note on 228 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S1/S3): items 1-2 (the billing_service_for wiring factory) overlap S3 beta3 (task 1147, the one billing_service constructor with a required as_of). S1 kappa (task 1214) builds on 1147. Re-scope this task to what 1147 leaves, or let S3's owner cancel it naming 1147."
- note on 342 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S1): adjacent to S1 zeta1/zeta2 (tasks 365, 519): an undated CLOSED account is not_supplied under supply_span. Build on supplied(); no second predicate."

## C. S4 leftover: 930 (currently deferred)
- add_dependency 930 -> 1162; note on 930: "COORDINATION NOTE 2026-10-15 (review-all S4, lead): depends on task 1162 (S4 epsilon), whose encode_word frame slots make the reg-5 inf/NaN refusal fall out structurally. After 1162 lands, re-check whether anything of this task remains; if not, cancel it naming 1162."

## D. No commit_planning in this batch. Never touch 1065. Never edit files. Never write memory.

Return ONLY JSON: {"cancelled": [...], "skipped": [...], "edges_added": [...], "notes_appended": [...], "errors": [...]}
