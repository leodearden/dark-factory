# Lead task-store actions, batch 8 (from S5 decompose + S1 Q4). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-14.

## A. Cancellations / closures (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled)
- 583 -> "subsumed by review-all stream S5 leaf epsilon (task 1174, docs/prds/review-all-ingest-ack-persist-ordering.md): under manual ack after commit (ruling R5, D1) rows wait unacknowledged and are judged when the DB recovers, which is the 2026-09-29 ruling's outcome; no PENDING decision type is built; lead cancellation 2026-10-14"
- 305 -> "human gate decided by ruling R5 and delivered by review-all stream S5 leaf epsilon (task 1174): manual ack after the batch commits, bounded in-flight window (D7), per-connection acks in receipt order (D3), one client per connection (D4), QoS fixed at 1 (D8), timer retry of a failed flush (D6); delta (1170) tests the upgrade guard and theta (1187) runs the real-broker kill-the-store gate on mosquitto; lead closure 2026-10-14"
- 866 -> "subsumed by review-all stream S5 leaf lambda (task 1176, which carries fk-0c0e3ef4e738): one StopDeadline bounds the disconnect; lead cancellation 2026-10-14"
- 865 -> "moot under review-all stream S5 D1 (task 1174): nothing is acknowledged before commit, so the acked-then-lost case cannot arise; lead cancellation 2026-10-14"
- 281 -> "subsumed by review-all stream S5 leaf zeta (task 1179, the persisted IngestTick field set) with the tile half on leaf eta (task 1185); lead cancellation 2026-10-14"
- 549 -> "subsumed by review-all stream S5 leaf zeta (task 1179); the per-RejectReason breakdown stays out of scope (PRD §12); lead cancellation 2026-10-14"
- 934 -> "subsumed by review-all stream S5 leaf eta (task 1185, the ops ingest tile over the new tick fields); lead cancellation 2026-10-14"

## B. Key stamp (metadata merge; 1174 is pending)
- 1174: get_task, then metadata merge {"x_finding_key": <its current list plus "fk-2426a5a9c9f7">} (send the complete list); details append: "2026-10-14 (review-all S1 ruling Q4): this leaf also carries fk-2426a5a9c9f7's ingestion half — Subscriber's consent guard becomes required and AllowAllConsentChecker is the one named opt-out (no consent_guard=None path)."

## C. Edge + notes
- add_dependency 1009 -> 1176; details append on 1009: "COORDINATION NOTE 2026-10-14 (review-all S5): depends on task 1176 (S5 lambda), which rewrites ingestion/service.py's stop under one StopDeadline and deletes tests/ingestion/test_ingest_shutdown_drain.py; lambda bounds only the disconnect, so the stop-during-broker-connect case stays yours — build it on the StopDeadline."
- 1066 (details append): "2026-10-14 (review-all S5, lead ruling Q7): docs/legibility/design-invariants.md gets an ELEVENTH entry, INV-11 'an ingest message is acked only after its fate is final, on its connection, in receipt order', heuristic h10, owning stream S5, enforcing test tests/integration/test_ingest_store_kill.py (task 1187, S5 theta); your T1 accepts eleven headings INV-1..INV-11."
- 1071 (details append): "2026-10-14 (review-all S5): S5 epsilon (task 1174) deletes INGESTION_BUFFER_MAX_DEPTH / StoreForwardBuffer(max_depth=...) and retargets your B2 refusal signal onto INGESTION_INFLIGHT_WINDOW=0; whichever lands second adapts that test."
- 213 (details append): "2026-10-14 (review-all S5): after S5 zeta (task 1179) the IngestTick field set retires dropped_* (replaced by messages_held/overflowed and per-fate counts); your ingest.dropped == 0 gate becomes overflowed == 0 and messages_held drained — adapt the summary assertion when 1179 has landed."

Return ONLY JSON: {"cancelled": [...], "skipped": [...], "stamped": [...], "edges_added": [...], "notes_appended": [...], "errors": [...]}
