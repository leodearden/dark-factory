# Lead task-store actions, batch 15 (from S11 decompose). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-15.

## A. Cancellations (get_task first; skip if in-progress/done/cancelled; metadata merge {x_acceptance_reason, x_finding_run} then set_task_status cancelled). For 785 additionally check `git -C /home/leo/src/solar-challenge-platform branch --list task/785` and `ls /home/leo/src/solar-challenge-platform/.worktrees/785`; if a branch with commits ahead of main exists (`git log --oneline main..task/785`), SKIP the cancellation and report it.
- 939 -> "subsumed by review-all stream S11 leaf alpha1 (task 1239: npm workspace + @scp/frontend-shared, token store and test helpers); lead cancellation 2026-10-15"
- 554 -> "subsumed by review-all stream S11 leaf alpha2 (task 1249: the session seam — storedSignIn, 401 handling, admission rule; integration gate B3-B6); lead cancellation 2026-10-15"
- 1049 -> "subsumed by review-all stream S11 leaf gamma (task 1258: one http client with a typed failure and the shared read lifecycle; integration gate B8-B10); lead cancellation 2026-10-15"
- 785 -> "subsumed by review-all stream S11 leaves delta1 (task 1262: bill explainer account picker and /me/accounts ordering — items 1 and 3) and delta2 (task 1264: energy dashboard on the shared read hook — item 4); item 2 (/me/energy reads only supplied days) landed on main via 2476fda and 5ecc31e; lead cancellation 2026-10-15"
- 1043 -> "subsumed by review-all stream S11 leaf delta2 (task 1264); lead cancellation 2026-10-15"
- 717 -> "subsumed by review-all stream S11 leaf delta2 (task 1264); lead cancellation 2026-10-15"
- 786 -> "subsumed by review-all stream S11 leaf zeta (task 1268: generated wire types and the INV-4 enforcing test tests/docs/test_ts_unions_equal_python_enums.py); lead cancellation 2026-10-15"
- 825 -> "subsumed by review-all stream S11 leaf zeta (task 1268); lead cancellation 2026-10-15"

## B. Edges + coordination notes (add_dependency(id=<dependent>, depends_on=<upstream>); details append with append=True and NO metadata in that call)
- 697 -> 1258; note on 697: "COORDINATION NOTE 2026-10-15 (review-all S11, lead ruling Q4): depends on task 1258 (S11 gamma). Build the governed-config read lifecycle on the shared useRetryableLoad / http client from @scp/frontend-shared; no SPA-local copy."
- 605 -> 1258; note on 605: "COORDINATION NOTE 2026-10-15 (review-all S11, lead ruling Q4): depends on task 1258 (S11 gamma). Build the console view on @scp/frontend-shared (session/http/read/format/wire) once 1249/1258 land; no SPA-local read lifecycle."
- note on 638 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S11): build the console view on @scp/frontend-shared (session/http/read/format/wire) once 1249 (S11 alpha2) and 1258 (S11 gamma) land; no SPA-local copies."
- note on 428 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S11): contended with task 1264 (S11 delta2) on frontend-portal AccountScreen.tsx; the second to land adapts."
- note on 474 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S11): a hand-kept frontend/src/api/billingTypes.ts would fail the INV-4 gate once task 1268 (S11 zeta) lands — import from @scp/frontend-shared/wire or regenerate; RunsPanel is contended with 1243 and 1270."
- note on 475 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S11): frontend/src/api/adminTypes.ts is deleted by task 1268 (S11 zeta); import the generated types from @scp/frontend-shared/wire."
- note on 442 (no edge; it is in progress — details append only): "COORDINATION NOTE 2026-10-15 (review-all S11): tasks 1249 (S11 alpha2) and 1268 (S11 zeta) now depend on this task because it turns Role into a set and edits both auth.ts, AdminApp, adminTypes.ts and the portal types.ts that those leaves rework."

## C. No commit_planning. Never touch 1065. Never edit files. Never write memory.

Return ONLY JSON: {"cancelled": [...], "skipped": [...], "edges_added": [...], "notes_appended": [...], "errors": [...]}
