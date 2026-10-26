# Lead task-store actions, batch 16 (from S7 decompose). project_root /home/leo/src/solar-challenge-platform. Run review-all-solar_challenge_platform-20261013. Today 2026-10-15.

## A. Edges + coordination notes (add_dependency(id=<dependent>, depends_on=<upstream>); details append with append=True and NO metadata in that call). No cancellations in this batch.
- 752 -> 1271; note on 752: "COORDINATION NOTE 2026-10-15 (review-all S7, lead ruling Q4): NOT cancelled — whether OPERATOR may read consent is Leo's ruling. Depends on task 1271 (S7 theta), which adds require_role(CUSTOMER, ADMIN) on every /consent route and one _consent_target from the scope; OPERATOR keeps 403, now with the cause 'Insufficient role'. Leo's ruling lands on 1271's resolver."
- 972 -> 1269; note on 972: "COORDINATION NOTE 2026-10-15 (review-all S7): depends on task 1269 (S7 eta). After it, the occupant comes from models/occupancy.py::occupant(session, property_id, at) and the denial reason names the occupancy outcome (no occupant at `at`)."
- 342 -> 1272; note on 342: "COORDINATION NOTE 2026-10-15 (review-all S7, lead): depends on task 1272 (S7 iota), which makes an undated CLOSED account impossible (ck_accounts_closure_stated on both lanes; PENDING -> CLOSED dates an empty window on the closing day). Re-scope this task to what remains after 1272 — if nothing remains, cancel it naming 1272 (plus 365/519 for the supplied() read)."
- note on 707 (no edge change; 1267 already depends on it): "COORDINATION NOTE 2026-10-15 (review-all S7): task 1267 (S7 zeta) supersedes this revision's properties/devices policies after it lands, including mpans (the history read re-binds its day with reaching_on); zeta's downgrade restores this revision's policies. Land this task as is."
- note on 442 (no edge; 1254 and 1271 depend on it): "COORDINATION NOTE 2026-10-15 (review-all S7): task 1254 (S7 beta) adds Principal.__post_init__ over the role set and scope_of supersedes this task's adaptation of TenantScope.from_principal; STAFF_ROLES stays P-Gate's. 1254 and 1271 depend on this task."
- note on 445 (no edge; 1260 depends on it): "COORDINATION NOTE 2026-10-15 (review-all S7): audited_session takes Depends(tenant_session) once task 1260 (S7 delta) lands; the reach is applied once, below the actor. 1260 depends on this task."
- note on 323 (no edge): "COORDINATION NOTE 2026-10-15 (review-all S7): task 1272 (S7 iota) dates PENDING -> CLOSED as an empty window on the closing day through SUPPLY_MOVES; status-move dating otherwise stays with S1 (this task)."

## B. No commit_planning. Never touch 1065. Never edit files. Never write memory.

Return ONLY JSON: {"edges_added": [...], "notes_appended": [...], "errors": [...]}
