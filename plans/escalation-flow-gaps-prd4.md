# Cross-PRD gaps — async-merge-request PRD (“PRD-4”) ↔ escalation-flow PRDs 1–3

Authored 2026-06-04 by the integrating session (not the PRD-4 author) after PRD-4
(`plans/async-merge-request-prd.md`, f210962c2f) landed outside the escalation-flow seam
register. PRD-4 was not a register party; these entries retrofit it. P1 interim tasks
1604–1607 (same incident, reify-3112) are already done/merged and are NOT PRD-4 scope.

1. **The L2 skill's “Merge Submissions — NEVER in the Foreground” hard rule is predicated on
   blocking `merge_request`.** `skills/escalation-watcher/SKILL.md:137-162` (and the esc-2831-78
   rationale) exists because `merge_request` blocks until the worker CAS-advances main (~30 min
   on big repos). PRD-4 P3 makes `merge_request(wait_secs=0)` return immediately with a
   `request_id`; once P3 lands, the rule's rationale inverts — foreground submission becomes
   safe and the background-sub-agent machinery becomes an optimization, not a survival rule.
   PRD-3 owns that prose (seam register). Its L2-skill rewrite task must either keep the rule
   with an explicit “until async merge_request (P3) lands” qualifier, or coordinate landing
   order with PRD-4's compat ladder. Do NOT delete the rule before P3 is deployed.

2. **PRD-2's B3 design rationale has the same predicate.** The background-sub-agent shape for
   `unblock-low-risk` (escalation-watcher/SKILL.md:211-224; unblock-low-risk merge step) exists
   because the merge call blocks. Post-P3 the sub-agent (or even the L2 foreground) can submit
   and poll `merge_status(request_id)`. PRD-2 tasks (1613–1616) should not author fresh prose
   re-asserting the blocking premise without a forward pointer to PRD-4 P3; the durable-cap and
   freshness work is unaffected.

3. **Seam-register row “merge_queue.py verify path — nobody” is now stale.** PRD-4 owns the
   `merge_queue.py` entry model (P4 multi-waiter/generations) and the `server.py`
   `merge_request`/`merge_status`/`merge_cancel` region. No collision with PRD-3's server.py
   claims (resolve_issue handler + submit chokepoint + CATEGORIES — disjoint regions) or PRD-1's
   startup-sweep block. All three escalation-flow PRDs treat the merge worker's post-merge
   re-rebase+verify as an invariant correctness backstop (PRD-2 explicitly); PRD-4 must preserve
   that property through P2–P4 or flag loudly.

4. **`in_flight` response-shape compat.** escalation-watcher/SKILL.md:158-159 documents the
   `merge_request status='in_flight'` duplicate backstop (“do NOT re-queue”). PRD-4 D8 renames
   the coalesce outcome to `attached` + `request_id` (pre-P4 attach semantics). Any PRD-1/2/3
   task or skill prose that hardcodes `'in_flight'` handling must tolerate both values through
   the compat ladder (P3's `wait_secs=None` default preserves blocking behaviour until callers
   migrate).

5. **Landing-order recommendation.** PRD-4 P2 (additive `request_id`) is safe to land anytime.
   P3 (behaviour flip via compat ladder) should land either before PRD-3's L2-skill rewrite task
   (so the rewrite documents the post-P3 world once) or after it with a follow-up prose sweep —
   not interleaved. PRD-4's decomposition should wire explicit task dependencies accordingly.
