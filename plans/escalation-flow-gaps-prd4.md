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

## Integration audit 2026-06-04

Wired by the integration-analyst pass after confirming PRD-4 is FULLY decomposed and queued
(tasks 1628–1642 all live; statuses re-verified 2026-06-04). This section records the concrete
resolution of entries 1–5 above now that real task ids exist on both sides of every seam.

**Entry 5 (landing order) — RESOLVED via explicit dependency.** PRD-4 β6 = task 1636
(retires the `skills/escalation-watcher/SKILL.md:137-162` merge-foreground hard rule). PRD-3's
L2-skill rewrite (the "eta" task) = task 1624 (η, owns everything except the B3 subsection,
which by ownership includes :137-162). Added dependency **1624 → 1636** so β6 lands the
post-P3 merge prose first and η then rewrites its owned sections against the already-migrated
file (the "land P3 before the L2-skill rewrite" arm of entry 5). Both were PENDING; no cycle
(1636 deps 1630/1631/1632; 1624 deps 1617/1620 — disjoint). Consequence: η's critical path now
runs behind the full P3 merge chain α2(1629)→β1(1631)→β2(1632)→β6(1636); accepted because
prose coherence on the shared file dominates latency here.

**Entries 1 + 4 (foreground rule + in_flight→attached) — note-injected.** Task 1624's body did
NOT enumerate the :137-162 region; an `integration_notes` entry was added telling η to treat
that region as β6-owned and to tolerate BOTH `in_flight` and `attached`+`request_id` in any
response-shape prose it touches. Reciprocal note on 1636: it lands first, owns the foreground
rule retirement, and must leave the B3 subsection (:199-242 / :256 "blocks on merge_request")
alone (PRD-2 territory, task 1616 DONE).

**Entry 2 (B3 blocking-premise prose) — residual sweep flagged, NOT wired.** β5 = task 1635
(unblock-low-risk submit→poll→merge_cancel-on-abort). The B3 "why background" rationale at
escalation-watcher/SKILL.md:256 is PRD-2-owned (task 1616, DONE) and is NOT swept by β6 or η;
PRD-2 tasks 1615/1616 landed without the forward-pointer entry 2 requested. This leaves a
post-P3 coherence gap (β5 inverts a premise the B3 rationale prose still asserts). An
`integration_notes` entry on 1635 flags it for surfacing; it is a follow-up prose sweep, NOT a
re-open of the DONE PRD-2 tasks. **If a single owner is wanted for that sweep, file it as a
new task depending on 1635 + 1638** — deliberately left unfiled here because β5/β6 may already
leave the rationale coherent enough; decide after β6/β5 land.

> **RESOLVED 2026-06-04 (human approved):** filed as **task 1648** (decide-and-implement-if-
> needed; deps 1635 + 1638 — 1638 transitively covers β3–β7 incl. β6). If the post-flip prose
> is already coherent, 1648 closes with a note and no edit; otherwise a minimal sweep confined
> to the B3 subsection. Acceptance: no remaining unbounded-blocking claim anywhere in
> escalation-watcher/SKILL.md; B3 rationale consistent with unblock-low-risk/SKILL.md and the
> final merge_request docstring shapes.

**Disjoint seams confirmed, no action (see no_action list):** merge_queue.py (PRD-4-exclusive,
corroborated by task 1645's reciprocal invariant); workflow.py:3815-3854 (γ3/1641) vs the
PRD-3 gates at :1186/:3093/:3442 (line-disjoint, module-locked); γ2/1640's blocking (NOT
born-at-L2) escalation shape vs escalation-repend α1 (1617, DONE); β4/1634 vs κ/1627 on
unblock/SKILL.md (disjoint regions, κ self-coordinates); β7/1637 vs θ/1625 on
escalation-watcher-auto/SKILL.md (disjoint sections); PRD-1 ε/1612 server.py startup-sweep vs
PRD-4's merge_request region (disjoint).
