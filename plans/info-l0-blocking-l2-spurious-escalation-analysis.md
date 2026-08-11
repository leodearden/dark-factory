# Analysis: the "info L0 → blocking L2" spurious-escalation pattern

**Date:** 2026-08-10 · **Author:** claude-interactive (investigation session, brief:
`~/.claude/spawn-briefs/info-l0-to-blocking-l2-investigation.md`)
**Status:** scratch analysis for operator discussion — NOT a PRD, no tasks filed,
no code changed. All file:line cites verified against the working tree this
session; queue numbers measured 2026-08-10T12:52Z from `data/escalations/`.

## TL;DR

The pattern is **partially covered**. Severity inflation (the `promote_to_l2`
default) is fully owned by **task 3976** (pending, unblocked). Moot-record
closure, co-parent orphaning, and pin safety are fully owned *in text* by
**task 3587** — but 3587 is un-dispatchable behind a 6-task state-graph chain,
and three of its adjacent gaps are owned by **no task at all**: the
merged-but-not-yet-done signal, the reaper's `cancelled`-subject blindness, and
the two chokepoint-bypass holes that let this class skip an
*already-implemented* terminal-subject auto-resolve. Current cost: **16 of 69
pending L2s (23%) are inflated from info sources**, and this class consumes
**~2 watcher//unblock sessions per day**. The severity default also has a
second-order effect no prior task names: it converts a non-pinning annotation
into a **done-flip-vetoing pin**, so the spurious L2 can prolong the very
merge→done gap that defeats the status-keyed moot sweep.

---

## 1. Root cause — the verified causal chain

Every step below was read from code this session (two independent agent passes
plus spot-checks); line numbers are from today's working tree.

1. **Filing.** `escalate_info` defaults `severity='info'`
   (`escalation/src/escalation/server.py:551`) and submits at `level=0`
   (`:609`); the submit chokepoint leaves info untouched (no downgrade at
   `:448`, no born-at-L2 stamp at `:486`). The record carries its
   self-resolution in `suggested_action` (e.g. esc-3667-1: *"No action needed
   to unblock…"*).

2. **Workflow ends; the L0 sits pending.** Nothing consumes an open info L0
   when its workflow finishes or its task merges. (This is the purpose
   mismatch: the channel exists to record something *without* demanding
   attention, but an unresolved record is indistinguishable from an unserviced
   handoff.)

3. **Orphan reaper promotes on age + liveness only.**
   `_reap_orphan_l0_escalations` (`orchestrator/src/orchestrator/harness.py:10961-11158`):
   predicate = `level == 0` (`:10985`), no in-memory workflow slot (`:10987`),
   not actively held (`:11002`), age ≥ `orphan_l0_timeout_secs` (600 s default;
   this repo runs pure defaults — no `orphan_l0_*` key in
   `dark-factory-orchestrator.yaml`). **No severity, category, subject-status,
   or git check.** It mints an L1 twin preserving severity (`severity=esc.severity`,
   `:11126`) but **overwrites** `suggested_action` with `'manual_intervention'`
   (`:11137`) — destroying the self-resolution hint rather than reading it —
   and files via **direct `queue.submit()`** (`:11142`).

4. **Bypass hole #1.** That direct submit skips the MCP chokepoint's gate 4
   (`server.py:501-538`), which **already** auto-resolves any escalation filed
   on a `done`/`cancelled` subject (`resolved_by='escalation-mcp-pre-submit-check'`,
   `resolution_class='benign'`). The terminal-subject check this incident class
   needed exists and is live — for agent filings only. Reaper filings never see it.
   (Narrow class exceptions exist: `_is_done_step_commit_orphan` `harness.py:623-641`
   and `_is_scope_divergence_orphan` `:697-724` — neither matches this class.
   The only status reader, `_is_terminal_merged` `:667-694`, requires
   `status == 'done'` *and* a merged done-provenance kind, so `cancelled`
   subjects fail it — task 3124 hazard H8, fix owned by no task.)

5. **No filter between L1 and the auto-watcher.** All three layers are
   level-only: supervisor precheck `_watcher_has_actionable_l1`
   (`harness.py:11381-11390`), the inotify wake
   (`escalation/src/escalation/watcher.py:57-67`), and the skill's drain loop
   (`skills/escalation-watcher-auto/SKILL.md:234-240`). No severity, age, or
   category filter anywhere.

6. **Promotion manufactures the severity.** `promote_to_l2` has
   `severity: str = 'blocking'` as a plain default parameter
   (`server.py:1300`) and stamps it verbatim onto the new L2 (`:1422`,
   `level=2` at `:1426`). **Member records are never loaded** — `member_ids`
   are opaque strings, deduped and stored without even an existence check
   (`:1372-1432`). The dedup/update path can't fix it either:
   `add_members_to_l2` never mutates severity (`queue.py:889-925`). The skill's
   entire severity guidance is one line — `severity="blocking",  # default; use
   "critical" for urgent` (`SKILL.md:130`) — with **no downward direction**;
   seven promote templates omit the argument, and the skill is promote-biased
   ("when unsure, PROMOTE", `SKILL.md:223`, `:517`). `severity='info'` *is* a
   legal argument (`KNOWN_SEVERITIES`, `models.py:79`) — nothing ever passes it
   by default.

7. **Bypass hole #2.** `promote_to_l2` calls `queue.submit()` directly, by
   documented design (`server.py:1318-1322`) — so gate 4's terminal-subject
   auto-resolve doesn't run at promotion time either. esc-3667-3 was filed 67
   minutes *after* its subject merged.

8. **Why it then sticks.** Three mutually reinforcing mechanisms:
   - **The inflated severity creates a pin.** `pins.py:228-230`: an info record
     is an ANNOTATION, `NON_PINNING`, at any level. `pins.py:246-247`: any
     non-info record at level ≠ 0 is a `QUEUE_HANDOFF` pin that **vetoes the
     subject's done-flip** (`harness.py:10552-10574`, consumed at `:10745`)
     regardless of liveness. So had the L2 inherited `info` it would never have
     pinned; born `blocking`, it can hold its own subject's row out of `done` —
     prolonging exactly the merge→done gap that hides the record from
     status-keyed sweeps. (A second, severity-blind veto exists at the
     ground-truth layer: row (f), `task_ground_truth.py:610-616`, LEAVEs a
     stranded in-progress task with on-main evidence whenever *any* pending
     escalation exists — `bool(get_by_task(tid, status='pending'))`, `:529`.
     Task 3541 plans to collapse this onto pin-class-aware `classify_pins`.)
   - **Only a human can close an L2.** The auto-watcher connection carries
     `X-Escalation-Levels: 0,1` (`harness.py:449`); the resolution cascade is
     strictly downward, parent→members (`queue.py:619-636`), with **no
     member→parent index anywhere** — closing members can never retire the L2.
   - **The L2 terminal-subject sweep misses this class by category.**
     `_revalidate_open_l2` (`harness.py:12229-12320`) is L2-only and gated on
     `escalation_revalidation_allowlist`, default
     `frozenset({'task_failure','stranded_blocked'})` (`config.py:3608-3610`).
     Promoted L2s default to `category='design_concern'` (`server.py:1299`) —
     esc-3667-3's root_cause was `design-concern:…` — so even after the subject
     flips `done`, the sweep never touches them. Reaper-filed **L1s** on done
     tasks are invisible to it entirely (level gate `:12279`).

**Non-determinism proof.** The queue contains three "clean pairs" — the same
info L1 fed into two L2s, one correctly info, one inflated: esc-3223-3 →
esc-3223-4 (info) *and* esc-3223-5 (blocking); esc-3508-2 → esc-3508-3/-4;
esc-3843-2 → esc-3843-3/-4. Same signal, different rotation, different
severity — exactly what an unguided default parameter produces. The watcher
*can* pass severity correctly (4 pending L2s are info); nothing makes it.

## 2. Coverage verdict, per facet

| Facet | Verdict | Owner | Notes |
|---|---|---|---|
| F1 severity inflation at promotion | **FULL** | **3976** (pending, medium, deps=[], unblocked) | Mandates deriving default severity from members ("e.g. max(member severities)"), keeps explicit upward-capable override, includes the skill file in scope (`skills/escalation-watcher-auto/SKILL.md` in `metadata.modules`). Mechanism "not pre-committed". |
| F2 reaper files L1 on a self-resolved info L0 | **NONE** | — | No task treats this as a defect — and 3976's own text rules the reaper *correct* ("The reaper behaved CORRECTLY on both counts… it PRESERVES severity"). The premise needs an operator decision, not silent task-filing (see Option C / open question 2). |
| F3 lifecycle on terminal or merged-but-not-flipped subject | **PARTIAL** | 3587, 3541 (+3550/3563), 3052, (3124 documents only) | Four named gaps below. |
| F4 moot closure below L2 + co-parent orphaning | **FULL in text, gated in practice** | **3587** (pending, high) | Precise textual match incl. the esc-3713-9/10/12 regression fixture. But un-dispatchable: 3587→3541→3540→3539→3538/3535→3537→3536; only 3535/3536 are dispatchable today. |
| F5 load-bearing-pin safety | **FULL** | 3587 (embedded constraint) | "The sweep must FLAG/annotate… rather than auto-dismiss, unless a positive safety predicate holds," citing the 3371 incident by name. |

**The F3 gaps, precisely:**

1. **Merged-but-not-yet-done signal: owned by nobody.** 3587 *argues for* it
   parenthetically ("This also argues for a merged-but-not-yet-flipped signal,
   not just a task-status one") but its WORK list does not commit to building
   it. Nothing in the reaper touches git; `_is_terminal_merged` is
   status+metadata only.
2. **`cancelled` subjects: owned by nobody.** `_is_terminal_merged` hard-requires
   `status=='done'` (`harness.py:686`), so cancelling a source task
   *manufactures* L1 queue load. Documented as hazard H8 in task 3124, whose
   deliverable is an unrelated `refile_task` tool.
3. **The two chokepoint bypasses (root-cause steps 4 and 7): owned by nobody.**
   Neither 3587 nor 3976 mentions that gate 4 already implements the
   terminal-subject auto-resolve and that both legs of this class skip it.
4. **3541's dead-L0 rule is inert without 3550 + 3563** (producer stamping and
   consumer run-id normalization), and neither is a formal dependency of 3541 —
   they can land out of order and leave the rule non-functional. Separately,
   note 3541 makes the reaper promote **more** (filing-incarnation death un-defers
   currently-immortal L0s), so landing 3541 *before* 3976 would increase the
   inflated-L2 manufacture rate.

## 3. Measured prevalence and cost (2026-08-10T12:52Z snapshot)

- **Queue:** 94 pending — L0=5, L1=20, **L2=69**. Severity: info=21,
  blocking=34, critical=39.
- **Headline: 16 of 69 pending L2s (23%) are inflated from info-severity
  sources** (blocking L2 with ≥1 info member): esc-3105-3, 3143-7, 3223-5,
  3236-5, 3508-4, 3561-6, 3561-7, 3576-6, 3576-7, 3580-3, 3641-4, 3641-5,
  3843-4, 3875-4, 3891-3, markup-tripwire-3. Ages 0.1–10.2 days. Four more L2s
  correctly carry info (the clean-pair halves).
- **Terminal-subject:** 15/94 pending sit on done/cancelled tasks (8 at L2) —
  statistically unchanged from the 08-10 baseline (16/100, 9 at L2). The class
  is in steady-state production, not historical debris: esc-3143-7 and
  esc-3236-5 were minted *during* the measurement session, going
  terminal-subject within minutes.
- **Merge→done gap: 0 live instances right now.** All 80 non-terminal-subject
  pending escalations were checked against main's merge markers — none has a
  landed branch. The gap is real but transient (3713's verified 2-day instance
  drained on 08-10); a filing-time check would still race it.
- **Cost:** the archive (3049 resolved records; resolved-timestamps span ~30
  days) contains **59 L2-level closures** in the moot/cascade class
  (`resolution` containing "moot" or `resolved_by` = `l2-cascade:*`) — **≈2
  consumed watcher//unblock sessions per day**, median survival 2.0 h, tail to
  87–198 h. 19 records carry the exact
  `resolution_class='moot-terminal-subject'` (incl. esc-3667-2/-3 and
  esc-3713-12). 525 total records across all levels are in the moot∪cascade
  class.
- **Fixture check:** esc-3667-1 (L0 info, dismissed benign) → esc-3667-2 (L1
  info, closed by `l2-cascade:esc-3667-3`, moot-terminal-subject) → esc-3667-3
  (L2 **blocking**, members=['esc-3667-2'], moot-terminal-subject) — both
  defects in one lineage, as the brief stated.
- **Side finding (data integrity):** `esc-508-96` sits at the **archive root**
  (not a dated subdir) with `status: "pending"`, 125 days old, missing modern
  schema fields — a pending record invisible to both the live queue and proper
  archive tooling. One-off cleanup candidate; excluded from counts above.

## 4. Candidate fixes and trade-offs

### Option A — inherit severity at the promotion seam (task 3976 as filed, sharpened)

Default the promoted L2's severity to `max(member severities)` by loading the
member records (which also buys member-existence validation — today ids are
never checked); keep the explicit parameter as an upward-capable override; fix
the skill line and templates (already in 3976's scope). Implementation note:
`_SEVERITY_RANK` (`queue.py:78`) lacks `critical`/`urgent` — any max-of-members
must extend it or reuse a complete ranking, or the same-level dedup fold's
latent gap gets replicated.

- **Fixes:** the 23% inflated class carries truthful severity; info L2s stop
  pinning (pins.py link 1) so the done-flip-veto feedback loop dies; ntfy stops
  paging `urgent` for FYIs; the clean-pair non-determinism disappears.
- **Misses:** the records are still *manufactured* — the reaper still files the
  L1, the watcher still burns a rotation, an info L2 still lands in the human
  queue (at correct priority). Terminal-subject staleness untouched.
- **Risks:** low and bounded. A genuine blocker mis-filed as info by its L0
  author would stay info — but that mis-filing is a filer error today too, and
  the watcher retains the explicit override. Cheapest option; unblocked now.

### Option B — stop filing on dead subjects (close the two bypass holes)

(i) Before promoting an L0, the reaper consults subject state: if the subject
is `done`/`cancelled` **or** its branch has a merge marker on main
(merged-but-not-flipped), dismiss rather than promote — **scoped to
`severity=='info'` records only**. (ii) `promote_to_l2` runs the gate-4
terminal-subject check (or stops bypassing the chokepoint for it).

- **Fixes:** stops *manufacturing* new records on dead subjects — the exact
  esc-3667 shape (L2 filed 67 min post-merge) becomes impossible; covers
  `cancelled`; covers the merge→done window via the git marker rather than
  status.
- **Misses:** the already-pending stock (needs 3587's sweep); live-subject
  info churn (needs A or C).
- **Risks:** the pin hazard is the sharp edge — the 3371 pin *was* an
  orphan-reaper record, and the reaper re-filing is what keeps a pin alive
  across workflow death. The info-only scoping is the positive safety
  predicate 3587 demands: **by the pin classifier's own spec, an info record
  never pins** (`pins.py:228-230`), so dismissing info-only records cannot
  destroy a pin. Non-info records on terminal subjects should be flagged into
  3587's triage bucket, never auto-dismissed. Second risk: suppressing a
  genuine "merged but broken" report — mitigated because that class should be
  (and is) filed at non-info severity, which this option never suppresses; gate
  4 already applies a *stronger* version of this policy to all agent filings.

### Option C — give `escalate_info` a real lifecycle (non-promotable observation class)

The principled fix for the purpose mismatch. Variants, combinable:

- **C1 — consume-on-terminal:** open info L0s auto-resolve
  (`observation-consumed`, archived and searchable — not deleted) when their
  subject task merges or goes terminal. The note's stated purpose ("legible at
  review time") has been served by then.
- **C2 — reaper disposition:** the reaper's action for an aged orphan info L0
  becomes resolve-as-`observation-expired` instead of promote-to-L1.
- **C3 — FYI rung in the watcher skill:** an all-info member set doesn't
  promote; the watcher (which holds authority at levels 0–1) closes or
  triage-stamps at L1 with a digest note.

- **Fixes:** the churn itself — no L1, no watcher rotation, no L2, no human
  session. Under C, F2's premise question dissolves: the reaper stays "correct,"
  info records simply never reach it.
- **Misses:** non-info moot records and co-parent orphaning (3587 still
  needed); severity for *mixed* clusters (3976 still needed — an info member
  folded into a blocking cluster must not drag it down, which max-of-members
  handles).
- **Risks:** an observation a human would have wanted escalated gets quietly
  archived. 3976's constraint must be preserved: a *cluster* of individually
  informational findings CAN be collectively blocking — so C must remove only
  *default/unattended* promotion, never the watcher's explicit judgment call.
  This is a contract change to the escalation ladder (info records' consumer
  becomes the review/audit trail, not the ladder) and deserves explicit
  operator sign-off.

### Option D — land the moot-closure sweep, and amend its gaps (task 3587)

Amend 3587 **in place** (never refile) to commit what it currently only argues
for: the merged-marker signal, `cancelled` subjects, and a disposition for the
`design_concern` category (the entire inflated class) — the flag-into-triage-
bucket design covers it without widening the auto-close allowlist. Separately,
decide whether the sweep half of 3587 genuinely needs the 3541→3540→…
classify_pins chain, or can be hoisted — at ~2 burned sessions/day, the queue
pays daily while six upstream tasks land.

- **Fixes:** the standing stock, and the co-parent orphaning that strands L2s
  after their members close.
- **Risks:** dependency reshuffling second-guesses the state-graph PRD's
  sequencing; the pin-safety constraint is already correctly embedded in
  3587's text and must survive any amendment.

## 5. Recommendation

Layered — these are complements, not alternatives; each targets a different
stage of the pipeline (severity → manufacture → stock → contract):

1. **Land 3976 first (Option A).** Unblocked, small, kills the worst
   consequence (blocking-pin manufacture + red pages) and the 23% inflation.
   Consider bumping it from medium: it should land **before 3541**, which will
   otherwise increase the manufacture rate. Fold in the `_SEVERITY_RANK`
   completion.
2. **Add the info-scoped filing-time gate (Option B) — ideally as an amendment
   to 3976's scope or 3587's**, since it shares the seam files with both. The
   info-only predicate is the pin-safe core; the merged-marker check is what
   actually closes the 3667-shaped race that status checks lose.
3. **Amend 3587 in place** per Option D and make the operator call on hoisting
   vs. waiting out the chain.
4. **Discuss Option C (C1 specifically) as the contract-level fix** — it is the
   only option under which `escalate_info` stops being a time bomb by
   construction rather than by compensating sweeps.

**Open questions for the operator:**

1. Should an **all-info member set promote to L2 at all** (3976 leaves this
   explicitly open)? If not: watcher close-at-L1 with triage stamp, or a
   digest artifact?
2. **Is F2 a defect?** 3976 says the reaper behaved correctly; the alternative
   view is that *any* unconsumed info record reaching the ladder is the defect
   (Option C). This determines whether the fix lives in the reaper or in the
   info lifecycle.
3. **Hoist 3587's sweep half** out of the state-graph chain, or accept ~2
   sessions/day until 3536→…→3541 land?
4. Should `promote_to_l2` make severity **mandatory** (no default) instead of
   inherited — trading a hard API break (and watcher-prompt reliance) for
   explicitness?
5. `esc-508-96` archive-root anomaly and the missing 3541→{3550,3563} formal
   dependencies: fold into existing tasks when filing is next appropriate?
   (Nothing filed this session, per the brief.)

## Appendix: data provenance

Four parallel agent passes (task-coverage sweep over 612 non-terminal tasks;
promotion-path code trace; reaper-path code trace; queue measurement over 94
live + 3049 archived records), synthesized with independent spot-checks of
`server.py` gate 4, `pins.py` links 1–3, `task_ground_truth.py` row (f), and
full reads of tasks 3587/3976/3541. Raw measurement intermediates (full
escalation index, per-task dumps, merge-id lists) were preserved in the
session scratchpad (`full_index.json`, `task2_strict_results.json`,
`task3_results.json`, `task4_results.json`, `task5_results.json`); the queue
mutated twice during measurement (esc-3624-6/-7 resolved, esc-3543-5 filed) —
all numbers are from the final consistent snapshot.

---

# Addendum (2026-08-11): Option C's premise re-examined, and a larger defect found

Operator discussion round 1 rejected Option C as written ("consume-on-terminal"
would archive info L0s silently — "not what I want") and posed two empirical
questions: (1) aren't late-lifecycle info L0s almost always non-blocking review
comments (test completeness / code quality) that should be *actioned* as
follow-up tasks, not archived? (2) doesn't deterministic conversion code for
that already exist? Two agent studies answered both; every load-bearing claim
below was spot-checked in code or against the live server.

## A1. What info L0s actually are (corpus study, N=1,501, sample=300)

Population: 1,307 info L0s + 194 reaper-minted info L1s, 2026-07-12→08-11.

| Class | Share | Disposition it deserves |
|---|---|---|
| STATUS_INFO (mechanical notices; ~⅔ of the orchestrator-filed ones are the one "done step commit orphaned" tripwire) | **61%** | class-specific auto-handling; never promote, never convert |
| FOLLOWUP_WORK (real adjacent bugs/gaps, out of scope) | **17%** | **convert to task** (the operator's ratification pattern) |
| DEVIATION_NOTE (plan-vs-reality audit notes, "no action needed") | **16%** | consume-on-terminal (original C1, correctly scoped to THIS class only) |
| REVIEW_TEST + REVIEW_CODE (the hypothesized class) | **5.3%** | convert to task |

So the hypothesis is **refuted on prevalence** (5%, not "almost always") but
**confirmed on tone and timing**: 86.6% are filed within 24h of the task's
merge (median 8.4h before). The quality leak is real but lives mostly in
FOLLOWUP_WORK: in the sample crosstab, 32/50 genuine follow-up-work notes died
DISMISSED_BENIGN vs 12 converted. Full-population conversion-to-task: 5.4%
(L0s) / 25.3% (reaper L1s — inflated by cascade credit); 79% dismissed benign;
the mechanical reaper touches 44.9% of all info-L0 dispositions, and 52.7% of
reaper-minted L1s are cascade-resolved without ever being independently read.
Of the 9 info records pending today: 4 FOLLOWUP_WORK worth keeping, 3
DEVIATION_NOTE, 2 STATUS_INFO, 0 review comments.

## A2. The conversion machinery exists — its deterministic core is a silent no-op

Task 2640 (done, `1ecbb4f430`, 2026-07-21) codified the owner decision (Leo,
2026-07-15): *work item → submit_task; information → escalate_info*. Inventory
of all 18 discovered paths from "follow-up content" to "task": the only paths
carrying real volume are 2640's **prompt** instruction to architect/implementer
(387 tasks since landing — LLM discretion, no detector for missed filings) and
the steward sweep-up (**1% coverage** — scoped to "YOUR OWN still-open
info-notes", but 99% of info L0s are filed by implementer/architect/debugger,
and it never runs when a workflow dies — both fixture incidents were exactly
that crash path).

**The genuinely deterministic paths are all dead behind one transport bug.**
`workflow.py`'s four raw MCP POSTs (`:15149, :15190, :15222, :15260` —
`_post_submit_tasks` and the completion/decisions/suggestions memory writes)
plus `merge_queue.py:15122` POST to `{mcp.url}/mcp/` (trailing slash); the
server answers **HTTP 307 → /mcp**; bare `httpx.AsyncClient()` defaults
`follow_redirects=False`; the response is never inspected; success is logged
regardless. Verified live this session (curl → `307 http://localhost:8002/mcp`).
Corroboration: **0 tickets ever** with the review-suggestion spawn_contexts in
6,087 all-time tickets; `steward-triage` ticket volume collapsed 1,668→96/month
when task 1367 (2026-05-15) cut the DONE path over to this transport.
Consequence: **every task's end-of-review non-blocking suggestions have been
silently discarded since ~2026-05-15** — they never become an escalation, a
ticket, or a memory; not even a human sees them. This is the actual
code-quality/test-completeness leak, and it sits *upstream* of the escalation
ladder entirely. Precedent: task 29 diagnosed this exact httpx/307 bug and
prescribed the fix ("follow_redirects=True … and correct the URL") but only
`dashboard/` was fixed; task 3644 fixed a third instance in a script. The
orchestrator's five sites were never swept; `test_suggestion_triage.py:260-266`
mocks the POST so the suite can't notice.

Other structural facts: no code anywhere reads an escalation and files a task
(`RESOLVE_ACTIONS` has no filing action; `escalation/src/` has zero references
to any task store). The L1 auto-watcher holds **no** filing tool
(`_WATCHER_ALLOWED_TOOLS`), yet its skill resolves `cleanup_needed` with
*"Cleanup queued… tracked in digest"* — nothing is queued and the digest dies
with the rotation, while the *same record* at L2 would get a real
`submit_task` — L1 resolves it first, so it never reaches L2. Ticket-pipeline
losses: 272 `failed` all-time (117 `tm_add_task_returned_no_id`, 49
`server_restart` — a fused-memory restart destroys in-flight candidates);
`refused` is a latent trap (0 occurrences, excluded from janitor sweep and
agent-facing status list); the completion-claim gate verifies "filed as tkt_X"
on row *existence*, so a failed ticket passes it.

## A3. Revised proposal: a disposition router, not one policy

Replacing Option C as written. "Never inflate to blocking" (3976) stays first
and orthogonal. For open info L0s at end of lifecycle:

1. **Work-shaped (FOLLOWUP_WORK + REVIEW, ~22%): convert-and-close.** File a
   candidate via the curator admission gate (which is what makes over-filing
   safe — it fails open to `create`), resolve the note citing the ticket/task
   id. Candidate mechanisms, cheapest first: (a) **fix the 307 transport** (5
   sites, one keyword + `raise_for_status()` — prerequisite for everything
   else and independently urgent); (b) give the L1 auto-watcher a filing tool
   + a convert-and-close rung (replaces the fictional digest); (c) a
   crash-path backstop so the steward sweep-up's conversion also happens when
   the workflow dies (the reaper — or the terminal-subject sweep — routing
   work-shaped info L0s to the curator instead of promoting them).
2. **DEVIATION_NOTE (~16%): consume-on-terminal** (original C1, now correctly
   scoped) — resolved as observation-consumed, archived, searchable.
3. **STATUS_INFO (~61%): class-specific handling** — the dominant tripwire
   class already has discriminators; extend the existing benign-class
   dismissal rather than inventing a new channel.
4. The work/information discriminator is a judgment call → it belongs with an
   LLM consumer (the L1 watcher) or the filing author, not a regex in the
   reaper; the reaper stays a lifecycle mechanism.

## A4. Updated open questions

1. The 307 transport fix is outside this investigation's no-implement rule but
   is an active fleet-wide loss (review suggestions + three memory-write
   classes + main-health auto-heal filing). Fix now as its own task?
2. Ratify the disposition router? Which mechanism for work-shaped conversion —
   L1 watcher rung (LLM judgment, needs tool grant), reaper→curator routing
   (deterministic reach, needs a classifier), or steward crash-backstop
   (narrowest)?
3. Is DISMISSED_BENIGN-by-default acceptable for STATUS_INFO (61%), given the
   dominant tripwire is itself an automation artifact worth reducing at source?
4. The L1/L2 skill asymmetry (`cleanup_needed` "digest" fiction; L2's "local
   todo" for design_concern/risk_identified never reaching the task tree) —
   fix the skills regardless of the router decision?
5. Ticket-pipeline hardening (server_restart flush, refused trap, claim gate
   row-existence check) — in scope for this remediation or separate?

---

# Addendum B (2026-08-11, round 3): rulings executed, L1 history, fictions verified, recovery map

Operator rulings: 307 fix filed as **task 4023** (priority high + scheduler
boost_tier=critical); disposition router **RATIFIED** (reaper→curator primary
+ steward crash-backstop); skill-fiction fixes verify-then-discuss; ticket
hardening in scope.

## B1. L1 filing-authority history (operator recollection: REFUTED with a kernel)

The auto-watcher **never** held a filing tool; `_WATCHER_ALLOWED_TOOLS` has
four all-additive commits. No restriction-for-cause exists anywhere. The
kernel: the **L1 tier** lost filing on 2026-05-27 when task 1505 moved the
human watcher (which could file) to L2 and handed L1 to the auto-watcher,
built 12 days earlier with a deliberately narrow AFK toolset — a loss by tier
reassignment, unnoticed until **task 3726 (pending, filed by Leo 2026-08-05:
grant submit_task/resolve_ticket to the auto-watcher)** — the grant is already
decided, just unimplemented (one dispatch bounced on missing plan-tools MCP).
Composition with the ratified router: reaper→curator = **coverage**
(deterministic; runs with no rotation up; but no content judgment, and only
fires on *orphaned L0s* — 12 of 20 pending L1s today were born at L1);
L1 filing = **quality + closure** (reads/RCAs the record, one task per root
cause across a cluster, file-and-close in one rotation). Cautions: the curator
no longer fails open to create on LLM failure (files a curator_failure L1 —
feedback edge into the watcher's own queue); a per-rotation cap would be
prompt-only (allowlist is advisory under --permission-bypass); 3726 collides
with pending 3465 (both edit the auto-watcher skill).

## B2. Fictions verification (ruling 3: discuss before actioning — NOT yet edited)

| Item | Verdict | Evidence core |
|---|---|---|
| auto `cleanup_needed` "tracked in digest" (SKILL.md:390-404) | **FICTION** | digest = the rotation's final chat message; supervisor reads only success/timed_out; allowlist has no write tool; `resume` archives the record. esc-3568-2 promised replay+memory correction — cited ids exist nowhere; esc-2868-2/esc-3088-2 self-document the gap. 12 archived specimens checked. |
| L2 `design_concern` "create a local task/todo" (:806-815) | **PARTIAL** | Ambiguous TodoWrite-vs-task; omits the write-decision line siblings carry (vs C8 policy). Mitigations: "leave pending" is real tracking; 18 sampled human-present rulings almost always filed/amended real tasks. Hazard = unattended AFK. |
| L2 `risk_identified` "track as todo" (:817-822) | **PARTIAL** | Same as above; thinner section. |
| Everything else (full sweep of both skills) | **REAL** | afk-digest.md is a real file; write-decision implemented; amend-fold close conditions checkable; L2 cleanup_needed two-phase pattern correct incl. refused/failed handling. |

No code parses any of these phrases (repo-wide grep) — rewording is safe.
Proposed changes (pending sign-off): #1 either grant filing (= task 3726) and
make the handler file-and-close via curator, or reword to state truthfully
that no durable follow-up exists; #2/#3 replace "local task/todo" with
explicit submit_task-for-work-items + restate write-decision, TodoWrite as
in-session convenience only.

## B3. Recovery map

**Reviewer suggestions (lost to the 307 since 05-15):** recoverable from
`.worktrees/.task-meta/<task_id>/reviews/*.json` (meta-root migration
`5067b2d230`, 2026-07-06; never reaped — survives worktree deletion).
Post-07-06 era: **88.5% of 745 done tasks recoverable; 477 tasks hold 1,329
verbatim suggestion objects, 0 parse errors**. Pre-07-06 era (706 tasks):
near-total loss (3 recoverable) — reviews lived inside since-deleted
worktrees. runs.db review events carry `data='{}'`; journal retention ~3d
(counts only). Server access log confirms the 307 **still firing today**.

**L1-died follow-ups:** escalation archive retention is **30 days**
(`DEFAULT_RETENTION_DAYS=30`) — everything before ~07-12 permanently pruned.
In-window: 43 L1-auto-resolved records in the 4 work categories; 36 legitimate
closes; 2 covered elsewhere; **5 genuine re-file candidates**: esc-2763-2
(dead legacy plan.lock fallback in task_ground_truth.py — verified still in
code), esc-2868-2 (one-off reify metadata.files reconcile, 6 tasks,
"a human should file this"), esc-2834-2 (reify sidecar note, only durable
copy, relevant later), esc-2608-2 (2 optional git_ops cancellation-reap
tests), esc-2243-2 (moot). The 12% miss-rate is a *rate*: pre-07-12 losses of
the same shape are unrecoverable.

**Pending decisions:** (1) backfill the 1,329 recovered suggestions through
the curator after 4023 lands (spawn_context tag, curator dedup as safety)?
(2) file which of the 5 re-file candidates? (3) 3726: proceed as the L1 half
of the ratified combination (and resolve the 3465 collision)? (4) skill edits
per B2 once signed off.
