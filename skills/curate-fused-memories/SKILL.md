---
name: curate-fused-memories
description: "Run the memory-consolidation curation sitting: batch-rule the backlog of 'Human gate: consolidate duplicate cluster' tasks and execute the resulting consolidations against the fused-memory Mem0 corpus under the gate-3200 retain-and-tag shape. Use when the user says 'run the curation sitting', 'consolidate duplicate memories', 'batch-rule the consolidation gates', 'clean up the Mem0 corpus', 'curate memories', 'adjudicate the duplicate-cluster gates', or wants the memory-consolidation backlog (task 3524 / esc-3524-1) worked. This is NOT for: ordinary memory writes (use /memory), the reconciliation escalation queues (recon-escalation-watcher), unblocking stuck tasks (/unblock), or a single ad-hoc near-exact dedup that recon Stage 1/2 can absorb inline — per the 2026-08-02 policy, new clusters route through inline Stage 1/2 consolidation, not new gate tasks."
---

# Memory-Consolidation Curation Sitting

You are running the **batch curation sitting**: adjudicating the backlog of
"Human gate: consolidate duplicate cluster" tasks in one pass and executing the
resulting consolidations against the fused-memory Mem0 corpus. This is the
sitting the 2026-08-02 policy decision called for (batch-ruling, explicitly
rejecting one-gate-at-a-time adjudication).

The moving parts, and where the truth lives:

- **Tracking vehicle**: task **3524** and its escalation **esc-3524-1** (born-at-L2,
  `milestone_gate`, **pending BY DESIGN** — it holds 3524 at `blocked` and closes
  only when the sitting is actually done).
- **Readiness gate**: task **3625** — a pure human gate whose dispatch means every
  precondition has landed. Its description is the authoritative statement of the
  sequencing trap and the citation-exposure measurement; read it in full.
- **The primitive**: task **3133** — the transactional `consolidate_memories` MCP op
  (validate → write short canonical → retain/delete → re-query closure → return
  survivors). As of 2026-08-11 it had **not yet landed** (3133 and 3523 still
  pending, so 3625 had not fired) — Phase 0 re-checks this; do not assume either way.
- **The ratified shape**: gate **3200** (resolved 2026-08-11) — see "The ratified
  corpus shape" below.

Everything dated in this skill is a **snapshot**. Where a number or list is
stamped with a date, re-derive it from the named source before acting on it.

---

## When NOT to use this skill

- **Ordinary memory writes** — `/memory` covers routine `add_memory`/`search` use.
- **The reconciliation escalation queues** — that is `recon-escalation-watcher`
  (port 8103), a different consumer with different semantics.
- **Unblocking tasks** — `/unblock`. The gate tasks this sitting closes are not
  "stuck"; they are deliberate recorded debt.
- **A single ad-hoc dedup** — a genuinely safe near-exact duplicate found in
  passing goes through inline Stage 1/2 consolidation, not this skill and not a
  new gate task. The 2026-08-02 policy stopped filing new consolidation gates.
- **Anything that merely wants `update_memory` authority** — see the next section.
  This skill is not a key you borrow.

---

## Authority: `curator-` is a declared role, not a credential

`update_memory` (the in-place Mem0 amend/patch tool) has **two independently
gated arms**:

- `content_amend` — rewrite a record's text in place (re-embeds; the point id survives)
- `metadata_patch` — patch/delete metadata keys (no re-embed; the point id survives)

Each arm sits behind an **agent_id-prefix allowlist** read **live off config on
every call**, plus a kill switch that outranks both. Read these two sources —
they are short and they are the law:

- `fused-memory/src/fused_memory/config/schema.py`, `Mem0UpdateConfig` — since
  2026-08-12 the `curator-` grant is the **schema default**: both arms'
  `default_factory` lists are `['recon-stage-', 'curator-']`, and
  `config.yaml`'s `mem0_update:` block ships fully commented out (an active
  block there is an operator override that trips a deliberate tripwire test).
  The field descriptions record Leo's ruling (b) on esc-3524-1 and why **both**
  arms must carry `curator-`: retain-and-tag needs the preserving half, not
  just the destructive half.
- `fused-memory/src/fused_memory/server/mem0_update_authz.py` — the resolver.
  Note `resolve_mem0_update_enabled` (the `mem0_update.enabled` kill switch,
  evaluated first, denies everyone when off) and that every fallback is
  fail-closed.

A session executing this skill adopts `agent_id='curator-<something-descriptive>'`
(e.g. `curator-sitting-2026-08-11`) on its memory write calls, and thereby holds
both arms.

**Say it plainly: `agent_id` is SELF-REPORTED.** The resolver's own docstring
says so — this gate "is a misuse deterrent for cooperating callers ... not a
security boundary." The `curator-` prefix is a **declaration of sanctioned
role**, not a password. That cuts both ways, and both directions are binding:

- **It is legitimate** to adopt `curator-` while actually executing this skill
  as a sanctioned sitting — that is precisely what the prefix was minted for
  (esc-3524-1 triage addendum, 2026-08-11).
- **It is NOT legitimate** to adopt it because you hit `Mem0UpdateNotAuthorized`
  in the middle of unrelated work and want past it. Renaming yourself to clear
  the gate is self-authorization for a silent-rewrite primitive. Stop, leave the
  record alone, and escalate to an operator. (The precedent to imitate: the
  2026-08-03 session that was refused as `claude-interactive` and correctly fell
  back rather than spoofing a prefix — recorded in esc-3524-1's triage note.)

Also for orientation: `recon-stage-` holds the same grant by a completely
different route (reconciliation Stage 1/2 runs automatically under it). It has
nothing to do with this skill; do not borrow it either.

---

## The ratified corpus shape (gate 3200, resolved 2026-08-11)

Read task **3200's description AND its `details` field** (the details carry a
post-resolution correction), and escalation **esc-3200-3's resolution**. The
ruling was a **split ratification**:

- **RATIFIED — Option C's WRITE shape**: short retained single-claim peers
  sharing `metadata.topic`, **exactly one `canonical: true` per (project, topic)**,
  **RETAIN-not-delete**.
- **NOT RATIFIED — any read transform.** Deliberately. The read-side choice was
  delegated to task 4004; nothing in this sitting should assume a grouped or
  anchored read exists.

What "consolidate" therefore means in this sitting:

1. Write a **SHORT canonical** — an index/summary claim, **not a concatenation**.
   Long concatenated canonicals are the founding pathology: task 3133's
   `x_deferred_reason` records a ~9k-char canonical (`bbc063a7`) absent from a
   limit-10 same-topic search while ten short siblings ranked 0.66–0.76 (the
   measurement is task 3111's).
2. **RETAIN the folded entries** as topic-tagged peers via a **metadata-only
   patch** (`update_memory` `metadata_patch`), preserving their Qdrant point ids —
   this is what keeps every live task citation valid.
3. Put **only genuinely absorbed (deleted) entries** in `supersedes`.
4. A **delete arm remains available** for true near-verbatim redundancy — but
   deletes are exactly where the sequencing trap below lives.

If anything you read at sitting time contradicts this (e.g. a later gate
re-opened the shape question), stop and re-derive from 3200/esc-3200-3 before
executing anything.

---

## THE SEQUENCING TRAP: close each gate task BEFORE running its own deletes

This is the single most important operational item in the sitting. The
authoritative statement is task **3625's description** ("SEQUENCING TRAP — READ
BEFORE THE FIRST DELETE"); re-read it at sitting time. In short:

A gate task that **enumerates** a cluster is itself a **live citer** of every
member UUID it names. Measured 2026-08-04 (snapshot — re-measure): **103 of the
201 cluster UUIDs (88 DF + 15 reify) have live citations ONLY from their own
gate task.** Task 3624's broadened `_citation_repoint_gate`
(`fused-memory/src/fused_memory/server/tools.py:1802`; live-citer filter at
`:1949`) now applies to non-recon callers too, so each such delete refuses with
`error_type=CitationRepointRequired` — **the sitting blocked by its own
bookkeeping**.

**The fix inverts the normal order: CLOSE each gate task BEFORE running its own
deletes.** A terminal task drops out of `live_citers`, the self-citations
evaporate, and the guard then fires on exactly the UUIDs with **genuine
third-party exposure** — turning it from an obstacle into a precise detector
that names the victim tasks for you in its refusal payload.

**Prohibition:** do **NOT** reach for `metadata={'allow_dangling_citations': True}`
(task 3624's deliberate escape hatch on `delete_memory`) as routine batch
posture. Applied reflexively across ~88 consecutive refusals, it is exactly how
the real victims — the tasks with genuine third-party citations — get destroyed
silently. The escape is for a single, individually-reasoned case where you have
decided dangling is acceptable, never a loop default.

---

## Expect the storm alarm — it is a feature firing, not a fault

Config `mem0_update.storm_threshold` (20) over `storm_window_seconds` (3600)
(snapshot — read the live values off the `mem0_update:` block; both are
green-tier hot-reloadable). A sitting of ~30 gates **WILL cross 20 content
amends in an hour**. By design the alarm **ESCALATES, NEVER BLOCKS** — the
config comment is explicit that a hard block would risk a legitimate large
consolidation failing mid-run on its own success count. Metadata-only calls
(the retain-and-tag stamps) do not count toward it.

So: when a `content_amend_storm` escalation appears mid-sitting
(`fused-memory/src/fused_memory/middleware/mem0_update_storm_escalator.py`),
acknowledge it in your closing report and resolve it as expected-by-design —
do not halt the sitting, and do not misread it as evidence of a runaway.

---

## Recurrence is real, and consolidation does not end it

Gate 3200's third decision recorded recurrence as **"REDUCED, STILL REAL"** on
measured evidence (esc-3200-3, decision 3): a reify cluster consolidated
11-into-1 (2026-07-05) regrew to 6 by 07-26, was re-consolidated 07-27 into a
canonical (`bf91bc5c`) carrying an explicit **in-content accretion warning** —
and regrew AGAIN within three days.

The load-bearing detail: the re-emissions arrived **UNSTAMPED**
(`count_memories_by_metadata` on the topic returned 1 — only the canonical). So
"N peers under a live canonical is the target state" holds only for **stamped**
peers; organic re-emission lands as an untracked near-duplicate under every
corpus shape. Task **3136** owns converting this into a measured per-topic rate.

Practical consequences for the sitting:

- Consider ending each canonical with a short accretion warning ("append to this
  entry / write a stamped peer under topic `<slug>`, do not write a new
  unstamped entry") — cheap, though measured to be insufficient on its own.
- **Never record a topic as "closed"** in a disposition. Record what was
  consolidated and when; recurrence measurement is 3136's job, not a promise
  this sitting can make.

---

## The procedure

### Phase 0 — Pre-flight: readiness, then authority

Run all of this BEFORE forming any per-gate plan. Discovering a missing grant
mid-sitting, after gates have been closed, is the failure mode this phase exists
to prevent.

**0a. Readiness.** The sitting is premature until the readiness gate has fired:

```
get_statuses(project_root="/home/leo/src/dark-factory",
             ids=["3133", "3523", "3623", "3624", "3625", "3524"])
get_escalation(escalation_id="esc-3524-1")     # expect: still pending
```

- 3133, 3523, 3623, 3624 all `done` (equivalently: 3625 dispatched and its
  born-at-L2 escalation exists) → proceed. Otherwise **stop**: 3625 exists
  precisely to convert "wait for the right moment" into an edge that fires once;
  do not hand-run the sitting ahead of it. (As of 2026-08-11: 3623/3624 done,
  3133/3523 pending — re-check, don't assume.)
- esc-3524-1 must still be `pending`. If someone resolved it without a sitting,
  investigate before proceeding — its pendency is what holds 3524 `blocked`.

**0b. Confirm the primitive.** Check whether `consolidate_memories` is present
in this session's fused-memory MCP tool list and read its landed signature
(3133's suggested shape: `consolidate_memories(canonical_content, supersedes=[UUIDs
to delete], retain=[UUIDs to topic-tag], topic, project_id, ...)` — verify
against the real tool, not this note). If 3133 landed, prefer it: it does the
validate → write → retain-tag → delete → **re-query closure** sequence
transactionally and returns `survivors` so you never claim an unverified
closure. Without it, you are hand-sequencing `add_memory` + `update_memory` +
`delete_memory` and must run the closure re-query yourself (Phase 4).

**0c. Read the live authority config.**

```bash
grep -n "allowed_agent_prefixes" \
  fused-memory/src/fused_memory/config/schema.py      # the shipped defaults
grep -n "^mem0_update:" fused-memory/config/config.yaml || echo "no override"
```

Expect `curator-` in BOTH arms' `default_factory` lists and NO active
`mem0_update:` override in config.yaml (snapshot 2026-08-12 — the grant is the
schema default; an uncommented YAML block means an operator has overridden it,
and what the running server holds is whatever config it booted/reloaded with,
which is exactly why the probe in 0d, not this read, is the ground truth).

**0d. The cheap authority probe.** Authorization is checked **before** any
record lookup or mutation (`server/tools.py` `update_memory` step (2), ~:4287 —
authz runs ahead of project canonicalization and store validation), so a probe
against a nonexistent id is a zero-mutation discriminator. Request **both arms
in one call**, under the **same** `agent_id` the sitting will use:

```
update_memory(
  memory_id="00000000-0000-0000-0000-000000000000",   # syntactically valid, guaranteed absent
  store="mem0",
  project_id="dark_factory",
  content="preflight authz probe — must never land",
  reason="curate-fused-memories Phase 0 authority probe",
  metadata_patch={"x_preflight_probe": True},
  agent_id="curator-<your-sitting-slug>",
)
```

Read the result by `error_type`:

- `Mem0UpdateNotAuthorized` → the grant is missing (config regressed or was
  never reloaded). **Stop.** Surface to the operator — the remedy is a config
  edit plus `reload_config` (read the returned `applied` dispositions, not just
  `reloaded`), and that is an operator decision, not something to self-serve.
- `Mem0UpdateToolDisabled` → the kill switch is on, which means an operator
  turned it off, possibly mid-incident. **Stop and ask.**
- Any other error (a not-found on the nonexistent id) → **both arms held**,
  nothing mutated. Proceed.

### Phase 1 — Derive the scope (read the field, never a number in prose)

- **DF scope count**: task **3524's `metadata.gate_task_count`** is the SINGLE
  SOURCE OF TRUTH (30 as of 2026-08-10 — snapshot). Task 3625's description
  carries a long history of superseded counts (14, 18, 22, 23, 25, 29, 30) and
  its own final addendum says they are historical snapshots that must NOT be
  treated as current. Never copy a count out of prose — including this skill's.
- **DF membership**: the gate tasks are wired `depends_on=3524`. Enumerate them
  by reading the dependency graph (e.g. `get_tasks` and filter for tasks whose
  `dependencies` include 3524), and cross-check the member count against
  `gate_task_count`. A mismatch means the scope drifted — reconcile (re-read
  3524/3625's latest addenda) before ruling on anything.
- **Reify side**: ~7 curator gates (5773, 5821, 5824, 5901, 5914, 5943, 5965 —
  2026-08-05 snapshot, explicitly pending re-verification). Re-enumerate against
  the reify checkout's own `project_root`. Note 3625's warning: these are mostly
  **stale-claim corrections, not dedups** — 5773's own text is itself stale (it
  predates the `update_memory` tool). Re-read each before ruling.

### Phase 2 — Adjudicate each gate

For each gate task, in whatever order you like (adjudication is read-only):

1. Read the gate task's description — the cluster UUIDs, the claimed
   duplication/staleness, any proposed disposition.
2. Fetch the members (`get_memory_by_id`) and read them. Do not rule on the
   gate's summary alone: clusters drift, members get tombstoned, claims go
   stale in both directions.
3. Classify into a disposition:
   - **CONSOLIDATE** (retain-and-tag): the C shape — short canonical, stamp the
     retained peers with `metadata.topic`, `supersedes` only what is genuinely
     absorbed.
   - **CORRECT**: a stale or wrong claim — `content_amend` with a `reason`, or a
     metadata repoint (`corrects`/`supplements`), usually the cheap
     `metadata_patch` arm. Most reify gates land here.
   - **DELETE**: true near-verbatim redundancy — the delete arm, citation gate
     permitting.
   - **NO-OP / OVERTAKEN**: already consolidated, members gone, or the ruling on
     3200's third decision makes the gate's premise moot. Record why.
4. Draft the disposition text now, per gate — Phase 3 writes it.

### Phase 3 — Execute, per gate, in the INVERTED order

For each gate with real work, in this order:

1. **Record the disposition on the gate task** (append to its `details`).
   Safe-append protocol: fetch the current text, compose the full replacement
   locally, resend, then read back and **byte-compare the original prefix** —
   `update_task`'s append semantics resend the whole field, and a transcription
   slip silently loses text.
2. **Close the gate task**: `set_task_status(id=<gate>, status='done',
   project_root=...)`. **Omit `done_provenance`** — none of the accepted kinds
   (`merged`, `found_on_main`, `deterministic-deploy`,
   `deterministic-deploy-scheduled`) truthfully describes a decision closure,
   and recording a false one feeds Stage-2 reconciliation a fabricated
   "shipped via" edge (task 3200's `details` documents this exact gap).
3. **Now run the cluster's writes**: `consolidate_memories` if landed (preferred),
   else hand-sequence — `add_memory` for the short canonical (stamped
   `topic` + `canonical: true`), `update_memory` `metadata_patch` stamps for each
   retained peer, `delete_memory` for each genuinely absorbed entry. All writes
   carry `project_id` and your `curator-<slug>` `agent_id`.
4. **If a delete refuses with `CitationRepointRequired`**: this is the detector
   working — with the gate task closed, the refusal names a **genuine
   third-party citer**. Read the named tasks, then either repoint their
   citations to the canonical (respecting the tombstone-ledger rules in
   `citation_verifier.py`) or change disposition to retain-and-tag for that
   member. **Never** answer a refusal loop with
   `allow_dangling_citations: True` (see the trap section).

### Phase 4 — Verify, per gate and at the end

- **Closure re-query** (per topic, deterministic — not a semantic top-k probe):

  ```
  get_memories_by_metadata(project_id=..., filters={"topic": "<slug>"})
  count_memories_by_metadata(project_id=..., filters={"topic": "<slug>", "canonical": True})
  ```

  Expect the canonical plus every retained peer; the canonical count must be
  **exactly 1**. `consolidate_memories`' returned `survivors` list is the same
  check done for you — `survivors: []` or a concrete leftover list, never a
  claimed closure.
- **Read back the canonical** (`get_memory_by_id`) and verify the content landed
  verbatim — verify per field, don't trust the write call's echo.
- **Citation exposure end-check**: the known victim list (12 DF + 13 reify live
  non-gate tasks, mostly via `_causation_id`, plus `consolidation_note` and
  `x_memory_write_caution`) is a **2026-08-04 snapshot** recorded in 3625's
  description and `metadata.x_citation_victim_tasks_*`; it drifts as tasks go
  terminal. Re-measure with
  `fused_memory.reconciliation.citation_verifier.find_live_citation_occurrences`
  (`citation_verifier.py:250`) — **NOT a regex**: the wrapper exists to exclude
  the tombstone ledger, which a regex would count as live. Confirm no deleted
  UUID is still cited by a live task.

### Phase 5 — Report and close

- **Per-gate ledger**: gate id → disposition, canonical id, retained-peer count,
  deleted count, any `CitationRepointRequired` hits and how each was repointed.
  This ledger lives durably in the gate tasks (Phase 3 step 1) and in task
  3524's details — the escalation records below summarize it, they are not its
  only home.
- **Resolve 3625's born-at-L2 escalation** recording the per-gate dispositions
  (its RESOLUTION clause asks for exactly this; `action='resume'` flips the
  deterministic gate).
- **Resolve esc-3524-1 LAST, and only if the sitting is genuinely done.** It is
  pending BY DESIGN and holds 3524 at `blocked`; a partial sitting leaves it
  pending with a triage-note addendum instead. When you do close it: draft the
  resolution prose first, then resolve in the very next call — the post-`done`
  revalidation sweep can overwrite a slowly-written resolution (~1 minute race).
- Flip **3524** to `done` (same no-fabricated-provenance rule as the gates),
  then `/reflect`.

---

## Hygiene the sitting must respect

- **Search before any `procedural_knowledge` write.** `add_memory` soft-blocks
  near-duplicates (`allow_near_duplicate: True` override only for genuinely
  distinct content) and rejects writes matching a known-contradictory topic
  cluster (`ProceduralKnowledgeKnownTopicClusterWriteRejected`) — see CLAUDE.md's
  Memory Usage section. A curation sitting that itself plants a near-duplicate
  is the ouroboros.
- **Never run `git stash`** in any dark-factory checkout — `refs/stash` is
  shared across worktrees and the merge worker consumes it (incident
  `13674d3c68`). This sitting should not need git at all; if it somehow does,
  park WIP as commits on a branch.
- **All task operations go through the fused-memory MCP tools**, never the
  Taskmaster CLI/MCP directly — status flips must emit reconciliation events.
- **Tag every memory write** with `project_id` and your `curator-<slug>`
  `agent_id`, consistently, for the whole sitting — the provenance trail is what
  makes the self-reported prefix auditable after the fact.
