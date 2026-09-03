# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added

#### `consolidate_memories` — one transactional op for folding a duplicate cluster (task 3133)

Replaces the hand-rolled write-then-delete choreography that made consolidation a
**ratchet**: a canonical write plus unordered deletes with no verification nets +1
entry per failed pass, which is how a cluster ends up containing the consolidator's
own prior canonicals. The cure is ordering plus a closure that is CORROBORATED by a
live re-read, never inferred from "the delete call returned ok".

Ordering is the contract, and each step sits where it does because of what its
failure would cost: (1) argument validation, pure and free to refuse; (2) fail-closed
`metadata_patch` authorization, inherited from task 3088's resolver and run
unconditionally — reparenting is only discovered after the canonical exists, so a late
denial would abort mid-transaction; (3) the same tool-layer citation gate `delete_memory`
runs, in its non-mutating `scan_only` pre-flight, so a set that cannot be cleared leaves
the corpus byte-identical; (4) the canonical write, before anything destructive; (5)
retained peers tagged, then per supersede read → re-home children → corroborate → delete;
(6) deterministic re-query; (7) tombstone; (8) structured report.

- **Retain-and-tag is the default arm**, ratified at gate 3200. Peers are stamped with
  the cluster's `topic` IN PLACE — never `canonical` (exactly one per project+topic) and
  never `parent_id` — so they keep their Qdrant point ids and every citation, parent
  pointer and supersedes edge already aimed at them stays valid. Retained ids never
  appear in the canonical's `metadata.supersedes`: they were not replaced.
  A peer is first PROVEN not to be a canonical already, and one that is gets refused
  with `RetainedPeerIsCanonical` rather than tagged. Not SETTING `canonical` is not the
  same as ensuring it is unset: the patch is a server-side payload merge, so a peer
  already holding the key would keep it and become a second claimant for
  (project, topic) — invisibly, since `_apply_canonical_uniqueness` runs only on the
  `add_memory` path. Refused rather than demoted, because a prior canonical in the
  retain list is usually what the caller should have put in `supersedes`; an unreadable
  peer fails closed the same way (`RetainCheckFailed`).
- **An id repeated within one arm is refused by name, naming both slots.** Not
  de-duplicated: silently rewriting the caller's set would make the op's own report
  describe a request nobody made — the reason the cross-arm overlap check refuses rather
  than picking an arm, and the reason `normalize_supersedes` never drops a member.
  Tolerating a repeat has three durable consequences, all avoidable for free at
  validation time: the delete arm awaits `delete_memory` TWICE for one record (a second
  `memory_deleted` event and a second WriteJournal row for a record already gone); the
  canonical's durable `metadata.supersedes` KEEPS the repeat, because the step (7)
  narrowing compares SETS and so never fires on a list differing only by duplication;
  `tombstones_written`/`tombstones_expected` BOTH count it while the recon ledger's
  five-part identity collapses the two rows into one, so the pair the envelope advertises
  as its audit-trail proof would overstate the ledger by one; and the single row that
  DOES survive is gutted — `victims_by_id` is keyed by id and reassigned per pass, so the
  repeat's pre-delete capture (running after the first pass already deleted the record)
  misses and overwrites the good capture with `metadata=None, created_at=None`, leaving
  the surviving tombstone stripped of exactly the victim-identity fields that make a dead
  id answerable. The last two are why this is refused rather than tolerated — an INFERRED
  count and a silently gutted audit row, in the op whose whole deliverable is corroborated
  facts.
- **`survivors` is the deliverable.** Computed only from a post-delete `get_memory_by_id`
  per id, so an id whose delete reported success but which still resolves is reported as
  a survivor, and an id whose delete raised but which is genuinely gone is not. Partial
  failure is a RETURNED envelope (`status='partial'` plus `failed_deletes`,
  `retain_failures`, `reparent_failures`), never a raise — `@mcp_tool_errors` would
  flatten an exception and destroy exactly those per-id dispositions.
- **Children are re-homed before their parent dies**, and a re-homing this call cannot
  PROVE complete refuses that delete instead of orphaning. The proof fails on a refused
  patch, on a truncated child listing (its count reads as a floor, "at least N"), or on a
  live post-reparent re-count that is still non-zero. The delete is never forced with
  `cascade=True`, which would destroy the children the re-homing exists to preserve.
  A truncated listing is decided BEFORE the re-homing loop runs, not after: truncation is
  known the instant the listing returns and refuses the delete unconditionally, so
  re-pointing the children that ARE visible first would buy nothing and cost a real write
  each — moving them onto the canonical while their actual parent stays alive and
  un-deleted, splitting that subtree across two live parents, with `reparented` reporting
  the moves exactly like ones that earned a delete. A refusal already determined costs
  zero mutations.
- **The delete arm stamps a task-3041 tombstone** per reaped supersede, carrying the new
  `absorbed_by` reverse pointer and the caller-supplied `run_id` as `deleting_run_id`
  (required whenever `supersedes` is non-empty; a delete that cannot be attributed is
  refused before anything is written). Tombstones are stamped ONLY for CONFIRMED-GONE
  victims — `deleted` minus `survivors` — because a tombstone over a record that still
  resolves would mint a durable audit row asserting a record is gone while it is alive.
  A shortfall is reported via `tombstones_written` / `tombstones_expected` and a WARNING;
  it deliberately does NOT flip `status`, since retrying a completed merge is the very
  ratchet this op ends.
- **The canonical claims only what it actually replaced.** `metadata.supersedes` is
  stamped at write time with the REQUESTED set — the write must precede every delete —
  so on a partial run it would name records that are still live. Step (7) patches it
  DOWN to the corroborated-gone set (the same set the tombstones are stamped for) and
  reports the narrowing as `supersedes_correction`; `canonical_supersedes` always shows
  what the record really carries, including when that patch itself failed. Nothing else
  in the system repairs this field, so an uncorrected claim would persist in the corpus
  pointing readers away from live records.
- **`partial` is not a retry signal, and there is no resume arm.** The op takes no
  existing canonical id, so re-running it for the same (project, topic) writes a SECOND
  canonical — censused but ADMITTED under the shipped warn-mode default, i.e. the very
  ratchet. The partial envelope therefore carries a `hint` with the by-hand recovery
  (fix what the failure lists name, then `delete_memory` per still-listed id, which runs
  the same citation gate and child guard, plus `update_memory` for any untagged peer).
  Related bound, stated rather than widened: "a refused consolidation leaves the corpus
  byte-identical" covers refusals from steps (1)-(4) only. The mutating citation repoint
  runs over the whole delete set right after the canonical write, so an id later refused
  for a non-citation reason has already had its citers rewritten onto the canonical while
  it is still in the corpus. Under `metadata.enforce=True` one shape refuses outright: a
  supersede that is itself the topic's incumbent canonical is still alive when the write
  probes uniqueness, so `CanonicalUniquenessViolation` names it and nothing is deleted —
  demote it (`metadata_patch={'canonical': False}`) and re-run, or leave it out of
  `supersedes`. Under the shipped `enforce=False` default the write proceeds and this
  op's own delete arm reaps the incumbent inside the same call.
- Closure reads are deterministic Qdrant work only (`get_memory_by_id`,
  `get_memories_by_metadata({'topic': T})`); `MemoryService.search` is never called and a
  test pins that negative. Registered in `DISALLOW_MEMORY_WRITES` (hence Stage 3) and in
  the orchestrator's dry-run `_DISALLOWED_TOOLS` in the same change that adds it.
- **No unguarded await after the canonical write.** Every Qdrant call here can propagate
  `TimeoutError` by contract, so each is guarded — otherwise one timeout would flatten the
  whole result to `{'error', 'error_type'}` and destroy the per-id dispositions for
  records already irreversibly gone. That covers the WRITE seams too: `update_memory`
  returns a structured rejection only for `MemoryNotFound` and re-raises every backend
  failure, so both the retain-arm tag and the reparent patch record a raise in the same
  per-id shape as a returned refusal (`retain_failures` / `reparent_failures`). The
  reparent patch matters most: it runs INTERLEAVED with the deletes, so a propagating
  raise would abandon already-deleted supersedes with neither a disposition nor a
  tombstone — un-attributable, i.e. indistinguishable from silent data loss.
- **A failed READ degrades the claim, never the envelope.** Enrichment reads degrade (a
  victim capture that fails still deletes and still tombstones, with `metadata`/
  `created_at` null; a closure scroll that fails reports `topic_members: []` with
  `topic_members_available: false`, so an empty listing is never misread as "this topic
  has no members"). Proof reads FAIL CLOSED (an unreadable child listing or post-reparent
  re-count refuses that delete with `ChildScanFailed` / `ReparentIncomplete` rather than
  reading silence as "no children"). A corroborating read that fails is a third outcome
  in its own `survivor_check_failed` list: the id is claimed neither alive nor gone, it
  is NOT tombstoned, and it makes the op `partial` — unlike a tombstone shortfall, an
  unprovable closure means the deliverable itself is missing.

Explicitly NOT claimed here: topic-cluster auto-seed (task 3135), the Stage-1 rewire and
`recon-stage-*` guard-exemption retirement (task 3134), `update_memory`'s
`_apply_memory_metadata_validation` bypass (task 3523 — this op validates the slug at
entry, bounding but not closing it), and `x_memory_citation_tombstones` on citing tasks
(task 3893 — a different object in a different store).

### Changed

#### `execution_class` blessed into Tier-A, and the Tier-A listing is now machine-checked (task 3780)

**Two changes, and the second matters more than the first.**

**`execution_class` is now a Tier-A blessed metadata key.** It had eight
production read sites across seven modules — `execution_class_guard`,
`operational_routing_guard` (which coerces `operational`/`decision` to
`task_kind='deterministic'` + `always_escalates`, a real dispatch consequence),
`routing_intent_guard`, `operational_suggestion_guard`,
`operational_ask_registry`, the `task_interceptor` gate-marker set and the task
curator's decision-cache key — while still emitting `code=unknown_key` on every
read. Measured 2026-08-18: **336 of 4204 dict-metadata tasks carry it**
(`code_tdd` 196, `operational` 126, `decision` 12, `implementation` 2), so that
is 336 census lines removed.

Blessed rather than promoted to a typed `Literal`, despite `EXECUTION_CLASSES`
looking like a closed vocabulary. Its validity rule is conditional on
recon-stage caller identity, which no pydantic field validator can see; and the
vocabulary is not closed in the data — tasks 3623/3624 carry `'implementation'`,
and both are `done` carrying `done_provenance`, so a `Literal` would raise on
every metadata write to them and they stay unrepairable until task 3777 lands.
That acceptance is now pinned by a test, so the constraint cannot be silently
re-tightened into a stranding bug. `docs/task-authoring.md` §8 "Promoting a
convention" records the generalised rule.

**`docs/task-authoring.md` §8's Tier-A listing is now pinned to the frozenset**
by `tests/scripts/test_task_authoring_blessed_keys_drift.py`, anchored on a
`tier-a-blessed-keys-mirror` marker pair. This is the structural half. The
listing is what a task author reads before deciding whether a key they are about
to write will manufacture census noise, and nothing kept it in step with the
code. The failure is measured, not hypothetical: task 4372 blessed two keys the
day before, then mirrored them into the doc by hand in a *separate* follow-up
commit whose own message names the hazard — "hand-maintained prose with no sync
test, so it drifts silently if not mirrored by hand". Under-listing is the
dangerous direction: an author sees a blessed key absent, concludes it is
unblessed, and either `x_`-renames a machine-written key — forking the
vocabulary against every sibling task and blinding its live reader — or files a
redundant blessing task. The same file's frozenset header comment had meanwhile
gone stale across two blessings, claiming 39 keys against 42; that count is now
dropped rather than re-derived, since a hand-maintained denominator needs a hand
re-count on every future blessing.

**This does NOT empty the census, and should not be read as if it had.**
Re-measured after the change: **1622 tasks still carry at least one unknown
key** (down from 1688 — the blessing removes 336 warning *lines* but clears only
66 tasks entirely, because most carry other unknown keys too), spread across
**975 distinct spellings**, 725 of which appear on exactly one task. The largest
single contributor is `related_tasks` at 445 tasks — which is the *canonical*
Tier-B spelling, not drift. Blessing one key does not move that number much, and
the remaining tail is deliberately left warning as a drift signal.

Also splits out the corpus `x_` sweep that this task originally carried, as task
4302: 23 of its 29 target tasks carry `done_provenance` and are unwritable until
task 3777 lands, and sweeping only the 6 writable ones would fork the vocabulary
for a fifth of the benefit. §8's Known-gaps table is re-pointed and re-measured
accordingly; the `execution_class` row there is now closed.

#### `migrate_task_metadata_to_x_namespace.py`: a snapshot per run, and a recovery pointer on every post-write exit (task 4125)

**Behaviour change, operator-visible.** Two changes to the pre-write snapshot
`--apply` takes, and one to what a failed run tells you.

**The default `--backup-path` is now stamped per run** —
`/tmp/task-<id>-metadata-before-<UTC-stamp>.json` (`%Y%m%dT%H%M%SZ`, the stamp
the rest of the repo already uses) rather than one fixed
`/tmp/task-<id>-metadata-before.json`. It is resolved at write time, so a dry
run now prints that shape and `(resolved at write time)` instead of a concrete
name the later `--apply` — running in a different second — would never write. `docs/task-authoring.md` §8 prescribes a
mechanical per-task re-run of this script with different `--keys`, and under
the fixed path run 2 wrote its already-partially-migrated row straight over run
1's TRUE pre-migration row — silently destroying the one artifact the SAFETY
section exists to produce, at the only moment it is wanted.

**`write_backup` now REFUSES an occupied path** instead of overwriting it (an
exclusive create, so the existence check and the create are one atomic
operation with no window for a concurrent run). This is the behaviour change to
flag: an operator reusing an explicit `--backup-path` across runs now gets the
run refused — `FileExistsError`, which the existing `except OSError` turns into
"Refusing to write without a recoverable snapshot" and exit 1 *before*
`update_task` is called — where previously the earlier snapshot was lost with
no sign. Move the file aside, or pass a different `--backup-path`. A collision
on the *default* path (only reachable from two `--apply` runs inside one
second) is not an operator error and does not abort: the run steps aside to
`...-2.json`. The create is exclusive either way, so no existing snapshot is
ever replaced.

**Every exit that leaves the stored row unverified now names that snapshot** —
the read-back that crashed, and the write call that never returned. Both used
to unwind out of `main_async` as a bare stack trace: `_fetch_task` on an
unexpected payload (or `_coerce_metadata` on a non-dict blob), and — the
likelier one in practice — a transport timeout on the `update_task` POST
itself, where the client is an `httpx.AsyncClient(timeout=30.0)` and the
payload is a whole-blob replace of a row that runs to tens of KB, so no reply
means no way to tell landed from lost. The recovery pointer was printed only on
the reported-drift exit, so it was absent from exactly the paths where the
operator has least to go on. The sentence now has a single source
(`recovery_pointer`) that every such exit prints verbatim, and the traceback is
kept and printed first: the diagnosis is not traded for the instruction. An
explicit server REJECTION is deliberately excluded — the server replied and
refused, nothing committed, and there is nothing to recover.

Extends the SAFETY section added with the script itself in task 3697 (below).

#### `record_mem0_deletion_tombstone(s)` gained an optional `absorbed_by` keyword (task 3133)

Additive and keyword-only, defaulting to `None`. It records the surviving canonical id
that absorbed a victim — the REVERSE pointer, which is what makes "where did its content
go?" answerable from the DEAD id alone (the recon-gate-165 audit dead-end: the survivor
carried a forward `consolidated_from`, but probing the victim returned `{'found': false}`
with no tombstone at all). Written as a top-level payload field rather than through
`victim_metadata`, whose `_VICTIM_IDENTITY_KEYS` projection is deliberately what the
VICTIM recorded about itself. The two existing GC/trim sweeps absorb nothing, need no
edit, and now write `absorbed_by: None` — present, not omitted, so absence reads as
"nothing absorbed it" rather than "this row predates the field".

#### `update_memory`'s default allowlists now admit `curator-` on both arms (esc-3524-1)

**Behaviour change, operator-visible.** `Mem0UpdateConfig`'s
`content_amend_allowed_agent_prefixes` and `metadata_patch_allowed_agent_prefixes`
default to `['recon-stage-', 'curator-']` instead of `['recon-stage-']`. The
`curator-` prefix is the dedicated opt-in identity for the interactive
memory-consolidation sitting (`skills/curate-fused-memories`), granted both arms
by Leo's ruling (b) on esc-3524-1 (2026-08-11) and promoted from a per-machine
`config.yaml` override to an all-deployments schema default on 2026-08-12 — the
sitting skill does not work without it. `config.yaml`'s `mem0_update:` block is
fully commented out again; the premise tripwire in
`test_recon_amend_tool_advertisement.py` (which fired on the 2026-08-11 override
commit `65b011ed8c` and turned main red) is re-armed: it still fails on any
active YAML override of these leaves. No recon-stage capability changed;
`agent_id` remains self-reported (a misuse deterrent, not a security boundary).

#### `delete_memory`'s citation gate now applies to EVERY caller (task 3624)

**Behaviour change, operator-visible.** A `store='mem0'` `delete_memory` call
that succeeded yesterday can now be REFUSED with
`error_type='CitationRepointRequired'`. The pre-delete citation-repoint gate
(task 3108) was scoped to `recon-stage-*` callers; it now keys on the RECORD —
`store == 'mem0'` and a scannable registered project — because "will this delete
dangle a live task pointer?" is a property of the entry and the task DB, not of
who issued the delete. Under the old predicate the identical delete issued from
an interactive session landed unguarded, and a caller with no `agent_id` at all
was the *least* guarded one. This is what gates the 25-gate memory-consolidation
batch tracked by task 3524 / esc-3524-1, which is driven from an interactive
session rather than a recon-stage agent. An uncited entry is unaffected: the scan
finds nothing live and the delete proceeds as before.

**Migration note:** the remedy is unchanged — retry with
`replacement_memory_id=<the surviving entry's full 36-char UUID>` and the live
citers are repointed before the delete runs. For a delete with *no* surviving
entry to repoint to — a plain drop rather than a consolidation, which
`replacement_memory_id` cannot express — pass
`metadata={'allow_dangling_citations': True}`. Only a literal boolean `True`
counts, matching the `allow_mcp_markup` convention below; a truthy `'yes'` or `1`
does not unlock an irreversible delete — and does not vanish either: a supplied
value that is not a literal boolean comes back as `ignored_override` on the
rejection, with a `hint` sentence saying why, rather than leaving the caller to
retry the flag they believe they already passed. (A literal `False` is a
deliberate "no" and is honoured without comment.)

The `hint` names the escape, so it is discoverable from the rejection itself —
but only on `CitationRepointRequired`. The `CitationReplacement*` refusals
deliberately do **not** advertise it: those are reached by *naming* a survivor,
so the caller demonstrably has one and their fix is to correct the UUID. A
consolidation delete has a survivor by definition, which is exactly the case the
escape is wrong for.

Two deliberate differences from `allow_mcp_markup`. First, the override is not
silent: it is recorded at `WARNING` naming the deleted id, the `agent_id` that
asked, and every live citer it strands — an override that lands silently is the
same class of defect as the gate that never ran. The same enumeration is
returned to the caller (`dangled_citations`, `dangled_citation_count`), because
an MCP caller never sees the server's log stream and a bare
`{'status': 'deleted'}` would be silent from the only vantage point that matters
to them. Supplying a `replacement_memory_id` *alongside* the flag is
contradictory; the override wins, and the ignored value is named both in that
log line and as `ignored_replacement_memory_id` on the response rather than
dropped in silence.
Second, `allow_mcp_markup` is "stripped before persistence" whereas this flag
needs no strip at all: `delete_memory` discards `_extract_causation`'s cleaned
dict and `memory_service.delete_memory(...)` takes no `metadata` parameter, so
the flag structurally cannot reach the store.

The escape is a property of stated intent, not of identity — it is available to
recon-stage callers too, since a second caller-identity check would reintroduce
exactly the scoping this task removed. It deliberately does **not** unlock the
fail-closed path: an override plus an unreadable task DB is still
`CitationScanFailed`, because the flag means "I accept dangling the citers you
just showed me" and with nothing enumerated there is nothing to knowingly accept.

**Cost.** Every `store='mem0'` delete now pays one `task_interceptor.get_tasks`
read plus a whole-tree metadata walk, so a 25-delete batch pays it 25 times. The
snapshot is deliberately *not* cached across calls: it is the last read before an
irreversible delete, and a task that starts citing the doomed id after a cached
snapshot was taken would be invisible to the gate — trading its fail-closed
guarantee for a race, on the exact operation the gate exists to protect. The cost
is made observable instead: each scan logs its task count and duration at `DEBUG`
(`citation gate: scanned N task(s) ... in X ms`), so a project large enough for
this to matter shows up as a measurement.

#### Leaked tool-call XML: root cause, and the tooling to sweep the corpus (task 3083)

**This task ships no write-time guard.** Live rejection at the MCP write
boundary is delivered by `fused_memory/server/markup_tripwire.py` (task 3141 —
see "MCP writes carrying raw envelope markup are now REJECTED" below), which is
deliberately the only place in the package that enumerates the markup literals.
An earlier revision of this task added a second, parallel guard; it was retired
before merge as a duplicate enumeration of that invariant. If you need to write
content that legitimately quotes a fragment, the override is 3141's
`metadata={'allow_mcp_markup': True}` and the rejection is
`error_type = 'McpEnvelopeMarkupWriteRejected'`.

**What this task establishes: the fragments do not originate in this repo.**
There is no XML parser anywhere in `fused-memory/src/`. They are evidence that
the *harness's* tool-call parser terminated a string argument early at a literal
closing tag inside that argument's value, which also **silently swallows the
sibling arguments that followed**. A leaked fragment in `description` means
`priority` may never have reached the MCP boundary at all, in which case
`sqlite_task_backend.py`'s `priority or 'medium'` substituted a plausible wrong
value with no log. That sibling-argument loss is the part of the failure that is
otherwise invisible — a corrupted record looks merely untidy, not wrong.

**Added:** `fused_memory/utils/toolcall_xml_leak.py`, the shared detector for
*already-stored* content, now the single implementation behind the corpus sweep,
`scan_memory_content`, the redaction path, and `scripts/scan_task_toolcall_leaks.py`
(which previously carried its own inline copy of the pattern).

It is calibrated in the **opposite** direction from 3141's tripwire, on purpose:
the tripwire over-reports to maximise recall at write time, where a false
positive costs one retry; this detector under-reports to stay precise, because
it runs over already-stored content where a false positive would silently
rewrite something a user wrote. That asymmetry is why both exist, and why this
one must not adopt the tripwire's pattern list.

**Also added:** `scan_memory_content` (read-only literal substring scan over
Qdrant payload text — semantic `search` provably cannot find these),
`redact_episode_content` (neutralise a leaked Graphiti episode without
`delete_episode(cascade=True)` destroying its valid extracted edges), and
`fused-memory/scripts/sweep_toolcall_xml_leak.py` (the corpus sweep; dry-run by
default).

**What the sweep's repair preserves.** The memory **text** in full, plus the
**payload metadata** that is the record's only metadata-scoped retrieval axis —
carry-over rule `payload keys - _MEM0_OWNED_KEYS`, with the scope identities
threaded back as `agent_id=` / `session_id=` arguments and anything carried
nowhere *named* per-record in the report's `metadata_dropped`. Without that,
a repaired record would silently vanish from `get_memories_by_metadata` /
`count_memories_by_metadata`, which match payload keys by equality.

**The sweep verifies the re-add persisted** rather than trusting a non-raising
`add_memory`: the service swallows a Mem0 write failure into
`AddMemoryResponse.message` as `[mem0_error: ...]` and returns normally, so a
returned response is not evidence of a write. Three per-record outcomes exit
non-zero and need a human — `content_lost_in_flight` (the delete landed, the
re-add did not persist; the original text now exists only in the printed report,
restore it from there before re-running), `skipped_not_mem0_routed` (a
repairable record whose category does not route to mem0, left entirely untouched
because neither a plain re-add nor `dual_write=True` is safe), and
`record_error` (that record's repair aborted on an unexpected error, so whether
its delete landed is unknown).

**The report always survives an abort**, since for a `content_lost_in_flight`
record it is the only remaining copy of the original text. Each record enters
the report *before* any store mutation is attempted and its repair runs under
its own `try`, so one record's transport failure is recorded as `record_error`
on that record and the sweep continues instead of unwinding and discarding every
earlier entry. If anything escapes anyway, the CLI prints the **partial** report
— same shape, plus `"aborted": true` — before exiting `2`.

**Full root cause, evidence, and operator runbook:**
[`docs/mcp-toolcall-xml-leak.md`](docs/mcp-toolcall-xml-leak.md). Run the sweep
**before** any further large consolidation pass — consolidation deletes
corrupted entries as a merge side effect and destroys the specimens.

#### First live-corpus sweep: ~0.19% incidence, and `--apply` must not run sandboxed (task 3567)

Ran 3083's sweep against the live Mem0/Qdrant corpus. This ships no runtime
behaviour; the deliverable is committed evidence in
[`docs/toolcall-xml-leak-sweep-2026-08-05/`](docs/toolcall-xml-leak-sweep-2026-08-05/)
— verbatim report, provenance sidecar, and `investigation.md`.

**The incidence measurement**, the first anyone has had: an exhaustive walk of
**21,080 points** found **41 leak-carrying records (~0.19%)** — 1
`repairable_duplicate`, 40 `manual_review`, 0 `repairable_tail`. Read it as a
statement about what the sweep walked, not a clean ratio over the collection:
the corpus takes concurrent writes, consolidation *deletes* entries so the count
is not monotonic, and `scan_payload_text` scopes by `group_id` while a
collection count does not. Both 2026-07-27 specimens were already consolidated
away before the run — the evidence-loss risk that motivated running this now was
real, and it had already fired.

**All 40 `manual_review` records were adjudicated by hand** (nothing mutated),
each verdict re-derived by running the production detector over that record's own
content. They share one shape, and it is not the shape 3083 anticipated: a stray
closing tag followed by a `parameter` continuation naming a **sibling argument**
whose value is the swallowed remainder — manifestation #1 (sibling-argument loss,
the silent one) in live data at scale. 35 swallowed a short `priority`/`category`
value; 3 are nested double leaks that are functionally `repairable_duplicate` but
invisible to the detector; **2 need a human**, holding ~1KB of real prose and a
JSON blob that a tail-drop would destroy. A third repairable shape would make 38
of the 40 auto-repairable — filed as follow-up, not done here.

**`--apply` must not be run from a sandboxed agent session.** The gated apply run
repaired 0 records and **deleted one record from Qdrant with no re-add**
(`7d073281`, flagged `record_error`). The sandbox denies file *creation* under
`~/.mem0`, so SQLite cannot create its rollback journal and mem0's history write
fails with *"attempt to write a readonly database"* — while the Qdrant delete,
being a network call, succeeds. That splits delete-then-re-add down the middle.
The record's text survives **only because the dry-run report was committed
first**, which is exactly the ordering constraint the runbook prescribes; that
safety net fired on its first real use. Two lessons now in the runbook: read the
per-record flags rather than the exit code (exit 1 was overdetermined here, and
the exit code alone would have hidden the mutation), and run `--apply` from an
ordinary interactive shell. Recovery of `7d073281` from the committed report is
**tracked as task 3686**, together with the sandbox-policy decision it depends
on; the blocking escalation was auto-dismissed on timeout rather than
adjudicated, which is why the ask was converted into a task.

`fused-memory/tests/test_toolcall_xml_leak_sweep_artifacts.py` guards any
committed sweep artifact against fabrication: it re-runs the production
`classify_record` over each record's own stored content and requires a provenance
sidecar naming the collection, sha, argv and bracketing point counts. It also
requires every unrepaired live-corpus mutation to carry a `recovery-tracking.json`
entry naming a task owner — and re-runs the detector over the recoverable payload
to prove the documented recovery is still executable, so deleting the artifact
holding the only surviving copy fails loudly instead of silently. Hermetic — no
live store — so it stays green whether or not Qdrant is up.

#### `allow_mcp_markup` documented; `last_blocked_at` blessed (task 3697)

**`allow_mcp_markup` is now documented in `docs/task-authoring.md` §8** as the
correct move for a write that deliberately quotes the MCP envelope literals. The
guard (task 3141, below) has been live and working since it shipped; what was
missing was the convention around it, so authors reached for a workaround
instead — paraphrase the literals, then park the evidence under a bespoke
timestamped metadata key. That workaround manufactures *both* failure classes at
once: a `code=unknown_key` census line for every such key, and a bounced write
for every author who quotes the literals without the flag. It was
self-perpetuating because it was documented inside task 3083's own `details`.
The new section records the scope of the gate (text arguments only — never the
metadata blob), that only a literal boolean `True` enables it, and that it is
write-time-only and stripped before the merge, so it never persists into stored
metadata. It deliberately does not restate the literals: doing so would oblige
every future task write quoting that section to set the override, which is the
loop it exists to break.

**`last_blocked_at` promoted to the Tier-A blessed-key allowlist**
(`_BLESSED_METADATA_KEYS`, mirrored into the hand-maintained listing in
`docs/task-authoring.md` §8 in the same commit — that listing is a manual copy
with no drift test guarding it). It is written by the orchestrator on every
block and read back by the briefing stale-check, and 78 tasks carry it, so every
one of them was minting an `unknown_key` line. Promotion rather than an `x_`
rename, because renaming a machine-written key on one task forks the vocabulary
against its siblings and the orchestrator would re-add the canonical spelling on
the next block anyway.

**Added `fused-memory/scripts/migrate_task_metadata_to_x_namespace.py`** for
retiring ad-hoc metadata keys into the `x_` namespace. The gotcha it exists for:
**`metadata_mode='merge'` cannot retire a key.** `_merge_metadata` is a shallow
`{**old, **new}` with no deletion sentinel, so merge can add the `x_` spelling
but never remove the old one — both would coexist and the warnings would
persist. The script is dry-run by default (`--apply` required), requires an
explicit `--task-id`, refuses on a collision rather than clobbering, and runs a
mandatory read-back proving the `x_` keys landed, the old spellings are gone,
sibling keys are byte-identical, and `description`/`details` sha256 and `status`
are unchanged. Two guards cover what the read-back structurally cannot, since
it verifies *intent* (did the rename land) and not *safety* (should it have):
`--keys` is validated up front and refuses an already-`x_`-prefixed, typed
`TaskMetadata` or Tier-A blessed key (`--force` overrides), and `--apply`
writes the full pre-write row — metadata *and* `description`/`details` — to
`--backup-path` first, refusing to write at all if that snapshot cannot be
saved. The read-back also discounts the backend's own reserved control keys
(`append`, `metadata_mode`, stripped from every incoming payload in all modes),
so migrating a task that carries a leaked control key reports the drop as
information rather than raising a false corruption alarm after the write has
already committed.

**This task's stated signal — zero `unknown_key` lines on task 3083 — is
PARTIALLY MET, and is recorded as such rather than claimed.** Blessing
`last_blocked_at` took it from 7 to 6. The remaining 6 need a live metadata
write that is currently impossible: `update_task` rejects any metadata payload
containing `done_provenance` (a presence-only write-authority floor evaluated
before `metadata_mode` is resolved), so a whole-blob `'replace'` cannot run
against any `done`/merged task, and task 3083 is one. Task 3083 is unchanged and
undamaged — metadata still 20124 bytes / 18 keys, `description` and `details`
sha256 identical to the pre-migration snapshot, status still `done`. Ticket
`tkt_0RS4WVMH1RSTSY88N781E70F5S` owns the write-path decision and the re-run.
Two further pre-existing gaps are measured and recorded in the new "Known gaps"
table in `docs/task-authoring.md` §8, owned by
`tkt_0RS4XDWJQ9PR8MFXY5DKW950WS`: `execution_class` is read by two live guards
but is neither blessed nor typed (272 tasks), and the `x_` sweep has not been
run corpus-wide.

#### Plan-decision cross-pairing: a re-runnable scanner, and where it's documented (task 3967)

A `design_decisions` entry whose `decision` and `rationale` are each well-formed
prose but wrongly **paired** — a different damage class from the envelope
leakage above, and disjoint from it at the detector (`shared.toolcall_markup.detect`
cannot see a mis-pairing by construction, not by oversight). The full account —
the shape, why repair is impossible, why no deterministic write-time predicate
can contain it, and the containment measurement — lives in
[`docs/plan-decision-cross-pairing.md`](docs/plan-decision-cross-pairing.md).
Treat every prevalence figure there as a dated, strict lower bound, not a
number to trust — the corpus keeps growing — and re-run the scanner instead of
citing it.

**Added `scripts/scan_plan_decision_pairing.py`**, a read-only, re-runnable CLI
(`--root`, `--json`, `--fail-on-hit`, `--require-scanned`; no `--apply`, ever).
The invocation safe for an unattended gate (CI job, timer):

```bash
python scripts/scan_plan_decision_pairing.py --fail-on-hit --require-scanned 1
```

`--require-scanned` must accompany `--fail-on-hit`: alone, `--fail-on-hit` keys
only on hits, so a `--root` that is mistyped, not yet created, or unlistable
yields zero hits over zero scanned files and exits 0 — indistinguishable from a
clean corpus. `--require-scanned N` states the coverage floor instead and exits
**3** when fewer than `N` plan files were actually read, which outranks
`--fail-on-hit`'s exit 1.

### Changed (BREAKING)

#### MCP writes carrying raw envelope markup are now REJECTED (task 3141)

Four fused-memory MCP write tools — `add_memory`, `add_episode`, `submit_task`,
`update_task` — now **hard-reject** a write whose payload contains a raw MCP
tool-call envelope fragment.  Writes that previously succeeded (and silently
persisted the fragment) now fail.

**Rejected fields:** `content` for the memory tools; `title`, `description`,
`details` and `prompt` for the task tools — the same four-field set
`premise_lint_guard` already lints, since all four reach the same description
parser.

**Response shape** (the write does NOT reach the store or the interceptor):

```
error       : mcp_markup_write_blocked
error_type  : McpEnvelopeMarkupWriteRejected
field       : which field tripped
matched_pattern : the exact literal that matched
content_excerpt : first 200 chars of the offending text
hint        : remediation + the override key + DF 3083
storm       : present ONLY on a rejection burst (count/threshold/window_seconds
              /hint, plus escalation_id when one was filed)
```

**Why:** a harness serialization bug leaks envelope fragments into write
payloads.  Two observed vectors: memory `content` arriving with a closing-tag
tail (permanent specimens now sitting in the mem0 and Graphiti corpora), and
task text arriving with a `<parameter name=`-shaped fragment that the
interceptor's description parser then mis-parsed **silently** — one reify task
was filed `priority=high` and stored as `medium`.  Loud rejection at the
boundary is strictly better than either outcome.  Rejecting is deliberately
scoped to CONTAINMENT: DF task 3083 owns the root cause, the Qdrant payload
text-match read tool and the retroactive corpus sweep.

**Migration note:** if you write text that quotes envelope markup on purpose —
documenting this very leak, for instance — pass
`metadata={'allow_mcp_markup': True}`.  Only a literal boolean `True` counts,
and the flag is write-time-only: it is stripped before persistence, so it never
enters stored memory metadata or the task metadata vocabulary.  An accidental
serialization leak never sets an explicit flag; an author can.  The matcher is
deliberately a bare case-sensitive substring scan and therefore over-reports
relative to the retrospective `scripts/scan_task_toolcall_leaks.py`; the
authoritative pattern list and the reasoning behind the differing calibration
live in `fused-memory/src/fused_memory/server/markup_tripwire.py`.

**Also new:** a per-server rolling-window storm counter.  Three rejections
within an hour emit one greppable `markup_tripwire_storm` ERROR log line and
file a best-effort `mcp_markup_write_storm` escalation (level 1, deduped against
an open one), because a burst means the upstream leak is *active* rather than
that the tripwire is misfiring.  Escalation is purely additive — a queue
failure never changes a rejection's outcome.

#### `reservation_installed` (reason=reserve_now) → `reserve_now_consumed` (task 1230)

The reserve-now short-circuit path in the scheduler **no longer emits**
`reservation_installed` with `data.reason == 'reserve_now'`.  It now emits the
dedicated `reserve_now_consumed` event instead.

**Old behaviour (pre-task-1230):**

```
event_type : reservation_installed
data       : {modules, priority, reason='reserve_now'}
```

**New behaviour:**

```
event_type : reserve_now_consumed
data       : {modules, priority}
```

**Commits:** `4d45eecd9b` (add `reserve_now_armed`/`reserve_now_consumed`
`EventType` members) · `deb8f426ab` (replace `reservation_installed` with
`reserve_now_consumed` at the scheduler short-circuit emit site, steps 5-6).

**Migration note:**  Any downstream consumer (dashboard query, log filter,
reconciliation tooling, external subscriber) that was filtering on
`event_type = 'reservation_installed' AND data->>'reason' = 'reserve_now'`
must migrate to `event_type = 'reserve_now_consumed'`.

**Threshold-based reservation path is UNCHANGED** — the scheduler still emits
`reservation_installed` when the skip-count threshold is exceeded
(`scheduler.py:1593`); that event's data payload has never contained a
`reason` key (`data = {modules, skip_count, priority}`).  The two events are
now discriminated by **event name**, not by a `reason` field:

| Event | Path | Data keys |
|---|---|---|
| `reservation_installed` | threshold (skip_count ≥ threshold) | `modules`, `skip_count`, `priority` |
| `reserve_now_consumed` | reserve-now short-circuit | `modules`, `priority` |

**Dual-emit rejected:** Option (b) — emitting both `reservation_installed` and
`reserve_now_consumed` during a deprecation window — was evaluated and
rejected.  It would protect zero in-repo consumers (see audit below) and would
require reverting the deliberately-added, merged locked-in regression test
`TestReserveNowConsumedShortCircuit` (part b, `test_scheduler_state.py:191-197`)
from task 1230, contradicting a merged decision for no benefit.

**Audit result (task 1333) — in-repo blast radius: ZERO:**

An exhaustive search across all `*.py`, `*.md`, `*.sql`, `*.js`, `*.ts`,
`*.html`, `*.yaml`, `*.yml`, `*.json` files (excluding `.venv/`, `.git/`,
`uv.lock`) found the following `reservation_installed` sites:

- `orchestrator/src/orchestrator/event_store.py:75` — enum definition only
- `orchestrator/src/orchestrator/scheduler.py:1593` — **threshold-path emit**
  (unrelated to reserve_now; no `reason` key; UNCHANGED)
- `orchestrator/src/orchestrator/scheduler.py:2030` — code comment only
- `orchestrator/tests/test_scheduler_state.py:155,191-197` — task-1230
  locked-in regression test (asserts short-circuit emits `reserve_now_consumed`
  and no legacy `reservation_installed` with `reason=='reserve_now'`)
- `orchestrator/tests/test_scheduler.py:2019,2206,2230-2234,3899,3952-3955,4022-4025`
  — tests of the threshold `reservation_installed` path (unaffected)

Dashboard (`dashboard/`): all `reserve_now` references concern the override
*flag* (UI badges, clear-fields allow-list, POST body, scheduler-snapshot
reads) — the dashboard does **not** consume `reservation_installed` events from
the event_store.  No `scripts/`, SQL, JS/TS, or documentation file consumes or
filters the event.

No emit site in the current (post-task-1230) tree sets `data.reason` on
`reservation_installed`.  (Before task 1230 the reserve-now short-circuit did
emit `reservation_installed` with `reason='reserve_now'`; removed in
`deb8f426ab` — historical event-store rows predating task 1230 may still
contain it.)  The renamed event's pre-1230 form (`reservation_installed` +
`data.reason='reserve_now'`) had zero in-repo consumers.
