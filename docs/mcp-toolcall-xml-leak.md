# The MCP tool-call XML serialization leak

**Task 3083 · root-cause artifact · 2026-07-29**

Corrupted fragments of serialized tool-call XML have been turning up inside
stored task descriptions and stored memories. This document records what
they are, where they come from, why they were invisible for so long, and what
this repo now does about them.

> **Notation.** Every sentinel below is written with the HTML escape
> `&#60;` in place of a literal `<`. This is not cosmetic. Writing the raw
> literal into a source, test, or doc file forces the authoring agent to emit
> that literal inside its own tool call — reproducing the exact bug this
> document describes and corrupting the file being written. The same rule is
> enforced in code with the `\x3c` escape; see the comment at the top of
> `fused-memory/src/fused_memory/utils/toolcall_xml_leak.py`.

---

## 1. Verdict: one bug, and it is not in this repo

Two failure shapes were reported separately and looked like two problems. They
are **one defect with two manifestations**, and it lives at the harness's
tool-call XML serialization boundary, **upstream of this repository**.

The mechanism: the model emits a parameter's **closing** tag in the wrong
dialect — echoing the parameter *name* (`&#60;/description>`, `&#60;/content>`,
`&#60;/rationale>`) instead of the canonical `&#60;/parameter>` — and frequently
continues the remaining parameters in that same wrong dialect. The harness's
tool-call parser does not find the closer it expects. It **over-consumes**: it
runs forward to the next available terminator — a later well-formed
`&#60;/parameter>`, else `&#60;/invoke>` — dumps everything it swallowed along
the way into the *current* parameter's value, and **silently drops every
parameter in between**.

That direction, not early termination, is what the evidence requires. The
observed signature is a fragment **inside** the stored value *and* sibling
parameters **missing**. Early termination is inconsistent with both halves at
once: it would end the argument *at* the tag, so the fragment would be
**absent** from the stored text rather than embedded in it, and the parser
would resume normally at the next parameter, so the siblings would be
**intact** rather than missing. Only over-consumption produces both halves
together — a fragment landing inside the value *and* everything between it
and the fallback terminator disappearing.

### The four specimen shapes

Four measured specimens (`plans/toolcall-markup-containment-prd.md` §2.1) show
the mechanism end to end — what the model emitted, what the parser actually
kept, and what silently never arrived:

| Call | Value tail (as parsed) | Keys received | Dropped |
|---|---|---|---|
| `submit_task` 07-30T16:47Z | `…direction.&#60;/description>\n&#60;priority>medium&#60;/priority>\n&#60;agent_id>…&#60;/agent_id>\n&#60;metadata">{…}&#60;/metadata">\n&#60;/invoke>` | `project_root, title, description` | **priority, agent_id, metadata** |
| `submit_task` 08-04T12:13Z | `…automatically).&#60;/description>\n&#60;parameter name="priority">low` | no `priority` | **priority** (intended `low`) |
| `add_memory` 08-04T16:58Z | `…by design.&#60;/content>\n&#60;/invoke>` | all | none — `content` was the **last** parameter, so over-consumption ran straight to the invoke closer with nothing in between |
| `update_memory` 08-02T21:26Z | `…scoped).&#60;/parameter>\n&#60;parameter name="agent_id">escalation-watcher-l2` | no `agent_id` | **agent_id** — that boundary is **ungated**, so the corrupted value landed |

Specimen 1's metadata tags are a literal blend of the two dialects — a stray
double quote before the angle bracket, the model interpolating between the
canonical `&#60;parameter name="X">` opener and the name-echoing `&#60;X>`
form. `shared.toolcall_markup`'s repairer carries a one-character blend
tolerance (the trailing `"?` in its closer/opener regexes) for exactly this
specimen — without it, the blended item is leftover text and the repair is
rejected.

### Blast radius

334 corrupted calls out of 128,066 tool calls measured = **0.26%**
(`plans/toolcall-markup-containment-prd.md` §2.3; measured from the archived
agent transcripts against the collection predicate stated in PRD §2 — the
tree grows live, so a later re-run will find a larger denominator). Per
server:

| Server | Corrupted calls | Guarded at measurement (2026-08-05) |
|---|---|---|
| orchestrator/plan-tools | 198 (59.3%) | **no** |
| fused-memory | 96 (28.7%) | 4 of 6 write tools |
| orchestrator/verdict-tools | 19 (5.7%) | **no** |
| escalation | 18 (5.4%) | **no** |
| Claude Code builtins (`Agent`, `Edit`) | 3 (0.9%) | out of scope — not our servers |

Top victims: `add_design_decision.rationale` (109), `add_memory.content` (90),
`add_design_decision.decision` (33), `add_reuse_item.how` (33),
`submit_review_verdict.summary` (19), `escalate_info.detail` (17).

The "no" rows are not permanent. Closing that coverage gap is owned by the
containment PRD's middleware-registration task (task 3690, registering
`MarkupGuardMiddleware` on all four servers) — once that lands, three of the
rows above go stale. The column is dated for exactly that reason: it
describes coverage **at measurement**, not coverage today, so this table
cannot silently rot into a false present-tense claim.

### The negative evidence that settles it

The original report blamed "the submit_task/update_task description parser."
That parser does not exist. Measured on the base branch:

| Probe | Result |
|---|---|
| `ElementTree` / `xml.etree` / `lxml` / `BeautifulSoup` / `html.parser` in `fused-memory/src/` | **zero hits** — there is no XML parser in fused-memory at all |
| Serialized tool-call literals in any production write path across `fused-memory/src/`, `orchestrator/src/`, `shared/src/`, `escalation/src/` | **zero hits** — nothing in this repo ever emits one |
| `submit_task` / `update_task` priority parameter | `priority: str \| None = None` — no enum, no `Literal[...]`, no pydantic `Field` |

Since nothing here parses or emits tool-call XML, a fragment appearing in
stored text is **positive evidence** that the corruption happened before the
MCP boundary was reached. fused-memory stores the corrupted string faithfully;
it is a witness, not a culprit.

The silent default that turns the corruption into a *wrong value* is real and
in this repo, though:

- `fused-memory/src/fused_memory/backends/sqlite_task_backend.py:2491` —
  `status, priority or 'medium', metadata, _now(),` in the INSERT
- `fused-memory/src/fused_memory/middleware/task_interceptor.py:1565` —
  `priority=str(kwargs.get('priority') or 'medium'),`

Both catch `None`. Neither logs anything.

### The single literal source (INV-5)

`shared/src/shared/toolcall_markup.py` (task 3688) is now the **one** place
the envelope literals are enumerated.
`fused_memory.server.markup_tripwire.MCP_MARKUP_PATTERNS`
(`markup_tripwire.py:63`) and
`fused_memory.utils.toolcall_xml_leak.PREFILTER_NEEDLES`
(`toolcall_xml_leak.py:124`) are both now **re-exports** of names defined
there — two named predicates over one literal set, not two literal sets —
and no third site enumerates them.

That arrangement preserves the write-time recall-first vs. read-time
precision calibration split: `MCP_MARKUP_PATTERNS` still over-reports on
purpose at the write boundary, where a false positive costs only a retry,
and `PREFILTER_NEEDLES` still under-reports on purpose over already-stored
content, where a false positive would silently rewrite a user's memory.
What the consolidation removes is the divergence that used to sit
underneath that split — the write-time list carried one closing tag while
the read-time list carried four. That divergence is exactly what made the
original diagnosis ambiguous: a mis-closed `description` could not report
its own tag, so the write-time guard blamed whatever happened to follow it.

---

## 2. The two manifestations

### Vector 1 — sibling-argument loss (silent, and the dangerous one)

Over-consumption swallows every parameter between the mis-closed tag and the
terminator it falls back to, so those parameters silently never reach the MCP
boundary. `priority` never reaches the MCP boundary, arrives as `None`, and
`priority or 'medium'` substitutes a plausible wrong value. The intended
priority survives only as *text inside the description*, where nothing reads
it.

Nothing is logged. Nothing looks broken. The task simply runs at the wrong
priority forever.

Evidence: reify tasks **#3210** and **#5219**; and three of the four live
fixtures in `scripts/tests/test_scan_task_toolcall_leaks.py` — tasks **992**
(`priority` = `low`), **1068** (`priority` = `high`), and **1067**
(`priority` = `polish`). The fourth, task **2691**, is the swallowed-`details`
variant: an entire serialized `details` argument leaked into `description`
while the real `details` column was left empty. That three of four recorded
specimens are a lost `priority` argument is the direct corroboration of this
vector.

### Vector 2 — content self-duplication (visible)

The text over-consumption dumped into the surviving value leaves a visible
tail plus a verbatim duplicate of the body text — the same shape the sweep
classifies as `repairable_duplicate` (§5).

Evidence: Mem0 records **9f2d2ae6** and **c759c53b** (both 2026-07-27).

These are the same defect seen from two sides: vector 1 is what happens to
the parameters over-consumption swallowed, vector 2 is what happens to the
text it dumped into the surviving value. Treating them as one is not a
rhetorical claim — it is enforced in code by a single shared detector,
`fused_memory.utils.toolcall_xml_leak` (whose literals are now
`shared.toolcall_markup` re-exports; see §1), which all three consumers
import.

---

## 3. Why `search` could never find these

The obvious remediation — search the corpus for the fragment — does not work,
and the reason matters because it is why the leak went unswept for months.

`search` is **semantic**. A leaked fragment is punctuation and tag names; it
carries almost no semantic signal, so it ranks nowhere. A live watcher probe
on **2026-07-26** searched for exactly these fragments and returned **zero
results** against a corpus that provably contained them.

The other two read paths did not help either. `get_memories_by_metadata` and
`count_memories_by_metadata` match **metadata equality**, not payload text —
and the leak is in the text. `get_memory_by_id` needs an id you do not have.

There was simply no way to ask "which memories contain this substring."

That is now `scan_memory_content` (WORK b): a **literal** substring scan over
Qdrant payload text. It uses a server-side `MatchText` prefilter for speed and
then re-verifies **every** returned record in Python with the shared detector,
which is the authoritative verdict. `exhaustive=True` skips the prefilter
entirely and paginates the whole collection — that is the mode for an
incidence-rate claim, so the number rests on nothing but the detector.

Pagination is mandatory in both modes. A silently-capped scan would answer
"what is the true incidence rate" *wrongly*, which is the same silent-wrong-value
failure class this whole task exists to kill.

---

## 4. What changed in this repo

| Change | What it does |
|---|---|
| `fused_memory/utils/toolcall_xml_leak.py` | The single shared detector. Promoted from `scripts/scan_task_toolcall_leaks.py` (task 2939) and generalized for the Mem0 specimens; its envelope literals are now `shared.toolcall_markup` re-exports (task 3688, §1). |
| `fused_memory/server/markup_tripwire.py` (task 3141 — NOT this task) | Live rejection at the MCP write boundary. Listed here only so the picture is complete; see "The boundary rejection" below for why this task ships no guard of its own. |
| `Mem0Backend.scan_payload_text` → `MemoryService.scan_memory_content` → `scan_memory_content` MCP tool | The missing read capability (§3). |
| `fused-memory/scripts/sweep_toolcall_xml_leak.py` | The corpus sweep (§5). |
| `GraphitiBackend.redact_episode_content` + MCP tool | The residual-episode path (§6). |

### The boundary rejection — owned by task 3141, not by this task

`submit_task`, `update_task`, `add_memory`, and `add_episode` **reject** a call
whose text carries a leaked fragment, returning
`error_type = 'McpEnvelopeMarkupWriteRejected'` before anything is persisted.
Opt-out: `metadata={'allow_mcp_markup': True}` — load-bearing rather than
theoretical, since this task's own description quotes every sentinel verbatim
and this document could not otherwise be filed as a task.

That guard is `fused_memory/server/markup_tripwire.py`, delivered by task 3141.
**This task deliberately ships no write-boundary guard of its own.** An earlier
revision of this branch did, and it had to be withdrawn: it would have been a
second enumeration of the envelope literals at the same four call sites, which
directly contradicts the invariant 3141 states in `markup_tripwire.py` and in
the `add_memory` docstring — that module is *"deliberately the only place in the
package that enumerates the literals."* Two rival guards at one boundary is
precisely the drift this task exists to close, not to reproduce.

The division of labour that survives is real and worth stating, because the two
detectors are calibrated in **opposite** directions on purpose:

| | `markup_tripwire` (3141) | `utils/toolcall_xml_leak` (this task) |
|---|---|---|
| Runs at | write time, before persistence | over already-stored content |
| Method | bare substring scan | precise regex, requires real whitespace |
| Calibration | over-reports to maximise recall | under-reports to avoid false positives |
| Cost of a false positive | a retry | an unnecessary rewrite of stored memory |

Neither is redundant, and they must not be collapsed. A write-time false
positive costs the caller one retry; a sweep-time false positive silently
rewrites content a user wrote. That asymmetry is the whole justification for
maintaining two detectors — but it justifies two *predicates*, not two
*literal lists*. Both now enumerate the same set, defined once in
`shared.toolcall_markup` (§1, "The single literal source"); what still
differs, on purpose, is method and calibration, exactly as the table above
states.

One diagnostic did not survive the withdrawal and is worth recovering later:
the retired guard's message named the **sibling-argument risk** explicitly —
that a sentinel in `description` means parameters such as `priority` may have
been silently dropped. That sentence converts the invisible vector-1 failure
into a visible one at the moment it happens, and `matched_pattern` plus a
200-character excerpt does not convey it. Folding it into
`markup_tripwire.build_markup_block` is a clean, self-contained follow-up.

---

## 5. Operator runbook — sweeping the corpus

> **Run this sweep BEFORE any further large consolidation pass.** Routine
> consolidation deletes corrupted entries as a merge side effect and silently
> destroys the specimens. Both 2026-07-27 instances (Mem0 `c759c53b` and
> `9f2d2ae6`) were lost exactly that way. Once a specimen is consolidated
> away, the incidence rate can no longer be measured.

**Captured runs.** `docs/toolcall-xml-leak-sweep-2026-08-05/` holds the first
authoritative live-corpus measurement (task 3567): an exhaustive walk of 21,080
points found **41 leak-carrying records (~0.19%)** — 1 `repairable_duplicate`,
40 `manual_review`, 0 `repairable_tail`. The verbatim report, a provenance
sidecar, and the per-record adjudication are committed there. Both 2026-07-27
specimens named above were already gone by then, so that directory is now the
corpus of record — and `dry-run-report.json` doubles as a ready-made regression
fixture of 41 real specimens with verified classifications.

From `fused-memory/`:

```bash
# 1. Dry run. This is the default and it mutates nothing.
#    The printed JSON report IS the investigation.
python scripts/sweep_toolcall_xml_leak.py --exhaustive

# 2. Read the report. Every record is classified as one of:
#      clean                 — no leak
#      repairable_tail       — fragment runs to end-of-content; removing it
#                              leaves non-empty text
#      repairable_duplicate  — the 9f2d2ae6 shape: text after the fragment is
#                              a verbatim duplicate of the text before it
#      manual_review         — carries a leak but matches neither shape
#
#    Review every `manual_review` entry BY HAND. They are never auto-mutated.

# 3. Apply. Repairs the confidently-classified records only.
python scripts/sweep_toolcall_xml_leak.py --apply --exhaustive
```

> **Do NOT run `--apply` from a sandboxed agent session.** Measured 2026-08-05
> (task 3567): the sandbox denies file *creation* under `~/.mem0`, so SQLite
> cannot create its rollback journal and mem0's history write fails with
> *"attempt to write a readonly database"* — while the Qdrant delete, being a
> network call to `localhost:6333`, succeeds. That splits delete-then-re-add
> exactly down the middle. One record was deleted from Qdrant with no re-add;
> its text survived only because the dry-run report had already been committed.
> The db file itself is mode `0644`, uid 1000, `W_OK=True` — this is an
> environment property, not a filesystem permissions defect, and not a sweep
> defect. Run `--apply` from an ordinary interactive shell.
>
> **This is now MACHINE-ENFORCED, not left to memory** (task 3686). Before it
> scans anything, an `--apply` run probes whether this process can actually
> create a file in mem0's history directory
> (`fused_memory.utils.store_mutation_preflight.assert_store_mutation_allowed`).
> If it cannot, the run refuses to start: nothing is scanned, nothing is
> mutated, and the refusal surfaces as `aborted: true` and exit 2. A dry run is
> deliberately **not** gated — it mutates nothing, so the classification report
> stays obtainable from anywhere.
>
> The general policy the guard encodes:
>
> > **Mutating memory operations go through the fused-memory MCP server — the
> > single, unsandboxed owner of the store. An in-sandbox script must never
> > mutate the shared store directly.**
>
> That works because `.mcp.json` declares fused-memory as a separate HTTP
> process (`http://127.0.0.1:8002/mcp`), so an MCP write executes outside the
> calling agent's landlock. Note the asymmetry that caused this incident is
> unfixable by sandbox configuration: landlock governs the **filesystem only**
> and can never block the Qdrant network delete, so no write-set widening could
> have made the two-phase mutation atomic. Adding `~/.mem0` to the sandbox
> write-set was considered and declined for that reason among others — see
> `docs/toolcall-xml-leak-sweep-2026-08-05/investigation.md` §"Decision (2)".
>
> **The one record lost to this has been recovered** (task 3686). `7d073281` was
> re-added via the MCP `add_memory` tool and lives under a new id; see
> `docs/toolcall-xml-leak-sweep-2026-08-05/recovery-tracking.json` for that id
> and how the write was verified. Do **not** re-run the sweep to recover a
> record — that is what the rest of this runbook warns against.

`--apply` exits **non-zero** if it left any `manual_review` record behind, so
a partial sweep can never be mistaken for a complete one, and likewise if the
scan was truncated (`--limit` reached), since it then covered an unknown
fraction of the corpus.

Repair is **delete + re-add**, never an in-place Qdrant payload `SET`. The
repaired text must be re-embedded; an in-place write would leave a stale vector
pointing at the corrupted string. The report records old id, new id, and both
before/after contents so every repair is auditable — the same mapping shape as
Stage 1's `2c47b5cb` → `0a4f4848`.

### What "content-preserving" means, precisely

The repair preserves two things: the memory **text** in full, and the **payload
metadata** that is the record's only metadata-scoped retrieval axis.

That second half is not decoration. `get_memories_by_metadata` and
`count_memories_by_metadata` match payload *keys* by equality via `MatchValue`,
so a repaired record that lost its `task_id` / `kind` / `x_*` keys would become
invisible to every metadata-scoped consumer that could previously find it — a
second, undisclosed mutation of stored state layered on the harness's first.

The carry-over rule is `payload keys - _MEM0_OWNED_KEYS`, where the owned set is
what mem0/Qdrant assign or re-derive on write (`id`, `hash`, the content keys,
`created_at`/`updated_at`, `user_id`/`agent_id`/`run_id`/`actor_id`/`role`, and
`category`, which the service overwrites anyway). The scope identities go back
in as **arguments** — `agent_id=` and `session_id=`, since mem0 writes its
`run_id` payload key from `Scope.session_id` — rather than being forged as
metadata. Every record in the report names its own `metadata_preserved` and
`metadata_dropped` key lists, so anything that could not be carried is *named*
rather than dropped silently.

### The sweep verifies persistence; it does not trust a returned response

`MemoryService.add_memory` **swallows a Mem0 write failure**: it catches
`Exception`, logs it, folds the failure into `AddMemoryResponse.message` as
`[mem0_error: ...]`, and returns *normally*. A returned response is therefore
not by itself evidence that anything was written. The sweep checks three
independent things before marking a record repaired — non-empty `memory_ids`,
`mem0` present in `stores_written`, and no `mem0_error` in `message` — and a
non-raising add that fails any of them is treated **identically to a throw**.

### The three non-zero exit conditions that need a human

| Record flag | What happened | What to do |
| --- | --- | --- |
| `content_lost_in_flight` | The delete landed but the re-add did **not** persist (raised, or returned without evidence of a mem0 write). The original text now exists **only in the printed JSON report**. | Restore it by hand from the report — it carries the old id, the original content, the repaired content, and `metadata_preserved` / `metadata_dropped` — **before** re-running the sweep. |
| `skipped_not_mem0_routed` | A repairable record whose `category` does not route to mem0 (or is absent/unrecognised). Left **entirely untouched**: nothing deleted, nothing added. | Needs a human decision. Neither option is safe unattended — a plain re-add would route the repaired text to Graphiti only and the Qdrant copy the delete removed would be gone, while `dual_write=True` would duplicate the Graphiti copy that the mem0-scoped delete deliberately left alive. |
| `record_error` | That record's repair aborted on an unexpected error (a `delete_memory` transport failure, a Qdrant outage). The sweep **continued** to the remaining records rather than unwinding. | Whether the delete landed is **unknown** — check the record's id in the store before re-running. The sweep deliberately does not guess. |

A worked example of that last row —
`docs/toolcall-xml-leak-sweep-2026-08-05/investigation.md` §4. There the delete
*had* landed, making it a `content_lost_in_flight` situation arriving under the
`record_error` flag. Two lessons generalise. First, **read the record flags, not
just the exit code**: exit 1 was overdetermined on that run (40 leftover
`manual_review` records *and* the `record_error`), so the exit code alone would
have hidden the mutation. Second, the id check in that cell is not optional — it
is what established the delete had landed, and it must be a **read-only** lookup.

### The report always survives, even a fatal abort

Because the only copy of a `content_lost_in_flight` record's original text is
the printed report, the report must never be discarded on the way out. Two
redundant mechanisms guarantee that:

- **Per-record isolation.** Each record is added to the report *before* any
  store mutation is attempted, and its repair runs under its own `try`. One
  record's transport error is recorded as `record_error` on that record and the
  sweep carries on; it can no longer void every earlier record's entry.
- **Caller-owned progress.** Should anything escape anyway, `main()` still holds
  the accumulated records and **prints the partial report** (with
  `"aborted": true`) before exiting `2`. The partial report uses exactly the
  same shape as a complete one, so there is no second format to drift out of
  step.

---

## 6. Residual-episode policy: redact, never cascade-delete

Graphiti episodes are a different store with a different transport; the
Qdrant sweep structurally cannot reach them. The known residual is episode
`d12b0eb4-f027-4d0c-a26c-096ccd0e75c2`.

**Do not use `delete_episode(cascade=True)` on it.** Cascade removes the
entities and edges exclusively sourced from that episode — which here includes
demonstrably-valid collateral such as edge `ea4072dc`. Those edges were
extracted from the *clean* portion of the text, before the truncation point;
the fragment contributed no facts. Deleting real knowledge to fix appearance
is strictly worse than the leak.

Use `redact_episode_content` instead. It sets **only** the `content` property
on **one** `Episodic` node, scoped by uuid *and* group_id. EpisodicNodes carry
no embedding of their own, so an in-place content set leaves no stale vector
behind (unlike the Mem0 path, which must delete-then-re-add to re-embed). It
refuses loudly on a blank replacement, on a replacement that itself still
carries a leak, and on an absent uuid — a typo can never read as a success.

Graphiti-side **discovery** (enumerating which episodes across a graph carry a
leak) is deliberately not provided: it needs a full episode scan with no
payload index to lean on. Tracked as follow-up work. `redact_episode_content`
is the remediation primitive for an episode you have already identified.

---

## 7. Standing caveat: the `MatchText` prefilter relies on un-indexed semantics

A read-only live probe (Qdrant 1.17.1, collection `fused_dark_factory`, 19,321
points) established that on an **un-indexed** payload field, `MatchText`
performs a **literal, case-sensitive, order-preserving substring** match —
mid-word hits, reversed-word-order misses, and case sensitivity together prove
`contains`, not tokenization. Today `data` is un-indexed: `payload_schema`
carries keyword indexes on `actor_id`/`agent_id`/`run_id`/`user_id` only, and
nothing in `src/`, `tests/`, or `scripts/` calls `create_payload_index`.

**If a text payload index is ever added to `data`, `MatchText` silently flips
to tokenized word-matching.** A silent semantic flip is precisely the failure
class this task exists to kill, so the design does not depend on it:

- The prefilter is only ever a **speed optimisation**. Every returned record is
  re-verified in Python by the shared detector, which is the authoritative
  verdict.
- The flip fails **safe**. Tokenized matching is strictly *more* permissive for
  these needles (text containing `&#60;parameter name=` necessarily contains
  the tokens `parameter` and `name`), so the prefilter remains a superset and
  the answer stays exact. Only speed degrades.
- `exhaustive=True` bypasses the prefilter entirely.

The tripwire is
`TestMem0BackendScanPayloadTextIntegration::test_matchtext_prefilter_is_a_literal_substring_match`
in `fused-memory/tests/test_mem0_client.py` — a `qdrant_skipif`-gated
integration test that fails **loudly** if the semantics ever change.

---

## 8. Known out of scope

`orchestrator/src/orchestrator/config.py:56-60` — `coerce_tier` silently maps
an unknown priority string to `'medium'` with no log, warning, or counter.
Different subsystem, and an intentional fail-safe, but the same silence
anti-pattern that turned this leak into a wrong value rather than an error.
Not addressed here.

---

## Related

- `docs/toolcall-xml-leak-sweep-2026-08-05/` (task 3567) — the first live-corpus
  sweep: verbatim report, provenance sidecar, and `investigation.md` carrying the
  incidence measurement, the hand-adjudication of all 40 `manual_review` records,
  and the partial-mutation incident that `--apply` hit from a sandboxed session
- `fused-memory/tests/test_toolcall_xml_leak_sweep_artifacts.py` — validates any
  committed sweep artifact by re-running `classify_record` over each record's own
  stored content; hermetic, so it does not need Qdrant up
- `fused-memory/src/fused_memory/utils/toolcall_xml_leak.py` — the detector,
  and the rationale for its deliberately conservative real-whitespace
  discriminator (which excludes the tasks 2938/2939 false-positive shape)
- `scripts/scan_task_toolcall_leaks.py` (task 2939) — the read-only Taskmaster
  task-DB sweep, where the detector was originally written and hardened
- `fused-memory/scripts/clear_malformed_empty_memory.py` — the cleanup-script
  template the sweep follows
- `docs/legibility/design-invariants.md` — `no-silent-fail-soft` and
  `structured-facts-at-failure`, both of which this leak violated end to end
