# Retrospective sweep: unverified completion claims in `temporal_facts` / `decisions_and_rationale`

**Task 3381 · esc-3085-1 second ask · swept 2026-08-15T11:13:35Z · dark-factory `64a9de0655`**

> **This artifact set replaces a 2026-08-12 run made with defective code.** The earlier
> sweep looked its derived-edge reader up by each record's Graphiti `group_id` against a
> dict keyed by the `--project` **graph names**, so an episode whose `group_id` differed
> from the graph it was read from got the wrong reader — or none at all — and the miss
> serialized as `derived_edge_uuids: []`, indistinguishable from a measured zero. Fixed in
> commits `b3b1ff99fb`, `dcfedcccd6`, `041b6277f1`, `994e6667f4`; the whole sweep was then
> re-run against the live store rather than the stale file being patched. What the fix
> actually changed is measured in *[What the re-run changed](#what-the-re-run-changed)* —
> it is **not** what the fix was expected to change.

Artifacts in this directory:

| file | what it is |
|---|---|
| `report.json` | byte-for-byte the sweep script's stdout, unedited |
| `provenance.json` | run metadata the report itself could not know |
| `investigation.md` | this file — the human adjudication |

---

## Headline

| measure | value |
|---|---|
| Episodes scanned | **7595** (dark_factory 3042 + reify 4553) |
| Episodes carrying ≥1 completion claim | **1653** |
| Raw `mismatch` verdicts | **167** |
| Raw `unverifiable` verdicts | **30** |
| Findings whose edges were **not** enumerated | **0** |
| **Confirmed new fabrications of the esc-3085-1 shape** | **0** |

`mismatch` and `unverifiable` are different facts and are **never summed**, here or in
`report.json`.

**The "two in 17 hours is a rate" hypothesis was NOT borne out — and the corrected edge
data does not change that.** This was re-checked rather than assumed: the fix altered only
the `derived_edge_uuids` column, which is *evidence about blast radius*, not about whether a
claim is true. Every verdict is produced by `verify_claims` from the three authority probes,
which the fix did not touch. The one finding that disappeared and the two that appeared are
adjudicated in *[What the re-run changed](#what-the-re-run-changed)*, and none is a
fabrication. The conclusion is unchanged, and that is asserted, not assumed.

Read the *Denominator* and *Caveats* sections before treating this as "the corpus is
clean" — this sweep is biased toward under-counting by construction.

The 167 raw mismatches are the **script's verdicts, not adjudicated fabrications**. All 167
fall into the three systematic false-positive classes below. `report.json` is committed
unedited so this adjudication can be re-checked against the raw output.

### Breakdown as reported

By category (all 197 findings): `decisions_and_rationale` 112 · `temporal_facts` 85
By claimed project: `dark_factory` 125 · `reify` 57 · `know_live` 10 · `knowlive` 4 · `solar_challenge_platform` 1
By graph actually swept: `dark_factory` 122 · `reify` 75
By subject: `task` 162 · `commit` 35 · **`ticket` 0**

`project_id` (what the record claims about itself) and `graph_name` (which graph was
queried) are **now reported separately on every finding**, because they disagree for 21 of
the 197 — and that disagreement is precisely what the old code got wrong.

**Zero ticket findings is a real negative.** The ticket store was available and initialized
for this run (`provenance.json` → `authorities_resolved`), so the absence is not an
unavailability artifact. esc-3085-1 instance (2)'s exact shape — a completion claim naming
a `tkt_` id the registry does not hold — **did not recur anywhere in the swept corpus.**

---

## What the re-run changed

The plan for the fix expected the pre-fix report to **understate** the graph. Measured
finding-by-finding on `(record_uuid, claim_kind, subject, ref)`, that is not what happened.
The pre-fix error ran in **both** directions.

### Edge column: 5 findings LOST their edges, 0 gained any

195 findings are common to both runs. Five of them changed `derived_edge_uuids`, and **all
five went from a non-empty list (6–11 edges each) to an empty one. None went the other
way.**

All five are episodes **read from the `reify` graph whose own `group_id` is
`dark_factory`**. The old lookup keyed on `group_id`, so it queried the *dark_factory*
graph — and found edges there. Verified directly over `GRAPH.RO_QUERY` on 2026-08-15 for
three of the five:

| episode | Episodic node in `dark_factory` | node in `reify` | `RELATES_TO` edges in `dark_factory` | edges in `reify` |
|---|---|---|---|---|
| `a887c958-0018-4715-8817-cf048c187e8d` | 0 | 1 | **8** | 0 |
| `52a37f90-1f21-462f-96f6-9bf1774f3c32` | 0 | 1 | **11** | 0 |
| `0c5b3f7b-38e1-4436-baa6-4d8419f88645` | 0 | 1 | **7** | 0 |

**So an episode's derived edges can live in a different graph from the episode node.** The
corrected sweep enumerates edges only in the graph the episode was read from — a correct
and predictable rule, and the one the reader lookup must follow — but it means an empty
list means *"none in **this** graph"*, not *"none anywhere"*. Those 8 / 11 / 7 edges still
exist; this report no longer names them.

That bound is now shipped as a first-class `CAVEATS` entry (Caveat 6 below) naming
`a887c958-…` as the worked example, so a reader of `report.json` alone cannot miss it.
Actually enumerating edges across graphs is **filed as follow-up work, not built here** — it
is new behaviour beyond this task's plan, and guessing at it would be exactly the kind of
unverified reach this task exists to audit.

### The 15 previously-unqueried findings are now genuinely queried

Every finding whose `project_id` is not one of the swept graph names — `know_live` 10,
`knowlive` 4, `solar_challenge_platform` 1 — still shows `[]`. **The value is unchanged;
its epistemic status is not.** Before the fix those were never asked. Now they were asked,
and the answer is a measured zero. `summary.edges_unqueried` is **0** for this run and the
sweep emitted no *"NOT enumerated"* warning — the machine-checkable statement that nothing
went un-enumerated. Had any reader failed to resolve, those findings would now carry
`derived_edge_uuids: null`, which no consumer can misread as a measured zero.

### Verdict delta: 1 dropped, 2 added — none of them a fabrication

The corpus is live and grew 7577 → 7595 episodes over the three days between runs, so some
delta is drift rather than the fix.

**Dropped — `e64dbe6f-5df7-43a3-be1f-398ce55278cf`, dark_factory task 3584.** The earlier
run observed `pending`. Read from `tasks.db` on 2026-08-15, task 3584 is now **`cancelled`**
— which is in `shared.task_statuses.TERMINAL`, so the claim now *verifies* and the finding
vanished. **This is Caveat 2 firing live and measured**, not a hypothetical: work that was
*cancelled* — i.e. never landed — reads as verified by this sweep. It is the sharpest
concrete evidence in this whole document that the under-count is real.

**Added — both episodes written on 2026-08-12, after the earlier run; both Class A.**

| episode | claim | claimed project | observed | why it is a false positive |
|---|---|---|---|---|
| `4c7a7e58-858c-4447-acc1-7c9f24a220a1` | `filing_dispatch`, task 4213 | `dark_factory` | `pending` | *"prevention … **filed as** dark_factory task 4213"* — the task exists; a filing claim's truth condition is existence, not completion |
| `5d84c539-74de-4eb3-b6dd-b53535199756` | `filing_dispatch`, task 6098 | `reify` | `pending` | *"Task 6098 **is dispatched** consistent with gate-6141's ratified option (a)"* — a dispatch statement about a task that exists |

Neither asserts that the named work is finished. **Conclusion unchanged: 0 confirmed new
fabrications.**

---

## Adjudication of all 167 mismatches

Class totals shift by exactly the delta above — Class A +2, Class B −1, Class C unchanged —
and still account for every mismatch: **127 + 10 + 30 = 167**. The commit-subject population
is byte-identical between the two runs (35 findings, 30 of them `mismatch`), so Class C's
out-of-band adjudication carries over unchanged and was not re-derived.

### Class A — `filing_dispatch` about a task that EXISTS but is not terminal — **127 findings**

Observed states in the earlier run's 125: 64 `pending`, 52 `deferred`, 7 `in-progress`,
1 `merge-deferred`, 1 `blocked`. The two findings added since (table above) are both
`pending`.

`_verify_task` (completion_claim_gate.py:550) verifies **only** a TERMINAL status, for every
claim kind. But a `filing_dispatch` claim asserts the work *"was filed/queued/cancelled
somewhere"* (:292) — its truth condition is that the artifact **exists**, not that it is
finished. A task reported `pending` is positive evidence the filing claim was TRUE.

The gate's own ticket path already adjudicates filing this way: `_verify_ticket` (:578)
returns `verified` when the row merely EXISTS, whatever its status. The task path does not
make that distinction, so every truthful "filed as task N" about still-open work is flagged.

Examples (full listing in `report.json`):

| claim | project | observed | episode | derived edges |
|---|---|---|---|---|
| task 4136 | reify | deferred | `1879d86f-6ec2-43d6-a5a8-88e6e445a068` | 7 |
| task 4263 | reify | pending | `60f904c8-4354-453a-9313-9555bc77b33b` | 4 |
| task 4746 | reify | pending | `994dec4c-5024-4afd-8afd-a8f66330e3ef` | 4 |

> "The user-observable signal was NOT dropped — it was **refiled as task 4263**"
> "To close the hex/wedge Phase A wiring gap, **filed task 4746** (…)"

Both statements are true, and both tasks exist. **Verdict: false positives.**

### Class B — `applied_work` about a task, prose false positives — **10 findings**

These required reading each episode. None asserts that the named task is complete; the task
id merely co-occurs with completion-ish phrasing.

| task | project | observed | episode | what the text actually says |
|---|---|---|---|---|
| 5623 | reify | in-progress | `1554dafd` | *"task 5623 has landed nothing (still pending)"* — asserts the **opposite** |
| 3267 | dark_factory | pending | `181595f7` | *"**filed** dark-factory task 3267 … as the durable fix"* — a filing statement |
| 3846 | dark_factory | pending | `448d8400` | *"task 3846 **was filed** to give recon's stranded-in-progress finding a landed-on-main check"* |
| 5208 | reify | pending | `2df7dda7` | *"…once 5208 **lands**"* — explicitly future |
| 5317 | reify | pending | `1eefbd1e` | *"task 5317 … **was EXPANDED** to also own …"* — a scope change |
| 4687 | reify | deferred | `b6153503` | the capability *manifest* landed; 4687 is the decompose task |
| 6164 | reify | pending | `3fde06f9` | the real claim is the commit `9ca5c6ad9f`, not the task |
| 3541, 3578, 3539 | dark_factory | pending | `56052308`, `baaebac9`, `448d8400` | prose about what tasks *own* or *assert*, not completion |

The eleventh member of this class in the earlier run — task 3584, episode `e64dbe6f` — is
**no longer a finding at all**: task 3584 has since gone `cancelled`, which is terminal, so
the claim now verifies. See *[Verdict delta](#verdict-delta-1-dropped-2-added--none-of-them-a-fabrication)*.
It was a false positive then and it is invisible now, for two unrelated reasons.

The `5623` case is the sharpest: the sentence states the task landed nothing and is still
pending, and the extractor flagged it as a completion claim. The negation stripper does not
reach the construction *"has landed nothing"*. **Verdict: false positives.**

### Class C — commit claims — **30 findings**

Checked out of band against both known checkouts with `git cat-file -e <sha>^{commit}`:

| outcome | count |
|---|---|
| Commit **PRESENT** in `dark_factory`, but the sweep probed `reify` | **12** |
| Absent from both known checkouts | 18 |

**The 12 are confirmed false positives, and they are the most instructive finding here.**
The claims are true; the sweep searched the wrong repository. The prose names its repo
explicitly and the probe ignored it:

| sha | probed in | actually present in | claimed text |
|---|---|---|---|
| `5353d209e4` | reify | **dark_factory** | "LANDED on **dark-factory** main (commit 5353d209e4…)" |
| `8face929e4b6…` | reify | **dark_factory** | "already landed via **dark_factory** task 2053 (done, merged commit …)" |
| `ba3626d5` | reify | **dark_factory** | "**dark_factory** task 2879 (…done via commit ba3626d5)" |
| `d9ea04876f` | reify | **dark_factory** | "**dark-factory** commit d9ea04876f merged as be8b778aa9" |
| `72ddf7d1e268…` (×2) | reify | **dark_factory** | cross-project rerouting note naming `dark_factory` |
| `6f29517823`, `4287de86d23bf` (×2), `a05549ab8597…`, `ed251c9b586b…`, `a51f8e4d5f` | reify | **dark_factory** | — |

A reify agent recording a fact about a dark-factory commit has its claim attributed to
`reify` (the writer's project), and the probe then searches reify's object store. The
commit is absent there, so the gate reports `no commit X in project 'reify'` — a
**fabrication accusation against a truthful writer**.

The remaining **18** are absent from both checkouts and are **indeterminate, not confirmed
fabrications**. The benign explanations dominate: dark-factory's merge lane rebases, so a
pre-merge task-branch sha cited in an episode is garbage-collected after the merge (several
of these texts cite exactly that — *"merged as …"*, *"Branch task/3135 tip commit …"*), and
some name a third repository this sweep never opened (`dark-factory/taskmaster-ai` for
`82dccbfa`). Two more (`42c08a8e60`, `f12ea587c5`) come from an episode *narrating* a
mistaken landing belief — meta-discussion about a fabrication, not one. Adjudicating the 18
would require per-sha reflog/third-repo archaeology; none is actionable as written.

---

## How far the denominator can be trusted

- **The corpus is live and the denominator moved during the investigation.** The
  `dark_factory` graph grew 2995 → 3030 Episodic nodes over ~2.5h during the earlier run's
  measurements, and 3030 → 3042 over the three days since (`provenance.json`). 7595 is a
  snapshot at run time, not a fixed population.
- **The sweep systematically under-counts** (Caveat 1). Authorities are read *today*, not at
  write time: a claim written while its task was in-progress verifies now that the task has
  gone terminal. Most of the corpus predates this run by weeks, so this bias is large and
  one-directional. **A low mismatch count is not evidence of a clean corpus.** Task 3584
  going `cancelled` between the two runs — and its finding vanishing as a result — is this
  bias caught in the act, measured rather than argued.
- **Detection needs a named ref** (Caveat 8). A fabricated completion phrased without a task
  id / sha / `tkt_` id is invisible here by construction.
- **Graphiti episodes only** (Caveats 4-5). Mem0-resident records, orphaned edges whose
  source episode was deleted, and `add_episode` writes carrying a caller-supplied
  description (hence no category) are all outside scope.
- **Edges are enumerated per graph** (Caveat 6). `derived_edge_uuids: []` means "none in the
  graph this episode was read from", not "none anywhere" — measured, with a worked example,
  in *[What the re-run changed](#what-the-re-run-changed)*. A `null` there would mean the
  edges were never enumerated at all; this run has **0** such findings.
- **The originating incident's own record is in the corpus and this sweep does not flag it.**
  See below — it is the sharpest available demonstration that the under-count is real
  rather than theoretical.

### The esc-3085-1 instance-(2) record was swept and came back clean

Observed 2026-08-12 at dark-factory `a138751e80`, and unchanged in the 2026-08-15 re-run:

- Episode `02090224-7bc9-4485-9291-6748e1042ac9` is **still present**, in the **reify**
  graph, with `source_description = add_memory:temporal_facts` — in scope for this sweep,
  and carrying **no** `[unverified_claim]` prefix (it predates the task-3142 gate).
- It appears in **none** of this sweep's 197 findings.
- Ticket `tkt_0RRRC5AASJ9Z630VP4PCN9H376` — the ref that record claims — **does exist** in
  `data/reconciliation/tickets.db`: `status='combined'`, `task_id='5638'`,
  `created_at=2026-07-27T08:30:56Z`, `resolved_at=2026-07-27T08:31:29Z`, and
  `reason` beginning *"drop: Candidate is fully covered by pool Task 5638 …"*.

So the sweep behaved **correctly**: `_verify_ticket` accepts a row that exists, the row
exists, and the claim verifies. The ticket was not invented.

*Hypothesis (not established here):* the residual falsehood in that record is not the
ticket's existence but its **disposition**. The candidate was *dropped* as a duplicate of
reify task 5638, and `task_id` on the row is `5638` — a **reify** task — so the record's
assertion that the work was *"re-filed into dark_factory's task tree"* did not
materialize as a dark_factory task. An existence-only ticket check cannot see that
distinction, so **the current write-time gate would not catch esc-3085-1 instance (2)
either.** Confirming this would require reading esc-3085-1 itself, which is outside this
task's scope; it is carried into the follow-up rather than asserted here.

## Caveats (reproduced verbatim from `report.json`)

1. **RETROSPECTIVE BIAS, FALSE NEGATIVES:** the write-time gate reads each authority AT WRITE TIME; this sweep reads it TODAY. A claim written while its task was still in-progress VERIFIES today if that task has since gone terminal, so the sweep systematically under-counts. A low mismatch count is therefore NOT evidence the corpus is clean.
2. **RETROSPECTIVE BIAS, FALSE POSITIVES:** 'cancelled' is in shared.task_statuses.TERMINAL, so a claim about work that was later CANCELLED reads as verified here even though the work never landed.
3. **UNVERIFIABLE INDICTS NOBODY:** a task deleted since, a garbage-collected commit, or an expired ticket all read as unverifiable through no fault of the writer. 'unverifiable' and 'mismatch' are different facts and are never summed in this report.
4. **COVERAGE, STORE:** this sweep covers GRAPHITI EPISODES only. A MEM0-resident record is NOT covered, because the two categories in scope are Graphiti-primary and Mem0 is not read at all.
5. **COVERAGE, ORPHANED EDGES:** a derived RELATES_TO edge whose source episode has been deleted is NOT covered, and neither is an episode ingested through add_episode with a caller-supplied description, which carries no category and so falls outside the category scope.
6. **COVERAGE, CROSS-GRAPH EDGES:** derived_edge_uuids is enumerated ONLY in the graph the episode was READ FROM (the finding's graph_name). An episode whose group_id differs from that graph — see any finding where those two fields disagree — may have its derived RELATES_TO edges in ANOTHER graph, where they are NOT counted here. Measured, not hypothetical: episode a887c958-0018-4715-8817-cf048c187e8d exists only in the reify graph with group_id dark_factory, and its 8 derived edges exist only in the dark_factory graph. An empty list therefore means "none in THIS graph", not "none anywhere".
7. **EDGES NOT ENUMERATED:** a finding whose derived_edge_uuids is null (and whose edges_unqueried is true) had its RELATES_TO edges NOT enumerated — no reader resolved for its graph, or the edge query failed. Its harm artefacts are UNKNOWN, not absent. An empty list means the opposite: the store WAS asked and returned nothing.
8. **DETECTION BOUND:** claims are detected by the deterministic lexical vocabulary in fused_memory.services.completion_claim_gate, which requires completion PHRASING and a concrete NAMED REF (task id / commit sha / tkt_ id) to co-occur in one clause. A fabricated completion phrased without a named ref is invisible to this sweep by construction.

---

## What this found instead

The sweep's real yield is not a list of fabrications — it is **three reproducible
false-positive classes in the write-time gate itself** (task 3142), each of which causes a
truthful episode to be stamped `[unverified_claim]` in production today:

1. **Filing claims are held to a completion standard.** `_verify_task` requires TERMINAL for
   every claim kind, so a truthful *"filed as task N"* about still-open work is a mismatch.
   `_verify_ticket` already treats existence as sufficient for the same kind of claim.
   Largest class here: **127 of 167**.
2. **Cross-repo commit claims are probed against the wrong repository.** A claim is
   attributed to the writer's project even when the prose names another repo, so the sha is
   sought in the wrong object store. **12 confirmed** — each one a false accusation against
   a writer who was telling the truth.
3. **The negation stripper misses at least one construction.** *"task 5623 has landed
   nothing (still pending)"* was extracted as a completion claim.
4. **A ticket check that tests only existence is necessary but not sufficient.** The
   originating incident's own ticket exists, was `combined`/dropped, and therefore verifies
   — so the gate that exists today would very likely not have caught the record that
   motivated building it (evidence and hypothesis above).

This matters more than the null result: on the write path these misfire *now*, on every new
episode, and the tag they stamp is the very signal a future sweep would trust. Classes 1 and
2 would each have produced far more noise than signal had this sweep been run as a gate.

Filed as follow-up work rather than fixed here — this task's scope is a read-only report,
and the fixes belong in the gate module.

## Invariant

**This sweep invalidated nothing and deleted nothing.** Every episode and edge named above
is left exactly as it was, for human adjudication. The script has no `--apply`,
`--invalidate` or `--delete` path — an absence pinned by `TestReadOnlyByConstruction` — and
reads exclusively over `GRAPH.RO_QUERY`, so read-only is server-enforced rather than
client-promised.

Given the adjudication above, **no edge or episode in this corpus is recommended for
invalidation.** Acting on the raw 167 would have destroyed truthful records.

That holds for the re-run too. Correcting the reader lookup changed only *which edges this
report names*; it invalidated nothing, deleted nothing, and moved nothing between graphs.
The 8 / 11 / 7 cross-graph edges this report stopped naming are still in the `dark_factory`
graph, untouched — they are no longer *listed*, not no longer *present*.

## Reproducing

```bash
cd fused-memory
PROJECT_ROOT=/home/leo/src/dark-factory \
DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory,/home/leo/src/reify \
RECONCILIATION_DATA_DIR=/home/leo/src/dark-factory/data/reconciliation \
uv run python scripts/audit_unverified_completion_claims.py \
    --project dark_factory --project reify --include-unverifiable
```

The three env vars are **required**, not optional. Without them the project registry
resolves empty and the ticket store is not found, and every task and ticket claim falls to
`unverifiable` — see `provenance.json` → `notes` for the discarded first run that did
exactly that. The script warns loudly on stderr in that case rather than failing silently,
which is how the mis-scoped run was caught.
