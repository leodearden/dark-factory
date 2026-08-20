# Retrospective sweep: unverified completion claims in `temporal_facts` / `decisions_and_rationale`

**Task 4230 · esc-3085-1 second ask · swept 2026-08-20T09:08:17Z · dark-factory `b543435a48`**

> **This artifact set is the fourth run, and it replaces three earlier ones.** A 2026-08-12 run
> looked its derived-edge reader up by each record's Graphiti `group_id` against a dict keyed
> by the `--project` **graph names**, so an episode whose `group_id` differed from the graph
> it was read from got the wrong reader — or none at all — and the miss serialized as
> `derived_edge_uuids: []`, indistinguishable from a measured zero (fixed in `b3b1ff99fb`,
> `dcfedcccd6`, `041b6277f1`, `994e6667f4`). The 2026-08-15T11:13 run that replaced it was
> correct but **narrow**: it enumerated edges only in the graph each episode was read from, so
> the entire cross-graph population shipped with an empty harm-artefact column — including
> five findings whose edges are real and live in the other graph. The 2026-08-15T16:45 run
> widened that to the record's own `group_id` graph as well, and recovered them — but only
> when that second graph happened to be swept, which left 15 of the 21 cross-graph findings
> still asked in exactly **one** graph. **This run (task 4230) keys enumeration on the episode
> uuid across EVERY swept graph**, so all 204 findings are asked in both, and ships two new
> columns — `derived_edges_by_graph` (which graph held each edge) and `edges_unqueried_in`
> (which targeted graphs did not answer). Measured in
> *[What the re-run changed](#what-the-re-run-changed)*. Each time the whole sweep was re-run
> against the live store; no stale file was ever patched to look fixed.

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
| Episodes scanned | **7693** (dark_factory 3105 + reify 4588) |
| Episodes carrying ≥1 completion claim | **1687** |
| Raw `mismatch` verdicts | **172** |
| Raw `unverifiable` verdicts | **32** |
| Findings whose edges were **not** enumerated | **0** |
| Findings whose edges were **partially** enumerated | **0** |
| **Confirmed new fabrications of the esc-3085-1 shape** | **0** |

`mismatch` and `unverifiable` are different facts and are **never summed**, here or in
`report.json`.

**The "two in 17 hours is a rate" hypothesis was NOT borne out — and no round of corrected
edge data changes that.** Re-checked rather than assumed, three times: every fix altered only
the edge columns, which are *evidence about blast radius*, not about whether a claim is true.
Every verdict comes from `verify_claims` via the three authority probes, which no fix
touched. This run's finding set does differ from the previous one — 204 vs 197 — but
**entirely because the corpus moved**, not because the rule changed: of the 184 findings
common to both runs, **zero** flipped between `mismatch` and `unverifiable`. The 20 added and
13 dropped are adjudicated in *[Verdict delta](#verdict-delta-vs-1645-20-added-13-dropped--all-corpus-movement)*
below. The conclusion is unchanged, and that is measured, not assumed.

Read the *Denominator* and *Caveats* sections before treating this as "the corpus is
clean" — this sweep is biased toward under-counting by construction.

The 172 raw mismatches are the **script's verdicts, not adjudicated fabrications**. All 172
fall into the three systematic false-positive classes below — re-derived against *this*
report, not carried over. `report.json` is committed unedited so this adjudication can be
re-checked against the raw output.

### Breakdown as reported

By category (all 204 findings): `decisions_and_rationale` 119 · `temporal_facts` 85
By record `group_id` (`summary.by_project`): `dark_factory` 120 · `reify` 69 · `know_live` 10 · `knowlive` 4 · `solar_challenge_platform` 1
By graph actually swept (`summary.by_graph`): `dark_factory` 117 · `reify` 87
By subject: `task` 169 · `commit` 35 · **`ticket` 0**

`project_id` (the record's own Graphiti `group_id`) and `graph_name` (which graph was
queried) are **reported separately on every finding**, because they disagree for 21 of the
204 — and that disagreement is precisely what the old code got wrong. Both tallies are now
in `summary`, so neither has to be recomputed by hand from the findings list; the 21
disagreeing rows also carry a derived `cross_graph: true`. (Note the label: `by_project` is
the record's `group_id`, **not** `claim_project_id` — the project a claim is *about* is a
different field on every row.)

**Zero ticket findings is a real negative.** The ticket store was readable for this run —
opened `mode=ro`, queried, no unavailability warning (`provenance.json` →
`authorities_resolved`) — so the absence is not an unavailability artifact. esc-3085-1 instance (2)'s exact shape — a completion claim naming
a `tkt_` id the registry does not hold — **did not recur anywhere in the swept corpus.**

---

## What the re-run changed

Three things happened here, and they are kept apart deliberately. The **2026-08-12 → 11:13**
re-run fixed a reader keyed on the wrong field; its measurement is preserved below because
it is what exposed the cross-graph fact. The **11:13 → 16:45** re-run then acted on that
fact. The **16:45 → 2026-08-20** re-run generalized it from a two-graph special case to a
rule. All three were measured finding-by-finding on `(record_uuid, claim_kind, subject, ref)`;
none changed a single verdict on a finding common to both sides of it.

### 2026-08-20 vs 16:45 — the rule generalized to every swept graph

The 16:45 rule asked a finding's read-from graph **and** its own `group_id` graph, the latter
only when that graph was swept. That "home pair" cannot reach a **third** swept graph at all,
so 15 of the 21 cross-graph findings — the ones whose `group_id` names an **unswept** graph
(`know_live` 10, `knowlive` 4, `solar_challenge_platform` 1) — were asked in exactly one
graph, and their empty edge lists were a measured zero over *half* the swept corpus rather
than over all of it. This run keys enumeration on the **episode uuid** across **every** swept
graph.

**The machine-checkable statement:** all **204** findings now carry
`edges_enumerated_in == ["dark_factory", "reify"]` and `edges_unqueried_in == []`. Under the
superseded rule the distribution was `(dark_factory)` 122 · `(reify)` 69 ·
`(reify, dark_factory)` 6 — i.e. **191 of 197 findings were asked in exactly one graph**.
`summary.edges_unqueried` and `summary.edges_partial` are both **0**, and the sweep emitted
no *"NOT enumerated"* and no *"scan failed"* warning.

**What the widening recovered: nothing — and that is the honest result.** Of the 184 findings
common to both runs, **183 have an identical edge set and 0 grew**. The 15 findings that were
asked in a second graph for the first time were answered, and that graph genuinely holds no
edges for them. Their empty lists are now measured over **both** swept graphs instead of one;
that is what the change bought, not a larger edge count. The 6 `reify`←`dark_factory`
findings keep the 5-with-edges / 1-empty split the previous run already had, because that
population was the only one the old rule already asked in both graphs.

**The one edge set that SHRANK is not this change — it is the corpus, and it was verified
rather than assumed.** Finding (`16cc46d8-cf33-4557-93ce-f46ceb900487`, `filing_dispatch`,
task 3910) went from 5 edge uuids to 4, losing `f2dad7cc-15a0-4c3c-8962-06c0617d1eb8`.
Re-measured directly over `GRAPH.RO_QUERY` on 2026-08-20: that uuid returns **count 0 in both
graphs**, and the script's own join (`MATCH ()-[r:RELATES_TO]->() WHERE '16cc46d8-…' IN
r.episodes`) returns exactly the 4 uuids the new report names. Widening the set of graphs
asked can only *add* edges, so it cannot subtract one that still exists — the edge was
deleted from the live store in the ~4.7 days since.

**Two new columns ship with the rule.** `derived_edges_by_graph` maps each **answering**
graph to the edge uuids *it* held (sorted, `[]` for a graph that answered holding nothing,
`null` — never `{}` — when no graph answered); `edges_unqueried_in` names the targeted graphs
whose scan did not answer, always present so `[]` reads as "no hole" rather than as an absent
key. `summary.edges_partial` is the denominator of the third state between fully-enumerated
and not-enumerated, and is disjoint from `summary.edges_unqueried` by construction.
`derived_edge_uuids` is unchanged in meaning — the flat sorted union — so existing consumers
are unaffected. Worked example from this run's own report: finding
`f76ba492-fd19-460b-8983-44d0ea0dd80c` (read from `reify`, `group_id` `dark_factory`) carries
`{"dark_factory": [9 uuids], "reify": []}` — the node lives in `reify` and **every one of its
edges lives in `dark_factory`**. That split was invisible in the superseded artifact, which
reported only the merged 9-uuid union.

### 16:45 vs 11:13 — the 5 lost findings got their edges back

**197 findings common, 0 added, 0 dropped.** 172 findings show a different
`derived_edge_uuids` *value*, but **167 of those changed order only** — edge uuids are now
sorted so the artifact is byte-stable across runs — and their edge **sets** are identical.
Only **5** changed as sets, and each is a strict **superset** of before: 6, 7, 8, 9 and 11
edges recovered where the earlier run showed `[]`. Across the whole report: **1116 → 1157**
edge uuids named, **176 → 181** findings carrying at least one edge. Nothing was lost.

Those 5 are exactly the `reify`←`dark_factory` episodes the 11:13 run had emptied.

### Why they were empty: an episode's edges can live in another graph

195 findings were common to the 2026-08-12 and 11:13 runs. Five changed
`derived_edge_uuids`, and **all five went from a non-empty list (6–11 edges each) to an
empty one** — the opposite of the "the pre-fix report understates the graph" expectation.

All five are episodes **read from the `reify` graph whose own `group_id` is
`dark_factory`**. The 2026-08-12 lookup keyed on `group_id`, so — by accident — it queried
the *dark_factory* graph, and found edges there. Verified directly over `GRAPH.RO_QUERY` on
2026-08-15 for three of the five:

| episode | Episodic node in `dark_factory` | node in `reify` | `RELATES_TO` edges in `dark_factory` | edges in `reify` |
|---|---|---|---|---|
| `a887c958-0018-4715-8817-cf048c187e8d` | 0 | 1 | **8** | 0 |
| `52a37f90-1f21-462f-96f6-9bf1774f3c32` | 0 | 1 | **11** | 0 |
| `0c5b3f7b-38e1-4436-baa6-4d8419f88645` | 0 | 1 | **7** | 0 |

**So an episode's derived edges can live in a different graph from the episode node** — and,
the 2026-08-20 run establishes, in a graph that is *neither* the node's graph nor its
`group_id` graph. That is why the rule is no longer keyed on any graph at all: enumeration
keys on the **episode uuid** and asks **every swept `--project` graph**. `edges_enumerated_in`
on every finding names which graphs actually answered — `(dark_factory, reify)` for all 204 —
and `derived_edges_by_graph` names which graph held each edge, so the coverage *and* the
attribution of each row are stated on the row rather than inferred. Caveat 6 is rewritten to
describe what is now enumerated, still with `a887c958-…` as the worked example.

### 16 findings still show `[]` — and it is now a zero measured over BOTH graphs

Still 16 of the 21 cross-graph findings, but the *reason* has changed and is now weaker in
exactly the right way — "its `group_id` graph was not swept" no longer implies "only one
graph was asked":

- **15** name a `group_id` graph this run never swept (`know_live` 10, `knowlive` 4,
  `solar_challenge_platform` 1). Under the old rule these were asked **only** in their
  read-from graph. They are now asked in **both** swept graphs, and neither holds any edge
  for them. Edges could still live in the unswept graph each one names; that would need those
  graphs added to `--project`, and that residual bound is Caveat 6, not a silent gap.
- **1** is a `reify`←`dark_factory` finding where both graphs answered with nothing — the
  one case whose reading is unchanged from the previous run.

`summary.edges_unqueried` and `summary.edges_partial` are both **0** and the sweep emitted no
*"NOT enumerated"* and no *"scan failed"* warning — the machine-checkable statement that no
finding went un-enumerated **or partially enumerated**. Had a scan raised, those findings
would carry `derived_edge_uuids: null` and `derived_edges_by_graph: null` with
`edges_unqueried: true`; had only *some* graph raised, they would carry a non-empty
`edges_unqueried_in` marking the list as a measured **lower bound**. Neither is a state a
consumer can misread as a measured zero.

### Verdict delta vs 16:45: 20 added, 13 dropped — all corpus movement

**184 findings common, 20 added, 13 dropped**, and **zero** of the 184 flipped between
`mismatch` and `unverifiable`. The corpus grew 7596 → 7693 Episodic nodes (`dark_factory`
3043 → 3105, `reify` 4553 → 4588) over the ~4.7 days between runs. None of this delta is
attributable to the enumeration rule, which touches only the edge columns.

**The 13 dropped are Caveat 1 firing live, verified individually** against the claimed
project's `tasks.db` over a `mode=ro` URI on 2026-08-20: **every one of the 13 refs is now
`done`**, so the claim verifies today and is correctly no longer a finding — `dark_factory`
3622, 3980, 4027, 4181, 4183, 4184, 4185, 4186, 4187, 4190, 4192 and 3381 (this sweep's own
predecessor task), plus `reify` 5623. Nothing was suppressed: the sweep reads each authority
*today*, so a claim whose task has since gone terminal stops being a mismatch by
construction. Twelve are Class A; the thirteenth (`reify` 5623, episode `1554dafd`) was the
sharpest member of Class B — *"task 5623 has landed nothing (still pending)"* — and is now
invisible for a reason unrelated to why it was a false positive.

**The 20 added were adjudicated, not counted.** Each was re-checked against the **claimed**
project (`claim_project_id`, not the record's own `group_id`), and the live status agrees
with the sweep's `observed` in every case:

- **18 mismatches** on episodes created 2026-08-12…08-19 whose claimed task is still
  non-terminal — 13 Class A (`reify` 6346, 6184, 6343, 6344, 6098, 6335, 6336, 5480 and
  `dark_factory` 3690, 4377, 4371, 4361, 3639) and 5 Class B (`reify` 5467, 6077 ×2, 5480 and
  `dark_factory` 3589).
- **2 unverifiable** on *older* episodes (`c8be5801`, created 2026-06-25, ref
  `dark_factory:4778`; `ca2820dd`, created 2026-07-28, ref `dark_factory:5493`). Both refs
  are now **ABSENT** from `dark_factory`'s `tasks.db`, and neither episode was a finding on
  2026-08-15. *Hypothesis:* the task records were removed from the tree since, flipping those
  claims from verified to unverifiable — the mechanism Caveat 3 states in as many words ("a
  task deleted since").

**No new false-positive class is needed for any of the 20.** The historical delta below, from
2026-08-12 to 11:13, is kept because it is this document's sharpest live evidence of the
under-count bias.

### Historical: the 2026-08-12 → 11:13 delta — 1 dropped / 2 added

Over this older interval the corpus grew 7577 → 7595.

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

## Adjudication of all 172 mismatches

**Re-derived against this report, not carried over.** Class totals shift by exactly the
verdict delta above — Class A 127 → **128** (+13 added, −12 dropped), Class B 10 → **14**
(+5 added, −1 dropped), Class C **30** unchanged — and still account for every mismatch:
**128 + 14 + 30 = 172**. The commit-subject population is byte-identical across all four runs
(35 findings, 30 of them `mismatch`, 5 `unverifiable`), so Class C's out-of-band adjudication
carries over unchanged and was not re-derived.

### Class A — `filing_dispatch` about a task that EXISTS but is not terminal — **128 findings**

Observed states in *this* run's 128, counted from `report.json`: 64 `pending`,
52 `deferred`, 8 `in-progress`, 3 `blocked`, 1 `merge-deferred`. (The 16:45 run recorded
64/52/7/1/1 for 125 of its 127, the remaining 2 being `pending`.) **Not one is terminal**,
which is the class's whole point.

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

### Class B — `applied_work` about a task, prose false positives — **14 findings**

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

**Five rows joined this class in the 2026-08-20 run**, each read individually and each the
same shape — a task id co-occurring with completion-ish phrasing that is *about something
else*:

| task | project | observed | episode | what the text actually says |
|---|---|---|---|---|
| 5480 | reify | pending | `7bd26155` | *"PDOCCOVER's DETECTOR landed as task #5478 (done) while its GATE, task #5480, is **STILL PENDING**"* — asserts the **opposite** |
| 3589 | dark_factory | pending | `4b8fc0f3` | *"the value 16 that task 5984 **landed** was chosen only to mirror dark_factory:3589"* — the completion word attaches to 5984, not 3589 |
| 5467 | reify | pending | `042b2e84` | *"because #6077 **waits on** #5467, the seven landed doc-sync rows stay rename-blind"* — a dependency statement |
| 6077 | reify | pending | `042b2e84` | *"select_infra_tests() was **FOLDED INTO** task #6077 rather than filed as its own task"* — a scope statement |
| 6077 | reify | pending | `fcf5da6d` | *"it **would put** a landed, drift-guarded mechanism into #6077's blast radius"* — hypothetical, about a rejected proposal |

The `5623` and `5480` cases are the sharpest, and the second one is new evidence for an
already-recorded defect: **both sentences state that the named task has *not* landed, and the
extractor flagged them as completion claims anyway.** The negation stripper reaches neither
*"has landed nothing"* nor *"is STILL PENDING"*. That the same construction recurred in a
fresh episode written five days later is the strongest signal in this document that the
defect is live on the write path, not historical.

The eleventh member of this class in the 16:45 run — task 3584, episode `e64dbe6f` — is **no
longer a finding at all**: task 3584 has since gone `cancelled`, which is terminal, so the
claim now verifies. The `5623` row above is in the same position as of 2026-08-20: task 5623
is now `done`, so it too has dropped out of the report, though it is kept in the table because
it is this class's canonical example. Both were false positives when observed and are
invisible now, for reasons unrelated to why they were false positives.

**Verdict: false positives**, all 14.

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
  `dark_factory` graph grew 2995 → 3030 Episodic nodes over ~2.5h during the earliest run's
  measurements, 3030 → 3042 over the three days after that, 3042 → 3043 in the 5.5h before
  the 16:45 run, and 3043 → 3105 in the ~4.7 days before this one (`reify` 4553 → 4588 over
  the same span; `provenance.json`). **7693 is a snapshot at run time, not a fixed
  population** — and the movement is not only additive: one derived edge named by the 16:45
  report (`f2dad7cc-…`) no longer exists in either graph.
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
- **Edges are enumerated only in the graphs this run swept** (Caveat 6). Enumeration is
  keyed on the **episode uuid** and asks **every** swept `--project` graph — for this run,
  both of them, for all 204 findings — so `derived_edge_uuids: []` means "none in
  **`dark_factory` or `reify`**", not "none anywhere". A graph never passed to `--project`
  is not read at all, and 15 cross-graph findings name exactly such a graph; that is the
  residual bound. `edges_enumerated_in` names which graphs answered and
  `derived_edges_by_graph` which held each edge — measured, with a worked example, in
  *[What the re-run changed](#what-the-re-run-changed)*. A `null` there would mean the edges
  were never enumerated at all, and a non-empty `edges_unqueried_in` that the list is a
  measured lower bound; this run has **0** of each.
- **The originating incident's own record is in the corpus and this sweep does not flag it.**
  See below — it is the sharpest available demonstration that the under-count is real
  rather than theoretical.

### The esc-3085-1 instance-(2) record was swept and came back clean

Observed 2026-08-12 at dark-factory `a138751e80`, and unchanged in the 2026-08-15 and
2026-08-20 re-runs:

- Episode `02090224-7bc9-4485-9291-6748e1042ac9` is **still present**, in the **reify**
  graph, with `source_description = add_memory:temporal_facts` — in scope for this sweep,
  and carrying **no** `[unverified_claim]` prefix (it predates the task-3142 gate).
- It appears in **none** of this sweep's 204 findings.
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
6. **COVERAGE, CROSS-GRAPH EDGES:** derived edges are enumerated by EPISODE UUID in EVERY swept --project graph — not only the graph a record was read from and the graph its own group_id names. An episode's edges live in the graph its INGEST ran against, which can be neither of those two, so a rule keyed on that pair cannot reach them at all. Measured rather than hypothetical: episode a887c958-0018-4715-8817-cf048c187e8d exists only in the reify graph with group_id dark_factory, and all 8 of its derived edges exist only in the dark_factory graph. derived_edges_by_graph names which graph held each edge; edges_enumerated_in names which graphs answered; derived_edge_uuids is their flat union. RESIDUAL BOUND: a graph NOT passed to --project is not read at all, so an empty list means "none in the graph(s) named by edges_enumerated_in", not "none anywhere".
7. **EDGES NOT ENUMERATED:** enumeration has THREE states. Fully enumerated: every targeted graph answered. NOT ENUMERATED: a finding whose derived_edge_uuids and derived_edges_by_graph are null (and whose edges_unqueried is true) had NO targeted graph answer — every edge query failed — so its harm artefacts are UNKNOWN, not absent; summary.edges_unqueried is that denominator. PARTIAL: edges_unqueried_in non-empty while edges_unqueried is false means SOME targeted graph failed to answer, so the edge list is a measured LOWER BOUND and edges_unqueried_in names the graphs that were not covered; summary.edges_partial is that denominator, and the two counters never overlap. An empty list with an empty edges_unqueried_in is the opposite of all of these: every targeted graph WAS asked and returned nothing.
8. **DETECTION BOUND:** claims are detected by the deterministic lexical vocabulary in fused_memory.services.completion_claim_gate, which requires completion PHRASING and a concrete NAMED REF (task id / commit sha / tkt_ id) to co-occur in one clause. A fabricated completion phrased without a named ref is invisible to this sweep by construction.

---

## What this found instead

The sweep's real yield is not a list of fabrications — it is **three reproducible
false-positive classes in the write-time gate itself** (task 3142), each of which causes a
truthful episode to be stamped `[unverified_claim]` in production today:

1. **Filing claims are held to a completion standard.** `_verify_task` requires TERMINAL for
   every claim kind, so a truthful *"filed as task N"* about still-open work is a mismatch.
   `_verify_ticket` already treats existence as sufficient for the same kind of claim.
   Largest class here: **128 of 172**.
2. **Cross-repo commit claims are probed against the wrong repository.** A claim is
   attributed to the writer's project even when the prose names another repo, so the sha is
   sought in the wrong object store. **12 confirmed** — each one a false accusation against
   a writer who was telling the truth.
3. **The negation stripper misses at least two constructions.** *"task 5623 has landed
   nothing (still pending)"* and — in an episode written 2026-08-19, five days later —
   *"its GATE, task #5480, is STILL PENDING"* were both extracted as completion claims. The
   recurrence is evidence this misfires on the write path **now**, not just historically.
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

That now covers the **sqlite** authorities too, which it did not in the earlier runs: task
statuses and ticket rows were read through `SqliteTaskBackend` / `TicketStore`, each of
which opens a read/write connection and applies schema work (WAL pragmas + migration;
`CREATE TABLE`/`ALTER TABLE`/`CREATE INDEX`) to whichever database a claim names — for a
cross-project claim, another project's live db while its own orchestrator is running. Both
are now read over explicit `file:...?mode=ro` URIs and closed in a `finally`.

Given the adjudication above, **no edge or episode in this corpus is recommended for
invalidation.** Acting on the raw 172 would have destroyed truthful records.

That holds for all three re-runs. Correcting the reader lookup, then widening it to the
`group_id` graph, then keying it on the episode uuid across every swept graph, changed only
*which edges this report names*; none invalidated anything, deleted anything, or moved
anything between graphs. The 8 / 11 / 7 cross-graph edges the 11:13 run had stopped naming
were never *gone* — only unlisted — and this run lists them again. Task 4230 added no CLI
flag and no mutation path; `TestReadOnlyByConstruction` passes unchanged.

The one edge this report *stops* naming (`f2dad7cc-…`, on finding `16cc46d8-…`) was **not**
removed by this sweep: it is absent from both graphs when queried directly, and this sweep
cannot delete anything. See
*[2026-08-20 vs 16:45](#2026-08-20-vs-1645--the-rule-generalized-to-every-swept-graph)*.

## Reproducing

```bash
cd fused-memory
PROJECT_ROOT=/home/leo/src/dark-factory \
DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory,/home/leo/src/reify \
RECONCILIATION_DATA_DIR=/home/leo/src/dark-factory/data/reconciliation \
uv run python scripts/audit_unverified_completion_claims.py \
    --project dark_factory --project reify --include-unverifiable
```

Run for this artifact set at dark-factory `b543435a48` (branch `task/4230`) on
2026-08-20T09:07:56Z, exit code 0; `report.json` is that run's stdout byte-for-byte.

The three env vars are **required**, not optional. Without them the project registry
resolves empty and the ticket store is not found, and every task and ticket claim falls to
`unverifiable` — see `provenance.json` → `notes` for the discarded first run that did
exactly that. The script warns loudly on stderr in that case rather than failing silently,
which is how the mis-scoped run was caught.
