# Ruling-scope overreach at edge extraction: re-measurement and a costed guard design

**Task 4716** · escalation `esc-4639-1` · instrument
`fused-memory/scripts/audit_ruling_overreach.py` · `graphiti_core` 0.28.2

**§1 and §2 are PRE-REGISTERED.** They were committed before any edge in the
sample below was labelled, so the rubric could not drift toward the results.
§3 onward were written after adjudication. Every number in §3–§7 is cited from
`report.json` in this directory by key path; run provenance is in
`provenance.json`.

## 1. Question

The esc-4639-1 ruling (Leo, 2026-08-24; durable record in task 4639's details)
measured ruling-scope overreach on 30 ruling episodes, using two
complete-coverage passes:

* 7.5% [4.4–12.4%] of minted edges were overreach;
* 30% of episodes carried at least one;
* under 40% of what was minted stated the holding;
* overreach was the most durable class: 92.3% still live, against 72.5% for
  holdings.

The ruling rejected option (a), the status quo, and refuted option (c) as
specified: a subject-keyed filter cannot see instance 1, whose ruling is the
edge's OBJECT. It named **provenance** as the axis: every fact edge carries
`episodes`, and every episode carries `source_description`. It also required
this task to **re-measure after 4715 landed**, because write-time tagging
changes the population a post-extraction guard would see.

This document answers three questions, plus that precondition:

* **Q0, re-measurement.** Did 4715 shrink the population? Does the
  out-of-sample rate replicate?
* **Q1, classifier.** Which episodes should a provenance-keyed guard key on:
  RULING-headed records, decision records generally, or both?
* **Q2, granularity.** What does the guard act on (per edge or per episode),
  and how is a quarantined edge marked?
* **Q3, lever.** Which intervention, at what cost, and staged how?

## 2. Method

### 2.1 Population and reads

The instrument reads both graphs (`dark_factory`, `reify`) in full: every
`Episodic` node and every `RELATES_TO` edge, live or not, because durability is
an output. Reads use GRAPH.RO_QUERY through
`fused_memory/backends/graphiti_client.py::_paged_ro_query`, paged in a total
`ORDER BY uuid` against a census of the identical MATCH. A bare MATCH silently
truncates at the server's 10,000-row cap, and both edge populations are
above 20,000. Any incomplete read aborts the run with nothing written.

The edge census is `count(r)`, not `count(*)`. Measured during this task:
with `r` unreferenced, FalkorDB's bare-pattern `count(*)` counts connected
node pairs, not edges. On dark_factory that gave 19,409 against the true
20,491, so a census of that kind would have passed a read truncated by the
multi-edge excess.

### 2.2 Candidate classifiers

These are evaluated, not adopted. `HEAD` means `content[:200]`.

| name | predicate |
|---|---|
| `category_decisions` | parsed `source_description` category is `decisions_and_rationale` |
| `header_ruling_paren` | content starts `RULING (` (the task's narrow header) |
| `header_ruling` | `^\s*RULING\b`, case-insensitive |
| `ruling_lexeme_head` | `\b(ruling\|ruled)\b` in HEAD, case-insensitive |
| `decision_anchor_head` | `\b(ruling\|ruled\|decision\|decisions\|decided)\b` in HEAD AND an anchor in HEAD: `esc-N-N` or a parenthesis containing an ISO date |

Specimen recall is checked against the four known overreach episodes in
`audit_ruling_overreach.py::SPECIMENS` (reify 59d2d750 and 5c0884a3;
dark_factory 9b33077f and cf03f276).

### 2.3 Strata

Strata are assigned first match wins, in this order:

1. `ruling_lexeme`: `header_ruling` or `ruling_lexeme_head`.
2. `decision_anchor`: `decision_anchor_head`.
3. `other_decisions`: `category_decisions`.

Episodes matching none are out of scope. The three strata partition the
candidates, so a cross-stratum comparison can say WHERE overreach lives (Q1).
A uniform sample would be dominated by `decision_anchor`-only and
`other_decisions` episodes.

### 2.4 Sample

* **Window.** `[2026-08-25T00:00Z, 2026-10-01T00:00Z)`, frozen. It starts the
  day after the ruling, so it is disjoint from the 30 episodes the esc-4639-1
  passes adjudicated: an out-of-sample replication. The frozen end means later
  writes cannot move it.
* **Selection.** Group by `(graph, stratum)`. Within each group, order by
  `sha256("<graph>:<uuid>")` hex and take the first **8**. The selection is
  deterministic, independent of read order, and re-derivable from the
  committed `verdicts.json` sample block.
* **Bounds.** At most 48 episodes (3 strata × 2 graphs × 8).

### 2.5 What is adjudicated

* **Minted only.** An edge is MINTED by `episodes[0]`. graphiti_core's
  `utils/maintenance/edge_operations.py::resolve_extracted_edge` appends the
  current episode on a dedupe hit, so later entries are corroborations. They
  are counted, never adjudicated. A sampled episode that minted nothing still
  counts in its stratum's episode denominator.
* **Served** means `invalid_at IS NULL`. That is the only filter on every read
  path on main.
* **live_strict** means both `invalid_at` and `expired_at` are null. It is
  reported only to compare with the prior "92.3% still live". `expired_at`
  alone is the restored shape task 4714 measured, and is NOT retirement.

### 2.6 Rubric

**Judge against the source episode text alone, all of it, not the head.**
Every minted edge gets exactly one label. Decide in this order:

1. **Content.** Does the fact assert anything the record did not decide or
   state? If yes, the label is `overreach`, whatever the edge is bound to.
2. **Binding.** The fact is faithful, but is one endpoint the wrong entity, so
   that reading it off that node attributes it to the wrong subject? If yes,
   the label is `misbound`.
3. **Kind.** Otherwise it is faithful and correctly bound. Label it
   `holding`, `bookkeeping` or `context`.
4. **`unjudgeable`.** Use this only when the episode text cannot decide.

#### Labels

* **holding.** States what the record decided (the adopted option, rule or
  disposition) within the scope the record gives it. A rejection that is
  itself the decision ("X was chosen over Y") is a holding.
* **bookkeeping.** Who, when, which escalation, task or commit, or a status.
  True as stated, and not itself a decision.
* **context.** Faithfully restates grounds, measurements, background or
  history, with the record's own qualification intact. A counter-signal or
  rejected option restated WITH its rejection is context.
* **overreach.** Asserts more than the record decided. That is any one of:
  * **(i)** it generalises the holding beyond its stated scope;
  * **(ii)** it states an implication or consequence the record did not
    adopt, including a vague relational verb ("impacts", "affects",
    "implies") linking two things the record does not link;
  * **(iii)** it presents a weighed, rejected, proposed or counter-signal
    option as adopted or true. A proposal stated as current fact counts;
  * **(iv)** it drops a qualifier the record attaches, where the qualifier
    changes the holding's scope (who, which, when, under what condition).
    Dropping an incidental detail that does not change scope is not
    overreach.
* **misbound.** Faithful, but attached to the wrong endpoint. This is task
  4717's fact-PLACEMENT family. It is excluded from overreach and counted
  separately, because it needs a different fix.
* **unjudgeable.** Cannot be decided from the episode text. For example, the
  fact rests on a referent the episode never names.

#### Worked examples

All are edges minted by the four specimen episodes. All predate the window, so
none is in the sample.

| label | edge | fact (abridged) | why |
|---|---|---|---|
| holding | `2789bcbf` (reify, 59d2d750) | "The angular gate for orient_exp is narrowed to accept ANGLE only, rejecting dimensionless." | The ruling's own scope: "orient_exp and transform_exp's angular gate narrow to ANGLE ONLY". |
| holding | `92a1bbda` (dark_factory, 9b33077f) | "...when an L2 escalation's task is being actively driven by a human's live interactive unblock session, the watcher stands down..." | The decision, with every qualifier kept. |
| bookkeeping | `ef53fb6a` (dark_factory, 9b33077f) | "Leo's unblock session was confirmed to be active while he was working on task/1907." | A confirmed status, not a decision. |
| context | `1fe39ae8` (reify, 5c0884a3) | "The presence of zero conformers repo-wide made the migration of AnalysisResult the cheapest ever." | A ground the record states: "Zero conformers repo-wide made it the cheapest-ever migration". |
| overreach (ii) | `4f99fbf2` (reify, 59d2d750) | "Task 6080's decision to narrow the gate implies that sin should not accept dimensionless arguments." | The record weighs trig's acceptance of both spellings as a counter-signal and rejects the inference: "sin is a classical-convention math function with a large corpus". |
| overreach (iii) | `df7b7746` (dark_factory, cf03f276) | "The recovery strategy involves discarding the stale branch and re-dispatching fresh..." | The record calls that lever "expensive" and adopts the reframe: "Reframed as REBASE-FIRST / regenerate-last". |
| overreach (iv) | `b2267a98` (dark_factory, 9b33077f) | "The L2-watcher decision generalizes the rule to live unblock sessions." | It loses "an L2 escalation's task", "a human's" and "interactive", all three of which sibling `92a1bbda` keeps. |
| misbound | `8a51e13b` (reify, 59d2d750) | "Task 6126 is landing to remove the last admission of dimensionless in the transform family..." | The fact is about task 6126 and is faithful to "#6126 is landing to remove...", but its endpoint is node `Task 6128`. |
| unjudgeable | none | — | No example is drawn from the specimens. The label is reserved for a fact the episode text cannot decide. |

#### Known blind spot: authored overreach

Reify `c6ac6d99` ("The analysis seed rows of α (6001) carry
Const(Scalar<PRESSURE>) with basis Ruling(#6165)") restates its episode
5c0884a3 faithfully: "α (6001)'s analysis seed rows carry
Const(Scalar<PRESSURE>) with basis Ruling("#6165")". Under this rubric it is
**context**. The over-assertion was AUTHORED in the record (task 4639
details), so no episode-only adjudication can see it, and no extraction-side
lever can prevent it. The measured rate is therefore a rate of EXTRACTION
overreach, not of all overreach.

### 2.7 Measures

These are computed by `audit_ruling_overreach.py::adjudicated_rates`, per
stratum and rolled up as `all`. Each rate carries a Wilson 95% interval, and
the instrument is pinned to reproduce the prior published intervals exactly.

* `overreach_rate_minted` = overreach / minted. Comparable with the prior 7.5%.
* `overreach_rate_substantive` = overreach / (minted − bookkeeping −
  unjudgeable). Comparable with the prior 14.3%.
* `episode_hit_rate` = episodes with ≥1 overreach / sampled episodes.
  Comparable with the prior 30%.
* `holding_share` = holding / minted. Comparable with the prior "under 40%".
  `holding_share_in_hit_episodes` restricts it to episodes with any overreach,
  which is the Q2 granularity input.
* `served_fraction_by_label` and `live_strict_fraction_by_label`. Comparable
  with the prior 92.3% / 72.5% durability.
* `per_classifier_rates`: the same, restricted to sampled episodes each
  classifier matches. Classifiers overlap, so these are not additive.
* `detector_catch`: of the adjudicated overreach edges, how many came from an
  episode on which each write-time detector fires, and on which any detector
  actually wired on `add_memory` fires.

The `all` roll-up is a mean over the stratified sample, **not** a corpus rate:
the strata are not population-weighted.

### 2.8 Sample drawn

Drawn by the instrument from the live graphs before any labelling (see
`provenance.json` for the run):

| graph | stratum | window population | sampled episodes | minted edges | corroborated |
|---|---|---|---|---|---|
| dark_factory | ruling_lexeme | 85 | 8 | 53 | 0 |
| dark_factory | decision_anchor | 81 | 8 | 52 | 0 |
| dark_factory | other_decisions | 205 | 8 | 46 | 0 |
| reify | ruling_lexeme | 69 | 8 | 53 | 0 |
| reify | decision_anchor | 65 | 8 | 55 | 0 |
| reify | other_decisions | 90 | 8 | 32 | 0 |
| **total** | | **595** | **48** | **291** | **0** |

Every group reached the cap of 8, and no sampled episode's edges were
corroborated by a later episode.

### 2.9 Adjudication procedure

There is a single adjudicator, applying §2.6 verbatim and reading each
episode's full content. Each verdict's rationale quotes, or points at, the
episode span it rests on. The work split is recorded in §2.10 once
adjudication has run. `verdicts.json` is validated fail-closed by
`audit_ruling_overreach.py::load_verdicts` against the re-derived sample: an
unknown label, a missing rationale, a duplicate, an out-of-sample edge, a
missing edge or a mismatched sample definition each aborts the report. The
worksheet itself is not committed: it carries full episode bodies and is
regenerable from the frozen window.

### 2.10 How the adjudication was done (added after labelling)

The work was not split. One adjudicator, the implementing agent for task 4716,
labelled all 291 minted edges in a single pass, one stratum at a time
(`ruling_lexeme`, then `decision_anchor`, then `other_decisions`). It read
each episode's full body beside its minted edges and applied §2.6 verbatim. No
sub-agents were used. It is therefore a **single-adjudicator** pass, against
the prior figure's two independent passes, and no inter-rater agreement can be
reported.

Garbled facts were labelled as follows:

* An inverted causal link, a misattributed actor, or a quantity assigned to
  the wrong thing asserts something the record did not state, so it is
  labelled `overreach` (ii) under the content-first ordering.
* A faithful fact whose endpoint node is an entity the episode never names, or
  the wrong one, is labelled `misbound`.

`verdicts.json` passed `load_verdicts` against the re-derived sample: no
missing, duplicate, out-of-sample or stale verdict (see `report.json`
`adjudicated.verdicts` and `adjudicated.stale_verdicts`).

---

*§3 onward were written after adjudication. Every number below comes from
`report.json` (`swept_at` 2026-10-01T09:01:39Z) and is cited by key path; `a.`
abbreviates `adjudicated.`.*

## 3. Re-measurement

### 3.1 What 4715 changed: almost nothing in this population

* **What runs on `add_memory`.** `wired_on_add_memory` = `completion_claim`
  and `unverified_claim_tag`. These are the completion-claim gate and the tag
  it writes. The two framing detectors stayed unwired (esc-4715-5).
* **What the gate actually did.** The tag is the gate's real output. In the
  full population it is on:
  * 0 `ruling_lexeme` and 0 `decision_anchor` episodes in either graph;
  * 1 of 1,713 dark_factory and 2 of 1,900 reify `other_decisions` episodes
    (`detector_census.<graph>.<stratum>.unverified_claim_tag`).
* **Extraction hits are not tags.** The `completion_claim` column counts
  episodes whose text contains a completion claim that names a ref, for
  example 80 of 196 dark_factory ruling episodes. Those are only CANDIDATES
  the gate verifies. It tags only a claim that fails verification.
* **The tag does not suppress.** It prefixes `source_description`, and
  graphiti extracts edges from a tagged episode exactly as before.
* **On the adjudicated overreach** (`a.detector_catch`), 34 edges:
  * 0 came from a tagged episode (`by_detector.unverified_claim_tag`);
  * 4 came from an episode where the gate would have extracted a claim
    (`by_detector.completion_claim`, `any_wired`);
  * counting the unwired detectors as well, 9 of 34 (`any_detector`). 6 of
    those 9 are `batch_plan`, the detector whose hit on specimen cf03f276 is
    a false positive: it reads "(2026-06-23)" as a task-id range.

**Conclusion: 4715 did not shrink the population a post-extraction guard
sees.** Its gate labels a different defect (unverified completion claims). On
the measured overreach it fires 0 times, and even its widest possible reach is
4/34. Write-time tagging is not a substitute for an overreach guard.

### 3.2 The out-of-sample rates, beside esc-4639-1

| measure | prior (task 4639, 30 episodes) | this pass (`a.rates.all`, 48 episodes) | CIs overlap? |
|---|---|---|---|
| overreach / minted | 7.5% [4.4–12.4%] (13/174) | **11.7% [8.5–15.9%]** (34/291) `overreach_rate_minted` | yes (8.5–12.4%) |
| overreach / substantive | 14.3% | 13.1% [9.5–17.7%] (34/260) `overreach_rate_substantive` | prior CI unpublished; point inside |
| episodes with ≥1 overreach | 30% (9/30; Wilson [16.7–47.9%]) | 45.8% [32.6–59.7%] (22/48) `episode_hit_rate` | yes |
| holding / minted | "under 40%" | 43.6% [38.1–49.4%] (127/291) `holding_share` | — |
| still live (`live_strict`): overreach vs holding | 92.3% vs 72.5% | 88.2% (30/34) vs 91.3% (116/127) `live_strict_fraction_by_label` | gap does not replicate |
| served (`invalid_at` null): overreach vs holding | — | 97.1% (33/34) vs 97.6% (124/127) `served_fraction_by_label` | — |

* **The rate replicates and does not fall.** Every interval overlaps the
  prior. The point estimate is higher. One plausible part of that is the
  pre-registered content-first rule, under which a garbled fact is overreach
  (ii) (§2.10). Another is that this pass labels every minted edge of 48
  episodes rather than 30.
* **The durability gap does NOT replicate, and that is an age effect, not a
  contradiction.** The window episodes are at most 37 days old, so later
  rulings have had little time to supersede their holdings. The prior's 72.5%
  came from older records. Overreach is served at the same rate as holdings
  (97.1% vs 97.6%), so nothing retires it inside the window either. The design
  below does not rely on the durability gap.
* **Misbound** (task 4717's family) is 6 of 291 = 2.1% [0.9–4.4%]
  (`a.rates.all.misbound_count`). It is kept out of every overreach figure.

### 3.3 Population drift and write volume

* **Drift since the ruling.** The 2026-08-24 ruling counted
  `decisions_and_rationale` at 2,354 reify and 2,068 dark_factory episodes.
  Today it is 2,652 reify and 2,573 dark_factory
  (`classifiers.category_decisions.<graph>.population`).
* **Write volume** inside the window (`classifiers.<name>.<graph>.window_per_day`,
  dark_factory + reify):

| classifier | episodes/day |
|---|---|
| `category_decisions` | 9.97 + 6.05 = **16.0** |
| `decision_anchor_head` | 3.65 + 3.16 = 6.8 |
| `ruling_lexeme_head` | 2.30 + 1.86 = 4.2 |
| `header_ruling` | 0.46 + 0.70 = 1.2 |
| `header_ruling_paren` | 0.16 + 0.38 = 0.5 |

At 6.06 minted edges per sampled episode (291/48), the decision class mints
about **97 edges/day**.

## 4. Q1: the classifier is the structured category

| classifier | population df / reify | window df / reify | specimen recall | adjudicated overreach / minted (`a.per_classifier.<name>`) |
|---|---|---|---|---|
| `category_decisions` | 2,573 / 2,652 | 369 / 224 | **4/4** | 11.7% [8.5–15.9%] (34/291) |
| `decision_anchor_head` | 817 / 727 | 135 / 117 | 4/4 | 13.4% [9.3–19.1%] (25/186) |
| `ruling_lexeme_head` | 196 / 146 | 85 / 69 | 2/4 | 12.3% [7.3–19.9%] (13/106) |
| `header_ruling` | 31 / 44 | 17 / 26 | 1/4 | 12.5% [3.5–36.0%] (2/16) |
| `header_ruling_paren` | 13 / 29 | 6 / 14 | 1/4 | not computed: no sampled episode matched |
| *stratum* `other_decisions` | 1,713 / 1,900 | 205 / 90 | — | **9.0% [4.4–17.4%]** (7/78) `a.rates.other_decisions` |

The strata are `ruling_lexeme` 12.3% [7.3–19.9%] and `decision_anchor` 13.1%
[8.0–20.8%] (`a.rates.<stratum>.overreach_rate_minted`).

**Pre-registered rule applied.** The `other_decisions` interval overlaps both
ruling strata. So the class boundary is **decision records**, and the
classifier is the STRUCTURED `category == decisions_and_rationale` with no
prose regex. The writer already holds the category as data
(`services/memory_service.py::MemoryService.add_memory` resolves it before
composing `source_description`), so the guard keys on that value in process
and never parses the stored string (heuristic 12). Recall is 4 of 4
specimens.

**What a narrower classifier would cost:**

* A ruling-lexeme classifier would miss 2 of 4 specimens (`specimens.recall`).
* It would also miss `other_decisions`' 7 overreach edges, 7/34 = 21% of
  those measured.
* The narrow `RULING (` header the task proposed covers 1 of 4 specimens and
  0.5 episodes/day.

**What the data does show about ruling shape.** It concentrates *which
episodes* carry overreach, not the per-edge rate. `ruling_lexeme` episodes hit
75% [50.5–89.8%] against `other_decisions`' 18.8% [6.6–43.0%]
(`a.rates.<stratum>.episode_hit_rate`), and those intervals do not overlap. A
per-edge guard sees the same per-edge rate in both. So the build should log
the matching classifier names with every judged edge. That keeps a narrower
activation re-evaluable from shadow data, without a second sampling pass.

The ruling in task 4639 called the category "too broad". Breadth costs volume,
about 16 episodes/day against 4.2, and §6 shows that volume is affordable. It
does not cost precision: the per-edge rate is the same across the class.

## 5. Q2: per edge, minted only, an edge-property marker

**Per edge, never per episode.** In episodes that carry any overreach, 40.4%
of what they minted is holding (61/151,
`a.rates.all.holding_share_in_hit_episodes`). Quarantining a hit episode would
bury 61 holdings to remove 34 overreach edges. The 59d2d750 precedent is the
same: its holdings 2789bcbf and d104c799 sit beside overreach 4f99fbf2. A
disposition acts only on edges whose `episodes[0]` is the judged episode
(§2.5). In this sample no minted edge had been corroborated by a later episode
(`sample.sizes.*.corroborated` = 0), but 744 multi-episode edges exist
graph-wide (architect's measurement). A provenance guard acting on any listed
episode would hide facts another episode minted.

### 5.1 Why not `invalid_at`

`invalid_at` means "superseded" to every component that touches it:

* graphiti_core's own contradiction resolution;
* recon Stage 1's inline invalidations (4f99fbf2 was retired this way);
* `backends/graphiti_client.py::GraphitiBackend.update_edge`
  (`invalid_at` / `clear_invalid_at`).

A quarantine written there could not be told apart from a supersession. It
would feed graphiti's temporal reasoning as if a later fact had replaced it,
and a later `clear_invalid_at` restore would silently un-quarantine it.

### 5.2 Why not `expired_at`

It is already taken:

* Task 4714 is blocked on the measured finding that graphiti_core sets
  `expired_at` only together with `invalid_at`.
* 3,024 dark_factory edges carry `expired_at` alone because fused-memory
  restored them.
* The premise registry holds `valid_edge_query_expired_at_filter_refuted`.
* This sample shows the same shape: 8 holdings are served but not
  `live_strict` (124 served vs 116 `live_strict`). Using `expired_at` as the
  marker would hide restored facts.

### 5.3 Marker on the edge, or a sidecar registry

**The read sites** any marker must reach are every valid-edge read:

* `services/memory_service.py::MemoryService._search_graphiti` (a Python-side
  `invalid_at` check on search results);
* `services/memory_service.py::MemoryService.get_entity`, whose exact branch is
  `backends/graphiti_client.py::GraphitiBackend.get_valid_edges_for_node` and
  whose fuzzy branch is a search;
* `GraphitiBackend.get_connected_entity_uuids` and
  `GraphitiBackend.dedup_valid_edges_for_node`;
* the module-level `_ALL_VALID_EDGES_MATCH` (`get_all_valid_edges`) and
  `_PROVENANCE_RANK_CLAUSE` in `graphiti_client.py`;
* downstream, every orchestrator briefing that embeds edge facts
  (`orchestrator/src/orchestrator/agents/briefing.py`), which reads through
  search and `get_entity`.

**Edge-property marker** (for example `r.quarantined_at` plus
`r.quarantine_reason`):

* It lives with the edge, so it survives merges and reassignments that copy
  edge properties.
* Every Cypher read can filter it in the same WHERE clause it already uses.
* Its cost is one predicate added at every read site above.
* Whether graphiti_core's `EntityEdge` hydration carries an unknown property
  through search results is NOT verified here. The build's first enforcement
  leaf must test it.

**Sidecar registry**, mirroring `services/planned_episode_registry.py`:

* It needs no graph write and is reversible by deleting a row.
* But it is filtered only where someone wires it. The precedent is measured
  in code: the planned-episode registry is consulted in `_search_graphiti`
  alone, and neither `get_entity` nor any `graphiti_client.py` accessor sees
  it.
* Every Cypher accessor would need a uuid-list parameter or a post-filter,
  across two stores that can drift.

**Recommendation: the edge-property marker**, enforced through ONE shared
predicate constant composed into every valid-edge read. Task 4714's pending
`_ACTIVE_EDGE_PREDICATE` (both stamps null) is the same shape. It is on 4714's
branch, not on main: a grep of `src/` at this worktree's base finds no
`_ACTIVE_EDGE_PREDICATE`.

* Whichever task lands first introduces the shared served-edge predicate. The
  other extends it, so "served" stays one definition (SPOT).
* Reversibility is a `clear_quarantine` path on `update_edge`, mirroring
  `clear_invalid_at`.

## 6. Q3: the lever

Volume: 16.0 decision episodes/day and about 97 minted edges/day (§3.3). The
model is `llm.model: gpt-4o-mini` (`fused-memory/config/config.yaml`). No
dollar figure is given, because no price source is cited here. Cost is stated
relative to the calls graphiti already makes.

For scale, graphiti already makes these calls per episode (graphiti_core
0.28.2):

* one `extract_nodes`;
* one `dedupe_nodes.nodes`;
* one `extract_edges.edge`;
* one `dedupe_edges.resolve_edge` per extracted edge;
* one `extract_summaries_batch`.

That is about 10 calls for a 6-edge decision episode. This is an estimate from
the call sites, not a measurement.

| option | LLM calls/day | added latency inside `_identity_lock_for` | code surface | verifiability | expected catch on `verdicts.json` |
|---|---|---|---|---|---|
| **O1** `custom_extraction_instructions` for the class | 0 extra calls; longer prompts | none beyond longer prompts | 1 param threaded through `_execute_graphiti_write` → `GraphitiBackend.add_episode` | replay only: re-extract the 48 episodes into a scratch graph and re-adjudicate | unknown until replayed. It cannot remove the vendored licence "clearly stated or unambiguously implied" (`prompts/extract_edges.py::edge`), only counter it, and it is spliced into all three node prompts (`prompts/extract_nodes.py`), changing entity extraction for the whole class |
| **O2** typed `RULES_ON` via `edge_types`/`edge_type_map` | +1 `extract_attributes` per typed edge with fields | +1 call per typed edge | entity/edge type models plus a map | replay only | **0 suppressed**: an unmatched relation is still minted under a free `relation_type` (`utils/maintenance/edge_operations.py::extract_edges`). It labels holdings at best; it is not a guard |
| **O3** post-extraction judge over MINTED edges | +1 per class episode ≈ **16/day** (≈97 edges judged) | **0 if run after the lock** (see below); one LLM round trip if run as a tenth `_run_pass` inside it | judge prompt + parser + marker write + the §5 predicate | **deterministic wiring tests** (fake LLM) plus an **offline eval against `verdicts.json`** (291 labelled edges, 34 positive) | directly measurable: the only lever with an offline eval today |
| **O4** recon-targeted review: a deterministic per-cycle list of edges minted by class episodes, handed to Stage 1 | inside recon's existing Stage 1 budget | none on the write path | list builder + Stage 1 context | list selection deterministic; judgement replay-only inside recon | depends on Stage 1; turns "an LLM stage happening to look" (4639) into always looking. The exposure window is still one cycle (4f99fbf2 lived 5d10h over 28 cycles untargeted) |
| **O5** authoring-side lint | 0 | none | a write-time warning | deterministic | **0 by construction**: all 34 labelled overreach edges are extraction-side. It reaches only AUTHORED overreach (c6ac6d99), which the rubric cannot see |

**Lock placement.** `_identity_lock_for` serialises ALL Graphiti writes for a
group. The judge reads only two things: the episode body, and the facts of the
edges the write just minted. It writes only a marker property; it never creates
or resolves an entity. So it needs no identity lock. Run it after
`_execute_graphiti_write` releases the lock, as a best-effort step in the same
`_run_pass` spirit: a judge failure never fails the committed write. A tenth
sub-pass inside the lock would add one LLM round trip to every decision write
for the group, for no consistency gain.

### 6.1 Recommendation, staged, with trigger and revisit clauses

**O3, the judge, keyed on the structured category and acting on minted edges
only. Ship it in SHADOW first, with O4 as the shadow's consumer.**

* **Stage S (shadow, record-only).**
  * The judge runs after the lock on every `decisions_and_rationale` write.
  * It records `(graph, episode, edge, label, rationale, classifier names,
    judge model, prompt version)` to a sidecar.
  * It writes NOTHING to the graph, and nothing is filtered.
  * Its overreach flags go to recon Stage 1 as the O4 targeted list. Stage 1
    may act through today's status-quo path (`update_edge invalid_at`), so
    the shadow period still shortens exposure.
* **Trigger clause, from S to enforce.** The judge writes the §5 marker, which
  every read site filters, only when BOTH hold:
  1. **Offline.** On `verdicts.json`: precision ≥ 0.80 with a Wilson lower
     bound ≥ 0.65, and recall ≥ 0.50. At precision 0.80, at most one true fact
     is hidden for every four overreach edges hidden. For scale, 28 correct of
     34 flagged gives [0.66–0.92].
  2. **Online.** On a fresh shadow window of at least 200 minted edges,
     sampled per §2.4 with a new frozen window and adjudicated under §2.6, the
     same precision bounds hold.

  If (1) fails, the judge never leaves shadow. Re-open the lever choice, with
  O1 measured by replay of the same 48 episodes.
* **Revisit clause, after enforcement.** Return to shadow if either holds:
  * 30 days after enforcing, a re-run of `audit_ruling_overreach.py` on a new
    adjudicated window does not show the SERVED overreach rate at least halved
    against 11.7%;
  * the judge's flag rate on minted class edges exceeds 2 × the measured
    overreach rate, that is more than 23% of minted edges, in any 7-day span.
    That is the signature of a judge hiding holdings.

## 7. Accepted residuals

* **Authored overreach.** For example c6ac6d99: the episode itself
  over-asserts. No extraction-side lever and no episode-only judge can see it.
  It stays the authoring convention task 4639 recorded, unenforced.
* **Misbound, 2.1%.** This is task 4717's family. A content judge must not
  absorb it, and §2.6's ordering keeps it separate.
* **Single adjudicator.** No inter-rater agreement exists for this pass. The
  online trigger's second adjudication is the first chance to measure it.
* **Classifier recall.** There is no loss against the specimens (4/4).
  Ruling-shaped records filed under another category are out of class. All 48
  sampled episodes were `decisions_and_rationale`, but the population was not
  split to measure how many ruling-lexeme episodes sit outside the category.
* **Garbles.** Under the content-first rule, garbled facts count as overreach
  (ii). A judge prompted only on "scope" may miss them, and its recall on
  `verdicts.json` will show whether it does.
* **The durability claim is unconfirmed.** The prior's 92.3% vs 72.5% gap did
  not replicate inside a young window (§3.2). The case for a guard rests on
  the rate and the exposure, not on overreach outliving holdings.

## 8. Proposed build (ordered leaves)

1. **Promote the classifier (SPOT).** Add
   `fused-memory/src/fused_memory/services/overreach_guard.py` exposing the
   class predicate over the resolved category (the `MemoryCategory` value, not
   the string). `fused-memory/scripts/audit_ruling_overreach.py` must then
   IMPORT it for `category_decisions`, as `audit_wrong_binding_edges.py`
   imports its vocabulary. A test in the style of
   `tests/test_audit_wrong_binding_edges.py::TestNoSecondVocabulary` pins one
   copy. The other four candidate classifiers stay in the script as
   evaluation-only.
2. **Judge core, pure, plus offline eval.** The judge prompt and its strict
   parser over (episode body, minted edge facts) emit `LABELS`. Add
   `fused-memory/scripts/eval_overreach_judge.py`, which scores the judge
   against `docs/ruling-overreach-guard-design-2026-10-01/verdicts.json`.
   Files: `services/overreach_guard.py`, `tests/test_overreach_guard.py`.
3. **Shadow wiring.** Make a best-effort call after the lock in
   `services/memory_service.py::MemoryService._execute_graphiti_write`'s caller
   path, for class writes only, over edges whose `episodes[0]` is the written
   episode. Record to a new sidecar store. The graph is untouched. Files:
   `memory_service.py`, a new sidecar module, tests.
4. **O4 handoff.** Each recon cycle hands Stage 1 the shadow flags since the
   last cycle as a deterministic list. Files: `fused_memory/reconciliation/`
   (context assembly) plus its tests.
5. **Measurement gate.** A deterministic, `always_escalates` task evaluates
   the §6.1 trigger: offline from leaf 2, online from a new frozen window
   adjudicated under §2.6, using `audit_ruling_overreach.py`.
6. **Enforcement** (only if leaf 5 passes):
   * the edge-property marker write;
   * the shared served-edge predicate at every §5.3 read site, extending 4714's
     predicate if it has landed;
   * `update_edge(clear_quarantine=True)`;
   * a test per read site, plus the `EntityEdge` hydration check.

   Files: `backends/graphiti_client.py`, `services/memory_service.py`, tests.

## 9. Follow-ups filed

Both were filed through `submit_task` with `spawned_from: 4716` and
`escalation_id: agent-followup-4716`. Each id is a curator ticket, not a task
id.

* **F1 (medium), build the guard per §8**, carrying §6.1's staging, trigger
  and revisit clauses verbatim: `tkt_0RV9ERANDWG7CC3N1GFFKHCBY8`. The first
  submission was rejected (`LockCharterViolation`) because its `files` named a
  directory. It was re-filed naming
  `fused_memory/reconciliation/context_assembler.py`.
* **F2 (low), consolidate the read-only FalkorDB reader seam** into one src
  module. The seam is duplicated in
  `scripts/audit_wrong_binding_edges.py::EdgeReader`,
  `scripts/audit_unverified_completion_claims.py::EpisodeReader` and
  `scripts/local_memory_models_eval/build_corpus.py::EpisodeReader`, and the
  migration also covers `scripts/audit_ruling_overreach.py::GraphReader`. The
  ticket carries the edge-census `count(r)` gotcha: `tkt_0RV9EQ11QMC9WK79B74XBJ549H`.
