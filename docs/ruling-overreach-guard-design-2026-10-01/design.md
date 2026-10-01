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
