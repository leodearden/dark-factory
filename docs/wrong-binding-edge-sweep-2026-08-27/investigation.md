# Wrong-binding in extracted edges — detection, quantification, cause

**Task 4717** · escalation `esc-4639-1` · swept 2026-08-27T04:50:22Z ·
branch `task/4717` · sweep code at `8bc5f763a2` · `graphiti_core` 0.28.2

A *wrong-binding* edge is one whose `fact` is a faithful restatement of its
source episode, but which is **attached to the wrong entity**. Reading it off
the node it hangs from therefore attributes a true statement to the wrong
subject. `esc-4639-1` separates this fact-PLACEMENT family from the
fact-CONTENT family (a fact asserting more than its episode said), which
`scripts/audit_unverified_completion_claims.py` answers.

Every number below is cited from `report.json` in this directory, which is
byte-for-byte the stdout of
`fused-memory/scripts/audit_wrong_binding_edges.py` at `8bc5f763a2`. Run
provenance, including the read-population census, is in `provenance.json`.

**On line numbers.** `CLAUDE.md` asks for `path/to/module.py::symbol`
citations rather than bare line pins, because pins go stale. This document
cites symbol-first and gives line numbers only as *measurement evidence* —
here the line is part of the observation. All line numbers were read at
`8bc5f763a2` (in-tree) and at `graphiti_core` 0.28.2 (the installed wheel);
if a pin has drifted, the symbol is authoritative.

**Reproduce the whole sweep:**

```bash
cd fused-memory && uv run python scripts/audit_wrong_binding_edges.py \
    --graph dark_factory --graph reify --json \
    --out-dir ../docs/wrong-binding-edge-sweep-2026-08-27
```

---

## 1. Specimen re-verification

All ten specimens named in the task description were re-checked against the
**live** graphs (`invalid_at IS NULL AND expired_at IS NULL`). All ten still
exist and are still live. Seven are Class A and are reproduced by the
detector; three are out of reach by construction and are recorded in
`report.json`'s `known_gaps` (§4).

### The canonical family — reify node `Task 6165`

| edge | fact (truncated) | detected |
|---|---|---|
| `63fa5c78` | "Task 6164 described landing the same artefact — ElasticResult.rotation." | yes |
| `9a8e780b` | "Ruling 6164's HALF 2 was described in task 6164." | yes |
| `317da2e2` | "Task 6164 explicitly forbids reducing the task to 'conformance verification'…" | yes |
| `6a79d29b` | "Task 6164 had the 6185 GUI-channel-bridge dependency wired on 2026-08-10." | yes |
| `9135f049` | "Task 6164's IMPLEMENTATION COORDINATION paragraph was rewritten…" | yes |
| `54f53beb` | "Task 6128 was cancelled and consolidated into task 6164…" | yes (**6th**, not in the task description) |

The task description named five. The sweep finds **six**: `54f53beb`
(`(Task 6128)->(Task 6165)`) belongs to the same family and the same source
episode. It was not a regression — the corpus simply grew.

The node carries **7 live edges in total**. Six are the mis-bound family
above; exactly one — `5f1baf7f`, "Task 6165's supersession paragraph demanded
a `Const(Scalar<PRESSURE>)` for task 6001", from a *different* episode
`c2c692d4` — is genuinely about 6165 and is correctly **not** flagged. So
**6 of the 7 things this graph says about "Task 6165" are actually about task
6164.**

### `8a51e13b` — why a subject-only detector fails

```
(Task 6080) -> (Task 6128)
"Task 6126 is landing to remove the last admission of dimensionless in the
 transform family, which task 6080's decision addresses."
```

The **subject** `6080` *is* named in the fact, so the rule the task
description proposes — "fact names an id differing from the id in its
SUBJECT node's name" — returns *clean* here. The mis-bound end is the
**OBJECT** (`Task 6128`, while the fact says `6126`).

This drove the design decision to check **both** endpoints. It is not an edge
case: **73 of 181 findings (40%) are object-end**
(`report.json` → `summary.by_end`). A subject-only detector would have missed
two fifths of the population.

### Reproduction

```bash
cd fused-memory && uv run python -c "
import asyncio, falkordb.asyncio as fa
async def m():
    g = fa.FalkorDB(host='localhost', port=6379).select_graph('reify')
    r = await g.ro_query(\"MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) \"
        \"WHERE (a.name='Task 6165' OR b.name='Task 6165') \"
        \"AND r.invalid_at IS NULL AND r.expired_at IS NULL \"
        \"RETURN r.uuid, a.name, b.name, r.episodes, r.fact\")
    [print(x) for x in r.result_set]
asyncio.run(m())"
```

---

## 2. Fresh quantification

Over the **complete live population of both graphs**:

| measure | value |
|---|---|
| rows scanned | **26 667** (dark_factory 11 411 + reify 15 256 live `RELATES_TO`) |
| qualifying population | **6 959** |
| unverifiable (fact names no task id) | 304 |
| **findings** | **181** |
| **rate** | **2.60 %** |
| by graph | dark_factory 76 · reify 105 |
| by end | subject 108 · object 73 |
| truncated | `null` — nothing silently capped |

**Pagination is load-bearing and the run proves it.** reify holds 15 256 live
`RELATES_TO` rows against a server `RESULTSET_SIZE` of 10 000. An unpaginated
`MATCH` — the shape `audit_unverified_completion_claims.py` uses, correctly,
for its smaller population — would have returned exactly 10 000 of them
*silently*, and every denominator here would be wrong. `EdgeReader` routes
both reads through the shared paged primitive, and `truncated_by` is a
first-class report key rather than a footnote.

### Relation to the task's 11.7 %

The task description cited 13/111 = 11.7 %. That figure was a **narrow
pocket** — edges whose subject is a *ruling task* — not the whole corpus.
**2.60 % is the whole-corpus rate over 6 959 qualifying edges.** Both stand;
they measure different denominators. Neither supersedes the other.

**2.60 % is a LOWER BOUND.** Endpoints and facts are read with the shared
vocabulary in `fused_memory/utils/canonical_labels.py`, which is documented
*precision over recall*. A node named with bare digits, a reference made by
task **title**, an alias/codename, and a hard-wrapped qualified ref are all
invisible by design.

### The 181 findings are NOT one homogeneous population

This is the most important analytic result for whoever adjudicates them, and
it is not visible from the headline rate. Crossing proximity against whether
the correctly-named node already exists:

| proximity | correct node absent | correct node present |
|---|---|---|
| `one_digit_diff` | **57** | 40 |
| `prefix` | 2 | 6 |
| `unrelated` | 11 | **65** |

Two distinct modes fall out:

**Mode 1 — near-miss substitution (the ~57-edge cell).** The correct node does
not exist, and the bound node's id differs by one digit. This is the
`Task 6165` family, and §3 traces its mechanism exactly. High confidence
these are true defects.

**Mode 2 — shared-sink collapse (the ~65-edge cell).** The bound node's id
isn't close to anything the fact names, *and* the correctly-named node
already exists. Worked example — dark_factory node `task 1155` carries 5 live
edges, of which **3 come from one episode `64315ec6` and are each about a
different task**:

```
4895becb (Task 1137)->(task 1155)  "Task 1137 completed at 2026-05-09T17:09 UTC via commit b4b4614…"
d5706134 (Task 1144)->(task 1155)  "Task 1144 completed at 2026-05-09T13:47 UTC via commit 750525…"
28891420 (Task 1173)->(task 1155)  "Task 1173 completed at 2026-05-09T13:53 UTC via commit e6c56c1…"
```

A multi-task episode produced N facts and the extractor used one arbitrary
task node as a shared object endpoint. The other two edges on that node are
legitimate (their episodes genuinely discuss 1155).

**Mode 2 is also where the false positives concentrate.** Some members are
legitimate cross-task relations that merely hang off a task-shaped node —
`52090eb1` on `Task 2273`, "The live cross-graph migration was executed by
sibling deterministic task 2456", is plausibly a correct relation. This is
exactly why every finding is a **candidate for human adjudication** and why
the script has no remediation path (§5).

**31 families** (a node with ≥2 findings) cover 88 findings; the remaining
**93 are singletons**.

---

## 3. Cause

Stated as measured fact. Each claim carries its reproduction.

### 3.1 The source episode never mentions the node it bound to

Episode `779b7b7d-bb31-436b-869b-b4cafd033282` (reify,
`source_description = add_memory:decisions_and_rationale`, created
2026-08-20T10:36:07Z, 1094 chars) is **wholly about tasks 6128 and 6164**.
The string `6165` **does not occur in it**. All six edges above nonetheless
bound to node `Task 6165`.

### 3.2 The correctly-named node does not exist — but a near-miss does

- `Task 6164` — **does not exist** in reify, in any spelling.
- `Task 6165` — **exists**, uuid `b500ca10…`, created **2026-08-17T17:29:35Z**,
  three days *before* the episode.

**Sharpening not previously recorded:** a node named **`ruling 6164`**
(uuid `39eff5b0…`, created 2026-08-10) *does* exist — and edge `9a8e780b`
points *to* it while its subject bound to `Task 6165`. So within one episode
the extractor materialised the correct referent under a non-canonical surface
form and *still* resolved the subject to the wrong task node. That is active
mis-resolution, not a coverage gap.

```bash
cd fused-memory && uv run python -c "
import asyncio, falkordb.asyncio as fa
async def m():
    g = fa.FalkorDB(host='localhost', port=6379).select_graph('reify')
    r = await g.ro_query(\"MATCH (e:Episodic) WHERE e.uuid STARTS WITH '779b7b7d' RETURN e.content\")
    c = r.result_set[0][0]; print('mentions 6165:', '6165' in c); print(c)
    r = await g.ro_query(\"MATCH (n:Entity) WHERE n.name CONTAINS '6164' RETURN n.name, n.created_at\")
    print('nodes containing 6164:', r.result_set)
asyncio.run(m())"
```

### 3.3 Corpus-wide, this is mis-resolution rather than a missing node

- **105/181 (58 %)** of mis-bound endpoint ids are a *near miss* of an id the
  fact names — 97 one-digit-different at equal length, 8 a strict prefix.
  Against 1 423 + 2 111 task-shaped nodes, the chance baseline is near zero.
- **111/181 (61 %)** have `correct_node_present = true`: the node the fact
  actually names **already exists** in that graph.

Both reproduce the planning-time measurement (62.5 % and 64 %) within the
movement of a live corpus.

**The canonical specimen is not representative** — because `Task 6164` is
absent, the `Task 6165` family sits in the 39 % *minority*. A reader
generalising from that family alone reaches the wrong diagnosis. The report
carries both counts so this is visible.

### 3.4 A statistic that must NOT be used as cause evidence

Planning recorded "100 % of the qualifying population traces to
`add_memory:*` episodes". **Re-measured, that is true and completely
vacuous:** *every* `Episodic` node in both graphs is `add_memory:*` —
dark_factory 3 261/3 261, reify 4 656/4 656. There is no other ingestion path
represented in these graphs at all.

It therefore establishes *which* write path is implicated, and carries
**zero discriminating power** about whether `add_memory` is riskier than
`add_episode`. Answering that needs a corpus containing both. Recorded here
so nobody re-derives it as evidence of an `add_memory`-specific defect.

```bash
cd fused-memory && uv run python -c "
import asyncio, falkordb.asyncio as fa
async def m():
    db = fa.FalkorDB(host='localhost', port=6379)
    for gr in ('dark_factory','reify'):
        r = await db.select_graph(gr).ro_query(
          'MATCH (e:Episodic) RETURN split(e.source_description,\":\")[0], count(*)')
        print(gr, r.result_set)
asyncio.run(m())"
```

### 3.5 The mechanism in `graphiti_core` 0.28.2

The `graphiti/` submodule is **not checked out** — `git submodule status`
reports `-dea9e86…` (leading `-` = uninitialised) and the directory is empty.
`graphiti_core` is consumed only as an installed wheel, so **it cannot be
patched in-tree**.

**Step 1 — the candidate pool is fuzzy-only.**
`graphiti_core/utils/maintenance/node_operations.py::_collect_candidate_nodes`
(def L209, body to L241) builds the pool solely by hybrid search keyed on
`query=node.name` (L219) with `config=NODE_HYBRID_SEARCH_RRF` (L222) —
BM25 + cosine, RRF-reranked (`search/search_config_recipes.py` L156-161) —
and `search_filter=SearchFilters()` (L221), i.e. **empty**. The function body
contains no Cypher, no exact-name lookup and no numeric constraint. `"Task
6164"` reliably surfaces `Task 6165`.

**Step 2 — the deterministic layer provably cannot decide it.**
In `graphiti_core/utils/maintenance/dedup_helpers.py::_resolve_with_similarity`
(L198-244) the gates run in this order:

1. **Entropy gate** (L208) — `_has_high_entropy('task 6164')` is **True**
   (`_name_entropy` = **2.75** ≥ `_NAME_ENTROPY_THRESHOLD` = **1.5**, L31).
   So it is *not* punted early.
2. **Exact-name map** (L212) — `indexes.normalized_existing.get(...)`. This
   is scoped to the **already-fuzzy candidate pool** from step 1, so a node
   that is absent (or that never entered the pool) **cannot be rescued here**.
3. **LSH + Jaccard** (L239) — 3-gram Jaccard of `'task 6164'` vs
   `'task 6165'` is **0.714** (shingle sets differ in exactly one element:
   `{164,616,ask,k61,sk6,tas}` vs `{165,616,ask,k61,sk6,tas}`; 5∩ / 7∪),
   below `_FUZZY_JACCARD_THRESHOLD` = **0.9** (L34). The fuzzy accept
   declines.

The decision therefore reaches the LLM dedupe pass **by construction**.

> **Correction to the planning note:** these four symbols
> (`_has_high_entropy`, `_NAME_ENTROPY_THRESHOLD`, `_FUZZY_JACCARD_THRESHOLD`,
> `_jaccard_similarity`/`_shingles`) live in **`dedup_helpers.py`**, not
> `node_operations.py`. Every numeric value (1.5, 0.9, 2.75, 0.714) is exact.

**Step 3 — the LLM's answer is accepted without a numeric check.**
`node_operations.py::_resolve_with_llm` builds
`existing_nodes_by_name = {node.name: node for node in indexes.existing_nodes}`
(L311-313) and then resolves (L379-382):

```python
if not duplicate_name:
    resolved_node = extracted_node
elif duplicate_name in existing_nodes_by_name:
    resolved_node = existing_nodes_by_name[duplicate_name]
```

The **only** validation is membership in that name map. There is no digit
comparison against `extracted_node.name`, no similarity re-check, no
numeric-suffix guard. L392-395 then unconditionally writes
`state.uuid_map[extracted_node.uuid] = resolved_node.uuid`.

**Step 4 — one map entry rewrites the edge.**
`graphiti_core/utils/bulk_utils.py::resolve_edge_pointers` (def L549) applies
it at L553-554:

```python
edge.source_node_uuid = uuid_map.get(source_uuid, source_uuid)
edge.target_node_uuid = uuid_map.get(target_uuid, target_uuid)
```

It is generic over `E = TypeVar('E', bound=Edge)`, so it rewrites **both**
endpoints — which is why the defect appears at the object end as often as the
subject end (§1, `8a51e13b`).

```bash
cd fused-memory && uv run python -c "
from graphiti_core.utils.maintenance.dedup_helpers import (
    _has_high_entropy, _name_entropy, _cached_shingles, _jaccard_similarity,
    _NAME_ENTROPY_THRESHOLD, _FUZZY_JACCARD_THRESHOLD)
print('entropy', _name_entropy('task 6164'), '>=', _NAME_ENTROPY_THRESHOLD,
      '->', _has_high_entropy('task 6164'))
print('jaccard', _jaccard_similarity(_cached_shingles('task 6164'),
      _cached_shingles('task 6165')), '<', _FUZZY_JACCARD_THRESHOLD)"
```

### 3.6 fused_memory supplies almost no resolution levers, and no config governs any threshold

`fused_memory/backends/graphiti_client.py` calls `Graphiti.add_episode` at
L1442-1454 with: `name`, `episode_body`, `source`, `group_id`,
`source_description`, `reference_time`, `entity_types`, `uuid`.

> **Correction to the planning note:** an `entity_types=` kwarg **is** passed
> (L1450). The substance nevertheless holds: it is a pure pass-through of the
> wrapper's own `entity_types: dict | None = None` (L1421), those two lines
> are the *only* occurrences of `entity_types` in `fused-memory/src/`, and the
> sole production caller (`memory_service.py` L3923-3932) omits it — so it is
> **always `None`** in production.

`excluded_entity_types`, `custom_extraction_instructions`, `edge_types` and
`edge_type_map` appear **nowhere** in `fused-memory/src/`, `shared/` or
`orchestrator/`, while upstream `graphiti_core/graphiti.py` L788-806 exposes
all four. **These levers exist and are entirely unused.**

**No config key governs the thresholds.** `_NAME_ENTROPY_THRESHOLD` and
`_FUZZY_JACCARD_THRESHOLD` exist only as hardcoded module constants in the
installed wheel. `fused_memory/config/schema.py` carries many `*_threshold`
keys, but the nearest (`procedural_knowledge_near_dup_threshold`,
`resolve_near_dup_threshold`) govern fused-memory's own Mem0 near-duplicate
logic, not `graphiti_core` node dedupe. There is **no override path**.

---

## 4. The heuristic that was measured and rejected

Two of the ten specimens are **direction reversals**: both endpoints are
correctly named in the fact, only the subject/object roles are swapped.

```
1cf19488 (Task 6346)->(Task 6347)  "The recurring-attention task #6347 depends on task #6346."
01e3ff5d (Task 5997)->(Task 6014)  "Task 6014 carries task 5997 as a hard dependency."
```

Set-membership is **satisfied at both ends**, so the Class A detector returns
clean — correctly, by construction.

The obvious cheap heuristic — *leftmost task id named in the fact == object id
!= subject id* — **was implemented and measured during planning**: it flags
**85 / 7 131** and does catch both specimens. It was **not shipped**, because
reading all 85 shows the overwhelming majority are benign **grammatical
voice**, not mis-binding:

- "Task 2660 depends on Task 2659 landing" on `(2659)->(2660)`
- "Task 846 is the companion task to Task 839" on `(839)->(846)`
- "Themes addressed by follow-up Task 2083 are related to Task 394" on `(394)->(2083)`

Precision is far too low to ship, and **the two true specimens are
indistinguishable from that noise by text alone** — because both of their
endpoints *are* named, the only thing that could adjudicate them is the
**authoritative task dependency graph**, which this sweep deliberately does
not consult.

This negative result is recorded in `report.json` → `known_gaps` (not only
here) so that "this run found no reversals" can never be read as "the corpus
holds no reversals", and so nobody re-proposes the heuristic.

The third out-of-reach specimen, `993a9a7b` `(Task 6004)->(Task 5997)`
— "Task 6004's rulings were ported verbatim into task 5997" — **contradicts
its source episode**. It is reachable by no text or topology rule at all:
adjudicating it requires re-reading the episode body and comparing meaning.
That is the fact-CONTENT family `esc-4639-1` separates from this one.

> **Errata.** The first cut of `KNOWN_GAPS` cited `01e3e75e` for the second
> reversal specimen. That uuid does not exist in reify in any state; the real
> edge is `01e3ff5d-ef73-4f52-9914-95533fca5cf9`, verified live and
> corroborated by `docs/unverified-completion-claim-sweep-2026-08-11/report.json`.
> Fixed at source in `8bc5f763a2` and the sweep re-run — the artifact was not
> hand-patched.

---

## 5. Proposed fix

**The fix is already designed in-tree and merely unwired.** This is the
central finding of §5: no new detection logic is needed.

### What already exists

`fused_memory/services/memory_service.py::MemoryService._verify_episode_referents`
(L3223-3599) **already performs exactly this check, post-write**, as a
set-membership test (L3524-3529):

```python
if endpoint_referent not in referent_set:
    check = 'set-membership'
elif cited_declared and endpoint_referent not in cited:
    check = 'per-edge-pairing'
else:
    continue
```

It computes `resolvable = len(candidates) == 1` (L3539), resolves
`new_endpoint_uuid` (L3569-3582) via the read-only
`_intended_endpoint_uuid` (L3602-3639) — and then **only logs a warning**
(L3596-3598). Its own docstring (L3235) says so: *"DETECTS AND RECORDS ONLY
— it performs no writes of any kind"*, pinned by
`tests/test_referent_verification.py` L213-226 (`_WRITE_PRIMITIVES` /
`assert_never_repaired()`).

The repair chain — **leaf ETA**, `ensure_entity_node` → `reassign_edge` →
`refresh_entity_summary` — is documented at `memory_service.py` L3236 and in
`plans/memory-referent-fidelity-prd.md` L123 and L321 (where η is still
listed as a **leaf**, depending on α and ζ). It has **no production wiring**:
`ensure_entity_node` is defined at
`fused_memory/backends/graphiti_client.py` L3156 and has **zero production
call sites** repo-wide — every occurrence outside its own body is either a
test (`test_write_time_identity.py`, `test_referent_verification.py`) or
prose.

So the system already *knows* about these 181 edges at write time and
deliberately does nothing.

### Recommendation

**The primary fix is already a filed, live task — do not re-file it.**

1. **Primary — wire leaf ETA: already tracked as task 3672**, *"Referent
   repair path: `ensure_entity_node` → `reassign_edge` →
   `refresh_entity_summary`, + streak escalation"*, **status
   `in-progress`**, priority high. This sweep does not propose new work here;
   it supplies 3672 with a measured population (181 edges) and one design
   constraint it should honour:

   > **Gate the repair on `resolvable`, and exclude mode 2.** 111/181 (61 %)
   > already have an existing correct node; mode 1 (§2) supplies the rest via
   > `ensure_entity_node`. **Do not** auto-repair the mode-2 cell
   > (`unrelated` proximity + node present, ~65 edges): it contains
   > legitimate cross-task relations, and it is precisely where an automated
   > verdict would corrupt good data.

2. **Remediation of the existing edges — already tracked as task 3673**
   (leaf θ, `pending`), which remediates measured live conflations. The 181
   findings here are candidate input to it, **not** a mandate: each still
   needs human adjudication (§5, below).

3. **Complementary prevention — NOT currently tracked; filed by this task.**
   `graphiti_client.py` L1442-1454 is the only place fused_memory can pass
   resolution levers into `add_episode`, and it passes none that bite (§3.6).
   A `custom_extraction_instructions` / `excluded_entity_types` constraint
   requiring an exact referent match for task-shaped names would attack the
   cause rather than the symptom. Note this **cannot** be fixed by patching
   `graphiti_core` in-tree — the submodule is not checked out (§3.5).

   Nothing in the backlog covers this: the nearest neighbours are task 2073
   (duplicate-node minting, done), 385 (noise-edge extraction filters, done),
   2110 (node-name canonicalisation, done) and 3335 (cross-project collapse,
   cancelled). None passes resolution levers into `add_episode`.

### Remediation of the existing ~181 edges stays human-gated

`reassign_edge` is the lossless remediation primitive for them, but
**remediation is deliberately out of scope for this task.** Every finding is
a *candidate*, not a verdict. This script has no `--apply`, `--invalidate`,
`--delete`, `--repair` or `--reassign` path, and
`TestReadOnlyByConstruction` enforces that absence mechanically — its
forbidden-call set deliberately names `reassign_edge` and `merge_entities`,
because those are exactly what a later editor would reach for on this defect
class. The graph is read only over `GRAPH.RO_QUERY`, where read-only is
**server-enforced** rather than client-promised.

**This sweep invalidated nothing, reassigned nothing and deleted nothing.**

---

## 6. Reader warning

**Any prior investigation that read "the Task 6165 ruling" out of the reify
graph was reading task 6164's ruling.** Six of the seven live edges on that
node are about 6164 (§1). The same hazard applies to all 181 findings and,
by the recall bound in §2, to an unknown number of edges the shared
vocabulary cannot see.

**Re-derivation must go through `r.episodes`.** It is populated on **100 % of
live `RELATES_TO` edges in both graphs** — reify 15 256/15 256, dark_factory
11 411/11 411, zero null-or-empty — so it is the right handle for recovering
what an edge was actually extracted from. Never trust the endpoint node name
alone.

**But `r.episodes` is not always sufficient.** Of the **116** distinct
episode uuids cited by the 181 findings, **7 (6 %) are dangling** — the
`Episodic` node no longer exists:

- dark_factory: `64315ec6` (1 of 51) — the `task 1155` shared-sink cluster in §2
- reify: `6ee043e1`, `4b64388f`, `71dcbcde`, `9ee72340`, `5735728d`,
  `15a1b133` (6 of 65)

Those edges are **permanently un-adjudicable from the graph alone**. A lookup
by episode uuid returning empty means the episode was deleted — not that the
query was malformed. This was discovered incidentally here; its corpus-wide
extent is **not** scoped by this sweep and is filed as follow-up work.

*(The `109/109 add_memory:*` census in §3.4 is over the 109 episodes that
still exist — 116 cited minus these 7. The figures reconcile exactly.)*
