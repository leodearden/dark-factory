# Memory referent fidelity — declared and derived entity referents for Graphiti writes

**Status:** active · authored 2026-08-05 · closes the item-2 decision on `esc-3375-1` (task 3375)

## Goal

A memory write that talks about `Task 3127` produces graph edges attached to the
`Task 3127` node — not to `Task 3129` because an LLM guessed. Concretely, after this
lands:

- An agent calls `add_memory(content="…Task 3127…", entities=[{"kind":"task","id":3127}])`
  and the resulting edge's endpoint is the `Task 3127` node, deterministically.
- An agent that declares a referent contradicting its own prose gets a structured
  error naming the conflict, not a silent bad write.
- An agent that declares nothing still gets correct endpoints, because the referent
  set is derived from the content by a deterministic scanner.
- An operator can query the declaration rate and see the transitional
  auto-derive fraction shrinking over time.

## Background

`esc-3375-1` reported a Graphiti edge whose fact was about `Task 3331` resolving onto
the pre-existing `Task 3330` node. A full-graph sweep on 2026-08-05 established the
mechanism and base rate. Measurements (all re-derived directly against FalkorDB, not
taken from the escalation record):

- **800** canonical `^Task N$` Entity nodes; **1,140** node names matching the task-label
  shape case-insensitively, over **1,067** distinct task numbers.
- **5 live** adjacent-number conflations out of **2,274** live task-mentioning edges
  (~0.22%); **16** over the graph's lifetime, 11 already tombstoned.
- The rate is **flat-to-declining**: 0.384% → 0.192% across exposure-matched halves of
  the graph's lifetime (Fisher p=0.266, n=12 — underpowered, but no rising trend).
- **38** normalized name keys carry more than one node (78 nodes). This matters
  causally, not just cosmetically — see "the coupling" below.
- **59.5%** of `dark_factory` episodes (1,567 / 2,635) cite a `Task N`-shaped reference
  in prose, overwhelmingly written by an agent that had just read that id as
  structured data from a tool result.

### Root cause, as read in code

1. `_MultiTenantFalkorDriver.build_indices_and_constraints` is a `pass` stub
   (`graphiti_client.py:291`) and the only index builder `_ensure_indices`
   (`:474`) calls exactly that method — then logs "Ensured indices". No fulltext index
   has ever existed on any graph in the instance.
2. So BM25 returns zero rows **silently, with no exception**, and
   `_collect_candidate_nodes` degrades to embedding-only against 800 `Task N` nodes at
   0.94–0.99 mutual cosine (0.046 separates rank 2 from rank 20).
3. `gpt-4o-mini` (`_resolve_with_llm`) picks a numerically-adjacent sibling. It is
   handed bare name strings — no summary, no attributes, because `entity_types` is
   always `None`.

graphiti's deterministic layer is **not** at fault: `Task 3331` vs `Task 3330` scores
Jaccard 0.333 against a 0.90 threshold. It correctly abstains and the LLM decides.

**Item 1 is not this PRD's problem.** The missing-index defect is real, systemic, and
being investigated separately; restoring BM25 would *not* have prevented this
conflation, because the query builder emits a union `(task | 3127)` and the rare token
matches zero documents when the node doesn't exist. See "Out of scope".

### Three failure modes, not one

Episode provenance separates them:

| Mode | Mechanism | Instances |
|---|---|---|
| (i) correct node absent | candidate pool held only siblings; LLM merged onto one | `a20a78a3`, `6719b8b2`, `bb4e1ca8` |
| (ii) correct node present | exact match was available and still lost | `88ec711b` (Task 1031 had existed 3 days) |
| (iii) intra-episode mis-pointing | both nodes minted by the *same* episode; edge extraction picked the wrong endpoint | `b901636b`, `f60f403a` |

`f60f403a` carries episode `8562760e` — the same episode that created both Task 3074
and Task 3075, 60 µs apart. Mode (iii) is immune to any entity-resolution guard,
because resolution never ran against an existing node. Only a post-write
content-vs-topology check catches it.

### The coupling

`dedup_helpers.py:220`: when `len(existing_matches) > 1`, the deterministic exact-match
fast path **abstains and punts to the LLM**. The key is `lower() + whitespace-collapse`.
With 38 such keys live, the under-merge defect permanently disables the deterministic
protection against the over-merge defect for those task numbers. One defect causes the
other. Repairing duplicate nodes is therefore not cosmetic hygiene — it restores a
guard.

### Why not fix this at the write substrate

`add_triplet` was evaluated as an LLM-free structured write path and **rejected**: it
routes every not-already-resolved entity through the same `dedupe_nodes` LLM prompt
(measured 9 LLM calls for 4 triplets on a scratch graph, with `Task 3627`/`Task 3628`
landing in the dead zone between the LSH candidate band and the 0.9 Jaccard threshold),
writes **zero** episode provenance (`add_nodes_and_edges_bulk(driver, [], [], …)`), has
no multi-tenant driver routing, and is unexported/undocumented/untested-in-CI.
`add_nodes_and_edges_bulk` is the correct substrate for a true structured path — but a
census found only **one** deterministic code path in the entire repo writes to a
Graphiti-primary category (`_emit_override_audit`, 16 of 2,713 episodes = 0.6%). A
triple-write migration would serve 0.6% of traffic. The leverage is in the 99.4% of
LLM-composed prose, which is why this PRD pins referents *alongside* prose rather than
replacing it.

## Sketch of approach

Prose stays. What changes is that every Graphiti write carries a **referent set** — the
canonical entities the content is about — and the post-write reconcile verifies that
every canonical endpoint the episode produced is in that set, repairing the ones that
aren't.

The referent set is populated from three sources, in precedence order:

1. **Declared** — a new `entities` parameter on `add_memory` / `add_episode`.
2. **Metadata-bridged** — `metadata.task_id`, which callers already set and which the
   Graphiti enqueue currently discards outright.
3. **Derived** — a deterministic scan of the content for canonical label shapes.

Verification runs as a new sub-pass of `_reconcile_episode_identity`, inside the
existing per-group identity lock, so no wrongly-attached state is ever externally
visible. Two complementary checks:

- **Set membership** — any `Task N` endpoint whose number is not in the referent set is
  a conflation. This catches all five measured live cases, whose defining signature is
  precisely that the landed-on number is never named in the fact.
- **Per-edge pairing** — the fact cites `Task M` but the endpoint is `Task N`, N≠M.
  This catches mode (iii), where both numbers are legitimately in the referent set.

Repair is `ensure_entity_node` (mint the correct node if absent) → `reassign_edge`
(lossless, atomic, stamps `reassigned_from_node_uuid`) → `refresh_entity_summary` (the
step today's invalidate-on-sighting posture omits, which is why `Task 3129`'s summary
still asserts the 3127 fact after that edge expired).

### The C′ veto folds in

An earlier design carried a separate "post-LLM veto" preventing `Task N` from merging
onto `Task M`. With the post-write placement chosen, that is not a second mechanism:
post-write, "extracted Task N was merged onto the Task M node" is observationally
identical to "an edge about Task M is attached to a Task N node", which the two checks
above already detect. Keeping it separate would create two sites that must agree
byte-for-byte (INV-5). It is therefore **not** a distinct leaf.

### Queue compatibility is free here

The referent set rides as an **additional key on the existing `add_memory_graphiti` /
`add_episode` payloads**, not a new operation. An old consumer ignores an unknown key;
a new consumer reading an old row finds it absent and treats it as "no referents",
which is exactly today's behaviour. No `payload_version`, no unknown-operation guard,
no migration. (Those would be required by a new queue *operation* — which the
out-of-scope triple-write path would need.)

## Resolved design decisions

1. **Post-write verification, not a graphiti_core monkeypatch.** `_resolve_with_llm` is
   a bare module-global call at `node_operations.py:424`, so `setattr` patching does
   work — but it is a private underscore API whose breakage on a 0.29 refactor would be
   **silent** (the wrapper simply stops being called). The post-write pass runs inside
   the same identity-lock critical section as the write, so it is equivalent from any
   external observer's point of view, with zero upgrade coupling.
2. **`entities` is optional, tri-state.** `None` = never considered; `[]` = considered,
   none apply; `[…]` = declared. It is **not** "all entities" — graphiti legitimately
   extracts `PRD decision D2`, `gitignore`, `MergeWorker`, which no caller can predict.
   Requiring a non-empty list would produce garbage declarations. Requiring the
   parameter in the tool signature would break every non-upgraded agent across `reify`,
   `know_live`, `autopilot_video`, and `solar_challenge_platform`.
3. **The gate rejects on conflict, not on absence.** If the caller declared nothing,
   auto-derive silently. If the caller declared something and the scan disagrees,
   reject with a structured hint (the INV-1 house pattern: ValidationError+hint at the
   submit boundary). Rejecting on mere absence would lose memories from agents that
   don't retry — notably `/reflect` at session end.
4. **Silent auto-derive is a transitional phase, not the SOP.** It weakens the feature
   if it becomes permanent. Declaration rate is tracked from day one, and a **dated
   census task** fires ~7 days after landing to report the undeclared fraction and
   drive the tightening decision. This is the INV-7 bound on the transitional state:
   the hold has a named owner and a deadline.
5. **One canonical-label vocabulary, one site.** The label regex would otherwise exist
   at four sites (`task_naming.canonicalize_task_node_name`, `cross_project_refs`, the
   new scanner, the verifier) needing byte-for-byte agreement. INV-5 requires
   extraction. `canonicalize_task_node_name`'s pattern `^\s*tasks?\s+(\d+)\s*$` is also
   structurally unable to match `task #1153`, one of the 53 measured variant splits —
   so the extraction is a fix, not just a refactor.
6. **Do not assume `task/3335` lands.** Its two leaves are cherry-picked as an explicit
   prerequisite. Empirically verified: the four constituent commits apply to current
   main with zero conflicts. The branch itself is unclaimed 3 days, never merged, no
   `verified_green`, 580 commits behind, and its tip is a pyright fix rather than a
   done-checkpoint. 3335 retains ownership of its `_split_cross_project_task_nodes`
   sub-pass; this PRD takes only the two dependency-free primitives.
7. **Set membership AND per-edge pairing, both.** Neither alone is sufficient: set
   membership misses mode (iii) (both numbers legitimately declared), per-edge pairing
   misses conflations whose fact cites no number the scanner can see.
8. **Scanner is precision-over-recall, with an explicit blind list.** It will not see
   bare-digit node names (`1251`), title-only references, or Greek-letter aliases
   (`Task θ2=2184`) — all measured as present. The signal is scoped accordingly; we do
   not claim completeness.

## Pre-conditions for activating

- **α (in this batch)** — `cross_project_refs.py` and `GraphitiBackend.ensure_entity_node`
  cherry-picked onto main. Everything downstream of the repair path depends on it.
  The four constituent commits on `task/3335`, in order:
  `104ce86ded` (RED: cross_project_refs tests) · `a930dc7111` (GREEN: cross_project_refs.py)
  · `c5629b3e65` (RED: ensure_entity_node tests) · `3432e0e284` (GREEN: ensure_entity_node).
  Verified 2026-08-05 against main tip `52e27ff13f`: all four apply with **zero conflicts**
  (one clean auto-merge in `graphiti_client.py`), 829 net inserted lines across 4 files,
  all parsing. Re-verify before applying — main moves. Do **not** take the branch's fifth
  concern (`_split_cross_project_task_nodes` in `memory_service.py`); that stays with 3335.
- Verified present on main, no prerequisite needed: `reassign_edge`
  (`graphiti_client.py:989`), `_resolve_or_create_entity` (`:2251`),
  `get_nodes_by_exact_name` (`:2043`), `find_duplicate_entity_nodes` (`:2087`),
  `refresh_entity_summary` (`:2439`), `canonicalize_task_node_name`
  (`utils/task_naming.py:26`), `_reconcile_episode_identity` (`memory_service.py:2075`),
  `_identity_lock_for` (`graphiti_client.py:438`), the `add_memory` gate stack
  (`server/tools.py:1933`).

## Cross-PRD relationship

| Other PRD / work | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/fm-memory-identity-prd.md` (W6, shipped) | extends | `_reconcile_episode_identity` sub-pass chain; `_identity_lock_for` contract | **this PRD** (W6 is shipped and closed) | wired |
| task 3335 / `task/3335` | consumes | `ensure_entity_node`, `cross_project_refs.find_cross_project_task_refs` | **this PRD** owns the cherry-pick (leaf α); 3335 retains `_split_cross_project_task_nodes` | queued |
| `plans/cross-graph-entity-leak-prd.md` | adjacent | `group_id` routing / `_canonicalize_group_args` | cross-graph-leak PRD | unchanged |
| `esc-3375-1` / task 3375 | closes | the item-2 "systemic mitigation or not" decision | **this PRD** (leaf θ) | queued |
| Missing-index investigation (`_ensure_indices` stub) | independent | `build_indices_and_constraints` | separate work — explicitly not this PRD | out of scope |

## Contract (B+H)

G5 fires: mechanism count ≥ 8, load-bearing seam (the memory write path), cross-PRD
consumers ≥ 2.

### Referent vocabulary

```python
# fused_memory/utils/canonical_labels.py  — the single normative site (INV-5)

@dataclass(frozen=True)
class Referent:
    kind: str          # 'task' today; registry-extensible ('escalation' next)
    project_id: str    # canonicalized; own-project by default
    number: str        # digits VERBATIM, never int-normalized ('0132' stays '0132')

    @property
    def node_name(self) -> str: ...   # 'Task 132' own-project; 'reify:132' cross-project

@dataclass(frozen=True)
class LabelScan:
    refs: list[Referent]
    ambiguous: list[Referent]   # consumer refuses rather than guessing

def parse_node_name(name: str) -> Referent | None: ...
def scan_content(content: str, *, group_id: str,
                 known_project_ids: Collection[str] | None = None) -> LabelScan: ...
```

Invariants:
- `parse_node_name` is **anchored**: `Task 3331 dashboard index` returns `None`.
- Digits are preserved verbatim; `Task 0132` and `Task 132` are distinct referents.
- A `reify:` or other project qualifier is a **different-project signal**, never
  normalized away.
- `scan_content` is precision-over-recall: source locations (`file.py:2091`), URL
  authorities, and clock times must not match.
- `canonicalize_task_node_name` and `cross_project_refs` are refactored to call this
  module; no second copy of the pattern survives.

### Referent-set resolution

```python
def resolve_referents(*, declared: list[dict] | None,
                      metadata: dict,
                      content: str,
                      group_id: str) -> ReferentResolution
# .referents        — the effective set
# .source           — 'declared' | 'metadata' | 'derived' | 'none'
# .conflicts        — declared refs contradicted by the scan  (gate rejects on these)
# .ambiguous        — scan could not disambiguate            (drops to manual)
```

Precedence: declared > metadata-bridged > derived. `source` is what the declaration-rate
telemetry counts.

### Verification sub-pass

```python
async def _verify_episode_referents(result, *, group_id: str,
                                    referents: ReferentSet) -> ReferentStats
```

Runs **last** in `_reconcile_episode_identity` — after `_normalize_task_node_names`,
which produces the canonical `Task N` names this pass keys on, and after 3335's
cross-project split if that ever lands. Ordering is load-bearing.

Postconditions:
- Every edge endpoint whose name parses as a canonical task referent is either in
  `referents`, or has been repointed to a node that is.
- Every repair emits a **structured** record (edge uuid, old endpoint uuid, new endpoint
  uuid, referent set, which check fired) — not a log line (INV-2).
- Repairs are counted; a consecutive-streak threshold escalates (INV-4). A repair
  storm means the scanner or the resolver has regressed, and must not be absorbed
  silently.
- A repair that cannot be resolved unambiguously (unary fact with no correct target,
  e.g. `b901636b`) is **recorded and left alone**, never guessed at.

### Boundary-test sketch

| Scenario | Preconditions | Postconditions |
|---|---|---|
| Declared referent, correct endpoint | `Task 3127` node exists; write declares `task:3127` | edge endpoint uuid == the `Task 3127` node; no repair recorded |
| Mode (i): correct node absent | no `Task 3127` node; content cites it; `Task 3129` exists | `Task 3127` node minted; edge endpoint is it; repair record emitted with check=`set-membership` |
| Mode (ii): correct node present, LLM picks sibling | both `Task 1030` and `Task 1031` exist; fact is about 1031 | edge repointed to `Task 1031`; `reassigned_from_node_uuid` stamped; both summaries refreshed |
| Mode (iii): both nodes in referent set | one episode mints `Task 3074` + `Task 3075`; fact cites 3075, lands on 3074 | set-membership passes; **per-edge pairing** fires; edge repointed to `Task 3075` |
| Declared/prose conflict | content cites `Task 3127`; caller declares `task:3129` | `add_memory` returns a structured error naming both; **no write occurs** |
| Undeclared, derivable | content cites `Task 3127`; `entities` omitted | write succeeds; `source='derived'`; declaration-rate counter increments the undeclared bucket |
| Undeclared, underivable | content says "the merge-lane hardening task" | write succeeds; `source='none'`; no repair attempted; counted |
| Ambiguous scan | same number appears both bare and project-qualified | ref routed to `.ambiguous`; treated as undeclared; recorded, not guessed |
| Unary fact, no correct target | fact is unary about 2519, endpoint is `Task 2520` | flagged, recorded, **left unrepaired** |
| Repair storm | 20 consecutive writes each needing repair | streak escalation fires; repairs continue |

## Decomposition plan

| Label | Title | Modules | Kind | Observable signal | Prereqs |
|---|---|---|---|---|---|
| **α** | Cherry-pick `cross_project_refs` + `ensure_entity_node` onto main | `fused-memory/{utils,backends,tests}` | intermediate | unlocks γ, η — the mint and scan primitives exist on main | — |
| **β** | Extract one canonical-label vocabulary; refactor `task_naming` + `cross_project_refs` onto it | `fused-memory/utils` | intermediate | unlocks γ, ζ; `task #1153` now parses where `canonicalize_task_node_name` could not | α |
| **γ** | Deterministic referent scanner + `resolve_referents` precedence | `fused-memory/utils`, `services` | intermediate | unlocks δ, ε | α, β |
| **δ** | `entities` param on `add_memory`/`add_episode` + metadata.task_id bridge + `_entities_gate` | `fused-memory/server/tools.py`, `services` | **leaf** | `add_memory(content="…Task 3127…", entities=[{"kind":"task","id":3129}])` returns a structured error naming the conflict; the write does not occur | γ |
| **ε** | Thread the referent set through the queue payload into `_execute_graphiti_write` | `fused-memory/services` | intermediate | unlocks ζ; an old-format queue row still executes unchanged | γ |
| **ζ** | Verification sub-pass: set-membership + per-edge pairing, structured records | `fused-memory/services/memory_service.py` | intermediate | unlocks η, ι | β, ε |
| **η** | Repair path: `ensure_entity_node` → `reassign_edge` → `refresh_entity_summary`, + streak escalation | `fused-memory/services`, `backends` | **leaf** | write a memory citing `Task N` where only `Task N±1` exists; `get_entity(name="Task N")` then returns the fact, and the `Task N±1` summary no longer contains it | α, ζ |
| **θ** | Remediate the 5 live conflations + refresh contaminated summaries; close `esc-3375-1` | operational | **leaf** (`task_kind=deterministic`) | `get_entity` for Tasks 2434/3220/1030/3074 no longer returns the foreign facts; esc-3375-1 resolved with the measured before/after | η |
| **ι** | Declaration-rate telemetry surfaced through the read path | `fused-memory/services`, dashboard | **leaf** | an operator can read declared / metadata / derived / none counts per project over a window | ζ |
| **κ** | Drive adoption: CLAUDE.md, `/memory`, `/reflect`, recon stage prompts, review-checkpoint template | docs, `skills/`, `reconciliation/prompts` | **leaf** | the documented `add_memory` examples show `entities=`; a fresh session following `/memory` declares referents | δ |
| **λ** | **Dated** undeclared-referent census (+7 days) → tightening decision | operational | **leaf** (`metadata.milestone` delayed) | a census report giving the undeclared fraction over ≥7 days of production data, with a recommendation to tighten or hold | ι, κ |

**G7 walk.** `storm-escape-required` — η carries the repair streak counter; δ's gate
rejections are counted by ι. `no-lockstep-duplication` — β exists specifically to
prevent the four-site regex; C′ was folded into ζ for the same reason.
`structured-facts-at-failure` — ζ emits structured repair records rather than the
`logger.debug`-only shape `ReconcileStats` currently uses. `corroborate-before-acting`
— η repoints via `reassign_edge`, which re-reads endpoints from topology and no-ops if
already moved. `holds-owned-and-bounded` — λ is the dated bound on decision 4's
transitional phase.

## Out of scope for this PRD

- **Full structured triple writes.** `add_nodes_and_edges_bulk` (`utils/bulk_utils.py:128`)
  is the right substrate — genuinely LLM-free, transactional, MERGEs on caller-supplied
  uuids, and accepts `episodic_nodes`/`episodic_edges` so provenance survives. It serves
  0.6% of current write traffic. Revisit when a deterministic writer population exists.
- **The missing-index defect.** `_ensure_indices` calling its own no-op override is a
  real, systemic, instance-wide defect, but it would not have prevented this conflation.
  Separate work, separate risk profile (the override exists to stop an index storm, and
  fulltext previously dead-lettered the write path — task 3334 / dead-letter 9950).
- **Derived (uuid5) entity uuids.** Attractive but incompatible with the corpus: the
  ~800 existing `Task N` nodes carry `uuid4`s, so a derived scheme would miss every one
  and mint duplicates en masse. Viable only behind a full backfill.
- **`entity_types` typed extraction.** Plumbed through `graphiti_client.add_episode` and
  never once called with a non-`None` value. Would improve the LLM's context but keeps
  the decision probabilistic; this PRD removes the decision instead.
- **Escalation-id referents.** The registry in β is built to extend, and esc-ids show the
  same signature (1/204), but shipping one kind first keeps the verification surface small.
- **Bulk remediation of the 38 duplicate-name keys.** `merge_entities` is irreversible,
  has no type check, discards the deprecated summary, and hard-deletes parallel
  duplicates leaving only a count in a best-effort journal. Needs per-instance human
  review; θ covers only the 5 conflations.

## Open questions (tactical)

1. **Streak threshold for the η repair storm escalation.** Base rate is ~0.22%, so any
   sustained streak is anomalous. **Suggested resolution:** reuse the consecutive-streak
   gate shape from `merge_liveness.py`; start at 10 consecutive repairs. Decide in η.
2. **Where declaration-rate counters live.** `write_journal.db` `write_ops` has free-text
   `operation` and JSON `params`, so no migration is needed either way. **Suggested
   resolution:** `write_ops.params`, read by a dashboard query. Decide in ι.
3. **Whether `add_episode` gets `entities` in the same leaf as `add_memory`.**
   `add_episode` explicitly does not persist metadata today. **Suggested resolution:**
   both in δ; split only if the gate stack diverges. Decide in δ.
4. **New `EventType` member for referent repairs.** The enum has 8 members and
   reconciliation consumers key off it. **Suggested resolution:** reuse
   `memory_updated`; add a member only if a consumer needs to filter. Decide in ζ.
