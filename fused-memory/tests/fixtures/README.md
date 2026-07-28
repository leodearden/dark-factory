# fused-memory test fixtures

Committed fixture data for `fused-memory/tests/`. Each fixture carries
per-record `provenance` so any label can be audited back to the artifact
that produced it.

---

## `write_triage_calibration.jsonl`

A labeled corpus of memory entries from the 2026-07-27 `reify` curator
milestone-gate session, used by
`fused-memory/scripts/calibrate_write_triage.py` to derive the write-triage
band thresholds `T_high` / `T_low` from **measured** similarity
distributions (PRD `docs/prds/memory-write-path-convergence.md` §9 leaf α,
contract C1, decision D1).

The point of the fixture is that the labels are *curator ground truth*: a
human-in-the-loop adjudicated, for each cluster of similar memories,
whether the members were genuine rediscoveries of one fact (duplicates) or
genuinely different claims that merely look alike (the hard negatives). No
threshold is assumed anywhere; the fixture supplies the population the
thresholds are measured over.

### Record schema

One JSON object per line:

```
{
  "memory_id":  "<full 36-char UUID>",       // the Mem0 entry id
  "content":    "<verbatim stored text>",    // required, non-empty
  "category":   "<fused-memory category>",   // e.g. procedural_knowledge
  "agent_id":   "<writer agent id|null>",
  "created_at": "<ISO-8601 timestamp|null>",
  "cluster_id": "<full 36-char UUID>",       // the CANONICAL's memory_id
  "label":      "duplicate" | "canonical" | "distinct" | "pseudo_contradiction",
  "provenance": {
    "gate_id":         "esc-NNNN",           // adjudicating milestone gate
    "transcript_line": <int>,                // 1-based line the content came from
    "source":          "<extraction route>"  // see "Fidelity order" below
  }
}
```

`cluster_id` is the **canonical entry's UUID, never the gate id.** Gates
`esc-5534`, `esc-5547` and `esc-5561` each produced *two* canonicals, so
keying clusters by gate would fuse two unrelated canonicals' members into
one cluster and inject pairs that are not duplicates into the positive
class — which would drag the derived `T_high` downward and manufacture
exactly the deterministic-band false positives the calibration exists to
measure. `gate_id` is retained only as per-record provenance.

### Labels

| label | meaning |
|---|---|
| `canonical` | the surviving entry the curator wrote to replace a cluster |
| `duplicate` | a member the curator merged into / superseded by that canonical — a curator-confirmed genuine rediscovery (the positive class) |
| `distinct` | same cluster, but the curator ruled the entries are *not* the same claim (`esc-5606`: "they are not three competing answers to one question") — a hard negative |
| `pseudo_contradiction` | entries that read as contradictory but were adjudicated *both correct* (`esc-5557`, `esc-5626`: the contradiction was an omission, not a disagreement) — a hard negative |

### Derivation provenance

Both source artifacts are **machine-local and outside this repo**, which is
precisely why the extracted fixture is committed rather than re-derived at
test time. The tests must run in CI with no access to either path.

**(a) Content / category / agent_id / created_at** — extracted from the
curator session transcript:

```
/home/leo/.claude/projects/-home-leo-src-reify/bceaf4a6-d79e-44f3-8422-b152906f70cb.jsonl
```

`tool_use` blocks pair to `tool_result` blocks on
`tool_use.id == tool_result.tool_use_id` (293 ↔ 293, zero orphans).
Note that `category` is **not** a top-level key on a
`mcp__fused-memory__get_memory_by_id` payload — it lives at
`metadata.category`, alongside `metadata.agent_id` / `metadata.created_at`.

Fidelity order for content recovery (recorded per record in
`provenance.source`):

| source | notes |
|---|---|
| `get_memory_by_id` | full record; highest fidelity |
| `search` | result rows carry content + category |
| `get_memories_by_metadata` | content lives at `metadata.data`; there is **no** top-level `content` key |
| `add_memory_input` | the curator's own write, recovered from the call input |

Gotcha: three Stage-1 suppression records have a `metadata.data` that is
only a short `STAGE 1 FLAG SUPPRESSION task_id=...` stub while the real
substance sits in `metadata.reason`. Both are read, or those records are
under-recovered.

**(b) Cluster membership and disposition** — derived from the explicit
edges the curator wrote into `add_memory` metadata in the same session
(`merged_from`, `supersedes`, `retired_suppressions`), each canonical
identified by `metadata.source == 'curator_gate_NNNN'`. These edges
attribute each member to a *specific* canonical, which is what makes
canonical-keyed clusters machine-exact.

The 21 resolved `milestone_gate` escalations at
`/home/leo/src/reify/data/escalations/archive/2026-07-27/esc-*.json`
(gates 5534, 5541, 5547, 5552, 5557, 5560, 5561, 5563, 5564, 5567, 5571,
5600, 5603, 5606, 5610, 5622, 5626, 5631, 5634, 5636, 5637) are the
cross-check for the same information; the 19 gates fetched in-session are
inlined in the transcript verbatim. **`resolution_class` and `root_cause`
are null on all 21** — in the archive store *and* in-transcript — so the
`distinct` / `pseudo_contradiction` dispositions were curated from the
`resolution` prose by hand, not machine-parsed. That is why every record
carries a `provenance` block.

### Exclusions

Deliberately **not** emitted:

- `8d79e0e4…` — an intra-session canonical: created and then superseded
  within the same session, so it is an artifact of the curation run rather
  than a member of the pre-existing population.
- `43a47400…` — a spent Stage-1 flag marker: a meta-record *about* a
  memory, not a topical entry, so it has no meaningful place in a
  similarity distribution.
- Any id whose content could not be recovered from the transcript. These
  are **excluded and tallied**, never emitted with placeholder content —
  a placeholder would corrupt the measured similarity distributions.

### Coverage

Measured at extraction (`step-2`); see `Known label ambiguities` below for
the caveats these counts carry.

<!-- COVERAGE:BEGIN -->
- Delete calls issued in session: _TBD_
- Ids with recovered content: _TBD_
- Records emitted: _TBD_
- Unrecovered (excluded): _TBD_
- Clusters: _TBD_
- Per-label counts: _TBD_
<!-- COVERAGE:END -->

There is deliberately **no test asserting a total record count.** The three
available counts disagree — the session issued 90 `delete_memory` calls,
the PRD says 89, and the topical population differs again after the
exclusions above — so any pinned total would be a test that could not be
made green without falsifying the data. The tests assert label and
referential invariants instead (one canonical per cluster, no dangling
members, the `esc-5606` distinct-triple, the `esc-5557`/`esc-5626`
pseudo-contradictions), which are the properties the calibration actually
depends on.

### Known label ambiguities

Recorded rather than silently guessed:

- **`esc-5561`** produced two canonicals (`0e81aa96…`, `c8ca5f55…`) but its
  9 members are all attributed to `0e81aa96…`; the 9-to-2 split is not
  machine-resolvable from the session edges.
- **`c759c53b…` / `9f2d2ae6…`** carry a secondary cross-reference from
  `esc-5600` in addition to their primary cluster attribution.
