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
`esc-5534`, `esc-5547`, `esc-5561` and `esc-5610` each produced *two*
canonicals, so keying clusters by gate would fuse two canonicals' members into
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
**Records emitted: 104** across **20 clusters**.

| label | count |
|---|---|
| `duplicate` | 75 |
| `canonical` | 20 |
| `pseudo_contradiction` | 6 |
| `distinct` | 3 |

Recovery source, per record:

| `provenance.source` | count |
|---|---|
| `get_memory_by_id` | 69 |
| `add_memory_input` | 20 (the canonicals) |
| `search` | 11 |
| `get_memories_by_metadata` | 4 |

Accounting against the session's 90 `delete_memory` calls — every id is
either emitted or explicitly tallied, none silently dropped:

| disposition | n | why |
|---|---|---|
| emitted as a cluster member | 84 | |
| excluded — intra-session canonical | 1 | `8d79e0e4…` |
| excluded — spent Stage-1 flag marker | 1 | `43a47400…` |
| excluded — canonical unrecoverable | 3 | gate `esc-5571` (below) |
| excluded — ambiguous attribution | 1 | `97557618…` (below) |
| **total** | **90** | |

Plus the 20 canonicals themselves (each keying its own cluster) = 104
records. **Content recovery was complete: 0 records were dropped for
unrecoverable content**, so no record carries placeholder text.
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
- **`97557618…` is excluded, not coin-flipped.** It is named in the
  `merged_from` of *both* `esc-5547` canonicals (`d4b39613…` at transcript
  line 386, `a063640d…` at line 389) — the only id in the session whose
  edges name two owners. Forcing it into one cluster would make its pairs
  against the other canonical's members read as *unrelated* negatives when
  they are in fact near-duplicates; because `T_high` must exceed every
  measured negative, one spuriously-high negative can drag the headline
  threshold up on its own. Excluding one record is the cheaper error.
- **`esc-5571`'s cluster is absent entirely.** Its canonical `add_memory`
  call (transcript line 527) returned `"The operation timed out."` rather
  than a `memory_ids` payload, and the curator never retried it, so the
  canonical's UUID is unrecoverable. Its 3 `retired_suppressions` members
  (`48c5cba2…`, `f5a52915…`, `f0cb8363…` — the Stage-1 stub records) are
  therefore dropped rather than emitted against a fabricated cluster id,
  which would have violated the no-dangling-members invariant.
- **The 6 members of the excluded `8d79e0e4…` are re-attributed to
  `bf91bc5c…`**, the `esc-5610` canonical that superseded it within the
  same session. An excluded canonical never takes ownership, so its
  members fall through to the surviving one rather than being orphaned.

---

## `memory_eval_topic_registry.json`

The probed-topic registry for the **E1 retrieval-health** eval
(`docs/prds/memory-eval-program.md` §5 leaf β, task 3208). Read by
`fused-memory/scripts/memory_eval_retrieval_probe.py` via
`load_topic_registry()`; its shape is contract-tested in
`fused-memory/tests/test_memory_eval_retrieval_probe.py`.

### Purpose

Each entry names a topic, the memory that topic's queries are *expected* to
return (its **canonical**), several query phrasings, and the substantive
claims that should come back. The probe issues every phrasing and records
what the store returned; it emits measurements only and never evaluates a
limit (D1/G6 — thresholds, grandfather sets and alarms all live in the
limits evaluator, not here).

### Record schema

Top level is `{"schema_version": 1, "_disclosures": {...}, "entries": [...]}`.
`_disclosures` is described under "What the registry does **not** cover"
below. Per entry:

| field | meaning |
|---|---|
| `topic` | Stable slug. **This is a persisted key** — the tripwire item key is `t-<topic>`, which the grandfather set stores. Renaming one silently re-releases a grandfathered item. |
| `project_id` | `dark_factory` or `reify` — which corpus to probe. |
| `derived_from` | Provenance tag: `curator_gate`, `census_topic`, `topic_guard_cluster`, `briefing_query`, `hand`. Lets a later run tell which entries auto-derivation has taken over from hand-authoring. |
| `canonical.content_hash` | `content_key()` of the expected entry — sha256 of whitespace-normalised content, `hexdigest()[:16]`. The **primary** matcher. |
| `canonical.last_known_id` | Fallback matcher. Memory UUIDs rot on re-consolidation (D5), so the probe tries the hash first and discloses which matcher fired. |
| `canonical.content_prefix` | Human anchor for the report. **Never** used for matching. |
| `phrasings[]` | `{text, held_out}`. At least three per topic, at least one `held_out`. |
| `claim_queries[]` | `{query, needles}`. A claim is recalled when **all** needles appear in some returned entry — deliberately weaker than canonical identity, so a consolidation that moved a claim into a different entry does not read as knowledge loss. |
| `members[]` | Content hashes of entries the curator adjudicated as the same claim. |
| `supersedes_pairs[]` | `{superseded_hash, successor_hash}`, recorded **offline**. |

Unknown keys on an entry load untouched (the loader is required-strict /
additive-tolerant), so 3201's widened derivation is an improvement rather
than a fixture rewrite.

### Why `held_out` exists

A held-out phrasing is authored **fresh** for this eval and was never used to
write, consolidate or retrieve the entry it probes. Without it, a fix that
tunes the known phrasings saturates canonical-presence and the metric stops
discriminating (the Goodhart guard, D5). Three invariants are test-enforced:
no held-out phrasing may duplicate a tuned phrasing anywhere in the registry,
none may equal a `_default_topic_guard_clusters()` phrase (those phrases were
used to *build* the entries they would retrieve), and held-out phrasings are
unique across topics.

### Why `supersedes_pairs` is recorded, not parsed

The probe never reads `metadata.supersedes`. That parser is
`normalize_supersedes()` (task 3196), leaf γ/E4's hard dependency, and a
second one here would be exactly the lockstep duplication INV-5 forbids. The
relation is therefore recorded at derivation time from committed sources, and
the runtime metric reduces to "is `index(superseded) < index(successor)` in
this one result list" — no pointer-shape knowledge at runtime at all.

### Provenance (32 topics)

- **20 `curator_gate`** — one per adjudicated cluster in
  `write_triage_calibration.jsonl` (17 `esc-55xx`/`56xx` gates). The
  `canonical`-labelled row is the topic canonical; `duplicate`-labelled rows
  become `members` and `supersedes_pairs`. Because that fixture commits
  `content` verbatim, **every hash here is computable offline** — no Qdrant,
  no embedder, no `OPENAI_API_KEY` — so a reviewer can re-derive and audit any
  entry.
- **4 `topic_guard_cluster`** — slugs from
  `fused_memory/config/schema.py:_default_topic_guard_clusters()`.
- **4 `census_topic`** — multi-entry topics from
  `plans/memory-metadata-census-report.json`.
- **1 `briefing_query`** — `g7-design-invariants`, carrying the four
  briefing-assembler queries (`briefing.py:978-1013`) as its phrasings. This
  is the highest-leverage query surface in the system: those four run against
  every dispatched task's context window.
- **3 `hand`** — single-entry dark_factory topics.

### What the registry does **not** cover (`_disclosures`)

32 topics is a *selection*. `scripts/memory_eval_retrieval_probe.py
--derive-registry` emits 74 candidates from the committed offline sources,
and the census tail it never offered at all is larger still. Every one of
those narrowings is recorded in the top-level `_disclosures` block and
rendered into the run report's registry-composition section, because a
narrowing nobody can see reads downstream as "there was nothing there".

| key | meaning |
|---|---|
| `curator_gate_clusters` / `census_topics_emitted` / `topic_guard_clusters` | Candidates derivation produced, per source. |
| `census_topics_skipped_singleton` | Census topics with `count <= 1`. A one-entry topic answers "is the canonical in the top k" by that entry's mere existence — presence, not retrieval. |
| `curator_clusters_without_canonical` | Calibration clusters carrying no `canonical`-labelled row, so no entry could be built. |
| `census_rows_malformed_value` / `census_rows_malformed_count` | Census rows whose `value` / `count` was the wrong shape. Counted **separately** from the singleton skip: a malformed row is a broken census, a singleton is a healthy one — folding them together reports a schema break as a corpus property. |
| `slug_collisions_dropped` | Candidates whose slug collided with one already emitted. |
| `derived_candidates_not_carried` | Candidates derivation emitted that this fixture does **not** contain. Mostly census topics whose canonical `content_hash` is unknown offline (derivation leaves it empty) and which no operator has resolved against a live read-only scroll yet. |

Regenerate the derivation half with `--derive-registry` and merge; the flag
prints and never overwrites, because the hand-authored held-out phrasings
are the part a machine cannot regenerate.

A `_disclosures` value that is not an integer is a **named load failure**,
not a silently dropped key — dropping it would erase the record that a
narrowing happened, which is exactly the state the block exists to prevent.

The `topic_guard_cluster`, `census_topic`, `hand` and `briefing_query`
canonicals were resolved by a **read-only Qdrant payload scroll** on
2026-07-30 (no embedder, no writes), because unlike the curator clusters their
content is not committed anywhere in this repo. Their hashes are therefore
re-derivable only against a live store; the curator-gate 20 are not.

### Exclusions

- **`architect-plan-revalidation-requeue-lock`** is a
  `_default_topic_guard_clusters()` slug but is **not** a registry topic: its
  named canonicals (`6a96a020…`, `974b0adb…`) were not resolvable in the
  scroll, and inventing a hash that can never match would manufacture a
  permanent tripwire failure indistinguishable from a real retrieval defect.
  Four of the five guard slugs are covered.
- **`distinct` and `pseudo_contradiction` rows are not members.** The curator
  adjudicated them as separate claims that only *read* as contradictory;
  folding them in would poison the contamination metric with entries that
  legitimately answer a different question.
- **Four curator canonicals no longer resolve by UUID**
  (`0e954870…`, `168c3a6b…`, `417d86d0…`, `c8ca5f55…`) — they have rotated
  since the calibration session. They are kept deliberately: their
  content hashes are still valid, so they exercise the hash-primary /
  id-fallback matcher on real decay rather than on a synthetic case.

---

## `transcript_corpus/`

A miniature **agent-transcript archive** for the retro replay-corpus
extractor (`docs/prds/memory-eval-program.md` §5 leaf θ, decision D9, task
3214). Consumed by `fused-memory/scripts/memory_eval_transcript_corpus.py`
(via `--archive-root` and `--transcript`) and asserted end-to-end in
`fused-memory/tests/test_memory_eval_transcript_corpus.py`.

Unlike the two fixtures above this one is a directory of JSONL files, not a
single file: the extractor's provenance parsing reads task/session/subagent
identity **out of the path**, so a flat fixture could not exercise it.

### Purpose

The fixture is the committed, reviewable stand-in for the live archive at
`<main-checkout>/data/orchestrator/agent-transcripts/` — untracked runtime
state that exists on one machine, cannot be committed, and grows under the
fleet. It is a hand-written miniature of the record **shapes** measured
there, deliberately **not** a pasted real transcript: a real one carries
incidental session content that would drift, bloat the diff, and make a
failure hard to read.

Its round-trip is one half of the capability manifest's
`coverage-report-discloses-failures` check
(`docs/prds/memory-eval-program.capability-manifest.yaml`); the other half is
an all-unparseable-input case, which is built in a temp dir because a
deliberately-corrupt transcript is not something to commit.

### Layout

Mirrors the two path shapes `shared/src/shared/transcript_archive.py`
writes:

```
transcript_corpus/
└── 4242/                                            # <task_id>
    └── -home-leo-src-dark-factory--worktrees-4242/  # <encoded cwd>
        ├── 11111111-…-555555555555.jsonl            # main session
        └── 11111111-…-555555555555/
            └── subagents/
                └── agent-abc123def4567890.jsonl     # subagent session
```

Task id, session id, `is_subagent` and the subagent id are all recovered
from these path components — never from the record bodies — so renaming a
directory changes what the extractor reports.

**Why plain `.jsonl` and not the live archive's `.jsonl.gz`.** The
extractor accepts both suffixes (`scan_archive` globs `*.jsonl.gz` and
`*.jsonl`; `load_transcript` / `iter_json_lines` pick the reader off the
suffix), so the committed copy is uncompressed for two reasons. First,
`.gz` is a binary blob: a fixture whose whole value is that its authored
facts are *reviewable* must be readable in a diff, and the line-by-line
table below is only auditable against a plain file. Second, the
`gz` extension is absent from the α/γ lock-charter extension allowlist
(`shared/src/shared/locking.py::CODE_EXTENSIONS`, mirrored in
`fused_memory.middleware.lock_charter_guard` and reify's
`scripts/lock-charter-guard.sh`), so committing a tracked `.gz` fails
`fused-memory/tests/test_lock_charter_guard.py
::test_every_tracked_extension_is_allowlisted` — widening that vector is a
coordinated cross-repo change (α leads, γ mirrors) that this fixture does
not need. The **gzip** read path is still covered end to end: every
archive-scan test writes real gzipped transcripts into a temp tree
(`_write_transcript`), including the corrupt-gz and partially-written
cases.

### What each line exercises

The main transcript is 9 physical lines yielding 7 records; the two missing
ones are the point:

| line | record | exercises |
|---|---|---|
| 1 | `{"type": "queue-operation", …}` | a non-message record interleaved mid-session — must be skipped, not crashed on |
| 2 | first `user` record | the briefing **Agent Identity** line, `agent_id:** ` + backticked `claude-task-4242-architect` — the only place caller identity is recoverable (see below) |
| 3 | *(blank)* | reader-level skip — **not** a parse failure |
| 4 | `{"type": "user", "message": {broken` | corrupt JSON — reader-level skip, **not** a parse failure. A fire-and-forget writer can truncate its last line; that must not lose the whole file, and must not inflate `parse_failures` (which counts unreadable **files**) |
| 5 | `assistant` + `tool_use` | an `mcp__fused-memory__search` call: `input.query` / `project_id` / `limit` |
| 6 | `user` + `tool_result` | its answer — `content` is a JSON **string** decoding to `{"results": [...]}`, two results at **distinct** `relevance_score`s so rank order is observable |
| 7 | `{"type": "queue-operation", …}` | a second interleaved skip, *between* a search and its successor |
| 8 | `assistant` + two `tool_use` | `Read` and `mcp__fused-memory__add_memory` — non-search tools that must contribute **nothing** |
| 9 | `assistant` + `tool_use` | a second search with **no** matching `tool_result` anywhere: the truncated-transcript case, which must be emitted with `result_status: "missing"` rather than dropped |

The subagent transcript is 3 records: a briefing whose agent_id
(`claude-task-4242-code-reviewer`) has a **hyphenated role**, plus one
answered search. Its records also carry `isSidechain` / `agentId` /
`gitBranch`, mirroring the live shape — the extractor ignores them and
derives `is_subagent` from the path, so the fixture keeps them available for
a future cross-check without asserting one today.

### The `caller` name collision

Every search `tool_use` block in the fixture carries its own
`"caller": {"type": "direct"}`, exactly as the live archive does. That is a
Claude Code harness field, uniform across every measured call, and it is
**not** agent identity — it is reproduced here precisely so the extractor is
tested against the collision. The corpus record's `caller` field means *who
issued the search*, recovered from the briefing text on line 2.

### Record schema

Unlike the two fixtures above, this fixture's own records are *transcripts*
(shapes tabulated under "What each line exercises"). The schema worth stating
here is the one the round-trip **produces** from them — the corpus record. The
normative definition, field by field, lives in the extractor's module
docstring (`fused-memory/scripts/memory_eval_transcript_corpus.py`, "Record
schema"); this is the reader's-digest version, kept here because the
round-trip test asserts against it.

One JSON object per line, keys sorted:

| field | meaning |
|---|---|
| `schema_version` | `SCHEMA_VERSION`, stamped per **record** (not in a header) so a consumer reading one line still knows what it holds. |
| `transcript` | Archive-**relative** posix path (in `--transcript` mode, the bare file name), so the corpus is portable across machines. |
| `task_id`, `session_id`, `is_subagent`, `subagent_id` | All derived from the archive **path** by `parse_archive_path` — never from record bodies. This is why the fixture must be a directory tree. |
| `tool_use_id` | Correlates the `tool_use` call to its later `tool_result` answer. |
| `tool_name` | The matched tool. A frozenset, widenable via `--tool-name`, because the archive spans months of tool-name history. |
| `query` | `input.query`, lifted out so it stays greppable. |
| `params` | Everything else that was passed (`project_id`, `limit`, …). |
| `caller` | `{agent_id, task_id, role}` — *who issued the search*, from the briefing's Agent Identity line. **Not** the `tool_use` block's own `caller` key (see above). Its nested `task_id` is who the agent thinks it is; the top-level `task_id` is how the archive filed it — keeping both makes a disagreement visible. |
| `result_status` | `ok` \| `error` \| `unparsed` \| `missing`. Only `ok` measures the store; the rest are gaps in the instrument, kept distinct from a genuine zero-hit search (`ok` with `result_count == 0`) so a decoding bug cannot read as a recall collapse. Line 9 of the main transcript pins `missing`. |
| `result_count` | `len(results)`. |
| `results[]` | `{rank, id, score, source_store, category, content_chars}`. `rank` is 1-based and positional — the order the agent **saw**. `content_chars` is a **length, never the text**: the corpus records what was retrieved and how well it scored, and copying result bodies would duplicate the live store into a gitignored file that stales the moment a memory is edited. Look the `id` up if you need text. |

A run writes three artifacts (named here in text because the lock-charter file
lint's extension allowlist has no `jsonl` entry, so they cannot be declared in
`metadata.files`):

```
fused-memory/data/memory-evals/transcript-corpus/corpus-<STAMP>.jsonl
fused-memory/data/memory-evals/transcript-corpus/coverage-<STAMP>.json
fused-memory/data/memory-evals/transcript-corpus/report-<STAMP>.txt
```

`fused-memory/data/` is gitignored, so no run dirties a diff — and no artifact
is committed. Declared consumers: PRD leaf η's **write-after-miss validation
set**; **operators** reading `report-<STAMP>.txt` for how much of the archive a
run covered and what it could not read; and future **E3 / E8 runners** and the
**shadow-replay harness**, which need a fixed, re-runnable set of historical
queries. Scope boundary (PRD §7): this leaf produces the corpus **only** — no
shadow-replay harness, no E3 golden query set, no E8 re-derivation audit, and
no write or mutation of the live memory corpus.

### Regenerating

The two `.jsonl` files are plain text and hand-authored (< 1 KB each) —
edit them directly, no decompress/recompress step. Keep them uncompressed:
see "Why plain `.jsonl` and not the live archive's `.jsonl.gz`" above.
Changing a query, an id, or a score will fail the round-trip test by
design — the test asserts the fixture's authored facts verbatim, which is
what makes it a fixture rather than a smoke test.
