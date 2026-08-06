# LME replay corpus

The committed replay corpus for PRD `plans/local-memory-models-eval-prd.md`
task **δ**: a stratified sample of real `dark_factory` episodes that is
**never conditioned on the incumbent extraction pipeline's outcome**.

| file | what it is |
|---|---|
| `build_corpus.py` | the read-only builder + verifier (`--verify`) |
| `corpus_manifest.json` | the committed artifact: 200 episode ids, content hashes, criteria, stratification report |

Declared consumers: **ε** (replay engine input), **ζ** (control replays),
**θ** (full arm replays). They read episode *content* from the store by uuid;
the manifest deliberately carries only the id and a content hash, so a second
copy of the bytes can never silently disagree with the first.

## Re-deriving the corpus

The one command a reviewer runs:

```bash
uv run python fused-memory/scripts/local_memory_models_eval/build_corpus.py --verify
```

It re-runs the sampler from the criteria the manifest records about **itself**
— seed, N, dimensions, allocation rule — and compares the result to the
recorded episode list, then re-hashes every episode's content. It never reads
the manifest's own `episodes` list as the answer to what should be selected;
that check would pass on any manifest, including a tampered one.

Exit codes are per failure class, because each has a different remedy:

| exit | status | means |
|---|---|---|
| 0 | `ok` | every id re-derived and every hash matched |
| 2 | `id_mismatch` | the recorded criteria no longer produce the recorded corpus |
| 3 | `hash_drift` | the selection is right, but episode bytes changed |
| 4 | `missing_episodes` | an id the manifest names is gone from the store |
| 5 | `bad_manifest` | the artifact is structurally unusable |
| 1 | — | the run never reached a verdict (store unreachable, unsatisfiable `--n`) |

To rebuild rather than check, drop `--verify`; add `--dry-run` to print the
stratification report and write nothing.

### `id_mismatch` on a grown store is expected, and is not tampering

`dark_factory` is written continuously by the running fleet — it grew from
2,770 episodes to 2,775 in the two hours between this task's census and its
build. Re-derivation runs against the population **as it is now**, so once new
episodes land the cell counts shift, largest-remainder allocation moves a seat
or two, and `--verify` reports `id_mismatch`.

That is loud rather than silent, which is the right failure — but it does mean
a green `--verify` is a statement about *today's* store, not a permanent
property of the artifact. The legs that stay durable are `missing_episodes` and
`hash_drift`: those answer "are the 200 episodes ε/ζ/θ replay still present,
with the same bytes?", which is the question the downstream arms actually need.

Verified green at build time (2026-08-06, population 2,775).

## Stratification

Two dimensions, crossed: **`month(created_at)` × `payload_kind(source_description)`**.
16 of the 20 cells are non-empty.

**Why `source_description` and not `source`.** `e.source` looks like the
payload-kind field and is not: measured read-only across the whole store, it is
uniformly `'text'` for **every** episode and discriminates nothing. The real
signal is `e.source_description`, which the writers shape as
`add_memory:<category>` (and `replay_from_mem0:<category>` on the Mem0 replay
path). A caller-supplied `add_episode` description buckets under a single
explicit `add_episode` kind rather than fanning the axis into one stratum per
caller string.

Allocation is **min-1 floor + largest-remainder proportional, capped at cell
size**. The floor is load-bearing, not tidiness: the
`('2026-04', 'procedural_knowledge')` cell holds exactly **one** episode of
2,775, so its proportional share at N=200 is 0.07 seats. Pure proportional
allocation rounds it to zero and deletes an entire payload kind from the corpus
— and therefore from every downstream arm comparison — while the stratification
report still reconciles perfectly. The floor costs 16 of 200 seats and makes
that invisible loss impossible.

Within a cell the draw is a **prefix of a seeded permutation**: sort the cell
canonically by uuid (so the result does not depend on the order FalkorDB
happens to return rows in, which is not guaranteed stable), permute under
`Random(f'{seed}:{month}:{kind}')`, take the first `allocate()[cell]`.

## N = 200 is provisional

200 is δ's choice: the midpoint of the PRD's 150–300 band, checked against the
measured census so that all 16 non-empty cells receive at least one seat.
**PRD Open Q4 defers the final corpus size to ζ**, to be settled from measured
control variance and wall-clock.

Re-tuning is cheap by construction. Because each cell's take is a *prefix* of
that cell's permutation, growing N only ever **appends** to a cell — so ζ can
re-run the builder at a different N without invalidating replays ε has already
completed at the smaller one. Note this is a **per-cell** guarantee, not a
global one: largest-remainder allocation can move a single seat between cells
as N changes, so a cell whose allocation *shrank* is the one case where an
earlier pick is dropped.

```bash
# what ζ runs to re-tune
uv run python fused-memory/scripts/local_memory_models_eval/build_corpus.py --n 300
```

## The binding hazard: no conditioning on the incumbent's outcome

Corpus membership must never depend on how well the **incumbent** extraction
pipeline did on an episode — otherwise every arm is compared on a corpus that
was pre-selected to flatter one of them. `e.entity_edges` is the per-episode
record of what that pipeline produced, so it is the exact outcome proxy that
must stay untouched. The guarantee is mechanized three ways, each tested in
`fused-memory/tests/test_local_memory_models_eval_corpus.py`:

1. **The projection cannot see it.** The single query is
   `MATCH (e:Episodic) RETURN e.uuid, e.name, e.group_id, e.source_description,
   e.created_at, e.content` — built from the `PROJECTED_FIELDS` constant, with
   no `WHERE`, no `LIMIT`, and no `entity_edges`. Tests assert the RETURN list
   by equality and assert the absence of a filter clause directly.
2. **The record cannot hold it.** `EpisodeRecord` has no outcome field, so no
   sampling rule *can* condition on one. A test pins the field set exactly.
3. **The sampler has nowhere to hide it.** Every input episode leaves
   `select()` as either `selected` or `not_drawn` — there is deliberately no
   `ineligible` or `low_quality` disposition, since either would be a place for
   outcome conditioning to live. The disposition accounting is exhaustive.

**The anchor.** Exactly one episode in the store —
`e622a9bf-f1c8-431b-ad36-92762d69436d` (`add_memory:temporal_facts`,
2026-05-16) — has `size(entity_edges) == 0`: the one the incumbent extracted
nothing from. It is the concrete eligibility anchor, and a test class
(`TestOutcomeFailedEpisodesStayEligible`) pins that it is fetched into the
population and is selectable. It competes in cell `2026-05|temporal_facts`
against 390 others on equal terms; at the committed seed it was not drawn, and
that is the point — being drawn is a coin flip, being *eligible* is the
guarantee.

## Read-only, and no graphiti driver

The builder issues exactly one `GRAPH.RO_QUERY`. Read-only here is
**server-enforced**, not a client-side promise: a `CREATE` issued through that
command path is refused by FalkorDB and materializes nothing. A client-side
guard sits on top of it so a violation surfaces as a typed `CorpusBuildError`
at the seam that owns the guarantee, rather than as a redis error three layers
down.

It connects with `falkordb.asyncio.FalkorDB(...).select_graph(...)` and
**never** constructs `graphiti_core.driver.falkordb_driver.FalkorDriver`, whose
`__init__` fire-and-forgets `build_indices_and_constraints()` — that would
create indices on `dark_factory` and destroy the protected no-index evidence
owned by `docs/prds/falkordb-index-provisioning.md`. Both halves are *measured*,
not promised: the offline tests booby-trap `FalkorDriver.__init__` and drive the
reader against a double whose write-capable methods are tripwires (with a
meta-test proving the tripwires fire), and the live smoke wraps the real graph
handle so only `ro_query` can be reached.

Test lane: the offline suite runs by default; the single live test is
`@pytest.mark.integration` (per-test, never module-level — `addopts` is
`-m 'not integration'`) and skips cleanly with no FalkorDB.

```bash
uv run pytest fused-memory/tests/test_local_memory_models_eval_corpus.py            # offline
uv run pytest fused-memory/tests/test_local_memory_models_eval_corpus.py -m integration  # live, read-only
```

## Self-check

The delivered_check for this PRD row
(`plans/local-memory-models-eval-prd.capability-manifest.yaml`) greps the
committed tree. Run it anchored to this directory's parent:

```bash
git grep -cE 'PRD[-]MARKER:local-memory-models-eval corpus[-]manifest' -- fused-memory/scripts/
```

Expected — two files, three matching lines:

```
fused-memory/scripts/local_memory_models_eval/build_corpus.py:2
fused-memory/scripts/local_memory_models_eval/corpus_manifest.json:1
```

The builder carries the literal marker twice (its docstring's Artifacts section
and the `PRD_MARKER` constant it serializes from); the manifest carries it once,
as a top-level `prd_marker` field so it survives JSON serialization intact.

This README deliberately spells the marker only in the bracketed `PRD[-]MARKER`
regex form, never literally, so it does not satisfy the check it documents —
the same trick the capability-manifest pair uses on itself. If you add a file
here that carries the literal, update the expected count above in the same
commit, or this self-check quietly stops meaning anything.
