#!/usr/bin/env python3
"""E2 storage-shape bake-off + D10 audit-recall measurement (task 3199, PRD leaf ζ).

WHAT THIS IS
------------
The arbitration experiment for ``docs/prds/memory-metadata-vocabulary.md``
D9: *"ζ implements eval-design E2 exactly (arms: status-quo / C-peers /
B-grouped / each ± 3111-pin; the 'hybrid' of eval-doc open question 5 *is*
C); η is a pure deterministic gate (3169 pattern) defaulting to ratify-C."*

It materialises the SAME knowledge in three storage shapes into three
seeded ephemeral Qdrant collections, queries each with one shared query
set, and emits a per-arm decision table to
``plans/e2-storage-shape-bakeoff-report.{json,md}``.  That artifact is the
signal gate leaf **η** (the shape-ratification gate) puts in front of an
operator: the PRD's choice between δ-as-default and peers-as-default gets
made by reading it.

``plans/memory-subsystem-eval-design.md`` §5 E2 specifies the arms and the
four metrics:

  * **claim recall@k** — does the *specific claim* a query targets surface
    (k=5 and k=10; the near-dup guard lives at 5),
  * **canonical/topic discoverability**,
  * **tokens returned per query** — the D4 cost of a grouped read vs N
    short hits,
  * **near-dup-guard candidate adequacy** — replay the pure guard over each
    arm's top-5: *would the write that became duplicate N+1 have been
    matched?*

PRD **D10** adds a second, independent deliverable (3136's deferral item
3): *"ζ also delivers the audit-recall measurement — run
``audit_duplicate_memories.py`` against α/3130's labeled fixture and report
recall on the paraphrase class — the number that decides how much to trust
the κ report."*  No threshold, rate or bound is asserted anywhere for it
(gate G6): the measurement informs a judgement, it does not gate a build.

MEASUREMENT DISCIPLINE — RANK-BASED, NEVER ABSOLUTE-SCORE-BASED
---------------------------------------------------------------
``plans/memory-subsystem-eval-design.md`` §1 states the program-wide rule
verbatim:

    "every retrieval metric in this program must be rank-based, never
    absolute-score-based.  Re-running 3111's probe today on the canonical's
    own topic phrase returned scores of 0.44-0.51 for the same corpus where
    the task record measured 0.72-0.90 — wording and embedding/config drift
    move the score scale wholesale.  Ranks and set-membership
    (present-in-top-k) survive that; thresholds on raw cosine do not."

Every metric here is therefore rank/set-based.  The ONE exception is
deliberate and is reported split in two, so the discipline is not quietly
violated: ``find_near_duplicate_memory`` *is* an absolute-threshold
selector in production, and E2's question is whether the guard would have
fired — so the replay cannot be made rank-based without ceasing to measure
the real guard.  ``guard_adequacy`` therefore returns

  1. ``candidate_present`` — rank/set-based and score-free: is a true
     cluster sibling in the arm's top-5 at all?  This is the drift-proof
     part, and the part that actually discriminates between storage shapes.
  2. ``guard_matched`` — the production selector's verdict at its
     configured threshold, flagged ``threshold_replay: True`` in the JSON
     so nobody trends it across embedding-config changes as if it were
     stable.

RANK-BASED IS NOT TRANSFORM-BLIND
--------------------------------
**The single statement of the credit mechanism.**  Every other site in this
module and its tests points HERE rather than restating it, because five
copies of one paragraph drift into five subtly different claims.

Discoverability is reported in BOTH forms.  ``canonical_in_top_5_rate`` is
measured over the read window, where ``apply_grouped_read`` has already
synthesised a grouped document carrying the CANONICAL's ``record_id`` — and
``topic_discoverability`` identifies the canonical BY ``record_id``.  So
under ``b_grouped`` any child hit that folds upward materialises a record
wearing the canonical's id and is scored as "the canonical was found", while
under ``c_peers``/``status_quo`` the canonical's own stored record must
itself have ranked.  ``apply_topic_anchor`` credits it the same way, by
injecting the canonical outright — so grouping is not the only transform
that can move that column.

``stored_canonical_in_top_5_rate`` (with its median rank and censored
denominator) is measured over the RAW store hits, before either transform,
and so answers the narrower question: did the canonical's own stored record
rank?  It is therefore identical across a shape's two pin variants by
construction.

Neither number is corrected toward the other — the transform-credited one is
arguably the right thing to credit, since a grouped read genuinely does put
the canonical body in the reader's window.  The gap between them is the read
transform's contribution, DISCLOSED rather than folded invisibly into one
column.  See ``render_markdown``'s reading guide for the operator-facing
statement of the same thing (the one legitimate second copy, because it is
output rather than commentary).

BLIND-AUTHORING PROTOCOL (resolves PRD §10's open tactical question
"Blind-authoring protocol for ζ's arms (two-agent cross-check vs
single-author-blind-to-metrics)")
-------------------------------------------------------------------
Resolved as **single-author-blind-to-metrics, MECHANIZED BY COMMIT
ORDERING**.  The eval doc names the experiment's own biggest weakness
(§5 E2): *"arm quality reflects authoring skill — the experiment is
gameable by authoring one arm well and another lazily."*

The mitigation implemented here:

  * ``tests/fixtures/e2_arm_claims.jsonl`` (the editorial decomposition of
    each cluster into short single-claim peers) and
    ``tests/fixtures/e2_query_set.jsonl`` (the query set) are authored and
    committed as PREREQUISITES — before a single metric function exists in
    this file.  **Git history is the audit trail**: it proves no metric
    code was in the tree when the arms were decomposed, so the arms cannot
    have been tuned toward a number.  The report records the fixture commit
    SHAs.
  * The anti-laziness floor is **claim-coverage parity**: every claim id
    must be realizable in every arm (asserted mechanically).  It is
    deliberately NOT total-content-length parity — arm (a)'s long originals
    versus arm (c)'s short peers differ *by construction*, and that
    difference IS the tokens-per-query metric.

ARM-LOCAL REFERENCE TRANSFORMS
------------------------------
``server/grouped_read.py`` (task 3129) and the topic-anchored pin in
``MemoryService.search`` (task 3111) do not exist: both are deferred BEHIND
gate η, which depends on THIS task.  A downstream task structurally cannot
supply an upstream premise, so the bake-off carries its own **arm-local**
reference implementations — ``apply_grouped_read`` (PRD V2/D6: upward
resolution mandatory, contested never suppressed) and
``apply_topic_anchor`` (PRD D1: additive ``topic == T AND canonical is
True`` pin).  Both are pure read-side transforms over an already-fetched
ranked hit list.  They stay arm-local by construction: PRD V2 explicitly
forbids the suppression filter leaking into ``MemoryService.search``, where
it would break ``mem0_dedup.find_prior_memories``' post-filter and hide
candidates from the write guard.

THE +1-RE-EMISSION REGROWTH PROBE (task 4012)
--------------------------------------------
esc-3200-3 was a SPLIT ratification: Option C's WRITE shape was ratified, no
read transform was, and the read choice was delegated to 4004 (which selected
the promoting topic pin).  Neither answered what ONE organic re-emission of
an already-canonical claim costs retrieval under that shape, or whether the
selected transform absorbs it.  So on top of the six arms, every default run
also seeds the ratified ``c_peers`` corpus PLUS one near-duplicate
re-emission per topic — twice, once ``unstamped`` (no ``topic`` key at all,
the organically observed case) and once ``stamped`` (the write path's best
case) — and scores each injected pass under three READ arms: ``flat``,
``additive_pin`` and 4004's selected ``promoting_pin``.  The un-injected
baseline is the ratified arm's OWN rankings, so the three read arms are
scored from ONE set of rankings and an arm-vs-arm comparison carries no
cross-seeding noise; a baseline-vs-after delta does span two collections,
and the artifact reports its own re-seeding noise floor beside the table
rather than arguing the difference away.  The
deltas, and the ``stamped - unstamped`` difference that is what task 4006's
stamping campaign buys, land in the report's ``regrowth`` block and its
``## Regrowth deltas`` section.  ``--no-regrowth`` skips it, and the block is
then an explicit ``None`` rather than an absent key.  NO threshold is
asserted on any delta (gate G6).

STRUCTURE
---------
A thick **pure core** (loaders, arm materialization, read transforms,
metrics, report rendering) with zero network, fully exercised in the merge
lane, plus a thin **live driver** (``seed_arm`` / ``run_arm`` /
``run_bake_off`` / ``_run``) that is the only part touching Qdrant or an
embedder.

WHY THERE IS NO STORE-MUTATION PREFLIGHT HERE (an observation, task 4293)
------------------------------------------------------------------------
``fused_memory.utils.store_mutation_preflight.assert_store_mutation_allowed``
is the fail-closed capability probe the shared-store mutators in
``fused-memory/scripts/`` call before their first write.  The live driver's
two mutations — ``drop_collections``' ``delete_collection`` and
``seed_arm``'s ``backend.add`` — are unprobed, and that is a decision rather
than an omission.  Measured against this file AS IT STANDS at task 4293; not
a standing exemption for whatever it becomes:

  * its blast radius is bounded to scratch substrate, statically.
    ``run_bake_off`` force-sets ``config.mem0.collection_prefix =
    ephemeral_collection_prefix()`` on its own config copy, and that prefix is
    ``load_cleanup_script().E2_BAKEOFF_PREFIX`` — the SAME constant the reaper
    ``scripts/cleanup_test_collections.py`` allowlists, read from the reaper
    itself.  No CLI flag reaches the prefix (``--project-suffix`` moves the
    suffix only), so no invocation can point these deletes or writes at a
    ``fused``-prefixed production collection;
  * it does not write mem0's shared SQLite history: ``seed_arm`` stubs
    ``instance.db.add_history`` per instance before the first ``add`` (that
    writer is process-shared, xdist-contended, and read-only in the sandbox —
    see ``seed_arm``'s own docstring).  So the capability the preflight probes
    for is not one the seeding path needs.

THE RESIDUAL, stated rather than papered over: ``run_bake_off`` builds a real
``MemoryService(config)`` and awaits ``memory.initialize()``, which runs
Graphiti startup maintenance — index creation plus the W6-ε dup-uuid-edge
scan-and-REPAIR — against the production FalkorDB, under no prefix bound at
all.  That is NOT covered by anything above, and it is not specific to this
script: no ``run()``-level guard in any of the scripts task 4293 guarded
dominates ``initialize()`` either, because ``initialize()`` runs before
``run()`` is called.  It is a systemic gap tracked by tasks 4318 and 4350, and
is deliberately not claimed as handled here.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------
#
# Package-relative, never resolved against the checkout this happens to run
# in: a per-task worktree path baked in here would break the moment the
# script runs anywhere else (the lesson pinned at
# tests/test_calibrate_write_triage.py:1267).

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES_DIR = _PACKAGE_ROOT / 'tests' / 'fixtures'

#: alpha/3130's labeled calibration fixture. CHARTER-LOCKED — read-only here
#: (shared/tests/test_locking.py:122, tests/test_lock_charter_guard.py:191).
DEFAULT_ALPHA_FIXTURE_PATH = _FIXTURES_DIR / 'write_triage_calibration.jsonl'
#: E1's topic registry, owned by memory_eval_retrieval_probe.run_derive_registry
#: — read-only here. Supplies E2's topic slugs (PRD D4: one namespace).
DEFAULT_REGISTRY_PATH = _FIXTURES_DIR / 'memory_eval_topic_registry.json'
#: The three E2-owned fixtures (this task's prerequisites pre-2/3/4).
DEFAULT_ARM_CLAIMS_PATH = _FIXTURES_DIR / 'e2_arm_claims.jsonl'
DEFAULT_QUERY_SET_PATH = _FIXTURES_DIR / 'e2_query_set.jsonl'
DEFAULT_DISTRACTOR_SLAB_PATH = _FIXTURES_DIR / 'e2_distractor_slab.jsonl'
#: The +1-re-emission regrowth probe's own fixture (task 4012, pre-1).
#: Package-relative for the same reason its five siblings are — a path
#: resolved against the checkout cwd works only from the tree it was
#: authored in.  Read ONLY on the regrowth path: a probe-less run must not
#: fail on a fixture it never opens.
DEFAULT_REGROWTH_INJECTION_PATH = _FIXTURES_DIR / 'e2_regrowth_injection.jsonl'

#: Where a reader is told to look when a fixture is missing or disagrees.
_FIXTURE_DOCS = 'fused-memory/tests/fixtures/README.md'


class FixtureError(RuntimeError):
    """A fixture is missing, malformed, or disagrees with its siblings.

    Named rather than a bare KeyError/FileNotFoundError so that a fixture
    problem is never mistaken for a retrieval defect: every message states
    WHICH fixture, WHICH record, and where the regeneration procedure is
    documented.
    """


# ---------------------------------------------------------------------------
# Loaded shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationCluster:
    """One alpha-fixture cluster: its canonical plus every member, labels intact."""

    cluster_id: str
    canonical: dict[str, Any]
    members: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RegistryTopic:
    """One curator-gate registry entry, keyed by the cluster it was derived from."""

    cluster_id: str
    topic: str
    phrasings: list[dict[str, Any]]
    raw: dict[str, Any]


@dataclass(frozen=True)
class ArmClaim:
    """One editorially-decomposed single-claim peer (pre-2)."""

    claim_id: str
    cluster_id: str
    topic: str
    text: str
    source_memory_id: str
    canonical: bool
    b_arm_role: str
    contested: bool


@dataclass(frozen=True)
class Query:
    """One E2 query (pre-3)."""

    query_id: str
    kind: str
    text: str
    topic: str
    cluster_id: str
    expects_claim_ids: list[str]
    held_out: bool


#: Query kinds the fixture encodes, and the order the artifact reports them
#: in.  Pinned here rather than derived from whatever the loaded fixture
#: happens to contain: a kind that stopped being generated would silently
#: vanish from the report instead of showing up as an empty, measured-nothing
#: row.  ``load_query_set`` validates every record's ``kind`` against this.
QUERY_KINDS: tuple[str, ...] = ('claim', 'topic_phrasing')

#: The cross-cutting subset reported alongside the kinds.  ``held_out``
#: phrasings are the ones the E1 registry was NOT derived from, so they are
#: the only queries that measure generalisation rather than recall of the
#: derivation input.  It overlaps a kind by construction — it is a SUBSET,
#: not a third kind — and is labelled that way in the artifact.
HELD_OUT_SUBSET = 'held_out'


@dataclass(frozen=True)
class RegrowthInjection:
    """One +1 re-emission of a topic's canonical claim (task 4012, pre-1).

    ``reemits_claim_id`` is carried in the fixture rather than derived
    because the materialised record gets ``claim_ids = [reemits_claim_id]``:
    a near-duplicate that restates the canonical's claim genuinely DOES
    realize it, and scoring it as realizing nothing would make claim recall
    able only to FALL under injection — rigging the probe toward the very
    conclusion it exists to test.  Naming the re-emitted claim in the
    fixture keeps that modelling choice auditable there rather than buried
    in the materializer, and ``cross_validate_regrowth_injections`` pins it
    to the cluster's TRUE ``canonical: true`` claim.
    """

    injection_id: str
    topic: str
    cluster_id: str
    reemits_claim_id: str
    text: str


@dataclass(frozen=True)
class Distractor:
    """One contamination-slab entry (pre-4).

    ``raw`` is kept so a caller can assert the row carries no vocabulary
    metadata key — a distractor that were topic-anchorable would stop being
    a distractor and start being a right answer for the pin to find.
    """

    distractor_id: str
    content: str
    category: str
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _read_jsonl(path: str | Path, *, what: str) -> list[dict[str, Any]]:
    """Read a JSONL fixture STRICTLY.

    Reuses the strict-line semantics of
    ``calibrate_write_triage.load_fixture`` rather than reimplementing
    lenient parsing: a malformed line raises with its 1-based line number
    instead of being skipped, because silently dropping a record would
    shrink the measured population without saying so.
    """
    path = Path(path)
    if not path.exists():
        raise FixtureError(
            f'{what} fixture not found at {path}. It is a FROZEN, committed '
            f'fixture — see {_FIXTURE_DOCS} for its derivation and how to '
            f'regenerate it. Refusing to continue with an empty {what}: that '
            f'would silently delete an experiment variable.'
        )
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FixtureError(f'{path}:{lineno}: malformed JSON line: {exc}') from exc
            if not isinstance(record, dict):
                raise FixtureError(
                    f'{path}:{lineno}: expected a JSON object, got {type(record).__name__}'
                )
            records.append(record)
    return records


def load_calibration_clusters(
    path: str | Path = DEFAULT_ALPHA_FIXTURE_PATH,
) -> dict[str, CalibrationCluster]:
    """Group the alpha fixture's records into clusters, keyed by cluster id.

    Labels are preserved on every member: the guard-adequacy probe selects
    the cluster's chronologically last ``label == 'duplicate'`` record as
    the write that became duplicate N+1, so a loader that dropped labels
    would silently change which record is measured.
    """
    records = _read_jsonl(path, what='alpha calibration')
    grouped: dict[str, list[dict[str, Any]]] = {}
    for lineno, record in enumerate(records, 1):
        cluster_id = record.get('cluster_id')
        if not cluster_id:
            raise FixtureError(f'{path}:{lineno}: record has no cluster_id')
        grouped.setdefault(cluster_id, []).append(record)

    clusters: dict[str, CalibrationCluster] = {}
    for cluster_id, members in grouped.items():
        canonicals = [r for r in members if r.get('label') == 'canonical']
        if len(canonicals) != 1:
            raise FixtureError(
                f'{path}: cluster {cluster_id} has {len(canonicals)} records labelled '
                f"'canonical', expected exactly 1"
            )
        clusters[cluster_id] = CalibrationCluster(
            cluster_id=cluster_id, canonical=canonicals[0], members=members,
        )
    return clusters


def load_registry_topics(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, RegistryTopic]:
    """Read E1's topic registry, keeping only the curator-gate entries.

    The registry also carries census/topic-guard/hand entries derived from
    other sources; only ``derived_from == 'curator_gate'`` entries overlap
    the alpha clusters (20/20), and only those supply E2's topic slugs.
    """
    path = Path(path)
    if not path.exists():
        raise FixtureError(f'topic registry not found at {path} — see {_FIXTURE_DOCS}')
    payload = json.loads(path.read_text())
    topics: dict[str, RegistryTopic] = {}
    for entry in payload.get('entries', []):
        if entry.get('derived_from') != 'curator_gate':
            continue
        cluster_id = entry.get('provenance', {}).get('cluster_id')
        if not cluster_id:
            raise FixtureError(f'{path}: curator_gate entry {entry.get("topic")!r} has no provenance.cluster_id')
        topics[cluster_id] = RegistryTopic(
            cluster_id=cluster_id,
            topic=entry['topic'],
            phrasings=list(entry.get('phrasings', [])),
            raw=entry,
        )
    return topics


def load_arm_claims(path: str | Path = DEFAULT_ARM_CLAIMS_PATH) -> list[ArmClaim]:
    """Read the blind-authored arm-claim decomposition (pre-2)."""
    claims = []
    seen: set[str] = set()
    for lineno, record in enumerate(_read_jsonl(path, what='E2 arm-claims'), 1):
        claim_id = record.get('claim_id')
        if not claim_id:
            raise FixtureError(f'{path}:{lineno}: claim has no claim_id')
        if claim_id in seen:
            raise FixtureError(f'{path}:{lineno}: duplicate claim_id {claim_id!r}')
        seen.add(claim_id)
        try:
            claims.append(ArmClaim(
                claim_id=claim_id,
                cluster_id=record['cluster_id'],
                topic=record['topic'],
                text=record['text'],
                source_memory_id=record['source_memory_id'],
                canonical=bool(record['canonical']),
                b_arm_role=record['b_arm_role'],
                contested=bool(record.get('contested', False)),
            ))
        except KeyError as exc:
            raise FixtureError(f'{path}:{lineno}: claim {claim_id!r} missing field {exc}') from exc
    return claims


def load_regrowth_injections(
    path: str | Path = DEFAULT_REGROWTH_INJECTION_PATH,
) -> list[RegrowthInjection]:
    """Read the +1-re-emission injection slab (task 4012, pre-1).

    Deliberately the same strict idiom as ``load_arm_claims`` — dedup on the
    id, ``KeyError`` translated into a ``FixtureError`` naming the path, the
    1-based line and the field — so the error surface a reader meets is
    uniform across all six fixtures.
    """
    injections = []
    seen: set[str] = set()
    for lineno, record in enumerate(_read_jsonl(path, what='E2 regrowth-injection'), 1):
        injection_id = record.get('injection_id')
        if not injection_id:
            raise FixtureError(f'{path}:{lineno}: injection has no injection_id')
        if injection_id in seen:
            raise FixtureError(f'{path}:{lineno}: duplicate injection_id {injection_id!r}')
        seen.add(injection_id)
        try:
            injections.append(RegrowthInjection(
                injection_id=injection_id,
                topic=record['topic'],
                cluster_id=record['cluster_id'],
                reemits_claim_id=record['reemits_claim_id'],
                text=record['text'],
            ))
        except KeyError as exc:
            raise FixtureError(
                f'{path}:{lineno}: injection {injection_id!r} missing field {exc}'
            ) from exc
    return injections


def load_query_set(path: str | Path = DEFAULT_QUERY_SET_PATH) -> list[Query]:
    """Read the blind-authored query set (pre-3)."""
    queries = []
    seen: set[str] = set()
    for lineno, record in enumerate(_read_jsonl(path, what='E2 query-set'), 1):
        query_id = record.get('query_id')
        if not query_id:
            raise FixtureError(f'{path}:{lineno}: query has no query_id')
        if query_id in seen:
            raise FixtureError(f'{path}:{lineno}: duplicate query_id {query_id!r}')
        seen.add(query_id)
        try:
            kind = record['kind']
            if kind not in QUERY_KINDS:
                raise FixtureError(f'{path}:{lineno}: unknown query kind {kind!r}')
            queries.append(Query(
                query_id=query_id,
                kind=kind,
                text=record['text'],
                topic=record['topic'],
                cluster_id=record['cluster_id'],
                expects_claim_ids=list(record['expects_claim_ids']),
                held_out=bool(record.get('held_out', False)),
            ))
        except KeyError as exc:
            raise FixtureError(f'{path}:{lineno}: query {query_id!r} missing field {exc}') from exc
    return queries


def load_distractor_slab(
    path: str | Path = DEFAULT_DISTRACTOR_SLAB_PATH,
) -> list[Distractor]:
    """Read the frozen contamination slab (pre-4).

    A missing slab raises rather than yielding an empty list — see
    ``_read_jsonl``. Seeding no distractors would leave every arm's
    collection containing only right answers, which is a different
    experiment that would nonetheless report as a clean success.
    """
    slab = []
    for lineno, record in enumerate(_read_jsonl(path, what='E2 distractor-slab'), 1):
        try:
            slab.append(Distractor(
                distractor_id=record['distractor_id'],
                content=record['content'],
                category=record['category'],
                raw=record,
            ))
        except KeyError as exc:
            raise FixtureError(f'{path}:{lineno}: distractor missing field {exc}') from exc
    return slab


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_fixtures(
    *,
    clusters: dict[str, CalibrationCluster],
    topics: dict[str, RegistryTopic],
    claims: list[ArmClaim],
    queries: list[Query],
) -> None:
    """Assert the fixtures AGREE with each other, naming any offender.

    Parsing cleanly is not enough: a claim pointing at a cluster that does
    not exist, or a query expecting a claim from another cluster, would
    otherwise surface deep inside the metrics as a KeyError or — worse — as
    a silently-zero recall indistinguishable from a genuine retrieval miss.
    """
    claims_by_id = {c.claim_id: c for c in claims}

    for claim in claims:
        if claim.cluster_id not in clusters:
            raise FixtureError(
                f'claim {claim.claim_id!r} names cluster {claim.cluster_id!r}, '
                f'which is not in the alpha fixture'
            )
        if claim.cluster_id not in topics:
            raise FixtureError(
                f'claim {claim.claim_id!r} names cluster {claim.cluster_id!r}, '
                f'which has no curator-gate registry topic'
            )
        if claim.topic != topics[claim.cluster_id].topic:
            raise FixtureError(
                f'claim {claim.claim_id!r} carries topic {claim.topic!r} but its '
                f'cluster\'s registry topic is {topics[claim.cluster_id].topic!r} '
                f'(PRD D4: one topic namespace — slugs are taken verbatim)'
            )
        member_ids = {r['memory_id'] for r in clusters[claim.cluster_id].members}
        if claim.source_memory_id not in member_ids:
            raise FixtureError(
                f'claim {claim.claim_id!r} cites source_memory_id '
                f'{claim.source_memory_id!r}, which is not a member of its own '
                f'cluster {claim.cluster_id!r}'
            )

    canonical_counts: dict[str, int] = {c.cluster_id: 0 for c in claims}
    for claim in claims:
        if claim.canonical:
            canonical_counts[claim.cluster_id] += 1
    for cluster_id, count in canonical_counts.items():
        if count != 1:
            raise FixtureError(
                f'cluster {cluster_id!r} has {count} claims marked canonical, expected exactly 1'
            )

    for query in queries:
        if not query.expects_claim_ids:
            raise FixtureError(f'query {query.query_id!r} expects no claim')
        for claim_id in query.expects_claim_ids:
            expected = claims_by_id.get(claim_id)
            if expected is None:
                raise FixtureError(
                    f'query {query.query_id!r} expects claim {claim_id!r}, which does not exist'
                )
            if expected.cluster_id != query.cluster_id:
                raise FixtureError(
                    f'query {query.query_id!r} (cluster {query.cluster_id!r}) expects claim '
                    f'{claim_id!r} from cluster {expected.cluster_id!r}'
                )


def cross_validate_regrowth_injections(
    *,
    injections: list[RegrowthInjection],
    claims: list[ArmClaim],
) -> None:
    """Assert the injection slab agrees with the claims fixture.

    A SEPARATE function rather than a branch inside
    ``cross_validate_fixtures`` on purpose: the five original fixtures are
    cross-validated on EVERY run, including runs with the probe disabled,
    and folding a sixth in would make a probe-less run fail on a fixture it
    never reads.

    Checks, in order: exactly one injection per topic; every topic covered;
    ``cluster_id`` agreeing with the claims fixture; ``reemits_claim_id``
    existing and naming that cluster's TRUE canonical claim.
    """
    claims_by_id = {c.claim_id: c for c in claims}
    topics = {c.topic for c in claims}
    cluster_by_topic = {c.topic: c.cluster_id for c in claims}

    per_topic: dict[str, list[RegrowthInjection]] = {}
    for injection in injections:
        if injection.topic not in topics:
            raise FixtureError(
                f'injection {injection.injection_id!r} names topic '
                f'{injection.topic!r}, which is not in the arm-claims fixture '
                f'(see {_FIXTURE_DOCS})'
            )
        per_topic.setdefault(injection.topic, []).append(injection)

    for topic in sorted(topics):
        found = per_topic.get(topic, [])
        if len(found) != 1:
            raise FixtureError(
                f'topic {topic!r} has {len(found)} regrowth injections, expected '
                f'exactly 1 — the "+1" re-emission quantity IS the experiment, '
                f'so a second injection is a fixture defect rather than a '
                f'bigger probe (see {_FIXTURE_DOCS})'
            )

    for injection in injections:
        expected_cluster = cluster_by_topic[injection.topic]
        if injection.cluster_id != expected_cluster:
            raise FixtureError(
                f'injection {injection.injection_id!r} names cluster '
                f'{injection.cluster_id!r} but topic {injection.topic!r} belongs '
                f'to cluster {expected_cluster!r} in the arm-claims fixture '
                f'(see {_FIXTURE_DOCS})'
            )
        reemitted = claims_by_id.get(injection.reemits_claim_id)
        if reemitted is None:
            raise FixtureError(
                f'injection {injection.injection_id!r} re-emits claim '
                f'{injection.reemits_claim_id!r}, which does not exist '
                f'(see {_FIXTURE_DOCS})'
            )
        if not reemitted.canonical or reemitted.cluster_id != injection.cluster_id:
            raise FixtureError(
                f'injection {injection.injection_id!r} re-emits claim '
                f'{injection.reemits_claim_id!r}, which is not the canonical '
                f'claim of cluster {injection.cluster_id!r} — the record gets '
                f'claim_ids=[reemits_claim_id], so pointing it at a peer would '
                f'make the probe measure something other than regrowth of the '
                f'canonical (see {_FIXTURE_DOCS})'
            )


# ---------------------------------------------------------------------------
# Arm materialization
# ---------------------------------------------------------------------------
#
# The three storage shapes E2 arbitrates between (eval-design §5 E2, PRD D9).
# Pinned as a tuple and asserted by EQUALITY in the tests: a fourth arm must
# not be able to appear without the decision table growing a column for it.

ARM_SHAPES: tuple[str, ...] = ('status_quo', 'c_peers', 'b_grouped')

#: Role marking a contamination-slab record, in every arm.
DISTRACTOR_ROLE = 'distractor'

#: Namespace for deriving this experiment's synthetic record ids.
#:
#: uuid5 (SHA-1 over namespace+name) rather than uuid4: arm records must be
#: REPRODUCIBLE across runs, or two runs seed different collections and the
#: report diff stops being signal.  The namespace is a fixed literal — a
#: value derived at runtime would reintroduce exactly the nondeterminism
#: this avoids.  It also keeps the ids canonical 36-char dashed UUIDs, which
#: is what the ``parent_id`` shape rule
#: (``fused_memory.utils.validation.is_full_uuid``, enforced by β's
#: ``validate_memory_metadata``) requires.
_E2_ID_NAMESPACE = uuid.UUID('6f2b7c14-9a3d-4e58-8b71-2c5d0a4f9e63')


def _derive_record_id(shape: str, key: str) -> str:
    """Deterministic canonical dashed UUID for a synthetic arm record."""
    return str(uuid.uuid5(_E2_ID_NAMESPACE, f'{shape}:{key}'))


@dataclass(frozen=True)
class ArmRecord:
    """One record as a single arm would store it.

    ``metadata`` carries ONLY what would really be written to Mem0 — the
    reserved vocabulary keys plus ``category`` — so that
    ``validate_memory_metadata`` can be run over it as a conformance oracle.
    Bookkeeping the experiment needs but the store does not (``record_id``,
    ``cluster_id``, ``claim_ids``, ``role``) lives on the dataclass instead,
    where it cannot manufacture ``unknown_key`` census noise or accidentally
    become a retrievable payload field that flatters one arm.
    """

    record_id: str
    content: str
    metadata: dict[str, Any]
    #: ``None`` for a distractor: it belongs to no cluster by construction.
    cluster_id: str | None
    #: Which blind-authored claims this record realizes.  The basis of the
    #: claim-coverage parity check and of grouped-read recall crediting.
    claim_ids: list[str]
    role: str


def _distractor_records(distractors: list[Distractor]) -> list[ArmRecord]:
    """The shared contamination slab, identical in every arm.

    Carries ``category`` (server-stamped, and what mem0 routes on) but NOT a
    single reserved vocabulary key: a distractor that were topic-anchorable
    would stop being a distractor and start being a right answer for the
    topic pin to find, quietly deleting the contamination variable.
    """
    return [
        ArmRecord(
            record_id=d.distractor_id,
            content=d.content,
            metadata={'category': d.category},
            cluster_id=None,
            claim_ids=[],
            role=DISTRACTOR_ROLE,
        )
        for d in distractors
    ]


def _materialize_status_quo(
    clusters: dict[str, CalibrationCluster], claims: list[ArmClaim],
) -> list[ArmRecord]:
    """Arm (a) — the corpus EXACTLY as it actually existed.

    Long original records, no vocabulary metadata at all (the α fixture
    genuinely carries no ``metadata`` key on any record).  The arm is
    deliberately not "cleaned up": it is the baseline the other two shapes
    are measured against, so any improvement made here would be silently
    subtracted from both of their results.

    A claim is realized in this arm by its ``source_memory_id`` record being
    present — that is what makes claim-coverage parity checkable for an arm
    that was never decomposed into claims.  One original may carry several
    claims, and an original that carries none keeps an empty list.
    """
    claims_by_source: dict[str, list[str]] = {}
    for claim in claims:
        claims_by_source.setdefault(claim.source_memory_id, []).append(claim.claim_id)

    records: list[ArmRecord] = []
    for cluster_id in sorted(clusters):
        for member in sorted(clusters[cluster_id].members, key=lambda r: r['memory_id']):
            memory_id = member['memory_id']
            records.append(ArmRecord(
                record_id=memory_id,
                content=member['content'],
                metadata={'category': member['category']},
                cluster_id=cluster_id,
                claim_ids=sorted(claims_by_source.get(memory_id, [])),
                # The α label IS this record's role in the corpus as it
                # existed, and the guard-adequacy probe selects on it.
                role=member['label'],
            ))
    return records


def _cluster_claims(claims: list[ArmClaim]) -> dict[str, list[ArmClaim]]:
    """Claims grouped by cluster, deterministically ordered by claim id."""
    grouped: dict[str, list[ArmClaim]] = {}
    for claim in claims:
        grouped.setdefault(claim.cluster_id, []).append(claim)
    return {
        cluster_id: sorted(grouped[cluster_id], key=lambda c: c.claim_id)
        for cluster_id in sorted(grouped)
    }


def _claim_categories(
    clusters: dict[str, CalibrationCluster], claims: list[ArmClaim],
) -> dict[str, str]:
    """Map each claim to the category of the α record it was decomposed from.

    Every arm record MUST carry a category, and it must be the SAME category
    the same knowledge has in the status-quo arm.  Two reasons, both
    load-bearing:

    * ``find_near_duplicate_memory`` defensively filters on ``category``
      (``near_duplicate_guard.py:78``).  An arm whose records carried no
      category, or a different one, would score zero guard adequacy for a
      reason that has nothing to do with its storage shape — a silent
      false negative attributable to materialization, not to the arm.
    * mem0 routes and filters on category, so a shifted category would make
      the arms retrieve from differently-shaped candidate pools.

    Deriving it from ``source_memory_id`` rather than assuming one category
    per cluster is deliberate: two of the twenty α clusters are genuinely
    category-mixed, and flattening them would edit the corpus.
    """
    by_memory_id = {
        record['memory_id']: record['category']
        for cluster in clusters.values()
        for record in cluster.members
    }
    categories: dict[str, str] = {}
    for claim in claims:
        category = by_memory_id.get(claim.source_memory_id)
        if category is None:
            raise FixtureError(
                f'claim {claim.claim_id!r} cites source_memory_id '
                f'{claim.source_memory_id!r}, which is not in the α fixture — '
                f'its category cannot be derived'
            )
        categories[claim.claim_id] = category
    return categories


def _materialize_c_peers(
    claims: list[ArmClaim],
    topics: dict[str, RegistryTopic],
    categories: dict[str, str],
) -> list[ArmRecord]:
    """Arm (c) — short single-claim peers, flat, sharing one topic.

    PRD's Option C.  Every peer of a cluster carries the SAME ``topic`` slug,
    taken verbatim from E1's registry (PRD D4: one namespace — E2 invents no
    slug of its own), and exactly one peer carries ``canonical: True``.

    No ``parent_id``: flatness is the shape.  No ``kind`` either — arm (c)'s
    peers are deliberately undifferentiated, which is precisely the property
    arm (b) is being compared against.

    The canonical here is the blind-authored INDEX claim, never a
    concatenation of its siblings (PRD §3).  A rolled-up canonical would win
    claim-recall trivially — every claim would literally be inside it — and
    the experiment would end up measuring a concatenation rather than a peer
    set.
    """
    records: list[ArmRecord] = []
    for cluster_id, cluster_claims in _cluster_claims(claims).items():
        topic = topics[cluster_id].topic
        for claim in cluster_claims:
            metadata: dict[str, Any] = {
                'category': categories[claim.claim_id], 'topic': topic,
            }
            if claim.canonical:
                # Bool identity, not truthiness: β's `invalid_canonical_type`
                # rule rejects a truthy 1 as fatal.
                metadata['canonical'] = True
            records.append(ArmRecord(
                record_id=_derive_record_id('c_peers', claim.claim_id),
                content=claim.text,
                metadata=metadata,
                cluster_id=cluster_id,
                claim_ids=[claim.claim_id],
                role='canonical' if claim.canonical else 'peer',
            ))
    return records


#: Injection modes the +1-re-emission regrowth probe measures (task 4012).
#:
#: Pinned as a tuple and asserted by EQUALITY in the tests, for the same
#: reason ``QUERY_KINDS`` is: a mode that stopped being generated would
#: otherwise vanish from the report as an empty row rather than as a
#: failure.
#:
#: ``unstamped`` is FIRST because it is the case that models reality today —
#: esc-3200-3 measured exactly one topic-stamped record for the topic whose
#: re-emissions it was reading, so every organic re-emission had arrived
#: with no ``topic`` key at all.  ``stamped`` is the write path's best case,
#: and the difference between the two deltas is what a topic-stamping
#: campaign actually buys against regrowth.
REGROWTH_MODES: tuple[str, ...] = ('unstamped', 'stamped')

#: Role marking an injected re-emission.  Distinct from ``DISTRACTOR_ROLE``
#: and ``GROUPED_ROLE``: a re-emission is neither contamination nor a read-
#: time synthetic, it is a record the live system genuinely would have.
REGROWTH_ROLE = 'regrowth'


def materialize_regrowth_injections(
    injections: list[RegrowthInjection],
    claims: list[ArmClaim],
    clusters: dict[str, CalibrationCluster],
    mode: str,
) -> list[ArmRecord]:
    """One +1 re-emission per topic, as the live system would have stored it.

    ``mode`` decides only whether the record carries the ``topic`` key:

    * ``unstamped`` — no ``topic`` key AT ALL (not a present-but-empty one).
      This is the organically observed case, and it is what makes the topic
      pin unable to reach the re-emission.
    * ``stamped`` — ``topic`` present, the write path's best case.

    Never writes ``canonical`` (a second canonical on a topic would make
    ``build_canonical_by_topic`` raise), never ``parent_id`` (the ratified
    shape is flat), never ``contested`` (which has no writer in the live
    system, so a re-emission cannot acquire it), and never ``kind`` (arm
    (c)'s peers are deliberately undifferentiated).

    ``claim_ids=[reemits_claim_id]`` credits the re-emission with realizing
    the claim it restates — see :class:`RegrowthInjection` for why crediting
    it is both the truthful and the conservative modelling.
    """
    if mode not in REGROWTH_MODES:
        raise ValueError(
            f'unknown regrowth mode {mode!r} — the probe measures exactly '
            f'{REGROWTH_MODES} (REGROWTH_MODES). Adding a mode means adding '
            f'rows to the delta table the report renders, not just a branch '
            f'here.'
        )

    categories = _claim_categories(clusters, claims)
    records: list[ArmRecord] = []
    for injection in injections:
        metadata: dict[str, Any] = {'category': categories[injection.reemits_claim_id]}
        if mode == 'stamped':
            metadata['topic'] = injection.topic
        records.append(ArmRecord(
            record_id=_derive_record_id(f'regrowth:{mode}', injection.injection_id),
            content=injection.text,
            metadata=metadata,
            cluster_id=injection.cluster_id,
            claim_ids=[injection.reemits_claim_id],
            role=REGROWTH_ROLE,
        ))
    return records


def regrowth_corpus(
    base_records: list[ArmRecord],
    injections: list[RegrowthInjection],
    claims: list[ArmClaim],
    clusters: dict[str, CalibrationCluster],
    *,
    mode: str,
) -> list[ArmRecord]:
    """The ratified arm's corpus plus one re-emission per topic.

    ORDER IS LOAD-BEARING, which is why it is asserted by a dedicated
    invariant test rather than left as an implementation detail.
    :func:`canonical_record_ids` takes the FIRST record whose ``claim_ids``
    carry a cluster's canonical claim (``index.setdefault``), and the
    injection deliberately carries exactly that claim id.  Prepending an
    injection would therefore silently rename the cluster's canonical to the
    re-emission, and every discoverability number in the regrowth block
    would be scored against the duplicate instead of the true canonical —
    the aliasing failure 3560 had to disclose after the fact for
    ``b_grouped``.

    Returns a NEW list; ``base_records`` is never mutated, so the same
    baseline arm can be reused for both modes.
    """
    return [
        *base_records,
        *materialize_regrowth_injections(injections, claims, clusters, mode),
    ]


def _materialize_b_grouped(
    claims: list[ArmClaim],
    topics: dict[str, RegistryTopic],
    categories: dict[str, str],
) -> list[ArmRecord]:
    """Arm (b) — one canonical per cluster plus ``parent_id`` children.

    PRD's δ / Option B.  Same claim bodies as arm (c) — the two arms differ
    only in how the claims are LINKED, which is what isolates the grouping
    variable from an authoring-quality variable.

    The canonical carries ``canonical: True`` and NO ``kind``: ``'canonical'``
    is not a member of β's ``KIND_REGISTRY`` and would be a fatal
    ``unknown_kind`` violation.  Canonicality is expressed by the
    ``canonical`` key; ``kind`` says what a CHILD is (``amendment`` /
    ``sighting``), and the two are not interchangeable.

    Child ``parent_id`` values are derived from the canonical CLAIM id via
    :func:`_derive_record_id`, never minted randomly, so the
    fixture→arm mapping is reproducible and a rerun seeds identical
    collections.
    """
    records: list[ArmRecord] = []
    for cluster_id, cluster_claims in _cluster_claims(claims).items():
        topic = topics[cluster_id].topic
        canonical_claims = [c for c in cluster_claims if c.canonical]
        if len(canonical_claims) != 1:
            raise FixtureError(
                f'cluster {cluster_id!r} has {len(canonical_claims)} canonical claims, '
                f'expected exactly 1 — arm b_grouped has no parent to point at'
            )
        parent_id = _derive_record_id('b_grouped', canonical_claims[0].claim_id)

        for claim in cluster_claims:
            base = {'category': categories[claim.claim_id], 'topic': topic}
            if claim.canonical:
                metadata: dict[str, Any] = {**base, 'canonical': True}
            else:
                metadata = {**base, 'kind': claim.b_arm_role, 'parent_id': parent_id}
            records.append(ArmRecord(
                record_id=_derive_record_id('b_grouped', claim.claim_id),
                content=claim.text,
                metadata=metadata,
                cluster_id=cluster_id,
                claim_ids=[claim.claim_id],
                role=claim.b_arm_role,
            ))
    return records


def materialize_arm(
    shape: str,
    clusters: dict[str, CalibrationCluster],
    claims: list[ArmClaim],
    topics: dict[str, RegistryTopic],
    distractors: list[Distractor],
) -> list[ArmRecord]:
    """Materialize the same knowledge into one of the three E2 storage shapes.

    Returns the arm's own records followed by the shared distractor slab,
    deterministically ordered so a rerun seeds an identical collection.

    The three arms are held to claim-coverage parity — every blind-authored
    claim is realizable in each of them — which is the mechanical floor
    against the weakness the eval doc names for this very experiment: *"arm
    quality reflects authoring skill — the experiment is gameable by
    authoring one arm well and another lazily."*  Parity is deliberately NOT
    extended to content length: arm (a)'s long originals versus arm (c)'s
    short peers differ by construction, and that difference IS the D4
    tokens-per-query metric.
    """
    if shape not in ARM_SHAPES:
        raise ValueError(
            f'unknown arm shape {shape!r} — E2 arbitrates between exactly '
            f'{ARM_SHAPES}. Adding a shape means adding a column to the '
            f'decision table gate η reads, not just a branch here.'
        )

    categories = _claim_categories(clusters, claims)
    if shape == 'status_quo':
        records = _materialize_status_quo(clusters, claims)
    elif shape == 'c_peers':
        records = _materialize_c_peers(claims, topics, categories)
    else:
        records = _materialize_b_grouped(claims, topics, categories)

    return records + _distractor_records(distractors)


# ---------------------------------------------------------------------------
# Read-side transform: the grouped read (arm-local reference for task 3129)
# ---------------------------------------------------------------------------
#
# ``server/grouped_read.py`` does not exist — 3129 is deferred behind gate η,
# which depends on THIS task, so the premise cannot be supplied from
# downstream.  What follows is the arm-local reference implementation of
# PRD V2/D6, and doubles as their executable specification.
#
# It is a PURE post-ranking transform over an already-fetched hit list: no
# store, no await, no config.  That is not a stylistic choice.  PRD V2
# explicitly forbids this suppression filter from living at
# ``MemoryService.search``, where it would break
# ``mem0_dedup.find_prior_memories``' post-filter and hide candidates from
# the write guard — i.e. it would relocate the very failure grouping exists
# to prevent one layer further down, where it is harder to see.

#: Role marking a synthesised grouped document (never a stored record).
GROUPED_ROLE = 'grouped'


def _render_grouped_document(
    canonical: ArmRecord,
    children: list[ArmRecord],
    sighting_count: int,
) -> str:
    """Canonical body, then child digests, then a sighting COUNT.

    Sightings are counted rather than pasted because that is precisely the
    cost claim grouping makes (PRD D4): a grouped read is supposed to be
    cheaper than N peers, and it can only be cheaper if the repetitive
    "seen it again" entries collapse to a number.  Pasting them would make
    the tokens-per-query metric measure a concatenation and quietly hand
    arm (b) a loss it did not earn.

    Each child is labelled with its OWN ``kind``, not with a fixed
    ``[amendment]``.  ``apply_grouped_read`` passes non-sighting children of
    every kind through this parameter, so a fixed label renamed a child of
    any third kind into an amendment — misattributing what the record is,
    inside the transform that calls itself the executable specification of
    PRD D6.  A child with no ``kind`` at all falls back to ``[child]``, which
    says only what is actually known.
    """
    parts = [canonical.content]
    for child in children:
        kind = child.metadata.get('kind') or 'child'
        parts.append(f'[{kind}] {child.content}')
    if sighting_count:
        parts.append(f'[sightings: {sighting_count}]')
    return '\n'.join(parts)


def apply_grouped_read(
    hits: list[ArmRecord],
    records_by_id: dict[str, ArmRecord],
    *,
    contested_ids: set[str],
) -> list[ArmRecord]:
    """Resolve child hits upward into their parents' grouped documents.

    PRD **D6 (mandatory)**: a child hit resolves UPWARD, so a child's content
    is never unreachable — that unreachability is the whole objection to
    δ/Option B, and a grouped read that did not fix it would be arbitrating
    a strawman.  Every claim whose text the grouped document actually
    RENDERS is credited to the group, or arm (b) would lose claim-recall
    precisely *for grouping correctly*.  Sightings are the exception: they
    collapse to a count, their bodies never reach the reader, and so their
    claims are NOT credited — the crediting rule and the rendering rule are
    held in agreement so arm (b) cannot bank recall and a token discount for
    the same content at once.

    PRD **V2**: a ``contested`` child is NEVER suppressed.  It survives as
    its own hit alongside the grouped document and its body is kept out of
    that document.  This is the esc-5712 shape: a contested amendment folded
    invisibly into a canonical is how a disagreement gets silently resolved
    in favour of whoever wrote the canonical.

    Ranking: a group lands at the BEST rank among its collapsed members.
    Demoting it to the worst would make grouping look worse at every k as an
    artifact of this transform rather than of the storage shape.

    A hit with no ``parent_id`` passes through unchanged and in place.  A
    child whose parent is missing from *records_by_id* also passes through
    unchanged: ``parent_id`` liveness is leaf δ's concern, and dropping the
    hit here would silently delete a real answer.

    Pure — never mutates *hits*, *records_by_id* or any record in them.
    """
    # Which parents are being formed, and at what rank each first appeared.
    group_members: dict[str, list[ArmRecord]] = {}
    group_rank: dict[str, int] = {}
    output: list[tuple[int, ArmRecord]] = []

    for rank, hit in enumerate(hits):
        parent_id = hit.metadata.get('parent_id')

        # V2: contested children never fold. Emitted as themselves, and
        # deliberately NOT registered as group members, so their body cannot
        # reach the grouped document either.
        if parent_id is not None and hit.record_id in contested_ids:
            output.append((rank, hit))
            continue

        if parent_id is None or parent_id not in records_by_id:
            # Parentless, or a dangling link: pass through in place.
            output.append((rank, hit))
            continue

        if parent_id not in group_members:
            group_members[parent_id] = []
            group_rank[parent_id] = rank
        group_members[parent_id].append(hit)

    # A parent that was ITSELF a hit becomes the anchor of its own group
    # rather than a second, duplicate entry.
    resolved: list[tuple[int, ArmRecord]] = []
    for rank, hit in output:
        if hit.record_id in group_members:
            group_rank[hit.record_id] = min(group_rank[hit.record_id], rank)
            continue
        resolved.append((rank, hit))

    for parent_id, members in group_members.items():
        canonical = records_by_id[parent_id]
        amendments = [m for m in members if m.metadata.get('kind') == 'amendment']
        sightings = [m for m in members if m.metadata.get('kind') == 'sighting']
        others = [
            m for m in members
            if m.metadata.get('kind') not in ('amendment', 'sighting')
        ]

        # Credit ONLY the claims whose text actually reaches the reader.
        # Sightings are collapsed to a bare count by
        # `_render_grouped_document`, so their bodies are NOT in the returned
        # payload; crediting them would hand arm (b) claim-recall AND the
        # token discount for the same content — a double advantage in exactly
        # the two columns gate η reads the decision table on.  The D6
        # justification above ("or arm (b) would lose claim-recall precisely
        # for grouping correctly") holds for amendments and other kinds,
        # whose text IS rendered; it does not hold for a counted sighting.
        # If a sighting's claim is meant to be recallable, the fix is to
        # render its body and pay the tokens — not to credit it unrendered.
        credited = [canonical, *amendments, *others]
        claim_ids: list[str] = []
        for member in credited:
            for claim_id in member.claim_ids:
                if claim_id not in claim_ids:
                    claim_ids.append(claim_id)

        resolved.append((group_rank[parent_id], ArmRecord(
            record_id=canonical.record_id,
            content=_render_grouped_document(
                canonical, amendments + others, len(sightings),
            ),
            # A fresh dict: the stored canonical's metadata is never mutated.
            metadata=dict(canonical.metadata),
            cluster_id=canonical.cluster_id,
            claim_ids=claim_ids,
            role=GROUPED_ROLE,
        )))

    # Stable sort on the original rank: equal ranks are impossible (each rank
    # is consumed once), so the order is total and the transform deterministic.
    resolved.sort(key=lambda pair: pair[0])
    return [record for _, record in resolved]


# ---------------------------------------------------------------------------
# Read-side transform: the topic anchor (arm-local reference for task 3111)
# ---------------------------------------------------------------------------
#
# 3111's pin does not exist at ``MemoryService.search`` (zero ``topic`` hits
# in that module) and is deferred behind gate η, so — as with the grouped
# read — the reference implementation lives here.
#
# Because this is a READ-side transform, arm (d) of E2 (each shape ± the pin)
# needs NO extra seeded collection: pin-on and pin-off run over identical
# stored state, making the comparison an exactly-controlled A/B rather than
# two corpora that could rank differently under ANN nondeterminism.


def build_canonical_by_topic(records: list[ArmRecord]) -> dict[str, ArmRecord]:
    """Index ``topic -> the topic's canonical record``.

    In the live driver this is populated via ``Mem0Backend.scroll_by_metadata``
    rather than a search: ``Mem0Backend.search`` exposes no arbitrary metadata
    filter, so ``topic == T AND canonical is True`` is structurally not
    expressible as a query.  Keeping the index construction pure means that
    seam stays testable with no store attached.

    A topic carrying two canonicals RAISES rather than picking a winner.
    Per-(project, topic) canonical uniqueness is leaf ε's rule and cannot be
    enforced from here, but silently choosing would make the pin's answer
    depend on scroll order — non-deterministic between runs, and invisible in
    the report.
    """
    index: dict[str, ArmRecord] = {}
    for record in records:
        topic = record.metadata.get('topic')
        # Bool identity, matching β's `invalid_canonical_type` rule: a truthy
        # 1 is a FATAL violation at the write boundary, so it must not be
        # treated as canonical here either.
        if topic is None or record.metadata.get('canonical') is not True:
            continue
        existing = index.get(topic)
        if existing is not None:
            raise ValueError(
                f'topic {topic!r} has two canonical records '
                f'({existing.record_id!r} and {record.record_id!r}). '
                f'Per-(project, topic) canonical uniqueness is leaf ε\'s rule; '
                f'picking one here would make the pin depend on scroll order.'
            )
        index[topic] = record
    return index


def apply_topic_anchor(
    hits: list[ArmRecord],
    canonical_by_topic: dict[str, ArmRecord],
) -> list[ArmRecord]:
    """Pin each present topic's canonical into the results (PRD D1).

    The pin fires when any hit carries ``metadata['topic'] == T``, and adds
    T's canonical if it is not already present.  It is **ADDITIVE, never
    subtractive**: no hit is dropped, reordered relative to its peers, or
    replaced.  A subtractive pin would hide precisely the candidates
    ``mem0_dedup.find_prior_memories`` needs to see, converting a
    discoverability feature into a write-guard regression.

    Pinned canonicals are appended after the original ranking rather than
    promoted to the front, so a measured improvement is attributable to the
    canonical becoming *reachable at all* rather than to the transform having
    hand-placed it high.

    Appending alone does NOT make that guarantee, though — what enforces it is
    :func:`read_path`'s post-transform truncation at *k*.  Without it an
    appended record simply widens the window, and every rank-based metric
    improves because the variant was handed an extra result.  This function
    stays additive-and-never-subtractive per PRD D1; the window budget is the
    reader's, and it is enforced at the read path.

    Returns the input list unchanged (same contents, same order) when nothing
    is pinnable — the pin is not a rewrite.

    ``canonical is True`` is NOT re-checked here.  β's
    ``invalid_canonical_type`` rule is bool identity — a truthy ``1`` is a
    FATAL violation at the write boundary — and that rule is enforced once,
    at index-construction time, by :func:`build_canonical_by_topic`, which
    skips any record whose flag is not the bool.  Re-asserting it on the
    already-filtered index would be a branch the module's own wiring cannot
    reach, guarding a state the index cannot contain.

    Pure: never mutates *hits*, *canonical_by_topic*, or any record.
    """
    present_ids = {hit.record_id for hit in hits}
    # dict-not-set: insertion order makes the pinned tail deterministic.
    present_topics: dict[str, None] = {}
    for hit in hits:
        topic = hit.metadata.get('topic')
        if topic is not None:
            present_topics.setdefault(topic, None)

    pinned: list[ArmRecord] = []
    for topic in present_topics:
        canonical = canonical_by_topic.get(topic)
        if canonical is None:
            continue
        if canonical.record_id in present_ids:
            continue  # already ranked — never duplicated
        pinned.append(canonical)
        present_ids.add(canonical.record_id)

    if not pinned:
        return hits
    return [*hits, *pinned]


# ---------------------------------------------------------------------------
# Retrieval metrics — rank and set membership ONLY
# ---------------------------------------------------------------------------
#
# WHY THESE DO NOT DELEGATE TO calibrate_write_triage.compute_recall_at_k
# (INV-5: a reader should not have to wonder why there are two recall@k
# implementations in this repo).
#
# That helper's unit is ID MEMBERSHIP: for each retrieval it asks whether one
# ``canonical_id`` appears in a flat list of candidate ids, and it drops
# retrievals flagged ``canonical_present: False`` as corpus gaps.  E2's unit
# is CLAIM REALIZATION: a single hit realizes a SET of claims, and a grouped
# document deliberately realizes many at once — which is the entire point of
# crediting arm (b) fairly.  There is no flat candidate-id list to test
# membership against, and E2 has no corpus-gap class (fixture
# cross-validation already guarantees every expected claim is realizable in
# every arm).  Reshaping E2's data into that signature would mean
# synthesising a fake id list per (query, claim) pair — more code, and it
# would hide the grouped-document case rather than express it.
#
# What IS reused from it, deliberately, is its rule that an empty denominator
# reports None rather than 0.0: "no measurement is not a measured zero".


def claim_recall_at_k(
    hits: list[ArmRecord],
    expected_claim_ids: list[str],
    k: int,
) -> float | None:
    """Fraction of *expected_claim_ids* realized by the top-*k* hits.

    Rank/set-based only (eval-design §1): the sole inputs are the ORDER of
    *hits* and the claim ids each realizes.  No score is read — and none
    could be, since :class:`ArmRecord` has no score field.

    A grouped document realizes every claim it absorbed, so arm (b) is not
    penalised for grouping correctly.  A claim realized by several hits
    counts once — this is recall over the expected SET, not a hit count.

    Returns ``None`` when *expected_claim_ids* is empty: no measurement is
    not a measured zero, and averaging one in would drag an arm's reported
    recall down with non-observations.  An empty *hits* list with a real
    expectation IS a measured zero — the query was scorable and returned
    nothing.
    """
    if not expected_claim_ids:
        return None
    realized: set[str] = set()
    for hit in hits[:k]:
        realized.update(hit.claim_ids)
    expected = set(expected_claim_ids)
    return len(expected & realized) / len(expected)


def topic_discoverability(
    hits: list[ArmRecord],
    topic: str,
    canonical_record_id: str,
    k: int,
) -> dict[str, Any]:
    """Can the topic's canonical be found, and how much of the topic surfaced?

    Returns ``canonical_in_top_k`` (bool), ``canonical_rank`` (**1-based**,
    or ``None`` when the canonical is absent from *hits* entirely) and
    ``topic_member_count`` (distinct records carrying ``metadata['topic'] ==
    topic`` within the top-*k*).

    ``canonical_rank`` is reported even when the canonical falls OUTSIDE the
    window: "absent from top-5" and "absent entirely" are different findings,
    and a decision table that conflated them would hide a shape that gets the
    canonical *nearly* there.

    ``None`` rather than ``0`` for a missing rank is deliberate — ``0`` would
    collide with a real rank under any 0-based reading and would silently
    average as "very good" in a mean-rank summary.

    Rank/set-based only: no score is read.
    """
    rank: int | None = None
    for position, hit in enumerate(hits, 1):
        if hit.record_id == canonical_record_id:
            rank = position
            break

    member_ids = {
        hit.record_id for hit in hits[:k] if hit.metadata.get('topic') == topic
    }
    return {
        'canonical_in_top_k': rank is not None and rank <= k,
        'canonical_rank': rank,
        'topic_member_count': len(member_ids),
    }


# ---------------------------------------------------------------------------
# The D4 cost metric — tokens returned per query
# ---------------------------------------------------------------------------
#
# D4 asks a COST question: a grouped read returns one long document where N
# peers return N short ones — which costs the reader more?  So the metric
# sums the PAYLOADS returned within top-k, which weighs the two shapes on the
# same footing.  A hit-count metric would answer "how many results?", which
# is not the question and would flatter whichever shape happens to fragment
# less.
#
# The number is COMPARATIVE ACROSS ARMS by design.  The report reads "arm (a)
# costs 3.1x arm (c)", never "arm (a) costs 812 tokens" — which is what makes
# the character-proxy fallback below acceptable: a proxy that is uniformly
# off by a constant factor leaves every ratio in the decision table intact.

#: Name reported when the real tokenizer ran.  cl100k_base is the encoding
#: for the OpenAI embedding/chat models this system actually uses.
TIKTOKEN_ESTIMATOR_NAME = 'tiktoken:cl100k_base'

#: Name reported when the fallback ran.  It names the DERIVATION, not merely
#: "fallback": a reader of the artifact has to be able to tell what produced
#: the numbers without reading this file.
CHAR_PROXY_ESTIMATOR_NAME = 'char-proxy:4-chars-per-token'


def character_proxy_tokens(text: str) -> int:
    """Estimate tokens from character count — the no-tiktoken fallback.

    Delegates to ``fused_memory.reconciliation.context_assembler.estimate_tokens``
    (``len(text) // 4``, "~4 chars per token for mixed English/JSON") rather
    than restating the divisor.  That helper is the repo's ONE chars-per-token
    constant and it already budgets exactly this domain — memory payload text
    on its way to a model.  A second literal ``// 4`` here would be a
    same-fact-two-places drift hazard (INV-5), and the two would be free to
    disagree without any test noticing.

    The import is function-local so importing this script costs nothing until
    a token is actually counted.
    """
    from fused_memory.reconciliation.context_assembler import (  # noqa: PLC0415
        estimate_tokens,
    )

    return estimate_tokens(text)


def resolve_token_estimator() -> tuple[str, Any]:
    """Pick the token estimator and NAME it: ``(name, encode)``.

    Prefers real tokenization and falls back to :func:`character_proxy_tokens`
    — but the fallback is never silent.  The name travels into every
    :func:`tokens_returned` result and from there into the report JSON, so an
    operator reading ``plans/e2-storage-shape-bakeoff-report.md`` can always
    tell which estimator produced its numbers.  Un-flagged proxy numbers in
    an operator-facing artifact are indistinguishable from measured ones,
    which is the failure this avoids.

    Two distinct failure modes both fall back:

      * tiktoken is not installed (the live path in this venv today);
      * tiktoken imports but ``get_encoding`` fails — it fetches its BPE
        file over the network on first use, so an offline or air-gapped box
        raises here.  A cost metric must not take the whole bake-off down
        with it.

    Deliberately NOT cached: a cached resolution would make every later
    caller in the process report whichever estimator the FIRST one happened
    to get, which is precisely the silent-substitution failure above wearing
    a different hat.  Resolution is cheap and happens once per run; the
    encoding object itself is captured in the closure, so the BPE table is
    built at most once per resolution rather than per call.
    """
    try:
        # tiktoken is an OPTIONAL dep, absent from this venv — the try/except
        # below is the whole point, so pyright's unresolved-import complaint is
        # expected rather than a defect.
        import tiktoken  # type: ignore[reportMissingImports]  # noqa: PLC0415

        encoding = tiktoken.get_encoding('cl100k_base')
    except Exception:  # noqa: BLE001 — absent, broken, or offline: same answer
        return CHAR_PROXY_ESTIMATOR_NAME, character_proxy_tokens

    def _encode(text: str) -> int:
        return len(encoding.encode(text))

    return TIKTOKEN_ESTIMATOR_NAME, _encode


def tokens_returned(
    hits: list[ArmRecord],
    k: int,
    estimator: tuple[str, Any] | None = None,
) -> dict[str, Any]:
    """Tokens a reader must consume for the top-*k* of *hits* (D4's cost).

    Sums over the returned payloads, so one grouped document and the N short
    peers carrying the same knowledge are weighed identically — that
    comparison IS the metric.

    Counts ``content`` only.  Metadata is excluded on purpose: how a result
    dict is rendered on the wire is a transport/formatting choice gate η is
    not deciding, and folding it in would make the arm comparison move with
    that choice.  It would also charge the shapes that carry β vocabulary
    keys for carrying them — a real question, but a different one from D4's
    payload cost.

    *estimator* is a ``(name, encode)`` pair, defaulting to
    :func:`resolve_token_estimator`.  The name is returned alongside the
    count; see that function for why it must be.

    Note on the character-proxy path: ``len // 4`` floors once per payload,
    so an N-hit arm can be undercounted by up to N-1 tokens relative to the
    same text returned as one document.  Bounded by ``k`` (≤10 here) against
    payloads of hundreds of characters, that is far below the effect sizes
    the decision table reads, and it biases AGAINST the fragmented shape's
    apparent cost — i.e. it cannot manufacture a win for the grouped arm.
    """
    name, encode = estimator if estimator is not None else resolve_token_estimator()
    window = hits[:k]
    return {
        'tokens': sum(encode(hit.content) for hit in window),
        'estimator': name,
        'payloads_counted': len(window),
    }


# ---------------------------------------------------------------------------
# Near-dup-guard candidate adequacy
# ---------------------------------------------------------------------------
#
# eval-doc :324-326: "would the write that became duplicate N+1 have been
# matched?"  Reported SPLIT IN TWO — see this module's docstring for why the
# threshold half cannot be made rank-based without ceasing to measure the
# real guard, and why it is flagged rather than dropped.

#: How many results the guard sees.  Production's pre-check search runs with
#: ``limit=5`` (``server/tools.py:1556``), so a replay over a wider window
#: would measure a more generous guard than the one that is running.
GUARD_TOP_K = 5

#: The ONLY category production's near-duplicate guard covers.
#:
#: ``find_near_duplicate_memory``'s ``category`` parameter defaults to
#: ``MemoryCategory.procedural_knowledge`` and filters candidates on it
#: unconditionally (``server/near_duplicate_guard.py:88-95``), and the
#: pre-check itself only runs on explicit procedural_knowledge writes.  So a
#: probe of any other category cannot produce ``guard_matched`` under ANY
#: storage shape, which puts an identical ceiling on every arm's
#: ``guard_matched_rate`` for a reason that is not about storage shape.
#:
#: The STRING, not the enum: this module's pure core must import nothing from
#: ``fused_memory`` (it is loaded by path, in tests with no package on the
#: path), and the fixture stores categories as strings anyway.  The value is
#: pinned against the real enum by a test.
GUARD_COVERED_CATEGORY = 'procedural_knowledge'


@dataclass(frozen=True)
class ScoredHit:
    """An :class:`ArmRecord` plus the score the store returned for it.

    A deliberately SEPARATE type rather than a score field on ``ArmRecord``:
    eval-design §1 bans absolute-score metrics, and the threshold replay is
    the single sanctioned exception.  Quarantining the score in a wrapper
    only this metric consumes makes that structural — the rank-based metrics
    could not read a score even by accident, because the type they take does
    not have one.
    """

    record: ArmRecord
    relevance_score: float


def as_memory_results(hits: list[ScoredHit]) -> list[Any]:
    """Adapt arm hits to the ``MemoryResult`` objects the selector takes.

    ``find_near_duplicate_memory`` reads ``.category`` and ``.source_store``
    as ENUMS and the per-store cosine from ``.metadata['store_score']``.
    Dicts would raise; dicts of plain strings would be worse — every enum
    comparison would silently fail and the report would state "the guard never
    fires" for every arm.

    Since task 3658 the cosine's home is ``metadata['store_score']``:
    ``MemoryService.search`` puts an ORDINAL Reciprocal Rank Fusion value in
    ``relevance_score`` (rank-1 is 1/61 ~ 0.0164), so the guard thresholds on
    the metadata field instead.  This adapter therefore reproduces the real
    post-fusion shape — the arm's cosine in ``store_score``, the fused ordinal
    in ``relevance_score`` — rather than the pre-3658 shape.  Putting the
    cosine anywhere else here would make guard_adequacy report
    ``guard_matched: False`` for every arm in the program: the same silent
    "the guard never fires" corruption this docstring already warns about,
    arrived at by a different route.

    Rank order is preserved: the selector takes the max by score, but the
    report quotes the matched id back against the ranked window, and a
    reordering adapter would make that evidence point at the wrong record.
    ``store_rank`` is the 1-based position in that same window.

    A category the enum does not know becomes ``None`` rather than a guess —
    that record is exactly what the selector's defensive filter exists to
    drop, and inventing a category would defeat it.

    ``source_store`` is ``mem0`` for every hit: the arms are seeded into
    mem0/Qdrant and production's pre-check searches ``stores=['mem0']``.
    """
    from fused_memory.models.enums import MemoryCategory, SourceStore  # noqa: PLC0415
    from fused_memory.models.memory import MemoryResult  # noqa: PLC0415
    from fused_memory.services.memory_service import RRF_K  # noqa: PLC0415

    results = []
    for rank, hit in enumerate(hits, start=1):
        raw_category = hit.record.metadata.get('category')
        try:
            category = MemoryCategory(raw_category)
        except ValueError:
            category = None
        results.append(MemoryResult(
            id=hit.record.record_id,
            content=hit.record.content,
            category=category,
            source_store=SourceStore.mem0,
            relevance_score=1.0 / (RRF_K + rank),
            metadata={'store_rank': rank, 'store_score': hit.relevance_score},
        ))
    return results


def resolve_guard_threshold(memory_service: Any = None) -> float:
    """The threshold the guard would really run at.

    Delegates to ``near_duplicate_guard.resolve_near_dup_threshold``, which
    navigates the config defensively and falls back to that module's
    ``_DEFAULT_NEAR_DUP_THRESHOLD``.  The value is NOT restated here — not
    even in this docstring — because a copy in this script would keep
    reporting the old number for months after production moved, and a report
    that names a threshold nobody is running is worse than one that does not
    name it at all.

    ``None`` is accepted for the pure tests and for a dry run — the delegate
    already treats every missing hop as "use the default", so no separate
    branch is needed.
    """
    from fused_memory.server.near_duplicate_guard import (  # noqa: PLC0415
        resolve_near_dup_threshold,
    )

    return resolve_near_dup_threshold(memory_service)


def guard_adequacy(
    top5_hits: list[ScoredHit],
    sibling_record_ids: set[str],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Would the guard have caught the write that became duplicate N+1?

    Returns both halves of the answer, which are independent by design:

    ``candidate_present``
        Rank/set-based and score-free: is any record from the probing write's
        own cluster in the arm's top-5 at all?  This is the drift-proof half
        and the one that discriminates between storage shapes.  A shape that
        put the right sibling at rank 1 did its job even if the cosine came
        in under threshold; collapsing the two halves would report that as a
        shape failure and the decision table would blame the shape for the
        threshold.

    ``guard_matched`` / ``guard_matched_id``
        The REAL selector's verdict, with ``threshold_replay: True`` and the
        threshold value alongside it so no reader mistakes it for something
        stable enough to trend across an embedding-config change.  The bool
        is the aggregable form (a rate across clusters); the id is the
        evidence a reader checks the rate against.

    *sibling_record_ids* are the arm's records for the probe's cluster.  A
    grouped document is credited correctly without special-casing: it carries
    its canonical's ``record_id``, and the canonical is a cluster sibling.

    Windowed to :data:`GUARD_TOP_K` defensively — a caller handing over ten
    hits must not accidentally measure a more generous guard than production
    runs.
    """
    from fused_memory.server import near_duplicate_guard  # noqa: PLC0415

    if threshold is None:
        threshold = resolve_guard_threshold()
    window = top5_hits[:GUARD_TOP_K]

    match = near_duplicate_guard.find_near_duplicate_memory(
        as_memory_results(window), threshold,
    )
    return {
        'candidate_present': any(
            hit.record.record_id in sibling_record_ids for hit in window
        ),
        'guard_matched': match is not None,
        'guard_matched_id': None if match is None else match.id,
        'threshold_replay': True,
        'threshold': threshold,
    }


def select_probing_write(cluster: CalibrationCluster) -> dict[str, Any] | None:
    """The write that became duplicate N+1 for *cluster*, or ``None``.

    The chronologically LAST ``label == 'duplicate'`` member: that is the
    write the guard would have had to catch, and its content is the probe
    query.  Ties break on ``memory_id`` (max) — the 20 canonicals carry
    ``created_at: null`` and nothing forbids two duplicates sharing a stamp,
    so an unstable tiebreak would move the probe query, and with it the whole
    guard column, between runs of an experiment whose report is supposed to
    be diffable.

    Only ``duplicate``-labelled members are eligible.  A ``distinct`` or
    ``pseudo_contradiction`` member is by definition NOT a duplicate N+1, and
    probing with one would measure whether the guard fires on content it is
    specifically supposed to let through.

    ``None`` — never a substituted canonical — when the cluster has no
    duplicate at all.  5 of the committed fixture's 20 clusters really are
    duplicate-free, and probing them with *something* would manufacture 5
    fake measurements in a 20-row table.

    Some measurable clusters have a probe whose category is not
    :data:`GUARD_COVERED_CATEGORY`, which production's guard does not cover at
    all (it runs only on explicit procedural_knowledge writes, and
    ``find_near_duplicate_memory`` filters candidates to that category
    unconditionally).  Those rows are measured the same way rather than
    dropped here — dropping them would hide that the guard's remit is narrower
    than the corpus.  The disclosure is ``guard_adequacy.guard_covered_probes``
    in the report JSON and the ``(n=covered/probes)`` suffix on the decision
    table's guard column: without it every arm carries an identical,
    invisible ceiling on ``guard_matched_rate``.
    """
    duplicates = [m for m in cluster.members if m.get('label') == 'duplicate']
    if not duplicates:
        return None
    return max(
        duplicates,
        key=lambda record: (record.get('created_at') or '', record.get('memory_id') or ''),
    )


# ---------------------------------------------------------------------------
# PRD D10 — audit-recall over alpha/3130's labeled fixture
# ---------------------------------------------------------------------------
#
# "ζ also delivers the audit-recall measurement — run
# audit_duplicate_memories.py against α/3130's labeled fixture and report
# recall on the paraphrase class — the number that decides how much to trust
# the κ report."
#
# The REAL detector and the REAL partition are loaded and run.  A
# reimplementation of either would measure this file's idea of the audit
# script, which is the one number nobody wants.  Both live in
# ``fused-memory/scripts/`` as standalone scripts rather than importable
# package modules, so they are loaded by path — package-relative, never
# against an absolute checkout path (the lesson pinned at
# tests/test_calibrate_write_triage.py:1267).
#
# NO THRESHOLD IS ASSERTED ANYWHERE ON THE RESULT (gate G6): the measurement
# informs a judgement about how far to trust the κ sweep; it does not gate a
# build.

_SCRIPTS_DIR = Path(__file__).resolve().parent

#: How many nearest-miss paraphrase pairs the payload shows.  Enough to audit
#: the band split by hand, few enough that the artifact stays readable.
_PARAPHRASE_EXEMPLAR_COUNT = 5


def _load_sibling_script(name: str) -> Any:
    """Load a sibling script in ``scripts/`` by path.

    Registered in ``sys.modules`` under its bare name: ``@dataclass`` and
    other reflection-based decorators look the defining module up there, so
    a module loaded without registering breaks on the way in.
    """
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    if name in sys.modules:
        return sys.modules[name]
    path = _SCRIPTS_DIR / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FixtureError(f'cannot load sibling script {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_audit_script() -> Any:
    """``scripts/audit_duplicate_memories.py`` — the detector under measure."""
    return _load_sibling_script('audit_duplicate_memories')


def load_calibration_script() -> Any:
    """``scripts/calibrate_write_triage.py`` — the labeled-pair partition."""
    return _load_sibling_script('calibrate_write_triage')


def load_read_transform_script() -> Any:
    """``scripts/read_transform_selection.py`` — 4004's read-transform arms.

    LAZY BY NECESSITY, not merely by style.  ``read_transform_selection``
    loads THIS module through its own lazy ``bake_off()`` accessor, so an
    import-time load in either direction closes the cycle.  Calling this at
    use time (inside :func:`read_path`) keeps both scripts importable on
    their own.

    Laziness is also sufficient for correctness.
    ``apply_promoting_topic_anchor`` is duck-typed over ``.record_id`` and
    ``.metadata`` with no ``isinstance`` check, so even if the two scripts
    end up holding distinct ``sys.modules`` entries for the bake-off,
    :class:`ArmRecord` instances from either copy still compare and
    transform correctly.

    The transform itself is 4004's deliverable and is NOT reimplemented
    here: two copies could drift, and the point of the regrowth probe is to
    measure the transform 4004's selection table actually picked.
    """
    return _load_sibling_script('read_transform_selection')


def load_labeled_fixture(path: str | Path = DEFAULT_ALPHA_FIXTURE_PATH) -> list[dict[str, Any]]:
    """The alpha fixture as a flat record list, via its own strict loader.

    Delegates to ``calibrate_write_triage.load_fixture`` rather than reusing
    this module's :func:`_read_jsonl`: D10's whole claim is that it measures
    the real detector over the real partition, and the partition builder is
    fed by that loader.  Reading the fixture through a second parser would
    leave room for the two to disagree about what the population even is.
    """
    return load_calibration_script().load_fixture(Path(path))


def resolve_audit_threshold() -> float:
    """The lexical threshold the audit script itself defaults to.

    Read from the detector's own signature rather than restated here — that
    script carries the value in three places already, and a fourth copy in a
    measurement script would go on reporting the old number for months after
    a retune.  A stale threshold in the artifact is worse than none: it makes
    the recall look like a fact about a detector nobody is running.
    """
    import inspect  # noqa: PLC0415

    detector = load_audit_script().find_near_duplicate_memory_groups
    params = inspect.signature(detector).parameters
    if 'threshold' not in params:
        # A bare `KeyError: 'threshold'` deep in a live D10 run says nothing
        # about which upstream moved or what to do.  Plain RuntimeError
        # rather than MeasurementError: nothing was mis-measured here, the
        # detector this script exists to replay has changed shape.
        raise RuntimeError(
            f'{detector.__module__}.{detector.__name__} has no `threshold` '
            f'parameter (signature: {sorted(params)}). D10 reads the audit '
            f'threshold off the detector\'s own signature so the artifact '
            f'cannot report a number nobody runs; a rename upstream has to '
            f'be followed here, not defaulted around.'
        )
    return float(params['threshold'].default)


def max_lexical_ratio(left: str, right: str) -> float | None:
    """Best ``SequenceMatcher.ratio()`` over BOTH argument orders, or ``None``.

    ``ratio()`` is order-sensitive — the known exemplar pair scores 0.0948
    one way and 0.2279 the other on normalised content (pinned at
    ``test_audit_duplicate_memories.py:1116-1119``).  A one-directional ruler
    would place a pair in the paraphrase band depending on which id happened
    to sort first, which is not a property of the pair.  Taking the max can
    only under-claim the paraphrase band, never over-claim it.

    Normalisation and the ratio computation are both the detector's own
    (``_normalized_contents`` / ``_max_lexical_ratio``), so the band split
    describes the detector that is actually running.  Crossing those helpers'
    leading underscore is deliberate: ``_normalized_contents``' own docstring
    asks callers to share it precisely because a second copy would
    "silently desynchronise numbers a reader is invited to compare" — and
    this report prints the band split beside the detector's verdict.

    ``None`` when either side normalises to empty: the detector refuses to
    cluster empty content (an unextractable memory must never be deleted as
    a duplicate), so there is no ratio to report.  ``SequenceMatcher`` would
    answer 1.0, which is exactly the wrong answer.
    """
    audit = load_audit_script()
    normalized = audit._normalized_contents([{'content': left}, {'content': right}])
    if not normalized[0] or not normalized[1]:
        return None
    forward, _ = audit._max_lexical_ratio(normalized, [0, 1])
    backward, _ = audit._max_lexical_ratio(normalized, [1, 0])
    candidates = [r for r in (forward, backward) if r is not None]
    return max(candidates) if candidates else None


def _rate(numerator: int, denominator: int) -> float | None:
    """A fraction, or ``None`` when nothing was measured.

    An empty denominator is NO measurement, not a measured zero — a 0.0
    false-grouping rate for a class with no members reads as a perfect score
    for something never tested.  Same rule ``calibrate_write_triage``'s own
    recall helper applies.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def audit_recall_over_labeled_fixture(
    records: list[dict[str, Any]],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Replay the real lexical audit detector over a labeled corpus (D10).

    Runs ``audit_duplicate_memories.find_near_duplicate_memory_groups`` and
    scores the grouping it recovers against
    ``calibrate_write_triage.build_pair_sets``' three-class partition: recall
    over the curator-confirmed positives, and false-grouping counts over the
    hard negatives (same cluster, curator-ruled NOT duplicates) and the
    unrelated pairs.  Recall alone would be unreadable — a detector that
    groups everything scores 1.0.

    The positive class is split into two bands by :func:`max_lexical_ratio`
    against *threshold*:

      ``lexical_band``
          The detector could reach these.  "Did it?" is a fair question about
          the detector.
      ``paraphrase_band``
          Structurally unreachable by a character-level threshold at any
          tuning short of changing kind.  Counting these as plain misses
          would read as "the audit script is broken" rather than "this class
          is invisible to it" — and the second is what D10 asks about.

    Emits the nearest-miss paraphrase pairs so the split can be audited by
    hand: a band assignment nobody can check is a number to be believed
    rather than read.

    No threshold, bound or rate is asserted on the result anywhere (G6).
    """
    if threshold is None:
        threshold = resolve_audit_threshold()

    audit = load_audit_script()
    calibration = load_calibration_script()

    groups = audit.find_near_duplicate_memory_groups(
        [{'id': r['memory_id'], 'content': r['content']} for r in records],
        threshold,
    )
    # Which group each record landed in; absent means it was never grouped.
    group_of: dict[str, int] = {}
    for index, group in enumerate(groups):
        for member in group:
            group_of[member['id']] = index

    def _grouped_together(pair: dict[str, str]) -> bool:
        left = group_of.get(pair['a'])
        return left is not None and left == group_of.get(pair['b'])

    pair_sets = calibration.build_pair_sets(records)
    contents = {r['memory_id']: r.get('content') or '' for r in records}

    lexical: list[dict[str, Any]] = []
    paraphrase: list[dict[str, Any]] = []
    for pair in pair_sets['true_dup_pairs']:
        ratio = max_lexical_ratio(contents[pair['a']], contents[pair['b']])
        entry = {'a': pair['a'], 'b': pair['b'], 'max_ratio': ratio}
        # An unmeasurable ratio (empty content) is unreachable by definition:
        # the detector declines to cluster it at ANY threshold.
        if ratio is not None and ratio >= threshold:
            lexical.append(entry)
        else:
            paraphrase.append(entry)

    def _band(entries: list[dict[str, Any]]) -> dict[str, Any]:
        recovered = sum(1 for e in entries if _grouped_together(e))
        return {
            'pairs': len(entries),
            'recovered': recovered,
            'recall': _rate(recovered, len(entries)),
        }

    def _negatives(pairs: list[dict[str, str]]) -> dict[str, Any]:
        false_groupings = sum(1 for p in pairs if _grouped_together(p))
        return {
            'pairs': len(pairs),
            'falsely_grouped': false_groupings,
            'rate': _rate(false_groupings, len(pairs)),
        }

    positives = pair_sets['true_dup_pairs']
    recovered = sum(1 for p in positives if _grouped_together(p))

    # Nearest misses first: "how far would the threshold have to fall to
    # reach this class at all?" is the decision-relevant end of the band.
    # Ratio is sorted descending with an id tiebreak so a rerun's diff is
    # signal rather than iteration-order noise; None sorts last (-1.0).
    exemplars = sorted(
        paraphrase,
        key=lambda e: (-(e['max_ratio'] if e['max_ratio'] is not None else -1.0),
                       e['a'], e['b']),
    )[:_PARAPHRASE_EXEMPLAR_COUNT]

    return {
        'detector': 'audit_duplicate_memories.find_near_duplicate_memory_groups',
        'threshold': threshold,
        'groups_found': len(groups),
        'true_dup': {
            'pairs': len(positives),
            'recovered': recovered,
            'recall': _rate(recovered, len(positives)),
            'lexical_band': _band(lexical),
            'paraphrase_band': _band(paraphrase),
        },
        'hard_negative': _negatives(pair_sets['hard_negative_pairs']),
        'unrelated': _negatives(pair_sets['unrelated_pairs']),
        'paraphrase_exemplars': exemplars,
    }


# ---------------------------------------------------------------------------
# The report — the artifact gate η reads
# ---------------------------------------------------------------------------

#: The six rows of the decision table: three storage shapes x pin on/off.
#:
#: The pin is a READ-side transform, so its variants reuse their shape's
#: seeded collection — pin-on and pin-off run over identical stored state,
#: making that comparison an exactly-controlled A/B rather than two corpora
#: that could rank differently under ANN nondeterminism.  They are still
#: separate ROWS: "does the pin help *this shape*?" is a question the table
#: has to answer per shape, not once globally.
ARM_VARIANTS: tuple[str, ...] = tuple(
    variant for shape in ARM_SHAPES for variant in (shape, f'{shape}+pin')
)

#: Metrics every arm must carry, and the nested keys within them that a
#: shallow ``in`` check would miss.  Enumerated ONCE and used by both the
#: completeness check and the renderer, so a metric cannot be validated into
#: the JSON and then quietly dropped from the table.
_REQUIRED_ARM_METRICS: dict[str, tuple[str, ...]] = {
    # Both halves: "was the pin on?" and "did it change anything?".  Two
    # identical rows are ambiguous without the second — "the pin is useless"
    # and "the pin never fired" are different findings.
    'pin': ('enabled', 'window_changed_rate'),
    'claim_recall': ('at_5', 'at_10'),
    # The rate, AND the censored denominator the median rank is taken over:
    # a median over successes only lets an arm that rarely finds the
    # canonical print the best rank, so the two must travel together.
    #
    # The `stored_` trio travels with them for the same class of reason — see
    # the module docstring, "Rank-based is not transform-blind".  Registered
    # rather than merely produced, because a table carrying only the
    # transform-credited half IS the undisclosed state it exists to fix.
    'discoverability': ('canonical_in_top_5_rate', 'median_canonical_rank',
                        'canonical_found_count', 'canonical_candidates',
                        'stored_canonical_in_top_5_rate',
                        'stored_canonical_median_rank',
                        'stored_canonical_found_count'),
    # eval-design §5 E2 names claim recall and discoverability as DISTINCT
    # metrics, and the fixture splits its queries accordingly. Required, not
    # optional: a pooled-only report cannot tell a shape that wins on claim
    # queries and loses on topic phrasings from one that ties on both.
    'by_query_kind': (*QUERY_KINDS, HELD_OUT_SUBSET),
    'tokens_per_query': ('mean', 'estimator'),
    # Both halves, always: the rank-based one and the threshold replay —
    # plus the best score the replay saw, which is what separates "no
    # candidate cleared the threshold" from "no score ever arrived" — plus
    # the covered-probe denominator, without which the replay column carries
    # an identical, invisible ceiling on every arm.
    'guard_adequacy': ('candidate_present_rate', 'guard_matched_rate',
                       'threshold_replay', 'threshold', 'max_observed_score',
                       'probes', 'guard_covered_probes'),
}

#: The regrowth block's completeness, enumerated ONCE and shared by the
#: check and the renderer — the same reason ``_REQUIRED_ARM_METRICS`` is a
#: constant rather than two hand-kept lists.
_REQUIRED_REGROWTH_DESCRIPTORS: tuple[str, ...] = (
    'shape', 'read_arms', 'modes', 'topics_injected', 'injections_per_topic',
    'injection_fixture',
)
#: The flat ``{arm: {metric: value}}`` tables, and the per-mode ones.
_REQUIRED_REGROWTH_KEYS: tuple[str, ...] = ('baseline', 'stamping_value')
_REQUIRED_REGROWTH_MODE_KEYS: tuple[str, ...] = ('after', 'deltas')

#: What each ``by_query_kind`` subset must carry, on top of the kind names
#: registered above.  The metric BLOCKS are named here; the keys inside them
#: are the same tuples the pooled block is checked against, reused rather than
#: retyped — a subset is the pooled block computed over fewer rows (one
#: ``_aggregate_queries`` call for both), so a second key list here could
#: drift into disagreeing with the code that produces both.
#:
#: Validating only the kind NAMES was a fail-obscure gap: the by-kind table
#: subscripts ``subset['discoverability'][...]`` and ``subset['queries']``
#: directly, so a subset missing one surfaced as a raw ``KeyError`` out of
#: ``render_markdown`` instead of an :class:`IncompleteReportError` naming the
#: arm, the kind and the key.
_REQUIRED_SUBSET_KEYS: tuple[str, ...] = ('queries',)
_REQUIRED_SUBSET_METRICS: tuple[str, ...] = ('claim_recall', 'discoverability')

#: What the artifact must say about how it was produced.  An arbitration
#: artifact whose provenance is not in it cannot be re-read six months later
#: by somebody who was not in the room.
_REQUIRED_PROTOCOL_KEYS: tuple[str, ...] = (
    'blind_authoring', 'fixtures', 'token_estimator', 'guard_threshold',
    'distractor_slab_size', 'embedder_model',
)

#: The decision table's columns, pinned in one place.  A column quietly
#: dropped from a decision table is a metric quietly dropped from the
#: decision, so the test asserts this by equality rather than substring.
DECISION_TABLE_COLUMNS: tuple[str, ...] = (
    'arm',
    'claim recall@5',
    'claim recall@10',
    'canonical in top-5',
    # The same question, asked of the RAW store hits — see the module
    # docstring, "Rank-based is not transform-blind".  ADJACENT on purpose:
    # the gap between the two IS the read transform's contribution, and a
    # qualifier placed anywhere else is not a qualifier on this number.
    'canonical in top-5 (stored)',
    'median canonical rank',
    'tokens/query',
    'guard candidate present',
    'guard matched (replay)',
    # Reads the other columns for you: a +pin row identical to its twin is
    # "the pin never fired" when this is 0.00, not "the pin is useless".
    'pin changed window',
)

#: The regrowth delta tables' metric labels, index-aligned with
#: ``REGROWTH_METRICS`` (defined with the probe, far below — a test pins the
#: two tuples to the same length, so a metric added without a label cannot
#: silently lose a column).  Both `canonical in top-5` columns are named
#: with their semantics in the header itself: `(stored)` is the SCORED
#: retrieval property, `(credited)` is what the read transform handed the
#: reader, and a header that distinguished them only by position would let
#: either be quoted as the other.
REGROWTH_METRIC_LABELS: tuple[str, ...] = (
    'claim recall@5',
    'claim recall@10',
    'canonical in top-5 (stored)',
    'median canonical rank (stored)',
    'canonical found (stored)',
    'canonical in top-5 (credited)',
    'tokens/query',
)

#: Which SIGN of a regrowth delta is the COST, per metric.  ``-1`` means a
#: NEGATIVE delta is the cost (recall and discoverability rates: losing them
#: is what regrowth does); ``+1`` means a POSITIVE delta is the cost
#: (``tokens/query``, and ``median canonical rank`` — a bigger rank NUMBER is
#: a worse rank).
#:
#: ENUMERATED rather than inferred, deliberately.  "Higher is better" is true
#: of five of these seven columns and false of the other two, so a helper
#: that guessed would get exactly those two silently wrong — in the one
#: sentence a gate reads.  ``_check_regrowth_cost_signs`` (called at import,
#: below :func:`_regrowth_metric_keys`) pins the key set to
#: ``REGROWTH_METRICS``, so adding a metric without declaring its direction
#: fails loudly instead of defaulting to "higher is better".
#:
#: Consumed by :func:`_absorption_phrase` and :func:`_cost_phrase`, which are
#: the only two places in this renderer that turn a delta into a DIRECTIONAL
#: claim.  The tables themselves render signed values through
#: :func:`_gap_cell` and make no directional claim, so they need none of this.
_REGROWTH_COST_SIGN: dict[str, int] = {
    'claim_recall.at_5': -1,
    'claim_recall.at_10': -1,
    'discoverability.stored_canonical_in_top_5_rate': -1,
    'discoverability.stored_canonical_median_rank': +1,
    'discoverability.stored_canonical_found_count': -1,
    'discoverability.canonical_in_top_5_rate': -1,
    'tokens_per_query.mean': +1,
}

#: The regrowth delta table's columns, pinned in one place for the same
#: reason ``DECISION_TABLE_COLUMNS`` is: a column quietly dropped from a
#: delta table is a metric quietly dropped from the decision, so the test
#: asserts this by equality rather than substring.  One row per
#: ``(mode, read arm)``; each metric cell carries baseline, after AND delta
#: (see :func:`_regrowth_cell` for why three numbers share a cell).
REGROWTH_TABLE_COLUMNS: tuple[str, ...] = (
    'mode', 'read arm', *REGROWTH_METRIC_LABELS,
)

#: The stamping-value table's columns — ``stamped delta - unstamped delta``,
#: one row per read arm.  Built from the SAME labels, so the two tables
#: cannot drift into naming the same metric differently.
REGROWTH_STAMPING_COLUMNS: tuple[str, ...] = ('read arm', *REGROWTH_METRIC_LABELS)

#: Rendered for a measurement that is None.  Deliberately NOT '0.00': the
#: whole point of carrying None through the pipeline is that "no measurement"
#: and "measured zero" are different findings, and a table that prints them
#: identically throws that away at the last step.
_NO_MEASUREMENT = '—'

DEFAULT_REPORT_JSON = _PACKAGE_ROOT.parent / 'plans' / 'e2-storage-shape-bakeoff-report.json'
DEFAULT_REPORT_MD = _PACKAGE_ROOT.parent / 'plans' / 'e2-storage-shape-bakeoff-report.md'

#: Bumped when the JSON's shape changes, so a reader of an old artifact is
#: never silently comparing two different schemas.
#: v2 — the `discoverability` block gained the transform-blind trio
#: (`stored_canonical_in_top_5_rate` / `_median_rank` / `_found_count`), so a
#: v1 artifact and a v2 one answer different questions in the same column
#: names and must not be diffed as if they were the same schema.
#: v3 — the artifact gained the `regrowth` block (the +1-re-emission probe,
#: task 4012).  A v2 artifact carries no block at all, so a v2 and a v3 one
#: answer different questions about the same six arms and must not be diffed
#: as if they were the same schema either.
REPORT_SCHEMA_VERSION = 3


class IncompleteReportError(RuntimeError):
    """A measurement is missing, so no table is emitted.

    Named rather than a bare ValueError because the failure mode it guards is
    specific: a decision table with a blank cell reads as "measured, and equal
    to its neighbour", and the reader has no way to tell that from "never
    measured".  Refusing to emit is the loud alternative.
    """


def _repo_relative(path: str | Path) -> str:
    """``fused-memory/tests/fixtures/…``, never an absolute home-directory path.

    Shared by :func:`fixture_provenance` and the regrowth block so the two
    cannot name the same file differently.  An artifact naming somebody's
    absolute checkout is neither reproducible nor readable by anyone else,
    and it leaks the worktree the run happened in.  Falls back to the bare
    filename for a path outside the repo, rather than emitting the absolute
    one.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_PACKAGE_ROOT.parent))
    except ValueError:
        return resolved.name


def _protocol_fixture_paths(
    base: list[Any], *, regrowth: dict[str, Any] | None,
) -> list[Any]:
    """The fixtures whose provenance this run is entitled to claim.

    The injection fixture joins the table only when the probe actually READ
    it.  Provenance for a file the run never opened is a false audit trail —
    the same failure :func:`fixture_provenance` avoids by reporting ``None``
    for an untracked path rather than falling back to HEAD.

    Gated on ``is not None`` rather than truthiness: an empty block is still
    a block, and a probe that measured nothing still opened the fixture.
    """
    if regrowth is None:
        return list(base)
    return [*base, DEFAULT_REGROWTH_INJECTION_PATH]


def fixture_provenance(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Repo-relative path + last-touching commit for each fixture.

    Git history IS the blind-authoring audit trail: the arm decomposition and
    query set were committed BEFORE any metric function existed in the tree,
    and these SHAs are what lets a reader verify that rather than take it on
    trust.

    Paths are reported repo-relative (``fused-memory/tests/fixtures/…``, never
    ``/home/<user>/src/dark-factory/…``) — an artifact naming somebody's
    absolute home-directory checkout is neither reproducible nor readable by
    anyone else, and it leaks the worktree the run happened in.

    ``commit`` is ``None`` for a path git does not track.  Reporting None
    rather than falling back to HEAD is the point: a wrong SHA in an audit
    trail is worse than an absent one, because it looks checkable.
    """
    import subprocess  # noqa: PLC0415

    repo_root = _PACKAGE_ROOT.parent
    provenance: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).resolve()
        relative = _repo_relative(path)
        commit: str | None = None
        try:
            completed = subprocess.run(
                ['git', 'log', '-1', '--format=%H', '--', str(path)],
                cwd=str(repo_root), capture_output=True, text=True, timeout=30,
                check=False,
            )
            commit = completed.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            # No git, or no repo: the report says so rather than inventing a
            # SHA. Provenance that cannot be verified must not look verified.
            commit = None
        provenance.append({'path': relative, 'commit': commit})
    return provenance


def _check_arms(arms: dict[str, Any]) -> None:
    """Every variant present, every metric present, or raise."""
    missing = [arm for arm in ARM_VARIANTS if arm not in arms]
    unknown = [arm for arm in arms if arm not in ARM_VARIANTS]
    if missing or unknown:
        # Reported TOGETHER, never missing-first: a misspelled arm name
        # produces both at once, and the unknown name is the actionable half
        # — it is the difference between "the run broke" and "you typed
        # c_peers_pin". An error that named only the absence would send a
        # reader looking for a failed arm that ran perfectly well.
        problems = []
        if missing:
            problems.append(f'no measurement for {missing}')
        if unknown:
            problems.append(f'unknown arm(s) {unknown}')
        raise IncompleteReportError(
            f'{"; ".join(problems)}. Expected exactly {list(ARM_VARIANTS)}. '
            f'Refusing to emit a decision table with a missing row: a reader '
            f'cannot tell an absent arm from one that scored the same as its '
            f'neighbour.'
        )

    for arm in ARM_VARIANTS:
        measurement = arms[arm]
        for metric, required_keys in _REQUIRED_ARM_METRICS.items():
            if metric not in measurement:
                raise IncompleteReportError(
                    f"arm '{arm}' is missing metric '{metric}'"
                )
            for key in required_keys:
                if key not in measurement[metric]:
                    raise IncompleteReportError(
                        f"arm '{arm}' metric '{metric}' is missing '{key}'"
                    )

        # The same rule, INSIDE each subset.  The loop above proves every kind
        # name is present; a subset that is present but hollow renders a
        # by-kind row out of `KeyError` rather than out of this error class,
        # and the whole completeness discipline here is that a missing
        # measurement is named, never discovered by a traceback.
        by_kind = measurement['by_query_kind']
        for kind in (*QUERY_KINDS, HELD_OUT_SUBSET):
            subset = by_kind[kind]
            for key in _REQUIRED_SUBSET_KEYS:
                if key not in subset:
                    raise IncompleteReportError(
                        f"arm '{arm}' by_query_kind '{kind}' is missing "
                        f"'{key}'"
                    )
            for metric in _REQUIRED_SUBSET_METRICS:
                if metric not in subset:
                    raise IncompleteReportError(
                        f"arm '{arm}' by_query_kind '{kind}' is missing "
                        f"metric '{metric}'"
                    )
                for key in _REQUIRED_ARM_METRICS[metric]:
                    if key not in subset[metric]:
                        raise IncompleteReportError(
                            f"arm '{arm}' by_query_kind '{kind}' metric "
                            f"'{metric}' is missing '{key}'"
                        )


def _check_regrowth(regrowth: dict[str, Any]) -> None:
    """Every mode, arm and metric present, or raise — modelled on `_check_arms`.

    Missing AND unknown are collected TOGETHER at each level, for the same
    reason ``_check_arms`` does it: a misspelled name produces both at once,
    and the unknown half is the actionable one.  Every message names the
    mode, the arm and the key.

    A ``None`` VALUE never raises here.  "Measured, no denominator" is a
    legitimate result in this pipeline and the renderer prints it as
    ``_NO_MEASUREMENT``; only an ABSENT key means the run broke.
    """
    expected_metrics = set(_regrowth_metric_keys())

    missing_descriptors = [
        key for key in _REQUIRED_REGROWTH_DESCRIPTORS if key not in regrowth
    ]
    if missing_descriptors:
        raise IncompleteReportError(
            f'regrowth block is missing descriptor(s) {missing_descriptors}. '
            f'A block whose shape, fixture and injected-topic count are not '
            f'in it cannot be re-read later.'
        )

    def _check_arm_table(table: dict[str, Any], *, where: str) -> None:
        missing = [arm for arm in REGROWTH_READ_ARMS if arm not in table]
        unknown = [arm for arm in table if arm not in REGROWTH_READ_ARMS]
        if missing or unknown:
            problems = []
            if missing:
                problems.append(f'no measurement for {missing}')
            if unknown:
                problems.append(f'unknown read arm(s) {unknown}')
            raise IncompleteReportError(
                f'regrowth {where}: {"; ".join(problems)}. Expected exactly '
                f'{list(REGROWTH_READ_ARMS)}.'
            )
        for arm in REGROWTH_READ_ARMS:
            absent = sorted(expected_metrics - set(table[arm]))
            if absent:
                raise IncompleteReportError(
                    f"regrowth {where}: read arm '{arm}' is missing metric(s) "
                    f"{absent}. A metric absent from a delta table is a metric "
                    f"absent from the decision."
                )

    for key in _REQUIRED_REGROWTH_KEYS:
        if key not in regrowth:
            raise IncompleteReportError(f'regrowth block is missing {key!r}')
        _check_arm_table(regrowth[key], where=key)

    for key in _REQUIRED_REGROWTH_MODE_KEYS:
        if key not in regrowth:
            raise IncompleteReportError(f'regrowth block is missing {key!r}')
        table = regrowth[key]
        missing_modes = [mode for mode in REGROWTH_MODES if mode not in table]
        unknown_modes = [mode for mode in table if mode not in REGROWTH_MODES]
        if missing_modes or unknown_modes:
            problems = []
            if missing_modes:
                problems.append(f'no measurement for mode(s) {missing_modes}')
            if unknown_modes:
                problems.append(f'unknown mode(s) {unknown_modes}')
            raise IncompleteReportError(
                f'regrowth {key}: {"; ".join(problems)}. Expected exactly '
                f'{list(REGROWTH_MODES)}.'
            )
        for mode in REGROWTH_MODES:
            _check_arm_table(table[mode], where=f'{key}[{mode!r}]')


def build_report(
    *,
    arms: dict[str, Any],
    audit_recall: dict[str, Any],
    protocol: dict[str, Any],
    regrowth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the E2 decision table + D10 measurement into one artifact.

    Raises :class:`IncompleteReportError` rather than emitting a partial
    table.  A ``None`` VALUE is fine and survives to the artifact — "measured,
    no denominator" is a legitimate result — but an ABSENT key means the run
    broke, and a broken run must not be published as a decision.

    Arms are ordered by :data:`ARM_VARIANTS`, not by the caller's dict order,
    so two runs produce diffable artifacts.

    ``regrowth`` is OPTIONAL — smoke runs and any ``--no-regrowth`` run
    legitimately produce no block — but the key is ALWAYS emitted, as an
    explicit ``None`` when the probe did not run.  Same convention as
    ``protocol['replayed_from']``, and for the same reason: an absent key
    would make "this build predates the probe" and "the probe was skipped"
    the same reading.  The hard requirement that a COMMITTED artifact carry
    a real block lives in the committed-artifact test, where the artifact
    actually is, in the merge lane, on every commit.
    """
    _check_arms(arms)
    if regrowth is not None:
        _check_regrowth(regrowth)
    missing_protocol = [k for k in _REQUIRED_PROTOCOL_KEYS if k not in protocol]
    if missing_protocol:
        raise IncompleteReportError(
            f'protocol block is missing {missing_protocol}. The artifact has '
            f'to say how it was produced or it cannot be re-read later.'
        )

    return {
        'schema_version': REPORT_SCHEMA_VERSION,
        'protocol': dict(protocol),
        'arms': {arm: arms[arm] for arm in ARM_VARIANTS},
        'audit_recall': audit_recall,
        # LAST, and always present: see the docstring.
        'regrowth': regrowth,
    }


def _cell(value: Any, *, precision: int = 2) -> str:
    """Format one measurement, keeping "no measurement" distinguishable."""
    if value is None:
        return _NO_MEASUREMENT
    if isinstance(value, float):
        return f'{value:.{precision}f}'
    return str(value)


def _rank_cell(discoverability: dict[str, Any]) -> str:
    """Median canonical rank, WITH the censored denominator it is taken over.

    The median is over successes only: a query whose canonical never surfaced
    contributes no rank at all.  Printing the bare median therefore lets an
    arm that almost never finds the canonical show the best rank in the
    table, because it was scored on the handful of queries where it did.  The
    ``n=found/candidates`` suffix is what makes two medians comparable.
    """
    median = _cell(discoverability.get('median_canonical_rank'))
    found = discoverability.get('canonical_found_count')
    candidates = discoverability.get('canonical_candidates')
    if found is None or candidates is None:
        return median
    return f'{median} (n={found}/{candidates})'


def _guard_cell(guard: dict[str, Any]) -> str:
    """The threshold replay, WITH the covered denominator it is ceilinged by.

    Same disclosure as :func:`_rank_cell`, for the same reason.  Production's
    guard runs only on ``procedural_knowledge`` writes and
    ``find_near_duplicate_memory`` filters candidates to that category
    unconditionally, so a probe of any other category can never contribute a
    match under ANY shape.  The rate is over ALL probes, so its ceiling is
    ``covered/probes`` — identical across every arm, and invisible without
    this suffix.  Today no candidate clears the threshold at all, so every
    arm reads 0.00 and the ceiling is moot; on a rerun where one does, an
    undisclosed 20% haircut on every arm's guard column is exactly the kind
    of silent censoring the rank column's denominator exists to prevent.
    """
    rate = _cell(guard.get('guard_matched_rate'))
    covered = guard.get('guard_covered_probes')
    probes = guard.get('probes')
    if covered is None or probes is None:
        return rate
    return f'{rate} (n={covered}/{probes})'


def pin_bullet_prefix(shape: str) -> str:
    """The stable, machine-findable anchor for a shape's pin bullet.

    The markdown carries several bullet lists (D10's paraphrase exemplars,
    among others), so a test that located this one by English wording — "a
    line starting `- ` that mentions the shape AND the word 'pin'" — would
    break on a REWORDING rather than on a wrong number, which is the opposite
    of what it is checking.  Renderer and test both go through here, so the
    bullet's prose is genuinely free to change.
    """
    return f'- **`{shape}`** —'


#: Rendered for a pin rate that is real but rounds to zero at 2 decimals.
#:
#: `0.00` is the load-bearing value in this column: the artifact's own
#: reading guide reads it as "the pin NEVER FIRED", which is one of the two
#: findings the diagnostic exists to separate.  So a rate that fired — 2 of
#: 487 windows, 0.0041 — must not be allowed to print as that value.  A
#: distinct token is used rather than more decimals because the reader's
#: question is categorical ("did it fire at all?"), and `0.00411` invites
#: reading a magnitude off a number whose only content is "yes, barely".
_PIN_RATE_UNDERFLOW = '<0.01'

#: Below this, a rate rounds to `0.00` at the table's 2-decimal precision.
_PIN_RATE_ROUNDS_TO_ZERO = 0.005


def _pin_cell(rate: Any) -> str:
    """The pin diagnostic, where a rounded-down nonzero would be a lie.

    Three outcomes, all distinguishable:

      * ``—``      — the pin was off; the question was never asked.
      * ``0.00``   — asked, and the pin changed NOTHING.  Exactly zero.
      * ``<0.01``  — asked, and it fired, on too few windows to round up.

    Reusing :func:`_cell` here collapsed the last two, which made the
    artifact assert the opposite of what was measured in the one column
    built to tell them apart.
    """
    if rate is None:
        return _NO_MEASUREMENT
    if isinstance(rate, (int, float)) and 0 < rate < _PIN_RATE_ROUNDS_TO_ZERO:
        return _PIN_RATE_UNDERFLOW
    return _cell(rate)


def stored_gap_bullet_prefix(arm: str) -> str:
    """The machine-findable anchor for one arm's stored-vs-credited bullet.

    Same contract as :func:`pin_bullet_prefix` — renderer and test both go
    through here, so the bullet's prose stays free to change and only the
    numbers are pinned.  Deliberately NOT the same string: that anchor is
    ``- **`{shape}`** —``, and a per-ARM bullet reusing it would give the pin
    test two bullets where it asserts exactly one.
    """
    return f'- `{arm}` stored vs credited:'


def regrowth_bullet_prefix(arm: str) -> str:
    """The machine-findable anchor for one read arm's regrowth bullet.

    Same contract as :func:`pin_bullet_prefix` and
    :func:`stored_gap_bullet_prefix` — renderer and test both go through
    here, so the bullet's prose stays free to change and only the numbers
    are pinned.  Deliberately distinct from BOTH: each of those asserts
    "exactly one" bullet over its own anchor, so a colliding prefix would
    break a test in a distant section rather than in the one that owns it.
    """
    return f'- `{arm}` regrowth:'


def _regrowth_cell(baseline: Any, after: Any, delta: Any) -> str:
    """``base → after (Δ)``, in the one cell the reader is looking at.

    Three numbers in a cell rather than three columns per metric: at seven
    metrics the split form is a 23-column markdown table, which is not a
    table anybody reads, and the delta is only interpretable beside the
    level it moved from.

    Nothing here is hand-formatted.  The levels go through :func:`_cell` and
    the delta through :func:`_gap_cell`, so ``—`` (never measured) stays
    distinguishable from ``0.00`` (measured, did not move) and from
    ``<0.01`` (moved, too little to round up) exactly as they are in the arm
    tables above.
    """
    return f'{_cell(baseline)} → {_cell(after)} ({_gap_cell(delta)})'


def _gap_cell(gap: float) -> str:
    """A stored-vs-credited gap, where a rounded-down nonzero would be a lie.

    Same rule as :func:`_pin_cell`, for the same reason: this block sorts arms
    by whether the two columns AGREE, so a real gap must never print as
    ``0.00`` — the value the surrounding sentence reads as "identical".

    Shared with the regrowth delta tables, which are the same kind of
    quantity: a difference whose whole content is "did this move at all",
    where a rounded-down nonzero reads as "the injection changed nothing".
    """
    if gap and abs(gap) < _PIN_RATE_ROUNDS_TO_ZERO:
        return _PIN_RATE_UNDERFLOW if gap > 0 else f'-{_PIN_RATE_UNDERFLOW}'
    return _cell(gap)


def _stored_gap_lines(report: dict[str, Any]) -> list[str]:
    """The stored-vs-credited comparison, DERIVED from the arms it describes.

    Every run-specific finding in this renderer is computed from ``report``
    (see the pin bullets below), and this one is no exception on purpose.
    This file is a rerunnable generator: a hand-typed sentence about a
    previous run's numbers silently becomes a false statement sitting three
    lines above the table that contradicts it, in the one artifact gate η
    reads.  The numbers here move when the table's do, or not at all.

    The causal claim is kept general for the same reason.  Agreement is NOT
    evidence that a shape "runs no grouping transform": ``apply_topic_anchor``
    also injects the canonical and so also diverges the two columns wherever
    the pin fires.  What agreement licenses is the narrower statement that no
    read-side transform changed what the credited column counted on that arm.
    """
    lines: list[str] = [
        '',
        'As measured in THIS run, per arm (derived from the table above, not '
        'asserted — no threshold is set on either column):',
        '',
    ]
    bullets: list[str] = []
    for arm in ARM_VARIANTS:
        discoverability = report['arms'][arm]['discoverability']
        credited = discoverability['canonical_in_top_5_rate']
        stored = discoverability['stored_canonical_in_top_5_rate']
        anchor = stored_gap_bullet_prefix(arm)
        if credited is None or stored is None:
            bullets.append(
                f'{anchor} {_cell(stored)} vs {_cell(credited)} — not '
                f'comparable; one of the two columns was never measured.'
            )
            continue
        if credited == stored:
            bullets.append(
                f'{anchor} {_cell(stored)} vs {_cell(credited)} — identical, '
                f'so no read-side transform changed what the credited column '
                f'counted on this arm.'
            )
            continue
        bullets.append(
            f'{anchor} {_cell(stored)} vs {_cell(credited)} — a gap of '
            f'{_gap_cell(credited - stored)}, which is what the read-side '
            f'transforms added on top of what retrieval put in front of the '
            f'reader.'
        )
    lines += bullets
    lines += [
        '',
        'Where the two agree, `canonical in top-5` is reporting retrieval '
        'alone on that arm.  Where they diverge, the gap is a read '
        'transform\'s contribution rather than a retrieval difference, and '
        '`pin changed window` is what says WHICH transform: a row whose pin '
        'never fired can only have been moved by grouping.  Whether the gap '
        'is worth crediting is gate η\'s call, and it is a different call '
        'from "this shape retrieves the canonical better".  Read the same two '
        'columns on the `held_out` rows of the by-kind table, which are the '
        'only rows measuring generalisation.',
    ]
    return lines


def _cost_phrase(delta: Any, *, cost_sign: int) -> str:
    """One delta, as a COST or a GAIN — the verb derived, never typed.

    ``cost_sign`` comes from :data:`_REGROWTH_COST_SIGN`, so the signed cost
    is ``delta * cost_sign``: positive means the injection took something,
    negative means it gave something back, zero means it did not move the
    column at all.

    This exists because the signal sentence hard-typed the word "costs"
    around :func:`_gap_cell`.  The committed artifact therefore read "one
    unstamped re-emission per topic costs <0.01 claim recall@5" for a
    MEASURED GAIN of +0.0042 — a false statement in the probe's headline
    sentence, produced by a verb that could not see the sign of the number
    it was attached to.

    :func:`_gap_cell` stays the number formatter: this changes the verb, not
    the formatting, so ``<0.01`` (moved, too little to round up) still never
    prints as ``0.00`` (measured, did not move).  The MAGNITUDE is rendered,
    because the verb already carries the direction and "gains -0.10" is not
    a sentence.
    """
    if delta is None:
        return f'was never measured on this column ({_NO_MEASUREMENT})'
    cost = delta * cost_sign
    if cost == 0:
        return f'does not move it ({_gap_cell(delta)})'
    if cost > 0:
        return f'costs {_gap_cell(abs(delta))}'
    return f'gains {_gap_cell(abs(delta))}'


def _absorption_phrase(flat: Any, selected: Any, *, cost_sign: int) -> str:
    """Does 4004's selected transform absorb the flat read's regrowth cost?

    DERIVED and deliberately COMPARATIVE: it reports how two measured deltas
    stand relative to each other and asserts no threshold on either (gate
    G6).  The alternative — a typed sentence about what the transform does —
    is a claim that goes stale the first time this generator is rerun, three
    lines above the table that then contradicts it.

    Branches on SIGNED COST (``delta * cost_sign``), NOT on ``abs()``.  The
    ``abs()`` form was the defect this replaced: distance-from-baseline
    cannot tell a gain from a loss, so a measured GAIN of 0.0042 and a
    measured LOSS of 0.0042 came out as "the same distance from baseline" —
    which is what the committed artifact published for the flat read
    (+0.0042) against ``promoting_pin`` (-0.0042) on claim recall@5.  Two
    opposite findings, declared equivalent, in the one sentence a gate
    reads.  Deriving the sentence (see this function's and
    :func:`_regrowth_lines`' rationale) is what keeps it from going stale;
    it is necessary and it was not sufficient — the derivation also has to
    be sign-correct.

    Branches on the STRICT sign, not on ``<= 0``, for the same reason: "the
    two arms moved in opposite directions" is reserved for
    ``cost_flat < 0 < cost_selected``, and a flat read that did not move at
    all (``cost_flat == 0``) gets its own sentence.  Folding the two together
    printed "moved the two arms in opposite directions — the flat read does
    not move it (0.00) …", a sentence contradicting the cell beside it.

    Not symmetric in its two arguments, on purpose: the flat read and the
    selected transform play different roles in the sentence, so a call site
    that transposed them could not be silent.
    """
    if flat is None or selected is None:
        return ('is not comparable on this column — one of the two arms was '
                'never measured')
    cost_flat = flat * cost_sign
    cost_selected = selected * cost_sign
    if cost_flat == 0 and cost_selected == 0:
        return 'had nothing to absorb: neither arm moved on this column'
    if cost_flat <= 0 and cost_selected <= 0:
        # Both moved AGAINST the cost direction (or one did not move).  Name
        # both values: a reader told "nothing happened" when both arms GAINED
        # cannot reconstruct the table from the sentence.
        return (f'had nothing to absorb — neither arm paid a cost (the flat '
                f'read moved {_gap_cell(flat)}, the selected transform '
                f'{_gap_cell(selected)})')
    if cost_flat == 0:  # and cost_selected > 0
        # SPLIT from the branch below on purpose.  One arm standing still is
        # not "two arms moving in opposite directions": that sentence, read
        # beside a `0.00` flat cell, contradicts the number it is describing.
        return (f'had no cost to absorb — the flat read did not move on this '
                f'column while the selected transform '
                f'{_cost_phrase(selected, cost_sign=cost_sign)} '
                f'({_gap_cell(flat)} against {_gap_cell(selected)})')
    if cost_flat < 0:  # and cost_selected > 0
        return (f'moved the two arms in opposite directions — the flat read '
                f'{_cost_phrase(flat, cost_sign=cost_sign)} while the '
                f'selected transform '
                f'{_cost_phrase(selected, cost_sign=cost_sign)} '
                f'({_gap_cell(flat)} against {_gap_cell(selected)}), so there '
                f'was no flat-read cost for it to absorb')
    if cost_selected == 0:
        return (f'absorbs all of it — the selected transform did not move '
                f'where the flat read '
                f'{_cost_phrase(flat, cost_sign=cost_sign)} '
                f'({_gap_cell(selected)} against {_gap_cell(flat)})')
    if cost_selected < 0:
        return (f'more than absorbs it, with a sign flip — the flat read '
                f'{_cost_phrase(flat, cost_sign=cost_sign)} while the '
                f'selected transform '
                f'{_cost_phrase(selected, cost_sign=cost_sign)} '
                f'({_gap_cell(flat)} against {_gap_cell(selected)})')
    if cost_selected < cost_flat:
        return (f'absorbs part of it ({_gap_cell(selected)} against the flat '
                f"read's {_gap_cell(flat)})")
    if cost_selected == cost_flat:
        return (f'does not change it ({_gap_cell(selected)}, the same cost '
                f'the flat read paid)')
    return (f'does not absorb it — it costs more than the flat read did '
            f'({_gap_cell(selected)} against {_gap_cell(flat)})')


def _reseeding_control_phrase(flat_stamping: dict[str, Any]) -> str:
    """What the `flat` stamping row came out as — the re-seeding noise floor.

    DERIVED, for the same reason :func:`_absorption_phrase` is: a typed
    "re-seeding contributes nothing" is a claim about a previous run, and the
    first re-measurement that moves it turns the sentence into a falsehood
    sitting beside the table that contradicts it.

    Why that row IS a control.  ``stamping_value`` cancels the baseline
    (``(after_stamped - base) - (after_unstamped - base)``), so the `flat`
    row compares two SEPARATELY SEEDED injected collections directly.  Their
    records carry byte-identical text and differ only in a ``topic`` metadata
    key, and a flat read consults no metadata — so nothing but the seeding
    can move it.  That is the same structural forcing
    ``REGROWTH_STAMPING_CEILING_DISCLOSURE`` warns makes the row useless as
    evidence ABOUT STAMPING; it is exactly what makes it useful here.

    Its reach is bounded and the sentence says so: it measures re-seeding
    between two INJECTED collections at this corpus size.  It is evidence
    about the noise floor, not a proof about the baseline collection.

    PER COLUMN, never collapsed to one scalar.  The seven metrics are in
    four different units — 0..1 rates, a raw count in the hundreds, a rank,
    and a mean token count in the thousands — so a max-by-magnitude across
    the row is a category error: it would bound a `claim recall@5` delta of
    `-0.01` by a `tokens/query` cell of `-3.13` and read as "that delta is
    entirely noise".  A cell bounds re-seeding in ITS OWN column and in no
    other, and the sentence this returns says exactly that, naming the
    columns that moved with the labels the table below uses.
    """
    metrics = _regrowth_metric_keys()
    labels = dict(zip(metrics, REGROWTH_METRIC_LABELS, strict=True))
    measured = {
        metric: flat_stamping[metric] for metric in metrics
        if flat_stamping.get(metric) is not None
    }
    if not measured:
        return 'was not measured in this run'
    moved = {
        metric: value for metric, value in measured.items() if value
    }
    if not moved:
        return ('came out as exactly `0.00` on every column, so at this '
                'corpus size a second seeding moved nothing any of these '
                'metrics can see')
    cells = ', '.join(
        f'`{labels[metric]}` {_gap_cell(value)}'
        for metric, value in moved.items()
    )
    return (f'did NOT come out flat — it moved on {len(moved)} of '
            f'{len(measured)} columns ({cells}), and those cells ARE the '
            f'noise floor, PER COLUMN.  Each bounds re-seeding in its own '
            f'column only: these metrics are in different units (rates, a '
            f'count, a rank, token counts), so no cell says anything about '
            f'how much of a delta in a DIFFERENT column is re-seeding, and '
            f'a column whose `flat` cell is `0.00` has no measured '
            f're-seeding contribution however large another column\'s cell '
            f'is')


def _regrowth_lines(report: dict[str, Any]) -> list[str]:
    """The +1-re-emission probe's section: what was injected, and what moved.

    Emitted on EVERY run, probed or not.  An absent section is exactly how
    this probe went missing the first time: esc-3200-3 asked for regrowth
    deltas and read an artifact that simply had no such section, which is
    indistinguishable from a probe that ran and found nothing.  So the
    heading is unconditional and the skipped case says so in words.

    Every number is DERIVED from ``report['regrowth']``, like every other
    run-specific finding in this renderer (see :func:`_stored_gap_lines`).
    """
    lines: list[str] = ['', '## Regrowth deltas', '']
    regrowth = report['regrowth']
    if regrowth is None:
        lines.append(
            '**Not probed in this run.**  The +1-re-emission probe was '
            'skipped (`--no-regrowth`), so this artifact carries no regrowth '
            'measurement at all.  The heading is emitted anyway: an ABSENT '
            'section reads identically to a probe that ran and moved '
            'nothing, and telling those two apart is the whole reason this '
            'section exists.'
        )
        return lines

    metrics = _regrowth_metric_keys()
    shape = regrowth['shape']
    baseline, after, deltas = (
        regrowth['baseline'], regrowth['after'], regrowth['deltas'],
    )
    arms = ', '.join(f'`{arm}`' for arm in regrowth['read_arms'])
    lines += [
        f"One near-duplicate **re-emission** was injected per topic "
        f"({regrowth['topics_injected']} topics, "
        f"{regrowth['injections_per_topic']} injection each, from "
        f"`{regrowth['injection_fixture']}`) into the RATIFIED write shape "
        f"`{shape}`, and only the READ arm was varied: {arms}.  Each "
        f"injected record restates that topic's canonical claim in different "
        f"words — the organic pattern esc-3200-3 documents — and is scored "
        f"as realizing the claim it re-emits, so claim recall can move in "
        f"either direction rather than only fall.",
        '',
        'Two injection modes, because the write path\'s best case and the '
        'case actually observed are different.  **`unstamped`** carries no '
        '`topic` key at all, which is how every reify warm-lane re-emission '
        'arrived and is therefore the mode that models reality today; '
        '**`stamped`** carries the topic a stamping write path would have '
        'set.  The gap between their deltas is the second table.',
        '',
        'Baseline is the SAME read arm over the un-injected corpus — the '
        'rankings the decision table above is built from.  Two comparisons '
        'live in this table and they are NOT equally strong.  An '
        'arm-vs-arm comparison (three rows of one mode) carries no '
        'cross-seeding noise at all: all three arms are scored from ONE set '
        'of rankings, varying only the read transform.  A '
        'baseline-vs-after delta is weaker — its two sides are two '
        'separately seeded collections (the ratified arm\'s, and the '
        'injected pass\'s own), so it carries whatever a re-seeding '
        'contributes on top of the re-emission.',
        '',
        'The control for that is in this artifact rather than argued for. '
        'The `flat` row of the stamping table below compares two separately '
        'seeded INJECTED collections whose records carry byte-identical text '
        'and differ only in a `topic` metadata key a flat read never '
        'consults — so nothing but the seeding can move it, and it reads as '
        'the re-seeding noise floor at this corpus size.  In this run it '
        f'{_reseeding_control_phrase(regrowth["stamping_value"]["flat"])}. '
        '**No threshold is asserted on any delta here** (gate G6): the probe '
        'informs gate η and task 4006\'s stamping campaign, it does not gate '
        'a build.',
        '',
        '| ' + ' | '.join(REGROWTH_TABLE_COLUMNS) + ' |',
        '| ' + ' | '.join('---' for _ in REGROWTH_TABLE_COLUMNS) + ' |',
    ]

    for mode in REGROWTH_MODES:
        for arm in REGROWTH_READ_ARMS:
            lines.append('| ' + ' | '.join([
                mode,
                arm,
                *(
                    _regrowth_cell(
                        baseline[arm][metric],
                        after[mode][arm][metric],
                        deltas[mode][arm][metric],
                    )
                    for metric in metrics
                ),
            ]) + ' |')

    recall, stored = 'claim_recall.at_5', (
        'discoverability.stored_canonical_in_top_5_rate'
    )
    selected = REGROWTH_READ_ARMS[-1]
    lines += [
        '',
        'As measured in THIS run, per read arm (derived from the table '
        'above, not asserted):',
        '',
    ]
    for arm in REGROWTH_READ_ARMS:
        lines.append(
            f'{regrowth_bullet_prefix(arm)} one re-emission per topic moves '
            f'claim recall@5 by {_gap_cell(deltas["unstamped"][arm][recall])} '
            f'unstamped / {_gap_cell(deltas["stamped"][arm][recall])} '
            f'stamped, and stored canonical-in-top-5 by '
            f'{_gap_cell(deltas["unstamped"][arm][stored])} unstamped / '
            f'{_gap_cell(deltas["stamped"][arm][stored])} stamped.  Stamping '
            f'every re-emission is worth '
            f'{_gap_cell(regrowth["stamping_value"][arm][recall])} on claim '
            f'recall@5 and '
            f'{_gap_cell(regrowth["stamping_value"][arm][stored])} on stored '
            f'canonical-in-top-5 under this arm.'
        )

    # The verb and the absorption verdict are both DIRECTIONAL claims, so
    # both go through the metric's declared cost direction.  Hard-typing
    # either one is what made the committed artifact call a measured GAIN a
    # cost, and call a gain and a loss of the same magnitude equivalent.
    recall_sign = _REGROWTH_COST_SIGN[recall]
    stored_sign = _REGROWTH_COST_SIGN[stored]
    lines += [
        '',
        f'**The signal sentence.**  Under the ratified `{shape}` write '
        f'shape, one unstamped re-emission per topic '
        f'{_cost_phrase(deltas["unstamped"]["flat"][recall], cost_sign=recall_sign)} '
        f'claim recall@5 and '
        f'{_cost_phrase(deltas["unstamped"]["flat"][stored], cost_sign=stored_sign)} '
        f'stored canonical-in-top-5 on a flat read.  4004\'s selected '
        f'transform (`{selected}`) '
        f'{_absorption_phrase(deltas["unstamped"]["flat"][recall], deltas["unstamped"][selected][recall], cost_sign=recall_sign)} '
        f'on claim recall@5, and '
        f'{_absorption_phrase(deltas["unstamped"]["flat"][stored], deltas["unstamped"][selected][stored], cost_sign=stored_sign)} '
        f'on stored canonical-in-top-5.  Stamped, the same injection '
        f'{_cost_phrase(deltas["stamped"]["flat"][recall], cost_sign=recall_sign)} '
        f'and '
        f'{_cost_phrase(deltas["stamped"]["flat"][stored], cost_sign=stored_sign)} '
        f'on a flat read.',
        '',
        # A module constant, spliced BETWEEN the delta table and the
        # stamping heading, for the same reason
        # `REGROWTH_STAMPING_CEILING_DISCLOSURE` is: it qualifies the two
        # `canonical in top-5` columns of the table above it, a test pins
        # its relative index, and a substring pin on its prose would have
        # passed for a paragraph gutted to the words it pinned — or, worse,
        # for one deleted entirely, since every word it used to be checked
        # for is also supplied by the table's own header row.
        REGROWTH_CREDITED_SEMANTICS_DISCLOSURE,
        '',
        '### What topic-stamping buys',
        '',
        '`stamped delta - unstamped delta`, per read arm.  This is the '
        'number task 4006\'s stamping campaign is owed: the unstamped delta '
        'is what one re-emission costs today, the stamped delta is what it '
        'would cost if the write path stamped every re-emission, and the '
        'difference is what the campaign buys against regrowth ON THE ARMS '
        'THAT CAN EXPRESS ONE — which is not all three of them.  Read the '
        'paragraph below before reading the table.  Sign is the '
        'underlying column\'s: on recall and discoverability a positive '
        'value is cost recovered, on `tokens/query` it is not.',
        '',
        # Spliced BETWEEN the heading and the table on purpose: this
        # qualifies the numbers in the rows immediately below it, and a
        # qualifier a section away is not a qualifier on this number.  A
        # test pins the relative index.
        REGROWTH_STAMPING_CEILING_DISCLOSURE,
        '',
        '| ' + ' | '.join(REGROWTH_STAMPING_COLUMNS) + ' |',
        '| ' + ' | '.join('---' for _ in REGROWTH_STAMPING_COLUMNS) + ' |',
    ]
    for arm in REGROWTH_READ_ARMS:
        lines.append('| ' + ' | '.join([
            arm,
            *(
                _gap_cell(regrowth['stamping_value'][arm][metric])
                for metric in metrics
            ),
        ]) + ' |')

    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """The operator-facing artifact: prose, decision table, D10, provenance.

    Byte-deterministic for identical input — every float goes through
    :func:`_cell` at a fixed precision and every collection is emitted in a
    pinned order — because the whole value of rerunning this experiment is
    that the diff between two artifacts is signal rather than formatting
    noise.
    """
    protocol = report['protocol']
    audit = report['audit_recall']
    true_dup = audit['true_dup']

    lines: list[str] = [
        '# E2 storage-shape bake-off',
        '',
        'Arbitration experiment for `docs/prds/memory-metadata-vocabulary.md` '
        'D9/D10 (task 3199, PRD leaf ζ), implementing '
        '`plans/memory-subsystem-eval-design.md` §5 E2.  This artifact is the '
        'signal gate leaf **η** reads: the choice between δ-as-default and '
        'peers-as-default is made from the table below.',
        '',
        '## How to read this table',
        '',
        'Every metric here is **rank/set-based** — ranks and set membership '
        '(present-in-top-k), never absolute cosine — per eval-design §1: '
        'wording and embedding-config drift move the score scale wholesale, '
        'so thresholds on raw cosine do not survive a re-measurement while '
        'ranks do.',
        '',
        'The ONE exception is the last column.  `guard matched (replay)` is a '
        '**threshold replay**: the production near-duplicate selector *is* an '
        'absolute-threshold selector, so replaying it is the only way to '
        'answer "would the guard have fired?".  Do not trend that column '
        'across an embedder or config change — trend '
        '`guard candidate present`, which is rank-based and asks the '
        'discriminating question (was a true cluster sibling in front of the '
        'guard at all?).',
        '',
        'A `—` cell is **no measurement**, not a measured zero.',
        '',
        '`guard matched (replay)` carries its denominator as `(n=covered/'
        'probes)`.  Production\'s guard runs only on `procedural_knowledge` '
        'writes, and `find_near_duplicate_memory` filters its candidates to '
        'that category unconditionally, so a probe of any other category '
        'cannot produce a match under ANY shape — a ceiling identical across '
        'all six arms and unrelated to storage shape.  Without the suffix '
        'that haircut is invisible in the one column a reader would otherwise '
        'read as a shape difference.',
        '',
        '`median canonical rank` carries its denominator as `(n=found/'
        'candidates)`.  The median is over the queries where the canonical '
        'surfaced AT ALL, so without that suffix an arm that almost never '
        'finds the canonical prints the best rank in the table — scored on '
        'the handful of queries where it did.  Rank is measured over the '
        'full fetch depth, not the k=5 read window, so "outside top-5" and '
        '"absent entirely" stay distinguishable.',
        '',
        '`canonical in top-5` is measured over the READ window, and under '
        '`b_grouped` that is not the same question it is under the other two '
        'shapes.  `apply_grouped_read` synthesises its grouped document '
        'carrying the CANONICAL\'s `record_id`, and the metric identifies the '
        'canonical by `record_id` — so any child hit that folds upward is '
        'scored as "the canonical was found", whereas under `c_peers` and '
        '`status_quo` the canonical\'s own stored record must itself have '
        'ranked.  Grouping is not the only transform that credits it: '
        '`apply_topic_anchor` injects the canonical outright, so a `+pin` row '
        'whose pin fired can move this column too.  Either way it is a '
        'property of the READ TRANSFORM, not purely of retrieval.  It is also '
        'arguably the right thing to credit: a grouped read genuinely does '
        'put the canonical body in the reader\'s window, which is what a '
        'reader of that window cares about.',
        '',
        '`canonical in top-5 (stored)` is the transform-blind counterpart — '
        'the canonical\'s OWN stored record, measured over the raw store hits '
        'before grouping and before the pin.  It is therefore identical '
        'across a shape\'s two pin variants by construction, and comparable '
        'across all six arms.  **Read the two together: the gap between them '
        'is what the read transforms added.**  This is DISCLOSURE, not '
        'correction — no '
        'arm, pin, window or threshold was re-tuned to move either column, '
        'and both numbers are recorded exactly as measured (gate G6/D10 '
        'assert no threshold on any of them).',
    ]

    # Run-specific, so DERIVED rather than typed — see `_stored_gap_lines`.
    lines += _stored_gap_lines(report)

    lines += [
        '',
        '`pin changed window` is the diagnostic that makes the `+pin` rows '
        'readable.  Every variant is scored over a window of the SAME size '
        '(k), so an additive pin can only pay off where a read-side transform '
        'left headroom in that budget — under grouping, which collapses the '
        'window.  A `+pin` row identical to its twin at `0.00` means the pin '
        'never fired, which is a different finding from "the pin does not '
        'help".  `<0.01` is a THIRD finding and not a rounded `0.00`: the pin '
        'fired, on too few windows to round up.  `—` on a pin-off row means '
        'the question was never asked.',
        '',
        '## Decision table',
        '',
        '| ' + ' | '.join(DECISION_TABLE_COLUMNS) + ' |',
        '| ' + ' | '.join('---' for _ in DECISION_TABLE_COLUMNS) + ' |',
    ]

    for arm in ARM_VARIANTS:
        measurement = report['arms'][arm]
        guard = measurement['guard_adequacy']
        lines.append('| ' + ' | '.join([
            arm,
            _cell(measurement['claim_recall']['at_5']),
            _cell(measurement['claim_recall']['at_10']),
            _cell(measurement['discoverability']['canonical_in_top_5_rate']),
            _cell(measurement['discoverability'][
                'stored_canonical_in_top_5_rate']),
            _rank_cell(measurement['discoverability']),
            _cell(measurement['tokens_per_query']['mean'], precision=1),
            _cell(guard['candidate_present_rate']),
            _guard_cell(guard),
            _pin_cell(measurement['pin']['window_changed_rate']),
        ]) + ' |')

    lines += [
        '',
        f"Token counts come from the `{protocol['token_estimator']}` "
        f'estimator (recorded because a substituted estimator would otherwise '
        f'be indistinguishable from a measured one).  '
        f"Guard threshold: {_cell(protocol['guard_threshold'])}.  "
        f"Distractor slab: {protocol['distractor_slab_size']} records, "
        f'identical in every arm.',
        '',
        '## What the pin column shows',
        '',
        "3111's topic anchor is ADDITIVE at the search seam (PRD D1): it "
        'appends a topic\'s canonical, it never promotes it over a ranked '
        'hit.  Both variants of a shape are scored over a window of the same '
        'size, so an additive pin can only pay off where a read-side '
        'transform freed a slot in that budget.  Per shape, as measured:',
        '',
    ]

    pin_blocks = ('claim_recall', 'discoverability', 'tokens_per_query',
                  'guard_adequacy')
    for shape in ARM_SHAPES:
        off = report['arms'][shape]
        on = report['arms'][f'{shape}+pin']
        rate = on['pin']['window_changed_rate']
        moved = [block for block in pin_blocks if on[block] != off[block]]
        lines.append(
            f'{pin_bullet_prefix(shape)} the pin changed {_pin_cell(rate)} of '
            f'the measured windows; '
            + (
                f"every metric column is unchanged from `{shape}`."
                if not moved
                else f"the {', '.join(f'`{b}`' for b in moved)} column(s) moved."
            )
        )

    lines += [
        '',
        'Read a `+pin` row against `pin changed window`, not against its twin '
        'alone.  A rate of exactly `0.00` beside identical columns means the '
        'pin never fired — at a full window there is no slot for an appended '
        'record — which is a different finding from "the pin does not help".  '
        'A rate above zero (including `<0.01`) beside identical columns means '
        'it fired and moved nothing these metrics measure.  Either way, a pin '
        'that is to pay off '
        'under a shape whose window is already full would have to PROMOTE '
        'rather than append, and that is a design choice for gate η, not a '
        'tuning knob for this experiment.',
        '',
        '## By query kind',
        '',
        'eval-design §5 E2 names claim recall and canonical/topic '
        'discoverability as DISTINCT metrics, and the query set is authored '
        'in two kinds for exactly that reason.  Pooled into one mean, a shape '
        'that wins on claim queries while losing on topic phrasings is '
        'indistinguishable from one that ties on both.  `held_out` is a '
        'SUBSET of `topic_phrasing` — the phrasings the E1 registry was NOT '
        'derived from, so the only ones that measure generalisation rather '
        'than recall of the derivation input.',
        '',
        '| arm | kind | queries | claim recall@5 | claim recall@10 | '
        'canonical in top-5 | canonical in top-5 (stored) | '
        'median canonical rank |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]

    for arm in ARM_VARIANTS:
        by_kind = report['arms'][arm]['by_query_kind']
        for kind in (*QUERY_KINDS, HELD_OUT_SUBSET):
            subset = by_kind[kind]
            lines.append('| ' + ' | '.join([
                arm,
                kind,
                str(subset['queries']),
                _cell(subset['claim_recall']['at_5']),
                _cell(subset['claim_recall']['at_10']),
                _cell(subset['discoverability']['canonical_in_top_5_rate']),
                # Per subset, from the subset's own block — `held_out` is the
                # row that measures generalisation, and it is exactly the row
                # where a transform-credited rate is least safe to read alone.
                _cell(subset['discoverability'][
                    'stored_canonical_in_top_5_rate']),
                _rank_cell(subset['discoverability']),
            ]) + ' |')

    lines += _regrowth_lines(report)

    lines += [
        '',
        '## D10 — audit-recall over the labeled fixture',
        '',
        f"Replay of `{audit['detector']}` at threshold "
        f"{_cell(audit['threshold'])} over α/3130's curator-labeled fixture, "
        f'scored against `calibrate_write_triage.build_pair_sets`.  '
        f'**No threshold is asserted on this number** (gate G6): it informs '
        f'how far to trust the κ duplicate sweep, it does not gate a build.',
        '',
        '| class | pairs | recovered | rate |',
        '| --- | --- | --- | --- |',
        f"| true duplicates | {true_dup['pairs']} | {true_dup['recovered']} | "
        f"{_cell(true_dup['recall'])} |",
        f"| — lexical band (reachable) | {true_dup['lexical_band']['pairs']} | "
        f"{true_dup['lexical_band']['recovered']} | "
        f"{_cell(true_dup['lexical_band']['recall'])} |",
        f"| — paraphrase band (unreachable) | "
        f"{true_dup['paraphrase_band']['pairs']} | "
        f"{true_dup['paraphrase_band']['recovered']} | "
        f"{_cell(true_dup['paraphrase_band']['recall'])} |",
        f"| hard negatives (falsely grouped) | {audit['hard_negative']['pairs']} | "
        f"{audit['hard_negative']['falsely_grouped']} | "
        f"{_cell(audit['hard_negative']['rate'])} |",
        f"| unrelated (falsely grouped) | {audit['unrelated']['pairs']} | "
        f"{audit['unrelated']['falsely_grouped']} | "
        f"{_cell(audit['unrelated']['rate'])} |",
        '',
        'The **paraphrase band** is the positive pairs no character-level '
        'threshold can reach at any tuning short of changing kind (split by '
        'max `SequenceMatcher` ratio over both argument orders, since '
        '`ratio()` is order-sensitive).  Counting those as plain detector '
        'misses would read as "the audit script is broken" rather than "this '
        'class is invisible to it".  Nearest misses, for hand-auditing the '
        'split:',
        '',
    ]
    for exemplar in audit['paraphrase_exemplars']:
        lines.append(
            f"- `{exemplar['a']}` / `{exemplar['b']}` — max ratio "
            f"{_cell(exemplar['max_ratio'], precision=4)}"
        )

    lines += [
        '',
        '## Protocol',
        '',
        f"**Blind authoring**: {protocol['blind_authoring']}.  The commits "
        f'below are the audit trail, and the anti-laziness floor is '
        f'claim-coverage parity (every '
        f'claim id realizable in every arm), deliberately not length parity — '
        f"arm (a)'s long originals versus arm (c)'s short peers differ by "
        f'construction, and that difference IS the tokens/query column.',
        '',
        # Emitted on EVERY run, probed or not: the disclosure describes how
        # the probe was AUTHORED, which a reader of a `--no-regrowth`
        # artifact still needs when they go looking for its numbers.
        REGROWTH_BLIND_AUTHORING_DISCLOSURE,
        '',
        f"**Embedder**: {protocol['embedder_model']}.",
        '',
        '| fixture | commit |',
        '| --- | --- |',
    ]
    for fixture in protocol['fixtures']:
        lines.append(
            f"| `{fixture['path']}` | "
            f"{fixture.get('commit') or _NO_MEASUREMENT} |"
        )

    lines.append('')
    return '\n'.join(lines)


def write_artifacts(
    report: dict[str, Any],
    json_out: str | Path = DEFAULT_REPORT_JSON,
    md_out: str | Path = DEFAULT_REPORT_MD,
) -> tuple[Path, Path]:
    """Write the JSON and markdown artifacts atomically.

    Mirrors ``memory_eval_retrieval_probe.write_report_text``: a plain
    ``write_text`` truncates first, so a crash, an ENOSPC or a concurrent
    reader mid-write leaves a half-written decision table on disk that still
    parses as markdown.  :func:`tempfile.mkstemp` gives an OS-guaranteed
    fresh, exclusively-created sibling — not a pid-derived name two writers
    could collide on — and ``os.replace`` is atomic within a filesystem, so a
    reader sees either the old artifact or the complete new one.

    The mechanism is copied rather than imported because the original is a
    module-private helper of another standalone script.

    Each file is atomic, and so is the PAIR: the markdown is rendered BEFORE
    either replace, so a raise mid-render aborts with both artifacts still on
    their previous run.  Writing the JSON first would leave a new JSON beside
    a stale markdown describing a different run — and `render_markdown`
    subscripts blocks (`audit_recall['true_dup']`, `protocol['fixtures']`)
    that `_check_arms` never validates, so that is reachable, not theoretical.
    The two files are gate eta's decision input and a reader has no way to
    tell they disagree.
    """
    json_path, md_path = Path(json_out), Path(md_out)
    markdown = render_markdown(report)
    _atomic_write_text(json_path, json.dumps(report, indent=2) + '\n')
    _atomic_write_text(md_path, markdown)
    return json_path, md_path


def _atomic_write_text(path: Path, text: str) -> None:
    import contextlib  # noqa: PLC0415
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix='.tmp', prefix=f'{path.name}.', dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------------------
# The live driver — the only part that touches Qdrant or an embedder
# ---------------------------------------------------------------------------
#
# Everything above is pure and runs in the merge lane.  Below, three seeded
# ephemeral collections are created, queried once each, measured twice each
# (pin off / pin on), and dropped.
#
# ISOLATION, in the order it matters:
#   - ``collection_prefix`` is set on the CONFIG, not merely the project_id.
#     Collections are ``f'{collection_prefix}_{project_id}'``
#     (``Scope.mem0_collection_name``), so scoping only the project_id would
#     write under the default ``fused`` prefix — which the reaper does not
#     match, by design.  That is a permanent leak, not a slow one.
#   - the prefix string itself lives in ``cleanup_test_collections`` and is
#     imported from there.  A collection named in one file and reaped by
#     another is one rename away from orphaning.
#   - every collection is dropped BEFORE and AFTER, so a swallowed teardown
#     self-heals on the next run rather than poisoning it.
#   - the project_id carries the xdist worker id, so concurrent workers
#     cannot seed each other's arms.
#
# SIX VARIANTS, THREE COLLECTIONS: the topic pin is a READ-side transform, so
# each shape is fetched ONCE and measured twice.  Pin-on and pin-off therefore
# compare identical stored state and an identical ranked list — an exactly
# controlled A/B rather than two ANN draws.

#: Concurrent seeding writes in flight.  Bounded rather than one gather over
#: the whole arm: an unbounded fan-out opens hundreds of simultaneous
#: embedding requests, which the provider answers with rate-limit retries —
#: a run whose wall clock is set by the throttler instead of the experiment,
#: and (worse) a partially-seeded arm if the retries give up.
SEED_CONCURRENCY = 8

#: Fetch depth for the claim-recall / discoverability / cost queries.
DEFAULT_SEARCH_LIMIT = 10

#: Fetch depth for a guard probe: production's window plus one.
#:
#: The extra slot pays for dropping the probing write's OWN content from its
#: own window.  In the real timeline that write had not landed when the guard
#: ran, and every arm carries it in some form — arm (a) verbatim, the peer
#: arms as whatever claims were decomposed OUT of it.  Leaving it in would
#: hand an arm a free near-1.0 self-match, and the decision table would read
#: that as a better guard.  The removal is by PROVENANCE
#: (:func:`probe_own_record_ids`), not by record id: only arm (a) reuses the
#: α memory_id as its record id, so an id filter would drop the self-match
#: for the baseline alone and bias the column the other way.
#: The replay still happens at :data:`GUARD_TOP_K`, which ``guard_adequacy``
#: enforces itself.
#:
#: This is the FLOOR, not the depth: it pays for the one removal the
#: status-quo arm makes.  A decomposed arm removes one record PER CLAIM the
#: probe was split into (up to four), so the depth is computed per probe by
#: :func:`guard_fetch_limit`.  A fixed +1 would leave a decomposed arm
#: replaying over a 2-record window while the baseline replayed over 5 —
#: the same unequal-window defect the query side was fixed for, biased
#: against exactly the two arms this experiment is arbitrating in favour of.
GUARD_FETCH_LIMIT = GUARD_TOP_K + 1


def guard_fetch_limit(own_record_ids: set[str]) -> int:
    """Fetch depth that still leaves :data:`GUARD_TOP_K` after the self-drop.

    The replay window must be the SAME SIZE for every arm, or the guard
    column measures how finely an arm was decomposed rather than how well it
    guards.  So the depth grows with the number of records the probe's own
    content occupies in THIS arm.
    """
    return GUARD_TOP_K + len(own_record_ids)

#: Agent id every seeded record is written under.
SEED_AGENT_ID = 'e2-bakeoff-seed'

#: Recorded verbatim in the artifact's protocol block.
BLIND_AUTHORING_PROTOCOL = (
    'single-author-blind-to-metrics, mechanized by commit ordering: the arm '
    'decomposition and query set were committed before any metric function '
    'existed in the tree'
)

#: Recorded verbatim in the artifact's Protocol section.  The +1-re-emission
#: probe (task 4012) does NOT carry the protection the six arms do, and the
#: artifact has to SAY so rather than let the section above imply it.
REGROWTH_BLIND_AUTHORING_DISCLOSURE = (
    '**Regrowth probe — not blind-authored**: the +1-re-emission probe does '
    'not carry the blind-authoring protection the six arms above do.  That '
    'protection was mechanized by commit ordering — the arm decomposition '
    'and query set were committed before any metric function existed in the '
    'tree — and it is UNRECOVERABLE for this probe, because the metric code '
    'was already in the tree when the injection corpus was authored.  The '
    'injection fixture was committed on its own, ahead of every line of '
    'probe code, as a PARTIAL audit trail and nothing more: it fixes WHAT '
    'was injected before any delta was seen, not before the metrics were '
    'written.  Its commit is in the fixture table below.  Read the regrowth '
    'section as a disclosed non-blind measurement, not as a blind one.'
)

#: Why two of the three rows in `### What topic-stamping buys` are zero, in
#: the report's own voice, spliced BESIDE that table rather than into a
#: distant section — a qualifier placed anywhere else is not a qualifier on
#: this number.
#:
#: The hazard it guards is the one 3560 and 4004 each had to correct after
#: publication: a number that could not have been anything else, read as a
#: measurement.  Every ceiling named here is asserted mechanically by
#: ``TestTheStampingTableCeilings``, so a ceiling that stops being true
#: fails a test rather than leaving this text explaining away a number that
#: has become real.
#:
#: The rows themselves STAY (review option (b)).  That is the convention
#: this artifact already uses for exactly this hazard — ``guard matched
#: (replay)`` publishes its ceiling as ``(n=covered/probes)`` plus a
#: paragraph naming a limit "identical across all six arms and unrelated to
#: storage shape", and ``median canonical rank`` carries
#: ``(n=found/candidates)``.  Neither drops the row: a dropped row hides
#: that the arm was measured at all, a disclosed row tells the reader both
#: the number and why it could not have been otherwise.
#: Neither `canonical in top-5` column can be quoted without the other, and
#: the disclosure saying so has to sit beside the numbers it qualifies rather
#: than three sections away.  Lifted out of :func:`_regrowth_lines` so its
#: PRESENCE and PLACEMENT are executable while its wording stays free — the
#: convention `REGROWTH_STAMPING_CEILING_DISCLOSURE` established below.
REGROWTH_CREDITED_SEMANTICS_DISCLOSURE = (
    'Both `canonical in top-5` columns travel together and neither can '
    'be quoted without the other.  The `(stored)` trio is the SCORED '
    'discoverability — `topic_discoverability` over the RAW store hits, '
    'before any `read_path` — so it is scored by the TRUE canonical '
    'record id and never by an aliased one, which is what makes it '
    'readable as retrieval.  The `(credited)` column is what the reader '
    'was handed AFTER the read transform ran, and under `promoting_pin` '
    'it is a PLACEMENT property: the transform injects the canonical '
    'into the window, in exactly the way `apply_grouped_read`\'s '
    'record-id aliasing did under `b_grouped`.  A credited column that '
    'holds while the stored one falls means the transform is masking '
    'regrowth at read time, not that retrieval survived it.'
)


REGROWTH_STAMPING_CEILING_DISCLOSURE = (
    '**Two of these three rows are forced to `0.00` by construction, not by '
    'measurement.**  `flat` cannot express a stamping difference at all: the '
    'two modes emit byte-identical record TEXT and differ only in a `topic` '
    'metadata key, mem0 embeds the content, and a flat read consults no '
    'metadata — a metadata-blind read arm has nothing to read the stamp '
    'with.  `additive_pin` is forced to `0.00` wherever the read window is '
    'full: the pin APPENDS its canonical and the window budget truncates it '
    'straight back off, which is the same thing `pin changed window = 0.00` '
    'reports elsewhere in this artifact.  `promoting_pin` is the only arm '
    'that can express one, and its value is bounded from ABOVE, because arm '
    '(c) stamps `topic` on every peer — so the pin usually fires from a '
    'sibling rather than from the injection\'s own stamp, and the probe is '
    'measuring stamping value against a 100%-stamped corpus.  That is the '
    'INVERSE of the live corpus esc-3200-3 documents, where '
    '`count_memories_by_metadata(topic=…)` returned one stamped record for '
    'the whole topic.  So a `0.00` in this table reads as "this arm could '
    'not express a difference", NOT as "stamping is worth nothing", and '
    'task 4006\'s campaign is not scored by this table alone.'
)


async def _gather_all(coroutines: Any) -> list[Any]:
    """``gather``, but never leaving a sibling coroutine in flight on failure.

    A bare ``asyncio.gather`` propagates the FIRST exception immediately and
    leaves every sibling running.  On the seeding side that is a real leak,
    not a tidiness point: control unwinds to :func:`run_bake_off`'s ``finally``,
    which closes the service and then drops the ephemeral collections — and an
    orphan write still in flight inside mem0's client can land AFTER the drop,
    recreating the collection under the reaped prefix.  That is precisely the
    leak the ISOLATION block above says cannot happen.

    ``return_exceptions=True`` makes every coroutine reach a terminal state
    before this returns, so nothing survives into teardown; the first failure
    is then re-raised UNCHANGED, so callers still see a plain
    :class:`SeedingError` / :class:`MeasurementError` rather than the
    ``ExceptionGroup`` a ``TaskGroup`` would hand them.

    Results keep input order, so callers can zip them back against their keys.
    """
    outcomes = await asyncio.gather(*coroutines, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return list(outcomes)


class SeedingError(RuntimeError):
    """A seeded record could not be mapped back to its stored id.

    Loud rather than skipped: a hit this script cannot resolve to an
    :class:`ArmRecord` is a hit that would silently vanish from every metric,
    and a shape that lost results would look like a shape that ranked badly.
    """


class MeasurementError(RuntimeError):
    """A query returned nothing measurable, so nothing was measured.

    ``Mem0Client.search`` swallows ``TimeoutError`` and returns ``{}``
    (``backends/mem0_client.py``), so a network failure and a shape that
    ranked nothing arrive here as the same empty list — which this module's
    own rule ("an empty hits list with a real expectation IS a measured
    zero") would then score as 0.0 recall, 0.0 discoverability, 0.0 guard
    across the board.  Every arm seeds the SHARED distractor slab, so an
    empty result list is never a legitimate outcome here: it is unambiguously
    a failed query, and it is raised rather than recorded.  Silently
    publishing a network failure as a measured zero in the artifact gate η
    reads is precisely the silent fail-soft the design invariants forbid.
    """


def load_cleanup_script() -> Any:
    """``scripts/cleanup_test_collections.py`` — the reaper, and the prefix's home."""
    return _load_sibling_script('cleanup_test_collections')


def ephemeral_collection_prefix() -> str:
    """The collection prefix, read from the reaper that has to match it."""
    return load_cleanup_script().E2_BAKEOFF_PREFIX


def worker_suffix() -> str:
    """The xdist worker id, or ``'main'`` outside a parallel test run.

    Never empty: an unsuffixed project_id would give two concurrent workers
    the same collection, and each would measure the other's arm.
    """
    import os  # noqa: PLC0415

    return os.environ.get('PYTEST_XDIST_WORKER') or 'main'


def arm_project_id(shape: str, *, suffix: str | None = None) -> str:
    """The ephemeral project id one arm seeds into."""
    return f'e2_bakeoff_{shape}_{suffix or worker_suffix()}'


def ephemeral_collections(
    shapes: tuple[str, ...] = ARM_SHAPES,
    *,
    suffix: str | None = None,
    prefix: str | None = None,
) -> dict[str, str]:
    """``shape -> qdrant collection name``, derived the way mem0 derives it.

    Through :meth:`Scope.mem0_collection_name` rather than an f-string here:
    ``Scope`` canonicalizes the project_id (lowercased, ``-`` to ``_``), so a
    hand-built name could differ from the collection mem0 actually writes to
    — and the reap would then miss it.
    """
    from fused_memory.models.scope import Scope  # noqa: PLC0415

    resolved_prefix = prefix or ephemeral_collection_prefix()
    return {
        shape: Scope(
            project_id=arm_project_id(shape, suffix=suffix),
        ).mem0_collection_name(resolved_prefix)
        for shape in shapes
    }


def drop_collections(names: Any, *, qdrant_url: str) -> None:
    """Delete each named collection, tolerating absence and per-name failure.

    Absence is the NORMAL case on the pre-run sweep, so it is not an error.
    A failure on one name must not abandon the rest: the whole point of the
    teardown is that nothing survives it.
    """
    from qdrant_client import QdrantClient  # noqa: PLC0415

    client = QdrantClient(url=qdrant_url, timeout=10)
    try:
        for name in names:
            try:
                client.delete_collection(name)
            except Exception as exc:  # noqa: BLE001 - absence is the normal case
                print(f'could not drop {name}: {exc}', file=sys.stderr)
    finally:
        client.close()


def canonical_record_ids(
    records: list[ArmRecord], claims: list[ArmClaim],
) -> dict[str, str]:
    """``cluster_id -> the record realizing that cluster's canonical claim``.

    Keyed off the CLAIM rather than off ``metadata['canonical']`` so that arm
    (a) — which carries no vocabulary metadata at all, because the α fixture
    genuinely has none — is measurable on the same footing as the two shapes
    that do.  Reading the metadata key would have reported arm (a) as having
    no canonical to discover, which is a statement about the measurement, not
    about the shape.
    """
    canonical_claims = {c.cluster_id: c.claim_id for c in claims if c.canonical}
    index: dict[str, str] = {}
    for record in records:
        if record.cluster_id is None:
            continue
        claim_id = canonical_claims.get(record.cluster_id)
        if claim_id is not None and claim_id in record.claim_ids:
            index.setdefault(record.cluster_id, record.record_id)
    return index


def contested_record_ids(
    records: list[ArmRecord], claims: list[ArmClaim],
) -> set[str]:
    """Records realizing a contested claim — never folded by the grouped read."""
    contested = {c.claim_id for c in claims if c.contested}
    return {
        record.record_id for record in records
        if any(claim_id in contested for claim_id in record.claim_ids)
    }


@dataclass
class SeededArm:
    """One materialized shape, live in its own ephemeral collection."""

    shape: str
    project_id: str
    collection: str
    records: list[ArmRecord]
    #: Mem0's assigned id -> the record it stored.  The join key between a
    #: search result and this experiment's bookkeeping.  Deliberately NOT a
    #: content match: mem0 is free to normalize what it echoes back, and a
    #: near-miss would silently drop the hit.
    by_stored_id: dict[str, ArmRecord]
    records_by_id: dict[str, ArmRecord]
    canonical_by_topic: dict[str, ArmRecord]
    contested_ids: set[str]
    canonical_by_cluster: dict[str, str]
    siblings_by_cluster: dict[str, set[str]]
    #: α ``memory_id`` -> every record in THIS arm that realizes a claim
    #: decomposed from it.  The guard probe drops its own write's content by
    #: PROVENANCE, not by record id: only ``_materialize_status_quo`` reuses
    #: the α memory_id as its record id, so an id-equality filter is a no-op
    #: in ``c_peers``/``b_grouped`` and would leave those arms searching a
    #: corpus that still contains the probing write's own claims.
    records_by_source: dict[str, set[str]]


def _index_arm(
    shape: str, project_id: str, collection: str,
    records: list[ArmRecord], claims: list[ArmClaim],
) -> SeededArm:
    """Every pure index an arm needs, built once before a byte is written."""
    siblings: dict[str, set[str]] = {}
    for record in records:
        if record.cluster_id is not None:
            siblings.setdefault(record.cluster_id, set()).add(record.record_id)
    source_by_claim = {claim.claim_id: claim.source_memory_id for claim in claims}
    records_by_source: dict[str, set[str]] = {}
    for record in records:
        for claim_id in record.claim_ids:
            source = source_by_claim.get(claim_id)
            if source is not None:
                records_by_source.setdefault(source, set()).add(record.record_id)
    return SeededArm(
        shape=shape,
        project_id=project_id,
        collection=collection,
        records=records,
        by_stored_id={},
        records_by_id={record.record_id: record for record in records},
        canonical_by_topic=build_canonical_by_topic(records),
        contested_ids=contested_record_ids(records, claims),
        canonical_by_cluster=canonical_record_ids(records, claims),
        siblings_by_cluster=siblings,
        records_by_source=records_by_source,
    )


async def seed_arm(
    backend: Any,
    seeded: SeededArm,
    *,
    concurrency: int = SEED_CONCURRENCY,
) -> SeededArm:
    """Write every record of one arm into its own ephemeral collection.

    Populates ``by_stored_id`` in place and returns the same object.

    ``instance.db.add_history`` is stubbed first: mem0's SQLite history
    writer is process-shared and xdist-contended (and read-only in the
    sandbox).  It is not the question under test, and its failure would mask
    the one that is — the same reason ``test_recon_dedup_premise.py:135`` and
    the E1 probe stub it.
    """
    from fused_memory.models.scope import Scope  # noqa: PLC0415

    scope = Scope(project_id=seeded.project_id, agent_id=SEED_AGENT_ID)
    instance = await backend._get_instance(scope)
    instance.db.add_history = lambda *args, **kwargs: None

    semaphore = asyncio.Semaphore(concurrency)

    async def _write(record: ArmRecord) -> None:
        async with semaphore:
            response = await backend.add(
                record.content, scope, metadata=dict(record.metadata),
            )
        results = (response or {}).get('results') or []
        if not results or not results[0].get('id'):
            raise SeedingError(
                f'mem0 returned no id for {seeded.shape} record '
                f'{record.record_id!r}. Continuing would measure an arm that '
                f'is quietly missing a record.'
            )
        seeded.by_stored_id[results[0]['id']] = record

    await _gather_all(_write(record) for record in seeded.records)
    return seeded


async def _search(
    backend: Any, seeded: SeededArm, text: str, *, limit: int,
) -> list[ScoredHit]:
    """One ranked list from the arm's collection, joined back to its records."""
    from fused_memory.models.scope import Scope  # noqa: PLC0415

    response = await backend.search(
        query=text, scope=Scope(project_id=seeded.project_id), limit=limit,
    )
    results = (response or {}).get('results') or []
    if not results:
        raise MeasurementError(
            f'{seeded.collection} returned no results for a {limit}-limit '
            f'query. Every arm seeds the shared distractor slab, so an empty '
            f'ranking is not a possible outcome — this is a failed query '
            f'(mem0 swallows TimeoutError and returns {{}}), and recording it '
            f'would publish a network failure as a measured zero. '
            f'Query text: {text[:120]!r}'
        )
    hits: list[ScoredHit] = []
    for item in results:
        record = seeded.by_stored_id.get(item.get('id'))
        if record is None:
            raise SeedingError(
                f"{seeded.collection} returned point {item.get('id')!r}, which "
                f'this run did not seed. The collection is not clean; dropping '
                f'the hit would silently shrink the measured ranking.'
            )
        score = item.get('score')
        if score is None:
            raise MeasurementError(
                f"{seeded.collection} returned point {item.get('id')!r} with "
                f'no score. Defaulting it to 0.0 would feed the threshold '
                f'replay a value the store never produced, and the guard '
                f'column would read as a shape finding rather than as '
                f'missing score plumbing.'
            )
        hits.append(ScoredHit(record=record, relevance_score=float(score)))
    return hits


def probe_own_record_ids(seeded: SeededArm, probe_memory_id: str) -> set[str]:
    """Every record in this arm that carries the probing write's own content.

    Provenance, not id equality.  ``record_id == memory_id`` holds ONLY in
    ``_materialize_status_quo``; ``c_peers`` and ``b_grouped`` derive every id
    through :func:`_derive_record_id`, so an id-equality filter silently
    removes nothing there and the probe would search a corpus that still
    contains its own claims — a free self-match for exactly the two arms the
    :data:`GUARD_FETCH_LIMIT` rationale says structurally cannot get one.

    The id itself stays in the set so the status-quo arm, whose originals may
    carry no decomposed claim at all, is still covered.
    """
    return {probe_memory_id} | set(seeded.records_by_source.get(probe_memory_id, ()))


async def fetch_arm(
    backend: Any,
    seeded: SeededArm,
    queries: list[Query],
    probes: list[tuple[str, dict[str, Any]]],
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
    concurrency: int = SEED_CONCURRENCY,
) -> dict[str, dict[str, list[ScoredHit]]]:
    """Every ranked list this arm needs, fetched ONCE.

    Both pin variants are measured from these same lists, so the pin
    comparison is exactly controlled and the run pays for one shape's
    embeddings rather than two.

    Guard probes drop the probing write's own record — see
    :data:`GUARD_FETCH_LIMIT`.

    SERIAL, deliberately, and *concurrency* is accepted only to keep the
    signature honest about the knob that was tried.

    The obvious efficiency win is to bound-concurrent this the way
    :func:`seed_arm` does — 236 queries + 15 probes per arm, 753 sequential
    embed+ANN round trips across the three, every one of them independent and
    keyed by ``query_id``/``cluster_id`` rather than by arrival order, so
    concurrency changes nothing about determinism.  It was implemented at
    ``asyncio.Semaphore(SEED_CONCURRENCY)`` and MEASURED: the live run aborted
    partway through the second arm with

        MeasurementError: _test_e2_bakeoff_e2_bakeoff_c_peers_main returned no
        results for a 10-limit query

    Hypothesis for that observation: eight concurrent searches queue behind
    each other in mem0's client and exceed
    ``config.queue.backend_read_timeout_seconds``, which
    ``Mem0Client.search`` SWALLOWS — it logs and returns ``{}``.  The read
    side is therefore the one place where concurrency converts a latency win
    into a silently-empty ranking, and this script's whole posture is that an
    empty ranking must never be published as a measured zero.  The write side
    keeps its bound because ``add`` propagates its timeout instead of
    swallowing it.

    Re-derive before re-trying: a smaller bound, or a read timeout raised for
    the run, would both be defensible — but neither is worth a report that
    cannot be regenerated.
    """
    del concurrency  # documented above; the serial path is the measured one

    fetched_queries: dict[str, list[ScoredHit]] = {}
    for query in queries:
        fetched_queries[query.query_id] = await _search(
            backend, seeded, query.text, limit=limit,
        )

    fetched_probes: dict[str, list[ScoredHit]] = {}
    for cluster_id, probe in probes:
        own = probe_own_record_ids(seeded, probe['memory_id'])
        raw = await _search(
            backend, seeded, probe['content'], limit=guard_fetch_limit(own),
        )
        fetched_probes[cluster_id] = [
            hit for hit in raw if hit.record.record_id not in own
        ]

    return {'queries': fetched_queries, 'probes': fetched_probes}


# ---------------------------------------------------------------------------
# The fetch cache — replaying a live run's rankings without a live run
# ---------------------------------------------------------------------------
#
# :func:`fetch_arm` above is the ONLY thing in this experiment that costs an
# embedder call and a live Qdrant collection.  Everything downstream of it —
# :func:`read_path`, :func:`rescore`, :func:`measure_arm`, :func:`build_report`,
# :func:`render_markdown` — is pure, and ``measure_arm``'s docstring says so
# explicitly (no store, no await).  Dumping ``fetch_arm``'s return value at
# this seam and replaying it therefore makes every read-side variant free
# forever, and makes the metric code unit-testable against REAL rankings
# rather than only hand-built ones.
#
# WHAT IS STORED, and why it is only this:
#
#   ``record_id`` and ``relevance_score``.  Nothing else.
#
#   NOT mem0's point id.  :func:`_search` joins its results back through
#   ``seeded.by_stored_id[item['id']]``, and ``Mem0Backend.add`` pins
#   ``infer=False`` — which routes to mem0's fresh-uuid ``_create_memory``
#   path.  ``item['id']`` is therefore newly minted on EVERY seeding run, so a
#   cache keyed on it is replayable exactly once and then rehydrates to an
#   empty ranking, which this script's whole posture forbids publishing as a
#   measured zero.  :func:`_derive_record_id` is uuid5-deterministic over
#   ``(shape, key)`` and :func:`materialize_arm` is pure over the committed
#   fixtures, so the join key is stable across runs by construction.
#
#   NOT the record body.  It is fully reconstructible from the committed
#   fixtures via ``materialize_arm``, so storing it would make the cache both
#   large and — the real objection — capable of DISAGREEING with the fixtures
#   it claims to describe, with no way for a reader to tell which won.

#: Bumped when the on-disk cache layout changes incompatibly.  A replay is a
#: measurement, so an unreadable cache must fail rather than degrade.
FETCH_CACHE_SCHEMA_VERSION = 1


class FetchCacheError(RuntimeError):
    """A fetch cache cannot be trusted to describe the corpus being replayed.

    A sibling of :class:`FixtureError`/:class:`SeedingError`/
    :class:`MeasurementError`, and raised for the same reason they are: a
    replayed report is indistinguishable from a live one in the artifact, so
    every way the cache can be wrong has to be loud at load time rather than
    silently producing a plausible table.
    """


def corpus_fingerprint(records: list[ArmRecord]) -> str:
    """sha256 over what a replay must agree with the dump about.

    ``(record_id, content, sorted metadata)`` — all three, because all three
    feed retrieval: the id is the cache's join key, the content is what was
    embedded, and the metadata is what every read transform branches on.  A
    fingerprint over ids alone would pass cleanly through a fixture edit that
    rewrote every body.

    SORTED, so the digest answers a set question ("does the cache describe
    THIS corpus?") rather than a sequence one.  ``materialize_arm`` is already
    deterministically ordered, so this is never exercised by the real
    pipeline; it is chosen so that a reordering cannot raise a false alarm
    while every content/id/metadata change still does.

    ``json.dumps(sort_keys=True)`` rather than ``str(dict)`` so the digest
    does not depend on metadata insertion order either.
    """
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    for record_id, content, metadata in sorted(
        (
            record.record_id,
            record.content,
            json.dumps(record.metadata, sort_keys=True, default=str),
        )
        for record in records
    ):
        # Length-prefixed, so two different corpora cannot collide by
        # straddling a separator that happens to occur inside a body.
        for part in (record_id, content, metadata):
            digest.update(f'{len(part)}:'.encode())
            digest.update(part.encode('utf-8'))
    return digest.hexdigest()


def fixture_digests(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Repo-relative path + sha256 of each fixture's CONTENT.

    Deliberately not folded into :func:`fixture_provenance`, which stamps the
    last-touching COMMIT.  The two answer different questions and the cache
    needs both: a commit sha is the blind-authoring audit trail, but it does
    not move when a fixture is edited in the working tree — which is exactly
    the drift that silently invalidates a dumped cache.

    Paths are reported repo-relative for the same reason
    :func:`fixture_provenance` does it (:1852): an artifact naming somebody's
    absolute home-directory checkout is neither reproducible nor readable by
    anyone else.
    """
    import hashlib  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).resolve()
        # Through :func:`_repo_relative`, like every other call site.  Not
        # cosmetic: `_check_fixture_digests` compares these rows to the
        # cache's as a `path -> sha256` MAPPING, and the regrowth block names
        # the same injection fixture through that helper — so a second
        # spelling that drifted would surface as an unexplained "cache
        # carries no digest for <path>" refusal rather than as a diff here.
        rows.append({
            'path': _repo_relative(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return rows


def fetch_cache_provenance(
    *,
    records_by_shape: dict[str, list[ArmRecord]],
    fixtures: list[str | Path],
    search_limit: int,
    guard_threshold: float,
    embedder_model: str,
) -> dict[str, Any]:
    """Everything a replayed report needs that the rankings do not carry.

    ``guard_threshold`` and ``embedder_model`` ride along because they are the
    only other LIVE-ONLY values that reach the report's protocol block
    (:3264, :3297): ``resolve_guard_threshold`` navigates a real
    ``MemoryService`` to find the threshold production would run at, and the
    embedder model comes off the live config.  Without them a replayed report
    is unbuildable — not merely less detailed — and defaulting either would
    feed the threshold replay a number the store never produced.
    """
    return {
        'corpus_fingerprints': {
            shape: corpus_fingerprint(records)
            for shape, records in sorted(records_by_shape.items())
        },
        'fixtures': fixture_digests(list(fixtures)),
        'search_limit': int(search_limit),
        'guard_threshold': float(guard_threshold),
        'embedder_model': embedder_model,
    }


def _serialize_ranking(hits: list[ScoredHit]) -> list[list[Any]]:
    """One ranked list as ``[[record_id, score], ...]``, in RANK ORDER.

    A list of pairs rather than an object keyed by record id: every metric in
    this experiment is rank-based, so order IS the data, and a JSON object
    would invite a re-sort that no downstream assertion could detect.
    """
    return [
        [hit.record.record_id, float(hit.relevance_score)] for hit in hits
    ]


def fetches_to_document(
    arms: dict[str, dict[str, dict[str, list[ScoredHit]]]],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """The on-disk form of one run's fetches: a pure ranking map.

    *arms* maps each shape to that shape's :func:`fetch_arm` result.  Pure —
    the caller is still measuring off these same lists (``run_arm`` calls
    ``measure_arm`` twice off one fetch), so nothing here may mutate them.

    *provenance* is REQUIRED, not defaulted.  A cache with no fingerprint can
    never be loaded (:func:`_check_corpus_fingerprint` refuses it), so making
    it optional would only move the failure to load time — i.e. to AFTER a
    full live seeding run had already been spent producing an unusable file.
    """
    if not provenance:
        raise FetchCacheError(
            'refusing to dump a fetch cache with no provenance block. Without '
            'a per-shape corpus fingerprint the cache can never be verified '
            'against the fixtures it is replayed over, and `load_fetches` '
            'would refuse it — after a full seeding run had been spent '
            'producing it. Build the block with `fetch_cache_provenance`.'
        )
    return {
        'schema_version': FETCH_CACHE_SCHEMA_VERSION,
        'provenance': dict(provenance),
        'arms': {
            shape: {
                'queries': {
                    query_id: _serialize_ranking(hits)
                    for query_id, hits in sorted(fetched['queries'].items())
                },
                'probes': {
                    cluster_id: _serialize_ranking(hits)
                    for cluster_id, hits in sorted(fetched['probes'].items())
                },
            }
            for shape, fetched in sorted(arms.items())
        },
    }


def dump_fetches(
    path: str | Path,
    arms: dict[str, dict[str, dict[str, list[ScoredHit]]]],
    *,
    provenance: dict[str, Any],
) -> Path:
    """Write the fetch cache atomically, byte-stably.

    ``sort_keys=True`` on top of the sorted comprehensions above because the
    cache is a COMMITTED artifact: a re-dump of the same fetches must diff
    cleanly, and JSON object order otherwise follows insertion order — which
    follows query-set iteration order, which is not a property anyone should
    have to keep stable by hand.
    """
    target = Path(path)
    _atomic_write_text(
        target,
        json.dumps(
            fetches_to_document(arms, provenance=provenance),
            indent=2, sort_keys=True,
        ) + '\n',
    )
    return target


def _rehydrate_ranking(
    seeded: SeededArm, entries: Any, *, shape: str, kind: str, key: str,
) -> list[ScoredHit]:
    """One cached ranking back to ``list[ScoredHit]``, joined to THIS arm."""
    hits: list[ScoredHit] = []
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            raise FetchCacheError(
                f'fetch cache entry for {shape}/{kind}/{key} is {entry!r}, not '
                f'a [record_id, score] pair. Skipping it would shrink a '
                f'measured ranking silently.'
            )
        record_id, score = entry
        record = seeded.records_by_id.get(record_id)
        if record is None:
            raise FetchCacheError(
                f'fetch cache names record {record_id!r} in {shape}/{kind}/'
                f'{key}, which this run did not materialize. The cache and the '
                f'fixtures describe different corpora — dropping the hit would '
                f'silently shrink the measured ranking (the same refusal '
                f'`_search` makes for an unseeded point). Re-dump the cache '
                f'against the current fixtures.'
            )
        hits.append(ScoredHit(record=record, relevance_score=float(score)))
    return hits


def _read_fetch_cache(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """The parsed cache document, or a named refusal."""
    target = Path(path)
    try:
        doc = json.loads(target.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FetchCacheError(
            f'no fetch cache at {target} — a replay measures nothing without '
            f'one. Produce it with --dump-fetches on a live run.'
        ) from exc
    except json.JSONDecodeError as exc:
        raise FetchCacheError(
            f'fetch cache at {target} is not valid JSON: {exc}'
        ) from exc

    version = doc.get('schema_version')
    if version != FETCH_CACHE_SCHEMA_VERSION:
        raise FetchCacheError(
            f'fetch cache at {target} is schema version {version!r}, but this '
            f'build reads version {FETCH_CACHE_SCHEMA_VERSION}.'
        )
    return target, doc


def load_fetch_provenance(path: str | Path) -> dict[str, Any]:
    """The dump's provenance block — the live-only half of a replayed report.

    Separate from :func:`load_fetches` because the driver needs it BEFORE it
    has materialized anything (``guard_threshold`` and ``embedder_model`` go
    into the protocol block), and because reading it must not require the
    caller to have already built every arm.
    """
    _, doc = _read_fetch_cache(path)
    provenance = doc.get('provenance')
    if not isinstance(provenance, dict):
        raise FetchCacheError(
            f'fetch cache at {Path(path)} carries no provenance block, so the '
            f'guard threshold and embedder model this run was measured at are '
            f'unknown. Defaulting either would feed the threshold replay a '
            f'number the store never produced.'
        )
    return provenance


def _check_corpus_fingerprint(
    target: Path, provenance: dict[str, Any], shape: str, seeded: SeededArm,
) -> None:
    """Refuse a cache whose corpus is not the one being replayed.

    The failure this exists for is silent by construction: a stale cache still
    LOADS, it just describes records that no longer exist, and the replayed
    report is indistinguishable from a fresh measurement.
    """
    cached = (provenance.get('corpus_fingerprints') or {}).get(shape)
    if cached is None:
        raise FetchCacheError(
            f'fetch cache at {target} carries no corpus fingerprint for shape '
            f'{shape!r}, so it cannot be checked against the fixtures this run '
            f'materialized — and an unverifiable cache replayed as a '
            f'measurement is exactly the silent staleness the fingerprint '
            f'exists to catch. Re-dump the cache with --dump-fetches.'
        )
    local = corpus_fingerprint(seeded.records)
    if cached != local:
        raise FetchCacheError(
            f'fetch cache at {target} was dumped against a different {shape!r} '
            f'corpus: cached fingerprint {cached}, but the fixtures in this '
            f'tree materialize {local} ({len(seeded.records)} records). THE '
            f'FIXTURES MOVED — the cached rankings name records that no longer '
            f'exist, or omit records that now do, so replaying them would '
            f'publish a stale ranking as a fresh measurement. Re-run the '
            f'seeding pass with --dump-fetches.'
        )


def _check_fixture_digests(
    target: Path, provenance: dict[str, Any], expect_fixtures: list[str | Path],
) -> None:
    """Refuse a cache whose fixtures are not the ones being replayed.

    The corpus fingerprint does NOT cover this: :func:`materialize_arm` takes
    no queries, so the fingerprint is structurally blind to the query set.
    Edit a query's ``text`` while keeping its ``query_id`` and every
    fingerprint still matches, while the cached rankings — fetched against the
    old text — become a measurement of a question nobody asked.

    Compared as a ``path -> sha256`` MAPPING, never positionally:
    :func:`fixture_digests` emits in the caller's argument order, so a caller
    that merely reordered its list would otherwise be told the fixtures moved.
    """
    cached = provenance.get('fixtures')
    if not isinstance(cached, list):
        raise FetchCacheError(
            f'fetch cache at {target} records no fixture digests, so the '
            f'fixtures it was fetched over cannot be compared with the ones in '
            f'this tree. The per-shape corpus fingerprint does not cover this: '
            f'`materialize_arm` takes no queries, so a query-set edit leaves '
            f'every fingerprint intact. Re-dump the cache with --dump-fetches.'
        )
    cached_by_path = {
        str(row.get('path')): str(row.get('sha256'))
        for row in cached if isinstance(row, dict)
    }
    for row in fixture_digests(list(expect_fixtures)):
        was = cached_by_path.get(row['path'])
        if was is None:
            raise FetchCacheError(
                f'fetch cache at {target} was dumped over a different fixture '
                f'set: it carries no digest for {row["path"]} (it carries '
                f'{sorted(cached_by_path)}). A cache that never saw a fixture '
                f'cannot answer for it.'
            )
        if was != row['sha256']:
            raise FetchCacheError(
                f'fetch cache at {target} was dumped against a different '
                f'{row["path"]}: cached sha256 {was}, but the file in this '
                f'tree hashes to {row["sha256"]}. THE FIXTURES MOVED — the '
                f'cached rankings answer the OLD file, and replaying them '
                f'would publish them as a fresh measurement. (The corpus '
                f'fingerprint cannot catch a query-set edit: `materialize_arm` '
                f'takes no queries.) Re-run the seeding pass with '
                f'--dump-fetches.'
            )


def load_fetches(
    path: str | Path,
    seeded_by_shape: dict[str, SeededArm],
    *,
    expect_query_ids: list[str] | None = None,
    expect_limit: int | None = None,
    expect_fixtures: list[str | Path] | None = None,
) -> dict[str, dict[str, dict[str, list[ScoredHit]]]]:
    """Replay a dumped run's rankings against locally materialized arms.

    Returns exactly what :func:`fetch_arm` would have returned for each
    requested shape, so ``measure_arm`` cannot tell a replay from a live run —
    which is the point, and also why every disagreement between the cache and
    the corpus has to raise HERE instead of surfacing downstream as a
    plausible-looking table.

    Iterates the CALLER's shapes, not the cache's: a cache carrying extra
    shapes is merely wider than this run needs, but a requested shape it does
    not carry would otherwise be measured as an arm with no queries.

    *expect_query_ids*, *expect_limit* and *expect_fixtures* are the
    truncation, depth and fixture-drift guards.  All three are opt-in so the
    pure round-trip tests can exercise the join without a full query set, and
    all three are supplied by the live driver.
    """
    target, doc = _read_fetch_cache(path)
    provenance = load_fetch_provenance(path)
    cached_arms = doc.get('arms')
    if not isinstance(cached_arms, dict):
        raise FetchCacheError(
            f'fetch cache at {target} carries no "arms" block.'
        )

    if expect_limit is not None:
        cached_limit = provenance.get('search_limit')
        if cached_limit is None:
            raise FetchCacheError(
                f'fetch cache at {target} does not record the search limit it '
                f'was fetched at, so it cannot be checked against the '
                f'requested depth of {expect_limit}.'
            )
        if int(cached_limit) < int(expect_limit):
            raise FetchCacheError(
                f'fetch cache at {target} was fetched at limit '
                f'{int(cached_limit)}, shallower than the requested limit '
                f'{int(expect_limit)}. Replaying it would measure every arm '
                f'over a {int(cached_limit)}-deep window while the report '
                f'claimed {int(expect_limit)} — a silently DIFFERENT '
                f'experiment, not a smaller one. (A deeper cache is fine: '
                f'`read_path` truncates to the reader budget anyway.)'
            )

    if expect_fixtures is not None:
        _check_fixture_digests(target, provenance, expect_fixtures)

    replayed: dict[str, dict[str, dict[str, list[ScoredHit]]]] = {}
    for shape, seeded in seeded_by_shape.items():
        block = cached_arms.get(shape)
        if block is None:
            raise FetchCacheError(
                f'fetch cache at {target} has no rankings for shape {shape!r} '
                f'(it carries {sorted(cached_arms)}). Measuring that arm from '
                f'an absent cache would publish an empty ranking as a zero.'
            )
        _check_corpus_fingerprint(target, provenance, shape, seeded)

        cached_queries = block.get('queries') or {}
        if expect_query_ids is not None:
            missing = [q for q in expect_query_ids if q not in cached_queries]
            if missing:
                raise FetchCacheError(
                    f'fetch cache at {target} is TRUNCATED for shape {shape!r}: '
                    f'{len(missing)} of {len(expect_query_ids)} requested '
                    f'queries have no cached ranking, e.g. {missing[:5]}. '
                    f'Measuring the arm anyway would score it over a subset of '
                    f'the query set while the report claimed the whole one.'
                )
        replayed[shape] = {
            kind: {
                key: _rehydrate_ranking(
                    seeded, entries, shape=shape, kind=kind, key=key,
                )
                for key, entries in sorted((block.get(kind) or {}).items())
            }
            for kind in ('queries', 'probes')
        }
    return replayed


def read_path(
    seeded: SeededArm, hits: list[ScoredHit], k: int, *, pin: bool,
    promote: bool = False,
) -> list[ArmRecord]:
    """What a reader of this arm variant actually receives for a top-*k* fetch.

    Truncated TWICE, and both truncations are load-bearing.

    BEFORE the transforms, because that is the order production runs in: the
    store returns k, and the read-side transforms act on what came back.
    Transforming first would let a record the store never returned into the
    window.

    AFTER them, because *k* is the READER's budget and pin-on/pin-off have to
    be scored over equal-size windows.  ``apply_topic_anchor`` is additive, so
    without this an arm variant would be measured over k+1 records and its
    every column would improve for having been handed an extra result — a
    "win" that says nothing about the storage shape.  With it, an append-only
    pin can only pay off where a read-side transform left HEADROOM in the
    budget.  Grouping, which SHRINKS the window, is untouched: a collapsed
    window stays short (its legitimate token win) and a canonical pinned into
    the slot grouping freed survives.

    ``promote`` selects 4004's PROMOTING variant of the pin
    (``read_transform_selection.apply_promoting_topic_anchor``) instead of
    the landed additive one.  Both truncations stay exactly as they are —
    the promoting pin's whole measurable difference is that it survives the
    second one.  It requires ``pin``: promotion is a variant of the pin's
    firing rule, not an independent transform.
    """
    if promote and not pin:
        raise ValueError(
            'promote=True requires pin=True — promotion is a variant of the '
            'topic pin\'s FIRING RULE (the two select the identical set of '
            'records and differ only in where the canonical lands), so '
            '"promote without pin" is a caller bug rather than a fourth, '
            'silent read mode.'
        )

    records = [hit.record for hit in hits[:k]]
    if seeded.shape == 'b_grouped':
        records = apply_grouped_read(
            records, seeded.records_by_id, contested_ids=seeded.contested_ids,
        )
    if pin:
        anchor = (
            load_read_transform_script().apply_promoting_topic_anchor
            if promote else apply_topic_anchor
        )
        records = anchor(records, seeded.canonical_by_topic)
    return records[:k]


def rescore(transformed: list[ArmRecord], hits: list[ScoredHit]) -> list[ScoredHit]:
    """Re-attach the store's scores after a read-side transform.

    Only the guard replay needs scores, and only because the production
    selector is threshold-based.

      * a record the store ranked keeps its own score;
      * a grouped document takes the BEST score available to it — its own if
        the canonical was itself a hit, its best child's, whichever is
        higher.  This is the same rule ``apply_grouped_read`` applies to
        RANK (``group_rank[parent] = min(...)``), and the two have to agree:
        a canonical hit at rank 5 / score 0.30 with a child at rank 1 /
        score 0.95 ranks the group 1st, so replaying the threshold guard at
        0.30 would be a false negative attributable to this helper rather
        than to the storage shape;
      * a PINNED canonical the store never returned scores 0.0 — the honest
        value, because the ANN produced no similarity evidence for it at all.
        It still counts for the rank-based ``candidate_present`` half, which
        is where a pin can legitimately show a win.
    """
    direct = {hit.record.record_id: hit.relevance_score for hit in hits}
    best_child: dict[str, float] = {}
    for hit in hits:
        parent_id = hit.record.metadata.get('parent_id')
        if parent_id is not None:
            best_child[parent_id] = max(
                best_child.get(parent_id, 0.0), hit.relevance_score,
            )
    return [
        ScoredHit(
            record=record,
            relevance_score=max(
                direct.get(record.record_id, 0.0),
                best_child.get(record.record_id, 0.0),
            ),
        )
        for record in transformed
    ]


def _mean(values: list[float | None]) -> float | None:
    """Mean over the MEASURED values, or ``None`` when there are none.

    ``None`` entries are non-observations, not zeros: averaging them in as 0
    would drag an arm's reported score down for questions it was never asked.
    """
    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return sum(measured) / len(measured)


def _aggregate_queries(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    """Claim recall + discoverability over one subset of the measured queries.

    Used for the pooled headline AND for each by-kind subset, so a subset can
    never be computed by a second, subtly different code path than the total.

    ``queries: 0`` is a legitimate result and every metric under it is
    ``None`` — an empty subset is "not asked", never a measured zero.
    """
    import statistics  # noqa: PLC0415

    canonical_ranks = [
        row['canonical_rank'] for row in rows if row['canonical_rank'] is not None
    ]
    median_rank: float | None = None
    if canonical_ranks:
        median_rank = float(statistics.median(canonical_ranks))
    #: The TRANSFORM-BLIND half of the same question — see the module
    #: docstring, "Rank-based is not transform-blind".  Same aggregation, over
    #: ranks the caller measured on the raw store hits.
    stored_ranks = [
        row['stored_canonical_rank'] for row in rows
        if row['stored_canonical_rank'] is not None
    ]
    stored_median_rank: float | None = None
    if stored_ranks:
        stored_median_rank = float(statistics.median(stored_ranks))
    return {
        'queries': len(rows),
        'claim_recall': {
            'at_5': _mean([row['recall_5'] for row in rows]),
            'at_10': _mean([row['recall_10'] for row in rows]),
        },
        'discoverability': {
            'canonical_in_top_5_rate': _mean([row['canonical_in_5'] for row in rows]),
            'median_canonical_rank': median_rank,
            'canonical_found_count': len(canonical_ranks),
            # Queries that HAD a canonical to look for — the honest
            # denominator for `canonical_found_count`.  The median rank is a
            # median over successes only, so without this an arm that almost
            # never finds the canonical prints the best median in the table.
            'canonical_candidates': sum(1 for row in rows if row['has_canonical']),
            # Ranks are censored at the fetch depth, not at the read window.
            # Stated so a reader knows what "absent entirely" here means —
            # and taken from the depth THIS run fetched at, never from the
            # module default: `--limit` is a CLI flag, and an artifact that
            # printed the constant would understate how far a `--limit 25`
            # run actually looked. That is the same class of defect as the
            # censored median this field exists to disclose.
            'canonical_rank_window': limit,
            # The transform-blind counterpart of the three keys above, in the
            # same order and with the same contracts: rate through `_mean`
            # (so a query with no canonical is a non-observation, not a 0.0),
            # median over successes only, and the censored denominator that
            # median is taken over.  `canonical_candidates` is shared — both
            # halves ask about the same queries, only through different
            # windows — so it is not duplicated.
            'stored_canonical_in_top_5_rate': _mean(
                [row['stored_canonical_in_5'] for row in rows],
            ),
            'stored_canonical_median_rank': stored_median_rank,
            'stored_canonical_found_count': len(stored_ranks),
        },
    }


def measure_arm(
    seeded: SeededArm,
    fetched: dict[str, dict[str, list[ScoredHit]]],
    *,
    pin: bool,
    promote: bool = False,
    queries: list[Query],
    probes: list[tuple[str, dict[str, Any]]],
    estimator: tuple[str, Any],
    guard_threshold: float,
    limit: int,
) -> dict[str, Any]:
    """The E2 metrics for one arm variant.  Pure — no store, no await.

    Every metric is scored at the LITERAL k, never at ``len(window)``.  The
    window a reader receives is capped at k by :func:`read_path`, and scoring
    at its length would hand an append-only pin a wider budget than its
    pin-off twin — see that function for the full rationale.

    The ``pin`` block reports whether the pin was enabled and, when it was,
    the fraction of measured read windows it actually changed.  Without it two
    identical rows read as "the pin is useless" when the honest statement may
    be "the pin never fired": at a full window an additive pin has nowhere to
    put anything.  It is ``None`` when the pin is off — the question was never
    asked, and a 0.0 would claim it was asked and answered.

    ``promote`` selects 4004's promoting pin for the transformed windows
    only.  The pin-off baseline inside ``_window`` and the transform-blind
    ``stored_`` trio stay promote-free BY CONSTRUCTION — the former because
    ``read_path`` refuses ``promote`` without ``pin``, the latter because it
    never calls ``read_path`` at all.
    """
    #: One row per query, kept individually rather than accumulated straight
    #: into means.  eval-design §5 E2 names "claim recall@k" and
    #: "canonical/topic discoverability" as DISTINCT metrics, and the fixture
    #: encodes two query kinds plus a held-out phrasing split for exactly that
    #: reason.  Pooling 236 queries into one mean makes a shape that wins on
    #: claim queries while losing on topic phrasings indistinguishable from
    #: one that ties on both — and the held-out subset, which exists to test
    #: generalisation beyond the phrasings the registry was derived from,
    #: disappears entirely.  Rows in, subsets out.
    rows: list[dict[str, Any]] = []
    #: Every read window this arm was measured over, pin-on vs pin-off.  Both
    #: query windows AND the guard probe window: a pin that fires only on a
    #: probe still moves the guard column, and a diagnostic that missed it
    #: would say "nothing changed" next to a column that did.
    windows_compared = 0
    windows_changed = 0

    def _window(hits: list[ScoredHit], k: int) -> list[ArmRecord]:
        """One read window, plus its pin-off counterfactual when pin is on."""
        nonlocal windows_compared, windows_changed
        records = read_path(seeded, hits, k, pin=pin, promote=promote)
        if pin:
            # The counterfactual stays pin-OFF, and therefore promote-free BY
            # CONSTRUCTION (`read_path` refuses promote without pin).  A
            # promote leak here would make the two windows agree and the
            # diagnostic would report 0.00 for an arm that reorders every
            # window it touches.
            baseline = read_path(seeded, hits, k, pin=False)
            windows_compared += 1
            if [r.record_id for r in records] != [r.record_id for r in baseline]:
                windows_changed += 1
        return records

    for query in queries:
        hits = fetched['queries'][query.query_id]
        top_5 = _window(hits, 5)
        top_10 = _window(hits, 10)

        canonical_in_5: float | None = None
        canonical_rank: int | None = None
        #: The SAME question asked of the raw store hits, before any
        #: read-side transform ran — see the module docstring, "Rank-based is
        #: not transform-blind".  Measured over `[hit.record for hit in
        #: hits]`, so this pair is blind to grouping AND to the pin.
        stored_canonical_in_5: float | None = None
        stored_canonical_rank: int | None = None
        canonical_id = seeded.canonical_by_cluster.get(query.cluster_id)
        if canonical_id is not None:
            found = topic_discoverability(top_5, query.topic, canonical_id, 5)
            canonical_in_5 = 1.0 if found['canonical_in_top_k'] else 0.0
            # RANK is measured over the deepest window this arm was fetched
            # at, NOT over the k=5 read window the rate is measured over.
            # Scoring the rank inside an already-truncated window censors it
            # at 5, collapsing "rank 7" and "absent entirely" into the same
            # None — exactly the conflation topic_discoverability's docstring
            # says would hide a shape that gets the canonical NEARLY there.
            deep = topic_discoverability(
                read_path(seeded, hits, len(hits), pin=pin, promote=promote),
                query.topic, canonical_id, 5,
            )
            canonical_rank = deep['canonical_rank']
            # ONE call, over the untransformed hits: `topic_discoverability`
            # already scans the full list for the rank and derives
            # `canonical_in_top_k` from it against the k it is passed, so a
            # second windowed call would be redundant.  No `read_path` here
            # is the whole point — neither `apply_grouped_read` nor
            # `apply_topic_anchor` runs, so nothing can materialise a record
            # the store did not return.  Read-only over hits already
            # fetched: no arm, pin, window or threshold is re-tuned.
            stored = topic_discoverability(
                [hit.record for hit in hits], query.topic, canonical_id, 5,
            )
            stored_canonical_in_5 = 1.0 if stored['canonical_in_top_k'] else 0.0
            stored_canonical_rank = stored['canonical_rank']

        rows.append({
            'kind': query.kind,
            'held_out': query.held_out,
            'recall_5': claim_recall_at_k(top_5, query.expects_claim_ids, 5),
            'recall_10': claim_recall_at_k(top_10, query.expects_claim_ids, 10),
            # None, not 0.0, when the query had no canonical to look for: the
            # question was never asked, and a zero would drag the rate down
            # for a non-observation.
            'canonical_in_5': canonical_in_5,
            'canonical_rank': canonical_rank,
            # Same None-not-0.0 discipline, for the same reason: a query with
            # no canonical was never asked either question.
            'stored_canonical_in_5': stored_canonical_in_5,
            'stored_canonical_rank': stored_canonical_rank,
            'has_canonical': canonical_id is not None,
            'tokens': float(tokens_returned(top_10, 10, estimator)['tokens']),
        })

    candidate_present: list[float | None] = []
    guard_matched: list[float | None] = []
    #: The highest score the threshold replay ever SAW.  Carried because a
    #: guard_matched_rate of 0.00 is otherwise indistinguishable from score
    #: plumbing that never delivered a score: "the best candidate scored 0.71
    #: against a 0.92 threshold" is a shape finding, "the best candidate
    #: scored 0.00" is a bug report.
    max_observed_score: float | None = None
    #: Probes production's guard would actually have RUN on.  The pre-check
    #: fires only on explicit ``procedural_knowledge`` writes, and
    #: ``find_near_duplicate_memory`` filters its candidates to that category
    #: unconditionally, so a probe of any other category can never produce
    #: ``guard_matched`` under ANY storage shape — for a reason that has
    #: nothing to do with storage shape.  Counted and disclosed rather than
    #: dropped: dropping would hide that the guard's remit is narrower than
    #: the corpus, and NOT disclosing would put an identical, invisible
    #: ceiling on every arm's ``guard_matched_rate``.
    guard_covered_probes = 0
    for cluster_id, _probe in probes:
        if _probe.get('category') == GUARD_COVERED_CATEGORY:
            guard_covered_probes += 1
        probe_hits = fetched['probes'][cluster_id]
        window = _window(probe_hits, GUARD_TOP_K)
        # The SAME slice both times.  `rescore` builds `best_child` by
        # iterating everything it is handed, so passing the full fetched list
        # here lets a child OUTSIDE the replayed top-5 donate its score to a
        # grouped document inside it — booking a `guard_matched` production,
        # whose pre-check runs at limit=5, structurally could not produce.
        # The leak happens before `guard_adequacy`'s defensive window, so
        # that window cannot catch it.
        rescored = rescore(window, probe_hits[:GUARD_TOP_K])
        for scored in rescored[:GUARD_TOP_K]:
            if max_observed_score is None or scored.relevance_score > max_observed_score:
                max_observed_score = scored.relevance_score
        verdict = guard_adequacy(
            rescored,
            seeded.siblings_by_cluster.get(cluster_id, set()),
            guard_threshold,
        )
        candidate_present.append(1.0 if verdict['candidate_present'] else 0.0)
        guard_matched.append(1.0 if verdict['guard_matched'] else 0.0)

    pooled = _aggregate_queries(rows, limit)
    by_query_kind = {
        kind: _aggregate_queries([r for r in rows if r['kind'] == kind], limit)
        for kind in QUERY_KINDS
    }
    by_query_kind[HELD_OUT_SUBSET] = _aggregate_queries(
        [r for r in rows if r['held_out']], limit,
    )

    return {
        'pin': {
            'enabled': pin,
            'window_changed_rate': (
                _rate(windows_changed, windows_compared) if pin else None
            ),
        },
        'claim_recall': pooled['claim_recall'],
        'discoverability': pooled['discoverability'],
        # The same two metrics, split the way the fixture and eval-design §5
        # E2 both say they differ.  Additive: the pooled blocks above are
        # unchanged, so the headline decision table still reads as before.
        # `held_out` is a SUBSET of `topic_phrasing`, not a third kind — it
        # is the phrasings the E1 registry was not derived from, and so the
        # only ones that measure generalisation.
        'by_query_kind': by_query_kind,
        'tokens_per_query': {
            'mean': _mean([row['tokens'] for row in rows]),
            'estimator': estimator[0],
            'window': 10,
        },
        'guard_adequacy': {
            'candidate_present_rate': _mean(candidate_present),
            'guard_matched_rate': _mean(guard_matched),
            # Carried per-arm, not only in the protocol block: a reader who
            # copies one row out of the table must not lose the flag that
            # says the last column is a threshold replay.
            'threshold_replay': True,
            'threshold': guard_threshold,
            'max_observed_score': max_observed_score,
            'probes': len(probes),
            # The rate's denominator is `probes`, but only these can ever be
            # nonzero — see the accumulator above.  Reported so the ceiling
            # is visible in the artifact gate eta reads instead of being an
            # identical, undisclosed haircut on all six arms.
            'guard_covered_probes': guard_covered_probes,
            'guard_covered_category': GUARD_COVERED_CATEGORY,
        },
    }


async def run_arm(
    backend: Any,
    seeded: SeededArm,
    *,
    queries: list[Query],
    probes: list[tuple[str, dict[str, Any]]],
    limit: int,
    estimator: tuple[str, Any],
    guard_threshold: float,
    fetched: dict[str, dict[str, list[ScoredHit]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """One shape's two rows of the decision table, off ONE set of fetches.

    *fetched* lets the caller supply the ranked lists instead of paying for
    them — either because it already fetched them in order to dump the cache,
    or because it replayed them from one.  ``measure_arm`` cannot tell the
    difference, which is exactly the property that makes the cache sound: a
    replayed row is computed by the same code over the same rankings.

    When it is ``None`` the arm fetches for itself, so every existing caller
    behaves exactly as before.  *backend* is then required; on the replay path
    it is unused and may be ``None``.
    """
    if fetched is None:
        fetched = await fetch_arm(backend, seeded, queries, probes, limit=limit)
    return {
        seeded.shape: measure_arm(
            seeded, fetched, pin=False, queries=queries, probes=probes,
            estimator=estimator, guard_threshold=guard_threshold, limit=limit,
        ),
        f'{seeded.shape}+pin': measure_arm(
            seeded, fetched, pin=True, queries=queries, probes=probes,
            estimator=estimator, guard_threshold=guard_threshold, limit=limit,
        ),
    }


# ---------------------------------------------------------------------------
# The +1-re-emission regrowth probe (task 4012)
# ---------------------------------------------------------------------------
#
# esc-3200-3 was a SPLIT ratification: Option C's WRITE shape was ratified,
# no read transform was, and the read choice was delegated to 4004 (which
# selected the promoting topic pin).  Neither decision answered what ONE
# organic re-emission of an already-canonical claim costs retrieval under
# that ratified shape, or whether the selected transform absorbs it.  This
# section measures it: the ratified write shape held fixed, the READ arm
# varied three ways, over two injection modes.

#: The probe is scoped to the RATIFIED write shape, and varies only the read
#: arm.  Probing all six ``ARM_VARIANTS`` would re-open a settled question
#: and triple the live cost; probing only ``c_peers`` and ``c_peers+pin``
#: would omit the transform 4004 actually selected.
REGROWTH_SHAPE = 'c_peers'


def regrowth_pass_key(mode: str) -> str:
    """The cache slot and collection name for one injected pass.

    Named a "pass key" rather than a shape because it is NOT one: the
    injected corpus is still the ratified ``c_peers`` write shape, and the
    ``SeededArm`` built over it keeps ``shape == REGROWTH_SHAPE`` so
    :func:`read_path` behaves identically.  What this key distinguishes is
    which CORPUS was seeded, which is exactly what the fetch cache and the
    Qdrant collection have to tell apart.

    Underscores only, already lowercase.  :func:`arm_project_id`
    interpolates this into a project id that ``Scope`` canonicalizes
    (lowercased, ``-`` to ``_``), so a key that canonicalized differently
    would have mem0 write to one collection name while the teardown swept
    another — a collection leaked on every run, under the reapable prefix
    but never actually reaped.

    Carries the mode in the name so a leaked collection is attributable in
    the reaper's output rather than merely present.
    """
    if mode not in REGROWTH_MODES:
        raise ValueError(
            f'unknown regrowth mode {mode!r} — the probe measures exactly '
            f'{REGROWTH_MODES} (REGROWTH_MODES)'
        )
    return f'{REGROWTH_SHAPE}_regrowth_{mode}'


#: The three read arms, pinned by EQUALITY like ``ARM_SHAPES``.
REGROWTH_READ_ARMS: tuple[str, ...] = ('flat', 'additive_pin', 'promoting_pin')

#: ``arm -> (pin, promote)``.  ONE definition, so the fan-out, the delta
#: table and the renderer cannot disagree about what an arm name means.
_REGROWTH_ARM_FLAGS: dict[str, tuple[bool, bool]] = {
    'flat': (False, False),
    'additive_pin': (True, False),
    'promoting_pin': (True, True),
}

#: The ``(block, key)`` pairs the delta table reports, pinned by EQUALITY for
#: the same reason ``DECISION_TABLE_COLUMNS`` is: a metric quietly dropped
#: from a delta table is a metric quietly dropped from the decision.
#:
#: The ``stored_`` trio is the SCORED discoverability — ``topic_discoverability``
#: over the RAW hits, before any ``read_path`` — which is the "TRUE canonical
#: id, never an aliased group record id" scoring 3560 landed and the task
#: mandates.
#:
#: The transform-CREDITED ``canonical_in_top_5_rate`` travels BESIDE it and
#: never alone.  ``apply_promoting_topic_anchor`` injects the canonical into
#: the window, so under the ``promoting_pin`` arm that column is a PLACEMENT
#: property in exactly the way ``apply_grouped_read``'s record-id aliasing
#: was under ``b_grouped`` (E2 report, "How to read this table";
#: read-transform-selection report, disclosure (a)).  Dropping it would hide
#: the transform's actual contribution; printing it alone would repeat the
#: misreading 3560 and 4004 each had to correct after publication.  Both, in
#: adjacent columns, with the semantics stated in the rendered section, is
#: the convention both sibling reports converged on — and it costs nothing,
#: because ``measure_arm`` already returns both.
REGROWTH_METRICS: tuple[tuple[str, str], ...] = (
    ('claim_recall', 'at_5'),
    ('claim_recall', 'at_10'),
    ('discoverability', 'stored_canonical_in_top_5_rate'),
    ('discoverability', 'stored_canonical_median_rank'),
    ('discoverability', 'stored_canonical_found_count'),
    ('discoverability', 'canonical_in_top_5_rate'),
    ('tokens_per_query', 'mean'),
)


def measure_regrowth_arms(
    seeded: SeededArm,
    fetched: dict[str, dict[str, list[ScoredHit]]],
    *,
    queries: list[Query],
    probes: list[tuple[str, dict[str, Any]]],
    estimator: tuple[str, Any],
    guard_threshold: float,
    limit: int,
) -> dict[str, dict[str, Any]]:
    """One full measurement per read arm, over ONE set of rankings.

    ``measure_arm`` is pure over ``(SeededArm, fetched)`` and takes ``pin`` /
    ``promote`` as flags, so all three arms are scored from the SAME
    rankings.  That also makes the baseline free — it is computed from the
    ``c_peers`` arm's own fetches, the very rankings the decision table is
    built from.

    Be exact about the REACH of that, because the artifact publishes it: what
    this buys is that an ARM-vs-ARM comparison is seeding-free, there being
    only one seeding behind the three measurements it returns.  It does NOT
    make a baseline-vs-after delta seeding-free — the driver measures
    ``after`` over a separate ``c_peers_regrowth_<mode>`` collection, so that
    delta does span two seedings.  What bounds THAT is a measurement rather
    than an argument: the ``flat`` row of the stamping table compares two
    separately seeded injected collections whose records differ only in a
    metadata key a flat read never consults, so it reads re-seeding and
    nothing else, and :func:`_reseeding_control_phrase` reports what it came
    out as beside the deltas it qualifies.
    """
    return {
        arm: measure_arm(
            seeded, fetched,
            pin=pin, promote=promote,
            queries=queries, probes=probes, estimator=estimator,
            guard_threshold=guard_threshold, limit=limit,
        )
        for arm, (pin, promote) in (
            (arm, _REGROWTH_ARM_FLAGS[arm]) for arm in REGROWTH_READ_ARMS
        )
    }


def _regrowth_metric_keys() -> tuple[str, ...]:
    """``('claim_recall.at_5', …)`` — the flat keys, in metric order.

    ONE spelling of the ``<block>.<key>`` join, shared by the projection,
    the completeness check and the renderer, so the three cannot disagree
    about what a metric is called or what order the table reports it in.
    """
    return tuple(f'{block}.{key}' for block, key in REGROWTH_METRICS)


def _check_regrowth_cost_signs() -> None:
    """``_REGROWTH_COST_SIGN`` covers exactly ``REGROWTH_METRICS``, or raise.

    Called at import (immediately below), so a metric added to
    ``REGROWTH_METRICS`` without a declared cost direction fails LOUDLY
    instead of being silently defaulted to "higher is better" — which is
    false of ``tokens_per_query.mean`` and ``stored_canonical_median_rank``,
    the two columns a default would get backwards.

    A raise rather than an ``assert`` so ``python -O`` cannot strip the one
    check standing between a new metric and a wrong directional claim.
    """
    declared = set(_REGROWTH_COST_SIGN)
    expected = set(_regrowth_metric_keys())
    if declared != expected:
        raise RuntimeError(
            f'_REGROWTH_COST_SIGN does not cover REGROWTH_METRICS: '
            f'undeclared {sorted(expected - declared)}, '
            f'unknown {sorted(declared - expected)}. Every regrowth metric '
            f'must declare which SIGN of its delta is the cost; the signal '
            f'sentence reads it to choose between "costs" and "gains".'
        )


_check_regrowth_cost_signs()


def _pluck_regrowth_metrics(measurement: dict[str, Any]) -> dict[str, Any]:
    """Project ``REGROWTH_METRICS`` to a flat ``{'<block>.<key>': value}``.

    One shape for the delta arithmetic and the renderer both, so a metric can
    only be added or removed in ``REGROWTH_METRICS``.
    """
    return {
        metric: measurement[block][key]
        for metric, (block, key) in zip(
            _regrowth_metric_keys(), REGROWTH_METRICS, strict=True,
        )
    }


def _plucked_regrowth_arms(
    measurements: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """:func:`measure_regrowth_arms`' full blocks, projected per arm.

    The seam between the two halves of the probe: the fan-out returns whole
    ``measure_arm`` blocks (which is what makes each arm's numbers auditable
    against the arm tables), while the delta arithmetic is defined over the
    flat ``REGROWTH_METRICS`` projection.  Both drivers go through here, so
    a live block and a replayed one cannot be projected differently.
    """
    return {
        arm: _pluck_regrowth_metrics(measurement)
        for arm, measurement in measurements.items()
    }


def _delta(after: Any, baseline: Any) -> float | None:
    """``after - baseline``, or ``None`` if EITHER side was never measured.

    The ``_NO_MEASUREMENT`` discipline one layer down: a delta table that
    printed an unmeasured pair as ``0.0`` would say the injection changed
    nothing, which is a finding rather than an absence.
    """
    if after is None or baseline is None:
        return None
    return float(after) - float(baseline)


def _check_same_arms(left: dict[str, Any], right: dict[str, Any], *, what: str) -> None:
    if set(left) != set(right):
        missing = sorted(set(left) ^ set(right))
        raise MeasurementError(
            f'{what}: the two sides disagree on which read arms were '
            f'measured — {missing!r} is present on only one. Silently '
            f'dropping it would delete a read arm from the decision table.'
        )


def regrowth_deltas(
    baseline: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """``after - baseline`` per read arm, per metric."""
    _check_same_arms(baseline, after, what='regrowth deltas')
    deltas: dict[str, dict[str, float | None]] = {}
    for arm in REGROWTH_READ_ARMS:
        if arm not in baseline:
            raise MeasurementError(
                f'regrowth deltas: read arm {arm!r} was never measured'
            )
        if set(baseline[arm]) != set(after[arm]):
            missing = sorted(set(baseline[arm]) ^ set(after[arm]))
            raise MeasurementError(
                f'regrowth deltas: read arm {arm!r} disagrees on metrics — '
                f'{missing!r} is present on only one side'
            )
        deltas[arm] = {
            metric: _delta(after[arm][metric], baseline[arm][metric])
            for metric in baseline[arm]
        }
    return deltas


def regrowth_stamping_value(
    deltas: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, float | None]]:
    """What topic-stamping coverage buys against regrowth, per arm.

    ``stamped delta - unstamped delta``.  This is the number task 4006's
    stamping campaign is owed: the unstamped delta is what one re-emission
    costs today, the stamped delta is what it would cost if the write path
    stamped every re-emission, and the difference is the campaign's value.
    """
    for mode in REGROWTH_MODES:
        if mode not in deltas:
            raise MeasurementError(
                f'regrowth stamping value: mode {mode!r} was never measured — '
                f'the value is stamped-minus-unstamped and cannot be formed '
                f'from one mode alone'
            )
    unstamped, stamped = deltas['unstamped'], deltas['stamped']
    _check_same_arms(unstamped, stamped, what='regrowth stamping value')
    return {
        arm: {
            metric: _delta(stamped[arm][metric], unstamped[arm][metric])
            for metric in unstamped[arm]
        }
        for arm in REGROWTH_READ_ARMS
    }


def build_regrowth_block(
    *,
    baseline: dict[str, dict[str, Any]],
    after_by_mode: dict[str, dict[str, dict[str, Any]]],
    injections: list[RegrowthInjection],
    fixture_path: str | Path,
) -> dict[str, Any]:
    """The report's ``regrowth`` block, descriptors and all four tables."""
    deltas = {
        mode: regrowth_deltas(baseline, after_by_mode[mode])
        for mode in REGROWTH_MODES
    }
    per_topic: dict[str, int] = {}
    for injection in injections:
        per_topic[injection.topic] = per_topic.get(injection.topic, 0) + 1
    counts = set(per_topic.values())
    return {
        'shape': REGROWTH_SHAPE,
        'read_arms': list(REGROWTH_READ_ARMS),
        'modes': list(REGROWTH_MODES),
        'topics_injected': len(per_topic),
        # A checkable property of the artifact, not a claim in its prose:
        # the "+1" in the probe's name IS the independent variable.
        #
        # DERIVED from the slab this block was built over, never the literal
        # `1`: a hard-coded 1 beside `topics_injected: 0` is a descriptor
        # that contradicts the table it heads.  `None` — explicit, the same
        # convention as `protocol['replayed_from']` — when there is nothing
        # to describe, or when the slab is not uniform;
        # `cross_validate_regrowth_injections` forbids the non-uniform case,
        # so reaching it means a caller skipped validation and the honest
        # answer is "no single number describes this".
        'injections_per_topic': counts.pop() if len(counts) == 1 else None,
        'injection_fixture': _repo_relative(fixture_path),
        'baseline': baseline,
        'after': {mode: after_by_mode[mode] for mode in REGROWTH_MODES},
        'deltas': deltas,
        'stamping_value': regrowth_stamping_value(deltas),
    }


def _subset(
    clusters: dict[str, CalibrationCluster],
    topics: dict[str, RegistryTopic],
    claims: list[ArmClaim],
    queries: list[Query],
    cluster_limit: int | None,
) -> tuple[dict[str, CalibrationCluster], dict[str, RegistryTopic],
           list[ArmClaim], list[Query]]:
    """The first *cluster_limit* clusters by id, with their claims and queries.

    Cluster-aligned so the subset stays internally consistent (a claim whose
    cluster went missing would fail cross-validation).  Deterministic by
    sorted id so a smoke run is reproducible.
    """
    if cluster_limit is None:
        return clusters, topics, claims, queries
    keep = sorted(clusters)[:cluster_limit]
    kept = set(keep)
    return (
        {cluster_id: clusters[cluster_id] for cluster_id in keep},
        {cluster_id: topics[cluster_id] for cluster_id in keep if cluster_id in topics},
        [claim for claim in claims if claim.cluster_id in kept],
        [query for query in queries if query.cluster_id in kept],
    )


async def run_bake_off(
    *,
    config: Any = None,
    alpha_fixture: str | Path = DEFAULT_ALPHA_FIXTURE_PATH,
    registry: str | Path = DEFAULT_REGISTRY_PATH,
    arm_claims: str | Path = DEFAULT_ARM_CLAIMS_PATH,
    query_set: str | Path = DEFAULT_QUERY_SET_PATH,
    distractor_slab: str | Path = DEFAULT_DISTRACTOR_SLAB_PATH,
    cluster_limit: int | None = None,
    distractor_limit: int | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
    seed_concurrency: int = SEED_CONCURRENCY,
    project_suffix: str | None = None,
    dump_fetches_to: str | Path | None = None,
    replay_fetches_from: str | Path | None = None,
    regrowth: bool = True,
) -> dict[str, Any]:
    """Seed, query, measure, tear down — and return the validated report.

    Fixtures are loaded and cross-validated FIRST, before a collection
    exists: a fixture checked after seeding would leave three live
    collections behind on every typo.  The injection slab is loaded on the
    same road and under the same rule, but only when *regrowth* is set — a
    probe-less run must not fail on a fixture it never reads.

    *regrowth* defaults to ON.  This artifact is gate leaf eta's input and
    esc-3200-3 asked for these deltas by name; a probe that had to be opted
    INTO would go missing again exactly the way it went missing the first
    time, and the artifact would carry no trace of the omission.  When it is
    off the report's ``regrowth`` key is an explicit ``None``.

    Raises :class:`FixtureError` for any fixture problem, and re-raises
    anything a live run throws — but never without tearing down, because the
    failure that leaks is the mid-run one.  The two INJECTED passes seed into
    two more ephemeral collections, and both ``drop_collections`` calls name
    them: a probe that leaked one would leave it live on every run while the
    report it returned looked perfect.
    """
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    clusters = load_calibration_clusters(alpha_fixture)
    topics = load_registry_topics(registry)
    claims = load_arm_claims(arm_claims)
    queries = load_query_set(query_set)
    distractors = load_distractor_slab(distractor_slab)
    cross_validate_fixtures(
        clusters=clusters, topics=topics, claims=claims, queries=queries,
    )
    # Loaded and cross-validated against the FULL claim set, before the
    # subset and before a collection exists — the same rule the five above
    # follow.  Validating only the subset would let a smoke run pass over a
    # slab that is broken outside its two clusters.
    injections: list[RegrowthInjection] = []
    if regrowth:
        injections = load_regrowth_injections(DEFAULT_REGROWTH_INJECTION_PATH)
        cross_validate_regrowth_injections(injections=injections, claims=claims)

    clusters, topics, claims, queries = _subset(
        clusters, topics, claims, queries, cluster_limit,
    )
    # The injections follow their clusters.  A `--clusters N` run that kept
    # the whole slab would inject re-emissions of claims it never seeded,
    # and every one of them would be scored as a miss.
    injections = [i for i in injections if i.cluster_id in clusters]
    # ONE predicate for "this run probes regrowth", and it is the POST-SUBSET
    # injection list — the same value `_replay_bake_off` decides from, so the
    # two paths cannot disagree about the same run.  `regrowth` alone is not
    # it: a `--clusters N` subset that retains no injected topic would make
    # the live path publish `regrowth_probed: true` beside a block describing
    # zero topics while a replay of that run's own cache published
    # `regrowth_probed: false` and a null block.  The descriptor exists so a
    # reader never has to infer coverage from whether a table looks
    # populated; two paths disagreeing about it defeats that outright.
    probe_regrowth = bool(injections)
    if distractor_limit is not None:
        distractors = distractors[:distractor_limit]

    probes: list[tuple[str, dict[str, Any]]] = []
    for cluster_id in sorted(clusters):
        probe = select_probing_write(clusters[cluster_id])
        if probe is not None:
            probes.append((cluster_id, probe))

    estimator = resolve_token_estimator()

    # The REPLAY short-circuit, taken before a single live resource is
    # acquired.  Not "the same driver with the searches skipped": no
    # `MemoryService` is constructed, so no durable write queue is recovered
    # and no config is deep-copied and repointed; no `ephemeral_collections`
    # are created and — the one that could damage a concurrent run — no
    # `drop_collections` is issued.  A replay is a pure computation over a
    # committed file, and anything less than that would make replaying unsafe
    # to do while a live bake-off is running.
    if replay_fetches_from is not None:
        return await _replay_bake_off(
            replay_fetches_from,
            clusters=clusters, topics=topics, claims=claims, queries=queries,
            distractors=distractors, probes=probes, limit=limit,
            estimator=estimator, project_suffix=project_suffix,
            alpha_fixture=alpha_fixture, registry=registry,
            arm_claims=arm_claims, query_set=query_set,
            distractor_slab=distractor_slab, seed_concurrency=seed_concurrency,
            injections=injections,
        )

    collections = ephemeral_collections(suffix=project_suffix)
    if probe_regrowth:
        # Merged into the SAME dict the pre-run sweep and the `finally`
        # teardown both iterate, rather than tracked alongside it: a second
        # collection list is a second thing to forget.
        collections.update(ephemeral_collections(
            shapes=tuple(regrowth_pass_key(mode) for mode in REGROWTH_MODES),
            suffix=project_suffix,
        ))

    config = (config or FusedMemoryConfig()).model_copy(deep=True)
    config.mem0.collection_prefix = ephemeral_collection_prefix()
    # Clearing the pinned key makes mem0's OpenAIEmbedding fall back to the
    # real OPENAI_API_KEY.  A stub constant vector would make every ranking in
    # this report meaningless.  `providers.openai` is optional in the schema
    # and its absence already means "fall back", so a missing provider is not
    # an error here.
    openai_provider = getattr(config.embedder.providers, 'openai', None)
    if openai_provider is not None:
        openai_provider.api_key = None
    qdrant_url = config.mem0.qdrant_url

    # NEVER let this run attach to the shared durable write queue.
    #
    # `MemoryService.initialize()` is not a mem0-only spin-up: it builds a
    # `DurableWriteQueue` on `config.queue.data_dir`, which defaults to a
    # RELATIVE path (`./data/queue`).  Run from the repo root — the natural
    # cwd — that is the very queue DB the live fused-memory server is using.
    # Initialising it would run `_recover_in_flight()`, flipping the other
    # process's in-flight rows back to pending, and start workers that drain
    # real user writes through `memory.mem0` — the backend whose
    # `collection_prefix` was just repointed at `_test_e2_bakeoff_*`.  Those
    # writes would land in this experiment's ephemeral collections, be marked
    # completed in the shared queue, and then be destroyed by the
    # `drop_collections` in the `finally` below: silent data loss on the
    # production memory store, invisible in the artifact and the logs.
    #
    # A per-run temp dir is used rather than dropping to a bare `Mem0Backend`
    # because `resolve_guard_threshold` navigates the SERVICE to find the
    # threshold production would really run at, and hard-coding or defaulting
    # that number is what its docstring exists to forbid.
    queue_dir = tempfile.mkdtemp(prefix='e2-bakeoff-queue-')
    config.queue.data_dir = queue_dir

    # Everything acquired above is released in the `finally` below, so the
    # acquisition sits INSIDE the try: `initialize()` raising — an unreachable
    # Qdrant, a missing key, a durable-queue recovery failure — is precisely
    # the case where the argument above about this queue never outliving the
    # run has to hold, and a teardown that only ran on success leaked the
    # directory and skipped `close()` on a half-built service.
    #
    # `close()` is called unguarded: `MemoryService.close()` routes every
    # sub-close through `_safe_close`, which is time-boxed and logs without
    # re-raising, so it is already safe on a partially-initialised service —
    # wrapping it in a blanket suppress would only hide a real close failure
    # on the happy path.
    memory = MemoryService(config)
    arms: dict[str, Any] = {}
    fetched_by_shape: dict[str, dict[str, dict[str, list[ScoredHit]]]] = {}
    records_by_shape: dict[str, list[ArmRecord]] = {}
    baseline: dict[str, dict[str, Any]] = {}
    after_by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        drop_collections(collections.values(), qdrant_url=qdrant_url)
        await memory.initialize()
        guard_threshold = resolve_guard_threshold(memory)
        base_records: list[ArmRecord] = []
        for shape in ARM_SHAPES:
            records = materialize_arm(shape, clusters, claims, topics, distractors)
            seeded = _index_arm(
                shape,
                arm_project_id(shape, suffix=project_suffix),
                collections[shape],
                records,
                claims,
            )
            await seed_arm(memory.mem0, seeded, concurrency=seed_concurrency)
            # Fetched HERE rather than inside `run_arm` only when the cache is
            # being written, so the unflagged path is byte-for-byte the run it
            # always was.  The rankings handed to `run_arm` are the same
            # objects that get dumped, so the cache cannot describe a fetch
            # that differs from the one measured.
            #
            # The probe additionally captures the RATIFIED arm's fetches,
            # because they ARE its baseline: `measure_arm` is pure over
            # `(SeededArm, fetched)`, so the un-injected side is scored from
            # the very rankings the decision table above is built from — and
            # the three read arms over it are scored from one seeding, which
            # is what makes an arm-vs-arm comparison seeding-free.  The
            # `after` side below is a separate collection, so a
            # baseline-vs-after delta is not; the `flat` stamping row is the
            # noise floor that bounds it (see `_reseeding_control_phrase`).
            capture = dump_fetches_to is not None or (
                probe_regrowth and shape == REGROWTH_SHAPE
            )
            fetched: dict[str, dict[str, list[ScoredHit]]] | None = None
            if capture:
                fetched = await fetch_arm(
                    memory.mem0, seeded, queries, probes, limit=limit,
                )
                fetched_by_shape[shape] = fetched
                records_by_shape[shape] = records
            arms.update(await run_arm(
                memory.mem0, seeded, queries=queries, probes=probes,
                limit=limit, estimator=estimator,
                guard_threshold=guard_threshold, fetched=fetched,
            ))
            if probe_regrowth and shape == REGROWTH_SHAPE:
                base_records = records
                # Read back through `fetched_by_shape` rather than the local
                # `fetched`: `capture` is true whenever this branch is, so the
                # two name the SAME object, but the dict's value type is
                # non-optional.  A broken invariant then surfaces as a
                # KeyError here instead of measuring against a silent `None`.
                baseline = _plucked_regrowth_arms(measure_regrowth_arms(
                    seeded, fetched_by_shape[shape], queries=queries, probes=probes,
                    estimator=estimator, guard_threshold=guard_threshold,
                    limit=limit,
                ))

        # The two injected passes, INSIDE the try so the `finally` below
        # still reaps their collections when one of them raises mid-seed.
        for mode in REGROWTH_MODES if probe_regrowth else ():
            key = regrowth_pass_key(mode)
            injected = regrowth_corpus(
                base_records, injections, claims, clusters, mode=mode,
            )
            # `shape` stays REGROWTH_SHAPE, never the pass key: the pass key
            # names the collection and the cache slot, and `read_path`
            # branches on `seeded.shape`.  Passing the key here would silently
            # take the grouped-read branch out of play for a corpus that is
            # still the ratified flat one.
            seeded_pass = _index_arm(
                REGROWTH_SHAPE,
                arm_project_id(key, suffix=project_suffix),
                collections[key],
                injected,
                claims,
            )
            await seed_arm(memory.mem0, seeded_pass, concurrency=seed_concurrency)
            injected_fetched = await fetch_arm(
                memory.mem0, seeded_pass, queries, probes, limit=limit,
            )
            fetched_by_shape[key] = injected_fetched
            records_by_shape[key] = injected
            after_by_mode[mode] = _plucked_regrowth_arms(measure_regrowth_arms(
                seeded_pass, injected_fetched, queries=queries, probes=probes,
                estimator=estimator, guard_threshold=guard_threshold,
                limit=limit,
            ))
    finally:
        await memory.close()
        drop_collections(collections.values(), qdrant_url=qdrant_url)
        shutil.rmtree(queue_dir, ignore_errors=True)

    # Pure, and computed before the dump so the cache's fixture list and the
    # protocol block's are decided by the same value.
    regrowth_block: dict[str, Any] | None = None
    if probe_regrowth:
        regrowth_block = build_regrowth_block(
            baseline=baseline,
            after_by_mode=after_by_mode,
            injections=injections,
            fixture_path=DEFAULT_REGROWTH_INJECTION_PATH,
        )

    # AFTER the measurement, and after teardown: a cache written mid-run would
    # survive a run that then died before reporting, and would then be
    # replayable as a complete measurement it never was.
    if dump_fetches_to is not None:
        dump_fetches(
            dump_fetches_to, fetched_by_shape,
            provenance=fetch_cache_provenance(
                records_by_shape=records_by_shape,
                # The injection slab joins the digest list only on a probed
                # run.  Additive: `load_fetches` compares fixtures as a
                # path -> sha MAPPING over the CALLER's list, so a wider cache
                # still satisfies `read_transform_selection`'s narrower
                # five-fixture check.
                fixtures=_protocol_fixture_paths([
                    alpha_fixture, registry, arm_claims, query_set,
                    distractor_slab,
                ], regrowth=regrowth_block),
                search_limit=limit,
                guard_threshold=guard_threshold,
                embedder_model=config.embedder.model,
            ),
        )

    return build_report(
        arms=arms,
        audit_recall=audit_recall_over_labeled_fixture(
            load_labeled_fixture(alpha_fixture),
        ),
        regrowth=regrowth_block,
        protocol={
            'blind_authoring': BLIND_AUTHORING_PROTOCOL,
            'fixtures': fixture_provenance(_protocol_fixture_paths([
                alpha_fixture, registry, arm_claims, query_set, distractor_slab,
            ], regrowth=regrowth_block)),
            'token_estimator': estimator[0],
            'guard_threshold': guard_threshold,
            'distractor_slab_size': len(distractors),
            'embedder_model': config.embedder.model,
            'search_limit': limit,
            # The FLOOR.  The live depth is per-probe-per-arm
            # (`guard_fetch_limit`), because a decomposed arm has to over-fetch
            # by the number of records its probe was split into to leave the
            # same GUARD_TOP_K replay window every other arm gets.
            'guard_fetch_limit_floor': GUARD_FETCH_LIMIT,
            'guard_replay_window': GUARD_TOP_K,
            'seed_concurrency': seed_concurrency,
            'clusters_measured': len(clusters),
            'queries_measured': len(queries),
            'guard_probes_measured': len(probes),
            # Same reason `clusters_measured` is here: a reader holding the
            # artifact must not have to infer the probe's coverage from
            # whether a table looks populated.  Read off the BLOCK, in the
            # identical spelling `_replay_bake_off` uses, so the descriptor
            # and the thing it describes can never disagree — and so a
            # replayed report is byte-identical to the live one it replays.
            'regrowth_probed': regrowth_block is not None,
            'regrowth_injections_measured': len(injections),
            'collections': sorted(collections.values()),
            # Explicit `None` rather than an absent key: every measured cell
            # in a replayed report is identical to a live one, so a reader has
            # no other way to tell which they are holding.  A key that is
            # merely absent on the live path would make "not replayed" and
            # "written by a build that predates the cache" the same reading.
            'replayed_from': None,
        },
    )


async def _replay_bake_off(
    cache_path: str | Path,
    *,
    clusters: dict[str, CalibrationCluster],
    topics: dict[str, RegistryTopic],
    claims: list[ArmClaim],
    queries: list[Query],
    distractors: list[Distractor],
    probes: list[tuple[str, dict[str, Any]]],
    limit: int,
    estimator: tuple[str, Any],
    project_suffix: str | None,
    alpha_fixture: str | Path,
    registry: str | Path,
    arm_claims: str | Path,
    query_set: str | Path,
    distractor_slab: str | Path,
    seed_concurrency: int,
    injections: list[RegrowthInjection],
) -> dict[str, Any]:
    """The same report, recomputed from a committed cache. No network at all.

    Every arm is materialized locally — ``materialize_arm`` is pure over the
    committed fixtures and ``_derive_record_id`` is uuid5-deterministic — and
    the cached rankings are joined back onto those records.  ``measure_arm``
    then runs exactly as it does live, which is what makes a replayed row a
    real measurement rather than a transcription of one.

    ``guard_threshold`` and ``embedder_model`` come from the cache's
    provenance because they are live-only values (:3264, :3297); defaulting
    either would feed the threshold replay a number the store never produced.

    The regrowth probe replays on the same terms: both injected corpora are
    rebuilt locally from *injections* and joined to their own cached pass
    keys, so a replayed block is a real measurement rather than a
    transcription of one and must come out byte-identical to the live one.
    An EMPTY *injections* list is how a ``--no-regrowth`` replay says so —
    the two pass keys are then never requested, which is what keeps a cache
    dumped before this probe existed replayable for the six arms.
    """
    provenance = load_fetch_provenance(cache_path)
    for key in ('guard_threshold', 'embedder_model'):
        if provenance.get(key) is None:
            raise FetchCacheError(
                f'fetch cache at {cache_path} does not record {key!r}, which '
                f'is a LIVE-ONLY value the report\'s protocol block needs '
                f'(:3264, :3297). Defaulting it would publish a number this '
                f'run never measured. Re-dump the cache with --dump-fetches.'
            )

    seeded_by_shape = {
        shape: _index_arm(
            shape,
            arm_project_id(shape, suffix=project_suffix),
            # No collection exists on this path, and naming the one a live run
            # WOULD have used would put a fabricated resource into any error
            # message raised from here.
            f'<replay:{shape}>',
            materialize_arm(shape, clusters, claims, topics, distractors),
            claims,
        )
        for shape in ARM_SHAPES
    }
    replayed = load_fetches(
        cache_path, seeded_by_shape,
        expect_query_ids=[query.query_id for query in queries],
        expect_limit=limit,
        # The same five paths, in the same order, that the protocol block
        # stamps below (:4197-4199) and that the live dump records.
        expect_fixtures=[
            alpha_fixture, registry, arm_claims, query_set, distractor_slab,
        ],
    )

    guard_threshold = float(provenance['guard_threshold'])
    arms: dict[str, Any] = {}
    for shape in ARM_SHAPES:
        # Through `run_arm` rather than calling `measure_arm` twice here, so
        # "an arm's two rows" has exactly one definition and a replayed row
        # cannot drift from a live one.  `backend` is None: with `fetched`
        # supplied, `run_arm` never reaches it.
        arms.update(await run_arm(
            None, seeded_by_shape[shape], queries=queries, probes=probes,
            limit=limit, estimator=estimator, guard_threshold=guard_threshold,
            fetched=replayed[shape],
        ))

    regrowth_block: dict[str, Any] | None = None
    if injections:
        seeded_by_pass = {
            regrowth_pass_key(mode): _index_arm(
                # REGROWTH_SHAPE, not the pass key — see the live driver.
                REGROWTH_SHAPE,
                arm_project_id(regrowth_pass_key(mode), suffix=project_suffix),
                f'<replay:{regrowth_pass_key(mode)}>',
                regrowth_corpus(
                    seeded_by_shape[REGROWTH_SHAPE].records,
                    injections, claims, clusters, mode=mode,
                ),
                claims,
            )
            for mode in REGROWTH_MODES
        }
        # A SECOND `load_fetches`, with the injection slab as its own
        # expectation, rather than folding a sixth path into the call above.
        # The five originals must keep verifying on their own: a cache dumped
        # before this probe existed (4004's committed `e2_fetch_cache.json`,
        # which `read_transform_selection` reads) carries no injection digest,
        # and a merged list would make it unloadable for the E2 arms.
        replayed_passes = load_fetches(
            cache_path, seeded_by_pass,
            expect_query_ids=[query.query_id for query in queries],
            expect_limit=limit,
            expect_fixtures=[DEFAULT_REGROWTH_INJECTION_PATH],
        )
        regrowth_block = build_regrowth_block(
            baseline=_plucked_regrowth_arms(measure_regrowth_arms(
                seeded_by_shape[REGROWTH_SHAPE], replayed[REGROWTH_SHAPE],
                queries=queries, probes=probes, estimator=estimator,
                guard_threshold=guard_threshold, limit=limit,
            )),
            after_by_mode={
                mode: _plucked_regrowth_arms(measure_regrowth_arms(
                    seeded_by_pass[regrowth_pass_key(mode)],
                    replayed_passes[regrowth_pass_key(mode)],
                    queries=queries, probes=probes, estimator=estimator,
                    guard_threshold=guard_threshold, limit=limit,
                ))
                for mode in REGROWTH_MODES
            },
            injections=injections,
            fixture_path=DEFAULT_REGROWTH_INJECTION_PATH,
        )

    return build_report(
        arms=arms,
        audit_recall=audit_recall_over_labeled_fixture(
            load_labeled_fixture(alpha_fixture),
        ),
        regrowth=regrowth_block,
        protocol={
            'blind_authoring': BLIND_AUTHORING_PROTOCOL,
            'fixtures': fixture_provenance(_protocol_fixture_paths([
                alpha_fixture, registry, arm_claims, query_set, distractor_slab,
            ], regrowth=regrowth_block)),
            'token_estimator': estimator[0],
            'guard_threshold': guard_threshold,
            'distractor_slab_size': len(distractors),
            'embedder_model': provenance['embedder_model'],
            'search_limit': limit,
            'guard_fetch_limit_floor': GUARD_FETCH_LIMIT,
            'guard_replay_window': GUARD_TOP_K,
            'seed_concurrency': seed_concurrency,
            'clusters_measured': len(clusters),
            'queries_measured': len(queries),
            'guard_probes_measured': len(probes),
            'regrowth_probed': regrowth_block is not None,
            'regrowth_injections_measured': len(injections),
            # EMPTY, not the names a live run would have used: this run
            # created no collection, and printing five it never touched would
            # put an unmeasured resource into the audit trail.
            'collections': [],
            'replayed_from': str(cache_path),
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> Any:
    """The CLI.  Every flag is a knob on the experiment, none on its verdict."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description=(
            'E2 storage-shape bake-off: seed three ephemeral collections, '
            'measure six arm variants, and emit the decision table gate eta '
            'reads.'
        ),
    )
    parser.add_argument('--alpha-fixture', default=str(DEFAULT_ALPHA_FIXTURE_PATH),
                        help='alpha/3130 curator-labeled fixture (JSONL)')
    parser.add_argument('--registry', default=str(DEFAULT_REGISTRY_PATH),
                        help='E1 topic registry (JSON)')
    parser.add_argument('--arm-claims', default=str(DEFAULT_ARM_CLAIMS_PATH),
                        help='blind-authored arm claim decomposition (JSONL)')
    parser.add_argument('--query-set', default=str(DEFAULT_QUERY_SET_PATH),
                        help='blind-authored query set (JSONL)')
    parser.add_argument('--distractor-slab', default=str(DEFAULT_DISTRACTOR_SLAB_PATH),
                        help='shared contamination slab (JSONL)')
    parser.add_argument('--clusters', type=int, default=None,
                        help='measure only the first N clusters (smoke runs)')
    parser.add_argument('--distractors', type=int, default=None,
                        help='cap the contamination slab at N records')
    parser.add_argument('--limit', type=int, default=DEFAULT_SEARCH_LIMIT,
                        help='fetch depth for the claim/cost queries')
    parser.add_argument('--seed-concurrency', type=int, default=SEED_CONCURRENCY,
                        help='concurrent seeding writes in flight')
    parser.add_argument('--project-suffix', default=None,
                        help='override the per-xdist-worker project id suffix')
    parser.add_argument('--json-out', default=str(DEFAULT_REPORT_JSON),
                        help='where to write the machine-readable report')
    parser.add_argument('--md-out', default=str(DEFAULT_REPORT_MD),
                        help='where to write the operator-facing report')
    # Both default to None, so an unflagged run is exactly the run it always
    # was — and `TestBuildParser`, which asserts defaults by attribute rather
    # than flag-set exhaustiveness, keeps passing untouched.
    parser.add_argument('--dump-fetches', default=None,
                        help=('after measuring, write every arm ranking to '
                              'this path so read-side variants can be scored '
                              'later without a reseed'))
    parser.add_argument('--replay-fetches', default=None,
                        help=('measure from a dumped ranking cache instead of '
                              'seeding: no Qdrant, no embedder, no collections'))
    # Default ON, and only turnable OFF.  The artifact is gate leaf eta's
    # input and esc-3200-3 asked for these deltas by name: an opt-IN probe
    # would go missing again exactly the way it went missing the first time,
    # and the artifact would carry no trace of the omission.
    parser.add_argument('--no-regrowth', dest='regrowth', action='store_false',
                        default=True,
                        help=('skip the +1-re-emission regrowth probe: two '
                              'fewer seeded collections and two fewer fetch '
                              'passes, and the report carries an explicit '
                              'null regrowth block'))
    return parser


async def _run(args: Any) -> int:
    """Exit 2 for a fixture problem, with NO artifact written.

    A fixture failure must not overwrite a good committed report with a
    partial one — the artifact is the gate's input, and a silently truncated
    decision table reads exactly like a measured one.

    Exit 3 for a fetch-cache problem, also with NO artifact written, and for
    the same reason: a replay against a cache that does not describe this
    corpus would publish a stale ranking as a fresh measurement, and the
    artifact gives a reader no way to tell.
    """
    if args.dump_fetches is not None and args.replay_fetches is not None:
        # Named, not silently resolved in favour of one. Preferring either
        # would make the run's provenance a coin flip: the artifact does not
        # say which won, so a reader could not tell a measurement from a
        # replay of one.
        print(
            'cannot pass --dump-fetches and --replay-fetches together: '
            '--dump-fetches records the rankings a LIVE run measured, while '
            '--replay-fetches measures from rankings a previous run already '
            'recorded. Doing both would either re-dump what was just replayed '
            '(a no-op that overwrites the cache with itself) or silently pick '
            'one and leave the report unable to say which.',
            file=sys.stderr,
        )
        return 2

    try:
        report = await run_bake_off(
            alpha_fixture=args.alpha_fixture,
            registry=args.registry,
            arm_claims=args.arm_claims,
            query_set=args.query_set,
            distractor_slab=args.distractor_slab,
            cluster_limit=args.clusters,
            distractor_limit=args.distractors,
            limit=args.limit,
            seed_concurrency=args.seed_concurrency,
            project_suffix=args.project_suffix,
            dump_fetches_to=args.dump_fetches,
            replay_fetches_from=args.replay_fetches,
            regrowth=args.regrowth,
        )
    except FixtureError as exc:
        print(f'fixture error: {exc}', file=sys.stderr)
        return 2
    except FetchCacheError as exc:
        print(f'fetch cache error: {exc}', file=sys.stderr)
        return 3

    json_path, md_path = write_artifacts(report, args.json_out, args.md_out)
    print(f'wrote {json_path}')
    print(f'wrote {md_path}')
    return 0


def main(argv: list[str] | None = None) -> int:
    """Injectable argv so the CLI is testable without a subprocess."""
    return asyncio.run(_run(build_parser().parse_args(argv)))


if __name__ == '__main__':
    raise SystemExit(main())
