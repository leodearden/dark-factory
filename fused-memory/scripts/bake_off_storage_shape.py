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

STRUCTURE
---------
A thick **pure core** (loaders, arm materialization, read transforms,
metrics, report rendering) with zero network, fully exercised in the merge
lane, plus a thin **live driver** (``seed_arm`` / ``run_arm`` /
``run_bake_off`` / ``_run``) that is the only part touching Qdrant or an
embedder.
"""
from __future__ import annotations

import json
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
            if kind not in ('topic_phrasing', 'claim'):
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
#: is what β's ``parent_id`` shape rule (``_is_full_uuid``) requires.
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
    amendments: list[ArmRecord],
    sighting_count: int,
) -> str:
    """Canonical body, then amendment digests, then a sighting COUNT.

    Sightings are counted rather than pasted because that is precisely the
    cost claim grouping makes (PRD D4): a grouped read is supposed to be
    cheaper than N peers, and it can only be cheaper if the repetitive
    "seen it again" entries collapse to a number.  Pasting them would make
    the tokens-per-query metric measure a concatenation and quietly hand
    arm (b) a loss it did not earn.
    """
    parts = [canonical.content]
    for amendment in amendments:
        parts.append(f'[amendment] {amendment.content}')
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
    a strawman.  Every claim the collapsed children realized is credited to
    the group, or arm (b) would lose claim-recall precisely *for grouping
    correctly*.

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

        claim_ids = list(canonical.claim_ids)
        for member in members:
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
    promoted to the front.  Appending is the conservative choice: it cannot
    flatter the pin on any rank-based metric, so a measured improvement is
    attributable to the canonical becoming *reachable at all* rather than to
    the transform having hand-placed it high.

    Returns the input list unchanged (same contents, same order) when nothing
    is pinnable — the pin is not a rewrite.

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
        if canonical.metadata.get('canonical') is not True:
            raise ValueError(
                f'the record registered as canonical for topic {topic!r} '
                f'({canonical.record_id!r}) does not carry canonical is True — '
                f'β\'s invalid_canonical_type rule is bool identity, so a truthy '
                f'value is a FATAL violation at the write boundary. Anchoring on '
                f'it would report a discoverability win for a shape production '
                f'cannot store.'
            )
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
        import tiktoken  # noqa: PLC0415

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
    as ENUMS and ``.relevance_score`` as a float.  Dicts would raise; dicts
    of plain strings would be worse — every enum comparison would silently
    fail and the report would state "the guard never fires" for every arm.

    Rank order is preserved: the selector takes the max by score, but the
    report quotes the matched id back against the ranked window, and a
    reordering adapter would make that evidence point at the wrong record.

    A category the enum does not know becomes ``None`` rather than a guess —
    that record is exactly what the selector's defensive filter exists to
    drop, and inventing a category would defeat it.

    ``source_store`` is ``mem0`` for every hit: the arms are seeded into
    mem0/Qdrant and production's pre-check searches ``stores=['mem0']``.
    """
    from fused_memory.models.enums import MemoryCategory, SourceStore  # noqa: PLC0415
    from fused_memory.models.memory import MemoryResult  # noqa: PLC0415

    results = []
    for hit in hits:
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
            relevance_score=hit.relevance_score,
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

    Note for the report: 3 of the 15 measurable clusters have a probe whose
    category is not ``procedural_knowledge``, which production's guard does
    not cover at all (it runs only on explicit procedural_knowledge writes).
    Those rows are measured the same way and flagged there rather than
    dropped here — dropping them would hide that the guard's remit is
    narrower than the corpus.
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
    return float(inspect.signature(detector).parameters['threshold'].default)


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
    'claim_recall': ('at_5', 'at_10'),
    'discoverability': ('canonical_in_top_5_rate',),
    'tokens_per_query': ('mean', 'estimator'),
    # Both halves, always: the rank-based one and the threshold replay.
    'guard_adequacy': ('candidate_present_rate', 'guard_matched_rate',
                       'threshold_replay', 'threshold'),
}

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
    'median canonical rank',
    'tokens/query',
    'guard candidate present',
    'guard matched (replay)',
)

#: Rendered for a measurement that is None.  Deliberately NOT '0.00': the
#: whole point of carrying None through the pipeline is that "no measurement"
#: and "measured zero" are different findings, and a table that prints them
#: identically throws that away at the last step.
_NO_MEASUREMENT = '—'

DEFAULT_REPORT_JSON = _PACKAGE_ROOT.parent / 'plans' / 'e2-storage-shape-bakeoff-report.json'
DEFAULT_REPORT_MD = _PACKAGE_ROOT.parent / 'plans' / 'e2-storage-shape-bakeoff-report.md'

#: Bumped when the JSON's shape changes, so a reader of an old artifact is
#: never silently comparing two different schemas.
REPORT_SCHEMA_VERSION = 1


class IncompleteReportError(RuntimeError):
    """A measurement is missing, so no table is emitted.

    Named rather than a bare ValueError because the failure mode it guards is
    specific: a decision table with a blank cell reads as "measured, and equal
    to its neighbour", and the reader has no way to tell that from "never
    measured".  Refusing to emit is the loud alternative.
    """


def fixture_provenance(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Repo-relative path + last-touching commit for each fixture.

    Git history IS the blind-authoring audit trail: the arm decomposition and
    query set were committed BEFORE any metric function existed in the tree,
    and these SHAs are what lets a reader verify that rather than take it on
    trust.

    Paths are reported repo-relative — an artifact naming somebody's absolute
    home-directory checkout is neither reproducible nor readable by anyone
    else, and it leaks the worktree the run happened in.  (Stated without an
    example literal on purpose: the fixture-path test forbids any absolute
    path in this file's source, docstrings included.)

    ``commit`` is ``None`` for a path git does not track.  Reporting None
    rather than falling back to HEAD is the point: a wrong SHA in an audit
    trail is worse than an absent one, because it looks checkable.
    """
    import subprocess  # noqa: PLC0415

    repo_root = _PACKAGE_ROOT.parent
    provenance: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).resolve()
        try:
            relative = str(path.relative_to(repo_root))
        except ValueError:
            relative = path.name
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


def build_report(
    *,
    arms: dict[str, Any],
    audit_recall: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the E2 decision table + D10 measurement into one artifact.

    Raises :class:`IncompleteReportError` rather than emitting a partial
    table.  A ``None`` VALUE is fine and survives to the artifact — "measured,
    no denominator" is a legitimate result — but an ABSENT key means the run
    broke, and a broken run must not be published as a decision.

    Arms are ordered by :data:`ARM_VARIANTS`, not by the caller's dict order,
    so two runs produce diffable artifacts.
    """
    _check_arms(arms)
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
    }


def _cell(value: Any, *, precision: int = 2) -> str:
    """Format one measurement, keeping "no measurement" distinguishable."""
    if value is None:
        return _NO_MEASUREMENT
    if isinstance(value, float):
        return f'{value:.{precision}f}'
    return str(value)


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
            _cell(measurement['discoverability'].get('median_canonical_rank')),
            _cell(measurement['tokens_per_query']['mean'], precision=1),
            _cell(guard['candidate_present_rate']),
            _cell(guard['guard_matched_rate']),
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
        f"**Blind authoring**: {protocol['blind_authoring']}.  The arm "
        f'decomposition and query set were committed BEFORE any metric '
        f'function existed in this tree; the commits below are the audit '
        f'trail, and the anti-laziness floor is claim-coverage parity (every '
        f'claim id realizable in every arm), deliberately not length parity — '
        f"arm (a)'s long originals versus arm (c)'s short peers differ by "
        f'construction, and that difference IS the tokens/query column.',
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
    """
    json_path, md_path = Path(json_out), Path(md_out)
    _atomic_write_text(json_path, json.dumps(report, indent=2) + '\n')
    _atomic_write_text(md_path, render_markdown(report))
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
