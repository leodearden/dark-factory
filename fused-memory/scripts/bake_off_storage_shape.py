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
