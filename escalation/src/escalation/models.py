"""Data model for escalations.

Consumer-per-level contract (invariant):
  L0  level=0  producer: agent           consumer: steward (interactive review)
  L1  level=1  producer: steward/workflow consumer: escalation-watcher-auto (automated triage)
  L2  level=2  producer: auto-watcher     consumer: human (direct, bypasses auto-watcher)

Escalations are born at L2 when their severity is in BORN_AT_L2_SEVERITIES.
All other escalations start at L0 (level=0) and are promoted by handlers.

L2 cluster fields (default-empty; L0/L1 are unaffected):
  members:    list of member L1 escalation ids forming this cluster
  root_cause: dedup key for pending-L2 lookup, matched on its CANONICAL form
              (`escalation.canonical.canonical_root_cause`; task 3998), so
              near-duplicates differing only in case/whitespace/punctuation fold.
              Stored and displayed VERBATIM — the canonical form is a
              compare-time value and is never persisted.
  options:    proposed resolution options (e.g. ['A: rollback', 'B: fix forward'])

Structured-evidence field (default-empty; task 2558):
  evidence:   list of EvidenceEntry {observation, measured_at, ref} raw
              OBSERVATIONS backing the escalation (not causal diagnoses)

Incoming-framing amendments (default-empty; task 3997):
  amendments: append-only list of Amendment entries -- the framing
              (root_cause/summary/detail/options) a fold carried in, preserved
              instead of discarded.  NEVER overwrites the record's own fields
              above: the original human-facing framing stays immutable.
  amendments_truncated:
              count of oldest entries shed to the `queue._MAX_AMENDMENTS` cap;
              a durable structured fact so the loss is assertable from the
              record rather than only by log-scrape.
  amendments_chars_elided:
              the BYTE-side counterpart: count of characters dropped by the
              per-field caps (`queue._MAX_AMENDMENT_LINE_CHARS` /
              `_MAX_AMENDMENT_DETAIL_CHARS` / `_MAX_AMENDMENT_OPTIONS`).  An
              entry count alone bounds nothing when one field is unbounded free
              text, so the two counters together are what make the list's size
              envelope assertable from the record.
  SOLE WRITER: `queue.add_members_to_l2`.

Over-fold evidence (default-empty; task 3998):
  root_cause_variants:
              the DISTINCT PRE-CANONICAL root_cause spellings that have folded
              into this L2, the record's own spelling seeded first.
              Canonicalising the match makes more promotes fold BY DESIGN, so
              its failure mode is OVER-folding — distinct causes silently merged
              under one canonical key — and this set is that failure's only
              observable signature.  NOT derivable from `amendments`, which
              sheds its oldest at the cap and would therefore under-report
              precisely during the runaway the count exists to catch.
  root_cause_variants_truncated:
              count of oldest spellings shed to the
              `queue._MAX_ROOT_CAUSE_VARIANTS` cap.  The TRUE distinct count is
              `len(root_cause_variants) + root_cause_variants_truncated`, so the
              loss stays assertable from the record rather than only by
              log-scrape.
  SOLE WRITER: `queue.add_members_to_l2` (also the sole trimmer).

Filing-identity field (default-None; task 3533, populated by task 3550):
  filing_claimant_run_id:
              the FILING incarnation's claimant id in
              `shared.task_claimant.compose_claimant_run_id` format;
              None = unknown.  STAMPED BY: `orchestrator.workflow.TaskWorkflow`,
              `orchestrator.harness.Harness`, and the
              `escalate_blocker`/`escalate_info` chokepoint in
              `escalation.server`.  For the producers that do NOT stamp it —
              and why each one's records are (or are not) moot — see the
              inventory on `escalation.pins.classify_pins`; do not restate or
              summarise it here, and in particular do not assume the
              non-stamping set is level-1/2-only.  Semantics and the fail-safe rule are stated
              once on `escalation.pins.classify_pins` (normative source:
              spec docs/task-escalation-state-spec.md S6) — do not restate
              them here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TypedDict

logger = logging.getLogger(__name__)


class TrainState(TypedDict):
    """Per-train context embedded in L1/L2 escalations for park-prefix derail triage.

    Shape mirrors the dict returned by TaskWorkflow._build_train_state (PRD § 9.8).
    TypedDict at runtime is a plain dict; existing from_dict / to_dict / asdict()
    paths are unaffected — round-trip fidelity is unchanged.
    """

    id: str               # train identifier
    order: int            # this task's position in the train (0-based)
    parked_members: list[str]  # sibling task_ids at merge-deferred, excluding self
    failing_member: str   # this task's id (the one that triggered BLOCKED)


class IndexHealthState(TypedDict):
    """FalkorDB index-provisioning drift context for `recon_missing_index` (task 3709).

    Shape mirrors the record produced by
    fused_memory.reconciliation.index_health.summarize_index_health(), projected
    by the drift detector (PRD δ, D8 + D11).

    The drifted graph and the missing specs are carried as FIRST-CLASS FIELDS so
    no consumer parses `summary`/`detail` prose to recover a fact the emitter
    already had in a variable (INV-2).

    Index specs are stored as JSON LISTS, never tuples: a tuple deserialises
    back as a list, so storing tuples would make the on-disk record fail to
    round-trip identically and break equality for any payload comparison.

    TypedDict at runtime is a plain dict; existing from_dict / to_dict /
    asdict() paths are unaffected — round-trip fidelity is unchanged.
    """

    group_id: str                  # the graph whose index state drifted
    missing: list[list[str]]       # expected-but-absent specs, sorted
    unexpected: list[list[str]]    # present-but-unexpected specs (reported, never acted on)
    expected_total: int            # size of the expected set this was diffed against


class EvidenceEntry(TypedDict):
    """A single structured raw-OBSERVATION backing an escalation (task 2558).

    Each entry records what was measured, when, and against which ref — NOT a
    causal diagnosis.  Filers put observations here (and in summary/detail) and
    keep any causal claim on a clearly-marked `Hypothesis:` line, so a reviewer
    never reads an unverified guess as fact (survey §1.7 precedent: a
    "last-green" rewind recommendation named a commit that ALSO failed).

    Shape mirrors the TrainState TypedDict.  TypedDict at runtime is a plain
    dict; existing from_dict / to_dict / asdict() paths are unaffected —
    round-trip fidelity is unchanged.  The server stores and returns it
    verbatim (no shape validation), so partial evidence is accepted.
    """

    observation: str   # what was measured, e.g. "main red at abc123 (exit 134)"
    measured_at: str   # when/where measured, e.g. "HEAD=abc123 @ 2026-07-14T00:00:00+00:00"
    ref: str           # ref/context this was measured against, e.g. "rerun#2" or "HEAD=abc123"


class Amendment(TypedDict):
    """Framing a fold carried IN, appended rather than discarded (task 3997).

    When `promote_to_l2` folds a new promote into an existing pending L2, the
    incoming root_cause/evidence/options/summary used to be dropped on the floor
    (measured: 336,875 characters lost).  Each such fold now appends one of these
    to `Escalation.amendments`.  The record's OWN framing is never overwritten —
    both the original and every incoming reframing survive, which is the point.

    `detail` holds `promote_to_l2`'s ``evidence`` ARGUMENT — the same argument the
    create path writes into the record's own `detail` — so a reader can diff an
    amendment against the record like-for-like.  It is deliberately NOT called
    `evidence`: `Escalation.evidence` is the semantically different structured
    EvidenceEntry list (task 2558), and two different things called `evidence` on
    one record would be a legibility trap.

    `root_cause` is the PRE-canonical incoming value, which is what makes an
    over-folding root-cause canonicaliser detectable after the fact.

    `timestamp` is stamped by the queue at write time, never author-supplied —
    the write chokepoint owns its clock (mirroring `stamp_triage`/`triaged_at`),
    so a caller cannot backdate an amendment.

    Every free-text field is stored ELIDED to the queue's per-field caps, with
    an in-band marker naming what was dropped (`queue._build_amendment`), and
    `options` is capped in length.  A field ending in that marker is the HEAD of
    what was submitted, not all of it; the record-level `amendments_chars_elided`
    holds the running character total.

    Shape mirrors the EvidenceEntry / TrainState TypedDicts.  TypedDict at
    runtime is a plain dict; existing from_dict / to_dict / asdict() paths are
    unaffected — round-trip fidelity is unchanged.  Stored verbatim with no
    shape validation, so a partial entry is accepted rather than rejected.
    """

    timestamp: str        # ISO, stamped by queue.add_members_to_l2 at write time
    agent_role: str       # the session that submitted this framing
    root_cause: str       # the PRE-canonical incoming root cause
    summary: str          # incoming one-line hypothesis
    detail: str           # incoming `evidence` argument (see docstring on the name)
    options: list[str]    # incoming proposed resolution options


# Severities that cause an escalation to be created directly at L2,
# bypassing the auto-watcher and routing straight to a human.
BORN_AT_L2_SEVERITIES: frozenset[str] = frozenset({'critical', 'urgent'})

# All valid severity values accepted by the MCP tools (escalate_blocker /
# escalate_info).  Used to validate caller input and return a clear error
# rather than silently misrouting escalations.
KNOWN_SEVERITIES: frozenset[str] = frozenset({'info', 'blocking'}) | BORN_AT_L2_SEVERITIES

# Urgency ordering over KNOWN_SEVERITIES, used by every severity FOLD in the
# system (dedupe-child promotion, L2 member inheritance, the L2 update-path
# floor).  Alphabetical comparison is wrong ('blocking' < 'info'), so the rank
# is explicit.
#
# TOTAL over KNOWN_SEVERITIES by contract: adding a severity to the vocabulary
# without adding it here is a bug, not a silent rank-0 — TestSeverityRank
# (escalation/tests/test_models.py) fails on the gap.  The BORN_AT_L2
# severities (critical/urgent) outrank 'blocking', which is what lets a
# born-at-L2 child promote a lower-severity parent.
#
# The ordering mirrors the repo's only other canonical escalation-severity
# ranking — cockpit/src/cockpit/priority.py's _ESCALATION_SEVERITIES and its
# severity_weights (urgent 6.0 > critical 5.0 > blocking 2.5 > info 0.25) — so
# the two are traceably one decision rather than two guesses that can drift.
SEVERITY_RANK: dict[str, int] = {'info': 0, 'blocking': 1, 'critical': 2, 'urgent': 3}


def max_severity(a: str, b: str) -> str:
    """Return the higher-urgency severity string between *a* and *b*.

    Fail-soft on unknown input: an unrecognised severity is WARNed about and
    treated as rank 0 (info-level) rather than raising, so malformed input can
    never crash a fold nor cause an unexpected upward promotion.  The ``>=``
    tie-break on *a* makes equal-rank comparisons — including unknown-vs-unknown
    — deterministic rather than arbitrary.
    """
    for val in (a, b):
        if val not in SEVERITY_RANK:
            logger.warning(
                'max_severity: unrecognised severity %r — treating as info-level '
                '(rank 0). Known values: %s',
                val,
                ', '.join(SEVERITY_RANK),
            )
    return a if SEVERITY_RANK.get(a, 0) >= SEVERITY_RANK.get(b, 0) else b

# Ladder levels an AGENT-side MCP filing (escalate_blocker) may be born at.
# 0 = agent→steward, 1 = steward re-escalation→escalation-watcher-auto.
# Level 2 is deliberately excluded: agents must not self-mint an L2 that
# bypasses the auto-watcher and pages a human.  The legitimate routes to L2
# are a born-at-L2 *severity* filed by a harness sentinel role (see
# BORN_AT_L2_SEVERITIES and the agent-role downgrade in server.py) or the
# promote_to_l2 handler tool.  This mirrors the existing policy that
# downgrades an agent-filed critical/urgent to 'blocking'.
AGENT_FILABLE_LEVELS: frozenset[int] = frozenset({0, 1})

# Legal values for Escalation.resolution_class (escalation-lifecycle-dashboard-prd.md
# Seam 1).  Used to validate the resolve/dismiss chokepoint's optional
# resolution_class param and return a clear error naming the legal
# values rather than silently accepting an arbitrary string.
#
# 'moot-terminal-subject' (task 2724) is the DISTINCT, non-benign stamp the
# escalation-revalidation sweep writes when it auto-closes an allowlisted L2
# whose subject task went terminal (done/cancelled).  It is deliberately
# neither 'benign' nor 'actionable' so swept records stay auditable — the
# dashboard's effective_benign/_origin_block/_workflow_block count it as
# classified+stamped but bucket it into neither, preserving the evidence the
# old 'benign' mis-label destroyed.
#
# 'stale-strand' (task 3172) is the DISTINCT, non-benign stamp
# EscalationQueue.dismiss_all_pending writes for a level-0 escalation that had
# already been pending for >= strand_age_secs when the orchestrator restarted
# and swept it.  Like 'moot-terminal-subject' it is deliberately neither
# 'benign' nor 'actionable' so a strand stays auditable after the sweep that
# closed it.  It EXTENDS this vocabulary rather than forking a second
# classification scheme (task 3172: coordinate, do not fork).
#
# The origin incident is what this member exists to make legible: esc-5189-7
# had been pending 20h58m with a workflow parked on it, while esc-5685-1 had
# been pending ~90s — a genuine strand and an ordinary restart artifact.  The
# old single 'benign' stamp made the two records indistinguishable, so the
# only durable trace of a 20h strand read as routine restart noise.
RESOLUTION_CLASSES: frozenset[str] = frozenset(
    {'benign', 'actionable', 'moot-terminal-subject', 'stale-strand'}
)


@dataclass
class Escalation:
    id: str  # "esc-{task_id}-{seq}"
    task_id: str
    agent_role: str
    severity: str  # "blocking" | "info" | "critical" | "urgent"
    category: str  # scope_violation, design_concern, cleanup_needed,
    # dependency_discovered, risk_identified, infra_issue,
    # reconciliation_stale_human_operator, reconciliation_stale_gate_backlog,
    # recon_missing_index
    # REFACTOR TRIGGER (task 3709): this vocabulary is prose, not a checked
    # contract — nothing rejects a typo'd category at submit time, and the
    # dedup correctness of every categorized detector depends on filer and
    # reader spelling it identically.  The NEXT category addition promotes
    # this comment to an enum (or a submit-time lint) instead of growing
    # another line.
    summary: str  # one-line
    detail: str = ''  # full context
    suggested_action: str = ''  # expand_scope, create_followup_task, abort_task, etc.
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = 'pending'  # pending, resolved, dismissed
    resolution: str | None = None  # filled by handler
    worktree: str | None = None  # path to worktree
    workflow_state: str | None = None  # what state the agent was in
    level: int = 0  # 0 = L0 agent→steward, 1 = L1 steward/workflow→escalation-watcher-auto, 2 = L2 escalation-watcher-auto→human
    resolved_at: str | None = None
    resolved_by: str | None = None  # "steward" | "interactive" | "auto-dismissed"
    resolution_turns: int | None = None  # conversation turns to resolve
    dedupe_count: int = 0  # number of duplicate submissions folded into this parent
    dedupe_children: list[str] = field(default_factory=list)  # ids of folded duplicates
    dedupe_fingerprint: str | None = None  # content fingerprint for A7a/A7b recon dedup
    # L2 cluster fields — empty defaults keep L0/L1 escalations bit-identical on disk.
    # Old JSON files (pre-L2) deserialise correctly via from_dict's __dataclass_fields__
    # filter: absent keys map to the dataclass defaults without any migration required.
    members: list[str] = field(default_factory=list)  # member L1 escalation ids (cluster composition)
    root_cause: str = ''  # root-cause hypothesis; dedup key for pending-L2 lookup, matched on its CANONICAL form (task 3998) but stored verbatim
    options: list[str] = field(default_factory=list)  # proposed resolution options ['A: ...', 'B: ...']
    # Structured raw-OBSERVATION entries backing this escalation (task 2558).
    # Each is a {observation, measured_at, ref} dict recording a measurement
    # (HEAD SHA, rerun result, raw exit code) — never a causal diagnosis.
    # Empty default keeps existing free-form escalations bit-identical on disk;
    # legacy JSON without the key deserialises to [] via the from_dict
    # __dataclass_fields__ filter — zero migration, same pattern as members /
    # train_state above.  Stored/returned verbatim (no shape validation).
    evidence: list[EvidenceEntry] = field(default_factory=list)
    # PRD § 9.8 — per-train context for park-prefix derail L2 escalations.
    # None for all non-train escalations; legacy JSON (pre-field) deserialises to None
    # via the from_dict __dataclass_fields__ filter — no migration required.
    train_state: TrainState | None = None
    # PRD δ (task 3709) — FalkorDB index-provisioning drift context for
    # `recon_missing_index` escalations filed by the reconciliation harness's
    # index drift detector.  None for every other escalation kind; legacy JSON
    # (pre-field) deserialises to None via the from_dict __dataclass_fields__
    # filter — zero migration, same pattern as members / train_state above.
    # to_dict's asdict() serialises it for free, and submit / submit_resolved /
    # _atomic_write / resolve / park / stamp_triage are field-agnostic
    # passthroughs that need no change.
    index_health: IndexHealthState | None = None
    # C1 action chosen at resolve_issue time (resume/restart/park/abandon/close_only).
    # None for records resolved before α1 or for L2 cascade members (β derives theirs
    # from the parent via resolved_by='l2-cascade:<id>' attribution).
    resolution_action: str | None = None
    # Benign/actionable classification stamp (escalation-lifecycle-dashboard-prd.md
    # Seam 1).  Written ONLY at a terminal-write chokepoint — queue.resolve() or
    # queue.submit_resolved() — never author-supplied at filing time.  None means
    # unstamped — readers fall back to the effective_benign() proxy (see
    # escalation.classify).
    resolution_class: str | None = None
    # Triage-ack annotation (NOT a resolution) — lets escalation-watcher-auto
    # rotations skip re-deriving the disposition of a still-pending L1/L2 item
    # every rotation. triaged_at/triaged_by/triage_note are stamped by
    # queue.stamp_triage(); updated_at is a separate "changed since I triaged
    # it" signal bumped elsewhere (e.g. add_members_to_l2 on real append).
    # All four are optional with zero-migration defaults: legacy JSON files
    # without these keys deserialise correctly via the from_dict
    # __dataclass_fields__ filter below — the same pattern used for
    # train_state/resolution_action/resolution_class above.
    triaged_at: str | None = None
    triaged_by: str | None = None  # server-attributed from X-Escalation-Identity when present
    triage_note: str = ''  # freshness contract: verified predicate + probe (see watcher SKILLs)
    updated_at: str | None = None  # last-substantive-change marker; None means never bumped
    # Steward's structured scope-expansion grant (task 2505) — file-level,
    # project-relative paths — consumed by the orchestrator resume path to
    # widen plan.files/metadata.files/locks. Distinct from the free-text
    # `resolution` rationale string, which stays human-readable prose.
    # Empty-list default keeps non-grant escalations bit-identical on disk;
    # legacy JSON without this key deserialises to [] via the from_dict
    # __dataclass_fields__ filter below — no migration required.
    granted_files: list[str] = field(default_factory=list)
    # The FILING incarnation's claimant identity (task 3533) — semantics are
    # documented once on `escalation.pins.classify_pins` (see the module
    # docstring's field summary above). Zero migration, same pattern as
    # members / evidence / train_state / the triage quad / granted_files
    # above: legacy JSON without this key deserialises to None via the
    # from_dict __dataclass_fields__ filter below, to_dict's asdict()
    # serialises it automatically, and queue.submit / submit_resolved /
    # _atomic_write / resolve / park / stamp_triage need NO change (they are
    # field-agnostic passthroughs or RMW-on-hydrated-record).
    # Stamped by TaskWorkflow, Harness, and the escalate_blocker/escalate_info
    # chokepoint (task 3550); None on legacy rows and from non-stamping
    # producers such as submit.py's detached CLI.
    filing_claimant_run_id: str | None = None
    # Preserved incoming framing (task 3997).  APPEND-ONLY: an amendment NEVER
    # overwrites this record's own root_cause / detail / options / summary — the
    # original human-facing framing is immutable and every incoming reframing is
    # kept alongside it.  `queue.add_members_to_l2` is the SOLE writer and also
    # the sole trimmer (it sheds the OLDEST past `queue._MAX_AMENDMENTS`,
    # counting each drop in amendments_truncated).  Zero migration, same pattern
    # as members / evidence / train_state / the triage quad / granted_files /
    # filing_claimant_run_id above: legacy JSON without these keys deserialises
    # to the defaults via the from_dict __dataclass_fields__ filter below,
    # to_dict's asdict() serialises them automatically, and queue.submit /
    # submit_resolved / _atomic_write / resolve / park / stamp_triage need NO
    # change (they are field-agnostic passthroughs or RMW-on-hydrated-record).
    amendments: list[Amendment] = field(default_factory=list)
    amendments_truncated: int = 0
    # BYTE-side counterpart of amendments_truncated: characters dropped by the
    # per-field caps that make the list's size (not merely its length) bounded.
    # Same zero-migration story as the two fields above.
    amendments_chars_elided: int = 0
    # The DISTINCT PRE-CANONICAL root_cause spellings that have folded into this
    # L2 (task 3998), the record's own spelling seeded first.  Canonicalising the
    # root-cause match makes MORE promotes fold BY DESIGN, so its failure mode is
    # OVER-folding — distinct causes silently merged under one canonical key —
    # and the set of mutually-distinct spellings one cluster has been addressed
    # by is that failure's only observable signature.
    #
    # NOT DERIVABLE FROM `amendments`, which is why this is a field and not a
    # scan: that list sheds its OLDEST entries at `queue._MAX_AMENDMENTS` and
    # elides each field at `_MAX_AMENDMENT_LINE_CHARS`, so a derived count would
    # silently plateau and then UNDER-report precisely during the runaway the
    # count exists to catch — reintroducing the silent suppression it is meant to
    # close.
    #
    # The count that matters is `len(root_cause_variants) +
    # root_cause_variants_truncated`, so the loss at the cap stays assertable
    # FROM THE RECORD and is never log-only (INV-8).  `queue.add_members_to_l2`
    # is the SOLE writer and sole trimmer (oldest-shed past
    # `queue._MAX_ROOT_CAUSE_VARIANTS`, each drop counted), exactly as for
    # `amendments`.  Zero migration by the same from_dict __dataclass_fields__
    # filter as every field above.
    root_cause_variants: list[str] = field(default_factory=list)
    root_cause_variants_truncated: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> Escalation:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, text: str) -> Escalation:
        return cls.from_dict(json.loads(text))
