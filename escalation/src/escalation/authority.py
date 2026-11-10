"""Server-side identity-derived escalation authority policy.

PRD ``plans/task-status-authority-prd.md`` contract C8 / decision D7
(findings 10.1/10.2). Closes the general-case hole left by 2041's
header-opt-in capability guard: without this module, ANY caller other than
the patched auto-watcher could resolve/promote at any level simply by
omitting the X-Escalation-Levels header (default-open).

These tables constrain ONLY identified callers whose identity is a key of
``ROLE_LEVEL_ALLOWLIST`` / a member of ``PROMOTE_ALLOWED``. A header-less
connection (no X-Escalation-Identity) — the esc-2087-2 human-channel
guarantee — is NEVER narrowed by this module and keeps full authority.
Policy is DATA at module scope (mirrors the existing RESOLVE_ACTIONS /
_DISMISS_ACTIONS convention in ``escalation.server`` and
``escalation/models.py::HARNESS_SENTINEL_ROLE_PREFIXES``), not an
``if resolved_by == watcher`` fork.

Layer direction: ``escalation`` is the lower fleet-wide package and must NOT
module-level import ``orchestrator`` (the existing orchestrator imports in
``escalation.server`` are deliberately lazy/in-function, optional). So the
canonical watcher identity string below is DUPLICATED — not imported — from
``orchestrator.harness._WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']``,
and pinned in lockstep by a cross-layer TEST import in
``tests/test_authority.py``. If either string changes, update both sides
together.

Task 2630 adds a second, narrower policy on top of the above: an
above-ceiling ``close_only`` carve-out for the auto-watcher at L2 (see
:func:`l2_auto_close_class` below). The ceiling above remains the PRIMARY
gate — ``escalation.server.resolve_issue`` only consults the carve-out when
the ceiling would otherwise deny the call (``rec.level not in effective``),
so a non-``close_only`` action, a non-watcher identity, or a level other
than 2 never reaches it. The carve-out enforces STRUCTURAL evidence
presence — case-insensitive regex markers required in the resolution text —
not semantic verification: the escalation server cannot itself run a git
merge-base, probe live infra, or call ``get_task``, so it can only require
that the watcher QUOTED the proof, trusting the read-only watcher to have
done the check. The denylist (``L2_AUTO_CLOSE_DENY_CATEGORIES`` /
``L2_AUTO_CLOSE_DENY_ROLES``) is applied BEFORE the allowlist, guaranteeing
the born-at-L2 human gates (the design_concern / milestone_gate /
milestone_check_failed / milestone_check_stalled / curator_adjudication_missing
/ review_due categories, and the ``orchestrator-deterministic`` sentinel role)
are never auto-closable even when a class predicate would otherwise match. That guarantee belongs to the
MCP ``resolve_issue`` path. Direct-queue paths also auto-close L2s, gated by
MEMBERSHIP rather than by class, so the category denylist does not bind them:

- ``orchestrator/src/orchestrator/sentinel_alarm_wrappers.py::discharge_sentinel_wrappers``,
  called by ``recovery_emission.py::resolve_recovery_veto_streak_escalation``
  and ``zero_progress_requeue.py::resolve_zero_progress_requeue_alert``,
  resolves an L2 of ANY category whose every member is an already-resolved
  alarm of the resolving detector's own class;
- ``orchestrator/src/orchestrator/offline_lane_escalations.py::release_lane_record``
  resolves an L2 whose only member is the lane's own released record.

Such a wrapper holds nothing the detector has not itself stood down.

Task 4541 adds ``PROMOTE_SENTINEL_BOUND_ROLES``, keyed on the agent_role of
the records being PROMOTED rather than on the caller's identity, so unlike
the tables above it binds header-less callers too. Tasks 6861 and 5880 bind
the offline lane's records and the zero-progress requeue alarm by the same
rule.

Task 8422 adds ``POLICY_TABLE`` (PRD ``plans/decision-stream-prd.md`` §5.1):
the ONE enumeration of escalation policy classes. Each :class:`PolicyRow`
names the records it applies to, the acts it authorises, and the
presence-only evidence the resolution must quote, exactly as the 2630
carve-out above does; the carve-out's three classes are rows of that table
rather than a second copy of their regexes. Whether a class may act on a
given queue is a per-queue :class:`ClassSwitch` held outside this module.
Two more identified callers read it: ``EXECUTOR_IDENTITY`` (levels {0,1,2},
every act gated by :func:`executor_refusal`) and ``COORDINATOR_IDENTITY``
(full levels, denied only by :func:`coordinator_denied`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType

from escalation.action_effects import ACTION_EFFECTS
from escalation.categories import EscalationCategory, require_registered_category
from escalation.models import Escalation
from escalation.store_identity import QueueKind

# Must equal orchestrator.harness._WATCHER_ESCALATION_HEADERS['X-Escalation-Identity'].
# Duplicated (not imported) to preserve the escalation -> orchestrator layer
# direction; pinned in lockstep by tests/test_authority.py.
_WATCHER_AUTO_IDENTITY = 'orchestrator-escalation-watcher-auto'

# The escalation executor rotation (PRD plans/decision-stream-prd.md §3
# decision 3): levels {0,1,2}, every act class-gated by executor_refusal below.
EXECUTOR_IDENTITY = 'orchestrator-escalation-executor'

# The human sitting: full levels, denied only on a record carrying a fresh
# executor claim (coordinator_denied below).
COORDINATOR_IDENTITY = 'interactive:coordinator'

# Identity -> the set of escalation levels that identity may resolve/park.
# A present X-Escalation-Levels header may only NARROW within this ceiling,
# never widen it (see escalation.server resolve_issue). Identities absent
# from this mapping fall back to the 2041 header-opt-in behaviour.
ROLE_LEVEL_ALLOWLIST: dict[str, frozenset[int]] = {
    _WATCHER_AUTO_IDENTITY: frozenset({0, 1}),
    EXECUTOR_IDENTITY: frozenset({0, 1, 2}),
}

# Identities permitted to mint a new L2 via promote_to_l2. Header-less
# (absent identity) callers are unaffected — this only gates identified
# callers not in this set. The coordinator is a member because PRD §3
# decision 3 gives it full levels, denied only on a fresh executor claim;
# refusing it promote_to_l2 would be a second, unstated denial.
PROMOTE_ALLOWED: frozenset[str] = frozenset({
    _WATCHER_AUTO_IDENTITY, EXECUTOR_IDENTITY, COORDINATOR_IDENTITY,
})

# MEMBER agent_roles (not caller identities — disjoint from PROMOTE_ALLOWED)
# whose records live under a SYNTHETIC sentinel task id. An L2 clustering only
# such records must be minted under one of its members' own task ids: minted
# under a caller-chosen REAL id, it becomes a record every veto predicate reads
# on that task, so the alarm about a hold becomes part of the hold
# (esc-5469-11).
# Must equal orchestrator.recovery_emission.RECOVERY_VETO_STREAK_ROLE,
# orchestrator.offline_lane_escalations.LANE_ESCALATION_ROLE and
# orchestrator.zero_progress_requeue.ZERO_PROGRESS_REQUEUE_ROLE; the full WHY is
# at orchestrator/src/orchestrator/recovery_emission.py::
# RECOVERY_VETO_STREAK_SENTINEL_PREFIX. Duplicated (not imported) to preserve
# the escalation -> orchestrator layer direction; each member is pinned in
# lockstep by its own test in tests/test_authority.py.
PROMOTE_SENTINEL_BOUND_ROLES: frozenset[str] = frozenset({
    'orchestrator-recovery-veto-streak',
    'orchestrator-offline-lane',
    'orchestrator-zero-progress-requeue',
})


# ---------------------------------------------------------------------------
# L2 auto-close carve-out (task 2630) — a narrow, evidence-gated exception
# ABOVE the {0,1} ceiling above, scoped to the auto-watcher identity only.
# ---------------------------------------------------------------------------

# The only resolve_issue action ever carved out above the ceiling. Every
# other action (resume/restart/park/abandon) stays level_forbidden at L2.
L2_AUTO_CLOSE_ACTION: str = 'close_only'

# Categories that are NEVER auto-closable regardless of class match — the
# born-at-L2 human-judgment categories (design decisions, and the THREE milestone
# categories the deterministic runner files: 'milestone_gate' — the gate
# itself — 'milestone_check_failed' — a failed predicate check — and
# 'milestone_check_stalled' — a re-arming predicate still "not yet" at its stall
# cap. The latter two are also covered by L2_AUTO_CLOSE_DENY_ROLES below (all
# file under agent_role='orchestrator-deterministic'), but are listed here too
# for defense-in-depth so a future filing under a different role stays blocked.
# 'curator_adjudication_missing' (filed by ``orchestrator.deterministic_runner``)
# is the re-ask raised when a 'human_curator_gate' task resumes with no
# 'human_curator_adjudicated_at' stamp; auto-closing it re-opens the task-3181
# incident verbatim (esc-3181-1 was auto-resolved by the watcher whose own
# resolution text said the curator work was "deliberately NOT executed"). Like
# 'milestone_check_failed' it is ALSO covered today by L2_AUTO_CLOSE_DENY_ROLES
# (it files under agent_role='orchestrator-deterministic'), and is listed here
# for that same defense-in-depth reason. 'review_due' is the runner-filed human
# gate of a trigger chain, held for the same reason.
#
# Every member is spelled through the registry, so a misspelling fails at
# import instead of silently ending the denial. The five RUNNER-FILED members
# are additionally held in lockstep with their named constants in
# ``orchestrator.deterministic_runner`` by the function-local orchestrator
# imports in ``tests/test_authority.py``, per the layer-direction rule in this
# module's docstring.
# Checked BEFORE the allowlist (denylist-first).
L2_AUTO_CLOSE_DENY_CATEGORIES: frozenset[str] = frozenset({
    EscalationCategory.DESIGN_CONCERN,
    EscalationCategory.MILESTONE_GATE,
    EscalationCategory.MILESTONE_CHECK_FAILED,
    # The stall escalation of a re-arming predicate (PRD completion-driven-triggers C1).
    EscalationCategory.MILESTONE_CHECK_STALLED,
    EscalationCategory.CURATOR_ADJUDICATION_MISSING,
    # The human gate of a trigger chain (PRD completion-driven-triggers C4).
    EscalationCategory.REVIEW_DUE,
})

# Agent roles that are NEVER auto-closable regardless of category/evidence —
# the born-at-L2 human-gate sentinel role (deterministic always_escalates /
# operator / acceptance / milestone predicate gates all file under this role
# per CLAUDE.md's "Deterministic task kind" section). Checked BEFORE the
# allowlist (denylist-first).
L2_AUTO_CLOSE_DENY_ROLES: frozenset[str] = frozenset({'orchestrator-deterministic'})

# The roles a host alarm files under. 'harness-dashboard-watchdog' is the role
# scripts/dashboard-watchdog.py::ESCALATION_AGENT_ROLE files under (the PRD's
# bare 'dashboard-watchdog' is not a role any filer writes);
# tests/test_authority.py pins it in lockstep. 'orchestrator-watchdog' (PRD
# boundary row 3) has no filer yet: it reserves the role for the task-8270
# host alarm.
HOST_FILER_ROLES: frozenset[str] = frozenset({
    'orchestrator-watchdog', 'harness-dashboard-watchdog',
})

# The executor's deny roles: the sentinel role above plus the host alarms.
# Its deny categories ARE L2_AUTO_CLOSE_DENY_CATEGORIES.
EXECUTOR_DENY_ROLES: frozenset[str] = L2_AUTO_CLOSE_DENY_ROLES | HOST_FILER_ROLES

# How long an executor claim stays fresh. The code home of the triage-ack
# freshness window that skills/escalation-watcher-auto/SKILL.md "Triage-ack
# freshness contract" states in prose.
CLAIM_TTL = timedelta(hours=6)


# ---------------------------------------------------------------------------
# The policy table (task 8422; PRD plans/decision-stream-prd.md §5.1).
# ---------------------------------------------------------------------------


class ClassSwitch(StrEnum):
    """A policy class's per-queue switch (PRD §3 decision 2).

    ``shadow`` only measures; ``live`` lets the class act; ``off`` withdraws it.
    """

    OFF = 'off'
    SHADOW = 'shadow'
    LIVE = 'live'


class PolicyAction(StrEnum):
    """Every act a policy row may authorise.

    ``close_only`` and ``resume`` are ``resolve_issue`` actions; the rest are
    task-side or infra acts the executor performs before it resolves.
    ``park``, ``restart`` and ``abandon`` are absent on purpose: no class is
    ever granted them. ``update_task_amendment`` is the shadow codec's existing
    spelling of "amend the task record", reused rather than re-coined.
    """

    CLOSE_ONLY = 'close_only'
    RESUME = 'resume'
    DONE_FLIP = 'done_flip'
    UPDATE_TASK_AMENDMENT = 'update_task_amendment'
    ADD_DEPENDENCY = 'add_dependency'
    FILE_TASK = 'file_task'
    RECOVER_MAIN = 'recover_main'
    B3_APPLY = 'b3_apply'
    REPLAY_DEAD_LETTERS = 'replay_dead_letters'
    REPROBE_QUARANTINE = 'reprobe_quarantine'


# The acts the ratified skeleton counts as reversible
# (docs/escalation-standing-policy.md '## Reversible actions').
REVERSIBLE_POLICY_ACTIONS: frozenset[PolicyAction] = frozenset({
    PolicyAction.CLOSE_ONLY,
    PolicyAction.RESUME,
    PolicyAction.ADD_DEPENDENCY,
    PolicyAction.UPDATE_TASK_AMENDMENT,
    PolicyAction.FILE_TASK,
})

# Not reversible, but stampable under a named waiver (Leo, 2026-11-04;
# plans/decision-stream-prd.md §3 decision 7).
REVERSIBILITY_WAIVED_ACTIONS: frozenset[PolicyAction] = frozenset({PolicyAction.DONE_FLIP})


_RESOLVE_ACTIONS: frozenset[str] = frozenset(action for action, _, _ in ACTION_EFFECTS)

# A file-writing or main-moving class is audited at 100% until it has this
# many closes (PRD §3 decisions 5 and 6).
FULL_AUDIT_MIN_N = 20

_POLICY_CLASS_SLUG = re.compile(r'[a-z0-9_]+')


@dataclass(frozen=True)
class PolicyMatch:
    """Which records a row applies to.

    ``categories`` must contain the record's category, and ``agent_role``
    must FULL-match its agent_role; the two are ANDed. ``None`` leaves that
    dimension unconstrained, so an all-``None`` match is any record outside
    the deny lists.
    """

    categories: frozenset[EscalationCategory] | None = None
    agent_role: re.Pattern[str] | None = None

    def __post_init__(self) -> None:
        for category in self.categories or ():
            require_registered_category(category)

    def matches(self, category: str, agent_role: str) -> bool:
        if self.categories is not None and category not in self.categories:
            return False
        return self.agent_role is None or self.agent_role.fullmatch(agent_role) is not None


@dataclass(frozen=True)
class PolicyRow:
    """One policy class: the records it covers, its acts, and its evidence.

    Every evidence pattern must search-match the resolution text; that is
    presence of quoted proof, not verification of it (see the 2630 paragraph
    in this module's docstring).
    """

    policy_class: str
    match: PolicyMatch
    actions: frozenset[PolicyAction]
    evidence: tuple[re.Pattern[str], ...]
    file_writing: bool = False
    queue_kind: QueueKind = 'project'
    initial_state: ClassSwitch = ClassSwitch.SHADOW

    def __post_init__(self) -> None:
        slug = self.policy_class
        if not _POLICY_CLASS_SLUG.fullmatch(slug):
            raise ValueError(f'policy_class {slug!r} must match ^[a-z0-9_]+$')
        if not self.actions:
            raise ValueError(f'policy row {slug!r} allows no action: {self.actions!r}')
        if _RESOLVE_ACTIONS.isdisjoint(self.actions):
            raise ValueError(
                f'policy row {slug!r} allows no resolve_issue action; '
                f'got {sorted(map(str, self.actions))}'
            )
        if not self.evidence:
            raise ValueError(f'policy row {slug!r} requires evidence; got {self.evidence!r}')
        for pattern in self.evidence:
            if not isinstance(pattern, re.Pattern):
                raise ValueError(
                    f'policy row {slug!r} evidence entry {pattern!r} is '
                    f'{type(pattern).__name__}, not a compiled re.Pattern'
                )

    @property
    def reversible(self) -> bool:
        """True iff every act the row allows is reversible; a waived act does not make a row reversible."""
        return self.actions <= REVERSIBLE_POLICY_ACTIONS

    @property
    def audit_full_until(self) -> int:
        """How many closes this class is audited at 100% for before random draw."""
        touches_files = self.file_writing or PolicyAction.RECOVER_MAIN in self.actions
        return FULL_AUDIT_MIN_N if touches_files else 0

    def missing_evidence(self, resolution: str) -> tuple[str, ...]:
        """The source of each evidence pattern *resolution* fails to match, in row order."""
        return tuple(pattern.pattern for pattern in self.evidence if not pattern.search(resolution))

    def evidence_present(self, resolution: str) -> bool:
        return not self.missing_evidence(resolution)


def _any_of(*alternatives: str) -> re.Pattern[str]:
    """One case-insensitive evidence pattern satisfied by any of *alternatives*."""
    return re.compile('|'.join(f'(?:{a})' for a in alternatives), re.IGNORECASE)


def _role(*agent_roles: str) -> re.Pattern[str]:
    """An agent_role pattern that full-matches exactly one of *agent_roles*."""
    return re.compile('|'.join(map(re.escape, agent_roles)))


def _categories(*members: EscalationCategory) -> PolicyMatch:
    return PolicyMatch(categories=frozenset(members))


def _actions(*members: PolicyAction) -> frozenset[PolicyAction]:
    return frozenset(members)


_C = EscalationCategory
_A = PolicyAction
_ANY_RECORD = PolicyMatch()

_SHA = r'\b[0-9a-f]{7,40}\b'
_ANCESTOR = (r'is[- ]ancestor', r'merge-base')
_TASK_CITE = r'task\s+#?\d+'
_ESC_ID = r'\besc-[\w-]*\d'
# A quoted liveness-probe token from a known key allowlist, NOT an arbitrary
# "word=word" token, so incidental '='s in prose ('note=ok') cannot satisfy it.
# Admits the '>' comparison harness.py itself emits (MainPID>0).
_PROBE_TOKEN = r'\b(?:paused|ActiveState|MainPID|ActiveEnterTimestamp)\b\s*[=:>]\s*\S+'

_FOUND_ON_MAIN_EVIDENCE = (_any_of(r'found_on_main'), _any_of(_SHA), _any_of(*_ANCESTOR))
_FOUND_ON_MAIN_ACTIONS = _actions(_A.DONE_FLIP, _A.CLOSE_ONLY)

# Moved classes (PRD §3 decision 7): acts already granted to the interactive
# session, moving to the executor after a shadow period that only measures.
_MOVED_ROWS: tuple[PolicyRow, ...] = (
    # Class 1b.
    PolicyRow(
        'landing_provenance', _categories(_C.PROVENANCE_UNATTRIBUTED),
        _FOUND_ON_MAIN_ACTIONS, _FOUND_ON_MAIN_EVIDENCE,
    ),
    # Class 3, self-execute item 12: a disposition, not a category.
    PolicyRow('found_on_main', _ANY_RECORD, _FOUND_ON_MAIN_ACTIONS, _FOUND_ON_MAIN_EVIDENCE),
    # Class 2's update_task half.
    PolicyRow(
        'delivered_check_repath', _categories(_C.DEPENDENCY_CAPABILITY),
        _actions(_A.UPDATE_TASK_AMENDMENT, _A.RESUME),
        (_any_of(r'delivered_checks?'), _any_of(rf'pass(?:es|ed)?\s+at\s+(?:main|{_SHA})')),
    ),
    # Item 1.
    PolicyRow(
        'read_only_probe_answered', _ANY_RECORD, _actions(_A.CLOSE_ONLY),
        (
            _any_of(r'\bprobe\b'),
            _any_of(r'->', r'=>', r'\brc\s*=', r'exit(?:ed)?\s*(?:code\s*)?\d'),
        ),
    ),
    # Item 3.
    PolicyRow(
        'premature_gate_followup', _ANY_RECORD,
        _actions(_A.FILE_TASK, _A.ADD_DEPENDENCY, _A.CLOSE_ONLY),
        (_any_of(_TASK_CITE), _any_of(r'\bdelayed\b'), _any_of(r'\bdepend')),
    ),
    # Item 4.
    PolicyRow(
        'preserved_payload_recovered', _ANY_RECORD,
        _actions(_A.UPDATE_TASK_AMENDMENT, _A.CLOSE_ONLY),
        (_any_of(r'\brecovered\b'), _any_of(_TASK_CITE)),
    ),
    # Item 5.
    PolicyRow(
        'owner_finding_appended', _ANY_RECORD,
        _actions(_A.UPDATE_TASK_AMENDMENT, _A.CLOSE_ONLY),
        (_any_of(_TASK_CITE), _any_of(r'status\s*[=:]\s*pending'), _any_of(r'\bappend')),
    ),
    # Class 1a's TRUE arm closes the gate record, as humans always did (Leo,
    # 2026-11-09, sitting item 522). Item 11, cancelling the router-spawned
    # task, has no row and stays interactive-only: docs/escalation-standing-policy.md
    # '## First-tranche candidate classes' says why.
    PolicyRow(
        'completion_claim_true', _categories(_C.UNVERIFIED_COMPLETION_CLAIM),
        _actions(_A.CLOSE_ONLY),
        (_any_of(r'status\s*[=:]\s*done'), _any_of(_SHA), _any_of(*_ANCESTOR)),
    ),
    # The three task-2630 carve-out classes; L2_AUTO_CLOSE_ALLOWLIST below
    # orders them for the auto-watcher.
    #
    # superseded_main_sweep mirrors harness._close_superseded_main_sweep_escalations:
    # it requires BOTH a newer sweep-escalation id (``esc-...``) AND proof the
    # failing tip is an ancestor of a now-green main.
    PolicyRow(
        'superseded_main_sweep', PolicyMatch(agent_role=_role('orchestrator-main-sweep')),
        _actions(_A.CLOSE_ONLY),
        (
            _any_of(r'esc-\S+'),
            _any_of(
                r'is[- ]ancestor', r'merge-base', r'ancestor of', r'verifies clean', r'gate re-run',
            ),
        ),
    ),
    # self_cleared_infra mirrors harness._revalidate_open_deterministic_escalation:
    # it requires a quoted liveness-probe token (see _PROBE_TOKEN).
    PolicyRow(
        'self_cleared_infra', _categories(_C.INFRA_ISSUE), _actions(_A.CLOSE_ONLY),
        (_any_of(_PROBE_TOKEN),),
    ),
    # stale_task_scoped — category-agnostic, requires a live get_task citation
    # showing the subject task went terminal / was re-scoped / was
    # re-dispatched. All THREE alternatives are citation-anchored (task 4192):
    # the terminal-status one on a 'status=' / 'status:' shape rather than a
    # free-standing "is done" English phrase, and the two moved-on ones on an
    # explicit preposition ('to'/'into'/'as' for re-scoped, 'as' only for
    # re-dispatched) plus a task/run identifier. So neither "the work is done"
    # nor "was rescoped last week" nor "re-dispatched it this morning" can
    # satisfy it — the identity that writes this text is the LLM-driven watcher
    # rotation, whose free-form close prose is exactly where such phrasing
    # appears incidentally.
    #
    # Precisely what the identifier requires, so the anchor is not read as
    # stronger than it is: a token containing a DIGIT, optionally wrapped in
    # '#' / quotes / backticks. It is not validated as a real task or run id,
    # and a bare number counts — so count-shaped prose that happens to use the
    # same preposition ("re-scoped into 2 subtasks") still matches. The
    # load-bearing part is the preposition+token STRUCTURE, which the casual
    # phrasing this anchor exists to reject never produces.
    #
    # This class shares its INTENT with harness._revalidate_open_l2 — a subject
    # that has gone terminal or moved on moots the escalation — but is NOT a
    # mirror of it, in three ways (verified against harness.py
    # _revalidate_open_l2 at main 6ee3eaec25):
    #   - Independent mechanism, no shared code: the harness sweep closes
    #     IN-PROCESS via self._escalation_queue.resolve(...) under the
    #     'harness-escalation-revalidation-sweep' role. It never issues an HTTP
    #     resolve_issue, so it never reaches l2_auto_close_class at all.
    #   - Strictly wider on CATEGORY: since task 2724 landed, the harness sweep
    #     is ALLOWLIST-gated on config.escalation_revalidation_allowlist
    #     (default {task_failure, stranded_blocked}); this class is
    #     category-AGNOSTIC modulo L2_AUTO_CLOSE_DENY_CATEGORIES.
    #   - Strictly wider on EVIDENCE: the harness closes only on a live batch
    #     get_statuses read showing the subject 'done'/'cancelled'; this class
    #     also accepts a cited re-scoped / re-dispatched citation, which the
    #     harness never checks.
    # It is therefore the WIDEST auto-close door in the carve-out — weigh any
    # further loosening of it against that.
    #
    # FOLLOW-UP (open, deliberate — not an oversight): this class gates
    # categories with a DENYLIST (L2_AUTO_CLOSE_DENY_CATEGORIES) while the
    # harness revalidation sweep gates with an ALLOWLIST. Task 2724 has landed
    # on main, so that divergence is a fact today, not a future one. Aligning
    # the two models is a behaviour change with its own blast radius — it would
    # narrow which L2s the watcher can auto-close at all — and was scoped OUT of
    # task 4192 by the 2026-08-12 operator ruling; it wants its own task and
    # review.
    PolicyRow(
        'stale_task_scoped', _ANY_RECORD, _actions(_A.CLOSE_ONLY),
        (
            _any_of(
                r'status\s*[=:]\s*(done|cancelled)',
                # The optional [#`"'] wrappers admit the spellings this repo
                # actually writes task references in — 'task #3821', a
                # markdown-quoted `3821`, a quoted "task 3821". They relax only
                # the citation's PUNCTUATION, never its structure: an explicit
                # preposition plus an identifier containing a digit is still
                # required, so every casual-prose negative stays rejected.
                r'''re-?scoped\s+(?:to|into|as)\s+[#`"']?(?:task\s+)?[#`"']?[a-z0-9_.:-]*\d''',
                r'''re-?dispatched\s+as\s+[#`"']?(?:run|task)?[- ]?[#`"']?[a-z0-9_.:-]*\d''',
            ),
        ),
    ),
    # Item 6's carve-out half and item 7: the PRD's "applies a named ruling"
    # key. A ruling citation lives in the resolution text, so it is evidence
    # on a match-anything row rather than a record match.
    PolicyRow(
        'applies_named_ruling', _ANY_RECORD, _actions(_A.CLOSE_ONLY, _A.RESUME),
        (
            _any_of(r'\b(?:Leo|ruled|ruling)\b[^\n]*\b20\d{2}-\d{2}-\d{2}\b'),
            _any_of(_ESC_ID, r'x_ruling'),
        ),
    ),
)

# New delegations (PRD §3 decision 7): start shadow, flip live per class at the
# η gate.
_NEW_DELEGATION_ROWS: tuple[PolicyRow, ...] = (
    # Class 5.
    PolicyRow(
        'curator_cause_cleared',
        _categories(
            _C.CURATOR_FAILURE, _C.CURATOR_CONSECUTIVE_DEGRADED, _C.CURATOR_ZERO_OUTPUT_HANG,
            _C.TICKET_FAILURE, _C.MCP_MARKUP_RESIDUE,
        ),
        _actions(_A.CLOSE_ONLY),
        (
            _any_of(r'get_curator_state', r'\bprobe\b'),
            _any_of(r'\bcleared\b', r'\brecovered\b', r'false positive'),
        ),
    ),
    # Class 1a's false-positive arm.
    PolicyRow(
        'completion_claim_false_positive', _categories(_C.UNVERIFIED_COMPLETION_CLAIM),
        _actions(_A.CLOSE_ONLY),
        (
            _any_of(r'status\s*[=:]\s*\w+'), _any_of(r'false positive'),
            _any_of(r'not an assertion', r'imperative'),
        ),
    ),
    # Detector-close confirmation.
    PolicyRow(
        'detector_stood_down',
        PolicyMatch(agent_role=_role(
            'orchestrator-recovery-veto-streak',
            'orchestrator-verify-host-monitor',
            'orchestrator-zero-progress-requeue',
        )),
        _actions(_A.CLOSE_ONLY),
        (_any_of(r'stood down', r'alarm (?:has )?cleared'), _any_of(_ESC_ID)),
    ),
    PolicyRow(
        'dead_letter_replayed', _categories(_C.DURABLE_WRITE_DEAD_LETTER),
        _actions(_A.REPLAY_DEAD_LETTERS, _A.CLOSE_ONLY),
        (_any_of(r'replay_dead_letters'), _any_of(r'replayed\s*[=:]\s*\d+')),
    ),
    # Quarantine re-probe.
    PolicyRow(
        'verify_host_reprobed', _categories(_C.VERIFY_HOST_UNREACHABLE),
        _actions(_A.REPROBE_QUARANTINE, _A.CLOSE_ONLY),
        (_any_of(r'\bprobe\b'), _any_of(r'\breachable\b', r'\brc\s*=\s*0\b')),
    ),
    # B3.
    PolicyRow(
        'b3_low_risk_unblock', _categories(_C.TASK_FAILURE, _C.REVIEW_ISSUES),
        _actions(_A.B3_APPLY, _A.RESUME, _A.CLOSE_ONLY),
        (_any_of(r'b3_gate'), _any_of(r'request_id\s*[=:]\s*\S+')),
        file_writing=True,
    ),
    # Red main (PRD §3 decision 6). File-writing because recover_main moves the
    # main checkout, which must be clean.
    PolicyRow(
        'red_main_recovery', _categories(_C.PREEXISTING_MAIN_BREAK),
        _actions(_A.RECOVER_MAIN, _A.CLOSE_ONLY),
        (
            _any_of(r'\bgate\b'), _any_of(r'\bsweep\b', r're-run'),
            _any_of(r'recover_main'), _any_of(_SHA),
        ),
        file_writing=True,
    ),
)

# Recon self-clears (PRD §3 decision 10): the only row the shared
# reconciliation queue serves.
_RECON_ROWS: tuple[PolicyRow, ...] = (
    PolicyRow(
        'recon_run_self_cleared', _categories(_C.RECON_STALE_RUN), _actions(_A.CLOSE_ONLY),
        (_any_of(r'(?:later|subsequent) run'), _any_of(r'run[_ -]?id\s*[=:]\s*\S+')),
        queue_kind='reconciliation',
    ),
)

# The four first-tranche classes the ratified skeleton shadowed before this
# table existed; their slugs are frozen by the shadow-stamp archive.
# design_concern_semantic_collision is executor-unreachable, because
# design_concern is a deny category; only the interactive arm measures it.
_LEGACY_FIRST_TRANCHE_ROWS: tuple[PolicyRow, ...] = (
    PolicyRow(
        'risk_identified_branch_behind_main', _categories(_C.RISK_IDENTIFIED),
        _actions(_A.FILE_TASK, _A.CLOSE_ONLY),
        (_any_of(*_ANCESTOR, r'behind main'),),
    ),
    PolicyRow(
        'risk_identified_recovery_veto_streak',
        PolicyMatch(
            categories=frozenset({_C.RISK_IDENTIFIED}),
            agent_role=_role('orchestrator-recovery-veto-streak'),
        ),
        _actions(_A.CLOSE_ONLY),
        (_any_of(r'4541', r'veto[- ]streak'),),
    ),
    PolicyRow(
        'design_concern_semantic_collision', _categories(_C.DESIGN_CONCERN),
        _actions(_A.UPDATE_TASK_AMENDMENT, _A.CLOSE_ONLY),
        (_any_of(r'collision', r'reconcil'),),
    ),
    PolicyRow(
        'infra_issue_transient_self_cleared', _categories(_C.INFRA_ISSUE),
        _actions(_A.CLOSE_ONLY), (_any_of(_PROBE_TOKEN),),
    ),
)

POLICY_TABLE: tuple[PolicyRow, ...] = (
    _MOVED_ROWS + _NEW_DELEGATION_ROWS + _RECON_ROWS + _LEGACY_FIRST_TRANCHE_ROWS
)


def _index_by_class(rows: Iterable[PolicyRow]) -> Mapping[str, PolicyRow]:
    by_class: dict[str, PolicyRow] = {}
    for row in rows:
        if row.policy_class in by_class:
            raise ValueError(f'duplicate policy_class {row.policy_class!r} in POLICY_TABLE')
        by_class[row.policy_class] = row
    return MappingProxyType(by_class)


_ROWS_BY_CLASS: Mapping[str, PolicyRow] = _index_by_class(POLICY_TABLE)


def policy_row(policy_class: str) -> PolicyRow | None:
    """The table row named *policy_class*, or ``None`` when there is none."""
    return _ROWS_BY_CLASS.get(policy_class)


# The task-2630 carve-out classes, ordered MOST-SPECIFIC-FIRST: a record can
# satisfy more than one row's match (an infra_issue filed by
# orchestrator-main-sweep satisfies both superseded_main_sweep's and
# self_cleared_infra's) — first-match-wins picks the sharpest attribution when
# its stricter evidence is present, falling back to the more general class
# otherwise.
L2_AUTO_CLOSE_ALLOWLIST: tuple[PolicyRow, ...] = tuple(
    _ROWS_BY_CLASS[slug]
    for slug in ('superseded_main_sweep', 'self_cleared_infra', 'stale_task_scoped')
)


def l2_auto_close_class(
    *,
    identity: str | None,
    level: int,
    action: str,
    category: str,
    agent_role: str,
    resolution: str | None,
) -> str | None:
    """Return the allowlisted auto-close class name matched by this call, or ``None``.

    Re-gates on ``identity``/``level``/``action`` even though
    ``escalation.server.resolve_issue`` only consults this function inside
    its existing ceiling-forbidden branch: a header-less human connection
    (``identity is None``) or any identity/level/action combination outside
    the narrow watcher/L2/close_only triple returns ``None`` immediately, so
    such callers are unaffected byte-for-byte by this carve-out.

    The denylist (``L2_AUTO_CLOSE_DENY_CATEGORIES`` / ``L2_AUTO_CLOSE_DENY_ROLES``)
    is applied BEFORE the allowlist, so a born-at-L2 human gate is never
    auto-closable even when a class predicate would otherwise match (e.g. an
    ``orchestrator-deterministic``-filed ``infra_issue`` is blocked by the
    role denylist despite matching ``self_cleared_infra``'s category). The
    rows' ``queue_kind`` is ignored here, exactly as before the table existed.

    Returns the matched row's ``policy_class`` (for the caller's audit trail /
    rotation-digest line) or ``None`` when no class matches — the caller
    (``resolve_issue``) treats ``None`` as "stay forbidden".
    """
    if identity != _WATCHER_AUTO_IDENTITY or level != 2 or action != L2_AUTO_CLOSE_ACTION:
        return None
    if category in L2_AUTO_CLOSE_DENY_CATEGORIES or agent_role in L2_AUTO_CLOSE_DENY_ROLES:
        return None
    text = resolution or ''
    for row in L2_AUTO_CLOSE_ALLOWLIST:
        if row.match.matches(category, agent_role) and row.evidence_present(text):
            return row.policy_class
    return None


# ---------------------------------------------------------------------------
# Executor and coordinator gates over POLICY_TABLE (task 8422; PRD §5.1).
# ---------------------------------------------------------------------------


class RefusalCode(StrEnum):
    """Why an executor act or claim was refused."""

    DENIED_RECORD = 'denied_record'
    NO_FRESH_CLAIM = 'no_fresh_claim'
    UNKNOWN_CLASS = 'unknown_class'
    ROW_MISMATCH = 'row_mismatch'
    ACTION_NOT_ALLOWED = 'action_not_allowed'
    CLASS_NOT_LIVE = 'class_not_live'
    EVIDENCE_MISSING = 'evidence_missing'
    CLASS_OFF = 'class_off'
    ALREADY_CLAIMED = 'already_claimed'
    PROJECT_ROOT_DIRTY = 'project_root_dirty'


@dataclass(frozen=True)
class PolicyRefusal:
    code: RefusalCode
    detail: str


def _claim_is_fresh(record: Escalation, now: datetime) -> bool:
    """True iff the executor claimed *record* less than CLAIM_TTL before *now*.

    An unparsable ``claimed_at`` is never fresh: the executor is refused and
    the human is not blocked, which is the safe reading on both sides.
    """
    if record.claimed_by != EXECUTOR_IDENTITY or record.claimed_at is None:
        return False
    try:
        age = now - datetime.fromisoformat(record.claimed_at)
    except (TypeError, ValueError):
        return False
    return timedelta(0) <= age < CLAIM_TTL


def _denied(record: Escalation) -> PolicyRefusal | None:
    if record.category in L2_AUTO_CLOSE_DENY_CATEGORIES:
        return PolicyRefusal(
            RefusalCode.DENIED_RECORD,
            f'category {record.category!r} is on the executor deny list',
        )
    if record.agent_role in EXECUTOR_DENY_ROLES:
        return PolicyRefusal(
            RefusalCode.DENIED_RECORD,
            f'agent_role {record.agent_role!r} is on the executor deny list',
        )
    return None


def _row_or_refusal(
    record: Escalation, policy_class: str | None, queue_kind: QueueKind,
) -> PolicyRow | PolicyRefusal:
    row = None if policy_class is None else policy_row(policy_class)
    if row is None:
        return PolicyRefusal(RefusalCode.UNKNOWN_CLASS, f'no policy row named {policy_class!r}')
    if row.queue_kind != queue_kind:
        return PolicyRefusal(
            RefusalCode.ROW_MISMATCH,
            f'{row.policy_class!r} serves {row.queue_kind!r} queues, not {queue_kind!r}',
        )
    if not row.match.matches(record.category, record.agent_role):
        return PolicyRefusal(
            RefusalCode.ROW_MISMATCH,
            f'{row.policy_class!r} does not cover category {record.category!r} '
            f'with agent_role {record.agent_role!r}',
        )
    return row


def executor_refusal(
    record: Escalation,
    action: str,
    resolution: str | None,
    state: Mapping[str, ClassSwitch],
    *,
    now: datetime,
    queue_kind: QueueKind,
) -> PolicyRefusal | None:
    """Why the executor may not resolve *record* with *action*, or ``None``.

    Checked in PRD §5.1 order: deny lists, a fresh executor claim, the claimed
    row's match and queue kind, the row's actions, the class being live on
    this queue (*state*; a class absent from it is at its row's initial
    state), and every evidence pattern present in *resolution*.
    """
    if (denied := _denied(record)) is not None:
        return denied
    if not _claim_is_fresh(record, now):
        return PolicyRefusal(
            RefusalCode.NO_FRESH_CLAIM,
            f'no executor claim younger than {CLAIM_TTL}: '
            f'claimed_by={record.claimed_by!r}, claimed_at={record.claimed_at!r}',
        )
    row = _row_or_refusal(record, record.policy_class, queue_kind)
    if isinstance(row, PolicyRefusal):
        return row
    if action not in row.actions:
        return PolicyRefusal(
            RefusalCode.ACTION_NOT_ALLOWED,
            f'{row.policy_class!r} does not allow {action!r}; '
            f'it allows {sorted(map(str, row.actions))}',
        )
    switch = state.get(row.policy_class, row.initial_state)
    if switch != ClassSwitch.LIVE:
        return PolicyRefusal(
            RefusalCode.CLASS_NOT_LIVE, f'{row.policy_class!r} is {switch}, not live',
        )
    if missing := row.missing_evidence(resolution or ''):
        return PolicyRefusal(
            RefusalCode.EVIDENCE_MISSING,
            f'{row.policy_class!r} resolution lacks required evidence: ' + '; '.join(missing),
        )
    return None


def executor_policy_class(
    record: Escalation,
    action: str,
    resolution: str | None,
    state: Mapping[str, ClassSwitch],
    *,
    now: datetime,
    queue_kind: QueueKind,
) -> str | None:
    """The class the executor acts under, or ``None`` meaning level_forbidden (PRD §5.1)."""
    refusal = executor_refusal(
        record, action, resolution, state, now=now, queue_kind=queue_kind,
    )
    return record.policy_class if refusal is None else None


def claim_refusal(
    record: Escalation,
    policy_class: str,
    state: Mapping[str, ClassSwitch],
    *,
    now: datetime,
    clean: bool,
    queue_kind: QueueKind,
) -> PolicyRefusal | None:
    """Why the executor may not claim *record* as *policy_class*, or ``None``.

    A shadow class is claimable, so the executor can act out its shadow
    measurement. *clean* is the executor's report that ``project_root`` has
    no uncommitted changes; a file-writing class needs it (PRD §3 decision 5).
    """
    if (denied := _denied(record)) is not None:
        return denied
    row = _row_or_refusal(record, policy_class, queue_kind)
    if isinstance(row, PolicyRefusal):
        return row
    if state.get(row.policy_class, row.initial_state) == ClassSwitch.OFF:
        return PolicyRefusal(RefusalCode.CLASS_OFF, f'{row.policy_class!r} is off on this queue')
    if _claim_is_fresh(record, now):
        return PolicyRefusal(
            RefusalCode.ALREADY_CLAIMED,
            f'already claimed as {record.policy_class!r} at {record.claimed_at}',
        )
    if row.file_writing and not clean:
        return PolicyRefusal(
            RefusalCode.PROJECT_ROOT_DIRTY,
            f'{row.policy_class!r} writes files and needs clean=True',
        )
    return None


def coordinator_denied(record: Escalation, now: datetime) -> bool:
    """True iff *record* carries a fresh executor claim the coordinator must not race."""
    return _claim_is_fresh(record, now)


# The only moves a class switch may make (PRD §5.1): off -> shadow -> live ->
# shadow. No writer moves a class to off.
CLASS_SWITCH_TRANSITIONS: frozenset[tuple[ClassSwitch, ClassSwitch]] = frozenset({
    (ClassSwitch.OFF, ClassSwitch.SHADOW),
    (ClassSwitch.SHADOW, ClassSwitch.LIVE),
    (ClassSwitch.LIVE, ClassSwitch.SHADOW),
})


class IllegalClassTransition(ValueError):
    """A class switch move outside :data:`CLASS_SWITCH_TRANSITIONS`."""


def check_class_transition(policy_class: str, current: ClassSwitch, target: ClassSwitch) -> None:
    if (current, target) not in CLASS_SWITCH_TRANSITIONS:
        raise IllegalClassTransition(
            f'policy class {policy_class!r} cannot move {current} -> {target}'
        )


def storm_escape_target(current: ClassSwitch, closes: int, cap: int) -> ClassSwitch | None:
    """SHADOW when a live class closed more than *cap* records in one rotation, else ``None``."""
    if current == ClassSwitch.LIVE and closes > cap:
        return ClassSwitch.SHADOW
    return None


# The audit flip rule (PRD §3 decision 8): below AUDIT_RATE_MIN_N audited
# closes any wrong one drops a live class; from there on, a wrong rate above
# the ceiling does.
AUDIT_RATE_MIN_N = 20
AUDIT_WRONG_RATE_CEILING = 0.05


def audit_escape_target(current: ClassSwitch, audited: int, wrong: int) -> ClassSwitch | None:
    """SHADOW when a live class's audited closes of one queue run wrong, else ``None``."""
    if not 0 <= wrong <= audited:
        raise ValueError(
            f'audit counts must satisfy 0 <= wrong <= audited; got audited={audited}, wrong={wrong}'
        )
    if current != ClassSwitch.LIVE:
        return None
    if audited < AUDIT_RATE_MIN_N:
        return ClassSwitch.SHADOW if wrong >= 1 else None
    return ClassSwitch.SHADOW if wrong / audited > AUDIT_WRONG_RATE_CEILING else None
