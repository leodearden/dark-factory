"""Shadow-mode L2 adjudication rulings — MEASUREMENT ONLY, no authority granted.

``docs/escalation-standing-policy.md`` proposes classes of L2 escalation that an
adjudicating session could one day rule without waiting for the human. NONE of
them is adopted. This module is the measurement half of that proposal: a watcher
records what it WOULD have ruled, takes no action, and a weekly count compares
the recorded proposal against what the human actually did. Nothing here decides
anything, closes anything, or widens any caller's authority — the shape of that
authority is owned by ``escalation/src/escalation/authority.py`` and is untouched
by this file.

**Why the payload rides inside ``triage_note``.** The tool that writes it,
``escalation/src/escalation/server.py::stamp_triage``, has exactly one free
parameter — ``triage_note: str``. There is no structured field for a shadow
ruling and this task deliberately adds none: a new ``Escalation`` field plus a
new tool argument would be a schema and API change in the very subsystem whose
authority the task must not touch.

**Why the payload is a LINE rather than the whole note.** ``triage_note``
already carries a freshness contract — a named world-facing predicate plus the
probe used to check it (``skills/escalation-watcher/SKILL.md``, "Reading a
triage-ack annotation"). A shadow stamp must COMPOSE with that contract, not
replace it, so the payload is one ``x_shadow_ruling: {...}`` line among the
note's other lines. Reading it is ``line.startswith(marker)`` plus
``json.loads`` — real structured data behind a fixed token, never an ad-hoc
grammar over prose.

**Promotion path.** If a class is ever adopted, ``x_shadow_ruling`` becomes a
first-class ``Escalation`` field and this codec is DELETED, not extended. The
string-carried form is the price of changing nothing while the measurement runs;
it is not a design to build on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from escalation.authority import L2_AUTO_CLOSE_DENY_CATEGORIES, L2_AUTO_CLOSE_DENY_ROLES
from escalation.models import Escalation

logger = logging.getLogger(__name__)

#: The fixed token that opens a shadow-ruling line inside a ``triage_note``.
#: A line qualifies only when it BEGINS with this token, so prose that merely
#: mentions it — this module's own guidance, or the skill's — is never mistaken
#: for a stamp.
SHADOW_RULING_MARKER = 'x_shadow_ruling:'

#: The first-tranche candidate classes, SHADOWED AND NOT ADOPTED. Each names a
#: recurring L2 shape the standing-policy skeleton proposes could be ruled by
#: the adjudicating session. Enumerated with their prose gloss in
#: ``docs/escalation-standing-policy.md``; that document and this frozenset are
#: held equal by ``tests/scripts/test_shadow_ruling_doc_contract.py``.
FIRST_TRANCHE_CLASSES: frozenset[str] = frozenset({
    'risk_identified_branch_behind_main',
    'risk_identified_recovery_veto_streak',
    'design_concern_semantic_collision',
    'infra_issue_transient_self_cleared',
})

#: The reversible actions the ratified skeleton permits. A SUPERSET of the C1
#: ``resolution_action`` vocabulary: ``close_only`` and ``resume`` are C1 values
#: a resolved record records, while ``add_dependency``, ``update_task_amendment``
#: and ``file_task`` are task-side operations that leave ``resolution_action``
#: unset — which is why the weekly count needs a third ``not_comparable``
#: outcome rather than folding them into agreement or divergence.
REVERSIBLE_ACTIONS: frozenset[str] = frozenset({
    'close_only',
    'resume',
    'add_dependency',
    'update_task_amendment',
    'file_task',
})

#: The seven classes of decision that stay with the human FOREVER, whatever any
#: shadow measurement shows. Only the first two have a record-level signal; see
#: :func:`mechanically_gated`. The authority for this list is
#: ``docs/escalation-standing-policy.md``, which glosses each one.
HUMAN_FOREVER_GATES: frozenset[str] = frozenset({
    'milestone_gate',
    'deterministic_runner_filing',
    'model_admission',
    'physical_operator_action',
    'irreversible_deletion',
    'spend_or_eval_launch',
    'post_breaker_resume_scheduler',
})

#: The two gate slugs :func:`mechanically_gated` can return. A PROPER subset of
#: :data:`HUMAN_FOREVER_GATES` — the remaining five gates are semantic and have
#: no record-level signal at all.
DETECTABLE_GATES: frozenset[str] = frozenset({'milestone_gate', 'deterministic_runner_filing'})

#: The categories and roles the detector matches against — THE IMPORTED
#: ``escalation.authority`` tables, bound under local names, never copies of
#: their literals. Re-exported so the SPOT property is assertable without
#: reaching into a private name: a member added to authority.py propagates here
#: for free, and replacing either binding with a literal fails the guard.
GATED_CATEGORIES: frozenset[str] = L2_AUTO_CLOSE_DENY_CATEGORIES
GATED_ROLES: frozenset[str] = L2_AUTO_CLOSE_DENY_ROLES

#: Wire keys of the JSON payload. ``class`` rather than ``ruling_class`` because
#: that is what a reader of the note sees; the Python attribute cannot be
#: spelled ``class``.
_PAYLOAD_KEYS = frozenset({'class', 'proposed_action', 'evidence', 'confidence'})


@dataclass(frozen=True)
class ShadowRuling:
    """What an adjudicating session WOULD have ruled, had the class been adopted.

    Validated on construction, so a rejected payload renders nothing and
    persists nothing. The four fields are exactly the four the standing policy
    requires of a proposal: which class it falls in, which reversible action it
    proposes, the evidence quoted verbatim, and how sure the adjudicator was.
    """

    ruling_class: str
    proposed_action: str
    evidence: str
    confidence: float

    def __post_init__(self) -> None:
        if self.ruling_class not in FIRST_TRANCHE_CLASSES:
            raise ValueError(
                f'unknown shadow ruling_class {self.ruling_class!r}; '
                f'expected one of {sorted(FIRST_TRANCHE_CLASSES)}'
            )
        if self.proposed_action not in REVERSIBLE_ACTIONS:
            raise ValueError(
                f'unknown shadow proposed_action {self.proposed_action!r}; '
                f'expected one of {sorted(REVERSIBLE_ACTIONS)}'
            )
        if not isinstance(self.evidence, str) or not self.evidence:
            raise ValueError(
                'shadow ruling evidence must be a non-empty string quoting the '
                'deciding evidence verbatim'
            )
        if not isinstance(self.confidence, int | float) or isinstance(self.confidence, bool):
            raise ValueError(f'shadow ruling confidence must be a number, got {self.confidence!r}')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f'shadow ruling confidence {self.confidence!r} outside [0.0, 1.0]'
            )

    def to_note_line(self) -> str:
        """Render the marker line to append to an existing ``triage_note``.

        Always ONE line: JSON escaping folds any newline in *evidence* into
        ``\\n``, so the caller can append this to a note without breaking the
        note's other lines.
        """
        payload = {
            'class': self.ruling_class,
            'proposed_action': self.proposed_action,
            'evidence': self.evidence,
            'confidence': self.confidence,
        }
        return f'{SHADOW_RULING_MARKER} {json.dumps(payload, sort_keys=True)}'


def parse_shadow_ruling(triage_note: str) -> ShadowRuling | None:
    """Return the :class:`ShadowRuling` carried by *triage_note*, or ``None``.

    ``None`` means "no usable shadow ruling here" and is the answer for a note
    with no marker line, an empty note, a marker line whose payload is not a
    JSON object, and a payload whose values are outside the policy's
    vocabularies. It NEVER raises: the weekly count sweeps every escalation in
    the queue and archive, the overwhelming majority of which carry no stamp,
    so an exception would turn one malformed note into a failed measurement.

    A marker line that IS present but unusable is logged at WARNING — an
    unreadable stamp is a lost sample, and losing samples silently is how a
    class's agreement rate drifts without anybody noticing.
    """
    for line in triage_note.splitlines():
        if not line.startswith(SHADOW_RULING_MARKER):
            continue
        raw = line[len(SHADOW_RULING_MARKER):]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning('unparsable %s payload: %s', SHADOW_RULING_MARKER, exc)
            return None
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
            logger.warning(
                '%s payload keys are %r; expected exactly %r',
                SHADOW_RULING_MARKER,
                sorted(payload) if isinstance(payload, dict) else type(payload).__name__,
                sorted(_PAYLOAD_KEYS),
            )
            return None
        try:
            return ShadowRuling(
                ruling_class=payload['class'],
                proposed_action=payload['proposed_action'],
                evidence=payload['evidence'],
                confidence=payload['confidence'],
            )
        except ValueError as exc:
            logger.warning('rejected %s payload: %s', SHADOW_RULING_MARKER, exc)
            return None
    return None


def mechanically_gated(record: Escalation) -> str | None:
    """Return the human-forever gate slug *record* trips, or ``None``.

    ``None`` means **no MECHANICAL gate was detected**, NEVER "not gated". Five
    of the seven gates in :data:`HUMAN_FOREVER_GATES` — model admission,
    physical operator actions, irreversible deletions, spend/eval launches and a
    post-breaker ``resume_scheduler`` — are semantic judgements with no signal
    on the record at all. ``docs/escalation-standing-policy.md`` remains the
    authority for the full list; this function covers only its detectable
    subset, and a caller that treats a ``None`` here as clearance is reading it
    wrong.

    Category and role are checked independently, mirroring the defence in depth
    ``escalation/src/escalation/authority.py`` already relies on: neither benign
    half can mask the other.

    THE ``design_concern`` TENSION, stated rather than hidden. The set this
    matches is authority.py's auto-close denylist, and ``design_concern`` is a
    member of it — so every ``design_concern`` shadow stamp lands in the weekly
    count's ``gated_stamps`` bucket, including one filed under the
    ``design_concern_semantic_collision`` first-tranche candidate. That is why
    the candidate is shadow-only. It is ALSO why adoption is a question about
    the INTERACTIVE arm rather than the auto arm: per authority.py's own
    docstring these tables constrain only identified callers in
    ``ROLE_LEVEL_ALLOWLIST`` — whose sole member is the auto-watcher identity —
    and a header-less interactive connection is never narrowed by that module.
    Extending a class to the AUTO watcher would need an authority.py change;
    adopting one for the interactive session would not.
    """
    if record.category in GATED_CATEGORIES:
        return 'milestone_gate'
    if record.agent_role in GATED_ROLES:
        return 'deterministic_runner_filing'
    return None
