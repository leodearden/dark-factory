"""Referent-set resolution: WHICH source is authoritative for one write.

The PRECEDENCE POLICY layer over utils/canonical_labels. That module stays THE
single normative site for the label VOCABULARY (INV-5 / PRD resolved decision
5); this one decides which of the available referent sources speaks for a given
write, and reports what the prose contradicted.

Precedence, strictly, one authoritative source per resolution::

    declared  >  metadata.task_id  >  derived scan  >  none

``.source`` is SINGULAR because exactly one source wins — this is an override
chain, not a union. A caller that declared its referents is believed over
ambient harness metadata, which is believed over what the prose happens to say.

This module compiles NO regex of its own and must never grow one. The derived
path IS :func:`~fused_memory.utils.canonical_labels.scan_content`, the declared
path builds :class:`~fused_memory.utils.canonical_labels.Referent` objects, and
the metadata bridge tries
:func:`~fused_memory.utils.canonical_labels.parse_node_name` before its single
bare-digit branch. A second copy of the label pattern here would be exactly the
lockstep duplication INV-5 forbids, and the drift would be invisible until a
destructive consumer acted on the stale half.

KNOWN BLIND SPOTS, inherited from ``scan_content`` and restated so no reader
assumes completeness (PRD resolved decision 8): a node named with bare digits
('1251'), a reference made by task TITLE rather than number, and Greek-letter
or codename aliases ('Task θ2=2184') are all invisible to the DERIVED path by
design. Precision over recall — consumers perform destructive edge surgery, so
a false positive misattributes a fact. The load-bearing consequence is in
:func:`resolve_referents`: an empty scan is UNINFORMATIVE, never contradictory,
so it can never produce a conflict and can never reject an honest write.

This module is a dependency-free leaf — stdlib plus utils/canonical_labels and
utils/validation, both themselves leaves — so leaf δ (``server/tools.py``) and
leaf ε (``services/memory_service.py``) can each import it without a cycle.
Mirrors utils/cross_project_refs.py, the pure policy consumer of
canonical_labels whose shape it copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fused_memory.utils.canonical_labels import Referent

#: The CLOSED vocabulary of resolution sources, in precedence order (strongest
#: first). Exported as one tuple so task ι's declaration-rate telemetry
#: iterates a single site rather than re-spelling four string literals; a
#: second copy would drift the same way two copies of the label vocabulary do.
#:
#: - ``'declared'``  — the caller stated its referents explicitly, INCLUDING
#:   the empty declaration ``[]`` ("considered, none apply").
#: - ``'metadata'``  — bridged from ambient ``metadata['task_id']``.
#: - ``'derived'``   — scanned out of the content by ``scan_content``.
#: - ``'none'``      — nothing declared, bridged or derivable. A real, COUNTED
#:   outcome, not a missing value.
REFERENT_SOURCES: tuple[str, ...] = ('declared', 'metadata', 'derived', 'none')

#: The type twin of :data:`REFERENT_SOURCES`. Single-sourced against it below
#: so the constant and the type cannot drift apart.
ReferentSource = Literal['declared', 'metadata', 'derived', 'none']

#: A resolved set of referents.
#:
#: Named HERE, at the producer, because the PRD's ζ signature
#: (``_verify_episode_referents(..., referents: ReferentSet)``) names this type
#: and nothing in the tree defines it — naming it at the producer stops ζ
#: inventing a competing spelling, the same INV-5 pressure that produced β.
#:
#: A SET by content — de-duplicated on ``(kind, project_id, number)`` with
#: first-seen order preserved, the same key and discipline ``scan_content``
#: uses — and a TUPLE by type, for :class:`LabelScan`'s stated reason: a list
#: would stay ``.append()``-able on an object that is evidence for destructive
#: graph surgery.
ReferentSet = tuple[Referent, ...]


@dataclass(frozen=True, kw_only=True)
class ReferentResolution:
    """What one write's referents resolved to, and from where.

    Frozen for the same reason :class:`Referent` and :class:`LabelScan` are: a
    resolution is EVIDENCE for destructive edge surgery, not a mutable
    accumulator. ``frozen=True`` blocks attribute rebinding only, which is why
    every referent field is a :data:`ReferentSet` (a tuple) rather than a list
    — otherwise ``resolution.referents.append(...)`` would let a consumer
    quietly add a referent the resolver refused to infer.

    Keyword-only because four same-shaped referent-ish fields read as noise
    positionally, and a call site that swapped ``conflicts`` for ``ambiguous``
    would type-check fine.
    """

    #: Which source was authoritative. REQUIRED — no default — because
    #: ``.source`` must be set on EVERY resolution including the empty one
    #: (``source='none'``), and a required field makes omission structurally
    #: impossible rather than merely inconvenient. One of
    #: :data:`REFERENT_SOURCES`.
    source: ReferentSource
    #: The resolved referents, from the authoritative source only. Empty is a
    #: legitimate answer for every source; an empty set carries nothing to
    #: verify membership against, so a downstream verifier must no-op on it
    #: regardless of ``.source``.
    referents: ReferentSet = ()
    #: DECLARED referents the scanned content contradicts — a declared referent
    #: of kind K whose kind the scan did see, but which the scan did not name.
    #:
    #: Populated on the ``'declared'`` path ONLY. Ambient ``metadata.task_id``
    #: is not a claim about the prose (an agent working on task 3668
    #: legitimately writes memories about Task 2500), and the derived path IS
    #: the scan, so neither can contradict it.
    #:
    #: This module REPORTS; it never rejects and never degrades ``.referents``
    #: because of a conflict. Whether a conflict rejects the write is leaf δ's
    #: gate to decide (``_entities_gate`` in ``server/tools.py``) — a resolver
    #: that silently fell back to the scan here would produce a write the
    #: caller never asked for.
    conflicts: ReferentSet = ()
    #: Referents the CONTENT is genuinely ambiguous about — a number claimed
    #: both by a bare own-project mention and by a foreign-qualified reference.
    #: Reported verbatim from the scan whatever the winning source, and NEVER
    #: promoted into :attr:`referents`: ambiguity is recorded, not guessed.
    ambiguous: ReferentSet = ()
