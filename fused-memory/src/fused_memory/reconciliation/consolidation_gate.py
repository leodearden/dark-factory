"""Consolidation-gate shape brief + mechanical closure predicate (task 3112).

## The two defects this module fixes

**Defect 1 — the gate-filing instruction is silent on the target END STATE.**
:func:`fused_memory.reconciliation.recon_self_model.render_source_completion_section`
is today the whole instruction a recon stage gets for filing a consolidation
gate, and it prescribes no end-state shape, no enumeration policy and no
re-search guard.  Every gate therefore invents its own (dark-factory gates
2969/2973/3011/3016/3036/3063/3092; 3036 hand-wrote its member enumeration
under an invented ``metadata.memory_ids`` key, which a later cycle then
extended 7→8 while it still defined "done").  :func:`render_end_state_brief`
and :func:`render_consolidation_gate_section` supply the missing shape.

**Defect 2 — nothing REFUSES to close a gate task over a malformed cluster.**
``consolidate_memories`` (task 3133) reports closure at *op* time, but the
user-observable signal is the gate TASK closing, and no gate stood between a
curator's claim and that transition.  :func:`evaluate_closure` is that
predicate; ``middleware/task_interceptor.py::TaskInterceptor._apply_status_transition``
is the seam that calls it.

## The end state: PRD §3 Option C, as ratified by gate 3200

A consolidated cluster is *N short single-claim peers sharing
``metadata.topic``, exactly one of which carries ``canonical: true``*.  The
surviving same-topic peers are therefore **correct**, not residue: a predicate
that treats a live same-topic peer as a defect makes every correctly executed
consolidation permanently uncloseable.  The predicate here refuses only on
cluster MALFORMEDNESS (wrong canonical count, an "absorbed" id still live, an
unstamped cluster member) or on a view too incomplete to judge — never on peer
count.

## Import-LEAF, deliberately

``middleware/task_interceptor.py`` imports this module, so this module's import
weight becomes the interceptor's.  Imports are restricted to the standard
library plus :mod:`fused_memory.memory_metadata`,
:mod:`fused_memory.utils.validation` and
``reconciliation.recon_self_model.EXECUTION_CLASSES``.  PRD D4 records a
*measured* hard import cycle from a careless import of exactly this kind
(``config/schema.py`` → ``memory_metadata`` → ``backends.mem0_client`` →
``config.schema`` raising ``ImportError: cannot import name
'FusedMemoryConfig'``), which is why ``TOPIC_SLUG_RE`` got its own stdlib-only
leaf module with a regression test.  Nothing here may import
``reconciliation.targeted``, ``reconciliation.harness``,
``services.memory_service`` or ``server.tools``;
``tests/test_consolidation_gate.py`` pins that in both import orders.
"""

from __future__ import annotations

from fused_memory.memory_metadata import EXPERIMENTAL_KEY_PREFIX

__all__ = [
    'GATE_METADATA_KEY',
    'render_end_state_brief',
]

# The Tier-C ``x_``-prefixed gate block under which a consolidation gate carries
# its working TOPIC (plus any inert provenance and audited waivers).  Composed
# from EXPERIMENTAL_KEY_PREFIX rather than re-spelling the ``x_`` literal — the
# precedent set by ``server/grouped_read.py::CONTESTED_METADATA_KEY``.  A Tier-C
# key passes the metadata boundary silently and generates no unknown-key census
# line, so it needs no amendment to RESERVED_VOCABULARY_KEYS.
GATE_METADATA_KEY = f'{EXPERIMENTAL_KEY_PREFIX}recon_consolidation_gate'


# --------------------------------------------------------------------------- #
# Defect 1 — the end-state brief.  Single-sourced: the filed gate's
# description, the stage prompt section and the closure predicate's docstring
# all read the SAME text, so the three cannot prescribe different targets.
# --------------------------------------------------------------------------- #


def render_end_state_brief() -> str:
    """Render the target-shape brief a filed consolidation gate carries.

    This is Defect 1's payload.  Until now
    ``recon_self_model.render_source_completion_section`` was the entire
    gate-filing instruction and it named no end state at all, so each gate
    invented its own.  Reused verbatim by
    :func:`render_consolidation_gate_section` and embedded in every gate built
    by :func:`build_consolidation_gate_task`, so the prompt, the filed gate and
    the closure predicate cannot drift apart.
    """
    return (
        'TARGET END STATE (PRD memory-metadata-vocabulary §3 Option C, '
        'ratified by gate 3200):\n'
        ' 1. Split the cluster into N SHORT single-claim peers. Each peer '
        'states ONE claim; none is a concatenation of the others.\n'
        ' 2. Every peer carries the same `metadata.topic=<slug>`. That shared '
        'topic — not a hand-written member list — is what makes the cluster '
        'findable and is the only working list the closure check reads.\n'
        ' 3. Exactly ONE peer carries `metadata.canonical: true`, and that '
        'canonical is itself SHORT: an index/summary claim pointing at its '
        'peers, never the concatenated body of the cluster.\n'
        ' 4. That canonical\'s `metadata.supersedes` lists ONLY ids genuinely '
        'deleted/absorbed. An id that is still live must not appear there.\n'
        ' 5. Execute the change through the `consolidate_memories` tool\'s '
        'ratified `retain` arm rather than by hand: retained ids are tagged '
        'in place with the cluster topic and are never deleted, never given '
        '`canonical`, never given `parent_id`. Surviving same-topic peers are '
        'the TARGET, not residue.\n\n'
        'THE "1 canonical + 1 appendix" ABSORBING END STATE IS RETIRED. Do not '
        'aim for it. PRD §3 measured the inversion twice: 168c3a6b ranked '
        '10/10 and was then deleted, and its ~9k-char replacement bbc063a7 was '
        'ABSENT from a limit=10 window while ten short siblings ranked '
        '0.66-0.76. Post-consolidation write rate measurably DOUBLED. The '
        'effect is a property of entry LENGTH, not of those particular '
        'entries — so an absorbing canonical becomes the least retrievable '
        'member of its own cluster, and writers who cannot retrieve it keep '
        'minting entry N+1. Absorbing the cluster into one long record '
        'therefore defeats the consolidation it was meant to perform.'
    )
