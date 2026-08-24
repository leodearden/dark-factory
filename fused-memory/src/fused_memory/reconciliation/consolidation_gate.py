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
]

# The Tier-C ``x_``-prefixed gate block under which a consolidation gate carries
# its working TOPIC (plus any inert provenance and audited waivers).  Composed
# from EXPERIMENTAL_KEY_PREFIX rather than re-spelling the ``x_`` literal — the
# precedent set by ``server/grouped_read.py::CONTESTED_METADATA_KEY``.  A Tier-C
# key passes the metadata boundary silently and generates no unknown-key census
# line, so it needs no amendment to RESERVED_VOCABULARY_KEYS.
GATE_METADATA_KEY = f'{EXPERIMENTAL_KEY_PREFIX}recon_consolidation_gate'
