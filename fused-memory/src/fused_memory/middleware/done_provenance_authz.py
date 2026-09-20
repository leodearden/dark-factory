"""Caller bar for `deterministic-*` done provenance (PRD C5 / D11, task 5241).

The `deterministic-*` provenance kinds assert that a MACHINE observed a
task's completion — a service restart with fresh PID evidence, a predicate
check that exited 0, a pure gate whose escalation resolved. Their only live
producer is DeterministicRunner (`orchestrator/src/orchestrator/
deterministic_runner.py`). A blob hand-passed by anything else is a claim
about an observation that never happened, which is how incident 5156 closed a
gate task from inside reconciliation Stage 2.

HONEST CAVEAT, carried over verbatim in spirit from
``server/mem0_update_authz.py``: the ``agent_id`` this bar reads is
SELF-REPORTED. A caller that sends none falls back to the clientInfo name
(``server/tools.py::_resolve_identity``), which the caller also chose. So this
DETERS a cooperating caller from reaching for a primitive it was not meant to
touch; it is not a security boundary, and a determined caller can claim any
prefix. The residual is made visible instead of prevented: every done write,
accepted or refused, leaves a write-journal row naming the resolved caller
(``middleware/task_interceptor.py::_log_write_op_row``).

Structure mirrors ``server/mem0_update_authz.py`` clause for clause:
module-level live-read resolvers, a defensive per-hop attribute navigator, and
module-level defaults for a missing config hop. It keeps that module's
deliberate INVERSION against the near-duplicate guard's permissive fallbacks:
this is a mutation-authorization gate, whose safe direction is DENY, so every
fallback here denies — including a missing config object entirely.

LIVE READ (``config/reload.py``'s reload-safety precondition): the resolvers
re-read ``config.reconciliation.deterministic_provenance_allowed_agent_prefixes``
on EVERY call and capture nothing at import or construction, which is what
makes the leaf registered at ``config/reload.py`` genuinely green-tier
hot-reloadable rather than restart-only in disguise.

Lives in its own module rather than inline in
``middleware/task_interceptor.py`` for two reasons: the live-read proof test
must call the resolver DIRECTLY (a body buried in a 6.8k-line module's
validator is reachable only through a full ``set_task_status`` round trip),
and that file is already far past the size at which another concern can be
added for free (heuristic 14).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_args

from shared.task_metadata import DoneProvenance

# Fail-CLOSED module default, used whenever a config hop is missing, None, or
# the wrong type. Deny, never permit — see the module docstring's inversion
# note. The 'orchestrator' literal is deliberately NOT minted here: it reaches
# this module only as the config default (ReconciliationConfig's
# default_factory), so there is exactly one spelling of the bar (SPOT).
_DEFAULT_ALLOWED_PREFIXES: tuple[str, ...] = ()

#: The dotted config path named in every refusal so the message is
#: self-remedying — an operator reads the refusal and knows which leaf to edit.
_CONFIG_PATH = 'reconciliation.deterministic_provenance_allowed_agent_prefixes'

#: The gated kind family, DERIVED from the shared Literal rather than
#: hand-listed — the same single-source-of-truth argument
#: ``task_interceptor._DONE_PROVENANCE_KINDS`` records. Deriving makes a 5th
#: `deterministic-*` kind added to ``DoneProvenance.kind`` gated BY DEFAULT
#: (the fail-closed direction) instead of silently escaping the bar until
#: someone notices a second list has drifted.
DETERMINISTIC_PROVENANCE_KINDS: frozenset[str] = frozenset(
    kind
    for kind in get_args(DoneProvenance.model_fields['kind'].annotation)
    if kind.startswith('deterministic-')
)

#: The one spelling of the machine-readable refusal, imported by
#: ``middleware/task_interceptor.py``. ONE token covers both the recon-stage
#: refusal and the allowlist refusal: the proposition is identical ("this
#: caller may not record deterministic provenance") and splitting it would
#: force consumers to enumerate spellings of the same verdict.
DETERMINISTIC_CALLER_ERROR_TYPE = 'DeterministicProvenanceCallerNotPermitted'


@dataclass(frozen=True)
class DoneProvenanceAuthzDecision:
    """The gate's verdict — a value, never an exception (INV-1).

    A denial is returned as structured data so the validator can hand the
    caller a machine-readable rejection (``error_type`` + human-readable
    ``error``) instead of raising through the MCP boundary.
    """

    allowed: bool
    error_type: str | None = None
    error: str | None = None


def _reconciliation_attr(config: Any, attr: str) -> Any:
    """Defensively navigate ``config.reconciliation.<attr>``.

    ``getattr`` at each hop with a ``None`` default, mirroring
    ``server/mem0_update_authz.py::_mem0_update_attr``, so a missing
    ``config``, a missing ``reconciliation`` section, or an unspecced test
    double never raises. Type validation is the caller's job — an unspecced
    ``Mock`` returns a Mock here, which the strict ``isinstance`` check below
    rejects.
    """
    reconciliation = getattr(config, 'reconciliation', None)
    return getattr(reconciliation, attr, None)


def resolve_deterministic_provenance_allowed_prefixes(config: Any) -> tuple[str, ...]:
    """Read the caller allowlist live off the shared config object.

    Returns :data:`_DEFAULT_ALLOWED_PREFIXES` (empty → deny everyone) unless
    the leaf is a real ``list`` of real ``str``. The ``isinstance(value, list)``
    check is load-bearing rather than defensive boilerplate: a bare STRING
    would still satisfy the ``startswith`` call below, so accepting one would
    silently treat the whole string as a single prefix — a mis-typed config
    value that reads as working while gating on something the operator never
    wrote.

    Empty and non-``str`` members are dropped rather than kept: an empty
    prefix would match EVERY caller, turning a typo into a silently open bar.
    """
    value = _reconciliation_attr(config, 'deterministic_provenance_allowed_agent_prefixes')
    if not isinstance(value, list):
        return _DEFAULT_ALLOWED_PREFIXES
    return tuple(p for p in value if isinstance(p, str) and p)


def resolve_deterministic_provenance_authorization(
    config: Any,
    *,
    agent_id: Any,
) -> DoneProvenanceAuthzDecision:
    """Decide whether *agent_id* may record `deterministic-*` provenance.

    Never raises. A non-``str`` or empty *agent_id* is denied on the same
    footing as an unlisted one: an unknown caller is not a permitted caller,
    and the empty list is the operator's live kill switch for the whole
    provenance family.

    This is only ONE of the two bars PRD D11 specifies. The other — an
    unconditional refusal for a ``recon-stage-`` caller, which no config edit
    may defeat — is evaluated FIRST and separately in
    ``middleware/task_interceptor.py::_validate_done_provenance``, because it
    is config-free by design.
    """
    prefixes = resolve_deterministic_provenance_allowed_prefixes(config)
    if (
        isinstance(agent_id, str)
        and agent_id
        and any(agent_id.startswith(prefix) for prefix in prefixes)
    ):
        return DoneProvenanceAuthzDecision(allowed=True)

    return DoneProvenanceAuthzDecision(
        allowed=False,
        error_type=DETERMINISTIC_CALLER_ERROR_TYPE,
        error=(
            f'caller {agent_id!r} may not record a deterministic-* '
            f'done_provenance. Permitted caller prefixes: {list(prefixes)!r} '
            f'(config {_CONFIG_PATH}). These kinds assert a machine '
            'observation made by DeterministicRunner, so the default bar is '
            'deliberately narrow; an empty list denies every caller.'
        ),
    )
