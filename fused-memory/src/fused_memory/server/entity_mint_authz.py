"""Authorization gate for the ``ensure_entity_node`` MCP tool (task 4932).

Minting a Graphiti Entity node is a write-time-IDENTITY primitive. A node
minted under a NON-CANONICAL name splits a referent instead of resolving it, and
nothing in this system sweeps orphan minted nodes — so the damage is quiet,
cumulative, and only visible to whoever later tries to resolve the referent. The
tool therefore ships behind a narrow agent-prefix allowlist plus a kill switch,
exactly as the sibling :mod:`fused_memory.server.mem0_update_authz` gates the
in-place ``update_memory`` tool.

HONEST CAVEAT, carried over verbatim in spirit from ``add_system_record``'s
docstring and from the sibling module: ``agent_id`` is SELF-REPORTED by the
caller. This gate is a misuse deterrent for cooperating callers — it stops an
agent from casually reaching for an identity-write primitive it was not meant to
touch — NOT a security boundary. A determined caller can claim any prefix. Do
not describe it as authorization in the cryptographic sense.

Structure mirrors :mod:`fused_memory.server.mem0_update_authz`: module-level
live-read resolvers, a shared defensive attribute navigator, and module-level
defaults for when a config hop is missing. Those defaults DENY, inverting
:mod:`fused_memory.server.near_duplicate_guard`'s permissive fallbacks — that
module fails open because fail-open is the safe direction for a soft-block
guard, whereas this is a mutation-authorization gate whose safe direction is
deny.

LIVE READ (``config/reload.py``'s reload-safety precondition): every resolver
re-reads ``memory_service.config.entity_mint.*`` on EVERY call and captures
nothing at import or construction, which is what makes the ``entity_mint.*``
leaves genuinely green-tier hot-reloadable rather than restart-only in disguise.
For ``enabled`` that is not a nicety: a restart-only kill switch is no kill
switch.

Lives in its own module rather than inline in ``server/tools.py`` because that
file's tool registration is a closure: a guard body added there would be neither
importable nor independently testable, and the live-read proof test must call
the resolver directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Fail-CLOSED module defaults, used whenever a config hop is missing, None, or
# the wrong type. Deny, never permit — see the module docstring's inversion note.
_DEFAULT_ENABLED: bool = False
_DEFAULT_ALLOWED_PREFIXES: tuple[str, ...] = ()

# Neither allowlisted prefix literal is minted here: both reach this module as
# config defaults (EntityMintConfig's default_factory), so there is exactly one
# spelling of the bar (INV-5). 'recon-stage-' is the same string
# add_system_record, the mem0_update bars and ~15 other sites in server/tools.py
# already gate on; 'curator-' is the dedicated opt-in prefix for the interactive
# consolidation sitting, which is the flow that discovers a dangling referent
# and needs the node it points at.


@dataclass(frozen=True)
class EntityMintAuthzDecision:
    """The gate's verdict — a value, never an exception (INV-1).

    A denial is returned as structured data so the calling tool can hand the
    caller a machine-readable rejection (``error_type`` + human-readable
    ``error``) instead of raising through the MCP boundary.
    """

    allowed: bool
    error_type: str | None = None
    error: str | None = None


def _entity_mint_attr(memory_service: Any, attr: str) -> Any:
    """Defensively navigate ``memory_service.config.entity_mint.<attr>``.

    ``getattr`` at each hop with a ``None`` default, mirroring
    ``mem0_update_authz._mem0_update_attr`` and
    ``near_duplicate_guard._reconciliation_attr``, so a missing ``config``, a
    missing ``entity_mint`` section, or an unspecced test double never raises.
    Type validation is the caller's job — an unspecced ``Mock`` returns a Mock
    here, which the strict ``isinstance`` checks below reject.
    """
    config = getattr(memory_service, 'config', None)
    entity_mint = getattr(config, 'entity_mint', None)
    return getattr(entity_mint, attr, None)


def resolve_entity_mint_enabled(memory_service: Any) -> bool:
    """Read the ``ensure_entity_node`` kill switch live off the shared config.

    Returns :data:`_DEFAULT_ENABLED` (``False``) unless the leaf is a real
    ``bool`` — a missing section, a ``None`` hop, a string ``'yes'`` or a Mock
    attribute all deny.
    """
    value = _entity_mint_attr(memory_service, 'enabled')
    if isinstance(value, bool):
        return value
    return _DEFAULT_ENABLED


def resolve_entity_mint_allowed_prefixes(memory_service: Any) -> tuple[str, ...]:
    """Read the agent-prefix allowlist live off the shared config.

    Returns :data:`_DEFAULT_ALLOWED_PREFIXES` (empty → deny everyone) unless the
    leaf is a real ``list`` of real ``str``. The ``isinstance(value, list)``
    check is load-bearing rather than defensive boilerplate: a bare STRING would
    still satisfy the ``startswith`` call below, so accepting one would silently
    treat the whole string as a single prefix — a mis-typed config value that
    reads as working while gating on something the operator never wrote.
    """
    value = _entity_mint_attr(memory_service, 'allowed_agent_prefixes')
    if not isinstance(value, list):
        return _DEFAULT_ALLOWED_PREFIXES
    return tuple(p for p in value if isinstance(p, str) and p)


def _agent_matches(agent_id: Any, prefixes: tuple[str, ...]) -> bool:
    """True iff *agent_id* is a non-empty ``str`` matching one of *prefixes*."""
    if not isinstance(agent_id, str) or not agent_id:
        return False
    return any(agent_id.startswith(prefix) for prefix in prefixes)


def resolve_entity_mint_authorization(
    memory_service: Any,
    *,
    agent_id: Any,
) -> EntityMintAuthzDecision:
    """Decide whether *agent_id* may mint an Entity node. Never raises.

    The kill switch is evaluated FIRST and outranks everything: when
    ``entity_mint.enabled`` is false, every caller is denied with
    ``error_type='EntityMintToolDisabled'`` regardless of ``agent_id``, so an
    operator has one knob that reliably stops a runaway minter — and because the
    leaf is read live here, flipping it denies the very NEXT call with no
    restart.
    """
    if not resolve_entity_mint_enabled(memory_service):
        return EntityMintAuthzDecision(
            allowed=False,
            error_type='EntityMintToolDisabled',
            error=(
                'The ensure_entity_node tool is disabled '
                '(config entity_mint.enabled=false). No caller may mint an '
                'Entity node while it is off; ask an operator to re-enable it '
                '(green-tier hot-reloadable, no restart needed).'
            ),
        )

    prefixes = resolve_entity_mint_allowed_prefixes(memory_service)
    if not _agent_matches(agent_id, prefixes):
        return EntityMintAuthzDecision(
            allowed=False,
            error_type='EntityMintNotAuthorized',
            error=(
                f'agent_id {agent_id!r} is not authorized to mint an Entity '
                f'node. Authorized prefixes: {list(prefixes)!r} '
                '(config entity_mint.allowed_agent_prefixes). Minting is a '
                'write-time-identity primitive — a node minted under the wrong '
                'name splits a referent instead of resolving it, and nothing '
                'sweeps orphan minted nodes — so the default bar is '
                'deliberately narrow.'
            ),
        )

    return EntityMintAuthzDecision(allowed=True)
