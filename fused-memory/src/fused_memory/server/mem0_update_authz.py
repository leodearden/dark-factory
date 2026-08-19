"""Authorization gate for the in-place ``update_memory`` tool (task 3088).

In-place content amendment is a SILENT-REWRITE primitive: it changes what a
record says while preserving its id, so nothing downstream can tell the text
changed. Decision doc ``plans/mem0-in-place-update-decision.md`` §4 therefore
ships the tool behind a narrow agent-prefix allowlist plus a kill switch, with
the two arms (content amend, metadata patch) gated independently.

HONEST CAVEAT, carried over verbatim in spirit from ``add_system_record``'s
docstring: ``agent_id`` is SELF-REPORTED by the caller. This gate is a misuse
deterrent for cooperating callers — it stops an agent from casually reaching for
a rewrite primitive it was not meant to touch — not a security boundary. A
determined caller can claim any prefix. Do not describe it as authorization in
the cryptographic sense.

Structure mirrors the sibling :mod:`fused_memory.server.near_duplicate_guard`:
module-level live-read resolvers, a shared defensive attribute navigator, and
module-level defaults for when a config hop is missing. ONE deliberate
inversion: the near-dup resolvers fall back to PERMISSIVE defaults because
fail-open is the safe direction for a soft-block guard, whereas this is a
mutation-authorization gate whose safe direction is DENY. Every fallback here
therefore denies.

LIVE READ (``config/reload.py``'s reload-safety precondition): every resolver
re-reads ``memory_service.config.mem0_update.*`` on EVERY call and captures
nothing at import or construction, which is what makes the five
``mem0_update.*`` leaves genuinely green-tier hot-reloadable rather than
restart-only in disguise.

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
# config defaults (Mem0UpdateConfig's default_factory), so there is exactly one
# spelling of the bar (INV-5). 'recon-stage-' is the same string
# add_system_record and ~15 other sites in server/tools.py already gate on;
# 'curator-' is the dedicated opt-in prefix for the interactive consolidation
# sitting (esc-3524-1, skills/curate-fused-memories).


@dataclass(frozen=True)
class Mem0UpdateAuthzDecision:
    """The gate's verdict — a value, never an exception (INV-1).

    A denial is returned as structured data so the calling tool can hand the
    caller a machine-readable rejection (``error_type`` + human-readable
    ``error``) instead of raising through the MCP boundary.
    """

    allowed: bool
    error_type: str | None = None
    error: str | None = None


def _mem0_update_attr(memory_service: Any, attr: str) -> Any:
    """Defensively navigate ``memory_service.config.mem0_update.<attr>``.

    ``getattr`` at each hop with a ``None`` default, mirroring
    ``near_duplicate_guard._reconciliation_attr``, so a missing ``config``, a
    missing ``mem0_update`` section, or an unspecced test double never raises.
    Type validation is the caller's job — an unspecced ``Mock`` returns a Mock
    here, which the strict ``isinstance`` checks below reject.
    """
    config = getattr(memory_service, 'config', None)
    mem0_update = getattr(config, 'mem0_update', None)
    return getattr(mem0_update, attr, None)


def resolve_mem0_update_enabled(memory_service: Any) -> bool:
    """Read the ``update_memory`` kill switch live off the shared config.

    Returns :data:`_DEFAULT_ENABLED` (``False``) unless the leaf is a real
    ``bool`` — a missing section, a ``None`` hop, a string ``'yes'`` or a Mock
    attribute all deny.
    """
    value = _mem0_update_attr(memory_service, 'enabled')
    if isinstance(value, bool):
        return value
    return _DEFAULT_ENABLED


def resolve_mem0_update_allowed_prefixes(
    memory_service: Any,
    arm: str,
) -> tuple[str, ...]:
    """Read one arm's agent-prefix allowlist live off the shared config.

    *arm* is ``'content_amend'`` or ``'metadata_patch'``, naming which of the two
    independently-configurable lists to read.

    Returns :data:`_DEFAULT_ALLOWED_PREFIXES` (empty → deny everyone) unless the
    leaf is a real ``list`` of real ``str``. The ``isinstance(value, list)``
    check is load-bearing rather than defensive boilerplate: a bare STRING would
    still satisfy the ``startswith`` call below, so accepting one would silently
    treat the whole string as a single prefix — a mis-typed config value that
    reads as working while gating on something the operator never wrote.
    """
    value = _mem0_update_attr(memory_service, f'{arm}_allowed_agent_prefixes')
    if not isinstance(value, list):
        return _DEFAULT_ALLOWED_PREFIXES
    return tuple(p for p in value if isinstance(p, str) and p)


def _agent_matches(agent_id: Any, prefixes: tuple[str, ...]) -> bool:
    """True iff *agent_id* is a non-empty ``str`` matching one of *prefixes*."""
    if not isinstance(agent_id, str) or not agent_id:
        return False
    return any(agent_id.startswith(prefix) for prefix in prefixes)


def resolve_mem0_update_authorization(
    memory_service: Any,
    *,
    agent_id: Any,
    content_amend: bool,
    metadata_patch: bool,
) -> Mem0UpdateAuthzDecision:
    """Decide whether *agent_id* may perform the requested arms. Never raises.

    *content_amend* / *metadata_patch* say which arms the call is REQUESTING.
    Each is checked against its own allowlist, and a call requesting both must
    clear BOTH bars — the metadata bar being wider (the supported way to admit an
    interactive curator-gate tagging flow) must not become a back door to content
    amendment.

    The kill switch is evaluated FIRST and outranks everything: when
    ``mem0_update.enabled`` is false, every caller is denied with
    ``error_type='Mem0UpdateToolDisabled'`` regardless of ``agent_id``, so an
    operator has one knob that reliably stops an in-flight incident.

    A request for NO arm is denied. There is nothing to authorize, and arm
    validation is a separate layer — an empty request must not slip past this one
    on the grounds that it asks for nothing.
    """
    if not resolve_mem0_update_enabled(memory_service):
        return Mem0UpdateAuthzDecision(
            allowed=False,
            error_type='Mem0UpdateToolDisabled',
            error=(
                'The in-place update_memory tool is disabled '
                '(config mem0_update.enabled=false). No caller may amend content '
                'or patch metadata while it is off; ask an operator to re-enable '
                'it (green-tier hot-reloadable, no restart needed).'
            ),
        )

    if not content_amend and not metadata_patch:
        return Mem0UpdateAuthzDecision(
            allowed=False,
            error_type='Mem0UpdateNoArmRequested',
            error=(
                'update_memory authorization was requested for no arm: supply '
                'content (amend) and/or metadata_patch/metadata_delete_keys.'
            ),
        )

    for arm, requested, knob in (
        ('content_amend', content_amend, 'content_amend_allowed_agent_prefixes'),
        ('metadata_patch', metadata_patch, 'metadata_patch_allowed_agent_prefixes'),
    ):
        if not requested:
            continue
        prefixes = resolve_mem0_update_allowed_prefixes(memory_service, arm)
        if not _agent_matches(agent_id, prefixes):
            return Mem0UpdateAuthzDecision(
                allowed=False,
                error_type='Mem0UpdateNotAuthorized',
                error=(
                    f'agent_id {agent_id!r} is not authorized for the '
                    f'{arm} arm of update_memory. Authorized prefixes: '
                    f'{list(prefixes)!r} (config mem0_update.{knob}). '
                    'In-place amendment is a silent-rewrite primitive, so the '
                    'default bar is deliberately narrow.'
                ),
            )

    return Mem0UpdateAuthzDecision(allowed=True)
