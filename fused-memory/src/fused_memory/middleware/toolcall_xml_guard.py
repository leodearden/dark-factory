"""Tool-call XML leak guard (task 3083).

Rejects an inbound MCP write whose text carries a leaked serialized tool-call
fragment, BEFORE any persistence happens. Wired into the ``submit_task``,
``update_task``, ``add_memory``, and ``add_episode`` prologues in
``server/tools.py``, alongside ``premise_lint_error``, ``lock_charter_error``,
``model_overrides_error``, and ``execution_class_error``.

## Why this exists

``scripts/scan_task_toolcall_leaks.py`` has detected this corruption since
task 2939 — but it is wired nowhere, so it can only report leaks
out-of-band, long after the corrupted text was persisted. This guard is the
same detector (``fused_memory.utils.toolcall_xml_leak``, the single source of
truth) applied at the receiving boundary, where the write can still be
refused.

The markers it looks for never originate in this repository; see the shared
module's docstring for the negative evidence. Their presence in an inbound
argument is positive evidence that the HARNESS's tool-call XML parser
terminated a string argument early at a literal closing tag inside that
argument's value — which means the damage is not confined to the field you
can see.

## Contract

``toolcall_xml_error(*, metadata=None, agent_id=None, **fields)`` returns a
structured error dict (``error_type='ToolCallXmlLeakError'``) naming the
offending field(s), the matched fragment, and its character offset, or
``None`` when every field is clean. A non-str field value (``None``, an int,
a dict) is treated as clean rather than raising, so callers can pass
unvalidated arguments straight through.

Each field is scanned INDEPENDENTLY — one detector call per field — rather
than concatenated into a single string first, mirroring
``premise_lint_guard``'s cross-field invariant. A match must therefore be
wholly contained within a single field: a leak cannot be assembled by
straddling the join between two otherwise-clean fields (e.g. a ``description``
ending at a closing tag plus a ``details`` beginning with a serialized
opening tag must NOT combine into a spurious cross-field rejection).

## Design commitment 1: it REJECTS, it does not sanitize

Stripping the fragment automatically would be a SECOND silent mutation of
user content layered on top of the first — precisely the failure class this
guard exists to kill ("a silent default to a wrong value produces
plausible-looking wrong state rather than an error"). Rejection also carries
information the caller genuinely needs, which is why the message says so
explicitly: a sentinel in ``description`` means the arguments that FOLLOWED
it in the same tool call — canonically ``priority`` — may never have reached
this boundary at all, and would then have silently defaulted (``priority or
'medium'``). The caller must re-send the whole call, not accept a partially
corrupted write.

## Design commitment 2: NOT scoped to recon callers

``premise_lint_error`` fires only for ``recon-stage-*`` callers because the
false premises it lints are a recon-authored failure mode. This leak is
caller-agnostic — it is a property of the harness serialization boundary, not
of any agent — so scoping it would leave every other write path unguarded.
``agent_id`` is accepted purely to mirror the sibling guards' call shape and
is deliberately ignored; it is an identity rather than caller content, so it
is never itself scanned as a text field.

## The opt-out

``metadata={'allow_toolcall_xml': True}`` suppresses the rejection. This is
load-bearing rather than decorative: legitimately quoting a leak fragment is
exactly what a bug report, a memory documenting this defect, and this task's
own description all have to do — without the escape hatch they would be
unfileable. It deliberately mirrors the established ``allow_near_duplicate``
convention (``server/near_duplicate_guard.py``), including its strict
``is True`` check, so agents already know the idiom and a stringly-typed
value cannot accidentally disable the guard.

## Known limitation: an opted-out write is not re-validated

The opt-out suppresses the REJECTION, not the underlying risk. A caller that
sets the flag is asserting "I meant to write this text"; it is not evidence
that the call's sibling arguments survived. If a genuinely truncated call is
submitted with the flag set, the sibling-argument loss is still silent. This
is accepted deliberately: the alternative — refusing every write that quotes
the shape — makes documenting the bug impossible, and the flag at least
leaves an auditable marker in the stored metadata.

## Sentinel-literal hazard

This module contains NO sentinel literals of its own; the pattern lives in
``fused_memory.utils.toolcall_xml_leak``, where it is spelled with the
``\\x3c`` escape for ``<``. Do not paste a verbatim tool-call literal into
this file — see that module's docstring for why.
"""
from __future__ import annotations

from typing import Any

from fused_memory.utils.toolcall_xml_leak import LeakHit, find_toolcall_xml_leak

__all__ = [
    'ALLOW_TOOLCALL_XML_KEY',
    'toolcall_xml_error',
]

# Mirrors metadata={'allow_near_duplicate': True} (near_duplicate_guard.py).
ALLOW_TOOLCALL_XML_KEY = 'allow_toolcall_xml'

_HINT = (
    'Do NOT simply strip the fragment and resend that one field: the fragment '
    'is evidence the harness truncated the tool call, so re-send the ENTIRE '
    'call and verify every argument (especially ones that followed the '
    'affected field) arrived with the value you intended. If you are '
    'deliberately quoting this shape — a bug report, a memory documenting the '
    "leak — override with metadata={'allow_toolcall_xml': True}."
)


def toolcall_xml_error(
    *,
    metadata: Any = None,
    agent_id: Any = None,
    **fields: Any,
) -> dict[str, Any] | None:
    """Reject *fields* carrying a leaked tool-call XML fragment.

    Returns ``None`` when every field is clean (the common case), else a
    structured error dict for the tools layer to return verbatim:

    * ``error`` — human-readable message naming the offending field(s) and
      warning that sibling parameters may have been silently dropped;
    * ``error_type`` — always ``'ToolCallXmlLeakError'``;
    * ``hint`` — remediation, including the opt-out;
    * ``field`` / ``fields`` — the first and all offending field names, in
      the caller's keyword order;
    * ``fragment`` / ``offset`` — the first offending field's matched
      fragment and its 0-based character offset within that field.

    Args:
        metadata: The write's metadata. When it is a dict whose
            ``'allow_toolcall_xml'`` value ``is True``, the rejection is
            suppressed. Any other shape (``None``, a JSON string, a
            non-dict) is tolerated and simply does not opt out.
        agent_id: Accepted to mirror the sibling guards' call shape and
            DELIBERATELY IGNORED — this leak is caller-agnostic. It is never
            scanned as a text field.
        **fields: Field name -> value, scanned independently in the order
            given. Non-str and falsy values are treated as clean.
    """
    if isinstance(metadata, dict) and metadata.get(ALLOW_TOOLCALL_XML_KEY) is True:
        return None

    offending: list[tuple[str, LeakHit]] = []
    for name, value in fields.items():
        hits = find_toolcall_xml_leak(value)
        if hits:
            offending.append((name, hits[0]))

    if not offending:
        return None

    names = [name for name, _hit in offending]
    first_name, first_hit = offending[0]
    field_list = ', '.join(names)
    return {
        'error': (
            f'Leaked serialized tool-call XML detected in: {field_list}. This '
            f'text was never written by fused-memory, so its presence means '
            f'the harness terminated a string argument early — and therefore '
            f'that SIBLING parameters of this same call (canonically '
            f'`priority`, which then silently defaults to "medium") may have '
            f'been dropped before reaching this boundary. The write was '
            f'REJECTED rather than silently cleaned, because auto-stripping '
            f'would layer a second silent mutation on the first. Offending '
            f'fragment in `{first_name}` at offset {first_hit.index}: '
            f'{first_hit.fragment!r}'
        ),
        'error_type': 'ToolCallXmlLeakError',
        'hint': _HINT,
        'field': first_name,
        'fields': names,
        'fragment': first_hit.fragment,
        'offset': first_hit.index,
    }
