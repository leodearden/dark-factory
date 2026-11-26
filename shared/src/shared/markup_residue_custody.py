"""Custody of a markup residue payload: per residue kind, the words every
residue sink states about the call and about who still holds the payload,
plus the last log floor for a payload no store took."""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from shared.mcp_markup_middleware import (
    MARKUP_RECURRENCE_ROUTING,
    MARKUP_UNRECOVERED_PARAM_ERROR_TYPE,
    MARKUP_UNREPAIRABLE_ERROR_TYPE,
    residue_payload_length,
)

#: All a ``MARKUP_UNRECOVERED_PARAM_ERROR_TYPE`` residue record can say about
#: its call's outcome. :meth:`MarkupGuardMiddleware._handle_markup` files the
#: dropped slice BEFORE it forwards the call, so no sink can know whether the
#: call then succeeded.
MARKUP_UNRECOVERED_PARAM_OUTCOME_CAVEAT = (
    'This record was filed BEFORE the call ran, so whether the call then '
    'succeeded is not known here; if it did, whatever it stored is missing '
    'this value.'
)

#: All a ``MARKUP_UNRECOVERED_PARAM_ERROR_TYPE`` residue record can say about
#: custody of its payload. Unlike a refusal, nothing bounced the caller, and
#: the drop rode only on the forwarded result's ``markup_repair`` meta.
MARKUP_UNRECOVERED_PARAM_CUSTODY_CAVEAT = (
    'Nothing bounced the caller: its call went through, and the drop was '
    'reported only in the result metadata, which an agent may never read. Do '
    'not wait on a resend of this value; the payload below is the only copy '
    'this boundary holds.'
)

#: All a ``MARKUP_UNREPAIRABLE_ERROR_TYPE`` residue record can say about
#: custody of its payload. The guard refuses at the boundary and reads nothing
#: back, so it knows the call wrote nothing and that its caller was told to
#: resend, never whether the caller did.
MARKUP_REFUSAL_CUSTODY_CAVEAT = (
    'This call wrote nothing, and its caller was told to resend from its own '
    'copy. Whether it did resend is not known here: a refusal proves one '
    'failed call, not lost data. Before recovering this payload, look at the '
    "call's destination for a later write of the same content, because "
    'recovering an already-resent payload writes a duplicate.'
)


class ResidueCall(enum.Enum):
    """What the boundary did with the call a residue record came from."""

    REFUSED = 'refused'
    FORWARDED_WITHOUT_PARAM = 'forwarded_without_param'
    UNRECOGNISED = 'unrecognised'


@dataclass(frozen=True)
class ResidueProse:
    """What a residue sink states: the call's outcome, then the payload's custody."""

    call: ResidueCall
    outcome: str
    custody: str


_KNOWN_KIND_PROSE: Mapping[str, ResidueProse] = {
    MARKUP_UNREPAIRABLE_ERROR_TYPE: ResidueProse(
        call=ResidueCall.REFUSED,
        outcome=(
            "THE CALL WAS REFUSED. The leaked fragment's own boundary could not "
            'be determined, so no repair was attempted: guessing a boundary '
            'would have silently dropped whatever arguments hide in the residue.'
        ),
        custody=MARKUP_REFUSAL_CUSTODY_CAVEAT,
    ),
    MARKUP_UNRECOVERED_PARAM_ERROR_TYPE: ResidueProse(
        call=ResidueCall.FORWARDED_WITHOUT_PARAM,
        outcome=(
            'THE CALL WAS FORWARDED WITHOUT THIS PARAMETER. The value below was '
            "recovered from the leak but could not be typed against the parameter's "
            'declared schema, so it was dropped from the call rather than passed '
            f'through untyped. {MARKUP_UNRECOVERED_PARAM_OUTCOME_CAVEAT}'
        ),
        custody=MARKUP_UNRECOVERED_PARAM_CUSTODY_CAVEAT,
    ),
}

_UNRECOGNISED_CUSTODY = (
    'Whether any other party still holds this payload is not known here, so '
    'treat the copy below as possibly the only one.'
)


def _unrecognised_prose(record: Mapping[str, Any]) -> ResidueProse:
    outcome = '\n'.join([
        f'UNRECOGNISED RESIDUE KIND {record.get("error_type")!r}. This sink does '
        "not know what this call's outcome was, so it states neither — the "
        'middleware grew a record kind that '
        'shared/src/shared/markup_residue_custody.py::residue_prose has no '
        'text for. Its own '
        'words for what happened follow; treat them, not this paragraph, as the '
        'account of the call.',
        '',
        f'summary: {record.get("summary")!r}',
        f'suggested_action: {record.get("suggested_action")!r}',
    ])
    return ResidueProse(
        call=ResidueCall.UNRECOGNISED, outcome=outcome, custody=_UNRECOGNISED_CUSTODY
    )


def residue_prose(record: Mapping[str, Any]) -> ResidueProse:
    """The outcome and custody text for *record*'s residue kind.

    A kind with no text here is still rendered, hedged and carrying the
    middleware's own ``summary`` and ``suggested_action``, so a sink files it
    visibly instead of dropping it.
    """
    error_type = record.get('error_type')
    known = _KNOWN_KIND_PROSE.get(error_type) if isinstance(error_type, str) else None
    return known if known is not None else _unrecognised_prose(record)


#: The greppable first token of :func:`log_unpreserved_residue`'s line.
RESIDUE_UNPRESERVED_LOG_EVENT = 'markup_residue_unpreserved'


def log_unpreserved_residue(
    log: logging.Logger,
    record: Mapping[str, Any],
    *,
    why: str,
    exc_info: bool = False,
    extra: Mapping[str, object] | None = None,
) -> None:
    """The last floor under a residue sink, for a host whose log is durable.

    Meant for a long-lived service under journald. The whole record rides as
    data in ``residue_record`` and as text in the message, which journald keeps.
    """
    whole = dict(record)
    log.error(
        '%s: the %r residue of %s.%s (%d chars) was preserved nowhere durable '
        'because %s, so this line is its copy. record=%r; %s',
        RESIDUE_UNPRESERVED_LOG_EVENT, record.get('error_type'),
        record.get('tool'), record.get('field'), residue_payload_length(record),
        why, whole, MARKUP_RECURRENCE_ROUTING,
        exc_info=exc_info,
        extra={'residue_record': whole, **(extra or {})},
    )
