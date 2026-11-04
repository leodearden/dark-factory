"""Loud-and-safe MCP envelope parser.

Provides a single shared primitive that replaces the silent ``parse_guard_fallthrough``
family (family A in the remediation doc).  Every abnormal envelope shape produces a
distinct ``logger.warning`` and a non-``None`` ``EnvelopeParseError`` — guaranteeing
that callers can always distinguish "got a value" from "something went wrong".

Public API::

    from shared.mcp_envelope import (
        parse_tool_result,
        resolver_failed,
        write_failure_reason,
        EnvelopeParseError,
        EnvelopeShape,
    )

The module is intentionally NOT re-exported from ``shared/__init__.py``.
Consumers import via the fully-qualified path (see above), consistent with the
``proc_group``/``config_dir``/``pytest_jobserver`` sub-module convention.
"""

from __future__ import annotations

import enum
import json
import logging
from typing import Any

__all__ = [
    'parse_tool_result',
    'resolver_failed',
    'write_failure_reason',
    'EnvelopeParseError',
    'EnvelopeShape',
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EnvelopeShape — closed set of abnormal shapes
# ---------------------------------------------------------------------------


class EnvelopeShape(enum.StrEnum):
    """Closed set of abnormal MCP envelope shapes.

    Mirrors the ``AgentFailureKind`` pattern in ``cli_invoke.py``.
    """

    NO_TEXT_BLOCK = 'NO_TEXT_BLOCK'
    """No ``type=='text'`` block found in ``result.content``."""

    JSON_DECODE_ERROR = 'JSON_DECODE_ERROR'
    """Text block found but ``json.loads`` raised ``ValueError``/``TypeError``."""

    INNER_NOT_DICT = 'INNER_NOT_DICT'
    """Parsed payload (after ``{data:{…}}`` unwrap) is not a ``dict``."""

    KEY_ABSENT = 'KEY_ABSENT'
    """Inner dict is a dict but the requested key is absent."""

    WRONG_TYPE = 'WRONG_TYPE'
    """``inner[key]`` is present but not an instance of ``expected_type``."""

    RAISED = 'RAISED'
    """An unexpected exception was raised during parsing."""

    IS_ERROR = 'IS_ERROR'
    """``result.isError`` is set; an error reply is never a value."""

    JSON_RPC_ERROR = 'JSON_RPC_ERROR'
    """The JSON-RPC body carries an ``error`` member, so there is no tool result."""


# ---------------------------------------------------------------------------
# EnvelopeParseError
# ---------------------------------------------------------------------------


class EnvelopeParseError(Exception):
    """Raised (or returned) whenever ``parse_tool_result`` cannot produce a value.

    Attributes:
        shape: The ``EnvelopeShape`` member that describes the failure.
        key: The key that was being looked up when the failure occurred.
        payload_prefix: A bounded (≤200 char) repr of the relevant payload
            fragment, suitable for log messages.
    """

    def __init__(self, shape: EnvelopeShape, *, key: str | None, payload_prefix: str) -> None:
        self.shape = shape
        self.key = key
        self.payload_prefix = payload_prefix
        super().__init__(f'MCP envelope parse error: shape={shape!r} key={key!r}')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(
    shape: EnvelopeShape,
    key: str | None,
    payload: Any,
) -> tuple[None, EnvelopeParseError]:
    """Emit ONE distinct WARNING and return ``(None, EnvelopeParseError(…))``."""
    prefix = repr(payload)[:200]
    logger.warning(
        'MCP envelope parse failed (shape=%s, key=%r): %s',
        shape,
        key,
        prefix,
    )
    err = EnvelopeParseError(shape, key=key, payload_prefix=prefix)
    return None, err


# ---------------------------------------------------------------------------
# parse_tool_result
# ---------------------------------------------------------------------------


def _first_text_block(content: Any) -> dict | None:
    if not isinstance(content, list):
        return None
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'text':
            return block
    return None


def _payload_of(result: dict, key: str | None) -> tuple[Any, EnvelopeParseError | None]:
    """The tool's decoded payload: a dict ``structuredContent``, else the first text block."""
    if result.get('error') is not None:
        return _fail(EnvelopeShape.JSON_RPC_ERROR, key, result)
    inner_result = result.get('result')
    if not isinstance(inner_result, dict):
        return _fail(EnvelopeShape.NO_TEXT_BLOCK, key, result)
    if inner_result.get('isError'):
        return _fail(EnvelopeShape.IS_ERROR, key, inner_result)

    structured = inner_result.get('structuredContent')
    if isinstance(structured, dict):
        return structured, None

    text_block = _first_text_block(inner_result.get('content', []))
    if text_block is None:
        return _fail(EnvelopeShape.NO_TEXT_BLOCK, key, result)
    raw_text = text_block.get('text', '')
    try:
        return json.loads(raw_text), None
    except (ValueError, TypeError):
        return _fail(EnvelopeShape.JSON_DECODE_ERROR, key, raw_text)


def parse_tool_result(
    result: Any,
    key: str | None,
    expected_type: type | tuple[type, ...],
) -> tuple[Any | None, EnvelopeParseError | None]:
    """Extract *key* from a standard MCP tool-result envelope.

    **Standard mode** (``key`` is a non-``None`` string):

    Returns ``(value, None)`` **only** when:
    - *result* is a dict with no JSON-RPC ``error`` member (``JSON_RPC_ERROR``)
      and whose ``result.isError`` is not set (``IS_ERROR``) — both are
      checked before anything else, so an error reply is never a value,
    - the payload is ``result.structuredContent`` when that is a dict, or
      otherwise ``json.loads`` of the first ``type=='text'`` block's ``text``,
    - the payload (after optional ``{data:{…}}`` unwrap) is a dict,
    - *key* is present in that dict, and
    - ``isinstance(inner[key], expected_type)`` holds.

    **Whole-inner-dict mode** (``key=None``):

    Returns ``(inner, None)`` after the ``{data:{…}}`` unwrap step, provided
    the inner payload is a dict and ``isinstance(inner, expected_type)`` holds
    (callers should pass ``expected_type=dict``; a mismatched type fails with
    ``WRONG_TYPE``).  ``KEY_ABSENT`` does not apply — the entire inner dict is
    returned as-is.  ``JSON_RPC_ERROR``, ``IS_ERROR``, ``NO_TEXT_BLOCK``,
    ``JSON_DECODE_ERROR``, ``INNER_NOT_DICT``, ``WRONG_TYPE``, and ``RAISED``
    still fail loud.

    For **every** other outcome (including non-raising malformed payloads)
    exactly **one** ``logger.warning`` is emitted naming the shape, and the
    tuple ``(None, EnvelopeParseError(shape=…, …))`` is returned.

    The invariant ``err is not None`` whenever ``value is None`` (and vice
    versa) is guaranteed — callers may branch on ``err is not None`` safely.

    When ``structuredContent`` is not a dict, only the first ``type=='text'``
    block is read.
    """
    try:
        if not isinstance(result, dict):
            return _fail(EnvelopeShape.NO_TEXT_BLOCK, key, result)

        data, err = _payload_of(result, key)
        if err is not None:
            return None, err

        # --- {data:{…}} unwrap -------------------------------------------------
        if isinstance(data, dict) and isinstance(data.get('data'), dict):
            inner = data['data']
        else:
            inner = data

        # --- inner must be a dict ----------------------------------------------
        if not isinstance(inner, dict):
            return _fail(EnvelopeShape.INNER_NOT_DICT, key, inner)

        # --- whole-inner-dict mode: key=None returns the entire inner dict -----
        if key is None:
            # Validate expected_type so the parameter retains meaning even when
            # key=None.  inner is already guaranteed a dict by the INNER_NOT_DICT
            # guard above, but an incorrect expected_type (e.g. list) would slip
            # through silently without this check.
            if not isinstance(inner, expected_type):
                return _fail(EnvelopeShape.WRONG_TYPE, key, inner)
            return inner, None

        # --- key presence (membership test — NOT .get) -------------------------
        if key not in inner:
            return _fail(EnvelopeShape.KEY_ABSENT, key, inner)

        # --- type check --------------------------------------------------------
        value = inner[key]
        if not isinstance(value, expected_type):
            return _fail(EnvelopeShape.WRONG_TYPE, key, value)

        return value, None

    except Exception as exc:  # noqa: BLE001 — catch-all for unexpected exceptions
        prefix = repr(exc)[:200]
        logger.warning(
            'MCP envelope parse raised unexpected exception (shape=%s, key=%r): %s',
            EnvelopeShape.RAISED,
            key,
            prefix,
        )
        err = EnvelopeParseError(EnvelopeShape.RAISED, key=key, payload_prefix=prefix)
        err.__cause__ = exc
        return None, err


# ---------------------------------------------------------------------------
# resolver_failed
# ---------------------------------------------------------------------------


def resolver_failed(value: Any, err: Exception | None) -> bool:
    """Fail-safe-WAIT predicate for MCP resolver results.

    Returns ``True`` iff *err* is not ``None`` **or** *value* is falsy.

    An empty resolution (``{}`` / ``[]`` / ``None`` with ``err=None``) is
    conservatively treated as *not yet usable* — the caller should warn and
    wait/retry rather than proceeding.  This is distinct from a fail-CLOSED
    gate: it is non-destructive (the next scheduler tick re-resolves).

    This function is a pure predicate — it emits **no** log records.  The
    caller is responsible for logging and deciding whether to escalate.

    Truth table::

        ({'a': 1}, None)  → False   # got a value, no error
        ({},       None)  → True    # empty → wait
        (None,     None)  → True    # falsy → wait
        (anything, <err>) → True    # error present → wait/escalate
    """
    return err is not None or not value


# ---------------------------------------------------------------------------
# write_failure_reason
# ---------------------------------------------------------------------------


def _clip(value: Any) -> str:
    return repr(value)[:200]


def write_failure_reason(reply: object) -> str | None:
    """Return why *reply* is not a successful write, or ``None`` iff it is one.

    *reply* is the tool's own reply — ``parse_tool_result(body, None, dict)``'s
    value — never the JSON-RPC envelope.  ``@mcp_tool_errors`` turns every
    server-side exception into an ordinary reply, so a rejection arrives here
    as a dict.  ``success`` defaults to True because ``update_task`` answers
    with the updated record, which carries no ``success`` key.
    """
    if not isinstance(reply, dict):
        return f'server reply was not a dict: {type(reply).__name__}: {_clip(reply)}'
    if not reply:
        return 'server reply was empty ({}) — it carries no positive write signal'
    if reply.get('error'):
        error_type = reply.get('error_type')
        named = f'{error_type}: ' if error_type else ''
        return f'server returned an error: {named}{reply["error"]}'
    if not reply.get('success', True):
        detail = reply.get('error_type') or reply.get('message') or _clip(reply)
        return f'server reported failure: {detail}'
    return None
