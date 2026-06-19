"""Loud-and-safe MCP envelope parser.

Provides a single shared primitive that replaces the silent ``parse_guard_fallthrough``
family (family A in the remediation doc).  Every abnormal envelope shape produces a
distinct ``logger.warning`` and a non-``None`` ``EnvelopeParseError`` — guaranteeing
that callers can always distinguish "got a value" from "something went wrong".

Public API::

    from shared.mcp_envelope import (
        parse_tool_result,
        resolver_failed,
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
    "parse_tool_result",
    "resolver_failed",
    "EnvelopeParseError",
    "EnvelopeShape",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EnvelopeShape — closed set of abnormal shapes
# ---------------------------------------------------------------------------


class EnvelopeShape(enum.StrEnum):
    """Closed set of abnormal MCP envelope shapes.

    Mirrors the ``AgentFailureKind`` pattern in ``cli_invoke.py``.
    """

    NO_TEXT_BLOCK = "NO_TEXT_BLOCK"
    """No ``type=='text'`` block found in ``result.content``."""

    JSON_DECODE_ERROR = "JSON_DECODE_ERROR"
    """Text block found but ``json.loads`` raised ``ValueError``/``TypeError``."""

    INNER_NOT_DICT = "INNER_NOT_DICT"
    """Parsed payload (after ``{data:{…}}`` unwrap) is not a ``dict``."""

    KEY_ABSENT = "KEY_ABSENT"
    """Inner dict is a dict but the requested key is absent."""

    WRONG_TYPE = "WRONG_TYPE"
    """``inner[key]`` is present but not an instance of ``expected_type``."""

    RAISED = "RAISED"
    """An unexpected exception was raised during parsing."""


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

    def __init__(self, shape: EnvelopeShape, *, key: str, payload_prefix: str) -> None:
        self.shape = shape
        self.key = key
        self.payload_prefix = payload_prefix
        super().__init__(f"MCP envelope parse error: shape={shape!r} key={key!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(
    shape: EnvelopeShape,
    key: str,
    payload: Any,
) -> tuple[None, EnvelopeParseError]:
    """Emit ONE distinct WARNING and return ``(None, EnvelopeParseError(…))``."""
    prefix = repr(payload)[:200]
    logger.warning(
        "MCP envelope parse failed (shape=%s, key=%r): %s",
        shape,
        key,
        prefix,
    )
    err = EnvelopeParseError(shape, key=key, payload_prefix=prefix)
    return None, err


# ---------------------------------------------------------------------------
# parse_tool_result
# ---------------------------------------------------------------------------


def parse_tool_result(
    result: Any,
    key: str | None,
    expected_type: type | tuple[type, ...],
) -> tuple[Any | None, EnvelopeParseError | None]:
    """Extract *key* from a standard MCP tool-result envelope.

    **Standard mode** (``key`` is a non-``None`` string):

    Returns ``(value, None)`` **only** when:
    - *result* is a dict with ``result.content`` containing at least one
      ``type=='text'`` block,
    - ``json.loads`` succeeds on its ``text`` field,
    - the parsed payload (after optional ``{data:{…}}`` unwrap) is a dict,
    - *key* is present in that dict, and
    - ``isinstance(inner[key], expected_type)`` holds.

    **Whole-inner-dict mode** (``key=None``):

    Returns ``(inner, None)`` after the ``{data:{…}}`` unwrap step, provided
    the inner payload is a dict (``expected_type=dict`` is implied by the
    INNER_NOT_DICT guard).  ``KEY_ABSENT`` and ``WRONG_TYPE`` do not apply —
    the entire inner dict is returned as-is.  ``NO_TEXT_BLOCK``,
    ``JSON_DECODE_ERROR``, ``INNER_NOT_DICT``, and ``RAISED`` still fail loud.

    For **every** other outcome (including non-raising malformed payloads)
    exactly **one** ``logger.warning`` is emitted naming the shape, and the
    tuple ``(None, EnvelopeParseError(shape=…, …))`` is returned.

    The invariant ``err is not None`` whenever ``value is None`` (and vice
    versa) is guaranteed — callers may branch on ``err is not None`` safely.

    Processes only the **first** ``type=='text'`` block ("first text block
    wins"), mirroring the scheduler's ``_parse_tool_text_result``.
    """
    try:
        # --- locate first text block -------------------------------------------
        if not isinstance(result, dict):
            return _fail(EnvelopeShape.NO_TEXT_BLOCK, key, result)

        inner_result = result.get("result")
        content = inner_result.get("content", []) if isinstance(inner_result, dict) else None
        if not isinstance(content, list):
            return _fail(EnvelopeShape.NO_TEXT_BLOCK, key, result)

        text_block = None
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_block = block
                break

        if text_block is None:
            return _fail(EnvelopeShape.NO_TEXT_BLOCK, key, result)

        raw_text = text_block.get("text", "")

        # --- JSON decode -------------------------------------------------------
        try:
            data = json.loads(raw_text)
        except (ValueError, TypeError):
            return _fail(EnvelopeShape.JSON_DECODE_ERROR, key, raw_text)  # noqa: B904

        # --- {data:{…}} unwrap -------------------------------------------------
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            inner = data["data"]
        else:
            inner = data

        # --- inner must be a dict ----------------------------------------------
        if not isinstance(inner, dict):
            return _fail(EnvelopeShape.INNER_NOT_DICT, key, inner)

        # --- whole-inner-dict mode: key=None returns the entire inner dict -----
        if key is None:
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
            "MCP envelope parse raised unexpected exception (shape=%s, key=%r): %s",
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
