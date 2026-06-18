"""Tests for shared.mcp_envelope — loud-and-safe MCP envelope parser.

TDD pair 1: happy-path (GREEN on impl step-2).
TDD pair 2: the 5 non-raising abnormal shapes (GREEN on impl step-4).
TDD pair 3: RAISED catch-all + invariant (GREEN on impl step-6).
TDD pair 4: resolver_failed truth table (GREEN on impl step-8).
"""
from __future__ import annotations

import json
import logging

import pytest

from shared.mcp_envelope import (
    EnvelopeParseError,
    EnvelopeShape,
    parse_tool_result,
    resolver_failed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(payload_obj):
    """Build a minimal MCP tool-result envelope around *payload_obj*."""
    return {
        "result": {
            "content": [
                {"type": "text", "text": json.dumps(payload_obj)},
            ]
        }
    }


# ---------------------------------------------------------------------------
# Pair 1 — Happy-path (step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_direct_key_returns_value_and_no_error(self, caplog):
        env = _env({"statuses": {"a": "done"}})
        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            value, err = parse_tool_result(env, "statuses", dict)
        assert err is None
        assert value == {"a": "done"}
        warn_records = [
            r
            for r in caplog.records
            if r.name == "shared.mcp_envelope" and r.levelno >= logging.WARNING
        ]
        assert warn_records == [], "Success path must emit zero WARNING+ records"

    def test_data_wrapped_envelope_unwraps(self, caplog):
        """Envelope with {data: {...}} wrapper is unwrapped before key lookup."""
        env = _env({"data": {"tasks": [1, 2, 3]}})
        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            value, err = parse_tool_result(env, "tasks", list)
        assert err is None
        assert value == [1, 2, 3]
        warn_records = [
            r
            for r in caplog.records
            if r.name == "shared.mcp_envelope" and r.levelno >= logging.WARNING
        ]
        assert warn_records == []

    def test_expected_type_tuple_accepted(self, caplog):
        """expected_type may be a tuple (mirrors isinstance semantics)."""
        env = _env({"items": ["x", "y"]})
        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            value, err = parse_tool_result(env, "items", (list, dict))
        assert err is None
        assert value == ["x", "y"]
        warn_records = [
            r
            for r in caplog.records
            if r.name == "shared.mcp_envelope" and r.levelno >= logging.WARNING
        ]
        assert warn_records == []

    def test_data_wrapper_not_applied_when_data_is_not_dict(self, caplog):
        """When payload['data'] is not a dict the payload itself is the inner dict."""
        env = _env({"data": "not-a-dict", "key": "val"})
        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            value, err = parse_tool_result(env, "key", str)
        assert err is None
        assert value == "val"


# ---------------------------------------------------------------------------
# Pair 2 — 5 non-raising abnormal shapes (step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------

# Each tuple: (shape_member, result_input, key, expected_type)
_ABNORMAL_CASES = [
    # NO_TEXT_BLOCK — result is not a dict
    (EnvelopeShape.NO_TEXT_BLOCK, None, "k", dict),
    # NO_TEXT_BLOCK — result dict has empty content
    (EnvelopeShape.NO_TEXT_BLOCK, {"result": {"content": []}}, "k", dict),
    # NO_TEXT_BLOCK — content only has non-text blocks
    (
        EnvelopeShape.NO_TEXT_BLOCK,
        {"result": {"content": [{"type": "image", "url": "x"}]}},
        "k",
        dict,
    ),
    # NO_TEXT_BLOCK — result completely missing result key
    (EnvelopeShape.NO_TEXT_BLOCK, {}, "k", dict),
    # JSON_DECODE_ERROR — text block present but invalid JSON
    (EnvelopeShape.JSON_DECODE_ERROR, _env.__func__ if hasattr(_env, "__func__") else None, "k", dict),
    # INNER_NOT_DICT — parsed JSON is a list (not a dict)
    (EnvelopeShape.INNER_NOT_DICT, _env([1, 2, 3]), "k", dict),
    # INNER_NOT_DICT — parsed JSON is a scalar
    (EnvelopeShape.INNER_NOT_DICT, _env(42), "k", dict),
    # KEY_ABSENT — inner dict lacks the requested key
    (EnvelopeShape.KEY_ABSENT, _env({"other": "val"}), "missing_key", dict),
    # WRONG_TYPE — value present but wrong type
    (EnvelopeShape.WRONG_TYPE, _env({"count": 5}), "count", str),
    # WRONG_TYPE — value is None (None is not an instance of dict)
    (EnvelopeShape.WRONG_TYPE, _env({"ptr": None}), "ptr", dict),
]

# Build the JSON_DECODE_ERROR case properly (None slot was placeholder)
_JSON_DECODE_ENV = {"result": {"content": [{"type": "text", "text": "{not json"}]}}
_ABNORMAL_CASES[4] = (EnvelopeShape.JSON_DECODE_ERROR, _JSON_DECODE_ENV, "k", dict)


@pytest.mark.parametrize(
    "shape,result_input,key,expected_type",
    _ABNORMAL_CASES,
    ids=[
        "no_text_block_none",
        "no_text_block_empty_content",
        "no_text_block_no_text_type",
        "no_text_block_missing_result",
        "json_decode_error",
        "inner_not_dict_list",
        "inner_not_dict_scalar",
        "key_absent",
        "wrong_type",
        "wrong_type_none_value",
    ],
)
class TestAbnormalShapes:
    def test_returns_none_and_envelope_error(
        self, shape, result_input, key, expected_type, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            value, err = parse_tool_result(result_input, key, expected_type)

        assert value is None, f"Expected value=None for shape {shape!r}"
        assert err is not None, f"Expected non-None err for shape {shape!r}"
        assert isinstance(err, EnvelopeParseError), (
            f"Expected EnvelopeParseError, got {type(err)}"
        )
        assert err.shape == shape, f"Expected shape={shape!r}, got {err.shape!r}"
        assert isinstance(err.payload_prefix, str) and len(err.payload_prefix) > 0, (
            "payload_prefix must be a non-empty str"
        )

    def test_exactly_one_warning_logged(
        self, shape, result_input, key, expected_type, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            parse_tool_result(result_input, key, expected_type)

        warn_records = [
            r
            for r in caplog.records
            if r.name == "shared.mcp_envelope" and r.levelno >= logging.WARNING
        ]
        assert len(warn_records) == 1, (
            f"Expected exactly 1 WARNING, got {len(warn_records)}: {warn_records}"
        )

    def test_warning_message_contains_shape_token(
        self, shape, result_input, key, expected_type, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            parse_tool_result(result_input, key, expected_type)

        warn_records = [
            r
            for r in caplog.records
            if r.name == "shared.mcp_envelope" and r.levelno >= logging.WARNING
        ]
        msg = warn_records[0].getMessage()
        # The shape member value (a StrEnum) should appear in the log message
        assert str(shape) in msg or shape.name in msg, (
            f"Expected shape token {shape!r} in warning message: {msg!r}"
        )


class TestSuccessPathNoRegressionAfterAbnormal:
    """Ensure the success path still emits zero warnings (regression guard)."""

    def test_success_path_zero_warnings(self, caplog):
        env = _env({"x": 42})
        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            value, err = parse_tool_result(env, "x", int)
        assert err is None
        assert value == 42
        warn_records = [
            r
            for r in caplog.records
            if r.name == "shared.mcp_envelope" and r.levelno >= logging.WARNING
        ]
        assert warn_records == []


# ---------------------------------------------------------------------------
# Pair 3 — RAISED catch-all + invariant (step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------


class TestRaisedShape:
    def test_unexpected_exception_yields_raised_shape(self, monkeypatch, caplog):
        """Monkeypatching json.loads to raise RecursionError triggers RAISED."""
        import shared.mcp_envelope as _mod

        def _bad_loads(_):
            raise RecursionError("deep recursion")

        monkeypatch.setattr(_mod, "json", type("_FakeJson", (), {"loads": staticmethod(_bad_loads)})())
        env = _env({"x": 1})  # built before monkeypatch changes _mod.json

        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            value, err = parse_tool_result(env, "x", dict)

        assert value is None
        assert err is not None
        assert isinstance(err, EnvelopeParseError)
        assert err.shape == EnvelopeShape.RAISED
        assert err.__cause__ is not None, "RAISED must chain the original exception"

    def test_raised_emits_exactly_one_warning(self, monkeypatch, caplog):
        import shared.mcp_envelope as _mod

        def _bad_loads(_):
            raise MemoryError("oom")

        monkeypatch.setattr(_mod, "json", type("_FakeJson", (), {"loads": staticmethod(_bad_loads)})())
        env = _env({})

        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            parse_tool_result(env, "k", dict)

        warn_records = [
            r
            for r in caplog.records
            if r.name == "shared.mcp_envelope" and r.levelno >= logging.WARNING
        ]
        assert len(warn_records) == 1


_ALL_FAILURE_INPUTS = [
    (None, "k", dict),
    ({"result": {"content": []}}, "k", dict),
    (_JSON_DECODE_ENV, "k", dict),
    (_env([1, 2]), "k", dict),
    (_env({"other": 1}), "missing", dict),
    (_env({"val": "str"}), "val", int),
    (_env({"val": None}), "val", dict),
]


@pytest.mark.parametrize(
    "result_input,key,expected_type",
    _ALL_FAILURE_INPUTS,
    ids=[
        "inv_none",
        "inv_empty_content",
        "inv_bad_json",
        "inv_list_payload",
        "inv_key_absent",
        "inv_wrong_type",
        "inv_none_value",
    ],
)
class TestInvariant:
    """Invariant: failure paths NEVER return (value, None) or (None, None)."""

    def test_err_slot_always_non_none_on_failure(
        self, result_input, key, expected_type, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="shared.mcp_envelope"):
            value, err = parse_tool_result(result_input, key, expected_type)
        assert err is not None, (
            f"Invariant violated: got (value={value!r}, err=None) for "
            f"result_input={result_input!r}, key={key!r}"
        )
        assert value is None, (
            f"Invariant violated: value must be None on failure, got {value!r}"
        )


# ---------------------------------------------------------------------------
# Pair 4 — resolver_failed truth table (step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


class TestResolverFailed:
    def test_non_empty_dict_no_err_returns_false(self):
        assert resolver_failed({"a": 1}, None) is False

    def test_non_empty_list_no_err_returns_false(self):
        assert resolver_failed(["x"], None) is False

    def test_empty_dict_no_err_returns_true(self):
        assert resolver_failed({}, None) is True

    def test_empty_list_no_err_returns_true(self):
        assert resolver_failed([], None) is True

    def test_none_value_no_err_returns_true(self):
        assert resolver_failed(None, None) is True

    def test_non_empty_value_with_err_returns_true(self):
        err = EnvelopeParseError(EnvelopeShape.WRONG_TYPE, key="k", payload_prefix="p")
        assert resolver_failed({"a": 1}, err) is True

    def test_empty_value_with_err_returns_true(self):
        err = EnvelopeParseError(EnvelopeShape.WRONG_TYPE, key="k", payload_prefix="p")
        assert resolver_failed({}, err) is True

    def test_resolver_failed_emits_no_log_records(self, caplog):
        """resolver_failed is a pure predicate — it must not log anything."""
        with caplog.at_level(logging.DEBUG, logger="shared.mcp_envelope"):
            resolver_failed({}, None)
            resolver_failed({"a": 1}, None)
            err = EnvelopeParseError(EnvelopeShape.KEY_ABSENT, key="k", payload_prefix="p")
            resolver_failed({"x": 1}, err)
        records = [r for r in caplog.records if r.name == "shared.mcp_envelope"]
        assert records == [], f"resolver_failed emitted unexpected log records: {records}"
