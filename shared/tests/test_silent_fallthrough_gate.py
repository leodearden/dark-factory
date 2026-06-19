"""pytest AST-scan lint gate: forbid silent-fallthrough-on-error signatures.

This test module enforces two anti-patterns that hide errors instead of
escalating them:

  Signature (a) — discarded resolver error slot:
      x, _ = [await] <resolver>(...)
    where <resolver> is one of KNOWN_VALUE_ERROR_RESOLVERS.

  Signature (b) — silent broad-except returning an empty literal:
      except (Exception|BaseException|bare): return None/{}/()/[]/set()
    with no WARN+ log call and no re-raise.

The gate scans all first-party source files (excluding tests, submodules)
and fails on any violation not present in the documented baseline ALLOWLIST.

References:
  - plans/silent-fallthrough-dedup-prd.md (PRD task σ)
  - shared/tests/silent_fallthrough_allowlist.py (baseline)
  - shared/tests/silent_fallthrough_scan.py (scanner)
"""
from __future__ import annotations

import textwrap

import pytest

from silent_fallthrough_scan import (
    KNOWN_VALUE_ERROR_RESOLVERS,
    Violation,
    find_violations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _violations(source: str, filename: str = "<test>") -> list[Violation]:
    """Run find_violations on a dedented source string."""
    return find_violations(textwrap.dedent(source).strip(), filename)


def _sig_a(v: Violation) -> bool:
    return v.signature == "a"


def _sig_b(v: Violation) -> bool:
    return v.signature == "b"


# ---------------------------------------------------------------------------
# Step 1 — Signature (a) unit tests
# ---------------------------------------------------------------------------


class TestSignatureAKnownResolvers:
    """KNOWN_VALUE_ERROR_RESOLVERS contains exactly the expected names."""

    def test_known_set_contains_parse_tool_result(self):
        assert "parse_tool_result" in KNOWN_VALUE_ERROR_RESOLVERS

    def test_known_set_contains_get_external_statuses(self):
        assert "get_external_statuses" in KNOWN_VALUE_ERROR_RESOLVERS

    def test_known_set_contains_get_statuses(self):
        assert "get_statuses" in KNOWN_VALUE_ERROR_RESOLVERS

    def test_known_set_excludes_or_warn_primitives(self):
        assert "parse_timestamp_or_warn" not in KNOWN_VALUE_ERROR_RESOLVERS
        assert "load_json_or_warn" not in KNOWN_VALUE_ERROR_RESOLVERS

    def test_known_set_excludes_ambiguous_names(self):
        assert "get_tasks" not in KNOWN_VALUE_ERROR_RESOLVERS
        assert "get_status" not in KNOWN_VALUE_ERROR_RESOLVERS


class TestSignatureAPositives:
    """Patterns that MUST be flagged as sig-a violations."""

    def test_await_parse_tool_result_discard_error(self):
        src = "x, _ = await parse_tool_result(r, 'k', dict)"
        violations = _violations(src)
        sig_a = [v for v in violations if _sig_a(v)]
        assert len(sig_a) == 1, f"Expected 1 sig-a violation, got {sig_a}"
        assert sig_a[0].lineno == 1

    def test_sync_get_external_statuses_discard_error(self):
        src = "s, _ = get_external_statuses(deps)"
        violations = _violations(src)
        sig_a = [v for v in violations if _sig_a(v)]
        assert len(sig_a) == 1, f"Expected 1 sig-a violation, got {sig_a}"

    def test_method_get_statuses_discard_error(self):
        src = "st, _ = self.get_statuses()"
        violations = _violations(src)
        sig_a = [v for v in violations if _sig_a(v)]
        assert len(sig_a) == 1, f"Expected 1 sig-a violation, got {sig_a}"

    def test_correct_line_number_reported(self):
        src = """\
        x = 1
        y, _ = get_external_statuses(deps)
        z = 2
        """
        violations = _violations(src)
        sig_a = [v for v in violations if _sig_a(v)]
        assert len(sig_a) == 1
        assert sig_a[0].lineno == 2

    def test_filename_is_preserved_in_violation(self):
        src = "x, _ = await parse_tool_result(r, 'k', dict)"
        violations = _violations(src, filename="mymodule.py")
        sig_a = [v for v in violations if _sig_a(v)]
        assert len(sig_a) == 1
        assert sig_a[0].filename == "mymodule.py"


class TestSignatureANegatives:
    """Patterns that must NOT be flagged as sig-a violations."""

    def test_error_bound_not_discarded(self):
        """tasks, tasks_err = parse_tool_result(...) — error is kept."""
        src = "tasks, tasks_err = parse_tool_result(r, 'k', dict)"
        sig_a = [v for v in _violations(src) if _sig_a(v)]
        assert sig_a == []

    def test_parse_timestamp_or_warn_not_flagged(self):
        """ts, _ = parse_timestamp_or_warn(raw) — or_warn primitive, excluded."""
        src = "ts, _ = parse_timestamp_or_warn(raw)"
        sig_a = [v for v in _violations(src) if _sig_a(v)]
        assert sig_a == []

    def test_load_json_or_warn_not_flagged(self):
        """d, _ = load_json_or_warn(p, default=None) — or_warn primitive, excluded."""
        src = "d, _ = load_json_or_warn(p, default=None)"
        sig_a = [v for v in _violations(src) if _sig_a(v)]
        assert sig_a == []

    def test_unknown_callee_not_flagged(self):
        """a, _ = some_other_func() — callee not in known set."""
        src = "a, _ = some_other_func()"
        sig_a = [v for v in _violations(src) if _sig_a(v)]
        assert sig_a == []

    def test_value_discarded_error_bound(self):
        """_, err = parse_tool_result(...) — value discarded, not error."""
        src = "_, err = parse_tool_result(r, 'k', dict)"
        sig_a = [v for v in _violations(src) if _sig_a(v)]
        assert sig_a == []

    def test_three_element_tuple_not_flagged(self):
        """Only 2-element tuples with _ at index 1 are flagged."""
        src = "a, b, _ = some_resolver()"
        sig_a = [v for v in _violations(src) if _sig_a(v)]
        assert sig_a == []

    def test_syntax_error_returns_empty(self):
        """Invalid syntax yields no violations (doesn't raise)."""
        src = "def ("
        violations = find_violations(src, "<bad>")
        assert violations == []


# ---------------------------------------------------------------------------
# Step 3 — Signature (b) unit tests
# ---------------------------------------------------------------------------


class TestSignatureBPositives:
    """Patterns that MUST be flagged as sig-b violations."""

    def test_broad_except_exception_return_empty_dict(self):
        src = """\
        def f():
            try:
                pass
            except Exception:
                return {}
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1, f"Expected 1 sig-b, got {sig_b}"

    def test_bare_except_return_empty_dict(self):
        src = """\
        def f():
            try:
                pass
            except:
                return {}
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1

    def test_except_base_exception_return_empty_dict(self):
        src = """\
        def f():
            try:
                pass
            except BaseException:
                return {}
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1

    def test_tuple_handler_containing_exception_flagged(self):
        src = """\
        def f():
            try:
                pass
            except (RuntimeError, Exception):
                return {}
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1

    def test_return_none_flagged(self):
        src = """\
        def f():
            try:
                pass
            except Exception:
                return None
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1

    def test_return_empty_list_flagged(self):
        src = """\
        def f():
            try:
                pass
            except Exception:
                return []
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1

    def test_return_empty_tuple_flagged(self):
        src = """\
        def f():
            try:
                pass
            except Exception:
                return ()
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1

    def test_return_bare_set_call_flagged(self):
        src = """\
        def f():
            try:
                pass
            except Exception:
                return set()
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1

    def test_return_bare_dict_call_flagged(self):
        src = """\
        def f():
            try:
                pass
            except Exception:
                return dict()
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1

    def test_return_bare_list_call_flagged(self):
        src = """\
        def f():
            try:
                pass
            except Exception:
                return list()
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1

    def test_return_bare_tuple_call_flagged(self):
        src = """\
        def f():
            try:
                pass
            except Exception:
                return tuple()
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1

    def test_return_empty_set_literal_flagged(self):
        """set() call is the only way to write an empty set literal."""
        src = """\
        def f():
            try:
                pass
            except Exception:
                return set()
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert len(sig_b) == 1


class TestSignatureBNegatives:
    """Patterns that must NOT be flagged as sig-b violations."""

    def test_narrow_typed_except_not_flagged(self):
        """except KeyError: return None — narrow handler, deliberate expected-condition."""
        src = """\
        def f():
            try:
                pass
            except KeyError:
                return None
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []

    def test_broad_except_with_warn_log_not_flagged(self):
        """except Exception with logger.warning(...) — logs WARN+, not silent."""
        src = """\
        def f():
            try:
                pass
            except Exception:
                logger.warning('oops %s', e)
                return {}
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []

    def test_broad_except_with_error_log_not_flagged(self):
        src = """\
        def f():
            try:
                pass
            except Exception:
                logger.error('oops')
                return {}
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []

    def test_broad_except_with_exception_log_not_flagged(self):
        src = """\
        def f():
            try:
                pass
            except Exception:
                logger.exception('oops')
                return {}
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []

    def test_broad_except_reraise_not_flagged(self):
        """except Exception: raise — re-raising is loud."""
        src = """\
        def f():
            try:
                pass
            except Exception:
                raise
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []

    def test_broad_except_reraise_with_empty_return_not_flagged(self):
        """If there's a raise anywhere in the handler, not flagged."""
        src = """\
        def f():
            try:
                pass
            except Exception:
                raise ValueError('x')
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []

    def test_broad_except_return_non_empty_object_not_flagged(self):
        """except Exception: return SomeObj() — non-empty return."""
        src = """\
        def f():
            try:
                pass
            except Exception:
                return SomeObj()
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []

    def test_broad_except_return_tuple_with_values_not_flagged(self):
        """except Exception: return (default, False) — non-empty tuple."""
        src = """\
        def f():
            try:
                pass
            except Exception:
                return (default, False)
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []

    def test_broad_except_return_false_not_flagged(self):
        """except Exception: return False — boolean fail-safe excluded."""
        src = """\
        def f():
            try:
                pass
            except Exception:
                return False
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []

    def test_broad_except_return_zero_not_flagged(self):
        """except Exception: return 0 — numeric excluded."""
        src = """\
        def f():
            try:
                pass
            except Exception:
                return 0
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []

    def test_tuple_handler_narrow_only_not_flagged(self):
        """except (KeyError, ValueError): return None — all narrow, excluded."""
        src = """\
        def f():
            try:
                pass
            except (KeyError, ValueError):
                return None
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []
