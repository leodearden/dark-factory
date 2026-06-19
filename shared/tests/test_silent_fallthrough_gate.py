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

import ast
import textwrap
from typing import NamedTuple

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

    def test_nonempty_set_literal_not_flagged(self):
        """except Exception: return {1} — non-empty set literal must NOT be flagged.

        Coverage for the ast.Set branch: an empty set() call is already covered
        by test_return_bare_set_call_flagged; this confirms the non-empty case
        is excluded (distinct from the set() call path).
        """
        src = """\
        def f():
            try:
                pass
            except Exception:
                return {1}
        """
        sig_b = [v for v in _violations(src) if _sig_b(v)]
        assert sig_b == []


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


# ---------------------------------------------------------------------------
# Step 5 — Whole-tree integration + gate self-integrity tests
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from silent_fallthrough_allowlist import ALLOWLIST_ENTRIES, ALLOWLIST_VIOLATIONS  # noqa: E402
from silent_fallthrough_scan import iter_first_party_files  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _TreeScanData(NamedTuple):
    """Cached result of a full first-party tree scan (session-scoped)."""

    files: list        # list[Path]
    violations: list   # list[Violation]
    violations_by_key: dict  # {(relpath, lineno): Violation}
    parse_failures: list     # list[str]


@pytest.fixture(scope="session")
def tree_scan_data() -> _TreeScanData:
    """Enumerate, read, parse, and scan the first-party tree once per test session.

    All integration and integrity tests consume this fixture rather than
    re-scanning ~215 files independently on each test invocation.
    """
    files = list(iter_first_party_files(_REPO_ROOT))
    violations: list[Violation] = []
    violations_by_key: dict[tuple[str, int], Violation] = {}
    parse_failures: list[str] = []
    for filepath in files:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        rel = str(filepath.relative_to(_REPO_ROOT))
        try:
            ast.parse(source, filename=str(filepath))
        except SyntaxError as e:
            parse_failures.append(f"{filepath}: {e}")
        for v in find_violations(source, rel):
            violations.append(v)
            violations_by_key[(v.filename, v.lineno)] = v
    return _TreeScanData(
        files=files,
        violations=violations,
        violations_by_key=violations_by_key,
        parse_failures=parse_failures,
    )


class TestGateSelfIntegrity:
    """The gate must not silently scan 0 files (a mis-resolved root would pass trivially)."""

    def test_scanned_file_count_exceeds_150(self, tree_scan_data):
        """Expect 150+ first-party source files (215 today)."""
        files = tree_scan_data.files
        assert len(files) > 150, (
            f"Only {len(files)} files found — is repo_root correct? "
            f"({_REPO_ROOT})"
        )

    def test_known_first_party_files_are_included(self, tree_scan_data):
        """Known first-party files must be present in the scan set."""
        files = {str(f) for f in tree_scan_data.files}
        expected = [
            "orchestrator/src/orchestrator/scheduler.py",
            "orchestrator/src/orchestrator/harness.py",
            "fused-memory/src/fused_memory/services/memory_service.py",
        ]
        for rel in expected:
            candidate = str(_REPO_ROOT / rel)
            assert candidate in files, f"Expected file missing from scan: {rel}"

    def test_excluded_paths_are_absent(self, tree_scan_data):
        """Submodule dirs (mem0/, graphiti/) and tests/ are excluded."""
        for f in tree_scan_data.files:
            parts = Path(f).parts
            assert "mem0" not in parts, f"mem0 submodule should be excluded: {f}"
            assert "graphiti" not in parts, f"graphiti submodule should be excluded: {f}"
            assert "tests" not in parts, f"tests/ dirs should be excluded: {f}"
            assert not Path(f).name.startswith("test_"), f"test_ files should be excluded: {f}"
            assert Path(f).name != "conftest.py", f"conftest.py should be excluded: {f}"


class TestWholeTreeGate:
    """Whole-tree gate: no non-allowlisted violations on the migrated tree."""

    def test_no_unparseable_files(self, tree_scan_data):
        """All first-party Python files must parse without SyntaxError."""
        if tree_scan_data.parse_failures:
            raise AssertionError(
                "Files failed to parse (fix syntax errors or exclude them):\n"
                + "\n".join(f"  {p}" for p in sorted(tree_scan_data.parse_failures))
            )

    def test_no_violations_outside_allowlist(self, tree_scan_data):
        """The set of violations across the first-party tree must be empty
        after subtracting the documented baseline ALLOWLIST.

        Any new violation means a silent-fallthrough-on-error was introduced.
        Either fix it (preferred) or add a documented entry to ALLOWLIST.
        """
        non_baseline = [
            v for v in tree_scan_data.violations
            if (v.filename, v.lineno) not in ALLOWLIST_VIOLATIONS
        ]

        if non_baseline:
            offender_list = "\n".join(
                f"  {v.filename}:{v.lineno} [sig-{v.signature}] {v.message}"
                for v in sorted(non_baseline, key=lambda v: (v.filename, v.lineno))
            )
            raise AssertionError(
                "Silent-fallthrough violations found outside the baseline allowlist.\n"
                "Fix the violation (preferred) or add a documented entry to\n"
                "shared/tests/silent_fallthrough_allowlist.py (non-silent, with reason).\n"
                f"\nOffending sites ({len(non_baseline)}):\n{offender_list}"
            )


# ---------------------------------------------------------------------------
# Step 7 — Allowlist integrity tests
# ---------------------------------------------------------------------------


class TestAllowlistIntegrity:
    """The allowlist baseline must be minimal, current, and fully documented.

    These tests enforce the ratchet: as violations are fixed, stale allowlist
    entries cause a failure here, forcing the baseline to shrink.
    """

    def test_every_entry_has_nonempty_reason(self):
        """No allowlist entry may have a blank reason (that would be a silent exemption)."""
        blank = [
            f"{relpath}:{lineno} ({qualname})"
            for relpath, lineno, qualname, reason in ALLOWLIST_ENTRIES
            if not reason.strip()
        ]
        if blank:
            raise AssertionError(
                "Allowlist entries with blank reason strings — add a reason:\n"
                + "\n".join(f"  {e}" for e in blank)
            )

    def test_every_entry_relpath_exists(self):
        """Every allowlist relpath must point to an existing file under repo root."""
        missing = [
            f"{relpath}:{lineno}"
            for relpath, lineno, _qualname, _reason in ALLOWLIST_ENTRIES
            if not (_REPO_ROOT / relpath).is_file()
        ]
        if missing:
            raise AssertionError(
                "Allowlist entries referencing non-existent files:\n"
                + "\n".join(f"  {e}" for e in missing)
            )

    def test_every_entry_qualname_is_parseable(self):
        """Every qualname must be a valid dotted Python identifier or '<module>'."""
        import re
        # Valid: 'foo', 'Foo.bar', 'A.B.c._d', '<module>'
        _DOTTED_IDENT = re.compile(r'^(?:<module>|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)$')
        invalid = [
            f"{relpath}:{lineno} qualname={qualname!r}"
            for relpath, lineno, qualname, _reason in ALLOWLIST_ENTRIES
            if not _DOTTED_IDENT.match(qualname)
        ]
        if invalid:
            raise AssertionError(
                "Allowlist entries with unparseable qualnames:\n"
                + "\n".join(f"  {e}" for e in invalid)
            )

    def test_no_duplicate_entries(self):
        """No two allowlist entries should share the same (relpath, lineno)."""
        seen: set[tuple[str, int]] = set()
        dupes: list[str] = []
        for relpath, lineno, qualname, _reason in ALLOWLIST_ENTRIES:
            key = (relpath, lineno)
            if key in seen:
                dupes.append(f"{relpath}:{lineno} ({qualname})")
            seen.add(key)
        if dupes:
            raise AssertionError(
                "Duplicate allowlist entries (same relpath:lineno):\n"
                + "\n".join(f"  {e}" for e in dupes)
            )

    def test_no_stale_entries(self, tree_scan_data):
        """Every allowlist entry must correspond to a real violation in the current tree.

        If a violation is fixed, the allowlist entry becomes stale.  This test
        fails on stale entries, forcing the baseline to shrink as code improves.
        """
        stale = [
            f"{relpath}:{lineno} ({qualname}) — {reason[:60]}"
            for relpath, lineno, qualname, reason in ALLOWLIST_ENTRIES
            if (relpath, lineno) not in tree_scan_data.violations_by_key
        ]
        if stale:
            raise AssertionError(
                "Stale allowlist entries: the violation no longer exists in the source tree.\n"
                "Remove these entries from shared/tests/silent_fallthrough_allowlist.py:\n"
                + "\n".join(f"  {e}" for e in stale)
            )
