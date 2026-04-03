"""Tests that MEM0_TELEMETRY is suppressed before mem0 imports in server/main.py.

These tests guard against regressions in the defense-in-depth pattern:
  os.environ.setdefault('MEM0_TELEMETRY', 'false')  # line N
  load_dotenv()                                       # line N+2
  from fused_memory...                               # line N+4+

The critical invariant: setdefault runs BEFORE load_dotenv() and BEFORE any
fused_memory/mem0 imports, ensuring telemetry is suppressed in all startup paths.
"""

import os
from pathlib import Path

MAIN_PY = Path(__file__).resolve().parents[1] / 'src' / 'fused_memory' / 'server' / 'main.py'

_SETDEFAULT_LITERAL = "os.environ.setdefault('MEM0_TELEMETRY', 'false')"


def _read_lines() -> list[str]:
    return MAIN_PY.read_text().splitlines()


class TestSetdefaultLiteralPresent:
    def test_setdefault_literal_present(self):
        """The exact setdefault call must exist verbatim in main.py source."""
        source = MAIN_PY.read_text()
        assert _SETDEFAULT_LITERAL in source, (
            f"Expected {_SETDEFAULT_LITERAL!r} in {MAIN_PY} — "
            "this call suppresses mem0 telemetry before any imports run"
        )


class TestSetdefaultPrecedesLoadDotenv:
    def test_setdefault_precedes_load_dotenv(self):
        """setdefault line must appear strictly before load_dotenv() by line number."""
        lines = _read_lines()
        setdefault_line: int | None = None
        load_dotenv_line: int | None = None

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if _SETDEFAULT_LITERAL in stripped:
                setdefault_line = i
            # First non-comment load_dotenv call
            if load_dotenv_line is None and 'load_dotenv(' in stripped and not stripped.startswith('#'):
                load_dotenv_line = i

        assert setdefault_line is not None, f"setdefault call not found in {MAIN_PY}"
        assert load_dotenv_line is not None, f"load_dotenv() call not found in {MAIN_PY}"
        assert setdefault_line < load_dotenv_line, (
            f"setdefault (line {setdefault_line}) must precede load_dotenv "
            f"(line {load_dotenv_line}) — setdefault must win over .env contents"
        )


class TestSetdefaultPrecedesFusedMemoryImports:
    def test_setdefault_precedes_fused_memory_imports(self):
        """setdefault must appear before any non-TYPE_CHECKING fused_memory/mem0 imports."""
        lines = _read_lines()
        setdefault_line: int | None = None
        earliest_import_line: int | None = None
        in_type_checking = False

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Track setdefault position
            if _SETDEFAULT_LITERAL in stripped:
                setdefault_line = i
                continue

            # Detect TYPE_CHECKING block start
            if stripped == 'if TYPE_CHECKING:':
                in_type_checking = True
                continue

            # Detect end of TYPE_CHECKING block: any non-blank, non-indented line
            if in_type_checking and line and not line[0].isspace():
                in_type_checking = False

            # Track earliest real fused_memory / mem0 import
            if not in_type_checking and earliest_import_line is None and (
                stripped.startswith('from fused_memory') or stripped.startswith('import mem0')
            ):
                earliest_import_line = i

        assert setdefault_line is not None, f"setdefault call not found in {MAIN_PY}"
        assert earliest_import_line is not None, (
            f"No fused_memory/mem0 imports found outside TYPE_CHECKING in {MAIN_PY}"
        )
        assert setdefault_line < earliest_import_line, (
            f"setdefault (line {setdefault_line}) must precede first fused_memory/mem0 import "
            f"(line {earliest_import_line}) — mem0 must not be imported before telemetry is disabled"
        )


class TestEnvVarIsFalseAtRuntime:
    def test_env_var_is_false_at_runtime(self):
        """After importing server.main, MEM0_TELEMETRY must equal 'false' in os.environ."""
        import fused_memory.server.main  # noqa: F401 — triggers setdefault side effect

        assert os.environ.get('MEM0_TELEMETRY') == 'false', (
            "MEM0_TELEMETRY should be 'false' after importing fused_memory.server.main"
        )
