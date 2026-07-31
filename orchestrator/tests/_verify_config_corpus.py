"""The single definition site for the verify-scoper suites' real-config corpus.

``test_verify_cmd.py``, ``test_verify_plan.py`` and ``test_verify_scope_kappa.py``
all assert byte-identical scoper outcomes over the repo's *real* orchestrator
config commands. Before task 3220 each file carried its own copy of those
literals, so "are these still the live values?" had three answers that could
silently disagree. They live here once instead, and
``test_verify_config_corpus.py`` — the drift gate — checks them against the
committed YAML on every run.

What the corpus is FOR: it is the set of command shapes ``split_chain_tail``'s
accept/reject gate must classify correctly. The lint chains are SIBLING-CHECKER
chains — a ruff/pyright invocation followed by an independent
``python3 fused-memory/scripts/check_*.py <dir>`` clause that would have run
anyway — so the scoper accepts them and preserves the tail. The root
type/test chains are cwd-sequenced same-tool fan-outs (``cd a && npx pyright &&
cd ../b && npx pyright``), where the tail is *more of the same tool in other
directories*; the scoper rejects those and keeps today's truncation.

A ``_``-prefixed, uniquely-named sibling module (like ``_orch_helpers.py``,
``_merge_queue_harness.py``): ``conftest.py`` inserts ``_TESTS_DIR`` on
``sys.path`` at import time, which is what makes a bare
``from _verify_config_corpus import ...`` resolve. The constants are NOT in
``conftest.py`` because non-fixture helpers imported from a conftest collide
across sibling subprojects under ``sys.modules['conftest']`` — see that file's
docstring.

Importing this module performs NO disk I/O: the constants are plain literals,
and ``load_config_scalar`` reads only when a drift test calls it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Worktree root. Identical expression to ``conftest.py:38`` and restated here
# rather than imported, per the conftest-collision note above (same directory
# depth, so the same ``parents[2]``). Resolving from ``__file__`` rather than
# the CWD is what makes the drift gate check THIS worktree's committed YAML —
# the copy the task's own verify run is gated on.
REPO_ROOT = Path(__file__).resolve().parents[2]

DF_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'
FM_CONFIG_PATH = REPO_ROOT / 'fused-memory' / 'orchestrator.yaml'


def load_config_scalar(path: Path, key: str) -> str:
    """Return the top-level scalar *key* from the committed YAML at *path*.

    Reads the config file directly rather than going through the consuming
    code path, mirroring ``tests/scripts/test_fallback_verify_config.py``'s
    ``_fleet_test_command`` — the point of a drift check is to compare against
    what is on disk, unmediated by the loader under test.
    """
    return yaml.safe_load(path.read_text(encoding='utf-8'))[key]


# --- The corpus -------------------------------------------------------------
#
# Provenance comments name ``<yaml file>::<key>``, deliberately NOT a line
# number: line numbers rot, and ``test_verify_config_corpus.py`` now pins each
# value against its key anyway.

# fused-memory/orchestrator.yaml::lint_command — the only 3-segment chain
# (two sibling checkers rather than one).
FM_LINT_COMMAND = (
    'uv run --project fused-memory --directory fused-memory ruff check src/ tests/'
    ' && python3 fused-memory/scripts/check_bare_magicmock_config.py fused-memory/tests'
    ' && python3 fused-memory/scripts/check_asyncmock_assertion_style.py fused-memory/tests'
)

# dark-factory-orchestrator.yaml::lint_command
ROOT_LINT_COMMAND = (
    'uv run ruff check shared escalation fused-memory orchestrator dashboard'
    ' && python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests'
    ' escalation/tests fused-memory/tests orchestrator/tests dashboard/tests'
)

# dark-factory-orchestrator.yaml::type_check_command
ROOT_TYPE_CHECK_COMMAND = (
    'cd fused-memory && npx pyright && cd ../orchestrator && npx pyright'
    ' && cd ../dashboard && npx pyright'
)

# dark-factory-orchestrator.yaml::test_command
ROOT_TEST_COMMAND = (
    'cd shared && uv run pytest tests/ --timeout=300'
    ' && cd ../escalation && uv run pytest tests/ --timeout=300'
    ' && cd ../orchestrator && uv run pytest tests/ --timeout=300'
    ' && cd ../fused-memory && uv run pytest tests/ --timeout=300'
    ' && cd ../dashboard && uv run pytest tests/ --timeout=300'
    ' && cd ../sampler && uv run pytest tests/ --timeout=300'
    ' && cd .. && ( [ -d cockpit ] || exit 0; cd cockpit && uv run pytest tests/ --timeout=300 )'
    ' && uv run --project shared pytest tests/scripts/ --timeout=300'
)
