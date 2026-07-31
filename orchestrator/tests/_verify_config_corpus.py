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

    Every failure at this boundary IS a drift, just one the caller's assertion
    never gets to describe: an unreadable file means the module was deleted or
    renamed, a missing key means the key was. Each is re-raised as an
    ``AssertionError`` carrying the same "fix the corpus, don't loosen the
    check" guidance the drift assertions spell out, so the gate reports the
    actionable thing rather than a bare ``FileNotFoundError``/``KeyError``
    traceback out of a parametrised case.
    """
    try:
        raw = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise AssertionError(
            f'cannot read {path} while drift-checking the verify-config corpus: '
            f'{exc.__class__.__name__}: {exc}.\n'
            f'FIX: if that config was deleted or renamed, drop the corpus entry pointing '
            f'at it — and the verify-scoper goldens that consume it — from '
            f'orchestrator/tests/_verify_config_corpus.py.'
        ) from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise AssertionError(
            f'{path} is not parseable YAML: {exc}.\n'
            f'FIX: repair the config. An unparseable config is a real failure, not an '
            f'absent one — the drift gate must fail loudly rather than skip past it.'
        ) from exc
    try:
        return data[key]
    except (KeyError, TypeError) as exc:
        available = sorted(data) if isinstance(data, dict) else f'<{type(data).__name__}>'
        raise AssertionError(
            f'{path} has no top-level {key!r} key (top level holds: {available}).\n'
            f'FIX: the key was renamed or removed. Point the corpus at its new name, or '
            f'drop the constant together with the verify-scoper goldens that consume it. '
            f'Do NOT fall back to a default — a silently-absent key is exactly the drift '
            f'this gate exists to catch.'
        ) from exc


def discover_lint_command_modules() -> set[str]:
    """Names of the immediate subdirs whose ``orchestrator.yaml`` defines a ``lint_command``.

    Modelled on ``tests/scripts/test_fallback_verify_config.py``'s
    ``_discover_per_module_configs`` (applied to ``lint_command`` rather than
    ``test_command``) and adopted for the same stated reason: a hardcoded list
    silently fails to cover a subproject added later. The corpus constants play
    that file's known-names-floor role, and the completeness check in
    ``test_verify_config_corpus.py`` is what ties the two together.

    Returns the module NAMES, not a ``{name: lint_command}`` mapping: the
    completeness check compares set membership, and the values are pinned
    independently by the forward checks via ``load_config_scalar``. Handing the
    forward half its values from here would collapse the gate's two halves into
    one code path — a discovery bug would then skew both.

    Configs defining only a ``test_command`` (e.g. ``scripts/orchestrator.yaml``)
    are omitted by construction. Dot-prefixed parents are skipped so ``.venv``,
    ``.task``, ``.claude`` — and any future dot-dir — stay deterministically out
    of the result.

    ``dark-factory-orchestrator.yaml`` is deliberately out of reach: it sits at
    the repo root rather than under ``*/``, and its scalars are pinned by the
    forward checks instead.

    An unreadable or unparseable config RAISES rather than being skipped —
    deliberately diverging from the precedent's bare ``continue``, because a
    silent skip defeats this check in the exact case it exists for: a new
    subproject whose ``orchestrator.yaml`` is present but broken would read as
    one declaring no ``lint_command`` at all, and the suite would go on claiming
    to cover every live lint_command.

    With ``load_config_scalar``, one of only two disk-reading entry points in
    this module — neither is called at import time.
    """
    found: set[str] = set()
    for path in sorted(REPO_ROOT.glob('*/orchestrator.yaml')):
        if path.parent.name.startswith('.'):
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
        except (OSError, yaml.YAMLError) as exc:
            raise AssertionError(
                f'cannot read or parse {path} while discovering the live lint_commands: '
                f'{exc.__class__.__name__}: {exc}.\n'
                f'FIX: repair that config. Skipping it would silently defeat the '
                f'completeness half of the drift gate — a module with a broken '
                f'orchestrator.yaml would be indistinguishable from one that declares no '
                f'lint_command, leaving the corpus claiming coverage it does not have.'
            ) from exc
        if isinstance(data, dict) and 'lint_command' in data:
            found.add(path.parent.name)
    return found


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

# cockpit/dashboard/escalation/orchestrator/sampler/shared
# orchestrator.yaml::lint_command — each the same 2-segment shape, differing
# only in the module name.
#
# Deliberately an explicit pinned literal rather than discovery from disk:
# ``test_verify_plan.py``'s goldens ITERATE this dict to build deterministic
# scope expectations, and a disk-derived dict would make those goldens vary
# with the checkout — a suite whose expected values are read from the same
# place as its inputs proves nothing. Discovery is used only by
# ``discover_lint_command_modules`` below, which cross-CHECKS this literal
# instead of replacing it.
MODULE_LINT_COMMANDS = {
    module: (
        f'uv run --project {module} --directory {module} ruff check src/ tests/'
        f' && python3 fused-memory/scripts/check_bare_magicmock_config.py {module}/tests'
    )
    for module in ('cockpit', 'dashboard', 'escalation', 'orchestrator', 'sampler', 'shared')
}

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
