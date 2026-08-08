"""conftest.py for scripts/ tests — inserts scripts/ onto sys.path.

Mirrors the repo root conftest.py sys.path-insertion pattern so that
`import reviewer_redundancy_diagnostic` resolves when pytest collects
scripts/tests/ under importlib import mode.  No package __init__.py is
needed (importlib mode does not require it).

Also inserts scripts/legibility/ so flat modules nested one level down
(e.g. `digest.py`, and its PRD-decomposition siblings — the sampler,
merger, trickle coder, etc. all planned to live under scripts/legibility/)
resolve via a bare `import digest` the same way top-level scripts/*.py
modules do. Without this, scripts/ on sys.path alone only makes
scripts/legibility/ importable as a namespace package (`import legibility`),
not its contents as bare top-level names.

Also home to the shared tasks.db test fixtures (`make_tasks_db`,
`project_root_with_tasks_db`). Each previously existed as three
near-identical private copies across the sweep-script test files, under
two different names and with two different return types (task 3336). They are
real pytest fixtures rather than importable helpers because fixtures
auto-resolve for every file in this directory, whereas a `from conftest import
...` spelling is fragile under the repo-wide `--import-mode=importlib` addopts.

`install_fake_httpx` (task 3376) follows the same convention, collapsing six
copies of one fake-httpx idiom spread across four of this directory's files.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent  # scripts/tests/../ = scripts/
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_LEGIBILITY_DIR = _SCRIPTS_DIR / 'legibility'
if str(_LEGIBILITY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGIBILITY_DIR))

# Same insert for scripts/local-model-serving/ (task 3713, LME-alpha).  That
# directory name is HYPHENATED, so unlike a dotted package name it can never be
# imported as a namespace package at all — a direct sys.path entry is the only
# way its flat `lms_*` modules resolve by bare name under --import-mode=importlib.
# Mirrored in the root pyproject.toml [tool.pyright] extraPaths, which is what
# `npx pyright scripts/` resolves against.
_LMS_DIR = _SCRIPTS_DIR / 'local-model-serving'
if str(_LMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LMS_DIR))

# Suite-wide git isolation (task 3355, incident esc-3072-3).  A run rooted at
# this directory does not load the repo-root conftest.py, so each test-root
# conftest wires the defence itself.  APPEND the repo root, never
# insert(0, ...): at sys.path[0] it would make the subproject directories
# resolve as namespace packages pointing at the project folder instead of
# src/<pkg>/, defeating the inserts above.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from df_pytest_isolation import (  # noqa: E402
    _df_git_ceiling_at_basetemp,  # noqa: F401  — the binding IS the wiring
    reject_unsafe_basetemp,
)


def pytest_configure(config):
    """Refuse a --basetemp aimed inside a live task worktree (esc-3072-3)."""
    reject_unsafe_basetemp(config)


# ---------------------------------------------------------------------------
# Shared tasks.db fixtures (task 3336).
#
# _TASKS_SCHEMA mirrors fused-memory's sqlite_task_backend.py _SCHEMA_SQL so
# tests exercise real column shapes and NOT NULL constraints rather than
# invented ones. It stays private to `make_tasks_db`, which executes it on
# every use — that is the executable check, so no test asserts on the literal.
#
# The schema is the SUPERSET of what the three sweep-script test files used
# privately: audit_wiped_metadata_files' copy carries `priority TEXT`, the two
# scanners' copies do not. A superset is inert for the scanners — neither
# inserts nor selects priority, and scan_db uses an explicit column list — and
# required by audit, so one schema serves all three.
# ---------------------------------------------------------------------------

_TASKS_SCHEMA = """
CREATE TABLE tasks (
    tag           TEXT NOT NULL DEFAULT 'master',
    id            INTEGER NOT NULL,
    title         TEXT NOT NULL,
    description   TEXT,
    details       TEXT,
    test_strategy TEXT,
    status        TEXT NOT NULL,
    priority      TEXT,
    metadata      TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (tag, id)
);
"""


@pytest.fixture
def make_tasks_db(tmp_path):
    """Factory: build a temp tasks.db seeded with *rows*, return its Path.

    Each row is a dict of column -> value. ``id`` is required; ``tag``,
    ``title``, ``status``, ``priority`` and ``updated_at`` (the NOT NULL
    columns, plus priority) fall back to placeholders, and the nullable
    columns stay NULL when omitted.

    ``metadata`` is passed through VERBATIM when it is a str or None — so a
    test can insert deliberately malformed JSON to exercise a decoder's error
    path — and json-encoded when it is a dict/list.

    *directory* defaults to ``tmp_path``; pass it to seed a db at a
    project-root-relative location such as ``<root>/.taskmaster/tasks/``.
    """
    def _make(rows, name='tasks.db', directory=None):
        db_path = (Path(directory) if directory is not None else tmp_path) / name
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(_TASKS_SCHEMA)
            for row in rows:
                metadata = row.get('metadata')
                if metadata is not None and not isinstance(metadata, str):
                    metadata = json.dumps(metadata)
                conn.execute(
                    'INSERT INTO tasks (tag, id, title, description, details, '
                    'test_strategy, status, priority, metadata, updated_at) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        row.get('tag', 'master'),
                        row['id'],
                        row.get('title', f"task {row['id']}"),
                        row.get('description'),
                        row.get('details'),
                        row.get('test_strategy'),
                        row.get('status', 'done'),
                        row.get('priority', 'medium'),
                        metadata,
                        row.get('updated_at', '2026-07-30T00:00:00+00:00'),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return db_path

    return _make


@pytest.fixture
def project_root_with_tasks_db():
    """Factory: create ``<root>/.taskmaster/tasks/tasks.db``, return its Path.

    The file is created empty — callers that need real rows use
    :func:`make_tasks_db` with ``directory=``.

    Idempotent AND non-destructive: an EXISTING tasks.db is left untouched.
    Both halves matter. A test may build a root and then re-touch it (hence
    ``exist_ok=True``), but ``exist_ok`` only makes the *mkdir* idempotent — an
    unconditional ``write_text('')`` would silently truncate a real database to
    zero bytes. Since this fixture auto-resolves for every file in
    ``scripts/tests/``, the natural ordering ``make_tasks_db(rows,
    directory=root / '.taskmaster' / 'tasks')`` then
    ``project_root_with_tasks_db(root)`` (reading as "now make this root
    discoverable") would otherwise blank the seeded rows and leave the
    downstream assertion passing vacuously.
    """
    def _make(root):
        db = Path(root) / '.taskmaster' / 'tasks' / 'tasks.db'
        db.parent.mkdir(parents=True, exist_ok=True)
        if not db.exists():
            db.write_text('')
        return db

    return _make


# ---------------------------------------------------------------------------
# Shared fake-httpx fixture (task 3376).
#
# A fixture rather than an importable helper for the same reason as the
# tasks.db fixtures above: fixtures auto-resolve for every file in this
# directory, whereas `from conftest import ...` is fragile under the repo-wide
# `--import-mode=importlib` addopts (pyproject.toml:47).
#
# Deduplicates six copies of the identical idiom across four files:
# test_census_trigger.py (x3), test_legibility_census.py,
# test_legibility_nightly.py and test_legibility_transcript_persistence.py.
# It injects only the MODULE, so each file keeps its own `_FakeHttpxResponse`
# — those shapes genuinely differ (payload-taking vs no-arg) and are
# orthogonal to this dedup.
# ---------------------------------------------------------------------------


@pytest.fixture
def install_fake_httpx(monkeypatch):
    """Factory: install a stub ``httpx`` module exposing *post*, return it.

    The seam exists for DETERMINISM, not because the package is missing.
    httpx IS installed here — a direct dependency of ``shared``
    (``shared/pyproject.toml``, ``httpx>=0.27``, added for task 2965) — so the
    lazy, function-local ``import httpx`` in the modules under test resolves to
    the real library and its POST genuinely goes out to
    ``$FUSED_MEMORY_MCP_URL`` (default ``localhost:8002``).  On any box running
    the fused-memory stack that POST SUCCEEDS, so a test leaning on ambient
    unreachability flaps with the host's listener state — the exact defect
    tasks 3237/3291 fixed by injecting a fake.  Injecting makes both the
    request shape assertable and the failure path deterministic regardless of
    what is listening, which makes this seam MORE necessary now, not less.

    Uses ``monkeypatch.setitem`` (never a bare ``sys.modules[...] = ...``) so
    the real module is restored at teardown and cannot leak into later tests.

    The stub exposes ``post``, and ``get`` when a caller supplies one (task
    3713: readiness polling hits GET ``/health`` and GET ``/v1/models``, which
    a POST-only stub cannot express).  ``get`` stays OPT-IN rather than a
    no-arg default, so a module that starts issuing GETs without its test
    saying so still lands in the loud-miss path below instead of silently
    receiving a stub response nobody wrote.

    Any OTHER attribute is a loud ``pytest.fail`` rather than an
    ``AttributeError``, because ``default_status_fetcher`` funnels every
    ``Exception`` into ``StatusFetchUnavailable``: a future production change
    reaching for e.g. ``httpx.Timeout`` or ``except httpx.HTTPError`` would
    otherwise keep its test green off the stub's own miss.  ``pytest.fail``
    raises a ``BaseException``, which such a handler cannot swallow.  Dunders
    stay ordinary ``AttributeError``s so import machinery and introspection can
    still probe the module.
    """
    def _make(post, get=None):
        fake = type(sys)('httpx')
        fake.post = post
        if get is not None:
            fake.get = get

        def _missing(name: str):
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(name)
            pytest.fail(
                f'install_fake_httpx stub has no {name!r}: the code under test now '
                'reaches for an httpx attribute this fixture does not provide. '
                'Extend the fixture (scripts/tests/conftest.py) instead of letting '
                'the miss be swallowed by a production `except Exception` path.',
                pytrace=False,
            )

        fake.__getattr__ = _missing  # PEP 562 module-level __getattr__
        monkeypatch.setitem(sys.modules, 'httpx', fake)
        return fake

    return _make
