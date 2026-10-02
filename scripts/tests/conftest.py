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

Also APPENDS scripts/tests/ itself, so non-test helper modules living beside
the tests (`cli_subprocess_timeout`, `write_triage_attach_fixtures`) resolve
by bare name: importlib mode keeps a test file's own directory off sys.path.
Like tests/scripts/conftest.py's `_THIS_DIR` entry, but appended rather than
inserted, so it can never shadow a scripts/ module or an installed package.

Also home to the shared tasks.db test fixtures (`make_tasks_db`,
`project_root_with_tasks_db`). Each previously existed as three
near-identical private copies across the sweep-script test files, under
two different names and with two different return types (task 3336). They are
real pytest fixtures rather than importable helpers because fixtures
auto-resolve for every file in this directory, whereas a `from conftest import
...` spelling is fragile under the repo-wide `--import-mode=importlib` addopts.

`install_fake_httpx` (task 3376) follows the same convention, collapsing six
copies of one fake-httpx idiom spread across four of this directory's files.

So do `runs_db_path` / `runs_db` (task 5441): a synthetic orchestrator runs.db
shared by the model-admission audit and review suites.

And `fake_claude_cli` / `pool_roster` (task 6042): a JSON-mode fake `claude`
binary keyed on the leased OAuth token, plus a hermetic account roster, for
the legibility suites that drive the shared session runner end to end.
"""
import json
import os
import shutil
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
# `uv run --project shared pyright scripts/` — this module's declared type gate
# since task 4358, `npx pyright scripts/` before it — resolves against.
_LMS_DIR = _SCRIPTS_DIR / 'local-model-serving'
if str(_LMS_DIR) not in sys.path:
    sys.path.insert(0, str(_LMS_DIR))

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

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
    _df_deploy_clocks_unwritten,  # noqa: F401  — the binding IS the wiring
    _df_fleet_deploy_clock_redirect,  # noqa: F401  — the binding IS the wiring
    _df_fleet_dir_redirect,  # noqa: F401  — the binding IS the wiring
    _df_git_ceiling_at_basetemp,  # noqa: F401  — the binding IS the wiring
    _df_git_env_hermetic,  # noqa: F401  — the binding IS the wiring
    _df_no_leaked_drain_processes,  # noqa: F401  — the binding IS the wiring
    _df_no_synthetic_heartbeats_in_live_fleet,  # noqa: F401  — binding IS wiring
    reject_unsafe_basetemp,
)


def pytest_configure(config):
    """Refuse a --basetemp aimed inside a live task worktree (esc-3072-3)."""
    reject_unsafe_basetemp(config)


# The fleet-deploy-clock redirect this directory relied on used to be defined
# HERE (task 3797). It is now the suite-wide default applied unconditionally by
# df_pytest_isolation._df_deploy_clocks_unwritten, autoused into every conftest
# that imports this module — not just this one (task 5299). This directory
# keeps only the thin `_df_fleet_deploy_clock_redirect` NAME above, imported
# rather than redefined, so `test_suite_never_stamps_the_repo_fleet_deploy_clock`
# can still take the resolved path by fixture rather than reading os.environ
# bare. See df_pytest_isolation.py's module docstring, SECOND DEFENCE, for the
# full history.


# ---------------------------------------------------------------------------
# Legibility trickle state isolation (task 4514).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_legibility_trickle_state(tmp_path_factory, monkeypatch):
    """Point the legibility trickle state root at a per-test tmp dir.

    WHAT IT PREVENTS. The operator's LIVE state files for two running
    pipelines sit at ``~/.local/state/dark-factory/legibility/{dark_factory,
    reify}/trickle-state.json``, carrying real ``last_productive_at`` stamps
    and streak history. Before task 4514 this directory isolated itself from
    them ONLY through ``XDG_STATE_HOME`` — ``test_legibility_nightly.py::
    _isolate_trickle_state``, ``test_check_trickle_progress.py::_run_probe``
    and ``::_seed`` (whose default ``project_id`` is the literal
    ``"dark_factory"``), and ~15 sites in ``test_trickle_state.py``. Task 4514
    makes ``scripts/legibility/trickle_state.py::trickle_state_path`` stop
    honouring ``XDG_STATE_HOME`` and ``HOME`` entirely, at which instant every
    one of those isolations becomes a silent no-op. Each is migrated to the
    variable set here in the same commit as that change; this autouse fixture
    is the belt to that pair of braces, catching any site the migration missed
    and any test added later that forgets.

    Directory-wide and autouse rather than opt-in, because the failure mode is
    a test that never says it touches trickle state — a plain ``pytest`` run
    overwriting the operator's live streak history is data loss, not a dirty
    tmp dir.

    A test that deliberately exercises the DEFAULT (passwd-anchored)
    resolution must ``monkeypatch.delenv('DARK_FACTORY_LEGIBILITY_STATE_ROOT',
    raising=False)`` first, and must then assert on the RESOLVED PATH only —
    never call ``record_run``, which would write to the real file.

    The literal is spelled out rather than imported because this fixture
    predates the constant it mirrors and must stay inert until that
    constant exists: ``scripts/legibility/trickle_state.py::STATE_ROOT_ENV``.
    """
    monkeypatch.setenv(
        'DARK_FACTORY_LEGIBILITY_STATE_ROOT',
        str(tmp_path_factory.mktemp('legibility-state')),
    )


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

    The stub exposes ``post``, and ``get``/``delete`` when a caller supplies
    one (task 3713: readiness polling hits GET ``/health`` and GET
    ``/v1/models``, which a POST-only stub cannot express; task 3644:
    ``census_trigger.post_mcp_envelope`` DELETEs the ``/mcp`` endpoint to
    terminate the MCP session it opened, so the server does not leak one
    session dict entry + one live anyio task per legibility escalation).
    Both stay OPT-IN rather than no-arg defaults, so a module that starts
    issuing a verb without its test saying so still lands in the loud-miss
    path below instead of silently receiving a stub response nobody wrote.

    Any OTHER attribute is a loud ``pytest.fail`` rather than an
    ``AttributeError``, because ``default_status_fetcher`` funnels every
    ``Exception`` into ``StatusFetchUnavailable``: a future production change
    reaching for e.g. ``httpx.Timeout`` or ``except httpx.HTTPError`` would
    otherwise keep its test green off the stub's own miss.  ``pytest.fail``
    raises a ``BaseException``, which such a handler cannot swallow.  Dunders
    stay ordinary ``AttributeError``s so import machinery and introspection can
    still probe the module.
    """
    def _make(post, get=None, delete=None):
        fake = type(sys)('httpx')
        fake.post = post
        if get is not None:
            fake.get = get
        if delete is not None:
            fake.delete = delete

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


# ---------------------------------------------------------------------------
# Shared synthetic runs.db fixtures (task 5441).
#
# Moved here from test_audit_model_admission.py so the model-admission audit
# and review suites seed ONE schema copy through one set of helpers. Fixtures
# for the same importlib-mode reason as the tasks.db fixtures above.
#
# RUNS_DB_SCHEMA is a VERBATIM copy of the three tables those scripts read,
# captured with
#
#     sqlite3 data/orchestrator/runs.db ".schema events invocations account_events"
#
# Copied rather than imported because this directory is collected by
# `uv run --project shared pytest` and imports NO first-party package (the
# comment on dark-factory-orchestrator.yaml::test_command says so), so the
# orchestrator's event store, which owns this DDL, is out of reach. Re-capture
# with that command rather than hand-editing if the writer's schema moves.
# ---------------------------------------------------------------------------

RUNS_DB_SCHEMA = """
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);
CREATE TABLE invocations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    task_id             TEXT,
    project_id          TEXT NOT NULL,
    account_name        TEXT NOT NULL,
    model               TEXT NOT NULL,
    role                TEXT NOT NULL,
    cost_usd            REAL NOT NULL DEFAULT 0.0,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cache_read_tokens   INTEGER,
    cache_create_tokens INTEGER,
    duration_ms         INTEGER NOT NULL DEFAULT 0,
    capped              INTEGER NOT NULL DEFAULT 0,
    started_at          TEXT NOT NULL,
    completed_at        TEXT NOT NULL
);
CREATE TABLE account_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    project_id   TEXT,
    run_id       TEXT,
    details      TEXT,
    created_at   TEXT NOT NULL
);
"""


def _payload(value):
    """JSON-encode a dict/list payload; pass a str or None through VERBATIM.

    The pass-through is what lets a test seed a deliberately malformed payload
    — the live store holds an ``account_events.details`` of the bare string
    ``'Escalation watcher (auto)'`` — so tolerant-parse paths are exercised
    against the real shape rather than a hypothetical one. Same convention as
    ``make_tasks_db``'s ``metadata`` handling above.
    """
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


class SeedingConnection(sqlite3.Connection):
    """A real, writable ``sqlite3.Connection`` that can also seed scenario rows.

    Still a Connection, so a test seeds through it and hands the SAME object
    straight to the scan under test. Each ``seed_*`` inserts one row and
    commits.
    """

    def seed_event(
        self,
        timestamp,
        event_type,
        *,
        run_id='run-1',
        task_id=None,
        phase=None,
        role=None,
        data=None,
        cost_usd=None,
        duration_ms=None,
    ):
        """Insert one `events` row, stating its payload as a dict rather than JSON text."""
        self.execute(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, phase, role, '
            'data, cost_usd, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                timestamp,
                run_id,
                task_id,
                event_type,
                phase,
                role,
                _payload({} if data is None else data),
                cost_usd,
                duration_ms,
            ),
        )
        self.commit()

    def seed_invocation(
        self,
        *,
        model,
        role,
        started_at,
        completed_at,
        run_id='run-1',
        task_id=None,
        project_id='dark_factory',
        account_name='max-a',
        cost_usd=0.0,
        input_tokens=None,
        output_tokens=None,
        cache_read_tokens=None,
        cache_create_tokens=None,
        duration_ms=0,
        capped=0,
    ):
        """Insert one `invocations` row."""
        self.execute(
            'INSERT INTO invocations (run_id, task_id, project_id, account_name, model, '
            'role, cost_usd, input_tokens, output_tokens, cache_read_tokens, '
            'cache_create_tokens, duration_ms, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                run_id,
                task_id,
                project_id,
                account_name,
                model,
                role,
                cost_usd,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_create_tokens,
                duration_ms,
                capped,
                started_at,
                completed_at,
            ),
        )
        self.commit()

    def seed_account_event(
        self,
        *,
        account_name,
        event_type,
        created_at,
        details=None,
        project_id='dark_factory',
        run_id='run-1',
    ):
        """Insert one `account_events` row; *details* follows :func:`_payload`."""
        self.execute(
            'INSERT INTO account_events (account_name, event_type, project_id, run_id, '
            'details, created_at) VALUES (?, ?, ?, ?, ?, ?)',
            (account_name, event_type, project_id, run_id, _payload(details), created_at),
        )
        self.commit()


@pytest.fixture
def runs_db_path(tmp_path):
    """Path to a fresh, empty runs.db carrying :data:`RUNS_DB_SCHEMA`."""
    path = tmp_path / 'runs.db'
    conn = sqlite3.connect(path)
    try:
        conn.executescript(RUNS_DB_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture
def runs_db(runs_db_path):
    """A WRITABLE :class:`SeedingConnection` on :func:`runs_db_path`.

    The scans under test take an open connection, so a test normally seeds
    through this fixture and hands the same connection straight to the function
    under test. Tests that exercise a read-only connection factory or a CLI take
    ``runs_db_path`` instead — both name the same file.
    """
    conn = sqlite3.connect(runs_db_path, factory=SeedingConnection)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fake JSON-mode `claude` CLI + hermetic account roster (task 6042).
#
# The legibility trickle and census spawn `claude` through the shared runner
# (shared.cli_invoke.invoke_with_cap_retry -> invoke_claude_agent), which
# resolves the BARE name against PATH. These fixtures put a fake first on PATH
# and give it a per-token script, so a suite can drive a real UsageGate through
# failover, auth rejection and caps without any real CLI being reachable.
# ---------------------------------------------------------------------------

_FAKE_CLAUDE_PLAN_ENV = 'FAKE_CLAUDE_PLAN'
_FAKE_CLAUDE_CALLS_ENV = 'FAKE_CLAUDE_CALLS'

_FAKE_CLAUDE_DEFAULT_RESPONSE = {
    'result': 'fake claude reply',
    'is_error': False,
    'api_error_status': None,
    'subtype': 'success',
    'total_cost_usd': 0.0,
    'num_turns': 1,
    'duration_ms': 1200,
    'stderr': '',
    'rc': 0,
    'sleep_secs': 0,
}

_FAKE_CLAUDE_SOURCE = """\
import json
import os
import sys
import time
import uuid
from pathlib import Path

plan = json.loads(Path(os.environ[{plan_env!r}]).read_text())
token = os.environ.get('CLAUDE_CODE_OAUTH_TOKEN', '')
response = {{**plan['default'], **plan['by_token'].get(token, {{}})}}

argv = sys.argv[1:]
stdin = sys.stdin.read()
system_prompt = None
if '--system-prompt-file' in argv:
    system_prompt = Path(argv[argv.index('--system-prompt-file') + 1]).read_text()
config_dir = os.environ.get('CLAUDE_CONFIG_DIR')
credentials = None
if config_dir:
    try:
        credentials = json.loads((Path(config_dir) / '.credentials.json').read_text())
    except (OSError, ValueError):
        credentials = None

record = {{
    'argv': argv,
    'cwd': os.getcwd(),
    'stdin': stdin,
    'system_prompt': system_prompt,
    'env': {{
        'CLAUDE_CONFIG_DIR': config_dir,
        'CLAUDE_CODE_OAUTH_TOKEN': os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'),
        'ANTHROPIC_API_KEY_present': 'ANTHROPIC_API_KEY' in os.environ,
        'HOME': os.environ.get('HOME'),
    }},
    'credentials': credentials,
}}
with open(os.environ[{calls_env!r}], 'a') as calls:
    calls.write(json.dumps(record) + '\\n')

time.sleep(response['sleep_secs'])
print(json.dumps({{
    'type': 'result',
    'subtype': response['subtype'],
    'is_error': response['is_error'],
    'result': response['result'],
    'session_id': str(uuid.uuid4()),
    'num_turns': response['num_turns'],
    'total_cost_usd': response['total_cost_usd'],
    'duration_ms': response['duration_ms'],
    'api_error_status': response['api_error_status'],
}}))
sys.stderr.write(response['stderr'])
sys.exit(response['rc'])
"""


class FakeClaudeCli:
    """A fake `claude` first on PATH, scripted per leased OAuth token.

    ``plan(by_token, default=...)`` maps a ``CLAUDE_CODE_OAUTH_TOKEN`` value to
    a response dict (any key of ``_FAKE_CLAUDE_DEFAULT_RESPONSE``); a token
    with no entry gets *default*, itself laid over those defaults. The fake
    prints real-CLI-shaped ``--output-format json`` (``subtype`` defaults to
    ``success`` even on an error, as measured on CLI 2.1.287), writes
    ``stderr`` and exits ``rc``. A failure meant to be read as an ordinary
    failure must cost something or run >=5s: the shared runner's heuristic
    net reads a zero-cost, sub-5s, <=1-turn failure as an unrecognised cap.

    ``calls()`` returns one dict per invocation, recorded BEFORE any scripted
    sleep so a timed-out call is still visible: ``argv``, ``cwd``, ``stdin``,
    ``system_prompt``, an ``env`` subset (``CLAUDE_CONFIG_DIR``,
    ``CLAUDE_CODE_OAUTH_TOKEN``, ``ANTHROPIC_API_KEY_present``, ``HOME``) and
    ``credentials``, the parsed ``$CLAUDE_CONFIG_DIR/.credentials.json`` as the
    child saw it.
    """

    def __init__(self, bin_dir: Path, state_dir: Path):
        self.bin_dir = bin_dir
        self.path = bin_dir / 'claude'
        self._plan_path = state_dir / 'fake-claude-plan.json'
        self._calls_path = state_dir / 'fake-claude-calls.jsonl'

    def plan(self, by_token=None, *, default=None):
        self._plan_path.write_text(json.dumps({
            'by_token': dict(by_token or {}),
            'default': {**_FAKE_CLAUDE_DEFAULT_RESPONSE, **(default or {})},
        }))

    def calls(self):
        if not self._calls_path.exists():
            return []
        return [json.loads(line) for line in self._calls_path.read_text().splitlines()]


@pytest.fixture
def fake_claude_cli(tmp_path, monkeypatch):
    """Install a :class:`FakeClaudeCli` as the only `claude` on PATH.

    Prepended to PATH and then PROVEN to win: ``shutil.which('claude')`` must
    resolve to the fake, so no test using this fixture can ever reach the real
    CLI (real LLM spend, and a test passing for the wrong reason). The shebang
    is ``sys.executable``, so the fake needs nothing else on PATH to start.
    Starts with the default plan; call ``.plan(...)`` to script it.
    """
    bin_dir = tmp_path / 'fake-claude-bin'
    bin_dir.mkdir()
    state_dir = tmp_path / 'fake-claude-state'
    state_dir.mkdir()
    fake = FakeClaudeCli(bin_dir, state_dir)
    fake.path.write_text(f'#!{sys.executable}\n' + _FAKE_CLAUDE_SOURCE.format(
        plan_env=_FAKE_CLAUDE_PLAN_ENV, calls_env=_FAKE_CLAUDE_CALLS_ENV,
    ))
    fake.path.chmod(0o755)
    fake.plan()
    monkeypatch.setenv(_FAKE_CLAUDE_PLAN_ENV, str(fake._plan_path))
    monkeypatch.setenv(_FAKE_CLAUDE_CALLS_ENV, str(fake._calls_path))
    monkeypatch.setenv('PATH', f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    assert shutil.which('claude') == str(fake.path), (
        f"the fake claude at {fake.path} must be the one PATH resolves; "
        f"got {shutil.which('claude')!r}"
    )
    return fake


class PoolRoster:
    """Writes ``config/usage-accounts.yaml``-shaped rosters for build_pool.

    ``pool_roster('a', 'b')`` returns ``(accounts_file, env_file)``: a roster
    naming each account, with its ``CLAUDE_OAUTH_TOKEN_<NAME>`` set to
    ``pool_roster.token(name)`` through monkeypatch (so nothing leaks past the
    test), and a guaranteed-empty ``.env`` — never the repo's, which carries
    real tokens when the suite runs from the main checkout.
    ``resolve_tokens=False`` deletes those vars instead, for an empty pool.
    """

    def __init__(self, directory: Path, monkeypatch):
        self._directory = directory
        self._monkeypatch = monkeypatch
        self._written = 0

    @staticmethod
    def token(name: str) -> str:
        return f'tok-{name}'

    @staticmethod
    def token_env(name: str) -> str:
        return 'CLAUDE_OAUTH_TOKEN_' + name.upper().replace('-', '_')

    def __call__(self, *names: str, resolve_tokens: bool = True):
        self._written += 1
        lines = ['accounts:']
        for name in names:
            env = self.token_env(name)
            lines += [f'  - name: {name}', f'    oauth_token_env: {env}']
            if resolve_tokens:
                self._monkeypatch.setenv(env, self.token(name))
            else:
                self._monkeypatch.delenv(env, raising=False)
        accounts_file = self._directory / f'usage-accounts-{self._written}.yaml'
        accounts_file.write_text('\n'.join(lines) + '\n')
        env_file = self._directory / f'empty-{self._written}.env'
        env_file.write_text('')
        return accounts_file, env_file


@pytest.fixture
def pool_roster(tmp_path, monkeypatch):
    """A :class:`PoolRoster` writing into this test's tmp dir."""
    directory = tmp_path / 'pool-roster'
    directory.mkdir()
    return PoolRoster(directory, monkeypatch)
