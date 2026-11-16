"""Guard test: ``durable_archive_path`` is the ONLY session-id-keyed locator
into the durable transcript archive (invariant I-E, task 3730 / boundary B7).

THE INVARIANT, as ``shared/src/shared/transcript_archive.py``'s module
docstring already states it in prose: exactly one function knows how to turn a
``(task_id, session_id)`` pair into a path inside the archive. Everything that
needs one CALLS it. Task 3578's ``restore_archived_transcript`` obeys this —
it invokes the locator rather than re-globbing, and says why at its own call
site — and ``Harness._archive_available`` obeys it too.

WHY IT NEEDS A GATE RATHER THAN A CONVENTION. The archive layout is not one
fact but four in a trench coat: the ``<task_id>/<encoded-cwd>/`` nesting, the
``.jsonl*`` suffix that spans task 3618's gzip drop, the ``is_file()`` filter
that excludes the subagent DIRECTORY the same pattern would otherwise match,
and the newest-mtime-with-path-tiebreak choice among duplicates. A second
finder would reproduce the easy half and silently drop the rest — most likely
the tiebreak, which is what makes a resume reproducible. Since task 3730 the
lookup is an ELIGIBILITY input rather than instrumentation, so a divergent
second finder now costs resumes, not just a telemetry field.

The risk is entirely that a future change ADDS a finder. That is a statement
about code which does not exist yet, so an assertion over the locator's
existing CALLERS cannot detect it: this gate scans for the SHAPE instead, and
fails when a new site appears.

WHAT COUNTS AS A SITE. A ``.glob()``/``.rglob()`` call whose pattern is an
f-string interpolating a session-id-bearing name — the only way to key a
filesystem lookup by session id. AST-based, never a text grep, so the prose
above (which is full of glob patterns) cannot trip it.

SCOPE. Production ``*/src`` trees only. That is the scope the invariant is
actually about, and it keeps the two known TEST-tree globs
(``shared/tests/test_startup_completion_fixtures.py`` and this package's
``test_session_resume_integration_gate.py``, both against the LIVE config tree)
out structurally rather than by an allowlist that would grow with every new
fixture. ``scripts/`` is out for the same reason: ``scripts/legibility/
inventory.py`` and ``fused-memory/scripts/memory_eval_transcript_corpus.py``
parse the archive's path STRUCTURE for enumeration and perform no session-id
lookup, which is the distinction this gate exists to draw.

THE ALLOWLIST is for sites that glob BY SESSION ID but not against the ARCHIVE
— the live CLI config tree is a different store with a different layout, and
conflating the two is the confusion the gate is meant to prevent, not one it
should silence.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import NamedTuple

import pytest
from _tree_scan import (
    REPO_ROOT,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
    python_files,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

# Applied to paths RELATIVE to REPO_ROOT, never to their absolute parts. This
# is not a stylistic choice: every checkout of this repo used by an agent lives
# under a path CONTAINING '.worktrees', so an absolute-parts filter skips
# literally every file and the sweep silently scans nothing. Measured while
# writing this gate — the first draft returned 0 files from a task worktree and
# passed every assertion below except the vacuity floor. That floor is what
# caught it, which is exactly what it is for.
_SKIP_PARTS = frozenset({
    '.worktrees', '.venv', 'node_modules', '__pycache__', '.git', 'plans',
})

# Minimum production .py files the sweep must visit. MEASURED at 425 on
# 2026-09-07, so the floor sits ~2x below reality: low enough to survive
# packages moving, high enough that a filter regression trips it. A raw-count
# floor is a COLLAPSE detector only and cannot see a partial hole — the same
# concession test_killpg_frozen_pgid_guard.py::_MIN_SCANNED_FILES records — but
# the collapse it does catch is the one that actually happened here.
_MIN_SCANNED_FILES = 200

_GLOB_METHODS = frozenset({'glob', 'rglob'})

# The ONE production site permitted to glob the archive by session id.
_ARCHIVE_LOCATOR = ('shared/src/shared/transcript_archive.py', 'durable_archive_path')

# Interpolated into the failure message to point a future author at where the
# invariant is stated in prose, and so at where a deliberate change of policy
# belongs. A message POINTER only: nothing here asserts on that docstring's
# wording, which would pin prose rather than behaviour and would go red on a
# rewrite that left the invariant perfectly intact. Cited as `path::symbol` per
# CLAUDE.md rather than a line pin.
_PROSE_HOME = 'shared/src/shared/transcript_archive.py (module docstring)'

# Sites that interpolate a session id into a glob against the LIVE CLI CONFIG
# TREE rather than the archive. Each carries its reason inline so a future
# reader can tell a legitimate live-tree lookup from the archive re-glob this
# gate forbids.
#
# KEYED BY (path, ENCLOSING FUNCTION), not by path and not by line. Not by line
# because line pins rot — this task's own revalidation absorbed a 140-line
# shift in a cited file that had not changed a byte. Not by path because the
# archival WRITE side lives in the SAME module as the locator, so a path-only
# allowlist for transcript_archive.py would exempt `durable_archive_path`
# itself and leave this gate asserting nothing at all.
#
# Note the census counts THREE glob calls but only TWO entries: the archival
# write side issues two adjacent globs (the session transcript and its
# subagents) inside one function, and the function is the unit that carries the
# reason.
_LIVE_CONFIG_TREE_ALLOWLIST: dict[tuple[str, str], str] = {
    ('shared/src/shared/cli_invoke.py', '_resolve_transcript_path'):
        'the CLI-facing locator for the CLI\'s OWN config tree '
        '(<config_dir>/projects/*/<sid>.jsonl) — a different store with a '
        'different layout, and the thing the archive is a copy OF.',
    ('shared/src/shared/transcript_archive.py', 'archive_task_transcripts'):
        'the archival WRITE side: it enumerates SOURCES in the live config '
        'tree to copy INTO the archive, so it reads the store the archive is '
        'derived from, never the archive itself.',
}


class _Site(NamedTuple):
    """One session-id-keyed glob call found in production source."""

    path: str      # POSIX path relative to REPO_ROOT
    function: str  # enclosing def, or '<module>' at module scope
    lineno: int
    pattern: str   # the f-string as source, for the failure message


# An identifier "names a session id" when one of its underscore-separated
# tokens is a known spelling. Token-wise rather than substring, so `resid` and
# `considered` are not session ids while `sid`, `session`, `session_id`,
# `sess_id` and `agent_session_id` all are. Deliberately generous about the
# spelling and strict about the boundary: a false POSITIVE costs one allowlist
# entry with a reason, while a false NEGATIVE is a second archive locator this
# gate was supposed to catch.
_SESSION_TOKENS = frozenset({'sid', 'sess', 'session', 'sessionid', 'sessid'})


def _looks_like_session_id(name: str) -> bool:
    tokens = [t for t in re.split(r'[^a-z0-9]+', name.lower()) if t]
    return any(t in _SESSION_TOKENS for t in tokens)


def _production_python_files() -> tuple[Path, ...]:
    """Every ``.py`` under a top-level ``<package>/src`` tree.

    Anchored at ``_tree_scan.py::REPO_ROOT``, which is resolved from a file
    and never from the process CWD: merge-verify runs pytest from
    ``orchestrator/`` while a plain ``pytest orchestrator/tests`` runs from the
    repo root, and the census must come out identical under both.  A src root
    whose own path crosses a skipped part is dropped whole, so the skip stays
    judged relative to REPO_ROOT end to end.
    """
    src_roots = sorted(
        root for root in REPO_ROOT.glob('*/src')
        if root.is_dir() and _SKIP_PARTS.isdisjoint(root.relative_to(REPO_ROOT).parts)
    )
    return python_files(*src_roots, exclude_parts=_SKIP_PARTS)


def _session_id_globs(source: str, relpath: str) -> list[_Site]:
    """Every session-id-keyed glob call in *source*.

    Fails SOFT on an unparseable module (returns ``[]`` rather than raising),
    matching the sibling guards: a file mid-edit, or a fixture malformed on
    purpose, must not turn this gate red.

    KNOWN LIMITATION, stated rather than papered over. A pattern built at
    runtime (``pattern = f'...'; root.glob(pattern)``), passed by keyword, or
    concatenated from parts is NOT matched, and neither is a lookup that walks
    the archive with ``os.scandir``/``os.listdir`` instead of globbing. The
    detector is a FLOOR on coverage, not a proof of totality; the falsifiability
    rows below are what stop that concession from becoming vacuous.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    return _session_id_globs_in(tree, relpath)


def _session_id_globs_in(tree: ast.Module, relpath: str) -> list[_Site]:
    """:func:`_session_id_globs` over an already-parsed module."""
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def _enclosing_function(node: ast.AST) -> str:
        cur = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.FunctionDef | ast.AsyncFunctionDef):
                return cur.name
        return '<module>'

    sites: list[_Site] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _GLOB_METHODS):
            continue
        if not node.args:
            continue
        pattern = node.args[0]
        if not isinstance(pattern, ast.JoinedStr):
            continue
        names: set[str] = set()
        for piece in pattern.values:
            if not isinstance(piece, ast.FormattedValue):
                continue
            for sub in ast.walk(piece.value):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
                elif isinstance(sub, ast.Attribute):
                    names.add(sub.attr)
        if not any(_looks_like_session_id(n) for n in names):
            continue
        sites.append(_Site(
            path=relpath,
            function=_enclosing_function(node),
            lineno=node.lineno,
            pattern=ast.unparse(pattern),
        ))
    return sites


def _archive_sites(module: ParsedModule) -> tuple[_Site, ...]:
    """Session-id-keyed globs in *module*: loud if it cannot be read, none if it cannot be parsed.

    ``may_name`` reads the source strictly, so an undecodable file raises there,
    before the parse's fail-soft ``ValueError`` clause could swallow its
    ``UnicodeDecodeError``.
    """
    if not module.may_name(*_GLOB_METHODS):
        return ()
    try:
        tree = module.tree
    except (SyntaxError, ValueError):
        return ()
    return tuple(_session_id_globs_in(tree, module.path.relative_to(REPO_ROOT).as_posix()))


ANALYSIS = TreeAnalysis(name=Path(__file__).stem, files=_production_python_files, extract=_archive_sites)
TREE_ANALYSES = (ANALYSIS,)


def _scan(tree_pass: TreePass) -> list[_Site]:
    return [site for sites in tree_pass.facts(ANALYSIS).values() for site in sites]


def _violation_message(unexpected: list[_Site]) -> str:
    """Build the failure text a future author actually reads.

    It names the invariant and then argues the remedy, because the reflex on
    seeing a red allowlist-shaped gate is to ADD AN ALLOWLIST ENTRY, and for
    THIS gate that is precisely the wrong move: a real second archive locator
    allowlisted is the invariant deleted, quietly, by the very person the guard
    was written for. So the message says CALL the locator instead, and says WHY
    — which parts of the layout a re-glob drops — because it has to win that
    argument against someone mid-refactor and in a hurry.
    """
    listed = '\n'.join(
        f'  - {s.path}::{s.function} (line {s.lineno}): {s.pattern}'
        for s in unexpected
    )
    return (
        'I-E VIOLATED: durable_archive_path must be the SINGLE session-id-keyed '
        'locator into the durable transcript archive, but these production '
        'sites glob by session id and are neither it nor an allowlisted '
        'live-config-tree lookup:\n'
        f'{listed}\n\n'
        'If the new site looks up a transcript IN THE ARCHIVE: do not allowlist '
        'it — CALL shared.transcript_archive.durable_archive_path instead, as '
        'restore_archived_transcript and Harness._archive_available both do. A '
        'second finder reproduces the easy half of the layout and drops the '
        'rest (the .jsonl* suffix spanning the gzip drop, the is_file() filter '
        'that excludes the subagent directory, the newest-mtime-with-path '
        'tiebreak that makes a resume reproducible). Since task 3730 that '
        'answer decides ELIGIBILITY, so a divergent finder costs resumes.\n'
        'If it globs the LIVE CLI config tree instead — a different store with '
        'a different layout — add it to _LIVE_CONFIG_TREE_ALLOWLIST keyed by '
        '(path, enclosing function) WITH the reason inline.\n'
        f'The invariant is stated in prose at {_PROSE_HOME}, which is where a '
        'change of policy belongs — not here.'
    )


def test_durable_archive_path_is_the_sole_archive_locator(tree_pass: TreePass) -> None:
    """I-E: exactly one production site globs the archive by session id."""
    sites = _scan(tree_pass)
    unexpected = [
        s for s in sites
        if (s.path, s.function) != _ARCHIVE_LOCATOR
        and (s.path, s.function) not in _LIVE_CONFIG_TREE_ALLOWLIST
    ]
    assert not unexpected, _violation_message(unexpected)

    # ...and the locator is still THERE. The assertion above is satisfied by an
    # empty tree, so without this the gate would go green if the locator were
    # deleted or renamed — the state in which a second finder is most likely to
    # be written next.
    locators = [s for s in sites if (s.path, s.function) == _ARCHIVE_LOCATOR]
    assert len(locators) == 1, (
        f'expected exactly one glob inside {_ARCHIVE_LOCATOR[0]}::'
        f'{_ARCHIVE_LOCATOR[1]}, found {len(locators)}: {locators}'
    )
    assert '.jsonl*' in locators[0].pattern, (
        'the locator lost its `.jsonl*` suffix — that trailing star is what '
        'spans task 3618\'s gzip drop, matching both `.jsonl.gz` today and a '
        'plain `.jsonl` tomorrow, so narrowing it silently un-finds every '
        f'archived transcript: {locators[0].pattern}'
    )

    # Smoke the message builder on a synthetic offender: it must INTERPOLATE
    # its argument, i.e. localise the offender at all. Exercised here rather
    # than in a test of its own because the green path never runs it — the
    # `assert not unexpected, _violation_message(unexpected)` above evaluates
    # its message lazily, so on a clean tree a refactor that broke these
    # f-strings (a bad field name raising at format time) would ship undetected
    # and surface only at the moment the gate was supposed to explain a real
    # violation. Deliberately NO assertion on the message's wording: that pins
    # the prose of a diagnostic, so any clearer rewrite would go red while
    # nothing had regressed.
    msg = _violation_message([
        _Site(
            path='pkg/src/pkg/rogue.py',
            function='find_it',
            lineno=42,
            pattern="f'{task_id}/*/{session_id}.jsonl*'",
        ),
    ])
    assert 'pkg/src/pkg/rogue.py::find_it' in msg
    assert 'line 42' in msg


def test_every_allowlist_entry_is_still_live(tree_pass: TreePass) -> None:
    """An allowlist entry whose site is gone must be DELETED, not left to rot.

    A stale entry is worse than no entry: it is standing permission for the
    next author to re-add a glob at that exact (path, function) — the one place
    a second archive locator would be least visible.
    """
    found = {(s.path, s.function) for s in _scan(tree_pass)}
    stale = sorted(set(_LIVE_CONFIG_TREE_ALLOWLIST) - found)
    assert not stale, (
        f'these allowlist entries no longer match any site: {stale}. Delete '
        'them — an entry with nothing behind it is standing permission for a '
        'future glob at the same spot.'
    )


def test_restore_calls_the_locator_rather_than_reglobbing() -> None:
    """POSITIVE half of I-E: task 3578's restore CALLS the locator.

    The census above fails when a second FINDER appears; this fails when the
    one existing consumer stops consuming — the same violation arriving from
    the other direction, and the likelier one under a refactor that inlines a
    "simple" glob for speed. Without it the gate would stay green while
    restore_archived_transcript quietly grew its own path arithmetic, because
    a hand-rolled `Path(root) / task / enc / f'{sid}.jsonl.gz'` composition
    interpolates no session id into a GLOB and so is invisible to the scan.
    """
    src = (REPO_ROOT / 'shared/src/shared/transcript_archive.py').read_text()
    tree = ast.parse(src)
    restore = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
         and n.name == 'restore_archived_transcript'),
        None,
    )
    assert restore is not None, (
        'restore_archived_transcript is gone from shared/transcript_archive.py '
        '— if it moved, re-point this gate at its new home rather than '
        'deleting the assertion.'
    )
    calls = [
        n for n in ast.walk(restore)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == 'durable_archive_path')
            or (isinstance(n.func, ast.Attribute)
                and n.func.attr == 'durable_archive_path')
        )
    ]
    assert calls, (
        'restore_archived_transcript no longer CALLS durable_archive_path. It '
        'must locate the archived transcript through the sole locator, not '
        'compose or re-glob the path itself — see that function\'s own comment '
        'on why it calls rather than re-globs.'
    )


def test_sweep_is_not_vacuous(tree_pass: TreePass) -> None:
    """A broken walk must not pass as a clean tree.

    The floor is load-bearing rather than ceremonial, with a measured witness:
    this gate's first draft filtered ``_SKIP_PARTS`` against ABSOLUTE path
    parts, and every checkout an agent works in lives under a directory named
    ``.worktrees`` — so the sweep visited 0 files and the census assertion
    above passed on an empty set. This is the assertion that caught it.
    """
    scanned = tree_pass.facts(ANALYSIS)
    assert len(scanned) >= _MIN_SCANNED_FILES, (
        f'sweep scanned only {len(scanned)} production .py files, expected '
        f'>= {_MIN_SCANNED_FILES} — the gate is vacuous. Most likely the '
        '_SKIP_PARTS filter is being applied to ABSOLUTE path parts (every '
        'task worktree lives under a `.worktrees` directory, so that skips '
        'everything), or the */src root glob no longer matches this layout.'
    )
    listed = _production_python_files()
    assert len(listed) == len(set(listed)), 'the walk visited a path twice'


# ── Falsifiability: a guard that cannot go red is measuring nothing ──────────
def test_detector_flags_a_second_archive_locator() -> None:
    """The exact shape this gate exists to forbid."""
    source = (
        'def find_it(archive_root, task_id, session_id):\n'
        "    return list(archive_root.glob(f'{task_id}/*/{session_id}.jsonl*'))\n"
    )
    sites = _session_id_globs(source, 'pkg/src/pkg/rogue.py')
    assert [(s.function, s.lineno) for s in sites] == [('find_it', 2)]


def test_detector_flags_an_alternative_session_id_spelling() -> None:
    """A rename must not buy an exemption: `sid`, `sess_id` and
    `agent_session_id` are all session ids, so a second locator cannot hide
    behind a variable name."""
    for name in ('sid', 'sess_id', 'agent_session_id', 'session'):
        source = (
            f'def find_it(root, {name}):\n'
            f"    return list(root.glob(f'*/{{{name}}}.jsonl*'))\n"
        )
        assert _session_id_globs(source, 'p/src/p/m.py'), name


def test_detector_ignores_a_glob_with_no_session_id() -> None:
    """Interpolating something else — an escalation id, a task id — is not a
    session-id-keyed lookup. session_registry.py's escalation-id glob is the
    live in-repo witness for this arm."""
    source = (
        'def by_escalation(root, escalation_id):\n'
        "    return list(root.glob(f'{escalation_id}/*.json'))\n"
    )
    assert _session_id_globs(source, 'p/src/p/m.py') == []


def test_detector_ignores_a_literal_pattern() -> None:
    """A constant pattern keys on nothing and cannot be a locator."""
    source = (
        'def all_of_them(root):\n'
        "    return list(root.glob('*/*.jsonl'))\n"
    )
    assert _session_id_globs(source, 'p/src/p/m.py') == []


def test_detector_ignores_prose_mentions() -> None:
    """AST, not grep: this module's own docstring is full of glob patterns
    interpolating a session id, and so is transcript_archive.py's."""
    source = (
        '"""Globs archive_root.glob(f\'{task_id}/*/{session_id}.jsonl*\').\n'
        '\n'
        'Do not re-glob: call durable_archive_path.\n'
        '"""\n'
        "# root.glob(f'{session_id}.jsonl') -- example only\n"
        'X = 1\n'
    )
    assert _session_id_globs(source, 'p/src/p/m.py') == []


def test_detector_fails_soft_on_a_syntax_error() -> None:
    """A file mid-edit must not turn the gate red."""
    assert _session_id_globs('def broken(:\n', 'p/src/p/m.py') == []


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('session_id', True),
        ('sid', True),
        ('sess_id', True),
        ('agent_session_id', True),
        ('session', True),
        ('task_id', False),
        ('escalation_id', False),
        ('resid', False),      # substring 'sid', but not a token
        ('considered', False),  # substring 'sid' AND 'sess'
        ('', False),
    ],
)
def test_looks_like_session_id_boundaries(name: str, expected: bool) -> None:
    """Token-wise, never substring — the arm that keeps the allowlist from
    growing false positives while still catching a renamed locator."""
    assert _looks_like_session_id(name) is expected
