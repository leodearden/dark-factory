"""Tests for ``scripts/sweep_toolcall_markup.py`` (task 3691, PRD delta).

## AUTHORING HAZARD — every envelope sentinel here is ``\\x3c``-escaped

Not one raw ``chr(60)`` + ``/`` sequence may appear in this file's SOURCE TEXT,
and :func:`test_this_module_spells_no_raw_envelope_literal` enforces that
mechanically by reading the module's own bytes.

The rationale is the one recorded at ``shared/src/shared/toolcall_markup.py``
lines 52-62 and ``fused_memory/utils/toolcall_xml_leak.py`` lines 77-86: writing
the literal verbatim would force any agent editing this file to emit it inside
its own tool-call envelope, which reproduces the very defect under test — the
agent's Write/Edit argument terminates early, truncating the file and silently
dropping the sibling arguments of that same call. ``\\x3c`` is byte-identical at
runtime and never appears verbatim in the file text, so it is immune. This is
not stylistic. Leave it escaped.

## Fixture byte formats are the REAL writers' formats

Escalation records are written by ``escalation.models.Escalation.to_json`` as
``json.dumps(obj, indent=2)`` with **no trailing newline** (models.py:194);
``plan.json`` is written by ``orchestrator.artifacts`` as
``json.dumps(obj, indent=2) + "\\n"`` (artifacts.py:1370). Both were verified to
round-trip byte-exactly against the live corpora at plan time, and the
round-trip-or-refuse precondition (design decision 5) keys on exactly this
distinction — so the fixtures reproduce both conventions rather than picking
one.

No test here touches the live ``data/escalations`` or ``.worktrees-orphaned``
trees: every fixture is built under ``tmp_path``.
"""
import json
import os
import sys
from pathlib import Path

import pytest

# shared/src bootstrap, same idiom and same precedence argument as
# scripts/scan_task_toolcall_leaks.py:113-115. Needed HERE, independently of
# the script under test, so the sentinel-bridge test below pins this file's
# spellings against the enumeration's OWNER rather than against the script's
# re-export of it. conftest.py appends the repo root, which only makes `shared`
# resolve as a namespace package — `shared.toolcall_markup` lives one level
# down at shared/src/shared/ and needs this entry.
_SHARED_SRC = Path(__file__).resolve().parents[2] / 'shared' / 'src'
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

# `import sweep_toolcall_markup` resolves because scripts/tests/conftest.py
# already inserts scripts/ onto sys.path under --import-mode=importlib.
import sweep_toolcall_markup as sweep  # noqa: E402
from shared.toolcall_markup import (  # noqa: E402
    CANONICAL_OPENER_PREFIX,
    PREFILTER_NEEDLES,
    detect,
)
from shared.toolcall_markup import INVOKE_CLOSER as SHARED_INVOKE_CLOSER  # noqa: E402

# ---------------------------------------------------------------------------
# Sentinel spellings. THE ONLY PLACE this file writes an envelope literal.
# ---------------------------------------------------------------------------

#: The bare closing ``invoke`` tag — the terminator the harness parser falls
#: back to when it cannot find the closer it expected (PRD section 2.1).
INVOKE_CLOSER = '\x3c/invoke>'


def _lit(name: str) -> str:
    """The CANONICAL opening tag for pseudo-parameter *name*.

    Spelled ``\\x3cparameter name="X">``. Its closer is named ``parameter``,
    not ``X`` — the one place the two harness dialects are not symmetric, which
    is why :func:`_closer` must be called with ``'parameter'`` to close it.
    """
    return '\x3cparameter name="' + name + '">'


def _closer(name: str) -> str:
    """The closing tag for *name*, spelled ``\\x3c/X>``."""
    return '\x3c/' + name + '>'


def _swallowed(prefix: str, misclose: str, param: str, value: str) -> str:
    """Build a corrupted string: *prefix*, mis-closed, then a dropped param.

    Reproduces the shape measured across the live ``data/escalations`` corpus:
    the caller's intended text, a closing tag for the field it was received as,
    then a canonical pseudo-parameter carrying the argument the harness parser
    silently dropped, then the invoke terminator.
    """
    return (
        prefix
        + _closer(misclose)
        + '\n'
        + _lit(param)
        + value
        + _closer('parameter')
        + '\n'
        + INVOKE_CLOSER
        + '\n'
    )


def _truncated(prefix: str, misclose: str) -> str:
    """Build the B4 last-parameter corruption: mis-close, then nothing.

    This is the shape ALL FIVE corrupted ``design_decisions[].rationale``
    strings in the live ``.worktrees-orphaned`` corpus carry (measured at plan
    time): the mis-closed field was the last parameter of its call, so the tail
    strips to empty and the repair recovers nothing. Empty ``recovered`` is a
    SUCCESS, not a refusal — the value is still truncated back to its prefix.
    """
    return prefix + _closer(misclose) + '\n' + INVOKE_CLOSER + '\n'


# ---------------------------------------------------------------------------
# Document builders, shaped after the real records.
# ---------------------------------------------------------------------------

#: The string-typed holes a real escalation record carries. Measured: every one
#: of the 26 live repairs lands its recovered value in one of these three.
_ESCALATION_HOLES = ('suggested_action', 'root_cause', 'triage_note')


def make_escalation(esc_id: str, status: str, detail: str, **overrides) -> dict:
    """An escalation record with the live field set and the live hole set."""
    record = {
        'id': esc_id,
        'task_id': esc_id.split('-')[1],
        'agent_role': 'implementer',
        'severity': 'blocking',
        'category': 'infra_issue',
        'summary': 'a one-line summary',
        'detail': detail,
        'evidence': [],
        'status': status,
        'resolution': '',
        'created_at': '2026-08-08T00:00:00+00:00',
    }
    for hole in _ESCALATION_HOLES:
        record[hole] = ''
    record.update(overrides)
    return record


def make_plan(task_id: str, rationales) -> dict:
    """A plan.json shaped after the real artifact, with the real nesting.

    The corruption in the live corpus is always at
    ``design_decisions[i].rationale``, whose containing object carries exactly
    two keys — so a repair there recovers nothing and merely truncates. That is
    why the plan lane exercises the B4 path and the escalation lane exercises
    the recover-into-a-hole path.
    """
    return {
        'task_id': task_id,
        'title': f'fixture plan {task_id}',
        'design_decisions': [
            {'decision': f'decision {i}', 'rationale': text}
            for i, text in enumerate(rationales)
        ],
        'steps': [
            {
                'id': 'step-1',
                'type': 'impl',
                'description': 'do the thing',
                'status': 'pending',
                'commit': None,
            }
        ],
        '_schema_version': 1,
    }


# ---------------------------------------------------------------------------
# Byte-exact writers — each mirrors ONE real writer's convention.
# ---------------------------------------------------------------------------


def write_escalation(path: Path, obj: dict) -> Path:
    """Write *obj* the way ``Escalation.to_json`` does: NO trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    return path


def write_plan(path: Path, obj: dict) -> Path:
    """Write *obj* the way ``artifacts`` does: WITH a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    return path


def read_bytes_map(root: Path) -> dict:
    """Snapshot every file under *root* as ``{relpath: bytes}``.

    Used by the "nothing was written" assertions. Symlinks are read THROUGH
    (their target's bytes), and the link-ness of each path is captured
    separately so a test can also prove a link was not replaced by a regular
    file — the ``os.replace``-onto-a-symlink hazard design decision 4 names.
    """
    snapshot = {}
    for path in sorted(root.rglob('*')):
        if path.is_dir():
            continue
        rel = str(path.relative_to(root))
        snapshot[rel] = (path.read_bytes(), path.is_symlink())
    return snapshot


# ---------------------------------------------------------------------------
# The shared tree every later test consumes.
# ---------------------------------------------------------------------------

#: Repo-relative path of a committed-evidence file that quotes real specimens
#: and is replicated into every worktree checkout. Must never be discovered,
#: and must additionally be refused when handed to the writer directly.
INVENTORY_REL = 'docs/task-recovery-2026-05-13/worktree-inventory.json'


@pytest.fixture
def sweep_root(tmp_path) -> Path:
    """A fake repo root carrying both lanes, the decoys, and the skips.

    Layout (every path is load-bearing for at least one test):

    * ``data/escalations/esc-1-1.json`` — status ``pending``, corrupted. The
      non-terminal skip: swept only if the terminal-status gate is missing.
    * ``data/escalations/archive/2026-08-08/esc-2-1.json`` — ``resolved``,
      ``detail`` swallowed ``suggested_action``. The canonical repair, nested
      under ``archive/<date>/`` so recursion is exercised.
    * ``data/escalations/archive/2026-08-08/esc-3-1.json`` — ``dismissed`` and
      clean. Must come out byte-identical: proof the sweep rewrites only what
      it repairs.
    * ``data/escalations/b3-state.json`` — a dict with no ``id``/``status``,
      carrying a flagged string anyway. The not-a-record skip.
    * ``data/escalations/.watch-fire.json`` — DOTFILE with a full record shape.
      Live watcher state; skipped by the explicit dot-component exclusion
      (design decision 8: ``glob.glob`` would skip it silently, ``Path.rglob``
      would sweep it silently — so the choice is made explicitly and tested).
    * ``.worktrees-orphaned/9001-2026/.task/plan.json`` — a REAL FILE with two
      corrupted ``rationale`` values. The plain-file plan repair.
    * ``.worktrees-orphaned/9002-2026/.task/plan.json`` — an ABSOLUTE SYMLINK
      into ``.worktrees/.task-meta/9002/plan.json``, matching all five live
      orphaned plans. Exercises realpath resolution and link preservation.
    * ``.worktrees-orphaned/9001-2026/docs/...worktree-inventory.json`` and
      ``.worktrees-orphaned/9001-2026/other.json`` — decoys that must never be
      discovered. The first is committed evidence replicated into the
      checkout; the second proves the plans glob is the exact
      ``.task/plan.json`` tail and not ``**/*.json``.
    """
    root = tmp_path / 'root'
    escalations = root / 'data' / 'escalations'
    archive = escalations / 'archive' / '2026-08-08'

    # --- escalations lane -------------------------------------------------
    write_escalation(
        escalations / 'esc-1-1.json',
        make_escalation(
            'esc-1-1',
            'pending',
            _swallowed('A pending record.', 'detail', 'suggested_action', 'do a thing'),
        ),
    )
    write_escalation(
        archive / 'esc-2-1.json',
        make_escalation(
            'esc-2-1',
            'resolved',
            _swallowed(
                'The detail the agent meant to write.',
                'detail',
                'suggested_action',
                'Re-run the merge worker after the lock clears.',
            ),
        ),
    )
    write_escalation(
        archive / 'esc-3-1.json',
        make_escalation('esc-3-1', 'dismissed', 'A perfectly clean detail.'),
    )
    write_escalation(
        escalations / 'b3-state.json',
        {
            'generated_at': '2026-08-08T00:00:00+00:00',
            'note': _swallowed('B3 state blob.', 'detail', 'suggested_action', 'nope'),
        },
    )
    write_escalation(
        escalations / '.watch-fire.json',
        make_escalation(
            'esc-9-9',
            'resolved',
            _swallowed('Live watcher state.', 'detail', 'suggested_action', 'nope'),
        ),
    )

    # --- plans lane -------------------------------------------------------
    orphaned = root / '.worktrees-orphaned'
    write_plan(
        orphaned / '9001-2026' / '.task' / 'plan.json',
        make_plan(
            '9001',
            [
                _truncated('The first rationale, mis-closed.', 'rationale'),
                'A clean rationale.',
                _truncated('The third rationale, mis-closed.', 'rationale'),
            ],
        ),
    )

    # The symlinked plan, exactly as all five live orphaned plans are shaped:
    # an ABSOLUTE symlink into the SHARED meta-root.
    meta_target = write_plan(
        root / '.worktrees' / '.task-meta' / '9002' / 'plan.json',
        make_plan('9002', [_truncated('The 9002 rationale, mis-closed.', 'rationale')]),
    )
    link = orphaned / '9002-2026' / '.task' / 'plan.json'
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(meta_target), str(link))

    # --- decoys that must never be discovered -----------------------------
    write_escalation(
        orphaned / '9001-2026' / INVENTORY_REL,
        {'specimen': _swallowed('Quoted specimen.', 'detail', 'suggested_action', 'x')},
    )
    write_escalation(
        orphaned / '9001-2026' / 'other.json',
        {'note': _swallowed('Not a plan.', 'detail', 'suggested_action', 'x')},
    )

    return root


# ---------------------------------------------------------------------------
# Scaffolding self-tests.
# ---------------------------------------------------------------------------


def test_this_module_spells_no_raw_envelope_literal():
    """This file's own SOURCE must never contain a raw ``chr(60)`` + ``/``.

    The mechanical half of the authoring-hazard note in the module docstring.
    Computed at runtime from :func:`chr` so the needle itself is not spelled
    here either — a test that had to write the literal to check for it would
    be the very hazard it guards.
    """
    needle = chr(60) + '/'
    source = Path(__file__).read_text(encoding='utf-8')
    assert needle not in source, (
        'A raw envelope literal was written into this test file. Spell it with '
        'the \\x3c escape instead — see this module\'s docstring for why.'
    )


def test_sentinel_helpers_agree_with_the_shared_enumeration():
    """The locally-spelled sentinels must equal ``shared.toolcall_markup``'s.

    The helpers above deliberately re-spell the literals rather than importing
    them: a test that built its fixtures from the same constant the code under
    test consumes would pass even if that constant were wrong. This test is the
    bridge — it pins the two spellings together ONCE, so every other test can
    use the local helpers and still be testing against the real enumeration.
    """
    assert INVOKE_CLOSER == SHARED_INVOKE_CLOSER
    assert _lit('x') == CANONICAL_OPENER_PREFIX + '"x">'
    assert _closer('parameter') in PREFILTER_NEEDLES


def test_fixture_bytes_match_the_real_writers(sweep_root):
    """Escalation records carry no trailing newline; plan.json carries one."""
    record = sweep_root / 'data' / 'escalations' / 'archive' / '2026-08-08' / 'esc-2-1.json'
    plan = sweep_root / '.worktrees-orphaned' / '9001-2026' / '.task' / 'plan.json'

    record_bytes = record.read_bytes()
    plan_bytes = plan.read_bytes()

    assert not record_bytes.endswith(b'\n'), 'models.py:194 writes no trailing newline'
    assert plan_bytes.endswith(b'\n'), 'artifacts.py:1370 appends a trailing newline'

    # Both must round-trip byte-exactly through json.dumps(indent=2), which is
    # the precondition design decision 5 makes a hard gate on every rewrite.
    assert json.dumps(json.loads(record_bytes), indent=2).encode() == record_bytes
    assert json.dumps(json.loads(plan_bytes), indent=2).encode() + b'\n' == plan_bytes


def test_fixture_corruption_is_actually_detectable(sweep_root):
    """The fixtures must really be corrupted, or every later test is vacuous."""
    record = json.loads(
        (sweep_root / 'data' / 'escalations' / 'archive' / '2026-08-08' / 'esc-2-1.json')
        .read_text(encoding='utf-8')
    )
    assert detect(record['detail']) is not None
    assert record['suggested_action'] == '', 'the recovery target must be a hole'

    plan = json.loads(
        (sweep_root / '.worktrees-orphaned' / '9001-2026' / '.task' / 'plan.json')
        .read_text(encoding='utf-8')
    )
    flagged = [
        d for d in plan['design_decisions'] if detect(d['rationale']) is not None
    ]
    assert len(flagged) == 2, 'two of the three rationales are corrupted by design'


def test_symlinked_plan_fixture_is_an_absolute_symlink(sweep_root):
    """The 9002 plan must be a link, mirroring all five live orphaned plans."""
    link = sweep_root / '.worktrees-orphaned' / '9002-2026' / '.task' / 'plan.json'
    assert link.is_symlink()
    assert Path(os.readlink(link)).is_absolute()
    assert Path(os.path.realpath(link)) == (
        sweep_root / '.worktrees' / '.task-meta' / '9002' / 'plan.json'
    )


# ---------------------------------------------------------------------------
# step-1 — discovery. The two pinned path sets, and NOTHING else.
# ---------------------------------------------------------------------------


def _discovered(root):
    """``{relpath: lane}`` for every target ``discover_targets`` yields."""
    return {
        str(target.path.relative_to(root)): target.lane
        for target in sweep.discover_targets(root)
    }


def test_discovery_yields_exactly_the_two_pinned_path_sets(sweep_root):
    """Discovery is an allowlist of two shapes, not a repo-wide .json walk.

    Pinning the WHOLE mapping (rather than asserting membership of a few
    interesting paths) is deliberate: the hazard this sweep carries is
    over-reach onto files it was never meant to rewrite, and a membership-only
    assertion cannot fail when a new path is wrongly swept in.
    """
    assert _discovered(sweep_root) == {
        'data/escalations/esc-1-1.json': sweep.LANE_ESCALATIONS,
        'data/escalations/b3-state.json': sweep.LANE_ESCALATIONS,
        'data/escalations/archive/2026-08-08/esc-2-1.json': sweep.LANE_ESCALATIONS,
        'data/escalations/archive/2026-08-08/esc-3-1.json': sweep.LANE_ESCALATIONS,
        '.worktrees-orphaned/9001-2026/.task/plan.json': sweep.LANE_PLANS,
        '.worktrees-orphaned/9002-2026/.task/plan.json': sweep.LANE_PLANS,
    }


def test_discovery_recurses_into_the_archive_date_directories(sweep_root):
    """59 of the 60 corrupted live records sit under ``archive/<date>/``.

    A non-recursive escalations glob would therefore find almost nothing while
    still reporting a clean, plausible-looking run.
    """
    found = _discovered(sweep_root)
    assert 'data/escalations/archive/2026-08-08/esc-2-1.json' in found


def test_discovery_skips_dot_prefixed_escalation_files(sweep_root):
    """``.watch-fire.json`` is live watcher state, not a terminal record.

    Design decision 8: it carries a full escalation-record shape, so nothing
    about its CONTENT excludes it. ``glob.glob`` would drop it silently and
    ``Path.rglob`` would sweep it silently, so the choice is made explicitly
    here and asserted rather than inherited from whichever globbing API the
    implementation happened to reach for.
    """
    assert 'data/escalations/.watch-fire.json' not in _discovered(sweep_root)


def test_discovery_never_yields_the_replicated_committed_evidence(sweep_root):
    """``worktree-inventory.json`` is git-tracked evidence that QUOTES specimens.

    It is replicated into every worktree checkout, so an orphaned-worktree walk
    that globbed ``**/*.json`` would find a 355 KB file full of verbatim leak
    text and "repair" the evidence. Hazard 1 in the plan.
    """
    found = _discovered(sweep_root)
    assert not [path for path in found if 'worktree-inventory.json' in path]


def test_discovery_plans_glob_is_the_exact_task_plan_tail(sweep_root):
    """Only ``<orphaned>/.task/plan.json`` — never any other .json beneath it."""
    found = _discovered(sweep_root)
    assert '.worktrees-orphaned/9001-2026/other.json' not in found
    for path, lane in found.items():
        if lane == sweep.LANE_PLANS:
            assert path.endswith('/.task/plan.json')


def test_discovery_is_deterministic_and_sorted(sweep_root):
    """Two calls agree, and the order is sorted.

    The report is diffed between runs by an operator, so an unstable order
    would manufacture churn that looks like new corruption.
    """
    first = sweep.discover_targets(sweep_root)
    second = sweep.discover_targets(sweep_root)
    assert first == second
    assert first == sorted(first)


def test_discovery_tolerates_absent_lane_directories(tmp_path):
    """A root with neither lane present yields nothing and does not raise.

    ``.worktrees-orphaned`` only exists once the reclaim timer has rotated at
    least one lane, so an empty or fresh checkout is an ordinary state, not an
    error.
    """
    assert sweep.discover_targets(tmp_path / 'empty-root') == []
