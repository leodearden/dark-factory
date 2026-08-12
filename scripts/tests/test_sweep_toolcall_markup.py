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


# ---------------------------------------------------------------------------
# step-3 — the must-not-touch guard, DECLARED and machine-checked.
# ---------------------------------------------------------------------------

#: Repo-relative path of the second committed-evidence file: 41 verbatim leak
#: records captured by the 2026-08-05 dry-run sweep.
DRY_RUN_REPORT_REL = 'docs/toolcall-xml-leak-sweep-2026-08-05/dry-run-report.json'


def test_never_touch_is_a_declared_frozenset_naming_both_evidence_files():
    """The guard is a DECLARED constant, not an emergent property of the glob.

    Discovery already fails to yield these two files, so this constant is
    strictly redundant against today's ``discover_targets`` — and that is the
    point. It is defence in depth against a future widening of the globs, and
    it makes the "never touch committed evidence" rule greppable rather than
    implicit in a path pattern.
    """
    assert isinstance(sweep.NEVER_TOUCH, frozenset)
    assert INVENTORY_REL in sweep.NEVER_TOUCH
    assert DRY_RUN_REPORT_REL in sweep.NEVER_TOUCH


def test_never_touch_refuses_a_directly_handed_target(sweep_root):
    """Handed straight to the writer gate, bypassing discovery, it still refuses.

    The realistic path to this bug is not a bad glob — it is an operator (or a
    future caller) passing a path in directly. The guard has to live at the
    write boundary, not only at discovery.
    """
    evidence = sweep_root / INVENTORY_REL
    write_escalation(evidence, {'specimen': 'quoted leak text'})
    before = evidence.read_bytes()

    outcome = sweep.resolve_write_target(
        sweep.Target(path=evidence, lane=sweep.LANE_ESCALATIONS), sweep_root
    )

    assert isinstance(outcome, sweep.Refusal)
    assert outcome.reason == sweep.REASON_NEVER_TOUCH
    assert str(evidence) in str(outcome.path), 'the refusal must name the path'
    assert evidence.read_bytes() == before


def test_never_touch_refuses_via_realpath_not_just_the_literal_path(sweep_root):
    """A symlink whose TARGET is committed evidence is refused too.

    Checking only the discovered path would be defeated by exactly the shape
    this corpus is full of: every live orphaned plan is a symlink. The guard
    resolves first, then matches.
    """
    evidence = sweep_root / DRY_RUN_REPORT_REL
    write_escalation(evidence, {'records': ['verbatim leak']})
    before = evidence.read_bytes()

    link = sweep_root / '.worktrees-orphaned' / '9003-2026' / '.task' / 'plan.json'
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(evidence), str(link))

    outcome = sweep.resolve_write_target(
        sweep.Target(path=link, lane=sweep.LANE_PLANS), sweep_root
    )

    assert isinstance(outcome, sweep.Refusal)
    assert outcome.reason == sweep.REASON_NEVER_TOUCH, (
        'never-touch must be checked BEFORE the plan-location gate, so the '
        'refusal names the real hazard rather than a generic location miss'
    )
    assert evidence.read_bytes() == before


def test_plan_symlink_outside_the_sanctioned_locations_is_refused(sweep_root, tmp_path):
    """A plan link resolving anywhere else gets a DISTINCT reason.

    The two sanctioned locations are the orphaned worktree's own ``.task/``
    (a plain file) and ``<root>/.worktrees/.task-meta/<id>/plan.json`` (the
    shared meta-root all five live orphaned plans point into). Anything else
    means the link is not what this sweep thinks it is, and following it would
    write through to an unknown file.
    """
    outside = tmp_path / 'elsewhere' / 'plan.json'
    write_plan(outside, make_plan('9004', ['clean']))
    before = outside.read_bytes()

    link = sweep_root / '.worktrees-orphaned' / '9004-2026' / '.task' / 'plan.json'
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(outside), str(link))

    outcome = sweep.resolve_write_target(
        sweep.Target(path=link, lane=sweep.LANE_PLANS), sweep_root
    )

    assert isinstance(outcome, sweep.Refusal)
    assert outcome.reason == sweep.REASON_UNSANCTIONED_PLAN_LOCATION
    assert outcome.reason != sweep.REASON_NEVER_TOUCH, 'a DISTINCT reason'
    assert outside.read_bytes() == before


def test_sanctioned_plan_targets_resolve(sweep_root):
    """Both sanctioned shapes resolve, and the meta-root link resolves THROUGH.

    The resolved ``write_path`` is what ``os.replace`` will land on, so for the
    symlinked plan it must be the meta-root FILE, never the link itself —
    replacing the link with a regular file re-forks the lane and meta-root
    copies (the esc-5205-9 divergence plan_tools:715 documents).
    """
    plain = sweep_root / '.worktrees-orphaned' / '9001-2026' / '.task' / 'plan.json'
    resolved = sweep.resolve_write_target(
        sweep.Target(path=plain, lane=sweep.LANE_PLANS), sweep_root
    )
    assert isinstance(resolved, sweep.ResolvedTarget)
    assert resolved.write_path == plain

    link = sweep_root / '.worktrees-orphaned' / '9002-2026' / '.task' / 'plan.json'
    resolved_link = sweep.resolve_write_target(
        sweep.Target(path=link, lane=sweep.LANE_PLANS), sweep_root
    )
    assert isinstance(resolved_link, sweep.ResolvedTarget)
    assert resolved_link.write_path == (
        sweep_root / '.worktrees' / '.task-meta' / '9002' / 'plan.json'
    )
    assert resolved_link.write_path != link, 'must resolve THROUGH the symlink'


def test_refusals_are_returned_not_raised(sweep_root):
    """Refusals are structured VALUES carrying (path, lane, reason).

    Never raised-and-swallowed: `scripts` is inside
    shared/tests/silent_fallthrough_scan.py's _SCOPE_ROOTS, so a broad
    `except Exception` funnelling into a default would trip the ratchet — and
    would also lose the reason a human needs to adjudicate the refusal.
    """
    evidence = sweep_root / INVENTORY_REL
    write_escalation(evidence, {'specimen': 'x'})
    refusal = sweep.resolve_write_target(
        sweep.Target(path=evidence, lane=sweep.LANE_ESCALATIONS), sweep_root
    )
    assert isinstance(refusal, tuple), 'a NamedTuple value, not an exception'
    assert isinstance(refusal, sweep.Refusal)
    assert refusal.path is not None
    assert refusal.lane == sweep.LANE_ESCALATIONS
    assert isinstance(refusal.reason, str) and refusal.reason


# ---------------------------------------------------------------------------
# step-5 — repair_document, the uniform per-OBJECT rule.
# ---------------------------------------------------------------------------


def _swallowed_echo(prefix: str, misclose: str, param: str, value: str) -> str:
    """The NAME-ECHOING dialect: the dropped param closes on its own name.

    The model drifts into this spelling, and it is not symmetric with the
    canonical one — a canonical opener closes with ``parameter`` while this one
    closes with ``X``. Both dialects appear in the live corpus, so both are
    exercised.
    """
    return (
        prefix
        + _closer(misclose)
        + '\n'
        + '\x3c'
        + param
        + '>'
        + value
        + _closer(param)
        + '\n'
        + INVOKE_CLOSER
        + '\n'
    )


def test_repair_restores_a_swallowed_sibling_into_its_string_hole():
    """Case (a): the canonical repair, and invariant D5 asserted DIRECTLY.

    The uniform rule: schema_params = the containing object's own keys, legal
    targets = sibling keys currently holding a string HOLE, supplied =
    everything else. So a recovered name must name a real field of THIS record
    and must land on a hole — never displacing authored content.
    """
    intended = 'The detail the agent meant to write.'
    action = 'Re-run the merge worker after the lock clears.'
    record = make_escalation(
        'esc-7-1', 'resolved', _swallowed_echo(intended, 'detail', 'suggested_action', action)
    )
    original = json.loads(json.dumps(record))

    repaired, outcomes = sweep.repair_document(record)

    assert repaired['detail'] == intended
    assert repaired['suggested_action'] == action

    # D5, asserted rather than assumed: recovery never fabricates.
    assert original['detail'].startswith(repaired['detail']), 'clean value is a PREFIX'
    assert repaired['suggested_action'] in original['detail'], 'recovered is a SUBSTRING'

    # Every other key identical by value AND type.
    for key in original:
        if key in ('detail', 'suggested_action'):
            continue
        assert repaired[key] == original[key]
        assert type(repaired[key]) is type(original[key])

    assert [o.action for o in outcomes] == [sweep.ACTION_REPAIRED]
    assert outcomes[0].field == 'detail'
    assert outcomes[0].recovered_names == ('suggested_action',)


def test_repair_refuses_when_the_recovered_name_is_a_list_typed_field():
    """Case (b): a tail naming ``evidence`` while ``evidence: []``.

    This is the single largest refusal class measured live. A bare str landing
    on a list-typed field would change that field's TYPE, so a non-string is
    never a legal target — it stays in ``supplied`` and repair()'s own B9
    condition does the refusing. The document comes back byte-identical.

    Uses the CANONICAL dialect deliberately, because that is what the live
    corpus carries for this class — and the dialect is load-bearing here. See
    :func:`test_echo_dialect_falls_through_to_a_later_candidate` for the
    measured asymmetry.
    """
    record = make_escalation(
        'esc-7-2', 'resolved', _swallowed('Detail.', 'detail', 'evidence', 'some text')
    )
    before = json.dumps(record, indent=2)

    repaired, outcomes = sweep.repair_document(record)

    assert json.dumps(repaired, indent=2) == before, 'byte-identical'
    assert [o.action for o in outcomes] == [sweep.ACTION_REFUSED]
    assert outcomes[0].reason == sweep.REASON_NO_STRING_HOLE_TARGET
    assert repaired['evidence'] == [], 'type unchanged'


def test_repair_refuses_a_recovered_name_absent_from_the_record():
    """Case (c): the recovered name is not a field of THIS record at all.

    Stricter than a parameter-name-keyed recovery would be: it is what stops a
    junk key being invented beside a real one.
    """
    record = make_escalation(
        'esc-7-3', 'resolved', _swallowed_echo('Detail.', 'detail', 'not_a_field', 'text')
    )
    before = json.dumps(record, indent=2)

    repaired, outcomes = sweep.repair_document(record)

    assert json.dumps(repaired, indent=2) == before
    assert [o.action for o in outcomes] == [sweep.ACTION_REFUSED]
    assert 'not_a_field' not in repaired, 'no key may be invented'
    assert outcomes[0].reason != sweep.REASON_NO_STRING_HOLE_TARGET, (
        'a name that is not a field at all is a DIFFERENT refusal from a name '
        'that is a field but holds content'
    )


def test_repair_refuses_rather_than_displacing_authored_text():
    """Case (d): the named sibling already holds authored content (B9).

    Design decision 2's restore-or-refuse rule. Overwriting here would destroy
    a real value to rescue another — strictly worse than leaving the corruption
    in place, where the swallowed text at least still survives.
    """
    authored = 'A root cause someone actually wrote.'
    record = make_escalation(
        'esc-7-4',
        'resolved',
        _swallowed('Detail.', 'detail', 'root_cause', 'the recovered one'),
        root_cause=authored,
    )
    before = json.dumps(record, indent=2)

    repaired, outcomes = sweep.repair_document(record)

    assert json.dumps(repaired, indent=2) == before
    assert repaired['root_cause'] == authored, 'authored text survives untouched'
    assert [o.action for o in outcomes] == [sweep.ACTION_REFUSED]
    assert outcomes[0].reason == sweep.REASON_NO_STRING_HOLE_TARGET


def test_repair_reaches_corruption_nested_in_a_list_of_objects():
    """Case (e): ``design_decisions[1].rationale``, via the recursive walk.

    Every corrupted string in the live orphaned-plan corpus is at this depth,
    so a top-level-only walk would find none of them.
    """
    plan = make_plan(
        '9005',
        [
            'clean first',
            _truncated('The second rationale.', 'rationale'),
            'clean third',
        ],
    )
    original = json.loads(json.dumps(plan))

    repaired, outcomes = sweep.repair_document(plan)

    decisions = repaired['design_decisions']
    assert decisions[1]['rationale'] == 'The second rationale.'
    assert decisions[0] == original['design_decisions'][0], 'siblings untouched'
    assert decisions[2] == original['design_decisions'][2], 'siblings untouched'
    assert decisions[1]['decision'] == original['design_decisions'][1]['decision']
    assert repaired['steps'] == original['steps']
    assert repaired['_schema_version'] == 1

    assert [o.action for o in outcomes] == [sweep.ACTION_REPAIRED]
    assert outcomes[0].json_path == 'design_decisions[1].rationale'
    assert outcomes[0].recovered_names == (), 'B4 last-parameter: nothing dropped'


def test_repair_of_a_clean_document_is_identity_with_no_outcomes():
    """Case (f): unchanged, zero outcomes, and the SAME object back.

    Identity matters beyond tidiness: the sweep decides whether to rewrite a
    file by whether anything changed, so a repairer that deep-copied every
    document unconditionally would make every file look dirty.
    """
    record = make_escalation('esc-7-5', 'dismissed', 'A perfectly clean detail.')

    repaired, outcomes = sweep.repair_document(record)

    assert outcomes == []
    assert repaired is record, 'no copy churn on the clean path'


def test_repair_leaves_quoting_prose_alone_when_it_cannot_be_repaired():
    """Prose that merely QUOTES an envelope literal is flagged but not rewritten.

    81 of the 144 detect()-flagged strings measured live are this: records
    ABOUT markup incidents, 39 of them in evidence[].observation. detect() is
    deliberately wider than repair(), which is why repair() — not detect() — is
    the gate on every write.
    """
    prose = 'The agent emitted ' + _closer('description') + ' and the argument was dropped.'
    record = make_escalation('esc-7-6', 'dismissed', prose)
    before = json.dumps(record, indent=2)

    repaired, outcomes = sweep.repair_document(record)

    assert json.dumps(repaired, indent=2) == before
    assert [o.action for o in outcomes] == [sweep.ACTION_REFUSED]


def test_echo_dialect_falls_through_to_a_later_candidate():
    """MEASURED asymmetry between the two dialects. Documented, not asserted away.

    When the tail names a sibling that is NOT a legal target, what repair()
    does next depends on which dialect the tail is written in:

    * CANONICAL — the later candidate is the ``parameter`` closer, and the
      clean prefix it would produce still contains the opener prefix, which IS
      an enumerated literal. repair()'s prefix-clean accept condition therefore
      rejects it and the whole value is refused. This is the live corpus's
      shape and the source of the 37 measured refusals.
    * NAME-ECHOING — the later candidate is the pseudo-parameter's own closer,
      and neither it nor its opener is an enumerated literal, so the prefix
      comes back clean and repair() ACCEPTS a truncation at that point.

    The echo outcome is cosmetic — it strips the trailing invoke terminator and
    leaves the earlier mis-close sitting in the value as prose. It is still
    LOSSLESS (invariant D5: the result is a prefix of the input, so no byte of
    the swallowed text is destroyed), and the result no longer trips detect(),
    so the second-run-zero invariant is unaffected.

    This sweep does NOT add a second policy layer to override it. repair() is
    the gate by design (decision 7), and re-deciding its verdict here would
    mean re-implementing the matching this module deliberately owns none of
    (INV-5). The behaviour belongs to ``shared.toolcall_markup`` (task 3688);
    this test pins what the sweep actually inherits so a future change there is
    visible here rather than silent.
    """
    record = make_escalation(
        'esc-7-7', 'resolved', _swallowed_echo('Detail.', 'detail', 'evidence', 'some text')
    )
    original_detail = record['detail']

    repaired, outcomes = sweep.repair_document(record)

    assert [o.action for o in outcomes] == [sweep.ACTION_REPAIRED]
    assert outcomes[0].recovered_names == (), 'nothing was restored — a truncation'
    assert original_detail.startswith(repaired['detail']), 'D5 still holds: a PREFIX'
    assert repaired['evidence'] == [], 'the list-typed field is still NOT displaced'
    assert detect(repaired['detail']) is None, 'and the result no longer trips detect()'


# ---------------------------------------------------------------------------
# step-7 — convergence, and the bound.
# ---------------------------------------------------------------------------


def test_repair_document_converges_and_is_idempotent_on_a_nested_leak():
    """Case (a): a recovered value that ITSELF still carries markup.

    The nested double-leak shape cancelled task 3654 classified. What actually
    happens is worth pinning precisely, because it is NOT "repaired twice":

    ``_parse_tail`` refuses outright when a recovered item contains a further
    mis-close (PRD boundary row B5 — the inner value's boundary would be a
    guess), so a STRICTLY nested leak is a refusal, not a two-round repair. The
    shape that DOES survive round one is the unterminated inner opener below:
    it carries no closing tag, so B5 does not fire, and the recovered value
    lands in its hole still tripping detect().

    Round two then REFUSES it (there is no qualifying mis-close to scan), so
    the residue is reported rather than silently left — and a second
    ``repair_document`` call applies zero repairs. That is the idempotence the
    second-run-zero invariant needs, and it holds here by measurement rather
    than by assumption.
    """
    intended = 'The intended detail.'
    inner = 'rc_text' + _lit('triage_note') + 'tn_text'
    record = make_escalation(
        'esc-8-1',
        'resolved',
        intended + _closer('detail') + '\n' + _lit('suggested_action') + inner
        + _closer('parameter') + '\n' + INVOKE_CLOSER + '\n',
    )

    repaired, outcomes = sweep.repair_document(record)

    assert repaired['detail'] == intended
    assert repaired['suggested_action'] == inner
    assert [o.action for o in outcomes].count(sweep.ACTION_REPAIRED) == 1
    assert sweep.ACTION_REFUSED in [o.action for o in outcomes], (
        'the residue must be REPORTED, never silently left'
    )

    # Idempotence: a second pass changes nothing and repairs nothing.
    before = json.dumps(repaired, indent=2)
    again, again_outcomes = sweep.repair_document(repaired)
    assert json.dumps(again, indent=2) == before
    assert [o.action for o in again_outcomes].count(sweep.ACTION_REPAIRED) == 0


def test_the_round_bound_is_a_named_module_constant():
    """Not a magic number inline — an operator has to be able to find it."""
    assert isinstance(sweep._MAX_REPAIR_ROUNDS, int)
    assert sweep._MAX_REPAIR_ROUNDS >= 2, 'a fixed point needs at least two passes'


def test_a_non_converging_document_reports_did_not_converge(monkeypatch):
    """Case (b): the bound truncates the LOOP, never a repair.

    Driven through a stubbed ``repair`` rather than a hand-built document, and
    deliberately so: with the real repairer no document can reach the bound —
    a recovered value can never contain a further mis-close (B5 refuses the
    parse), so one round always converges on every shape repair() accepts. That
    is exactly why the loop is INSURANCE rather than routine machinery, and why
    its bound has to be tested at the seam instead of through a fixture that
    cannot exist today.

    What must hold when the bound is hit: the document is still valid, every
    repair already applied is INTACT, and the failure is reported loudly with
    the path that was still changing — never silently truncated into a
    plausible-looking clean result.
    """
    def _never_converges(value, param, schema_params, supplied):
        # Always "repairs", never reaching a fixed point.
        return sweep.Repair(
            clean_value=value,
            recovered={'root_cause': 'restored by the stub'},
            pattern=INVOKE_CLOSER,
            misclose=INVOKE_CLOSER,
        )

    monkeypatch.setattr(sweep, 'repair', _never_converges)
    record = make_escalation('esc-8-2', 'resolved', 'detail ' + INVOKE_CLOSER)

    repaired, outcomes = sweep.repair_document(record)

    stalled = [o for o in outcomes if o.action == sweep.ACTION_DID_NOT_CONVERGE]
    assert len(stalled) == 1, 'exactly one loud did-not-converge outcome'
    assert stalled[0].json_path == 'detail', 'it names the path still changing'
    assert stalled[0].reason == sweep.REASON_ROUND_BOUND_EXCEEDED

    # The document survives, and the applied repair was not rolled back.
    assert json.loads(json.dumps(repaired)) == repaired, 'still valid JSON'
    assert repaired['root_cause'] == 'restored by the stub', 'repair intact'
    assert len([o for o in outcomes if o.action == sweep.ACTION_REPAIRED]) == (
        sweep._MAX_REPAIR_ROUNDS
    ), 'one repair per round, and the loop stopped AT the bound'


# ---------------------------------------------------------------------------
# step-9 — the escalation lane's terminal-status gate.
# ---------------------------------------------------------------------------


def _load(path, lane=None):
    """Run one path through the load/gate stage."""
    if lane is None:
        lane = sweep.LANE_PLANS if path.name == 'plan.json' else sweep.LANE_ESCALATIONS
    return sweep.load_target(sweep.Target(path=path, lane=lane))


@pytest.mark.parametrize('status', ['resolved', 'dismissed'])
def test_terminal_escalation_records_are_swept(tmp_path, status):
    """Both terminal statuses load. These are the 59 archived corrupted files."""
    path = write_escalation(
        tmp_path / f'esc-{status}.json',
        make_escalation('esc-1-1', status, _swallowed('D.', 'detail', 'root_cause', 'r')),
    )
    loaded = _load(path)
    assert isinstance(loaded, sweep.LoadedDocument)
    assert loaded.obj['status'] == status


@pytest.mark.parametrize('status', ['pending', 'in-progress', 'some-future-status'])
def test_non_terminal_and_unknown_statuses_are_skipped(tmp_path, status):
    """Anything not terminal is skipped and REPORTED, never silently clean.

    An unknown status skips in the fail-safe direction: this sweep's atomicity
    prevents a torn READ but not a lost UPDATE against the live queue writer,
    so a record whose lifecycle state it does not understand is left alone.

    Byte-identity under ``--apply`` is asserted end-to-end in the second-run
    test; what is pinned here is that the gate refuses before a write is ever
    considered.
    """
    path = write_escalation(
        tmp_path / f'esc-{status}.json',
        make_escalation('esc-1-1', status, _swallowed('D.', 'detail', 'root_cause', 'r')),
    )
    refusal = _load(path)
    assert isinstance(refusal, sweep.Refusal)
    assert refusal.reason == sweep.REASON_NON_TERMINAL


def test_a_non_record_json_document_is_skipped_even_when_flagged(tmp_path):
    """``b3-state.json`` shape: no ``id``/``status``, but flagged content.

    It lives inside data/escalations and matches the glob, so only a SHAPE
    check excludes it. Skipping it on "no detectable markup" would be luck,
    not a rule — hence the flagged string in the fixture.
    """
    path = write_escalation(
        tmp_path / 'b3-state.json',
        {'generated_at': 'now', 'note': _swallowed('B3.', 'detail', 'root_cause', 'r')},
    )
    refusal = _load(path)
    assert isinstance(refusal, sweep.Refusal)
    assert refusal.reason == sweep.REASON_NOT_AN_ESCALATION_RECORD


def test_an_unparseable_file_is_reported_not_raised(tmp_path):
    """A malformed file yields a reason, so the sweep can continue past it.

    Aborting the whole run on one bad file would leave the corpus half-swept
    with no record of where it stopped.
    """
    path = tmp_path / 'broken.json'
    path.write_text('{"id": "esc-1-1", "status": "resolved",', encoding='utf-8')
    refusal = _load(path)
    assert isinstance(refusal, sweep.Refusal)
    assert refusal.reason == sweep.REASON_UNPARSEABLE


def test_an_unreadable_file_is_reported_not_raised(tmp_path):
    """Undecodable bytes are a reported refusal too, not a traceback."""
    path = tmp_path / 'binary.json'
    path.write_bytes(b'\xff\xfe\x00 not utf-8 at all')
    refusal = _load(path)
    assert isinstance(refusal, sweep.Refusal)
    assert refusal.reason == sweep.REASON_UNPARSEABLE


def test_the_plans_lane_is_not_subject_to_the_terminal_status_gate(tmp_path):
    """A plan has no ``status`` field to read — its gate is lane liveness.

    Applying the escalation gate to plans would skip every plan in the corpus
    while reporting a clean run, which is the failure this test exists to stop.
    """
    path = write_plan(
        tmp_path / 'plan.json',
        make_plan('9006', [_truncated('R.', 'rationale')]),
    )
    loaded = _load(path, lane=sweep.LANE_PLANS)
    assert isinstance(loaded, sweep.LoadedDocument)
    assert 'status' not in loaded.obj


def test_loaded_documents_carry_the_raw_bytes_for_the_round_trip_check(tmp_path):
    """The raw source text travels with the parse.

    Re-reading the file later to compare formatting would race the live queue
    writer; the round-trip precondition has to check the SAME bytes the parse
    came from.
    """
    path = write_escalation(
        tmp_path / 'esc-2-2.json', make_escalation('esc-2-2', 'resolved', 'clean')
    )
    loaded = _load(path)
    assert isinstance(loaded, sweep.LoadedDocument)
    assert loaded.raw == path.read_text(encoding='utf-8')
    assert json.loads(loaded.raw) == loaded.obj


# ---------------------------------------------------------------------------
# step-11 — plans-lane symlink resolution, liveness, dedupe.
# ---------------------------------------------------------------------------


def _plan_target(root, lane_dir):
    return sweep.Target(
        path=root / '.worktrees-orphaned' / lane_dir / '.task' / 'plan.json',
        lane=sweep.LANE_PLANS,
    )


def test_a_symlinked_plan_resolves_to_the_meta_root_and_stays_a_link(sweep_root):
    """(a) The write lands on the FILE; the LINK is never the write target.

    ``os.replace`` onto the link path would replace it with a regular file and
    re-fork the lane and meta-root copies — the esc-5205-9 stale-plan
    divergence ``plan_tools._atomic_write_plan`` documents at line 715. The
    link must therefore still be a link after resolution, and forever after.
    """
    target = _plan_target(sweep_root, '9002-2026')
    resolved = sweep.resolve_write_target(target, sweep_root)

    assert isinstance(resolved, sweep.ResolvedTarget)
    assert resolved.write_path == (
        sweep_root / '.worktrees' / '.task-meta' / '9002' / 'plan.json'
    )
    assert target.path.is_symlink(), 'resolution must not disturb the link'


def test_a_plan_whose_meta_root_a_live_lane_shares_is_skipped(sweep_root):
    """(b) PRD D4's terminal-vs-live split, enforced MECHANICALLY.

    The meta-root is SHARED with the live lane. Measured at plan time:
    ``.worktrees/3415`` exists right now, so sweeping "the orphaned 3415 plan"
    would in fact rewrite a plan a RUNNING task is reading. PRD D4 assigns
    those to task 3692's lazy write-back, not to this sweep — and this skip is
    how that assignment is enforced instead of assumed.
    """
    (sweep_root / '.worktrees' / '9002').mkdir(parents=True)
    link = _plan_target(sweep_root, '9002-2026').path
    meta = sweep_root / '.worktrees' / '.task-meta' / '9002' / 'plan.json'
    before = meta.read_bytes()

    refusal = sweep.resolve_write_target(_plan_target(sweep_root, '9002-2026'), sweep_root)

    assert isinstance(refusal, sweep.Refusal)
    assert refusal.reason == sweep.REASON_LIVE_LANE_PRESENT
    assert meta.read_bytes() == before
    assert link.is_symlink()


def test_two_orphaned_lanes_sharing_a_realpath_are_swept_once(sweep_root):
    """(c) Dedupe by REALPATH, so a shared meta-root is never written twice.

    Writing the same file twice in one run is not merely wasteful: the second
    pass would re-read the first pass's output, and any asymmetry between the
    two would show up as churn an operator cannot explain.
    """
    meta = sweep_root / '.worktrees' / '.task-meta' / '9002' / 'plan.json'
    second = sweep_root / '.worktrees-orphaned' / '9002-dup' / '.task' / 'plan.json'
    second.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(meta), str(second))

    targets = sweep.discover_targets(sweep_root)
    assert len([t for t in targets if t.path in (second, _plan_target(sweep_root, '9002-2026').path)]) == 2

    deduped = sweep.dedupe_by_realpath(targets)
    resolved_paths = [Path(os.path.realpath(t.path)) for t in deduped]
    assert len(resolved_paths) == len(set(resolved_paths)), 'no realpath twice'
    assert sum(1 for p in resolved_paths if p == meta) == 1


def test_a_dangling_plan_symlink_is_reported_and_materialises_nothing(sweep_root):
    """(d) A dangling link is a reported refusal, never a created file.

    The hazard is specific: an unguarded ``os.replace`` onto a dangling link
    path CREATES a regular file there, inventing a plan for a lane whose
    meta-root was already reclaimed.
    """
    link = sweep_root / '.worktrees-orphaned' / '9007-2026' / '.task' / 'plan.json'
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(sweep_root / '.worktrees' / '.task-meta' / '9007' / 'plan.json'), str(link))

    refusal = sweep.resolve_write_target(_plan_target(sweep_root, '9007-2026'), sweep_root)

    assert isinstance(refusal, sweep.Refusal)
    assert refusal.reason == sweep.REASON_DANGLING_SYMLINK
    assert link.is_symlink() and not link.exists(), 'still dangling, nothing created'
    assert not (sweep_root / '.worktrees' / '.task-meta' / '9007').exists()


def test_a_plain_file_plan_is_repaired_in_place(sweep_root):
    """(e) Not every orphaned plan is a link; a real file writes to itself."""
    target = _plan_target(sweep_root, '9001-2026')
    resolved = sweep.resolve_write_target(target, sweep_root)

    assert isinstance(resolved, sweep.ResolvedTarget)
    assert resolved.write_path == target.path
    assert not target.path.is_symlink()


def test_liveness_is_checked_only_for_meta_root_resolutions(sweep_root):
    """A plain-file plan has no lane id to check, so liveness cannot apply.

    Guards against a liveness check that derived an "id" from whatever
    directory happened to be the parent — for a plain file that is ``.task``,
    and ``<root>/.worktrees/.task`` is not a lane.
    """
    (sweep_root / '.worktrees' / '.task').mkdir(parents=True, exist_ok=True)
    resolved = sweep.resolve_write_target(_plan_target(sweep_root, '9001-2026'), sweep_root)
    assert isinstance(resolved, sweep.ResolvedTarget)


# ---------------------------------------------------------------------------
# step-13 — round-trip-or-refuse.
# ---------------------------------------------------------------------------


def test_serialize_like_reproduces_both_real_writers_byte_for_byte(sweep_root):
    """The UNMODIFIED parse of each fixture must re-serialize to its own bytes.

    This is the mechanical guarantee behind "each byte-diff is confined to the
    corrupted strings": if dumps(before) == raw, then dumps(after) differs from
    raw exactly where the parsed object differs. The property is proven PER
    FILE at run time rather than asserted once in a test.
    """
    record = sweep_root / 'data' / 'escalations' / 'archive' / '2026-08-08' / 'esc-2-1.json'
    plan = sweep_root / '.worktrees-orphaned' / '9001-2026' / '.task' / 'plan.json'

    for path in (record, plan):
        raw = path.read_text(encoding='utf-8')
        assert sweep.serialize_like(raw, json.loads(raw)) == raw, path.name
        assert sweep.round_trips(raw, json.loads(raw)) is True


def test_the_trailing_newline_convention_is_taken_from_the_source(tmp_path):
    """One writer appends a newline and the other does not; both are honoured."""
    obj = {'a': 1}
    assert sweep.serialize_like(json.dumps(obj, indent=2), obj).endswith('}')
    assert sweep.serialize_like(json.dumps(obj, indent=2) + '\n', obj).endswith('}\n')


@pytest.mark.parametrize(
    'raw',
    [
        json.dumps({'id': 'e', 'status': 'resolved'}, indent=4),
        json.dumps({'id': 'e', 'status': 'resolved'}),
        json.dumps({'id': 'e', 'status': 'resolved'}, indent=2) + '\n\n',
    ],
    ids=['indent-4', 'compact', 'double-trailing-newline'],
)
def test_unreproducible_formatting_is_refused(raw):
    """A hand-edited or unusually-formatted file fail-safes for free.

    Rewriting it would put changes in the diff that the corrupted strings did
    not cause — the exact reason plan_tools._atomic_write_plan was NOT reused
    here (it stamps _schema_version and re-indents).
    """
    assert sweep.round_trips(raw, json.loads(raw)) is False


def test_non_ascii_round_trips_byte_exactly(tmp_path):
    """ensure_ascii stays at its default, so escapes survive unchanged.

    The live corpus is full of em-dashes and smart quotes written as \\uXXXX
    escapes. Flipping ensure_ascii=False would rewrite every one of them and
    turn a two-field repair into a whole-file diff.
    """
    obj = {'id': 'e', 'status': 'resolved', 'detail': 'em—dash and “quotes” and é'}
    raw = json.dumps(obj, indent=2)
    assert '\\u2014' in raw, 'the fixture must actually exercise escaping'
    assert sweep.serialize_like(raw, json.loads(raw)) == raw
    assert sweep.round_trips(raw, json.loads(raw)) is True


def test_key_order_is_reproduced_from_the_file_not_re_sorted():
    """A file written with sorted keys still round-trips — and that is CORRECT.

    Worth pinning because it is non-obvious: ``json.loads`` preserves the
    order the keys appear in the FILE, so re-dumping without ``sort_keys``
    reproduces whatever order that file had. The round-trip check is a
    statement about BYTES, not about which writer produced them, so a
    sorted-key file is reproducible and safe to rewrite rather than a format
    this sweep must refuse.
    """
    raw = json.dumps({'z': 1, 'a': 2}, indent=2, sort_keys=True)
    assert raw.index('"a"') < raw.index('"z"')
    assert sweep.round_trips(raw, json.loads(raw)) is True


# ---------------------------------------------------------------------------
# step-15 — atomicity, boundary row B11.
# ---------------------------------------------------------------------------


def _resolved(root, relpath, lane=sweep.LANE_ESCALATIONS):
    target = sweep.Target(path=root / relpath, lane=lane)
    resolved = sweep.resolve_write_target(target, root)
    assert isinstance(resolved, sweep.ResolvedTarget)
    return resolved


def _tmp_leftovers(directory):
    return sorted(p.name for p in directory.iterdir() if p.name.startswith('.sweep.'))


def test_the_temp_file_is_created_in_the_resolved_targets_own_directory(sweep_root, monkeypatch):
    """os.replace is only atomic WITHIN a filesystem, so /tmp will not do.

    Captured at the mkstemp seam rather than inferred from the result, because
    a cross-device temp would still produce a correct-looking file — it would
    just stop being an atomic swap, which is the entire contract.
    """
    seen = {}
    real_mkstemp = sweep.tempfile.mkstemp

    def _spy(*args, **kwargs):
        seen['dir'] = kwargs.get('dir')
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(sweep.tempfile, 'mkstemp', _spy)
    resolved = _resolved(sweep_root, 'data/escalations/archive/2026-08-08/esc-3-1.json')

    assert sweep.write_repaired(resolved, '{}', {'a': 1}) is None
    assert Path(seen['dir']) == resolved.write_path.parent


def test_a_successful_write_preserves_the_targets_permission_bits(sweep_root):
    """A 0600 mkstemp artifact must not become the record's new mode."""
    resolved = _resolved(sweep_root, 'data/escalations/archive/2026-08-08/esc-3-1.json')
    os.chmod(resolved.write_path, 0o640)

    assert sweep.write_repaired(resolved, '{}', {'a': 1}) is None
    assert oct(resolved.write_path.stat().st_mode & 0o777) == oct(0o640)


def test_a_failed_replace_leaves_the_target_byte_identical(sweep_root, monkeypatch):
    """The interrupt-between-write-and-replace case. Nothing half-written."""
    resolved = _resolved(sweep_root, 'data/escalations/archive/2026-08-08/esc-2-1.json')
    before = resolved.write_path.read_bytes()

    def _boom(src, dst):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(sweep.os, 'replace', _boom)
    failure = sweep.write_repaired(resolved, before.decode(), {'a': 1})

    assert isinstance(failure, sweep.WriteFailure)
    assert str(resolved.write_path) in str(failure.path)
    assert 'No space left' in failure.error, 'the underlying error is carried'
    assert resolved.write_path.read_bytes() == before
    assert json.loads(resolved.write_path.read_text(encoding='utf-8'))
    assert _tmp_leftovers(resolved.write_path.parent) == [], 'no .sweep. debris'


def test_a_failed_verify_leaves_the_target_byte_identical(sweep_root, monkeypatch):
    """Verify-before-replace: a payload that will not re-parse never lands.

    Forced through serialize_like so the whole write-then-verify path really
    runs, rather than stubbing the verifier and proving only that a stub was
    called.
    """
    resolved = _resolved(sweep_root, 'data/escalations/archive/2026-08-08/esc-2-1.json')
    before = resolved.write_path.read_bytes()

    monkeypatch.setattr(sweep, 'serialize_like', lambda raw, obj: '{"truncated": ')
    failure = sweep.write_repaired(resolved, before.decode(), {'a': 1})

    assert isinstance(failure, sweep.WriteFailure)
    assert failure.reason == sweep.REASON_VERIFY_FAILED
    assert resolved.write_path.read_bytes() == before
    assert json.loads(resolved.write_path.read_text(encoding='utf-8'))
    assert _tmp_leftovers(resolved.write_path.parent) == []


def test_the_swap_lands_on_the_meta_root_file_never_on_the_link(sweep_root):
    """The symlink survives as a symlink; its TARGET receives the bytes."""
    link = sweep_root / '.worktrees-orphaned' / '9002-2026' / '.task' / 'plan.json'
    resolved = _resolved(
        sweep_root, '.worktrees-orphaned/9002-2026/.task/plan.json', sweep.LANE_PLANS
    )
    raw = link.read_text(encoding='utf-8')

    assert sweep.write_repaired(resolved, raw, {'swapped': True}) is None
    assert link.is_symlink(), 'the LINK must never be replaced by a regular file'
    assert json.loads(link.read_text(encoding='utf-8')) == {'swapped': True}
    assert json.loads(resolved.write_path.read_text(encoding='utf-8')) == {'swapped': True}


# ---------------------------------------------------------------------------
# step-17 — the CLI. Dry run is the DEFAULT.
# ---------------------------------------------------------------------------


def _run_json(capsys, *argv):
    code = sweep.main(list(argv) + ['--json'])
    return code, json.loads(capsys.readouterr().out)


def test_dry_run_is_the_default_and_writes_nothing(sweep_root, capsys):
    """No --apply, no writes. Every byte AND every link-ness preserved.

    The snapshot compares the whole tree rather than the interesting files,
    because "writes nothing" is only meaningful as a statement about
    everything the run could have touched.
    """
    before = read_bytes_map(sweep_root)
    code, summary = _run_json(capsys, '--root', str(sweep_root))
    assert read_bytes_map(sweep_root) == before
    assert code == sweep.EXIT_REPAIRABLE_REMAINS
    assert summary['repaired'] > 0, 'the fixture must have repairable work'


def test_dry_run_emits_a_unified_diff_per_repairable_file(sweep_root, capsys):
    """An operator reviews the diff before ever passing --apply."""
    sweep.main(['--root', str(sweep_root)])
    out = capsys.readouterr().out
    assert '@@' in out, 'a unified diff hunk header'
    assert 'esc-2-1.json' in out
    assert '---' in out and '+++' in out


def test_the_summary_carries_every_named_counter(sweep_root, capsys):
    """The report's shape is part of the contract, not incidental output."""
    _code, summary = _run_json(capsys, '--root', str(sweep_root))
    for key in (
        'files_scanned', 'strings_detected', 'repaired',
        'leak_unrepaired', 'quoted_only', 'skipped', 'failed',
    ):
        assert key in summary, key
    assert isinstance(summary['skipped'], dict), 'skips are broken out BY REASON'
    assert summary['skipped'].get(sweep.REASON_NON_TERMINAL) == 1, 'the pending record'
    assert summary['skipped'].get(sweep.REASON_NOT_AN_ESCALATION_RECORD) == 1


def test_a_tree_with_nothing_repairable_exits_clean(tmp_path, capsys):
    """Exit 0 is the green signal the second run has to be able to produce."""
    write_escalation(
        tmp_path / 'data' / 'escalations' / 'esc-9-1.json',
        make_escalation('esc-9-1', 'resolved', 'perfectly clean'),
    )
    code, summary = _run_json(capsys, '--root', str(tmp_path))
    assert code == sweep.EXIT_CLEAN
    assert summary['repaired'] == 0


def test_residue_is_split_into_leak_shaped_and_merely_quoting(tmp_path, capsys):
    """detect() flags 144 strings live but only 63 carry the leak signature.

    Reporting all refusals undifferentiated would bury the ones that are real
    corruption under prose that merely QUOTES a tag — 39 of which are
    evidence[].observation fields in records ABOUT markup incidents.
    """
    escalations = tmp_path / 'data' / 'escalations'
    write_escalation(
        escalations / 'esc-leak.json',
        make_escalation(
            'esc-leak', 'resolved',
            _swallowed('D.', 'detail', 'evidence', 'x'),
        ),
    )
    write_escalation(
        escalations / 'esc-quote.json',
        make_escalation(
            'esc-quote', 'resolved',
            'The agent emitted ' + _closer('description') + ' and lost an argument.',
        ),
    )
    _code, summary = _run_json(capsys, '--root', str(tmp_path))
    assert summary['leak_unrepaired'] == 1
    assert summary['quoted_only'] == 1


def test_residue_never_sets_a_non_zero_exit(tmp_path, capsys):
    """37 refusals legitimately remain after --apply, forever.

    Keying the exit code on residue would make the second run red permanently
    and destroy the very signal this task is measured by.
    """
    write_escalation(
        tmp_path / 'data' / 'escalations' / 'esc-r.json',
        make_escalation('esc-r', 'resolved', _swallowed('D.', 'detail', 'evidence', 'x')),
    )
    code, summary = _run_json(capsys, '--root', str(tmp_path), '--apply')
    assert summary['leak_unrepaired'] == 1
    assert code == sweep.EXIT_CLEAN, 'unrepairable residue is reported, not failed'


def test_the_lane_flag_narrows_the_sweep(sweep_root, capsys):
    """--lane lets an operator run one corpus at a time."""
    _c, plans_only = _run_json(capsys, '--root', str(sweep_root), '--lane', 'plans')
    _c2, esc_only = _run_json(capsys, '--root', str(sweep_root), '--lane', 'escalations')
    assert plans_only['files_scanned'] == 2
    assert esc_only['files_scanned'] == 4


def test_the_script_source_spells_no_raw_envelope_literal():
    """Mirrors this test module's own self-check, for the script under test.

    Every sentinel it needs is imported from shared.toolcall_markup, so a raw
    chr(60)+'/' in its source would mean someone re-spelled one — which is both
    a third enumeration site (INV-5) and the authoring hazard itself.
    """
    source = Path(sweep.__file__).read_text(encoding='utf-8')
    assert (chr(60) + '/') not in source


# ---------------------------------------------------------------------------
# step-19 — --apply end-to-end, and the BINDING second-run-zero invariant.
# ---------------------------------------------------------------------------


def test_apply_repairs_both_lanes_and_rewrites_only_what_it_repaired(sweep_root, capsys):
    """The whole pipeline, on both corpora at once."""
    record_path = (
        sweep_root / 'data' / 'escalations' / 'archive' / '2026-08-08' / 'esc-2-1.json'
    )
    clean_path = record_path.parent / 'esc-3-1.json'
    plan_path = sweep_root / '.worktrees-orphaned' / '9001-2026' / '.task' / 'plan.json'
    link_path = sweep_root / '.worktrees-orphaned' / '9002-2026' / '.task' / 'plan.json'

    original_detail = json.loads(record_path.read_text(encoding='utf-8'))['detail']
    original_plan = json.loads(plan_path.read_text(encoding='utf-8'))
    clean_before = clean_path.read_bytes()
    pending_before = (sweep_root / 'data' / 'escalations' / 'esc-1-1.json').read_bytes()

    code, summary = _run_json(capsys, '--root', str(sweep_root), '--apply')
    assert code == sweep.EXIT_CLEAN
    assert summary['failed'] == 0

    # --- the escalation record ---------------------------------------------
    record = json.loads(record_path.read_text(encoding='utf-8'))
    assert original_detail.startswith(record['detail']), 'detail is the PREFIX'
    assert record['suggested_action'] in original_detail, 'recovered VERBATIM'
    assert record['suggested_action'] == 'Re-run the merge worker after the lock clears.'
    for key, value in json.loads(clean_before.decode()).items():
        if key in ('detail', 'suggested_action'):
            continue
        assert record[key] == value or key in ('id', 'task_id', 'status')
    assert type(record['evidence']) is list, 'type preserved'
    assert not record_path.read_bytes().endswith(b'\n'), 'no newline added'

    # --- the plans lane -----------------------------------------------------
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert plan['_schema_version'] == 1, 'no schema stamp, no re-indent'
    assert plan['steps'] == original_plan['steps']
    assert plan['design_decisions'][0]['rationale'] == 'The first rationale, mis-closed.'
    assert plan['design_decisions'][1] == original_plan['design_decisions'][1]
    assert plan_path.read_bytes().endswith(b'\n'), 'trailing newline preserved'

    assert link_path.is_symlink(), 'the link is still a link'
    assert detect(json.loads(link_path.read_text(encoding='utf-8'))
                  ['design_decisions'][0]['rationale']) is None

    # --- nothing else was touched ------------------------------------------
    assert clean_path.read_bytes() == clean_before, 'a clean file is NOT rewritten'
    assert (sweep_root / 'data' / 'escalations' / 'esc-1-1.json').read_bytes() == (
        pending_before
    ), 'the non-terminal record is untouched even under --apply'


def test_the_second_run_reports_zero_and_changes_nothing(sweep_root, capsys):
    """THE binding invariant. A second --apply is a total no-op.

    This is the signal the whole task is measured by, so it is asserted three
    ways at once: zero repairs, a clean exit, and byte-identity of every file
    in the tree against the post-first-run state.
    """
    first_code, _first = _run_json(capsys, '--root', str(sweep_root), '--apply')
    assert first_code == sweep.EXIT_CLEAN
    after_first = read_bytes_map(sweep_root)

    second_code, second = _run_json(capsys, '--root', str(sweep_root), '--apply')

    assert second['repaired'] == 0
    assert second_code == sweep.EXIT_CLEAN
    assert read_bytes_map(sweep_root) == after_first


def test_a_write_failure_exits_2_names_the_path_and_does_not_abort(sweep_root, capsys, monkeypatch):
    """One failing target must not cost the others their repair."""
    record_path = (
        sweep_root / 'data' / 'escalations' / 'archive' / '2026-08-08' / 'esc-2-1.json'
    )
    plan_path = sweep_root / '.worktrees-orphaned' / '9001-2026' / '.task' / 'plan.json'
    record_before = record_path.read_bytes()

    real_replace = os.replace

    def _selective(src, dst):
        if Path(dst) == record_path:
            raise OSError(13, 'Permission denied')
        return real_replace(src, dst)

    monkeypatch.setattr(sweep.os, 'replace', _selective)
    code, summary = _run_json(capsys, '--root', str(sweep_root), '--apply')

    assert code == sweep.EXIT_WRITE_FAILED
    assert summary['failed'] == 1
    assert record_path.read_bytes() == record_before, 'the failed target is intact'
    assert json.loads(record_path.read_text(encoding='utf-8')), 'still valid JSON'
    # ...and the sweep carried on to the other lane.
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert detect(plan['design_decisions'][0]['rationale']) is None, 'others repaired'
