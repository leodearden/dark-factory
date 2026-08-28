"""Tests for the write_triage flip gate's item 1 (task 4810).

Two units are under test, and they are deliberately separate files:

  * ``scripts/check_write_triage_attach_target.py`` — the PROBE. It imports a
    judge module out of a bare source tree and decides, by EXECUTING it,
    whether the judge path binds a verdict to a determinate candidate.
  * ``scripts/check_write_triage_flip_preconditions.sh`` — the GATE. Item 1
    delegates to the probe through the ``CHECK_WRITE_TRIAGE_ATTACH_TARGET_PY``
    env seam (modelled on ``scripts/check_sandbox_soak.sh``'s
    ``CHECK_SANDBOX_SOAK_PY``).

WHY THE FIXTURES ARE STANDALONE. The probe is pointed at a ``--src-root`` and
imports ``fused_memory.server.write_triage_judge`` from it. A fixture module
that imported pydantic — or anything else the real judge module pulls in —
would make these tests depend on the fused-memory virtualenv. Every fixture
here is stdlib-only, so the suite runs under whatever interpreter collected it.

WHY THE SLATE IS BUILT BY THE FIXTURE'S OWN ``select_judge_candidates``. The
fixture mirrors main's selection EXACTLY, including the winner-rescue APPEND
(``selected = [*selected[: max(n - 1, 0)], winner]``). On a hoisted-parent
slate the rescued evidence child therefore lands LAST — measured on main
f474347580: ``select_judge_candidates(results, 3, canonical_id='parent-1')``
returned ``['m0', 'm1', 'child-1']``. An implementation that marks
``candidates[0]`` as the attach target is WRONG on exactly that slate, so the
probe must be driven on it or the gate would bless the bug it exists to catch.

Assertions are on EXIT CODES and on the probe's own structured stdout
vocabulary — never on the wording of a fixture's rendered prompt. Pinning
prompt prose is the meta-test class this repo deletes (task 3128 steps 23-25).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GATE_SCRIPT = _REPO_ROOT / 'scripts' / 'check_write_triage_flip_preconditions.sh'
_PROBE = _REPO_ROOT / 'scripts' / 'check_write_triage_attach_target.py'

# The probe's stable stdout vocabulary. Kept as named constants so the tests
# and the probe cannot drift apart silently, and so it is obvious at a glance
# that no PROMPT wording is being pinned — only the probe's own verdict lines.
_INDETERMINATE = 'INDETERMINATE'
_UNVERIFIABLE = 'UNVERIFIABLE'
_RESCUE_PATH = 'select_judge_candidates'
_BYTE_IDENTICAL = 'byte-identical'
_MARKS_FIRST = 'distinguishes slate[0]'
_SWAP_HELD = 'rendering depends on which candidate is the attach target'


# ---------------------------------------------------------------------------
# Fixture judge modules
# ---------------------------------------------------------------------------

_JUDGE_PREAMBLE = r'''"""Standalone stand-in for fused_memory.server.write_triage_judge.

Dependency-free by construction: the probe imports this out of a bare
directory tree, so it may import nothing but the stdlib.

select_judge_candidates mirrors main's EXACTLY, including the winner-rescue
APPEND that puts a rescued hoisted-parent's evidence child LAST.
"""
from __future__ import annotations

import json

PARENT_ID_KEY = 'parent_id'
VERDICT_KEY = 'verdict'
JUDGE_VERDICTS = {
    'distinct': 'stored',
    'restates': 'restated',
    'amends': 'amended',
    'contests': 'contested',
}

_DEFAULT_JUDGE_CANDIDATE_COUNT = 5


def _cosine_of(result):
    value = (result.metadata or {}).get('store_score')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def select_judge_candidates(results, n, *, canonical_id):
    scored = []
    for result in results or ():
        cosine = _cosine_of(result)
        if cosine is not None:
            scored.append((cosine, result))
    if not scored:
        return []
    scored.sort(key=lambda pair: pair[0], reverse=True)
    ordered = [result for _, result in scored]
    if n <= 0:
        n = _DEFAULT_JUDGE_CANDIDATE_COUNT
    selected = ordered[:n]
    if canonical_id is not None and all(r.id != canonical_id for r in selected):
        winner = next(
            (r for r in ordered if r.id == canonical_id), None,
        ) or next(
            (
                r
                for r in ordered
                if (r.metadata or {}).get(PARENT_ID_KEY) == canonical_id
            ),
            None,
        )
        if winner is not None and winner not in selected:
            selected = [*selected[: max(n - 1, 0)], winner]
    return selected


def _render_candidate(candidate):
    return ['- id: ' + str(candidate.id), '  text: ' + str(candidate.content)]


def _parse_bare_str(raw):
    """main's shape: a bare verdict word, naming no candidate."""
    payload = json.loads(raw)
    word = payload.get(VERDICT_KEY)
    if not isinstance(word, str):
        raise ValueError('no string verdict key')
    verdict = JUDGE_VERDICTS.get(word.strip().lower())
    if verdict is None:
        raise ValueError('not a judge verdict word')
    return verdict
'''


_VARIANT_TAILS = {
    # main's shape exactly: an undifferentiated candidate list and a bare-str
    # verdict. Neither remedy present -> the attach target is indeterminate.
    'flat': r'''

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # Importable, but blows up when the probe renders. An unverifiable
    # invariant is not a satisfied one: this must FAIL, never pass.
    'raises': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    raise RuntimeError('fixture: build_judge_prompt blows up')


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # Takes an attach-target argument but labels candidates[0] regardless. On a
    # hoisted-parent slate that is the WRONG candidate — the rescued evidence
    # child is LAST — so this must FAIL. Its rendering is independent of the
    # argument, and slate[0]'s id is mentioned one extra time.
    'positional_target': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '']
    lines.append('ATTACH TARGET: ' + str(candidates[0].id))
    lines.append('')
    lines.append('EXISTING CANDIDATES:')
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # The PROSE-ONLY fix: the signature grew an attach-target argument and the
    # prompt grew a sentence, but the rendering is byte-identical whichever
    # candidate is named. A source-text grep would pass this; the swap test
    # cannot.
    'ignores_target': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    lines.append('')
    lines.append('One of the candidates above is the one this write will attach to.')
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # A real option (b): whichever candidate matches the passed target id is
    # the one lifted out, wherever it sits in the slate.
    'by_id': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    target = [c for c in candidates if c.id == attach_target_id]
    others = [c for c in candidates if c.id != attach_target_id]
    lines = ['NEW ENTRY:', str(content), '', 'THE CANDIDATE THIS WRITE WILL ATTACH TO:']
    for candidate in target:
        lines.extend(_render_candidate(candidate))
    lines.append('')
    lines.append('OTHER CANDIDATES (CONTEXT ONLY):')
    for candidate in others:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # Byte-different heading text, identical id-level behaviour. Proves the
    # gate pins no prompt wording: reword freely, the check still passes.
    'by_id_reworded': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    target = [c for c in candidates if c.id == attach_target_id]
    others = [c for c in candidates if c.id != attach_target_id]
    lines = ['submitted:', str(content), '', '>>> the record this attaches to <<<']
    for candidate in target:
        lines.extend(_render_candidate(candidate))
    lines.append('')
    lines.append('~~~ background only, do not attach ~~~')
    for candidate in others:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
}


def _write_fake_judge(src_root: Path, *, variant: str) -> Path:
    """Lay down a standalone judge module at *src_root* and return *src_root*.

    *src_root* is what ``--src-root`` takes: the directory that CONTAINS the
    ``fused_memory`` package (i.e. the analogue of ``fused-memory/src``).

    ``variant='missing'`` writes no module at all — the fail-closed case where
    the ref carries nothing importable.
    """
    src_root.mkdir(parents=True, exist_ok=True)
    if variant == 'missing':
        return src_root
    server = src_root / 'fused_memory' / 'server'
    server.mkdir(parents=True, exist_ok=True)
    (src_root / 'fused_memory' / '__init__.py').write_text('')
    (server / '__init__.py').write_text('')
    (server / 'write_triage_judge.py').write_text(
        _JUDGE_PREAMBLE + _VARIANT_TAILS[variant],
    )
    return src_root


def _run_probe(src_root: Path, *, env: dict[str, str] | None = None):
    """Drive the probe directly under this interpreter."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(_PROBE), '--src-root', str(src_root)],
        capture_output=True,
        text=True,
        timeout=120,
        env=full_env,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAttachTargetProbe:
    """The probe decides the attach-target invariant by EXECUTING the judge
    module it is pointed at, never by inspecting its source text."""

    def test_gate_script_exists_and_is_executable(self):
        # DeterministicRunner's _default_run_script executes the predicate
        # directly, not via `bash`, so the +x bit and the shebang are
        # load-bearing.
        assert _GATE_SCRIPT.exists(), f'gate script missing at {_GATE_SCRIPT}'
        assert os.access(_GATE_SCRIPT, os.X_OK), f'gate script not executable: {_GATE_SCRIPT}'

    def test_probe_exists_and_is_executable(self):
        assert _PROBE.exists(), f'probe missing at {_PROBE}'
        assert os.access(_PROBE, os.X_OK), f'probe not executable: {_PROBE}'

    def test_flat_judge_fails_with_indeterminate_target(self, tmp_path):
        """main's shape: no attach-target parameter and a bare-str verdict."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='flat')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe passed on main\'s shape:\n{proc.stdout}\n{proc.stderr}'
        assert _INDETERMINATE in proc.stdout, proc.stdout

    def test_missing_module_fails_closed(self, tmp_path):
        """Nothing importable at --src-root is UNVERIFIABLE, so it FAILS."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='missing')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe passed on an empty tree:\n{proc.stdout}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout

    def test_raising_module_fails_closed(self, tmp_path):
        """A judge module that raises when rendered is UNVERIFIABLE, not OK."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='raises')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe passed on a raising module:\n{proc.stdout}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout


class TestSwapAndRescuePath:
    """The two properties that stop item 1 being satisfiable by prose, or by
    the measured ``candidates[0]`` bug.

    SWAP TEST — render the SAME slate twice against two different attach
    targets and require the two renderings to DIFFER. A sentence added to the
    prompt renders identically whichever candidate is the target, so a
    prose-only change cannot pass by construction. Nothing here matches
    heading text, so any rewording survives.

    RESCUE PATH — the fixture's ``select_judge_candidates`` appends a rescued
    hoisted-parent winner LAST (mirroring main). An implementation that labels
    ``candidates[0]`` therefore names the WRONG candidate on this slate, which
    is the same harm item 1 exists to prevent, merely relocated.
    """

    def test_marking_candidates_zero_fails_and_names_the_rescue_path(self, tmp_path):
        src_root = _write_fake_judge(tmp_path / 'src', variant='positional_target')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe blessed candidates[0]:\n{proc.stdout}'
        assert _MARKS_FIRST in proc.stdout, proc.stdout
        # The operator has to be told WHY slate[0] is not the attach target.
        assert _RESCUE_PATH in proc.stdout, proc.stdout

    def test_prose_only_change_fails_the_swap_test(self, tmp_path):
        src_root = _write_fake_judge(tmp_path / 'src', variant='ignores_target')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe passed a prose-only change:\n{proc.stdout}'
        assert _BYTE_IDENTICAL in proc.stdout, proc.stdout

    def test_marking_by_id_passes(self, tmp_path):
        src_root = _write_fake_judge(tmp_path / 'src', variant='by_id')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe failed a real fix:\n{proc.stdout}\n{proc.stderr}'
        assert _SWAP_HELD in proc.stdout, proc.stdout

    def test_rewording_the_headings_still_passes(self, tmp_path):
        """No prompt wording is pinned — only id-level behaviour."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='by_id_reworded')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe pinned prompt wording:\n{proc.stdout}\n{proc.stderr}'
        assert _SWAP_HELD in proc.stdout, proc.stdout
