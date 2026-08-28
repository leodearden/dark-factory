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
_OPTION_A_HELD = 'the verdict itself names its candidate'


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
    # OPTION (a): the prompt marks nothing — main's two-argument
    # build_judge_prompt and its flat list — but the VERDICT carries the judged
    # candidate, so the model's answer names its own candidate and slate
    # position becomes irrelevant. The shape task 4762's own design decision
    # says option (a) produces: an (outcome, candidate_id) pair.
    'option_a': r'''

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw, candidate_ids=None):
    payload = json.loads(raw)
    word = payload.get(VERDICT_KEY)
    verdict = JUDGE_VERDICTS.get(str(word).strip().lower())
    if verdict is None:
        raise ValueError('not a judge verdict word')
    return (verdict, payload.get('candidate_id'))
''',
    # A STRICTER option (a): the parser refuses a verdict that carries no
    # candidate id at all, and refuses one naming a candidate outside the
    # slate. It therefore RAISES on a bare {"verdict": ...} payload — which
    # must be read as inconclusive for the branch, not as a failure.
    'option_a_rejects_dangling': r'''

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw, candidate_ids=None):
    payload = json.loads(raw)
    word = payload.get(VERDICT_KEY)
    verdict = JUDGE_VERDICTS.get(str(word).strip().lower())
    if verdict is None:
        raise ValueError('not a judge verdict word')
    candidate_id = payload.get('candidate_id')
    if not isinstance(candidate_id, str):
        raise ValueError('verdict carries no candidate_id')
    if candidate_ids is not None and candidate_id not in candidate_ids:
        raise ValueError('verdict names a candidate outside the slate')
    return (verdict, candidate_id)
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


class TestOptionAAcceptance:
    """The gate is MECHANISM-AGNOSTIC: it asserts the invariant, not which fix
    landed.

    Task 4810's root defect is that item 1 hard-required option (a) while its
    only dependency (4762) implements option (b). Pinning option (b) instead
    would reproduce that defect mirrored — task 4798 item 7 still carries
    option (a) as the better long-term design, and an option-(b)-shaped gate
    would fail it and re-block 3169 all over again. So EITHER remedy closes
    the gate, and a later option-(a) landing needs no further gate edit.
    """

    def test_verdict_carrying_its_candidate_passes_with_an_unmarked_prompt(self, tmp_path):
        """Position is irrelevant once the answer names its own candidate."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='option_a')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe failed option (a):\n{proc.stdout}\n{proc.stderr}'
        assert _OPTION_A_HELD in proc.stdout, proc.stdout

    def test_stricter_parser_demanding_a_candidate_id_also_passes(self, tmp_path):
        """A parser that REJECTS a verdict carrying no candidate id — and one
        naming a candidate outside the slate — is a stronger option (a), not a
        failure. Raising on the probe's plainest payload must not be misread."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='option_a_rejects_dangling')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe failed a strict option (a):\n{proc.stdout}\n{proc.stderr}'
        assert _OPTION_A_HELD in proc.stdout, proc.stdout

    def test_todays_main_shape_still_fails(self, tmp_path):
        """Regression guard: the option-(a) branch must not accidentally pass
        a bare-str verdict paired with an unmarked prompt, i.e. main today."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='flat')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'option (a) branch passed main:\n{proc.stdout}'
        assert _INDETERMINATE in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# The gate script, end to end against hermetic fixture repos
# ---------------------------------------------------------------------------

# Committed identity so a fixture commit does not depend on a global git
# identity being configured in the test environment (the idiom from
# fused-memory/tests/test_predicate_contradiction.py).
_GIT_ENV_ARGS = [
    '-c',
    'user.email=write-triage-gate-test@example.com',
    '-c',
    'user.name=write-triage-gate-test',
    '-c',
    'commit.gpgsign=false',
]

#: The fixture repo's branch, pinned so the test does not depend on whatever
#: `init.defaultBranch` the host git is configured with.
_FIXTURE_REF = 'fixture-main'

# Item 2 greps for the frozenset being iterated directly; item 4 for the
# self-overwriting markdown sibling. These two fixtures straddle both.
_EVAL_FAILING = '''\
"""Fixture stand-in for fused-memory/scripts/eval_write_triage_judge.py."""
CONFUSION_COLUMNS = list(TRIAGE_OUTCOMES)


def publish(report_path):
    return report_path.with_suffix('.md')
'''

_EVAL_FIXED = '''\
"""Fixture stand-in with both cycle-2 findings fixed."""
EVAL_OUTCOMES = tuple(sorted(TRIAGE_OUTCOMES))
CONFUSION_COLUMNS = EVAL_OUTCOMES


def publish(report_path):
    sibling = report_path.parent / (report_path.stem + '.md')
    assert sibling != report_path
    return sibling
'''

_CONFIG_YAML = 'write_triage:\n  enabled: false\n  judge_enabled: true\n'


def _make_gate_repo(tmp_path: Path, *, judge: str = 'flat', eval_src: str = 'failing') -> Path:
    """A throwaway git repo the gate script can be run from.

    The script derives ``REPO`` from ``BASH_SOURCE/..`` with no env override,
    so BOTH scripts are copied into ``<repo>/scripts/`` — that is what makes
    the fixture repo, rather than the real checkout, the thing item 1's
    ``git archive`` reads.
    """
    repo = tmp_path / 'gate-repo'
    (repo / 'scripts').mkdir(parents=True)
    for script in (_GATE_SCRIPT, _PROBE):
        dest = repo / 'scripts' / script.name
        dest.write_bytes(script.read_bytes())
        dest.chmod(0o755)

    _write_fake_judge(repo / 'fused-memory' / 'src', variant=judge)
    (repo / 'fused-memory' / 'scripts').mkdir(parents=True)
    (repo / 'fused-memory' / 'scripts' / 'eval_write_triage_judge.py').write_text(
        _EVAL_FIXED if eval_src == 'fixed' else _EVAL_FAILING,
    )
    (repo / 'fused-memory' / 'config').mkdir(parents=True)
    (repo / 'fused-memory' / 'config' / 'config.yaml').write_text(_CONFIG_YAML)

    subprocess.run(
        ['git', 'init', '-q', '-b', _FIXTURE_REF],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(['git', 'add', '-A', '-f'], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ['git', *_GIT_ENV_ARGS, 'commit', '-q', '-m', 'fixture'],
        cwd=repo, check=True, capture_output=True,
    )
    return repo


def _run_gate(script: Path, *, ref: str, probe_py: str | None = None):
    """Execute the gate script DIRECTLY, as DeterministicRunner does."""
    full_env = dict(os.environ)
    full_env['WRITE_TRIAGE_GATE_REF'] = ref
    full_env['CHECK_WRITE_TRIAGE_ATTACH_TARGET_PY'] = probe_py or sys.executable
    return subprocess.run(
        [str(script)],
        capture_output=True,
        text=True,
        timeout=240,
        env=full_env,
    )


def _assert_no_bare_grep_verdict(stdout: str) -> None:
    """Item 1 must no longer report a source-text grep as its verdict.

    A grep asserts which MECHANISM landed, not whether the invariant holds: it
    fails a correct option-(b) fix and passes prose that changes no behaviour.
    """
    assert 'candidate_id ABSENT' not in stdout, stdout
    assert 'candidate_id present' not in stdout, stdout


class TestFlipPreconditionsScript:
    """The gate script end to end. Item 1 delegates to the probe; items 2 and
    4 keep their existing eval-source patterns, unchanged."""

    def test_flat_judge_fails_item_1(self, tmp_path):
        repo = _make_gate_repo(tmp_path, judge='flat')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        assert 'FAIL  item 1' in proc.stdout, proc.stdout
        _assert_no_bare_grep_verdict(proc.stdout)

    def test_by_id_judge_with_fixed_eval_passes_everything(self, tmp_path):
        repo = _make_gate_repo(tmp_path, judge='by_id', eval_src='fixed')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'
        assert 'PASS  item 1' in proc.stdout, proc.stdout
        _assert_no_bare_grep_verdict(proc.stdout)

    def test_marking_candidates_zero_does_not_satisfy_the_gate(self, tmp_path):
        """The gate must not bless candidates[0]-as-attach-target."""
        repo = _make_gate_repo(tmp_path, judge='positional_target', eval_src='fixed')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        assert 'FAIL  item 1' in proc.stdout, proc.stdout
        assert _RESCUE_PATH in proc.stdout, proc.stdout

    def test_unreadable_ref_fails_closed(self):
        """NEGATIVE CONTROL, mandated by the script's own comment.

        A `$(...)` command substitution runs in a SUBSHELL, so a `fail=1`
        assigned inside one is discarded — the bug caught 2026-08-27, where an
        unreadable ref skipped its whole check block and the gate exited 0 on
        unverifiable input. Run against the REAL checkout, read-only.
        """
        proc = _run_gate(_GATE_SCRIPT, ref='no-such-ref')
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout

    def test_unresolvable_probe_interpreter_fails_closed(self, tmp_path):
        """A judge that WOULD pass, but no interpreter to prove it with.

        An unverifiable invariant is not a satisfied one — this must never be
        the difference between exit 1 and exit 0.
        """
        repo = _make_gate_repo(tmp_path, judge='by_id', eval_src='fixed')
        proc = _run_gate(
            repo / 'scripts' / _GATE_SCRIPT.name,
            ref=_FIXTURE_REF,
            probe_py=str(tmp_path / 'no-such-python3'),
        )
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout

    def test_items_2_and_4_still_fail_on_their_patterns(self, tmp_path):
        repo = _make_gate_repo(tmp_path, judge='by_id', eval_src='failing')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        assert 'FAIL  item 2' in proc.stdout, proc.stdout
        assert 'FAIL  item 4' in proc.stdout, proc.stdout

    def test_items_2_and_4_still_pass_on_the_fixed_patterns(self, tmp_path):
        repo = _make_gate_repo(tmp_path, judge='flat', eval_src='fixed')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert 'PASS  item 2' in proc.stdout, proc.stdout
        assert 'PASS  item 4' in proc.stdout, proc.stdout
