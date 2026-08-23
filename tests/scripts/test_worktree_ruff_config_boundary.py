"""Executable rationale for task 3922's two rejected/adopted remedies.

WHY THIS FILE EXISTS. Task 3922's decision — fix the CACHE half of ruff's
worktree escape unconditionally, and make the CONFIG half LOUD rather than
"fixed" — rests on exactly two measurements. Both are properties of ruff's
config-resolution behaviour, so neither survives as prose: a future author
reading only the code would see a `RUFF_CACHE_DIR` injection plus a warning and
reasonably ask "why not just pass `--config` and be done?". This file answers
that in the only currency that cannot rot — a measurement that re-runs on every
merge.

THE DEFECT BEING REASONED ABOUT. ruff resolves its config and its cache by
walking parent directories up from each linted file, and a git VCS root does
NOT halt that walk. A task worktree at ``<parent>/.worktrees/<id>`` whose own
root ``pyproject.toml`` carries no ``[tool.ruff]`` therefore resolves the PARENT
checkout's table — reading a rule set out of another checkout's working tree,
which the branch does not control and which may be uncommitted. Measured on
this host at filing time: 286 of 567 worktrees carrying a pyproject.toml had no
``[tool.ruff]``.

(a) WHY ``--config`` WAS REJECTED. Aimed at a pyproject that declares no
``[tool.ruff]``, ``--config`` neither merges nor resumes the walk-up: it pins
that file and falls through to ruff's BUILT-IN defaults, silently dropping
``UP``/``B``/``SIM``/``I``. On the 286 stale worktrees that converts a
transiently OVER-strict gate into a permanently UNDER-strict green — "a rule set
nobody chose applies silently", the exact failure the repo-root ``[tool.ruff]``
block (task 3457) exists to eliminate, and the same carve-out flag shape
``test_scripts_module_config.py::_narrowing_flag_args`` already refuses for
pyright's ``-p /tmp/lax.json``.

(b) WHY ``RUFF_CACHE_DIR`` COULD BE APPLIED UNCONDITIONALLY. It is rule-NEUTRAL:
it moves ``cache_dir`` and changes nothing else — not the resolved settings
path, not the emitted rule codes. That neutrality is the whole licence for
applying it to every worktree at any branch base age without auditing which
ones are stale.

WHY SETS, NEVER COUNTS OR EXIT CODES. ``ruff check`` exits 0 both when the right
rules ran and when a narrower set ran clean; an exit code cannot carry the
claim, and a count cannot distinguish "one fewer finding" from "a different
rule". ``test_nonmember_ruff_config.py`` records that lesson at length; this
file inherits it.

WHY SYNTHETIC AND NOT THE AMBIENT CHECKOUT. An assertion bound to the live
worktree would be born green in any worktree whose base already declares
``[tool.ruff]`` (this one does) and would prove nothing about the stale case
that motivated the task.

WHY HERE AND NOT IN ``orchestrator/tests/``. ``tests/scripts/`` carries its own
module config (``tests/scripts/orchestrator.yaml``) with
``merge_verify_breadth: full``, so this guard actually runs on the merge gate
rather than only when the orchestrator package is in scope.

COST. Exactly five ruff subprocesses for the whole module, computed once in a
module-scoped fixture — this directory is a merge gate.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_MEMBER_SELECT = '["E", "F", "UP", "B", "SIM", "I"]'

# Unsorted imports (I001, in the members' set, NOT in ruff's built-in defaults)
# plus an unused import (F401, in BOTH). The pair is what makes the two rule
# sets distinguishable: a probe emitting only F-codes could not tell them apart.
_PROBE_SOURCE = 'import sys\nimport os\n\nprint(os.getcwd())\n'

_SETTINGS_PATH_PREFIX = 'Settings path:'
_CACHE_DIR_PREFIX = 'cache_dir ='


def _ruff(args: list[str], *, cwd: Path, env_extra: dict[str, str] | None = None):
    """Run ``ruff`` as ``sys.executable -m ruff``.

    Deterministic by construction, and a missing module FAILS the guard rather
    than skipping it — the never-skip doctrine ``test_nonmember_ruff_config.py``
    states for the sibling probe. A guard that skips itself on a broken
    environment silently stops defending the decision it exists to defend.
    """
    env = dict(os.environ)
    env.pop('RUFF_CACHE_DIR', None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--no-cache', *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, check=False,
    )
    assert 'No module named ruff' not in proc.stderr, (
        f'`{sys.executable} -m ruff` is unavailable: {proc.stderr}'
    )
    return proc


def _rule_codes(proc) -> set[str]:
    """The SET of rule codes ruff emitted. Never a count, never an exit code."""
    try:
        return {finding['code'] for finding in json.loads(proc.stdout)}
    except json.JSONDecodeError as exc:  # pragma: no cover - only on a broken env
        raise AssertionError(
            f'ruff did not emit parseable JSON (rc={proc.returncode}).\n'
            f'stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}'
        ) from exc


def _show_settings_field(proc, prefix: str) -> str:
    """Read one ``--show-settings`` field, matched by PREFIX not by line index."""
    for line in proc.stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip().strip('"').strip("'").strip('"')
    raise AssertionError(
        f'no {prefix!r} line in --show-settings output:\n{proc.stdout}'
    )


@pytest.fixture(scope='module')
def measurements(tmp_path_factory):
    """Build the M1 geometry once and take all five measurements.

    Module-scoped so the whole file costs five ruff spawns rather than five per
    test. The geometry: a parent checkout DECLARING ``[tool.ruff]``, a worktree
    at ``<parent>/.worktrees/wt`` declaring NONE, both ``git init``-ed (a VCS
    root does not stop the walk-up — omitting the init would prove something
    weaker than the real defect), and a real .py file under the worktree.
    """
    root = tmp_path_factory.mktemp('ruffgeo')
    parent = (root / 'parent').resolve()
    worktree = parent / '.worktrees' / 'wt'
    (worktree / 'scripts').mkdir(parents=True)

    (parent / 'pyproject.toml').write_text(
        '[project]\nname = "parentproj"\nversion = "0.1.0"\n\n'
        '[tool.ruff]\nline-length = 100\n\n'
        f'[tool.ruff.lint]\nselect = {_MEMBER_SELECT}\n'
    )
    # The STALE worktree root: a pyproject.toml with NO [tool.ruff] table. This
    # is the 286-of-567 case, and it is what ruff walks straight past.
    (worktree / 'pyproject.toml').write_text(
        '[project]\nname = "wtproj"\nversion = "0.1.0"\n'
    )
    for repo in (parent, worktree):
        subprocess.run(['git', 'init', '-q', str(repo)], check=True, capture_output=True)

    probe = worktree / 'scripts' / 's.py'
    probe.write_text(_PROBE_SOURCE)

    rel = 'scripts/s.py'
    local_cache = str(worktree / '.ruff_cache')
    json_args = ['--output-format', 'json']
    return {
        'parent': parent,
        'worktree': worktree,
        'baseline_codes': _rule_codes(_ruff([*json_args, rel], cwd=worktree)),
        'pinned_codes': _rule_codes(
            _ruff(['--config', 'pyproject.toml', *json_args, rel], cwd=worktree)
        ),
        'cached_codes': _rule_codes(
            _ruff([*json_args, rel], cwd=worktree,
                  env_extra={'RUFF_CACHE_DIR': local_cache})
        ),
        'baseline_settings': _ruff(['--show-settings', rel], cwd=worktree),
        'cached_settings': _ruff(['--show-settings', rel], cwd=worktree,
                                 env_extra={'RUFF_CACHE_DIR': local_cache}),
        'local_cache': local_cache,
    }


class TestTheEscapeReproduces:
    """The premise both remedies were chosen against."""

    def test_stale_worktree_resolves_the_parent_checkouts_config(self, measurements):
        settings = _show_settings_field(
            measurements['baseline_settings'], _SETTINGS_PATH_PREFIX,
        )
        assert Path(settings) == measurements['parent'] / 'pyproject.toml', (
            'the walk-up escape did not reproduce; if ruff changed its config '
            'resolution, task 3922\'s premise needs re-measuring before this '
            'guard is edited'
        )
        assert measurements['baseline_codes'] == {'I001', 'F401'}


class TestConfigPinIsAFalseGreen:
    """(a) Why ``--config`` was REJECTED — do not "simplify" the fix into it."""

    def test_pinning_a_ruleless_pyproject_shrinks_the_rule_set(self, measurements):
        baseline = measurements['baseline_codes']
        pinned = measurements['pinned_codes']

        # STRICT subset, asserted as a set relation: `--config` at a pyproject
        # declaring no [tool.ruff] does not merge and does not resume the
        # walk-up — it falls through to ruff's built-in defaults, which carry F
        # but not I/UP/B/SIM.
        assert pinned < baseline, (
            f'expected --config to STRICTLY narrow the rule set; '
            f'baseline={sorted(baseline)} pinned={sorted(pinned)}'
        )
        assert pinned == {'F401'}
        # Name the dropped rule explicitly: this is the false green. The pinned
        # run reports clean on a violation the branch's real rule set catches.
        assert 'I001' in baseline - pinned

    def test_the_false_green_is_invisible_in_the_exit_code(self, measurements):
        # Why this guard compares SETS. Both runs exit non-zero here only
        # because F401 survives; had the file carried ONLY an I001 violation the
        # pinned run would exit 0 and look green. An exit code cannot carry the
        # claim, so no assertion in this file may rest on one.
        assert measurements['pinned_codes'] != measurements['baseline_codes']


class TestCacheRedirectIsRuleNeutral:
    """(b) Why ``RUFF_CACHE_DIR`` may be applied to EVERY worktree, unconditionally."""

    def test_redirect_moves_the_cache_dir(self, measurements):
        before = _show_settings_field(
            measurements['baseline_settings'], _CACHE_DIR_PREFIX,
        )
        after = _show_settings_field(
            measurements['cached_settings'], _CACHE_DIR_PREFIX,
        )
        assert before != after
        assert Path(after) == Path(measurements['local_cache'])
        # ...and the escaping default is precisely the coupling being cut: the
        # parent checkout's cache, shared with every sibling worktree.
        assert Path(before) == measurements['parent'] / '.ruff_cache'

    def test_redirect_changes_neither_the_settings_path_nor_the_rules(
        self, measurements,
    ):
        # THE licence for applying the lever unconditionally. If this ever goes
        # red, the cache injection in verify._target_subprocess_env is no longer
        # rule-neutral and must become conditional (or be reconsidered) — it is
        # not merely this assertion that needs updating.
        assert _show_settings_field(
            measurements['cached_settings'], _SETTINGS_PATH_PREFIX,
        ) == _show_settings_field(
            measurements['baseline_settings'], _SETTINGS_PATH_PREFIX,
        )
        assert measurements['cached_codes'] == measurements['baseline_codes']
