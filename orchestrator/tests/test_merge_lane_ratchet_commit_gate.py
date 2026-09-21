"""The ratchet gate that reads the COMMITTED DIFF, not a fresh measurement.

``scripts/merge_lane_metrics.py``'s write gate is the ratchet's enforcement
point for a *regeneration*: it refuses to absorb a raise into a baseline it can
read first. What it cannot see is the baseline deleted before the write, or
rendered at a scratch path and copied over -- with nothing at the destination to
compare against, every frozen measure resets and every ceiling is
re-grandfathered, unrefused and unrecorded. Two such raises reached main this
way (merge_lane/ports.py at 0b04534c7b, orchestrator/tests/conftest.py at
52d98220ad) with the ledger still reading ``"raises": []``.

``scripts/check_staged_ratchet_raise.py`` closes that by asking a different
question: not "does the tree match the baseline" but "does this COMMIT raise
anything". Every route to a widened baseline -- regenerate, delete-then-write,
write-elsewhere-and-copy, hand-edit -- lands as the same staged diff against
HEAD's blob, so one comparison covers them all.

Its git plumbing is why these tests live in their own module rather than in
``test_merge_lane_ratchet.py``: each case needs a real throwaway repo, and only
a module under ``orchestrator/tests`` can import ``_orch_helpers``'s
``assert_isolated_git_repo`` / ``git_env_with_ceiling`` -- the two per-call
layers of the esc-3072-3 defence that every git-in-tmp_path test here must use.
The instrument's own pure helpers stay unit-tested next to the instrument.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

# Same bootstrap as test_merge_lane_ratchet.py, and for the same reason:
# orchestrator/pyproject.toml's [tool.pyright] extraPaths already carries
# "../scripts", so importing the instrument needs no new config.
_SCRIPTS = Path(__file__).parents[2] / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import merge_lane_metrics as metrics  # type: ignore[import-not-found]  # noqa: E402

# The seed report is IMPORTED, never re-declared: a second synthetic baseline
# would drift from the one the instrument's own unit tests pin, and the two
# would then disagree about what a raise looks like.
from test_merge_lane_ratchet import _synthetic_report  # noqa: E402

_REPO_ROOT = Path(__file__).parents[2]
_GATE = _REPO_ROOT / 'scripts' / 'check_staged_ratchet_raise.py'


def _report_with(mutate: Callable[[dict], None]) -> dict:
    """The seed report with one measure perturbed. Never the live measurement."""
    report = copy.deepcopy(_synthetic_report())
    mutate(report)
    return report


def _raise_lines(report: dict) -> None:
    report['files']['a.py']['lines'] = 1005


def _lower_lines(report: dict) -> None:
    report['files']['a.py']['lines'] = 995


class _Repo:
    """A throwaway repo carrying the two ratchet artifacts at their real paths.

    Every git invocation goes through :meth:`git`, so ``cwd`` and the ceiling
    environment are established in exactly one place and no case builder can
    shell out unguarded (esc-3072-3).
    """

    def __init__(self, root: Path) -> None:
        # FIRST, before any subprocess: a rejected root must write nothing.
        assert_isolated_git_repo(root)
        self.root = root
        self.env = git_env_with_ceiling(root)

    @classmethod
    def seeded(cls, tmp_path: Path, report: dict | None = None) -> _Repo:
        """``git init`` plus one commit carrying a baseline and an empty ledger."""
        root = tmp_path / 'repo'
        root.mkdir()
        # The pre-flight cannot precede `git init` -- it refuses any directory
        # that is not ALREADY a repo root, which is what this call creates. The
        # ceiling is applied to the bootstrap on its own, and `git init` only
        # ever writes into its own cwd.
        bootstrap = subprocess.run(
            ['git', 'init', '-b', 'main'],
            cwd=str(root),
            capture_output=True,
            text=True,
            env=git_env_with_ceiling(root),
            check=False,
        )
        assert bootstrap.returncode == 0, bootstrap.stderr
        repo = cls(root)
        repo.git('config', 'user.name', 'ratchet gate test')
        repo.git('config', 'user.email', 'ratchet@example.invalid')
        # Pinned explicitly so an operator's GLOBAL core.hooksPath cannot reach
        # into a throwaway repo and run real hooks over it.
        repo.git('config', 'core.hooksPath', str(root / '.git' / 'hooks'))
        repo.write_baseline(report if report is not None else _synthetic_report())
        repo.write_ledger(metrics.empty_ledger())
        repo.commit_all('seed the ratchet artifacts')
        return repo

    def git(self, *args: str, check: bool = True) -> str:
        proc = subprocess.run(
            ['git', *args],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )
        if check:
            assert proc.returncode == 0, (
                f'git {" ".join(args)} failed (rc={proc.returncode}): '
                f'{proc.stderr.strip()}'
            )
        return proc.stdout.strip()

    def commit_all(self, message: str) -> str:
        self.git('add', '-A')
        self.git('commit', '-m', message)
        return self.git('rev-parse', 'HEAD')

    def _write(self, relpath: str, text: str) -> Path:
        target = self.root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
        return target

    def write_baseline(self, report: dict) -> Path:
        return self._write(metrics.BASELINE_RELPATH, metrics.render_baseline(report))

    def write_ledger(self, ledger: dict) -> Path:
        return self._write(metrics.LEDGER_RELPATH, metrics.render_ledger(ledger))

    def stage(self, *relpaths: str) -> None:
        self.git('add', '--', *relpaths)

    def gate(self) -> subprocess.CompletedProcess[str]:
        """Drive the auditor through its real entry point, as the hook does."""
        return subprocess.run(
            [sys.executable, str(_GATE), '--root', str(self.root)],
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )


def _ledger_with(*records: dict) -> dict:
    return {**metrics.empty_ledger(), 'raises': list(records)}


def _record(raises: list[metrics.Violation], task_id: str = '5722') -> dict:
    return metrics.authorization_record(
        metrics.RaiseAuthorization(task_id=task_id, reason=f'net-additive {task_id}'),
        raises,
    )


class TestTheStagedDiffIsAudited:
    """The gate's core arm: HEAD's baseline blob against the staged one."""

    def test_an_unrelated_staged_file_is_not_audited(self, tmp_path: Path) -> None:
        repo = _Repo.seeded(tmp_path)
        (repo.root / 'unrelated.py').write_text('x = 1\n', encoding='utf-8')
        repo.stage('unrelated.py')

        result = repo.gate()

        # An ordinary commit must never be ambushed by pre-existing debt, and
        # must not pay for a comparison it does not need.
        assert result.returncode == 0, result.stderr
        assert result.stdout == '' and result.stderr == ''

    def test_a_staged_baseline_that_moved_no_measure_is_clean(
        self, tmp_path: Path
    ) -> None:
        repo = _Repo.seeded(tmp_path)
        # Re-serialized rather than rewritten with identical bytes: identical
        # bytes stage NOTHING, so that variant would exercise the cheap
        # short-circuit above instead of the comparison this case is about.
        (repo.root / metrics.BASELINE_RELPATH).write_text(
            json.dumps(json.loads(
                (repo.root / metrics.BASELINE_RELPATH).read_text(encoding='utf-8')
            ), indent=4) + '\n',
            encoding='utf-8',
        )
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stderr

    def test_a_fall_needs_no_ceremony(self, tmp_path: Path) -> None:
        repo = _Repo.seeded(tmp_path)
        repo.write_baseline(_report_with(_lower_lines))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        # Lowering is the point of a ratchet: it must never need a ledger entry.
        assert result.returncode == 0, result.stderr

    def test_an_unrecorded_rise_is_refused_naming_the_measure_and_both_numbers(
        self, tmp_path: Path
    ) -> None:
        repo = _Repo.seeded(tmp_path)
        repo.write_baseline(_report_with(_raise_lines))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode != 0
        message = result.stderr
        assert 'lines' in message
        assert 'a.py' in message
        assert '1000' in message and '1005' in message
        # The ONE copy of what to do next, printed verbatim rather than
        # paraphrased into a fourth near-copy (task 5342, esc-5342-1).
        assert metrics.RAISE_REMEDY in message

    def test_a_covering_ledger_append_in_the_same_commit_allows_the_rise(
        self, tmp_path: Path
    ) -> None:
        repo = _Repo.seeded(tmp_path)
        repo.write_baseline(_report_with(_raise_lines))
        repo.write_ledger(
            _ledger_with(
                _record([
                    metrics._violation('lines', 'a.py', 1000, 1005),
                    metrics._violation(
                        'total:lines', metrics.CLUSTER_TOTAL_KEY, 1200, 1205
                    ),
                ])
            )
        )
        repo.stage(metrics.BASELINE_RELPATH, metrics.LEDGER_RELPATH)

        result = repo.gate()

        # THE SANCTIONED PATH. A raise is not forbidden -- it is never silent.
        assert result.returncode == 0, result.stderr

    def test_a_stale_ledger_entry_does_not_cover_the_rise(
        self, tmp_path: Path
    ) -> None:
        repo = _Repo.seeded(tmp_path)
        repo.write_baseline(_report_with(_raise_lines))
        # Right measure, right key, WRONG landing value -- the shape a
        # hand-written or copied-forward entry actually has.
        repo.write_ledger(
            _ledger_with(_record([metrics._violation('lines', 'a.py', 1000, 1003)]))
        )
        repo.stage(metrics.BASELINE_RELPATH, metrics.LEDGER_RELPATH)

        result = repo.gate()

        assert result.returncode != 0
        assert '1005' in result.stderr

    def test_a_staged_deletion_of_the_baseline_is_refused(
        self, tmp_path: Path
    ) -> None:
        repo = _Repo.seeded(tmp_path)
        repo.git('rm', '--quiet', '--', metrics.BASELINE_RELPATH)

        result = repo.gate()

        # The first half of "delete the destination first", closed at the
        # cheapest possible point -- before any comparison is attempted.
        assert result.returncode != 0
        assert 'delete' in result.stderr.lower()
        assert metrics.BASELINE_RELPATH in result.stderr

    def test_a_malformed_staged_baseline_names_its_cause(
        self, tmp_path: Path
    ) -> None:
        repo = _Repo.seeded(tmp_path)
        (repo.root / metrics.BASELINE_RELPATH).write_text(
            '{"files": {', encoding='utf-8'
        )
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode != 0
        # load_baseline's named hard failure, surfaced as a line a reader can
        # act on -- never a traceback, which reads as a broken instrument and
        # sends them hunting the wrong thing (INV-11).
        assert 'not valid JSON' in result.stderr
        assert 'Traceback' not in result.stderr
