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

EVERY REFUSAL PINS THE EXIT CODE rather than merely non-zero. The ladder -- 0
clean, 1 a policy refusal this commit must fix, 2 the gate could not do its job
-- is the gate's only machine-readable output, and a caller that reads 2 as
"instrument down, retry or ignore" must never be handed a policy refusal under
it. Asserting ``!= 0`` everywhere is what let exactly that drift in unseen: a
rewritten ledger exited 2 with a fully green suite.
"""
from __future__ import annotations

import copy
import enum
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

# The seed report is IMPORTED, never re-declared: a second synthetic baseline
# would drift from the one the instrument's own unit tests pin, and the two
# would then disagree about what a raise looks like. It sits in a fixtures
# module of its own rather than in either suite -- see that module's docstring.
from _merge_lane_ratchet_fixtures import synthetic_report
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

# Same bootstrap as test_merge_lane_ratchet.py, and for the same reason:
# orchestrator/pyproject.toml's [tool.pyright] extraPaths already carries
# "../scripts", so importing the instrument needs no new config.
_SCRIPTS = Path(__file__).parents[2] / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import merge_lane_metrics as metrics  # type: ignore[import-not-found]  # noqa: E402

_REPO_ROOT = Path(__file__).parents[2]
_GATE = _REPO_ROOT / metrics.COMMIT_GATE_RELPATH


def _report_with(mutate: Callable[[dict], None]) -> dict:
    """The seed report with one measure perturbed. Never the live measurement."""
    report = copy.deepcopy(synthetic_report())
    mutate(report)
    return report


def _raise_lines(report: dict) -> None:
    report['files']['a.py']['lines'] = 1005


def _lower_lines(report: dict) -> None:
    report['files']['a.py']['lines'] = 995


def _with_a_py_lines(lines: int) -> dict:
    """The seed report with a.py's `lines` at *lines* (the seed holds 1000)."""
    return _report_with(lambda report: report['files']['a.py'].__setitem__('lines', lines))


class _Baseline(enum.Enum):
    """A side's baseline move that is not a report."""

    #: `git rm`ed on that side -- the hook-less deletion a branch can carry.
    REMOVED = enum.auto()


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
        repo.write_baseline(report if report is not None else synthetic_report())
        repo.write_ledger(metrics.empty_ledger())
        repo.commit_all('seed the ratchet artifacts')
        return repo

    @classmethod
    def unborn(cls, tmp_path: Path) -> _Repo:
        """An initialized repo with NO commit yet -- HEAD names nothing."""
        root = tmp_path / 'repo'
        root.mkdir()
        bootstrap = subprocess.run(
            ['git', 'init', '-b', 'main'],
            cwd=str(root),
            capture_output=True,
            text=True,
            env=git_env_with_ceiling(root),
            check=False,
        )
        assert bootstrap.returncode == 0, bootstrap.stderr
        return cls(root)

    @classmethod
    def mid_merge(
        cls,
        tmp_path: Path,
        *,
        ours: dict | _Baseline | None = None,
        theirs: dict | _Baseline | None = None,
        ours_ledger: dict | None = None,
        theirs_ledger: dict | None = None,
    ) -> _Repo:
        """The state a merge resolver finds: `git merge main` STOPPED on a conflict.

        Branch 'task' (HEAD) commits *ours*; main (MERGE_HEAD) then commits
        *theirs* with no hooks, which is exactly main's hook-less route. A
        sentinel file both sides wrote is what stops the merge, and it is
        resolved and staged here. The ratchet artifacts are left exactly as git
        merged them, for each test to stage the resolution it is about.
        """
        repo = cls.seeded(tmp_path)
        repo.git('switch', '--quiet', '-c', 'task')
        repo._commit_side('ours', ours, ours_ledger)
        repo.git('switch', '--quiet', 'main')
        repo._commit_side('theirs', theirs, theirs_ledger)
        repo.git('switch', '--quiet', 'task')
        repo.git('merge', 'main', check=False)
        merge_head = repo.git('rev-parse', '-q', '--verify', 'MERGE_HEAD', check=False)
        assert merge_head, '`git merge main` did not stop on the sentinel conflict'
        repo._write('conflict.txt', 'resolved\n')
        repo.stage('conflict.txt')
        return repo

    def _commit_side(
        self, side: str, baseline: dict | _Baseline | None, ledger: dict | None
    ) -> None:
        if baseline is _Baseline.REMOVED:
            self.git('rm', '--quiet', '--', metrics.BASELINE_RELPATH)
        elif baseline is not None:
            self.write_baseline(baseline)
        if ledger is not None:
            self.write_ledger(ledger)
        self._write('conflict.txt', f'{side}\n')
        self.commit_all(f'{side}: move the ratchet artifacts')

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

    def write_baseline_text(self, text: str) -> Path:
        """Raw bytes, for a blob ``render_baseline`` would never produce."""
        return self._write(metrics.BASELINE_RELPATH, text)

    def write_ledger(self, ledger: dict) -> Path:
        return self._write(metrics.LEDGER_RELPATH, metrics.render_ledger(ledger))

    def stage(self, *relpaths: str) -> None:
        self.git('add', '--', *relpaths)

    def attempt_commit(self, message: str) -> subprocess.CompletedProcess[str]:
        """A real `git commit`, hooks and all, whose outcome is the assertion."""
        return subprocess.run(
            ['git', 'commit', '-m', message],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )

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

        assert result.returncode == 1, result.stdout + result.stderr
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
                    metrics.Violation.rose('lines', 'a.py', 1000, 1005),
                    metrics.Violation.rose(
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
            _ledger_with(_record([metrics.Violation.rose('lines', 'a.py', 1000, 1003)]))
        )
        repo.stage(metrics.BASELINE_RELPATH, metrics.LEDGER_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert '1005' in result.stderr

    def test_a_recorded_raise_in_heads_ledger_is_not_a_standing_permission(
        self, tmp_path: Path
    ) -> None:
        # LEDGER_README, in terms: "It is NOT a permission list: nothing in this
        # file grants a future raise, and no entry here will ever let
        # --write-baseline absorb one." The gate honours that by asking only
        # THIS COMMIT's appended entries to cover the rise, so a record already
        # committed -- naming the exact measure, key and landing value -- must
        # license nothing. Every other covering case writes its record in the
        # same commit, so a refactor that passed the whole staged ledger (or
        # HEAD's) to `unrecorded_raises` would turn the file into the standing
        # permission list its own README forbids with all of them still green.
        repo = _Repo.seeded(tmp_path)
        repo.write_ledger(
            _ledger_with(
                _record([
                    metrics.Violation.rose('lines', 'a.py', 1000, 1005),
                    metrics.Violation.rose(
                        'total:lines', metrics.CLUSTER_TOTAL_KEY, 1200, 1205
                    ),
                ])
            )
        )
        repo.commit_all('record a raise in a commit that does not take it')

        repo.write_baseline(_report_with(_raise_lines))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert '1005' in result.stderr

    def test_a_staged_deletion_of_the_baseline_is_refused(
        self, tmp_path: Path
    ) -> None:
        repo = _Repo.seeded(tmp_path)
        repo.git('rm', '--quiet', '--', metrics.BASELINE_RELPATH)

        result = repo.gate()

        # The first half of "delete the destination first", closed at the
        # cheapest possible point -- before any comparison is attempted.
        assert result.returncode == 1, result.stdout + result.stderr
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

        assert result.returncode == 2, result.stdout + result.stderr
        # load_baseline's named hard failure, surfaced as a line a reader can
        # act on -- never a traceback, which reads as a broken instrument and
        # sends them hunting the wrong thing (INV-11).
        assert 'not valid JSON' in result.stderr
        assert 'Traceback' not in result.stderr


# ---------------------------------------------------------------------------
# The two real incidents, carried as FIXTURES rather than read out of this
# checkout's live git history. A regression pin that reads the repo it runs in
# stops pinning anything the moment that history is rewritten or shallow-cloned,
# and it would read production state from a test (esc-3072-3's whole subject).

_CONFTEST = 'orchestrator/tests/conftest.py'
_PORTS = 'orchestrator/src/orchestrator/merge_lane/ports.py'


def _incident_report() -> dict:
    """The seed report plus the two paths the incidents moved, at their OLD numbers."""
    report = copy.deepcopy(synthetic_report())
    report['files'][_CONFTEST] = {
        'lines': 1172,
        'prose_lines': 752,
        'cognitive': 42,
        'function_local_imports': 7,
        'reexport_names': 5,
    }
    report['files'][_PORTS] = {
        'lines': 255,
        'prose_lines': 47,
        'cognitive': 1,
        'function_local_imports': 0,
        'reexport_names': 0,
    }
    return report


def _conftest_rise(report: dict) -> None:
    """The 52d98220ad shape: conftest.py 1172 -> 1187 lines, 752 -> 770 prose."""
    report['files'][_CONFTEST].update(lines=1187, prose_lines=770)


def _ports_rise(report: dict) -> None:
    """The 0b04534c7b shape: ports.py 255 -> 274 lines, 47 -> 59 prose."""
    report['files'][_PORTS].update(lines=274, prose_lines=59)


class TestRestoreCarveOut:
    """Undoing the LAST change to the baseline is not a new raise.

    Ruled by Leo and verified exact on the real revert: `3e7d55ce47:<baseline>`,
    `5f577b9613^:<baseline>` and `52d98220ad:<baseline>` are all blob
    a0fb5cc8e0fb1e9ce363c81538a350dbfc12c191. The measured shape is one step
    back: a0fb5cc8e0 was the value at b79315a31c, 5f577b9613 replaced it with
    0a42c2ee7d, and 3e7d55ce47 put a0fb5cc8e0 straight back --
    `compare_baseline_files(0a42c2ee7d, a0fb5cc8e0)` measures 4 raises, so the
    carve-out really does fire for it. Without it, undoing a revert would demand
    a fresh authorization for a raise nobody re-introduced.

    NOT "a blob this path ever carried", and not "already reviewed". Both were
    wrong and this class once said both. The images that motivated this task
    landed at 0b04534c7b and 52d98220ad with the ledger reading `"raises": []`,
    so they were never reviewed as raises at all; and admitting any historical
    blob is a wholesale ratchet reset, which ``TestTheCarveOutIsOneStepBack``
    now refutes directly. The honest reason is narrower: the path's
    immediately-previous value is the state this repository held one commit ago.
    """

    @staticmethod
    def _history_with_a_revert(tmp_path: Path) -> tuple[_Repo, str, dict]:
        """low -> high -> low, the exact shape the conftest revert left behind."""
        low = _incident_report()
        high = copy.deepcopy(low)
        _conftest_rise(high)

        repo = _Repo.seeded(tmp_path, report=low)
        repo.write_baseline(high)
        absorbed = repo.commit_all('absorb the conftest rise')
        repo.write_baseline(low)
        repo.commit_all('revert the absorbed rise')
        return repo, absorbed, high

    def test_restoring_the_conftest_blob_is_allowed_and_names_its_commit(
        self, tmp_path: Path
    ) -> None:
        repo, absorbed, high = self._history_with_a_revert(tmp_path)
        repo.write_baseline(high)
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stderr
        # NAMING the provenance commit is what keeps this from reading as the
        # gate simply failing to notice: a reviewer can go look at it.
        assert absorbed in (result.stdout + result.stderr)

    def test_the_ports_incident_is_refused_naming_both_measures(
        self, tmp_path: Path
    ) -> None:
        repo, _absorbed, _high = self._history_with_a_revert(tmp_path)
        raised = _incident_report()
        _ports_rise(raised)
        repo.write_baseline(raised)
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        # A blob never recorded at this path, with an empty ledger: the
        # 0b04534c7b case, which reached main unexamined.
        assert result.returncode == 1, result.stdout + result.stderr
        message = result.stderr
        assert _PORTS in message
        assert 'lines' in message and 'prose_lines' in message
        assert '255' in message and '274' in message
        assert '47' in message and '59' in message

    def test_a_blob_one_byte_off_the_historical_one_is_refused(
        self, tmp_path: Path
    ) -> None:
        # WITHOUT THIS THE CARVE-OUT IS UNFALSIFIABLE. A test that only ever
        # stages the exact historical blob cannot tell "identity" from "these
        # measures were once seen high".
        repo, _absorbed, high = self._history_with_a_revert(tmp_path)
        repo.write_baseline_text(metrics.render_baseline(high) + ' ')
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert _CONFTEST in result.stderr

    def test_the_same_blob_at_another_path_does_not_license_the_baseline(
        self, tmp_path: Path
    ) -> None:
        # PATH SCOPING. Identity is scoped to blobs recorded at the BASELINE's
        # own path, so a blob that merely exists somewhere in the object store
        # -- trivially arranged by committing it anywhere -- licenses nothing.
        low = _incident_report()
        high = copy.deepcopy(low)
        _conftest_rise(high)
        repo = _Repo.seeded(tmp_path, report=low)
        (repo.root / 'scratch_baseline.json').write_text(
            metrics.render_baseline(high), encoding='utf-8'
        )
        repo.commit_all('park a widened baseline at an unrelated path')

        repo.write_baseline(high)
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert _CONFTEST in result.stderr

    def test_a_fall_is_clean_without_consulting_history(
        self, tmp_path: Path
    ) -> None:
        repo, _absorbed, _high = self._history_with_a_revert(tmp_path)
        repo.write_baseline(_report_with(_lower_lines))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stderr
        # The carve-out is consulted only after a raise was measured, so a fall
        # must not announce a restore it never looked for.
        assert 'restore' not in result.stdout.lower()

    def test_an_unchanged_baseline_is_clean_and_silent(self, tmp_path: Path) -> None:
        repo, _absorbed, _high = self._history_with_a_revert(tmp_path)
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stderr
        assert result.stdout == '' and result.stderr == ''


class TestLedgerIsAppendOnlyAtTheGate:
    """The arm where the LEDGER moves, whether or not the baseline did.

    ``append_authorization``'s docstring promises that "a reviewer reading the
    file reads every raise this baseline has ever absorbed". It can only keep
    that for its OWN writes, while the file is also edited by rebases, merges
    and hands. A gate that looked at the ledger only when the baseline also
    moved would leave a commit that quietly deletes historical entries
    unexamined -- so the promise would stay unenforced in exactly the case that
    breaks it (heuristic 10, uniformly).
    """

    @staticmethod
    def _with_history(tmp_path: Path) -> tuple[_Repo, dict]:
        repo = _Repo.seeded(tmp_path)
        history = _record([metrics.Violation.rose('lines', 'a.py', 990, 1000)], '5485')
        repo.write_ledger(_ledger_with(history))
        repo.commit_all('record a historical raise')
        return repo, history

    def test_appending_to_the_ledger_alone_is_clean(self, tmp_path: Path) -> None:
        repo, history = self._with_history(tmp_path)
        repo.write_ledger(_ledger_with(history, _record([], '5722')))
        repo.stage(metrics.LEDGER_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stderr

    def test_dropping_a_historical_entry_is_refused(self, tmp_path: Path) -> None:
        repo, _history = self._with_history(tmp_path)
        repo.write_ledger(_ledger_with())
        repo.stage(metrics.LEDGER_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'append-only' in result.stderr
        assert '1' in result.stderr

    def test_rewriting_a_historical_entry_in_place_is_refused(
        self, tmp_path: Path
    ) -> None:
        # Same COUNT, changed content: a check that compared lengths would pass
        # this, and a rewritten `reason` is exactly how a raise stops reading as
        # what it was.
        repo, _history = self._with_history(tmp_path)
        forged = _record([metrics.Violation.rose('lines', 'a.py', 990, 1000)], '5485')
        forged['reason'] = 'actually it was a refactor'
        repo.write_ledger(_ledger_with(forged))
        repo.stage(metrics.LEDGER_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'append-only' in result.stderr

    def test_a_covering_append_does_not_launder_a_rewritten_history(
        self, tmp_path: Path
    ) -> None:
        # A CORRECTLY COVERING APPEND DOES NOT LAUNDER A REWRITTEN HISTORY.
        # The raise is authorized exactly as the sanctioned path asks and a
        # recorded entry is gone in the same commit; the append-only refusal
        # TAKES PRECEDENCE, which is what this pins. The two are not reported
        # together: the ledger arm is resolved before the baseline arm and
        # returns on its own, because a rewritten history cannot be excused by
        # anything else in the commit.
        repo, _history = self._with_history(tmp_path)
        repo.write_baseline(_report_with(_raise_lines))
        repo.write_ledger(
            _ledger_with(
                _record([
                    metrics.Violation.rose('lines', 'a.py', 1000, 1005),
                    metrics.Violation.rose(
                        'total:lines', metrics.CLUSTER_TOTAL_KEY, 1200, 1205
                    ),
                ])
            )
        )
        repo.stage(metrics.BASELINE_RELPATH, metrics.LEDGER_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'append-only' in result.stderr

    def test_a_malformed_staged_ledger_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        repo, _history = self._with_history(tmp_path)
        (repo.root / metrics.LEDGER_RELPATH).write_text(
            '{"raises": ', encoding='utf-8'
        )
        repo.stage(metrics.LEDGER_RELPATH)

        result = repo.gate()

        # A ledger that cannot be parsed is one whose history cannot be audited,
        # and it must NEVER read as "nothing was authorized" (load_ledger's own
        # stated polarity).
        assert result.returncode == 2, result.stdout + result.stderr
        assert 'not valid JSON' in result.stderr
        assert 'Traceback' not in result.stderr

    def test_an_unborn_head_with_nothing_staged_does_no_archaeology(
        self, tmp_path: Path
    ) -> None:
        # The cheap filter must decide BEFORE anything consults HEAD: a repo
        # whose HEAD names nothing is the sharpest way to assert that, because
        # every history read would fail on it.
        result = _Repo.unborn(tmp_path).gate()

        assert result.returncode == 0, result.stderr
        assert result.stdout == '' and result.stderr == ''


class TestAConflictedMergeLedgerIsAuditedAgainstBothParents:
    """A merge's LEDGER descends from two recorded histories, not one.

    esc-3620-11: `git commit` finishing task/3620's conflicted merge with main
    was refused for 18 "raises" that main itself had made. Measured on git 2.43:
    a conflicted merge finished with `git commit` runs PRE-COMMIT with
    MERGE_HEAD set -- not pre-merge-commit -- so this gate sees every merge
    resolver's commit, and HEAD is only one of its two parents. Both parents'
    recorded entries must survive whole, and only what the merge ITSELF
    appends is its own.
    """

    @pytest.mark.parametrize(
        'theirs_first', [True, False], ids=['theirs-then-ours', 'ours-then-theirs']
    )
    def test_both_sides_appended_entries_resolve_in_either_order(
        self, tmp_path: Path, theirs_first: bool
    ) -> None:
        ours_own, theirs_own = _record([], '5722'), _record([], '3620')
        repo = _Repo.mid_merge(
            tmp_path,
            ours_ledger=_ledger_with(ours_own),
            theirs_ledger=_ledger_with(theirs_own),
        )
        blocks = (theirs_own, ours_own) if theirs_first else (ours_own, theirs_own)
        repo.write_ledger(_ledger_with(*blocks))
        repo.stage(metrics.LEDGER_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stdout + result.stderr

    def test_keeping_heads_ledger_over_merge_heads_entry_is_refused(
        self, tmp_path: Path
    ) -> None:
        # NOTHING DIFFERS FROM HEAD, which is why a HEAD-only filter never
        # audited this: the resolution kept HEAD's bytes and silently dropped
        # the entry main recorded.
        repo = _Repo.mid_merge(tmp_path, theirs_ledger=_ledger_with(_record([], '3620')))
        repo.git('checkout', 'HEAD', '--', metrics.LEDGER_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'append-only' in result.stderr

    def test_an_entry_merge_head_recorded_is_not_a_standing_permission(
        self, tmp_path: Path
    ) -> None:
        # Main RECORDED 1005 -> 1009 without taking it. Counted as "appended
        # vs HEAD", that record would license the merge to take the raise --
        # the standing permission LEDGER_README forbids.
        recorded = _ledger_with(
            _record(
                [
                    metrics.Violation.rose('lines', 'a.py', 1005, 1009),
                    metrics.Violation.rose(
                        'total:lines', metrics.CLUSTER_TOTAL_KEY, 1205, 1209
                    ),
                ],
                '3620',
            )
        )
        repo = _Repo.mid_merge(
            tmp_path, theirs=_with_a_py_lines(1005), theirs_ledger=recorded
        )
        repo.write_baseline(_with_a_py_lines(1009))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert '1009' in result.stderr

    def test_a_baseline_merge_head_carries_may_not_be_dropped(
        self, tmp_path: Path
    ) -> None:
        # The task branch removed the baseline hook-lessly and main never
        # touched it, so git's merge carries the deletion -- invisible against
        # HEAD, which lacks the file too.
        repo = _Repo.mid_merge(tmp_path, ours=_Baseline.REMOVED)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'delete' in result.stderr.lower()


class TestAConflictedMergeBaselineIsBoundedByBothParents:
    """A merge's staged BASELINE is compared against a 3-way bound of its parents.

    esc-3620-11's 18 refused "raises" were main's own moves, carried in by git's
    clean file-level merge while the resolver made no choice about the ratchet
    files at all. Against HEAD alone each read as new; against either parent,
    keeping one side's stale value would re-absorb the other side's lowering.
    The bound is git's own rule per measure: a measure one side moved stands at
    that side's value, and where both moved, the higher side bounds it. Only
    entries the merge ITSELF appends cover anything above it.
    """

    @staticmethod
    def _unrelated_mid_merge(tmp_path: Path, *, ours: dict, theirs: dict) -> _Repo:
        """Two histories with NO common ancestor, stopped on a baseline add/add."""
        repo = _Repo.seeded(tmp_path, report=theirs)
        repo.git('switch', '--quiet', '--orphan', 'task')
        repo.write_baseline(ours)
        repo.write_ledger(metrics.empty_ledger())
        repo.commit_all('ours: an unrelated history')
        repo.git('merge', '--allow-unrelated-histories', 'main', check=False)
        merge_head = repo.git('rev-parse', '-q', '--verify', 'MERGE_HEAD', check=False)
        assert merge_head, 'the unrelated merge did not stop on the baseline add/add'
        return repo

    def test_the_incident_takes_merge_heads_unrecorded_move_cleanly(
        self, tmp_path: Path
    ) -> None:
        # Main moved a.py 1000 -> 1005 hook-lessly, with no ledger entry, and
        # git's merge carried that move in untouched.
        repo = _Repo.mid_merge(tmp_path, theirs=_with_a_py_lines(1005))
        baseline = metrics.BASELINE_RELPATH
        assert repo.git('rev-parse', f':{baseline}') == repo.git(
            'rev-parse', f'MERGE_HEAD:{baseline}'
        )

        result = repo.gate()

        assert result.returncode == 0, result.stdout + result.stderr

    def test_exceeding_both_parents_unrecorded_is_refused(self, tmp_path: Path) -> None:
        repo = _Repo.mid_merge(tmp_path, theirs=_with_a_py_lines(1005))
        repo.write_baseline(_with_a_py_lines(1009))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'a.py' in result.stderr and '1009' in result.stderr
        assert metrics.RAISE_REMEDY in result.stderr

    def test_a_covering_entry_the_merge_itself_appends_allows_it(
        self, tmp_path: Path
    ) -> None:
        repo = _Repo.mid_merge(tmp_path, theirs=_with_a_py_lines(1005))
        repo.write_baseline(_with_a_py_lines(1009))
        repo.write_ledger(
            _ledger_with(
                _record(
                    [
                        metrics.Violation.rose('lines', 'a.py', 1005, 1009),
                        metrics.Violation.rose(
                            'total:lines', metrics.CLUSTER_TOTAL_KEY, 1205, 1209
                        ),
                    ],
                    '5792',
                )
            )
        )
        repo.stage(metrics.BASELINE_RELPATH, metrics.LEDGER_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stdout + result.stderr

    def test_keeping_heads_stale_value_over_a_merge_head_lowering_is_refused(
        self, tmp_path: Path
    ) -> None:
        # Clean against HEAD, which is why a HEAD-only audit let it through.
        repo = _Repo.mid_merge(tmp_path, theirs=_with_a_py_lines(995))
        repo.git('checkout', 'HEAD', '--', metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'a.py' in result.stderr

    def test_taking_merge_heads_stale_value_over_a_head_lowering_is_refused(
        self, tmp_path: Path
    ) -> None:
        # The single-parent restore carve-out reads this as "undoing the last
        # change": the seed blob IS the path's previous value in HEAD's history.
        # In a merge it discards HEAD's own lowering, so it must not be asked.
        repo = _Repo.mid_merge(tmp_path, ours=_with_a_py_lines(995))
        repo.write_baseline(synthetic_report())
        repo.stage(metrics.BASELINE_RELPATH)
        baseline = metrics.BASELINE_RELPATH
        assert repo.git('rev-parse', f':{baseline}') == repo.git(
            'rev-parse', f'HEAD~1:{baseline}'
        )

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'restores the value this path held' not in result.stdout

    def test_where_both_parents_moved_a_measure_the_higher_bounds_it(
        self, tmp_path: Path
    ) -> None:
        repo = _Repo.mid_merge(
            tmp_path, ours=_with_a_py_lines(990), theirs=_with_a_py_lines(1005)
        )
        repo.write_baseline(_with_a_py_lines(1005))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stdout + result.stderr

    def test_unrelated_histories_are_bounded_by_the_higher_parent(
        self, tmp_path: Path
    ) -> None:
        # No common ancestor: git merges such histories against the empty tree.
        repo = self._unrelated_mid_merge(
            tmp_path, ours=synthetic_report(), theirs=_with_a_py_lines(1005)
        )
        repo.write_baseline(_with_a_py_lines(1005))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_head_without_a_baseline_does_not_make_the_merge_a_first_write(
        self, tmp_path: Path
    ) -> None:
        # The incident's LEDGER shape, applied to the baseline: HEAD lacks the
        # file, so the single-parent arm has nothing to compare and would pass
        # ANY staged baseline as a first write.
        repo = _Repo.mid_merge(tmp_path, ours=_Baseline.REMOVED)
        repo.write_baseline(_with_a_py_lines(1009))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        assert '1009' in result.stderr

    def test_a_head_without_a_baseline_may_take_merge_heads(
        self, tmp_path: Path
    ) -> None:
        repo = _Repo.mid_merge(tmp_path, ours=_Baseline.REMOVED)
        repo.git('checkout', 'MERGE_HEAD', '--', metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stdout + result.stderr


#: The four real files a miniature repo needs before its hooks mean anything.
#: Copied VERBATIM rather than restated: if someone renames the auditor or drops
#: the project-checks section, the copied hook stops refusing and these go red.
_WIRED_FILES = (
    'hooks/pre-commit',
    'hooks/project-checks',
    'scripts/merge_lane_metrics.py',
    'scripts/check_staged_ratchet_raise.py',
)


def _hook_repo(tmp_path: Path) -> _Repo:
    """A throwaway repo whose `git commit` runs THIS checkout's real hooks.

    The hooks resolve their own repo root, so pointing a throwaway repo's
    ``core.hooksPath`` at this checkout's ``hooks/`` would make project-checks
    look for ``scripts/`` inside the throwaway and find nothing. Copying is what
    makes the test pin the WIRING rather than a re-stated value.

    The branch is switched BEFORE the hooks are installed, and is never main: on
    main the copied project-checks goes on to run `uv run ruff check` against a
    tmp repo with no packages. That this test only works on a non-main branch is
    itself the reachability property the task is about.
    """
    repo = _Repo.seeded(tmp_path)
    repo.git('switch', '--quiet', '-c', 'task/5722-gate')
    _install_real_hooks(repo)
    return repo


def _install_real_hooks(repo: _Repo) -> None:
    """Copy THIS checkout's wired files in, point git at them, and commit them.

    The commit that installs them already runs them, harmlessly: it stages no
    ratchet artifact, and it must be made off main.
    """
    for relpath in _WIRED_FILES:
        target = repo.root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO_ROOT / relpath, target)
        target.chmod(0o755)
    repo.git('config', 'core.hooksPath', 'hooks')
    repo.commit_all('install the real hooks and the instrument')


def _hook_repo_mid_merge(tmp_path: Path, upstream: dict) -> _Repo:
    """A hooked task branch stopped mid-merge with an upstream that moved the baseline.

    The upstream branch moves the baseline BEFORE any hook exists, which is the
    hook-less route main's moves arrive by. A sentinel both branches wrote stops
    `git merge upstream`, and it is resolved and staged here; the baseline is
    left as git merged it. Everything stays OFF main, for `_hook_repo`'s reason.
    """
    repo = _Repo.seeded(tmp_path)
    repo.git('switch', '--quiet', '-c', 'upstream')
    repo.write_baseline(upstream)
    (repo.root / 'conflict.txt').write_text('upstream\n', encoding='utf-8')
    repo.commit_all('upstream: move the baseline hook-lessly')
    repo.git('switch', '--quiet', 'main')
    repo.git('switch', '--quiet', '-c', 'task/5792-merge')
    (repo.root / 'conflict.txt').write_text('task\n', encoding='utf-8')
    _install_real_hooks(repo)
    repo.git('merge', 'upstream', check=False)
    merge_head = repo.git('rev-parse', '-q', '--verify', 'MERGE_HEAD', check=False)
    assert merge_head, '`git merge upstream` did not stop on the sentinel conflict'
    (repo.root / 'conflict.txt').write_text('resolved\n', encoding='utf-8')
    repo.stage('conflict.txt')
    return repo


class TestTheHookActuallyRunsTheGate:
    """A gate wired into a path nothing runs is the failure this task is about.

    Both real incidents are MERGE commits onto main ("Merge task/5485 into
    main", "Merge task/5675 into main"), and main is advanced by the merge
    worker with `git update-ref`, which runs no hooks at all. Raises are
    INTRODUCED by commits on task branches, so that is where a commit-time gate
    has to bite -- and until this task, hooks/pre-commit returned 0 on every
    branch that is not main, before project-checks was ever reached.
    """

    def test_an_unrecorded_rise_fails_the_commit_on_a_task_branch(
        self, tmp_path: Path
    ) -> None:
        repo = _hook_repo(tmp_path)
        before = repo.git('rev-parse', 'HEAD')
        repo.write_baseline(_report_with(_raise_lines))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.attempt_commit('absorb a rise nobody recorded')

        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert 'lines' in output and 'a.py' in output
        assert '1000' in output and '1005' in output
        assert metrics.RAISE_REMEDY in output
        # It really REFUSED the commit rather than merely printing at it.
        assert repo.git('rev-parse', 'HEAD') == before

    def test_a_covering_ledger_append_commits_on_a_task_branch(
        self, tmp_path: Path
    ) -> None:
        repo = _hook_repo(tmp_path)
        repo.write_baseline(_report_with(_raise_lines))
        repo.write_ledger(
            _ledger_with(
                _record([
                    metrics.Violation.rose('lines', 'a.py', 1000, 1005),
                    metrics.Violation.rose(
                        'total:lines', metrics.CLUSTER_TOTAL_KEY, 1200, 1205
                    ),
                ])
            )
        )
        repo.stage(metrics.BASELINE_RELPATH, metrics.LEDGER_RELPATH)

        result = repo.attempt_commit('authorize a net-additive raise')

        assert result.returncode == 0, result.stdout + result.stderr

    def test_an_unrelated_commit_never_spawns_the_auditor(
        self, tmp_path: Path
    ) -> None:
        repo = _hook_repo(tmp_path)
        (repo.root / 'unrelated.py').write_text('x = 1\n', encoding='utf-8')
        repo.stage('unrelated.py')

        result = repo.attempt_commit('an ordinary change')

        assert result.returncode == 0, result.stdout + result.stderr
        # The bash staged-path filter must short-circuit BEFORE python is
        # spawned, so an ordinary commit pays nothing and is never ambushed.
        assert 'ratchet' not in (result.stdout + result.stderr).lower()

    def test_a_fall_commits_cleanly_on_a_task_branch(self, tmp_path: Path) -> None:
        repo = _hook_repo(tmp_path)
        repo.write_baseline(_report_with(_lower_lines))
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.attempt_commit('lower a measure')

        assert result.returncode == 0, result.stdout + result.stderr


class TestTheHookAuditsTheCommitThatFinishesAConflictedMerge:
    """`git commit` finishing a conflicted merge runs PRE-COMMIT -- pinned, not assumed.

    hooks/pre-commit used to say merge commits run pre-merge-commit instead.
    That holds for a CLEAN `git merge` only: a conflicted merge finished with
    `git commit` -- the merge resolver's path, and esc-3620-11's -- runs this
    hook with MERGE_HEAD set. Driven through a real `git commit` on the copied
    hooks, so no rewording of either hook can fool it.
    """

    def test_taking_upstreams_unrecorded_move_commits_the_merge(
        self, tmp_path: Path
    ) -> None:
        repo = _hook_repo_mid_merge(tmp_path, _with_a_py_lines(1005))
        before = repo.git('rev-parse', 'HEAD')

        result = repo.attempt_commit('merge upstream into the task branch')

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        # PRE-COMMIT ran and reached the audit: no other hook prints this.
        assert 'merge-lane ratchet audit' in output
        assert repo.git('rev-parse', 'HEAD') != before
        assert len(repo.git('rev-list', '--parents', '-n', '1', 'HEAD').split()) == 3

    def test_keeping_heads_stale_baseline_over_an_upstream_lowering_is_refused(
        self, tmp_path: Path
    ) -> None:
        # Byte-identical to HEAD, so a filter listing staged artifacts against
        # HEAD alone never spawns the auditor and the commit lands.
        repo = _hook_repo_mid_merge(tmp_path, _with_a_py_lines(995))
        repo.git('checkout', 'HEAD', '--', metrics.BASELINE_RELPATH)
        before = repo.git('rev-parse', 'HEAD')

        result = repo.attempt_commit('merge upstream, keeping the stale baseline')

        assert result.returncode != 0
        assert 'a.py' in result.stdout + result.stderr
        assert repo.git('rev-parse', 'HEAD') == before


class TestTheWiringIsStructurallyPinned:
    """REFERENTIAL INTEGRITY only. Reachability is pinned behaviourally above.

    hooks/project-checks is a shell script, so it cannot import
    ``BASELINE_RELPATH`` / ``LEDGER_RELPATH`` and carries hardcoded duplicates of
    them. This asserts the duplicates still name the real constants and the real
    auditor (heuristic 11, SPOT) -- a rename that updated the Python and not the
    hook would otherwise leave a filter that silently matches nothing.

    It survives arbitrary rewording of the surrounding script, which is what
    distinguishes it from a shape pin. The sibling that asserted on
    hooks/pre-commit's literal SOURCE TEXT was deleted: its negative assertion
    stayed green against any rewording that KEPT the bug (``[ "$branch" = "main"
    ] || exit 0``), and its positive one went spuriously red on a
    behaviour-preserving rewrite (``"${branch}"``). Reachability is pinned by
    ``TestTheHookActuallyRunsTheGate`` instead, which drives a real `git commit`
    on a task branch through real copied hooks and asserts HEAD did not move --
    and no rewording of pre-commit can fool that.
    """

    def test_project_checks_audits_both_artifacts_through_the_auditor(self) -> None:
        source = (_REPO_ROOT / 'hooks' / 'project-checks').read_text(encoding='utf-8')

        assert metrics.BASELINE_RELPATH in source
        assert metrics.LEDGER_RELPATH in source
        assert metrics.COMMIT_GATE_RELPATH in source


class TestTheCarveOutIsOneStepBack:
    """The carve-out admits the path's IMMEDIATELY-PREVIOUS value, nothing older.

    WITHOUT THESE CASES THE RULE IS UNFALSIFIABLE. The sibling class above only
    ever stages the blob from the revert immediately preceding it, so it passes
    identically under "the previous value" and under "any value this path ever
    carried" -- and the second is a wholesale ratchet reset wearing a carve-out's
    clothes. Measured against this branch before the narrowing: staging the
    baseline blob from 60e954b608 exited 0 with "absorbing 97 measure(s)".

    WHY ONE STEP BACK IS THE RIGHT BOUND. The immediately-previous value is the
    state this repository held one commit ago, so restoring it re-raises nothing
    the tree has not just been running with. Reaching further back does: every
    image in between is a value the ratchet moved through, and returning to one
    behind them re-absorbs every measure they lowered.
    """

    @staticmethod
    def _image(n: int) -> dict:
        """A distinct baseline image, each raising `lines` above the last."""
        return _report_with(lambda report: report['files']['a.py'].__setitem__(
            'lines', 1000 + n
        ))

    @classmethod
    def _history(cls, tmp_path: Path, *images: dict) -> _Repo:
        """A repo whose baseline walked through *images*, oldest first."""
        repo = _Repo.seeded(tmp_path, report=images[0])
        for index, image in enumerate(images[1:], start=1):
            repo.write_baseline(image)
            repo.commit_all(f'move the baseline to image {index}')
        return repo

    def test_reaching_back_past_an_intervening_value_is_refused(
        self, tmp_path: Path
    ) -> None:
        # low -> high -> low (a real revert) -> mid. `high` IS a blob this path
        # carried, so the unbounded rule allows it; the path's previous value is
        # `low`, so the narrowed rule must refuse.
        low, high, mid = self._image(0), self._image(9), self._image(4)
        repo = self._history(tmp_path, low, high, low, mid)
        repo.write_baseline(high)
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr

    def test_the_deep_reach_is_refused_naming_the_measures(
        self, tmp_path: Path
    ) -> None:
        # The measured real-repo exploit in miniature: four distinct images,
        # stage the OLDEST. Every measure the three later images lowered would
        # be re-absorbed, unrecorded.
        images = [self._image(n) for n in (0, 3, 6, 9)]
        repo = self._history(tmp_path, *reversed(images))
        repo.write_baseline(images[3])
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr
        # It refuses through the ordinary unrecorded-raise arm, so the message
        # names what rose and how to authorize it.
        assert 'lines' in result.stderr and 'a.py' in result.stderr
        assert metrics.RAISE_REMEDY in result.stderr

    def test_a_deletion_is_not_a_span_to_reach_across(
        self, tmp_path: Path
    ) -> None:
        # The walk must STOP at the first entry that is not a blob rather than
        # skipping it: a commit that deleted the baseline is not a value the
        # path held, so what precedes it is not the previous value.
        # `high` is staged, so undoing back to it RAISES against HEAD's `mid`
        # -- without that the comparator exits clean and the carve-out is never
        # consulted at all, which would make this case vacuous.
        high, mid = self._image(9), self._image(4)
        repo = _Repo.seeded(tmp_path, report=high)
        repo.git('rm', '--quiet', '--', metrics.BASELINE_RELPATH)
        repo.commit_all('delete the baseline')
        repo.write_baseline(mid)
        repo.commit_all('reintroduce a baseline')

        repo.write_baseline(high)
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 1, result.stdout + result.stderr

    def test_the_immediately_previous_value_is_still_allowed(
        self, tmp_path: Path
    ) -> None:
        """The real 3e7d55ce47 revert shape, and it must keep working.

        Measured on this repo's own history: `git rev-list --full-history` over
        the baseline shows blob a0fb5cc8e0 at b79315a31c, then 0a42c2ee7d
        introduced by 5f577b9613 (task 5668), then a0fb5cc8e0 restored by
        3e7d55ce47. `compare_baseline_files(0a42c2ee7d, a0fb5cc8e0)` measures 4
        raises -- conftest.py lines 1172->1187, prose_lines 752->770, and both
        derived totals -- so the carve-out really does fire for it, and
        narrowing the rule must not take that away.
        """
        # PREVIOUS IS THE HIGHER IMAGE, matching the real revert: 5f577b9613
        # LOWERED the baseline and 3e7d55ce47 put the higher one back, so the
        # restore is a RAISE and the carve-out is what lets it through. Were
        # previous the lower image this would be a fall -- exit 0 without ever
        # consulting the carve-out, and the case would pin nothing.
        previous, current = self._image(9), self._image(0)
        repo = self._history(tmp_path, previous, current)
        repo.write_baseline(previous)
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stderr
        # Reached the carve-out rather than exiting clean as a fall.
        assert 'restores the value this path held' in result.stdout

    def test_the_allowed_line_names_what_came_back(self, tmp_path: Path) -> None:
        # A COUNT IS NOT ENOUGH. A reviewer reading "absorbing 97 measure(s)"
        # learns nothing about what was reabsorbed; the measures and keys are
        # what let them judge whether undoing the last change was right.
        previous, current = self._image(9), self._image(0)
        repo = self._history(tmp_path, previous, current)
        repo.write_baseline(previous)
        repo.stage(metrics.BASELINE_RELPATH)

        result = repo.gate()

        assert result.returncode == 0, result.stderr
        assert 'lines' in result.stdout
        assert 'a.py' in result.stdout
        assert '1009' in result.stdout
