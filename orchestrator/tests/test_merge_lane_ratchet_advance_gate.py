"""The ratchet gate on main's MOVE: one landing audited as one change.

``scripts/check_staged_ratchet_raise.py`` audits a COMMIT, so it never sees a
raise that reaches main without one: a ``git rebase`` replay, a clean merge, or
a hook-less commit. Every route to main still ends in a move of
``refs/heads/main``, and ``hooks/reference-transaction`` hands that move to
``scripts/check_ratchet_advance.py`` as OLD and NEW. The auditor reads the move
as a single change, merge-base(OLD, NEW) -> NEW, under the commit gate's own
rules: a raise is covered by the sum of the entry files the move adds, and the
ledger is append-only.

The commit-time half is pinned in ``test_merge_lane_ratchet_commit_gate.py``.
This suite lives under ``orchestrator/tests`` for the same reason that one
does: each case builds real history in a throwaway repo, which must go through
``_orch_helpers``' esc-3072-3 isolation (``RatchetRepo`` applies it).

HOW A LANDING IS BUILT. Hooks are off. A task branch commits the change and is
merged ``--no-ff`` onto a side ref cut from main. OLD is main's sha, NEW is the
landing's, and main itself never moves, so the auditor is driven exactly as the
hook drives it. EVERY CASE PINS THE EXIT CODE: 0 clean, 1 a policy refusal,
2 the auditor could not do its job.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from _merge_lane_ratchet_fixtures import (
    RatchetRepo,
    a_py_cognitive_raise,
    metrics,
    record,
    synthetic_report,
)

_REPO_ROOT = Path(__file__).parents[2]

_TASK_BRANCH = 'task/5722'
_LANDING = 'landing'


def _a_py_cognitive(value: int) -> dict:
    """The seed report with a.py's `cognitive` at *value* (seed: 120)."""
    report = synthetic_report()
    report['files']['a.py']['cognitive'] = value
    return report


def _advance(repo: RatchetRepo, old: str, new: str) -> subprocess.CompletedProcess[str]:
    """Drive the auditor through its real entry point, as the hook does."""
    return subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / metrics.ADVANCE_GATE_RELPATH),
            '--root',
            str(repo.root),
            old,
            new,
        ],
        capture_output=True,
        text=True,
        env=repo.env,
        check=False,
    )


def _fork(repo: RatchetRepo, branch: str = _TASK_BRANCH, start: str = 'main') -> None:
    repo.git('switch', '--quiet', '-c', branch, start)


def _commit(
    repo: RatchetRepo,
    message: str,
    *,
    baseline: dict | None = None,
    entries: tuple[dict, ...] = (),
) -> str:
    """Commit *baseline* and *entries* on HEAD's branch, hooks off; the new sha."""
    if baseline is not None:
        repo.write_baseline(baseline)
    for entry in entries:
        repo.write_entry(entry)
    if baseline is None and not entries:
        repo.write_file('unrelated.py', f'# {message}\n')
    return repo.commit_all(message)


def _merge(repo: RatchetRepo, branch: str, *, composed: dict | None = None) -> None:
    """Merge *branch* --no-ff into HEAD; with *composed*, record that baseline instead."""
    message = f'Merge {branch} into main'
    if composed is None:
        repo.git('merge', '--quiet', '--no-ff', '--no-verify', '-m', message, branch)
        return
    # Stops whether git merged the baseline clean or conflicted on it.
    repo.git('merge', '--quiet', '--no-ff', '--no-commit', branch, check=False)
    assert repo.git('rev-parse', '-q', '--verify', 'MERGE_HEAD', check=False), branch
    repo.write_baseline(composed)
    repo.git('add', '-A')
    repo.git('commit', '--quiet', '--no-verify', '-m', message)


def _land(
    repo: RatchetRepo, *branches: str, composed: dict | None = None
) -> tuple[str, str]:
    """Land *branches*, in order, on a side ref cut from main: (OLD, NEW).

    *composed* is the baseline the LAST merge records, for a train whose members
    both moved it.
    """
    old = repo.git('rev-parse', 'main')
    _fork(repo, _LANDING)
    for branch in branches[:-1]:
        _merge(repo, branch)
    _merge(repo, branches[-1], composed=composed)
    return old, repo.git('rev-parse', 'HEAD')


def _how_to_lines(stderr: str, base: str) -> list[str]:
    """The advance-specific how-to lines: they restore *base*'s baseline on the branch."""
    restore = f'git checkout {base} -- {metrics.BASELINE_RELPATH}'
    return [
        line
        for line in stderr.splitlines()
        if line.startswith('ratchet advance gate:') and restore in line
    ]


class TestTheAdvanceAuditor:
    """main's move, merge-base(OLD, NEW) -> NEW, under the commit gate's rules."""

    def test_an_advance_touching_no_artifact_is_clean_and_silent(
        self, tmp_path: Path
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        _commit(repo, 'an ordinary change')
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout == '' and result.stderr == ''

    def test_an_unrecorded_raise_is_refused_with_the_landing_how_to(
        self, tmp_path: Path
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        _commit(repo, 'raise a.py with no entry', baseline=_a_py_cognitive(125))
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'cognitive rose 120 -> 125 for a.py' in result.stderr
        assert metrics.RAISE_REMEDY in result.stderr
        how_to = _how_to_lines(result.stderr, old)
        assert how_to, result.stderr
        assert '--write-baseline' in how_to[0] and '--authorize-raise' in how_to[0]

    def test_a_covering_entry_the_branch_adds_allows_the_raise(
        self, tmp_path: Path
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        _commit(
            repo,
            'authorize a net-additive raise',
            baseline=_a_py_cognitive(125),
            entries=(a_py_cognitive_raise(120, 125, '5722'),),
        )
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 0, result.stdout + result.stderr

    def test_entries_from_successive_commits_cover_their_sum(
        self, tmp_path: Path
    ) -> None:
        # The Details' coverage rule: the landing's delta reads 120 -> 125, which
        # neither commit's entry reaches alone.
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        _commit(
            repo,
            'first half',
            baseline=_a_py_cognitive(123),
            entries=(a_py_cognitive_raise(120, 123, '5722'),),
        )
        _commit(
            repo,
            'second half',
            baseline=_a_py_cognitive(125),
            entries=(a_py_cognitive_raise(123, 125, '5722'),),
        )
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_stale_entry_claiming_less_than_the_rise_is_refused(
        self, tmp_path: Path
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        _commit(
            repo,
            'record less than the raise',
            baseline=_a_py_cognitive(125),
            entries=(a_py_cognitive_raise(120, 123, '5722'),),
        )
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'cognitive rose 120 -> 125 for a.py' in result.stderr

    def test_an_entry_old_already_holds_is_not_a_standing_permission(
        self, tmp_path: Path
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _commit(repo, 'record a raise main never took',
                entries=(a_py_cognitive_raise(120, 125, '5722'),))
        _fork(repo)
        _commit(repo, 'take the raise on the old entry', baseline=_a_py_cognitive(125))
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'cognitive rose 120 -> 125 for a.py' in result.stderr

    def test_a_train_is_covered_by_every_members_entry(self, tmp_path: Path) -> None:
        # Two members each raise a.py by 5 against main; the second merge records
        # the composed baseline (A + B - O), as the ratchet-baseline driver does.
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo, 'task/5722')
        _commit(
            repo,
            'first member',
            baseline=_a_py_cognitive(125),
            entries=(a_py_cognitive_raise(120, 125, '5722'),),
        )
        _fork(repo, 'task/5723')
        _commit(
            repo,
            'second member',
            baseline=_a_py_cognitive(125),
            entries=(a_py_cognitive_raise(120, 125, '5723'),),
        )
        old, new = _land(repo, 'task/5722', 'task/5723', composed=_a_py_cognitive(130))

        result = _advance(repo, old, new)

        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_fall_needs_no_ceremony(self, tmp_path: Path) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        _commit(repo, 'lower a.py', baseline=_a_py_cognitive(115))
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 0, result.stdout + result.stderr

    def test_an_advance_deleting_the_baseline_is_refused(self, tmp_path: Path) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        repo.git('rm', '--quiet', '--', metrics.BASELINE_RELPATH)
        repo.commit_all('delete the baseline')
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'delete' in result.stderr.lower()
        assert metrics.BASELINE_RELPATH in result.stderr

    @pytest.mark.parametrize('rewrite', ['modify', 'delete'])
    def test_rewriting_a_recorded_entry_is_a_policy_refusal(
        self, tmp_path: Path, rewrite: str
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        history = repo.write_entry(
            record([metrics.Violation.rose('cognitive', 'a.py', 110, 120)], '5485')
        )
        repo.commit_all('record a historical raise')
        _fork(repo)
        if rewrite == 'delete':
            repo.git('rm', '--quiet', '--', history)
        else:
            repo.write_file(history, '{"reason": "actually it was a refactor"}\n')
        repo.commit_all(f'{rewrite} a recorded entry')
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'append-only' in result.stderr

    def test_burying_an_entry_in_the_retired_ledger_is_refused(
        self, tmp_path: Path
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        repo.write_file(metrics.LEGACY_LEDGER_RELPATH, '{"raises": []}\n')
        repo.commit_all('stage the retired ledger back')
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 1, result.stdout + result.stderr
        assert metrics.LEGACY_LEDGER_RELPATH in result.stderr

    def test_an_entry_not_named_for_its_content_is_an_instrument_failure(
        self, tmp_path: Path
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        repo.write_file(
            f'{metrics.LEDGER_DIR_RELPATH}/5722-guessed.json',
            metrics.render_entry(record([], '5722')),
        )
        repo.commit_all('hand-write an entry under a guessed name')
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 2, result.stdout + result.stderr
        assert any(
            '5722-guessed.json' in line and 'not named for its content' in line
            for line in result.stderr.splitlines()
        ), result.stderr
        assert 'Traceback' not in result.stderr

    def test_a_baseline_old_lacks_is_an_announced_first_write(
        self, tmp_path: Path
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        repo.git('rm', '--quiet', '--', metrics.BASELINE_RELPATH)
        repo.commit_all('main without a baseline')
        _fork(repo)
        _commit(repo, 'introduce the baseline', baseline=_a_py_cognitive(125))
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 0, result.stdout + result.stderr
        assert 'first write' in result.stdout + result.stderr

    def test_a_rewind_across_a_baseline_change_passes(self, tmp_path: Path) -> None:
        # recover_red_main's backward move. Here it RAISES a.py back to 120 over
        # OLD's 115, and still passes: main held that value before.
        repo = RatchetRepo.seeded(tmp_path)
        new = repo.git('rev-parse', 'HEAD')
        old = _commit(repo, 'lower a.py on main', baseline=_a_py_cognitive(115))

        result = _advance(repo, old, new)

        assert result.returncode == 0, result.stdout + result.stderr
        assert 'nothing reaches main' in result.stdout + result.stderr

    @pytest.mark.parametrize('covered', [False, True], ids=['uncovered', 'covered'])
    def test_a_sideways_move_is_audited_over_the_merge_base(
        self, tmp_path: Path, covered: bool
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        base = repo.git('rev-parse', 'HEAD')
        _fork(repo)
        entries = (a_py_cognitive_raise(120, 125, '5722'),) if covered else ()
        new = _commit(repo, 'raise a.py', baseline=_a_py_cognitive(125), entries=entries)
        repo.git('switch', '--quiet', 'main')
        old = _commit(repo, 'main moves on')

        result = _advance(repo, old, new)

        if covered:
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            assert result.returncode == 1, result.stdout + result.stderr
            assert _how_to_lines(result.stderr, base), result.stderr
        assert base in result.stdout + result.stderr

    def test_a_later_commits_entry_covers_an_earlier_commits_raise(
        self, tmp_path: Path
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        _commit(repo, 'raise a.py', baseline=_a_py_cognitive(125))
        _commit(repo, 'record the raise', entries=(a_py_cognitive_raise(120, 125, '5722'),))
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_malformed_baseline_at_new_names_its_cause(self, tmp_path: Path) -> None:
        repo = RatchetRepo.seeded(tmp_path)
        _fork(repo)
        repo.write_baseline_text('{"files": {')
        repo.commit_all('truncate the baseline')
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 2, result.stdout + result.stderr
        assert any(
            line.startswith('ratchet advance gate:') and 'not valid JSON' in line
            for line in result.stderr.splitlines()
        ), result.stderr
        assert 'Traceback' not in result.stderr

    def test_an_old_naming_no_commit_is_an_instrument_failure(
        self, tmp_path: Path
    ) -> None:
        repo = RatchetRepo.seeded(tmp_path)

        result = _advance(repo, 'refs/heads/no-such-branch', repo.git('rev-parse', 'HEAD'))

        assert result.returncode == 2, result.stdout + result.stderr
        assert 'no-such-branch' in result.stderr
        assert 'Traceback' not in result.stderr


class TestTheAdvanceHonoursTheRestoreCarveOut:
    """Undoing main's LAST baseline change is not a new raise, at a landing too.

    At a move of main, "the state this repository held one commit ago" is
    main's history, so the carve-out reads it from the merge base: NEW's
    baseline may be the image the path held immediately before the merge base's
    last baseline change, and nothing older.
    """

    @staticmethod
    def _main_with_a_lowering(tmp_path: Path) -> tuple[RatchetRepo, str]:
        """main's a.py `cognitive` walks 125 -> 129 -> 120; the commit that recorded 129.

        The last change LOWERS, as the real revert 3e7d55ce47 did, so restoring
        the previous image is a raise and the carve-out is what decides it.
        """
        repo = RatchetRepo.seeded(tmp_path, report=_a_py_cognitive(125))
        previous = _commit(repo, 'raise a.py on main', baseline=_a_py_cognitive(129))
        _commit(repo, 'lower a.py on main', baseline=_a_py_cognitive(120))
        return repo, previous

    def test_restoring_mains_previous_image_is_allowed_and_named(
        self, tmp_path: Path
    ) -> None:
        repo, previous = self._main_with_a_lowering(tmp_path)
        _fork(repo)
        _commit(repo, "undo main's last baseline change", baseline=_a_py_cognitive(129))
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 0, result.stdout + result.stderr
        assert previous in result.stdout
        assert 'cognitive rose 120 -> 129 for a.py' in result.stdout

    def test_restoring_an_image_two_changes_back_is_refused(self, tmp_path: Path) -> None:
        repo, _previous = self._main_with_a_lowering(tmp_path)
        _fork(repo)
        _commit(repo, 'reach past the previous image', baseline=_a_py_cognitive(125))
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 1, result.stdout + result.stderr

    def test_a_blob_one_byte_off_the_previous_image_is_refused(
        self, tmp_path: Path
    ) -> None:
        repo, _previous = self._main_with_a_lowering(tmp_path)
        _fork(repo)
        repo.write_baseline_text(metrics.render_baseline(_a_py_cognitive(129)) + '\n')
        repo.commit_all('restore the previous measures under other bytes')
        old, new = _land(repo, _TASK_BRANCH)

        result = _advance(repo, old, new)

        assert result.returncode == 1, result.stdout + result.stderr

    def test_a_restore_main_has_moved_past_is_bounded_like_a_raise(
        self, tmp_path: Path
    ) -> None:
        # The branch's commit restores the previous image while the lowering is
        # main's last change, which the commit gate allows. main then lowers b.py,
        # so the landing composes a.py 129 with b.py 25: an image main never held.
        repo, _previous = self._main_with_a_lowering(tmp_path)
        _fork(repo)
        _commit(repo, "undo main's last baseline change", baseline=_a_py_cognitive(129))
        repo.git('switch', '--quiet', 'main')
        moved_on = _a_py_cognitive(120)
        moved_on['files']['b.py']['cognitive'] = 25
        _commit(repo, 'lower b.py on main', baseline=moved_on)
        composed = _a_py_cognitive(129)
        composed['files']['b.py']['cognitive'] = 25
        old, new = _land(repo, _TASK_BRANCH, composed=composed)

        result = _advance(repo, old, new)

        assert result.returncode == 1, result.stdout + result.stderr
        assert 'cognitive rose 120 -> 129 for a.py' in result.stderr
        assert _how_to_lines(result.stderr, old), result.stderr
