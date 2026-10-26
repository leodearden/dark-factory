"""The committed selection report is what the stock CLI makes of the cache.

``plans/read-transform-selection-report.{json,md}`` must equal a stock-flag
run of ``scripts/read_transform_selection.py`` over the committed fixtures
and fetch cache, so a metric change that lands without a regeneration fails
here.  ``test_read_transform_selection.py`` pins the same artifact AS DATA;
this file pins that the data is still what the generator produces, and that
every commit the report stamps is on this history.

The two guards split by cause: the comparison ignores commit stamps, so it
fails only when content moved; the ancestry guard fails only when a stamp is
off HEAD's history.  Masking does not make a rebase harmless.  A branch that
commits a change to a stamped fixture (the fetch cache, say) and then
regenerates stamps its own commit; a rebase rewrites that commit, and the
ancestry guard fails.  The merge lane rebases
(``orchestrator/src/orchestrator/git_ops.py::GitOps.advance_main``), so such
a regeneration holds only once the branch sits on current main.

Merge lane, offline: no network, Qdrant or OPENAI_API_KEY.  Git is needed
only to resolve the stamps.

On failure, re-run the stock command (``REGENERATE_COMMAND``) and commit BOTH
files.  Never hand-edit either one (PRD G6/D10).
"""
from __future__ import annotations

import contextlib
import copy
import functools
import io
import json
import subprocess
import tempfile
import types
from pathlib import Path

import pytest
from _fm_helpers import _init_git_repo, load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'read_transform_selection.py'
THIS_CHECKOUT = Path(__file__).resolve().parent

REGENERATE_COMMAND = (
    'uv run --project fused-memory python '
    'fused-memory/scripts/read_transform_selection.py'
)


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='read_transform_selection')


@functools.cache
def _committed_report() -> dict:
    return json.loads(_mod().DEFAULT_SELECTION_JSON.read_text(encoding='utf-8'))


@functools.cache
def _committed_markdown() -> str:
    return _mod().DEFAULT_SELECTION_MD.read_text(encoding='utf-8')


def _without_commit_stamps(report: dict) -> dict:
    """A copy minus ``fixture_provenance[*].commit``, so a comparison sees content only.

    Each stamp is the fixture's last-touching commit when the report was
    generated (``fused-memory/scripts/bake_off_storage_shape.py::fixture_provenance``).
    Masking tolerates a stamp that is still on this history but is no longer
    the fixture's latest commit, because a later commit touched the fixture
    without moving a metric.  A stamp that a rebase rewrote also passes here,
    and ``TestEveryCommittedFixtureStampIsOnThisHistory`` fails it.
    """
    masked = copy.deepcopy(report)
    for entry in masked.get('fixture_provenance') or []:
        entry.pop('commit', None)
    return masked


_UNTRACKED_REASON = (
    'no commit stamped: the fixture was untracked when the report was '
    'generated, so the stamp cannot be verified'
)

_OFF_HISTORY_REASON = (
    'git has this commit but HEAD does not descend from it; usually a commit '
    "a rebase rewrote (task 4004's report named two), so regenerate"
)


def _unresolvable_reason(returncode: int, stderr: str) -> str:
    return (
        f'git cannot resolve this commit here (rc={returncode}: '
        f'{stderr.strip() or "no stderr"}); a shallow clone lacks it (deepen '
        'and re-run), otherwise it is a rewritten commit this clone never '
        'had, so regenerate'
    )


def _stamps_not_on_head(
    provenance: list[dict], *, repo: Path = THIS_CHECKOUT,
) -> list[dict]:
    """The stamps ``repo``'s HEAD history cannot vouch for, each with a kind and reason.

    ``git merge-base --is-ancestor`` exits 1 only for a commit git has but HEAD
    does not descend from (``off_history``); any other nonzero exit means git
    could not resolve the commit at all (``unresolvable``).
    """
    off_head = []
    for entry in provenance:
        commit = entry.get('commit')
        if not commit:
            off_head.append({
                'path': entry['path'], 'commit': None,
                'kind': 'untracked', 'reason': _UNTRACKED_REASON,
            })
            continue
        result = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', commit, 'HEAD'],
            cwd=repo, capture_output=True, text=True, check=False,
        )
        if result.returncode == 1:
            off_head.append({
                'path': entry['path'], 'commit': commit,
                'kind': 'off_history', 'reason': _OFF_HISTORY_REASON,
            })
        elif result.returncode != 0:
            off_head.append({
                'path': entry['path'], 'commit': commit, 'kind': 'unresolvable',
                'reason': _unresolvable_reason(result.returncode, result.stderr),
            })
    return off_head


_STALE_REMEDY = (
    f'the committed report is stale: regenerate with `{REGENERATE_COMMAND}` '
    'and commit BOTH plans/read-transform-selection-report.{json,md}; never '
    'hand-edit (PRD G6/D10), even if the recommendation moves'
)


@functools.cache
def _regenerated_pair() -> tuple[dict, str]:
    """The stock CLI's output, with only its two destinations redirected."""
    with tempfile.TemporaryDirectory(prefix='read-transform-regen-') as tmp:
        out = Path(tmp)
        json_out, md_out = out / 'report.json', out / 'report.md'
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = _mod().main(['--json-out', str(json_out), '--md-out', str(md_out)])
        assert code == 0, (
            f'`{REGENERATE_COMMAND}` exited {code}:\n{stderr.getvalue()}'
        )
        return (
            json.loads(json_out.read_text(encoding='utf-8')),
            md_out.read_text(encoding='utf-8'),
        )


def _differing_leaves(committed, regenerated, path: str = '') -> list[str]:
    """One line per differing leaf, for a failure message only."""
    if isinstance(committed, dict) and isinstance(regenerated, dict):
        lines = []
        for key in sorted(committed.keys() | regenerated.keys()):
            where = f'{path}.{key}' if path else str(key)
            if key not in regenerated:
                lines.append(f'{where}: only in committed')
            elif key not in committed:
                lines.append(f'{where}: only in regenerated')
            else:
                lines += _differing_leaves(committed[key], regenerated[key], where)
        return lines
    if (
        isinstance(committed, list) and isinstance(regenerated, list)
        and len(committed) == len(regenerated)
    ):
        lines = []
        for index, (a, b) in enumerate(zip(committed, regenerated, strict=True)):
            lines += _differing_leaves(a, b, f'{path}[{index}]')
        return lines
    if committed != regenerated:
        return [f'{path}: committed {committed!r} != regenerated {regenerated!r}']
    return []


class TestTheCommitStampMaskHidesOnlyTheStamps:
    def test_a_restamped_report_compares_equal(self):
        committed = _committed_report()
        restamped = copy.deepcopy(committed)
        for entry in restamped['fixture_provenance']:
            entry['commit'] = 'f' * 40

        assert _without_commit_stamps(restamped) == _without_commit_stamps(committed)

    def test_a_moved_metric_still_compares_unequal(self):
        committed = _committed_report()
        moved = copy.deepcopy(committed)
        moved['arms'][_mod().ARM_KEYS[0]]['e2']['tokens_per_query'] += 1

        assert _without_commit_stamps(moved) != _without_commit_stamps(committed)

    def test_a_changed_fixture_path_still_compares_unequal(self):
        committed = _committed_report()
        moved = copy.deepcopy(committed)
        moved['fixture_provenance'][0]['path'] += '.renamed'

        assert _without_commit_stamps(moved) != _without_commit_stamps(committed)

    def test_masking_leaves_its_input_untouched(self):
        _without_commit_stamps(_committed_report())

        assert _committed_report()['fixture_provenance'][0]['commit']


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], capture_output=True, text=True,
        check=True,
    ).stdout.strip()


def _commit_files(repo: Path, message: str, **files: str) -> None:
    for name, text in files.items():
        (repo / name).write_text(text)
    _git(repo, 'add', '--', *files)
    _git(repo, 'commit', '-q', '-m', message)


class TestABlobHeadDoesNotHoldIsReported:
    def test_a_blob_head_holds_is_not_reported_and_every_other_kind_is(
        self, tmp_path,
    ):
        _init_git_repo(tmp_path)
        _commit_files(
            tmp_path, 'fixtures',
            **{'held.json': '{"held": 1}\n', 'differs.json': '{"differs": 1}\n'},
        )
        entries = [
            {'path': 'held.json', 'blob': _git(tmp_path, 'rev-parse', 'HEAD:held.json')},
            {'path': 'differs.json', 'blob': 'f' * 40},
            {'path': 'absent.json', 'blob': 'e' * 40},
            {'path': 'unstamped.json', 'blob': None},
        ]

        reported = _blobs_head_does_not_hold(entries, repo=tmp_path)

        assert [(item['path'], item['kind']) for item in reported] == [
            ('differs.json', 'differs'),
            ('absent.json', 'absent_at_head'),
            ('unstamped.json', 'unstamped'),
        ]
        for item in reported:
            assert {'path', 'blob', 'kind', 'reason'} <= item.keys()
            assert item['reason']
        assert len({item['reason'] for item in reported}) == 3

    def test_a_stamp_survives_a_rebase_that_rewrites_its_commit(self, tmp_path):
        _init_git_repo(tmp_path)
        _git(tmp_path, 'checkout', '-q', '-b', 'task')
        _commit_files(tmp_path, 'stamped fixture', **{'fixture.json': '{"v": 2}\n'})
        stamp = {
            'path': 'fixture.json',
            'commit': _git(tmp_path, 'rev-parse', 'HEAD'),
            'blob': _git(tmp_path, 'rev-parse', 'HEAD:fixture.json'),
        }
        _git(tmp_path, 'checkout', '-q', 'main')
        _commit_files(tmp_path, 'unrelated', **{'unrelated.txt': 'moved main\n'})
        _git(tmp_path, 'checkout', '-q', 'task')
        _git(tmp_path, 'rebase', '-q', 'main')

        rewritten = subprocess.run(
            ['git', '-C', str(tmp_path), 'merge-base', '--is-ancestor',
             stamp['commit'], 'HEAD'],
            capture_output=True, text=True, check=False,
        )
        assert rewritten.returncode == 1
        assert _blobs_head_does_not_hold([stamp], repo=tmp_path) == []


@pytest.mark.xdist_group('read_transform_regeneration')
class TestTheCommittedReportIsWhatTheCacheProduces:
    """Grouped onto one xdist worker so the regeneration is paid once."""

    def test_the_json_is_a_stock_regeneration_except_for_commit_stamps(self):
        regenerated = _without_commit_stamps(_regenerated_pair()[0])
        committed = _without_commit_stamps(_committed_report())

        assert regenerated == committed, '\n'.join([
            *_differing_leaves(committed, regenerated)[:25],
            _STALE_REMEDY,
        ])

    def test_the_markdown_is_a_stock_regeneration_byte_for_byte(self):
        stamped = [
            entry['commit'] for entry in _committed_report()['fixture_provenance']
            if entry.get('commit')
        ]
        printed = [sha for sha in stamped if sha in _committed_markdown()]
        assert not printed, (
            f'the renderer now prints a commit stamp ({printed}); mask it here '
            'as the JSON test does, or this comparison fails on every rebase'
        )

        assert _regenerated_pair()[1] == _committed_markdown(), _STALE_REMEDY


@pytest.mark.reads_git_history
class TestEveryCommittedFixtureStampIsOnThisHistory:
    def test_every_stamp_is_an_ancestor_of_head(self):
        stamps = _committed_report()['fixture_provenance']
        assert stamps

        off = _stamps_not_on_head(stamps)

        assert off == [], '\n'.join([
            *(f'{item["path"]}: {item["commit"]} — {item["reason"]}' for item in off),
            f'to regenerate, run `{REGENERATE_COMMAND}` on the CURRENT history '
            'and commit both files — do not hand-edit the sha',
        ])
