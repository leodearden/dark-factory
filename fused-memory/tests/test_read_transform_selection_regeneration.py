"""The committed selection report is what the stock CLI makes of the cache.

``plans/read-transform-selection-report.{json,md}`` must equal a stock-flag
run of ``scripts/read_transform_selection.py`` over the committed fixtures
and fetch cache, so a metric change that lands without a regeneration fails
here.  ``test_read_transform_selection.py`` pins the same artifact AS DATA;
this file pins that the data is still what the generator produces, and that
every commit the report stamps is on this history.

Merge lane, offline: no network, Qdrant or OPENAI_API_KEY.  Git is needed
only to resolve the stamps.

On failure, re-run the stock command (``REGENERATE_COMMAND``) and commit BOTH
files.  Never hand-edit either one (PRD G6/D10).
"""
from __future__ import annotations

import copy
import functools
import json
import subprocess
import types
from pathlib import Path

from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'read_transform_selection.py'

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
    """A copy minus ``fixture_provenance[*].commit``, which a pre-merge rebase rewrites.

    Each is the fixture's LAST-TOUCHING commit, per
    ``fused-memory/scripts/bake_off_storage_shape.py::fixture_provenance``.
    """
    masked = copy.deepcopy(report)
    for entry in masked.get('fixture_provenance') or []:
        entry.pop('commit', None)
    return masked


_UNTRACKED_REASON = (
    'no commit stamped: the fixture was untracked when the report was '
    'generated, so the stamp cannot be verified'
)


def _stamps_not_on_head(provenance: list[dict]) -> list[dict]:
    """The stamps HEAD's history cannot vouch for, each with its reason."""
    off_head = []
    for entry in provenance:
        commit = entry.get('commit')
        if not commit:
            off_head.append(
                {'path': entry['path'], 'commit': None, 'reason': _UNTRACKED_REASON}
            )
            continue
        result = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', commit, 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip()
            off_head.append({
                'path': entry['path'],
                'commit': commit,
                'reason': (
                    f'not an ancestor of HEAD (git rc={result.returncode}'
                    f'{": " + detail if detail else ""})'
                ),
            })
    return off_head


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


class TestAStampOffThisHistoryIsReported:
    def test_an_unresolvable_or_missing_commit_is_reported_and_head_is_not(self):
        head = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        entries = [
            {'path': 'a', 'commit': '0' * 40},
            {'path': 'b', 'commit': None},
            {'path': 'c', 'commit': head},
        ]

        assert [p['path'] for p in _stamps_not_on_head(entries)] == ['a', 'b']

    def test_each_report_names_its_commit_and_a_reason(self):
        entries = [
            {'path': 'a', 'commit': '0' * 40},
            {'path': 'b', 'commit': None},
        ]

        reported = _stamps_not_on_head(entries)

        assert len(reported) == 2
        for item in reported:
            assert isinstance(item, dict)
            assert {'path', 'commit', 'reason'} <= item.keys()
            assert item['reason']
        assert [item['commit'] for item in reported] == ['0' * 40, None]
