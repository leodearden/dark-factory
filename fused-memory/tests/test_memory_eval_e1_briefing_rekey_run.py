"""Contract tests for the frozen E1 run after the per-template briefing re-key.

`plans/memory-eval-e1-briefing-rekey-run/` holds a verbatim copy of one live,
read-only E1 probe run (task 4856). It was taken after the registry retired the
collapsed ``g7-design-invariants`` topic in favour of one topic per briefing
query template in ``shared/src/shared/briefing_queries.py``. It is the evidence
for that task's user-observable signal: each briefing topic is adjudicated
individually, and the report splits the tripwire by census canonical presence.

The run stamp is discovered rather than written down, so exactly one run may
live here.

**Lane discipline.** File reads, one ``git`` subprocess, and an importlib load
of the probe: no network, no Qdrant, no OPENAI_API_KEY, and no ``integration``
marker, so the merge lane runs it.
"""
from __future__ import annotations

import functools
import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from shared.briefing_queries import QUERY_SPECS
from shared.memory_eval_metrics import (
    load_metric_series,
    parse_metric_series,
    serialize_metric_series,
)

REPO_ROOT = Path(__file__).parents[2]
RUN_ROOT = REPO_ROOT / 'plans' / 'memory-eval-e1-briefing-rekey-run'
SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'memory_eval_retrieval_probe.py'

EVAL_ID = 'e1-retrieval-health'
RETIRED_COLLAPSED_ITEM = 't-g7-design-invariants'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='memory_eval_retrieval_probe')


def _the_run() -> tuple[Path, Path]:
    """``(metrics, report)`` of the single frozen run."""
    found = sorted((RUN_ROOT / EVAL_ID).glob('metrics-*.json'))
    assert len(found) == 1, f'expected exactly one frozen run under {RUN_ROOT}, found {found!r}'
    metrics = found[0]
    stamp = metrics.name.removeprefix('metrics-').removesuffix('.json')
    return metrics, metrics.with_name(f'report-{stamp}.txt')


def _tripwire_items() -> dict[str, bool]:
    """``{item_key: passed}`` of the frozen run's topic-canonical-present metric."""
    series = load_metric_series(_the_run()[0])
    (tripwire,) = [
        m for m in series.metrics if m.metric_id == _mod().METRIC_TOPIC_CANONICAL_PRESENT
    ]
    return {item.item_key: item.passed for item in tripwire.items or []}


def _census_split_lines(report: str) -> list[str]:
    """The report's census-split block: its header line to the next blank line.

    The header is found by the section's own title, the one piece of the block
    that names it; every other assertion below is on slugs, which are data.
    """
    lines = report.splitlines()
    (start,) = [
        i for i, line in enumerate(lines)
        if line.startswith('tripwire split by census canonical presence')
    ]
    end = next((i for i in range(start + 1, len(lines)) if not lines[i].strip()), len(lines))
    return lines[start:end]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git at the repo root, or skip where git cannot answer at all."""
    if shutil.which('git') is None:
        pytest.skip('git is not available; cannot check tracked-ness')
    inside = subprocess.run(
        ['git', 'rev-parse', '--is-inside-work-tree'],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if inside.returncode != 0 or inside.stdout.strip() != 'true':
        pytest.skip('not a git working tree; cannot check tracked-ness')
    return subprocess.run(
        ['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )


class TestTheRunIsDurablyCommitted:
    def test_the_metrics_and_report_are_present(self):
        metrics, report = _the_run()

        assert metrics.is_file()
        assert report.is_file()

    def test_the_metrics_and_report_are_tracked_by_git(self):
        for path in _the_run():
            result = _git('ls-files', '--error-unmatch', '--', str(path))
            assert result.returncode == 0, (
                f'{path.relative_to(REPO_ROOT)} is NOT tracked by git: '
                f'{result.stderr.strip()}'
            )

    def test_the_readme_is_present(self):
        assert (RUN_ROOT / 'README.md').is_file()


class TestTheRunIsAValidM1Artifact:
    def test_the_metrics_artifact_validates_as_this_eval(self):
        assert load_metric_series(_the_run()[0]).eval_id == EVAL_ID

    def test_reemitting_the_artifact_is_byte_identical(self):
        """The anti-hand-edit guard: the copy is verbatim, not reconciled."""
        text = _the_run()[0].read_text(encoding='utf-8')

        assert serialize_metric_series(parse_metric_series(json.loads(text))) == text


class TestEachBriefingTopicIsAdjudicatedIndividually:
    def test_every_briefing_template_is_its_own_tripwire_item(self):
        items = _tripwire_items()

        for spec in QUERY_SPECS:
            assert f't-{spec.slug}' in items

    def test_the_collapsed_topic_is_gone(self):
        assert RETIRED_COLLAPSED_ITEM not in _tripwire_items()

    def test_the_report_names_every_briefing_topic(self):
        report = _the_run()[1].read_text(encoding='utf-8')

        for spec in QUERY_SPECS:
            assert spec.slug in report

    def test_the_census_split_names_every_failing_item(self):
        section = _census_split_lines(_the_run()[1].read_text(encoding='utf-8'))
        prefix = _mod().TRIPWIRE_ITEM_PREFIX

        for item_key, passed in _tripwire_items().items():
            if not passed:
                assert f'    - {item_key.removeprefix(prefix)}' in section


class TestTheRunCannotBeMistakenForALiveRun:
    def test_the_run_root_is_disjoint_from_the_live_artifact_root(self):
        """A committed artifact under the live root would make
        :func:`is_initial_run` report every fresh clone as already run."""
        frozen = RUN_ROOT.resolve()
        live = _mod().DEFAULT_OUT_ROOT.resolve()

        assert frozen != live
        assert not frozen.is_relative_to(live)
        assert not live.is_relative_to(frozen)
