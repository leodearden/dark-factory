"""Contract tests for the committed first-live-run E1 exemplar.

`plans/memory-eval-e1-first-live-run/` holds a byte-for-byte copy of the
FIRST live run of the E1 retrieval-health probe (task 3208, run_stamp
``20260805T093831Z``) — the only run that carries the D1 initial-state
snapshot, and therefore the only evidence that 3208's user_observable_signal
("a live read-only run produces the M1 artifact + initial-state report") was
ever actually met.

**Why these tests exist.** The probe writes to ``DEFAULT_OUT_ROOT``
(``fused-memory/data/memory-evals/``), which is gitignored
(``fused-memory/.gitignore:9``). The 2026-08-05 run therefore existed on
exactly one host, on a path no fresh clone could reach — which is what left
three PRD provenance citations pointing at a file their readers could not
open. An existence check would have passed the entire time the bug was live;
only a check of the git INDEX distinguishes "the file is here" from "the file
survives a clone". That is the assertion in
:class:`TestTheExemplarIsDurablyCommitted`, and it is the one that encodes
this defect.

Everything else here is anti-rot: schema validity, canonical byte-identity,
and path/payload agreement, so the committed bytes can never silently drift
from what ``write_metric_series`` emits. The pattern is lifted from
``shared/tests/test_memory_eval_boundary.py``'s
``TestCommittedExemplarsAreProducible``.

:class:`TestTheExemplarCannotBeMistakenForALiveRun` guards the other half:
the exemplar must stay provably disjoint from the probe's live artifact root,
because ``is_initial_run()`` globs that root and an exemplar living inside it
would burn the one-shot D1 snapshot for the next genuine first run. A README
saying "don't put this in the live root" is unenforceable; that test is.

**Lane discipline.** File reads, one ``git`` subprocess, and an importlib load
of the probe: no network, no Qdrant, no OPENAI_API_KEY. Deliberately carries
NO ``integration`` marker so it runs under the merge lane's default
``-m 'not integration'`` selection — a durability guard that the merge lane
deselects guards nothing.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest
from shared.memory_eval_metrics import (
    parse_metric_series,
    serialize_metric_series,
    validate_metric_series,
)

REPO_ROOT = Path(__file__).parents[2]
EXEMPLAR_ROOT = REPO_ROOT / 'plans' / 'memory-eval-e1-first-live-run'
SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'memory_eval_retrieval_probe.py'

EVAL_ID = 'e1-retrieval-health'
RUN_STAMP = '20260805T093831Z'

METRICS_PATH = EXEMPLAR_ROOT / EVAL_ID / f'metrics-{RUN_STAMP}.json'
REPORT_PATH = EXEMPLAR_ROOT / EVAL_ID / f'report-{RUN_STAMP}.txt'

ARTIFACTS = (METRICS_PATH, REPORT_PATH)
_IDS = [p.name for p in ARTIFACTS]


def _load_module() -> types.ModuleType:
    """Load memory_eval_retrieval_probe.py from its file path.

    Same loader as ``test_memory_eval_retrieval_probe.py`` (and
    ``test_calibrate_write_triage.py``, ``test_audit_duplicate_memories.py``):
    ``scripts/`` has no ``__init__.py``, so the probe is not importable as a
    package. Registered in ``sys.modules`` under its own name because
    ``@dataclass`` and other reflection-based decorators look the module up
    there.
    """
    import sys  # noqa: PLC0415

    mod_name = 'memory_eval_retrieval_probe'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


@functools.cache
def _mod() -> types.ModuleType:
    """Lazy, so the pure file-contract tests above never load the probe."""
    return _load_module()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git at the repo root, or skip if git is unavailable.

    ``check=False`` throughout: ``git check-ignore``'s exit code IS the
    answer being asserted on, so a non-zero exit must reach the test rather
    than raise. Skipping (not failing) on a missing binary keeps this from
    becoming an environment-dependent flake in a container without git.
    """
    if shutil.which('git') is None:
        pytest.skip('git is not available; cannot check tracked-ness')
    return subprocess.run(
        ['git', *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class TestTheExemplarIsDurablyCommitted:
    """The assertion that encodes the bug: tracked, not merely present.

    The original artifact was present on disk for its whole lifetime. What
    made it useless — and what stalled esc-3253-1 — was that it was untracked
    and unreachable from a fresh clone or a second host. These tests check the
    git index, not the filesystem, because that is the only thing that tells
    the two states apart.
    """

    @pytest.mark.parametrize('path', ARTIFACTS, ids=_IDS)
    def test_the_artifact_is_present(self, path):
        # Guards the tests below from passing vacuously against a path that
        # was renamed or never copied.
        assert path.is_file(), f'{path} is missing'

    @pytest.mark.parametrize('path', ARTIFACTS, ids=_IDS)
    def test_the_artifact_is_tracked_by_git(self, path):
        """Present-on-disk is not enough; it must survive a clone."""
        result = _git('ls-files', '--error-unmatch', '--', str(path))
        assert result.returncode == 0, (
            f'{path.relative_to(REPO_ROOT)} is NOT tracked by git — it would '
            f'not survive a fresh clone or `git clean -xdf`, which is the '
            f'exact defect this exemplar exists to close. git said: '
            f'{result.stderr.strip()}'
        )
        assert result.stdout.strip(), 'git ls-files named no file'

    @pytest.mark.parametrize('path', ARTIFACTS, ids=_IDS)
    def test_the_artifact_is_not_gitignored(self, path):
        """No ignore rule may reach the exemplar.

        Asserts exit code 1 exactly — git's documented "no path is ignored" —
        rather than merely non-zero, so a 128 (fatal invocation error) cannot
        masquerade as a clean answer.
        """
        result = _git('check-ignore', '--', str(path))
        assert result.returncode == 1, (
            f'`git check-ignore` did not report {path.relative_to(REPO_ROOT)} '
            f'as unignored (exit {result.returncode}). matched rule: '
            f'{result.stdout.strip()!r} stderr: {result.stderr.strip()!r}'
        )
        assert not result.stdout.strip()


class TestTheExemplarIsAValidM1Artifact:
    """Anti-rot: the committed bytes stay what the writer would emit."""

    def test_the_metrics_artifact_validates(self):
        series = parse_metric_series(json.loads(METRICS_PATH.read_text(encoding='utf-8')))
        validate_metric_series(series)

    def test_the_artifact_names_its_own_eval_and_stamp(self):
        """Payload, filename and containing directory must agree.

        A rename that left the payload behind would relabel this run in every
        downstream window, silently.
        """
        series = parse_metric_series(json.loads(METRICS_PATH.read_text(encoding='utf-8')))
        assert series.eval_id == EVAL_ID
        assert series.eval_id == METRICS_PATH.parent.name
        assert series.run_stamp == RUN_STAMP
        assert METRICS_PATH.name == f'metrics-{series.run_stamp}.json'
        assert REPORT_PATH.name == f'report-{series.run_stamp}.txt'

    def test_reemitting_the_exemplar_is_byte_identical(self):
        """The anti-hand-edit guard.

        Mirrors ``TestCommittedExemplarsAreProducible.
        test_reemitting_a_committed_series_is_byte_identical``. The source was
        already in exact canonical form when copied, so this passes verbatim —
        and goes RED the day anyone reconciles the bytes by hand instead of
        regenerating through ``write_metric_series``.
        """
        text = METRICS_PATH.read_text(encoding='utf-8')
        assert serialize_metric_series(parse_metric_series(json.loads(text))) == text


class TestItIsTheFirstRunReport:
    """What makes this pair the evidence 3208 promised, not just any run."""

    def test_the_report_names_the_same_run_as_the_metrics_beside_it(self):
        header = REPORT_PATH.read_text(encoding='utf-8').splitlines()[:3]
        assert f'memory-eval run: {EVAL_ID}' in header[0]
        assert RUN_STAMP in header[1]

    def test_the_report_carries_the_d1_initial_state_section(self):
        """D1's one-shot snapshot is the whole reason this run is special.

        A later run against the same (now non-empty) root emits no such
        section, so its presence is what proves the committed pair is the
        FIRST run and not a re-run standing in for one. Asserts on the section
        header token only — never on the surrounding prose, which is free to
        be reworded.
        """
        assert 'INITIAL STATE — known-bad items' in REPORT_PATH.read_text(encoding='utf-8')


class TestTheExemplarCannotBeMistakenForALiveRun:
    """The separation invariant, asserted structurally rather than in prose.

    ``is_initial_run()`` globs ``metrics-*.json`` under the probe's artifact
    root. An exemplar committed inside that root would make every fresh clone
    look like it had already run, permanently suppressing the one-shot D1
    initial-state snapshot for the next genuine first run — a strictly worse
    outcome than the unreachability bug this exemplar fixes. So the two roots
    must be disjoint, and that has to be a test: a README asking future
    readers not to relocate the artifact cannot enforce anything.
    """

    def test_the_probe_names_the_exemplar_root(self):
        """The constant and the on-disk location cannot drift apart.

        The test computes the path independently from ``Path(__file__)``; if
        the probe's constant were edited to point elsewhere (or the directory
        moved without it), the two would disagree here.
        """
        assert _mod().COMMITTED_EXEMPLAR_ROOT.resolve() == EXEMPLAR_ROOT.resolve()

    def test_the_exemplar_root_is_disjoint_from_the_live_artifact_root(self):
        """Neither root may be, or contain, the other."""
        exemplar = _mod().COMMITTED_EXEMPLAR_ROOT.resolve()
        live = _mod().DEFAULT_OUT_ROOT.resolve()
        assert exemplar != live
        assert not exemplar.is_relative_to(live), (
            f'{exemplar} sits inside the live artifact root {live}; a '
            f'committed artifact there would burn the one-shot D1 '
            f'initial-state snapshot for the next genuine first run'
        )
        assert not live.is_relative_to(exemplar)

    def test_a_pristine_root_is_a_first_run_but_the_exemplar_root_is_not(self, tmp_path):
        """The consequence that makes the disjointness matter.

        Two things at once, both against the real ``is_initial_run`` rather
        than a mock: the exemplar root holds a genuine, discoverable run (so
        it is not a first run), and an untouched root still reports as one. It
        is precisely because of the disjointness above that the former can
        never influence the latter when the probe next runs for real against
        ``DEFAULT_OUT_ROOT``.
        """
        is_initial_run = _mod().is_initial_run
        assert is_initial_run(tmp_path) is True
        assert is_initial_run(_mod().COMMITTED_EXEMPLAR_ROOT) is False
