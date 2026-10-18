"""Tests for eval_write_triage_judge.py — the runner, the committed-artifact guard and the CLI.

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf γ, decision D10.

Structure mirrors ``test_calibrate_write_triage.py``: each script is loaded by
path through importlib (``scripts/`` is not an importable package), under its
file stem, by the loaders in ``_eval_write_triage_fixtures.py``, and each
loader is invoked LAZILY so a test loads only the scripts it exercises.

Every test in this file is free of ``OPENAI_API_KEY``, network and Qdrant —
the judge is injected as a plain callable, exactly as ``run_calibration``
injects ``embed_fn``/``search_fn``.

WHAT THIS SUITE DELIBERATELY DOES NOT ASSERT: any accuracy FLOOR. D10 makes
the committed report the arbiter and the human at the task-3169 flip gate the
decision-maker. A floor asserted here would silently become that gate,
pre-empting a decision this task is explicitly told not to make. The
assertions are about report SHAPE, per-class presence with an explicit ``n``,
and traceability — never about whether a number is large enough.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest
from _eval_write_triage_fixtures import ALIASES as _ALIASES
from _eval_write_triage_fixtures import ALIASES_PATH, FIXTURE_PATH
from _eval_write_triage_fixtures import PROVENANCE as _PROVENANCE
from _eval_write_triage_fixtures import aliased_row as _aliased_row
from _eval_write_triage_fixtures import cases_mod as _cases_mod
from _eval_write_triage_fixtures import cli_mod as _mod
from _eval_write_triage_fixtures import corpus as _corpus
from _eval_write_triage_fixtures import crossed_retrieved_plan as _crossed_retrieved_plan
from _eval_write_triage_fixtures import live as _live
from _eval_write_triage_fixtures import one_cluster_corpus as _one_cluster_corpus
from _eval_write_triage_fixtures import rec as _rec
from _eval_write_triage_fixtures import scored as _scored
from _eval_write_triage_fixtures import scoring_mod as _scoring
from _eval_write_triage_fixtures import slate as _slate
from _eval_write_triage_fixtures import wording_mod as _wording

from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
)
from fused_memory.server.write_triage_judge import JudgePromptElisions, judge_prompt_elisions

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'eval_write_triage_judge.py'


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def _fake_judge(answer: str = OUTCOME_RESTATED):
    """A judge_fn that always answers *answer*, recording every call.

    Deliberately not a Mock: the recorded `.calls` list is asserted on
    positionally, and a Mock's call objects would let a signature change pass
    unnoticed.
    """
    def judge_fn(case, candidates):
        judge_fn.calls.append((
            case['memory_id'],
            case['expected_class'],
            [c['memory_id'] for c in candidates],
        ))
        return answer

    judge_fn.calls = []
    return judge_fn


def _run(
    tmp_path: Path,
    *,
    judge=None,
    corpus=None,
    distractors: int = 2,
    provenance=None,
):
    return _mod().run_judge_eval(
        plan=_cases_mod().seeded_plan(
            corpus if corpus is not None else _corpus(),
            distractors=distractors,
        ),
        judge_fn=judge if judge is not None else _fake_judge(),
        report_path=tmp_path / 'report.json',
        provenance=dict(_PROVENANCE if provenance is None else provenance),
    )


class TestRunJudgeEval:
    """The runner: build cases, call the judge once each, score, write, return."""

    def test_writes_the_json_report(self, tmp_path: Path) -> None:
        report = _run(tmp_path)
        assert json.loads((tmp_path / 'report.json').read_text()) == report

    def test_writes_a_markdown_sibling(self, tmp_path: Path) -> None:
        _run(tmp_path)
        assert (tmp_path / 'report.md').exists()

    # --- the markdown sibling's composition (gate item 4) --------------------
    #
    # `run_judge_eval` writes the JSON and then a markdown sibling. Deriving
    # that sibling with `Path.with_suffix('.md')` REPLACES the last suffix, so
    # `--report-path foo.md` composes back to `foo.md` and the markdown
    # silently overwrites the JSON that was just written — with no error, on a
    # script whose output is a committed artifact. `guard_committed_report`
    # does not cover it: that guard addresses dry-run/`--limit` publishing and
    # returns early for any non-committed path.

    @staticmethod
    def _sibling_run(tmp_path: Path, name: str):
        return _mod().run_judge_eval(
            plan=_cases_mod().seeded_plan(_corpus(), distractors=2),
            judge_fn=_fake_judge(),
            report_path=tmp_path / name,
            provenance=dict(_PROVENANCE),
        )

    def test_a_json_report_path_gets_a_dot_md_sibling(self, tmp_path: Path) -> None:
        """The anchor case: `.json` in, `.md` beside it, JSON still parseable."""
        report = self._sibling_run(tmp_path, 'r.json')
        assert json.loads((tmp_path / 'r.json').read_text()) == report
        assert (tmp_path / 'r.md').exists()
        assert (tmp_path / 'r.md').read_text().startswith('# ')

    def test_a_multi_suffix_path_keeps_its_json_and_gains_a_sibling(
        self, tmp_path: Path,
    ) -> None:
        """`foo.tar.gz` — the sibling is composed from the STEM, not a suffix swap."""
        report = self._sibling_run(tmp_path, 'foo.tar.gz')
        assert json.loads((tmp_path / 'foo.tar.gz').read_text()) == report
        assert (tmp_path / 'foo.tar.md').exists()

    def test_a_suffixless_path_gains_a_dot_md_sibling(self, tmp_path: Path) -> None:
        """`--report-path /tmp/smoke` is a reasonable ad-hoc spelling."""
        report = self._sibling_run(tmp_path, 'smoke')
        assert json.loads((tmp_path / 'smoke').read_text()) == report
        assert (tmp_path / 'smoke.md').exists()

    def test_a_dot_md_report_path_raises_instead_of_eating_the_json(
        self, tmp_path: Path,
    ) -> None:
        """The destructive case, and the reason this is gate item 4.

        `with_suffix('.md')` composes `foo.md` back to `foo.md`: the report is
        written and then overwritten by its own markdown, losing every number
        the run paid for, silently. `markdown_sibling` raises before the run
        does ANY work — not merely before the two writes — so the mistake
        costs nothing at all rather than costing a whole corpus of LLM calls
        for no artifact. That stronger claim is what
        :meth:`test_a_dot_md_report_path_costs_nothing_because_it_raises_first`
        pins; this one stays on the files. A warning instead of a raise, on a
        script whose output is a committed artifact, would be read after the
        loss.
        """
        target = tmp_path / 'foo.md'
        with pytest.raises(ValueError) as excinfo:
            self._sibling_run(tmp_path, 'foo.md')
        message = str(excinfo.value)
        assert str(target) in message, message
        assert not target.exists() or json.loads(target.read_text()), (
            f'{target} was left holding markdown where the JSON should be'
        )

    def test_a_dot_md_report_path_costs_nothing_because_it_raises_first(
        self, tmp_path: Path,
    ) -> None:
        """The guard's whole claim is that the mistake is free. Pin the COST.

        Raising before the two WRITES only protects the two files. The
        precondition is a function of *report_path* alone — knowable before
        `build_judge_cases`, before the per-case `judge_fn` loop, before
        `score_cases`. Evaluated after them, `--report-path foo.md` on a LIVE
        run spends every LLM call for the whole 102-case corpus and then
        raises with no artifact written at all: the operator pays for the run
        and gets nothing. The unit tests above cannot see that, because
        `_sibling_run` injects a free stub judge — so this asserts on the stub
        judge's CALL LOG instead of on the files.
        """
        judge = _fake_judge()
        with pytest.raises(ValueError):
            _mod().run_judge_eval(
                plan=_cases_mod().seeded_plan(_corpus(), distractors=2),
                judge_fn=judge,
                report_path=tmp_path / 'foo.md',
                provenance=dict(_PROVENANCE),
            )
        assert judge.calls == [], (
            f'the guard raised only after {len(judge.calls)} judge call(s); on a '
            f'live run that is the whole corpus paid for and nothing written'
        )

    def test_creates_the_report_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / 'calibration' / 'nested'
        _mod().run_judge_eval(
            plan=_cases_mod().seeded_plan(_corpus(), distractors=2),
            judge_fn=_fake_judge(),
            report_path=nested / 'report.json',
            provenance=dict(_PROVENANCE),
        )
        assert (nested / 'report.json').exists()

    def test_calls_the_judge_exactly_once_per_case(self, tmp_path: Path) -> None:
        """Every call is paid for; a re-ask would double the bill silently."""
        judge = _fake_judge()
        _run(tmp_path, judge=judge)
        expected = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        assert len(judge.calls) == len(expected)

    def test_the_judge_is_handed_resolved_candidate_records_not_bare_ids(
        self, tmp_path: Path,
    ) -> None:
        """A judge shown ids has no text to compare and is answering noise —
        the same defect `TestTheRealJudgeIsWiredAtTheToolSeam` pins at the
        production seam.
        """
        judge = _fake_judge()
        _run(tmp_path, judge=judge)
        for memory_id, _class, candidate_ids in judge.calls:
            assert candidate_ids, f'{memory_id} was handed an empty slate'

        # Keyed on (memory_id, class) because one record produces BOTH a
        # labelled case and, if it is its cluster's first, a control — and the
        # whole difference between them is the slate.
        by_case = {(mid, cls): ids for mid, cls, ids in judge.calls}
        assert by_case[('c1-dup-1', 'duplicate')][0] == 'c1-canon', (
            'the attach target must lead the resolved slate'
        )
        control = by_case[('c1-dup-1', _scoring().CLASS_DISTRACTOR)]
        assert 'c1-canon' not in control, (
            'the control slate must carry no correct attach target at all'
        )

    def test_a_judge_that_is_wrong_on_every_case_still_produces_a_report(
        self, tmp_path: Path,
    ) -> None:
        """No pass/fail verdict, at any accuracy. D10 makes the report the
        arbiter and the human the gate; a runner that raised or flagged on a
        low score would be that gate, made of code.
        """
        report = _run(tmp_path, judge=_fake_judge(OUTCOME_CONTESTED))
        assert report['per_class']['duplicate']['accuracy'] == 0.0
        assert 'verdict' not in report
        assert 'passed' not in report and 'failed' not in report

    def test_a_judge_failure_propagates_rather_than_scoring_a_wrong_answer(
        self, tmp_path: Path,
    ) -> None:
        """The same discipline `run_calibration` applies to embed_fn/search_fn.

        A swallowed judge error is indistinguishable from a genuine
        misclassification, so it would silently shrink the measured population
        AND depress the accuracy it reports — a doubly wrong number.
        """
        def boom(case, candidates):
            raise RuntimeError('the judge exploded')

        with pytest.raises(RuntimeError, match='exploded'):
            _run(tmp_path, judge=boom)

    def test_provenance_records_the_measured_population(self, tmp_path: Path) -> None:
        report = _run(tmp_path)
        provenance = report['provenance']
        assert provenance['record_count'] == len(_corpus())
        assert provenance['case_count'] == len(
            _cases_mod().build_judge_cases(_corpus(), distractors=2),
        )
        assert provenance['distractor_count'] == 2
        assert provenance['judge_model'] == 'gpt-4o-mini'

    @staticmethod
    def _short_pool() -> list[dict]:
        """`_corpus()`'s c1 and c2 clusters: each draws from the other's 3 records.

        At ``distractors=4`` that narrows every labelled slate to 4 and every
        control to 3 — short, but never empty. A single cluster would empty
        the control's slate, which `EvalPlan` refuses outright.
        """
        return [r for r in _corpus() if r['cluster_id'] in {'c1-canon', 'c2-canon'}]

    def test_slate_width_provenance_is_measured_not_asserted(
        self, tmp_path: Path,
    ) -> None:
        """The report must publish the width it BUILT, not the one asked for.

        `_rotated` truncates when the distractor pool is short, so
        `distractors + 1` is a REQUEST that can silently exceed what the
        slates actually carried. Recording the request as `candidate_count`
        is the "silently narrows a slate the report claims was 5 wide"
        failure `run_judge_eval`'s docstring says its KeyError prevents,
        arriving by the other door — and the artifact is the operator's
        stated input at the task-3169 flip gate.

        A short distractor pool is the concrete trigger: `_distractor_pool`
        draws only from OTHER clusters, so two clusters of three records
        cannot fill a 5-wide slate.
        """
        short = self._short_pool()
        report = _run(tmp_path, corpus=short, distractors=4)
        provenance = report['provenance']

        widths = {
            len(c['candidates'])
            for c in _cases_mod().build_judge_cases(short, distractors=4)
        }
        assert widths == {4, 3}, 'precondition: labelled slates 4 wide, controls 3'
        assert provenance['candidate_count'] == max(widths)
        assert provenance['candidate_count_min'] == min(widths)
        assert provenance['candidate_count'] != 5, (
            'the report must not claim a width it never built'
        )
        assert provenance['distractor_count_requested'] == 4
        assert provenance['distractor_count'] == max(widths) - 1

    def test_a_short_pool_is_logged_loudly(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Silence here reads as "covered everything" when it did not."""
        with caplog.at_level(logging.WARNING):
            _run(tmp_path, corpus=self._short_pool(), distractors=4)
        assert any(
            record.levelno >= logging.WARNING for record in caplog.records
        ), 'a narrowed slate must warn, not pass silently'

    def test_a_full_width_run_records_the_requested_width(
        self, tmp_path: Path,
    ) -> None:
        """Measuring must not CHANGE the numbers when the pool is adequate.

        A seeded run whose pool can supply `--distractors` publishes exactly
        the width asked for, or measuring would silently restate the
        operator's own input.
        """
        report = _run(tmp_path, distractors=2)
        provenance = report['provenance']
        assert provenance['candidate_count'] == 3
        assert provenance['candidate_count_min'] == 3
        assert provenance['distractor_count'] == 2
        assert provenance['distractor_count_requested'] == 2

    def test_the_effective_cap_is_recorded_beside_the_built_width(
        self, tmp_path: Path,
    ) -> None:
        """`candidate_count` is the slate BUILT. The model sees it TRIMMED.

        `judge_write` re-trims via
        `select_judge_candidates(..., resolve_judge_candidate_count(...))`, so
        a slate built 3 wide against a cap of 2 reaches the model 2 wide and
        `candidate_count: 3` overstates what was measured. With the shipped
        `judge_candidate_count: 5` and the default `--distractors 4` the two
        agree, which is why this is latent rather than visible.

        Recorded ALONGSIDE the measurement, never in place of it: the cap is
        a config value and `candidate_count` stays the width that was built.
        """
        report = _run(tmp_path, distractors=2, provenance={
            **_PROVENANCE, 'judge_candidate_count': 2,
        })
        provenance = report['provenance']
        assert provenance['candidate_count'] == 3, 'precondition: built 3 wide'
        assert provenance['judge_candidate_count'] == 2

    def test_a_slate_wider_than_the_effective_cap_is_logged_loudly(
        self, tmp_path: Path, caplog,
    ) -> None:
        """The same silence the short-pool warning exists to break.

        A run that builds wider than the model can see is measuring a
        different slate from the one it publishes, and nothing in the
        artifact says so.
        """
        with caplog.at_level(logging.WARNING):
            _run(tmp_path, distractors=2, provenance={
                **_PROVENANCE, 'judge_candidate_count': 2,
            })
        warnings = [
            record.getMessage()
            for record in caplog.records if record.levelno >= logging.WARNING
        ]
        assert any('measured' in message for message in warnings), warnings
        assert any('3' in message and '2' in message for message in warnings), (
            'the warning must name the width built and the cap it exceeds'
        )

    def test_a_slate_within_the_effective_cap_does_not_warn(
        self, tmp_path: Path, caplog,
    ) -> None:
        """A warning that always fires is a warning nobody reads."""
        with caplog.at_level(logging.WARNING):
            _run(tmp_path, distractors=2, provenance={
                **_PROVENANCE, 'judge_candidate_count': 5,
            })
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestRunResolvesTheJudgeConfigIntoProvenance:
    """`_run` records the two config values that decide what was measured.

    `run_judge_eval` measures the width it BUILT; what the model saw is that
    width trimmed by `judge_candidate_count`. And whether the model saw
    anything at all is `judge_enabled`: a run against a config with the kill
    switch off returns `stored` on `judge_write`'s FIRST line, spends nothing,
    and still writes a committed-looking report — in which `distractor` scores
    1.0 (its only acceptable outcome IS `stored`) and `duplicate` scores 0.0.
    Neither fact is recoverable from the artifact today.

    Driven through `_run` with `run_judge_eval` captured, so this pins the
    resolution wiring rather than the runner it hands off to.
    """

    def _handed(self, tmp_path: Path, monkeypatch, **overrides) -> dict:
        """The kwargs `_run` hands `run_judge_eval`, with *overrides* on the args."""
        handed: dict = {}

        def fake_run_judge_eval(**kwargs):
            handed.update(kwargs)
            return _scoring().build_report(
                scored=_scoring().score_cases([], []),
                provenance=dict(kwargs['provenance']),
            )

        monkeypatch.setattr(_mod(), 'run_judge_eval', fake_run_judge_eval)
        args = types.SimpleNamespace(**{
            'config': None,
            'report_path': str(tmp_path / 'report.json'),
            'fixture': str(FIXTURE_PATH),
            'distractors': 2,
            'limit': None,
            'dry_run': True,
            'slate_mode': _scoring().SLATE_SEEDED,
            'project_id': 'reify',
            'field_chars': None,
            'judge_candidate_count': None,
            'cases_path': None,
            'canonical_aliases': None,
            'wording': 'shipped',
            **overrides,
        })
        assert _mod()._run(args) == 0
        return handed

    def _captured(self, tmp_path: Path, monkeypatch, **overrides) -> dict:
        return dict(self._handed(tmp_path, monkeypatch, **overrides)['provenance'])

    @staticmethod
    def _service():
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

        return types.SimpleNamespace(config=FusedMemoryConfig())

    def test_provenance_records_the_effective_candidate_cap(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from fused_memory.server import write_triage_judge  # noqa: PLC0415

        captured = self._captured(tmp_path, monkeypatch)
        expected = write_triage_judge.resolve_judge_candidate_count(self._service())
        assert captured['judge_candidate_count'] == expected
        assert isinstance(captured['judge_candidate_count'], int)

    def test_provenance_records_the_effective_field_width(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from fused_memory.server import write_triage_judge  # noqa: PLC0415

        captured = self._captured(tmp_path, monkeypatch)
        expected = write_triage_judge.resolve_judge_field_chars(self._service())
        assert captured['field_chars'] == expected
        assert isinstance(captured['field_chars'], int)

    def test_a_requested_width_is_recorded(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        captured = self._captured(tmp_path, monkeypatch, field_chars=40)
        assert captured['field_chars'] == 40

    def test_provenance_records_whether_the_judge_arm_was_live(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from fused_memory.server import write_triage_judge  # noqa: PLC0415

        captured = self._captured(tmp_path, monkeypatch)
        expected = write_triage_judge.resolve_judge_enabled(self._service())
        assert captured['judge_enabled'] is expected
        assert isinstance(captured['judge_enabled'], bool)

    def test_a_disabled_judge_is_machine_detectable_in_the_artifact(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The kill switch is the one that costs the operator a decision.

        `judge_write` returns `stored` on its first line when the switch is
        off, so every case is answered without a provider call and the report
        still looks like a measurement.
        """
        from fused_memory.server import write_triage_judge  # noqa: PLC0415

        monkeypatch.setattr(
            write_triage_judge, 'resolve_judge_enabled', lambda service: False,
        )
        captured = self._captured(tmp_path, monkeypatch)
        assert captured['judge_enabled'] is False

    def test_provenance_names_the_alias_sidecar_and_its_size(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        captured = self._captured(tmp_path, monkeypatch, canonical_aliases=str(ALIASES_PATH))
        assert captured['canonical_aliases_path'] == (
            'tests/fixtures/write_triage_calibration.canonical_aliases.json'
        )
        assert captured['canonical_aliases_count'] == 3

    def test_no_sidecar_is_recorded_as_none_and_zero(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        captured = self._captured(tmp_path, monkeypatch)
        assert (captured['canonical_aliases_path'], captured['canonical_aliases_count']) == (
            None, 0,
        )

    def test_every_case_is_built_with_the_sidecars_aliases(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        aliases = json.loads(ALIASES_PATH.read_text())
        plan = self._handed(tmp_path, monkeypatch, canonical_aliases=str(ALIASES_PATH))['plan']
        for case in plan.cases:
            assert case['canonical_alias_id'] == aliases.get(case['cluster_id']), (
                case['memory_id']
            )
        assert {c['canonical_alias_id'] for c in plan.cases} - {None} == set(aliases.values())

    def test_provenance_names_the_cases_file_package_relative(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        never_written = SCRIPT_PATH.parent.parent / 'calibration' / 'never-written.cases.jsonl'
        captured = self._captured(tmp_path, monkeypatch, cases_path=str(never_written))
        assert captured['cases_path'] == 'calibration/never-written.cases.jsonl'
        assert not never_written.exists()

    def test_no_dump_is_recorded_as_no_cases_path(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        assert self._captured(tmp_path, monkeypatch)['cases_path'] is None

    def test_a_malformed_sidecar_is_refused_by_the_calibrators_parser(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        sidecar = tmp_path / 'aliases.json'
        sidecar.write_text(json.dumps({'old-canonical': 123}))
        with pytest.raises(ValueError, match='str -> str'):
            self._captured(tmp_path, monkeypatch, canonical_aliases=str(sidecar))

    def test_the_three_keys_are_provenance_vocabulary(self) -> None:
        assert {'canonical_aliases_path', 'canonical_aliases_count', 'cases_path'} <= set(
            _scoring().PROVENANCE_KEYS,
        )

    def test_provenance_records_the_system_prompt_it_measured(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from fused_memory.server.write_triage_judge import (  # noqa: PLC0415
            JUDGE_SYSTEM_PROMPT,
        )

        recorded = self._captured(tmp_path, monkeypatch)['judge_system_prompt_sha256']
        assert recorded == hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode('utf-8')).hexdigest()
        assert len(recorded) == 64 and set(recorded) <= set('0123456789abcdef')

    def test_a_pre_psi_run_records_the_pre_psi_prompt_it_measured(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        recorded = self._captured(tmp_path, monkeypatch, wording='pre-psi')[
            'judge_system_prompt_sha256'
        ]
        assert recorded == hashlib.sha256(
            _wording().PRE_PSI_JUDGE_SYSTEM_PROMPT.encode('utf-8'),
        ).hexdigest()


class TestADotMdReportPathIsRejectedAtArgumentTime:
    """`--report-path foo.md` is a bad ARGUMENT, so `_run` refuses it up front.

    Nothing about the collision depends on what the run measures, so making
    the operator wait for a paid corpus-wide run to be told is pure waste.
    Driven with a fixture that does NOT exist: if the check were ordered
    after the fixture load — never mind after the judge loop — the failure
    would be a `FileNotFoundError` naming the fixture instead of the
    `ValueError` naming the report path.
    """

    @staticmethod
    def _args(tmp_path: Path, name: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            config=None,
            report_path=str(tmp_path / name),
            fixture=str(tmp_path / 'no-such-fixture.jsonl'),
            distractors=2,
            limit=None,
            dry_run=True,
            slate_mode=_scoring().SLATE_SEEDED,
            project_id='reify',
            field_chars=None,
            judge_candidate_count=None,
            cases_path=None,
            canonical_aliases=None,
            wording='shipped',
        )

    def test_run_refuses_it_before_it_even_reads_the_fixture(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            _mod()._run(self._args(tmp_path, 'foo.md'))
        assert str(tmp_path / 'foo.md') in str(excinfo.value), str(excinfo.value)

    def test_a_json_report_path_is_not_refused_by_that_check(
        self, tmp_path: Path,
    ) -> None:
        """The control: a well-formed path gets past it and fails LATER, on
        the missing fixture, which is the next real precondition."""
        with pytest.raises(FileNotFoundError):
            _mod()._run(self._args(tmp_path, 'report.json'))


# ---------------------------------------------------------------------------
# The committed artifact
# ---------------------------------------------------------------------------

class TestGuardCommittedReport:
    """A run that did not measure must not publish itself as the measurement.

    The committed report is what the operator reads at the task-3169 flip
    gate. The regression this class pins: the overwrite guard used to live
    INSIDE ``_run``'s ``if args.limit is not None:`` block, so a bare
    ``--dry-run`` — the FIRST invocation in the module docstring, and the one
    that measures nothing at all — defaulted to the committed path and
    rewrote both artifacts with fixed-answer numbers under a `dry-run`
    provider nobody was obliged to notice.

    The committed report is also the RETRIEVED-slate arbiter (PRD C2'), so a
    seeded measurement aimed at it is refused outright. A seeded run anywhere
    else is untouched: that is how historical artifacts are reproduced.
    """

    def test_an_unrelated_path_is_returned_untouched(self, tmp_path: Path) -> None:
        target = str(tmp_path / 'somewhere-else.json')
        assert _mod().guard_committed_report(
            target, dry_run=True, limit=None, slate_mode=_scoring().SLATE_SEEDED) == target

    def test_a_dry_run_is_redirected_off_the_committed_artifact(self) -> None:
        committed = _mod()._DEFAULT_REPORT_PATH
        for slate_mode in _scoring().SLATE_MODES:
            got = _mod().guard_committed_report(
                committed, dry_run=True, limit=None, slate_mode=slate_mode)
            assert Path(got).resolve() != Path(committed).resolve(), (
                f'a {slate_mode} --dry-run scores a fixed-answer stub; publishing '
                f'it puts fabricated numbers in front of the flip-gate operator'
            )
            assert Path(got).name == _mod()._DRY_RUN_REPORT_NAME

    def test_a_relative_spelling_of_the_committed_path_is_also_caught(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A string ``==`` misses this; ``resolve()`` on both sides catches it."""
        committed = Path(_mod()._DEFAULT_REPORT_PATH)
        monkeypatch.chdir(committed.parent.parent)
        relative = str(Path('calibration') / committed.name)
        got = _mod().guard_committed_report(
            relative, dry_run=True, limit=None, slate_mode=_scoring().SLATE_SEEDED)
        assert Path(got).resolve() != committed.resolve()

    def test_the_dry_run_redirect_is_logged_loudly(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            _mod().guard_committed_report(
                _mod()._DEFAULT_REPORT_PATH, dry_run=True, limit=None,
                slate_mode=_scoring().SLATE_SEEDED)
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.parametrize('limit', [None, 5])
    def test_a_seeded_measurement_cannot_overwrite_the_retrieved_arbiter(
        self, limit: int | None,
    ) -> None:
        with pytest.raises(ValueError, match='--report-path'):
            _mod().guard_committed_report(
                _mod()._DEFAULT_REPORT_PATH, dry_run=False, limit=limit,
                slate_mode=_scoring().SLATE_SEEDED)

    def test_nor_can_it_through_a_relative_spelling(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        committed = Path(_mod()._DEFAULT_REPORT_PATH)
        monkeypatch.chdir(committed.parent.parent)
        with pytest.raises(ValueError, match='--report-path'):
            _mod().guard_committed_report(
                str(Path('calibration') / committed.name), dry_run=False, limit=None,
                slate_mode=_scoring().SLATE_SEEDED)

    def test_a_seeded_measurement_elsewhere_is_untouched(self, tmp_path: Path) -> None:
        target = str(tmp_path / 'historical.json')
        assert _mod().guard_committed_report(
            target, dry_run=False, limit=None, slate_mode=_scoring().SLATE_SEEDED) == target

    def test_the_slate_mode_must_be_named(self) -> None:
        """No default: a caller that forgot it must not be waved through as retrieved."""
        with pytest.raises(TypeError, match='slate_mode'):
            _mod().guard_committed_report(
                _mod()._DEFAULT_REPORT_PATH, dry_run=False, limit=None)

    def test_a_limit_run_still_writes_the_committed_path_but_warns(
        self, caplog,
    ) -> None:
        """`--limit` measures REALLY, just partially.

        Its numbers are the judge's own and ``provenance.limit`` records the
        truncation in the artifact, so this one warns and proceeds rather
        than redirecting — the artifact-level tell is what
        :meth:`TestCommittedJudgeAccuracyReportIsTraceable.test_the_committed_report_is_a_full_measurement`
        reads.
        """
        committed = _mod()._DEFAULT_REPORT_PATH
        with caplog.at_level(logging.WARNING):
            got = _mod().guard_committed_report(
                committed, dry_run=False, limit=5, slate_mode=_scoring().SLATE_RETRIEVED)
        assert got == committed
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_a_full_retrieved_run_is_neither_redirected_nor_warned(
        self, caplog,
    ) -> None:
        committed = _mod()._DEFAULT_REPORT_PATH
        with caplog.at_level(logging.WARNING):
            got = _mod().guard_committed_report(
                committed, dry_run=False, limit=None, slate_mode=_scoring().SLATE_RETRIEVED)
        assert got == committed
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_a_non_shipped_wording_cannot_overwrite_the_arbiter(self) -> None:
        with pytest.raises(ValueError, match='--report-path'):
            _mod().guard_committed_report(
                _mod()._DEFAULT_REPORT_PATH, dry_run=False, limit=None,
                slate_mode=_scoring().SLATE_RETRIEVED, wording='pre-psi')

    def test_the_shipped_wording_may_write_the_arbiter(self) -> None:
        committed = _mod()._DEFAULT_REPORT_PATH
        assert _mod().guard_committed_report(
            committed, dry_run=False, limit=None,
            slate_mode=_scoring().SLATE_RETRIEVED, wording='shipped') == committed

    def test_a_pre_psi_run_elsewhere_is_untouched(self, tmp_path: Path) -> None:
        target = str(tmp_path / 'fixture-pre-psi.json')
        assert _mod().guard_committed_report(
            target, dry_run=False, limit=None,
            slate_mode=_scoring().SLATE_RETRIEVED, wording='pre-psi') == target


class TestABareDryRunCannotReachTheCommittedArtifact:
    """The guard must be consulted on EVERY path, driven rather than read.

    Replaces a test that asserted `'guard_committed_report(' in
    inspect.getsource(_run).split('if args.limit is not None:')[0]`. That
    passes UNCONDITIONALLY the moment the split literal is reworded or moved:
    `str.split` on an absent separator returns the WHOLE function body, so the
    guard could sink back inside the `--limit` block — the exact structural
    mistake being pinned — and the assertion would still find it. Per the repo
    norm it is replaced with behaviour, not hardened into a better grep.

    Driven through `main()` rather than a hand-built args namespace, so the
    argparse DEFAULT for `--report-path` is what gets guarded — that default
    is the whole hazard: a bare `--dry-run`, the first invocation in the
    module docstring, aims at the committed artifact without the operator
    naming it.
    """

    def test_a_bare_dry_run_leaves_both_committed_artifacts_byte_identical(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import sys  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        fixture = tmp_path / 'synthetic_corpus.jsonl'
        fixture.write_text(
            ''.join(json.dumps(record) + '\n' for record in _corpus()),
        )

        committed = Path(_mod()._DEFAULT_REPORT_PATH)
        sibling = committed.parent / (committed.stem + '.md')
        assert committed.exists() and sibling.exists(), (
            'precondition: both committed artifacts are on disk to be clobbered'
        )
        before = {path: path.read_bytes() for path in (committed, sibling)}
        before_mtimes = {path: path.stat().st_mtime_ns for path in before}

        # The redirect target, pointed somewhere hermetic. `guard_committed_report`
        # imports `tempfile` inside itself, so the module attribute is the seam.
        redirect_dir = tmp_path / 'redirected'
        redirect_dir.mkdir()
        monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(redirect_dir))
        monkeypatch.setattr(sys, 'argv', [
            'eval_write_triage_judge.py', '--dry-run', '--fixture', str(fixture),
        ])

        assert _mod().main() == 0

        for path, content in before.items():
            assert path.read_bytes() == content, (
                f'a --dry-run rewrote {path.name} with fixed-answer numbers — '
                f'the guard is not reached on the bare --dry-run path'
            )
            assert path.stat().st_mtime_ns == before_mtimes[path], (
                f'{path.name} was rewritten (identically, this time) — the '
                f'guard must not let a dry run touch it at all'
            )

        redirected = redirect_dir / _mod()._DRY_RUN_REPORT_NAME
        assert redirected.exists(), (
            'the dry run must still publish its throwaway where the redirect '
            'warning says it did, or "prove the pipeline" proves nothing'
        )
        assert json.loads(redirected.read_text())['provenance'][
            'judge_provider'
        ] == 'dry-run'

    def test_a_bare_live_run_is_refused_before_it_reads_the_fixture(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The old "full measured pass" command: seeded, aimed at the arbiter.

        Given a fixture that does not exist, so a refusal ordered after the
        fixture load would surface as `FileNotFoundError` instead.
        """
        import sys  # noqa: PLC0415

        monkeypatch.setattr(sys, 'argv', [
            'eval_write_triage_judge.py', '--fixture', str(tmp_path / 'absent.jsonl'),
        ])
        with pytest.raises(ValueError, match='--report-path'):
            _mod().main()


# ---------------------------------------------------------------------------
# Option C: the per-case dump, the field-width override, the retrieved slate
# ---------------------------------------------------------------------------

class TestTheCasesDump:
    """Per-case evidence, written as it is bought."""

    def _dump(self, tmp_path: Path, judge=None) -> list[dict]:
        _mod().run_judge_eval(
            plan=_cases_mod().seeded_plan(_corpus(), distractors=2),
            judge_fn=judge if judge is not None else _fake_judge(),
            report_path=tmp_path / 'report.json',
            provenance=dict(_PROVENANCE),
            cases_path=tmp_path / 'cases.jsonl',
        )
        return [
            json.loads(line)
            for line in (tmp_path / 'cases.jsonl').read_text().splitlines()
            if line.strip()
        ]

    def test_one_line_per_case(self, tmp_path: Path) -> None:
        rows = self._dump(tmp_path)
        assert len(rows) == len(_cases_mod().build_judge_cases(_corpus(), distractors=2))

    def test_a_row_carries_what_the_aggregates_cannot_be_reread_from(
        self, tmp_path: Path,
    ) -> None:
        row = self._dump(tmp_path)[0]
        assert set(row) >= {
            'memory_id', 'cluster_id', 'expected_class', 'candidates',
            'band_winner_id', 'verdict_candidate_id', 'judged_candidate_id',
            'attach_target_id', 'attach_target_is_canonical', 'verdict',
            'outcome', 'band', 'similarity', 'canonical_in_slate',
            'canonical_present', 'entry_elided', 'candidates_elided',
        }

    def test_a_bare_word_names_no_candidate_so_attaches_to_the_band_winner(
        self, tmp_path: Path,
    ) -> None:
        rows = self._dump(tmp_path, judge=_fake_judge(OUTCOME_AMENDED))
        assert {
            (row['verdict_candidate_id'], row['judged_candidate_id']) for row in rows
        } == {(None, None)}
        assert all(row['attach_target_id'] == row['band_winner_id'] for row in rows)

    def test_the_slate_is_recorded_in_prompt_order(self, tmp_path: Path) -> None:
        rows = {r['memory_id']: r for r in self._dump(tmp_path)
                if r['expected_class'] == 'duplicate'}
        row = rows['c1-dup-1']
        assert row['candidates'][0] == 'c1-canon'
        assert row['attach_target_id'] == 'c1-canon'
        assert row['attach_target_is_canonical'] is True
        assert row['canonical_in_slate'] is True

    def test_a_control_case_records_that_no_correct_target_was_shown(
        self, tmp_path: Path,
    ) -> None:
        control = [
            r for r in self._dump(tmp_path)
            if r['expected_class'] == _scoring().CLASS_DISTRACTOR
        ][0]
        assert control['canonical_in_slate'] is False
        assert control['attach_target_is_canonical'] is False

    def test_each_row_lands_before_the_next_case_is_asked(
        self, tmp_path: Path,
    ) -> None:
        """A 429 partway through must not cost the cases already paid for."""
        seen: list[int] = []

        def judge_fn(case, candidates):
            path = tmp_path / 'cases.jsonl'
            seen.append(
                len(path.read_text().splitlines()) if path.exists() else 0,
            )
            return OUTCOME_RESTATED

        self._dump(tmp_path, judge=judge_fn)
        assert seen == list(range(len(seen))), (
            f'rows were buffered rather than flushed per case: {seen}'
        )

    def test_a_provider_error_mid_run_keeps_every_row_already_answered(
        self, tmp_path: Path,
    ) -> None:
        """Task 5277 A4: the error propagates, and cases 1..N-1 stay on disk."""
        answers = [OUTCOME_RESTATED, OUTCOME_STORED, OUTCOME_AMENDED]
        asked: list[str] = []

        def judge_fn(case, candidates):
            asked.append(case['memory_id'])
            if len(asked) > len(answers):
                raise RuntimeError('429 Too Many Requests')
            return answers[len(asked) - 1]

        with pytest.raises(RuntimeError, match='429'):
            self._dump(tmp_path, judge=judge_fn)
        *answered, failed = asked
        rows = [
            json.loads(line)
            for line in (tmp_path / 'cases.jsonl').read_text().splitlines()
        ]
        assert [(r['memory_id'], r['outcome']) for r in rows] == list(
            zip(answered, answers, strict=True),
        ), f'the rows bought before case {len(asked)} ({failed}) raised were lost'

    def test_no_dump_is_written_without_a_cases_path(self, tmp_path: Path) -> None:
        _run(tmp_path)
        assert list(tmp_path.glob('*.jsonl')) == []

    def test_an_attach_to_the_alias_is_an_attach_to_the_canonical(self) -> None:
        row = _aliased_row([_live('new-canon')], 'new-canon')
        assert row['canonical_alias_id'] == 'new-canon'
        assert row['attach_target_is_canonical'] is True
        assert row['canonical_in_slate'] is True

    def test_a_sighting_child_of_the_alias_puts_the_canonical_on_the_slate(self) -> None:
        row = _aliased_row([_live('child', canonical_id='new-canon')], 'new-canon')
        assert row['canonical_in_slate'] is True

    def test_an_alias_does_not_make_an_unrelated_attach_canonical(self) -> None:
        row = _aliased_row([_live('zz')], 'zz')
        assert row['attach_target_is_canonical'] is False
        assert row['canonical_in_slate'] is False

    def test_a_case_with_no_alias_yields_todays_row(self) -> None:
        records = [
            _rec('c1-canon', 'c1-canon', 'canonical'),
            _rec('c1-dup-1', 'c1-canon', 'duplicate'),
        ]
        slate = _slate('c1-dup-1', candidates=[_live('c1-canon'), _live('zz')],
                       band_winner_id='c1-canon', band=OUTCOME_JUDGE)
        plan = _cases_mod().plan_from_slates(records, [slate], provenance={}, aliases=_ALIASES)
        row = _scoring().case_row(
            3, plan.cases[0],
            _scoring().JudgeAnswer(outcome=OUTCOME_AMENDED, verdict=OUTCOME_AMENDED),
            plan.candidate_records[0], plan.fixture_by_id,
        )
        assert row == {
            'index': 3,
            'memory_id': 'c1-dup-1',
            'cluster_id': 'c1-canon',
            'category': 'procedural_knowledge',
            'expected_class': 'duplicate',
            'acceptable_outcomes': sorted([OUTCOME_AMENDED, OUTCOME_RESTATED]),
            'candidates': ['c1-canon', 'zz'],
            'band_winner_id': 'c1-canon',
            'verdict_candidate_id': None,
            'judged_candidate_id': None,
            'attach_target_id': 'c1-canon',
            'attach_target_cluster_id': 'c1-canon',
            'attach_target_category': 'procedural_knowledge',
            'attach_target_label': 'canonical',
            'attach_target_is_canonical': True,
            'canonical_in_slate': True,
            'canonical_present': True,
            'band': OUTCOME_JUDGE,
            'similarity': 0.7,
            'verdict': OUTCOME_AMENDED,
            'outcome': OUTCOME_AMENDED,
            'correct': True,
            'entry_elided': None,
            'candidates_elided': None,
            'usage': None,
            'canonical_alias_id': None,
        }


class TestTheAttachTargetIsDescribedFromTheFixture:
    """A null cluster or label means the target is not a fixture record.

    A retrieved row carries only what the store holds, so a target described
    from the slate alone published both as null in the arbiter's own mode,
    fixture record or not.
    """

    RECORDS = (
        _rec('canon', 'canon', 'canonical', category='preferences_and_norms'),
        _rec('dup-a', 'canon', 'duplicate'),
        _rec('dup-b', 'canon', 'duplicate', category='observations_and_summaries'),
    )

    @classmethod
    def _dup_a_target(cls, candidates, band_winner_id, *, records=None, **kwargs) -> dict:
        """How dup-a's dump row describes its attach target."""
        population = list(cls.RECORDS if records is None else records)
        slates = [
            _slate(r['memory_id'], candidates=candidates,
                   band_winner_id=band_winner_id, band=OUTCOME_JUDGE)
            for r in population if r['label'] != 'canonical'
        ]
        plan = _cases_mod().plan_from_slates(population, slates, provenance={}, **kwargs)
        row = _scoring().case_row(
            1, plan.cases[0], _scoring().JudgeAnswer(outcome=OUTCOME_AMENDED),
            plan.candidate_records[0], plan.fixture_by_id,
        )
        return {key: row[f'attach_target_{key}'] for key in ('cluster_id', 'label', 'category')}

    def test_a_wrong_attach_to_a_fixture_sibling_names_its_cluster_and_label(self) -> None:
        shown = [_live('dup-b', category='observations_and_summaries')]
        assert self._dup_a_target(shown, 'dup-b') == {
            'cluster_id': 'canon', 'label': 'duplicate', 'category': 'observations_and_summaries',
        }

    def test_a_hoisted_parent_off_the_slate_is_described_from_the_fixture(self) -> None:
        shown = [_live('child', canonical_id='canon')]
        assert self._dup_a_target(shown, 'canon') == {
            'cluster_id': 'canon', 'label': 'canonical', 'category': 'preferences_and_norms',
        }

    def test_a_target_outside_the_fixture_is_described_by_its_row_alone(self) -> None:
        shown = [_live('zz', category='observations_and_summaries')]
        assert self._dup_a_target(shown, 'zz') == {
            'cluster_id': None, 'label': None, 'category': 'observations_and_summaries',
        }

    def test_a_limited_run_describes_a_target_as_the_full_run_does(self) -> None:
        """`--limit` narrows the population, not the fixture a target is read from."""
        limited = [r for r in self.RECORDS if r['memory_id'] != 'dup-b']
        shown = [_live('dup-b', category='observations_and_summaries')]
        assert self._dup_a_target(
            shown, 'dup-b', records=limited, fixture=self.RECORDS,
        ) == self._dup_a_target(shown, 'dup-b')


class TestTheAttachTargetIsTheJudgedCandidate:
    """A judged attach lands on the candidate the verdict NAMED, hoisted.

    `write_triage.py::_apply_judge_verdict` files a middle-band attach against
    `canonical_id_of(named record)`, and against the band winner only when
    the verdict named nothing. The row's `attach_target_*` describe that
    record, so `score_attachments` scores production's attach.
    """

    RECORDS = (
        _rec('c1-canon', 'c1-canon', 'canonical'),
        _rec('c1-dup-1', 'c1-canon', 'duplicate'),
    )

    @classmethod
    def _row(cls, candidates, band_winner_id, answer) -> dict:
        slate = _slate('c1-dup-1', candidates=candidates,
                       band_winner_id=band_winner_id, band=OUTCOME_JUDGE)
        plan = _cases_mod().plan_from_slates(list(cls.RECORDS), [slate], provenance={})
        return _scoring().case_row(
            1, plan.cases[0], answer, plan.candidate_records[0], plan.fixture_by_id,
        )

    @staticmethod
    def _naming(candidate_id: str):
        return _scoring().JudgeAnswer(
            outcome=OUTCOME_AMENDED, verdict=OUTCOME_AMENDED, candidate_id=candidate_id,
        )

    def test_the_judge_naming_the_canonical_overrules_the_band_winner(self) -> None:
        row = self._row([_live('zz'), _live('c1-canon')], 'zz', self._naming('c1-canon'))
        assert {key: row[key] for key in (
            'band_winner_id', 'verdict_candidate_id', 'judged_candidate_id',
            'attach_target_id', 'attach_target_is_canonical', 'attach_target_label',
        )} == {
            'band_winner_id': 'zz',
            'verdict_candidate_id': 'c1-canon',
            'judged_candidate_id': 'c1-canon',
            'attach_target_id': 'c1-canon',
            'attach_target_is_canonical': True,
            'attach_target_label': 'canonical',
        }

    def test_a_named_sighting_child_is_hoisted_to_its_canonical(self) -> None:
        row = self._row(
            [_live('zz'), _live('child', canonical_id='c1-canon')], 'zz', self._naming('child'),
        )
        assert (row['verdict_candidate_id'], row['judged_candidate_id']) == ('child', 'c1-canon')
        assert row['attach_target_is_canonical'] is True
        assert row['attach_target_cluster_id'] == 'c1-canon'

    def test_the_judge_can_move_a_correct_band_attach_off_the_canonical(self) -> None:
        row = self._row([_live('c1-canon'), _live('zz')], 'c1-canon', self._naming('zz'))
        assert row['attach_target_id'] == 'zz'
        assert row['attach_target_is_canonical'] is False

    def test_the_alias_rule_applies_to_the_judged_candidate(self) -> None:
        row = _aliased_row(
            [_live('zz'), _live('new-canon')], 'zz', self._naming('new-canon'),
        )
        assert row['attach_target_is_canonical'] is True

    @pytest.mark.parametrize('answer', [
        {'outcome': OUTCOME_AMENDED, 'verdict': OUTCOME_AMENDED},
        {'outcome': OUTCOME_RESTATED},
    ], ids=['a-verdict-naming-none', 'a-band-that-decided-itself'])
    def test_naming_nothing_attaches_to_the_band_winner(self, answer: dict) -> None:
        row = self._row(
            [_live('zz'), _live('c1-canon')], 'zz', _scoring().JudgeAnswer(**answer),
        )
        assert (row['verdict_candidate_id'], row['judged_candidate_id']) == (None, None)
        assert row['attach_target_id'] == row['band_winner_id'] == 'zz'

    def test_a_named_id_off_the_cases_own_slate_raises(self) -> None:
        """Production fails such a write open to `stored`; it never attaches it."""
        with pytest.raises(ValueError) as excinfo:
            self._row([_live('zz'), _live('c1-canon')], 'zz', self._naming('ghost'))
        message = str(excinfo.value)
        assert 'c1-dup-1' in message and 'ghost' in message, message

    def test_a_seeded_case_naming_a_distractor_attaches_off_the_canonical(self) -> None:
        plan = _cases_mod().seeded_plan(_corpus(), distractors=2)
        index, case = next(
            (i, c) for i, c in enumerate(plan.cases) if c['expected_class'] == 'duplicate'
        )
        distractor = case['candidates'][1]
        row = _scoring().case_row(
            1, case, self._naming(distractor),
            plan.candidate_records[index], plan.fixture_by_id,
        )
        assert row['judged_candidate_id'] == distractor
        assert row['attach_target_is_canonical'] is False

    def test_the_production_shape_scores_the_judged_attach(self, tmp_path: Path) -> None:
        records = [
            _rec('c1-canon', 'c1-canon', 'canonical'),
            _rec('c1-dup-1', 'c1-canon', 'duplicate'),
            _rec('c2-canon', 'c2-canon', 'canonical'),
            _rec('c2-dup-1', 'c2-canon', 'duplicate'),
        ]
        slates = [
            _slate('c1-dup-1', candidates=[_live('zz'), _live('c1-canon')],
                   band_winner_id='zz', band=OUTCOME_JUDGE),
            _slate('c2-dup-1', candidates=[_live('zz'), _live('c2-canon')],
                   band_winner_id='zz', band=OUTCOME_JUDGE),
        ]
        named = {'c1-dup-1': 'c1-canon', 'c2-dup-1': 'zz'}
        report = _mod().run_judge_eval(
            plan=_cases_mod().plan_from_slates(records, slates, provenance={}),
            judge_fn=lambda case, candidates: self._naming(named[case['memory_id']]),
            report_path=tmp_path / 'report.json', provenance={},
            cases_path=tmp_path / 'cases.jsonl',
        )
        shape = report['production_shape']
        assert shape['duplicate_attach']['strict'] == 1, 'the band winner would score 0'
        assert [
            (case['memory_id'], case['attach_target_id'])
            for case in shape['wrong_record_attach']['cases']
        ] == [('c2-dup-1', 'zz')]
        rows = [
            json.loads(line) for line in (tmp_path / 'cases.jsonl').read_text().splitlines()
        ]
        assert [row['judged_candidate_id'] for row in rows] == ['c1-canon', 'zz']


class TestRunJudgeEvalOverABuiltPlan:
    """`run_judge_eval` over plans built by eval_write_triage_cases.py.

    How a plan is built, and which plans are refused, is
    test_eval_write_triage_cases.py's subject. These pin what the runner does
    with one.
    """

    def test_a_retrieved_run_publishes_no_distractor_count(self, tmp_path: Path) -> None:
        """Distractors are the seeded construction's; a retrieved slate draws none."""
        plan, _ = _crossed_retrieved_plan()
        report = _mod().run_judge_eval(
            plan=plan, judge_fn=_fake_judge(),
            report_path=tmp_path / 'report.json', provenance={},
        )
        provenance = report['provenance']
        assert (provenance['candidate_count'], provenance['distractor_count']) == (2, None)

    def test_each_case_is_handed_its_own_rows(self, tmp_path: Path) -> None:
        """`judge_write` re-sorts a slate by its rows' cosines, so a row pooled
        from another case's retrieval would hand this prompt that query's order."""
        plan, _ = _crossed_retrieved_plan()
        handed: list[list[tuple[str, float]]] = []

        def judge_fn(case, candidates):
            handed.append(_scored(candidates))
            return OUTCOME_AMENDED

        _mod().run_judge_eval(
            plan=plan, judge_fn=judge_fn,
            report_path=tmp_path / 'report.json', provenance=dict(_PROVENANCE),
        )
        assert handed == [
            [('canon', 0.80), ('other', 0.70)],
            [('other', 0.85), ('canon', 0.75)],
        ]

    def test_the_judge_is_never_asked_and_nothing_is_written(self, tmp_path: Path) -> None:
        """Task 5277 A3: a judge-band case with an empty slate is refused before any spend."""
        judge = _fake_judge()
        with pytest.raises(ValueError):
            _run(tmp_path, judge=judge, corpus=_one_cluster_corpus(), distractors=4)
        assert judge.calls == []
        assert list(tmp_path.iterdir()) == []


class TestApplyFieldChars:
    """`--field-chars` moves the run's own in-memory `write_triage.judge_field_chars`."""

    @staticmethod
    def _resolved(config) -> int:
        from fused_memory.server.write_triage_judge import (  # noqa: PLC0415
            resolve_judge_field_chars,
        )

        return resolve_judge_field_chars(types.SimpleNamespace(config=config))

    @staticmethod
    def _config():
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

        return FusedMemoryConfig()

    def test_a_requested_width_is_written_to_the_leaf(self) -> None:
        config = self._config()
        assert _mod().apply_field_chars(config, 40) == 40
        assert self._resolved(config) == 40

    def test_zero_means_no_elision_rather_than_everything_elided(self) -> None:
        config = self._config()
        text = 'x' * 20_000
        slate = [MemoryResult(id='a', content=text, source_store=SourceStore.mem0)]
        assert _mod().apply_field_chars(config, 0) == 0
        width = self._resolved(config)
        assert width >= len(text)
        assert judge_prompt_elisions(text, slate, field_chars=width) == JudgePromptElisions(
            entry=False, candidates={'a': False},
        )

    def test_a_negative_width_is_refused_rather_than_recorded(self) -> None:
        """The resolver would fall back to the default, so provenance would lie."""
        config = self._config()
        before = config.write_triage.judge_field_chars
        with pytest.raises(ValueError, match='--field-chars'):
            _mod().apply_field_chars(config, -5)
        assert config.write_triage.judge_field_chars == before

    def test_none_changes_nothing_and_reports_the_configs_own_width(self) -> None:
        config = self._config()
        before = config.write_triage.judge_field_chars
        assert _mod().apply_field_chars(config, None) == self._resolved(config) == 4_000
        assert config.write_triage.judge_field_chars == before


# ---------------------------------------------------------------------------
# The split: one contract path, and imports that point downward only
# ---------------------------------------------------------------------------

class TestScoreAttachmentsKeepsItsContractPath:
    """PRD C2' and task 5807 call it at eval_write_triage_judge.py::score_attachments."""

    def test_the_cli_binds_the_scoring_modules_function(self) -> None:
        assert _mod().score_attachments is _scoring().score_attachments


#: Loads ONE ``scripts/`` module by path under its stem and prints the
#: ``scripts/`` stems then in ``sys.modules``, as one JSON list.
_LAYERING_PROBE = """
import importlib.util
import json
import sys
from pathlib import Path

scripts = Path('scripts')
stems = {path.stem for path in scripts.glob('*.py')}
stem = sys.argv[1]
spec = importlib.util.spec_from_file_location(stem, scripts / f'{stem}.py')
module = importlib.util.module_from_spec(spec)
sys.modules[stem] = module
spec.loader.exec_module(module)
print(json.dumps(sorted(stems & set(sys.modules))))
"""

_SCORING_CLOSURE = frozenset({'eval_write_triage_scoring', 'calibrate_write_triage'})


class TestTheEvalScriptsImportOnlyDownward:
    """Each eval module loads exactly what lies beneath it: never the CLI, never a sibling.

    The loads go by path, so no linter sees an upward or sideways one. Each
    module is loaded in its OWN fresh interpreter, so a sibling cannot hide
    behind a ``sys.modules`` entry an earlier load left, and the ``scripts/``
    modules it brings in must equal its downward closure. The CLI is not
    loaded: it may load everything.
    """

    @pytest.mark.parametrize(('stem', 'closure'), [
        ('eval_write_triage_scoring', _SCORING_CLOSURE),
        ('eval_write_triage_cases', _SCORING_CLOSURE | {'eval_write_triage_cases'}),
        ('eval_write_triage_live_judge', _SCORING_CLOSURE | {'eval_write_triage_live_judge'}),
    ])
    def test_a_module_loads_exactly_its_downward_closure(
        self, stem: str, closure: frozenset[str],
    ) -> None:
        result = subprocess.run(
            [sys.executable, '-c', _LAYERING_PROBE, stem],
            cwd=SCRIPT_PATH.parent.parent, capture_output=True, text=True, check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        loaded = set(json.loads(result.stdout))
        assert loaded == closure, (
            f'loading {stem} also loaded {sorted(loaded - closure)} '
            f'and not {sorted(closure - loaded)}'
        )
