"""Tests that the committed judge accuracy report is traceable to the run that made it.

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf γ, decision D10;
plans/write-triage-flip-readiness-prd.md C2'.

Reads only committed files: ``config/config.yaml``, the report and its
markdown sibling under ``calibration/``, the per-case dump and the curator
fixture. Nothing is executed against a network, a key or Qdrant.
"""
from __future__ import annotations

import functools
import hashlib
import json
import types
from pathlib import Path

import pytest
from _eval_write_triage_fixtures import ALIASES_PATH, FIXTURE_PATH, committed_records
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'eval_write_triage_judge.py', 'eval_write_triage_judge')


@functools.cache
def _scoring() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'eval_write_triage_scoring.py', 'eval_write_triage_scoring')


@pytest.fixture(scope='module')
def records() -> list[dict]:
    return committed_records()


class TestCommittedJudgeAccuracyReportIsTraceable:
    """A number gating a decision must be a number some run measured.

    Modelled on `test_calibrate_write_triage.py::TestCommittedCalibrationIsTraceable`,
    and for the same reason: `judge_accuracy_report_path` exists so a reader
    can get from "the judge was evaluated" back to the run that evaluated it,
    without taking anyone's word that the figures were not typed in. Committed
    artifacts only — no network, no script execution, no Qdrant.

    EXPLICITLY NO ASSERTION ON ANY ACCURACY VALUE, and that is the point of
    this docstring. D10 makes this report the operator's input at the
    task-3169 flip gate. A floor asserted here would silently BECOME that
    gate — every future run would have to clear a bar this task invented,
    pre-empting a judgment it is explicitly told to leave to a human. What is
    asserted is only that each accuracy is WELL-FORMED: `None`, or a float in
    [0.0, 1.0]. Never that it is large enough.
    """

    CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'

    # Decorator order is load-bearing: `staticmethod` OUTERMOST, so
    # `self._committed()` resolves through the descriptor and calls the cached
    # function with zero arguments. Stacked the other way, `functools.cache`'s
    # plain-function wrapper is the class attribute, binds `self` as its first
    # argument, and raises TypeError on every access. Same note as the
    # sibling suite's.
    @staticmethod
    @functools.cache
    def _committed():
        import yaml  # noqa: PLC0415

        cls = TestCommittedJudgeAccuracyReportIsTraceable
        block = yaml.safe_load(cls.CONFIG_PATH.read_text()).get('write_triage') or {}
        report_path = block.get('judge_accuracy_report_path')
        report = None
        resolved = None
        if report_path is not None:
            resolved = Path(__file__).parent.parent / report_path
            if resolved.exists():
                report = json.loads(resolved.read_text())
        return block, report, resolved

    def test_the_report_path_is_set_and_relative(self) -> None:
        block, _report, _resolved = self._committed()
        assert block.get('judge_accuracy_report_path'), (
            'a judge shipped without a measurement is an untraceable judge'
        )
        assert not Path(block['judge_accuracy_report_path']).is_absolute(), (
            'the path must not bake in the checkout it was produced in — this '
            'script runs in per-task worktrees that get reset'
        )

    def test_the_path_resolves_to_a_committed_report(self) -> None:
        block, report, resolved = self._committed()
        assert report is not None, (
            f'judge_accuracy_report_path '
            f'{block.get("judge_accuracy_report_path")!r} does not resolve to a '
            f'committed report (looked at {resolved})'
        )

    def test_the_report_carries_every_key_the_operator_reads(self) -> None:
        _block, report, _resolved = self._committed()
        assert report is not None
        assert set(report) >= {
            'per_class', 'confusion', 'duplicate_outcome_split', 'false_contested',
            'contested_ground_truth', 'caveats', 'provenance',
        }

    def test_all_four_classes_are_present_with_an_integer_n(self) -> None:
        _block, report, _resolved = self._committed()
        assert report is not None
        per_class = report['per_class']
        assert set(per_class) == set(_scoring().EVAL_CLASSES)
        for name, entry in per_class.items():
            assert isinstance(entry['n'], int), f'{name}: {entry!r}'

    def test_the_small_classes_match_the_fixtures_own_label_census(
        self, records,
    ) -> None:
        """`distinct` and `pseudo_contradiction` are the two the task names by
        hand — n=3 and n=6 today.

        Recounted from the JSONL rather than hardcoded, so editing the fixture
        surfaces as a mismatch demanding a re-measurement instead of leaving a
        stale constant agreeing with a stale report.
        """
        _block, report, _resolved = self._committed()
        assert report is not None
        for label in (_scoring().LABEL_DISTINCT, _scoring().LABEL_PSEUDO_CONTRADICTION):
            expected = sum(1 for r in records if r['label'] == label)
            assert expected > 0, f'{label} vanished from the fixture'
            assert report['per_class'][label]['n'] == expected, (
                f'{label}: report says {report["per_class"][label]["n"]}, the '
                f'fixture holds {expected} — re-run the eval'
            )

    def test_every_accuracy_is_well_formed_and_none_is_asserted_to_be_good(
        self,
    ) -> None:
        """The only shape assertion this class makes about a number."""
        _block, report, _resolved = self._committed()
        assert report is not None
        for name, entry in report['per_class'].items():
            accuracy = entry['accuracy']
            if accuracy is None:
                assert entry['n'] == 0, f'{name}: measured but unscored'
                continue
            assert isinstance(accuracy, float), f'{name}: {accuracy!r}'
            assert 0.0 <= accuracy <= 1.0, f'{name}: {accuracy!r}'

    def test_the_contested_caveat_survived_into_the_artifact(self) -> None:
        _block, report, _resolved = self._committed()
        assert report is not None
        ground_truth = report['contested_ground_truth']
        assert ground_truth['available'] is False
        assert ground_truth['reason'].split(':')[0].strip(), 'a reason code is required'

    def test_a_markdown_sibling_was_committed_beside_the_json(self) -> None:
        """The operator reads the markdown, so the artifact must exist.

        Only existence is checked. Deliberately NOT a ``CAVEATS``-identity
        check like the renderer's: ``CAVEATS`` is source, whereas this ``.md``
        is a measured artifact regenerable only by a live LLM run, so coupling
        the two would make a one-word caveat edit require a paid
        re-measurement to get back to green. The caveat's substance is pinned
        in the JSON instead, by
        :meth:`test_the_contested_caveat_survived_into_the_artifact`.
        """
        _block, report, resolved = self._committed()
        assert report is not None and resolved is not None
        sibling = resolved.with_suffix('.md')
        assert sibling.exists(), f'no markdown sibling at {sibling}'

    def test_the_committed_markdown_is_the_render_of_the_committed_json(
        self,
    ) -> None:
        """The operator reads the `.md`; the `.json` is what a reviewer diffs.

        `run_judge_eval` writes both from ONE report dict in ONE process, so
        a committed pair that disagrees provably did not come from a single
        run — and the markdown the task-3169 flip operator acts on is then
        not the render of the evidence beside it. Measured 2026-08-27: the
        committed `.json` keyed its confusion rows
        `[stored, amended, contested, restated]` while the committed `.md`
        header read `| class | stored | restated | amended | contested |`,
        which is a frozenset's iteration order moving between two processes.
        """
        _block, report, resolved = self._committed()
        assert report is not None and resolved is not None
        sibling = resolved.with_suffix('.md')
        assert sibling.exists(), f'no markdown sibling at {sibling}'
        assert sibling.read_text() == _scoring().render_markdown(report), (
            f'{sibling.name} is not the render of {resolved.name} — the '
            f'markdown the task-3169 operator reads must provably be the '
            f'render of the committed JSON, not a second artifact that drifted'
        )

    def test_the_committed_report_is_a_full_measurement(self) -> None:
        """Not a `--dry-run` stub, and not a truncated `--limit` smoke.

        The guard in the script stops the first case at write time; this
        stops BOTH at review time, so a clobbered artifact is a red test
        rather than a silent substitution of fabricated (or partial) numbers
        for the measurement the task-3169 flip gate reads.

        `judge_provider` is checked against the shipped `_KNOWN_PROVIDERS`
        tuple, which the `dry-run` sentinel is deliberately not a member of.
        Still no assertion on any accuracy VALUE — this is about whether a
        run happened, never about whether it scored well.
        """
        from fused_memory.server import write_triage_judge  # noqa: PLC0415

        _block, report, _resolved = self._committed()
        assert report is not None
        provenance = report['provenance']
        assert provenance['limit'] is None, (
            f'the committed report is a --limit {provenance["limit"]!r} smoke, '
            f'not the corpus-wide measurement — re-run the eval without '
            f'--limit. `guard_committed_report` only WARNS on a --limit run '
            f'(deliberately: its numbers are the judge\'s own, just partial), '
            f'so nothing but this assertion stands between a partial artifact '
            f'and the task-3169 flip gate'
        )
        assert provenance['case_count'] == sum(
            entry['n'] for entry in report['per_class'].values()
        ), (
            'the population the provenance CLAIMS was measured and the one the '
            'per-class table actually scores disagree — the artifact is not '
            'internally consistent'
        )
        assert provenance['judge_provider'] in write_triage_judge._KNOWN_PROVIDERS, (
            f'judge_provider {provenance["judge_provider"]!r} is not a real '
            f'provider {list(write_triage_judge._KNOWN_PROVIDERS)} — this '
            f'artifact was written by a stub, not measured'
        )

    def test_no_provenance_key_outside_the_vocabulary(self) -> None:
        """Nothing the artifact carries may be UNKNOWN.

        A RENAMED key would otherwise sit in the artifact under its old
        spelling and read as disclosed, so every key the file carries has to
        still be vocabulary.
        """
        _block, report, _resolved = self._committed()
        assert report is not None
        stray = set(report['provenance']) - set(_scoring().PROVENANCE_KEYS)
        assert not stray, (
            f'the committed report carries {sorted(stray)}, which PROVENANCE_KEYS '
            f'no longer names — a renamed key reads as disclosed and is not'
        )

    def test_the_committed_provenance_carries_the_whole_vocabulary(self) -> None:
        """A field added to `PROVENANCE_KEYS` must reach the artifact too.

        `build_report` backfills every key in the vocabulary, so a FRESH
        report cannot omit one — `test_provenance_carries_every_key_even_when_nothing_was_measured`
        pins that. The committed artifact is the gap: it is regenerated only
        by a paid live run, so a field added afterwards leaves the file the
        task-3169 operator actually reads silent on it, with the per-key
        assertions below unable to notice because they each name one key by
        hand. Measured 2026-08-27: `judge_candidate_count` and `judge_enabled`
        were added and the committed report still carried the ten keys that
        predated them.

        A superset check: keys beyond the vocabulary are
        `test_no_provenance_key_outside_the_vocabulary`'s concern.
        """
        _block, report, _resolved = self._committed()
        assert report is not None
        assert set(_scoring().PROVENANCE_KEYS) <= set(report['provenance']), (
            f'the committed report does not disclose '
            f'{sorted(set(_scoring().PROVENANCE_KEYS) - set(report["provenance"]))} — '
            f're-run the eval'
        )

    def test_the_committed_report_says_the_judge_arm_was_live(self) -> None:
        """The field exists because a disabled judge reads like a measurement.

        `judge_write` returns `stored` on its first line when the kill switch
        is off: every case answered with no provider call, `distractor`
        scoring 1.0 and `duplicate` 0.0, and an artifact otherwise
        indistinguishable from the corpus-wide run the flip gate reads. An
        artifact silent on the switch reproduces exactly the hazard the field
        was added to close.

        `judge_candidate_count` is asserted only for WELL-FORMEDNESS, never
        against the shipped config value: the knob is hot-reloadable, and
        binding the two would make an operator's config edit demand a paid
        re-measurement to get back to green.
        """
        _block, report, _resolved = self._committed()
        assert report is not None
        provenance = report['provenance']
        assert provenance['judge_enabled'] is True, (
            f'judge_enabled={provenance["judge_enabled"]!r}: this artifact was '
            f'produced with the judge arm off, so its per-class figures are '
            f'the kill switch being measured, not the judge'
        )
        cap = provenance['judge_candidate_count']
        assert isinstance(cap, int) and not isinstance(cap, bool) and cap > 0, (
            f'judge_candidate_count={cap!r} — the width the model actually saw '
            f'must be a positive integer'
        )

    def test_the_published_slate_width_is_one_the_model_could_actually_see(
        self,
    ) -> None:
        """`candidate_count` must not overstate the evidence the judge had.

        Two different numbers, and the artifact publishes both:
        `candidate_count` is the widest slate `build_judge_cases` BUILT, while
        `judge_candidate_count` is the cap `judge_write` re-trims to before
        the prompt is composed. When the first exceeds the second the model
        saw the narrower slate, and a reader taking `candidate_count` as "what
        the judge was shown" is reading an overstatement of the evidence base.

        `run_judge_eval` already notices this — it logs "slate widths ran N..M
        but judge_candidate_count caps the prompt at K". A log line is read
        once, by whoever happened to be watching the run; the artifact is read
        at the task-3169 flip gate, months later, by someone who was not. This
        is the assertion that moves the property from the log to the artifact.

        Silent today because the shipped cap (5) and the default
        `--distractors 4` agree, so this is GREEN ON ARRIVAL — it exists to
        own a property nothing owned, and to check the regeneration rather
        than merely follow it.

        DERIVED FROM THE ARTIFACT ALONE. Both numbers come from the same
        provenance block, so this cannot go red because someone edited a
        hot-reloadable knob in `config.yaml` after the run — the hazard the
        sibling test above declines the config comparison to avoid, and the
        same one that keeps `CAVEATS` uncoupled from the committed report.
        """
        _block, report, _resolved = self._committed()
        assert report is not None
        provenance = report['provenance']
        built = provenance['candidate_count']
        cap = provenance['judge_candidate_count']
        assert built <= cap, (
            f'the report publishes candidate_count={built} but the judge '
            f'trimmed every prompt to judge_candidate_count={cap} — the model '
            f'never saw {built} candidates, so the artifact overstates the '
            f'evidence the task-3169 operator is reading it for'
        )

    def test_provenance_names_the_model_and_the_fixture(self) -> None:
        _block, report, _resolved = self._committed()
        assert report is not None
        provenance = report['provenance']
        for key in ('judge_provider', 'judge_model'):
            assert isinstance(provenance[key], str) and provenance[key], key
        fixture = Path(__file__).parent.parent / provenance['fixture_path']
        assert fixture.resolve() == FIXTURE_PATH.resolve(), (
            f'the report measured {provenance["fixture_path"]!r}, not the '
            f'committed fixture'
        )

    def test_the_committed_report_is_the_retrieved_slate_arbiter(self) -> None:
        """PRD C2': `retrieved` is the arbiter; `seeded` reproduces history."""
        _block, report, _resolved = self._committed()
        assert report is not None
        assert report['provenance']['slate_mode'] == _scoring().SLATE_RETRIEVED

    def test_the_committed_report_measured_the_shipped_system_prompt(self) -> None:
        """The prompt is source, not a hot-reloadable knob, so an artifact that
        measured another prompt does not describe production (PRD C2').

        Unlike `judge_candidate_count`, which the sibling test above
        deliberately leaves unbound to the shipped config.
        """
        from fused_memory.server.write_triage_judge import (  # noqa: PLC0415
            JUDGE_SYSTEM_PROMPT,
        )

        _block, report, _resolved = self._committed()
        assert report is not None
        shipped = hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode('utf-8')).hexdigest()
        measured = report['provenance'].get('judge_system_prompt_sha256')
        assert measured == shipped, (
            f'the committed report measured system prompt {measured!r}, but the '
            f'shipped JUDGE_SYSTEM_PROMPT hashes to {shipped!r} — re-run the '
            f'committed arbiter command from the module docstring of '
            f'scripts/eval_write_triage_judge.py'
        )

    def test_every_duplicate_is_in_the_attach_population(self, records) -> None:
        """Recounted from the fixture rather than hardcoded (75 today; gate Γ1 reads it)."""
        _block, report, _resolved = self._committed()
        assert report is not None
        expected = sum(1 for r in records if r['label'] == _scoring().LABEL_DUPLICATE)
        assert report['production_shape']['duplicate_attach']['n'] == expected

    def test_the_alias_sidecar_is_named_and_committed(self) -> None:
        _block, report, _resolved = self._committed()
        assert report is not None
        named = report['provenance']['canonical_aliases_path']
        assert named == 'tests/fixtures/write_triage_calibration.canonical_aliases.json'
        assert (Path(__file__).parent.parent / named).resolve() == ALIASES_PATH.resolve()
        assert ALIASES_PATH.exists()

    def test_the_production_shape_is_the_score_of_the_committed_cases(self) -> None:
        """C2' computes `production_shape` ONCE, by `score_attachments`, and the
        per-case evidence it was computed from is committed beside it."""
        _block, report, _resolved = self._committed()
        assert report is not None
        provenance = report['provenance']
        assert isinstance(provenance['cases_path'], str), 'no per-case evidence is named'
        cases = Path(__file__).parent.parent / provenance['cases_path']
        assert cases.exists(), f'{provenance["cases_path"]} is not committed'
        rows = [json.loads(line) for line in cases.read_text().splitlines()]
        assert len(rows) == provenance['case_count']
        assert _mod().score_attachments(rows) == report['production_shape']

    def test_no_retrieval_came_back_degraded(self) -> None:
        """Each degraded retrieval is a fail-open that measured nothing."""
        _block, report, _resolved = self._committed()
        assert report is not None
        assert report['provenance']['degraded_retrievals'] == 0

    def test_the_arbiter_publishes_no_distractor_count(self) -> None:
        """A retrieved slate draws no distractors, so it must not report a count."""
        _block, report, _resolved = self._committed()
        assert report is not None
        assert report['provenance']['distractor_count'] is None

    def test_a_null_attach_target_cluster_means_not_a_fixture_record(self, records) -> None:
        """Every dumped target that IS a fixture record names its cluster and label."""
        _block, report, _resolved = self._committed()
        assert report is not None
        fixture = {record['memory_id']: record for record in records}
        cases = Path(__file__).parent.parent / report['provenance']['cases_path']
        for row in (json.loads(line) for line in cases.read_text().splitlines()):
            target = fixture.get(row['attach_target_id'])
            assert (row['attach_target_cluster_id'], row['attach_target_label']) == (
                (target['cluster_id'], target['label']) if target else (None, None)
            ), row['memory_id']
