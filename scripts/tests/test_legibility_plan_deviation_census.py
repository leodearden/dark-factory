"""Tests for scripts/legibility/plan_deviation_census.py — the nightly
plan-deviation census (plans/plan-deviation-recording-prd.md D8-D11).

Imports use the package form, `from legibility import plan_deviation_census
as mod` (scripts/tests/conftest.py puts scripts/ on sys.path), so every class
this file builds is the very class the module uses.

Every MCP side effect (`call_tool`) is ALWAYS a recording fake in this file:
no test reaches a real escalation server. State files land under the per-test
legibility state root conftest.py's autouse fixture installs.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from escalation.disposition import ObservationConsumed
from escalation.models import NOTE_TO_FILE_CLASS, Escalation
from legibility import config as config_mod
from legibility import plan_deviation_census as mod
from shared.plan_deviation import DEVIATION_KINDS, DeviationRecord

_LOGGER = 'legibility.plan_deviation_census'

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
HORIZON = NOW - timedelta(days=28)

SHIPPED_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / 'docs' / 'legibility' / 'legibility.yaml'
)


def _write_config(tmp_path: Path, *, block: object = None) -> Path:
    doc: dict[str, object] = {
        'project_id': 'testproj',
        'project_root': str(tmp_path),
        'escalation_port': 8199,
        'cwd_prefixes': [str(tmp_path / 'work')],
    }
    if block is not None:
        doc['plan_deviation_census'] = block
    path = tmp_path / 'legibility.yaml'
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    return path


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == _LOGGER and record.levelno == logging.WARNING
    ]


# ---------------------------------------------------------------------------
# settings validation — settings_from_mapping never raises and falls back
# per field, reporting every rejection in one WARNING
# ---------------------------------------------------------------------------

def test_settings_none_returns_defaults_silently(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)

    settings = mod.settings_from_mapping(None)

    assert settings == config_mod.PlanDeviationCensus()
    assert settings.enabled is True
    assert settings.rate_rise_factor == 2.0
    assert settings.min_records == 10
    assert settings.pattern_share == 0.5
    assert settings.prose_only_min == 3
    assert settings.cooldown_days == 7
    assert settings.horizon_days == 28
    assert settings.meta_roots == ('.worktrees/.task-meta',)
    assert _warnings(caplog) == []


def test_settings_valid_overrides_apply_verbatim(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)

    settings = mod.settings_from_mapping({
        'enabled': False,
        'rate_rise_factor': 3,
        'min_records': 5,
        'pattern_share': 0.6,
        'prose_only_min': 2,
        'cooldown_days': 3,
        'horizon_days': 35,
        'meta_roots': ['a', '/abs/b'],
    })

    assert isinstance(settings, config_mod.PlanDeviationCensus)
    assert settings.enabled is False
    assert settings.rate_rise_factor == 3
    assert settings.min_records == 5
    assert settings.pattern_share == 0.6
    assert settings.prose_only_min == 2
    assert settings.cooldown_days == 3
    assert settings.horizon_days == 35
    assert settings.meta_roots == ('a', '/abs/b')
    assert _warnings(caplog) == []


def test_settings_bad_fields_fall_back_per_field_in_one_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    defaults = config_mod.PlanDeviationCensus()

    settings = mod.settings_from_mapping({
        'rate_rise_factor': '2.0',
        'min_records': True,
        'pattern_share': 1.5,
        'prose_only_min': -1,
        'horizon_days': 7,
        'meta_roots': 'x',
        'enabled': 'yes',
        'rate_rise_factr': 4.0,
        'cooldown_days': 4,
    })

    assert settings.rate_rise_factor == defaults.rate_rise_factor
    assert settings.min_records == defaults.min_records
    assert settings.pattern_share == defaults.pattern_share
    assert settings.prose_only_min == defaults.prose_only_min
    assert settings.horizon_days == defaults.horizon_days
    assert settings.meta_roots == defaults.meta_roots
    assert settings.enabled == defaults.enabled
    assert settings.cooldown_days == 4
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    for key in (
        'rate_rise_factor', 'min_records', 'pattern_share', 'prose_only_min',
        'horizon_days', 'meta_roots', 'enabled', 'rate_rise_factr',
    ):
        assert key in message
    assert 'cooldown_days' not in message


def test_settings_zero_cooldown_is_rejected(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)

    settings = mod.settings_from_mapping({'cooldown_days': 0})

    assert settings.cooldown_days == config_mod.PlanDeviationCensus().cooldown_days
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert 'cooldown_days' in warnings[0].getMessage()


def test_settings_non_mapping_block_returns_defaults_with_one_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)

    settings = mod.settings_from_mapping(['enabled', False])

    assert settings == config_mod.PlanDeviationCensus()
    assert len(_warnings(caplog)) == 1


def test_garbage_block_never_fails_config_load(tmp_path):
    block = {
        'enabled': 'sometimes',
        'rate_rise_factor': None,
        'min_records': [1, 2],
        'meta_roots': {'not': 'a list'},
    }

    cfg = config_mod.load_config(_write_config(tmp_path, block=block))

    assert cfg.plan_deviation_census == block


def test_shipped_config_block_is_accepted_without_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    cfg = config_mod.load_config(SHIPPED_CONFIG_PATH)

    assert isinstance(cfg.plan_deviation_census, dict)
    settings = mod.settings_from_mapping(cfg.plan_deviation_census)

    assert settings == config_mod.PlanDeviationCensus()
    assert _warnings(caplog) == []
    assert 'plan_deviation_census:' in SHIPPED_CONFIG_PATH.read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# readers — plan.json step deviations (stat before read) and archived
# plan-deviation notes in the escalation queue
# ---------------------------------------------------------------------------

def _deviation(
    *, recorded_at: datetime, kind: str = 'wrong_inference', prose_only: bool = False,
) -> dict:
    return DeviationRecord(
        kind=kind,
        observed='observed',
        expected='expected',
        action_taken='action',
        prose_only=prose_only,
        note='note',
        recorded_at=recorded_at.isoformat(),
        recorded_by=None,
    ).to_dict()


def _set_mtime(path: Path, mtime: datetime) -> None:
    os.utime(path, (mtime.timestamp(), mtime.timestamp()))


def _write_plan(
    root: Path,
    lane: str,
    *,
    steps: Sequence[Sequence[object]] = (),
    prerequisites: Sequence[Sequence[object]] = (),
    mtime: datetime = NOW,
) -> Path:
    """A plan.json under ``<root>/<lane>/`` whose items each carry one
    ``deviations`` list; its ``task_id`` deliberately differs from *lane*."""
    path = root / lane / 'plan.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        'task_id': f'doc-{lane}',
        'prerequisites': [
            {'id': f'pre-{n}', 'deviations': list(entries)}
            for n, entries in enumerate(prerequisites, start=1)
        ],
        'steps': [
            {'id': f'step-{n}', 'deviations': list(entries)}
            for n, entries in enumerate(steps, start=1)
        ],
    }
    path.write_text(json.dumps(doc), encoding='utf-8')
    _set_mtime(path, mtime)
    return path


def _write_raw_plan(root: Path, lane: str, data: bytes, *, mtime: datetime) -> Path:
    path = root / lane / 'plan.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    _set_mtime(path, mtime)
    return path


def _sorted(records: Sequence[mod.CensusRecord]) -> list[mod.CensusRecord]:
    return sorted(records, key=lambda r: (r.task_id, r.recorded_at))


def test_plan_reader_reads_step_and_prerequisite_deviations_across_roots(tmp_path):
    root_a = tmp_path / '.worktrees' / '.task-meta'
    root_b = tmp_path / 'other-meta'
    step_at = NOW - timedelta(days=1)
    pre_at = NOW - timedelta(days=2)
    other_at = NOW - timedelta(days=3)
    _write_plan(
        root_a, '101',
        steps=[[_deviation(recorded_at=step_at, kind='wrong_inference')], []],
        prerequisites=[[_deviation(recorded_at=pre_at, kind='other', prose_only=True)]],
    )
    _write_plan(root_b, '102', steps=[[_deviation(recorded_at=other_at, kind='never_checked_fact')]])
    _write_plan(root_b, '103')

    corpus = mod.read_plan_deviations([root_a, root_b], horizon=HORIZON)

    assert _sorted(corpus.records) == [
        mod.CensusRecord(
            task_id='101', recorded_at=pre_at, kind='other', prose_only=True,
            source=mod.RecordSource.PLAN_STEP,
        ),
        mod.CensusRecord(
            task_id='101', recorded_at=step_at, kind='wrong_inference', prose_only=False,
            source=mod.RecordSource.PLAN_STEP,
        ),
        mod.CensusRecord(
            task_id='102', recorded_at=other_at, kind='never_checked_fact', prose_only=False,
            source=mod.RecordSource.PLAN_STEP,
        ),
    ]
    for record in corpus.records:
        assert record.recorded_at.utcoffset() == timedelta(0)
    assert corpus.opened == 3
    assert corpus.skipped == 0


def test_plan_reader_skips_files_older_than_the_horizon_without_opening_them(tmp_path):
    root = tmp_path / '.worktrees' / '.task-meta'
    stale = HORIZON - timedelta(days=1)
    _write_plan(
        root, '201',
        steps=[[_deviation(recorded_at=NOW - timedelta(days=1))]],
        mtime=stale,
    )
    _write_raw_plan(root, '202', b'\xff not json at all {', mtime=stale)
    _write_plan(root, '203', steps=[[_deviation(recorded_at=NOW - timedelta(days=1))]])

    corpus = mod.read_plan_deviations([root], horizon=HORIZON)

    assert [record.task_id for record in corpus.records] == ['203']
    assert corpus.opened == 1
    assert corpus.skipped == 0


def test_plan_reader_counts_bad_files_and_bad_entries_without_aborting(tmp_path):
    root = tmp_path / '.worktrees' / '.task-meta'
    kept_at = NOW - timedelta(days=1)
    _write_raw_plan(root, '301', b'{"steps": [', mtime=NOW)
    _write_raw_plan(root, '302', b'\xff\xfe\x00garbage', mtime=NOW)
    _write_plan(
        root, '303',
        steps=[[
            'not a dict',
            {**_deviation(recorded_at=kept_at), 'recorded_at': 'not-a-date'},
            {**_deviation(recorded_at=kept_at), 'kind': 'bogus_kind'},
            {**_deviation(recorded_at=kept_at), 'prose_only': 'yes'},
            _deviation(recorded_at=kept_at, kind='prediction_falsified'),
        ]],
    )

    corpus = mod.read_plan_deviations([root], horizon=HORIZON)

    assert corpus.records == (
        mod.CensusRecord(
            task_id='303', recorded_at=kept_at, kind='prediction_falsified',
            prose_only=False, source=mod.RecordSource.PLAN_STEP,
        ),
    )
    assert corpus.opened == 1
    assert corpus.skipped == 6


def test_plan_reader_absent_root_is_an_empty_corpus(tmp_path):
    corpus = mod.read_plan_deviations([tmp_path / 'no-such-root'], horizon=HORIZON)

    assert corpus == mod.CorpusRead(records=(), opened=0, skipped=0)


def _escalation(
    esc_id: str,
    *,
    task_id: str,
    agent_role: str = 'implementer',
    severity: str = 'info',
    category: str = 'design_concern',
    resolution_class: str | None = None,
    status: str = 'pending',
    timestamp: datetime,
) -> Escalation:
    return Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role=agent_role,
        severity=severity,
        category=category,
        summary=f'summary of {esc_id}',
        timestamp=timestamp.isoformat(),
        status=status,
        resolution_class=resolution_class,
        note_to_file=resolution_class == NOTE_TO_FILE_CLASS,
    )


def _write_escalation(directory: Path, esc: Escalation) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{esc.id}.json'
    path.write_text(esc.to_json(), encoding='utf-8')
    return path


def test_note_reader_counts_noted_classes_and_the_implementer_fallback(tmp_path):
    queue_dir = tmp_path / 'data' / 'escalations'
    archive = queue_dir / 'archive'
    stamps = {name: NOW - timedelta(days=n) for n, name in enumerate(
        ['n2f', 'obs', 'impl-pending', 'impl-resolved', 'blk', 'arch', 'infra'], start=1,
    )}
    counted = [
        (archive / '2026-10-20', _escalation(
            'esc-401-1', task_id='401', agent_role='reviewer',
            resolution_class=NOTE_TO_FILE_CLASS, status='resolved', timestamp=stamps['n2f'],
        )),
        (archive / '2026-10-20', _escalation(
            'esc-402-1', task_id='402', agent_role='architect',
            resolution_class=ObservationConsumed.resolution_class, status='resolved',
            timestamp=stamps['obs'],
        )),
        (queue_dir, _escalation('esc-403-1', task_id='403', timestamp=stamps['impl-pending'])),
        (archive / '2026-10-21', _escalation(
            'esc-404-1', task_id='404', status='resolved', timestamp=stamps['impl-resolved'],
        )),
    ]
    not_counted = [
        (queue_dir, _escalation(
            'esc-405-1', task_id='405', severity='blocking', timestamp=stamps['blk'],
        )),
        (archive / '2026-10-21', _escalation(
            'esc-406-1', task_id='406', agent_role='architect', status='resolved',
            timestamp=stamps['arch'],
        )),
        (queue_dir, _escalation(
            'esc-407-1', task_id='407', category='infra_issue', timestamp=stamps['infra'],
        )),
    ]
    for directory, esc in [*counted, *not_counted]:
        _write_escalation(directory, esc)

    corpus = mod.read_archived_notes(queue_dir)

    assert _sorted(corpus.records) == [
        mod.CensusRecord(
            task_id=esc.task_id, recorded_at=datetime.fromisoformat(esc.timestamp),
            kind=None, prose_only=False, source=mod.RecordSource.ESCALATION,
        )
        for _, esc in counted
    ]
    assert corpus.opened == 7
    assert corpus.skipped == 0


def test_note_reader_counts_an_unparsable_record_as_skipped(tmp_path):
    queue_dir = tmp_path / 'data' / 'escalations'
    queue_dir.mkdir(parents=True)
    (queue_dir / 'esc-501-1.json').write_text('{"id": ', encoding='utf-8')
    _write_escalation(queue_dir, _escalation(
        'esc-502-1', task_id='502', timestamp=NOW - timedelta(days=1),
    ))

    corpus = mod.read_archived_notes(queue_dir)

    assert [record.task_id for record in corpus.records] == ['502']
    assert corpus.skipped == 1


def test_note_reader_absent_queue_dir_is_an_empty_corpus(tmp_path):
    corpus = mod.read_archived_notes(tmp_path / 'data' / 'escalations')

    assert corpus == mod.CorpusRead(records=(), opened=0, skipped=0)


# ---------------------------------------------------------------------------
# evaluate — pure; three independent triggers over windows measured from now
# ---------------------------------------------------------------------------

_KINDS = sorted(DEVIATION_KINDS)


def _daily(
    per_day: int,
    days: range,
    *,
    kind: str | None = None,
    kindless: bool = False,
    prose_only: bool = False,
    task_id: str = 'task-a',
) -> list[mod.CensusRecord]:
    """*per_day* records on each day *d* in *days*, at ``now - d days - (1+k) h``.
    Kinds run round-robin over the vocabulary unless *kind* or *kindless* is
    given; a kindless record comes from the escalation source."""
    return [
        mod.CensusRecord(
            task_id=task_id,
            recorded_at=NOW - timedelta(days=d, hours=1 + k),
            kind=None if kindless else (kind or _KINDS[k % len(_KINDS)]),
            prose_only=prose_only,
            source=mod.RecordSource.ESCALATION if kindless else mod.RecordSource.PLAN_STEP,
        )
        for d in days
        for k in range(per_day)
    ]


def _at(stamp: datetime, *, task_id: str = 'task-a') -> mod.CensusRecord:
    return mod.CensusRecord(
        task_id=task_id, recorded_at=stamp, kind='other', prose_only=False,
        source=mod.RecordSource.PLAN_STEP,
    )


def _evaluate(records: Sequence[mod.CensusRecord], *, now: datetime = NOW) -> mod.Verdict:
    return mod.evaluate(records, now=now, config=config_mod.PlanDeviationCensus())


def _evidence(verdict: mod.Verdict, ref: str) -> dict:
    (entry,) = [entry for entry in verdict.evidence if entry['ref'] == ref]
    return entry


BASELINE_DAYS = range(7, 28)
RECENT_DAYS = range(0, 7)


def test_evaluate_positive_control_fires_on_a_rate_rise():
    verdict = _evaluate(_daily(3, BASELINE_DAYS) + _daily(8, RECENT_DAYS))

    assert verdict.fire is True
    window = _evidence(verdict, 'window')
    assert window['recent_count'] == 56
    assert window['recent_mean'] == 8.0
    assert window['baseline_count'] == 63
    assert window['baseline_mean'] == 3.0
    assert any(reason.startswith('rate-rise') and reason.endswith(' -> FIRE')
               for reason in verdict.reasons)
    for entry in verdict.evidence:
        assert {'observation', 'measured_at', 'ref'} <= entry.keys()
        assert entry['measured_at'] == NOW.isoformat()


def test_evaluate_negative_control_flat_rate_does_not_fire():
    verdict = _evaluate(_daily(3, range(0, 28)))

    assert verdict.fire is False
    assert len(verdict.reasons) == 3
    assert not any(reason.endswith(' -> FIRE') for reason in verdict.reasons)


def test_evaluate_rate_rise_below_min_records_does_not_fire():
    verdict = _evaluate(_daily(1, range(0, 5)))

    assert verdict.fire is False
    assert _evidence(verdict, 'window')['recent_count'] == 5


def test_evaluate_one_dominant_kind_fires_the_pattern_trigger():
    baseline = _daily(1, BASELINE_DAYS)
    recent = (
        _daily(1, range(0, 7), kind='wrong_inference')
        + _daily(1, range(0, 3), kind='wrong_inference')
        + _daily(1, range(3, 5), kind='other')
    )
    assert len(recent) == 12

    verdict = _evaluate(baseline + recent)

    assert verdict.fire is True
    kinds = _evidence(verdict, 'kinds')
    assert set(kinds['kind_counts']) == DEVIATION_KINDS
    assert kinds['kind_counts']['wrong_inference'] == 10
    assert kinds['kind_counts']['never_checked_fact'] == 0
    rate_rise = [reason for reason in verdict.reasons if reason.startswith('rate-rise')]
    assert rate_rise and not rate_rise[0].endswith(' -> FIRE')


def test_evaluate_dominant_kind_below_min_records_does_not_fire():
    verdict = _evaluate(_daily(1, BASELINE_DAYS) + _daily(1, range(0, 6), kind='other'))

    assert verdict.fire is False


def test_evaluate_kindless_records_never_form_a_pattern():
    verdict = _evaluate(_daily(3, range(0, 28), kindless=True))

    assert verdict.fire is False
    assert _evidence(verdict, 'kinds')['unkinded'] == 21


@pytest.mark.parametrize(('prose_only_count', 'fires'), [(3, True), (2, False)])
def test_evaluate_prose_only_records_fire_at_the_minimum(prose_only_count, fires):
    flat = _daily(3, range(0, 28))
    prose = _daily(1, range(0, prose_only_count), kind='other', prose_only=True)

    verdict = _evaluate(flat + prose)

    assert verdict.fire is fires
    assert _evidence(verdict, 'prose_only')['prose_only_count'] == prose_only_count


def test_evaluate_windows_are_measured_from_now_not_a_day_boundary():
    second = timedelta(seconds=1)
    records = [
        _at(NOW + second),
        _at(NOW - timedelta(days=28) - second),
        _at(NOW - timedelta(days=7) + second),
        _at(NOW - timedelta(days=7) - second),
    ]

    window = _evidence(_evaluate(records), 'window')

    assert window['recent_count'] == 1
    assert window['baseline_count'] == 1


def test_evaluate_naive_now_is_read_as_utc():
    records = _daily(3, BASELINE_DAYS) + _daily(8, RECENT_DAYS)

    assert _evaluate(records, now=NOW.replace(tzinfo=None)) == _evaluate(records, now=NOW)


def test_evaluate_top_task_ids_are_the_five_busiest_recent_tasks():
    counts = {'t1': 1, 't2': 5, 't3': 3, 't4': 3, 't5': 7, 't6': 2, 't7': 4}
    records = [
        _at(NOW - timedelta(hours=1 + n), task_id=task_id)
        for task_id, count in counts.items()
        for n in range(count)
    ]

    top = _evidence(_evaluate(records), 'top_task_ids')['top_task_ids']

    assert top == [
        {'task_id': 't5', 'count': 7},
        {'task_id': 't2', 'count': 5},
        {'task_id': 't7', 'count': 4},
        {'task_id': 't3', 'count': 3},
        {'task_id': 't4', 'count': 3},
    ]
