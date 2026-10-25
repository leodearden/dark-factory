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
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path

import pytest
import yaml
from escalation.disposition import ObservationConsumed
from escalation.models import NOTE_TO_FILE_CLASS, Escalation
from legibility import config as config_mod
from legibility import nightly, state_docs, trickle_state
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


# ---------------------------------------------------------------------------
# run — readers -> evaluate -> cooldown -> pending check -> filing -> state,
# one journal line, never raises
# ---------------------------------------------------------------------------

PROJECT_ID = 'testproj'
SYNTHETIC_TASK_ID = f'plan-deviation-census-{PROJECT_ID}'
MINTED_ID = f'esc-{SYNTHETIC_TASK_ID}-1'
_FIRE_PREFIX = 'plan-deviation census: FIRE -- '
_NO_FIRE_PREFIX = 'plan-deviation census: NO-FIRE -- '


class _FakeCallTool:
    """Records every MCP call; answers each tool from a canned response, or
    raises it when the canned response is an exception."""

    def __init__(self, **responses: object) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.responses: dict[str, object] = {
            'get_pending_escalations': {'result': []},
            'escalate_info': {'id': MINTED_ID, 'status': 'queued', 'level': 0},
            **responses,
        }

    def __call__(self, tool_name: str, arguments: dict) -> object:
        self.calls.append((tool_name, arguments))
        response = self.responses[tool_name]
        if isinstance(response, BaseException):
            raise response
        return response

    @property
    def tool_names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _load_cfg(tmp_path: Path, *, block: object = None) -> config_mod.LegibilityConfig:
    return config_mod.load_config(_write_config(tmp_path, block=block))


def _meta_root(tmp_path: Path) -> Path:
    return tmp_path / '.worktrees' / '.task-meta'


def _write_daily_plans(tmp_path: Path, *, baseline_per_day: int, recent_per_day: int) -> None:
    """Step deviations at the given daily rates, spread one lane per slot of
    the day so several plans contribute and kinds run round-robin."""
    lanes: dict[int, list[dict]] = {}
    for d in range(28):
        per_day = recent_per_day if d < 7 else baseline_per_day
        for k in range(per_day):
            lanes.setdefault(k, []).append(_deviation(
                recorded_at=NOW - timedelta(days=d, hours=1 + k), kind=_KINDS[k % len(_KINDS)],
            ))
    for k, entries in lanes.items():
        _write_plan(_meta_root(tmp_path), str(6000 + k), steps=[entries])


def _write_positive_control(tmp_path: Path) -> None:
    _write_daily_plans(tmp_path, baseline_per_day=3, recent_per_day=8)


def _census_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == _LOGGER and record.getMessage().startswith('plan-deviation census: ')
    ]


def _run(
    tmp_path: Path,
    call_tool: _FakeCallTool,
    *,
    block: object = None,
    state_path: Path | None = None,
) -> str:
    return mod.run(
        _load_cfg(tmp_path, block=block), now=NOW, call_tool=call_tool, state_path=state_path,
    )


def _seed_state(path: Path, *, last_filed_at: datetime, escalation_id: str = 'esc-old-1') -> None:
    state_docs.write_state(
        path, mod.CensusState(last_filed_at=last_filed_at, escalation_id=escalation_id),
    )


def test_run_positive_control_files_one_blocking_escalation(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    _write_positive_control(tmp_path)
    fake = _FakeCallTool()

    line = _run(tmp_path, fake)

    assert fake.tool_names == ['get_pending_escalations', 'escalate_info']
    pending_args = fake.calls[0][1]
    assert pending_args['task_id'] == SYNTHETIC_TASK_ID
    filed = fake.calls[1][1]
    json.dumps(filed)
    assert filed['task_id'] == SYNTHETIC_TASK_ID
    assert filed['agent_role'] == 'plan-deviation-census'
    assert filed['category'] == 'design_concern'
    assert filed['severity'] == 'blocking'
    assert isinstance(filed['evidence'], list)
    (window,) = [entry for entry in filed['evidence'] if entry['ref'] == 'window']
    assert window['recent_mean'] == 8.0
    assert window['baseline_mean'] == 3.0
    assert line.startswith(_FIRE_PREFIX)
    assert MINTED_ID in line

    state_path = mod.default_state_path(PROJECT_ID)
    assert state_path.parent == trickle_state.project_state_dir(PROJECT_ID)
    state = json.loads(state_path.read_text(encoding='utf-8'))
    assert datetime.fromisoformat(state['last_filed_at']) == NOW
    assert state['escalation_id'] == MINTED_ID

    records = _census_records(caplog)
    assert [record.getMessage() for record in records] == [line]
    assert records[0].levelno == logging.INFO


def test_run_negative_control_files_nothing(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    _write_daily_plans(tmp_path, baseline_per_day=3, recent_per_day=3)
    fake = _FakeCallTool()

    line = _run(tmp_path, fake)

    assert fake.calls == []
    assert line.startswith(_NO_FIRE_PREFIX)
    assert not mod.default_state_path(PROJECT_ID).exists()
    assert [record.getMessage() for record in _census_records(caplog)] == [line]


def test_run_reads_both_corpora_into_one_count(tmp_path):
    _write_plan(_meta_root(tmp_path), '7001', steps=[[_deviation(recorded_at=NOW - timedelta(days=1))]])
    _write_escalation(tmp_path / 'data' / 'escalations' / 'archive' / '2026-10-24', _escalation(
        'esc-7001-1', task_id='7001', resolution_class=NOTE_TO_FILE_CLASS, status='resolved',
        timestamp=NOW - timedelta(days=1),
    ))

    line = _run(tmp_path, _FakeCallTool())

    assert line.startswith(_NO_FIRE_PREFIX)
    assert 'corpus: 1 plan-step + 1 escalation record(s), 0 skipped' in line


def test_run_inside_the_cooldown_makes_no_call(tmp_path):
    _write_positive_control(tmp_path)
    state_path = mod.default_state_path(PROJECT_ID)
    _seed_state(state_path, last_filed_at=NOW - timedelta(days=2))
    before = state_path.read_bytes()
    fake = _FakeCallTool()

    line = _run(tmp_path, fake)

    assert fake.calls == []
    assert line.startswith(_FIRE_PREFIX)
    assert 'cooldown' in line
    assert 'esc-old-1' in line
    assert state_path.read_bytes() == before


def test_run_after_the_cooldown_files_again(tmp_path):
    _write_positive_control(tmp_path)
    state_path = mod.default_state_path(PROJECT_ID)
    _seed_state(state_path, last_filed_at=NOW - timedelta(days=8))
    fake = _FakeCallTool()

    line = _run(tmp_path, fake)

    assert fake.tool_names == ['get_pending_escalations', 'escalate_info']
    assert MINTED_ID in line
    assert json.loads(state_path.read_text(encoding='utf-8'))['escalation_id'] == MINTED_ID


def test_run_a_pending_census_escalation_blocks_a_refile(tmp_path):
    _write_positive_control(tmp_path)
    fake = _FakeCallTool(get_pending_escalations={
        'result': [{'id': 'esc-x', 'task_id': SYNTHETIC_TASK_ID, 'status': 'pending'}],
    })

    line = _run(tmp_path, fake)

    assert fake.tool_names == ['get_pending_escalations']
    assert line.startswith(_FIRE_PREFIX)
    assert 'esc-x' in line
    assert not mod.default_state_path(PROJECT_ID).exists()


@pytest.mark.parametrize('response', [
    ConnectionError('connection refused'),
    {'unexpected': 'shape'},
    [{'id': 'esc-x'}],
    {'result': 'not a list'},
])
def test_run_a_failed_pending_check_files_nothing(tmp_path, caplog, response):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    _write_positive_control(tmp_path)
    fake = _FakeCallTool(get_pending_escalations=response)

    line = _run(tmp_path, fake)

    assert fake.tool_names == ['get_pending_escalations']
    assert line.startswith(_FIRE_PREFIX)
    assert 'pending check failed' in line
    assert [record.levelno for record in _census_records(caplog)] == [logging.WARNING]


@pytest.mark.parametrize('response', [
    RuntimeError('server down'),
    {'error': 'refused', 'error_type': 'Boom'},
    {'id': MINTED_ID, 'status': 'accepted_unpersisted', 'level': 0},
    ['not', 'a', 'dict'],
])
def test_run_a_failed_filing_writes_no_state(tmp_path, caplog, response):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    _write_positive_control(tmp_path)
    fake = _FakeCallTool(escalate_info=response)

    line = _run(tmp_path, fake)

    assert fake.tool_names == ['get_pending_escalations', 'escalate_info']
    assert line.startswith(_FIRE_PREFIX)
    assert 'filing failed' in line
    assert not mod.default_state_path(PROJECT_ID).exists()
    assert [record.levelno for record in _census_records(caplog)] == [logging.WARNING]


def test_run_a_malformed_state_file_files_nothing(tmp_path):
    _write_positive_control(tmp_path)
    state_path = tmp_path / 'census-state.json'
    state_path.write_text('{not json', encoding='utf-8')
    fake = _FakeCallTool()

    line = _run(tmp_path, fake, state_path=state_path)

    assert fake.calls == []
    assert line.startswith(_FIRE_PREFIX)
    assert str(state_path) in line
    assert state_path.read_text(encoding='utf-8') == '{not json'


def test_run_disabled_reads_and_calls_nothing(tmp_path):
    _write_positive_control(tmp_path)
    fake = _FakeCallTool()

    line = _run(tmp_path, fake, block={'enabled': False})

    assert line == 'plan-deviation census: NO-FIRE -- disabled by plan_deviation_census.enabled'
    assert fake.calls == []


class _BrokenTz(tzinfo):
    def utcoffset(self, dt):
        raise RuntimeError('boom')


def test_run_an_evaluation_failure_is_a_no_fire_line_not_a_raise(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    fake = _FakeCallTool()

    line = mod.run(
        _load_cfg(tmp_path),
        now=datetime(2026, 10, 25, 12, 0, tzinfo=_BrokenTz()),
        call_tool=fake,
    )

    assert line.startswith('plan-deviation census: NO-FIRE -- evaluation failed (')
    assert fake.calls == []
    assert [record.levelno for record in _census_records(caplog)] == [logging.WARNING]


# ---------------------------------------------------------------------------
# nightly hook — the census runs from the trickle's one recording point, so a
# red night still carries its line; a retry pass does not run it
# ---------------------------------------------------------------------------

def _never_invoked(prompt, model):
    raise AssertionError('the coder must not be reached on a red night')


def _run_red_night(tmp_path: Path, *, pass_kind: nightly.PassKind) -> nightly.NightlyResult:
    projects_root = tmp_path / 'projects'
    projects_root.mkdir()
    posted: list[tuple[str, dict]] = []
    return nightly.run_nightly(
        config_path=_write_config(tmp_path),
        projects_root=projects_root,
        target_date=date(2026, 10, 24),
        now=NOW,
        invoke=_never_invoked,
        status_fetcher=lambda: {'statuses': {}},
        poster=lambda url, envelope: posted.append((url, envelope)),
        pass_kind=pass_kind,
    )


def _crashed_extractor(selected, **kwargs):
    return [], [('session-1', 'boom')]


def test_nightly_red_night_early_return_still_runs_the_census(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    monkeypatch.setattr(nightly, 'build_digests', _crashed_extractor)

    result = _run_red_night(tmp_path, pass_kind=nightly.PassKind.MANUAL)

    assert result.exit_code == 1
    assert len(_census_records(caplog)) == 1


def test_nightly_unexpected_raise_still_runs_the_census(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)

    def _raise(*args, **kwargs):
        raise RuntimeError('sampler exploded')

    monkeypatch.setattr(nightly, 'select_digest_sessions', _raise)

    result = _run_red_night(tmp_path, pass_kind=nightly.PassKind.MANUAL)

    assert result.exit_code == 1
    assert len(_census_records(caplog)) == 1


def test_nightly_retry_pass_does_not_run_the_census(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    monkeypatch.setattr(nightly, 'build_digests', _crashed_extractor)

    result = _run_red_night(tmp_path, pass_kind=nightly.PassKind.RETRY)

    assert result.exit_code == 1
    assert _census_records(caplog) == []
