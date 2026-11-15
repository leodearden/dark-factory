"""Tests for scripts/sitting/return_brief.py — the 05:30 cross-project return brief (task 5376)."""
from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from escalation.authority import EXECUTOR_IDENTITY, policy_row
from escalation.models import Escalation
from orchestrator.session_registry import (
    AuditOutcome,
    AuditVerdict,
    DecisionRecord,
    normalize_escalations_dir,
)
from shared.project_layout import KNOWN_PROJECT_ROOTS_ENV
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from sitting import apply as apply_mod
from sitting import audits, brief, fleet_state, payloads
from sitting import ledger as ledger_mod
from sitting import preparation as prep_mod
from sitting import return_brief as mod
from sitting import trigger_state as trigger_mod
from sitting.inventory import AUDIT_KIND, audit_key, escalation_key, key_str, parse_key
from sitting.payloads import AgreedMarker, PreparedMarker

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 9, 26, 5, 30, tzinfo=UTC)
NOW_ISO = NOW.isoformat()
IN_WINDOW = '2026-09-25T18:00:00+00:00'
BEFORE_WINDOW = '2026-09-20T18:00:00+00:00'
CLOSED_IN_WINDOW = '2026-09-25T20:00:00+00:00'
NIGHT_TWO = (NOW + timedelta(days=1)).isoformat()
NIGHT_THREE = (NOW + timedelta(days=2)).isoformat()
TITLES = (
    '1. Decisions needed',
    '2. Rulings made under standing policy',
    '3. Landed',
    '4. Stuck and why',
    '5. Spend and cap hits',
    '6. Autonomous closes',
)
HOST = 'Host'
HEADINGS = (HOST, *TITLES)
DONE_LINE = re.compile(
    r'^done \(decisions=(\w+), standing_policy=(\w+), landed=(\w+), stuck=(\w+), spend=(\w+), closes=(\w+)\)$'
)
EVIDENCE = 'Leo 2026-09-25: "take option A"\nquoted from esc-5580-4'


@dataclass
class Env:
    tmp: Path
    df: Path
    fleet: Path
    sessions: Path
    preparation: Path
    out: Path

    @property
    def queue(self) -> Path:
        return self.df / 'data' / 'escalations'

    @property
    def handover(self) -> Path:
        return self.queue / 'l2-handover.md'

    @property
    def output(self) -> Path:
        return self.out / 'nested' / 'return-brief.md'

    @property
    def ledger_out(self) -> Path:
        return self.out / 'sitting' / 'ledger-nightly.json'

    @property
    def trigger_state(self) -> Path:
        return self.tmp / 'state' / 'trigger-last-run.json'

    def argv(self, *roots: Path) -> list[str]:
        return [
            *[arg for root in (roots or (self.df,)) for arg in ('--project-root', str(root))],
            *self.store_argv(),
        ]

    def store_argv(self) -> list[str]:
        """Every argument but the roots, so a test can leave root discovery to main."""
        return [
            '--decisions-root', str(self.fleet),
            '--sessions-root', str(self.sessions),
            '--preparation', str(self.preparation),
            '--output', str(self.output),
            '--ledger-out', str(self.ledger_out),
            '--trigger-state', str(self.trigger_state),
            '--now', NOW_ISO,
        ]

    def build(self, *roots: Path) -> mod.ReturnBriefDocument:
        return mod.build(
            project_roots=[str(root) for root in (roots or (self.df,))],
            roots_source='--project-root',
            decisions_root=self.fleet,
            sessions_root=self.sessions,
            handover_override=None,
            preparation_path=self.preparation,
            previous_ledger=self.ledger_out,
            trigger_state_path=self.trigger_state,
            window=fleet_state.Window.trailing(NOW, days=1),
            now=NOW,
        )


def _closed_decision(fleet: Path, decision_id: str, *, closed_at: str, **fields) -> DecisionRecord:
    """An evidence-carrying close of a decision filed BEFORE the window."""
    fields.setdefault('closing_evidence', f'the evidence for {decision_id}')
    record = DecisionRecord(id=decision_id, project='dark_factory', text=f'close {decision_id}?',
                            filed_at=BEFORE_WINDOW, state='answered', closed_at=closed_at, **fields)
    decisions = fleet / 'decisions'
    decisions.mkdir(parents=True, exist_ok=True)
    (decisions / f'{decision_id}.json').write_text(record.to_json())
    return record


def _esc(queue: Path, **fields) -> Escalation:
    fields.setdefault('task_id', fields['id'].split('-')[1])
    fields.setdefault('agent_role', 'escalation-watcher-auto')
    fields.setdefault('severity', 'blocking')
    fields.setdefault('category', 'design_concern')
    fields.setdefault('summary', f"summary of {fields['id']}")
    fields.setdefault('level', 2)
    fields.setdefault('options', ['close it', 'hold it'])
    fields.setdefault('timestamp', '2026-09-24T05:30:00+00:00')
    esc = Escalation(**fields)
    queue.mkdir(parents=True, exist_ok=True)
    (queue / f'{esc.id}.json').write_text(esc.to_json())
    return esc


def _trial_note() -> str:
    prepared = PreparedMarker(recommendation='A', no_lean_reason='', sitting_id='s0', prepared_at=IN_WINDOW)
    agreed = AgreedMarker(answer='A', agreed=True, answer_rounds=1, at=IN_WINDOW)
    return payloads.append_markers('', payloads.render_prepared_marker(prepared), payloads.render_agreed_marker(agreed))


def _runs_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE events (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, run_id TEXT NOT NULL, task_id TEXT,
            event_type TEXT NOT NULL, phase TEXT, role TEXT, data TEXT DEFAULT '{}', cost_usd REAL, duration_ms INTEGER);
        CREATE TABLE invocations (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT, project_id TEXT NOT NULL,
            account_name TEXT NOT NULL, model TEXT NOT NULL, role TEXT NOT NULL, cost_usd REAL NOT NULL DEFAULT 0.0,
            duration_ms INTEGER NOT NULL DEFAULT 0, capped INTEGER NOT NULL DEFAULT 0, started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL);
        CREATE TABLE task_results (run_id TEXT NOT NULL, task_id TEXT NOT NULL, project_id TEXT NOT NULL, title TEXT,
            outcome TEXT NOT NULL, completed_at TEXT, PRIMARY KEY (run_id, task_id));
    """)
    conn.execute("INSERT INTO events (timestamp, run_id, task_id, event_type, data) "
                 "VALUES (?, 'r1', '120', 'task_completed', '{\"outcome\": \"done\"}')", (IN_WINDOW,))
    conn.execute("INSERT INTO invocations (run_id, task_id, project_id, account_name, model, role, cost_usd, capped, "
                 "started_at, completed_at) VALUES ('r1', '120', 'dark_factory', 'a', 'claude-opus', 'implementer', "
                 "2.5, 1, ?, ?)", (IN_WINDOW, IN_WINDOW))
    conn.execute("INSERT INTO task_results VALUES ('r1', '120', 'dark_factory', 't', 'done', ?)", (IN_WINDOW,))
    conn.commit()
    conn.close()


@pytest.fixture
def env(tmp_path, make_tasks_db, project_root_with_tasks_db) -> Env:
    df = tmp_path / 'src' / 'dark-factory'
    e = Env(tmp_path, df, tmp_path / 'fleet', tmp_path / 'fleet' / 'sessions',
            tmp_path / 'state' / 'preparation.json', tmp_path / 'out')
    make_tasks_db([
        *({'id': n, 'status': 'blocked'} for n in range(101, 108)),
        {'id': 109, 'status': 'blocked', 'title': 'blocked behind nothing'},
        {'id': 120, 'status': 'done'},
    ], directory=project_root_with_tasks_db(df).parent)
    _runs_db(fleet_state.runs_db_path(df))
    for n in range(101, 107):
        _esc(e.queue, id=f'esc-{n}-1')
    _esc(e.queue, id='esc-107-1', pin_declared_by=['leo'])
    _esc(e.queue, id='esc-108-1', status='resolved', resolved_at=IN_WINDOW, resolved_by='interactive',
         triage_note=_trial_note(), resolution_turns=1)
    e.sessions.mkdir(parents=True)
    e.handover.write_text('# handover\n\nnothing relevant here\n')
    _closed_decision(e.fleet, 'dec-close-1', closed_at=CLOSED_IN_WINDOW, escalation_id='esc-104-9',
                     closing_evidence=EVIDENCE)
    prep_mod.record(e.preparation, [{
        'item': ['esc', normalize_escalations_dir(e.queue), 'esc-101-1'],
        'question': 'Close esc-101-1 as ruled?',
        'options': [{'label': 'A', 'text': 'close', 'ramification': 'task 101 resumes'},
                    {'label': 'B', 'text': 'hold', 'ramification': 'task 101 stays blocked'}],
        'recommendation': {'option': 'A', 'evidence_chain': 'the description opens RULED'},
        'on_apply': 'esc-101-1 resolves',
        'prepared_at': '2026-09-26T05:40:00+00:00',
        'prepared_by': 'nightly-fable',
    }])
    return e


def _partition(text: str) -> tuple[list[str], list[list[str]]]:
    """Level-2 headings outside fences, and the lines under each, fence-aware."""
    headings: list[str] = []
    sections: list[list[str]] = []
    fence = ''
    for line in text.splitlines():
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if marker and (not fence or marker.group(1).startswith(fence)):
            fence = '' if fence else marker.group(1)
        if not fence and not marker and line.startswith('## '):
            headings.append(line[3:])
            sections.append([])
        elif sections:
            sections[-1].append(line)
    return headings, sections


def _sections(text: str) -> dict[str, str]:
    headings, bodies = _partition(text)
    return {heading: '\n'.join(body) for heading, body in zip(headings, bodies, strict=True)}


def _header(text: str) -> str:
    return text.split('\n## ', 1)[0]


def _run(capsys, argv: list[str]) -> tuple[int, str]:
    rc = mod.main(argv)
    return rc, capsys.readouterr().out


def _key(env: Env, esc_id: str) -> str:
    return json.dumps(list(escalation_key(normalize_escalations_dir(env.queue), esc_id)))


def _numbers(env: Env) -> dict[str, int]:
    """Every live (open or standing) item's number in the ledger the last render wrote."""
    ledger = ledger_mod.load(env.ledger_out)
    assert ledger is not None
    return {key: entry.number for key, entry in ledger.entries.items() if entry.state != 'done'}


class TestSections:
    def test_the_host_block_then_the_six_mandated_sections_in_order_as_level_two_headings(self, env):
        text = mod.render(env.build())

        assert _partition(text)[0] == list(HEADINGS)

    def test_section_one_is_the_brief_renderer_byte_for_byte_with_its_footer_and_docket_line(self, env):
        document = env.build()
        sitting = document.sitting

        direct = brief.render_brief(sitting.numbered, sitting.standing, sitting.done, glossary=sitting.glossary,
                                    generated_at=document.generated_at, level=2, title=TITLES[0])
        text = mod.render(document)

        section_one = text[text.index(f'## {TITLES[0]}'):text.index(f'## {TITLES[1]}')]
        assert section_one.startswith(direct)
        assert '### Standing / no action' in section_one and 'esc-107-1' in section_one
        assert 'Docket page: recommended' in section_one
        assert len(sitting.numbered) == 6

    def test_the_nightly_is_recommend_only_whatever_the_preparation_says(self, env):
        document = env.build()

        assert all(c.bucket != 'closeable' for c in document.sitting.classified)

    def test_every_section_shows_its_own_measured_at(self, env):
        sections = _sections(mod.render(env.build()))

        for title in TITLES:
            assert NOW_ISO in sections[title], title

    def test_section_bodies_carry_the_measurements(self, env):
        sections = _sections(mod.render(env.build()))

        assert fleet_state.SHADOW_ONLY in sections[TITLES[1]]
        assert f'1 in {brief.AUDIT_SAMPLE_EVERY}' in sections[TITLES[1]]
        assert 'dark_factory' in sections[TITLES[2]] and '1 task' in sections[TITLES[2]]
        assert fleet_state.NO_OPEN_ESCALATION in sections[TITLES[3]] and 'task 109' in sections[TITLES[3]]
        assert 'claude-opus' in sections[TITLES[4]] and '$2.50' in sections[TITLES[4]]
        assert EVIDENCE in sections[TITLES[5]] and 'dec-close-1' in sections[TITLES[5]]
        assert 'lifetime: 1' in sections[TITLES[5]]


class TestAutonomousClosesSection:
    def test_a_close_filed_before_the_window_renders_with_its_close_stamp(self, env):
        section = _sections(mod.render(env.build()))[TITLES[5]]

        assert f'closed {CLOSED_IN_WINDOW}; filed {BEFORE_WINDOW}' in section
        assert 'No autonomous close in the window.' not in section
        assert 'Undated closes' not in section

    def test_an_undated_close_is_listed_stating_closed_at_is_missing(self, env):
        _closed_decision(env.fleet, 'dec-undated', closed_at='')

        section = _sections(mod.render(env.build()))[TITLES[5]]

        undated = section.split('Undated closes', 1)[1]
        assert 'dec-undated' in undated and 'closed_at' in undated
        assert 'dec-close-1' not in undated
        assert 'lifetime: 2' in section

    def test_with_nothing_closed_in_the_window_the_undated_are_still_listed(self, env):
        _closed_decision(env.fleet, 'dec-close-1', closed_at=BEFORE_WINDOW, closing_evidence=EVIDENCE)
        _closed_decision(env.fleet, 'dec-undated', closed_at='')

        section = _sections(mod.render(env.build()))[TITLES[5]]

        assert 'No autonomous close in the window.' in section
        assert 'dec-undated' in section.split('Undated closes', 1)[1]

    def test_the_audit_sample_is_drawn_from_window_closes_only(self, env):
        for n in range(2, brief.AUDIT_SAMPLE_EVERY):
            _closed_decision(env.fleet, f'dec-close-{n}', closed_at=f'2026-09-25T21:{10 * n:02d}:00+00:00')
        for n in range(3):
            _closed_decision(env.fleet, f'dec-undated-{n}', closed_at='')

        short = _sections(mod.render(env.build()))[TITLES[5]]
        _closed_decision(env.fleet, 'dec-close-z', closed_at='2026-09-26T05:00:00+00:00')
        full = _sections(mod.render(env.build()))[TITLES[5]]

        assert 'AUDIT SAMPLE' not in short
        assert full.count('AUDIT SAMPLE') == 1
        assert f'### Close {brief.AUDIT_SAMPLE_EVERY}: decision dec-close-z — AUDIT SAMPLE' in full


class TestHeader:
    def test_header_states_generation_queues_preparation_age_and_the_trial(self, env):
        header = _header(mod.render(env.build()))

        assert NOW_ISO in header
        assert normalize_escalations_dir(env.queue) in header
        assert 'newest prepared_at 2026-09-26T05:40:00+00:00' in header
        assert '5 of 6 numbered items awaiting preparation' in header
        trial = next(line for line in header.splitlines() if 'dark_factory' in line and 'agreement' in line)
        assert 'agreement 1/1' in trial and 'resolution_turns n=1' in trial and 'median 1' in trial
        assert 'recommend-only' in header

    def test_no_preparation_recorded_is_stated(self, env):
        env.preparation.unlink()

        header = _header(mod.render(env.build()))

        assert 'no preparation recorded' in header
        assert '6 of 6 numbered items awaiting preparation' in header

    def test_an_unavailable_store_is_a_stated_shortfall_and_no_section_is_omitted(self, env):
        bare = env.tmp / 'src' / 'reify'
        bare.mkdir(parents=True)

        text = mod.render(env.build(env.df, bare))

        sections = _sections(text)
        assert list(sections) == list(HEADINGS)
        assert 'reify' in sections[TITLES[2]] and 'source_missing' in sections[TITLES[2]]
        assert 'runs_db' in sections[TITLES[2]]
        assert 'tasks_db' in sections[TITLES[3]]


class TestHandoverPerProject:
    @pytest.fixture
    def reify(self, env) -> Path:
        reify = env.tmp / 'src' / 'reify'
        _esc(reify / 'data' / 'escalations', id='esc-101-1')
        (reify / 'data' / 'escalations' / 'l2-handover.md').write_text(
            '# handover\n\nesc-101-1 task 101: the reify watcher holds this\n'
        )
        return reify

    def test_build_reads_each_projects_own_handover(self, env, reify):
        document = mod.build(
            project_roots=[str(env.df), str(reify)], roots_source='--project-root', decisions_root=env.fleet,
            sessions_root=env.sessions, handover_override=None, preparation_path=env.preparation,
            previous_ledger=env.ledger_out, trigger_state_path=env.trigger_state,
            window=fleet_state.Window.trailing(NOW, days=1), now=NOW,
        )

        key = escalation_key(normalize_escalations_dir(reify / 'data' / 'escalations'), 'esc-101-1')
        (classified,) = [c for c in document.sitting.classified if c.item.key == key]
        (probe,) = [p for p in classified.ownership.probes if p.probe == 'handover']
        assert probe.status == 'mentions'
        assert 'the reify watcher' in probe.evidence

    def test_the_flag_takes_project_equals_path(self, env, capsys, reify):
        override = env.tmp / 'override.md'
        override.write_text('esc-101-1 task 101: read from the override\n')

        rc, _ = _run(capsys, [*env.argv(env.df, reify), '--handover', f'reify={override}'])

        assert rc == 0
        assert 'read from the override' in env.output.read_text()


class TestSupersession:
    def test_names_the_digest_once_and_never_touches_it(self, env, capsys):
        digest = env.df / 'data' / 'afk-digest.md'
        digest.write_text('# old digest\n')
        before = (digest.read_bytes(), digest.stat().st_mtime_ns)

        assert _run(capsys, env.argv())[0] == 0

        assert env.output.read_text().count('afk-digest.md') == 1
        assert (digest.read_bytes(), digest.stat().st_mtime_ns) == before


class TestMain:
    def test_writes_the_page_and_a_fresh_nightly_ledger_atomically(self, env, capsys, monkeypatch):
        replaced: list[str] = []
        real_replace = os.replace

        def recording_replace(src, dst):
            replaced.append(str(dst))
            real_replace(src, dst)

        monkeypatch.setattr(os, 'replace', recording_replace)

        rc, _ = _run(capsys, env.argv())

        assert rc == 0
        assert env.output.read_text().startswith('# Return brief')
        assert {str(env.output), str(env.ledger_out)} <= set(replaced)
        ledger = ledger_mod.load(env.ledger_out)
        assert ledger is not None
        assert ledger.sitting_id == f'sitting-{NOW_ISO}'
        assert sorted(entry.number for _, entry in ledger.in_state('open')) == [1, 2, 3, 4, 5, 6, 7]

    def test_an_item_present_on_consecutive_nights_keeps_its_number(self, env, capsys):
        _run(capsys, env.argv())
        first = _numbers(env)
        _esc(env.queue, id='esc-101-1', status='resolved', resolved_at=NIGHT_TWO, resolved_by='interactive')
        _esc(env.queue, id='esc-110-1', severity='critical')

        _run(capsys, [*env.argv(), '--now', NIGHT_TWO])

        assert env.output.read_text().count('# Return brief') == 1
        second = _numbers(env)
        resolved = _key(env, 'esc-101-1')
        assert {key: number for key, number in second.items() if key in first} == {
            key: number for key, number in first.items() if key != resolved}
        assert second[_key(env, 'esc-110-1')] == max(first.values()) + 1
        ledger = ledger_mod.load(env.ledger_out)
        assert ledger is not None
        assert [(key, entry.number) for key, entry in ledger.in_state('done')] == [(resolved, first[resolved])]

    def test_a_done_item_is_not_carried_a_further_night_and_its_number_is_never_reissued(self, env, capsys):
        _run(capsys, env.argv())
        retired = _numbers(env)[_key(env, 'esc-101-1')]
        _esc(env.queue, id='esc-101-1', status='resolved', resolved_at=NIGHT_TWO, resolved_by='interactive')
        _run(capsys, [*env.argv(), '--now', NIGHT_TWO])
        _esc(env.queue, id='esc-111-1')

        _run(capsys, [*env.argv(), '--now', NIGHT_THREE])

        third = _numbers(env)
        assert _key(env, 'esc-101-1') not in third
        assert retired not in third.values()

    def test_an_unreadable_previous_ledger_numbers_afresh_and_says_so(self, env, capsys):
        env.ledger_out.parent.mkdir(parents=True)
        env.ledger_out.write_text('{not json')

        rc, out = _run(capsys, env.argv())

        assert rc == 0
        section = _sections(env.output.read_text())[TITLES[0]]
        assert f'shortfall: nightly_ledger {env.ledger_out}' in section and 'numbered afresh' in section
        assert sorted(_numbers(env).values()) == [1, 2, 3, 4, 5, 6, 7]
        match = DONE_LINE.match(out.splitlines()[-1])
        assert match is not None and match.group(1) == 'degraded'

    def test_default_paths_are_under_this_checkouts_data_dir(self):
        assert mod.DEFAULT_OUTPUT == REPO_ROOT / 'data' / 'return-brief.md'
        assert mod.DEFAULT_LEDGER_OUT == REPO_ROOT / 'data' / 'sitting' / 'ledger-nightly.json'
        assert mod.DEFAULT_PREPARATION == REPO_ROOT / 'data' / 'sitting' / 'preparation.json'

    def test_exits_zero_and_names_each_sections_status(self, env, capsys):
        rc, out = _run(capsys, env.argv())

        assert rc == 0
        match = DONE_LINE.match(out.splitlines()[-1])
        assert match is not None
        assert set(match.groups()) <= {'ok', 'degraded'}

    def test_a_degraded_night_still_exits_zero_with_the_degraded_sections_named(self, env, capsys):
        bare = env.tmp / 'src' / 'reify'
        bare.mkdir(parents=True)

        rc, out = _run(capsys, env.argv(env.df, bare))

        assert rc == 0
        match = DONE_LINE.match(out.splitlines()[-1])
        assert match is not None
        statuses = dict(zip(('decisions', 'standing_policy', 'landed', 'stuck', 'spend', 'closes'),
                            match.groups(), strict=True))
        assert statuses['landed'] == statuses['spend'] == statuses['stuck'] == 'degraded'
        assert statuses['closes'] == 'ok'
        assert env.output.is_file()

    def test_a_root_folding_onto_a_measured_project_is_a_shortfall_not_a_missing_page(self, env, capsys):
        twin = env.tmp / 'elsewhere' / 'dark_factory'
        twin.mkdir(parents=True)

        rc, out = _run(capsys, env.argv(env.df, twin))

        assert rc == 0
        sections = _sections(env.output.read_text())
        assert list(sections) == list(HEADINGS)
        for title in TITLES[1:5]:
            assert f'shortfall: project_root {twin}: folds to project' in sections[title], title
        match = DONE_LINE.match(out.splitlines()[-1])
        assert match is not None and match.group(3) == 'degraded'

    def test_a_measurement_that_raises_degrades_its_section_and_the_page_is_written(self, env, capsys, monkeypatch):
        def pathological(*_args, **_kwargs):
            raise RuntimeError('pathological registry: simulated')

        monkeypatch.setattr(fleet_state, 'autonomous_closes', pathological)

        rc, out = _run(capsys, env.argv())

        assert rc == 0
        section = _sections(env.output.read_text())[TITLES[5]]
        assert '(unreadable)' in section and 'RuntimeError: pathological registry: simulated' in section
        match = DONE_LINE.match(out.splitlines()[-1])
        assert match is not None and match.group(6) == 'degraded'

    def test_no_subprocess_and_no_network(self, env, capsys, monkeypatch):
        calls: list[str] = []

        def forbid(name):
            def recorder(*args, **kwargs):
                calls.append(name)
                raise AssertionError(f'{name} called')
            return recorder

        for name in ('run', 'Popen', 'call', 'check_call', 'check_output'):
            monkeypatch.setattr(subprocess, name, forbid(f'subprocess.{name}'))
        monkeypatch.setattr(socket, 'create_connection', forbid('socket.create_connection'))
        monkeypatch.setattr(socket.socket, 'connect', forbid('socket.connect'))

        rc, _ = _run(capsys, env.argv())

        assert rc == 0
        assert calls == []

    def test_a_pinned_now_renders_byte_identical_files(self, env, capsys):
        _run(capsys, env.argv())
        first = (env.output.read_bytes(), env.ledger_out.read_bytes())

        _run(capsys, env.argv())

        assert (env.output.read_bytes(), env.ledger_out.read_bytes()) == first

    def test_the_window_is_configurable(self, env, capsys):
        rc, _ = _run(capsys, [*env.argv(), '--window-days', '7'])

        assert rc == 0
        assert (NOW - timedelta(days=7)).isoformat() in _header(env.output.read_text())


SAMPLED_CLASS = 'landing_provenance'
FULL_CLASS = 'b3_low_risk_unblock'
AUDIT_ITEM = re.compile(r'^#### audit (\d+): decision (\S+)', re.MULTILINE)


def _hours_before_now(hours: float) -> str:
    return (NOW - timedelta(hours=hours)).isoformat()


def _executor_close(env: Env, n: int, *, policy_class: str = SAMPLED_CLASS, closed_at: str | None = None,
                    audited: AuditVerdict | None = None) -> DecisionRecord:
    """An archived record the executor closed under *policy_class*, and the DecisionRecord that closed it."""
    esc_id = f'esc-{700 + n}-1'
    _esc(env.queue / 'archive' / '2026-09-25', id=esc_id, status='resolved', resolved_by=EXECUTOR_IDENTITY,
         resolved_at=closed_at or _hours_before_now(8 * n), policy_class=policy_class,
         claimed_by=EXECUTOR_IDENTITY, claimed_at=closed_at or _hours_before_now(8 * n))
    closed_at = closed_at or _hours_before_now(8 * n)
    outcome = None if audited is None else AuditOutcome(audited, '', NOW_ISO, policy_class, closed_at=closed_at)
    return _closed_decision(env.fleet, f'dec-exec-{policy_class}-{n}', closed_at=closed_at,
                            escalation_id=esc_id, escalations_dir=normalize_escalations_dir(env.queue),
                            closing_evidence=f'executor evidence for close {n}\nfound_on_main abc1234',
                            audit_outcomes=() if outcome is None else (outcome,))


def _twelve_executor_closes(env: Env) -> list[DecisionRecord]:
    """Twelve closes inside the audit week, three of them inside the one-day --window-days."""
    return [_executor_close(env, n) for n in range(12)]


def _audit_items(section: str) -> dict[str, int]:
    """Each rendered audit item's decision id and its number."""
    return {decision_id: int(number) for number, decision_id in AUDIT_ITEM.findall(section)}


def _audit_entries(env: Env) -> dict[str, int]:
    """The open audit entries of the ledger the last render wrote, decision id to number."""
    ledger = ledger_mod.load(env.ledger_out)
    assert ledger is not None
    return {parse_key(key)[1]: entry.number for key, entry in ledger.entries.items()
            if parse_key(key)[0] == AUDIT_KIND and entry.state == 'open'}


def _item_block(section: str, number: int) -> str:
    start = section.index(f'#### audit {number}:')
    rest = section[start + 1:]
    end = rest.find('\n#### ')
    return section[start:] if end == -1 else section[start:start + 1 + end]


class TestExecutorAuditDraw:
    def test_ten_of_twelve_executor_closes_are_drawn_and_numbered_into_the_ledger(self, env, capsys):
        records = {record.id: record for record in _twelve_executor_closes(env)}

        rc, _ = _run(capsys, env.argv())

        assert rc == 0
        section = _sections(env.output.read_text())[TITLES[5]]
        drawn = _audit_items(section)
        assert len(drawn) == audits.SAMPLE_SIZE
        assert len(set(drawn.values())) == audits.SAMPLE_SIZE
        assert set(drawn) <= set(records)
        assert _audit_entries(env) == drawn
        questions = _numbers(env)
        question_numbers = [n for key, n in questions.items() if parse_key(key)[0] != AUDIT_KIND]
        assert min(drawn.values()) > max(question_numbers)
        for decision_id, number in drawn.items():
            block = _item_block(section, number)
            record = records[decision_id]
            assert record.escalation_id is not None
            assert SAMPLED_CLASS in block and record.escalation_id in block
            assert brief.fence_safe(record.closing_evidence) in block
            assert f'answer slot: `{number}=ok|wrong[:note]`' in block

    def test_the_draw_reaches_closes_outside_the_one_day_window(self, env, capsys):
        _twelve_executor_closes(env)

        _run(capsys, env.argv())

        section = _sections(env.output.read_text())[TITLES[5]]
        outside_the_day = [f'dec-exec-{SAMPLED_CLASS}-{n}' for n in range(4, 12)]
        assert set(_audit_items(section)) & set(outside_the_day)

    def test_a_rerender_at_the_same_now_keeps_the_same_ten_under_the_same_numbers(self, env, capsys):
        _twelve_executor_closes(env)
        _run(capsys, env.argv())
        first = _audit_entries(env)
        assert len(first) == audits.SAMPLE_SIZE

        _run(capsys, env.argv())

        assert _audit_entries(env) == first
        assert _audit_items(_sections(env.output.read_text())[TITLES[5]]) == first

    def test_a_render_a_night_later_keeps_each_carried_items_number(self, env, capsys):
        _twelve_executor_closes(env)
        _run(capsys, env.argv())
        first = _audit_entries(env)

        _run(capsys, [*env.argv(), '--now', NIGHT_TWO])

        second = _audit_entries(env)
        assert len(second) == audits.SAMPLE_SIZE
        assert {d: n for d, n in second.items() if d in first} == {d: n for d, n in first.items() if d in second}
        assert all(n > max(first.values()) for d, n in second.items() if d not in first)

    def test_a_fresh_install_states_no_executor_close_and_leaves_the_carve_out_closes_alone(self, env, capsys):
        for n in range(2, brief.AUDIT_SAMPLE_EVERY + 1):
            _closed_decision(env.fleet, f'dec-close-{n}', closed_at=f'2026-09-25T21:{10 * n:02d}:00+00:00')

        _run(capsys, env.argv())

        section = _sections(env.output.read_text())[TITLES[5]]
        assert 'no executor close in window' in section
        assert _audit_items(section) == {}
        assert section.count('AUDIT SAMPLE') == 1
        assert f'### Close {brief.AUDIT_SAMPLE_EVERY}: decision dec-close-{brief.AUDIT_SAMPLE_EVERY} — AUDIT SAMPLE' in section
        assert '### Close 1: decision dec-close-1' in section
        assert _audit_entries(env) == {}

    def test_every_close_already_audited_states_so_and_draws_nothing(self, env, capsys):
        for n in range(3):
            _executor_close(env, n, audited=AuditVerdict.OK)

        _run(capsys, env.argv())

        section = _sections(env.output.read_text())[TITLES[5]]
        assert 'every executor close in the window is audited' in section
        assert 'no executor close in window' not in section
        assert _audit_items(section) == {}

    def test_a_full_audit_class_is_drawn_whole_beside_the_ten(self, env, capsys):
        row = policy_row(FULL_CLASS)
        assert row is not None and row.audit_full_until == 20
        _twelve_executor_closes(env)
        full = [_executor_close(env, 20 + n, policy_class=FULL_CLASS, closed_at=_hours_before_now(2 + n))
                for n in range(3)]

        _run(capsys, env.argv())

        section = _sections(env.output.read_text())[TITLES[5]]
        drawn = _audit_items(section)
        assert len(drawn) == audits.SAMPLE_SIZE + 3
        assert {record.id for record in full} <= set(drawn)
        full_numbers = [drawn[record.id] for record in full]
        assert max(full_numbers) < min(n for d, n in drawn.items() if d not in {r.id for r in full})
        assert all('FULL AUDIT' in _item_block(section, number) for number in full_numbers)

    def test_an_unreadable_executor_escalation_degrades_the_section_and_is_printed(self, env, capsys):
        _twelve_executor_closes(env)
        broken = _executor_close(env, 30, closed_at=_hours_before_now(3))
        path = env.queue / 'archive' / '2026-09-25' / f'{broken.escalation_id}.json'
        path.write_text('{not json')

        rc, out = _run(capsys, env.argv())

        assert rc == 0
        section = _sections(env.output.read_text())[TITLES[5]]
        assert f'shortfall: executor_close {path}: {broken.id}' in section
        assert len(_audit_items(section)) == audits.SAMPLE_SIZE
        match = DONE_LINE.match(out.splitlines()[-1])
        assert match is not None and match.group(6) == 'degraded'

    def test_pruned_or_unreadable_closes_outside_the_audit_week_leave_the_section_ok(self, env, capsys):
        _twelve_executor_closes(env)
        archive = env.queue / 'archive' / '2026-09-25'
        pruned = _executor_close(env, 40, closed_at=_hours_before_now(40 * 24))
        corrupt = _executor_close(env, 41, closed_at=_hours_before_now(40 * 24))
        (archive / f'{pruned.escalation_id}.json').unlink()
        (archive / f'{corrupt.escalation_id}.json').write_text('{not json')

        rc, out = _run(capsys, env.argv())

        assert rc == 0
        section = _sections(env.output.read_text())[TITLES[5]]
        assert 'shortfall: executor_close' not in section
        assert len(_audit_items(section)) == audits.SAMPLE_SIZE
        match = DONE_LINE.match(out.splitlines()[-1])
        assert match is not None and match.group(6) == 'ok'

    def test_an_in_window_close_whose_escalation_aged_out_is_stated_once_not_a_shortfall(self, env, capsys):
        _twelve_executor_closes(env)
        pruned = _executor_close(env, 30, closed_at=_hours_before_now(3))
        (env.queue / 'archive' / '2026-09-25' / f'{pruned.escalation_id}.json').unlink()

        rc, out = _run(capsys, env.argv())

        assert rc == 0
        section = _sections(env.output.read_text())[TITLES[5]]
        assert section.count(mod.AGED_OUT.format(count=1)) == 1
        assert 'shortfall: executor_close' not in section
        match = DONE_LINE.match(out.splitlines()[-1])
        assert match is not None and match.group(6) == 'ok'

    def test_with_nothing_aged_out_the_aged_out_line_is_absent(self, env, capsys):
        _twelve_executor_closes(env)

        _run(capsys, env.argv())

        section = _sections(env.output.read_text())[TITLES[5]]
        assert mod.AGED_OUT.format(count=1) not in section
        assert mod.AGED_OUT.format(count=0) not in section

    def test_a_drawn_audit_is_answerable_by_the_audit_verb_and_leaves_the_next_draw(self, env, capsys):
        _twelve_executor_closes(env)
        _run(capsys, env.argv())
        drawn = _audit_entries(env)
        decision_id, number = min(drawn.items(), key=lambda pair: pair[1])

        report = apply_mod.run_audit(
            env.ledger_out, [ledger_mod.Answer(str(number), 'ok', '', NOW_ISO, 'terminal')], decisions_root=env.fleet,
        )
        assert report.exit_code == apply_mod.EXIT_OK
        _run(capsys, env.argv())

        ledger = ledger_mod.load(env.ledger_out)
        assert ledger is not None
        entry = ledger.entries[key_str(audit_key(decision_id))]
        assert entry.state == 'done' and entry.number == number
        assert decision_id not in _audit_items(_sections(env.output.read_text())[TITLES[5]])


def _line_naming(text: str, needle: str) -> str:
    (line,) = [line for line in text.splitlines() if needle in line]
    return line


class TestHostBlock:
    def test_host_filer_records_of_any_level_are_listed_between_the_header_and_section_one(self, env):
        _esc(env.queue, id='esc-201-1', agent_role='orchestrator-watchdog', severity='critical',
             summary='orchestrator unit is down', timestamp='2026-09-25T05:30:00+00:00')
        _esc(env.queue, id='esc-202-1', agent_role='harness-dashboard-watchdog', level=0,
             summary='dashboard is stale', timestamp='2026-09-26T03:30:00+00:00')

        text = mod.render(env.build())

        assert text.index('\n## Host\n') < text.index(f'\n## {TITLES[0]}\n')
        assert text.index('# Return brief') < text.index('\n## Host\n')
        host = _sections(text)[HOST]
        down, stale = _line_naming(host, 'esc-201-1'), _line_naming(host, 'esc-202-1')
        assert 'L2' in down and 'critical' in down and '1.0d' in down and 'orchestrator unit is down' in down
        assert 'L0' in stale and 'blocking' in stale and '0.1d' in stale and 'dashboard is stale' in stale
        assert host.index('esc-201-1') < host.index('esc-202-1')
        assert 'esc-101-1' not in host

    def test_no_host_filer_record_pending_reads_so(self, env):
        host = _sections(mod.render(env.build()))[HOST]

        assert 'No host alarm pending.' in host

    def test_a_resolved_host_record_is_not_an_alarm(self, env):
        _esc(env.queue, id='esc-203-1', agent_role='orchestrator-watchdog', status='resolved',
             resolved_at=IN_WINDOW, resolved_by='interactive')

        host = _sections(mod.render(env.build()))[HOST]

        assert 'esc-203-1' not in host and 'No host alarm pending.' in host

    def test_a_host_record_in_a_second_roots_queue_names_that_project(self, env):
        reify = env.tmp / 'src' / 'reify'
        _esc(reify / 'data' / 'escalations', id='esc-301-1', agent_role='harness-dashboard-watchdog')

        host = _sections(mod.render(env.build(env.df, reify)))[HOST]

        assert 'reify' in _line_naming(host, 'esc-301-1')

    def test_an_unreadable_root_record_is_a_stated_shortfall_of_the_block(self, env):
        (env.queue / 'esc-299-1.json').write_text('{not json')

        host = _sections(mod.render(env.build()))[HOST]

        assert f'shortfall: escalation_queue {env.queue / "esc-299-1.json"}' in host


class TestUnansweredSince:
    def test_every_open_numbered_item_is_listed_oldest_first_with_its_first_presented_stamp(self, env, capsys):
        _run(capsys, env.argv())
        night_one = _numbers(env)
        _esc(env.queue, id='esc-101-1', status='resolved', resolved_at=NIGHT_TWO, resolved_by='interactive')
        _esc(env.queue, id='esc-110-1')

        _run(capsys, [*env.argv(), '--now', NIGHT_TWO])

        section = _sections(env.output.read_text())[TITLES[0]]
        block = section[section.index('### Unanswered since'):]
        carried = _line_naming(block, 'esc-102-1')
        assert carried.startswith(f'- {night_one[_key(env, "esc-102-1")]} (esc-102-1)')
        assert NOW_ISO in carried and '1.0d' in carried
        fresh = _line_naming(block, 'esc-110-1')
        assert NIGHT_TWO in fresh and '0.0d' in fresh
        assert 'esc-101-1' not in block
        assert 'esc-107-1' not in block
        assert block.index('esc-102-1') < block.index('esc-110-1')
        listed = [line for line in block.splitlines() if line.startswith('- ')]
        assert len(listed) == 6


class TestTriggerStateLine:
    def test_no_trigger_file_reads_evaluator_has_not_run(self, env):
        header = _header(mod.render(env.build()))

        assert 'evaluator has not run' in _line_naming(header, 'sitting trigger')

    def test_a_saved_last_run_names_its_stamp_age_outcome_reason_and_last_launch(self, env):
        ran_at = (NOW - timedelta(hours=2)).isoformat()
        launched_at = (NOW - timedelta(hours=26)).isoformat()
        trigger_mod.save(env.trigger_state, trigger_mod.TriggerLastRun(
            ran_at=ran_at, outcome=trigger_mod.TriggerOutcome.LAUNCHED, reason='red main on dark_factory',
            last_launch_at=launched_at, lane='t3',
        ))

        line = _line_naming(_header(mod.render(env.build())), 'sitting trigger')

        assert ran_at in line and '2.0h' in line
        assert 'launched' in line and 'red main on dark_factory' in line
        assert launched_at in line

    def test_a_last_run_that_never_launched_says_never(self, env):
        trigger_mod.save(env.trigger_state, trigger_mod.TriggerLastRun(
            ran_at=NOW_ISO, outcome=trigger_mod.TriggerOutcome.NOT_FIRED, reason='nothing red',
            last_launch_at=None, lane=None,
        ))

        line = _line_naming(_header(mod.render(env.build())), 'sitting trigger')

        assert 'not_fired' in line and 'last launch never' in line

    def test_an_unreadable_trigger_file_is_rendered_unreadable_with_a_shortfall(self, env):
        env.trigger_state.parent.mkdir(parents=True, exist_ok=True)
        env.trigger_state.write_text('{not json')

        header = _header(mod.render(env.build()))

        line = _line_naming(header, 'sitting trigger')
        assert 'unreadable' in line and 'JSONDecodeError' in line
        assert f'shortfall: trigger_state {env.trigger_state}' in header

    def test_main_reads_the_trigger_file_it_is_pointed_at(self, env, capsys):
        trigger_mod.save(env.trigger_state, trigger_mod.TriggerLastRun(
            ran_at=NOW_ISO, outcome=trigger_mod.TriggerOutcome.SUPPRESSED, reason='cooldown', last_launch_at=None,
            lane=None,
        ))

        _run(capsys, env.argv())

        assert 'suppressed' in _line_naming(_header(env.output.read_text()), 'sitting trigger')

    def test_the_default_trigger_path_is_the_evaluators(self):
        assert mod.DEFAULT_TRIGGER_STATE == trigger_mod.DEFAULT_PATH


class TestRootsStatement:
    def test_roots_passed_on_the_command_line_are_named_with_that_source(self, env, capsys):
        reify = env.tmp / 'src' / 'reify'
        reify.mkdir(parents=True)

        _run(capsys, env.argv(env.df, reify))

        line = _line_naming(_header(env.output.read_text()), 'project roots')
        assert '--project-root' in line and str(env.df) in line and str(reify) in line

    def test_roots_from_the_environment_name_the_variable(self, env, capsys, monkeypatch):
        monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(env.df))

        _run(capsys, env.store_argv())

        line = _line_naming(_header(env.output.read_text()), 'project roots')
        assert KNOWN_PROJECT_ROOTS_ENV in line and str(env.df) in line

    def test_with_neither_the_source_is_default(self, env, capsys, monkeypatch):
        monkeypatch.delenv(KNOWN_PROJECT_ROOTS_ENV, raising=False)
        monkeypatch.setattr(mod, 'discover_project_roots', lambda: [str(env.df)])

        _run(capsys, env.store_argv())

        line = _line_naming(_header(env.output.read_text()), 'project roots')
        assert 'default' in line and str(env.df) in line

    def test_a_queue_reached_only_through_a_decision_record_is_named(self, env):
        elsewhere = env.tmp / 'src' / 'elsewhere' / 'data' / 'escalations'
        elsewhere.mkdir(parents=True)
        _closed_decision(env.fleet, 'dec-elsewhere', closed_at=CLOSED_IN_WINDOW,
                         escalations_dir=normalize_escalations_dir(elsewhere), escalation_id='esc-401-1')

        header = _header(mod.render(env.build()))

        outside = _line_naming(header, 'outside those roots')
        assert normalize_escalations_dir(elsewhere) in outside
        assert normalize_escalations_dir(env.queue) not in outside

    def test_with_every_queue_under_a_measured_root_none_is_outside(self, env):
        outside = _line_naming(_header(mod.render(env.build())), 'outside those roots')

        assert outside.endswith(': none')
