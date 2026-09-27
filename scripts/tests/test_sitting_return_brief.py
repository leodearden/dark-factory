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
from escalation.models import Escalation
from orchestrator.session_registry import DecisionRecord, normalize_escalations_dir
from sitting import brief, fleet_state, payloads
from sitting import ledger as ledger_mod
from sitting import preparation as prep_mod
from sitting import return_brief as mod
from sitting.inventory import escalation_key
from sitting.payloads import AgreedMarker, PreparedMarker

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 9, 26, 5, 30, tzinfo=UTC)
NOW_ISO = NOW.isoformat()
IN_WINDOW = '2026-09-25T18:00:00+00:00'
BEFORE_WINDOW = '2026-09-20T18:00:00+00:00'
CLOSED_IN_WINDOW = '2026-09-25T20:00:00+00:00'
TITLES = (
    '1. Decisions needed',
    '2. Rulings made under standing policy',
    '3. Landed',
    '4. Stuck and why',
    '5. Spend and cap hits',
    '6. Autonomous closes',
)
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
    handover: Path
    preparation: Path
    out: Path

    @property
    def queue(self) -> Path:
        return self.df / 'data' / 'escalations'

    @property
    def output(self) -> Path:
        return self.out / 'nested' / 'return-brief.md'

    @property
    def ledger_out(self) -> Path:
        return self.out / 'sitting' / 'ledger-nightly.json'

    def argv(self, *roots: Path) -> list[str]:
        return [
            *[arg for root in (roots or (self.df,)) for arg in ('--project-root', str(root))],
            '--decisions-root', str(self.fleet),
            '--sessions-root', str(self.sessions),
            '--handover', str(self.handover),
            '--preparation', str(self.preparation),
            '--output', str(self.output),
            '--ledger-out', str(self.ledger_out),
            '--now', NOW_ISO,
        ]

    def build(self, *roots: Path) -> mod.ReturnBriefDocument:
        return mod.build(
            project_roots=[str(root) for root in (roots or (self.df,))],
            decisions_root=self.fleet,
            sessions_root=self.sessions,
            handover_path=self.handover,
            preparation_path=self.preparation,
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
    e = Env(tmp_path, df, tmp_path / 'fleet', tmp_path / 'fleet' / 'sessions', tmp_path / 'handover.md',
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


class TestSections:
    def test_the_six_mandated_sections_in_order_as_level_two_headings(self, env):
        text = mod.render(env.build())

        assert _partition(text)[0] == list(TITLES)

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
        assert list(sections) == list(TITLES)
        assert 'reify' in sections[TITLES[2]] and 'source_missing' in sections[TITLES[2]]
        assert 'runs_db' in sections[TITLES[2]]
        assert 'tasks_db' in sections[TITLES[3]]


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

    def test_a_second_run_overwrites_and_the_ledger_starts_fresh_each_night(self, env, capsys):
        _run(capsys, env.argv())
        _esc(env.queue, id='esc-101-1', status='resolved', resolved_at=NOW_ISO, resolved_by='interactive')

        _run(capsys, env.argv())

        assert env.output.read_text().count('# Return brief') == 1
        ledger = ledger_mod.load(env.ledger_out)
        assert ledger is not None
        assert list(ledger.in_state('done')) == []
        assert escalation_key(normalize_escalations_dir(env.queue), 'esc-101-1') not in {
            tuple(json.loads(key)) for key in ledger.entries}

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
