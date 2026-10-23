"""``gen_brief.py`` driven through its CLI, over a run home whose project_root is a temp checkout.

Each case copies the fixture program with ``project_root`` pointed at a temp
``project/`` checkout, initialises the home with the real ``record.py``, writes
the run record there (the fixture record plus one filings row for an S1 key),
builds history with the real ``apply_return.py`` and ``record.py rule``, and
serves a writable copy of the fixture store that the fake store re-reads per
request. Every script runs as an ``-I -S`` subprocess from a temp cwd. A brief
is read by its ``## `` headings, named by ``gen_brief.Section``. Fixture files
are only ever copied.
"""
from __future__ import annotations

import contextlib
import json
import re
import shlex
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from review_all_program_cli import RUN_ID, ProgramEnv, load_program_module
from review_all_program_fixture import FIXTURE_DIR, PROGRAM_DIR, running_fake_store
from review_all_program_seeded_run import StoreCopy

SCRIPT = 'gen_brief.py'
S1_PRD = 'plans/review-all-fixture-s1-prd.md'
S1_FILING = {
    'key': 'fk-0000000000a1', 'sources': ['review-all'], 'step': 'open',
    'ticket_or_task': None, 'disposition': 'open → deliberation',
}
"""The filings row the record copy gains: the fixture record files only the routed key fk-…d1."""


def _section_heading(name: str) -> Any:
    """``gen_brief.Section.<name>``, loaded when a test reads a brief."""
    return getattr(load_program_module('gen_brief').Section, name)


@dataclass(frozen=True)
class BriefRun:
    """One test's run home, its store copy, and the temp checkout its program names."""

    env: ProgramEnv
    store: StoreCopy
    project: Path
    record_path: str

    @property
    def home(self) -> Path:
        return self.env.home

    @property
    def record(self) -> Path:
        return self.project / self.record_path

    def brief(self, *args: str | Path) -> subprocess.CompletedProcess[str]:
        return self.env.run(SCRIPT, '--run', RUN_ID, *args)

    def applied(self, fixture_return: str, *flags: str) -> None:
        proc = self.env.run('apply_return.py', self.env.put_inbox(fixture_return), *flags)
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def written(self, *args: str | Path) -> str:
        """The brief ``gen_brief.py`` wrote for *args*, which must exit 0 and print its path."""
        proc = self.brief(*args)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        briefs = self.home / 'briefs'
        (path,) = [Path(token) for token in proc.stdout.split() if Path(token).parent == briefs]
        return path.read_text(encoding='utf-8')


@contextlib.contextmanager
def _brief_run(tmp_path: Path, *store_flags: str) -> Iterator[BriefRun]:
    copy = StoreCopy.of('store.json', tmp_path)
    with running_fake_store(*store_flags, store=copy.path) as fake:
        env = ProgramEnv.under(tmp_path / 'run', fake.url)
        project = tmp_path / 'project'
        (project / 'plans').mkdir(parents=True)
        program = json.loads((FIXTURE_DIR / 'program.json').read_text())
        program_json = tmp_path / 'program.json'
        program_json.write_text(json.dumps({**program, 'project_root': str(project)}, indent=2))
        proc = env.run('record.py', 'init', program_json)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        record = json.loads((FIXTURE_DIR / 'record.json').read_text())
        record['filings'].append(S1_FILING)
        (project / program['record_path']).write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8',
        )
        yield BriefRun(env, copy, project, program['record_path'])


@pytest.fixture
def run(tmp_path: Path) -> Iterator[BriefRun]:
    with _brief_run(tmp_path) as brief_run:
        yield brief_run


def section(text: str, heading: str) -> list[str]:
    """The lines under ``## <heading>``, up to the next ``## `` heading."""
    lines = text.splitlines()
    marker = f'## {heading}'
    assert marker in lines, f'no {marker!r} in:\n{text}'
    body = lines[lines.index(marker) + 1:]
    ends = [i for i, line in enumerate(body) if line.startswith('## ')]
    return body[:ends[0]] if ends else body


def has_section(text: str, heading: str) -> bool:
    return f'## {heading}' in text.splitlines()


def fenced(lines: list[str], lang: str) -> list[str]:
    """The body of the first ```<lang> block in *lines*."""
    start = lines.index(f'```{lang}')
    end = lines.index('```', start + 1)
    return lines[start + 1:end]


def example_of(text: str) -> dict[str, Any]:
    return json.loads('\n'.join(fenced(section(text, _section_heading('RETURN')), 'json')))


def check_argv(text: str) -> list[str]:
    (line,) = fenced(section(text, _section_heading('RETURN')), 'sh')
    return shlex.split(line)


def run_check_line(run: BriefRun, argv: list[str]) -> subprocess.CompletedProcess[str]:
    """The brief's check line, run as the env runs a program script (argv[0] is that script)."""
    assert argv[0] == str(PROGRAM_DIR / 'apply_return.py'), argv
    return run.env.run('apply_return.py', *argv[1:])


def pass_keys(pass_name: str) -> frozenset[str]:
    state = load_program_module('state')
    keys = state.RETURN_KEYS[state.Pass(pass_name)]
    return keys.required | keys.optional


def test_an_author_brief_is_written_under_briefs_and_only_its_path_is_printed(run: BriefRun) -> None:
    proc = run.brief('S1', 'author')

    assert proc.returncode == 0, proc.stdout + proc.stderr
    path = run.home / 'briefs' / 'S1.author.md'
    assert path.is_file()
    assert str(path) in proc.stdout.split()
    assert not (run.home / 'sessions.json').exists()
    assert not has_section(path.read_text(encoding='utf-8'), _section_heading('SPAWN'))
    assert 'spawn-claude.sh' not in proc.stdout


def test_the_return_section_names_the_schema_and_every_key_of_the_pass(run: BriefRun) -> None:
    state = load_program_module('state')
    text = run.written('S1', 'author')

    body = '\n'.join(section(text, _section_heading('RETURN')))
    assert state.RETURN_SCHEMA in body
    for key in pass_keys('author'):
        assert f'`{key}`' in body, key


def test_the_author_example_passes_the_briefs_own_check_line(run: BriefRun) -> None:
    text = run.written('S1', 'author')
    example = example_of(text)
    inbox = run.home / 'inbox' / 'S1.author.json'

    assert str(inbox) in '\n'.join(section(text, _section_heading('RETURN')))
    inbox.write_text(json.dumps(example, indent=2, ensure_ascii=False), encoding='utf-8')
    argv = check_argv(text)
    assert '--check' in argv and str(inbox) in argv
    proc = run_check_line(run, argv)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'check ok' in proc.stdout
    assert set(example) == pass_keys('author')


def test_a_brief_after_the_author_return_is_applied_checks_with_amend_and_the_recorded_prd(
    run: BriefRun,
) -> None:
    run.applied('S1.author.json')

    text = run.written('S1', 'author')

    assert check_argv(text)[-1] == '--amend'
    assert example_of(text)['prd_path'] == S1_PRD


def test_an_unknown_stream_is_refused_and_nothing_is_written(run: BriefRun) -> None:
    proc = run.brief('S9', 'author')

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert 'S9' in proc.stderr
    assert not (run.home / 'briefs').exists()


D1_KEY = 'fk-0000000000d1'
GREP_RULE = 'grep -c over src/fixture'


def rule(run: BriefRun, *args: str) -> str:
    """Record one human ruling with the real ``record.py rule``; returns its minted id."""
    proc = run.env.run(
        'record.py', 'rule', '--run', RUN_ID, '--by', 'human', '--question', 'Which way?',
        '--ruling', 'this way', '--source', 'the sitting', '--option', 'this way',
        '--option', 'that way', '--consequence', 'the PRD goes this way', *args,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout.split()[1]


def bullets(lines: list[str]) -> list[str]:
    return [line[2:] for line in lines if line.startswith('- ')]


def stream_cells(text: str) -> dict[str, str]:
    """The stream section's ``- <column>: <cell>`` lines, by column."""
    return dict(entry.split(': ', 1) for entry in bullets(section(text, _section_heading('STREAM'))))


def finding_text(run: BriefRun, key: str) -> str:
    (finding,) = [f for f in json.loads(run.record.read_text())['findings'] if f['key'] == key]
    return finding['statement']


def filings_rows(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in fenced(section(text, _section_heading('FILINGS')), 'jsonl')]


def test_the_stream_row_findings_and_filings_follow_the_ledgers_amendments(run: BriefRun) -> None:
    state = load_program_module('state')
    home = state.Home(run.home)
    state.append_ledger(home, [{
        'seq': state.next_seq(state.read_ledger(home)), 'at': state.now_utc(), 'by': 'test',
        'op': str(state.LedgerOp.AMEND_STREAM), 'stream': 'S1', 'field': str(state.StreamField.KEYS),
        'change': str(state.AmendChange.ADD), 'value': D1_KEY, 'ruling': 'D1',
    }])

    text = run.written('S1', 'author')

    cells = stream_cells(text)
    s1 = json.loads((FIXTURE_DIR / 'program.json').read_text())['streams'][0]
    for value in (s1['name'], s1['slug'], s1['remedy']):
        assert value in cells.values(), value
    assert cells['Wave'] == '1'
    assert set(cells['Finding keys'].split(', ')) == {*s1['keys'], D1_KEY}
    assert finding_text(run, D1_KEY) in '\n'.join(section(text, _section_heading('FINDINGS')))
    assert [row['disposition'] for row in filings_rows(text) if row['key'] == D1_KEY] == ['filed:107']


def test_the_findings_carry_statements_measurements_and_the_dispute_beside_the_verdict(
    run: BriefRun,
) -> None:
    text = run.written('S1', 'author')

    findings = section(text, _section_heading('FINDINGS'))
    body = '\n'.join(findings)
    for key in ('fk-0000000000a1', 'fk-0000000000a2'):
        assert finding_text(run, key) in body, key
    assert sum(GREP_RULE in line for line in findings) == 2
    (disputed,) = [line for line in findings if 'weakened' in line]
    assert 'confirmed' in disputed
    assert 'two of the three private imports are test-only' in disputed
    assert finding_text(run, 'fk-0000000000b1') not in text


def test_the_filings_rows_are_the_records_rows_for_the_streams_keys_verbatim(run: BriefRun) -> None:
    text = run.written('S1', 'author')

    assert filings_rows(text) == [S1_FILING]


def test_the_xrefs_addressed_to_the_stream_are_carried(run: BriefRun) -> None:
    run.applied('S2.author.json')

    xrefs = '\n'.join(section(run.written('S1', 'author'), _section_heading('XREFS')))

    assert 'C2 is delivered by S2 α and releases S1 ζ.' in xrefs
    assert 'S2' in xrefs


def test_only_current_non_reask_rulings_binding_the_stream_are_carried(run: BriefRun) -> None:
    assert rule(run, '--stream', 'S1') == 'D1'
    assert rule(run, '--supersedes', 'D1', '--stream', 'S1') == 'D2'
    assert rule(run, '--stream', 'S3') == 'D3'
    assert rule(run) == 'D4'
    run.applied('S1.author.json')
    assert rule(run, '--answers', 'S1:Q1', '--reask') == 'D5'

    rulings = bullets(section(run.written('S1', 'author'), _section_heading('RULINGS')))

    assert {line.split()[0] for line in rulings} == {'D2', 'D4'}
    (d2,) = [line for line in rulings if line.split()[0] == 'D2']
    assert 'D1' in d2.replace(',', ' ').split()


def test_the_author_brief_states_the_sitting_rule(run: BriefRun) -> None:
    sitting = '\n'.join(section(run.written('S1', 'author'), _section_heading('SITTING')))

    assert 'lead_rulable' in sitting


def test_a_brief_is_rebuilt_from_state_on_every_run(run: BriefRun) -> None:
    run.written('S1', 'author')
    ruling_id = rule(run, '--stream', 'S1')

    rulings = bullets(section(run.written('S1', 'author'), _section_heading('RULINGS')))

    assert ruling_id in {line.split()[0] for line in rulings}


def test_an_absent_run_record_is_refused_naming_it(run: BriefRun) -> None:
    run.record.unlink()

    proc = run.brief('S1', 'author')

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert str(run.record) in proc.stderr
    assert not (run.home / 'briefs' / 'S1.author.md').exists()


def test_a_finding_without_a_statement_is_refused_naming_the_field(run: BriefRun) -> None:
    record = json.loads(run.record.read_text())
    del record['findings'][0]['statement']
    run.record.write_text(json.dumps(record), encoding='utf-8')

    proc = run.brief('S1', 'author')

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert 'findings[0].statement' in proc.stderr
    assert not (run.home / 'briefs' / 'S1.author.md').exists()


S1_UPSTREAM_TASKS = {101, 102, 103, 104, 109, 113}
"""S1's program tasks in the fixture store; 104 and 113 appear in no return."""


@pytest.fixture
def decomposed(run: BriefRun) -> BriefRun:
    run.applied('S1.author.json')
    run.applied('S1.decompose.json')
    run.applied('S2.author.json')
    return run


def task_ids(lines: list[str]) -> set[int]:
    """The task id each ``- <id> …`` line opens with."""
    return {int(entry.split()[0]) for entry in bullets(lines)}


def task_line(lines: list[str], task_id: int) -> str:
    (line,) = [entry for entry in bullets(lines) if entry.split()[0] == str(task_id)]
    return line


class NoRows:
    """A StoreView whose every read answers no rows."""

    def task(self, task_id: int) -> Any:
        state = load_program_module('state')
        return state.StoreRead((), state.now_utc())

    def gate(self, run_id: str) -> Any:
        state = load_program_module('state')
        return state.StoreRead((), state.now_utc())


def test_the_decompose_brief_reads_the_upstream_program_tasks_from_the_store(
    decomposed: BriefRun,
) -> None:
    proc = decomposed.brief('S2', 'decompose')

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert re.search(r'^store read \S+$', proc.stdout, re.MULTILINE), proc.stdout
    upstream = section(decomposed.written('S2', 'decompose'), _section_heading('UPSTREAM'))
    assert task_ids(upstream) == S1_UPSTREAM_TASKS
    assert 'held' in task_line(upstream, 103).split()


def test_the_gate_section_names_the_gate_or_says_none_carries_the_run(decomposed: BriefRun) -> None:
    gate = '\n'.join(section(decomposed.written('S2', 'decompose'), _section_heading('GATE')))
    assert re.search(r'\b110\b', gate), gate

    decomposed.store.edit({110: None})
    gate = '\n'.join(section(decomposed.written('S2', 'decompose'), _section_heading('GATE')))

    assert f'no human_gate task carries trigger_chain.run_id = {RUN_ID}' in gate


def test_the_release_duties_are_the_unreleased_held_leaves_the_stream_owns(
    decomposed: BriefRun,
) -> None:
    duties = section(decomposed.written('S2', 'decompose'), _section_heading('RELEASE_DUTIES'))

    assert task_ids(duties) == {103}


def test_the_decompose_brief_carries_the_xrefs_addressed_to_the_stream(decomposed: BriefRun) -> None:
    xrefs = '\n'.join(section(decomposed.written('S2', 'decompose'), _section_heading('XREFS')))

    assert "S2 consumes C2 once S1's alpha task 101 lands." in xrefs


def test_the_decompose_brief_has_the_seat_sections_and_no_spawn(decomposed: BriefRun) -> None:
    proc = decomposed.brief('S2', 'decompose')

    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = (decomposed.home / 'briefs' / 'S2.decompose.md').read_text(encoding='utf-8')
    for name in ('STREAM', 'RULINGS', 'FILINGS', 'RETURN'):
        assert has_section(text, _section_heading(name)), name
    assert not has_section(text, _section_heading('SPAWN'))
    assert not (decomposed.home / 'sessions.json').exists()
    assert 'spawn-claude.sh' not in proc.stdout


def test_the_decompose_example_passes_the_form_checks(decomposed: BriefRun) -> None:
    state = load_program_module('state')
    home = state.Home(decomposed.home)
    example = example_of(decomposed.written('S2', 'decompose'))

    checked = state.validate_return(
        json.dumps(example),
        program=state.parse_program(json.loads(home.program_json.read_text())),
        ledger=state.read_ledger(home), store=NoRows(),
    )

    assert [r for r in checked.rejections if r.kind is state.RejectionClass.FORM] == []
    assert set(example) == pass_keys('decompose')


def test_a_decompose_before_the_streams_author_return_is_refused(run: BriefRun) -> None:
    state = load_program_module('state')
    home = state.Home(run.home)
    views = state.stream_views(
        state.parse_program(json.loads(home.program_json.read_text())), state.read_ledger(home),
    )

    proc = run.brief('S2', 'decompose')

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert state.unauthored_reason(views['S2']) in proc.stderr
    assert not (run.home / 'briefs' / 'S2.decompose.md').exists()


def test_a_decompose_before_its_upstream_is_decomposed_is_refused(run: BriefRun) -> None:
    run.applied('S1.author.json')
    run.applied('S2.author.json')

    proc = run.brief('S2', 'decompose')

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert 'S1' in proc.stderr and 'authored' in proc.stderr
    assert not (run.home / 'briefs' / 'S2.decompose.md').exists()


def test_a_store_without_include_rows_is_refused_naming_it(tmp_path: Path) -> None:
    with _brief_run(tmp_path, '--no-include-rows') as run:
        run.applied('S3.author.json')

        proc = run.brief('S3', 'decompose')

        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert 'include_rows' in proc.stderr
        assert not (run.home / 'briefs' / 'S3.decompose.md').exists()
