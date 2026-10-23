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
