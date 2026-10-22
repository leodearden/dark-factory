"""``program_doc.py`` driven through its CLI, over a run home whose project_root is a temp checkout.

Each case copies the fixture program with ``project_root`` pointed at a temp
``project/`` checkout, initialises the home with the real ``record.py``, builds
history with the real ``apply_return.py`` and serves a writable copy of the
fixture store that the fake store re-reads per request. Every script runs as an
``-I -S`` subprocess from a temp cwd. Fixture files are only ever copied.
"""
from __future__ import annotations

import contextlib
import json
import re
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from review_all_program_cli import RUN_ID, ProgramEnv
from review_all_program_fixture import FIXTURE_DIR, PROGRAM_DIR, load_state, running_fake_store
from review_all_program_seeded_run import StoreCopy

state = load_state()

SCRIPT = 'program_doc.py'
TEMPLATE = PROGRAM_DIR / 'program_doc_template.md'
DOC_NAME = 'review-all-program-fixture-20261019.md'
STATE_FILES = ('program.json', 'rulings.jsonl', 'ledger.jsonl')
FILED = '## FILED — program status'
COULD_NOT_RUN = 2


@dataclass(frozen=True)
class DocRun:
    """One test's run home, its store copy, and the temp checkout its program publishes into."""

    env: ProgramEnv
    store: StoreCopy
    project: Path

    @property
    def home(self) -> Path:
        return self.env.home

    @property
    def render_doc(self) -> Path:
        return self.home / 'render' / DOC_NAME

    def doc(self, *args: str | Path) -> subprocess.CompletedProcess[str]:
        return self.env.run(SCRIPT, '--run', RUN_ID, *args)

    def applied(self, fixture_return: str, *flags: str) -> None:
        proc = self.env.run('apply_return.py', self.env.put_inbox(fixture_return), *flags)
        assert proc.returncode == 0, proc.stdout + proc.stderr


@contextlib.contextmanager
def _doc_run(tmp_path: Path, *store_flags: str) -> Iterator[DocRun]:
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
        yield DocRun(env, copy, project)


@pytest.fixture
def run(tmp_path: Path) -> Iterator[DocRun]:
    with _doc_run(tmp_path) as doc_run:
        yield doc_run


@pytest.fixture
def decomposed(run: DocRun) -> DocRun:
    run.applied('S1.author.json')
    run.applied('S1.decompose.json')
    return run


def _rendered(run: DocRun, *args: str | Path) -> str:
    proc = run.doc(*args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return run.render_doc.read_text(encoding='utf-8')


def _headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith('## ')]


def _section(text: str, heading: str) -> list[str]:
    """The lines between *heading* and the next ``## `` heading."""
    lines = text.splitlines()
    assert heading in lines, f'no {heading!r} in:\n{text}'
    body = lines[lines.index(heading) + 1:]
    ends = [i for i, line in enumerate(body) if line.startswith('## ')]
    return body[:ends[0]] if ends else body


def _render_files(run: DocRun) -> dict[str, bytes]:
    render = run.home / 'render'
    return {
        str(path.relative_to(render)): path.read_bytes()
        for path in sorted(render.rglob('*')) if path.is_file()
    }


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in re.split(r'(?<!\\)\|', row.strip())[1:-1]]


def _table(section: list[str]) -> list[dict[str, str]]:
    """The section's markdown table as one mapping per body row, keyed by the header cells."""
    rows = [line for line in section if line.startswith('|')]
    assert len(rows) >= 2, 'no table in:\n' + '\n'.join(section)
    header = _cells(rows[0])
    return [dict(zip(header, _cells(row), strict=True)) for row in rows[2:]]


def _filed_row(text: str, stream: str) -> Mapping[str, str]:
    (row,) = [row for row in _table(_section(text, FILED)) if row['Stream'] == stream]
    return row


def _entries(cell: str) -> dict[int, str]:
    """Each ``[<label> ]<id>[ <programme state>]`` entry of a Tasks cell, by its task id."""
    entries = {}
    for entry in cell.split(', '):
        (task_id,) = [int(token) for token in entry.split() if token.isdigit()]
        entries[task_id] = entry
    return entries


def _ids(cell: str) -> set[int]:
    return {int(token) for token in re.findall(r'\b\d+\b', cell)}


def _store_read_stamp(proc: subprocess.CompletedProcess[str]) -> str:
    (stamp,) = re.findall(r'^store read (\S+)$', proc.stdout, re.MULTILINE)
    return stamp


def test_a_render_writes_the_doc_and_a_byte_copy_snapshot_under_the_template_headings(
    decomposed: DocRun,
) -> None:
    text = _rendered(decomposed)

    snapshot = decomposed.home / 'render' / '.state'
    for name in STATE_FILES:
        home_file = decomposed.home / name
        expected = home_file.read_bytes() if home_file.exists() else b''
        assert (snapshot / name).read_bytes() == expected, name
    assert not (decomposed.home / 'rulings.jsonl').exists()
    assert _headings(text) == _headings(TEMPLATE.read_text(encoding='utf-8'))


def test_b5_a_render_after_the_store_moved_statuses_is_byte_identical(decomposed: DocRun) -> None:
    first = decomposed.doc()
    assert first.returncode == 0, first.stdout + first.stderr
    before = _render_files(decomposed)

    decomposed.store.edit({101: {'status': 'done'}, 105: {'status': 'in-progress'}})
    second = decomposed.doc()

    assert second.returncode == 0, second.stdout + second.stderr
    assert _render_files(decomposed) == before
    text = decomposed.render_doc.read_text(encoding='utf-8')
    for line in state.read_ledger(state.Home(decomposed.home)):
        text = text.replace(line['at'], '')
    assert _store_read_stamp(second) not in text


def test_filed_has_one_row_per_stream_with_programme_states(decomposed: DocRun) -> None:
    text = _rendered(decomposed)

    s1 = _filed_row(text, 'S1')
    s1_entries = _entries(s1['Tasks'])
    assert set(s1_entries) == {101, 102, 103, 104, 109, 113}
    assert s1_entries[101] == 'α 101'
    assert s1_entries[103] == 'γ 103 held'
    assert s1_entries[113].split()[-1] == 'released'
    assert s1_entries[109].split()[-1] == 'cancelled-as-accepted'
    assert _ids(s1['Gate deps']) == {101, 104}
    assert set(_entries(_filed_row(text, 'S2')['Tasks'])) == {105, 106}
    assert set(_entries(_filed_row(text, 'S3')['Tasks'])) == {108}
    assert set(_entries(_filed_row(text, 'unstreamed')['Tasks'])) == {107}
    assert any(re.search(r'human gate 110\b', line) for line in _section(text, FILED))

    decomposed.store.edit({102: {'status': 'cancelled'}})
    recancelled = _entries(_filed_row(_rendered(decomposed), 'S1')['Tasks'])
    assert recancelled[102].split()[-1] == 'cancelled-without-reason'


def test_inv13_a_store_without_program_tasks_says_so_instead_of_an_empty_table(run: DocRun) -> None:
    run.store.keep_only([120, 121, 201, 202, 203, 204, 205])

    filed = _section(_rendered(run), FILED)

    assert any('no program task is in the store yet' in line for line in filed)
    assert any(f'no human_gate task carries trigger_chain.run_id = {RUN_ID}' in line
               for line in filed)
    assert not [line for line in filed if line.startswith('|')]


def test_b11_a_store_that_dropped_include_rows_renders_nothing(tmp_path: Path) -> None:
    with _doc_run(tmp_path, '--no-include-rows') as run:
        proc = run.doc()

        assert proc.returncode == COULD_NOT_RUN, proc.stdout + proc.stderr
        assert 'include_rows' in proc.stderr
        assert not (run.home / 'render').exists()


DECISIONS = '## Resolved decisions (do not relitigate)'
HELD = '## Held leaves'
LEDGER = '## Ledger'


def _rule(run: DocRun, ruling: str, *args: str) -> None:
    proc = run.env.run(
        'record.py', 'rule', '--run', RUN_ID, '--question', 'q?', '--ruling', ruling,
        '--source', 'sitting 1', '--option', 'a', '--option', 'b', '--consequence', 'c', *args,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _append(run: DocRun, op: str, **fields: object) -> None:
    home = state.Home(run.home)
    state.append_ledger(home, [{
        'seq': state.next_seq(state.read_ledger(home)), 'at': state.now_utc(), 'by': 'test',
        'op': op, **fields,
    }])


def _bullet(section: list[str], ident: str) -> str:
    """The section's one bullet that opens with *ident*."""
    (line,) = [line for line in section if line.startswith(f'- {ident} ')]
    return line


def test_b12_a_superseded_ruling_shows_only_its_successor_naming_it(run: DocRun) -> None:
    _rule(run, 'alpha owns C1', '--by', 'human')
    _rule(run, 'beta owns C1', '--by', 'human', '--supersedes', 'D1')

    decisions = _section(_rendered(run), DECISIONS)

    d2 = _bullet(decisions, 'D2')
    assert 'beta owns C1' in d2
    assert 'supersedes D1' in d2
    assert not [line for line in decisions if line.startswith('- D1 ')]
    assert not [line for line in decisions if 'alpha owns C1' in line]


def test_a_reask_ruling_is_marked_as_one(run: DocRun) -> None:
    run.applied('S1.author.json')
    _rule(run, 'probe it first', '--by', 'human', '--answers', 'S1:Q1', '--reask')

    assert 're-ask' in _bullet(_section(_rendered(run), DECISIONS), 'D1')


def test_held_leaves_lists_every_task_carrying_a_seam_wait(decomposed: DocRun) -> None:
    rows = _table(_section(_rendered(decomposed), HELD))

    by_leaf = {next(iter(_ids(row['Leaf']))): row for row in rows}
    assert set(by_leaf) == {103, 113}
    held = by_leaf[103]
    assert (held['Waits on'], held['Contract as specified'], held['Release owner']) == (
        'S2', 'C1', 'S2')
    assert held['Released onto'] == 'not released'
    released = by_leaf[113]
    assert _ids(released['Released onto']) == {106}
    assert released['Contract as delivered'] == 'C2: beta.contract() returns a frozen view'


def test_the_ledger_section_has_one_bullet_per_ledger_line_in_seq_order(decomposed: DocRun) -> None:
    bullets = [line for line in _section(_rendered(decomposed), LEDGER) if line.startswith('- ')]

    ledger = state.read_ledger(state.Home(decomposed.home))
    assert len(bullets) == len(ledger)
    for bullet, line in zip(bullets, ledger, strict=True):
        fields = bullet.removeprefix('- ').split(' · ')
        assert (fields[0], fields[3]) == (str(line['seq']), line['op']), bullet


def test_amendments_from_the_ledger_are_rendered(run: DocRun) -> None:
    _rule(run, 'gamma gains the c2 key', '--by', 'lead')
    _append(run, 'amend_program', section='invariants', value={
        'n': 3, 'statement': 'Gamma never re-exports alpha.', 'owner': 'S3', 'inv_id': None})
    _append(run, 'amend_stream', stream='S3', field='keys', change='add',
            value='fk-0000000000c2', ruling='L1')

    text = _rendered(run)

    assert any('Gamma never re-exports alpha.' in line for line in _section(text, '## Invariants'))
    (s3,) = [row for row in _table(_section(text, '## Streams')) if row['ID'] == 'S3']
    assert 'fk-0000000000c2' in s3['Finding keys']


REFUSED = 1


def _scratch_template(tmp_path: Path, *, drop: str | None = None, extra: str = '') -> Path:
    """A copy of the live template, less the line *drop*, plus *extra* at its end."""
    lines = TEMPLATE.read_text(encoding='utf-8').splitlines(keepends=True)
    scratch = tmp_path / 'scratch-template.md'
    scratch.write_text(''.join(line for line in lines if line.rstrip('\n') != drop) + extra,
                       encoding='utf-8')
    return scratch


def _published(run: DocRun, *, extra: str = '') -> Path:
    """A prior publish, simulated: a default render copied to the program's doc_path, plus *extra*."""
    published = run.project / 'plans' / DOC_NAME
    published.write_text(_rendered(run) + extra, encoding='utf-8')
    return published


def test_with_no_published_doc_a_render_can_lose_no_heading(run: DocRun, tmp_path: Path) -> None:
    proc = run.doc('--template', _scratch_template(tmp_path, drop=LEDGER))

    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_b6_a_render_that_drops_a_published_heading_is_refused_and_writes_nothing(
    run: DocRun, tmp_path: Path,
) -> None:
    published = _published(run)
    before = _render_files(run)

    proc = run.doc('--template', _scratch_template(tmp_path, drop=LEDGER))

    assert proc.returncode == REFUSED, proc.stdout + proc.stderr
    assert LEDGER in proc.stderr
    assert str(published) in proc.stderr
    assert _render_files(run) == before


def test_b6_allow_drop_names_the_heading_it_lets_go(run: DocRun, tmp_path: Path) -> None:
    _published(run)

    proc = run.doc('--template', _scratch_template(tmp_path, drop=LEDGER), '--allow-drop', LEDGER)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert any(LEDGER in line for line in proc.stdout.splitlines())
    assert LEDGER not in run.render_doc.read_text(encoding='utf-8').splitlines()


def test_an_allowance_for_a_heading_the_render_keeps_is_refused(run: DocRun) -> None:
    proc = run.doc('--allow-drop', '## Nope')

    assert proc.returncode == REFUSED, proc.stdout + proc.stderr
    assert '## Nope' in proc.stderr
    assert not (run.home / 'render').exists()


def test_a_hand_added_published_heading_refuses_a_default_render(run: DocRun) -> None:
    _published(run, extra='\n## Coordinator interventions\n\n- by hand\n')

    proc = run.doc()

    assert proc.returncode == REFUSED, proc.stdout + proc.stderr
    assert '## Coordinator interventions' in proc.stderr


def test_an_unknown_template_placeholder_is_named(run: DocRun, tmp_path: Path) -> None:
    proc = run.doc('--template', _scratch_template(tmp_path, extra='\n${no_such_section}\n'))

    assert proc.returncode == COULD_NOT_RUN, proc.stdout + proc.stderr
    assert 'no_such_section' in proc.stderr
    assert not (run.home / 'render').exists()


@pytest.mark.parametrize(('args', 'flag'), [
    (('--roster', '--allow-drop', LEDGER), '--allow-drop'),
    (('--sitting', '1', '--template', 'x'), '--template'),
])
def test_render_flags_outside_a_render_are_usage_errors(
    run: DocRun, args: tuple[str, ...], flag: str,
) -> None:
    proc = run.doc(*args)

    assert proc.returncode == COULD_NOT_RUN, proc.stdout + proc.stderr
    assert flag in proc.stderr


DOC_PATHS = (
    f'plans/{DOC_NAME}',
    'plans/review-all-program-fixture-20261019.state/program.json',
    'plans/review-all-program-fixture-20261019.state/rulings.jsonl',
    'plans/review-all-program-fixture-20261019.state/ledger.jsonl',
)
S1_DRAFTS = ('review-all-fixture-s1-prd.md', 'review-all-fixture-s1-prd.capability-manifest.yaml')


def _publish(run: DocRun) -> list[str]:
    """``--publish``'s path list: the lines between its ``published`` header and ``store read``."""
    proc = run.doc('--publish')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    (header,) = [i for i, line in enumerate(lines) if line.startswith('published ')]
    (footer,) = [i for i, line in enumerate(lines) if line.startswith('store read ')]
    return lines[header + 1:footer]


def _draft(run: DocRun, stream: str, name: str) -> Path:
    path = run.home / 'prds' / stream / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{stream} draft {name}\n', encoding='utf-8')
    return path


def _published_files(run: DocRun) -> dict[str, bytes]:
    return {
        str(path.relative_to(run.project)): path.read_bytes()
        for path in sorted(run.project.rglob('*')) if path.is_file()
    }


def test_b6_publish_copies_the_render_and_its_snapshot_and_lists_the_paths(decomposed: DocRun) -> None:
    assert _publish(decomposed) == list(DOC_PATHS)

    render = decomposed.home / 'render'
    assert (decomposed.project / DOC_PATHS[0]).read_bytes() == decomposed.render_doc.read_bytes()
    for path, name in zip(DOC_PATHS[1:], STATE_FILES, strict=True):
        assert (decomposed.project / path).read_bytes() == (render / '.state' / name).read_bytes()
    assert not list(decomposed.project.rglob('.git'))


def test_publish_copies_each_streams_drafts_into_plans(decomposed: DocRun) -> None:
    drafts = [_draft(decomposed, 'S1', name) for name in S1_DRAFTS]

    paths = _publish(decomposed)

    assert paths == [*DOC_PATHS, *(f'plans/{name}' for name in sorted(S1_DRAFTS))]
    for draft in drafts:
        assert (decomposed.project / 'plans' / draft.name).read_bytes() == draft.read_bytes()


def test_a_second_publish_over_an_unchanged_program_changes_nothing(decomposed: DocRun) -> None:
    _draft(decomposed, 'S1', S1_DRAFTS[0])
    first = _publish(decomposed)
    published = _published_files(decomposed)

    assert _publish(decomposed) == first
    assert _published_files(decomposed) == published


def test_publish_self_checks_against_the_doc_it_would_overwrite(decomposed: DocRun) -> None:
    published = _published(decomposed, extra='\n## Coordinator interventions\n\n- by hand\n')
    before = published.read_bytes()

    proc = decomposed.doc('--publish')

    assert proc.returncode == REFUSED, proc.stdout + proc.stderr
    assert '## Coordinator interventions' in proc.stderr
    assert published.read_bytes() == before


def test_two_streams_publishing_one_name_publish_nothing(decomposed: DocRun) -> None:
    _draft(decomposed, 'S1', 'shared-name.md')
    _draft(decomposed, 'S3', 'shared-name.md')

    proc = decomposed.doc('--publish')

    assert proc.returncode == COULD_NOT_RUN, proc.stdout + proc.stderr
    assert 'shared-name.md' in proc.stderr
    assert 'S1' in proc.stderr and 'S3' in proc.stderr
    assert not (decomposed.project / DOC_PATHS[0]).exists()


def test_publish_without_the_checkout_names_project_root(decomposed: DocRun) -> None:
    (decomposed.project / 'plans').rmdir()
    decomposed.project.rmdir()

    proc = decomposed.doc('--publish')

    assert proc.returncode == COULD_NOT_RUN, proc.stdout + proc.stderr
    assert 'project_root' in proc.stderr


def _roster(run: DocRun) -> list[str]:
    proc = run.doc('--roster')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout.splitlines()


def _roster_line(lines: list[str], opening: str) -> str:
    (line,) = [line for line in lines if line.startswith(opening)]
    return line


def test_the_roster_prints_one_line_per_stream_each_held_leaf_and_the_gate(decomposed: DocRun) -> None:
    lines = _roster(decomposed)

    assert _roster_line(lines, 'S1 · ') == (
        'S1 · wave 1 · decomposed · filed 6 · held 1 · gate deps 2 · open actions 2 · '
        'open questions 2')
    assert _roster_line(lines, 'S2 · ') == (
        'S2 · wave 2 · no status · filed 2 · held 0 · gate deps 0 · open actions 0 · '
        'open questions 0')
    assert _roster_line(lines, 'S3 · ').startswith('S3 · wave 1 · no status · filed 1 · ')
    assert _roster_line(lines, 'unstreamed · ') == 'unstreamed · filed 1 · gate deps 0'
    assert [line for line in lines if line.startswith('held ')] == [
        'held 103 (S1 γ) waits on S2 C1 · release owner S2 · S2 filed: 105, 106']
    assert _roster_line(lines, 'human gate ').startswith('human gate 110 ')
    assert lines[-1].startswith('store read ')


def test_an_action_result_closes_the_action_on_the_roster(decomposed: DocRun) -> None:
    _append(decomposed, 'action_result', id='A1', state='applied', batch=1, detail='noted')

    assert ' · open actions 1 · ' in _roster_line(_roster(decomposed), 'S1 · ')


def test_a_ruling_closes_the_question_on_the_roster(decomposed: DocRun) -> None:
    _rule(decomposed, 'alpha owns C1', '--by', 'human', '--answers', 'S1:Q1')

    assert _roster_line(_roster(decomposed), 'S1 · ').endswith(' · open questions 1')
