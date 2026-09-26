"""Tests for scripts/sitting/brief.py — the pure sitting-brief renderer (task 5376)."""
from __future__ import annotations

import json
import re

import pytest
from sitting import brief as mod
from sitting.gates import CarveoutFacts, Fact, evaluate_carveout
from sitting.inventory import Glossary, OpenItem, decision_key, escalation_key
from sitting.ownership import PROBES, OwnershipFinding, ProbeResult
from sitting.payloads import (
    Finding,
    Ruling,
    resolve_issue_payload,
    route_finding_to_owner,
    update_task_payload,
)
from sitting.preparation import (
    Cites,
    EscalationClosed,
    Manual,
    NoLean,
    Option,
    Preparation,
    Recommended,
    Standing,
    TaskStatusIs,
)

QUEUE = '/src/dark-factory/data/escalations'
PROJECT = 'dark_factory'
NOW = '2026-09-26T09:00:00+00:00'
MEASURED = '2026-09-26T08:59:00+00:00'
HANDOVER = 'l2-watcher-handover-2026-09-22.md'
GLOSSARY = Glossary(
    escalations={QUEUE: {
        'esc-3881-3': 'Retarget\n3881 onto the sitting preparer?',
        'esc-3881-2': 'the earlier retarget ask',
        'esc-4001-1': 'cap the nightly spend?',
        'esc-4002-1': 'drop the legacy flag?',
        'esc-4003-1': 'rename the timer?',
    }},
    tasks={PROJECT: {'3881': 'Sitting preparer', '4811': 'Fleet redeploy lease'}},
    shortfalls=(),
)
FREE_ESC_RE = re.compile(r'(?<![\w-])esc-\d+-\d+')


def _item(n: int = 3881, seq: int = 3, **overrides) -> OpenItem:
    esc_id = f'esc-{n}-{seq}'
    fields = {
        'key': escalation_key(QUEUE, esc_id),
        'escalation_id': esc_id,
        'queue_dir': QUEUE,
        'project': PROJECT,
        'task_id': str(n),
        'severity': 'blocking',
        'text': f'raw question for {esc_id}',
        'options': ('close it', 'hold it'),
    }
    return OpenItem(**{**fields, **overrides})


def _finding(item: OpenItem, *, handover: bool = True, unavailable: tuple[str, ...] = ()) -> OwnershipFinding:
    probes = []
    for probe in PROBES:
        if probe in unavailable:
            probes.append(ProbeResult(probe, 'unavailable', MEASURED, evidence='no sessions root at /nowhere'))
        elif probe == 'handover' and handover:
            probes.append(ProbeResult(
                probe, 'mentions', MEASURED, owner=HANDOVER,
                evidence=f'## Untriaged — start here\n{item.escalation_id} still needs a ruling',
            ))
        else:
            probes.append(ProbeResult(probe, 'empty', MEASURED, evidence='nothing'))
    return OwnershipFinding(item.key, tuple(probes))


def _prep(item: OpenItem, **overrides) -> Preparation:
    fields = {
        'item_key': item.key,
        'question': f'Close {item.escalation_id} against the option-C ruling Leo already made?',
        'options': (
            Option('A', 'close as ruled', 'task 3881 keeps its retargeted scope; esc-3881-2 stays closed'),
            Option('B', 'hold for a re-ruling', 'task 3881 stays blocked until Leo rules again'),
        ),
        'recommendation': Recommended('A', 'the description of task 3881 opens RETARGETED 2026-09-01'),
        'on_apply': f'{item.escalation_id} resolves close_only; task 3881 x_ruling stamped',
        'prepared_at': '2026-09-26T05:40:00+00:00',
        'prepared_by': 'nightly-fable',
        'cites': Cites(escalation_ids=('esc-3881-2',), task_ids=('3881',)),
    }
    return Preparation(**{**fields, **overrides})


def _entry(number: int, item: OpenItem | None = None, *, prepared: bool = True, **prep_overrides) -> mod.BriefEntry:
    item = item or _item(4000 + number, 1)
    return mod.BriefEntry(
        number=number,
        item=item,
        preparation=_prep(item, **prep_overrides) if prepared else None,
        ownership=_finding(item),
    )


def _payloads(item: OpenItem):
    ruling = Ruling(item.escalation_id or '', 'close_only', 'Leo: option C', 'leo', NOW)
    return (
        resolve_issue_payload(item.escalation_id or '', 'ruled: option C', 'close_only',
                              resolved_by='leo', resolution_turns=1),
        update_task_payload('3881', '/src/dark-factory', ruling, existing='esc-5580-4 / Leo 2026-09-21: action D'),
    )


def _standing(number: int, standing: Standing, release: mod.ReleaseProbe | None, **kw) -> mod.StandingEntry:
    return mod.StandingEntry(number=number, item=_item(4100 + number, 1), standing=standing, release=release, **kw)


def _section(text: str, title: str) -> str:
    after = text.split(f'## {title}', 1)[1]
    return re.split(r'^## ', after, maxsplit=1, flags=re.MULTILINE)[0]


def _headings(text: str) -> int:
    count, fence = 0, ''
    for line in text.splitlines():
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if marker and (not fence or marker.group(1).startswith(fence)):
            fence = '' if fence else marker.group(1)
        elif not fence and re.match(r'^\s{0,3}#{1,6}(\s|$)', line):
            count += 1
    return count


def _all_held_verdict(esc_id: str):
    held = Fact(held=True, evidence=f'Leo 2026-09-21: close {esc_id}, option C', source_kind='task_description')
    return evaluate_carveout(CarveoutFacts(
        escalation_id=esc_id,
        ruling=held,
        executed=Fact(held=True, evidence='commit abc123 landed the retarget'),
        session_terminated=Fact(held=True, evidence='session 5376-x ended done'),
        pins_recovery=(),
        do_not_close_companions=(),
        sideways=Fact(held=True, evidence='no member chain'),
    ), recommend_only=True)


def _render(numbered=(), standing=(), done=()):
    return mod.render_brief(list(numbered), list(standing), list(done), glossary=GLOSSARY, generated_at=NOW)


class TestSections:
    def test_three_sections_in_order_with_ledger_numbers(self):
        numbered = [_entry(1), _entry(3), _entry(4)]
        done = [mod.DoneEntry(2, escalation_key(QUEUE, 'esc-4002-1'), MEASURED)]

        text = _render(numbered, [], done)

        positions = [text.index(f'## {title}') for title in ('Decisions needed', 'Standing / no action', 'Done')]
        assert positions == sorted(positions)
        assert [int(n) for n in re.findall(r'^### (\d+)\. ', text, flags=re.MULTILINE)] == [1, 3, 4]
        assert '**2.**' in _section(text, 'Done')
        assert '**2.**' not in _section(text, 'Decisions needed')

    def test_empty_input_says_so(self):
        text = _render()

        assert 'No decisions needed' in _section(text, 'Decisions needed')
        assert _section(text, 'Standing / no action').strip()
        assert _section(text, 'Done').strip()

    def test_the_generated_at_stamp_is_injected(self):
        assert NOW in _render()


class TestNumberedEntry:
    def test_parts_render_in_order(self):
        item = _item()
        entry = mod.BriefEntry(3, item, _prep(item), _finding(item), _payloads(item))

        text = _render([entry])

        markers = [
            'against the option-C ruling Leo already made?',
            'close as ruled',
            'keeps its retargeted scope',
            'Recommendation:',
            'On apply:',
            'nothing owns this',
            '```json',
        ]
        positions = [text.index(marker) for marker in markers]
        assert positions == sorted(positions)

    def test_every_option_carries_its_ramification(self):
        text = _render([_entry(1)])

        assert 'task 3881 keeps its retargeted scope' in text
        assert 'task 3881 stays blocked until Leo rules again' in text

    def test_a_recommendation_shows_its_evidence_chain(self):
        text = _render([_entry(1)])

        assert re.search(r'Recommendation:\*?\*? A\b.*RETARGETED 2026-09-01', text)

    def test_no_lean_is_rendered_explicitly(self):
        text = _render([_entry(1, recommendation=NoLean('both options are reversible'))])

        assert 'No lean — both options are reversible' in text
        assert 'Recommendation:' not in text

    def test_a_superseded_value_is_quoted(self):
        item = _item()

        text = _render([mod.BriefEntry(3, item, _prep(item), _finding(item), _payloads(item))])

        assert 'esc-5580-4 / Leo 2026-09-21: action D' in text
        assert 'replaces' in text

    def test_the_ownership_line_names_every_probe_that_returned_empty(self):
        text = _render([_entry(1)])

        assert ('nothing owns this: checked spawned_session, unblock_run, coalesce_fold, task_ruling, '
                'spawned_followup: empty; handover: mentions in ' + HANDOVER) in text
        assert 'Untriaged — start here' in text

    def test_an_unavailable_probe_is_never_read_as_nothing_owns_this(self):
        item = _item(4001, 1)
        entry = mod.BriefEntry(1, item, _prep(item), _finding(item, unavailable=('spawned_session',)))

        text = _render([entry])

        assert 'nothing owns this' not in text
        assert 'could not check spawned_session: no sessions root at /nowhere' in text

    def test_each_payload_is_a_fenced_json_block_of_tool_and_args(self):
        item = _item()
        payloads = _payloads(item)

        text = _render([mod.BriefEntry(3, item, _prep(item), _finding(item), payloads)])

        blocks = [json.loads(block) for block in re.findall(r'^```json\n(.*?)\n```$', text, re.S | re.M)]
        assert blocks == [{'tool': p.tool, 'args': p.args} for p in payloads]

    def test_an_unprepared_item_is_shown_awaiting_preparation_never_dropped(self):
        item = _item(4001, 1)

        text = _render([mod.BriefEntry(1, item, None, _finding(item))])

        assert 'awaiting preparation' in text
        assert 'close it' in text and 'hold it' in text
        assert 'nothing owns this' in text

    def test_a_decision_only_item_is_headed_by_its_decision_id(self):
        item = OpenItem(key=decision_key('df-esc-2683-1'), decision_id='df-esc-2683-1', project=PROJECT,
                        text='keep the cap?', options=('keep', 'lift'))

        text = _render([mod.BriefEntry(1, item, None, OwnershipFinding(item.key, ()))])

        assert re.search(r'^### 1\. decision df-esc-2683-1', text, flags=re.MULTILINE)


class TestGlossing:
    def test_cite_glosses_escalations_in_their_queue_and_tasks_in_their_project(self):
        assert mod.cite(mod.Citation('esc', 'esc-3881-3', QUEUE), GLOSSARY) == (
            'esc-3881-3 (Retarget 3881 onto the sitting preparer?)'
        )
        assert mod.cite(mod.Citation('task', '3881', PROJECT), GLOSSARY) == 'task 3881 (Sitting preparer)'

    @pytest.mark.parametrize('ref', [
        mod.Citation('esc', 'esc-9-9', QUEUE),
        mod.Citation('esc', 'esc-3881-3', '/elsewhere/data/escalations'),
        mod.Citation('task', '9', PROJECT),
        mod.Citation('task', '3881', 'solar'),
    ])
    def test_an_id_with_no_gloss_says_why(self, ref):
        text = mod.cite(ref, GLOSSARY)

        assert re.fullmatch(rf'(task )?{re.escape(ref.id)} \(no gloss: [^)]+\)', text)

    def test_every_escalation_id_is_glossed_on_its_first_occurrence(self):
        item = _item()
        numbered = [
            mod.BriefEntry(3, item, _prep(item), _finding(item), _payloads(item)),
            _entry(4, recommendation=NoLean('esc-4003-1 settles it either way')),
            _entry(5, prepared=False),
        ]
        standing = [_standing(6, Standing('hold', 'Leo', EscalationClosed('esc-4001-1'), 'Leo HOLD on esc-4002-1'),
                              mod.ReleaseProbe('pending', False, MEASURED))]
        done = [mod.DoneEntry(2, escalation_key(QUEUE, 'esc-4002-1'), MEASURED)]

        text = _render(numbered, standing, done)

        seen: set[str] = set()
        for match in FREE_ESC_RE.finditer(text):
            if match.group(0) not in seen:
                seen.add(match.group(0))
                assert text[match.end():match.end() + 2] == ' (', f'{match.group(0)} first appears unglossed'
        assert {'esc-3881-3', 'esc-3881-2', 'esc-5580-4', 'esc-4003-1', 'esc-4001-1', 'esc-4002-1'} <= seen

    def test_cited_tasks_are_glossed(self):
        assert 'task 3881 (Sitting preparer)' in _render([_entry(1)])

    def test_an_escalation_id_inside_a_decision_id_is_left_alone(self):
        text = _render([_entry(1, question='Close decision df-esc-2683-1 now?')])

        assert 'df-esc-2683-1 now?' in text


class TestStandingFooter:
    def test_machine_probeable_releases_show_the_measured_state(self):
        hold = Standing('hold', 'Leo', TaskStatusIs('4811', ('done',)), 'Leo HOLD 2026-08-27')
        pin = Standing('pin', 'esc-3105-5', EscalationClosed('esc-4001-1'), 'veto-pin-do-not-close:3105')

        text = _render([], [
            _standing(7, hold, mod.ReleaseProbe('in-progress', False, MEASURED)),
            _standing(8, pin, mod.ReleaseProbe('resolved', True, MEASURED)),
        ])
        footer = _section(text, 'Standing / no action')

        assert 'task 4811 (Fleet redeploy lease)' in footer
        assert 'in-progress' in footer and MEASURED in footer
        assert 'RELEASED' in footer.split('**8.**', 1)[1]
        assert 'RELEASED' not in footer.split('**8.**', 1)[0]

    def test_a_manual_release_is_labelled_not_machine_probeable(self):
        leo = Standing('leo_owned', 'Leo', Manual('Leo says when'), 'handover: with a human already on it')

        footer = _section(_render([], [_standing(7, leo, None)]), 'Standing / no action')

        assert 'not machine-probeable — re-probe by hand: Leo says when' in footer

    def test_owner_and_kind_are_shown_with_any_routing_payload(self):
        finding = Finding('esc-4107-1', 'cap overrun seen', 'the nightly ran past its cap twice')
        routing = route_finding_to_owner('5255', 'pending', finding, '/src/dark-factory')
        owned = Standing('owned', 'task 5255', TaskStatusIs('5255', ('done', 'cancelled')), 'x_coalesced_into=5255')

        footer = _section(
            _render([], [_standing(7, owned, mod.ReleaseProbe('pending', False, MEASURED), routing=routing)]),
            'Standing / no action',
        )

        assert isinstance(routing.args, dict)
        assert 'owned' in footer and 'task 5255' in footer
        assert json.dumps(routing.args['details']) in footer

    def test_a_standing_item_never_appears_in_the_numbered_list(self):
        hold = Standing('hold', 'Leo', Manual('x'), 'Leo HOLD')

        text = _render([_entry(1)], [_standing(7, hold, None)])

        assert '**7.**' not in _section(text, 'Decisions needed')
        assert not re.search(r'^### 7\. ', text, flags=re.MULTILINE)

    def test_a_manual_release_cannot_carry_a_probe(self):
        with pytest.raises(ValueError):
            _standing(7, Standing('hold', 'Leo', Manual('x'), 'e'), mod.ReleaseProbe('open', False, MEASURED))


class TestEntryInvariants:
    def test_an_owned_item_is_refused_from_the_numbered_list(self):
        item = _item()
        owns = OwnershipFinding(item.key, (ProbeResult('coalesce_fold', 'owns', MEASURED, owner='task 5255'),))

        with pytest.raises(ValueError):
            mod.BriefEntry(1, item, None, owns)

    def test_a_standing_preparation_is_refused_from_the_numbered_list(self):
        item = _item()
        prep = _prep(item, standing=Standing('hold', 'Leo', Manual('x'), 'e'))

        with pytest.raises(ValueError):
            mod.BriefEntry(1, item, prep, _finding(item))

    def test_the_parts_must_describe_the_same_item(self):
        item, other = _item(), _item(4001, 1)

        with pytest.raises(ValueError):
            mod.BriefEntry(1, item, _prep(other), _finding(item))
        with pytest.raises(ValueError):
            mod.BriefEntry(1, item, None, _finding(other))


class TestCloses:
    def _closes(self, count: int) -> list[mod.CloseRecord]:
        closes = []
        for n in range(1, count + 1):
            item = _item(4200 + n, 1)
            closes.append(mod.CloseRecord(item, _all_held_verdict(item.escalation_id or '')))
        return closes

    def test_each_close_shows_all_six_gates_with_evidence_verbatim_in_a_fence(self):
        text = mod.render_closes(self._closes(1), glossary=GLOSSARY)

        for gate in ('ruling_is_leos_own', 'ruling_names_this_record', 'ruling_was_executed',
                     'session_terminated', 'record_is_not_a_pin', 'sideways_check_ran'):
            assert gate in text
        assert re.search(r'^```\nLeo 2026-09-21: close esc-4201-1, option C\n```$', text, flags=re.MULTILINE)

    def test_every_fifth_close_is_the_audit_sample(self):
        text = mod.render_closes(self._closes(11), glossary=GLOSSARY)

        sampled = [int(n) for n in re.findall(r'^### Close (\d+):.*AUDIT SAMPLE', text, flags=re.MULTILINE)]
        assert sampled == [5, 10]
        assert mod.AUDIT_SAMPLE_EVERY == 5

    def test_recommend_only_closes_say_so(self):
        assert 'recommend-only' in mod.render_closes(self._closes(1), glossary=GLOSSARY)

    def test_a_close_needs_all_six_gates_held(self):
        item = _item()
        missed = evaluate_carveout(CarveoutFacts(escalation_id=item.escalation_id or ''), recommend_only=True)

        with pytest.raises(ValueError):
            mod.CloseRecord(item, missed)


def _heavy(n: int) -> mod.BriefEntry:
    options = tuple(Option(label, f'option {label}', 'first paragraph\n\nsecond paragraph') for label in 'ABC')
    return _entry(n, options=options, recommendation=NoLean('all three are costly'))


class TestDocket:
    @pytest.mark.parametrize(('entries', 'multi_sitting', 'expected'), [
        ([_entry(n) for n in range(1, 6)], False, False),
        ([_entry(n) for n in range(1, 7)], False, True),
        ([_heavy(1), _heavy(2), _entry(3)], False, False),
        ([_heavy(1), _heavy(2), _heavy(3)], False, True),
        ([_entry(1)], True, True),
        ([], False, False),
    ])
    def test_the_docket_threshold(self, entries, multi_sitting, expected):
        assert mod.needs_docket(entries, multi_sitting=multi_sitting) is expected

    def test_the_reason_names_the_rule_that_fired(self):
        assert '6' in (mod.docket_reason([_entry(n) for n in range(1, 7)], multi_sitting=False) or '')
        assert mod.docket_reason([_entry(1)], multi_sitting=False) is None

    def test_three_options_with_one_paragraph_ramifications_are_not_heavy(self):
        options = tuple(Option(label, f'option {label}', 'one paragraph') for label in 'ABC')

        entries = [_entry(n, options=options, recommendation=NoLean('x')) for n in (1, 2, 3)]

        assert mod.needs_docket(entries, multi_sitting=False) is False

    def test_one_row_per_numbered_entry_with_empty_decision_cells(self):
        rows = mod.docket_rows([_entry(1), _entry(3, prepared=False)])

        assert [row['item'] for row in rows] == [1, 3]
        assert all(set(row) == {'item', 'esc_id', 'issue', 'options', 'recommendation', 'decision', 'note'}
                   for row in rows)
        assert rows[0]['esc_id'] == 'esc-4001-1'
        assert rows[0]['options'][0] == {
            'label': 'A', 'text': 'close as ruled',
            'ramification': 'task 3881 keeps its retargeted scope; esc-3881-2 stays closed',
        }
        assert rows[0]['recommendation'] == 'A'
        assert (rows[0]['decision'], rows[0]['note']) == ('', '')
        assert rows[1]['options'][1] == {'label': 'B', 'text': 'hold it', 'ramification': ''}
        assert json.loads(json.dumps(rows)) == rows

    def test_options_by_label_matches_what_the_brief_shows(self):
        assert mod.options_by_label(_entry(1)) == {'A': 'close as ruled', 'B': 'hold for a re-ruling'}
        assert mod.options_by_label(_entry(1, prepared=False)) == {'A': 'close it', 'B': 'hold it'}


class TestMarkdownSafety:
    @pytest.mark.parametrize('hostile', [
        'see\n```\nunclosed fence',
        'a | b | c',
        '# not a heading\n## nor this',
        'underlined\n===',
        'breaks\n---',
    ])
    def test_agent_text_never_changes_the_heading_count(self, hostile):
        baseline = _render([_entry(1), _entry(2)])
        options = (Option('A', hostile, hostile), Option('B', 'hold', 'stays blocked'))

        text = _render([_entry(1, options=options, on_apply=hostile), _entry(2)])

        assert _headings(text) == _headings(baseline)

    def test_a_payload_containing_a_fence_stays_one_block(self):
        item = _item()
        payload = resolve_issue_payload(item.escalation_id or '', 'quoted:\n```\nx\n```', 'close_only',
                                        resolved_by='leo', resolution_turns=1)

        text = _render([mod.BriefEntry(3, item, _prep(item), _finding(item), (payload,))])

        assert _headings(text) == _headings(_render([_entry(3, item)]))
