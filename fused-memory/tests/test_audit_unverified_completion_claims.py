"""Tests for audit_unverified_completion_claims.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_audit_found_on_main_provenance.py
/ test_audit_duplicate_memories.py.
"""
from __future__ import annotations

import importlib.util
import json
import re
import types
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'audit_unverified_completion_claims.py'
)


def _load_module() -> types.ModuleType:
    """Load audit_unverified_completion_claims.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'audit_unverified_completion_claims'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
parse_category = _mod.parse_category
IN_SCOPE_CATEGORIES = _mod.IN_SCOPE_CATEGORIES
CorpusRecord = _mod.CorpusRecord
ScannedRecord = _mod.ScannedRecord
scan_records = _mod.scan_records
adjudicate = _mod.adjudicate
Finding = _mod.Finding
UNRESOLVABLE = _mod.UNRESOLVABLE
build_report = _mod.build_report
CAVEATS = _mod.CAVEATS
_build_parser = _mod._build_parser
RO_COMMAND = _mod.RO_COMMAND

KNOWN_PROJECTS = frozenset({'dark_factory', 'reify'})

#: esc-3085-1 instance (2), verbatim: a filing/dispatch claim naming a ticket
#: id that does not exist in the registry.
ESC_3085_1_INSTANCE_2 = (
    'reify task 5638 was reported unactionable and re-filed into '
    "dark_factory's task tree as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376"
)

#: esc-3085-1 instance (1): an applied-work claim about a still-open task.
ESC_3085_1_INSTANCE_1 = "task 5422's de-flake fix has been applied"


def _record(
    uuid: str = 'ep-1',
    text: str = '',
    category: str | None = 'temporal_facts',
    project_id: str = 'reify',
    created_at: str = '2026-07-26T00:00:00Z',
    source_description: str = 'add_memory:temporal_facts',
    name: str = 'episode',
) -> object:
    """Build a CorpusRecord without touching a store."""
    return CorpusRecord(
        uuid=uuid,
        kind='episode',
        text=text,
        source_description=source_description,
        category=category,
        project_id=project_id,
        created_at=created_at,
        name=name,
    )


class TestParseCategory:
    """The category label survives ONLY as the Episodic source_description.

    Graphiti persists no ``category`` property anywhere — not on Entity nodes,
    not on RELATES_TO edges. ``MemoryService`` stamps ``add_memory:<category>``
    (memory_service.py:2804) and ``replay_from_mem0:<category>`` (:3302), and
    ``graphiti_client`` may prepend ``[temporal:<ctx>] `` (:700) and — new from
    task 3142 — ``[unverified_claim] `` (:702). Recovering the label from that
    string is the only way to honour this task's category scope.
    """

    def test_add_memory_temporal_facts(self) -> None:
        assert parse_category('add_memory:temporal_facts') == 'temporal_facts'

    def test_add_memory_decisions_and_rationale(self) -> None:
        assert (
            parse_category('add_memory:decisions_and_rationale')
            == 'decisions_and_rationale'
        )

    def test_replay_from_mem0_prefix(self) -> None:
        assert parse_category('replay_from_mem0:temporal_facts') == 'temporal_facts'

    def test_temporal_context_prefix_is_stripped(self) -> None:
        assert (
            parse_category('[temporal:sprint-4] add_memory:temporal_facts')
            == 'temporal_facts'
        )

    def test_unverified_claim_prefix_is_stripped(self) -> None:
        """THE case that matters most.

        Task 3142 stamps ``[unverified_claim] `` at graphiti_client.py:702.
        Without stripping it, every episode the new write-time gate ALREADY
        flagged falls out of this sweep's population — precisely the records
        most worth reading.
        """
        assert (
            parse_category('[unverified_claim] add_memory:temporal_facts')
            == 'temporal_facts'
        )

    def test_both_prefixes_any_order(self) -> None:
        assert (
            parse_category(
                '[unverified_claim] [temporal:x] add_memory:decisions_and_rationale'
            )
            == 'decisions_and_rationale'
        )
        assert (
            parse_category(
                '[temporal:x] [unverified_claim] add_memory:decisions_and_rationale'
            )
            == 'decisions_and_rationale'
        )

    def test_caller_supplied_add_episode_description_is_uncategorized(self) -> None:
        assert parse_category('reconciliation stage 2 cycle summary') is None

    def test_unknown_category_does_not_mint_a_phantom_bucket(self) -> None:
        """Validated against models.enums.MemoryCategory, so a typo yields None."""
        assert parse_category('add_memory:not_a_real_category') is None

    def test_empty_and_none(self) -> None:
        assert parse_category('') is None
        assert parse_category(None) is None
        assert parse_category('   ') is None

    def test_in_scope_categories(self) -> None:
        assert IN_SCOPE_CATEGORIES == frozenset(
            {'temporal_facts', 'decisions_and_rationale'}
        )


class TestScanRecords:
    """The imported extractor, run over the corpus projection.

    The script contributes NO claim-detection regex of its own — the whole
    point of reusing completion_claim_gate is that a second, drifting copy of
    the negation/aspirational strippers is what stops "has not yet landed"
    from being read as a completion (the gate's own docstring, :20-26).
    """

    def _scan(self, records: list[object]) -> list[object]:
        return scan_records(
            records,
            default_project_id='reify',
            known_project_ids=KNOWN_PROJECTS,
            categories=IN_SCOPE_CATEGORIES,
        )

    def test_esc_3085_1_instance_2_yields_the_ticket_claim(self) -> None:
        """Ticket > commit > task precedence (completion_claim_gate.py:405-411).

        The clause names BOTH task 5638 and the ticket; it is ONE claim about
        the ticket — the most specific ref is what the writer asserted into
        existence.
        """
        scanned = self._scan([_record(text=ESC_3085_1_INSTANCE_2)])
        assert len(scanned) == 1
        claims = scanned[0].claims
        assert len(claims) == 1
        assert claims[0].kind == 'filing_dispatch'
        assert claims[0].subject == 'ticket'
        assert claims[0].ref == 'tkt_0RRRC5AASJ9Z630VP4PCN9H376'

    def test_esc_3085_1_instance_1_yields_the_task_claim(self) -> None:
        scanned = self._scan([_record(text=ESC_3085_1_INSTANCE_1)])
        assert len(scanned) == 1
        claims = scanned[0].claims
        assert len(claims) == 1
        assert claims[0].kind == 'applied_work'
        assert claims[0].subject == 'task'
        assert claims[0].ref == '5422'
        assert claims[0].project_id == 'reify'

    def test_negated_claim_yields_nothing(self) -> None:
        """Proves the imported strippers are LIVE, not re-derived."""
        scanned = self._scan(
            [_record(text='the fix has not been applied for task 5422')]
        )
        assert scanned == []

    def test_out_of_scope_category_is_excluded(self) -> None:
        scanned = self._scan([
            _record(uuid='ep-a', text=ESC_3085_1_INSTANCE_1,
                    category='entities_and_relations'),
            _record(uuid='ep-b', text=ESC_3085_1_INSTANCE_1, category=None),
        ])
        assert scanned == []

    def test_records_without_claims_are_dropped(self) -> None:
        scanned = self._scan([
            _record(uuid='ep-a', text='an ordinary observation about the tree'),
            _record(uuid='ep-b', text=''),
            _record(uuid='ep-c', text=ESC_3085_1_INSTANCE_1),
        ])
        assert [s.record.uuid for s in scanned] == ['ep-c']

    def test_output_order_is_deterministic(self) -> None:
        """Sorted by (category, created_at, uuid) — never dict iteration order."""
        records = [
            _record(uuid='ep-z', text=ESC_3085_1_INSTANCE_1,
                    category='temporal_facts', created_at='2026-07-27T00:00:00Z'),
            _record(uuid='ep-a', text=ESC_3085_1_INSTANCE_1,
                    category='temporal_facts', created_at='2026-07-26T00:00:00Z'),
            _record(uuid='ep-m', text=ESC_3085_1_INSTANCE_1,
                    category='decisions_and_rationale',
                    created_at='2026-07-28T00:00:00Z'),
            _record(uuid='ep-b', text=ESC_3085_1_INSTANCE_1,
                    category='temporal_facts', created_at='2026-07-26T00:00:00Z'),
        ]
        expected = ['ep-m', 'ep-a', 'ep-b', 'ep-z']
        assert [s.record.uuid for s in self._scan(records)] == expected
        assert [s.record.uuid for s in self._scan(list(reversed(records)))] == expected

    def test_record_project_id_overrides_the_default(self) -> None:
        scanned = self._scan(
            [_record(text=ESC_3085_1_INSTANCE_1, project_id='dark_factory')]
        )
        assert scanned[0].claims[0].project_id == 'dark_factory'


class TestAdjudicate:
    """Verdicts, via the IMPORTED verify_claims with hand-written probes.

    Every probe is injected, so the whole adjudication layer is exercised with
    no Taskmaster, no ticket DB and no git.
    """

    def _adjudicate(
        self,
        text: str,
        *,
        task_status: object = None,
        ticket: object = None,
        commit: object = None,
        category: str = 'temporal_facts',
        uuid: str = 'ep-1',
        created_at: str = '2026-07-26T00:00:00Z',
    ) -> list[object]:
        scanned = scan_records(
            [_record(uuid=uuid, text=text, category=category, created_at=created_at)],
            default_project_id='reify',
            known_project_ids=KNOWN_PROJECTS,
            categories=IN_SCOPE_CATEGORIES,
        )
        return adjudicate(
            scanned,
            task_status_probe=lambda ref, project_id: task_status,
            ticket_probe=lambda ref: ticket,
            commit_probe=lambda ref, project_id: commit,
        )

    def test_absent_ticket_is_a_mismatch(self) -> None:
        """esc-3085-1 instance (2), reproduced end-to-end.

        ``None`` from the ticket probe means the registry ANSWERED and said no
        such ticket — that is evidence the writer was wrong.
        """
        findings = self._adjudicate(ESC_3085_1_INSTANCE_2, ticket=None)
        assert len(findings) == 1
        assert findings[0].status == 'mismatch'
        assert findings[0].subject == 'ticket'
        assert findings[0].ref == 'tkt_0RRRC5AASJ9Z630VP4PCN9H376'

    def test_unresolvable_ticket_registry_is_unverifiable_not_mismatch(self) -> None:
        """THE CRITICAL ASSERTION — the deliberate divergence from the live gate.

        ``TaskInterceptor.get_ticket_row`` returns None BOTH for "no such
        ticket" and for "no ticket store configured" (task_interceptor.py:
        3006-3012), and ``_verify_ticket`` maps a None row to 'mismatch'
        (completion_claim_gate.py:575). On the write path that conflation is
        contained upstream by the _taskmaster_configured guard and costs one
        spurious tag. In a BATCH sweep it would print a fabrication accusation
        against every ticket claim in the corpus whenever tickets.db is merely
        absent. The gate's own module makes this distinction load-bearing at the
        sentinel level (:123-127, INV-2); honouring it here follows that intent.
        """
        findings = self._adjudicate(ESC_3085_1_INSTANCE_2, ticket=UNRESOLVABLE)
        assert len(findings) == 1
        assert findings[0].status == 'unverifiable'
        assert findings[0].status != 'mismatch'

    def test_existing_ticket_verifies_and_is_dropped(self) -> None:
        findings = self._adjudicate(
            ESC_3085_1_INSTANCE_2,
            ticket={'project_id': 'dark_factory', 'status': 'open'},
        )
        assert findings == []

    def test_terminal_task_status_verifies_and_is_dropped(self) -> None:
        """Verified claims never appear — the report is a report of problems."""
        assert self._adjudicate(ESC_3085_1_INSTANCE_1, task_status='done') == []

    def test_open_task_status_is_a_mismatch(self) -> None:
        findings = self._adjudicate(ESC_3085_1_INSTANCE_1, task_status='in-progress')
        assert len(findings) == 1
        assert findings[0].status == 'mismatch'
        assert findings[0].observed == 'in-progress'

    def test_unresolvable_task_status_is_unverifiable(self) -> None:
        findings = self._adjudicate(ESC_3085_1_INSTANCE_1, task_status=None)
        assert len(findings) == 1
        assert findings[0].status == 'unverifiable'

    def test_unknown_task_status_is_unverifiable_not_mismatch(self) -> None:
        """'unknown' is get_statuses' NULL sentinel, not a live status.

        It cannot contradict the claim, but it cannot confirm it either.
        """
        findings = self._adjudicate(ESC_3085_1_INSTANCE_1, task_status='unknown')
        assert len(findings) == 1
        assert findings[0].status == 'unverifiable'

    def test_absent_commit_is_a_mismatch_unresolvable_is_not(self) -> None:
        text = 'task 3142 has landed in commit 23f1c27ddf'
        absent = self._adjudicate(text, commit=False)
        assert len(absent) == 1
        assert absent[0].status == 'mismatch'
        assert absent[0].subject == 'commit'

        unresolvable = self._adjudicate(text, commit=None)
        assert len(unresolvable) == 1
        assert unresolvable[0].status == 'unverifiable'

        present = self._adjudicate(text, commit=True)
        assert present == []

    def test_finding_is_self_contained(self) -> None:
        """A reader can re-check the verdict without re-running the sweep."""
        findings = self._adjudicate(
            ESC_3085_1_INSTANCE_2,
            ticket=None,
            category='decisions_and_rationale',
            uuid='02090224-7bc9-4485-9291-6748e1042ac9',
            created_at='2026-07-27T05:00:00Z',
        )
        finding = findings[0]
        assert finding.record_uuid == '02090224-7bc9-4485-9291-6748e1042ac9'
        assert finding.record_kind == 'episode'
        assert finding.category == 'decisions_and_rationale'
        assert finding.created_at == '2026-07-27T05:00:00Z'
        assert finding.project_id == 'reify'
        assert finding.claim_kind == 'filing_dispatch'
        assert 'tkt_0RRRC5AASJ9Z630VP4PCN9H376' in finding.claimed_text
        assert 'no ticket' in finding.observed
        assert finding.derived_edge_uuids == ()
        assert finding.to_json()['status'] == 'mismatch'

    def test_mismatches_sort_before_unverifiables(self) -> None:
        """A mismatch is evidence a writer was wrong; an unverifiable is merely
        unchecked. The ordering is what makes the head of the report the part
        worth reading."""
        scanned = scan_records(
            [
                _record(uuid='ep-unver', text=ESC_3085_1_INSTANCE_2,
                        category='decisions_and_rationale'),
                _record(uuid='ep-mism', text=ESC_3085_1_INSTANCE_1,
                        category='temporal_facts'),
            ],
            default_project_id='reify',
            known_project_ids=KNOWN_PROJECTS,
            categories=IN_SCOPE_CATEGORIES,
        )
        findings = adjudicate(
            scanned,
            task_status_probe=lambda ref, project_id: 'in-progress',
            ticket_probe=lambda ref: UNRESOLVABLE,
            commit_probe=lambda ref, project_id: None,
        )
        assert [f.status for f in findings] == ['mismatch', 'unverifiable']


def _finding(
    status: str = 'mismatch',
    uuid: str = 'ep-1',
    category: str = 'temporal_facts',
    project_id: str = 'reify',
    subject: str = 'ticket',
    ref: str = 'tkt_X',
    created_at: str = '2026-07-26T00:00:00Z',
) -> object:
    return Finding(
        record_uuid=uuid,
        record_kind='episode',
        category=category,
        project_id=project_id,
        created_at=created_at,
        claim_kind='filing_dispatch',
        subject=subject,
        ref=ref,
        claim_project_id=None,
        status=status,
        observed=f'observed for {ref}',
        claimed_text='some claiming clause',
    )


class TestBuildReport:
    """Volume control WITHOUT a silent cap, and the retrospective bias as data."""

    def _build(self, findings: list[object], include_unverifiable: bool = False):
        return build_report(
            findings,
            swept_at='2026-08-11T12:00:00Z',
            scanned_count=100,
            records_with_claims=7,
            projects=['dark_factory', 'reify'],
            categories=sorted(IN_SCOPE_CATEGORIES),
            include_unverifiable=include_unverifiable,
        )

    def test_summary_counts_both_buckets_even_when_not_listing(self) -> None:
        """Counting-but-not-listing must never read as a clean corpus."""
        report = self._build([
            _finding('mismatch', uuid='ep-a'),
            _finding('unverifiable', uuid='ep-b'),
            _finding('unverifiable', uuid='ep-c'),
        ])
        assert report['summary']['mismatch'] == 1
        assert report['summary']['unverifiable'] == 2
        assert report['summary']['by_category']
        assert report['summary']['by_project']
        assert report['summary']['by_subject']

    def test_mismatch_and_unverifiable_are_never_summed(self) -> None:
        """Different facts — the gate makes the same distinction load-bearing
        at the sentinel level (completion_claim_gate.py:123-127)."""
        report = self._build([_finding('mismatch'), _finding('unverifiable')])
        assert 'total' not in report['summary']
        assert 'findings_total' not in report['summary']

    def test_default_lists_only_mismatches_and_says_what_it_withheld(self) -> None:
        report = self._build([
            _finding('mismatch', uuid='ep-a'),
            _finding('unverifiable', uuid='ep-b'),
        ])
        assert [f['status'] for f in report['findings']] == ['mismatch']
        assert report['summary']['unverifiable'] == 1
        truncated = report['truncated_by']
        assert truncated is not None
        assert truncated['withheld'] == 1
        assert truncated['status'] == 'unverifiable'
        assert '--include-unverifiable' in truncated['flag']

    def test_include_unverifiable_lists_everything(self) -> None:
        report = self._build(
            [_finding('mismatch', uuid='ep-a'), _finding('unverifiable', uuid='ep-b')],
            include_unverifiable=True,
        )
        assert [f['status'] for f in report['findings']] == [
            'mismatch', 'unverifiable',
        ]
        assert report['truncated_by'] is None

    def test_denominators_are_present(self) -> None:
        """A rate can be computed rather than guessed."""
        report = self._build([_finding('mismatch')])
        assert report['scanned'] == 100
        assert report['records_with_claims'] == 7
        assert report['projects'] == ['dark_factory', 'reify']
        assert report['categories'] == sorted(IN_SCOPE_CATEGORIES)
        assert report['swept_at'] == '2026-08-11T12:00:00Z'

    def test_caveats_pin_the_retrospective_bias_as_data(self) -> None:
        """A write-time gate reads the authority at write time; this sweep reads
        it TODAY. The gap biases the result in BOTH directions, and a reader who
        takes the headline as a clean measurement draws a wrong conclusion
        either way. Pinning the literal text against a module constant means a
        later editor cannot quietly drop one while the report still looks
        authoritative."""
        report = self._build([_finding('mismatch')])
        caveats = report['caveats']
        assert caveats
        assert list(caveats) == list(CAVEATS)
        blob = ' '.join(caveats).lower()
        assert 'under-count' in blob or 'under-counts' in blob
        assert 'cancelled' in blob
        assert 'mem0' in blob

    def test_report_is_json_serializable_and_byte_stable(self) -> None:
        findings = [
            _finding('unverifiable', uuid='ep-b', category='decisions_and_rationale'),
            _finding('mismatch', uuid='ep-a'),
        ]
        first = json.dumps(self._build(findings, True), indent=2, default=str)
        second = json.dumps(self._build(findings, True), indent=2, default=str)
        assert first == second
        assert json.loads(first)['summary']['mismatch'] == 1

    def test_empty_corpus_reports_zero_without_truncation(self) -> None:
        report = self._build([])
        assert report['findings'] == []
        assert report['summary']['mismatch'] == 0
        assert report['summary']['unverifiable'] == 0
        assert report['truncated_by'] is None


class TestReadOnlyByConstruction:
    """The scope note, turned into a test.

    "read-only report first; do NOT auto-delete or auto-invalidate edges on a
    regex verdict" — asserted mechanically so it survives a later editor who
    never reads the task description. audit_duplicate_memories.py and
    invalidate_fabricated_shipping_edges.py both HAVE an --apply; this script
    deliberately has none, and the ABSENCE is what must be asserted.
    """

    #: Any option whose dest or option string contains one of these is a
    #: mutation affordance this script must never grow.
    FORBIDDEN = ('apply', 'invalidate', 'delete', 'repair', 'fix', 'write', 'mutate')

    def test_parser_exposes_no_mutation_option(self) -> None:
        parser = _build_parser()
        offenders = []
        for action in parser._actions:
            names = [str(action.dest or '')] + [str(s) for s in action.option_strings]
            for name in names:
                lowered = name.lower()
                if any(word in lowered for word in self.FORBIDDEN):
                    offenders.append(name)
        assert offenders == [], f'mutation affordance(s) present: {offenders}'

    def test_ro_command_is_the_only_falkordb_command(self) -> None:
        assert RO_COMMAND == 'GRAPH.RO_QUERY'
        source = SCRIPT_PATH.read_text()
        assert 'GRAPH.QUERY' not in source
        assert source.count("'GRAPH.RO_QUERY'") == 1

    def test_source_contains_no_mutation_call(self) -> None:
        source = SCRIPT_PATH.read_text()
        for symbol in (
            'update_edge',
            'delete_episode',
            'bulk_remove_edges',
            'remove_edge',
            'delete_entity',
            'add_memory(',
            'add_episode(',
        ):
            assert symbol not in source, f'mutation call present: {symbol}'

    def test_no_cypher_write_keyword_in_any_query_constant(self) -> None:
        cypher_constants = [
            value
            for name, value in vars(_mod).items()
            if name.endswith('_CYPHER') and isinstance(value, str)
        ]
        assert cypher_constants, 'expected at least one *_CYPHER constant'
        # Word-boundary matched, not substring: the projected field
        # ``e.created_at`` legitimately contains the letters of CREATE, and a
        # substring test would fail on a read-only query.
        write_keyword_re = re.compile(
            r'\b(?:CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP)\b', re.IGNORECASE
        )
        for query in cypher_constants:
            hit = write_keyword_re.search(query)
            assert hit is None, f'write keyword {hit and hit.group()!r} in {query!r}'

    def test_project_is_repeatable(self) -> None:
        """esc-3085-1's two instances were written by reify agents about work in
        two different trees, so a single-project sweep would have missed half
        the incident."""
        parser = _build_parser()
        args = parser.parse_args(['--project', 'dark_factory', '--project', 'reify'])
        assert args.project == ['dark_factory', 'reify']

    def test_volume_and_gate_flags_exist(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['--include-unverifiable', '--fail-on-mismatch'])
        assert args.include_unverifiable is True
        assert args.fail_on_mismatch is True
        defaults = parser.parse_args([])
        assert defaults.include_unverifiable is False
        assert defaults.fail_on_mismatch is False
