"""Tests for scripts/audit_delivered_checks.py — the READ-ONLY, STATUS-AWARE
retroactive sweep over checked-in ``delivered_checks`` (task 3500, SCOPE item 5).

Mirrors test_audit_combine_gate_marker_loss.py: pure functions get direct
pytest coverage; ``main()`` gets subprocess coverage.

NO TEST HERE ASSERTS A COUNT OR TASK ID DERIVED FROM THE LIVE DATABASES.
tasks.db is mutated continuously by the running orchestrator, so a test
pinning "the live DB yields N findings" would be a guessed threshold that goes
red the moment another task lands. Every assertion runs against synthetic temp
databases and synthetic temp git repos whose contents the test controls
exactly — the same discipline that file's docstring requires.

WHY STATUS IS THE AXIS, and why this sweep cannot be the shared-suite ratchet
(shared/tests/test_capability_manifest.py::TestCheckedInGrepDescriptorHygiene).
Evaluating a descriptor against MAIN TODAY yields a bit, not a verdict: the
same "``expect: present`` and it matches" observation is the SUCCESS state of a
landed producer and a never-fires vacuous gate on a live one. Measured over the
checked-in corpus, a status-blind rule flags 313/548 (57%) of descriptors —
overwhelmingly correctly delivered work. The producer's STATUS is what turns
the bit into a disposition, and status lives in tasks.db, which is why this is
a script and not a test.
"""
from __future__ import annotations

import pytest
from audit_delivered_checks import (
    DISPOSITION_BROKEN,
    DISPOSITION_DELIVERED,
    DISPOSITION_HEALTHY,
    DISPOSITION_INERT,
    DISPOSITION_NO_TASK,
    DISPOSITION_SUPERSEDED,
    DISPOSITION_UNEVALUABLE,
    DISPOSITION_VACUOUS_LIVE_GATE,
    AuditCoverage,
    DescriptorRow,
    Finding,
    ProjectAudit,
    audit_project,
    classify_descriptor,
    evaluate_row,
    format_report,
    load_metadata_checks,
)
from shared.delivered_check_polarity import CheckOutcome

# ---------------------------------------------------------------------------
# classify_descriptor — the pure core, and the whole point of the script.
#
# The three inputs are the descriptor's OUTCOME against main today, the
# producer's STATUS, and (only for a would-be defect) the id of a later task
# that reintroduced the pattern. Nothing else: no history, no heuristics.
# ---------------------------------------------------------------------------


def _row(**overrides):
    """A minimal DescriptorRow; every test overrides only what it asserts on."""
    fields = {
        'task_id': 42,
        'tag': 'master',
        'status': 'done',
        'name': 'cap-one',
        'kind': 'grep',
        'pattern': 'SomeSymbol',
        'expect': 'present',
        'paths': ('src/',),
        'source': 'metadata',
        'manifest': None,
        'task_label': None,
    }
    fields.update(overrides)
    return DescriptorRow(**fields)


class TestClassifyDescriptor:
    def test_done_and_expect_present_with_no_match_is_broken(self):
        """(a) The mode-2/mode-3 defect class: the producer closed, and the
        capability its check asserts is nowhere on main. Whatever the check was
        supposed to gate was never gated."""
        assert classify_descriptor(CheckOutcome.FAIL, status='done') == DISPOSITION_BROKEN

    def test_done_and_expect_absent_still_matching_is_broken(self):
        """(b) The same defect from the other polarity. FAIL already encodes
        the polarity — interpret_grep_rc inverts on `expect` — so the
        classifier never re-derives it and the two cells collapse to one rule."""
        assert classify_descriptor(CheckOutcome.FAIL, status='done') == DISPOSITION_BROKEN

    @pytest.mark.parametrize('status', ['pending', 'blocked', 'in-progress', 'deferred'])
    def test_non_terminal_and_already_passing_is_a_vacuous_live_gate(self, status):
        """(c) THE DISPOSITION THAT JUSTIFIES THE WHOLE SWEEP. A live task
        whose check ALREADY passes on main is gating nothing: landing the
        producer cannot change the verdict, so any dependent is either released
        for the wrong reason or held for one that will never clear."""
        assert (
            classify_descriptor(CheckOutcome.PASS, status=status)
            == DISPOSITION_VACUOUS_LIVE_GATE
        )

    @pytest.mark.parametrize('status', ['pending', 'blocked', 'in-progress', 'deferred'])
    def test_non_terminal_and_failing_is_healthy(self, status):
        """(d) The normal majority: a forward-looking check on unbuilt work.
        This cell MUST stay silent or the report is unreadable."""
        assert classify_descriptor(CheckOutcome.FAIL, status=status) == DISPOSITION_HEALTHY

    def test_done_and_passing_is_delivered(self):
        """(e) The success state, and the one a status-blind sweep misreads as
        vacuity for 57% of the corpus."""
        assert classify_descriptor(CheckOutcome.PASS, status='done') == DISPOSITION_DELIVERED

    def test_supersession_outranks_broken(self):
        """(f) SUPERSESSION IS NOT A DEFECT. The measured case: task 3618's
        `expect: absent` gzip checks fail on main today only because task 3578
        deliberately RESTORED gzip reading afterwards. The descriptor was
        correct, delivered, and then legitimately undone by later work — so it
        must never be counted as an authoring defect."""
        assert (
            classify_descriptor(CheckOutcome.FAIL, status='done', superseded_by='3578')
            == DISPOSITION_SUPERSEDED
        )

    def test_supersession_does_not_rewrite_a_healthy_verdict(self):
        # A supersession signal only ever explains away a would-be DEFECT.
        # Letting it relabel a delivered or healthy row would launder a real
        # disposition into a footnote.
        assert (
            classify_descriptor(CheckOutcome.PASS, status='done', superseded_by='3578')
            == DISPOSITION_DELIVERED
        )
        assert (
            classify_descriptor(CheckOutcome.FAIL, status='pending', superseded_by='3578')
            == DISPOSITION_HEALTHY
        )

    def test_cancelled_producer_is_inert_never_broken(self):
        # A cancelled task promised nothing and gates nothing. Reporting its
        # failing check as `broken` would be a false positive on abandoned
        # work; reporting it as a live gate would be false too.
        assert classify_descriptor(CheckOutcome.FAIL, status='cancelled') == DISPOSITION_INERT
        assert classify_descriptor(CheckOutcome.PASS, status='cancelled') == DISPOSITION_INERT

    def test_unevaluable_is_its_own_disposition(self):
        # ERRORED is not FAIL. git being unable to answer must never be
        # rendered as "the capability was never delivered" — that is the
        # no-silent-fail-soft invariant, and it is the exact confusion the
        # runtime gate's own rc>=2 -> ERRORED boundary exists to prevent.
        for status in ('done', 'pending', 'cancelled', None):
            assert (
                classify_descriptor(CheckOutcome.ERRORED, status=status)
                == DISPOSITION_UNEVALUABLE
            )

    def test_descriptor_reaching_no_task_is_reported_not_dropped(self):
        """(g) COVERAGE, never silence. A sidecar capability whose task_id
        resolves to no row in tasks.db cannot be classified — but dropping it
        would present a partial sweep as a complete one."""
        assert classify_descriptor(CheckOutcome.PASS, status=None) == DISPOSITION_NO_TASK
        assert classify_descriptor(CheckOutcome.FAIL, status=None) == DISPOSITION_NO_TASK


# ---------------------------------------------------------------------------
# load_metadata_checks — tasks.db -> DescriptorRow, read-only.
# ---------------------------------------------------------------------------


class TestLoadMetadataChecks:
    def test_reads_grep_descriptors_with_the_producer_status(self, make_tasks_db):
        db = make_tasks_db([
            {
                'id': 10,
                'status': 'done',
                'metadata': {
                    'delivered_checks': [
                        {
                            'name': 'cap-one',
                            'kind': 'grep',
                            'pattern': 'SomeSymbol',
                            'expect': 'present',
                            'paths': ['src/'],
                        }
                    ]
                },
            },
        ])

        rows = load_metadata_checks(str(db))

        assert len(rows) == 1
        row = rows[0]
        assert (row.task_id, row.status, row.name) == (10, 'done', 'cap-one')
        assert (row.pattern, row.expect, row.paths) == ('SomeSymbol', 'present', ('src/',))
        assert row.source == 'metadata'

    def test_script_and_manual_kinds_are_not_swept(self, make_tasks_db):
        # The sweep is a statement about grep POLARITY against a tree. A script
        # check has no pattern to evaluate and belongs to the script-target
        # guard in shared/tests/test_capability_manifest.py instead.
        db = make_tasks_db([
            {
                'id': 11,
                'status': 'done',
                'metadata': {
                    'delivered_checks': [
                        {'name': 'c', 'kind': 'script', 'script': 'scripts/x.py'},
                    ]
                },
            },
        ])

        assert load_metadata_checks(str(db)) == []

    def test_malformed_metadata_is_skipped_not_raised(self, make_tasks_db):
        # A single undecodable row must not abort a whole-project sweep.
        db = make_tasks_db([
            {'id': 12, 'status': 'done', 'metadata': '{not json'},
            {'id': 13, 'status': 'pending', 'metadata': None},
            {
                'id': 14,
                'status': 'done',
                'metadata': {
                    'delivered_checks': [
                        {'name': 'ok', 'kind': 'grep', 'pattern': 'X', 'expect': 'present'}
                    ]
                },
            },
        ])

        rows = load_metadata_checks(str(db))

        assert [r.task_id for r in rows] == [14]

    def test_connection_is_read_only(self, make_tasks_db):
        # READ-ONLY/REPORT-ONLY is a structural guarantee, not a convention:
        # the URI mode makes writing impossible rather than merely unwritten.
        import sqlite3

        db = make_tasks_db([{'id': 15, 'status': 'done'}])
        conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("UPDATE tasks SET status = 'cancelled'")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# evaluate_row — the SAME primitive the runtime gate runs.
# ---------------------------------------------------------------------------


def _init_repo(root, files):
    import subprocess

    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-b', 'main', str(root)], check=True, capture_output=True)
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    subprocess.run(['git', '-C', str(root), 'add', '-A'], check=True, capture_output=True)
    subprocess.run(
        ['git', '-C', str(root), '-c', 'user.email=t@e', '-c', 'user.name=t',
         'commit', '-m', 'seed'],
        check=True, capture_output=True,
    )
    return root


class TestEvaluateRow:
    def test_present_pattern_that_matches_the_tree_passes(self, tmp_path):
        root = _init_repo(tmp_path / 'repo', {'src/a.py': 'class SomeSymbol:\n    pass\n'})
        row = _row(pattern='SomeSymbol', expect='present', paths=('src/',))
        assert evaluate_row(row, repo_root=str(root)) is CheckOutcome.PASS

    def test_present_pattern_with_no_match_fails(self, tmp_path):
        root = _init_repo(tmp_path / 'repo', {'src/a.py': 'pass\n'})
        row = _row(pattern='SomeSymbol', expect='present', paths=('src/',))
        assert evaluate_row(row, repo_root=str(root)) is CheckOutcome.FAIL

    def test_non_repo_root_is_errored_never_a_verdict(self, tmp_path):
        plain = tmp_path / 'not-a-repo'
        plain.mkdir()
        assert evaluate_row(_row(), repo_root=str(plain)) is CheckOutcome.ERRORED


# ---------------------------------------------------------------------------
# audit_project — the three-source join, and the report's sections.
# ---------------------------------------------------------------------------


class TestAuditProject:
    def test_joins_status_from_tasks_db_with_evaluation_against_the_tree(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        root = _init_repo(tmp_path / 'proj', {'src/a.py': 'class Landed:\n    pass\n'})
        # The fixture first: it creates <root>/.taskmaster/tasks/ (and an empty
        # file there, which make_tasks_db then opens and schemas). Reversed,
        # make_tasks_db has no directory to write into.
        project_root_with_tasks_db(root)
        make_tasks_db(
            [
                # done + matches -> delivered
                {'id': 20, 'status': 'done', 'metadata': {'delivered_checks': [
                    {'name': 'landed', 'kind': 'grep', 'pattern': 'Landed',
                     'expect': 'present', 'paths': ['src/']}]}},
                # done + no match -> broken
                {'id': 21, 'status': 'done', 'metadata': {'delivered_checks': [
                    {'name': 'never-built', 'kind': 'grep', 'pattern': 'NeverBuilt',
                     'expect': 'present', 'paths': ['src/']}]}},
                # pending + already matches -> vacuous_live_gate
                {'id': 22, 'status': 'pending', 'metadata': {'delivered_checks': [
                    {'name': 'already-green', 'kind': 'grep', 'pattern': 'Landed',
                     'expect': 'present', 'paths': ['src/']}]}},
                # pending + no match -> healthy
                {'id': 23, 'status': 'pending', 'metadata': {'delivered_checks': [
                    {'name': 'forward', 'kind': 'grep', 'pattern': 'NotYet',
                     'expect': 'present', 'paths': ['src/']}]}},
            ],
            directory=root / '.taskmaster' / 'tasks',
        )

        audit = audit_project(str(root))
        by_task = {f.row.task_id: f.disposition for f in audit.findings}

        assert by_task[20] == DISPOSITION_DELIVERED
        assert by_task[21] == DISPOSITION_BROKEN
        assert by_task[22] == DISPOSITION_VACUOUS_LIVE_GATE
        assert by_task[23] == DISPOSITION_HEALTHY

    def test_manifest_capability_with_no_task_row_lands_in_coverage(
        self, tmp_path, make_tasks_db, project_root_with_tasks_db
    ):
        """(g) again, end to end: a sidecar naming a task that is not in this
        project's tasks.db is REPORTED, never dropped."""
        root = _init_repo(
            tmp_path / 'proj',
            {
                'src/a.py': 'pass\n',
                'plans/x-prd.capability-manifest.yaml': (
                    'prd: plans/x-prd.md\n'
                    'schema_version: 1\n'
                    'tasks:\n'
                    '  - label: α\n'
                    '    task_id: 999\n'
                    '    capabilities:\n'
                    '      - name: orphan\n'
                    '        binding: b\n'
                    '        verdict: PASS\n'
                    '        delivered_check:\n'
                    '          kind: grep\n'
                    '          pattern: Orphaned\n'
                    '          expect: present\n'
                ),
            },
        )
        project_root_with_tasks_db(root)
        make_tasks_db([{'id': 20, 'status': 'done'}],
                      directory=root / '.taskmaster' / 'tasks')

        audit = audit_project(str(root))

        orphans = [f for f in audit.findings if f.disposition == DISPOSITION_NO_TASK]
        assert [f.row.task_id for f in orphans] == [999]
        assert audit.coverage.descriptors_without_task == 1

    def test_report_renders_supersession_in_its_own_section(self):
        # A superseded row must not sit in the DEFECTS section: it is a
        # correctly-authored descriptor that later work legitimately undid, and
        # filing it beside real defects is what makes a report get ignored.
        audit = ProjectAudit(
            project_root='/synthetic',
            findings=[
                Finding(row=_row(task_id=3618, name='gzip-gone'),
                        disposition=DISPOSITION_SUPERSEDED, superseded_by='3578'),
                Finding(row=_row(task_id=21, name='never-built'),
                        disposition=DISPOSITION_BROKEN),
            ],
            coverage=AuditCoverage(
                descriptors_total=2,
                descriptors_without_task=0,
                unevaluable=0,
                sidecars_unloadable=0,
            ),
        )

        report = format_report([audit])
        superseded_at = report.index('SUPERSEDED')
        broken_at = report.index('BROKEN')

        # Both rows are rendered, each under its OWN heading — the superseded
        # one carries the later task that explains it.
        assert 'gzip-gone' in report and 'never-built' in report
        assert '3578' in report
        assert superseded_at != broken_at
        assert 'COVERAGE' in report
