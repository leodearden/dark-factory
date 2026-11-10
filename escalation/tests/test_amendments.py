"""The amendment-framing seam: ``escalation.amendments`` (task 6319).

Every writer of ``Escalation.amendments`` goes through this module's one
append-and-trim policy.  The queue-level suites (test_queue.py's incoming-
framing tests, test_amend_escalation.py) prove the policy through the queue's
public methods; this file states what only the module's own seam can state, on
in-memory records with no queue and no filesystem: that the module is a leaf,
how ``elide`` marks a cut, and how ``append_amendment_capped`` records, refuses
and sheds.

Bounds are derived from the constants' NAMES, never their values, so a retune
of any cap leaves these tests meaningful.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from escalation.amendments import (
    MAX_AMENDMENT_DETAIL_CHARS,
    MAX_AMENDMENT_LINE_CHARS,
    MAX_AMENDMENT_OPTIONS,
    MAX_AMENDMENTS,
    amendment_room,
    append_amendment_capped,
    elide,
    has_framing,
)
from escalation.models import Escalation


def _esc(**overrides: Any) -> Escalation:
    """A real ``escalation.models.Escalation`` carrying its own non-empty framing."""
    fields: dict[str, Any] = {
        'id': 'esc-T1-1',
        'task_id': 'T1',
        'agent_role': 'escalation-watcher-auto',
        'severity': 'blocking',
        'category': 'design_concern',
        'level': 2,
        'root_cause': 'own root cause',
        'summary': 'own summary',
        'detail': 'own detail',
        'options': ['A: own option'],
    }
    fields.update(overrides)
    return Escalation(**fields)


class TestAmendmentsModuleIsALeaf:
    """The structural property that lets queue.py import this module downward."""

    def test_importing_amendments_pulls_in_only_models(self):
        """Fresh-interpreter probe: the sys.modules delta is this module plus models.

        Leaf-ness is a runtime property, so it is probed at runtime: a source
        scan is defeated by relative imports and ``importlib.import_module``,
        and falsely tripped by prose.  ``escalation.models`` is stdlib-only, so
        it is the one intra-package import this module may carry.
        """
        src_dir = Path(__file__).resolve().parent.parent / 'src'
        probe = (
            'import json\n'
            'import sys\n'
            f'sys.path.insert(0, {str(src_dir)!r})\n'
            'import escalation\n'
            "before = {m for m in sys.modules if m.startswith('escalation')}\n"
            'import escalation.amendments\n'
            "after = {m for m in sys.modules if m.startswith('escalation')}\n"
            'print(json.dumps(sorted(after - before)))\n'
        )
        proc = subprocess.run(
            [sys.executable, '-c', probe],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # A probe that dies during import reports nothing and would otherwise
        # pass vacuously, so survival is asserted first.
        assert proc.returncode == 0, (
            f'leaf-ness probe failed to run (rc={proc.returncode}).\n'
            f'stdout: {proc.stdout}\nstderr: {proc.stderr}'
        )
        delta = json.loads(proc.stdout)
        expected = ['escalation.amendments', 'escalation.models']
        assert delta == expected, (
            'escalation.amendments must import nothing from the escalation package '
            'but escalation.models — never queue or server — so that queue.py can '
            f'import it downward. It now also pulls in: {sorted(set(delta) - set(expected))}'
        )


class TestElide:
    def test_text_within_the_limit_is_returned_unchanged(self):
        assert elide('abcde', 5) == ('abcde', 0)

    def test_text_over_the_limit_keeps_the_head_and_counts_the_rest(self):
        text = 'x' * 15
        kept, dropped = elide(text, 10)
        assert dropped == len(text) - 10
        assert kept.startswith(text[:10])

    def test_marker_names_the_count_the_limit_and_the_cap(self):
        kept, dropped = elide('y' * 27, 20, what='late-resolution')
        marker = kept[20:]
        assert str(dropped) in marker
        assert '20-char' in marker
        assert 'late-resolution' in marker
        assert kept.endswith(' ...]')


class TestAppendAmendmentCapped:
    def test_all_empty_framing_records_nothing(self):
        esc = _esc()
        assert not has_framing(root_cause='', summary='', evidence='', options=None)

        result = append_amendment_capped(
            esc, root_cause='', summary='', evidence='', options=None,
            agent_role='r', caller='t',
        )

        assert result == (False, 0)
        assert esc.amendments == []

    def test_framing_identical_to_the_records_own_is_a_repeat(self):
        esc = _esc()

        result = append_amendment_capped(
            esc, root_cause=esc.root_cause, summary=esc.summary,
            evidence=esc.detail, options=list(esc.options),
            agent_role='r', caller='t',
        )

        assert result == (False, 0)
        assert esc.amendments == []

    def test_over_long_framing_records_one_entry_capped_per_field(self):
        esc = _esc()
        over = 50  # comfortably past every cap, whatever they are retuned to

        result = append_amendment_capped(
            esc,
            root_cause='R' * (MAX_AMENDMENT_LINE_CHARS + over),
            summary='S' * (MAX_AMENDMENT_LINE_CHARS + over),
            evidence='D' * (MAX_AMENDMENT_DETAIL_CHARS + over),
            options=['O' * (MAX_AMENDMENT_LINE_CHARS + over)] * (MAX_AMENDMENT_OPTIONS + 2),
            agent_role='escalation-watcher-auto',
            caller='t',
        )

        assert result == (True, 0)
        assert len(esc.amendments) == 1
        entry = esc.amendments[0]
        assert entry['root_cause'].count('R') == MAX_AMENDMENT_LINE_CHARS
        assert entry['summary'].count('S') == MAX_AMENDMENT_LINE_CHARS
        assert entry['detail'].count('D') == MAX_AMENDMENT_DETAIL_CHARS
        assert len(entry['options']) == MAX_AMENDMENT_OPTIONS
        shed_options = 2 * (MAX_AMENDMENT_LINE_CHARS + over)
        assert esc.amendments_chars_elided == over * (3 + MAX_AMENDMENT_OPTIONS) + shed_options
        assert entry['agent_role'] == 'escalation-watcher-auto'
        assert entry['timestamp']

    def test_past_the_cap_the_oldest_entry_is_shed_loudly(
        self, caplog: pytest.LogCaptureFixture,
    ):
        esc = _esc()
        for n in range(MAX_AMENDMENTS):
            assert amendment_room(esc) == MAX_AMENDMENTS - n
            recorded, dropped = append_amendment_capped(
                esc, root_cause=f'cause {n}', summary='', evidence='', options=None,
                agent_role='r', caller='t',
            )
            assert (recorded, dropped) == (True, 0)
        assert len(esc.amendments) == MAX_AMENDMENTS
        assert amendment_room(esc) == 0
        first_cause = esc.amendments[0]['root_cause']

        with caplog.at_level(logging.WARNING, logger='escalation.amendments'):
            result = append_amendment_capped(
                esc, root_cause='newest cause', summary='', evidence='', options=None,
                agent_role='r', caller='shed-probe',
            )

        assert result == (True, 1)
        assert len(esc.amendments) == MAX_AMENDMENTS
        assert first_cause not in [a['root_cause'] for a in esc.amendments]
        assert esc.amendments[-1]['root_cause'] == 'newest cause'
        assert esc.amendments_truncated == 1
        shed = [
            r for r in caplog.records
            if r.name == 'escalation.amendments' and r.levelno == logging.WARNING
        ]
        assert shed, 'shedding an amendment must WARN on the escalation.amendments logger'
        assert any(esc.id in r.getMessage() and 'shed-probe' in r.getMessage() for r in shed)
