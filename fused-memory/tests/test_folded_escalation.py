"""The ONE home for the pending-anchor-fold escalation filer skeleton (INV-5).

`middleware/_folded_escalation.file_folded_escalation` is the extracted body
that seven fused-memory filers had each been carrying verbatim: defensive
optional-`escalation`-package import, guarded queue construction, a
`get_by_task(anchor, status='pending')` fold, and a never-raise
`Escalation(...)` + `queue.submit(...)`.

WHAT MUST NOT COLLAPSE.  `anchor_task_id` is a REQUIRED keyword-only parameter
with NO default, and every caller keeps its own `_ANCHOR_TASK_ID` constant in
its own module.  A filer that dedupes against an anchor somebody else keeps
open never files again, and that silence is indistinguishable from health —
see `TestNoTwoFilersShareAnAnchor` for the measured incident.

NOT THE HOME FOR the `submit_or_dedupe` content-fingerprint family
(`scope_violation_escalator`, `mem0_update_storm_escalator`,
`entity_mint_storm_escalator`).  Those dedupe on a content fingerprint over a
cached per-project queue, not on a pending anchor; routing them through this
helper would be a behaviour change, not a refactor.
"""

from __future__ import annotations

import json
import logging

import pytest

from fused_memory.middleware import _folded_escalation
from fused_memory.middleware._folded_escalation import file_folded_escalation

pytestmark = pytest.mark.skipif(
    not _folded_escalation.HAS_ESCALATION,
    reason='escalation package unavailable (minimal env); the HAS_ESCALATION '
           'no-op arm is covered separately below',
)

_ANCHOR = 'folded-escalation-test'
_ROLE = 'fused-memory/folded-escalation-test'
_CATEGORY = 'folded_escalation_test'

logger = logging.getLogger('fused_memory.tests.folded_escalation')


def _emit(tmp_path, **overrides) -> str | None:
    """Call the helper with every required keyword supplied."""
    kwargs = {
        'anchor_task_id': _ANCHOR,
        'agent_role': _ROLE,
        'category': _CATEGORY,
        'severity': 'blocking',
        'summary': 'a folded summary',
        'detail': 'a folded detail\nwith two lines',
        'suggested_action': 'do the thing',
        'logger': logger,
        'log_label': 'folded_escalation_test',
    }
    kwargs.update(overrides)
    root = kwargs.pop('project_root', str(tmp_path))
    return file_folded_escalation(root, **kwargs)


def _filed(tmp_path) -> list[dict]:
    queue_dir = tmp_path / 'data' / 'escalations'
    if not queue_dir.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(queue_dir.glob('esc-*.json'))]


class TestTheFiledEscalation:
    """One escalation, into the caller-named project's OWN queue."""

    def test_files_exactly_one_escalation_and_returns_its_id(self, tmp_path):
        esc_id = _emit(tmp_path)

        assert isinstance(esc_id, str)
        filed = _filed(tmp_path)
        assert len(filed) == 1
        assert filed[0]['id'] == esc_id

    def test_lands_in_the_named_projects_own_queue(self, tmp_path):
        """`{project_root}/data/escalations` — never the server cwd, where no
        operator watches."""
        _emit(tmp_path)
        assert (tmp_path / 'data' / 'escalations').is_dir()

    def test_the_id_is_minted_off_the_anchor(self, tmp_path):
        esc_id = _emit(tmp_path)
        assert esc_id is not None
        assert esc_id.startswith(f'esc-{_ANCHOR}-'), (
            'the id must be minted by queue.make_id off the caller anchor, so '
            'the series is greppable and the dedupe lookup can find it'
        )

    def test_the_caller_fields_pass_through_unmodified(self, tmp_path):
        """The helper owns the SKELETON, not the content: every caller-supplied
        field lands on the record byte-for-byte."""
        esc_id = _emit(
            tmp_path,
            summary='a very specific summary',
            detail='line one\nline two\nline three',
            suggested_action='grep for write_triage: in the server log',
        )
        assert esc_id is not None
        record = _filed(tmp_path)[0]

        assert record['task_id'] == _ANCHOR
        assert record['agent_role'] == _ROLE
        assert record['category'] == _CATEGORY
        assert record['severity'] == 'blocking'
        assert record['summary'] == 'a very specific summary'
        assert record['detail'] == 'line one\nline two\nline three'
        assert record['suggested_action'] == (
            'grep for write_triage: in the server log'
        )

    def test_severity_is_the_callers_not_the_helpers(self, tmp_path):
        """Two of the seven callers file at `info`, five at `blocking`; the
        helper must never impose one."""
        esc_id = _emit(tmp_path, severity='info')
        assert esc_id is not None
        assert _filed(tmp_path)[0]['severity'] == 'info'

    def test_is_born_at_l1_by_default(self, tmp_path):
        """Every migrated filer is a background server process filing under a
        synthetic anchor that is never dispatched and therefore never has a
        steward, so an L0 entry would have no consumer at all."""
        esc_id = _emit(tmp_path)
        assert esc_id is not None
        assert _filed(tmp_path)[0]['level'] == 1

    def test_an_explicit_level_is_honoured(self, tmp_path):
        """`emit_markup_residue_escalation` carries a caller-supplied level
        (defaulting to 2), so the default must be overridable."""
        esc_id = _emit(tmp_path, level=2)
        assert esc_id is not None
        assert _filed(tmp_path)[0]['level'] == 2


class TestTheAnchorIsRequired:
    """The one property the consolidation must not collapse."""

    def test_anchor_task_id_is_required_keyword_only(self, tmp_path):
        """No default: forgetting the anchor is a TypeError at call time, not a
        silently disabled alarm sharing somebody else's anchor."""
        with pytest.raises(TypeError):
            file_folded_escalation(  # type: ignore[call-arg]
                str(tmp_path),
                agent_role=_ROLE,
                category=_CATEGORY,
                severity='blocking',
                summary='s',
                detail='d',
                suggested_action='a',
                logger=logger,
                log_label='folded_escalation_test',
            )

    def test_the_anchor_cannot_be_passed_positionally(self, tmp_path):
        """Keyword-only, so a caller cannot drift the anchor into another
        parameter's slot by reordering."""
        with pytest.raises(TypeError):
            file_folded_escalation(str(tmp_path), _ANCHOR)  # type: ignore[misc]
