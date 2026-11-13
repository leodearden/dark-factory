"""Unit tests for fused_memory.middleware.operational_routing_guard.

Step 1 (RED → step-2 GREEN): the coercion-table matrix for
``inject_operational_routing`` — the unbypassable operational→deterministic
submit-boundary transform (PRD operational-ask-routing, task β).

The module does not exist until step-2, so every test imports the transform
(and the marker constant) LOCALLY inside its body — mirroring
test_execution_class_guard.py's TestInjectExecutionClass style — so this
file still COLLECTS at RED (each test fails on the in-body import) rather
than erroring out at import time.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys

import pytest
from shared.pure_gate_provenance import PURE_GATE_FROM_KEY


def _consolidation_gate_metadata() -> dict:
    from fused_memory.reconciliation.consolidation_gate import (
        build_consolidation_gate_task,
    )

    return build_consolidation_gate_task(topic='pure-gate-provenance-probe').metadata


def _declared_decision_gate_metadata() -> dict:
    return {'task_kind': 'deterministic', 'always_escalates': True, 'execution_class': 'decision'}


class TestInjectOperationalRoutingMatrix:
    """Contract table for inject_operational_routing(metadata) -> Any.

    | execution_class | operational_mode   | result                            |
    |-----------------|--------------------|-----------------------------------|
    | operational     | gate (or absent)   | pure gate                         |
    | decision        | (any)              | pure gate                         |
    | operational     | llm                | pure gate + x_ llm marker         |
    | code_tdd/absent | (any)              | untouched (original object)       |
    """

    # -- coercion branch: operational + gate/absent, decision --------------

    def test_operational_gate_coerces_to_pure_gate(self):
        """(a) operational + operational_mode='gate' → deterministic pure gate."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        result = inject_operational_routing(
            {'execution_class': 'operational', 'operational_mode': 'gate'}
        )
        assert result['task_kind'] == 'deterministic'
        assert result['always_escalates'] is True
        assert 'before_done' not in result

    def test_operational_absent_mode_coerces_to_pure_gate(self):
        """(b) operational with NO operational_mode key (absent≡gate) → pure gate."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        result = inject_operational_routing({'execution_class': 'operational'})
        assert result['task_kind'] == 'deterministic'
        assert result['always_escalates'] is True
        assert 'before_done' not in result

    def test_decision_coerces_to_pure_gate(self):
        """(c) execution_class='decision' → pure gate."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        result = inject_operational_routing({'execution_class': 'decision'})
        assert result['task_kind'] == 'deterministic'
        assert result['always_escalates'] is True
        assert 'before_done' not in result

    # -- coercion branch: operational + llm (marker sub-case) --------------

    def test_operational_llm_stamps_marker_and_preserves_mode(self):
        """(d) operational + operational_mode='llm' → pure gate PLUS the x_ llm
        marker True, and the raw operational_mode='llm' is preserved."""
        from fused_memory.middleware.operational_routing_guard import (
            OPERATIONAL_LLM_GATE_MARKER_KEY,
            inject_operational_routing,
        )

        result = inject_operational_routing(
            {'execution_class': 'operational', 'operational_mode': 'llm'}
        )
        assert result['task_kind'] == 'deterministic'
        assert result['always_escalates'] is True
        assert 'before_done' not in result
        assert result[OPERATIONAL_LLM_GATE_MARKER_KEY] is True
        assert result['operational_mode'] == 'llm'

    def test_decision_llm_does_not_stamp_marker(self):
        """decision + operational_mode='llm' is still a plain pure gate — the
        llm marker is stamped ONLY for execution_class='operational'."""
        from fused_memory.middleware.operational_routing_guard import (
            OPERATIONAL_LLM_GATE_MARKER_KEY,
            inject_operational_routing,
        )

        result = inject_operational_routing(
            {'execution_class': 'decision', 'operational_mode': 'llm'}
        )
        assert result['task_kind'] == 'deterministic'
        assert OPERATIONAL_LLM_GATE_MARKER_KEY not in result

    # -- non-coercion branch: untouched ------------------------------------

    def test_code_tdd_returned_unchanged_same_object(self):
        """(e) code_tdd → returned UNCHANGED (same object; task_kind not forced)."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        original = {'execution_class': 'code_tdd', 'task_kind': 'normal'}
        result = inject_operational_routing(original)
        assert result is original
        assert result.get('task_kind') == 'normal'

    def test_no_execution_class_returned_unchanged(self):
        """(f) no execution_class key → returned unchanged (same object)."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        original = {'foo': 'bar'}
        result = inject_operational_routing(original)
        assert result is original

    def test_none_metadata_returned_unchanged(self):
        """None carries no execution_class → returned unchanged (byte-identical)."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        assert inject_operational_routing(None) is None

    # -- idempotency -------------------------------------------------------

    def test_idempotent_operational_gate(self):
        """(g) idempotent for operational+gate: f(f(m)) == f(m)."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        m = {'execution_class': 'operational', 'operational_mode': 'gate'}
        once = inject_operational_routing(m)
        twice = inject_operational_routing(once)
        assert twice == once

    def test_idempotent_operational_llm(self):
        """(g) idempotent for operational+llm: f(f(m)) == f(m) (marker stable)."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        m = {'execution_class': 'operational', 'operational_mode': 'llm'}
        once = inject_operational_routing(m)
        twice = inject_operational_routing(once)
        assert twice == once

    # -- override-safety ---------------------------------------------------

    def test_override_safe_normal_and_before_done_overwritten(self):
        """(h) a hand-supplied task_kind='normal' + before_done + always_escalates=False
        on an operational task is overwritten to the pure-gate shape.

        Scoped to the transform in isolation. At the full submit_task tool
        boundary, deterministic_task_error (invariants 3/3b) rejects an
        operational submission pre-declaring before_done / always_escalates=True
        under the default task_kind='normal' param *before* this coercion runs —
        see the guard module's "Override-safe" note."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        result = inject_operational_routing(
            {
                'execution_class': 'operational',
                'task_kind': 'normal',
                'before_done': {'script': 'deploy.sh', 'timeout_secs': 60},
                'always_escalates': False,
            }
        )
        assert result['task_kind'] == 'deterministic'
        assert result['always_escalates'] is True
        assert 'before_done' not in result

    # -- purity: caller dict not mutated -----------------------------------

    def test_caller_dict_not_mutated_on_coercion(self):
        """(i) the caller's dict is not mutated on the coercion branch; the
        result is a distinct object."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        original = {
            'execution_class': 'operational',
            'operational_mode': 'llm',
            'task_kind': 'normal',
        }
        original_copy = dict(original)
        result = inject_operational_routing(original)
        assert original == original_copy, 'caller dict must not be mutated'
        assert result is not original, 'result must be a distinct object on coercion'

    # -- JSON-string input -------------------------------------------------

    def test_json_string_metadata_parsed_and_coerced(self):
        """(j) a JSON-string metadata carrying execution_class='operational' is
        parsed and coerced to a deterministic pure gate."""
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        result = inject_operational_routing(
            json.dumps({'execution_class': 'operational'})
        )
        assert isinstance(result, dict)
        assert result['task_kind'] == 'deterministic'
        assert result['always_escalates'] is True

    # -- provenance stamp: pure_gate_from names the trigger -----------------

    @pytest.mark.parametrize(
        ('metadata', 'expected'),
        [
            ({'execution_class': 'operational', 'operational_mode': 'gate'}, 'execution_class=operational'),
            ({'execution_class': 'operational'}, 'execution_class=operational'),
            ({'execution_class': 'operational', 'operational_mode': 'llm'}, 'execution_class=operational'),
            ({'execution_class': 'decision'}, 'execution_class=decision'),
        ],
    )
    def test_coercion_stamps_the_triggering_execution_class(self, metadata, expected):
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        assert inject_operational_routing(metadata)[PURE_GATE_FROM_KEY] == expected

    def test_stamp_keeps_sibling_keys_unchanged(self):
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        result = inject_operational_routing(
            {'execution_class': 'operational', 'files': ['a.py'], 'prd_path': 'plans/x.md', 'x_probe': 1}
        )

        assert result[PURE_GATE_FROM_KEY] == 'execution_class=operational'
        assert result['files'] == ['a.py']
        assert result['prd_path'] == 'plans/x.md'
        assert result['x_probe'] == 1

    def test_json_string_operational_input_is_stamped(self):
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        result = inject_operational_routing(json.dumps({'execution_class': 'operational'}))

        assert result[PURE_GATE_FROM_KEY] == 'execution_class=operational'

    # -- provenance stamp: only a rewrite of the routing is stamped ---------

    @pytest.mark.parametrize(
        'factory',
        [_consolidation_gate_metadata, _declared_decision_gate_metadata],
        ids=['consolidation_gate', 'declared_decision_gate'],
    )
    def test_declared_pure_gate_is_not_stamped_and_gets_no_notice(self, factory):
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
            pure_gate_routing_notice,
        )

        result = inject_operational_routing(factory())

        assert PURE_GATE_FROM_KEY not in result
        assert result['task_kind'] == 'deterministic'
        assert result['always_escalates'] is True
        assert pure_gate_routing_notice(result) == {}

    @pytest.mark.parametrize(
        'metadata',
        [
            {'execution_class': 'operational', 'task_kind': 'deterministic'},
            {
                'execution_class': 'operational',
                'task_kind': 'deterministic',
                'always_escalates': True,
                'before_done': {'kind': 'script', 'script': 'x.sh'},
            },
        ],
        ids=['no_always_escalates', 'has_before_done'],
    )
    def test_partially_declared_gate_is_stamped(self, metadata):
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        assert inject_operational_routing(metadata)[PURE_GATE_FROM_KEY] == 'execution_class=operational'

    def test_second_application_keeps_the_first_stamp(self):
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
            pure_gate_routing_notice,
        )

        once = inject_operational_routing({'execution_class': 'operational'})
        twice = inject_operational_routing(once)

        assert twice[PURE_GATE_FROM_KEY] == 'execution_class=operational'
        assert pure_gate_routing_notice(twice)['routing_notice']['pure_gate_from'] == 'execution_class=operational'

    # -- marker constant hygiene -------------------------------------------

    def test_marker_key_is_census_clean(self):
        """OPERATIONAL_LLM_GATE_MARKER_KEY is x_-prefixed (census-clean: no
        code=unknown_key schema warning)."""
        from fused_memory.middleware.operational_routing_guard import (
            OPERATIONAL_LLM_GATE_MARKER_KEY,
        )

        assert OPERATIONAL_LLM_GATE_MARKER_KEY.startswith('x_')



class TestPureGateRoutingNotice:
    """pure_gate_routing_notice(metadata): the submit-response notice for a converted submission."""

    @pytest.mark.parametrize(
        ('execution_class', 'expected'),
        [('operational', 'execution_class=operational'), ('decision', 'execution_class=decision')],
    )
    def test_converted_submission_yields_one_structured_notice(self, execution_class, expected):
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
            pure_gate_routing_notice,
        )

        notice = pure_gate_routing_notice(inject_operational_routing({'execution_class': execution_class}))

        assert list(notice) == ['routing_notice']
        body = notice['routing_notice']
        assert set(body) == {'pure_gate_from', 'detail', 'hint'}
        assert body['pure_gate_from'] == expected
        assert isinstance(body['detail'], str) and body['detail']
        assert isinstance(body['hint'], str) and body['hint']
        assert "task_kind='normal'" in body['hint']
        assert "complexity='simple'" in body['hint']

    def test_notice_is_informational_never_an_error(self):
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
            pure_gate_routing_notice,
        )

        notice = pure_gate_routing_notice(inject_operational_routing({'execution_class': 'operational'}))

        assert 'error' not in notice and 'error_type' not in notice
        assert 'error' not in notice['routing_notice']
        assert 'error_type' not in notice['routing_notice']

    @pytest.mark.parametrize(
        'metadata',
        [
            None,
            '{"execution_class": "operational"}',
            {'execution_class': 'code_tdd'},
            {'task_kind': 'deterministic', 'always_escalates': True},
            {'pure_gate_from': 'execution_class=operational', 'task_kind': 'normal'},
            {'pure_gate_from': 'bogus', 'task_kind': 'deterministic', 'always_escalates': True},
        ],
        ids=[
            'none',
            'str',
            'code_tdd',
            'hand_authored_gate_without_stamp',
            'stamp_on_non_gate',
            'unknown_stamp_value',
        ],
    )
    def test_no_notice_without_a_valid_stamp_on_a_gate(self, metadata):
        from fused_memory.middleware.operational_routing_guard import (
            pure_gate_routing_notice,
        )

        assert pure_gate_routing_notice(metadata) == {}


_SCHEMA_WARNING = 'task_metadata.schema_warning'
_SOURCE = 'source=operational_routing_guard'


def _discard_lines(caplog) -> list[str]:
    messages = (r.getMessage() for r in caplog.records)
    return [m for m in messages if _SCHEMA_WARNING in m and _SOURCE in m]


class TestUnreadableMetadataIsDiscardedLoudly:
    """Unreadable metadata takes the non-coercion branch, and is logged."""

    def test_malformed_string_is_returned_unchanged_and_logs_one_census_line(self, caplog):
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        original = '{not json'
        with caplog.at_level(logging.WARNING):
            assert inject_operational_routing(original) is original
        assert len(_discard_lines(caplog)) == 1

    @pytest.mark.parametrize('metadata', [None, ''])
    def test_absent_metadata_logs_no_census_line(self, metadata, caplog):
        from fused_memory.middleware.operational_routing_guard import (
            inject_operational_routing,
        )

        with caplog.at_level(logging.WARNING):
            assert inject_operational_routing(metadata) is metadata
        assert _discard_lines(caplog) == []


def test_coercion_runs_without_loading_the_interceptor():
    """The guard sits below the interceptor, so coercing must not import it.

    Probed in a FRESH interpreter: this test process has already imported the
    interceptor transitively, so an in-process ``sys.modules`` check would
    pass vacuously. The ``task_kind`` check keeps the probe from silently
    no-opping if the call ever stops coercing.
    """
    probe = (
        'import sys; '
        'from fused_memory.middleware.operational_routing_guard import inject_operational_routing; '
        "result = inject_operational_routing({'execution_class': 'operational'}); "
        "assert result['task_kind'] == 'deterministic', result; "
        "assert 'fused_memory.middleware.task_interceptor' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, '-c', probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
