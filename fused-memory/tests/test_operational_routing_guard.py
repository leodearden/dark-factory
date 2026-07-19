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

    # -- marker constant hygiene -------------------------------------------

    def test_marker_key_is_census_clean(self):
        """OPERATIONAL_LLM_GATE_MARKER_KEY is x_-prefixed (census-clean: no
        code=unknown_key schema warning)."""
        from fused_memory.middleware.operational_routing_guard import (
            OPERATIONAL_LLM_GATE_MARKER_KEY,
        )

        assert OPERATIONAL_LLM_GATE_MARKER_KEY.startswith('x_')
