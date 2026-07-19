"""Config substrate for the μ OFAT→matrix→confirm driver (task 2478, step-01/02).

Pure-data assertions over ``evals/configs.py`` — no LLM, no I/O. The OFAT
methodology (PRD §μ, contract C4) varies exactly ONE role per candidate with
every other role pinned to its incumbent baseline; the matrix stage crosses
architect×implementer survivors INCLUDING same-family diagonals (the plan-style
/ implementer coupling hypothesis, PRD decision 9). This module pins:

  - ``ofat_candidates()``  — the one-role-each OFAT set (>=2 implementer cloud
    incumbents Opus/Sonnet, the G2 >=2-config floor, + the architect candidates)
  - ``INCUMBENTS``         — the pinned incumbent baseline per varied role
  - ``matrix_pairs()``     — the FULL architect×implementer cross product
  - ``same_family()``      — flags a same-model/backend diagonal (never excludes)

Imports are inside each test (the test_eval_architect RED convention) so an
absent symbol fails that test alone rather than erroring collection of the file.
"""

from __future__ import annotations


class TestOfatCandidates:
    def test_returns_candidates_each_varying_one_of_the_two_ofat_roles(self):
        from orchestrator.evals.configs import ofat_candidates

        candidates = ofat_candidates()
        assert candidates  # non-empty
        # Each candidate varies exactly ONE role — either the implementer or the
        # architect; no candidate is multi-role.
        assert all(c.role in ('implementer', 'architect') for c in candidates)
        # Candidate names are unique (they key results / lookup).
        names = [c.name for c in candidates]
        assert len(names) == len(set(names))

    def test_implementer_subset_has_opus_and_sonnet_cloud_incumbents(self):
        from orchestrator.evals.configs import ofat_candidates

        impl = [c for c in ofat_candidates() if c.role == 'implementer']
        # G2 >=2-config floor: at least one Opus AND one Sonnet cloud incumbent.
        assert len(impl) >= 2
        models = {c.model for c in impl}
        assert 'opus' in models
        assert 'sonnet' in models
        # Cloud incumbents: native claude backend, no proxy env_overrides (that
        # is what distinguishes a cloud baseline from a vLLM-proxied config).
        assert all(c.backend == 'claude' for c in impl)
        assert all(not c.env_overrides for c in impl)

    def test_architect_subset_reuses_architect_eval_configs(self):
        from orchestrator.evals.configs import ARCHITECT_EVAL_CONFIGS, ofat_candidates

        arch = [c for c in ofat_candidates() if c.role == 'architect']
        assert arch
        assert all(c.role == 'architect' for c in arch)
        # Reuses ARCHITECT_EVAL_CONFIGS verbatim (θ substrate), never a fresh set.
        assert {c.name for c in arch} == {c.name for c in ARCHITECT_EVAL_CONFIGS}


class TestIncumbents:
    def test_maps_each_varied_role_to_its_pinned_incumbent_baseline(self):
        from orchestrator.evals.configs import INCUMBENTS

        # build_eval_orch_config pins architect='opus' and reviewer='opus'; those
        # are the incumbent baselines every OFAT candidate holds the OTHER roles
        # at while it varies its own.
        assert INCUMBENTS['architect'] == 'opus'
        assert INCUMBENTS['reviewer'] == 'opus'


def _arch_cfgs():
    from orchestrator.evals.configs import EvalConfig

    return [
        EvalConfig('architect-opus-high', 'claude', 'opus', 'high', role='architect'),
        EvalConfig('architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect'),
    ]


def _impl_cfgs():
    from orchestrator.evals.configs import EvalConfig

    return [
        EvalConfig('claude-opus-high', 'claude', 'opus', 'high'),
        EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max'),
    ]


class TestMatrixPairs:
    def test_full_cross_product_length(self):
        from orchestrator.evals.configs import matrix_pairs

        arch, impl = _arch_cfgs(), _impl_cfgs()
        pairs = matrix_pairs(arch, impl)
        assert len(pairs) == len(arch) * len(impl)

    def test_pairs_are_architect_then_implementer_tuples(self):
        from orchestrator.evals.configs import matrix_pairs

        arch, impl = _arch_cfgs(), _impl_cfgs()
        pairs = matrix_pairs(arch, impl)
        arch_names = {c.name for c in arch}
        impl_names = {c.name for c in impl}
        for a, i in pairs:
            assert a.role == 'architect'
            assert a.name in arch_names
            assert i.name in impl_names
        # Every (arch, impl) combination is present — the FULL cross product.
        assert {(a.name, i.name) for a, i in pairs} == {
            (a.name, i.name) for a in arch for i in impl
        }

    def test_includes_same_family_diagonal(self):
        from orchestrator.evals.configs import matrix_pairs, same_family

        arch, impl = _arch_cfgs(), _impl_cfgs()
        pairs = matrix_pairs(arch, impl)
        # The sonnet-architect × sonnet-implementer diagonal is deliberately IN.
        diagonal = [
            (a, i) for a, i in pairs
            if a.model == 'sonnet' and i.model == 'sonnet'
        ]
        assert diagonal, 'same-family diagonal was excluded from the matrix'
        a, i = diagonal[0]
        assert same_family(a, i)


class TestSameFamily:
    def test_same_model_and_backend_is_true(self):
        from orchestrator.evals.configs import EvalConfig, same_family

        a = EvalConfig('architect-sonnet-high', 'claude', 'sonnet', 'high', role='architect')
        b = EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max')
        assert same_family(a, b) is True

    def test_different_model_is_false(self):
        from orchestrator.evals.configs import EvalConfig, same_family

        a = EvalConfig('architect-opus-high', 'claude', 'opus', 'high', role='architect')
        b = EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max')
        assert same_family(a, b) is False

    def test_different_backend_is_false(self):
        from orchestrator.evals.configs import EvalConfig, same_family

        a = EvalConfig('claude-sonnet', 'claude', 'sonnet', 'high')
        b = EvalConfig('codex-sonnet', 'codex', 'sonnet', 'high')
        assert same_family(a, b) is False
