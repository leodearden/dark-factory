"""Deep merge-ahead α: the ``merge_deep.chain_cap`` knob + kill switch.

Task 3183; PRD ``plans/deep-merge-ahead-prd.md`` task α (Phase 1 foundation).
Covers decision #6 (cap staging 0 -> 6 -> 32, with ``chain_cap=0`` as the kill
switch) and decision #7 (the knob is green-tier hot-reloadable, so
enable/retune/kill never needs a process restart).

This task ships ONLY the knob — β (3184, chain builder) and γ (3185, dispatch
gate) are the consumers, so nothing here asserts dispatch behaviour.

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_psi_admission_reload.py:4-7's stated rationale). Harness helpers
are likewise replicated rather than imported (test_harness_config_reload.py:31-34).
"""

from __future__ import annotations

import json
import sqlite3
from importlib import resources as pkg_resources
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    MergeDeepConfig,
    OrchestratorConfig,
    apply_reload,
    census_config_keys,
    diff_config,
    load_config,
)
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore

# ---------------------------------------------------------------------------
# Harness helpers replicated (not imported) from test_harness_config_reload.py:37-71
# — the established pattern for these harness test helpers.
# ---------------------------------------------------------------------------


def _make_harness(tmp_path: Path) -> tuple[Harness, EventStore]:
    """Build a Harness with a MagicMock RunStore and a real EventStore.

    ``Harness(config)``'s ``__init__`` only wires up in-process collaborators, so
    it is safe to call from a test — no MCP server, no full-startup side effects.
    """
    harness = Harness(OrchestratorConfig(project_root=tmp_path))
    harness._run_store = MagicMock(spec=RunStore)
    harness._run_id = 'run-test-0001'
    event_store = EventStore(tmp_path / 'events.db', 'run-test-0001')
    harness.event_store = event_store
    return harness, event_store


def _load_package_defaults() -> dict:
    """Read the shipped defaults.yaml so tests stay in sync automatically.

    Copied from test_chronic_flake.py:35-38.
    """
    defaults_file = pkg_resources.files('orchestrator') / 'defaults.yaml'
    return yaml.safe_load(defaults_file.read_text())


def _query_events(event_store: EventStore, event_type_str: str) -> list[dict]:
    """Query all rows matching event_type_str from the EventStore's DB."""
    conn = sqlite3.connect(str(event_store.db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT * FROM events WHERE event_type = ? ORDER BY id',
        (event_type_str,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


class TestMergeDeepKnobInProjectYaml:
    """The operator deploy path that ``scripts/merge-deep-set-cap.sh`` (already
    on main) writes: ``merge_deep:\\n  chain_cap: <cap>`` in the project YAML.
    """

    def test_project_yaml_override_loads_through(self, monkeypatch, tmp_path):
        """Project layer beats package defaults (test_config.py:982 precedent)."""
        project_cfg = tmp_path / 'orchestrator.yaml'
        project_cfg.write_text(yaml.dump({'merge_deep': {'chain_cap': 6}}))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        cfg = load_config(project_cfg)
        assert cfg.merge_deep.chain_cap == 6

    def test_knob_is_a_known_census_key(self, monkeypatch, tmp_path):
        """Registering the submodel makes ``merge_deep.chain_cap`` a KNOWN key,
        so the deploy script's edit is neither silently dropped by pydantic
        ``extra='ignore'`` nor censused as a phantom key (which would file a
        born-at-L2 every time the canary is deployed).
        """
        project_cfg = tmp_path / 'orchestrator.yaml'
        project_cfg.write_text(yaml.dump({'merge_deep': {'chain_cap': 6}}))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        assert census_config_keys(project_cfg).unknown == []

    def test_typo_under_the_knob_still_censuses(self, monkeypatch, tmp_path):
        """Negative control: the census must still descend INTO merge_deep, so a
        typo'd leaf is reported rather than swallowed by the new known parent.
        """
        project_cfg = tmp_path / 'orchestrator.yaml'
        project_cfg.write_text(yaml.dump({'merge_deep': {'chain_capp': 6}}))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        paths = [uk.path for uk in census_config_keys(project_cfg).unknown]
        assert 'merge_deep.chain_capp' in paths, (
            f'expected the typo to be censused under the known merge_deep parent; got {paths!r}'
        )


class TestChainCapBoundValidation:
    """``chain_cap`` must be >= 0, with deliberately NO upper bound."""

    @pytest.mark.parametrize('bad_value', [-1, -6, -32])
    def test_negative_cap_rejected(self, bad_value):
        """A negative cap is meaningless in
        ``target_depth = min(len(queue), cap, halving_state)`` — it would silently
        win that min() and disable or underflow the gate, so it must fail loudly
        at construction (loud over silent degradation).
        """
        with pytest.raises(ValidationError):
            MergeDeepConfig(chain_cap=bad_value)

    def test_negative_cap_rejected_at_load(self, monkeypatch, tmp_path):
        """Covers the 'rejected at load' signal: OrchestratorConfig builds the
        merge_deep submodel from the nested dict shape YAML deserializes into, so
        a bad project YAML fails at load rather than reaching dispatch.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError):
            OrchestratorConfig(merge_deep={'chain_cap': -1})  # type: ignore[arg-type]

    @pytest.mark.parametrize('good_value', [0, 1, 6, 32])
    def test_whole_staging_ladder_is_representable(self, good_value):
        """0 = kill switch, 1 = the halving floor (d=1, byte-identical to today's
        adjacent verify), 6 = the ζ canary depth, 32 = η2's "uncapped in
        practice". No upper bound is imposed — θ owns the ceiling question.
        """
        assert MergeDeepConfig(chain_cap=good_value).chain_cap == good_value


class TestMergeDeepReloadDisposition:
    """PRD decision #7: the knob is green-tier, so enable/retune/kill never needs
    a process restart. Mirrors TestPsiAdmissionReloadDisposition
    (test_config_psi_admission_reload.py:74).
    """

    @pytest.mark.parametrize('leaf', list(MergeDeepConfig.model_fields))
    def test_every_leaf_is_reloadable(self, leaf):
        """Parametrized over model_fields, not the literal name, so any knob
        β/γ/θ add to the submodel later is covered by this same test with no edit.
        """
        assert f'merge_deep.{leaf}' in RELOADABLE_FIELDS, (
            f'merge_deep.{leaf!r} is expected to be green-tier reloadable '
            f'but is missing from RELOADABLE_FIELDS'
        )

    def test_cap_edit_lands_in_applied_candidates_not_restart_required(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(merge_deep=MergeDeepConfig(chain_cap=0))
        fresh = OrchestratorConfig(merge_deep=MergeDeepConfig(chain_cap=6))
        diff = diff_config(live, fresh)
        assert 'merge_deep.chain_cap' in diff.applied_candidates
        assert 'merge_deep.chain_cap' not in diff.restart_required

    def test_apply_reload_applies_in_place(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(merge_deep=MergeDeepConfig(chain_cap=0))
        fresh = OrchestratorConfig(merge_deep=MergeDeepConfig(chain_cap=6))
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert report['applied']['merge_deep.chain_cap'] == {'old': 0, 'new': 6}
        assert 'merge_deep.chain_cap' not in report['restart_required']
        assert live.merge_deep.chain_cap == 6

    def test_reload_preserves_submodel_identity(self, monkeypatch, tmp_path):
        """Invariant I3: β's chain builder and γ's dispatch gate may hold a
        reference to the submodel, so a hot-reload must MUTATE it in place rather
        than rebind the field — otherwise a held reference keeps reading a stale
        cap and an operator's kill (cap -> 0) would not take effect.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(merge_deep=MergeDeepConfig(chain_cap=0))
        fresh = OrchestratorConfig(merge_deep=MergeDeepConfig(chain_cap=6))
        submodel = live.merge_deep
        apply_reload(live, fresh)
        assert live.merge_deep is submodel, (
            'apply_reload must mutate the merge_deep submodel in place (I3), not rebind it'
        )
        assert submodel.chain_cap == 6


class TestHarnessReloadAppliesChainCap:
    """The user-observable signal named in the task description: an operator
    hot-reload reports the knob under the ``applied`` disposition.

    This is exactly the assertion ``scripts/merge-deep-set-cap.sh`` (already on
    main) makes over the MCP wire after upserting the cap, expressed here at
    unit level.
    """

    @pytest.mark.asyncio
    async def test_reload_config_reports_chain_cap_as_applied(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        harness, event_store = _make_harness(tmp_path)
        assert harness.config.merge_deep.chain_cap == 0

        fresh = OrchestratorConfig(project_root=tmp_path)
        fresh.merge_deep.chain_cap = 6

        config_path = tmp_path / 'orchestrator.yaml'
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))

        with patch('orchestrator.harness.load_config', return_value=fresh):
            report = await harness.reload_config()

        assert report['reloaded'] is True
        assert report['applied']['merge_deep.chain_cap'] == {'old': 0, 'new': 6}
        assert 'merge_deep.chain_cap' not in report['restart_required']
        assert harness.config.merge_deep.chain_cap == 6

        rows = _query_events(event_store, 'config_reload')
        assert len(rows) == 1, f'Expected exactly one config_reload event row; got {rows!r}'
        data = json.loads(rows[0]['data'])
        assert data['applied']['merge_deep.chain_cap'] == {'old': 0, 'new': 6}


class TestDefaultsYamlMergeDeepBlock:
    """The shipped defaults.yaml declares the ``merge_deep:`` block explicitly so
    the knob is discoverable and retunable in a project's
    dark-factory-orchestrator.yaml without guessing at pydantic defaults
    (the git:/chronic_flake:/psi_admission: precedent).
    """

    def test_defaults_yaml_does_not_drift_from_the_field_default(self, monkeypatch, tmp_path):
        """The single canonical pin for "the shipped default is the kill switch".

        Drift guard, and deliberately the ONLY place the value 0 is asserted:
        ``YamlSettingsSource`` layers defaults.yaml UNDER the project YAML
        (config.py:132), so the STANZA — not the pydantic ``Field(default=...)``
        — is the value that actually applies. A stanza that drifted from the
        Field default would silently win, and the shipped kill switch would be
        off on paper only. Asserting all three together (loaded config attribute,
        defaults.yaml stanza, Field default) subsumes each of them separately:
        it fails loudly if the stanza is dropped or retuned, if the Field default
        changes, or if the knob stops being reachable through a loaded config.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        defaults = _load_package_defaults()
        assert 'merge_deep' in defaults, (
            'the shipped defaults.yaml must declare the merge_deep: block so the '
            'knob is discoverable by an operator reading it'
        )
        assert (
            OrchestratorConfig().merge_deep.chain_cap
            == defaults['merge_deep']['chain_cap']
            == MergeDeepConfig.model_fields['chain_cap'].default
            == 0
        )


class TestChainCapZeroIsANoOp:
    """The α-scoped byte-identity leg of the PRD's kill-switch contract.

    SCOPING (deliberate): the PRD's stronger wording — "cap=0 ⇒ no chain code
    executes on any dispatch path" — belongs to γ (3185) and the ι gate (3187),
    which own the dispatch code. α ships NO dispatch code, so there is no chain
    code path to assert against here; a test asserting the dispatch-level
    property at this scope could not be turned green by this task. At α's scope
    byte-identity means (a) zero config-diff leaves at cap=0, below, and (b) the
    existing merge suites re-run unchanged (step-8).

    Both tests here are KNOB-ANCHORED on purpose. ``OrchestratorConfig`` sets
    ``extra='ignore'``, so an unrecognized ``merge_deep=`` kwarg is silently
    dropped: without the anchor, "no diff / equal dumps" would hold trivially on
    a tree where the field and the defaults.yaml stanza do not exist at all, and
    the tests could not detect the regression they are named for (a later edit
    that removes or renames the knob, or an override that never reaches the
    model). The anchor asserts the leaf is actually PRESENT at 0 first, and the
    override side is built from a typed ``MergeDeepConfig`` instance.
    """

    def _assert_kill_switch_is_live(self, cfg: OrchestratorConfig) -> None:
        """Anchor: the knob really exists on the loaded config, at 0.

        Subscripted rather than compared to the exact dict ``{'chain_cap': 0}``
        so β/γ/θ can add a leaf to the submodel without editing this file — the
        no-op claim itself is carried by the whole-surface assertions below.
        """
        dumped = cfg.model_dump()
        assert 'merge_deep' in dumped, (
            'merge_deep is missing from the config surface — the no-op assertions '
            'below would pass vacuously (extra=ignore drops the kwarg)'
        )
        assert dumped['merge_deep']['chain_cap'] == 0

    def test_explicit_zero_stanza_moves_no_leaf(self, monkeypatch, tmp_path):
        """An explicit kill-switch stanza must be indistinguishable from an absent
        one, so shipping the defaults.yaml block cannot move any config leaf.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig()
        self._assert_kill_switch_is_live(live)
        fresh = OrchestratorConfig(merge_deep=MergeDeepConfig(chain_cap=0))
        diff = diff_config(live, fresh)
        assert diff.applied_candidates == {}
        assert diff.restart_required == {}

    def test_whole_config_surface_is_unperturbed(self, monkeypatch, tmp_path):
        """Whole-config equality: adding the stanza at 0 perturbs nothing else in
        the config surface — the only way α could have changed merge behaviour.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig()
        self._assert_kill_switch_is_live(live)
        fresh = OrchestratorConfig(merge_deep=MergeDeepConfig(chain_cap=0))
        assert live.model_dump() == fresh.model_dump()
