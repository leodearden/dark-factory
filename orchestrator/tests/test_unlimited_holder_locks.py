"""Lock keys that take unlimited holders by rule (task 5467).

``unlimited_holder_locks`` names keys by directory prefix and by root-level
Markdown; the lock table gives such a key no holder limit, an exact
``module_overrides`` entry still wins, and the rule is off unless a project
sets it.  Everything goes through ``ModuleLockTable`` and
``OrchestratorConfig``'s public seams.

Every test runs on the code defaults, not the operational yaml: pydantic-settings
merges a dict field such as ``module_overrides`` across the kwarg and the yaml,
so the yaml's own entries would otherwise leak into each table.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    OrchestratorConfig,
    UnlimitedHolderLocksConfig,
    apply_reload,
    census_unknown_config_keys,
    diff_config,
    load_config,
)
from orchestrator.lock_table import ModuleLockTable

pytestmark = pytest.mark.usefixtures('code_default_config')

DOCS_AND_ROOT_MARKDOWN = UnlimitedHolderLocksConfig(prefixes=('docs/',), root_markdown=True)
OFF = UnlimitedHolderLocksConfig()
MANY = 120

#: Must stay exclusive whatever a project sets with docs/ and root Markdown.
EXCLUSIVE_KEYS = (
    'crates/reify-mcp/src/tools/chunks/a.md',
    'skills/x/SKILL.md',
    'dark-factory-orchestrator.yaml',
    'docsite/x.md',
    'x/docs/y.md',
)


def _table(
    rule: UnlimitedHolderLocksConfig = OFF,
    *,
    lock_depth: int = 12,
    module_overrides: dict[str, int] | None = None,
) -> ModuleLockTable:
    return ModuleLockTable(OrchestratorConfig(
        max_per_module=1,
        lock_depth=lock_depth,
        module_overrides=module_overrides or {},
        unlimited_holder_locks=rule,
    ))


def _admits_many(table: ModuleLockTable, key: str) -> bool:
    return all(table.try_acquire(f'T{n}', [key]) for n in range(MANY))


def _is_exclusive(table: ModuleLockTable, key: str) -> bool:
    assert table.try_acquire('first', [key])
    return not table.try_acquire('second', [key])


class TestOffUnlessSet:
    def test_the_code_default_rule_is_empty_and_off(self):
        assert OrchestratorConfig().unlimited_holder_locks == OFF

    @pytest.mark.parametrize('key', ['OPERATIONS.md', 'docs/x.md', 'docs'])
    def test_without_the_rule_a_documentation_key_takes_one_holder(self, key):
        assert _is_exclusive(_table(), key)


class TestRuleOn:
    @pytest.mark.parametrize('key', ['OPERATIONS.md', 'docs/x.md', 'docs/a/b/c.md'])
    def test_any_number_of_tasks_hold_a_documentation_key(self, key):
        assert _admits_many(_table(DOCS_AND_ROOT_MARKDOWN), key)

    def test_a_task_whose_other_files_are_free_is_admitted_beside_a_documentation_holder(self):
        table = _table(DOCS_AND_ROOT_MARKDOWN)
        assert table.try_acquire('A', ['OPERATIONS.md', 'docs/x.md', 'pkg/a.py'])

        assert table.blockers('B', ['OPERATIONS.md', 'docs/x.md', 'pkg/b.py']) == []
        assert table.try_acquire('B', ['OPERATIONS.md', 'docs/x.md', 'pkg/b.py'])
        assert not table.try_acquire('C', ['OPERATIONS.md', 'pkg/a.py']), (
            'the rule must not widen the code file the first holder still holds'
        )

    @pytest.mark.parametrize('key', EXCLUSIVE_KEYS)
    def test_keys_outside_the_rule_stay_exclusive(self, key):
        assert _is_exclusive(_table(DOCS_AND_ROOT_MARKDOWN), key)

    def test_a_shallow_lock_depth_keys_the_directory_itself_and_it_still_matches(self):
        table = _table(DOCS_AND_ROOT_MARKDOWN, lock_depth=1)

        assert _admits_many(table, 'docs/a/b.md')

    def test_root_markdown_alone_leaves_docs_exclusive(self):
        table = _table(UnlimitedHolderLocksConfig(root_markdown=True))

        assert _admits_many(table, 'README.md')
        assert _is_exclusive(table, 'docs/x.md')

    def test_prefixes_alone_leave_root_markdown_exclusive(self):
        table = _table(UnlimitedHolderLocksConfig(prefixes=('docs/',)))

        assert _admits_many(table, 'docs/x.md')
        assert _is_exclusive(table, 'README.md')

    def test_root_markdown_is_case_sensitive(self):
        assert _is_exclusive(_table(DOCS_AND_ROOT_MARKDOWN), 'NOTES.MD')


class TestExactOverrideBeatsTheRule:
    def test_an_override_pins_one_documentation_file_back_to_one_holder(self):
        table = _table(DOCS_AND_ROOT_MARKDOWN, module_overrides={'docs/exclusive.md': 1})

        assert _is_exclusive(table, 'docs/exclusive.md')
        assert _admits_many(table, 'docs/other.md')

    def test_an_override_still_raises_a_key_outside_the_rule(self):
        table = _table(DOCS_AND_ROOT_MARKDOWN, module_overrides={'pkg/hub.py': 2})

        assert table.try_acquire('A', ['pkg/hub.py'])
        assert table.try_acquire('B', ['pkg/hub.py'])
        assert not table.try_acquire('C', ['pkg/hub.py'])


class TestModuleConfigsAroundTheRule:
    def test_a_module_override_pins_back_and_a_module_width_yields_to_the_rule(self, tmp_path: Path):
        """Loaded as production loads it: ``ORCH_PROJECT_ROOT`` is this test's tmp_path."""
        (tmp_path / 'docs').mkdir()
        (tmp_path / 'docs' / 'orchestrator.yaml').write_text(
            'max_per_module: 1\nmodule_overrides:\n  docs/pinned.md: 1\n'
        )
        config_file = tmp_path / 'project.yaml'
        config_file.write_text(
            'lock_depth: 12\nmax_per_module: 1\n'
            'unlimited_holder_locks:\n  prefixes: ["docs/"]\n'
        )
        config = load_config(config_file)
        assert config.for_module('docs/x.md') is not None, 'the docs/ module config must load'
        table = ModuleLockTable(config)

        assert _admits_many(table, 'docs/x.md')
        assert _is_exclusive(table, 'docs/pinned.md')


class TestReservationsOnAnUnlimitedKey:
    def test_a_park_on_a_documentation_key_blocks_no_one(self):
        table = _table(DOCS_AND_ROOT_MARKDOWN)
        table.install_parks('P', ['OPERATIONS.md'], 'critical')

        assert table.park_owners_blocking('C', ['OPERATIONS.md']) == []
        assert _admits_many(table, 'OPERATIONS.md')
        assert table.try_acquire('P', ['OPERATIONS.md'])

    def test_a_park_on_a_limit_one_key_still_blocks(self):
        table = _table(DOCS_AND_ROOT_MARKDOWN)
        table.install_parks('P', ['pkg/a.py'], 'critical')

        assert table.park_owners_blocking('C', ['pkg/a.py']) == ['P']

    def test_a_reserving_task_blocked_only_on_documentation_is_free_to_start(self):
        table = _table(DOCS_AND_ROOT_MARKDOWN)
        assert table.try_acquire('H', ['OPERATIONS.md', 'docs/x.md'])
        table.install_parks('P', ['OPERATIONS.md', 'docs/x.md', 'pkg/p.py'], 'critical')

        assert table.blockers('P', ['OPERATIONS.md', 'docs/x.md', 'pkg/p.py']) == []
        assert table.try_acquire('P', ['OPERATIONS.md', 'docs/x.md', 'pkg/p.py'])


class TestPrefixValidation:
    @pytest.mark.parametrize('prefix', [
        '', 'docs', '/docs/', 'docs\\x/', './docs/', 'docs/../x/', 'docs//x/', '/',
        'docs/**/', 'docs/*.md/', 'doc?/', 'docs/[ab]/', 'docs /', ' docs/', 'docs/\tx/',
    ])
    def test_a_malformed_prefix_is_refused_by_name(self, prefix):
        with pytest.raises(ValidationError) as caught:
            UnlimitedHolderLocksConfig(prefixes=(prefix,))

        assert repr(prefix) in str(caught.value)

    def test_a_duplicate_prefix_is_refused_by_name(self):
        with pytest.raises(ValidationError, match="'docs/'"):
            UnlimitedHolderLocksConfig(prefixes=('docs/', 'docs/'))

    @pytest.mark.parametrize(('block', 'named'), [
        ('prefixes: ["docs"]', "'docs'"),
        ('prefix: ["docs/"]', 'prefix'),
    ])
    def test_loading_a_config_file_with_a_bad_block_fails(self, tmp_path: Path, block, named):
        config_file = tmp_path / 'orchestrator.yaml'
        config_file.write_text(f'unlimited_holder_locks:\n  {block}\n')

        with pytest.raises(ValidationError, match=named):
            load_config(config_file)

    def test_a_prefix_as_deep_as_lock_depth_is_accepted(self):
        rule = UnlimitedHolderLocksConfig(prefixes=('docs/prds/',))

        assert OrchestratorConfig(lock_depth=2, unlimited_holder_locks=rule).unlimited_holder_locks == rule

    def test_a_prefix_deeper_than_lock_depth_is_refused_naming_it_and_the_depth(self):
        rule = UnlimitedHolderLocksConfig(prefixes=('docs/prds/old/',))

        with pytest.raises(ValidationError, match=r"'docs/prds/old/'.*lock_depth 2"):
            OrchestratorConfig(lock_depth=2, unlimited_holder_locks=rule)

    def test_a_nested_well_formed_prefix_is_accepted(self):
        assert UnlimitedHolderLocksConfig(prefixes=('docs/prds/',)).prefixes == ('docs/prds/',)


class TestConfigLifecycle:
    def test_a_change_to_the_rule_is_restart_required_and_never_applied(self):
        live = OrchestratorConfig(unlimited_holder_locks=OFF)
        fresh = OrchestratorConfig(unlimited_holder_locks=DOCS_AND_ROOT_MARKDOWN)

        assert set(diff_config(live, fresh).restart_required) == {
            'unlimited_holder_locks.prefixes', 'unlimited_holder_locks.root_markdown',
        }
        report = apply_reload(live, fresh)
        assert report['applied'] == {}
        assert live.unlimited_holder_locks == OFF

    def test_the_unknown_key_census_accepts_the_block(self, tmp_path: Path):
        config_file = tmp_path / 'orchestrator.yaml'
        config_file.write_text(
            'unlimited_holder_locks:\n  prefixes: ["docs/"]\n  root_markdown: true\n'
        )

        assert census_unknown_config_keys(config_file) == []
