"""Every task read has one access path: the AST-executed half of PRD decision 17.

``plans/dashboard-one-datum-one-path-prd.md`` decision 17(b) replaces token
greps with one executed check: ``fetch_tasks``, ``fetch_task_page``,
``fetch_statuses`` and ``fetch_task`` are used only where an
:class:`AccessGrant` below says so, and the one named exemption is
``app.py::_fanout_probe_completion``. The apparatus — a finder, fixture tests
of its matcher, then acceptance tests over a scan set that fails loudly — is
``test_clock_discipline.py``'s.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# The finder
# ---------------------------------------------------------------------------

GUARDED = frozenset({'fetch_tasks', 'fetch_task_page', 'fetch_statuses', 'fetch_task'})


@dataclass(frozen=True, slots=True)
class AccessUse:
    """One Load of a guarded read, and the def it sits in (``'<module>'`` at top level)."""

    line: int
    qualname: str
    name: str


@dataclass(frozen=True, slots=True)
class AccessGrant:
    """Where a guarded read may be used: a module, optionally one def in it.

    *module* is a path relative to ``src/dashboard``. *scope* is ``None`` for
    the whole module, or the exact qualname of the one def the grant covers.
    """

    module: str
    scope: str | None
    names: frozenset[str]
    reason: str


def find_access_path_uses(source: str) -> list[AccessUse]:
    raise NotImplementedError


def violations(
    uses_by_module: Mapping[str, Sequence[AccessUse]],
    grants: Iterable[AccessGrant],
) -> list[tuple[str, AccessUse]]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Matcher unit tests (fixtures, not the real tree)
# ---------------------------------------------------------------------------


def test_a_bare_call_reports_its_enclosing_def():
    source = (
        'async def f(client, config, root):\n'
        '    return await fetch_tasks(client, config, root)\n'
    )
    assert find_access_path_uses(source) == [AccessUse(2, 'f', 'fetch_tasks')]


def test_an_attribute_call_is_reported():
    source = (
        'from dashboard.data import tasks\n'
        'async def g(client, config, root):\n'
        '    return await tasks.fetch_statuses(client, config, root)\n'
    )
    assert find_access_path_uses(source) == [AccessUse(3, 'g', 'fetch_statuses')]


def test_an_aliased_import_is_reported_under_the_guarded_name():
    source = (
        'from dashboard.data.tasks import fetch_tasks as ft\n'
        'async def h(client, config, root):\n'
        '    return await ft(client, config, root)\n'
    )
    assert find_access_path_uses(source) == [AccessUse(3, 'h', 'fetch_tasks')]


def test_a_reference_that_is_not_a_call_is_a_use():
    source = (
        'import functools\n'
        'from dashboard.data.tasks import fetch_task_page\n'
        'def k(client, config, root):\n'
        '    return functools.partial(fetch_task_page, client, config, root)\n'
    )
    assert find_access_path_uses(source) == [AccessUse(4, 'k', 'fetch_task_page')]


def test_the_import_statement_itself_is_not_a_use():
    source = (
        'from dashboard.data.tasks import fetch_tasks, fetch_task\n'
        'from dashboard.data.tasks import fetch_statuses as fs\n'
        'import dashboard.data.tasks\n'
    )
    assert find_access_path_uses(source) == []


def test_prose_naming_a_read_is_not_a_use():
    source = (
        '"""Module docstring naming fetch_tasks."""\n'
        '# a comment naming fetch_task_page(...)\n'
        'def m():\n'
        '    """Calls fetch_statuses, says the docstring."""\n'
        "    return 'fetch_task is only a string here'\n"
    )
    assert find_access_path_uses(source) == []


def test_nested_defs_and_methods_report_dotted_qualnames():
    source = (
        'def outer():\n'
        '    def inner():\n'
        '        return fetch_tasks\n'
        '    return inner\n'
        'class Cls:\n'
        '    async def meth(self):\n'
        '        return await fetch_task(None, None, None, 1)\n'
        'TOP = fetch_statuses\n'
    )
    assert find_access_path_uses(source) == [
        AccessUse(3, 'outer.inner', 'fetch_tasks'),
        AccessUse(7, 'Cls.meth', 'fetch_task'),
        AccessUse(8, '<module>', 'fetch_statuses'),
    ]


def test_a_def_named_like_a_read_is_its_definition_not_a_use():
    source = (
        'async def fetch_tasks(client, config, project_root):\n'
        '    return []\n'
        'async def fetch_task(client, config, project_root, task_id):\n'
        '    return {}\n'
    )
    assert find_access_path_uses(source) == []


def test_lookalike_names_are_not_reported():
    source = (
        'from dashboard.data.tasks import fetch_task_prose, fetch_external_statuses\n'
        'from dashboard.data import tasks\n'
        'async def n(client, config, root):\n'
        '    await fetch_task_prose(client, config, root, 1)\n'
        '    await tasks.fetch_external_statuses(client, config, [])\n'
        '    return tasks.fetch_tasks_later\n'
    )
    assert find_access_path_uses(source) == []


def test_a_use_covered_by_its_grant_is_excused():
    use = AccessUse(10, 'acquire', 'fetch_tasks')
    grant = AccessGrant('data/snap.py', 'acquire', frozenset({'fetch_tasks'}), 'test')

    assert violations({'data/snap.py': [use]}, [grant]) == []


def test_a_function_scoped_grant_does_not_cover_another_def_in_its_module():
    granted = AccessUse(10, 'probe', 'fetch_tasks')
    elsewhere = AccessUse(20, 'handler', 'fetch_tasks')
    grant = AccessGrant('app.py', 'probe', frozenset({'fetch_tasks'}), 'test')

    assert violations({'app.py': [granted, elsewhere]}, [grant]) == [('app.py', elsewhere)]


def test_a_module_scoped_grant_covers_every_def_but_only_its_named_reads():
    named = [
        AccessUse(3, 'a', 'fetch_tasks'),
        AccessUse(9, 'Cls.b', 'fetch_tasks'),
        AccessUse(12, '<module>', 'fetch_tasks'),
    ]
    unnamed = AccessUse(15, 'a', 'fetch_task')
    grant = AccessGrant('data/snap.py', None, frozenset({'fetch_tasks'}), 'test')

    assert violations({'data/snap.py': [*named, unnamed]}, [grant]) == [
        ('data/snap.py', unnamed),
    ]


def test_a_grant_for_one_module_does_not_cover_another():
    use = AccessUse(4, 'f', 'fetch_statuses')
    grant = AccessGrant('data/snap.py', None, frozenset({'fetch_statuses'}), 'test')

    assert violations({'data/other.py': [use]}, [grant]) == [('data/other.py', use)]
