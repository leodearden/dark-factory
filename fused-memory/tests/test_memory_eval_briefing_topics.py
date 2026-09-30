"""The E1 registry's briefing topics are pinned to the briefing's own query source.

PRD ``docs/prds/memory-briefing-and-fusion.md`` D9 (INV-5): the registry keys
one topic per briefing query spec, and its tuned phrasings are the queries the
briefing ACTUALLY fires, rendered from ``shared/src/shared/briefing_queries.py``
rather than hand-copied. :func:`briefing_drift` is the single definition of
that pin. A reworded template, a renamed slug, or a changed search scope in
the source fails here instead of silently leaving the probe measuring a query
nobody issues.

Pure: no network, no store, no OPENAI_API_KEY.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from shared.briefing_queries import QUERY_SPECS, BriefingScope, queries_for

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'memory_eval_retrieval_probe.py'
REGISTRY_PATH = Path(__file__).parent / 'fixtures' / 'memory_eval_topic_registry.json'

BRIEFING_DERIVATION = 'briefing_query'
_SCOPE_KEYS = frozenset({'task_id', 'title', 'files'})


def _mod():
    return load_script_module(SCRIPT_PATH, mod_name='memory_eval_retrieval_probe')


def _briefing_scope(raw: dict) -> BriefingScope:
    kwargs = {key: raw[key] for key in _SCOPE_KEYS & raw.keys()}
    if 'files' in kwargs:
        kwargs['files'] = tuple(kwargs['files'])
    return BriefingScope(**kwargs)


def _scope_drift(entry) -> tuple[list[str], list[BriefingScope]]:
    raw_scopes = entry.extra.get('briefing_scopes')
    if not isinstance(raw_scopes, list) or not raw_scopes:
        return [f'{entry.topic}: briefing_scopes must be a non-empty list of objects'], []
    drift: list[str] = []
    scopes: list[BriefingScope] = []
    for raw in raw_scopes:
        if not isinstance(raw, dict) or set(raw) - _SCOPE_KEYS:
            drift.append(f'{entry.topic}: briefing scope {raw!r} is not a BriefingScope object')
            continue
        scope = _briefing_scope(raw)
        if entry.topic not in {spec.slug for spec, _ in queries_for(scope)}:
            drift.append(f'{entry.topic}: briefing scope {raw!r} does not fire this spec')
        scopes.append(scope)
    return drift, scopes


def _search_scope_drift(entry, spec) -> list[str]:
    declared = entry.search_scope
    if not spec.stores and not spec.categories:
        if declared is not None:
            return [f'{entry.topic}: search_scope is declared but the spec searches unscoped']
        return []
    if declared is None:
        return [f'{entry.topic}: search_scope is missing; the spec searches scoped']
    if (declared.stores, declared.categories) != (spec.stores, spec.categories):
        return [
            f'{entry.topic}: search_scope {(declared.stores, declared.categories)!r} '
            f'!= the spec scope {(spec.stores, spec.categories)!r}'
        ]
    return []


def briefing_drift(registry) -> list[str]:
    """Every way *registry*'s briefing topics disagree with the briefing's source.

    Empty when the registry is in step with ``shared.briefing_queries``: one
    topic per spec slug, each scope firing its spec, tuned phrasings equal to
    the rendered queries in order, and the spec's own search scope.
    """
    specs = {spec.slug: spec for spec in QUERY_SPECS}
    entries = {
        entry.topic: entry for entry in registry.entries
        if entry.derived_from == BRIEFING_DERIVATION
    }
    drift = [f'{slug}: no registry topic for this briefing spec' for slug in sorted(specs.keys() - entries.keys())]
    drift += [f'{topic}: briefing_query topic has no spec' for topic in sorted(entries.keys() - specs.keys())]
    for topic in sorted(entries.keys() & specs.keys()):
        entry, spec = entries[topic], specs[topic]
        scope_drift, scopes = _scope_drift(entry)
        drift += scope_drift
        rendered = [q for scope in scopes for s, q in queries_for(scope) if s.slug == topic]
        tuned = [p.text for p in entry.phrasings if not p.held_out]
        if tuned != rendered:
            drift.append(f'{topic}: tuned phrasings {tuned!r} != rendered queries {rendered!r}')
        drift += _search_scope_drift(entry, spec)
    return drift


@pytest.fixture(scope='module')
def registry():
    return _mod().load_topic_registry(REGISTRY_PATH)


def _briefing_entries(registry):
    return [e for e in registry.entries if e.derived_from == BRIEFING_DERIVATION]


def _reword_area_phrasing(payload: dict) -> None:
    entry = next(e for e in payload['entries'] if e['topic'] == 'briefing-conventions-area')
    tuned = next(p for p in entry['phrasings'] if not p['held_out'])
    tuned['text'] = f"{tuned['text']} reworded"


def _drop_generic_search_scope(payload: dict) -> None:
    entry = next(e for e in payload['entries'] if e['topic'] == 'briefing-conventions-generic')
    del entry['search_scope']


def _delete_task_semantic(payload: dict) -> None:
    payload['entries'] = [
        e for e in payload['entries'] if e['topic'] != 'briefing-task-semantic'
    ]


class TestBriefingTopicsArePinnedToTheirSource:

    def test_the_committed_registry_has_no_briefing_drift(self, registry):
        assert briefing_drift(registry) == []

    @pytest.mark.parametrize(('mutate', 'topic'), [
        pytest.param(_reword_area_phrasing, 'briefing-conventions-area', id='reworded-phrasing'),
        pytest.param(_drop_generic_search_scope, 'briefing-conventions-generic', id='dropped-scope'),
        pytest.param(_delete_task_semantic, 'briefing-task-semantic', id='deleted-topic'),
    ])
    def test_drift_is_caught(self, tmp_path, mutate, topic):
        payload = json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
        mutate(payload)
        path = tmp_path / 'registry.json'
        path.write_text(json.dumps(payload), encoding='utf-8')

        drift = briefing_drift(_mod().load_topic_registry(path))

        assert drift
        assert any(line.startswith(f'{topic}:') for line in drift), drift

    def test_rendered_phrasings_carry_no_placeholder(self, registry):
        for entry in _briefing_entries(registry):
            for phrasing in entry.phrasings:
                if phrasing.held_out:
                    continue
                assert '{' not in phrasing.text and '}' not in phrasing.text, (
                    f'{entry.topic}: {phrasing.text!r} carries a template placeholder'
                )

    def test_every_briefing_topic_has_a_held_out_that_is_not_a_rendered_query(
        self, registry,
    ):
        for entry in _briefing_entries(registry):
            rendered = {p.text for p in entry.phrasings if not p.held_out}
            held_out = [p.text for p in entry.held_out_phrasings if p.text not in rendered]
            assert held_out, f'{entry.topic} has no held-out phrasing of its own'
