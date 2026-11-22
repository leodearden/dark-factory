"""The production litter cleanup: an allowlist of graph names it may delete, and nothing else.

The offline cases drive the script through its own GraphServer Protocol with an
in-file fake, and aim ``main`` at refused ports. The one live case runs on the
scratch server only, over two graphs it creates; run it with
``uv run pytest -m integration tests/test_cleanup_litter_graphs.py``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

CLEANUP = load_script_module(
    Path(__file__).resolve().parents[1] / 'scripts' / 'cleanup_litter_graphs.py'
)

TARGET = 'redis://localhost:6379'


class FakeGraphServer:
    """A GraphServer over a dict of graph name to node count, recording every call.

    *on_run_id* runs at the start of each ``run_id()`` call, after
    ``run_id_calls`` counts it, so a test can restart the server or shrink a
    graph mid-run.
    """

    def __init__(
        self,
        counts: Mapping[str, int],
        run_id: str = 'run-a',
        *,
        on_run_id: Callable[[FakeGraphServer], None] | None = None,
    ) -> None:
        self.counts = dict(counts)
        self.current_run_id = run_id
        self.on_run_id = on_run_id
        self.run_id_calls = 0
        self.counted: list[str] = []
        self.deleted: list[str] = []

    def graph_names(self) -> tuple[str, ...]:
        return tuple(self.counts)

    def node_count(self, name: str) -> int:
        self.counted.append(name)
        return self.counts[name]

    def delete(self, name: str) -> None:
        self.deleted.append(name)
        del self.counts[name]

    def run_id(self) -> str:
        self.run_id_calls += 1
        if self.on_run_id is not None:
            self.on_run_id(self)
        return self.current_run_id

LITTER = (
    '_probe4748b_bbd3e6b1',
    '_probe4748b_0a1b2c3d',
    '_test_3334_falkor_fulltext_1f2e3d4c',
    '_test_530_list_indices_integration_5a6b7c8d',
    '_test_4340_read_pagination_03b38704',
    '_test_3561_episode_uuid_c8c384d4',
    '_test_3769_drop_vector_indices_9e8f7a6b',
    '_test_2207_merge_entities_deadbeef',
    'probe_e1_gw0',
    'probe_e1_gw29',
    'probe_e1_master',
    'sweep_e4_gw2',
    'evalmem_test_0123abcd',
    '_probe',
)

KEPT = (
    'dark_factory',
    'reify',
    'stream_touch',
    'solar_challenge',
    'solar_challenge_platform',
    'scoping_intake',
    'know_live',
    'knowlive',
    'autopilot_video',
    'pump_web_ui',
    'market_predictor',
    'default_db',
    'evalmem_lme_ref_incumbent_a',
)

NEAR_MISSES = (
    '_probe4748b_bbd3e6b',
    '_probe4748b_BBD3E6B1',
    '_probe_extra',
    '_probe\n',
    'probe_e1_gw',
    'probe_e1_master2',
    '_test_2084_edge_dedup_33c83367',
    '_test_7793_kept_0123abcd',
    'evalmem_test_',
    'xevalmem_test_0123abcd',
    'dark_factory_test',
)


def _interleaved(first: tuple[str, ...], second: tuple[str, ...]) -> list[str]:
    longest = max(len(first), len(second))
    return [group[i] for i in range(longest) for group in (first, second) if i < len(group)]


# --- the allowlist ----------------------------------------------------------------------


def test_the_plan_deletes_exactly_the_litter_and_keeps_the_rest_in_listing_order():
    names = _interleaved(LITTER, KEPT)

    plan = CLEANUP.plan_cleanup(names)

    assert plan.delete == tuple(name for name in names if name in LITTER)
    assert plan.keep == tuple(name for name in names if name in KEPT)


@pytest.mark.parametrize('name', NEAR_MISSES)
def test_a_near_miss_is_kept(name):
    plan = CLEANUP.plan_cleanup([name])

    assert plan.delete == ()
    assert plan.keep == (name,)
    assert not CLEANUP.is_litter(name)


def test_the_plan_is_immutable():
    plan = CLEANUP.plan_cleanup(['_probe', 'dark_factory'])

    with pytest.raises(AttributeError):
        plan.delete = ()


def test_the_allowlist_is_exactly_these_seven_patterns():
    assert tuple(pattern.pattern for pattern in CLEANUP.LITTER_PATTERNS) == (
        '_probe',
        '_probe4748b_[0-9a-f]{8}',
        'probe_e1_gw[0-9]+',
        'probe_e1_master',
        'sweep_e4_gw[0-9]+',
        '_test_(?:3334_falkor_fulltext|530_list_indices_integration|4340_read_pagination'
        '|3561_episode_uuid|3769_drop_vector_indices|2207_merge_entities)_[0-9a-f]{8}',
        'evalmem_test_[0-9a-f]{8}',
    )


def test_require_litter_refuses_a_kept_graph_by_name():
    with pytest.raises(CLEANUP.NotLitter, match="'dark_factory'"):
        CLEANUP.require_litter('dark_factory')


def test_require_litter_returns_a_litter_name_unchanged():
    assert CLEANUP.require_litter('probe_e1_gw29') == 'probe_e1_gw29'


# --- the census and the reviewed listing ------------------------------------------------


def test_the_census_reads_the_run_id_and_only_the_named_graphs():
    server = FakeGraphServer({'_probe': 0, 'dark_factory': 41, 'probe_e1_gw0': 3}, 'run-a')

    census = CLEANUP.take_census(server, ['dark_factory', '_probe'])

    assert census.run_id == 'run-a'
    assert dict(census.counts) == {'dark_factory': 41, '_probe': 0}
    assert server.counted == ['dark_factory', '_probe']


def test_the_census_is_immutable():
    census = CLEANUP.take_census(FakeGraphServer({'dark_factory': 41}), ['dark_factory'])

    with pytest.raises(AttributeError):
        census.run_id = 'run-b'
    with pytest.raises(TypeError):
        census.counts['dark_factory'] = 0


def test_the_listing_splits_every_graph_into_delete_and_keep_counts():
    server = FakeGraphServer(
        {'_probe': 0, 'dark_factory': 41, 'probe_e1_gw0': 3, 'default_db': 0}, 'run-a'
    )

    listing = CLEANUP.build_listing(TARGET, server)

    assert listing.target == TARGET
    assert listing.run_id == 'run-a'
    assert datetime.fromisoformat(listing.taken_at).tzinfo is not None
    assert dict(listing.delete) == {'_probe': 0, 'probe_e1_gw0': 3}
    assert dict(listing.keep) == {'dark_factory': 41, 'default_db': 0}


def test_a_written_listing_reads_back_equal(tmp_path):
    listing = CLEANUP.build_listing(
        TARGET, FakeGraphServer({'evalmem_test_0123abcd': 2, 'reify': 7})
    )
    path = tmp_path / 'listing.json'

    CLEANUP.write_listing(path, listing)

    assert CLEANUP.read_listing(path) == listing


def _valid_listing_json() -> dict[str, object]:
    return {
        'target': TARGET,
        'run_id': 'run-a',
        'taken_at': '2026-11-21T00:00:00+00:00',
        'delete': {'_probe': 0},
        'keep': {'dark_factory': 41},
    }


@pytest.mark.parametrize(
    'content',
    [
        pytest.param('{"target": ', id='malformed-json'),
        pytest.param(
            json.dumps({k: v for k, v in _valid_listing_json().items() if k != 'delete'}),
            id='no-delete-key',
        ),
        pytest.param(
            json.dumps({k: v for k, v in _valid_listing_json().items() if k != 'target'}),
            id='no-target-key',
        ),
        pytest.param(json.dumps([_valid_listing_json()]), id='not-an-object'),
        pytest.param(json.dumps({**_valid_listing_json(), 'delete': ['_probe']}), id='delete-not-a-map'),
        pytest.param(json.dumps({**_valid_listing_json(), 'delete': {'_probe': '0'}}), id='count-not-int'),
        pytest.param(json.dumps({**_valid_listing_json(), 'target': 6379}), id='target-not-str'),
    ],
)
def test_an_unreadable_listing_is_refused_naming_its_path(tmp_path, content):
    path = tmp_path / 'listing.json'
    path.write_text(content)

    with pytest.raises(CLEANUP.UnreadableListing, match=re.escape(str(path))):
        CLEANUP.read_listing(path)


def test_a_missing_listing_is_refused_naming_its_path(tmp_path):
    path = tmp_path / 'absent.json'

    with pytest.raises(CLEANUP.UnreadableListing, match=re.escape(str(path))):
        CLEANUP.read_listing(path)
