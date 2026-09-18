"""Guards for scripts/gen_dashboard_task_vocab.py and the file it generates.

The generator is a pure RENDERER: every fact in its output comes from
``shared.task_statuses`` and ``dashboard.data.census``. These tests derive
their expectations from those same two modules and restate no vocabulary of
their own — a second hand-written copy here would make the byte-equality
parity test below prove nothing about the vocabulary.

``import gen_dashboard_task_vocab`` resolves because tests/scripts/conftest.py
inserts ``scripts/`` onto sys.path; pytest's importlib mode deliberately does
not.
"""

import json
import pathlib
import re

import gen_dashboard_task_vocab
from dashboard.data import census
from shared.task_statuses import TaskStatus

REPO_ROOT = pathlib.Path(__file__).parents[2]

# A top-level (column-zero) JS binding. The generated file must declare exactly
# one, because classic_script_scope.test.mjs loads every classic script into
# ONE shared global lexical scope, where a second binding is a live collision
# with whatever the next module declares.
TOP_LEVEL_BINDING_RE = re.compile(
    r"^(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)", re.MULTILINE
)


def test_payload_has_exactly_the_four_contract_keys() -> None:
    """The generated vocabulary is closed: MEMBERS, VIEWS, SUB_VIEWS, TONES."""
    payload = gen_dashboard_task_vocab.task_vocab_payload()
    assert set(payload) == {"MEMBERS", "VIEWS", "SUB_VIEWS", "TONES"}


def test_payload_members_are_the_statuses_in_declaration_order() -> None:
    """MEMBERS mirrors the enum, order included, so the SPA can iterate it."""
    payload = gen_dashboard_task_vocab.task_vocab_payload()
    assert payload["MEMBERS"] == [member.value for member in TaskStatus]


def test_payload_views_derive_from_the_census_constants() -> None:
    """Each view lists its members' values, sorted so the output is stable."""
    payload = gen_dashboard_task_vocab.task_vocab_payload()
    assert payload["VIEWS"] == {
        view.value: sorted(member.value for member in members)
        for view, members in census.VIEWS.items()
    }
    assert payload["SUB_VIEWS"] == {
        view.value: sorted(member.value for member in members)
        for view, members in census.SUB_VIEWS.items()
    }


def test_payload_tones_derive_from_the_census_constant() -> None:
    """One tone per status, keyed by the plain status string."""
    payload = gen_dashboard_task_vocab.task_vocab_payload()
    assert payload["TONES"] == {member.value: census.TONES[member] for member in TaskStatus}


def test_payload_is_json_serialisable() -> None:
    """The renderer embeds this via json.dumps, so nothing may be enum-shaped."""
    payload = gen_dashboard_task_vocab.task_vocab_payload()
    assert json.loads(json.dumps(payload)) == payload


def test_render_declares_exactly_one_top_level_binding() -> None:
    """A second top-level const would collide in the shared classic-script scope."""
    rendered = gen_dashboard_task_vocab.render_task_vocab_js()
    assert TOP_LEVEL_BINDING_RE.findall(rendered) == ["TASK_VOCAB_API"]


def test_render_ends_with_the_dual_export_block() -> None:
    """node resolves the file as CommonJS; the browser gets the window global."""
    rendered = gen_dashboard_task_vocab.render_task_vocab_js()
    assert "module.exports = TASK_VOCAB_API" in rendered
    assert "window.DF_TASK_VOCAB = TASK_VOCAB_API" in rendered


def test_render_terminates_with_a_newline() -> None:
    """A committed text artifact ends with a newline; byte equality pins it."""
    assert gen_dashboard_task_vocab.render_task_vocab_js().endswith("\n")
