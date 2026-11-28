"""Mirror contract: three human-path docs carry ``DEFECT_TEST_RULE`` verbatim.

Task 7047 (PRD ``plans/defect-tests-prd.md`` decision 17). The rule's one home is
``orchestrator/src/orchestrator/agents/roles.py::DEFECT_TEST_RULE``, the text the
factory's fixing agents are given. A human fixing a defect by hand reads it from
``skills/unblock/SKILL.md``, ``skills/unblock-low-risk/SKILL.md`` or
``CONTRIBUTING.md`` instead, so each carries a rendered excerpt between
``defect-test-rule-mirror`` markers, and this guard holds every excerpt equal to
the constant.

It asserts equality of the marked span only. The pointer sentence above it and
every other line of the three docs are outside the marker and unpinned; this
guard must not be extended to them.

The marked span comes from ``tests/scripts/verify_command_invariants.py::marked_body``,
whose ``marked_span`` docstring says why every marker failure is loud.
"""
from __future__ import annotations

import pathlib

import pytest
import verify_command_invariants as vci
from orchestrator.agents.roles import DEFECT_TEST_RULE
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]

MIRROR_SLUG = "defect-test-rule-mirror"

MIRRORING_DOCS = (
    "skills/unblock/SKILL.md",
    "skills/unblock-low-risk/SKILL.md",
    "CONTRIBUTING.md",
)

_FENCE = "```"


def _documented_rule(markdown_text, source):
    """The fence body inside *source*'s ``defect-test-rule-mirror`` span, stripped."""
    marked = vci.marked_body(
        markdown_text,
        slug=MIRROR_SLUG,
        source=source,
        label=(
            "the fenced excerpt that mirrors DEFECT_TEST_RULE in "
            "orchestrator/src/orchestrator/agents/roles.py"
        ),
        task="7047",
    )
    body_lines = [
        line for line in marked.splitlines() if not line.strip().startswith(_FENCE)
    ]
    assert "".join(body_lines).strip(), (
        f"the fenced excerpt inside the {MIRROR_SLUG!r} marker in {source} is empty "
        "(task 7047): this mirror would pass vacuously against an emptied rule"
    )
    return "\n".join(body_lines).strip()


_HAPPY_DOC = """\
The rule's one home is `DEFECT_TEST_RULE`; the excerpt below is rendered from it.

<!-- defect-test-rule-mirror:begin  rendered from roles.py::DEFECT_TEST_RULE -->
```text
## A heading

First paragraph.

Second paragraph.
```
<!-- defect-test-rule-mirror:end -->

Prose after the marker, unpinned.
"""

_HAPPY_RULE = "## A heading\n\nFirst paragraph.\n\nSecond paragraph."

_NO_MARKER_DOC = """\
```text
## A heading
```
"""

_EMPTY_FENCE_DOC = """\
<!-- defect-test-rule-mirror:begin -->
```text
```
<!-- defect-test-rule-mirror:end -->
"""


def test_the_marked_fence_body_is_extracted():
    assert _documented_rule(_HAPPY_DOC, "synthetic.md") == _HAPPY_RULE


def test_a_missing_marker_is_refused_naming_the_marker_and_the_source():
    with pytest.raises(AssertionError) as excinfo:
        _documented_rule(_NO_MARKER_DOC, "synthetic.md")

    message = str(excinfo.value)
    assert MIRROR_SLUG in message
    assert "synthetic.md" in message


def test_a_duplicated_marked_block_is_refused():
    with pytest.raises(AssertionError):
        _documented_rule(_HAPPY_DOC + _HAPPY_DOC, "synthetic.md")


def test_an_empty_fence_is_refused():
    with pytest.raises(AssertionError):
        _documented_rule(_EMPTY_FENCE_DOC, "synthetic.md")


@pytest.mark.parametrize("rel", MIRRORING_DOCS)
def test_the_doc_carries_the_rule_verbatim(rel):
    documented = _documented_rule((REPO_ROOT / rel).read_text(encoding="utf-8"), rel)

    assert documented == DEFECT_TEST_RULE.strip(), (
        f"{rel}'s {MIRROR_SLUG!r} excerpt differs from "
        "orchestrator/src/orchestrator/agents/roles.py::DEFECT_TEST_RULE. Paste "
        "DEFECT_TEST_RULE.strip() verbatim inside the marker's fence; never paraphrase it."
    )
