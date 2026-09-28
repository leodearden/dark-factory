"""Render the dashboard's task vocabulary into a classic browser script.

This generator OWNS NO VOCABULARY. Every fact it emits comes from
``shared/src/shared/task_statuses.py`` (the members) and
``dashboard/src/dashboard/data/census.py`` (the views and tones), so the
committed JS cannot disagree with the Python the server shapes payloads
with. That property is the whole point: a generator that invented a fact of
its own would be a SECOND source, and byte-equality between a file and the
generator that produced it would then prove nothing about the vocabulary.

Committed output:
    dashboard/src/dashboard/static/redux/task_vocab.js

Regenerate it with::

    python3 scripts/gen_dashboard_task_vocab.py \\
        --output dashboard/src/dashboard/static/redux/task_vocab.js

Guarded by ``tests/scripts/test_dashboard_task_vocab.py``, which regenerates
to a temp file and asserts byte equality with the committed copy. There is
deliberately no pre-commit hook, no CI regeneration step and no ``--check``
mode: the repo root sets ``merge_verify_breadth: full``, so that test runs on
every merge regardless of which files a branch touched — which is exactly the
case that matters, a ``census.py`` edit landing without regenerating the JS.

The output declares exactly ONE top-level binding, ``TASK_VOCAB_API``. That
is a hard requirement rather than tidiness:
``dashboard/tests/js/classic_script_scope.test.mjs`` loads every classic
script into ONE shared global lexical scope, so a generated top-level
``const MEMBERS``/``VIEWS``/``TONES`` would be a live collision with whatever
the next module declares.
"""

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence

from dashboard.data import census
from shared.task_statuses import TaskStatus

BINDING_NAME = "TASK_VOCAB_API"

WINDOW_GLOBAL = "DF_TASK_VOCAB"

HEADER = """// GENERATED FILE — DO NOT EDIT BY HAND.
//
// Rendered by scripts/gen_dashboard_task_vocab.py from the Python vocabulary:
// shared/src/shared/task_statuses.py (members) and
// dashboard/src/dashboard/data/census.py (views, tones).
//
// Regenerate with:
//   python3 scripts/gen_dashboard_task_vocab.py \\
//       --output dashboard/src/dashboard/static/redux/task_vocab.js
//
// tests/scripts/test_dashboard_task_vocab.py fails if this file and that
// vocabulary have drifted.
"""


def task_vocab_payload() -> dict[str, object]:
    """Build the vocabulary the SPA reads, entirely from the Python sources.

    Returns:
        ``{MEMBERS, VIEWS, SUB_VIEWS, TONES}``. ``MEMBERS`` keeps the enum's
        declaration order; each view's member list is SORTED so the rendered
        bytes are stable across runs (a ``frozenset``'s iteration order is
        not), which is what makes byte-equality a meaningful guard.
    """
    return {
        "MEMBERS": [member.value for member in TaskStatus],
        "VIEWS": {
            view.value: sorted(member.value for member in members)
            for view, members in census.VIEWS.items()
        },
        "SUB_VIEWS": {
            view.value: sorted(member.value for member in members)
            for view, members in census.SUB_VIEWS.items()
        },
        "TONES": {member.value: census.TONES[member] for member in TaskStatus},
    }


def render_task_vocab_js() -> str:
    """Render the committed classic script as text. Pure; touches no file."""
    payload = json.dumps(task_vocab_payload(), indent=2, sort_keys=False)
    return (
        f"{HEADER}\n"
        f"const {BINDING_NAME} = {payload};\n"
        f"\n"
        f"if (typeof module !== 'undefined' && module.exports) {{\n"
        f"  module.exports = {BINDING_NAME}\n"
        f"}}\n"
        f"if (typeof window !== 'undefined') {{\n"
        f"  window.{WINDOW_GLOBAL} = {BINDING_NAME}\n"
        f"}}\n"
    )


def main(argv: "Sequence[str] | None" = None) -> int:
    """Write the rendered vocabulary to --output.

    ``--output`` is REQUIRED and there is no script-relative default: a second
    path-resolution rule is a second thing that can be wrong, and the caller
    already knows where the artifact lives (see this module's docstring). All
    filesystem access lives here so the two render functions stay pure and
    directly testable.
    """
    parser = argparse.ArgumentParser(
        description="Render the dashboard task vocabulary into a classic browser script"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the rendered task_vocab.js to",
    )
    args = parser.parse_args(argv)

    pathlib.Path(args.output).write_text(render_task_vocab_js(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
