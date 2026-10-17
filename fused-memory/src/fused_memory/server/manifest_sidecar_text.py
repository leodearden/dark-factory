"""Surgical ``task_id`` stamping of a capability-manifest sidecar's YAML text.

:func:`stamp_task_ids` rewrites only the ``task_id`` values of the labelled
task blocks it is given, inserting the key where a block has none. Every other
byte — comments, quoting, key order, blank lines, a BOM, LF, CRLF or CR line
endings — is preserved, because the text is edited at the spans PyYAML's
composer reports rather than re-dumped.

The result is checked before it is returned: re-parsed, it must equal the
original document with exactly those blocks' ``task_id`` set. A shape the edit
cannot handle (a flow-style block, an anchor shared between blocks, anything
unforeseen) fails that check and raises :class:`SidecarStampRefused` instead
of producing a sidecar that says something else.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

_LINE_BREAK = re.compile(r'\r\n|\r|\n')


class SidecarStampRefused(ValueError):
    """The sidecar cannot be stamped without changing more than task_id values."""


@dataclass(frozen=True)
class SidecarStamp:
    """A stamped sidecar's ``text``, and the ``document`` that text parses to.

    ``document`` is the verifier's own parse, so a caller validating the
    stamped sidecar need not parse it again. It is ``None`` when the stamp
    changed nothing — ``text`` is then the input, and there is nothing to write.
    """

    text: str
    document: Any = None


def stamp_task_ids(text: str, task_ids: Mapping[str, int]) -> SidecarStamp:
    """Stamp *text* so each ``label`` in *task_ids* is bound to its task_id.

    Labels absent from the sidecar are ignored. Stamping a value a block
    already carries leaves *text* unchanged.

    Raises:
        SidecarStampRefused: the text is not a sidecar-shaped YAML document,
            a bound block is flow-style, or the edited text would not parse
            back to the original with only those task_ids changed.
    """
    edits: dict[int, tuple[int, str]] = {}
    bound_labels: list[str] = []
    for entry, label_key, label_value in _bound_entries(text, task_ids):
        start, end, replacement = _task_id_edit(
            text, entry, label_key, label_value, task_ids[label_value.value],
        )
        edits[start] = (end, replacement)
        bound_labels.append(label_value.value)

    stamped = text
    for start in sorted(edits, reverse=True):
        end, replacement = edits[start]
        stamped = stamped[:start] + replacement + stamped[end:]
    if stamped == text:
        return SidecarStamp(text)
    return SidecarStamp(
        stamped, _verify_only_task_ids_changed(text, stamped, task_ids, bound_labels),
    )


def _bound_entries(
    text: str, task_ids: Mapping[str, int]
) -> list[tuple[MappingNode, Node, ScalarNode]]:
    """``(entry, label key, label value)`` for every task block whose label is bound."""
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise SidecarStampRefused(f'sidecar is not parseable YAML: {exc}') from exc
    tasks_item = _item(root, 'tasks') if isinstance(root, MappingNode) else None
    if tasks_item is None or not isinstance(tasks_item[1], SequenceNode):
        raise SidecarStampRefused('sidecar has no top-level `tasks` sequence')

    bound: list[tuple[MappingNode, Node, ScalarNode]] = []
    for entry in tasks_item[1].value:
        if not isinstance(entry, MappingNode):
            continue
        label_item = _item(entry, 'label')
        if label_item is None:
            continue
        label_key, label_value = label_item
        if not isinstance(label_value, ScalarNode) or label_value.value not in task_ids:
            continue
        if entry.flow_style:
            raise SidecarStampRefused(
                f'task block {label_value.value!r} is flow-style; only a block-style '
                f'entry can be stamped in place'
            )
        bound.append((entry, label_key, label_value))
    return bound


def _item(mapping: MappingNode, key: str) -> tuple[Node, Node] | None:
    for key_node, value_node in mapping.value:
        if isinstance(key_node, ScalarNode) and key_node.value == key:
            return key_node, value_node
    return None


def _task_id_edit(
    text: str, entry: MappingNode, label_key: Node, label_value: Node, task_id: int,
) -> tuple[int, int, str]:
    """``(start, end, replacement)`` setting *entry*'s task_id to *task_id*."""
    existing = _item(entry, 'task_id')
    if existing is not None:
        value = existing[1]
        start, end = value.start_mark.index, value.end_mark.index
        # An empty value's span sits right after the colon, and `task_id:7`
        # would be a different scalar.
        separator = ' ' if start == end and text[start - 1 : start] == ':' else ''
        return start, end, f'{separator}{task_id}'
    indent = ' ' * label_key.start_mark.column
    return _line_after(text, label_value.end_mark.index, f'{indent}task_id: {task_id}')


def _line_after(text: str, index: int, line: str) -> tuple[int, int, str]:
    """An insertion of *line* after the line holding *index*, in that line's own ending.

    An unterminated final line takes the ending of the line before it.
    """
    line_break = _LINE_BREAK.search(text, index)
    if line_break is None:
        earlier_breaks = _LINE_BREAK.findall(text, 0, index)
        eol = earlier_breaks[-1] if earlier_breaks else '\n'
        return len(text), len(text), f'{eol}{line}{eol}'
    return line_break.end(), line_break.end(), f'{line}{line_break.group()}'


def _verify_only_task_ids_changed(
    original: str, stamped: str, task_ids: Mapping[str, int], bound_labels: list[str],
) -> Any:
    """The parse of *stamped*, once it is shown to differ from *original* only in task_ids."""
    expected: Any = yaml.safe_load(original)
    for entry in expected['tasks']:
        label = entry.get('label') if isinstance(entry, dict) else None
        if isinstance(label, str) and label in task_ids:
            entry['task_id'] = task_ids[label]
    try:
        actual = yaml.safe_load(stamped)
    except yaml.YAMLError:
        actual = None
    if actual != expected:
        raise SidecarStampRefused(
            f'stamping task_id for label(s) {", ".join(map(repr, bound_labels))} would '
            f'change more than their task_id values'
        )
    return actual
