"""The sweep's CLI accepts only exact long-option spellings.

``scripts/fused-memory-flag-marker-check.sh`` is the read-only census
wrapper: it passes caller args straight through to this script and refuses
``--apply`` before invoking it (task 3498). That refusal is complete only
because the sweep's parser accepts ``--apply`` and nothing shorter. Under
argparse's default ``allow_abbrev=True``, ``--a``/``--ap``/``--app``/
``--appl`` all reached ``apply=True`` and walked through the wrapper's guard
into a destructive drain.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'sweep_orphan_flag_markers.py'

_mod = load_script_module(SCRIPT_PATH, mod_name='sweep_orphan_flag_markers')


@pytest.mark.parametrize('prefix', ['--a', '--ap', '--app', '--appl'])
def test_prefix_of_apply_is_rejected(prefix, capsys):
    with pytest.raises(SystemExit) as exc_info:
        _mod._parse_args([prefix])
    assert exc_info.value.code == 2
    assert prefix in capsys.readouterr().err


def test_prefix_of_a_value_option_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _mod._parse_args(['--project', 'reify'])
    assert exc_info.value.code == 2
    assert '--project' in capsys.readouterr().err


def test_exact_spellings_still_parse():
    args = _mod._parse_args(['--apply', '--terminal-drain', '--project-id=reify'])
    assert args.apply is True
    assert args.terminal_drain is True
    assert args.project_id == 'reify'
