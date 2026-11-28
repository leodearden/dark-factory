"""Contract tests for the stand-in /proc trees in ``_synthetic_proc``."""
from __future__ import annotations

import os

from _synthetic_proc import real_proc_view


def test_the_view_lists_only_the_named_pids_and_self(tmp_path):
    view = real_proc_view(tmp_path / 'proc', [os.getpid()])

    assert sorted(e.name for e in view.iterdir()) == sorted([str(os.getpid()), 'self'])


def test_entries_serve_the_live_kernel_entries(tmp_path):
    view = real_proc_view(tmp_path / 'proc', [os.getpid()])

    assert os.readlink(view / str(os.getpid()) / 'cwd') == os.getcwd()
    assert (view / 'self' / 'stat').read_text().split()[0] == str(os.getpid())
