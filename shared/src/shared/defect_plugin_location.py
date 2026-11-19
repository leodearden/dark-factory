"""Where the injected pytest plugin ``_df_defect_plugin`` is installed (plans/defect-tests-prd.md decision 19).

verify appends ``DEFECT_PLUGIN_DIR`` to a target test leg's ``PYTHONPATH`` so
that ``-p _df_defect_plugin`` imports there. This module sits beside the
directory it names, never inside it: that directory holds the plugin module
alone, so putting it on a target's path shadows nothing.
"""

from __future__ import annotations

from pathlib import Path

DEFECT_PLUGIN_DIR = Path(__file__).parent / '_pytest_inject'
