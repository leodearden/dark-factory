"""cockpit.panes.placeholders — the text the cockpit renders for an absent value.

Fail-soft rendering (PRD §2) means every surface has to put SOMETHING on
screen for a field the record left empty, and the operator meets the same
absent field on several of them -- the detail pane's session and decision
renders, and the queue row's clipboard payload. A placeholder shared by
more than one module is spelled here once so those surfaces cannot drift
apart (SPOT); a placeholder only one surface can ever show (the detail
pane's empty-selection and no-result-file text) stays private to that
module, where its meaning is local.
"""

from __future__ import annotations

# Never interpolate a possibly-None value straight into a rendered line:
# the literal 'None' reads as a value the record actually carries.
ABSENT_PLACEHOLDER = '(none)'
