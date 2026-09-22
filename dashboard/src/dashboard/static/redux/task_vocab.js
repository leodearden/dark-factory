// GENERATED FILE — DO NOT EDIT BY HAND.
//
// Rendered by scripts/gen_dashboard_task_vocab.py from the Python vocabulary:
// shared/src/shared/task_statuses.py (members) and
// dashboard/src/dashboard/data/census.py (views, tones).
//
// Regenerate with:
//   python3 scripts/gen_dashboard_task_vocab.py \
//       --output dashboard/src/dashboard/static/redux/task_vocab.js
//
// tests/scripts/test_dashboard_task_vocab.py fails if this file and that
// vocabulary have drifted.

const TASK_VOCAB_API = {
  "MEMBERS": [
    "pending",
    "in-progress",
    "blocked",
    "deferred",
    "review",
    "merge-deferred",
    "infra-hold",
    "done",
    "cancelled"
  ],
  "VIEWS": {
    "in_flight": [
      "blocked",
      "in-progress",
      "infra-hold",
      "merge-deferred",
      "review"
    ],
    "backlog": [
      "deferred",
      "pending"
    ],
    "terminal": [
      "cancelled",
      "done"
    ]
  },
  "SUB_VIEWS": {
    "running": [
      "in-progress"
    ]
  },
  "TONES": {
    "pending": "warn",
    "in-progress": "accent",
    "blocked": "bad",
    "deferred": "fg3",
    "review": "info",
    "merge-deferred": "warn",
    "infra-hold": "stranded",
    "done": "ok",
    "cancelled": "fg3"
  }
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TASK_VOCAB_API
}
if (typeof window !== 'undefined') {
  window.DF_TASK_VOCAB = TASK_VOCAB_API
}
