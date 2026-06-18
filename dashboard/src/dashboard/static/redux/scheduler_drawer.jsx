/* scheduler_drawer.jsx — slide-in override drawer for a selected task.
   Behavioral invariants (no JS test runner; verified manually per task spec).

   Exports: window.DF_SCHED_DRAWER = { SchedulerDrawer }
*/
const { useState: sdUseState, useCallback: sdUseCallback, useRef: sdUseRef, useEffect: sdUseEffect } = React;
const { timeago, fmtDateTime } = window.DF_SHELL;
// Shared helpers loaded by scheduler_utils.jsx (must come before this script).
const { fmtAge, totalEvents, avgWaitSeconds, disambiguateLabels } = window.DF_SCHED_UTILS;

// Derive which modules are currently blocking this task.
// A module blocks this task when: it is in the task's lock_set AND it has a
// holder.  Modules are project-scoped — a same-named path in another project
// is unrelated, so we filter to entries whose project matches the task's.
function blockingModules(task, modules) {
  const lockSet = new Set(task.lock_set || []);
  return (modules || []).filter(m =>
    (!m.project || !task.project || m.project === task.project) &&
    lockSet.has(m.path) &&
    m.holder &&
    m.holder !== task.task_id
  );
}

// ── Holder ETA section ──
function HolderEtas({ task, modules, spark }) {
  const blocked = blockingModules(task, modules);
  if (blocked.length === 0) {
    return (
      <div style={{ color: 'var(--fg-3)', fontSize: 11, fontStyle: 'italic' }}>
        No modules currently held by another task.
      </div>
    );
  }
  const elapsed = fmtAge(task.age_seconds);
  const n = totalEvents(spark);
  const p50 = n >= 10 ? avgWaitSeconds(spark) : null;
  const labelMap = disambiguateLabels ? disambiguateLabels(blocked.map(m => m.path)) : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {blocked.map(m => (
        <div key={`${m.project || ''}/${m.path}`} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 11 }}>
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <div className="mono" style={{ fontSize: 10, color: 'var(--fg-3)', overflow: 'hidden', textOverflow: 'ellipsis' }} title={m.path}>
              {labelMap ? labelMap.get(m.path) : m.path}
            </div>
            <div style={{ color: 'var(--fg-2)' }}>
              held by <span className="mono" style={{ color: 'var(--accent)' }}>T-{m.holder}</span>
            </div>
          </div>
          <div style={{ textAlign: 'right', flexShrink: 0 }}>
            <div style={{ color: 'var(--warn)', fontFamily: 'var(--mono)', fontSize: 10 }}>
              waiting {elapsed}
            </div>
            {p50 != null && (
              <div style={{ color: 'var(--fg-3)', fontSize: 10 }}>
                avg wait ~{fmtAge(p50)}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Waiters-behind-me section ──
// Other tasks whose lock_set overlaps with this task's lock_set.
function WaitersList({ task, allRows }) {
  const lockSet = new Set(task.lock_set || []);
  // Scope to the task's own project — competing rows from a different
  // project share no lock domain even if their lock_set paths coincide.
  // The (task_id, project) pair is the project-qualified identity, so we
  // exclude only the exact same task, not numeric-id collisions across
  // projects.
  const others = (allRows || []).filter(r =>
    !(r.task_id === task.task_id && r.project === task.project) &&
    (!task.project || !r.project || r.project === task.project) &&
    (r.lock_set || []).some(p => lockSet.has(p))
  );
  if (others.length === 0) {
    return (
      <div style={{ color: 'var(--fg-3)', fontSize: 11, fontStyle: 'italic' }}>
        No other tasks competing for the same modules.
      </div>
    );
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {others.slice(0, 8).map(r => {
        const shared = (r.lock_set || []).filter(p => lockSet.has(p));
        return (
          <div key={`${r.project || ''}/${r.task_id}`} style={{ fontSize: 11, display: 'flex', gap: 8, alignItems: 'baseline' }}>
            <span className="mono" style={{ fontSize: 10, color: 'var(--accent)', flexShrink: 0 }}>T-{r.task_id}</span>
            <span style={{ color: 'var(--fg-2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={r.title}>
              {r.title || '—'}
            </span>
            <span style={{ color: 'var(--fg-3)', fontSize: 10, flexShrink: 0 }}>
              ({shared.length} shared)
            </span>
          </div>
        );
      })}
      {others.length > 8 && (
        <div style={{ color: 'var(--fg-3)', fontSize: 10 }}>+{others.length - 8} more…</div>
      )}
    </div>
  );
}

// ── Actions section ──
// Three buttons: Boost, Pin, Reserve-Now. Optional TTL field for pin.
function ActionsPanel({ task, onSubmitOverride, onClose }) {
  const [ttl, setTtl] = sdUseState('');
  const [submitting, setSubmitting] = sdUseState(false);

  const submit = sdUseCallback(async (overrideType) => {
    setSubmitting(true);
    const body = { task_id: task.task_id, project_root: task.project_root || '' };
    if (overrideType === 'boost') {
      body.boost_tier = 'high';
    } else if (overrideType === 'pin') {
      body.pinned = true;
      if (ttl && !isNaN(Number(ttl)) && Number(ttl) > 0) {
        body.ttl_minutes = Number(ttl);
      }
    } else if (overrideType === 'reserve') {
      body.reserve_now = true;
    }
    try {
      await onSubmitOverride(body);
    } finally {
      setSubmitting(false);
    }
  }, [task, ttl, onSubmitOverride]);

  const btnStyle = {
    padding: '6px 14px', borderRadius: 4, fontSize: 12,
    border: '1px solid', cursor: submitting ? 'not-allowed' : 'pointer',
    opacity: submitting ? 0.6 : 1,
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button
          style={{ ...btnStyle, borderColor: 'var(--warn)', color: 'var(--warn)' }}
          onClick={() => submit('boost')}
          disabled={submitting}
          title="Raise effective priority to high"
        >
          ↑ Boost
        </button>
        <button
          style={{ ...btnStyle, borderColor: 'var(--ok)', color: 'var(--ok)' }}
          onClick={() => submit('pin')}
          disabled={submitting}
          title="Pin to top of scheduler queue"
        >
          📌 Pin
        </button>
        <button
          style={{ ...btnStyle, borderColor: 'var(--bad)', color: 'var(--bad)' }}
          onClick={() => submit('reserve')}
          disabled={submitting}
          title="Reserve-now: acquire slot immediately"
        >
          ⚡ Reserve-Now
        </button>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
        <label style={{ color: 'var(--fg-3)' }}>TTL (minutes, optional)</label>
        <input
          type="number"
          min="1"
          max="1440"
          placeholder="∞"
          value={ttl}
          onChange={e => setTtl(e.target.value)}
          style={{
            width: 70, padding: '3px 6px', borderRadius: 3,
            background: 'var(--bg-2)', border: '1px solid var(--line)',
            color: 'var(--fg-0)', fontSize: 11,
          }}
        />
        <span style={{ color: 'var(--fg-3)', fontSize: 10 }}>applies to Pin</span>
      </div>
    </div>
  );
}

// ── Audit history strip ──
// Renders recent override events for this task from eventsForTask.
function AuditHistory({ spark }) {
  if (!spark || !spark.labels || spark.labels.length === 0) {
    return (
      <div style={{ color: 'var(--fg-3)', fontSize: 11, fontStyle: 'italic' }}>
        No skip history in the last hour.
      </div>
    );
  }
  const total = totalEvents(spark);
  if (total === 0) {
    return (
      <div style={{ color: 'var(--fg-3)', fontSize: 11, fontStyle: 'italic' }}>
        Zero skips recorded in the last hour.
      </div>
    );
  }

  // Show bucketed sparkline as a small bar chart
  const maxVal = Math.max(...spark.values, 1);
  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 6 }}>
        {total} skip{total !== 1 ? 's' : ''} in the last hour
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 32 }}>
        {spark.values.map((v, i) => (
          <div
            key={i}
            title={`${spark.labels[i] ? fmtDateTime(spark.labels[i]) : i}: ${v} skips`}
            style={{
              flex: 1,
              height: `${Math.round((v / maxVal) * 100)}%`,
              minHeight: v > 0 ? 2 : 0,
              background: v > 0 ? 'var(--warn)' : 'var(--bg-3)',
              borderRadius: 1,
              transition: 'height 0.15s',
            }}
          />
        ))}
      </div>
    </div>
  );
}

// ── Main SchedulerDrawer component ──
//
// Props:
//   task            dict    — a task row from SCHEDULER.rows (the selected task)
//   modules         list    — SCHEDULER.modules (all contention modules)
//   eventsForTask   dict    — {labels, values} sparkline (or null)
//   allRows         list    — SCHEDULER.rows (all tasks, for waiter computation)
//   onClose         fn()    — close the drawer
//   onSubmitOverride fn(body) — called with POST body; returns a Promise
function SchedulerDrawer({ task, modules, eventsForTask, allRows, onClose, onSubmitOverride }) {
  if (!task) return null;

  const spark = eventsForTask || { labels: [], values: [] };

  return (
    <div className="sched-drawer" role="dialog" aria-label={`Override controls for T-${task.task_id}`}>
      {/* Header */}
      <div className="sched-drawer-head">
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span className="mono" style={{ fontSize: 11, color: 'var(--fg-3)' }}>T-{task.task_id}</span>
            {task.priority_differs && (
              <span className="badge warn" style={{ fontSize: 9, padding: '1px 5px' }}>
                {task.effective_priority}
              </span>
            )}
            {task.pinned && <span className="badge ok" style={{ fontSize: 9, padding: '1px 5px' }}>pinned</span>}
            {task.reserve_now && <span className="badge bad" style={{ fontSize: 9, padding: '1px 5px' }}>rsv-now</span>}
          </div>
          <div style={{ fontSize: 13, color: 'var(--fg-0)', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={task.title}>
            {task.title || '—'}
          </div>
          <div style={{ display: 'flex', gap: 12, marginTop: 4, fontSize: 10, color: 'var(--fg-3)' }}>
            <span>age: <span className="mono">{fmtAge(task.age_seconds)}</span></span>
            <span>skipped: <span className="mono">{task.skip_count || 0}</span></span>
            {task.declared_priority && (
              <span>priority: <span className="mono">{task.declared_priority}</span></span>
            )}
          </div>
        </div>
        <button
          className="sched-drawer-close"
          onClick={onClose}
          title="Close"
          aria-label="Close drawer"
        >
          ×
        </button>
      </div>

      {/* Body */}
      <div className="sched-drawer-body">

        {/* Holder ETAs */}
        <div className="sched-drawer-section">
          <div className="sched-drawer-section-lbl">Lock holders</div>
          <HolderEtas task={task} modules={modules} spark={spark} />
        </div>

        {/* Waiters behind me */}
        <div className="sched-drawer-section">
          <div className="sched-drawer-section-lbl">Competing tasks</div>
          <WaitersList task={task} allRows={allRows} />
        </div>

        {/* Actions */}
        <div className="sched-drawer-section">
          <div className="sched-drawer-section-lbl">Override actions</div>
          <ActionsPanel task={task} onSubmitOverride={onSubmitOverride} onClose={onClose} />
        </div>

        {/* Audit history */}
        <div className="sched-drawer-section">
          <div className="sched-drawer-section-lbl">Skip history (1h)</div>
          <AuditHistory spark={spark} />
        </div>
      </div>
    </div>
  );
}

window.DF_SCHED_DRAWER = { SchedulerDrawer };
