/* Curator tab — surfaces CURATOR_STATE from /api/v2/dashboard/curator
   and provides optimistic-UI cancel buttons wired to /api/v2/dashboard/curator/cancel. */
const { Sparkline, StepSpark, PALETTE } = window.DF_CHARTS;
const { ChipList } = window.DF_TABS;
const { timeago } = window.DF_SHELL;
const D = window.DF_DATA;
const { useState, useEffect, useCallback, useRef } = React;

// ── Helpers ──

// Derive a short display id from a ticket_id (tkt_<random>).
// Returns first 8 chars of the random tail, or the raw id if no prefix.
function shortTicketId(ticket_id) {
  if (!ticket_id) return '—';
  const tail = ticket_id.replace(/^tkt_/, '');
  return tail.slice(0, 8);
}

// Group an array of items by a key function, preserving insertion order.
function groupBy(arr, keyFn) {
  const map = new Map();
  for (const item of arr) {
    const k = keyFn(item);
    if (!map.has(k)) map.set(k, []);
    map.get(k).push(item);
  }
  return map;
}

// ── Latency panel — three stacked Sparklines for p50/p90/p99 ──
function LatencyPanel({ latency_spark }) {
  // backend seed also includes a `labels` key; no chart consumer reads it so it is omitted.
  const { p50 = [], p90 = [], p99 = [] } = latency_spark || {};
  const hasData = p50.length > 0 || p90.length > 0 || p99.length > 0;
  return (
    <div className="panel" style={{ flex: 1, minWidth: 0 }}>
      <div className="panel-head">
        <span className="title">Queue latency (ms)</span>
      </div>
      <div className="panel-body tight">
        {!hasData ? (
          <span style={{ color: 'var(--fg-3)', fontSize: 11 }}>No latency data yet</span>
        ) : (
          <>
            <div style={{ display: 'flex', gap: 8, marginBottom: 4, fontSize: 10, color: 'var(--fg-3)' }}>
              <span style={{ color: PALETTE.accent }}>■</span><span>p50</span>
              <span style={{ color: PALETTE.accent2 }}>■</span><span>p90</span>
              <span style={{ color: PALETTE.fg3 }}>■</span><span>p99</span>
            </div>
            <div style={{ height: 28, position: 'relative', marginBottom: 2 }}>
              <Sparkline values={p99} color={PALETTE.fg3} strokeWidth={1} area={false} />
            </div>
            <div style={{ height: 28, position: 'relative', marginBottom: 2 }}>
              <Sparkline values={p90} color={PALETTE.accent2} strokeWidth={1} area={false} />
            </div>
            <div style={{ height: 28, position: 'relative' }}>
              <Sparkline values={p50} color={PALETTE.accent} strokeWidth={1.5} area={true} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Capped panel — step sparkline for capped_spark ──
function CappedPanel({ capped_spark }) {
  const { values = [] } = capped_spark || {};
  return (
    <div className="panel" style={{ flex: 1, minWidth: 0 }}>
      <div className="panel-head">
        <span className="title">Capped events</span>
      </div>
      <div className="panel-body tight">
        {values.length === 0 ? (
          <span style={{ color: 'var(--fg-3)', fontSize: 11 }}>No capped events recorded</span>
        ) : (
          <div style={{ height: 56, position: 'relative' }}>
            <StepSpark values={values} color={PALETTE.bad} strokeWidth={1.5} />
          </div>
        )}
      </div>
    </div>
  );
}

// ── Pending panel — smooth sparkline for pending_spark ──
function PendingPanel({ pending_spark }) {
  const { values = [] } = pending_spark || {};
  return (
    <div className="panel" style={{ flex: 1, minWidth: 0 }}>
      <div className="panel-head">
        <span className="title">Pending over time</span>
      </div>
      <div className="panel-body tight">
        {values.length === 0 ? (
          <span style={{ color: 'var(--fg-3)', fontSize: 11 }}>No pending data yet</span>
        ) : (
          <div style={{ height: 56, position: 'relative' }}>
            <Sparkline values={values} color={PALETTE.accent} strokeWidth={1.5} area={true} />
          </div>
        )}
      </div>
    </div>
  );
}

// ── Queue table row ──
function QueueRow({ ticket, onCancel }) {
  const { ticket_id, title, files, project_id, age_seconds, created_at } = ticket;
  const age = age_seconds != null
    ? (age_seconds < 60 ? `${age_seconds}s` : age_seconds < 3600 ? `${Math.round(age_seconds / 60)}m` : `${Math.round(age_seconds / 3600)}h`)
    : timeago(created_at);
  // Lifted out of the render body: compute once per files-identity change, not per render.
  // React.useMemo avoids the O(n^2 * segments) scan on re-renders unrelated to file changes.
  const labelMap = React.useMemo(
    () => (window.DF_SCHED_UTILS || {}).disambiguateLabels
      ? window.DF_SCHED_UTILS.disambiguateLabels(files || [])
      : null,
    [files]
  );
  return (
    <tr>
      <td className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>{shortTicketId(ticket_id)}</td>
      <td style={{ maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis' }} title={title}>{title || '—'}</td>
      <td>
        {files && files.length > 0 ? (
          <ChipList
            items={files}
            renderChip={(f, i) => (
              <span key={i} className="chip" title={f} style={{ fontFamily: 'var(--mono)', fontSize: 10 }}>
                {labelMap ? labelMap.get(f) : f}
              </span>
            )}
            maxInline={2}
            persistKey={`df.curator.files.${ticket_id}`}
          />
        ) : (
          <span style={{ color: 'var(--fg-3)' }}>—</span>
        )}
      </td>
      <td className="mono" style={{ color: 'var(--fg-3)', fontSize: 11 }}>{age}</td>
      <td>
        <button
          className="badge bad"
          style={{ cursor: 'pointer', fontSize: 10, padding: '2px 8px', border: '1px solid' }}
          onClick={() => onCancel(ticket_id, title)}
          title={`Cancel ticket ${ticket_id}`}
        >
          Cancel
        </button>
      </td>
    </tr>
  );
}

// ── Main CuratorTab component ──
function CuratorTab({ projectFilter = [] }) {
  const cs = D.CURATOR_STATE || {};
  const { pending = [], latency_spark, pending_spark, capped_spark, state = {} } = cs;
  const { capped_now = 0, paused_reason = null, pending_total = 0, accounts_summary = { total: 0, capped: 0, available: 0, capped_accounts: [] } } = state;

  // Optimistic cancellation: set of ticket_ids removed from local view
  const [cancelled, setCancelled] = useState(() => new Set());
  // Toast list: [{ id, msg }]
  const [toasts, setToasts] = useState([]);
  // Track toast timers so we can clear them on unmount to avoid setState-after-unmount warnings
  const toastTimers = useRef([]);

  // Clear all pending toast timers on unmount
  useEffect(() => {
    return () => {
      toastTimers.current.forEach(clearTimeout);
    };
  }, []);

  // Prune stale cancelled IDs whenever pending changes (each 3 s poll).
  // Prevents unbounded Set growth and handles unlikely ticket_id reuse.
  useEffect(() => {
    if (cancelled.size === 0) return;
    const pendingIds = new Set(pending.map(t => t.ticket_id));
    setCancelled(prev => {
      const pruned = new Set([...prev].filter(id => pendingIds.has(id)));
      return pruned.size === prev.size ? prev : pruned;
    });
  }, [pending]); // eslint-disable-line react-hooks/exhaustive-deps

  const addToast = useCallback((msg) => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, msg }]);
    const timer = setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
      const idx = toastTimers.current.indexOf(timer);
      if (idx >= 0) toastTimers.current.splice(idx, 1);
    }, 5000);
    toastTimers.current.push(timer);
  }, []);

  // title arg is used in failure toasts so operators know which ticket failed.
  const handleCancel = useCallback(async (ticket_id, title) => {
    // Optimistically hide the row
    setCancelled(prev => new Set([...prev, ticket_id]));
    try {
      const resp = await fetch('/api/v2/dashboard/curator/cancel', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ ticket_id }),
      });
      if (!resp.ok) {
        // Revert: restore the row
        setCancelled(prev => { const s = new Set(prev); s.delete(ticket_id); return s; });
        const label = title ? `"${title}"` : ticket_id;
        addToast(`Cancel failed (${resp.status}): ${label}`);
      }
    } catch (err) {
      // Network error — revert
      setCancelled(prev => { const s = new Set(prev); s.delete(ticket_id); return s; });
      const label = title ? `"${title}"` : ticket_id;
      addToast(`Cancel error: ${label} — ${err.message || String(err)}`);
    }
  }, [addToast]);

  // Filter by project and by local cancellation set
  const visible = pending.filter(t =>
    !cancelled.has(t.ticket_id) &&
    (projectFilter.length === 0 || projectFilter.includes(t.project_id))
  );

  const grouped = groupBy(visible, t => t.project_id || '(unknown)');

  const stateBadgeClass = capped_now === 0 ? 'ok' : 'bad';
  let stateLabel;
  if (capped_now === 1) {
    stateLabel = paused_reason ? `Capped: ${paused_reason}` : 'Capped';
  } else if (accounts_summary.total > 0) {
    // Intentional: show N/M even when capped===0 (all healthy). "4/4 available" is
    // more informative than a plain "Open" badge and makes the denominator visible.
    stateLabel = `${accounts_summary.available}/${accounts_summary.total} available`;
  } else {
    stateLabel = 'Open';
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* State pill + summary */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span className={`badge ${stateBadgeClass}`} style={{ fontSize: 12, padding: '3px 10px' }}>
          {stateLabel}
        </span>
        <span style={{ color: 'var(--fg-3)', fontSize: 11 }}>
          {pending_total} ticket{pending_total !== 1 ? 's' : ''} pending
        </span>
      </div>

      {/* Spark panels row */}
      <div style={{ display: 'flex', gap: 12 }}>
        <LatencyPanel latency_spark={latency_spark} />
        <PendingPanel pending_spark={pending_spark} />
        <CappedPanel capped_spark={capped_spark} />
      </div>

      {/* Queue table */}
      <div className="panel">
        <div className="panel-head">
          <span className="title">Pending queue</span>
          <span className="meta">{visible.length} shown</span>
        </div>
        <div className="panel-body flush">
          {visible.length === 0 ? (
            <div style={{ padding: '20px 14px', color: 'var(--fg-3)', fontSize: 12 }}>
              Queue empty
            </div>
          ) : (
            <table className="tbl" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Title</th>
                  <th>Files</th>
                  <th>Age</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {Array.from(grouped.entries()).map(([project, tickets]) => (
                  <React.Fragment key={project}>
                    <tr>
                      <td colSpan={5} style={{ background: 'var(--bg-2)', color: 'var(--fg-3)', fontSize: 10, padding: '4px 10px', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                        {project}
                      </td>
                    </tr>
                    {tickets.map(ticket => (
                      <QueueRow key={ticket.ticket_id} ticket={ticket} onCancel={handleCancel} />
                    ))}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Toast container */}
      {toasts.length > 0 && (
        <div className="curator-toasts">
          {toasts.map(t => (
            <div key={t.id} className="curator-toast">
              {t.msg}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

window.DF_CURATOR = { CuratorTab };
