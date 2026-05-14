/* Curator tab — surfaces CURATOR_STATE from /api/v2/dashboard/curator
   and provides optimistic-UI cancel buttons wired to /api/v2/dashboard/curator/cancel. */
const { Sparkline: SP_CUR, StepSpark: SS_CUR, PALETTE: PAL_CUR } = window.DF_CHARTS;
const { ChipList: CL_CUR } = window.DF_TABS;
const { timeago: ta_CUR } = window.DF_SHELL;
const D_CUR = window.DF_DATA;
const { useState: uS_CUR, useEffect: uE_CUR, useCallback: uCB_CUR } = React;

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
  const { p50 = [], p90 = [], p99 = [], labels = [] } = latency_spark || {};
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
              <span style={{ color: PAL_CUR.accent }}>■</span><span>p50</span>
              <span style={{ color: PAL_CUR.accent2 }}>■</span><span>p90</span>
              <span style={{ color: PAL_CUR.fg3 }}>■</span><span>p99</span>
            </div>
            <div style={{ height: 28, position: 'relative', marginBottom: 2 }}>
              <SP_CUR values={p99} color={PAL_CUR.fg3} strokeWidth={1} area={false} />
            </div>
            <div style={{ height: 28, position: 'relative', marginBottom: 2 }}>
              <SP_CUR values={p90} color={PAL_CUR.accent2} strokeWidth={1} area={false} />
            </div>
            <div style={{ height: 28, position: 'relative' }}>
              <SP_CUR values={p50} color={PAL_CUR.accent} strokeWidth={1.5} area={true} />
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
            <SS_CUR values={values} color={PAL_CUR.bad} strokeWidth={1.5} />
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
    : ta_CUR(created_at);
  return (
    <tr>
      <td className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>{shortTicketId(ticket_id)}</td>
      <td style={{ maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis' }} title={title}>{title || '—'}</td>
      <td>
        {files && files.length > 0 ? (
          <CL_CUR
            items={files}
            renderChip={(f, i) => (
              <span key={i} className="chip" title={f} style={{ fontFamily: 'var(--mono)', fontSize: 10 }}>
                {f.split('/').pop()}
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
          onClick={() => onCancel(ticket_id, ticket)}
          title={`Cancel ticket ${ticket_id}`}
        >
          Cancel
        </button>
      </td>
    </tr>
  );
}

// ── Main CuratorTab component ──
function CuratorTab({ projectFilter }) {
  const cs = D_CUR.CURATOR_STATE || {};
  const { pending = [], latency_spark, capped_spark, state = {} } = cs;
  const { capped_now = 0, paused_reason = null, pending_total = 0 } = state;

  // Optimistic cancellation: set of ticket_ids removed from local view
  const [cancelled, setCancelled] = uS_CUR(() => new Set());
  // Toast list: [{ id, msg }]
  const [toasts, setToasts] = uS_CUR([]);

  const addToast = uCB_CUR((msg) => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, msg }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000);
  }, []);

  const handleCancel = uCB_CUR(async (ticket_id, ticket) => {
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
        addToast(`Cancel failed (${resp.status}): ${ticket_id}`);
      }
    } catch (err) {
      // Network error — revert
      setCancelled(prev => { const s = new Set(prev); s.delete(ticket_id); return s; });
      addToast(`Cancel error: ${err.message || String(err)}`);
    }
  }, [addToast]);

  // Filter by project and by local cancellation set
  const visible = pending.filter(t =>
    !cancelled.has(t.ticket_id) &&
    (projectFilter.length === 0 || projectFilter.includes(t.project_id))
  );

  const grouped = groupBy(visible, t => t.project_id || '(unknown)');

  const stateBadgeClass = capped_now === 0 ? 'ok' : 'bad';
  const stateLabel = capped_now === 0
    ? 'Open'
    : paused_reason ? `Capped: ${paused_reason}` : 'Capped';

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
