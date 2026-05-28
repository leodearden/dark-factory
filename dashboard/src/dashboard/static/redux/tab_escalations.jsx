/* Escalations tab — read-only queue view.
 *
 * No JS test runner in this project (see scheduler_drawer.jsx comment).
 * Wiring contracts are verified via Python source-assertion tests in
 * dashboard/tests/test_tab_escalations.py.
 *
 * Load order: tabs.jsx → tab_escalations.jsx → app.jsx
 * Export:     window.DF_TABS.EscalationsTab  (additive mutation of the object
 *             created by tabs.jsx; app.jsx destructures it last)
 */
const { useState: uS, useEffect: uE } = React;
const { ProjectGroup, taskId } = window.DF_SHELL;
const DF = window.DF_DATA;

// ── Local helpers (tabs.jsx-compatible copies; not exported from any namespace) ──

function useOpenSet(ids, defaultOpen = true, storageKey = null) {
  const [openMap, setOpenMap] = uS(() => {
    let stored = {};
    if (storageKey) {
      try { stored = JSON.parse(localStorage.getItem(storageKey) || '{}') || {}; } catch (e) {}
    }
    const init = {};
    for (const id of ids) init[id] = id in stored ? !!stored[id] : defaultOpen;
    return init;
  });
  // Backfill ids that arrive after mount (ESCALATIONS.subsections starts [] and
  // is populated by the first poll, so groups would otherwise render collapsed).
  const idsKey = ids.join('\0');
  uE(() => {
    setOpenMap(m => {
      let patch = null;
      for (const id of ids) {
        if (!(id in m)) { if (!patch) patch = {}; patch[id] = defaultOpen; }
      }
      return patch ? { ...m, ...patch } : m;
    });
  }, [idsKey]); // eslint-disable-line react-hooks/exhaustive-deps
  uE(() => {
    if (storageKey) {
      try { localStorage.setItem(storageKey, JSON.stringify(openMap)); } catch (e) {}
    }
  }, [storageKey, openMap]);
  const toggle = id => setOpenMap(m => ({ ...m, [id]: !m[id] }));
  const setAll = v => setOpenMap(Object.fromEntries(ids.map(id => [id, v])));
  return [openMap, toggle, setAll];
}

function usePersistedState(storageKey, defaultValue) {
  const [v, setV] = uS(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      return raw === null ? defaultValue : JSON.parse(raw);
    } catch (e) { return defaultValue; }
  });
  uE(() => { try { localStorage.setItem(storageKey, JSON.stringify(v)); } catch (e) {} }, [storageKey, v]);
  return [v, setV];
}

function GroupAllToggle({ allOpen, onSetAll }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 6 }}>
      <button className="seg" onClick={() => onSetAll(!allOpen)}
        style={{ cursor: 'pointer', padding: '4px 10px', fontSize: 11, color: 'var(--fg-2)' }}>
        {allOpen ? '⌃ collapse all' : '⌄ expand all'}
      </button>
    </div>
  );
}

// ── Level / severity badge helpers ──

function levelClass(level) {
  if (level === 0) return 'esc-level-0';
  if (level === 1) return 'esc-level-1';
  return 'esc-level-2';
}

function sevClass(sev) {
  if (!sev) return '';
  const s = String(sev).toLowerCase();
  if (s === 'critical' || s === 'high') return 'esc-sev-high';
  if (s === 'medium') return 'esc-sev-medium';
  return 'esc-sev-low';
}

// ── EscalationsTab ──

function EscalationsTab({ projectFilter }) {
  const escalations = DF.ESCALATIONS || { subsections: [], summary: { by_level: {}, by_status: {} } };

  // Filter subsections by project when projectFilter is active.
  // Orchestrator subsections: filter by subsection label (label == project name).
  // Reconciliation subsections: always include — their rows are filtered per-row
  // by row.project below, because the subsection label is 'fused-memory' (not a
  // project name) and rows may belong to different owning projects.
  const subsections = escalations.subsections.filter(s => {
    if (!projectFilter || projectFilter.length === 0) return true;
    if (s.kind === 'reconciliation') return true;
    return projectFilter.includes(s.label);
  });
  const subsectionIds = subsections.map(s => s.id);

  const [openMap, toggle, setAll] = useOpenSet(subsectionIds, true, 'df.open.esc');
  const allOpen = subsectionIds.length > 0 && subsectionIds.every(id => openMap[id]);

  // Sort state (key: 'task' | 'timestamp', dir: 'asc' | 'desc')
  const [sort, setSort] = usePersistedState('df.esc.sort', { key: 'task', dir: 'asc' });
  const flipDir = () => setSort(s => ({ ...s, dir: s.dir === 'asc' ? 'desc' : 'asc' }));

  // Filter state (levels 0/1/2, statuses pending/resolved/dismissed)
  const [filter, setFilter] = usePersistedState('df.esc.filter', {
    levels: { 0: true, 1: true, 2: true },
    statuses: { pending: true, resolved: false, dismissed: false },
  });
  const toggleLevel = lv => setFilter(f => ({ ...f, levels: { ...f.levels, [lv]: !f.levels[lv] } }));
  const toggleStatus = st => setFilter(f => ({ ...f, statuses: { ...f.statuses, [st]: !f.statuses[st] } }));

  function matchesFilter(row) {
    const lvKey = row.level != null ? row.level : 0;
    const stKey = row.status || 'pending';
    return !!(filter.levels[lvKey] && filter.statuses[stKey]);
  }

  function sortRows(rows) {
    const mul = sort.dir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      // Primary: numeric task_id (NaN/null sorts last regardless of direction)
      const aId = Number(a.task_id);
      const bId = Number(b.task_id);
      const aValid = !isNaN(aId);
      const bValid = !isNaN(bId);
      if (!aValid && !bValid) {
        // Both invalid: secondary tie-break by timestamp, respecting sort direction
        const ts = (a.timestamp || '') < (b.timestamp || '') ? -1 : (a.timestamp || '') > (b.timestamp || '') ? 1 : 0;
        return mul * ts;
      }
      if (!aValid) return 1;  // invalid always last, regardless of direction
      if (!bValid) return -1; // invalid always last, regardless of direction
      if (aId !== bId) return mul * (aId - bId);
      // Tie-break by timestamp, respecting sort direction
      const ts = (a.timestamp || '') < (b.timestamp || '') ? -1 : (a.timestamp || '') > (b.timestamp || '') ? 1 : 0;
      return mul * ts;
    });
  }

  // Selected row for sidebar
  const [selected, setSelected] = uS(null);

  // Global summary from top-level data
  const gs = escalations.summary || {};
  const byLevel = gs.by_level || {};
  const byStatus = gs.by_status || {};

  return (
    <div style={{ position: 'relative' }}>
      {/* Controls header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
        {/* Level filter chips */}
        <span style={{ fontSize: 11, color: 'var(--fg-3)' }}>Level:</span>
        {[0, 1, 2].map(lv => (
          <button key={lv} className={`chip${filter.levels[lv] ? ' on' : ''}`}
            onClick={() => toggleLevel(lv)}
            title={`Level ${lv}`}>
            L{lv}
          </button>
        ))}
        <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--fg-3)' }}>Status:</span>
        {['pending', 'resolved', 'dismissed'].map(st => (
          <button key={st} className={`chip${filter.statuses[st] ? ' on' : ''}`}
            onClick={() => toggleStatus(st)}
            title={st}>
            {st}
          </button>
        ))}
        {/* Sort toggle */}
        <button className="seg" onClick={flipDir}
          style={{ marginLeft: 8, cursor: 'pointer', padding: '4px 10px', fontSize: 11, color: 'var(--fg-2)' }}>
          task {sort.dir === 'asc' ? '↑' : '↓'}
        </button>
        {/* Summary pills */}
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--fg-3)' }}>
          {byStatus.pending || 0} pending · {byLevel[1] || 0} L1 · {byLevel[2] || 0} L2
        </span>
      </div>

      {/* Expand / collapse all */}
      <GroupAllToggle allOpen={allOpen} onSetAll={setAll} />

      {/* Subsection groups */}
      {subsections.map(sec => {
        const secSummary = sec.summary || {};
        const secByLevel = secSummary.by_level || {};
        const secByStatus = secSummary.by_status || {};
        const filteredRows = sortRows((sec.escalations || []).filter(row => {
          if (!matchesFilter(row)) return false;
          // Reconciliation subsections: filter by row.project (not subsection label).
          if (sec.kind === 'reconciliation' && projectFilter && projectFilter.length > 0) {
            return !row.project || projectFilter.includes(row.project);
          }
          return true;
        }));

        const summary = (
          <>
            <span className="pip" style={{ fontSize: 10 }}>{secByStatus.pending || 0} pending</span>
            {(secByLevel[1] || 0) > 0 && (
              <span className="pip"><span className="badge warn" style={{ fontSize: 9 }}>L1 · {secByLevel[1]}</span></span>
            )}
            {(secByLevel[2] || 0) > 0 && (
              <span className="pip"><span className="badge bad" style={{ fontSize: 9 }}>L2 · {secByLevel[2]}</span></span>
            )}
            <span className="mono" style={{ color: 'var(--fg-3)', fontSize: 10 }}>{sec.kind}</span>
          </>
        );

        return (
          <div key={sec.id} style={{ marginBottom: 8 }}>
            <ProjectGroup
              id={sec.id}
              label={sec.label}
              open={!!openMap[sec.id]}
              onToggle={() => toggle(sec.id)}
              summary={summary}
            >
              {filteredRows.length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--fg-3)', padding: '8px 0' }}>
                  No escalations match current filters.
                </div>
              ) : (
                <table className="tbl" style={{ tableLayout: 'fixed', width: '100%' }}>
                  <colgroup>
                    <col style={{ width: 70 }} />
                    <col style={{ width: 60 }} />
                    <col style={{ width: 80 }} />
                    <col style={{ width: 80 }} />
                    <col />
                    <col style={{ width: 100 }} />
                  </colgroup>
                  <thead>
                    <tr>
                      <th style={{ cursor: 'pointer', userSelect: 'none' }} onClick={flipDir}>
                        Task {sort.dir === 'asc' ? '↑' : '↓'}
                      </th>
                      <th>Level</th>
                      <th>Status</th>
                      <th>Severity</th>
                      <th>Summary</th>
                      <th>Project</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRows.map(row => (
                      <tr key={row.id} style={{ cursor: 'pointer' }}
                        onClick={() => setSelected(row)}>
                        <td className="mono">{row.task_id ? `T-${taskId(String(row.task_id))}` : '—'}</td>
                        <td>
                          <span className={`badge ${levelClass(row.level)}`}
                            style={{ fontSize: 10 }}>
                            L{row.level ?? 0}
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${row.status === 'pending' ? 'warn' : row.status === 'resolved' ? 'ok' : ''}`}
                            style={{ fontSize: 10 }}>
                            {row.status || '—'}
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${sevClass(row.severity)}`}
                            style={{ fontSize: 10 }}>
                            {row.severity || '—'}
                          </span>
                        </td>
                        <td style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={row.summary}>
                          {row.summary || '—'}
                        </td>
                        <td style={{ fontSize: 11, color: 'var(--fg-3)' }}>
                          {row.project || '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </ProjectGroup>
          </div>
        );
      })}

      {/* Detail sidebar */}
      {selected && (
        <EscalationSidebar row={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

// ── EscalationSidebar — read-only detail panel ──

function EscalationSidebar({ row, onClose }) {
  if (!row) return null;
  const task = row.task || null;
  return (
    <div className="sched-drawer" role="dialog" aria-label={`Escalation detail for ${row.id}`}>
      {/* Header */}
      <div className="sched-drawer-head">
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span className={`badge ${levelClass(row.level)}`} style={{ fontSize: 10 }}>L{row.level ?? 0}</span>
            <span className={`badge ${row.status === 'pending' ? 'warn' : row.status === 'resolved' ? 'ok' : ''}`}
              style={{ fontSize: 10 }}>{row.status || '—'}</span>
            <span className={`badge ${sevClass(row.severity)}`} style={{ fontSize: 10 }}>{row.severity || '—'}</span>
          </div>
          <div style={{ fontSize: 13, color: 'var(--fg-0)', fontWeight: 500,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            title={row.summary}>
            {row.summary || '—'}
          </div>
          <div style={{ display: 'flex', gap: 12, marginTop: 4, fontSize: 10, color: 'var(--fg-3)' }}>
            {row.category && <span>category: <span className="mono">{row.category}</span></span>}
            {row.agent_role && <span>agent: <span className="mono">{row.agent_role}</span></span>}
            {row.worktree && <span>worktree: <span className="mono">{row.worktree}</span></span>}
          </div>
        </div>
        <button className="sched-drawer-close" onClick={onClose} title="Close" aria-label="Close drawer">×</button>
      </div>

      {/* Body */}
      <div className="sched-drawer-body">
        {/* Detail */}
        {row.detail && (
          <div className="sched-drawer-section">
            <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Detail</div>
            <div style={{ fontSize: 12, color: 'var(--fg-1)', whiteSpace: 'pre-wrap' }}>{row.detail}</div>
          </div>
        )}

        {/* Suggested action */}
        {row.suggested_action && (
          <div className="sched-drawer-section">
            <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Suggested Action</div>
            <div style={{ fontSize: 12, color: 'var(--fg-1)' }}>{row.suggested_action}</div>
          </div>
        )}

        {/* Resolution */}
        {row.resolution && (
          <div className="sched-drawer-section">
            <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Resolution</div>
            <div style={{ fontSize: 12, color: 'var(--fg-1)' }}>{row.resolution}</div>
          </div>
        )}

        {/* Workflow state */}
        {row.workflow_state && (
          <div className="sched-drawer-section">
            <div style={{ display: 'flex', gap: 16, fontSize: 11 }}>
              <span style={{ color: 'var(--fg-3)' }}>Workflow:</span>
              <span className="mono">{row.workflow_state}</span>
            </div>
          </div>
        )}

        {/* Project */}
        {row.project && (
          <div className="sched-drawer-section">
            <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 2 }}>Project</div>
            <div style={{ fontSize: 12, color: 'var(--fg-1)' }}>{row.project}</div>
          </div>
        )}

        {/* Linked task */}
        <div className="sched-drawer-section">
          <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Linked Task</div>
          {row.task_id && row.task_unresolved ? (
            // task_id present but could not be resolved — distinct from "no task linked"
            <div style={{ fontSize: 12, color: 'var(--fg-3)' }}>
              Task ID <span className="mono">{row.task_id}</span> could not be resolved
              {row.worktree && <span> (worktree: <span className="mono">{row.worktree}</span>)</span>}.
            </div>
          ) : task ? (
            <div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
                <span className="mono" style={{ fontSize: 11, color: 'var(--fg-3)' }}>T-{taskId(String(task.id || ''))}</span>
                <span className={`badge ${task.status === 'in-progress' ? 'ok' : task.status === 'blocked' ? 'bad' : ''}`}
                  style={{ fontSize: 10 }}>{task.status || '—'}</span>
              </div>
              <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--fg-0)', marginBottom: 4 }}>{task.title || '—'}</div>
              {task.description && (
                <div style={{ fontSize: 11, color: 'var(--fg-2)', whiteSpace: 'pre-wrap', maxHeight: 120, overflow: 'auto' }}>
                  {task.description}
                </div>
              )}
            </div>
          ) : (
            <div style={{ fontSize: 12, color: 'var(--fg-3)' }}>No linked task.</div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Additive export (mutates the object created by tabs.jsx) ──
window.DF_TABS.EscalationsTab = EscalationsTab;
