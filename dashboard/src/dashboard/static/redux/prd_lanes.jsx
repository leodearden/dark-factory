// prd_lanes.jsx — the Tasks tab's PRD lanes page (plans/dashboard-prd-lanes-prd.md
// §4.11): one project's served PRD view as lanes of rows, its dark matter and
// footer, and its contention line. Every decision is prd_lanes_view.js's; every
// figure is one reading of the parent view Datum (ViewFigure), titled with the
// rule the server serves for it.
//
// A LOWER LAYER than tab_tasks.jsx: it loads before it and never reads it.
// TasksTab passes its task graph in as renderMembers(rows).

const { DatumReading: DatumReading_PL, timeago: timeago_PL } = window.DF_SHELL;
const { PALETTE: CP_PL } = window.DF_CHARTS;
const {
  readyCell,
  barSegments,
  doneOfTotal,
  signalCell,
  contentionEntryText,
  heldShareText,
  neverStartedText,
  countsText,
  darkMatterPage,
  membersDrawn,
  inputPip,
} = window.DF_PRD_LANES_VIEW;
const { useState: uS_PL } = React;

function ViewFigure({ view, figure, format, detail }) {
  const rule = view.value.rules[figure];
  return <DatumReading_PL datum={view} format={format} rule={detail ? `${detail}\n\n${rule}` : rule} />;
}

function ProjectPrdLanes({ project, view, pool, renderMembers, onSelect }) {
  if (view.state === 'unknown') return <div className="empty prd-unknown">{view.reason}</div>;
  const { prds, lanes, segment_legend: legend } = view.value;
  const byKey = new Map(prds.map(row => [row.key, row]));
  const rowProps = { view, pool, project, renderMembers };
  return (
    <div className="prd-lanes">
      <SegmentLegend legend={legend} />
      {lanes.map(entry => (
        <LaneSection key={entry.lane} entry={entry} rows={entry.prd_keys.map(key => byKey.get(key))} view={view} rowProps={rowProps} />
      ))}
      <DarkMatterStrip view={view} pool={pool} project={project} onSelect={onSelect} />
      <ProjectFooter view={view} />
    </div>
  );
}

function SegmentLegend({ legend }) {
  return (
    <div className="prd-legend">
      {legend.map(item => (
        <span key={item.segment} className="prd-legend-item">
          <span className="prd-legend-swatch" style={{ background: CP_PL[item.tone] }}></span>{item.label}
        </span>
      ))}
    </div>
  );
}

// The tail lane — every PRD whose signal is delivered — starts collapsed.
function LaneSection({ entry, rows, view, rowProps }) {
  const [open, setOpen] = uS_PL(entry.lane !== 'tail');
  return (
    <section className="prd-lane" data-lane={entry.lane}>
      <div className="prd-lane-head">
        <DatumReading_PL datum={view} format={() => `${entry.label} (${entry.prd_keys.length})`} rule={entry.rule} />
      </div>
      {open
        ? rows.map(row => <PrdRowView key={row.key} row={row} {...rowProps} />)
        : (
          <button type="button" className="prd-tail-line" onClick={() => setOpen(true)}>
            {`${rows.length} delivered: ${rows.map(row => row.title).join(' · ')}`}
          </button>
        )}
    </section>
  );
}

function PrdRowView({ row, view, pool, project, renderMembers }) {
  const [open, setOpen] = uS_PL(false);
  const total = doneOfTotal(row);
  const signal = signalCell(row.signal);
  const ready = readyCell(row.ready);
  const action = row.next_action;
  return (
    <div className="prd-row" data-key={row.key}>
      <div className="prd-row-head" role="button" onClick={() => setOpen(was => !was)}>
        <span className="prd-title"><ViewFigure view={view} figure="title" detail={row.key} format={() => row.title} /></span>
        <span className="prd-state"><ViewFigure view={view} figure="state_line" format={() => row.state_line} /></span>
        <SegmentBar row={row} legend={view.value.segment_legend} />
        <span className="prd-total"><ViewFigure view={view} figure="total" detail={total.detail} format={() => total.text} /></span>
        <span className="prd-signal"><ViewFigure view={view} figure="signal" detail={signal.detail} format={() => signal.text} /></span>
        <span className={ready.overdue ? 'prd-ready overdue' : 'prd-ready'}>
          <ViewFigure view={view} figure="ready" format={() => ready.text} />
        </span>
        {action.kind !== 'none' && (
          <span className={`prd-action prd-action-${action.kind}`}>
            <ViewFigure view={view} figure="next_action" format={() => action.text} />
          </span>
        )}
      </div>
      {open && <PrdRowDetail row={row} view={view} pool={pool} project={project} renderMembers={renderMembers} />}
    </div>
  );
}

// Zero-count segments stay in the DOM, titled, but take no width or gap.
function SegmentBar({ row, legend }) {
  return (
    <span className="prd-bar">
      {barSegments(row, legend).map(seg => (
        <span
          key={seg.segment}
          data-segment={seg.segment}
          className={seg.count === 0 ? 'prd-bar-seg prd-bar-seg-empty' : 'prd-bar-seg'}
          title={seg.title}
          style={{ flexGrow: seg.count, background: CP_PL[seg.tone] }}
        ></span>
      ))}
    </span>
  );
}

function PrdRowDetail({ row, view, pool, project, renderMembers }) {
  const openIds = [...row.direct_open_ids, ...row.followup_open.map(followup => followup.task_id)];
  const { rows, notDrawn } = membersDrawn(pool, project, openIds);
  const { blocker_refs: blockers, oldest_ready_high_ids: readyHigh, deferred_unrecorded_ids: unrecorded } = row.detail;
  const { direct, followup } = row.activity;
  return (
    <div className="prd-row-body">
      <div className="prd-activity">
        <ViewFigure view={view} figure="activity" format={() => `active: ${direct} direct, ${followup} follow-up`} />
      </div>
      {renderMembers(rows)}
      {notDrawn.length > 0 && <div className="prd-not-drawn muted">not drawn: {notDrawn.map(id => `#${id}`).join(' ')}</div>}
      <div className="prd-detail" title={view.value.rules.detail}>
        <ChipSection className="prd-blockers" label="blocked by" chips={blockers.map(ref => `#${ref.task_id} (${ref.prd ?? 'no PRD'})`)} />
        <ChipSection className="prd-ready-high" label="ready high, oldest first" chips={readyHigh.map(id => `#${id}`)} />
        <ChipSection className="prd-deferred-unrecorded" label="deferred with no record" chips={unrecorded.map(id => `#${id}`)} />
      </div>
    </div>
  );
}

function ChipSection({ className, label, chips }) {
  if (chips.length === 0) return null;
  return (
    <div className={className}>
      <span className="muted">{label}</span>
      {chips.map(chip => <span key={chip} className="chip">{chip}</span>)}
    </div>
  );
}

function DarkMatterStrip({ view, pool, project, onSelect }) {
  const [open, setOpen] = uS_PL(false);
  const darkMatter = view.value.dark_matter;
  const { hidden_high: hiddenHigh, ready_high_ids: readyHigh } = darkMatter;
  const summary = [
    `${darkMatter.total} open tasks no PRD owns`,
    ...['by_reason', 'by_status', 'by_priority', 'by_source'].map(by => countsText(darkMatter[by])).filter(Boolean),
    ...(readyHigh.length > 0 ? [`ready high ${readyHigh.map(id => `#${id}`).join(' ')}`] : []),
  ].join(' · ');
  const hidden = hiddenHigh.refs.map(ref => `#${ref.task_id} (${ref.prd})`).join(', ');
  return (
    <div className="dark-matter">
      <div className="dark-matter-head">
        <button type="button" className="dark-matter-toggle" onClick={() => setOpen(was => !was)}>{open ? '▾' : '▸'}</button>
        <ViewFigure view={view} figure="dark_matter" format={() => summary} />
      </div>
      {hiddenHigh.count > 0 && (
        <div className="dark-matter-hidden-high">
          <ViewFigure view={view} figure="hidden_high" format={() => `${hiddenHigh.count} high open under the collapsed tail: ${hidden}`} />
        </div>
      )}
      {open && Object.entries(darkMatter.members).map(([reason, ids]) => (
        <DarkMatterGroup key={reason} reason={reason} ids={ids} pool={pool} project={project} onSelect={onSelect} />
      ))}
    </div>
  );
}

function DarkMatterGroup({ reason, ids, pool, project, onSelect }) {
  const [pages, setPages] = uS_PL(1);
  const page = darkMatterPage(ids, pages);
  const { rows, notDrawn } = membersDrawn(pool, project, page.ids);
  return (
    <div className="dark-matter-group" data-reason={reason}>
      <div className="dark-matter-reason muted">{`${reason} (${ids.length})`}</div>
      {rows.map(row => (
        <button key={row.id} type="button" className="dark-matter-row" data-row-id={row.id} onClick={() => onSelect(row.id)}>
          {row.title}
        </button>
      ))}
      {notDrawn.length > 0 && <div className="prd-not-drawn muted">not drawn: {notDrawn.map(id => `#${id}`).join(' ')}</div>}
      {page.more > 0 && (
        <button type="button" className="dark-matter-more" onClick={() => setPages(was => was + 1)}>{`more (${page.more})`}</button>
      )}
    </div>
  );
}

function ProjectFooter({ view }) {
  const { closed, signal_coverage: coverage, multi_link: multiLink, inputs } = view.value;
  return (
    <div className="prd-footer">
      <span className="prd-closed">
        <ViewFigure view={view} figure="closed" format={() => `${closed.prds} PRDs closed (${closed.tasks} tasks)`} />
      </span>
      <span className="prd-coverage">
        <ViewFigure view={view} figure="signal_coverage" format={() => `${coverage.named} of ${coverage.of} open-work PRDs name a signal leaf`} />
      </span>
      {multiLink > 0 && (
        <span className="prd-multi-link">
          <ViewFigure view={view} figure="multi_link" format={() => `${multiLink} tasks carry more than one lineage link`} />
        </span>
      )}
      {inputs.map(input => <PrdInputPip key={input.name} input={input} />)}
    </div>
  );
}

function PrdInputPip({ input }) {
  const pip = inputPip(input);
  return (
    <span className="pip" data-input={input.name} title={pip.title}>
      <span className="pip-dot" style={{ background: CP_PL[pip.tone] }}></span>{pip.text}
    </span>
  );
}

function PrdContentionLine({ contention }) {
  if (contention.state === 'unknown') return <div className="prd-contention muted">{contention.reason}</div>;
  const { lock_depth: depth, snapshot_at: snapshotAt, top, held_share: heldShare, never_started: neverStarted, rules } = contention.value;
  const reading = (format, rule) => <DatumReading_PL datum={contention} format={format} rule={rule} />;
  return (
    <div className="prd-contention">
      <span className="prd-contention-meta" title={`${rules.lock_depth}\n\n${rules.snapshot_at}`}>
        {`locks at depth ${depth} · scheduler state changed ${timeago_PL(snapshotAt)}`}
      </span>
      {top.map(entry => (
        <span key={entry.key} className="prd-contention-entry">{reading(() => contentionEntryText(entry), rules.top)}</span>
      ))}
      <span className="prd-held-share">{reading(() => heldShareText(heldShare), rules.held_share)}</span>
      <span className="prd-never-started">{reading(() => neverStartedText(neverStarted), rules.never_started)}</span>
    </div>
  );
}

window.DF_PRD_LANES = { ProjectPrdLanes, PrdContentionLine };
