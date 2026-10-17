import { Fragment, useState } from 'react'
import type {
  CompareCaseCell,
  CompareCaseRow,
  ComparedRunMeta,
} from '../../api/types.ts'
import { Badge } from '../common/Badge.tsx'

function formatVal(val: unknown): string {
  if (val === null || val === undefined) return '—'
  if (typeof val === 'number') {
    if (Number.isInteger(val)) return val.toLocaleString()
    return Math.abs(val) >= 10 ? val.toFixed(2) : val.toFixed(3)
  }
  if (typeof val === 'boolean') return val ? 'true' : 'false'
  if (typeof val === 'string') return val
  try {
    return JSON.stringify(val)
  } catch {
    return String(val)
  }
}

export function CaseDiffTable({
  cases,
  totalCasesCount,
  runs,
  suiteFilter,
  onSuiteFilterChange,
  availableSuites,
}: {
  cases: CompareCaseRow[]
  totalCasesCount: number
  runs: ComparedRunMeta[]
  suiteFilter: string
  onSuiteFilterChange: (suite: string) => void
  availableSuites: string[]
}) {
  const [search, setSearch] = useState('')
  const [caseStatusFilter, setCaseStatusFilter] = useState('all')
  const [expandedKey, setExpandedKey] = useState<string | null>(null)

  const findCaseCell = (
    row: CompareCaseRow,
    r: ComparedRunMeta
  ): CompareCaseCell | undefined =>
    row.values.find((c) => c.run_id === r.run_id && c.result_id === r.result_id) ||
    row.values.find((c) => c.run_id === r.run_id)

  const visibleCases = cases.filter((c) => {
    if (suiteFilter !== 'all' && c.suite !== suiteFilter) {
      return false
    }
    if (caseStatusFilter === 'status_regressed' && !c.has_status_regression) {
      return false
    }
    if (caseStatusFilter === 'metric_regressed' && !c.has_metric_regression) {
      return false
    }
    if (caseStatusFilter === 'improved' && !c.has_improvement) {
      return false
    }
    if (caseStatusFilter === 'missing' && !c.missing_in_some) {
      return false
    }
    if (search.trim()) {
      const q = search.toLowerCase()
      const matchesId = c.case_id.toLowerCase().includes(q)
      const matchesSuite = c.suite.toLowerCase().includes(q)
      const matchesDetail = c.values.some((v) =>
        JSON.stringify(v.detail || {}).toLowerCase().includes(q)
      )
      return matchesId || matchesSuite || matchesDetail
    }
    return true
  })

  const toggleExpand = (key: string) => {
    setExpandedKey((prev) => (prev === key ? null : key))
  }

  return (
    <div className="card compare-section-card" data-testid="case-diff-table-section">
      <div className="compare-section-header">
        <div>
          <h3 className="section-title" style={{ marginBottom: '2px' }}>
            Per-Case Diff View ({visibleCases.length}
            {visibleCases.length !== totalCasesCount ? ` of ${totalCasesCount}` : ''})
          </h3>
          <span className="section-desc">
            Aligned test cases across runs with status transitions, per-case metric deltas, and detail diff
          </span>
        </div>
      </div>

      <div className="cases-filter-bar">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="search-input"
            placeholder="Search cases by ID, suite, or detail..."
            aria-label="Search cases"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="compare-case-diff-filter" className="filter-label">
            Case Diff:
          </label>
          <select
            id="compare-case-diff-filter"
            className="filter-select"
            value={caseStatusFilter}
            onChange={(e) => setCaseStatusFilter(e.target.value)}
          >
            <option value="all">All Shown ({cases.length})</option>
            <option value="status_regressed">
              Status Regressed ({cases.filter((c) => c.has_status_regression).length})
            </option>
            <option value="metric_regressed">
              Metric Regressed ({cases.filter((c) => c.has_metric_regression).length})
            </option>
            <option value="improved">
              Improved ({cases.filter((c) => c.has_improvement).length})
            </option>
            <option value="missing">
              Added / Missing ({cases.filter((c) => c.missing_in_some).length})
            </option>
          </select>
        </div>

        {availableSuites.length > 0 && (
          <div className="filter-group">
            <label htmlFor="compare-suite-filter" className="filter-label">
              Suite:
            </label>
            <select
              id="compare-suite-filter"
              className="filter-select"
              value={suiteFilter}
              onChange={(e) => onSuiteFilterChange(e.target.value)}
            >
              <option value="all">All Suites ({availableSuites.length})</option>
              {availableSuites.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {visibleCases.length === 0 ? (
        <div className="text-muted empty-inline-msg">
          No test cases match the active filters.
        </div>
      ) : (
        <div className="table-responsive">
          <table className="data-table case-diff-table" data-testid="case-diff-table">
            <thead>
              <tr>
                <th style={{ width: '36px' }} />
                <th>Case ID</th>
                <th>Suite</th>
                <th>Diff Status</th>
                {runs.map((r) => (
                  <th key={`${r.run_id}-${r.result_id}`}>
                    <div className="th-run-header">
                      <span>
                        Run #{r.run_id} {r.is_baseline ? '(Baseline)' : ''}
                      </span>
                      {(r.model || r.target_name) && (
                        <span className="th-run-sub font-mono">
                          {r.model || r.target_name}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
                <th>Inspect</th>
              </tr>
            </thead>
            <tbody>
              {visibleCases.map((c) => {
                const rowKey = `${c.suite}:${c.case_id}`
                const isExpanded = expandedKey === rowKey
                const rowClass = c.has_regression
                  ? 'row-regression'
                  : c.has_improvement
                    ? 'row-improvement'
                    : ''

                // Collect all detail keys across runs for this case
                const allDetailKeys: string[] = []
                const seenDetailKeys = new Set<string>()
                for (const cell of c.values) {
                  if (cell.detail && typeof cell.detail === 'object') {
                    for (const k of Object.keys(cell.detail)) {
                      if (!seenDetailKeys.has(k)) {
                        allDetailKeys.push(k)
                        seenDetailKeys.add(k)
                      }
                    }
                  }
                }

                return (
                  <Fragment key={rowKey}>
                    <tr
                      className={`case-row ${rowClass} ${isExpanded ? 'expanded' : ''}`}
                      onClick={() => toggleExpand(rowKey)}
                      style={{ cursor: 'pointer' }}
                      data-testid={`case-diff-row-${c.case_id}`}
                    >
                      <td className="expand-indicator">
                        <span className="expand-arrow">{isExpanded ? '▼' : '▶'}</span>
                      </td>
                      <td className="font-mono font-semibold">{c.case_id}</td>
                      <td>
                        <span className="kind-badge">{c.suite}</span>
                      </td>
                      <td>
                        <div className="case-diff-flags">
                          {c.has_status_regression && (
                            <span className="delta-badge delta-regression">
                              ▼ STATUS REGRESSION
                            </span>
                          )}
                          {!c.has_status_regression && c.has_metric_regression && (
                            <span className="delta-badge delta-regression">
                              ▼ METRIC REGRESSION
                            </span>
                          )}
                          {!c.has_regression && c.has_improvement && (
                            <span className="delta-badge delta-improvement">▲ IMPROVED</span>
                          )}
                          {c.missing_in_some && (
                            <span className="delta-badge delta-missing">
                              {c.status_change === 'added' ? '+ ADDED' : 'MISSING'}
                            </span>
                          )}
                          {!c.has_regression &&
                            !c.has_improvement &&
                            !c.missing_in_some &&
                            c.has_change && (
                              <span className="delta-badge delta-neutral">CHANGED</span>
                            )}
                          {!c.has_change && (
                            <span className="delta-badge delta-unchanged">UNCHANGED</span>
                          )}
                        </div>
                      </td>

                      {runs.map((r) => {
                        const cell = findCaseCell(c, r)
                        if (!cell || !cell.present || !cell.status) {
                          return (
                            <td
                              key={`${r.run_id}-${r.result_id}`}
                              className="cell-missing font-mono"
                            >
                              <span className="delta-badge delta-missing">— (missing)</span>
                            </td>
                          )
                        }

                        const metricDeltaEntries = Object.values(cell.metric_deltas || {})

                        return (
                          <td
                            key={`${r.run_id}-${r.result_id}`}
                            className={
                              cell.regression
                                ? 'cell-regression'
                                : cell.improvement
                                  ? 'cell-improvement'
                                  : ''
                            }
                          >
                            <div className="case-cell-top">
                              <Badge status={cell.status} size="sm" />
                              {!cell.is_baseline && cell.status_changed && (
                                <span
                                  className={`status-transition-tag font-mono ${
                                    cell.status_regression
                                      ? 'text-danger'
                                      : cell.status_improvement
                                        ? 'text-success'
                                        : 'text-muted'
                                  }`}
                                >
                                  {c.baseline_status || 'none'} → {cell.status}
                                </span>
                              )}
                            </div>

                            {metricDeltaEntries.length > 0 && (
                              <div className="case-cell-metrics font-mono">
                                {metricDeltaEntries.slice(0, 3).map((md) => {
                                  if (!md.present || md.value === null) {
                                    return (
                                      <span key={md.name} className="case-metric-pill missing">
                                        {md.name}: —
                                      </span>
                                    )
                                  }
                                  const pctText =
                                    !cell.is_baseline && md.pct_delta !== null
                                      ? ` (${md.pct_delta > 0 ? '+' : ''}${md.pct_delta.toFixed(1)}%)`
                                      : ''
                                  return (
                                    <span
                                      key={md.name}
                                      className={`case-metric-pill ${
                                        md.regression
                                          ? 'regression'
                                          : md.improvement
                                            ? 'improvement'
                                            : ''
                                      }`}
                                    >
                                      {md.name}={formatVal(md.value)}
                                      {md.unit ? ` ${md.unit}` : ''}
                                      {pctText}
                                    </span>
                                  )
                                })}
                              </div>
                            )}
                          </td>
                        )
                      })}

                      <td>
                        <button
                          type="button"
                          className="btn-link"
                          onClick={(e) => {
                            e.stopPropagation()
                            toggleExpand(rowKey)
                          }}
                        >
                          {isExpanded ? 'Hide Diff' : 'Inspect Diff'}
                        </button>
                      </td>
                    </tr>

                    {isExpanded && (
                      <tr className="case-diff-drawer-row">
                        <td colSpan={runs.length + 5} style={{ padding: 0 }}>
                          <div
                            className="case-detail-drawer"
                            style={{ margin: '8px 12px 16px 12px' }}
                            data-testid={`case-diff-drawer-${c.case_id}`}
                          >
                            <div className="drawer-header">
                              <h4>
                                Per-Case Diff: <code>{c.case_id}</code> ({c.suite})
                              </h4>
                              <span className="font-mono text-muted">
                                Status Change: {c.status_change.toUpperCase()}
                              </span>
                            </div>

                            {/* Per-Case Metrics Alignment Table */}
                            {c.metrics && c.metrics.length > 0 && (
                              <div style={{ marginBottom: '16px' }}>
                                <h5 style={{ margin: '0 0 8px 0', color: 'var(--text-secondary)' }}>
                                  Per-Case Metric Deltas
                                </h5>
                                <div className="table-responsive">
                                  <table className="data-table">
                                    <thead>
                                      <tr>
                                        <th>Metric</th>
                                        {runs.map((r) => (
                                          <th key={`${r.run_id}-cm`}>
                                            Run #{r.run_id} {r.is_baseline ? '(Baseline)' : ''}
                                          </th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {c.metrics.map((cm) => (
                                        <tr key={cm.name}>
                                          <td className="font-mono font-semibold">
                                            {cm.name}
                                            {cm.unit ? ` (${cm.unit})` : ''}
                                          </td>
                                          {runs.map((r) => {
                                            const mCell = cm.values.find(
                                              (v) => v.run_id === r.run_id
                                            )
                                            if (!mCell || !mCell.present || mCell.value === null) {
                                              return (
                                                <td key={r.run_id} className="font-mono text-muted">
                                                  — (missing)
                                                </td>
                                              )
                                            }
                                            return (
                                              <td key={r.run_id} className="font-mono">
                                                <span>
                                                  {formatVal(mCell.value)}
                                                  {cm.unit ? ` ${cm.unit}` : ''}
                                                </span>
                                                {!mCell.is_baseline && mCell.abs_delta !== null && (
                                                  <span
                                                    className={`delta-badge ${
                                                      mCell.regression
                                                        ? 'delta-regression'
                                                        : mCell.improvement
                                                          ? 'delta-improvement'
                                                          : 'delta-neutral'
                                                    }`}
                                                    style={{ marginLeft: '8px' }}
                                                  >
                                                    {mCell.abs_delta > 0 ? '+' : ''}
                                                    {formatVal(mCell.abs_delta)}
                                                    {mCell.pct_delta !== null
                                                      ? ` (${mCell.pct_delta > 0 ? '+' : ''}${mCell.pct_delta.toFixed(1)}%)`
                                                      : ''}
                                                  </span>
                                                )}
                                              </td>
                                            )
                                          })}
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            )}

                            {/* Detail Keys Diff Table */}
                            {allDetailKeys.length > 0 && (
                              <div style={{ marginBottom: '16px' }}>
                                <h5 style={{ margin: '0 0 8px 0', color: 'var(--text-secondary)' }}>
                                  Case Detail Field Comparison
                                </h5>
                                <div className="table-responsive">
                                  <table className="data-table">
                                    <thead>
                                      <tr>
                                        <th>Detail Key</th>
                                        <th>Diff</th>
                                        {runs.map((r) => (
                                          <th key={`${r.run_id}-dk`}>
                                            Run #{r.run_id} {r.is_baseline ? '(Baseline)' : ''}
                                          </th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {allDetailKeys.map((dk) => {
                                        const vals = runs.map((r) => {
                                          const cell = findCaseCell(c, r)
                                          return cell?.detail ? cell.detail[dk] : undefined
                                        })
                                        const serialized = new Set(
                                          vals.map((v) => JSON.stringify(v))
                                        )
                                        const keyDiffers = serialized.size > 1

                                        return (
                                          <tr
                                            key={dk}
                                            className={keyDiffers ? 'config-row-changed' : ''}
                                          >
                                            <td className="font-mono font-semibold">{dk}</td>
                                            <td>
                                              <span
                                                className={`diff-pill ${
                                                  keyDiffers ? 'diff-pill-warn' : 'diff-pill-neutral'
                                                }`}
                                              >
                                                {keyDiffers ? 'DIFFERS' : 'SAME'}
                                              </span>
                                            </td>
                                            {runs.map((r, idx) => (
                                              <td
                                                key={`${r.run_id}-${dk}`}
                                                className="font-mono text-break"
                                              >
                                                {formatVal(vals[idx])}
                                              </td>
                                            ))}
                                          </tr>
                                        )
                                      })}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            )}

                            {/* Side-by-Side Raw Detail JSON */}
                            <div
                              className="drawer-grid"
                              style={{
                                gridTemplateColumns: `repeat(${Math.min(runs.length, 3)}, minmax(0, 1fr))`,
                              }}
                            >
                              {runs.map((r) => {
                                const cell = findCaseCell(c, r)
                                return (
                                  <div key={`${r.run_id}-raw`} className="drawer-col">
                                    <h5>
                                      Run #{r.run_id} {r.is_baseline ? '(Baseline)' : ''} Detail JSON
                                    </h5>
                                    <pre className="json-block font-mono">
                                      {cell?.present
                                        ? JSON.stringify(cell.detail || {}, null, 2)
                                        : '// Case not present in this run'}
                                    </pre>
                                  </div>
                                )
                              })}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
