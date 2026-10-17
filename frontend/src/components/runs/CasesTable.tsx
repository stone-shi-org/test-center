import { useState } from 'react'
import type { CaseResult } from '../../api/types.ts'
import { Badge } from '../common/Badge.tsx'

export function CasesTable({
  cases,
}: {
  cases: CaseResult[]
}) {
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [suiteFilter, setSuiteFilter] = useState('all')
  const [expandedCaseId, setExpandedCaseId] = useState<number | null>(null)

  if (!cases || cases.length === 0) {
    return <div className="text-muted empty-inline-msg">No test cases recorded for this result.</div>
  }

  const uniqueSuites = Array.from(new Set(cases.map((c) => c.suite)))
  const uniqueStatuses = Array.from(new Set(cases.map((c) => c.status)))

  const filteredCases = cases.filter((c) => {
    const matchesStatus = statusFilter === 'all' || c.status.toLowerCase() === statusFilter.toLowerCase()
    const matchesSuite = suiteFilter === 'all' || c.suite === suiteFilter
    const matchesSearch =
      c.case_id.toLowerCase().includes(search.toLowerCase()) ||
      c.suite.toLowerCase().includes(search.toLowerCase()) ||
      JSON.stringify(c.metrics).toLowerCase().includes(search.toLowerCase())
    return matchesStatus && matchesSuite && matchesSearch
  })

  const toggleExpand = (id: number) => {
    setExpandedCaseId((prev) => (prev === id ? null : id))
  }

  return (
    <div className="cases-section">
      <div className="cases-filter-bar">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="search-input"
            placeholder="Search cases..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="cases-status-filter" className="filter-label">Status:</label>
          <select
            id="cases-status-filter"
            className="filter-select"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">All Statuses ({cases.length})</option>
            {uniqueStatuses.map((st) => (
              <option key={st} value={st}>
                {st.toUpperCase()} ({cases.filter((c) => c.status === st).length})
              </option>
            ))}
          </select>
        </div>

        {uniqueSuites.length > 1 && (
          <div className="filter-group">
            <label htmlFor="cases-suite-filter" className="filter-label">Suite:</label>
            <select
              id="cases-suite-filter"
              className="filter-select"
              value={suiteFilter}
              onChange={(e) => setSuiteFilter(e.target.value)}
            >
              <option value="all">All Suites ({uniqueSuites.length})</option>
              {uniqueSuites.map((s) => (
                <option key={s} value={s}>
                  {s} ({cases.filter((c) => c.suite === s).length})
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div className="table-responsive">
        <table className="data-table cases-table">
          <thead>
            <tr>
              <th style={{ width: '40px' }} />
              <th>Case ID</th>
              <th>Suite</th>
              <th>Status</th>
              <th>Metrics</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {filteredCases.map((c) => {
              const isExpanded = expandedCaseId === c.id
              const hasMetrics = c.metrics && Object.keys(c.metrics).length > 0

              return (
                <tr
                  key={c.id}
                  className={`case-row ${isExpanded ? 'expanded' : ''}`}
                  onClick={() => toggleExpand(c.id)}
                  style={{ cursor: 'pointer' }}
                >
                  <td className="expand-indicator">
                    <span className="expand-arrow">{isExpanded ? '▼' : '▶'}</span>
                  </td>
                  <td className="font-mono font-semibold">{c.case_id}</td>
                  <td>
                    <span className="kind-badge">{c.suite}</span>
                  </td>
                  <td>
                    <Badge status={c.status} size="sm" />
                  </td>
                  <td>
                    {hasMetrics ? (
                      <span className="metrics-summary-tag font-mono">
                        {Object.entries(c.metrics)
                          .map(([k, v]) => `${k}=${v}`)
                          .join(', ')}
                      </span>
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                  <td>
                    <button
                      type="button"
                      className="btn-link"
                      onClick={(e) => {
                        e.stopPropagation()
                        toggleExpand(c.id)
                      }}
                    >
                      {isExpanded ? 'Hide' : 'Inspect'}
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {expandedCaseId !== null && (
        <div className="case-detail-drawer">
          {(() => {
            const c = cases.find((item) => item.id === expandedCaseId)
            if (!c) return null
            return (
              <div className="drawer-inner">
                <div className="drawer-header">
                  <h4>
                    Case Detail: <code>{c.case_id}</code> ({c.suite})
                  </h4>
                  <Badge status={c.status} />
                </div>
                <div className="drawer-grid">
                  <div className="drawer-col">
                    <h5>Metrics</h5>
                    <pre className="json-block font-mono">
                      {JSON.stringify(c.metrics || {}, null, 2)}
                    </pre>
                  </div>
                  <div className="drawer-col">
                    <h5>Detail Payload</h5>
                    <pre className="json-block font-mono">
                      {JSON.stringify(c.detail || {}, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            )
          })()}
        </div>
      )}
    </div>
  )
}
