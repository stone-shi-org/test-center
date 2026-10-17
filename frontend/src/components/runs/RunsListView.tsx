import { useEffect, useState } from 'react'
import { api, type Run } from '../../api/index.ts'
import { Link, useRouter } from '../../router/index.ts'
import { Badge } from '../common/Badge.tsx'
import { Spinner } from '../common/Spinner.tsx'

export function RunsListView() {
  const { navigate } = useRouter()

  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState('all')
  const [testerFilter, setTesterFilter] = useState('all')
  const [selectedRunIds, setSelectedRunIds] = useState<number[]>([])

  const toggleRunSelection = (runId: number) => {
    setSelectedRunIds((prev) =>
      prev.includes(runId) ? prev.filter((id) => id !== runId) : [...prev, runId]
    )
  }

  const handleCompareSelected = () => {
    if (selectedRunIds.length > 0) {
      navigate(`/compare?run_ids=${selectedRunIds.join(',')}`)
    } else {
      navigate('/compare')
    }
  }

  const fetchRuns = async (showLoadingSpinner: boolean) => {
    try {
      if (showLoadingSpinner) setLoading(true)
      setError(null)
      const data = await api.runs.list({ limit: 100 })
      setRuns(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      if (showLoadingSpinner) setLoading(false)
    }
  }

  useEffect(() => {
    let ignore = false
    api.runs
      .list({ limit: 100 })
      .then((data) => {
        if (!ignore) {
          setRuns(data)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (!ignore) {
          setError(err instanceof Error ? err.message : String(err))
          setLoading(false)
        }
      })

    return () => {
      ignore = true
    }
  }, [])

  // Auto-refresh when any run is queued or running
  useEffect(() => {
    const hasActiveRuns = runs.some((r) => r.status === 'queued' || r.status === 'running')
    if (!hasActiveRuns) return

    const timer = setInterval(() => {
      fetchRuns(false)
    }, 3000)

    return () => clearInterval(timer)
  }, [runs])

  const uniqueTesters = Array.from(
    new Set(runs.map((r) => (r.config_snapshot?.tester as string) || 'unknown'))
  )

  const filteredRuns = runs.filter((r) => {
    const matchesStatus = statusFilter === 'all' || r.status === statusFilter
    const testerName = (r.config_snapshot?.tester as string) || 'unknown'
    const matchesTester = testerFilter === 'all' || testerName === testerFilter
    return matchesStatus && matchesTester
  })

  const formatDuration = (r: Run): string => {
    if (r.started_at && r.finished_at) {
      const ms = new Date(r.finished_at).getTime() - new Date(r.started_at).getTime()
      return `${(ms / 1000).toFixed(1)}s`
    }
    if (r.status === 'running') {
      return 'Running...'
    }
    if (r.status === 'queued') {
      return 'Queued'
    }
    return '—'
  }

  return (
    <div className="view-container runs-view">
      <div className="view-header">
        <div>
          <h2>Test Execution History</h2>
          <p className="view-subtitle">
            Monitor real-time runs, inspect historical benchmarks, and review results.
          </p>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary" onClick={handleCompareSelected}>
            ⚖️ {selectedRunIds.length >= 2 ? `Compare Selected (${selectedRunIds.length})` : 'Compare Runs'}
          </button>
          <button className="btn btn-secondary" onClick={() => fetchRuns(true)}>
            ↻ Refresh
          </button>
          <button className="btn btn-primary" onClick={() => navigate('/trigger')}>
            <span className="btn-icon">🚀</span> Trigger Run
          </button>
        </div>
      </div>

      {error && <div className="banner error-banner">{error}</div>}

      {selectedRunIds.length > 0 && (
        <div className="compare-selection-bar card">
          <div className="compare-selection-info">
            <span className="font-semibold">Selected for comparison ({selectedRunIds.length}):</span>
            <div className="selected-run-pills">
              {selectedRunIds.map((id, idx) => (
                <span key={id} className="selected-run-pill font-mono">
                  #{id}
                  {idx === 0 && <span className="baseline-mini-tag">Baseline</span>}
                  <button
                    type="button"
                    className="pill-remove-btn"
                    aria-label={`Remove run #${id}`}
                    onClick={() => toggleRunSelection(id)}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>
          <div className="compare-selection-actions">
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={() => setSelectedRunIds([])}
            >
              Clear
            </button>
            <button
              type="button"
              className="btn btn-sm btn-primary"
              onClick={handleCompareSelected}
            >
              ⚖️ {selectedRunIds.length >= 2 ? `Compare ${selectedRunIds.length} Runs →` : 'Open in Compare →'}
            </button>
          </div>
        </div>
      )}

      <div className="filter-bar">
        <div className="filter-group">
          <label htmlFor="runs-status-filter" className="filter-label">Status:</label>
          <select
            id="runs-status-filter"
            className="filter-select"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">All Statuses ({runs.length})</option>
            <option value="running">Running</option>
            <option value="queued">Queued</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="runs-tester-filter" className="filter-label">Tester:</label>
          <select
            id="runs-tester-filter"
            className="filter-select"
            value={testerFilter}
            onChange={(e) => setTesterFilter(e.target.value)}
          >
            <option value="all">All Testers ({uniqueTesters.length})</option>
            {uniqueTesters.map((t) => (
              <option key={t} value={t}>
                {t.toUpperCase()}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <Spinner message="Loading runs..." />
      ) : filteredRuns.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">⏱️</div>
          <h3>No test runs found</h3>
          <p>
            {runs.length === 0
              ? 'No test suites have been executed yet. Trigger a run to see live logs and benchmarks.'
              : 'No runs match the selected filters.'}
          </p>
          <button className="btn btn-primary" onClick={() => navigate('/trigger')}>
            Trigger First Run
          </button>
        </div>
      ) : (
        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '40px' }}>Compare</th>
                <th>Run ID</th>
                <th>Status</th>
                <th>Tester</th>
                <th>Target</th>
                <th>Duration</th>
                <th>Queued / Started</th>
                <th>Result</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredRuns.map((r) => {
                const tester = (r.config_snapshot?.tester as string) || 'unknown'
                const target = (r.target_snapshot?.name as string) || 'None'
                const primaryRes = r.results && r.results.length > 0 ? r.results[0] : null
                const isSelected = selectedRunIds.includes(r.id)

                return (
                  <tr key={r.id} className={isSelected ? 'row-selected' : ''}>
                    <td>
                      <input
                        type="checkbox"
                        aria-label={`Select run #${r.id} for comparison`}
                        checked={isSelected}
                        onChange={() => toggleRunSelection(r.id)}
                      />
                    </td>
                    <td>
                      <Link href={`/runs/${r.id}`} className="font-mono font-bold table-link">
                        #{r.id}
                      </Link>
                    </td>
                    <td>
                      <Badge status={r.status} size="sm" />
                    </td>
                    <td>
                      <span className="kind-badge">{tester}</span>
                    </td>
                    <td className="font-mono text-muted">{target}</td>
                    <td className="font-mono">{formatDuration(r)}</td>
                    <td className="text-muted">
                      {r.started_at
                        ? new Date(r.started_at).toLocaleString()
                        : r.queued_at
                          ? new Date(r.queued_at).toLocaleString()
                          : '—'}
                    </td>
                    <td>
                      {primaryRes ? (
                        <div className="result-pill-group">
                          <Badge status={primaryRes.status} size="sm" />
                          {primaryRes.cases && primaryRes.cases.length > 0 && (
                            <span className="cases-badge text-muted">
                              {primaryRes.cases.filter((c) => c.status === 'pass').length}/
                              {primaryRes.cases.length} passed
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                    </td>
                    <td>
                      <div className="row-actions">
                        <Link href={`/runs/${r.id}`} className="btn btn-sm btn-secondary">
                          View Detail →
                        </Link>
                        <button
                          type="button"
                          className="btn btn-sm btn-secondary"
                          title={`Compare Run #${r.id}`}
                          onClick={() => {
                            const ids = selectedRunIds.includes(r.id)
                              ? selectedRunIds
                              : [...selectedRunIds, r.id]
                            navigate(`/compare?run_ids=${ids.join(',')}`)
                          }}
                        >
                          ⚖️ Compare
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
