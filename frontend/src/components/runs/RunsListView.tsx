import { useEffect, useState } from 'react'
import { api, type Run } from '../../api/index.ts'
import { Link, useRouter } from '../../router/index.ts'
import {
  formatDateTime,
  formatTime,
  parseLocalDateEnd,
  parseLocalDateStart,
  parseUtcDate,
} from '../../utils/date.ts'
import { Badge } from '../common/Badge.tsx'
import { Spinner } from '../common/Spinner.tsx'
import { MetricTrendChart } from './MetricTrendChart.tsx'
import { RetentionSettingsPanel } from './RetentionSettingsPanel.tsx'

function getRunTesters(r: Run): string[] {
  const list: string[] = []
  const add = (val: unknown) => {
    if (typeof val === 'string') {
      for (const part of val.split(',')) {
        const trimmed = part.trim()
        if (trimmed && !list.includes(trimmed)) {
          list.push(trimmed)
        }
      }
    }
  }

  if (Array.isArray(r.config_snapshot?.testers)) {
    r.config_snapshot.testers.forEach(add)
  }
  add(r.config_snapshot?.tester)
  if (Array.isArray(r.config_snapshot?.definitions)) {
    for (const d of r.config_snapshot.definitions) {
      if (d && typeof d === 'object') add(d.tester)
    }
  }
  for (const res of r.results || []) {
    add(res.tester)
  }
  if (list.length === 0) {
    list.push('unknown')
  }
  return list
}

function getRunTargets(r: Run): string[] {
  const list: string[] = []
  const add = (val: unknown) => {
    if (typeof val === 'string' && val.trim() && !list.includes(val.trim())) {
      list.push(val.trim())
    }
  }

  add(r.target_snapshot?.name)
  if (Array.isArray(r.target_snapshot?.targets)) {
    for (const t of r.target_snapshot.targets) {
      if (t && typeof t === 'object') add(t.name)
    }
  }
  for (const res of r.results || []) {
    add(res.target_name)
  }
  if (list.length === 0) {
    list.push('None')
  }
  return list
}

function getRunModels(r: Run): string[] {
  const list: string[] = []
  const add = (val: unknown) => {
    if (typeof val === 'string' && val.trim() && !list.includes(val.trim())) {
      list.push(val.trim())
    }
  }

  const cfg = (r.config_snapshot?.config || {}) as Record<string, unknown>
  const extra = (r.target_snapshot?.extra || {}) as Record<string, unknown>
  add(cfg.model)
  add(cfg.transcription_model)
  if (Array.isArray(cfg.models)) {
    cfg.models.forEach(add)
  }
  add(extra.model)
  add(extra.transcription_model)
  add(r.target_snapshot?.model)
  for (const res of r.results || []) {
    add(res.model)
  }
  return list
}

function getRunJobLabel(r: Run): { key: string; label: string } {
  const jobId =
    r.job_id ??
    (typeof r.config_snapshot?.job_id === 'number' ? r.config_snapshot.job_id : null)
  if (jobId === null || jobId === undefined) {
    return { key: 'manual', label: 'Manual / Ad-hoc' }
  }
  const jobName =
    r.job_name ||
    (typeof r.config_snapshot?.job_name === 'string' ? r.config_snapshot.job_name : null)
  return {
    key: String(jobId),
    label: jobName ? `${jobName} (#${jobId})` : `Job #${jobId}`,
  }
}

function getRunTimestampMs(r: Run): number | null {
  const raw = r.started_at || r.queued_at || r.finished_at
  const d = parseUtcDate(raw)
  return d ? d.getTime() : null
}

export function RunsListView() {
  const { route, navigate } = useRouter()

  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [statusFilter, setStatusFilter] = useState<string>(route.params.status || 'all')
  const [testerFilter, setTesterFilter] = useState<string>(route.params.tester || 'all')
  const [targetFilter, setTargetFilter] = useState<string>(route.params.target || 'all')
  const [jobFilter, setJobFilter] = useState<string>(route.params.job || 'all')
  const [dateFrom, setDateFrom] = useState<string>(route.params.date_from || '')
  const [dateTo, setDateTo] = useState<string>(route.params.date_to || '')
  const [baselinesOnly, setBaselinesOnly] = useState<boolean>(route.params.baseline === 'true')

  const [selectedRunIds, setSelectedRunIds] = useState<number[]>([])
  const [showTrendChart, setShowTrendChart] = useState(true)
  const [showRetention, setShowRetention] = useState(false)

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

  const handleStartCompareFromTrend = (runIds: number[], baselineRunId?: number) => {
    const query = new URLSearchParams()
    query.set('run_ids', runIds.join(','))
    if (baselineRunId !== undefined) {
      query.set('baseline_run_id', String(baselineRunId))
    }
    navigate(`/compare?${query.toString()}`)
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
    new Set(runs.flatMap((r) => getRunTesters(r)))
  )

  const uniqueTargets = Array.from(
    new Set(runs.flatMap((r) => getRunTargets(r)))
  )

  const uniqueJobsMap = new Map<string, string>()
  for (const r of runs) {
    const jInfo = getRunJobLabel(r)
    if (jInfo.key !== 'manual' && !uniqueJobsMap.has(jInfo.key)) {
      uniqueJobsMap.set(jInfo.key, jInfo.label)
    }
  }

  const fromMs = parseLocalDateStart(dateFrom)
  const toMs = parseLocalDateEnd(dateTo)

  const filteredRuns = runs.filter((r) => {
    // 0. Baselines only filter
    if (baselinesOnly && (!r.baselines || r.baselines.length === 0)) {
      return false
    }

    // 1. Status filter
    let matchesStatus = true
    if (statusFilter !== 'all') {
      if (statusFilter === 'pass') {
        matchesStatus =
          r.results && r.results.length > 0 && r.results.every((res) => res.status === 'pass')
      } else if (statusFilter === 'fail') {
        matchesStatus =
          r.status === 'failed' ||
          Boolean(r.results && r.results.some((res) => res.status === 'fail'))
      } else if (statusFilter === 'failed') {
        matchesStatus =
          r.status === 'failed' ||
          Boolean(r.results && r.results.some((res) => res.status === 'fail'))
      } else {
        matchesStatus = r.status === statusFilter
      }
    }

    // 2. Tester filter
    const runTesters = getRunTesters(r)
    const matchesTester =
      testerFilter === 'all' ||
      runTesters.some((t) => t.toLowerCase() === testerFilter.toLowerCase())

    // 3. Target filter
    const runTargets = getRunTargets(r)
    const matchesTarget =
      targetFilter === 'all' ||
      runTargets.some((t) => t.toLowerCase() === targetFilter.toLowerCase())

    // 4. Job filter
    const jInfo = getRunJobLabel(r)
    const matchesJob =
      jobFilter === 'all' ||
      jInfo.key === jobFilter ||
      jInfo.label.toLowerCase().includes(jobFilter.toLowerCase())

    // 5. Date range filter
    let matchesDate = true
    if (fromMs !== null || toMs !== null) {
      const ts = getRunTimestampMs(r)
      if (ts === null) {
        matchesDate = false
      } else {
        if (fromMs !== null && ts < fromMs) matchesDate = false
        if (toMs !== null && ts > toMs) matchesDate = false
      }
    }

    return matchesStatus && matchesTester && matchesTarget && matchesJob && matchesDate
  })

  const hasActiveFilters =
    statusFilter !== 'all' ||
    testerFilter !== 'all' ||
    targetFilter !== 'all' ||
    jobFilter !== 'all' ||
    dateFrom !== '' ||
    dateTo !== '' ||
    baselinesOnly

  const clearFilters = () => {
    setStatusFilter('all')
    setTesterFilter('all')
    setTargetFilter('all')
    setJobFilter('all')
    setDateFrom('')
    setDateTo('')
    setBaselinesOnly(false)
  }

  const formatDuration = (r: Run): string => {
    const start = parseUtcDate(r.started_at)
    const finish = parseUtcDate(r.finished_at)
    if (start && finish) {
      const ms = finish.getTime() - start.getTime()
      if (ms < 0) return '0s'
      if (ms < 1000) return `${ms}ms`
      const sec = Math.round(ms / 1000)
      if (sec < 60) return `${sec}s`
      const min = Math.floor(sec / 60)
      const remSec = sec % 60
      return `${min}m ${remSec}s`
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
            Monitor real-time runs, analyze metric trends over time, and manage retention.
          </p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className={`btn ${showTrendChart ? 'btn-primary' : 'btn-secondary'}`}
            data-testid="toggle-trend-chart-btn"
            onClick={() => setShowTrendChart((prev) => !prev)}
          >
            📈 {showTrendChart ? 'Hide Trends' : 'Trend Chart'}
          </button>
          <button
            type="button"
            className={`btn ${showRetention ? 'btn-primary' : 'btn-secondary'}`}
            data-testid="toggle-retention-btn"
            onClick={() => setShowRetention((prev) => !prev)}
          >
            🗄️ Retention Settings
          </button>
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

      {showRetention && (
        <RetentionSettingsPanel
          onCleanedUp={() => fetchRuns(false)}
          onClose={() => setShowRetention(false)}
        />
      )}

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

      {/* Filter Bar: Tester, Target, Job, Status, Date */}
      <div className="filter-bar runs-filter-bar" data-testid="runs-filter-bar">
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

        <div className="filter-group">
          <label htmlFor="runs-target-filter" className="filter-label">Target:</label>
          <select
            id="runs-target-filter"
            className="filter-select"
            value={targetFilter}
            onChange={(e) => setTargetFilter(e.target.value)}
          >
            <option value="all">All Targets ({uniqueTargets.length})</option>
            {uniqueTargets.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="runs-job-filter" className="filter-label">Job:</label>
          <select
            id="runs-job-filter"
            className="filter-select"
            value={jobFilter}
            onChange={(e) => setJobFilter(e.target.value)}
          >
            <option value="all">All Jobs</option>
            <option value="manual">Manual / Ad-hoc</option>
            {Array.from(uniqueJobsMap.entries()).map(([jKey, jLabel]) => (
              <option key={jKey} value={jKey}>
                {jLabel}
              </option>
            ))}
          </select>
        </div>

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
            <option value="pass">Pass</option>
            <option value="fail">Fail</option>
            <option value="failed">Failed</option>
            <option value="error">Error</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="runs-date-from" className="filter-label">From Date:</label>
          <input
            id="runs-date-from"
            type="date"
            aria-label="From Date"
            className="filter-select"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <label htmlFor="runs-date-to" className="filter-label">To Date:</label>
          <input
            id="runs-date-to"
            type="date"
            aria-label="To Date"
            className="filter-select"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
          />
        </div>

        <div className="filter-group" style={{ display: 'flex', alignItems: 'center' }}>
          <label className="checkbox-label" style={{ cursor: 'pointer', userSelect: 'none' }}>
            <input
              type="checkbox"
              data-testid="runs-baselines-only-toggle"
              aria-label="Baselines only"
              checked={baselinesOnly}
              onChange={(e) => setBaselinesOnly(e.target.checked)}
            />
            <span>★ Baselines only</span>
          </label>
        </div>

        {hasActiveFilters && (
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            data-testid="clear-runs-filters-btn"
            onClick={clearFilters}
          >
            ✕ Clear Filters ({filteredRuns.length}/{runs.length})
          </button>
        )}
      </div>

      {/* Per-Metric Trend Chart Section */}
      {!loading && showTrendChart && runs.length > 0 && (
        <MetricTrendChart
          runs={filteredRuns}
          selectedRunIds={selectedRunIds}
          onToggleRunSelection={toggleRunSelection}
          onOpenRun={(runId) => navigate(`/runs/${runId}`)}
          onStartCompare={handleStartCompareFromTrend}
        />
      )}

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
          {hasActiveFilters ? (
            <button className="btn btn-secondary" onClick={clearFilters}>
              Reset Filters
            </button>
          ) : (
            <button className="btn btn-primary" onClick={() => navigate('/trigger')}>
              Trigger First Run
            </button>
          )}
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
                <th>Target / Model</th>
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
                const models = getRunModels(r)
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
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                          <Link href={`/runs/${r.id}`} className="font-mono font-bold table-link">
                            #{r.id}
                          </Link>
                          {r.baselines &&
                            r.baselines.map((b) => (
                              <span
                                key={b.id}
                                className="baseline-mini-tag"
                                data-testid={`runs-list-baseline-badge-${b.id}`}
                              >
                                ★ {b.name}
                              </span>
                            ))}
                        </div>
                        {r.job_id && (
                          <div className="text-muted text-sm">
                            Job: {r.job_name || (r.config_snapshot?.job_name as string) || `#${r.job_id}`}
                          </div>
                        )}
                      </div>
                    </td>
                    <td>
                      <Badge status={r.status} size="sm" />
                    </td>
                    <td>
                      <span className="kind-badge">{tester}</span>
                    </td>
                    <td className="font-mono text-muted">
                      <div>{target}</div>
                      {models.length > 0 && (
                        <div className="text-sm text-muted">{models.join(', ')}</div>
                      )}
                    </td>
                    <td className="font-mono">{formatDuration(r)}</td>
                    <td className="text-muted">
                      {r.started_at ? (
                        <div>
                          <div>{formatDateTime(r.started_at)}</div>
                          {r.queued_at && r.queued_at !== r.started_at && (
                            <div
                              className="text-xs text-muted"
                              title={`Queued: ${formatDateTime(r.queued_at)}`}
                            >
                              queued {formatTime(r.queued_at)}
                            </div>
                          )}
                        </div>
                      ) : r.queued_at ? (
                        <div>
                          <div>{formatDateTime(r.queued_at)}</div>
                          <div className="text-xs text-muted">(queued)</div>
                        </div>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td>
                      {r.results && r.results.length > 1 ? (
                        <div className="result-pill-group">
                          {r.results.map((res, idx) => (
                            <span key={res.id} className="grouped-mini-pill">
                              <span className="font-mono text-muted">
                                {res.definition_name || `${idx + 1}:${res.tester}`}:
                              </span>{' '}
                              <Badge status={res.status} size="sm" />
                            </span>
                          ))}
                        </div>
                      ) : primaryRes ? (
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
