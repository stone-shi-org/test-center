import { useEffect, useState } from 'react'
import {
  api,
  type BaselineItem,
  type CompareResponse,
  type Run,
} from '../../api/index.ts'
import { Link, useRouter } from '../../router/index.ts'
import { Badge } from '../common/Badge.tsx'
import { Spinner } from '../common/Spinner.tsx'
import { ArtifactsCompareSection } from './ArtifactsCompareSection.tsx'
import { CaseDiffTable } from './CaseDiffTable.tsx'
import { ConfigDiffHeader } from './ConfigDiffHeader.tsx'
import { MetricDeltaTable } from './MetricDeltaTable.tsx'
import { MetricsChart } from './MetricsChart.tsx'
import {
  buildCompareUrlPath,
  parseCompareUrlParams,
  type CompareFilterMode,
} from './urlParams.ts'

function getRunTester(r: Run): string {
  if (r.results && r.results.length > 0 && r.results[0].tester) {
    return r.results[0].tester
  }
  return (r.config_snapshot?.tester as string) || 'unknown'
}

function getRunModel(r: Run): string | null {
  const cfg = (r.config_snapshot?.config as Record<string, unknown>) || {}
  const extra = (r.target_snapshot?.extra as Record<string, unknown>) || {}
  const explicitModels = Array.isArray(cfg.models)
    ? cfg.models.find((m) => typeof m === 'string' && m.trim())
    : undefined
  const model =
    cfg.model ||
    cfg.transcription_model ||
    explicitModels ||
    extra.model ||
    extra.transcription_model ||
    r.target_snapshot?.model
  return typeof model === 'string' && model.trim() ? model.trim() : null
}

export function CompareView() {
  const { route, navigate } = useRouter()
  const initial = parseCompareUrlParams(route.params)

  const [selectedRunIds, setSelectedRunIds] = useState<number[]>(initial.runIds)
  const [baselineRunId, setBaselineRunId] = useState<number | null>(
    initial.baselineRunId ?? (initial.runIds.length > 0 ? initial.runIds[0] : null)
  )
  const [hasExplicitReference, setHasExplicitReference] = useState<boolean>(
    initial.baselineRunId !== null || Boolean(initial.baselines && initial.baselines.length > 0)
  )
  const [threshold, setThreshold] = useState<number>(initial.threshold)
  const [filterMode, setFilterMode] = useState<CompareFilterMode>(initial.filterMode)
  const [suiteFilter, setSuiteFilter] = useState<string>(initial.suite)
  const [allowDifferentTesters, setAllowDifferentTesters] = useState<boolean>(false)

  // Baselines state (independent of 100-run history limit)
  const [savedBaselines, setSavedBaselines] = useState<BaselineItem[]>([])
  const [loadingBaselines, setLoadingBaselines] = useState<boolean>(true)
  const [baselineSearch, setBaselineSearch] = useState<string>('')
  const [baselineTesterFilter, setBaselineTesterFilter] = useState<string>('all')
  const [unknownBaselines, setUnknownBaselines] = useState<string[]>([])

  // Run Picker state
  const [historyRuns, setHistoryRuns] = useState<Run[]>([])
  const [loadingRuns, setLoadingRuns] = useState<boolean>(true)
  const [pickerExpanded, setPickerExpanded] = useState<boolean>(
    initial.runIds.length + (initial.baselines?.length || 0) < 2
  )
  const [pickerSearch, setPickerSearch] = useState<string>('')
  const [pickerTesterFilter, setPickerTesterFilter] = useState<string>('all')

  // Comparison API state derived by request key
  const [compareState, setCompareState] = useState<{
    key: string
    data: CompareResponse | null
    error: string | null
  } | null>(null)

  // Shareable URL copy state
  const [copiedUrl, setCopiedUrl] = useState<boolean>(false)

  // Load saved baselines for the Baselines Picker and resolve ?baselines= URL param
  useEffect(() => {
    let ignore = false
    const urlBaselines = parseCompareUrlParams(route.params).baselines || []
    const urlBaseRunId = parseCompareUrlParams(route.params).baselineRunId
    api.baselines
      .list()
      .then((data) => {
        if (ignore) return
        setSavedBaselines(data)
        if (urlBaselines.length > 0) {
          const resolvedIds: number[] = []
          const missing: string[] = []
          for (const rawName of urlBaselines) {
            const match = data.find(
              (b) => b.name.toLowerCase() === rawName.toLowerCase()
            )
            if (match) {
              if (!resolvedIds.includes(match.run_id)) {
                resolvedIds.push(match.run_id)
              }
            } else {
              missing.push(rawName)
            }
          }
          setUnknownBaselines(missing)
          if (resolvedIds.length > 0) {
            setSelectedRunIds((prev) => {
              const merged = [...resolvedIds]
              for (const id of prev) {
                if (!merged.includes(id)) {
                  merged.push(id)
                }
              }
              return merged
            })
            if (urlBaseRunId === null) {
              setBaselineRunId(resolvedIds[0])
              setHasExplicitReference(true)
            }
          }
        }
        setLoadingBaselines(false)
      })
      .catch(() => {
        if (!ignore) {
          setLoadingBaselines(false)
        }
      })
    return () => {
      ignore = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Load runs history for the Run Picker
  useEffect(() => {
    let ignore = false
    api.runs
      .list({ limit: 100 })
      .then((data) => {
        if (!ignore) {
          setHistoryRuns(data)
          setLoadingRuns(false)
        }
      })
      .catch(() => {
        if (!ignore) {
          setLoadingRuns(false)
        }
      })
    return () => {
      ignore = true
    }
  }, [])

  const getRunBaselineNames = (runId: number): string[] => {
    const names: string[] = []
    const addName = (n: string) => {
      if (n && !names.includes(n)) names.push(n)
    }
    for (const b of savedBaselines) {
      if (b.run_id === runId) addName(b.name)
    }
    const histRun = historyRuns.find((r) => r.id === runId)
    if (histRun?.baselines) {
      for (const b of histRun.baselines) addName(b.name)
    }
    const cmpRun = compareState?.data?.runs.find((r) => r.run_id === runId)
    if (cmpRun?.baselines) {
      for (const b of cmpRun.baselines) addName(b.name)
    }
    return names
  }

  const effectiveBaselineId =
    baselineRunId !== null && selectedRunIds.includes(baselineRunId)
      ? baselineRunId
      : selectedRunIds.length > 0
        ? selectedRunIds[0]
        : null

  // Partition selected runs into baseline names (for runs that are baselines) and regular run_ids
  const selectedBaselineNames: string[] = []
  const nonBaselineRunIds: number[] = []
  for (const id of selectedRunIds) {
    const bNames = getRunBaselineNames(id)
    if (bNames.length > 0) {
      if (!selectedBaselineNames.includes(bNames[0])) {
        selectedBaselineNames.push(bNames[0])
      }
    } else {
      nonBaselineRunIds.push(id)
    }
  }

  // Sync state changes to shareable URL
  const shareablePath = buildCompareUrlPath({
    baselines: selectedBaselineNames.length > 0 ? selectedBaselineNames : undefined,
    runIds: nonBaselineRunIds,
    baselineRunId: effectiveBaselineId,
    threshold,
    filterMode,
    suite: suiteFilter,
  })

  useEffect(() => {
    if (loadingBaselines && initial.baselines && initial.baselines.length > 0) {
      return
    }
    const currentFullPath = buildCompareUrlPath(parseCompareUrlParams(route.params))
    if (currentFullPath !== shareablePath) {
      navigate(shareablePath, { replace: true })
    }
  }, [shareablePath, route.params, navigate, loadingBaselines, initial.baselines])

  const requestKey =
    selectedRunIds.length >= 2
      ? `${selectedRunIds.join(',')}:${effectiveBaselineId}:${threshold}:${allowDifferentTesters}`
      : null

  // Fetch comparison whenever 2+ runs are selected
  useEffect(() => {
    if (!requestKey) {
      return
    }

    let ignore = false

    api.compare
      .get({
        run_ids: selectedRunIds,
        baseline_run_id: effectiveBaselineId,
        threshold,
        allow_different_testers: allowDifferentTesters,
      })
      .then((res) => {
        if (!ignore) {
          setCompareState({ key: requestKey, data: res, error: null })
        }
      })
      .catch((err: unknown) => {
        if (!ignore) {
          setCompareState({
            key: requestKey,
            data: null,
            error: err instanceof Error ? err.message : String(err),
          })
        }
      })

    return () => {
      ignore = true
    }
  }, [requestKey, selectedRunIds, effectiveBaselineId, threshold, allowDifferentTesters])

  const loadingCompare = requestKey !== null && compareState?.key !== requestKey
  const compareData =
    requestKey !== null && compareState?.key === requestKey ? compareState.data : null
  const compareError =
    requestKey !== null && compareState?.key === requestKey ? compareState.error : null

  const handleToggleRun = (runId: number, isFromBaselineSection = false) => {
    setSelectedRunIds((prev) => {
      if (prev.includes(runId)) {
        const next = prev.filter((id) => id !== runId)
        if (baselineRunId === runId) {
          setBaselineRunId(next.length > 0 ? next[0] : null)
          setHasExplicitReference(false)
        }
        return next
      } else {
        const next = [...prev, runId]
        const isBaselineRun = isFromBaselineSection || getRunBaselineNames(runId).length > 0
        if (isBaselineRun && (!hasExplicitReference || baselineRunId === null)) {
          setBaselineRunId(runId)
          setHasExplicitReference(true)
        } else if (next.length === 1) {
          setBaselineRunId(runId)
        }
        return next
      }
    })
  }

  const handleSelectBaseline = (runId: number) => {
    if (!selectedRunIds.includes(runId)) {
      setSelectedRunIds((prev) => [runId, ...prev])
    }
    setBaselineRunId(runId)
    setHasExplicitReference(true)
  }

  const handleCopyShareableUrl = async () => {
    const fullUrl =
      typeof window !== 'undefined'
        ? `${window.location.origin}${shareablePath}`
        : shareablePath
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(fullUrl)
      }
    } catch {
      // Fallback if clipboard API is restricted
    }
    setCopiedUrl(true)
    setTimeout(() => setCopiedUrl(false), 2000)
  }

  // Determine active tester from first selected run to assist in Run Picker filtering
  const firstSelectedRun =
    selectedRunIds.length > 0
      ? historyRuns.find((r) => r.id === selectedRunIds[0])
      : undefined
  const firstSelectedBaseline =
    selectedRunIds.length > 0
      ? savedBaselines.find((b) => b.run_id === selectedRunIds[0])
      : undefined
  const activeTester = firstSelectedRun
    ? getRunTester(firstSelectedRun)
    : firstSelectedBaseline?.tester || null

  const uniqueTesters = Array.from(new Set(historyRuns.map((r) => getRunTester(r))))
  const uniqueBaselineTesters = Array.from(
    new Set(
      savedBaselines
        .map((b) => b.tester)
        .filter((t): t is string => Boolean(t))
    )
  )

  const filteredSavedBaselines = savedBaselines.filter((b) => {
    if (baselineTesterFilter !== 'all' && b.tester !== baselineTesterFilter) {
      return false
    }
    if (baselineSearch.trim()) {
      const q = baselineSearch.toLowerCase()
      return (
        b.name.toLowerCase().includes(q) ||
        (b.description || '').toLowerCase().includes(q) ||
        (b.tester || '').toLowerCase().includes(q) ||
        (b.target || '').toLowerCase().includes(q) ||
        (b.model || '').toLowerCase().includes(q) ||
        `#${b.run_id}`.includes(q) ||
        String(b.run_id).includes(q)
      )
    }
    return true
  })

  const filteredPickerRuns = historyRuns.filter((r) => {
    const tester = getRunTester(r)
    if (pickerTesterFilter !== 'all' && tester !== pickerTesterFilter) {
      return false
    }
    if (pickerSearch.trim()) {
      const q = pickerSearch.toLowerCase()
      const targetName = ((r.target_snapshot?.name as string) || '').toLowerCase()
      const modelName = (getRunModel(r) || '').toLowerCase()
      const matchesId = `#${r.id}`.includes(q) || String(r.id).includes(q)
      const matchesBaseline = getRunBaselineNames(r.id).some((n) =>
        n.toLowerCase().includes(q)
      )
      return (
        matchesId ||
        matchesBaseline ||
        tester.toLowerCase().includes(q) ||
        targetName.includes(q) ||
        modelName.includes(q)
      )
    }
    return true
  })

  // Apply Changed Only / Regressed Only filter to metrics and cases
  const allMetrics = compareData?.metrics || []
  const filteredMetrics = allMetrics.filter((m) => {
    if (filterMode === 'regressed') return m.has_regression
    if (filterMode === 'changed') return m.has_change
    return true
  })

  const allCases = compareData?.cases || []
  const availableSuites = Array.from(new Set(allCases.map((c) => c.suite)))
  const filteredCases = allCases.filter((c) => {
    if (filterMode === 'regressed') return c.has_regression
    if (filterMode === 'changed') return c.has_change
    return true
  })

  return (
    <div className="view-container compare-view" data-testid="compare-view">
      {/* Page Header */}
      <div className="view-header">
        <div>
          <h2>Compare Test Runs</h2>
          <p className="view-subtitle">
            Select 2 or more runs to compare top-level metrics, regression flags, per-case diffs, and artifacts side by side.
          </p>
        </div>

        <div className="header-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setPickerExpanded((prev) => !prev)}
            data-testid="toggle-run-picker-btn"
          >
            {pickerExpanded
              ? `Hide Run Picker (${selectedRunIds.length} selected)`
              : `📋 Pick Runs (${selectedRunIds.length} selected)`}
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleCopyShareableUrl}
            data-testid="copy-shareable-url-btn"
            title="Copy shareable comparison URL with selected run IDs"
          >
            {copiedUrl ? '✓ Link Copied!' : '🔗 Copy Shareable URL'}
          </button>
        </div>
      </div>

      {unknownBaselines.length > 0 && (
        <div
          className="banner error-banner"
          style={{ marginBottom: '12px' }}
          data-testid="unknown-baselines-banner"
        >
          Unknown baseline{unknownBaselines.length === 1 ? '' : 's'}:{' '}
          <strong>{unknownBaselines.join(', ')}</strong>. Check the name or pick from saved baselines below.
        </div>
      )}

      {/* Shareable URL & Controls Bar */}
      <div className="card compare-toolbar-card">
        <div className="compare-toolbar-row">
          {/* Selected Runs Pills */}
          <div className="compare-selected-group">
            <span className="filter-label font-semibold">Selected Runs:</span>
            {selectedRunIds.length === 0 ? (
              <span className="text-muted">None selected — pick 2+ runs below</span>
            ) : (
              <div className="selected-run-pills">
                {selectedRunIds.map((id) => {
                  const isBase = id === effectiveBaselineId
                  const runObj = historyRuns.find((r) => r.id === id)
                  const baseObj = savedBaselines.find((b) => b.run_id === id)
                  const model = (runObj ? getRunModel(runObj) : null) || baseObj?.model || null
                  const baselineNames = getRunBaselineNames(id)
                  return (
                    <span
                      key={id}
                      className={`selected-run-pill font-mono ${isBase ? 'is-baseline' : ''}`}
                      data-testid={`selected-run-pill-${id}`}
                    >
                      <Link href={`/runs/${id}`} className="pill-run-link">
                        #{id}
                      </Link>
                      {baselineNames.map((bName) => (
                        <span
                          key={bName}
                          className="pill-baseline-name"
                          data-testid={`pill-baseline-name-${id}`}
                          style={{
                            color: '#fbbf24',
                            fontWeight: 600,
                            marginLeft: '4px',
                          }}
                        >
                          ★ {bName}
                        </span>
                      ))}
                      {model && <span className="pill-model-text">({model})</span>}
                      {isBase ? (
                        <span className="baseline-mini-tag">Reference run</span>
                      ) : (
                        <button
                          type="button"
                          className="pill-base-btn"
                          onClick={() => handleSelectBaseline(id)}
                          title={`Set Run #${id} as Reference run`}
                        >
                          Set Reference
                        </button>
                      )}
                      <button
                        type="button"
                        className="pill-remove-btn"
                        aria-label={`Remove run #${id}`}
                        onClick={() => handleToggleRun(id)}
                      >
                        ×
                      </button>
                    </span>
                  )
                })}
              </div>
            )}
          </div>

          {/* Shareable URL display */}
          <div className="shareable-url-box" data-testid="shareable-url-box">
            <span className="filter-label">Shareable URL:</span>
            <code className="shareable-url-code font-mono">{shareablePath}</code>
          </div>
        </div>

        {/* Controls Row: Reference Run Picker, Threshold, Filter Mode */}
        <div className="compare-controls-row">
          <div className="filter-group">
            <label htmlFor="compare-baseline-select" className="filter-label">
              Reference run:
            </label>
            <select
              id="compare-baseline-select"
              aria-label="Reference run"
              className="filter-select font-mono"
              value={effectiveBaselineId ?? ''}
              disabled={selectedRunIds.length === 0}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10)
                if (!Number.isNaN(val)) handleSelectBaseline(val)
              }}
              data-testid="baseline-select"
            >
              {selectedRunIds.length === 0 && <option value="">—</option>}
              {selectedRunIds.map((id) => {
                const r = historyRuns.find((item) => item.id === id)
                const bObj = savedBaselines.find((item) => item.run_id === id)
                const meta = compareData?.runs.find((item) => item.run_id === id)
                const labelTarget =
                  meta?.target_name ||
                  (r?.target_snapshot?.name as string) ||
                  bObj?.target ||
                  ''
                const labelModel =
                  meta?.model || (r ? getRunModel(r) : '') || bObj?.model || ''
                const details = [labelTarget, labelModel].filter(Boolean).join(' • ')
                const bNames = getRunBaselineNames(id)
                return (
                  <option key={id} value={id}>
                    Run #{id}
                    {bNames.length > 0 ? ` ★ ${bNames.join(', ')}` : ''}
                    {details ? ` (${details})` : ''}
                  </option>
                )
              })}
            </select>
          </div>

          <div className="filter-group">
            <label htmlFor="compare-threshold-input" className="filter-label">
              Threshold (%):
            </label>
            <input
              id="compare-threshold-input"
              aria-label="Threshold (%)"
              type="number"
              min="0"
              max="100"
              step="0.5"
              className="filter-select font-mono"
              style={{ width: '80px' }}
              value={threshold}
              onChange={(e) => {
                const v = parseFloat(e.target.value)
                setThreshold(Number.isNaN(v) || v < 0 ? 0 : v)
              }}
              data-testid="threshold-input"
            />
          </div>

          <div className="filter-group" role="group" aria-label="Diff Filter">
            <span className="filter-label">Show:</span>
            <div className="btn-group">
              <button
                type="button"
                className={`btn btn-sm ${filterMode === 'all' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setFilterMode('all')}
                data-testid="filter-all-btn"
              >
                All
              </button>
              <button
                type="button"
                className={`btn btn-sm ${filterMode === 'changed' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setFilterMode('changed')}
                data-testid="filter-changed-btn"
              >
                Changed Only
                {compareData ? ` (${compareData.summary.metrics_changed + compareData.summary.cases_changed})` : ''}
              </button>
              <button
                type="button"
                className={`btn btn-sm ${filterMode === 'regressed' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setFilterMode('regressed')}
                data-testid="filter-regressed-btn"
              >
                ⚠ Regressed Only
                {compareData ? ` (${compareData.summary.metrics_regressed + compareData.summary.cases_regressed})` : ''}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Run Picker Panel */}
      {(pickerExpanded || selectedRunIds.length < 2) && (
        <div className="card run-picker-card" data-testid="run-picker-panel">
          {/* Baselines Section (independent of 100-run limit) */}
          <div
            className="picker-baselines-section"
            data-testid="picker-baselines-section"
            style={{
              marginBottom: '20px',
              paddingBottom: '16px',
              borderBottom: '1px solid var(--border-subtle, #334155)',
            }}
          >
            <div className="compare-section-header" style={{ marginBottom: '8px' }}>
              <div>
                <h3 className="section-title" style={{ marginBottom: '2px' }}>
                  ★ Baselines
                </h3>
                <span className="section-desc">
                  Named baselines preserved across all runs (independent of recent run history limit). Picking a baseline sets it as the Reference run by default.
                </span>
              </div>
            </div>

            <div className="filter-bar" style={{ marginBottom: '10px' }}>
              <div className="search-input-wrapper">
                <span className="search-icon">🔍</span>
                <input
                  type="text"
                  className="search-input"
                  placeholder="Search baselines by name..."
                  aria-label="Search baselines by name"
                  value={baselineSearch}
                  onChange={(e) => setBaselineSearch(e.target.value)}
                  data-testid="baseline-picker-search"
                />
              </div>

              <div className="filter-group">
                <label htmlFor="baseline-picker-tester" className="filter-label">
                  Tester:
                </label>
                <select
                  id="baseline-picker-tester"
                  aria-label="Filter baselines by tester"
                  className="filter-select"
                  value={baselineTesterFilter}
                  onChange={(e) => setBaselineTesterFilter(e.target.value)}
                  data-testid="baseline-picker-tester"
                >
                  <option value="all">All Testers ({uniqueBaselineTesters.length})</option>
                  {uniqueBaselineTesters.map((t) => (
                    <option key={t} value={t}>
                      {t.toUpperCase()}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {loadingBaselines ? (
              <div className="text-muted empty-inline-msg">Loading baselines...</div>
            ) : filteredSavedBaselines.length === 0 ? (
              <div className="text-muted empty-inline-msg" data-testid="no-baselines-msg">
                No saved baselines match the filter criteria.
              </div>
            ) : (
              <div className="table-responsive run-picker-table-wrapper">
                <table className="data-table" data-testid="baselines-picker-table">
                  <thead>
                    <tr>
                      <th style={{ width: '44px' }}>Pick</th>
                      <th>Baseline</th>
                      <th>Run</th>
                      <th>Result</th>
                      <th>Tester</th>
                      <th>Target</th>
                      <th>Model</th>
                      <th>Reference run</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredSavedBaselines.map((b) => {
                      const isChecked = selectedRunIds.includes(b.run_id)
                      const isBase = b.run_id === effectiveBaselineId
                      const tester = b.tester || 'unknown'
                      const isTesterMismatch =
                        !allowDifferentTesters &&
                        activeTester !== null &&
                        !isChecked &&
                        tester !== activeTester

                      return (
                        <tr
                          key={b.id}
                          className={`${isChecked ? 'row-selected' : ''} ${isTesterMismatch ? 'row-disabled' : ''}`}
                          data-testid={`picker-baseline-row-${b.name}`}
                        >
                          <td>
                            <input
                              type="checkbox"
                              aria-label={`Select baseline ${b.name}`}
                              checked={isChecked}
                              disabled={isTesterMismatch}
                              onChange={() => handleToggleRun(b.run_id, true)}
                              data-testid={`pick-baseline-checkbox-${b.id}`}
                            />
                          </td>
                          <td>
                            <span
                              className="font-semibold"
                              style={{ color: '#fbbf24' }}
                            >
                              ★ {b.name}
                            </span>
                            {b.description && (
                              <div className="text-muted" style={{ fontSize: '11px' }}>
                                {b.description}
                              </div>
                            )}
                          </td>
                          <td className="font-mono font-bold">#{b.run_id}</td>
                          <td>
                            <Badge status={b.status || 'completed'} size="sm" />
                          </td>
                          <td>
                            <span className="kind-badge">{tester}</span>
                          </td>
                          <td className="font-mono">{b.target || 'None'}</td>
                          <td className="font-mono text-muted">{b.model || '—'}</td>
                          <td>
                            {isChecked ? (
                              isBase ? (
                                <span className="baseline-badge">REFERENCE RUN</span>
                              ) : (
                                <button
                                  type="button"
                                  className="btn btn-sm btn-secondary"
                                  onClick={() => handleSelectBaseline(b.run_id)}
                                >
                                  Set Reference run
                                </button>
                              )
                            ) : (
                              <button
                                type="button"
                                className="btn btn-sm btn-secondary"
                                disabled={isTesterMismatch}
                                onClick={() => handleToggleRun(b.run_id, true)}
                                data-testid={`pick-baseline-btn-${b.name}`}
                              >
                                + Pick Baseline
                              </button>
                            )}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="compare-section-header">
            <div>
              <h3 className="section-title" style={{ marginBottom: '2px' }}>
                Pick Runs from History
              </h3>
              <span className="section-desc">
                Select at least 2 runs of the same tester to compare side by side.
                {activeTester && !allowDifferentTesters
                  ? ` Active tester: ${activeTester.toUpperCase()}`
                  : ''}
              </span>
            </div>
            {selectedRunIds.length > 0 && (
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() => {
                  setSelectedRunIds([])
                  setBaselineRunId(null)
                  setHasExplicitReference(false)
                }}
              >
                Clear Selection
              </button>
            )}
          </div>

          <div className="filter-bar" style={{ marginBottom: '12px' }}>
            <div className="search-input-wrapper">
              <span className="search-icon">🔍</span>
              <input
                type="text"
                className="search-input"
                placeholder="Search runs by ID, target, model, or tester..."
                aria-label="Search runs in picker"
                value={pickerSearch}
                onChange={(e) => setPickerSearch(e.target.value)}
              />
            </div>

            <div className="filter-group">
              <label htmlFor="picker-tester-filter" className="filter-label">
                Tester:
              </label>
              <select
                id="picker-tester-filter"
                className="filter-select"
                value={pickerTesterFilter}
                onChange={(e) => setPickerTesterFilter(e.target.value)}
              >
                <option value="all">All Testers ({uniqueTesters.length})</option>
                {uniqueTesters.map((t) => (
                  <option key={t} value={t}>
                    {t.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <label className="checkbox-label" style={{ alignItems: 'center', fontSize: '12px' }}>
              <input
                type="checkbox"
                checked={allowDifferentTesters}
                onChange={(e) => setAllowDifferentTesters(e.target.checked)}
              />
              <span>Allow different testers</span>
            </label>
          </div>

          {loadingRuns ? (
            <Spinner message="Loading run history..." />
          ) : filteredPickerRuns.length === 0 ? (
            <div className="text-muted empty-inline-msg">
              No runs found matching the filter criteria.
            </div>
          ) : (
            <div className="table-responsive run-picker-table-wrapper">
              <table className="data-table" data-testid="run-picker-table">
                <thead>
                  <tr>
                    <th style={{ width: '44px' }}>Pick</th>
                    <th>Run</th>
                    <th>Result</th>
                    <th>Tester</th>
                    <th>Target</th>
                    <th>Model</th>
                    <th>Executed</th>
                    <th>Reference run</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPickerRuns.map((r) => {
                    const isChecked = selectedRunIds.includes(r.id)
                    const isBase = r.id === effectiveBaselineId
                    const tester = getRunTester(r)
                    const target = (r.target_snapshot?.name as string) || 'None'
                    const model = getRunModel(r) || '—'
                    const primaryRes = r.results && r.results.length > 0 ? r.results[0] : null
                    const isTesterMismatch =
                      !allowDifferentTesters &&
                      activeTester !== null &&
                      !isChecked &&
                      tester !== activeTester
                    const runBaselineNames = getRunBaselineNames(r.id)

                    return (
                      <tr
                        key={r.id}
                        className={`${isChecked ? 'row-selected' : ''} ${isTesterMismatch ? 'row-disabled' : ''}`}
                        data-testid={`picker-row-${r.id}`}
                      >
                        <td>
                          <input
                            type="checkbox"
                            aria-label={`Select run #${r.id}`}
                            checked={isChecked}
                            disabled={isTesterMismatch}
                            onChange={() => handleToggleRun(r.id)}
                          />
                        </td>
                        <td className="font-mono font-bold">
                          #{r.id}
                          {runBaselineNames.map((bName) => (
                            <span
                              key={bName}
                              style={{
                                marginLeft: '6px',
                                color: '#fbbf24',
                                fontSize: '11px',
                                fontWeight: 600,
                              }}
                            >
                              ★ {bName}
                            </span>
                          ))}
                        </td>
                        <td>
                          <Badge
                            status={primaryRes ? primaryRes.status : r.status}
                            size="sm"
                          />
                        </td>
                        <td>
                          <span className="kind-badge">{tester}</span>
                        </td>
                        <td className="font-mono">{target}</td>
                        <td className="font-mono text-muted">{model}</td>
                        <td className="text-muted">
                          {r.started_at
                            ? new Date(r.started_at).toLocaleString()
                            : r.queued_at
                              ? new Date(r.queued_at).toLocaleString()
                              : '—'}
                        </td>
                        <td>
                          {isChecked ? (
                            isBase ? (
                              <span className="baseline-badge">REFERENCE RUN</span>
                            ) : (
                              <button
                                type="button"
                                className="btn btn-sm btn-secondary"
                                onClick={() => handleSelectBaseline(r.id)}
                              >
                                Set Reference run
                              </button>
                            )
                          ) : (
                            <button
                              type="button"
                              className="btn btn-sm btn-secondary"
                              disabled={isTesterMismatch}
                              onClick={() => handleToggleRun(r.id)}
                            >
                              + Add Run
                            </button>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Comparison Content */}
      {selectedRunIds.length < 2 ? (
        <div className="empty-state" data-testid="compare-empty-prompt">
          <div className="empty-icon">⚖️</div>
          <h3>
            {selectedRunIds.length === 0
              ? 'Select 2 or more runs to compare'
              : `Run #${selectedRunIds[0]} selected — pick at least 1 more run`}
          </h3>
          <p>
            Choose runs from the baselines or history table above to view aligned metric deltas, regression flags, per-case diffs, and side-by-side artifacts.
          </p>
        </div>
      ) : loadingCompare ? (
        <Spinner message="Computing metric deltas and aligning test cases..." />
      ) : compareError ? (
        <div className="banner error-banner" data-testid="compare-error-banner">
          {compareError}
        </div>
      ) : compareData ? (
        <div className="compare-results-stack" data-testid="compare-results">
          {/* Summary Banner */}
          <div
            className={`card compare-summary-banner ${
              compareData.summary.has_regression
                ? 'summary-has-regression'
                : compareData.summary.has_improvement
                  ? 'summary-has-improvement'
                  : ''
            }`}
            data-testid="compare-summary-banner"
          >
            <div className="compare-summary-status">
              <span className="compare-summary-icon">
                {compareData.summary.has_regression ? '⚠️' : '✅'}
              </span>
              <div>
                <h4 style={{ margin: 0 }}>
                  {compareData.summary.has_regression
                    ? `Regressions Detected (${compareData.summary.metrics_regressed} metric${compareData.summary.metrics_regressed === 1 ? '' : 's'}, ${compareData.summary.cases_regressed} case${compareData.summary.cases_regressed === 1 ? '' : 's'})`
                    : 'No Regressions Detected'}
                </h4>
                <span className="text-muted" style={{ fontSize: '12px' }}>
                  Comparing {compareData.runs.length} runs for tester{' '}
                  <strong className="font-mono">{compareData.tester.toUpperCase()}</strong> against
                  Reference run <strong>Run #{compareData.baseline_run_id}</strong>
                  {compareData.threshold > 0
                    ? ` (threshold: ±${compareData.threshold}%)`
                    : ''}
                </span>
              </div>
            </div>

            <div className="compare-summary-stats">
              <div className="summary-stat-pill">
                <span className="text-muted">Metrics:</span>
                <strong>{compareData.summary.total_metrics}</strong>
                {compareData.summary.metrics_regressed > 0 && (
                  <span className="delta-badge delta-regression">
                    {compareData.summary.metrics_regressed} regressed
                  </span>
                )}
                {compareData.summary.metrics_improved > 0 && (
                  <span className="delta-badge delta-improvement">
                    {compareData.summary.metrics_improved} improved
                  </span>
                )}
              </div>

              <div className="summary-stat-pill">
                <span className="text-muted">Cases:</span>
                <strong>{compareData.summary.total_cases}</strong>
                {compareData.summary.cases_regressed > 0 && (
                  <span className="delta-badge delta-regression">
                    {compareData.summary.cases_regressed} regressed
                  </span>
                )}
                {compareData.summary.cases_improved > 0 && (
                  <span className="delta-badge delta-improvement">
                    {compareData.summary.cases_improved} improved
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* 1. Config & Target Diff Header */}
          <ConfigDiffHeader
            runs={compareData.runs}
            configDiff={compareData.config_diff}
            onSelectBaseline={handleSelectBaseline}
            onRemoveRun={handleToggleRun}
          />

          {/* 2. Bar / Line Charts for Key Metrics */}
          <MetricsChart metrics={allMetrics} runs={compareData.runs} />

          {/* 3. Metric Delta Table */}
          <MetricDeltaTable
            metrics={filteredMetrics}
            totalMetricsCount={allMetrics.length}
            runs={compareData.runs}
          />

          {/* 4. Per-Case Diff View */}
          <CaseDiffTable
            cases={filteredCases}
            totalCasesCount={allCases.length}
            runs={compareData.runs}
            suiteFilter={suiteFilter}
            onSuiteFilterChange={setSuiteFilter}
            availableSuites={availableSuites}
          />

          {/* 5. Side-by-Side Artifacts */}
          <ArtifactsCompareSection runs={compareData.runs} />
        </div>
      ) : null}
    </div>
  )
}
