import { useState } from 'react'
import type { Run, RunResult, TrendMetricOption, TrendPoint, TrendSeries } from '../../api/types.ts'
import { Badge } from '../common/Badge.tsx'

const SERIES_COLORS = [
  '#38bdf8', // sky
  '#10b981', // emerald
  '#f59e0b', // amber
  '#a855f7', // purple
  '#f43f5e', // rose
  '#3b82f6', // blue
  '#14b8a6', // teal
  '#eab308', // yellow
]

export type MetricCategoryId = 'latency' | 'throughput' | 'accuracy' | 'cases' | 'other'
export type MetricFilterTab = 'key' | 'all' | MetricCategoryId

export interface MetricCategoryMeta {
  id: MetricCategoryId
  label: string
  shortLabel: string
}

export const METRIC_CATEGORIES: MetricCategoryMeta[] = [
  { id: 'latency', label: 'Latency & Timing', shortLabel: 'Latency' },
  { id: 'throughput', label: 'Throughput & Speed', shortLabel: 'Throughput' },
  { id: 'accuracy', label: 'Accuracy & Quality', shortLabel: 'Accuracy' },
  { id: 'cases', label: 'Cases & Coverage', shortLabel: 'Cases' },
  { id: 'other', label: 'System & Other', shortLabel: 'Other' },
]

const HEADLINE_METRIC_NAMES = new Set([
  'pass_rate',
  'success_rate',
  'latency_mean_ms',
  'latency_p50_ms',
  'latency_p95_ms',
  'ttft_ms',
  'ttft_mean_ms',
  'ttft_p50_ms',
  'ttft_p95_ms',
  'ttfp_mean_ms',
  'ttfp_p50_ms',
  'grpc_ttfp_mean_ms',
  'tts_ttfb_mean_ms',
  'rtf',
  'rtf_mean',
  'grpc_rtf_mean',
  'tokens_per_sec',
  'tps',
  'decode_tps_mean',
  'throughput',
  'throughput_seq_docs_per_sec',
  'wer',
  'wer_pct',
  'http_wer',
  'streaming_wer',
  'cpwer',
  'cpwer_pct',
  'der',
  'der_pct',
  'speaker_count_error',
  'judge_score_mean',
  'judge_pass_rate',
  'judge_mean',
  'score',
  'solve_at_1',
  'solve_at_k',
  'test_case_pass_rate',
  'recall@1',
  'recall_at_1',
  'mrr',
])

const COMPACT_PILL_THRESHOLD = 12
const MAX_COLLAPSED_PILLS = 12

export function categorizeMetric(
  metric: Pick<TrendMetricOption, 'name' | 'unit'>
): MetricCategoryId {
  const n = metric.name.toLowerCase()
  const u = (metric.unit || '').toLowerCase()

  if (
    n === 'pass_rate' ||
    n === 'success_rate' ||
    n.includes('cases') ||
    n.includes('meetings') ||
    n.includes('cache_hits') ||
    n.includes('cache_misses') ||
    u === 'cases' ||
    u === 'meetings'
  ) {
    return 'cases'
  }

  if (
    /(^|_)(wer|cpwer|der)($|_)/.test(n) ||
    n.startsWith('speaker_') ||
    n.startsWith('judge_') ||
    n === 'judge_score' ||
    n === 'score' ||
    n.startsWith('score_') ||
    n.startsWith('solve_at_') ||
    n === 'test_case_pass_rate' ||
    n.startsWith('recall') ||
    n === 'mrr' ||
    n === 'mean_rank' ||
    n.includes('similarity') ||
    n.startsWith('consistency_') ||
    n.startsWith('norm_') ||
    n.includes('accuracy') ||
    n.includes('cer') ||
    n.includes('bleu') ||
    n.includes('rouge')
  ) {
    return 'accuracy'
  }

  if (
    /(^|_)(rtf|tps|throughput)($|_)/.test(n) ||
    n === 'tokens_per_sec' ||
    n.endsWith('_per_sec') ||
    u === 'x' ||
    u === 'tokens/s' ||
    u === 'tok/s' ||
    u === 'docs/s' ||
    u === 'req/s'
  ) {
    return 'throughput'
  }

  if (
    n.includes('latency') ||
    n.includes('ttft') ||
    n.includes('ttfp') ||
    n.includes('ttfb') ||
    n.includes('duration') ||
    n.endsWith('_wall_ms') ||
    n.endsWith('_ms') ||
    n.endsWith('_sec') ||
    u === 'ms' ||
    u === 's' ||
    u === 'sec'
  ) {
    return 'latency'
  }

  return 'other'
}

export function isKeyMetric(metric: Pick<TrendMetricOption, 'name'>): boolean {
  return HEADLINE_METRIC_NAMES.has(metric.name.toLowerCase())
}

export function groupMetricsByCategory(
  metrics: TrendMetricOption[]
): Array<{ category: MetricCategoryMeta; metrics: TrendMetricOption[] }> {
  const buckets = new Map<MetricCategoryId, TrendMetricOption[]>()
  for (const cat of METRIC_CATEGORIES) {
    buckets.set(cat.id, [])
  }
  for (const m of metrics) {
    const catId = categorizeMetric(m)
    buckets.get(catId)!.push(m)
  }
  return METRIC_CATEGORIES.map((category) => ({
    category,
    metrics: buckets.get(category.id) || [],
  })).filter((g) => g.metrics.length > 0)
}

function parseMetricEntry(raw: unknown): {
  value: number | null
  unit: string | null
  better: 'higher' | 'lower' | null
} {
  if (raw === null || raw === undefined) {
    return { value: null, unit: null, better: null }
  }
  if (typeof raw === 'number' && !Number.isNaN(raw)) {
    return { value: raw, unit: null, better: null }
  }
  if (typeof raw === 'object') {
    const obj = raw as Record<string, unknown>
    const val = typeof obj.value === 'number' && !Number.isNaN(obj.value) ? obj.value : null
    const unit = typeof obj.unit === 'string' ? obj.unit : null
    const better =
      obj.better === 'higher' || obj.better === 'lower' ? obj.better : null
    return { value: val, unit, better }
  }
  return { value: null, unit: null, better: null }
}

function resolveResultTargetAndModel(
  run: Run,
  res: RunResult
): {
  targetId: number | null
  targetName: string | null
  model: string | null
  seriesLabel: string
} {
  let targetId: number | null =
    typeof res.target_id === 'number' ? res.target_id : null
  let targetName: string | null =
    typeof res.target_name === 'string' && res.target_name.trim()
      ? res.target_name.trim()
      : null
  let model: string | null =
    typeof res.model === 'string' && res.model.trim() ? res.model.trim() : null

  const cfgSnap = run.config_snapshot || {}
  const tgtSnap = run.target_snapshot || {}

  // Check definition steps if multi-definition job
  let matchedDef: Record<string, unknown> | null = null
  if (Array.isArray(cfgSnap.definitions)) {
    matchedDef =
      (cfgSnap.definitions.find(
        (d) =>
          d &&
          typeof d === 'object' &&
          ((res.test_definition_id &&
            (d.test_definition_id === res.test_definition_id ||
              d.id === res.test_definition_id)) ||
            d.tester === res.tester)
      ) as Record<string, unknown> | undefined) || null
  }

  if (targetId === null) {
    const stepTgt = matchedDef?.target_snapshot as Record<string, unknown> | undefined
    if (typeof stepTgt?.id === 'number') {
      targetId = stepTgt.id
    } else if (typeof tgtSnap.id === 'number') {
      targetId = tgtSnap.id
    }
  }

  if (!targetName) {
    const stepTgt = matchedDef?.target_snapshot as Record<string, unknown> | undefined
    if (typeof stepTgt?.name === 'string' && stepTgt.name.trim()) {
      targetName = stepTgt.name.trim()
    } else if (typeof tgtSnap.name === 'string' && tgtSnap.name.trim()) {
      targetName = tgtSnap.name.trim()
    }
  }

  if (!model) {
    const stepCfg = (matchedDef?.config || cfgSnap.config || {}) as Record<string, unknown>
    const stepTgt = (matchedDef?.target_snapshot || tgtSnap || {}) as Record<string, unknown>
    const stepExtra = (stepTgt.extra || {}) as Record<string, unknown>
    const explicitModels = Array.isArray(stepCfg.models)
      ? stepCfg.models.find((m) => typeof m === 'string' && m.trim())
      : undefined
    const candidate =
      stepCfg.model ||
      stepCfg.transcription_model ||
      explicitModels ||
      stepExtra.model ||
      stepExtra.transcription_model ||
      stepTgt.model
    if (typeof candidate === 'string' && candidate.trim()) {
      model = candidate.trim()
    } else if (Array.isArray(res.cases)) {
      const caseModels: string[] = []
      for (const c of res.cases) {
        const cm = c.detail?.model
        if (typeof cm === 'string' && cm.trim() && !caseModels.includes(cm.trim())) {
          caseModels.push(cm.trim())
        }
      }
      if (caseModels.length > 0) {
        model = caseModels[0]
      }
    }
  }

  let seriesLabel: string
  if (targetName && model) {
    seriesLabel = `${targetName} / ${model}`
  } else if (targetName) {
    seriesLabel = targetName
  } else if (model) {
    seriesLabel = model
  } else {
    seriesLabel = `${res.tester} (default)`
  }

  return { targetId, targetName, model, seriesLabel }
}

export function extractTrendsFromRuns(
  runs: Run[],
  selectedMetricName?: string | null
): {
  metric: string | null
  unit: string | null
  better: 'higher' | 'lower' | null
  availableMetrics: TrendMetricOption[]
  series: TrendSeries[]
  points: TrendPoint[]
} {
  // Sort chronologically (oldest first)
  const chronologicalRuns = [...runs].sort((a, b) => {
    const tA = new Date(a.started_at || a.queued_at || a.finished_at || 0).getTime()
    const tB = new Date(b.started_at || b.queued_at || b.finished_at || 0).getTime()
    if (tA !== tB) return tA - tB
    return a.id - b.id
  })

  const metricCatalog = new Map<string, TrendMetricOption>()
  const items: Array<{
    run: Run
    res: RunResult
    targetId: number | null
    targetName: string | null
    model: string | null
    seriesLabel: string
    parsedMetrics: Map<
      string,
      { value: number; unit: string | null; better: 'higher' | 'lower' | null }
    >
  }> = []

  for (const run of chronologicalRuns) {
    for (const res of run.results || []) {
      const { targetId, targetName, model, seriesLabel } = resolveResultTargetAndModel(run, res)
      const parsedMetrics = new Map<
        string,
        { value: number; unit: string | null; better: 'higher' | 'lower' | null }
      >()

      for (const [mName, rawEntry] of Object.entries(res.metrics || {})) {
        const parsed = parseMetricEntry(rawEntry)
        if (parsed.value === null) continue
        parsedMetrics.set(mName, {
          value: parsed.value,
          unit: parsed.unit,
          better: parsed.better,
        })

        const existing = metricCatalog.get(mName)
        if (!existing) {
          metricCatalog.set(mName, {
            name: mName,
            unit: parsed.unit,
            better: parsed.better,
            testers: [res.tester],
            count: 1,
          })
        } else {
          if (!existing.unit && parsed.unit) existing.unit = parsed.unit
          if (!existing.better && parsed.better) existing.better = parsed.better
          if (!existing.testers.includes(res.tester)) existing.testers.push(res.tester)
          existing.count += 1
        }
      }

      if (parsedMetrics.size > 0) {
        items.push({
          run,
          res,
          targetId,
          targetName,
          model,
          seriesLabel,
          parsedMetrics,
        })
      }
    }
  }

  const availableMetrics = Array.from(metricCatalog.values())
  const activeMetric =
    selectedMetricName && metricCatalog.has(selectedMetricName)
      ? selectedMetricName
      : availableMetrics.length > 0
        ? availableMetrics[0].name
        : null

  const activeMeta = activeMetric ? metricCatalog.get(activeMetric) : undefined
  const unit = activeMeta?.unit ?? null
  const better = activeMeta?.better ?? null

  const seriesMap = new Map<string, TrendSeries>()
  const points: TrendPoint[] = []

  if (activeMetric) {
    for (const item of items) {
      const mVal = item.parsedMetrics.get(activeMetric)
      if (!mVal) continue

      const pt: TrendPoint = {
        run_id: item.run.id,
        result_id: item.res.id,
        job_id: item.run.job_id,
        job_name:
          item.run.job_name ||
          (typeof item.run.config_snapshot?.job_name === 'string'
            ? item.run.config_snapshot.job_name
            : null),
        tester: item.res.tester,
        status: item.res.status,
        run_status: item.run.status,
        timestamp: item.run.started_at || item.run.queued_at || item.run.finished_at,
        metric: activeMetric,
        value: mVal.value,
        unit: mVal.unit ?? unit,
        better: mVal.better ?? better,
        target_id: item.targetId,
        target_name: item.targetName,
        model: item.model,
        series_key: item.seriesLabel,
        label: item.seriesLabel,
      }
      points.push(pt)

      let s = seriesMap.get(item.seriesLabel)
      if (!s) {
        s = {
          series_key: item.seriesLabel,
          label: item.seriesLabel,
          target_id: item.targetId,
          target_name: item.targetName,
          model: item.model,
          tester: item.res.tester,
          points: [],
        }
        seriesMap.set(item.seriesLabel, s)
      }
      s.points.push(pt)
    }
  }

  return {
    metric: activeMetric,
    unit,
    better,
    availableMetrics,
    series: Array.from(seriesMap.values()),
    points,
  }
}

function formatMetricNum(val: number): string {
  if (Number.isInteger(val)) return val.toLocaleString()
  if (Math.abs(val) >= 100) return val.toFixed(1)
  if (Math.abs(val) >= 1) return val.toFixed(2)
  return val.toFixed(4)
}

function formatShortTime(ts: string | null): string {
  if (!ts) return ''
  try {
    const d = new Date(ts)
    if (Number.isNaN(d.getTime())) return ''
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  } catch {
    return ''
  }
}

export function MetricTrendChart({
  runs,
  selectedRunIds,
  onToggleRunSelection,
  onOpenRun,
  onStartCompare,
}: {
  runs: Run[]
  selectedRunIds: number[]
  onToggleRunSelection: (runId: number) => void
  onOpenRun: (runId: number) => void
  onStartCompare: (runIds: number[], baselineRunId?: number) => void
}) {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<MetricFilterTab | null>(null)
  const [metricSearch, setMetricSearch] = useState('')
  const [pillsExpanded, setPillsExpanded] = useState(false)
  const [hiddenSeries, setHiddenSeries] = useState<string[]>([])
  const [activePoint, setActivePoint] = useState<TrendPoint | null>(null)
  const [clickMode, setClickMode] = useState<'menu' | 'open' | 'compare'>('menu')

  const {
    metric: currentMetric,
    unit,
    better,
    availableMetrics,
    series,
    points,
  } = extractTrendsFromRuns(runs, selectedMetric)

  if (availableMetrics.length === 0) {
    return (
      <div className="card trend-chart-card trend-chart-empty" data-testid="metric-trend-section">
        <div className="compare-section-header">
          <div>
            <h3 className="section-title" style={{ marginBottom: '2px' }}>
              📈 Per-Metric Trend Chart
            </h3>
            <span className="section-desc">
              No numeric metrics available in the currently filtered runs to plot a trend.
            </span>
          </div>
        </div>
      </div>
    )
  }

  const groupedMetrics = groupMetricsByCategory(availableMetrics)
  const explicitKeyMetrics = availableMetrics.filter((m) => isKeyMetric(m))
  const keyMetrics =
    explicitKeyMetrics.length > 0 ? explicitKeyMetrics : availableMetrics.slice(0, 8)
  const showKeyTab =
    availableMetrics.length > COMPACT_PILL_THRESHOLD &&
    keyMetrics.length < availableMetrics.length

  const defaultCategory: MetricFilterTab = showKeyTab ? 'key' : 'all'
  const activeCategory: MetricFilterTab = (() => {
    if (selectedCategory === 'all') return 'all'
    if (selectedCategory === 'key' && showKeyTab) return 'key'
    if (
      selectedCategory &&
      groupedMetrics.some((g) => g.category.id === selectedCategory)
    ) {
      return selectedCategory
    }
    return defaultCategory
  })()

  const trimmedSearch = metricSearch.trim().toLowerCase()
  const categoryFilteredMetrics: TrendMetricOption[] = (() => {
    if (trimmedSearch) {
      return availableMetrics.filter((m) => {
        const cat = METRIC_CATEGORIES.find((c) => c.id === categorizeMetric(m))
        return (
          m.name.toLowerCase().includes(trimmedSearch) ||
          (m.unit || '').toLowerCase().includes(trimmedSearch) ||
          (cat?.label.toLowerCase().includes(trimmedSearch) ?? false)
        )
      })
    }
    if (activeCategory === 'key') return keyMetrics
    if (activeCategory === 'all') return availableMetrics
    return (
      groupedMetrics.find((g) => g.category.id === activeCategory)?.metrics ||
      availableMetrics
    )
  })()

  const activeMetricOption = availableMetrics.find((m) => m.name === currentMetric)
  const shouldCollapsePills =
    !trimmedSearch &&
    !pillsExpanded &&
    categoryFilteredMetrics.length > MAX_COLLAPSED_PILLS

  const visibleMetricPills: TrendMetricOption[] = (() => {
    let list = shouldCollapsePills
      ? categoryFilteredMetrics.slice(0, MAX_COLLAPSED_PILLS)
      : [...categoryFilteredMetrics]

    if (!trimmedSearch && activeMetricOption && !list.some((m) => m.name === activeMetricOption.name)) {
      if (shouldCollapsePills && list.length >= MAX_COLLAPSED_PILLS) {
        list = [activeMetricOption, ...list.slice(0, MAX_COLLAPSED_PILLS - 1)]
      } else {
        list = [activeMetricOption, ...list]
      }
    }
    return list
  })()

  const hiddenPillsCount = Math.max(
    0,
    categoryFilteredMetrics.length -
      visibleMetricPills.filter((m) =>
        categoryFilteredMetrics.some((cm) => cm.name === m.name)
      ).length
  )

  const visibleSeries = series.filter((s) => !hiddenSeries.includes(s.series_key))
  const visiblePoints = visibleSeries.flatMap((s) => s.points)

  // Compute unique chronological X slots by (run_id, result_id)
  const xSlots = points.map((p) => `${p.run_id}:${p.result_id}`)
  const values = visiblePoints.map((p) => p.value)
  const rawMin = values.length > 0 ? Math.min(...values) : 0
  const rawMax = values.length > 0 ? Math.max(...values) : 1
  const padRange = rawMax === rawMin ? Math.max(1, Math.abs(rawMax) * 0.2) : (rawMax - rawMin) * 0.12
  const minVal = rawMin >= 0 ? Math.max(0, rawMin - padRange) : rawMin - padRange
  const maxVal = rawMax + padRange
  const valRange = maxVal - minVal || 1

  const svgWidth = 760
  const svgHeight = 230
  const padLeft = 64
  const padRight = 32
  const padTop = 28
  const padBottom = 44
  const plotW = svgWidth - padLeft - padRight
  const plotH = svgHeight - padTop - padBottom

  const getX = (pt: TrendPoint): number => {
    const slotKey = `${pt.run_id}:${pt.result_id}`
    const idx = xSlots.indexOf(slotKey)
    const count = xSlots.length
    if (count <= 1 || idx === -1) return padLeft + plotW / 2
    return padLeft + (idx / (count - 1)) * plotW
  }

  const getY = (val: number): number => {
    return padTop + plotH - ((val - minVal) / valRange) * plotH
  }

  const toggleSeriesVisibility = (seriesKey: string) => {
    setHiddenSeries((prev) =>
      prev.includes(seriesKey) ? prev.filter((k) => k !== seriesKey) : [...prev, seriesKey]
    )
  }

  const resolveCompareIdsForPoint = (pt: TrendPoint): { ids: number[]; baselineId?: number } => {
    // 1. If other runs are already selected for comparison, include this run with them
    const otherSelected = selectedRunIds.filter((id) => id !== pt.run_id)
    if (otherSelected.length > 0) {
      return {
        ids: [...otherSelected, pt.run_id],
        baselineId: otherSelected[0],
      }
    }

    // 2. Otherwise, check if the same target/model series has a previous run to compare against
    const ptSeries = series.find((s) => s.series_key === pt.series_key)
    if (ptSeries && ptSeries.points.length >= 2) {
      const idx = ptSeries.points.findIndex(
        (p) => p.run_id === pt.run_id && p.result_id === pt.result_id
      )
      if (idx > 0) {
        const prevRunId = ptSeries.points[idx - 1].run_id
        if (prevRunId !== pt.run_id) {
          return { ids: [prevRunId, pt.run_id], baselineId: prevRunId }
        }
      }
      const firstOther = ptSeries.points.find((p) => p.run_id !== pt.run_id)
      if (firstOther) {
        return { ids: [firstOther.run_id, pt.run_id], baselineId: firstOther.run_id }
      }
    }

    // 3. Fallback: compare against another point on the chart or open Compare view with this run
    const anyOther = points.find((p) => p.run_id !== pt.run_id)
    if (anyOther) {
      return { ids: [anyOther.run_id, pt.run_id], baselineId: anyOther.run_id }
    }
    return { ids: [pt.run_id], baselineId: pt.run_id }
  }

  const handlePointClick = (pt: TrendPoint) => {
    setActivePoint(pt)
    if (clickMode === 'open') {
      onOpenRun(pt.run_id)
    } else if (clickMode === 'compare') {
      onToggleRunSelection(pt.run_id)
    }
  }

  const resolvedActivePoint =
    activePoint &&
    points.find((p) => p.run_id === activePoint.run_id && p.result_id === activePoint.result_id)
      ? points.find(
          (p) => p.run_id === activePoint.run_id && p.result_id === activePoint.result_id
        )!
      : null

  return (
    <div className="card trend-chart-card" data-testid="metric-trend-section">
      <div className="compare-section-header">
        <div>
          <h3 className="section-title" style={{ marginBottom: '2px' }}>
            📈 Per-Metric Trend Chart
          </h3>
          <span className="section-desc">
            Metric progression over time per target / model — click any data point to open the run or start a comparison
          </span>
        </div>

        <div className="chart-controls">
          <div className="filter-group">
            <label htmlFor="trend-metric-select" className="filter-label">
              Metric:
            </label>
            <select
              id="trend-metric-select"
              aria-label="Trend Metric"
              className="filter-select"
              value={currentMetric || ''}
              onChange={(e) => {
                setSelectedMetric(e.target.value)
                setActivePoint(null)
              }}
            >
              {groupedMetrics.length > 1
                ? groupedMetrics.map((g) => (
                    <optgroup
                      key={g.category.id}
                      label={`${g.category.label} (${g.metrics.length})`}
                    >
                      {g.metrics.map((m) => (
                        <option key={m.name} value={m.name}>
                          {m.name}
                          {m.unit ? ` (${m.unit})` : ''}
                          {m.better === 'lower' ? ' ↓' : m.better === 'higher' ? ' ↑' : ''}
                        </option>
                      ))}
                    </optgroup>
                  ))
                : availableMetrics.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                      {m.unit ? ` (${m.unit})` : ''}
                      {m.better === 'lower' ? ' ↓' : m.better === 'higher' ? ' ↑' : ''}
                    </option>
                  ))}
            </select>
          </div>

          <div className="filter-group">
            <label htmlFor="trend-click-mode" className="filter-label">
              On Point Click:
            </label>
            <select
              id="trend-click-mode"
              aria-label="Point Click Action"
              className="filter-select"
              value={clickMode}
              onChange={(e) => setClickMode(e.target.value as 'menu' | 'open' | 'compare')}
            >
              <option value="menu">Inspect & Choose Action</option>
              <option value="open">Open Run Directly</option>
              <option value="compare">Toggle Compare Selection</option>
            </select>
          </div>
        </div>
      </div>

      {/* Metric Quick Selector & Category / Search Navigator */}
      {availableMetrics.length > 1 && (
        <div className="trend-metric-browser" data-testid="trend-metric-browser">
          {(groupedMetrics.length > 1 || availableMetrics.length > 6) && (
            <div className="trend-metric-toolbar" data-testid="trend-metric-toolbar">
              <div
                className="trend-category-tabs"
                role="tablist"
                aria-label="Metric categories"
              >
                {showKeyTab && (
                  <button
                    type="button"
                    role="tab"
                    aria-selected={activeCategory === 'key' && !trimmedSearch}
                    className={`trend-category-tab ${activeCategory === 'key' && !trimmedSearch ? 'active' : ''}`}
                    data-testid="trend-category-key"
                    onClick={() => {
                      setSelectedCategory('key')
                      setMetricSearch('')
                      setPillsExpanded(false)
                    }}
                  >
                    ★ Key ({keyMetrics.length})
                  </button>
                )}
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeCategory === 'all' && !trimmedSearch}
                  className={`trend-category-tab ${activeCategory === 'all' && !trimmedSearch ? 'active' : ''}`}
                  data-testid="trend-category-all"
                  onClick={() => {
                    setSelectedCategory('all')
                    setMetricSearch('')
                    setPillsExpanded(false)
                  }}
                >
                  All ({availableMetrics.length})
                </button>
                {groupedMetrics.map((g) => (
                  <button
                    key={g.category.id}
                    type="button"
                    role="tab"
                    aria-selected={activeCategory === g.category.id && !trimmedSearch}
                    className={`trend-category-tab ${activeCategory === g.category.id && !trimmedSearch ? 'active' : ''}`}
                    data-testid={`trend-category-${g.category.id}`}
                    onClick={() => {
                      setSelectedCategory(g.category.id)
                      setMetricSearch('')
                      setPillsExpanded(false)
                    }}
                  >
                    {g.category.label} ({g.metrics.length})
                  </button>
                ))}
              </div>

              <div className="search-input-wrapper trend-metric-search-wrap">
                <span className="search-icon">🔍</span>
                <input
                  type="text"
                  className="search-input trend-metric-search-input"
                  placeholder={`Filter ${availableMetrics.length} metrics...`}
                  aria-label="Search trend metrics"
                  data-testid="trend-metric-search"
                  value={metricSearch}
                  onChange={(e) => setMetricSearch(e.target.value)}
                />
                {metricSearch && (
                  <button
                    type="button"
                    className="trend-search-clear-btn"
                    aria-label="Clear metric search"
                    data-testid="trend-metric-search-clear"
                    onClick={() => setMetricSearch('')}
                  >
                    ✕
                  </button>
                )}
              </div>
            </div>
          )}

          {visibleMetricPills.length === 0 ? (
            <div className="text-muted text-sm" data-testid="trend-metric-no-matches">
              No metrics match &ldquo;{metricSearch}&rdquo;.{' '}
              <button
                type="button"
                className="btn-link"
                onClick={() => setMetricSearch('')}
              >
                Clear filter
              </button>
            </div>
          ) : (
            <div className="chart-metric-pills">
              {visibleMetricPills.map((m) => {
                const isSelected = m.name === currentMetric
                return (
                  <button
                    key={m.name}
                    type="button"
                    data-testid={`trend-metric-pill-${m.name}`}
                    className={`pill-btn ${isSelected ? 'selected' : ''}`}
                    onClick={() => {
                      setSelectedMetric(m.name)
                      setActivePoint(null)
                    }}
                  >
                    <span>{m.name}</span>
                    {m.unit && <span className="text-muted text-sm">({m.unit})</span>}
                  </button>
                )
              })}
              {hiddenPillsCount > 0 && (
                <button
                  type="button"
                  className="pill-btn pill-btn-more"
                  data-testid="trend-pills-expand-btn"
                  onClick={() => setPillsExpanded(true)}
                >
                  +{hiddenPillsCount} more
                </button>
              )}
              {pillsExpanded &&
                !trimmedSearch &&
                categoryFilteredMetrics.length > MAX_COLLAPSED_PILLS && (
                  <button
                    type="button"
                    className="pill-btn pill-btn-more"
                    data-testid="trend-pills-collapse-btn"
                    onClick={() => setPillsExpanded(false)}
                  >
                    Show fewer
                  </button>
                )}
            </div>
          )}
        </div>
      )}

      {/* Target / Model Series Legend */}
      <div className="trend-series-legend" data-testid="trend-series-legend">
        <div className="trend-meta-badges">
          <span className="font-mono font-bold">
            {currentMetric}
            {unit ? ` (${unit})` : ''}
          </span>
          {better && (
            <span className="better-hint">
              {better === 'lower' ? '↓ lower is better' : '↑ higher is better'}
            </span>
          )}
        </div>

        <div className="trend-legend-items">
          {series.map((s, idx) => {
            const color = SERIES_COLORS[idx % SERIES_COLORS.length]
            const isHidden = hiddenSeries.includes(s.series_key)
            const latestPt = s.points[s.points.length - 1]
            return (
              <button
                key={s.series_key}
                type="button"
                className={`trend-legend-btn ${isHidden ? 'series-muted' : ''}`}
                onClick={() => toggleSeriesVisibility(s.series_key)}
                title={`Toggle series ${s.label}`}
                data-testid={`trend-series-toggle-${s.series_key}`}
              >
                <span
                  className="trend-legend-dot"
                  style={{ backgroundColor: isHidden ? '#64748b' : color }}
                />
                <span className="font-mono">{s.label}</span>
                <span className="text-muted text-sm">
                  ({s.points.length} {s.points.length === 1 ? 'run' : 'runs'}
                  {latestPt ? ` · latest ${formatMetricNum(latestPt.value)}` : ''})
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* SVG Time-Series Line Chart */}
      <div className="trend-svg-wrapper">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="trend-line-svg"
          role="img"
          aria-label={`Trend chart for ${currentMetric}`}
          data-testid="metric-trend-svg"
        >
          {/* Horizontal Y-axis grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const val = minVal + ratio * valRange
            const y = getY(val)
            return (
              <g key={ratio}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={svgWidth - padRight}
                  y2={y}
                  stroke="#1e293b"
                  strokeDasharray={ratio === 0 ? undefined : '4 4'}
                  strokeWidth="1"
                />
                <text
                  x={padLeft - 8}
                  y={y + 4}
                  textAnchor="end"
                  fill="#94a3b8"
                  fontSize="10"
                  fontFamily="monospace"
                >
                  {formatMetricNum(val)}
                </text>
              </g>
            )
          })}

          {/* X-axis tick labels */}
          {points.map((pt) => {
            const x = getX(pt)
            const dateStr = formatShortTime(pt.timestamp)
            return (
              <g key={`x-${pt.run_id}-${pt.result_id}`}>
                <text
                  x={x}
                  y={svgHeight - 20}
                  textAnchor="middle"
                  fill="#cbd5e1"
                  fontSize="10"
                  fontFamily="monospace"
                >
                  Run #{pt.run_id}
                </text>
                {dateStr && (
                  <text
                    x={x}
                    y={svgHeight - 7}
                    textAnchor="middle"
                    fill="#64748b"
                    fontSize="9"
                    fontFamily="sans-serif"
                  >
                    {dateStr}
                  </text>
                )}
              </g>
            )
          })}

          {/* Series Polylines and Data Points */}
          {series.map((s, sIdx) => {
            if (hiddenSeries.includes(s.series_key)) return null
            const color = SERIES_COLORS[sIdx % SERIES_COLORS.length]
            const polyCoords = s.points.map((pt) => `${getX(pt)},${getY(pt.value)}`).join(' ')

            return (
              <g key={s.series_key} data-testid={`trend-series-line-${s.series_key}`}>
                {s.points.length > 1 && (
                  <polyline
                    fill="none"
                    stroke={color}
                    strokeWidth="2.5"
                    points={polyCoords}
                  />
                )}

                {s.points.map((pt) => {
                  const cx = getX(pt)
                  const cy = getY(pt.value)
                  const isCompareSelected = selectedRunIds.includes(pt.run_id)
                  const isFocused =
                    resolvedActivePoint?.run_id === pt.run_id &&
                    resolvedActivePoint?.result_id === pt.result_id

                  return (
                    <g
                      key={`pt-${pt.run_id}-${pt.result_id}`}
                      role="button"
                      tabIndex={0}
                      className="trend-point-group"
                      style={{ cursor: 'pointer' }}
                      data-testid={`trend-point-${pt.run_id}`}
                      aria-label={`Run #${pt.run_id} (${s.label}): ${formatMetricNum(pt.value)}${pt.unit ? ` ${pt.unit}` : ''}`}
                      onClick={() => handlePointClick(pt)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          handlePointClick(pt)
                        }
                      }}
                    >
                      {(isCompareSelected || isFocused) && (
                        <circle
                          cx={cx}
                          cy={cy}
                          r="9"
                          fill="none"
                          stroke={isFocused ? '#f8fafc' : '#f59e0b'}
                          strokeWidth="2"
                        />
                      )}
                      <circle
                        cx={cx}
                        cy={cy}
                        r="5.5"
                        fill={pt.status === 'fail' || pt.status === 'error' ? '#ef4444' : color}
                        stroke="#0f172a"
                        strokeWidth="1.5"
                      >
                        <title>
                          {`Run #${pt.run_id} • ${s.label} • ${pt.metric}: ${formatMetricNum(pt.value)}${pt.unit ? ` ${pt.unit}` : ''}`}
                        </title>
                      </circle>
                      <text
                        x={cx}
                        y={Math.max(14, cy - 10)}
                        textAnchor="middle"
                        fill="#f8fafc"
                        fontSize="10"
                        fontFamily="monospace"
                      >
                        {formatMetricNum(pt.value)}
                      </text>
                    </g>
                  )
                })}
              </g>
            )
          })}
        </svg>
      </div>

      {/* Selected Point Action Popover / Panel */}
      {resolvedActivePoint && (
        <div
          className="trend-point-action-bar"
          data-testid="trend-point-action-bar"
          role="region"
          aria-label={`Actions for Run #${resolvedActivePoint.run_id}`}
        >
          <div className="trend-point-info">
            <span className="font-mono font-bold">Run #{resolvedActivePoint.run_id}</span>
            <Badge status={resolvedActivePoint.status} size="sm" />
            <span className="kind-badge">{resolvedActivePoint.label}</span>
            <span className="font-mono">
              {resolvedActivePoint.metric}:{' '}
              <strong>
                {formatMetricNum(resolvedActivePoint.value)}
                {resolvedActivePoint.unit ? ` ${resolvedActivePoint.unit}` : ''}
              </strong>
            </span>
            {resolvedActivePoint.timestamp && (
              <span className="text-muted text-sm">
                {new Date(resolvedActivePoint.timestamp).toLocaleString()}
              </span>
            )}
          </div>

          <div className="trend-point-actions">
            <button
              type="button"
              className="btn btn-sm btn-primary"
              data-testid="trend-point-open-run"
              onClick={() => onOpenRun(resolvedActivePoint.run_id)}
            >
              Open Run #{resolvedActivePoint.run_id} →
            </button>
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              data-testid="trend-point-start-compare"
              onClick={() => {
                const { ids, baselineId } = resolveCompareIdsForPoint(resolvedActivePoint)
                onStartCompare(ids, baselineId)
              }}
            >
              ⚖️ Start Compare →
            </button>
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              data-testid="trend-point-toggle-select"
              onClick={() => onToggleRunSelection(resolvedActivePoint.run_id)}
            >
              {selectedRunIds.includes(resolvedActivePoint.run_id)
                ? '✓ In Compare Selection'
                : '+ Add to Compare'}
            </button>
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              aria-label="Close point actions"
              onClick={() => setActivePoint(null)}
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
