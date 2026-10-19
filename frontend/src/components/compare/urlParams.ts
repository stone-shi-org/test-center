export type CompareFilterMode = 'all' | 'changed' | 'regressed'

export interface ParsedCompareParams {
  runIds: number[]
  baselineRunId: number | null
  threshold: number
  filterMode: CompareFilterMode
  suite: string
  baselines?: string[]
}

export function parseCompareUrlParams(
  params: Record<string, string>
): ParsedCompareParams {
  const rawRuns = params.run_ids || params.runs || ''
  const runIds: number[] = []
  for (const part of rawRuns.split(',')) {
    const trimmed = part.trim()
    if (!trimmed) continue
    const num = parseInt(trimmed, 10)
    if (!Number.isNaN(num) && !runIds.includes(num)) {
      runIds.push(num)
    }
  }

  const rawBaselines = params.baselines || ''
  const baselines: string[] = []
  for (const part of rawBaselines.split(',')) {
    const trimmed = part.trim()
    if (trimmed && !baselines.includes(trimmed)) {
      baselines.push(trimmed)
    }
  }

  const rawBase = params.baseline_run_id || params.baseline_id || params.baseline || ''
  const parsedBase = rawBase ? parseInt(rawBase, 10) : NaN
  const baselineRunId = !Number.isNaN(parsedBase) ? parsedBase : null

  const rawThresh = params.threshold || params.threshold_pct || '0'
  const parsedThresh = parseFloat(rawThresh)
  const threshold = !Number.isNaN(parsedThresh) && parsedThresh >= 0 ? parsedThresh : 0

  let filterMode: CompareFilterMode = 'all'
  if (params.regressed_only === 'true' || params.filter === 'regressed') {
    filterMode = 'regressed'
  } else if (params.changed_only === 'true' || params.filter === 'changed') {
    filterMode = 'changed'
  }

  const suite = params.suite || 'all'

  return {
    runIds,
    baselineRunId,
    threshold,
    filterMode,
    suite,
    ...(baselines.length > 0 ? { baselines } : {}),
  }
}

export function buildCompareUrlPath(opts: {
  runIds: number[]
  baselineRunId?: number | null
  threshold?: number
  filterMode?: CompareFilterMode
  suite?: string
  baselines?: string[]
}): string {
  const query = new URLSearchParams()
  if (opts.baselines && opts.baselines.length > 0) {
    query.set('baselines', opts.baselines.join(','))
  }
  if (opts.runIds.length > 0) {
    query.set('run_ids', opts.runIds.join(','))
  }
  if (opts.baselineRunId !== undefined && opts.baselineRunId !== null) {
    query.set('baseline_run_id', String(opts.baselineRunId))
  }
  if (opts.threshold !== undefined && opts.threshold > 0) {
    query.set('threshold', String(opts.threshold))
  }
  if (opts.filterMode === 'regressed') {
    query.set('regressed_only', 'true')
  } else if (opts.filterMode === 'changed') {
    query.set('changed_only', 'true')
  }
  if (opts.suite && opts.suite !== 'all') {
    query.set('suite', opts.suite)
  }
  const qs = query.toString()
  return `/compare${qs ? `?${qs}` : ''}`
}
