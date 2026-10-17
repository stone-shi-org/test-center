const CRON_MACROS: Record<string, string> = {
  '@yearly': '0 0 1 1 *',
  '@annually': '0 0 1 1 *',
  '@monthly': '0 0 1 * *',
  '@weekly': '0 0 * * 0',
  '@daily': '0 0 * * *',
  '@midnight': '0 0 * * *',
  '@hourly': '0 * * * *',
}

const MONTH_NAMES: Record<string, number> = {
  jan: 1,
  feb: 2,
  mar: 3,
  apr: 4,
  may: 5,
  jun: 6,
  jul: 7,
  aug: 8,
  sep: 9,
  oct: 10,
  nov: 11,
  dec: 12,
}

const DOW_NAMES: Record<string, number> = {
  sun: 0,
  mon: 1,
  tue: 2,
  wed: 3,
  thu: 4,
  fri: 5,
  sat: 6,
}

const MAX_DAYS_PER_MONTH: Record<number, number> = {
  1: 31,
  2: 29,
  3: 31,
  4: 30,
  5: 31,
  6: 30,
  7: 31,
  8: 31,
  9: 30,
  10: 31,
  11: 30,
  12: 31,
}

const MANUAL_VALUES = new Set(['', 'manual', 'none', 'off', 'disabled'])

export interface ClientCronValidation {
  valid: boolean
  isManual: boolean
  normalized: string
  description: string
  nextRunAt: string | null
  error: string | null
}

export const SCHEDULE_PRESETS = [
  { label: 'Manual', value: 'manual', desc: 'Manual trigger only' },
  { label: 'Every 5m', value: '*/5 * * * *', desc: 'Every 5 minutes' },
  { label: 'Every 15m', value: '*/15 * * * *', desc: 'Every 15 minutes' },
  { label: 'Hourly', value: '0 * * * *', desc: 'Every hour at :00' },
  { label: 'Daily 02:00', value: '0 2 * * *', desc: 'Daily at 02:00 UTC' },
  { label: 'Weekly Sun', value: '0 3 * * 0', desc: 'Sundays at 03:00 UTC' },
]

export function isManualSchedule(schedule: string | null | undefined): boolean {
  if (!schedule) return true
  return MANUAL_VALUES.has(schedule.trim().toLowerCase())
}

function parseTokenValue(
  token: string,
  fieldName: string,
  minVal: number,
  maxVal: number,
  names?: Record<string, number>,
  isRangeEndDow = false
): number {
  const t = token.trim().toLowerCase()
  if (!t) {
    throw new Error(`Empty value in ${fieldName} field`)
  }
  let val: number
  if (names && t in names) {
    val = names[t]
    if (isRangeEndDow && val === 0) val = 7
  } else {
    if (!/^-?\d+$/.test(t)) {
      throw new Error(`Invalid value '${token}' in ${fieldName} field (expected ${minVal}-${maxVal})`)
    }
    val = parseInt(t, 10)
  }
  if (val < minVal || val > maxVal) {
    throw new Error(`Value ${val} out of range in ${fieldName} field (allowed ${minVal}-${maxVal})`)
  }
  return val
}

function parseField(
  fieldStr: string,
  fieldName: string,
  minVal: number,
  maxVal: number,
  names?: Record<string, number>,
  isDow = false
): { values: Set<number>; isWildcard: boolean } {
  const raw = fieldStr.trim()
  if (!raw) throw new Error(`Empty ${fieldName} field`)
  const isWildcard = raw === '*'
  const values = new Set<number>()

  for (const rawPart of raw.split(',')) {
    const part = rawPart.trim()
    if (!part) throw new Error(`Empty list item in ${fieldName} field`)

    let step = 1
    let basePart = part
    if (part.includes('/')) {
      const slashParts = part.split('/')
      if (slashParts.length !== 2 || !slashParts[0] || !slashParts[1]) {
        throw new Error(`Invalid step syntax '${part}' in ${fieldName} field`)
      }
      basePart = slashParts[0].trim()
      const stepStr = slashParts[1].trim()
      if (!/^-?\d+$/.test(stepStr)) {
        throw new Error(`Invalid step value '${stepStr}' in ${fieldName} field`)
      }
      step = parseInt(stepStr, 10)
      if (step <= 0) {
        throw new Error(`Step value must be positive in ${fieldName} field`)
      }
    }

    let start: number
    let end: number
    if (basePart === '*') {
      start = minVal
      end = isDow ? 6 : maxVal
    } else if (basePart.includes('-')) {
      const rangeParts = basePart.split('-')
      if (rangeParts.length !== 2 || !rangeParts[0] || !rangeParts[1]) {
        throw new Error(`Invalid range syntax '${basePart}' in ${fieldName} field`)
      }
      start = parseTokenValue(rangeParts[0], fieldName, minVal, maxVal, names)
      end = parseTokenValue(rangeParts[1], fieldName, minVal, maxVal, names, isDow && start > 0)
      if (start > end) {
        throw new Error(`Invalid range '${basePart}' in ${fieldName} field: ${start} > ${end}`)
      }
    } else {
      const single = parseTokenValue(basePart, fieldName, minVal, maxVal, names)
      start = single
      end = part.includes('/') ? (isDow ? 6 : maxVal) : single
    }

    for (let v = start; v <= end; v += step) {
      values.add(isDow ? v % 7 : v)
    }
  }

  return { values, isWildcard }
}

export function validateCronSchedule(
  schedule: string | null | undefined,
  now: Date = new Date()
): ClientCronValidation {
  if (isManualSchedule(schedule)) {
    return {
      valid: true,
      isManual: true,
      normalized: 'manual',
      description: 'Manual trigger only',
      nextRunAt: null,
      error: null,
    }
  }

  const raw = (schedule || '').trim().replace(/\s+/g, ' ')
  const lower = raw.toLowerCase()

  try {
    let expanded = raw
    if (lower.startsWith('@')) {
      if (!(lower in CRON_MACROS)) {
        throw new Error(`Unknown cron macro '${raw}'`)
      }
      expanded = CRON_MACROS[lower]
    }

    const parts = expanded.split(' ')
    if (parts.length !== 5) {
      throw new Error(
        `Cron expression must have 5 fields (minute hour day month weekday), got ${parts.length}`
      )
    }

    const minutes = parseField(parts[0], 'minute', 0, 59)
    const hours = parseField(parts[1], 'hour', 0, 23)
    const dom = parseField(parts[2], 'day_of_month', 1, 31)
    const months = parseField(parts[3], 'month', 1, 12, MONTH_NAMES)
    const dow = parseField(parts[4], 'day_of_week', 0, 7, DOW_NAMES, true)

    if (!dom.isWildcard && dow.isWildcard) {
      const maxPossibleDom = Math.max(...Array.from(months.values).map((m) => MAX_DAYS_PER_MONTH[m]))
      if (Array.from(dom.values).every((d) => d > maxPossibleDom)) {
        throw new Error('Impossible day_of_month for selected month(s)')
      }
    }

    const description = describeSchedule(raw, expanded)
    const nextDate = computeNextCronDate(
      minutes.values,
      hours.values,
      dom.values,
      months.values,
      dow.values,
      dom.isWildcard,
      dow.isWildcard,
      now
    )

    return {
      valid: true,
      isManual: false,
      normalized: raw,
      description,
      nextRunAt: nextDate ? nextDate.toISOString() : null,
      error: null,
    }
  } catch (err: unknown) {
    return {
      valid: false,
      isManual: false,
      normalized: raw,
      description: '',
      nextRunAt: null,
      error: err instanceof Error ? err.message : String(err),
    }
  }
}

function describeSchedule(raw: string, expanded: string): string {
  const lower = raw.toLowerCase()
  if (lower === '@hourly' || expanded === '0 * * * *') return 'Every hour (at minute 0)'
  if (lower === '@daily' || lower === '@midnight' || expanded === '0 0 * * *')
    return 'Daily at 00:00 UTC'
  if (lower === '@weekly' || expanded === '0 0 * * 0') return 'Weekly on Sunday at 00:00 UTC'
  if (lower === '@monthly' || expanded === '0 0 1 * *') return 'Monthly on day 1 at 00:00 UTC'
  if (lower === '@yearly' || lower === '@annually' || expanded === '0 0 1 1 *')
    return 'Yearly on Jan 1 at 00:00 UTC'
  if (expanded === '* * * * *') return 'Every minute'

  const [m, h, dom, mon, dow] = expanded.split(' ')
  if (m.startsWith('*/') && h === '*' && dom === '*' && mon === '*' && dow === '*') {
    return `Every ${m.split('/')[1]} minutes`
  }
  if (/^\d+$/.test(m) && /^\d+$/.test(h) && dom === '*' && mon === '*' && dow === '*') {
    return `Daily at ${h.padStart(2, '0')}:${m.padStart(2, '0')} UTC`
  }
  return `Cron (${raw})`
}

function computeNextCronDate(
  minutes: Set<number>,
  hours: Set<number>,
  dom: Set<number>,
  months: Set<number>,
  dow: Set<number>,
  domWild: boolean,
  dowWild: boolean,
  after: Date
): Date | null {
  const candidate = new Date(after.getTime())
  candidate.setUTCSeconds(0, 0)
  candidate.setUTCMinutes(candidate.getUTCMinutes() + 1)

  const maxIterations = 525600 // 1 year in minutes for fast client preview
  for (let i = 0; i < maxIterations; i++) {
    const m = candidate.getUTCMonth() + 1
    if (!months.has(m)) {
      candidate.setUTCMonth(candidate.getUTCMonth() + 1, 1)
      candidate.setUTCHours(0, 0, 0, 0)
      continue
    }

    const d = candidate.getUTCDate()
    const w = candidate.getUTCDay()
    const domMatch = dom.has(d)
    const dowMatch = dow.has(w)
    const dayMatches =
      domWild && dowWild
        ? true
        : domWild
          ? dowMatch
          : dowWild
            ? domMatch
            : domMatch || dowMatch

    if (!dayMatches) {
      candidate.setUTCDate(candidate.getUTCDate() + 1)
      candidate.setUTCHours(0, 0, 0, 0)
      continue
    }

    const hr = candidate.getUTCHours()
    if (!hours.has(hr)) {
      candidate.setUTCHours(hr + 1, 0, 0, 0)
      continue
    }

    const min = candidate.getUTCMinutes()
    if (!minutes.has(min)) {
      candidate.setUTCMinutes(min + 1, 0, 0)
      continue
    }

    return candidate
  }

  return null
}

export function formatUtcTimestamp(iso: string | null | undefined): string {
  if (!iso) return '—'
  const dt = new Date(iso)
  if (Number.isNaN(dt.getTime())) return iso
  return dt.toISOString().replace('T', ' ').slice(0, 16) + ' UTC'
}
