/**
 * Date and time formatting utilities ensuring user's browser local time is always used.
 */

/**
 * Safely parse a date/time representation into a Date object.
 *
 * If the input is a string without an explicit timezone offset (no trailing 'Z'
 * or '[+-]HH:MM'), it is interpreted as UTC to avoid the browser assuming it
 * is in local wall-clock time.
 */
export function parseUtcDate(raw: string | number | Date | null | undefined): Date | null {
  if (raw === null || raw === undefined || raw === '') return null
  if (raw instanceof Date) return Number.isNaN(raw.getTime()) ? null : raw
  if (typeof raw === 'number') {
    const d = new Date(raw)
    return Number.isNaN(d.getTime()) ? null : d
  }
  const str = String(raw).trim()
  if (!str) return null

  // If already contains a timezone indicator ('Z', '+HH:MM', '-HH:MM', etc.)
  const hasTimezone = /Z|[+-]\d{2}(?::?\d{2})?$/i.test(str)
  // If it's a date-time string without timezone, append 'Z' to interpret as UTC
  const normalized = hasTimezone ? str : `${str}Z`
  const d = new Date(normalized)
  if (!Number.isNaN(d.getTime())) {
    return d
  }
  const fallback = new Date(str)
  return Number.isNaN(fallback.getTime()) ? null : fallback
}

/**
 * Format a timestamp into the user's browser local date and time.
 * e.g., "10/21/2026, 2:38:18 PM"
 */
export function formatDateTime(
  raw: string | number | Date | null | undefined,
  fallback = '—',
  options?: Intl.DateTimeFormatOptions
): string {
  const d = parseUtcDate(raw)
  if (!d) return fallback
  return d.toLocaleString(undefined, options)
}

/**
 * Format a timestamp into the user's browser local time only.
 * e.g., "2:38:18 PM"
 */
export function formatTime(
  raw: string | number | Date | null | undefined,
  fallback = '—',
  options?: Intl.DateTimeFormatOptions
): string {
  const d = parseUtcDate(raw)
  if (!d) return fallback
  return d.toLocaleTimeString(undefined, options)
}

/**
 * Format a timestamp into the user's browser local date only.
 * e.g., "10/21/2026"
 */
export function formatDate(
  raw: string | number | Date | null | undefined,
  fallback = '—',
  options?: Intl.DateTimeFormatOptions
): string {
  const d = parseUtcDate(raw)
  if (!d) return fallback
  return d.toLocaleDateString(undefined, options)
}

/**
 * Parse a 'YYYY-MM-DD' input string into a local timestamp representing start of day (00:00:00.000).
 */
export function parseLocalDateStart(dateStr: string | null | undefined): number | null {
  if (!dateStr) return null
  const parts = dateStr.trim().split('-').map(Number)
  if (parts.length !== 3 || parts.some(Number.isNaN)) return null
  const [y, m, d] = parts
  return new Date(y, m - 1, d, 0, 0, 0, 0).getTime()
}

/**
 * Parse a 'YYYY-MM-DD' input string into a local timestamp representing end of day (23:59:59.999).
 */
export function parseLocalDateEnd(dateStr: string | null | undefined): number | null {
  if (!dateStr) return null
  const parts = dateStr.trim().split('-').map(Number)
  if (parts.length !== 3 || parts.some(Number.isNaN)) return null
  const [y, m, d] = parts
  return new Date(y, m - 1, d, 23, 59, 59, 999).getTime()
}
