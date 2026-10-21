import { describe, expect, it } from 'vitest'
import {
  formatDate,
  formatDateTime,
  formatTime,
  parseLocalDateEnd,
  parseLocalDateStart,
  parseUtcDate,
} from './date.ts'

describe('date utils (browser local time)', () => {
  it('parses naive ISO strings without timezone as UTC', () => {
    const naive = '2026-10-21T21:30:00.000'
    const d = parseUtcDate(naive)
    expect(d).not.toBeNull()
    expect(d?.toISOString()).toBe('2026-10-21T21:30:00.000Z')
  })

  it('parses ISO strings with Z or offset correctly', () => {
    const withZ = '2026-10-21T21:30:00.000Z'
    const withOffset = '2026-10-21T21:30:00.000+00:00'
    expect(parseUtcDate(withZ)?.toISOString()).toBe('2026-10-21T21:30:00.000Z')
    expect(parseUtcDate(withOffset)?.toISOString()).toBe('2026-10-21T21:30:00.000Z')
  })

  it('handles null, undefined, and invalid values gracefully', () => {
    expect(parseUtcDate(null)).toBeNull()
    expect(parseUtcDate(undefined)).toBeNull()
    expect(parseUtcDate('')).toBeNull()
    expect(parseUtcDate('invalid')).toBeNull()
    expect(formatDateTime(null)).toBe('—')
    expect(formatTime(undefined)).toBe('—')
    expect(formatDate(null)).toBe('—')
  })

  it('formats timestamp to user local date and time string', () => {
    const ts = '2026-10-21T12:00:00Z'
    const formatted = formatDateTime(ts)
    expect(formatted).not.toBe('—')
    expect(formatted.length).toBeGreaterThan(0)
    // Matches toLocaleString output of equivalent Date
    expect(formatted).toBe(new Date(ts).toLocaleString())
  })

  it('formats timestamp to user local time string', () => {
    const ts = '2026-10-21T12:00:00Z'
    const formatted = formatTime(ts)
    expect(formatted).not.toBe('—')
    expect(formatted).toBe(new Date(ts).toLocaleTimeString())
  })

  it('formats timestamp to user local date string', () => {
    const ts = '2026-10-21T12:00:00Z'
    const formatted = formatDate(ts)
    expect(formatted).not.toBe('—')
    expect(formatted).toBe(new Date(ts).toLocaleDateString())
  })

  it('computes local day start and end timestamps', () => {
    const start = parseLocalDateStart('2026-10-21')
    const end = parseLocalDateEnd('2026-10-21')
    expect(start).not.toBeNull()
    expect(end).not.toBeNull()
    expect(end!).toBeGreaterThan(start!)

    const startDate = new Date(start!)
    const endDate = new Date(end!)
    expect(startDate.getHours()).toBe(0)
    expect(startDate.getMinutes()).toBe(0)
    expect(endDate.getHours()).toBe(23)
    expect(endDate.getMinutes()).toBe(59)
  })
})
