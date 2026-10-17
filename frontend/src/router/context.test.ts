import { describe, expect, it } from 'vitest'
import { matchRoute, parsePath } from './context.ts'

describe('Router context and matching', () => {
  it('parses hash routes correctly', () => {
    expect(parsePath('/', '#/targets')).toBe('/targets')
    expect(parsePath('/', '#/runs/123')).toBe('/runs/123')
    expect(parsePath('/runs', '')).toBe('/runs')
  })

  it('matches dashboard routes', () => {
    expect(matchRoute('/').name).toBe('dashboard')
    expect(matchRoute('/dashboard').name).toBe('dashboard')
  })

  it('matches targets route', () => {
    expect(matchRoute('/targets').name).toBe('targets')
  })

  it('matches trigger and jobs route', () => {
    expect(matchRoute('/trigger').name).toBe('trigger')
    expect(matchRoute('/jobs').name).toBe('trigger')
  })

  it('matches runs list route', () => {
    expect(matchRoute('/runs').name).toBe('runs')
  })

  it('matches run-detail route with id param', () => {
    const match = matchRoute('/runs/42')
    expect(match.name).toBe('run-detail')
    expect(match.params.id).toBe('42')
  })

  it('falls back to dashboard on unrecognized path', () => {
    expect(matchRoute('/some-random-route').name).toBe('dashboard')
  })
})
