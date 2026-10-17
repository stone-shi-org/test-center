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

  it('matches trigger and jobs routes', () => {
    expect(matchRoute('/trigger').name).toBe('trigger')
    expect(matchRoute('/jobs').name).toBe('jobs')
    const jobDetail = matchRoute('/jobs/7')
    expect(jobDetail.name).toBe('jobs')
    expect(jobDetail.params.id).toBe('7')
  })

  it('matches runs list and history routes with query filter params', () => {
    expect(matchRoute('/runs').name).toBe('runs')
    expect(matchRoute('/history').name).toBe('runs')
    const filtered = matchRoute('/runs?tester=llm&target=vllm&status=pass')
    expect(filtered.name).toBe('runs')
    expect(filtered.params.tester).toBe('llm')
    expect(filtered.params.target).toBe('vllm')
    expect(filtered.params.status).toBe('pass')
  })

  it('matches run-detail route with id param', () => {
    const match = matchRoute('/runs/42')
    expect(match.name).toBe('run-detail')
    expect(match.params.id).toBe('42')
  })

  it('matches compare route and parses query params for shareable URLs', () => {
    const path = parsePath('/compare', '', '?run_ids=10,12&baseline_run_id=10&threshold=5')
    expect(path).toBe('/compare?run_ids=10,12&baseline_run_id=10&threshold=5')

    const match = matchRoute(path)
    expect(match.name).toBe('compare')
    expect(match.params.run_ids).toBe('10,12')
    expect(match.params.baseline_run_id).toBe('10')
    expect(match.params.threshold).toBe('5')
  })

  it('matches compare route with repeated run_ids and path-based ids', () => {
    const repeated = matchRoute('/compare?run_ids=1&run_ids=2&regressed_only=true')
    expect(repeated.name).toBe('compare')
    expect(repeated.params.run_ids).toBe('1,2')
    expect(repeated.params.regressed_only).toBe('true')

    const pathBased = matchRoute('/compare/3,4')
    expect(pathBased.name).toBe('compare')
    expect(pathBased.params.run_ids).toBe('3,4')
  })

  it('falls back to dashboard on unrecognized path', () => {
    expect(matchRoute('/some-random-route').name).toBe('dashboard')
  })
})
