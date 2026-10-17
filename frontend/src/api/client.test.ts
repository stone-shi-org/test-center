import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError } from './client.ts'

describe('API client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('fetches targets list with query params', async () => {
    const mockTargets = [
      { id: 1, name: 'tgt-1', kind: 'embedding', base_url: 'http://localhost', auth_env_var: null, extra: {} },
    ]

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockTargets,
    } as Response)

    const result = await api.targets.list({ kind: 'embedding', limit: 10 })
    expect(fetch).toHaveBeenCalledWith('/api/targets?kind=embedding&limit=10', expect.any(Object))
    expect(result).toEqual(mockTargets)
  })

  it('creates target and sends JSON headers', async () => {
    const newTarget = {
      name: 'tgt-new',
      kind: 'openai-compatible',
      base_url: 'http://localhost:8000',
    }

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: async () => ({ id: 2, ...newTarget }),
    } as Response)

    const res = await api.targets.create(newTarget)
    expect(res.id).toBe(2)
    expect(fetch).toHaveBeenCalledWith(
      '/api/targets',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(newTarget),
      })
    )
  })

  it('handles 409 conflict and throws ApiError with message', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: false,
      status: 409,
      statusText: 'Conflict',
      json: async () => ({ detail: "Target with name 'tgt-1' already exists" }),
    } as Response)

    try {
      await api.targets.create({ name: 'tgt-1', kind: 'kind', base_url: 'url' })
      expect.fail('Should have thrown')
    } catch (err: unknown) {
      expect(err).toBeInstanceOf(ApiError)
      expect((err as ApiError).status).toBe(409)
      expect((err as ApiError).message).toBe("Target with name 'tgt-1' already exists")
    }
  })

  it('generates stream and download URLs', () => {
    expect(api.runs.getLogStreamUrl(5)).toBe('/api/runs/5/logs/stream')
    expect(api.runs.getArtifactDownloadUrl(5, 12)).toBe('/api/runs/5/artifacts/12/download')
  })
})
