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

  it('tests target capabilities by ID and ad-hoc', async () => {
    const mockTestResponse = {
      target_id: 1,
      target_name: 'tgt-1',
      kind: 'openai-compatible',
      base_url: 'http://localhost:8000/v1',
      auth_env_var: null,
      auth_configured: true,
      ok: true,
      capabilities_tested: ['chat', 'embedding'],
      results: [
        { capability: 'chat', ok: true, status_code: 200, latency_ms: 12.5, message: 'Chat OK' },
        { capability: 'embedding', ok: true, status_code: 200, latency_ms: 8.2, message: 'Embed OK' },
      ],
      tested_at: '2026-10-18T00:00:00Z',
    }

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockTestResponse,
    } as Response)

    const res = await api.targets.test(1, { capabilities: ['chat', 'embedding'] })
    expect(res.ok).toBe(true)
    expect(res.capabilities_tested).toEqual(['chat', 'embedding'])
    expect(fetch).toHaveBeenCalledWith(
      '/api/targets/1/test',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ capabilities: ['chat', 'embedding'] }),
      })
    )
  })

  it('generates stream and download URLs', () => {
    expect(api.runs.getLogStreamUrl(5)).toBe('/api/runs/5/logs/stream')
    expect(api.runs.getArtifactDownloadUrl(5, 12)).toBe('/api/runs/5/artifacts/12/download')
  })

  it('fetches compare results with run_ids, baseline, and threshold', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ tester: 'llm', run_ids: [1, 2], baseline_run_id: 1 }),
    } as Response)

    const res = await api.compare.get({
      run_ids: [1, 2],
      baseline_run_id: 1,
      threshold: 5,
      regressed_only: true,
    })
    expect(fetch).toHaveBeenCalledWith(
      '/api/compare?run_ids=1%2C2&baseline_run_id=1&threshold=5&regressed_only=true',
      expect.any(Object)
    )
    expect(res.tester).toBe('llm')
  })

  it('fetches artifact text content', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => 'report line 1\nreport line 2',
    } as Response)

    const text = await api.runs.getArtifactContent(1, 10)
    expect(fetch).toHaveBeenCalledWith('/api/runs/1/artifacts/10/download')
    expect(text).toBe('report line 1\nreport line 2')
  })

  it('manages jobs via CRUD and triggers job runs', async () => {
    const mockJob = {
      id: 7,
      name: 'bundle-job',
      description: 'Test bundle',
      schedule: '',
      enabled: true,
      run_mode: 'parallel',
      definition_ids: [1, 2],
      overrides: {},
      definitions: [],
      created_at: '2026-10-16T00:00:00Z',
      updated_at: '2026-10-16T00:00:00Z',
    }

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: async () => mockJob,
    } as Response)

    const created = await api.jobs.create({
      name: 'bundle-job',
      run_mode: 'parallel',
      definition_ids: [1, 2],
    })
    expect(created.id).toBe(7)
    expect(fetch).toHaveBeenCalledWith(
      '/api/jobs',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          name: 'bundle-job',
          run_mode: 'parallel',
          definition_ids: [1, 2],
        }),
      })
    )

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 202,
      json: async () => ({ id: 42, job_id: 7, status: 'queued', results: [] }),
    } as Response)

    const run = await api.jobs.trigger(7)
    expect(run.id).toBe(42)
    expect(fetch).toHaveBeenCalledWith(
      '/api/jobs/7/run',
      expect.objectContaining({
        method: 'POST',
      })
    )

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ ...mockJob, enabled: false, next_run_at: null }),
    } as Response)

    const disabledJob = await api.jobs.disable(7)
    expect(disabledJob.enabled).toBe(false)
    expect(fetch).toHaveBeenCalledWith(
      '/api/jobs/7/disable',
      expect.objectContaining({ method: 'POST' })
    )

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        valid: true,
        schedule: '0 2 * * *',
        normalized: '0 2 * * *',
        is_manual: false,
        description: 'Daily at 02:00 UTC',
        next_run_at: '2026-10-17T02:00:00Z',
        upcoming_runs: ['2026-10-17T02:00:00Z'],
        error: null,
      }),
    } as Response)

    const cronPreview = await api.jobs.validateCron('0 2 * * *')
    expect(cronPreview.valid).toBe(true)
    expect(cronPreview.description).toBe('Daily at 02:00 UTC')
  })

  it('fetches filtered run history, metric trends, and manages retention settings', async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => [{ id: 10, status: 'completed', results: [] }],
    } as Response)

    const runs = await api.runs.list({
      tester: 'llm',
      target: 'vllm-prod',
      job: 'manual',
      status: 'completed',
      date_from: '2026-10-01',
      date_to: '2026-10-16',
    })
    expect(runs).toHaveLength(1)
    expect(fetch).toHaveBeenCalledWith(
      '/api/runs?status=completed&tester=llm&target=vllm-prod&job=manual&date_from=2026-10-01&date_to=2026-10-16',
      expect.any(Object)
    )

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        metric: 'ttft_p50_ms',
        unit: 'ms',
        better: 'lower',
        available_metrics: [],
        series: [],
        points: [],
      }),
    } as Response)

    const trends = await api.runs.getTrends({ metric: 'ttft_p50_ms', tester: 'llm' })
    expect(trends.metric).toBe('ttft_p50_ms')
    expect(fetch).toHaveBeenCalledWith(
      '/api/runs/trends?metric=ttft_p50_ms&tester=llm',
      expect.any(Object)
    )

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        enabled: true,
        max_age_days: 30,
        retention_days: 30,
        max_runs: 50,
        artifacts_only: false,
        keep_failed: true,
        last_cleanup_at: null,
        stats: {
          total_runs: 10,
          terminal_runs: 10,
          active_runs: 0,
          total_artifacts: 20,
          artifacts_size_bytes: 4096,
          eligible_runs_count: 2,
          eligible_run_ids: [1, 2],
        },
      }),
    } as Response)

    const ret = await api.runs.getRetention()
    expect(ret.max_age_days).toBe(30)
    expect(fetch).toHaveBeenCalledWith('/api/runs/retention', expect.any(Object))

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        enabled: true,
        max_age_days: 14,
        retention_days: 14,
        max_runs: 25,
        artifacts_only: true,
        keep_failed: false,
        last_cleanup_at: null,
        stats: ret.stats,
      }),
    } as Response)

    const updatedRet = await api.runs.updateRetention({
      enabled: true,
      max_age_days: 14,
      max_runs: 25,
      artifacts_only: true,
    })
    expect(updatedRet.max_age_days).toBe(14)
    expect(fetch).toHaveBeenCalledWith(
      '/api/runs/retention',
      expect.objectContaining({
        method: 'PUT',
      })
    )

    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        dry_run: false,
        artifacts_only: true,
        keep_failed: false,
        max_age_days: 14,
        max_runs: 25,
        matched_run_ids: [1, 2],
        cleaned_runs: 2,
        deleted_runs: 0,
        deleted_artifacts: 4,
        freed_bytes: 2048,
      }),
    } as Response)

    const cleanupRes = await api.runs.cleanupRetention({ max_runs: 25, artifacts_only: true })
    expect(cleanupRes.cleaned_runs).toBe(2)
    expect(fetch).toHaveBeenCalledWith(
      '/api/runs/retention/cleanup',
      expect.objectContaining({
        method: 'POST',
      })
    )
  })
})



