import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { RetentionSettings, Run } from '../../api/types.ts'
import { matchRoute, RouterContext } from '../../router/index.ts'
import { RunsListView } from './RunsListView.tsx'

const mockRuns: Run[] = [
  {
    id: 201,
    job_id: 10,
    job_name: 'Nightly LLM Suite',
    status: 'completed',
    queued_at: '2026-10-10T10:00:00Z',
    started_at: '2026-10-10T10:00:01Z',
    finished_at: '2026-10-10T10:00:05Z',
    config_snapshot: {
      tester: 'llm',
      config: { model: 'Qwen2.5-7B' },
    },
    target_snapshot: {
      id: 1,
      name: 'vllm-prod',
    },
    git_info: null,
    results: [
      {
        id: 301,
        tester: 'llm',
        status: 'pass',
        metrics: {
          ttft_p50_ms: { value: 42.5, unit: 'ms', better: 'lower' },
          tokens_per_sec: { value: 85.0, unit: 'tok/s', better: 'higher' },
        },
        summary: '1/1 passed',
        cases: [
          {
            id: 1,
            suite: 'latency',
            case_id: 'prompt-1',
            status: 'pass',
            metrics: {},
            detail: {},
          },
        ],
        artifacts: [],
      },
    ],
  },
  {
    id: 202,
    job_id: 10,
    job_name: 'Nightly LLM Suite',
    status: 'completed',
    queued_at: '2026-10-12T10:00:00Z',
    started_at: '2026-10-12T10:00:01Z',
    finished_at: '2026-10-12T10:00:06Z',
    config_snapshot: {
      tester: 'llm',
      config: { model: 'Qwen2.5-7B' },
    },
    target_snapshot: {
      id: 1,
      name: 'vllm-prod',
    },
    git_info: null,
    results: [
      {
        id: 302,
        tester: 'llm',
        status: 'pass',
        metrics: {
          ttft_p50_ms: { value: 38.0, unit: 'ms', better: 'lower' },
          tokens_per_sec: { value: 94.2, unit: 'tok/s', better: 'higher' },
        },
        summary: '1/1 passed',
        cases: [
          {
            id: 2,
            suite: 'latency',
            case_id: 'prompt-1',
            status: 'pass',
            metrics: {},
            detail: {},
          },
        ],
        artifacts: [],
      },
    ],
  },
  {
    id: 203,
    job_id: null,
    job_name: null,
    status: 'failed',
    queued_at: '2026-10-15T14:00:00Z',
    started_at: '2026-10-15T14:00:01Z',
    finished_at: '2026-10-15T14:00:04Z',
    config_snapshot: {
      tester: 'embedding',
      config: { model: 'bge-m3' },
    },
    target_snapshot: {
      id: 2,
      name: 'embed-staging',
    },
    git_info: null,
    results: [
      {
        id: 303,
        tester: 'embedding',
        status: 'fail',
        metrics: {
          latency_p50_ms: { value: 18.4, unit: 'ms', better: 'lower' },
        },
        summary: '0/1 passed',
        cases: [
          {
            id: 3,
            suite: 'quality',
            case_id: 'embed-1',
            status: 'fail',
            metrics: {},
            detail: {},
          },
        ],
        artifacts: [],
      },
    ],
  },
]

const mockRetentionSettings: RetentionSettings = {
  enabled: false,
  max_age_days: 30,
  retention_days: 30,
  max_runs: 100,
  artifacts_only: false,
  keep_failed: true,
  last_cleanup_at: null,
  last_cleanup_summary: null,
  cutoff_date: null,
  stats: {
    total_runs: 3,
    terminal_runs: 3,
    active_runs: 0,
    total_artifacts: 6,
    artifacts_size_bytes: 8192,
    eligible_runs_count: 1,
    eligible_run_ids: [201],
  },
}

describe('RunsListView — Filterable Run History, Per-Metric Trend Charts & Retention', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('filters run history by tester, target, job, status, and date range', async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.startsWith('/api/runs')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockRuns,
        } as Response
      }
      throw new Error(`Unexpected fetch URL: ${url}`)
    })

    const navigate = vi.fn()
    render(
      <RouterContext.Provider value={{ route: matchRoute('/runs'), navigate }}>
        <RunsListView />
      </RouterContext.Provider>
    )

    await waitFor(() => {
      expect(screen.getByText('#201')).toBeTruthy()
    })
    expect(screen.getByText('#202')).toBeTruthy()
    expect(screen.getByText('#203')).toBeTruthy()

    // 1. Filter by Tester: embedding
    const testerSelect = screen.getByLabelText('Tester:')
    fireEvent.change(testerSelect, { target: { value: 'embedding' } })
    expect(screen.queryByText('#201')).toBeNull()
    expect(screen.queryByText('#202')).toBeNull()
    expect(screen.getByText('#203')).toBeTruthy()

    // Reset filters
    fireEvent.click(screen.getByTestId('clear-runs-filters-btn'))
    expect(screen.getByText('#201')).toBeTruthy()
    expect(screen.getByText('#203')).toBeTruthy()

    // 2. Filter by Target: vllm-prod
    const targetSelect = screen.getByLabelText('Target:')
    fireEvent.change(targetSelect, { target: { value: 'vllm-prod' } })
    expect(screen.getByText('#201')).toBeTruthy()
    expect(screen.getByText('#202')).toBeTruthy()
    expect(screen.queryByText('#203')).toBeNull()

    // 3. Filter by Job: Manual / Ad-hoc
    fireEvent.change(targetSelect, { target: { value: 'all' } })
    const jobSelect = screen.getByLabelText('Job:')
    fireEvent.change(jobSelect, { target: { value: 'manual' } })
    expect(screen.queryByText('#201')).toBeNull()
    expect(screen.getByText('#203')).toBeTruthy()

    // Filter by specific job ID (#10)
    fireEvent.change(jobSelect, { target: { value: '10' } })
    expect(screen.getByText('#201')).toBeTruthy()
    expect(screen.getByText('#202')).toBeTruthy()
    expect(screen.queryByText('#203')).toBeNull()

    // 4. Filter by Status: fail
    fireEvent.change(jobSelect, { target: { value: 'all' } })
    const statusSelect = screen.getByLabelText('Status:')
    fireEvent.change(statusSelect, { target: { value: 'fail' } })
    expect(screen.queryByText('#201')).toBeNull()
    expect(screen.getByText('#203')).toBeTruthy()

    // 5. Filter by Date range (From Date / To Date)
    fireEvent.change(statusSelect, { target: { value: 'all' } })
    const fromDateInput = screen.getByLabelText('From Date')
    const toDateInput = screen.getByLabelText('To Date')
    fireEvent.change(fromDateInput, { target: { value: '2026-10-11' } })
    fireEvent.change(toDateInput, { target: { value: '2026-10-13' } })
    expect(screen.queryByText('#201')).toBeNull()
    expect(screen.getByText('#202')).toBeTruthy()
    expect(screen.queryByText('#203')).toBeNull()
  })

  it('renders per-metric trend chart grouped by target/model and supports clicking points to open run or start compare', async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.startsWith('/api/runs')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockRuns,
        } as Response
      }
      throw new Error(`Unexpected fetch URL: ${url}`)
    })

    const navigate = vi.fn()
    render(
      <RouterContext.Provider value={{ route: matchRoute('/runs'), navigate }}>
        <RunsListView />
      </RouterContext.Provider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('metric-trend-section')).toBeTruthy()
    })

    // Default metric should be ttft_p50_ms with series "vllm-prod / Qwen2.5-7B"
    const metricSelect = screen.getByLabelText('Trend Metric') as HTMLSelectElement
    expect(metricSelect.value).toBe('ttft_p50_ms')
    expect(screen.getByTestId('trend-series-legend').textContent).toContain(
      'vllm-prod / Qwen2.5-7B'
    )

    // Switch metric to tokens_per_sec
    fireEvent.change(metricSelect, { target: { value: 'tokens_per_sec' } })
    expect(metricSelect.value).toBe('tokens_per_sec')
    expect(screen.getByTestId('trend-series-legend').textContent).toContain('tokens_per_sec')

    // Click data point for Run #202 to open the point action bar
    const point202 = screen.getByTestId('trend-point-202')
    fireEvent.click(point202)

    const actionBar = screen.getByTestId('trend-point-action-bar')
    expect(actionBar.textContent).toContain('Run #202')
    expect(actionBar.textContent).toContain('94.20 tok/s')

    // Click "Start Compare" from the point action bar -> should compare #201 (previous in series) and #202
    fireEvent.click(screen.getByTestId('trend-point-start-compare'))
    expect(navigate).toHaveBeenCalledWith('/compare?run_ids=201%2C202&baseline_run_id=201')

    // Click "Open Run #202" from the point action bar -> should navigate to /runs/202
    fireEvent.click(screen.getByTestId('trend-point-open-run'))
    expect(navigate).toHaveBeenCalledWith('/runs/202')
  })

  it('opens retention settings panel, saves retention policy, and runs dry-run/immediate cleanup', async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/runs/retention' && (!init || !init.method || init.method === 'GET')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockRetentionSettings,
        } as Response
      }
      if (url === '/api/runs/retention' && init?.method === 'PUT') {
        const body = JSON.parse(String(init.body || '{}'))
        return {
          ok: true,
          status: 200,
          json: async () => ({
            ...mockRetentionSettings,
            ...body,
          }),
        } as Response
      }
      if (url === '/api/runs/retention/cleanup' && init?.method === 'POST') {
        const body = JSON.parse(String(init.body || '{}'))
        return {
          ok: true,
          status: 200,
          json: async () => ({
            dry_run: Boolean(body.dry_run),
            artifacts_only: Boolean(body.artifacts_only),
            keep_failed: Boolean(body.keep_failed),
            max_age_days: body.max_age_days ?? 14,
            max_runs: body.max_runs ?? 50,
            matched_run_ids: [201],
            cleaned_runs: 1,
            deleted_runs: body.dry_run ? 0 : 1,
            deleted_artifacts: 2,
            freed_bytes: 4096,
          }),
        } as Response
      }
      if (url.startsWith('/api/runs')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockRuns,
        } as Response
      }
      throw new Error(`Unexpected fetch URL: ${url}`)
    })

    const navigate = vi.fn()
    render(
      <RouterContext.Provider value={{ route: matchRoute('/runs'), navigate }}>
        <RunsListView />
      </RouterContext.Provider>
    )

    await waitFor(() => {
      expect(screen.getByText('#201')).toBeTruthy()
    })

    // Open Retention Settings Panel
    fireEvent.click(screen.getByTestId('toggle-retention-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('retention-settings-panel')).toBeTruthy()
      expect(screen.getByTestId('retention-stats-bar').textContent).toContain('3 total runs')
    })

    // Update Max Age (days) to 14 and save policy
    const maxAgeInput = screen.getByLabelText('Max Age (days):') as HTMLInputElement
    expect(maxAgeInput.value).toBe('30')
    fireEvent.change(maxAgeInput, { target: { value: '14' } })

    fireEvent.click(screen.getByTestId('retention-save-btn'))
    await waitFor(() => {
      expect(screen.getByTestId('retention-status-message').textContent).toContain(
        'Retention settings saved.'
      )
    })

    // Preview cleanup (Dry Run)
    fireEvent.click(screen.getByTestId('retention-preview-btn'))
    await waitFor(() => {
      expect(screen.getByTestId('retention-cleanup-summary').textContent).toContain('[DRY RUN]')
      expect(screen.getByTestId('retention-cleanup-summary').textContent).toContain(
        'Cleaned runs: 1'
      )
    })

    // Execute cleanup now
    fireEvent.click(screen.getByTestId('retention-cleanup-now-btn'))
    await waitFor(() => {
      expect(screen.getByTestId('retention-cleanup-summary').textContent).toContain('[EXECUTED]')
      expect(screen.getByTestId('retention-status-message').textContent).toContain(
        'Deleted 1 old runs and 2 artifacts'
      )
    })
  })
})
