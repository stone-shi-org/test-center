import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/index.ts'
import { RouterProvider } from '../../router/index.ts'
import { RunDetailView } from '../runs/RunDetailView.tsx'
import { JobsView } from './JobsView.tsx'

describe('JobsView & Grouped Run Results', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    window.history.pushState({}, '', '/jobs')
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const mockTesters = [
    {
      name: 'dummy',
      description: 'Dummy tester plugin',
      suites: ['smoke', 'load'],
      config_schema: {
        type: 'object' as const,
        properties: {
          duration_seconds: {
            type: 'number' as const,
            default: 0.1,
            description: 'Duration in seconds',
          },
          fail_case: {
            type: 'boolean' as const,
            default: false,
            description: 'Fail a test case',
          },
        },
      },
    },
  ]

  const mockDefinitions = [
    {
      id: 10,
      name: 'smoke-def',
      tester: 'dummy',
      suites: ['smoke'],
      config: { duration_seconds: 0.05 },
      target_id: null,
      created_at: '2026-10-16T00:00:00Z',
      updated_at: '2026-10-16T00:00:00Z',
    },
    {
      id: 20,
      name: 'load-def',
      tester: 'dummy',
      suites: ['load'],
      config: { duration_seconds: 0.1 },
      target_id: null,
      created_at: '2026-10-16T00:00:00Z',
      updated_at: '2026-10-16T00:00:00Z',
    },
  ]

  const mockJobs = [
    {
      id: 1,
      name: 'nightly-suite',
      description: 'Nightly sequential bundle',
      schedule: '0 2 * * *',
      enabled: true,
      run_mode: 'sequential' as const,
      definition_ids: [10, 20],
      overrides: {},
      definitions: [
        {
          order: 0,
          id: 10,
          test_definition_id: 10,
          name: 'smoke-def',
          tester: 'dummy',
          suites: ['smoke'],
          config: { duration_seconds: 0.05 },
          target_id: null,
          overrides: {},
          effective_suites: ['smoke'],
          effective_config: { duration_seconds: 0.05 },
          effective_target_id: null,
        },
        {
          order: 1,
          id: 20,
          test_definition_id: 20,
          name: 'load-def',
          tester: 'dummy',
          suites: ['load'],
          config: { duration_seconds: 0.1 },
          target_id: null,
          overrides: {},
          effective_suites: ['load'],
          effective_config: { duration_seconds: 0.1 },
          effective_target_id: null,
        },
      ],
      created_at: '2026-10-16T00:00:00Z',
      updated_at: '2026-10-16T00:00:00Z',
    },
  ]

  it('lists jobs and triggers Run Now', async () => {
    vi.spyOn(api.jobs, 'list').mockResolvedValue(mockJobs)
    vi.spyOn(api.definitions, 'list').mockResolvedValue(mockDefinitions)
    vi.spyOn(api.testers, 'list').mockResolvedValue(mockTesters)
    vi.spyOn(api.targets, 'list').mockResolvedValue([])
    vi.spyOn(api.runs, 'list').mockResolvedValue([])
    const triggerSpy = vi.spyOn(api.jobs, 'trigger').mockResolvedValue({
      id: 99,
      job_id: 1,
      status: 'queued',
      queued_at: '2026-10-16T00:00:00Z',
      started_at: null,
      finished_at: null,
      config_snapshot: {},
      target_snapshot: {},
      git_info: null,
      results: [],
    })

    render(
      <RouterProvider>
        <JobsView />
      </RouterProvider>
    )

    await waitFor(() => {
      expect(screen.getByText('nightly-suite')).toBeDefined()
    })

    expect(screen.getByText('SEQUENTIAL')).toBeDefined()
    expect(screen.getByText(/smoke-def/i)).toBeDefined()
    expect(screen.getByText(/load-def/i)).toBeDefined()

    const runBtn = screen.getByRole('button', { name: /Run Now/i })
    fireEvent.click(runBtn)

    await waitFor(() => {
      expect(triggerSpy).toHaveBeenCalledWith(1)
    })
  })

  it('creates a job with ordered definitions, run mode, and schema-generated overrides', async () => {
    vi.spyOn(api.jobs, 'list').mockResolvedValue([])
    vi.spyOn(api.definitions, 'list').mockResolvedValue(mockDefinitions)
    vi.spyOn(api.testers, 'list').mockResolvedValue(mockTesters)
    vi.spyOn(api.targets, 'list').mockResolvedValue([])
    vi.spyOn(api.runs, 'list').mockResolvedValue([])

    const createSpy = vi.spyOn(api.jobs, 'create').mockImplementation(async (payload) => ({
      id: 2,
      name: payload.name,
      description: payload.description || null,
      schedule: payload.schedule || '',
      enabled: payload.enabled ?? true,
      run_mode: payload.run_mode || 'sequential',
      definition_ids: payload.definition_ids || [],
      overrides: (payload.overrides as Record<string, unknown>) || {},
      definitions: [],
      created_at: '2026-10-16T00:00:00Z',
      updated_at: '2026-10-16T00:00:00Z',
    }))

    render(
      <RouterProvider>
        <JobsView />
      </RouterProvider>
    )

    await waitFor(() => {
      expect(screen.getByText(/No jobs found/i)).toBeDefined()
    })

    fireEvent.click(screen.getByRole('button', { name: /\+ New Job/i }))

    fireEvent.change(screen.getByLabelText(/Job Name/i), {
      target: { value: 'parallel-bundle' },
    })
    fireEvent.change(screen.getByLabelText(/Run Mode \*/i), {
      target: { value: 'parallel' },
    })

    // Add first definition (#10)
    fireEvent.click(screen.getByRole('button', { name: /\+ Add Definition/i }))

    // Select and add second definition (#20)
    fireEvent.change(screen.getByLabelText(/Select Test Definition/i), {
      target: { value: '20' },
    })
    fireEvent.click(screen.getByRole('button', { name: /\+ Add Definition/i }))

    // SchemaForm is rendered for the expanded step (#20): toggle fail_case in schema form
    const failCaseCheckbox = screen.getByRole('checkbox', { name: /fail_case/i })
    fireEvent.click(failCaseCheckbox)

    // Move step 2 up to position 1
    const moveUpBtn = screen.getByRole('button', { name: /Move step 2 up/i })
    fireEvent.click(moveUpBtn)

    // Save job
    fireEvent.click(screen.getByRole('button', { name: /^Save Job$/i }))

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledTimes(1)
    })

    expect(createSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'parallel-bundle',
        run_mode: 'parallel',
        definition_ids: [20, 10],
        overrides: expect.objectContaining({
          '20': expect.objectContaining({
            config: expect.objectContaining({
              fail_case: true,
            }),
          }),
        }),
      })
    )
  })

  it('displays grouped per-tester results in RunDetailView', async () => {
    vi.spyOn(api.runs, 'get').mockResolvedValue({
      id: 55,
      job_id: 1,
      job_name: 'nightly-suite',
      status: 'completed',
      queued_at: '2026-10-16T00:00:00Z',
      started_at: '2026-10-16T00:00:01Z',
      finished_at: '2026-10-16T00:00:03Z',
      config_snapshot: {
        job_id: 1,
        job_name: 'nightly-suite',
        run_mode: 'sequential',
        tester: 'dummy',
        testers: ['dummy', 'dummy'],
      },
      target_snapshot: {},
      git_info: null,
      results: [
        {
          id: 101,
          test_definition_id: 10,
          definition_name: 'smoke-def',
          tester: 'dummy',
          status: 'pass',
          summary: 'Smoke suite passed',
          metrics: { latency_ms: 12.5 },
          cases: [],
          artifacts: [],
        },
        {
          id: 102,
          test_definition_id: 20,
          definition_name: 'load-def',
          tester: 'dummy',
          status: 'fail',
          summary: 'Load suite failed threshold',
          metrics: { latency_ms: 95.0 },
          cases: [],
          artifacts: [],
        },
      ],
    })
    vi.spyOn(api.runs, 'getLogs').mockResolvedValue({
      run_id: 55,
      status: 'completed',
      lines: ['Step 1 complete', 'Step 2 complete'],
      raw: 'Step 1 complete\nStep 2 complete',
    })

    render(
      <RouterProvider>
        <RunDetailView runId={55} />
      </RouterProvider>
    )

    await waitFor(() => {
      expect(screen.getByText(/Job: nightly-suite \(SEQUENTIAL\)/i)).toBeDefined()
    })

    const resultsTabBtn = screen.getByRole('button', { name: /Results & Metrics \(2 TESTERS\)/i })
    fireEvent.click(resultsTabBtn)

    expect(screen.getByText(/Grouped Tester Results \(2\)/i)).toBeDefined()
    expect(screen.getByText('Smoke suite passed')).toBeDefined()
    expect(screen.getByText('Load suite failed threshold')).toBeDefined()

    // Filter to just the second tester result
    const loadPill = screen.getByRole('button', { name: /load-def \(DUMMY\)/i })
    fireEvent.click(loadPill)

    expect(screen.queryByText('Smoke suite passed')).toBeNull()
    expect(screen.getByText('Load suite failed threshold')).toBeDefined()
  })
})
