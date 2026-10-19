import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  api,
  type Target,
  type TestDefinition,
  type TesterInfo,
} from '../../api/index.ts'
import { RouterProvider } from '../../router/index.ts'
import { TriggerView } from './TriggerView.tsx'

describe('TriggerView', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    window.history.pushState({}, '', '/trigger')
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const mockTesters: TesterInfo[] = [
    {
      name: 'dummy',
      description: 'Dummy tester plugin',
      suites: ['smoke', 'load'],
      config_schema: {
        type: 'object',
        properties: {
          duration_seconds: {
            type: 'number',
            default: 0.1,
            description: 'Duration in seconds',
          },
          fail_case: {
            type: 'boolean',
            default: false,
            description: 'Fail a test case',
          },
        },
      },
    },
    {
      name: 'embedding',
      description: 'Embedding benchmark plugin',
      suites: ['retrieval', 'sts'],
      config_schema: {
        type: 'object',
        properties: {
          batch_size: {
            type: 'integer',
            default: 32,
            description: 'Batch size',
          },
        },
      },
    },
  ]

  const mockTargets: Target[] = [
    {
      id: 2,
      name: 'embed-server',
      kind: 'openai-compatible',
      base_url: 'http://localhost:8002/v1',
      auth_env_var: null,
      extra: {},
      created_at: '2026-10-16T00:00:00Z',
      updated_at: '2026-10-16T00:00:00Z',
    },
  ]

  const mockTemplates: TestDefinition[] = [
    {
      id: 15,
      name: 'embed-prod-template',
      tester: 'embedding',
      suites: ['retrieval'],
      config: { batch_size: 64 },
      target_id: 2,
      created_at: '2026-10-16T00:00:00Z',
      updated_at: '2026-10-17T00:00:00Z',
    },
  ]

  it('loads a template into the form, keeps values editable, and starts a run without creating a definition', async () => {
    vi.spyOn(api.testers, 'list').mockResolvedValue(mockTesters)
    vi.spyOn(api.targets, 'list').mockResolvedValue(mockTargets)
    vi.spyOn(api.definitions, 'list').mockResolvedValue(mockTemplates)

    const createDefSpy = vi.spyOn(api.definitions, 'create')
    const updateDefSpy = vi.spyOn(api.definitions, 'update')
    const createRunSpy = vi.spyOn(api.runs, 'create').mockResolvedValue({
      id: 42,
      job_id: null,
      status: 'queued',
      queued_at: '2026-10-17T12:00:00Z',
      started_at: null,
      finished_at: null,
      config_snapshot: {
        tester: 'embedding',
        suites: ['retrieval', 'sts'],
        config: { batch_size: 128 },
      },
      target_snapshot: { id: 2, name: 'embed-server' },
      git_info: null,
      results: [],
    })

    render(
      <RouterProvider>
        <TriggerView />
      </RouterProvider>
    )

    await waitFor(() => {
      expect(screen.getByLabelText(/Load from template/i)).toBeTruthy()
    })

    // Ensure Definition Name and Save Definition Only are removed
    expect(screen.queryByLabelText(/Definition Name/i)).toBeNull()
    expect(screen.queryByRole('button', { name: /Save Definition Only/i })).toBeNull()

    // Select template from dropdown
    const templateSelect = screen.getByLabelText(/Load from template/i)
    fireEvent.change(templateSelect, { target: { value: '15' } })

    // Verify tester, target, suites, and config were populated from the template
    const testerSelect = screen.getByLabelText(/Tester Plugin/i) as HTMLSelectElement
    expect(testerSelect.value).toBe('embedding')

    const targetSelect = screen.getByLabelText(/Target Endpoint/i) as HTMLSelectElement
    expect(targetSelect.value).toBe('2')

    const retrievalCheckbox = screen.getByRole('checkbox', { name: 'retrieval' }) as HTMLInputElement
    const stsCheckbox = screen.getByRole('checkbox', { name: 'sts' }) as HTMLInputElement
    expect(retrievalCheckbox.checked).toBe(true)
    expect(stsCheckbox.checked).toBe(false)

    // Edit values in the form (toggle sts suite, change batch_size and timeout)
    fireEvent.click(stsCheckbox)
    const batchSizeInput = screen.getByLabelText(/batch_size/i) as HTMLInputElement
    expect(batchSizeInput.value).toBe('64')
    fireEvent.change(batchSizeInput, { target: { value: '128' } })

    const timeoutInput = screen.getByLabelText(/Timeout \(Seconds\)/i)
    fireEvent.change(timeoutInput, { target: { value: '90' } })

    // Template object in memory is not mutated
    expect(mockTemplates[0].suites).toEqual(['retrieval'])
    expect(mockTemplates[0].config).toEqual({ batch_size: 64 })

    // Click Start Run
    fireEvent.click(screen.getByRole('button', { name: /^Start Run$/i }))

    await waitFor(() => {
      expect(createRunSpy).toHaveBeenCalledWith({
        tester: 'embedding',
        suites: ['retrieval', 'sts'],
        config: { batch_size: 128 },
        target_id: 2,
        timeout: 90,
      })
    })

    expect(createDefSpy).not.toHaveBeenCalled()
    expect(updateDefSpy).not.toHaveBeenCalled()
    expect(window.location.pathname).toBe('/runs/42')
  })

  it('lists starters in Load from template dropdown and loads starter config directly', async () => {
    vi.spyOn(api.testers, 'list').mockResolvedValue(mockTesters)
    vi.spyOn(api.targets, 'list').mockResolvedValue(mockTargets)
    vi.spyOn(api.definitions, 'list').mockResolvedValue(mockTemplates)

    const mockStarters = [
      {
        tester: 'dummy',
        name: 'Smoke',
        description: 'Dummy smoke suite',
        suites: ['smoke'],
        config: { duration_seconds: 0.05 },
      },
    ]
    vi.spyOn(api.definitions, 'listStarters').mockResolvedValue(mockStarters)

    render(
      <RouterProvider>
        <TriggerView />
      </RouterProvider>
    )

    await waitFor(() => {
      expect(screen.getByLabelText(/Load from template/i)).toBeTruthy()
    })

    // Verify Starters optgroup and option exist
    expect(screen.getByRole('group', { name: /Starters/i })).toBeTruthy()
    expect(screen.getByRole('group', { name: /Saved Templates/i })).toBeTruthy()

    // Select the starter option
    const templateSelect = screen.getByLabelText(/Load from template/i)
    fireEvent.change(templateSelect, { target: { value: 'starter:0' } })

    // Verify fields populated
    const testerSelect = screen.getByLabelText(/Tester Plugin/i) as HTMLSelectElement
    expect(testerSelect.value).toBe('dummy')

    const targetSelect = screen.getByLabelText(/Target Endpoint/i) as HTMLSelectElement
    expect(targetSelect.value).toBe('')

    const smokeCheckbox = screen.getByRole('checkbox', { name: 'smoke' }) as HTMLInputElement
    const loadCheckbox = screen.getByRole('checkbox', { name: 'load' }) as HTMLInputElement
    expect(smokeCheckbox.checked).toBe(true)
    expect(loadCheckbox.checked).toBe(false)

    const durationInput = screen.getByLabelText(/duration_seconds/i) as HTMLInputElement
    expect(durationInput.value).toBe('0.05')
  })

  it('omits schema suites property from SchemaForm and strips stray suites from config when triggering a run', async () => {
    const testersWithSchemaSuites: TesterInfo[] = [
      {
        name: 'live-stt',
        description: 'live-stt HTTP and accuracy tester',
        suites: ['transcriptions_json', 'accuracy_wer'],
        config_schema: {
          type: 'object',
          properties: {
            suites: {
              type: 'array',
              items: { type: 'string', enum: ['transcriptions_json', 'accuracy_wer'] },
              default: ['transcriptions_json'],
              description: 'Sub-suites to execute',
            },
            dataset_split: {
              type: 'string',
              default: 'eval_small',
              description: 'NOTSOFAR split',
            },
          },
        },
      },
    ]

    const legacyTemplates: TestDefinition[] = [
      {
        id: 19,
        name: 'legacy-accuracy-template',
        tester: 'live-stt',
        suites: ['accuracy_wer'],
        config: {
          suites: ['transcriptions_json'],
          dataset_split: 'eval_small',
        },
        target_id: null,
        created_at: '2026-10-18T00:00:00Z',
        updated_at: '2026-10-18T00:00:00Z',
      },
    ]

    vi.spyOn(api.testers, 'list').mockResolvedValue(testersWithSchemaSuites)
    vi.spyOn(api.targets, 'list').mockResolvedValue([])
    vi.spyOn(api.definitions, 'list').mockResolvedValue(legacyTemplates)
    vi.spyOn(api.definitions, 'listStarters').mockResolvedValue([])

    const createRunSpy = vi.spyOn(api.runs, 'create').mockResolvedValue({
      id: 99,
      job_id: null,
      status: 'queued',
      queued_at: '2026-10-18T12:00:00Z',
      started_at: null,
      finished_at: null,
      config_snapshot: {
        tester: 'live-stt',
        suites: ['accuracy_wer'],
        config: { dataset_split: 'eval_small' },
      },
      target_snapshot: {},
      git_info: null,
      results: [],
    })

    render(
      <RouterProvider>
        <TriggerView />
      </RouterProvider>
    )

    await waitFor(() => {
      expect(screen.getByLabelText(/Load from template/i)).toBeTruthy()
    })

    // Load legacy template with stray config.suites
    const templateSelect = screen.getByLabelText(/Load from template/i)
    fireEvent.change(templateSelect, { target: { value: '19' } })

    const werCheckbox = screen.getByRole('checkbox', { name: 'accuracy_wer' }) as HTMLInputElement
    const jsonCheckbox = screen.getByRole('checkbox', { name: 'transcriptions_json' }) as HTMLInputElement
    expect(werCheckbox.checked).toBe(true)
    expect(jsonCheckbox.checked).toBe(false)

    // Click Reset to Defaults in SchemaForm and verify it does not inject 'suites' into config
    fireEvent.click(screen.getByRole('button', { name: /Reset to Defaults/i }))

    // Click Start Run
    fireEvent.click(screen.getByRole('button', { name: /^Start Run$/i }))

    await waitFor(() => {
      expect(createRunSpy).toHaveBeenCalledTimes(1)
    })

    const runPayload = createRunSpy.mock.calls[0][0]
    expect(runPayload.suites).toEqual(['accuracy_wer'])
    expect(runPayload.config).toEqual({ dataset_split: 'eval_small' })
    expect('suites' in (runPayload.config ?? {})).toBe(false)
  })
})


