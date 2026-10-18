import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  ApiError,
  api,
  type Target,
  type TestDefinition,
  type TesterInfo,
} from '../../api/index.ts'
import { TemplatesView } from './TemplatesView.tsx'

describe('TemplatesView', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
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
      id: 1,
      name: 'vllm-prod',
      kind: 'openai-compatible',
      base_url: 'http://localhost:8000/v1',
      auth_env_var: null,
      extra: {},
      created_at: '2026-10-16T00:00:00Z',
      updated_at: '2026-10-16T00:00:00Z',
    },
  ]

  const mockTemplates: TestDefinition[] = [
    {
      id: 10,
      name: 'smoke-baseline',
      tester: 'dummy',
      suites: ['smoke'],
      config: { duration_seconds: 0.25, fail_case: false },
      target_id: 1,
      created_at: '2026-10-16T10:00:00Z',
      updated_at: '2026-10-17T12:30:00Z',
    },
    {
      id: 20,
      name: 'embed-retrieval',
      tester: 'embedding',
      suites: ['retrieval', 'sts'],
      config: { batch_size: 64 },
      target_id: null,
      created_at: '2026-10-16T11:00:00Z',
      updated_at: '2026-10-17T14:00:00Z',
    },
  ]

  it('lists templates with name, tester, target, suites, and updated time', async () => {
    vi.spyOn(api.definitions, 'list').mockResolvedValue(mockTemplates)
    vi.spyOn(api.testers, 'list').mockResolvedValue(mockTesters)
    vi.spyOn(api.targets, 'list').mockResolvedValue(mockTargets)
    vi.spyOn(api.jobs, 'list').mockResolvedValue([])

    render(<TemplatesView />)

    await waitFor(() => {
      expect(screen.getByText('smoke-baseline')).toBeTruthy()
    })

    expect(screen.getByText('embed-retrieval')).toBeTruthy()
    expect(screen.getByText('vllm-prod')).toBeTruthy()
    expect(screen.getByText('smoke')).toBeTruthy()
    expect(screen.getByText('retrieval')).toBeTruthy()
    expect(screen.getByText('sts')).toBeTruthy()
  })

  it('creates and edits a template with SchemaForm and suite checkboxes', async () => {
    const currentList = [...mockTemplates]
    vi.spyOn(api.definitions, 'list').mockImplementation(async () => [...currentList])
    vi.spyOn(api.testers, 'list').mockResolvedValue(mockTesters)
    vi.spyOn(api.targets, 'list').mockResolvedValue(mockTargets)
    vi.spyOn(api.jobs, 'list').mockResolvedValue([])

    const createSpy = vi.spyOn(api.definitions, 'create').mockImplementation(async (data) => {
      const created = {
        id: 30,
        name: data.name,
        tester: data.tester,
        suites: data.suites || [],
        config: data.config || {},
        target_id: data.target_id ?? null,
        created_at: '2026-10-17T15:00:00Z',
        updated_at: '2026-10-17T15:00:00Z',
      }
      currentList.push(created)
      return created
    })

    const updateSpy = vi.spyOn(api.definitions, 'update').mockImplementation(async (id, data) => {
      const updated = {
        ...mockTemplates[0],
        id,
        name: data.name ?? mockTemplates[0].name,
        tester: data.tester ?? mockTemplates[0].tester,
        suites: data.suites ?? mockTemplates[0].suites,
        config: data.config ?? mockTemplates[0].config,
        target_id: data.target_id !== undefined ? data.target_id : mockTemplates[0].target_id,
      }
      return updated
    })

    render(<TemplatesView />)

    await waitFor(() => {
      expect(screen.getByText('smoke-baseline')).toBeTruthy()
    })

    // Open New Template modal
    fireEvent.click(screen.getByRole('button', { name: /\+ New Template/i }))

    fireEvent.change(screen.getByLabelText(/Template Name/i), {
      target: { value: 'custom-dummy-tpl' },
    })
    fireEvent.change(screen.getByLabelText(/Target Endpoint/i), {
      target: { value: '1' },
    })

    // Uncheck 'load' suite so only 'smoke' remains
    const dialog = screen.getByRole('dialog')
    const loadCheckbox = within(dialog).getByRole('checkbox', { name: 'load' })
    fireEvent.click(loadCheckbox)

    // Toggle fail_case in SchemaForm
    const failCaseCheckbox = within(dialog).getByRole('checkbox', { name: /fail_case/i })
    fireEvent.click(failCaseCheckbox)

    fireEvent.click(within(dialog).getByRole('button', { name: /Create Template/i }))

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledWith({
        name: 'custom-dummy-tpl',
        tester: 'dummy',
        suites: ['smoke'],
        config: { fail_case: true },
        target_id: 1,
      })
    })

    // Edit existing template
    const editButtons = screen.getAllByRole('button', { name: 'Edit' })
    fireEvent.click(editButtons[0])

    const editDialog = screen.getByRole('dialog')
    const nameInput = within(editDialog).getByLabelText(/Template Name/i) as HTMLInputElement
    expect(nameInput.value).toBe('smoke-baseline')

    fireEvent.change(nameInput, { target: { value: 'smoke-baseline-v2' } })
    fireEvent.click(within(editDialog).getByRole('button', { name: /Save Changes/i }))

    await waitFor(() => {
      expect(updateSpy).toHaveBeenCalledWith(
        10,
        expect.objectContaining({
          name: 'smoke-baseline-v2',
          tester: 'dummy',
          suites: ['smoke'],
          target_id: 1,
        })
      )
    })
  })

  it('shows 422 validation errors from the API inside the form', async () => {
    vi.spyOn(api.definitions, 'list').mockResolvedValue(mockTemplates)
    vi.spyOn(api.testers, 'list').mockResolvedValue(mockTesters)
    vi.spyOn(api.targets, 'list').mockResolvedValue(mockTargets)
    vi.spyOn(api.jobs, 'list').mockResolvedValue([])

    vi.spyOn(api.definitions, 'create').mockRejectedValue(
      new ApiError(422, "Unknown suites for tester 'dummy': ['invalid']")
    )

    render(<TemplatesView />)

    await waitFor(() => {
      expect(screen.getByText('smoke-baseline')).toBeTruthy()
    })

    fireEvent.click(screen.getByRole('button', { name: /\+ New Template/i }))
    const dialog = screen.getByRole('dialog')

    fireEvent.change(within(dialog).getByLabelText(/Template Name/i), {
      target: { value: 'bad-template' },
    })
    fireEvent.click(within(dialog).getByRole('button', { name: /Create Template/i }))

    await waitFor(() => {
      expect(within(dialog).getByRole('alert').textContent).toContain(
        "Unknown suites for tester 'dummy'"
      )
    })
  })

  it('duplicates and deletes a template, warning and surfacing errors when used by a job', async () => {
    vi.spyOn(api.definitions, 'list').mockResolvedValue(mockTemplates)
    vi.spyOn(api.testers, 'list').mockResolvedValue(mockTesters)
    vi.spyOn(api.targets, 'list').mockResolvedValue(mockTargets)
    vi.spyOn(api.jobs, 'list').mockResolvedValue([
      {
        id: 5,
        name: 'nightly-job',
        description: null,
        schedule: '',
        enabled: true,
        run_mode: 'sequential',
        definition_ids: [10],
        overrides: {},
        definitions: [],
        created_at: '2026-10-16T00:00:00Z',
        updated_at: '2026-10-16T00:00:00Z',
      },
    ])

    const duplicateSpy = vi.spyOn(api.definitions, 'create').mockResolvedValue({
      ...mockTemplates[0],
      id: 99,
      name: 'smoke-baseline (copy)',
    })

    const deleteSpy = vi.spyOn(api.definitions, 'delete').mockImplementation(async (id) => {
      if (id === 10) {
        throw new ApiError(
          409,
          "Cannot delete template 'smoke-baseline' because it is used by job(s): nightly-job (#5)"
        )
      }
    })

    render(<TemplatesView />)

    await waitFor(() => {
      expect(screen.getByText('smoke-baseline')).toBeTruthy()
    })

    // Duplicate first template
    const duplicateBtns = screen.getAllByRole('button', { name: 'Duplicate' })
    fireEvent.click(duplicateBtns[0])

    await waitFor(() => {
      expect(duplicateSpy).toHaveBeenCalledWith({
        name: 'smoke-baseline (copy)',
        tester: 'dummy',
        suites: ['smoke'],
        config: { duration_seconds: 0.25, fail_case: false },
        target_id: 1,
      })
    })

    // Attempt to delete first template (used by nightly-job)
    const deleteBtns = screen.getAllByRole('button', { name: 'Delete' })
    fireEvent.click(deleteBtns[0])

    const confirmDialog = screen.getByRole('dialog')
    expect(confirmDialog.textContent).toContain('nightly-job (#5)')

    fireEvent.click(within(confirmDialog).getByRole('button', { name: /Delete Template/i }))

    await waitFor(() => {
      expect(deleteSpy).toHaveBeenCalledWith(10)
    })
    await waitFor(() => {
      expect(screen.getByRole('alert').textContent).toContain(
        "Cannot delete template 'smoke-baseline' because it is used by job(s): nightly-job (#5)"
      )
    })

    // Delete second template (not used by any job)
    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' })[1])
    const confirmDialog2 = screen.getByRole('dialog')
    fireEvent.click(within(confirmDialog2).getByRole('button', { name: /Delete Template/i }))

    await waitFor(() => {
      expect(deleteSpy).toHaveBeenCalledWith(20)
    })
  })

  it('renders built-in starter templates and copies one via Use as template', async () => {
    vi.spyOn(api.definitions, 'list').mockResolvedValue(mockTemplates)
    vi.spyOn(api.testers, 'list').mockResolvedValue(mockTesters)
    vi.spyOn(api.targets, 'list').mockResolvedValue(mockTargets)
    vi.spyOn(api.jobs, 'list').mockResolvedValue([])

    const mockStarters = [
      {
        tester: 'dummy',
        name: 'Smoke',
        description: 'Fast built-in dummy smoke suite',
        suites: ['smoke'],
        config: {},
      },
      {
        tester: 'llm',
        name: 'Reasoning, high effort',
        description: 'Reasoning benchmark with high effort',
        suites: ['reasoning'],
        config: { effort: 'high', max_tokens: 8192 },
      },
    ]
    vi.spyOn(api.definitions, 'listStarters').mockResolvedValue(mockStarters)

    const createSpy = vi.spyOn(api.definitions, 'create').mockResolvedValue({
      id: 50,
      name: 'Reasoning, high effort',
      tester: 'llm',
      suites: ['reasoning'],
      config: { effort: 'high', max_tokens: 8192 },
      target_id: null,
      created_at: '2026-10-18T00:00:00Z',
      updated_at: '2026-10-18T00:00:00Z',
    })

    render(<TemplatesView />)

    // Starters section and starter cards are visible
    await waitFor(() => {
      expect(screen.getByText('Starter Templates')).toBeTruthy()
      expect(screen.getByText('Reasoning, high effort')).toBeTruthy()
      expect(screen.getByText('Fast built-in dummy smoke suite')).toBeTruthy()
    })

    // Click 'Use as template' on the reasoning starter
    const useBtns = screen.getAllByRole('button', { name: /Use as template/i })
    expect(useBtns.length).toBe(2)

    fireEvent.click(useBtns[1])

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledWith({
        name: 'Reasoning, high effort',
        tester: 'llm',
        suites: ['reasoning'],
        config: { effort: 'high', max_tokens: 8192 },
        target_id: null,
      })
    })

    await waitFor(() => {
      expect(screen.getByText(/Created template "Reasoning, high effort" from starter/i)).toBeTruthy()
    })
  })
})

