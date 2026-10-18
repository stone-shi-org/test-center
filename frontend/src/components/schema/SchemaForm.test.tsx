import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, type JSONSchemaObject, type TargetModelInfo } from '../../api/index.ts'
import { SchemaForm } from './SchemaForm.tsx'

describe('SchemaForm component', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const mockSchema: JSONSchemaObject = {
    type: 'object',
    properties: {
      concurrency: {
        type: 'integer',
        default: 8,
        description: 'Concurrency worker count',
      },
      fail_fast: {
        type: 'boolean',
        default: false,
        description: 'Stop immediately on error',
      },
      model: {
        type: 'string',
        default: 'text-embedding-3-small',
        description: 'Target model name',
      },
      suites: {
        type: 'array',
        items: {
          type: 'string',
          enum: ['latency', 'throughput'],
        },
        default: ['latency'],
      },
    },
  }

  it('renders fields generated from JSON Schema', () => {
    const onChange = vi.fn()
    render(<SchemaForm schema={mockSchema} values={{}} onChange={onChange} />)

    expect(screen.getByText(/Tester Configuration Options/i)).toBeDefined()
    expect(screen.getByLabelText(/fail_fast/i)).toBeDefined()
    expect(screen.getByText('latency')).toBeDefined()
    expect(screen.getByText('throughput')).toBeDefined()
  })

  it('triggers onChange when a field value is modified', () => {
    const onChange = vi.fn()
    render(
      <SchemaForm
        schema={mockSchema}
        values={{ concurrency: 8, fail_fast: false }}
        onChange={onChange}
      />
    )

    const checkbox = screen.getByRole('checkbox', { name: /fail_fast/i })
    fireEvent.click(checkbox)

    expect(onChange).toHaveBeenCalledWith({
      concurrency: 8,
      fail_fast: true,
    })
  })

  it('toggles array enum pills when clicked', () => {
    const onChange = vi.fn()
    render(
      <SchemaForm
        schema={mockSchema}
        values={{ suites: ['latency'] }}
        onChange={onChange}
      />
    )

    const throughputBtn = screen.getByRole('button', { name: /throughput/i })
    fireEvent.click(throughputBtn)

    expect(onChange).toHaveBeenCalledWith({
      suites: ['latency', 'throughput'],
    })
  })

  it('resets to defaults when clicking Reset to Defaults', () => {
    const onChange = vi.fn()
    render(
      <SchemaForm
        schema={mockSchema}
        values={{ concurrency: 32, fail_fast: true }}
        onChange={onChange}
      />
    )

    const resetBtn = screen.getByRole('button', { name: /Reset to Defaults/i })
    fireEvent.click(resetBtn)

    expect(onChange).toHaveBeenCalledWith({
      concurrency: 8,
      fail_fast: false,
      model: 'text-embedding-3-small',
      suites: ['latency'],
    })
  })

  it('hides x-advanced fields by default and toggles them on button click', () => {
    const onChange = vi.fn()
    const advancedSchema: JSONSchemaObject = {
      type: 'object',
      properties: {
        model: {
          type: 'string',
          default: 'default-model',
          description: 'Essential model field',
        },
        hidden_param: {
          type: 'integer',
          default: 10,
          description: 'Advanced hidden parameter',
          'x-advanced': true,
        },
        another_hidden: {
          type: 'boolean',
          default: false,
          description: 'Another advanced field',
          'x-advanced': true,
        },
      },
    }

    render(<SchemaForm schema={advancedSchema} values={{}} onChange={onChange} />)

    // Essential field is visible
    expect(screen.getByLabelText(/model/i)).toBeDefined()

    // Advanced fields are hidden by default
    expect(screen.queryByLabelText(/hidden_param/i)).toBeNull()
    expect(screen.queryByLabelText(/another_hidden/i)).toBeNull()

    // Toggle button shows count of advanced options (2)
    const toggleBtn = screen.getByRole('button', { name: /Show advanced options \(2\)/i })
    expect(toggleBtn).toBeDefined()

    // Click toggle button -> reveals advanced fields
    fireEvent.click(toggleBtn)
    expect(screen.getByLabelText(/hidden_param/i)).toBeDefined()
    expect(screen.getByLabelText(/another_hidden/i)).toBeDefined()
    expect(screen.getByRole('button', { name: /Hide advanced options \(2\)/i })).toBeDefined()

    // Click toggle button again -> hides them
    fireEvent.click(screen.getByRole('button', { name: /Hide advanced options \(2\)/i }))
    expect(screen.queryByLabelText(/hidden_param/i)).toBeNull()
  })

  it('displays changed badge and auto-expands when external values contain non-default advanced options', () => {
    const onChange = vi.fn()
    const advancedSchema: JSONSchemaObject = {
      type: 'object',
      properties: {
        model: { type: 'string', default: 'base' },
        rate_limit: {
          type: 'integer',
          default: 100,
          'x-advanced': true,
        },
      },
    }

    // Render with rate_limit changed from default 100 -> 250
    render(
      <SchemaForm
        schema={advancedSchema}
        values={{ rate_limit: 250 }}
        onChange={onChange}
      />
    )

    // Badge is visible
    expect(screen.getByText(/1 advanced option changed/i)).toBeDefined()

    // Section is auto-expanded
    expect(screen.getByLabelText(/rate_limit/i)).toBeDefined()
    expect(screen.getByRole('button', { name: /Hide advanced options \(1\)/i })).toBeDefined()
  })

  it('supports unset option for string enum fields without default', () => {
    const onChange = vi.fn()
    const enumSchema: JSONSchemaObject = {
      type: 'object',
      properties: {
        effort: {
          type: 'string',
          enum: ['none', 'low', 'medium', 'high'],
          description: 'Reasoning effort',
        },
      },
    }

    const { rerender } = render(<SchemaForm schema={enumSchema} values={{}} onChange={onChange} />)

    const select = screen.getByLabelText(/effort/i) as HTMLSelectElement
    expect(select.value).toBe('')
    expect(screen.getByRole('option', { name: /— Unset —/i })).toBeDefined()

    // Change to 'high'
    fireEvent.change(select, { target: { value: 'high' } })
    expect(onChange).toHaveBeenCalledWith({ effort: 'high' })

    // Re-render with effort='high' then switch back to unset
    rerender(<SchemaForm schema={enumSchema} values={{ effort: 'high' }} onChange={onChange} />)
    fireEvent.change(select, { target: { value: '' } })
    expect(onChange).toHaveBeenCalledWith({})
  })

  it('preserves hidden advanced values when editing essential fields', () => {
    const onChange = vi.fn()
    const advancedSchema: JSONSchemaObject = {
      type: 'object',
      properties: {
        model: { type: 'string', default: 'base' },
        secret_ratio: {
          type: 'number',
          default: 1.0,
          'x-advanced': true,
        },
      },
    }

    render(
      <SchemaForm
        schema={advancedSchema}
        values={{ model: 'base', secret_ratio: 2.5 }}
        onChange={onChange}
      />
    )

    // Modify essential field 'model'
    const modelInput = screen.getByLabelText(/model/i)
    fireEvent.change(modelInput, { target: { value: 'updated-model' } })

    // Secret ratio is preserved in the emitted onChange object
    expect(onChange).toHaveBeenCalledWith({
      model: 'updated-model',
      secret_ratio: 2.5,
    })
  })

  describe('model-picker widget (TES-42)', () => {
    const liveSttSchema: JSONSchemaObject = {
      type: 'object',
      properties: {
        models: {
          description: 'Model preset or explicit list of model IDs.',
          default: 'per_engine',
          'x-widget': 'model-picker',
          oneOf: [
            { type: 'string', enum: ['per_engine', 'all', 'default'] },
            { type: 'array', items: { type: 'string' } },
          ],
        },
        language: {
          type: 'string',
          default: 'en',
        },
        grpc_addr: {
          type: 'string',
          default: '',
          'x-advanced': true,
        },
      },
    }

    const mockDiscoveredModels: TargetModelInfo[] = [
      {
        id: 'parakeet-ctc-0.6b',
        type: 'asr',
        engine: 'parakeet',
        is_default: true,
        sample_rate: 16000,
        supports_streaming: true,
        supports_num_speakers_hint: false,
        default_voice: null,
        supported_voices: [],
        grpc_streaming: true,
      },
      {
        id: 'parakeet-ctc-1.1b',
        type: 'asr',
        engine: 'parakeet',
        is_default: false,
        sample_rate: 16000,
        supports_streaming: true,
        supports_num_speakers_hint: false,
        default_voice: null,
        supported_voices: [],
        grpc_streaming: true,
      },
      {
        id: 'whisper-large-v3-turbo',
        type: 'asr',
        engine: 'whisper',
        is_default: false,
        sample_rate: 16000,
        supports_streaming: true,
        supports_num_speakers_hint: false,
        default_voice: null,
        supported_voices: [],
        grpc_streaming: false,
      },
      {
        id: 'pyannote/speaker-diarization-community-1',
        type: 'diarization',
        engine: 'pyannote',
        is_default: true,
        sample_rate: 16000,
        supports_streaming: false,
        supports_num_speakers_hint: true,
        default_voice: null,
        supported_voices: [],
        grpc_streaming: false,
      },
      {
        id: 'kokoro-82m',
        type: 'tts',
        engine: 'kokoro',
        is_default: true,
        sample_rate: 24000,
        supports_streaming: true,
        supports_num_speakers_hint: false,
        default_voice: 'af_heart',
        supported_voices: ['af_heart', 'bf_emma'],
        grpc_streaming: false,
      },
    ]

    it('renders the model picker outside Advanced options', () => {
      const onChange = vi.fn()
      render(<SchemaForm schema={liveSttSchema} values={{}} onChange={onChange} />)

      const picker = screen.getByTestId('model-picker-models')
      expect(picker).toBeTruthy()
      expect(picker.closest('.essential-fields')).not.toBeNull()
      expect(picker.closest('.advanced-fields')).toBeNull()

      // Advanced option grpc_addr is hidden until toggled
      expect(screen.queryByLabelText(/grpc_addr/i)).toBeNull()
    })

    it('stores preset and previews checkboxes when clicking All, Default, or Per engine', async () => {
      vi.spyOn(api.targets, 'listModels').mockResolvedValue(mockDiscoveredModels)
      const onChange = vi.fn()

      const { rerender } = render(
        <SchemaForm
          schema={liveSttSchema}
          values={{}}
          onChange={onChange}
          context={{ targetId: 7 }}
        />
      )

      await waitFor(() => {
        expect(screen.getByText('whisper-large-v3-turbo')).toBeTruthy()
      })

      // Default value is 'per_engine' -> banner shows "Per engine (resolved when the run starts)"
      expect(screen.getByTestId('model-picker-preset-status').textContent).toBe(
        'Per engine (resolved when the run starts)'
      )

      // Badges rendered: batch only on whisper-large-v3-turbo, voices on kokoro-82m
      expect(screen.getByText('batch only')).toBeTruthy()
      expect(screen.getByText(/voices: af_heart, bf_emma/i)).toBeTruthy()

      const getCheckboxFor = (modelId: string) => {
        const label = screen.getByText(modelId).closest('label')!
        return within(label).getByRole('checkbox') as HTMLInputElement
      }

      // Per engine previews: parakeet-ctc-0.6b (default parakeet), whisper-large-v3-turbo (whisper), pyannote, kokoro
      expect(getCheckboxFor('parakeet-ctc-0.6b').checked).toBe(true)
      expect(getCheckboxFor('parakeet-ctc-1.1b').checked).toBe(false)
      expect(getCheckboxFor('whisper-large-v3-turbo').checked).toBe(true)
      expect(getCheckboxFor('pyannote/speaker-diarization-community-1').checked).toBe(true)
      expect(getCheckboxFor('kokoro-82m').checked).toBe(true)

      // Click 'All' preset button
      fireEvent.click(screen.getByRole('button', { name: 'All' }))
      expect(onChange).toHaveBeenCalledWith({ models: 'all' })

      rerender(
        <SchemaForm
          schema={liveSttSchema}
          values={{ models: 'all' }}
          onChange={onChange}
          context={{ targetId: 7 }}
        />
      )
      expect(screen.getByTestId('model-picker-preset-status').textContent).toBe(
        'All (resolved when the run starts)'
      )
      expect(getCheckboxFor('parakeet-ctc-1.1b').checked).toBe(true)

      // Click 'Default' preset button
      fireEvent.click(screen.getByRole('button', { name: 'Default' }))
      expect(onChange).toHaveBeenCalledWith({ models: 'default' })

      rerender(
        <SchemaForm
          schema={liveSttSchema}
          values={{ models: 'default' }}
          onChange={onChange}
          context={{ targetId: 7 }}
        />
      )
      expect(screen.getByTestId('model-picker-preset-status').textContent).toBe(
        'Default (resolved when the run starts)'
      )
      expect(getCheckboxFor('parakeet-ctc-0.6b').checked).toBe(true)
      expect(getCheckboxFor('whisper-large-v3-turbo').checked).toBe(false)
      expect(getCheckboxFor('pyannote/speaker-diarization-community-1').checked).toBe(true)
      expect(getCheckboxFor('kokoro-82m').checked).toBe(true)
    })

    it('switches to an explicit list when toggling any checkbox or clicking Clear', async () => {
      vi.spyOn(api.targets, 'listModels').mockResolvedValue(mockDiscoveredModels)
      const onChange = vi.fn()

      render(
        <SchemaForm
          schema={liveSttSchema}
          values={{ models: 'default' }}
          onChange={onChange}
          context={{ targetId: 7 }}
        />
      )

      await waitFor(() => {
        expect(screen.getByText('whisper-large-v3-turbo')).toBeTruthy()
      })

      // Toggle 'whisper-large-v3-turbo' checkbox ON while on 'default' preset
      const whisperLabel = screen.getByText('whisper-large-v3-turbo').closest('label')!
      fireEvent.click(within(whisperLabel).getByRole('checkbox'))

      expect(onChange).toHaveBeenCalledWith({
        models: [
          'parakeet-ctc-0.6b',
          'pyannote/speaker-diarization-community-1',
          'kokoro-82m',
          'whisper-large-v3-turbo',
        ],
      })

      // Clicking Clear sets models to []
      fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
      expect(onChange).toHaveBeenCalledWith({ models: [] })
    })

    it('shows built-in defaults and manual add input when no target is selected or target fails', async () => {
      const onChange = vi.fn()
      const { rerender } = render(
        <SchemaForm schema={liveSttSchema} values={{}} onChange={onChange} />
      )

      // Built-in defaults are shown
      expect(screen.getByText('realtime_eou_120m-v1')).toBeTruthy()
      expect(screen.getByText('pyannote/speaker-diarization-community-1')).toBeTruthy()
      expect(screen.getByText('kokoro-82m-v1')).toBeTruthy()

      // Manual add input is visible
      const addInput = screen.getByLabelText('Add model ID')
      fireEvent.change(addInput, { target: { value: 'whisper-large-v3-turbo' } })
      fireEvent.click(screen.getByRole('button', { name: 'Add' }))

      expect(onChange).toHaveBeenCalledWith({
        models: [
          'realtime_eou_120m-v1',
          'pyannote/speaker-diarization-community-1',
          'kokoro-82m-v1',
          'whisper-large-v3-turbo',
        ],
      })

      // Also test unreachable target fallback
      vi.spyOn(api.targets, 'listModels').mockRejectedValue(
        new Error('Failed to reach target /v1/models')
      )
      rerender(
        <SchemaForm
          schema={liveSttSchema}
          values={{}}
          onChange={onChange}
          context={{ targetId: 99 }}
        />
      )

      await waitFor(() => {
        expect(screen.getByText(/Target models unavailable/i)).toBeTruthy()
      })
      expect(screen.getByLabelText('Add model ID')).toBeTruthy()
    })

    it('shows "not on this target" warning badge when an explicit model is missing from the target', async () => {
      vi.spyOn(api.targets, 'listModels').mockResolvedValue(mockDiscoveredModels)
      const onChange = vi.fn()

      render(
        <SchemaForm
          schema={liveSttSchema}
          values={{ models: ['parakeet-ctc-0.6b', 'custom-missing-asr-model'] }}
          onChange={onChange}
          context={{ targetId: 7 }}
        />
      )

      await waitFor(() => {
        expect(screen.getByText('custom-missing-asr-model')).toBeTruthy()
      })

      const missingLabel = screen.getByText('custom-missing-asr-model').closest('label')!
      expect(within(missingLabel).getByText('not on this target')).toBeTruthy()
      expect((within(missingLabel).getByRole('checkbox') as HTMLInputElement).checked).toBe(
        true
      )
    })
  })
})

