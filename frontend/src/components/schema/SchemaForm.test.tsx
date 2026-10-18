import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { JSONSchemaObject } from '../../api/types.ts'
import { SchemaForm } from './SchemaForm.tsx'

describe('SchemaForm component', () => {
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
})

