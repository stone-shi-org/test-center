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
})
