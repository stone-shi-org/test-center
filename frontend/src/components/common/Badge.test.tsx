import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Badge } from './Badge.tsx'

describe('Badge component', () => {
  it('renders pass status with success class', () => {
    render(<Badge status="pass" />)
    const el = screen.getByText('PASS')
    expect(el).toBeDefined()
    expect(el.className).toContain('badge-success')
  })

  it('renders running status with pulsing dot', () => {
    const { container } = render(<Badge status="running" />)
    expect(screen.getByText('RUNNING')).toBeDefined()
    const dot = container.querySelector('.pulsing-dot')
    expect(dot).not.toBeNull()
  })

  it('renders failed status with error class', () => {
    render(<Badge status="failed" />)
    const el = screen.getByText('FAILED')
    expect(el.className).toContain('badge-error')
  })

  it('renders queued status with warning class', () => {
    render(<Badge status="queued" />)
    const el = screen.getByText('QUEUED')
    expect(el.className).toContain('badge-warning')
  })
})
