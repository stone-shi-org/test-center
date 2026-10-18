import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../api/index.ts'
import { TargetsView } from './TargetsView.tsx'

describe('TargetsView capability selection and Test button', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const mockTargets = [
    {
      id: 1,
      name: 'vllm-unified',
      kind: 'openai-compatible',
      base_url: 'http://localhost:8000/v1',
      auth_env_var: 'OPENAI_API_KEY',
      extra: { model: 'qwen2.5-7b', capabilities: ['chat', 'embedding'] },
      capabilities: ['chat', 'embedding'],
      created_at: '2026-10-17T00:00:00Z',
      updated_at: '2026-10-17T00:00:00Z',
    },
  ]

  it('renders target capabilities and runs capability test when Test button is clicked', async () => {
    vi.spyOn(api.targets, 'list').mockResolvedValue(mockTargets)
    const testSpy = vi.spyOn(api.targets, 'test').mockResolvedValue({
      target_id: 1,
      target_name: 'vllm-unified',
      kind: 'openai-compatible',
      base_url: 'http://localhost:8000/v1',
      auth_env_var: 'OPENAI_API_KEY',
      auth_configured: true,
      ok: true,
      capabilities_tested: ['chat', 'embedding'],
      results: [
        {
          capability: 'chat',
          ok: true,
          status_code: 200,
          latency_ms: 14.2,
          model: 'qwen2.5-7b',
          message: 'Chat completion succeeded (14.2 ms)',
          detail: { response_preview: 'OK', endpoint: 'http://localhost:8000/v1/chat/completions' },
        },
        {
          capability: 'embedding',
          ok: true,
          status_code: 200,
          latency_ms: 9.8,
          model: 'qwen2.5-7b',
          message: 'Embedding succeeded (768-d vector in 9.8 ms)',
          detail: { dimensions: 768, endpoint: 'http://localhost:8000/v1/embeddings' },
        },
      ],
      tested_at: '2026-10-18T00:00:00Z',
    })

    render(<TargetsView />)

    await waitFor(() => {
      expect(screen.getByText('vllm-unified')).toBeTruthy()
    })

    // Verify capabilities are displayed on the row
    expect(screen.getByText('chat')).toBeTruthy()
    expect(screen.getByText('embedding')).toBeTruthy()

    // Click the Test button on the target row
    const testBtn = screen.getByRole('button', { name: 'Test' })
    fireEvent.click(testBtn)

    await waitFor(() => {
      expect(testSpy).toHaveBeenCalledWith(1, {
        capabilities: ['chat', 'embedding'],
        model: 'qwen2.5-7b',
      })
    })

    await waitFor(() => {
      expect(screen.getByText(/Chat completion succeeded/i)).toBeTruthy()
      expect(screen.getByText(/Embedding succeeded/i)).toBeTruthy()
    })
  })

  it('allows picking capabilities when creating a new target', async () => {
    vi.spyOn(api.targets, 'list').mockResolvedValue(mockTargets)
    const createSpy = vi.spyOn(api.targets, 'create').mockResolvedValue({
      ...mockTargets[0],
      id: 2,
      name: 'new-multi-target',
    })

    render(<TargetsView />)

    await waitFor(() => {
      expect(screen.getByText('vllm-unified')).toBeTruthy()
    })

    fireEvent.click(screen.getByRole('button', { name: /\+ Add Target/i }))

    const nameInput = screen.getByLabelText(/Target Name/i)
    const urlInput = screen.getByLabelText(/Base URL/i)
    fireEvent.change(nameInput, { target: { value: 'new-multi-target' } })
    fireEvent.change(urlInput, { target: { value: 'http://localhost:8001/v1' } })

    // Toggle 'embedding' capability in addition to default 'chat'
    const capGroup = screen.getByRole('group', { name: /Target Capabilities/i })
    const embedBtn = Array.from(capGroup.querySelectorAll('button')).find((b) =>
      b.textContent?.includes('embedding')
    )
    expect(embedBtn).toBeTruthy()
    fireEvent.click(embedBtn!)

    fireEvent.click(screen.getByRole('button', { name: /Create Target/i }))

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'new-multi-target',
          kind: 'openai-compatible',
          base_url: 'http://localhost:8001/v1',
          capabilities: ['chat', 'embedding'],
        })
      )
    })
  })
})
