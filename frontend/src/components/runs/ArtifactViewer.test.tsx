import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Artifact, Run } from '../../api/types.ts'
import { RouterContext, matchRoute } from '../../router/context.ts'
import {
  ArtifactViewer,
  detectArtifactKind,
  formatArtifactDisplayPath,
} from './ArtifactViewer.tsx'
import { RunDetailView } from './RunDetailView.tsx'

const mockArtifacts: Artifact[] = [
  {
    id: 701,
    name: 'report.html',
    path: '/app/data/artifacts/42/report.html',
    mime: 'text/html',
    size_bytes: 1024,
    created_at: '2026-10-17T01:00:00Z',
  },
  {
    id: 702,
    name: 'metrics.json',
    path: '/app/data/artifacts/42/metrics.json',
    mime: 'application/json',
    size_bytes: 512,
    created_at: '2026-10-17T01:00:01Z',
  },
  {
    id: 703,
    name: 'utterance_01.wav',
    path: '/app/data/artifacts/42/utterance_01.wav',
    mime: 'audio/wav',
    size_bytes: 32768,
    created_at: '2026-10-17T01:00:02Z',
  },
  {
    id: 704,
    name: 'execution.log',
    path: '/app/data/artifacts/42/execution.log',
    mime: 'text/plain',
    size_bytes: 2048,
    created_at: '2026-10-17T01:00:03Z',
  },
]

const mockHtmlContent =
  '<!doctype html><html><body><h1>Evaluation Summary</h1><script>window.evil=true</script></body></html>'

const mockJsonContent = JSON.stringify(
  {
    model: 'whisper-large-v3',
    summary: {
      wer: 0.035,
      rtf: 0.18,
    },
    utterances: [
      { id: 'utt_1', status: 'pass' },
      { id: 'utt_2', status: 'fail' },
    ],
  },
  null,
  2
)

const mockLogContent = [
  '[10:00:01] [INFO] Starting evaluation run #42',
  '[10:00:02] [WARNING] High jitter on chunk #3 (145ms)',
  '[10:00:03] [ERROR] Timeout waiting for final transcript on utt_2',
  '[10:00:04] [INFO] Run finished with 1 error',
].join('\n')

describe('TES-35 ArtifactViewer & RunArtifactsSection', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === 'string' ? input : input.toString()

        if (url.includes('/artifacts/701/download')) {
          return new Response(mockHtmlContent, {
            status: 200,
            headers: { 'Content-Type': 'text/html' },
          })
        }
        if (url.includes('/artifacts/702/download')) {
          return new Response(mockJsonContent, {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        }
        if (url.includes('/artifacts/704/download')) {
          return new Response(mockLogContent, {
            status: 200,
            headers: { 'Content-Type': 'text/plain' },
          })
        }
        if (url.endsWith('/api/runs/42/artifacts')) {
          return new Response(JSON.stringify(mockArtifacts), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        }
        if (url.endsWith('/api/runs/42')) {
          const run: Run = {
            id: 42,
            job_id: null,
            status: 'completed',
            queued_at: '2026-10-17T00:59:00Z',
            started_at: '2026-10-17T00:59:05Z',
            finished_at: '2026-10-17T01:00:05Z',
            config_snapshot: {
              tester: 'live_stt',
              suites: ['realtime'],
              config: { model: 'whisper-large-v3' },
            },
            target_snapshot: {},
            git_info: null,
            results: [
              {
                id: 901,
                tester: 'live_stt',
                status: 'pass',
                metrics: { wer: 0.035 },
                summary: 'Completed realtime suite',
                cases: [],
                artifacts: mockArtifacts,
              },
            ],
          }
          return new Response(JSON.stringify(run), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        }
        return new Response('Not Found', { status: 404 })
      })
    )
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('classifies HTML, JSON, audio, and log artifacts and formats data/artifacts/[run_id]/ paths', () => {
    expect(detectArtifactKind(mockArtifacts[0])).toBe('html')
    expect(detectArtifactKind(mockArtifacts[1])).toBe('json')
    expect(detectArtifactKind(mockArtifacts[2])).toBe('audio')
    expect(detectArtifactKind(mockArtifacts[3])).toBe('log')

    expect(formatArtifactDisplayPath(42, mockArtifacts[0])).toBe(
      'data/artifacts/42/report.html'
    )
  })

  it('renders HTML artifact in a strict sandboxed iframe with source toggle', async () => {
    render(<ArtifactViewer runId={42} artifact={mockArtifacts[0]} />)

    await waitFor(() => {
      expect(screen.getByTestId('artifact-html-viewer')).toBeTruthy()
    })

    expect(screen.getByTestId('html-sandbox-badge').textContent).toContain(
      'Sandboxed HTML (scripts disabled)'
    )

    const iframe = screen.getByTestId('artifact-html-iframe') as HTMLIFrameElement
    expect(iframe).toBeTruthy()
    // Strict sandbox="" attribute prevents script execution & same-origin access
    expect(iframe.getAttribute('sandbox')).toBe('')
    expect(iframe.getAttribute('srcdoc')).toContain('<h1>Evaluation Summary</h1>')

    // Toggle to HTML Source mode
    fireEvent.click(screen.getByTestId('html-mode-source'))
    expect(screen.getByTestId('artifact-html-source').textContent).toContain(
      '<h1>Evaluation Summary</h1>'
    )
  })

  it('renders interactive JSON tree with expand/collapse, search, and raw mode', async () => {
    render(<ArtifactViewer runId={42} artifact={mockArtifacts[1]} />)

    await waitFor(() => {
      expect(screen.getByTestId('artifact-json-viewer')).toBeTruthy()
    })

    expect(screen.getByTestId('artifact-json-tree')).toBeTruthy()
    expect(screen.getByTestId('json-leaf-root.model').textContent).toContain(
      '"whisper-large-v3"'
    )
    expect(screen.getByTestId('json-leaf-root.summary.wer').textContent).toContain(
      '0.035'
    )

    // Collapse All hides deep leaves
    fireEvent.click(screen.getByTestId('json-collapse-all'))
    expect(screen.queryByTestId('json-leaf-root.model')).toBeNull()

    // Expand All reveals deep leaves inside arrays
    fireEvent.click(screen.getByTestId('json-expand-all'))
    expect(
      screen.getByTestId('json-leaf-root.utterances.0.id').textContent
    ).toContain('"utt_1"')

    // Search filters tree to matching keys/values
    fireEvent.change(screen.getByTestId('json-tree-search'), {
      target: { value: '0.18' },
    })
    expect(screen.getByTestId('json-leaf-root.summary.rtf').textContent).toContain(
      '0.18'
    )
    expect(screen.queryByTestId('json-leaf-root.model')).toBeNull()

    // Switch to Raw JSON mode
    fireEvent.click(screen.getByTestId('json-mode-raw'))
    expect(screen.getByTestId('artifact-json-raw').textContent).toContain(
      '"whisper-large-v3"'
    )
  })

  it('renders audio player with speed selector and loop toggle', () => {
    render(<ArtifactViewer runId={42} artifact={mockArtifacts[2]} />)

    expect(screen.getByTestId('artifact-audio-viewer')).toBeTruthy()
    const audioEl = screen.getByTestId('artifact-audio-element') as HTMLAudioElement
    expect(audioEl).toBeTruthy()
    expect(audioEl.getAttribute('src')).toBe('/api/runs/42/artifacts/703/preview')
    expect(audioEl.loop).toBe(false)

    // Toggle loop
    fireEvent.click(screen.getByTestId('audio-loop-toggle'))
    expect(audioEl.loop).toBe(true)

    // Change playback speed
    fireEvent.change(screen.getByTestId('audio-speed-select'), {
      target: { value: '1.5' },
    })
    expect(audioEl.playbackRate).toBe(1.5)
  })

  it('renders log viewer with line numbers, level filter, and text search', async () => {
    render(<ArtifactViewer runId={42} artifact={mockArtifacts[3]} />)

    await waitFor(() => {
      expect(screen.getByTestId('artifact-log-viewer')).toBeTruthy()
    })

    expect(screen.getByTestId('log-viewer-line-count').textContent).toContain(
      '4 / 4 lines'
    )

    // Filter by ERROR level
    fireEvent.change(screen.getByTestId('artifact-log-level'), {
      target: { value: 'ERROR' },
    })
    expect(screen.getByTestId('log-viewer-line-count').textContent).toContain(
      '1 / 4 lines'
    )
    expect(screen.getByTestId('artifact-log-terminal').textContent).toContain(
      'Timeout waiting for final transcript on utt_2'
    )

    // Reset level to ALL and filter by text query
    fireEvent.change(screen.getByTestId('artifact-log-level'), {
      target: { value: 'ALL' },
    })
    fireEvent.change(screen.getByTestId('artifact-log-filter'), {
      target: { value: 'jitter' },
    })
    expect(screen.getByTestId('log-viewer-line-count').textContent).toContain(
      '1 / 4 lines'
    )
    expect(screen.getByTestId('artifact-log-terminal').textContent).toContain(
      'High jitter on chunk #3'
    )
  })

  it('lists run artifacts in RunArtifactsSection and RunDetailView with preview & download', async () => {
    render(
      <RouterContext.Provider
        value={{
          route: matchRoute('/runs/42'),
          navigate: vi.fn(),
        }}
      >
        <RunDetailView runId={42} />
      </RouterContext.Provider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('run-detail-artifacts-tab')).toBeTruthy()
    })
    expect(screen.getByTestId('run-detail-artifacts-tab').textContent).toContain(
      'Artifacts (4)'
    )

    // Switch to the Artifacts tab
    fireEvent.click(screen.getByTestId('run-detail-artifacts-tab'))

    await waitFor(() => {
      expect(screen.getByTestId('run-artifacts-section')).toBeTruthy()
    })

    // Verify all 4 artifacts have rows and download links
    for (const art of mockArtifacts) {
      expect(screen.getByTestId(`run-artifact-row-${art.name}`)).toBeTruthy()
      const dlBtn = screen.getByTestId(
        `download-artifact-btn-${art.name}`
      ) as HTMLAnchorElement
      expect(dlBtn.getAttribute('href')).toBe(
        `/api/runs/42/artifacts/${art.id}/download`
      )
      expect(dlBtn.getAttribute('download')).toBe(art.name)
    }

    // Open preview for report.html
    fireEvent.click(screen.getByTestId('preview-artifact-btn-report.html'))
    await waitFor(() => {
      expect(screen.getByTestId('artifact-html-iframe')).toBeTruthy()
    })

    // Switch preview to metrics.json via switcher pill
    fireEvent.click(screen.getByTestId('switch-artifact-pill-metrics.json'))
    await waitFor(() => {
      expect(screen.getByTestId('artifact-json-tree')).toBeTruthy()
    })
  })
})
