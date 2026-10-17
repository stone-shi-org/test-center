import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { CompareResponse, Run } from '../../api/types.ts'
import { RouterContext, matchRoute } from '../../router/context.ts'
import { RunDetailView } from '../runs/RunDetailView.tsx'
import { RunsListView } from '../runs/RunsListView.tsx'
import { CompareView } from './CompareView.tsx'
import { buildCompareUrlPath, parseCompareUrlParams } from './urlParams.ts'

const mockHistoryRuns: Run[] = [
  {
    id: 1,
    job_id: null,
    status: 'completed',
    queued_at: '2026-10-16T20:00:00Z',
    started_at: '2026-10-16T20:00:01Z',
    finished_at: '2026-10-16T20:00:10Z',
    config_snapshot: {
      tester: 'llm',
      suites: ['coding', 'reasoning'],
      config: { model: 'Qwen2.5-7B-Instruct', temperature: 0.05 },
    },
    target_snapshot: {
      id: 10,
      name: 'vllm-a100',
      kind: 'openai-compatible',
      base_url: 'http://target-a:8000/v1',
      extra: { model: 'Qwen2.5-7B-Instruct' },
    },
    git_info: null,
    results: [
      {
        id: 101,
        tester: 'llm',
        status: 'pass',
        metrics: {
          latency_p95_ms: { value: 500, unit: 'ms', better: 'lower' },
          tps: { value: 80, unit: 'tokens/s', better: 'higher' },
        },
        summary: 'All passed on A100',
        cases: [],
        artifacts: [
          {
            id: 501,
            name: 'llm_report.txt',
            path: '/tmp/1/llm_report.txt',
            mime: 'text/plain',
            size_bytes: 256,
            created_at: '2026-10-16T20:00:10Z',
          },
        ],
      },
    ],
  },
  {
    id: 2,
    job_id: null,
    status: 'completed',
    queued_at: '2026-10-16T20:05:00Z',
    started_at: '2026-10-16T20:05:01Z',
    finished_at: '2026-10-16T20:05:12Z',
    config_snapshot: {
      tester: 'llm',
      suites: ['coding', 'reasoning'],
      config: { model: 'Qwen2.5-32B-Instruct', temperature: 0.1 },
    },
    target_snapshot: {
      id: 11,
      name: 'vllm-h100',
      kind: 'openai-compatible',
      base_url: 'http://target-b:8000/v1',
      extra: { model: 'Qwen2.5-32B-Instruct' },
    },
    git_info: null,
    results: [
      {
        id: 102,
        tester: 'llm',
        status: 'fail',
        metrics: {
          latency_p95_ms: { value: 650, unit: 'ms', better: 'lower' },
          tps: { value: 100, unit: 'tokens/s', better: 'higher' },
        },
        summary: '1 coding case failed on H100',
        cases: [],
        artifacts: [
          {
            id: 502,
            name: 'llm_report.txt',
            path: '/tmp/2/llm_report.txt',
            mime: 'text/plain',
            size_bytes: 312,
            created_at: '2026-10-16T20:05:12Z',
          },
        ],
      },
    ],
  },
]

const mockCompareResponse: CompareResponse = {
  tester: 'llm',
  threshold: 0,
  baseline_run_id: 1,
  baseline_result_id: 101,
  run_ids: [1, 2],
  result_ids: [101, 102],
  runs: [
    {
      run_id: 1,
      result_id: 101,
      job_id: null,
      test_definition_id: null,
      is_baseline: true,
      tester: 'llm',
      status: 'pass',
      result_status: 'pass',
      run_status: 'completed',
      summary: 'All passed on A100',
      target_id: 10,
      target_name: 'vllm-a100',
      target_kind: 'openai-compatible',
      base_url: 'http://target-a:8000/v1',
      model: 'Qwen2.5-7B-Instruct',
      models: ['Qwen2.5-7B-Instruct'],
      suites: ['coding', 'reasoning'],
      config: { model: 'Qwen2.5-7B-Instruct', temperature: 0.05 },
      config_snapshot: {},
      target_snapshot: {},
      git_info: null,
      queued_at: '2026-10-16T20:00:00Z',
      started_at: '2026-10-16T20:00:01Z',
      finished_at: '2026-10-16T20:00:10Z',
      case_count: 2,
      artifact_count: 1,
      artifacts: [
        {
          id: 501,
          result_id: 101,
          run_id: 1,
          name: 'llm_report.txt',
          path: '/tmp/1/llm_report.txt',
          mime: 'text/plain',
          size_bytes: 256,
          created_at: '2026-10-16T20:00:10Z',
          download_url: '/api/runs/1/artifacts/501/download',
        },
      ],
    },
    {
      run_id: 2,
      result_id: 102,
      job_id: null,
      test_definition_id: null,
      is_baseline: false,
      tester: 'llm',
      status: 'fail',
      result_status: 'fail',
      run_status: 'completed',
      summary: '1 coding case failed on H100',
      target_id: 11,
      target_name: 'vllm-h100',
      target_kind: 'openai-compatible',
      base_url: 'http://target-b:8000/v1',
      model: 'Qwen2.5-32B-Instruct',
      models: ['Qwen2.5-32B-Instruct'],
      suites: ['coding', 'reasoning'],
      config: { model: 'Qwen2.5-32B-Instruct', temperature: 0.1 },
      config_snapshot: {},
      target_snapshot: {},
      git_info: null,
      queued_at: '2026-10-16T20:05:00Z',
      started_at: '2026-10-16T20:05:01Z',
      finished_at: '2026-10-16T20:05:12Z',
      case_count: 2,
      artifact_count: 1,
      artifacts: [
        {
          id: 502,
          result_id: 102,
          run_id: 2,
          name: 'llm_report.txt',
          path: '/tmp/2/llm_report.txt',
          mime: 'text/plain',
          size_bytes: 312,
          created_at: '2026-10-16T20:05:12Z',
          download_url: '/api/runs/2/artifacts/502/download',
        },
      ],
    },
  ],
  config_diff: {
    targets_differ: true,
    models_differ: true,
    suites_differ: false,
    config_keys_changed: ['model', 'temperature'],
    config_changes: {
      model: [
        { run_id: 1, result_id: 101, value: 'Qwen2.5-7B-Instruct' },
        { run_id: 2, result_id: 102, value: 'Qwen2.5-32B-Instruct' },
      ],
      temperature: [
        { run_id: 1, result_id: 101, value: 0.05 },
        { run_id: 2, result_id: 102, value: 0.1 },
      ],
    },
    targets: [
      {
        run_id: 1,
        result_id: 101,
        target_id: 10,
        target_name: 'vllm-a100',
        base_url: 'http://target-a:8000/v1',
        model: 'Qwen2.5-7B-Instruct',
      },
      {
        run_id: 2,
        result_id: 102,
        target_id: 11,
        target_name: 'vllm-h100',
        base_url: 'http://target-b:8000/v1',
        model: 'Qwen2.5-32B-Instruct',
      },
    ],
  },
  summary: {
    has_regression: true,
    has_improvement: true,
    total_metrics: 3,
    metrics_changed: 2,
    metrics_regressed: 1,
    metrics_improved: 1,
    metrics_unchanged: 1,
    metrics_missing: 0,
    total_cases: 2,
    cases_changed: 1,
    cases_status_changed: 1,
    cases_regressed: 1,
    cases_status_regressed: 1,
    cases_metric_regressed: 1,
    cases_improved: 0,
    cases_status_improved: 0,
    cases_unchanged: 1,
    cases_missing: 0,
    regressions: [],
  },
  metrics: [
    {
      name: 'latency_p95_ms',
      unit: 'ms',
      better: 'lower',
      threshold: 0,
      baseline_run_id: 1,
      baseline_result_id: 101,
      baseline_value: 500,
      target_value: 650,
      delta: 150,
      abs_delta: 150,
      pct_delta: 30,
      regression: true,
      has_regression: true,
      improvement: false,
      has_improvement: false,
      changed: true,
      has_change: true,
      missing_in_some: false,
      flag: 'regression',
      values: [
        {
          run_id: 1,
          result_id: 101,
          is_baseline: true,
          value: 500,
          present: true,
          delta: 0,
          abs_delta: 0,
          pct_delta: 0,
          regression: false,
          improvement: false,
          flag: 'baseline',
        },
        {
          run_id: 2,
          result_id: 102,
          is_baseline: false,
          value: 650,
          present: true,
          delta: 150,
          abs_delta: 150,
          pct_delta: 30,
          regression: true,
          improvement: false,
          flag: 'regression',
        },
      ],
      comparisons: [],
    },
    {
      name: 'tps',
      unit: 'tokens/s',
      better: 'higher',
      threshold: 0,
      baseline_run_id: 1,
      baseline_result_id: 101,
      baseline_value: 80,
      target_value: 100,
      delta: 20,
      abs_delta: 20,
      pct_delta: 25,
      regression: false,
      has_regression: false,
      improvement: true,
      has_improvement: true,
      changed: true,
      has_change: true,
      missing_in_some: false,
      flag: 'improvement',
      values: [
        {
          run_id: 1,
          result_id: 101,
          is_baseline: true,
          value: 80,
          present: true,
          delta: 0,
          abs_delta: 0,
          pct_delta: 0,
          regression: false,
          improvement: false,
          flag: 'baseline',
        },
        {
          run_id: 2,
          result_id: 102,
          is_baseline: false,
          value: 100,
          present: true,
          delta: 20,
          abs_delta: 20,
          pct_delta: 25,
          regression: false,
          improvement: true,
          flag: 'improvement',
        },
      ],
      comparisons: [],
    },
    {
      name: 'prompt_tokens',
      unit: 'tokens',
      better: null,
      threshold: 0,
      baseline_run_id: 1,
      baseline_result_id: 101,
      baseline_value: 250,
      target_value: 250,
      delta: 0,
      abs_delta: 0,
      pct_delta: 0,
      regression: false,
      has_regression: false,
      improvement: false,
      has_improvement: false,
      changed: false,
      has_change: false,
      missing_in_some: false,
      flag: 'unchanged',
      values: [
        {
          run_id: 1,
          result_id: 101,
          is_baseline: true,
          value: 250,
          present: true,
          delta: 0,
          abs_delta: 0,
          pct_delta: 0,
          regression: false,
          improvement: false,
          flag: 'baseline',
        },
        {
          run_id: 2,
          result_id: 102,
          is_baseline: false,
          value: 250,
          present: true,
          delta: 0,
          abs_delta: 0,
          pct_delta: 0,
          regression: false,
          improvement: false,
          flag: 'unchanged',
        },
      ],
      comparisons: [],
    },
  ],
  metric_map: {},
  cases: [
    {
      case_id: 'coding_lru_cache',
      suite: 'coding',
      baseline_run_id: 1,
      baseline_result_id: 101,
      baseline_status: 'pass',
      target_status: 'fail',
      status_change: 'regression',
      status_changed: true,
      status_regression: true,
      status_improvement: false,
      has_status_regression: true,
      has_status_improvement: false,
      has_metric_regression: true,
      has_metric_improvement: false,
      has_metric_change: true,
      regression: true,
      has_regression: true,
      improvement: false,
      has_improvement: false,
      changed: true,
      has_change: true,
      missing_in_some: false,
      metrics: [
        {
          name: 'ttft_ms',
          unit: 'ms',
          better: 'lower',
          threshold: 0,
          baseline_run_id: 1,
          baseline_result_id: 101,
          baseline_value: 95,
          target_value: 133,
          delta: 38,
          abs_delta: 38,
          pct_delta: 40,
          regression: true,
          has_regression: true,
          improvement: false,
          has_improvement: false,
          changed: true,
          has_change: true,
          missing_in_some: false,
          flag: 'regression',
          values: [
            {
              run_id: 1,
              result_id: 101,
              is_baseline: true,
              value: 95,
              present: true,
              delta: 0,
              abs_delta: 0,
              pct_delta: 0,
              regression: false,
              improvement: false,
              flag: 'baseline',
            },
            {
              run_id: 2,
              result_id: 102,
              is_baseline: false,
              value: 133,
              present: true,
              delta: 38,
              abs_delta: 38,
              pct_delta: 40,
              regression: true,
              improvement: false,
              flag: 'regression',
            },
          ],
          comparisons: [],
        },
      ],
      metric_changes: {},
      values: [
        {
          run_id: 1,
          result_id: 101,
          case_result_id: 201,
          original_case_id: 'coding_lru_cache',
          is_baseline: true,
          present: true,
          status: 'pass',
          status_change: 'baseline',
          status_changed: false,
          status_regression: false,
          status_improvement: false,
          metric_regression: false,
          metric_improvement: false,
          regression: false,
          improvement: false,
          metrics: { ttft_ms: { value: 95, unit: 'ms', better: 'lower' } },
          metric_deltas: {
            ttft_ms: {
              name: 'ttft_ms',
              unit: 'ms',
              better: 'lower',
              baseline_value: 95,
              run_id: 1,
              result_id: 101,
              is_baseline: true,
              value: 95,
              present: true,
              delta: 0,
              abs_delta: 0,
              pct_delta: 0,
              regression: false,
              improvement: false,
              flag: 'baseline',
            },
          },
          detail: { passed: true, eval_message: 'All unit tests passed' },
        },
        {
          run_id: 2,
          result_id: 102,
          case_result_id: 202,
          original_case_id: 'coding_lru_cache',
          is_baseline: false,
          present: true,
          status: 'fail',
          status_change: 'regression',
          status_changed: true,
          status_regression: true,
          status_improvement: false,
          metric_regression: true,
          metric_improvement: false,
          regression: true,
          improvement: false,
          metrics: { ttft_ms: { value: 133, unit: 'ms', better: 'lower' } },
          metric_deltas: {
            ttft_ms: {
              name: 'ttft_ms',
              unit: 'ms',
              better: 'lower',
              baseline_value: 95,
              run_id: 2,
              result_id: 102,
              is_baseline: false,
              value: 133,
              present: true,
              delta: 38,
              abs_delta: 38,
              pct_delta: 40,
              regression: true,
              improvement: false,
              flag: 'regression',
            },
          },
          detail: { passed: false, eval_message: 'AssertionError on LRU eviction' },
        },
      ],
      comparisons: [],
    },
    {
      case_id: 'reasoning_steady',
      suite: 'reasoning',
      baseline_run_id: 1,
      baseline_result_id: 101,
      baseline_status: 'pass',
      target_status: 'pass',
      status_change: 'unchanged',
      status_changed: false,
      status_regression: false,
      status_improvement: false,
      has_status_regression: false,
      has_status_improvement: false,
      has_metric_regression: false,
      has_metric_improvement: false,
      has_metric_change: false,
      regression: false,
      has_regression: false,
      improvement: false,
      has_improvement: false,
      changed: false,
      has_change: false,
      missing_in_some: false,
      metrics: [],
      metric_changes: {},
      values: [
        {
          run_id: 1,
          result_id: 101,
          case_result_id: 203,
          original_case_id: 'reasoning_steady',
          is_baseline: true,
          present: true,
          status: 'pass',
          status_change: 'baseline',
          status_changed: false,
          status_regression: false,
          status_improvement: false,
          metric_regression: false,
          metric_improvement: false,
          regression: false,
          improvement: false,
          metrics: {},
          metric_deltas: {},
          detail: { passed: true },
        },
        {
          run_id: 2,
          result_id: 102,
          case_result_id: 204,
          original_case_id: 'reasoning_steady',
          is_baseline: false,
          present: true,
          status: 'pass',
          status_change: 'unchanged',
          status_changed: false,
          status_regression: false,
          status_improvement: false,
          metric_regression: false,
          metric_improvement: false,
          regression: false,
          improvement: false,
          metrics: {},
          metric_deltas: {},
          detail: { passed: true },
        },
      ],
      comparisons: [],
    },
  ],
  case_map: {},
}

describe('CompareView and Compare UI components', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('parses and builds shareable compare URLs accurately', () => {
    const parsed = parseCompareUrlParams({
      run_ids: '1,2,3',
      baseline_run_id: '2',
      threshold: '5',
      regressed_only: 'true',
      suite: 'coding',
    })
    expect(parsed).toEqual({
      runIds: [1, 2, 3],
      baselineRunId: 2,
      threshold: 5,
      filterMode: 'regressed',
      suite: 'coding',
    })

    const built = buildCompareUrlPath(parsed)
    expect(built).toBe(
      '/compare?run_ids=1%2C2%2C3&baseline_run_id=2&threshold=5&regressed_only=true&suite=coding'
    )
  })

  it('picks runs from history in CompareView and renders full comparison', async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.startsWith('/api/runs?')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockHistoryRuns,
        } as Response
      }
      if (url.startsWith('/api/compare')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockCompareResponse,
        } as Response
      }
      throw new Error(`Unexpected fetch URL: ${url}`)
    })

    const navigate = vi.fn()
    render(
      <RouterContext.Provider value={{ route: matchRoute('/compare'), navigate }}>
        <CompareView />
      </RouterContext.Provider>
    )

    // Initially shows run picker and empty prompt
    expect(screen.getByTestId('compare-empty-prompt')).toBeDefined()
    await waitFor(() => {
      expect(screen.getByTestId('picker-row-1')).toBeDefined()
      expect(screen.getByTestId('picker-row-2')).toBeDefined()
    })

    // Pick Run #1 and Run #2 from the Run Picker
    const check1 = screen.getByRole('checkbox', { name: /Select run #1/i })
    const check2 = screen.getByRole('checkbox', { name: /Select run #2/i })
    fireEvent.click(check1)
    fireEvent.click(check2)

    // Wait for Compare results to render
    await waitFor(() => {
      expect(screen.getByTestId('compare-results')).toBeDefined()
    })

    // Verify Shareable URL includes run_ids=1,2 and baseline_run_id=1
    const urlBox = screen.getByTestId('shareable-url-box')
    expect(urlBox.textContent).toContain('run_ids=1%2C2')
    expect(urlBox.textContent).toContain('baseline_run_id=1')
  })

  it('renders config diff header, metric delta table, charts, per-case diff, and side-by-side artifacts', async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.startsWith('/api/runs?')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockHistoryRuns,
        } as Response
      }
      if (url.startsWith('/api/compare')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockCompareResponse,
        } as Response
      }
      if (url === '/api/runs/1/artifacts/501/download') {
        return {
          ok: true,
          status: 200,
          text: async () => 'Run 1 Report\nStatus: PASS\nLatency: 500ms',
        } as Response
      }
      if (url === '/api/runs/2/artifacts/502/download') {
        return {
          ok: true,
          status: 200,
          text: async () => 'Run 2 Report\nStatus: FAIL\nLatency: 650ms',
        } as Response
      }
      throw new Error(`Unexpected fetch URL: ${url}`)
    })

    const navigate = vi.fn()
    render(
      <RouterContext.Provider
        value={{
          route: matchRoute('/compare?run_ids=1,2&baseline_run_id=1'),
          navigate,
        }}
      >
        <CompareView />
      </RouterContext.Provider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('compare-results')).toBeDefined()
    })

    // 1. Config & Target Diff Header
    const configHeader = screen.getByTestId('config-diff-header')
    expect(configHeader.textContent).toContain('Targets Differ')
    expect(configHeader.textContent).toContain('Models Differ')
    expect(configHeader.textContent).toContain('vllm-a100')
    expect(configHeader.textContent).toContain('vllm-h100')
    expect(configHeader.textContent).toContain('Qwen2.5-7B-Instruct')
    expect(configHeader.textContent).toContain('Qwen2.5-32B-Instruct')
    expect(screen.getByTestId('config-diff-table').textContent).toContain('config.temperature')

    // 2. Bar and Line Charts for Key Metrics
    expect(screen.getByTestId('metrics-chart-section')).toBeDefined()
    expect(screen.getByRole('img', { name: 'latency_p95_ms bar chart' })).toBeDefined()

    // Switch to Line Chart
    fireEvent.click(screen.getByTestId('chart-type-line'))
    expect(screen.getByRole('img', { name: 'latency_p95_ms line chart' })).toBeDefined()

    // 3. Metric Delta Table & Regression Highlighting
    const latRow = screen.getByTestId('metric-row-latency_p95_ms')
    expect(latRow.className).toContain('row-regression')
    expect(latRow.textContent).toContain('+150')
    expect(latRow.textContent).toContain('+30.00%')

    const tpsRow = screen.getByTestId('metric-row-tps')
    expect(tpsRow.className).toContain('row-improvement')
    expect(tpsRow.textContent).toContain('+20')
    expect(tpsRow.textContent).toContain('+25.00%')

    // 4. Filter to Changed Only / Regressed Only
    expect(screen.getByTestId('metric-row-prompt_tokens')).toBeDefined()
    expect(screen.getByTestId('case-diff-row-reasoning_steady')).toBeDefined()

    // Click "Changed Only" -> prompt_tokens and reasoning_steady should be hidden
    fireEvent.click(screen.getByTestId('filter-changed-btn'))
    expect(screen.queryByTestId('metric-row-prompt_tokens')).toBeNull()
    expect(screen.queryByTestId('case-diff-row-reasoning_steady')).toBeNull()
    expect(screen.getByTestId('metric-row-latency_p95_ms')).toBeDefined()
    expect(screen.getByTestId('metric-row-tps')).toBeDefined()

    // Click "Regressed Only" -> tps (improved) should be hidden, latency_p95_ms and coding_lru_cache remain
    fireEvent.click(screen.getByTestId('filter-regressed-btn'))
    expect(screen.queryByTestId('metric-row-tps')).toBeNull()
    expect(screen.getByTestId('metric-row-latency_p95_ms')).toBeDefined()
    expect(screen.getByTestId('case-diff-row-coding_lru_cache')).toBeDefined()

    // Reset filter to All
    fireEvent.click(screen.getByTestId('filter-all-btn'))

    // 5. Per-Case Diff View & Expandable Detail Diff Drawer
    const lruCaseRow = screen.getByTestId('case-diff-row-coding_lru_cache')
    expect(lruCaseRow.textContent).toContain('pass → fail')
    fireEvent.click(lruCaseRow)

    const caseDrawer = screen.getByTestId('case-diff-drawer-coding_lru_cache')
    expect(caseDrawer.textContent).toContain('AssertionError on LRU eviction')
    expect(caseDrawer.textContent).toContain('All unit tests passed')

    // 6. Artifacts Side by Side
    const artSection = screen.getByTestId('artifacts-compare-section')
    expect(artSection.textContent).toContain('llm_report.txt')

    const compareArtBtn = screen.getByTestId('compare-artifact-btn-llm_report.txt')
    fireEvent.click(compareArtBtn)

    await waitFor(() => {
      const sbsDrawer = screen.getByTestId('artifact-side-by-side-llm_report.txt')
      expect(sbsDrawer.textContent).toContain('Contents Differ')
      expect(sbsDrawer.textContent).toContain('Run 1 Report')
      expect(sbsDrawer.textContent).toContain('Run 2 Report')
    })

    // 7. Switching baseline re-queries /api/compare with new baseline_run_id
    const baselineSelect = screen.getByTestId('baseline-select')
    fireEvent.change(baselineSelect, { target: { value: '2' } })

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('baseline_run_id=2'),
        expect.any(Object)
      )
    })
  })

  it('supports picking runs in RunsListView and clicking Compare in RunDetailView', async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.startsWith('/api/runs/1/logs')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ run_id: 1, status: 'completed', lines: [], raw: '' }),
        } as Response
      }
      if (url.startsWith('/api/runs/1')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockHistoryRuns[0],
        } as Response
      }
      if (url.startsWith('/api/runs')) {
        return {
          ok: true,
          status: 200,
          json: async () => mockHistoryRuns,
        } as Response
      }
      throw new Error(`Unexpected URL: ${url}`)
    })

    const navigate = vi.fn()
    const { unmount } = render(
      <RouterContext.Provider value={{ route: matchRoute('/runs'), navigate }}>
        <RunsListView />
      </RouterContext.Provider>
    )

    await waitFor(() => {
      expect(screen.getByLabelText('Select run #1 for comparison')).toBeDefined()
    })

    fireEvent.click(screen.getByLabelText('Select run #1 for comparison'))
    fireEvent.click(screen.getByLabelText('Select run #2 for comparison'))

    const compareBtn = screen.getByRole('button', { name: /Compare Selected \(2\)/i })
    fireEvent.click(compareBtn)
    expect(navigate).toHaveBeenCalledWith('/compare?run_ids=1,2')

    unmount()
    navigate.mockClear()

    // Test Compare button on RunDetailView
    render(
      <RouterContext.Provider value={{ route: matchRoute('/runs/1'), navigate }}>
        <RunDetailView runId={1} />
      </RouterContext.Provider>
    )

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Compare/i })).toBeDefined()
    })
    fireEvent.click(screen.getByRole('button', { name: /Compare/i }))
    expect(navigate).toHaveBeenCalledWith('/compare?run_ids=1&baseline_run_id=1')
  })
})
