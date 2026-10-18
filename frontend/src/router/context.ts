import { createContext, useContext } from 'react'

export type RouteName =
  | 'dashboard'
  | 'targets'
  | 'templates'
  | 'jobs'
  | 'trigger'
  | 'runs'
  | 'run-detail'
  | 'compare'

export interface RouteMatch {
  name: RouteName
  params: Record<string, string>
  path: string
}

export interface RouterContextType {
  route: RouteMatch
  navigate: (path: string, options?: { replace?: boolean }) => void
}

export function parsePath(pathname: string, hash: string, search = ''): string {
  if (hash.startsWith('#/')) {
    return hash.slice(1)
  }
  const base = pathname || '/'
  if (search && !base.includes('?')) {
    return `${base}${search.startsWith('?') ? search : `?${search}`}`
  }
  return base
}

export function parseQueryParams(path: string): Record<string, string> {
  const qIndex = path.indexOf('?')
  if (qIndex === -1) return {}
  const rawQuery = path.slice(qIndex + 1)
  const searchParams = new URLSearchParams(rawQuery)
  const params: Record<string, string> = {}
  searchParams.forEach((value, key) => {
    if (params[key]) {
      params[key] = `${params[key]},${value}`
    } else {
      params[key] = value
    }
  })
  return params
}

export function matchRoute(path: string): RouteMatch {
  const cleanPath = path.split('?')[0].replace(/\/+$/, '') || '/'
  const queryParams = parseQueryParams(path)

  if (cleanPath === '/' || cleanPath === '/dashboard') {
    return { name: 'dashboard', params: queryParams, path: cleanPath }
  }
  if (cleanPath === '/targets') {
    return { name: 'targets', params: queryParams, path: cleanPath }
  }
  if (
    cleanPath === '/templates' ||
    cleanPath === '/definitions' ||
    cleanPath === '/test-definitions'
  ) {
    return { name: 'templates', params: queryParams, path: '/templates' }
  }
  if (cleanPath === '/jobs') {
    return { name: 'jobs', params: queryParams, path: cleanPath }
  }
  const jobMatch = cleanPath.match(/^\/jobs\/(\d+)$/)
  if (jobMatch) {
    return {
      name: 'jobs',
      params: { ...queryParams, id: jobMatch[1] },
      path: cleanPath,
    }
  }
  if (cleanPath === '/trigger') {
    return { name: 'trigger', params: queryParams, path: cleanPath }
  }
  if (cleanPath === '/runs' || cleanPath === '/history') {
    return { name: 'runs', params: queryParams, path: '/runs' }
  }
  if (cleanPath === '/compare' || cleanPath === '/runs/compare') {
    return { name: 'compare', params: queryParams, path: '/compare' }
  }

  const compareIdsMatch = cleanPath.match(/^\/compare\/([\d,]+)$/)
  if (compareIdsMatch) {
    return {
      name: 'compare',
      params: { ...queryParams, run_ids: compareIdsMatch[1] },
      path: '/compare',
    }
  }

  const runMatch = cleanPath.match(/^\/runs\/(\d+)$/)
  if (runMatch) {
    return {
      name: 'run-detail',
      params: { ...queryParams, id: runMatch[1] },
      path: cleanPath,
    }
  }

  return { name: 'dashboard', params: queryParams, path: cleanPath }
}

export const RouterContext = createContext<RouterContextType | null>(null)

export function useRouter(): RouterContextType {
  const context = useContext(RouterContext)
  if (!context) {
    throw new Error('useRouter must be used within a RouterProvider')
  }
  return context
}
