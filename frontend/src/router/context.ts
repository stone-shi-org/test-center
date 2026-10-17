import { createContext, useContext } from 'react'

export type RouteName = 'dashboard' | 'targets' | 'trigger' | 'runs' | 'run-detail'

export interface RouteMatch {
  name: RouteName
  params: Record<string, string>
  path: string
}

export interface RouterContextType {
  route: RouteMatch
  navigate: (path: string) => void
}

export function parsePath(pathname: string, hash: string): string {
  if (hash.startsWith('#/')) {
    return hash.slice(1)
  }
  return pathname || '/'
}

export function matchRoute(path: string): RouteMatch {
  const cleanPath = path.split('?')[0].replace(/\/+$/, '') || '/'

  if (cleanPath === '/' || cleanPath === '/dashboard') {
    return { name: 'dashboard', params: {}, path: cleanPath }
  }
  if (cleanPath === '/targets') {
    return { name: 'targets', params: {}, path: cleanPath }
  }
  if (cleanPath === '/trigger' || cleanPath === '/jobs') {
    return { name: 'trigger', params: {}, path: cleanPath }
  }
  if (cleanPath === '/runs') {
    return { name: 'runs', params: {}, path: cleanPath }
  }

  const runMatch = cleanPath.match(/^\/runs\/(\d+)$/)
  if (runMatch) {
    return { name: 'run-detail', params: { id: runMatch[1] }, path: cleanPath }
  }

  return { name: 'dashboard', params: {}, path: cleanPath }
}

export const RouterContext = createContext<RouterContextType | null>(null)

export function useRouter(): RouterContextType {
  const context = useContext(RouterContext)
  if (!context) {
    throw new Error('useRouter must be used within a RouterProvider')
  }
  return context
}
