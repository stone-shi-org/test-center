import { useEffect, useState, type ReactNode } from 'react'
import {
  matchRoute,
  parsePath,
  RouterContext,
  useRouter,
} from './context.ts'

export function RouterProvider({ children }: { children: ReactNode }) {
  const [path, setPath] = useState<string>(() =>
    typeof window !== 'undefined'
      ? parsePath(window.location.pathname, window.location.hash)
      : '/'
  )

  useEffect(() => {
    const handleLocationChange = () => {
      setPath(parsePath(window.location.pathname, window.location.hash))
    }

    window.addEventListener('popstate', handleLocationChange)
    window.addEventListener('hashchange', handleLocationChange)
    return () => {
      window.removeEventListener('popstate', handleLocationChange)
      window.removeEventListener('hashchange', handleLocationChange)
    }
  }, [])

  const navigate = (newPath: string) => {
    if (window.location.hash.startsWith('#/')) {
      window.location.hash = `#${newPath}`
    } else {
      window.history.pushState(null, '', newPath)
    }
    setPath(newPath)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const route = matchRoute(path)

  return (
    <RouterContext.Provider value={{ route, navigate }}>
      {children}
    </RouterContext.Provider>
  )
}

export function Link({
  href,
  className,
  children,
  onClick,
}: {
  href: string
  className?: string
  children: ReactNode
  onClick?: () => void
}) {
  const { navigate } = useRouter()

  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) {
      return
    }
    e.preventDefault()
    if (onClick) onClick()
    navigate(href)
  }

  return (
    <a href={href} className={className} onClick={handleClick}>
      {children}
    </a>
  )
}
