import { useEffect, useRef } from 'react'
import { Link, useRouter, type RouteName } from '../../router/index.ts'

interface NavTab {
  name: RouteName
  path: string
  label: string
  icon: string
}

const TABS: NavTab[] = [
  { name: 'dashboard', path: '/', label: 'Dashboard', icon: '📊' },
  { name: 'targets', path: '/targets', label: 'Targets', icon: '🎯' },
  { name: 'templates', path: '/templates', label: 'Templates', icon: '📋' },
  { name: 'jobs', path: '/jobs', label: 'Jobs', icon: '📦' },
  { name: 'trigger', path: '/trigger', label: 'Run Trigger', icon: '🚀' },
  { name: 'runs', path: '/runs', label: 'Runs History', icon: '⏱️' },
  { name: 'compare', path: '/compare', label: 'Compare Runs', icon: '⚖️' },
]

export function Navigation() {
  const { route } = useRouter()
  const containerRef = useRef<HTMLDivElement>(null)

  // On narrow screens the tab strip scrolls horizontally; keep the active tab visible.
  useEffect(() => {
    const active = containerRef.current?.querySelector<HTMLElement>('.nav-link.active')
    active?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' })
  }, [route.name])

  const isActive = (tab: NavTab) => {
    if (tab.name === 'runs' && route.name === 'run-detail') {
      return true
    }
    return route.name === tab.name
  }

  return (
    <nav className="main-nav" aria-label="Main Navigation">
      <div className="nav-container" ref={containerRef}>
        {TABS.map((tab) => (
          <Link
            key={tab.name}
            href={tab.path}
            className={`nav-link ${isActive(tab) ? 'active' : ''}`}
          >
            <span className="nav-icon">{tab.icon}</span>
            <span className="nav-label">{tab.label}</span>
          </Link>
        ))}
      </div>
    </nav>
  )
}
