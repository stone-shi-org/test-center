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
  { name: 'trigger', path: '/trigger', label: 'Run Trigger', icon: '🚀' },
  { name: 'runs', path: '/runs', label: 'Runs History', icon: '⏱️' },
]

export function Navigation() {
  const { route } = useRouter()

  const isActive = (tab: NavTab) => {
    if (tab.name === 'runs' && route.name === 'run-detail') {
      return true
    }
    return route.name === tab.name
  }

  return (
    <nav className="main-nav" aria-label="Main Navigation">
      <div className="nav-container">
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
