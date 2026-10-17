import { Header } from './components/layout/Header.tsx'
import { Navigation } from './components/layout/Navigation.tsx'
import { DashboardView } from './components/dashboard/DashboardView.tsx'
import { TargetsView } from './components/targets/TargetsView.tsx'
import { TriggerView } from './components/trigger/TriggerView.tsx'
import { RunsListView } from './components/runs/RunsListView.tsx'
import { RunDetailView } from './components/runs/RunDetailView.tsx'
import { RouterProvider, useRouter } from './router/index.ts'
import './App.css'

function AppContent() {
  const { route } = useRouter()

  return (
    <div className="app-shell">
      <Header />
      <Navigation />

      <main className="app-main">
        {route.name === 'dashboard' && <DashboardView />}
        {route.name === 'targets' && <TargetsView />}
        {route.name === 'trigger' && <TriggerView />}
        {route.name === 'runs' && <RunsListView />}
        {route.name === 'run-detail' && (
          <RunDetailView runId={parseInt(route.params.id, 10)} />
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <span>Test Center • Unified Evaluation & Benchmark Runner</span>
          <span className="footer-links">
            <a href="/docs" target="_blank" rel="noreferrer">
              API Docs
            </a>
            {' • '}
            <a href="/api/health" target="_blank" rel="noreferrer">
              Health Status
            </a>
          </span>
        </div>
      </footer>
    </div>
  )
}

function App() {
  return (
    <RouterProvider>
      <AppContent />
    </RouterProvider>
  )
}

export default App
