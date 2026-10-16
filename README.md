# Test Center

Internal web platform for configuring and running test suites (live-stt, embedding, LLM) as jobs, storing results in a persistent database, and analyzing or comparing test runs side by side.

See [docs/PLAN.md](file:///data/homes/stoneshi/src/test-center/.worktrees/feature-infra/docs/PLAN.md) for architecture and product roadmap.

---

## Quickstart with Docker Compose

Ensure Docker is installed and running.

1. **(Optional) Configure environment:**
   ```bash
   cp .env.example .env
   ```

2. **Ensure test-data directory exists:**
   ```bash
   mkdir -p test-data
   ```

3. **Start the application container:**
   ```bash
   docker compose up --build
   ```

4. **Verify acceptance:**
   - **Frontend UI:** Open [http://localhost:8000](http://localhost:8000) in your browser.
   - **Health Check API:** `curl http://localhost:8000/api/health`
     ```json
     {
       "status": "ok",
       "app": "Test Center",
       "version": "0.1.0",
       "data_dir": "/app/data",
       "data_dir_exists": true,
       "test_data_dir": "/app/test-data",
       "test_data_dir_exists": true
     }
     ```

---

## Local Development Setup

For faster development and debugging, the backend and frontend can be run independently:

### 1. Backend (FastAPI + pytest)

Requires Python 3.11+:

```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt

# Run unit tests
pytest backend

# Start backend dev server with auto-reload
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 2. Frontend (React + Vite + TypeScript)

Requires Node.js 20+:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server starts on [http://localhost:5173](http://localhost:5173) and automatically proxies `/api` calls to `http://localhost:8000`.

To build the static frontend bundle:
```bash
cd frontend
npm run build
```

---

## Project Structure

```text
├── Dockerfile              # Multi-stage build (Node frontend build -> Python runtime)
├── docker-compose.yml      # Service config with data volume and read-only test-data mount
├── .env.example            # Environment template
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI application entry & SPA router
│   │   ├── config.py       # Pydantic configuration & env loading
│   │   └── api/
│   │       └── health.py   # Health check API (/api/health)
│   ├── tests/              # Backend test suite (pytest)
│   ├── requirements.txt    # Runtime dependencies
│   └── requirements-dev.txt# Test and dev dependencies
├── frontend/               # React + TypeScript + Vite single-page app
│   ├── src/
│   │   ├── App.tsx         # Dashboard and backend health status monitor
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts      # API proxy configuration
├── data/                   # Persistent SQLite db and run artifacts (volume target)
├── test-data/              # Test datasets e.g. NOTSOFAR-1 (read-only mount)
└── docs/
    └── PLAN.md             # Architecture and implementation plan
```
