# Stage 1: Build frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci --include=dev

COPY frontend/ ./
RUN npm run build

# Stage: backend unit tests (offline; no real services). Writes JUnit XML to
# /app/test-reports -- mount a host dir there. Built with `--target backend-test`.
FROM python:3.12-slim AS backend-test
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRPC_POLL_STRATEGY=poll
COPY backend/requirements-dev.txt backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r ./backend/requirements-dev.txt
COPY proto/ ./proto
COPY scripts/ ./scripts
COPY backend/ ./backend
COPY docs/ ./docs
RUN bash ./scripts/gen_proto.sh
CMD ["sh", "-c", "mkdir -p /app/test-reports && python -m pytest backend/tests -v --junitxml=/app/test-reports/backend-results.xml --junit-prefix=backend"]

# Stage: frontend unit tests. Writes JUnit XML to /app/test-reports.
# Built with `--target frontend-test`.
FROM node:22-alpine AS frontend-test
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --include=dev
COPY frontend/ ./
CMD ["sh", "-c", "mkdir -p /app/test-reports && NODE_ENV=development npx vitest run --reporter=default --reporter=junit --outputFile.junit=/app/test-reports/frontend-results.xml"]

# Stage 2: Runtime backend
FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRPC_POLL_STRATEGY=poll

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

COPY proto/ ./proto
COPY scripts/ ./scripts
COPY backend/ ./backend
RUN bash ./scripts/gen_proto.sh

COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV HOST=0.0.0.0
ENV PORT=8000
ENV DATA_DIR=/app/data
ENV TEST_DATA_DIR=/app/test-data
ENV STATIC_DIR=/app/frontend/dist

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
