#!/usr/bin/env bash
#
# Build (and optionally push) the test-center container image.
#
#   ./build.sh                          build test-center:latest locally
#   ./build.sh -p -r registry.example/x build and push (registry required)
#   TC_REGISTRY=registry.example/x ./build.sh -p   same, via environment
#
# There is deliberately no hardcoded registry: this repo is public.
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

REGISTRY="${TC_REGISTRY:-}"
PUSH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--push)     PUSH=1; shift ;;
        -r|--registry) REGISTRY="$2"; shift 2 ;;
        -h|--help)     sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ "$PUSH" -eq 1 ] && [ -z "$REGISTRY" ]; then
    echo "ERROR: --push needs a registry (-r or TC_REGISTRY)." >&2
    exit 1
fi

REPO="${REGISTRY:+${REGISTRY}/}test-center"
HASH="$(git rev-parse --short=7 HEAD 2>/dev/null || echo dev)"
[ -n "$(git status --porcelain 2>/dev/null)" ] && HASH="${HASH}-dev"

echo "==> Building ${REPO}:latest and :${HASH}"
docker build --target runtime -t "${REPO}:latest" -t "${REPO}:${HASH}" .

if [ "$PUSH" -eq 1 ]; then
    echo "==> Pushing"
    docker push "${REPO}:latest"
    docker push "${REPO}:${HASH}"
fi
