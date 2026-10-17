#!/usr/bin/env bash
#
# Generate Python protobuf & gRPC stubs in backend/app/testers/live_stt/pb/
# from the vendored proto/livestt/v1/asr.proto schema.
#
# Invoked during Docker builds (Dockerfile), package builds (backend/setup.py),
# or manually in local development.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PB_DIR="backend/app/testers/live_stt/pb"

PYTHON="python3"
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
fi

rm -rf "$PB_DIR/livestt"
mkdir -p "$PB_DIR"

"$PYTHON" -m grpc_tools.protoc \
    -I proto \
    --python_out="$PB_DIR" \
    --grpc_python_out="$PB_DIR" \
    --pyi_out="$PB_DIR" \
    proto/livestt/v1/asr.proto

# Ensure package __init__.py markers exist for mirrored directories
find "$PB_DIR/livestt" -type d -exec sh -c 'test -f "$1/__init__.py" || printf "\"\"\"Generated package namespace.\"\"\"\n" > "$1/__init__.py"' _ {} \;

echo "==> Generated Python gRPC stubs in $PB_DIR/livestt from proto/livestt/v1/asr.proto"
