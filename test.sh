#!/usr/bin/env bash
#
# Test runner. Emits JUnit XML into test-reports/ for the Atlassian Bamboo
# JUnit Parser task (result pattern: **/test-reports/*.xml).
#
#   ./test.sh --docker     backend + frontend unit tests in containers (CI path)
#   ./test.sh --backend    backend only (docker)
#   ./test.sh --frontend   frontend only (docker)
#
# Deliberately NOT `set -e`: a failing run must still leave its XML on disk,
# or Bamboo reports "no test results found" instead of naming the failure.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
REPORT_DIR="$SCRIPT_DIR/test-reports"

RUN_BACKEND=1
RUN_FRONTEND=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker)   shift ;;  # accepted for symmetry; Docker is the only path
        --backend)  RUN_FRONTEND=0; shift ;;
        --frontend) RUN_BACKEND=0; shift ;;
        -h|--help)  sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Bamboo fails a build when result files predate the build start; always start clean.
rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"

RC=0
# Reports are copied out with `docker cp` rather than a bind mount, so this
# also works when DOCKER_HOST points at a remote daemon (host paths would not
# exist there).
run_suite() {
    local name="$1" rc=0
    local image="test-center-$name-test" ctr="test-center-$name-test-$$"
    echo ""
    echo "================================================================"
    echo "  $name tests (in Docker)"
    echo "================================================================"
    docker build --target "$name-test" -t "$image" . || return 1
    docker run --name "$ctr" "$image" || rc=$?
    docker cp "$ctr:/app/test-reports/." "$REPORT_DIR/" || echo "WARNING: no reports copied for $name" >&2
    docker rm -f "$ctr" >/dev/null 2>&1 || true
    return "$rc"
}

if [ "$RUN_BACKEND" -eq 1 ]; then run_suite backend || RC=1; fi
if [ "$RUN_FRONTEND" -eq 1 ]; then run_suite frontend || RC=1; fi

echo ""
echo "  tests exit=$RC"
echo "  Reports:"; ls -1 "$REPORT_DIR" | sed 's/^/    /'
exit "$RC"
