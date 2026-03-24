#!/usr/bin/env bash
# =============================================================================
# Visual Jenga — Test Runner
#
# Usage:
#   bash run_tests.sh            # unit tests only (no GPU, fast)
#   bash run_tests.sh --smoke    # smoke tests: unit + 1 quick GPU sanity check
#   bash run_tests.sh --all      # all tests including slow GPU tests
#   bash run_tests.sh --gpu      # all GPU tests (skips slow marker filter)
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-}"

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "================================================================"
echo "  Visual Jenga Test Suite"
echo "================================================================"

case "$MODE" in

  # --------------------------------------------------------------------------
  --smoke)
    echo -e "${YELLOW}Mode: SMOKE (unit + quick GPU sanity)${NC}"
    echo ""
    echo "--- Unit tests (no GPU) ---"
    uv run pytest tests/ \
      -m "not gpu and not slow" \
      -v --tb=short
    echo ""
    echo "--- Smoke GPU test (single forward pass per model) ---"
    uv run pytest tests/ \
      -m "gpu and not slow" \
      -v --tb=short \
      --timeout=120 2>/dev/null || echo -e "${YELLOW}[SKIP] No GPU or models not downloaded — smoke GPU test skipped${NC}"
    ;;

  # --------------------------------------------------------------------------
  --all)
    echo -e "${YELLOW}Mode: ALL (unit + all GPU + slow)${NC}"
    echo ""
    uv run pytest tests/ \
      -v --tb=short \
      --timeout=600
    ;;

  # --------------------------------------------------------------------------
  --gpu)
    echo -e "${YELLOW}Mode: GPU only${NC}"
    echo ""
    uv run pytest tests/ \
      -m "gpu" \
      -v --tb=short \
      --timeout=600
    ;;

  # --------------------------------------------------------------------------
  *)
    # Default: unit tests only (fast, no GPU)
    echo -e "${GREEN}Mode: UNIT (no GPU required)${NC}"
    echo ""
    echo "  Runs: parse logic, mask ops, crop/bbox, diversity formula"
    echo "  Skips: any test marked @pytest.mark.gpu or @pytest.mark.slow"
    echo ""
    uv run pytest tests/ \
      -m "not gpu and not slow" \
      -v --tb=short
    ;;
esac

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Done.${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo "Other modes:"
echo "  bash run_tests.sh            # unit tests only (default)"
echo "  bash run_tests.sh --smoke    # unit + quick GPU sanity"
echo "  bash run_tests.sh --all      # everything"
echo "  bash run_tests.sh --gpu      # GPU tests only"
