#!/bin/bash

# MolinAntro DAW - Test Runner Script
# Usage: ./scripts/run_tests.sh [options]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BUILD_DIR="${BUILD_DIR:-build}"
TEST_FILTER=""
TEST_LABELS=""
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_LABELS="-L unit"
            shift
            ;;
        --integration)
            TEST_LABELS="-L integration"
            shift
            ;;
        --e2e)
            TEST_LABELS="-L e2e"
            shift
            ;;
        --filter)
            TEST_FILTER="-R $2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "MolinAntro DAW - Test Runner"
            echo ""
            echo "Usage: ./scripts/run_tests.sh [options]"
            echo ""
            echo "Options:"
            echo "  --unit          Run only unit tests"
            echo "  --integration   Run only integration tests"
            echo "  --e2e           Run only E2E tests"
            echo "  --filter <pat>  Run tests matching pattern"
            echo "  --verbose       Verbose output"
            echo "  --help          Show this help"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Build directory not found: $BUILD_DIR${NC}"
    echo -e "${YELLOW}Run ./scripts/build.sh first${NC}"
    exit 1
fi

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MolinAntro DAW - Test Suite           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

cd "$BUILD_DIR"

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
echo ""

if [ "$VERBOSE" = true ]; then
    ctest --output-on-failure --verbose $TEST_LABELS $TEST_FILTER
else
    ctest --output-on-failure $TEST_LABELS $TEST_FILTER
fi

TEST_RESULT=$?

echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   ✓ All Tests Passed!                    ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${RED}║   ✗ Tests Failed                         ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════════╝${NC}"
fi

exit $TEST_RESULT
