#!/bin/bash

# MolinAntro DAW - Build Script
# Usage: ./scripts/build.sh [options]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Options
BUILD_TESTS=ON
BUILD_DOCS=OFF
ENABLE_ASAN=OFF
VERBOSE=OFF
ACME_EDITION=OFF
ENABLE_GPU=OFF

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        --release)
            BUILD_TYPE=Release
            shift
            ;;
        --no-tests)
            BUILD_TESTS=OFF
            shift
            ;;
        --with-docs)
            BUILD_DOCS=ON
            shift
            ;;
        --asan)
            ENABLE_ASAN=ON
            BUILD_TYPE=Debug
            shift
            ;;
        --verbose)
            VERBOSE=ON
            shift
            ;;
        --target)
            if [ "$2" = "ACME_Edition" ]; then
                ACME_EDITION=ON
                echo -e "${GREEN}★ ACME Edition Enabled${NC}"
            fi
            shift 2
            ;;
        --enable-gpu)
            ENABLE_GPU=ON
            echo -e "${GREEN}★ GPU Acceleration Enabled${NC}"
            shift
            ;;
        --clean)
            echo -e "${YELLOW}Cleaning build directory...${NC}"
            rm -rf "$BUILD_DIR"
            echo -e "${GREEN}✓ Clean complete${NC}"
            exit 0
            ;;
        --help)
            echo "MolinAntro DAW - Build Script"
            echo ""
            echo "Usage: ./scripts/build.sh [options]"
            echo ""
            echo "Options:"
            echo "  --debug         Build in Debug mode"
            echo "  --release       Build in Release mode (default)"
            echo "  --no-tests      Don't build tests"
            echo "  --with-docs     Build documentation"
            echo "  --asan          Enable AddressSanitizer (Debug mode)"
            echo "  --verbose       Verbose build output"
            echo "  --clean         Clean build directory"
            echo "  --target [trg]  Set build target (e.g., ACME_Edition)"
            echo "  --enable-gpu    Enable GPU acceleration"
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

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MolinAntro DAW - Build System          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""
echo -e "Build Type:       ${YELLOW}$BUILD_TYPE${NC}"
echo -e "Build Directory:  ${YELLOW}$BUILD_DIR${NC}"
echo -e "Install Prefix:   ${YELLOW}$INSTALL_PREFIX${NC}"
echo -e "Parallel Jobs:    ${YELLOW}$NUM_JOBS${NC}"
echo -e "Build Tests:      ${YELLOW}$BUILD_TESTS${NC}"
echo -e "Build Docs:       ${YELLOW}$BUILD_DOCS${NC}"
echo -e "AddressSanitizer: ${YELLOW}$ENABLE_ASAN${NC}"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 not found"
        return 1
    fi
}

DEPS_OK=true
check_command cmake || DEPS_OK=false
check_command git || DEPS_OK=false

if [ "$DEPS_OK" = false ]; then
    echo -e "${RED}Missing required dependencies. Please install them first.${NC}"
    exit 1
fi

# Create build directory
echo -e "\n${YELLOW}Creating build directory...${NC}"
mkdir -p "$BUILD_DIR"

# Configure
echo -e "\n${YELLOW}Configuring CMake...${NC}"
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DENABLE_ASAN="$ENABLE_ASAN" \
    -DACME_EDITION="$ACME_EDITION" \
    -DENABLE_AI_MODULES="$ACME_EDITION" \
    -DENABLE_GPU="$ENABLE_GPU" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Configuration failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration complete${NC}"

# Build
echo -e "\n${YELLOW}Building...${NC}"

if [ "$VERBOSE" = "ON" ]; then
    cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$NUM_JOBS" -- VERBOSE=1
else
    cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$NUM_JOBS"
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build complete${NC}"

# Run tests if enabled
if [ "$BUILD_TESTS" = "ON" ]; then
    echo -e "\n${YELLOW}Running tests...${NC}"
    cd "$BUILD_DIR"
    ctest -C "$BUILD_TYPE" --output-on-failure
    TEST_RESULT=$?
    cd ..

    if [ $TEST_RESULT -ne 0 ]; then
        echo -e "${RED}✗ Tests failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ All tests passed${NC}"
fi

# Summary
echo -e "\n${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Build Complete!                        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""
echo -e "Executable: ${YELLOW}$BUILD_DIR/bin/MolinAntroDaw${NC}"
echo ""
echo -e "To install system-wide, run:"
echo -e "  ${YELLOW}sudo cmake --install $BUILD_DIR${NC}"
echo ""
echo -e "To run the application:"
echo -e "  ${YELLOW}./$BUILD_DIR/bin/MolinAntroDaw${NC}"
echo ""
echo -e "To run tests:"
echo -e "  ${YELLOW}cd $BUILD_DIR && ctest${NC}"
echo ""
