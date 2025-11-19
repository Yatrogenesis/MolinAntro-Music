#!/bin/bash
# MolinAntro DAW - Packaging Script
# Generates installers for multiple platforms

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build-release"
PACKAGE_DIR="${PROJECT_ROOT}/packages"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     MolinAntro DAW - Package Generator v2.0              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Parse command line arguments
BUILD_TYPE="Release"
CLEAN_BUILD=false
RUN_TESTS=true
GENERATE_SOURCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --no-tests)
            RUN_TESTS=false
            shift
            ;;
        --source)
            GENERATE_SOURCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug] [--clean] [--no-tests] [--source]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Build Type: $BUILD_TYPE"
echo "  Clean Build: $CLEAN_BUILD"
echo "  Run Tests: $RUN_TESTS"
echo "  Generate Source: $GENERATE_SOURCE"
echo ""

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "🧹 Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
mkdir -p "$PACKAGE_DIR"

# Configure
echo "⚙️  Configuring project..."
cd "$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX=/usr \
      "$PROJECT_ROOT"

# Build
echo ""
echo "🔨 Building project..."
cmake --build . -j $(nproc) --config $BUILD_TYPE

# Run tests
if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo "🧪 Running tests..."
    ctest --output-on-failure -j $(nproc)

    # Check test results
    if [ $? -ne 0 ]; then
        echo "❌ Tests failed! Aborting package generation."
        exit 1
    fi
    echo "✅ All tests passed!"
fi

# Generate packages
echo ""
echo "📦 Generating packages..."

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "   Platform: Linux"

    # Check which package generators are available
    GENERATORS="DEB;TGZ"
    if command -v rpmbuild &> /dev/null; then
        echo "   Generating DEB, RPM, and TGZ packages..."
        GENERATORS="DEB;RPM;TGZ"
    else
        echo "   Generating DEB and TGZ packages (rpmbuild not found)..."
    fi

    cpack -G "$GENERATORS"

    # Move packages
    mv *.deb "$PACKAGE_DIR/" 2>/dev/null || true
    mv *.rpm "$PACKAGE_DIR/" 2>/dev/null || true
    mv *.tar.gz "$PACKAGE_DIR/" 2>/dev/null || true

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "   Platform: macOS"
    echo "   Generating DMG package..."

    cpack -G "DragNDrop;TGZ"

    # Move packages
    mv *.dmg "$PACKAGE_DIR/" 2>/dev/null || true
    mv *.tar.gz "$PACKAGE_DIR/" 2>/dev/null || true

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "   Platform: Windows"
    echo "   Generating NSIS installer..."

    cpack -G "NSIS;ZIP"

    # Move packages
    mv *.exe "$PACKAGE_DIR/" 2>/dev/null || true
    mv *.zip "$PACKAGE_DIR/" 2>/dev/null || true
fi

# Generate source package
if [ "$GENERATE_SOURCE" = true ]; then
    echo ""
    echo "📦 Generating source package..."
    cpack -G "TGZ;ZIP" --config CPackSourceConfig.cmake
    mv *-Source.* "$PACKAGE_DIR/" 2>/dev/null || true
fi

# List generated packages
echo ""
echo "✅ Package generation complete!"
echo ""
echo "Generated packages:"
ls -lh "$PACKAGE_DIR"/*

# Calculate checksums
echo ""
echo "🔐 Generating checksums..."
cd "$PACKAGE_DIR"
sha256sum * > SHA256SUMS 2>/dev/null || shasum -a 256 * > SHA256SUMS

echo ""
echo "Checksums saved to: $PACKAGE_DIR/SHA256SUMS"
cat SHA256SUMS

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     Packaging Complete! 🎉                               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Packages available in: $PACKAGE_DIR"
