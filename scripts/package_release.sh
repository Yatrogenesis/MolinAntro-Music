#!/bin/bash
set -e

# MolinAntro DAW - Release Packaging Script
# Creates a distribution-ready package (DMG/Installer/Deb/Rpm)

VERSION=$(grep "set(MOLINANTRO_VERSION" cmake/Version.cmake | awk '{print $2}' | tr -d ')')
BUILD_DIR="build_release"

echo "========================================"
echo "Packaging MolinAntro DAW v${VERSION}"
echo "========================================"

# 1. Clean and Configure
echo "[1/4] Configuring Release Build (ACME Edition)..."
rm -rf ${BUILD_DIR}
cmake -B ${BUILD_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    -DACME_EDITION=ON \
    -DENABLE_AI_MODULES=ON \
    -DENABLE_GPU_ACCELERATION=OFF 

# 2. Build
echo "[2/4] Building Project..."
cmake --build ${BUILD_DIR} --config Release -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)

# 3. Test (Quick verification)
echo "[3/4] Verifying Build..."
if [ -f "${BUILD_DIR}/bin/MolinAntroDaw" ]; then
    echo "Binary created successfully: ${BUILD_DIR}/bin/MolinAntroDaw"
else
    echo "Error: Binary not found!"
    exit 1
fi

# 4. Package
echo "[4/4] Creating Package..."
cd ${BUILD_DIR}
cpack -C Release

echo "========================================"
echo "Packaging Complete!"
echo "Find artifacts in: ${BUILD_DIR}"
ls -lh *.dmg *.zip *.tar.gz 2>/dev/null || true
echo "========================================"
