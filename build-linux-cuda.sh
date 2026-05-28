#!/usr/bin/env bash
# Build Totem on Linux with MLX CUDA backend (GPU) or CPU fallback.
#
# Requirements:
#   - Swift 6.3+
#   - CUDA toolkit 12.x installed at /usr/local/cuda
#   - Run setup-cuda-ubuntu.sh first to install all dependencies
#
# Usage:
#   ./build-linux-cuda.sh          # CUDA GPU build (default)
#   ./build-linux-cuda.sh --cpu    # CPU-only build
#   ./build-linux-cuda.sh --debug  # CUDA debug build

set -euo pipefail

BUILD_CONFIG="release"
CUDA_ENABLED=1

for arg in "$@"; do
    case $arg in
        --cpu)   CUDA_ENABLED=0 ;;
        --debug) BUILD_CONFIG="debug" ;;
    esac
done

if [[ $CUDA_ENABLED -eq 1 ]]; then
    if [[ ! -d /usr/local/cuda ]]; then
        echo "ERROR: CUDA toolkit not found at /usr/local/cuda"
        echo "Install with: sudo apt-get install cuda-toolkit-12-9 libcudnn9-dev-cuda-12"
        exit 1
    fi
    export PATH=/usr/local/cuda/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
    export CUDA_ARCH=${CUDA_ARCH:-sm_86}
    echo "Building with CUDA GPU backend (arch=${CUDA_ARCH})..."
    export SPM_CUDA=1
else
    echo "Building with CPU backend..."
    export SPM_CUDA=0
fi

swift build -c "$BUILD_CONFIG" --jobs 2

echo ""
echo "Done. Run with:"
if [[ $BUILD_CONFIG == "release" ]]; then
    echo "  .build/release/totem --use-mlx --mlx-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
else
    echo "  .build/debug/totem --use-mlx --mlx-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
fi
