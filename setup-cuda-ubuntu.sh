#!/usr/bin/env bash
# Install CUDA toolkit and dependencies required to build Totem with MLX on Ubuntu 24.04.
# Run once on a fresh machine before using build-linux-cuda.sh.

set -euo pipefail

CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
KEYRING_DEB="cuda-keyring_1.1-1_all.deb"

echo "==> Adding NVIDIA CUDA repository..."
wget -q "$CUDA_KEYRING_URL" -O "$KEYRING_DEB"
sudo dpkg -i "$KEYRING_DEB"
rm -f "$KEYRING_DEB"
sudo apt-get update -q

echo "==> Installing CUDA toolkit..."
# libcudnn9-dev-cuda-12 is intentionally omitted — cuda-toolkit-12-9 pulls in
# nvidia-cudnn (8.9.x) which owns the same files and the two conflict.
# cuDNN headers come from the toolkit; the graph API is supplied by the
# cudnn-frontend clone below.
sudo apt-get install -y cuda-toolkit-12-9

echo "==> Installing BLAS/LAPACK..."
sudo apt-get install -y libopenblas-dev liblapack-dev liblapacke-dev gfortran

echo "==> Cloning cudnn-frontend v1.16.0 to /usr/local/cudnn-frontend..."
# mlx requires cudnn-frontend v1.x graph API; Ubuntu's libcudnn-frontend-dev is v0.x.
# Install to /usr/local so the path is consistent across all user accounts.
if [[ ! -d /usr/local/cudnn-frontend ]]; then
    sudo git clone --depth 1 --branch v1.16.0 https://github.com/NVIDIA/cudnn-frontend.git /usr/local/cudnn-frontend
else
    echo "  /usr/local/cudnn-frontend already exists, skipping clone."
fi

PROFILE="$HOME/.bashrc"
CUDA_PATH_LINE='export PATH=/usr/local/cuda/bin:$PATH'
CUDA_LIB_LINE='export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}'

echo "==> Adding CUDA to PATH in $PROFILE..."
grep -qxF "$CUDA_PATH_LINE" "$PROFILE" || echo "$CUDA_PATH_LINE" >> "$PROFILE"
grep -qxF "$CUDA_LIB_LINE" "$PROFILE" || echo "$CUDA_LIB_LINE" >> "$PROFILE"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo ""
echo "==> Verifying installation..."
nvcc --version
ls /usr/local/cuda/lib64/libcudart.so* 2>/dev/null && echo "libcudart: OK" || echo "WARNING: libcudart not found"

echo ""
echo "Setup complete. Run './build-linux-cuda.sh' to build Totem with CUDA."
