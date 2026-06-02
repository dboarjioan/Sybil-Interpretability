#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ENV_NAME="sybil_interpretability"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 1. Initializing Git Submodules ==="
cd "${SCRIPT_DIR}"

# Initialize top-level submodules first
git submodule update --init

# Traverse into MedGS to properly patch the nested simple-knn URL
cd submodules/MedGS
git config submodule.submodules/simple-knn.url https://github.com/camenduru/simple-knn.git
git submodule update --init --recursive
cd "${SCRIPT_DIR}"

echo "=== 2. Setting up Conda Environment ==="
# Bootstrap conda shell functions reliably
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda env update -f environment.yml --prune

# Temporarily disable strict unbound variable checking for Conda's hooks
set +u
conda activate "${ENV_NAME}"
set -u

echo "=== 3. Installing PyTorch (CUDA 13.0) ==="
# Install PyTorch wheels corresponding to the CUDA 13.0 toolkit in environment.yml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Torch's shared libraries need to be visible to native MedGS extensions.
TORCH_LIB_DIR="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"

echo "=== 4. Configuring Compiler Environment ==="
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# Restore the hermetic compilers from the original script
echo "Installing hermetic GCC/G++ compilers..."
# Suspend strict unbound variable checking for Conda-forge's compiler hooks
set +u
conda install -y -c conda-forge "gcc_linux-64=11.*" "gxx_linux-64=11.*"
set -u

# Point explicitly to the isolated Conda compilers, NOT the host system
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

# Force distutils to use the Conda headers instead of probing the host's /usr/include
export C_INCLUDE_PATH="${CUDA_HOME}/include:${CONDA_PREFIX}/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${C_INCLUDE_PATH}"

# Build for the detected GPU architecture (RTX 5060 Ti = sm_120)
GPU_ARCH="$(python -c 'import torch; cc = torch.cuda.get_device_capability(0); print(f"{cc[0]}.{cc[1]}+PTX")')"
export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"

echo "=== 5. Building C++/CUDA Extensions ==="

# Install ninja for significantly faster, parallelized C++ builds
pip install ninja

# Patch the rasterizer to use C++17 instead of C++20 to bypass glibc rsqrt conflicts
echo "Patching diff-gaussian-rasterization for glibc compatibility..."
sed -i 's/-std=c++20/-std=c++17/g' "${SCRIPT_DIR}/submodules/MedGS/submodules/diff-gaussian-rasterization/setup.py" || true

# --no-build-isolation ensures pip uses the active environment's PyTorch/CUDA headers
echo "Building diff-gaussian-rasterization..."
pip install --no-build-isolation -e "${SCRIPT_DIR}/submodules/MedGS/submodules/diff-gaussian-rasterization"

echo "Building fused-ssim (with GPU hidden)..."
CUDA_VISIBLE_DEVICES="" pip install --no-build-isolation -e "${SCRIPT_DIR}/submodules/MedGS/submodules/fused-ssim"

echo "Building simple-knn..."
pip install --no-build-isolation "${SCRIPT_DIR}/submodules/MedGS/submodules/simple-knn"

echo "=== Setup Complete! ==="