#!/bin/bash
# GPU CFR Installation Script
#
# Installs JAX with CUDA support and cfrx library for GPU-accelerated CFR
#
# Usage:
#   source ~/open_spiel/venv/bin/activate
#   bash install_gpu_cfr.sh

set -e  # Exit on error

echo "================================================================================"
echo "GPU-Accelerated CFR Installation"
echo "================================================================================"
echo ""

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "ERROR: No virtual environment detected!"
    echo "Please activate OpenSpiel environment first:"
    echo "  source ~/open_spiel/venv/bin/activate"
    echo ""
    exit 1
fi

echo "✓ Virtual environment: $VIRTUAL_ENV"
echo ""

# Check NVIDIA driver
echo "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found!"
    echo "Please install NVIDIA drivers first."
    exit 1
fi

echo "✓ NVIDIA driver detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Detect CUDA version to use
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)

if [[ $DRIVER_MAJOR -ge 580 ]]; then
    CUDA_VERSION="cuda13"
    echo "✓ Driver $DRIVER_VERSION supports CUDA 13"
elif [[ $DRIVER_MAJOR -ge 525 ]]; then
    CUDA_VERSION="cuda12"
    echo "✓ Driver $DRIVER_VERSION supports CUDA 12"
else
    echo "ERROR: Driver $DRIVER_VERSION is too old!"
    echo "Need driver >= 525 for CUDA 12 or >= 580 for CUDA 13"
    echo "Update with: sudo ubuntu-drivers autoinstall"
    exit 1
fi
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install JAX with CUDA support
echo "Installing JAX with $CUDA_VERSION support..."
pip install --upgrade "jax[${CUDA_VERSION}]"
echo ""

# Install cfrx
echo "Installing cfrx..."
pip install cfrx
echo ""

# Verify installation
echo "Verifying GPU detection..."
python -c "import jax; print('JAX devices:', jax.devices()); print('Backend:', jax.default_backend())"
echo ""

# Test GPU
echo "Testing GPU computation..."
python -c "import jax.numpy as jnp; x = jnp.ones((1000, 1000)); y = x @ x; print('✓ GPU matrix multiplication successful')"
echo ""

# Check cfrx
echo "Verifying cfrx installation..."
python -c "from cfrx.envs import KuhnPoker; from cfrx.policy import TabularPolicy; from cfrx.training import MCCFRTrainer; print('✓ cfrx imported successfully')"
echo ""

echo "================================================================================"
echo "Installation Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Run quick test:       python gpu_cfr_solver.py"
echo "  2. Run benchmark:        python benchmark_gpu_cpu.py"
echo "  3. Full comparison:      python compare_cfr_gpu_cpu.py --iterations 100000"
echo ""
echo "See GPU_CFR_README.md for detailed usage instructions."
echo "================================================================================"
