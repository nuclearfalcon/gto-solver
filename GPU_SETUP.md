# GPU-Accelerated CFR Setup Guide

This guide walks through setting up GPU-accelerated CFR training using JAX and the cfrx library on your RTX 4060 Ti.

## Hardware Requirements

- ✅ **GPU**: NVIDIA GeForce RTX 4060 Ti (Ada Lovelace architecture)
- ✅ **PCIe**: Gen 1 @ 8x (sufficient for CFR workloads)
- **Driver**: NVIDIA driver version >= 525 (for CUDA 12) or >= 580 (for CUDA 13)

## Installation Steps

### 1. Check NVIDIA Driver Version

```bash
nvidia-smi
```

Ensure driver version is >= 525. If not, update using:
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 2. Activate OpenSpiel Environment

```bash
source ~/open_spiel/venv/bin/activate
```

### 3. Install JAX with CUDA Support

**For CUDA 12 (recommended for RTX 4060 Ti):**
```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda12]"
```

**For CUDA 13 (if driver >= 580):**
```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda13]"
```

### 4. Install cfrx Library

```bash
pip install cfrx
```

### 5. Verify GPU Detection

```bash
python -c "import jax; print(jax.devices())"
```

Expected output:
```
[cuda(id=0)]
```

### 6. Run Quick GPU Test

```bash
python -c "import jax.numpy as jnp; x = jnp.ones((1000, 1000)); print(x @ x)"
```

This should run a matrix multiplication on the GPU.

## Performance Expectations

Based on the research and cfrx benchmarks:

| Game | CPU (OpenSpiel) | GPU (cfrx/JAX) | Speedup |
|------|----------------|----------------|---------|
| 3-player Kuhn | ~1000 it/s | ~10,000-50,000 it/s | 10-50x |
| Leduc Poker | ~100 it/s | ~5,000-10,000 it/s | 50-100x |
| Large games | Baseline | Scales better | 100-400x |

Your current `compare_dcfr_research_3p.py` runs at ~2000 it/s on CPU. With GPU acceleration, expect:
- **10-20x speedup**: 20,000-40,000 it/s
- **1M iterations**: ~25 seconds (vs ~8 minutes on CPU)
- **100k iterations**: ~2.5 seconds (vs ~50 seconds on CPU)

## Troubleshooting

### JAX Not Detecting GPU

1. Check LD_LIBRARY_PATH is not set:
   ```bash
   echo $LD_LIBRARY_PATH
   ```
   If set, unset it temporarily:
   ```bash
   unset LD_LIBRARY_PATH
   ```

2. Verify CUDA driver:
   ```bash
   nvidia-smi
   ```

3. Check JAX CUDA installation:
   ```bash
   python -c "import jax; print(jax.default_backend())"
   ```
   Should print `gpu` or `cuda`.

### Memory Issues

RTX 4060 Ti has either 8GB or 16GB VRAM. If running out of memory:

1. Reduce batch size in cfrx training
2. Use gradient checkpointing (if using Deep CFR)
3. Monitor GPU memory:
   ```bash
   watch -n 1 nvidia-smi
   ```

### Performance Not Improving

1. Ensure game is large enough (small games may be slower on GPU)
2. Check if computation is actually running on GPU:
   ```bash
   nvidia-smi dmon
   ```
   GPU utilization should be > 0% during training.

3. For very small games (< 1000 nodes), CPU may be faster due to GPU overhead.

## Next Steps

Once installation is complete:

1. **Test cfrx basics**: Run the example in `cfrx` documentation for Kuhn poker
2. **Create GPU wrapper**: Build `gpu_cfr_solver.py` to match your existing `UnifiedPokerSolver` interface
3. **Port research script**: Create `compare_dcfr_research_3p_gpu.py`
4. **Benchmark**: Compare GPU vs CPU performance with `benchmark_gpu_cpu.py`

## References

- JAX Installation: https://docs.jax.dev/en/latest/installation.html
- cfrx GitHub: https://github.com/Egiob/cfrx
- GPU-Accelerated CFR Paper: https://arxiv.org/abs/2408.14778
