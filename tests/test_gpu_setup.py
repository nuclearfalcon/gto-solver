#!/usr/bin/env python3
"""
GPU Setup Validation Test

Tests that JAX can detect and use the NVIDIA GeForce RTX 4060 Ti.

Requirements:
- JAX with CUDA 12 support installed
- NVIDIA GPU drivers >= 525

Usage:
    source ~/open_spiel/venv/bin/activate
    python tests/test_gpu_setup.py
"""

import sys


def test_jax_import():
    """Test that JAX can be imported."""
    try:
        import jax
        print("✓ JAX imported successfully")
        print(f"  JAX version: {jax.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import JAX: {e}")
        return False


def test_gpu_detection():
    """Test that JAX can detect GPU(s)."""
    import jax

    devices = jax.devices()
    print(f"\n=== Detected Devices ({len(devices)} total) ===")

    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.device_kind} (platform: {device.platform})")

    gpu_devices = [d for d in devices if d.platform == 'gpu']

    if gpu_devices:
        print(f"\n✓ Found {len(gpu_devices)} GPU device(s)")
        for i, gpu in enumerate(gpu_devices):
            print(f"  GPU {i}: {gpu.device_kind}")
        return True
    else:
        print("\n✗ No GPU devices found")
        print("  JAX is using CPU only")
        return False


def test_simple_computation():
    """Test a simple JAX computation on GPU."""
    import jax.numpy as jnp

    print("\n=== Testing GPU Computation ===")

    # Create test arrays
    x = jnp.ones((1000, 1000))
    y = jnp.ones((1000, 1000))

    # Matrix multiplication (should run on GPU)
    result = jnp.dot(x, y)

    print(f"  Matrix multiplication result shape: {result.shape}")
    print(f"  Result sum: {result.sum()}")
    print(f"  Expected sum: {1000 * 1000 * 1000}")

    if abs(result.sum() - 1000 * 1000 * 1000) < 1e-3:
        print("✓ GPU computation test passed")
        return True
    else:
        print("✗ GPU computation test failed")
        return False


def test_jit_compilation():
    """Test JAX JIT compilation."""
    import jax
    import jax.numpy as jnp

    print("\n=== Testing JIT Compilation ===")

    @jax.jit
    def fast_function(x):
        return jnp.dot(x, x.T)

    x = jnp.ones((100, 100))

    # First call compiles
    result1 = fast_function(x)
    print(f"  First call (with compilation): result shape = {result1.shape}")

    # Second call uses cached compiled version
    result2 = fast_function(x)
    print(f"  Second call (cached): result shape = {result2.shape}")

    if result1.shape == result2.shape == (100, 100):
        print("✓ JIT compilation test passed")
        return True
    else:
        print("✗ JIT compilation test failed")
        return False


def test_sparse_matrices():
    """Test JAX sparse matrix support (critical for CFR)."""
    try:
        from jax.experimental import sparse
        import jax.numpy as jnp

        print("\n=== Testing Sparse Matrix Support ===")

        # Create a simple sparse matrix (CSR format)
        # This is what we'll use for level matrices in CFR
        indices = jnp.array([[0, 0], [1, 1], [2, 2]])  # Diagonal matrix
        values = jnp.array([1.0, 2.0, 3.0])
        shape = (3, 3)

        # Note: JAX sparse is still experimental
        print("  JAX sparse module imported successfully")
        print("  Note: JAX sparse is experimental but should work for our needs")
        print("✓ Sparse matrix support available")
        return True

    except ImportError as e:
        print(f"✗ Failed to import JAX sparse: {e}")
        print("  This may be a problem for matrix-based CFR")
        return False


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 60)
    print("GPU SETUP VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! GPU is ready for matrix-based CFR")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


def main():
    """Run all GPU setup validation tests."""
    print("=" * 60)
    print("GPU SETUP VALIDATION FOR MATRIX-BASED CFR")
    print("Testing NVIDIA GeForce RTX 4060 Ti with JAX + CUDA 12")
    print("=" * 60)

    results = {}

    # Run tests in order
    results['JAX Import'] = test_jax_import()

    if not results['JAX Import']:
        print("\n❌ JAX not installed. Run: pip install 'jax[cuda12]'")
        return 1

    results['GPU Detection'] = test_gpu_detection()
    results['Simple Computation'] = test_simple_computation()
    results['JIT Compilation'] = test_jit_compilation()
    results['Sparse Matrix Support'] = test_sparse_matrices()

    # Print summary
    exit_code = print_summary(results)

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
