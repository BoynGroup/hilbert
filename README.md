# Hilbert

Hilbert is a collection of quantum-chemistry plugins for
[Psi4](https://psicode.org/), developed by the DePrince group. It includes DOCI, pair-density-matrix methods,
v2RDM-DOCI, v2RDM-CASSCF, polaritonic methods, and GPU-accelerated BPSDP for v2RDM added by us here.

## Installation

1. Install Psi4 (ideally into a new conda environment). A source or nightly build is recommended when developing
   Hilbert. The configure line produced by `psi4 --plugin-compile` may need an
   explicit `CMAKE_INSTALL_PREFIX`.
2. Clone Hilbert:

   ```bash
   git clone git@github.com:edeprince3/hilbert.git
   cd hilbert
   ```

3. Configure, build, and install:

   ```bash
   conda activate your-conda-environment
   psi4 --plugin-compile
   cmake <psi4-generated-options> -B objdir \
     -DCMAKE_INSTALL_PREFIX=/path/to/hilbert
   cmake --build objdir --parallel N
   cmake --install objdir
   ```

4. Run the test suite:

   ```bash
   cd tests
   make
   ```

The matrix-free GPU-BPSDP solver additionally requires a CUDA-enabled PyTorch
installation, a compatible CUDA toolkit and compiler for the JIT extensions. Primal sharding uses
PyTorch's NCCL bindings.

This code was tested with:
cuda 12.1.1
ompi 5.0.8
gcc 11.3.0
ninja 1.11.1
torch 2.5.1 

## Use

Hilbert methods are available from Psi4 input files and the Psi4 Python API.
The inputs under `tests/` provide complete examples. General Hilbert option
definitions live in `src/plugin.cc`.

### GPU-BPSDP 
All critical settings are set by default when GPUs are available. 
Simply select gpu_admm as the SDP solver. It works with arbitrary symmetry/point groups. 
NOTE ONLY DQG POSITIVITY IS SUPPORTED. 

```python
set hilbert {
  sdp_solver gpu_admm
  positivity dqg
}
```

### GPU-ADMM settings

The defaults below are the production defaults registered with Psi4. The
matrix-free CUDA operator, fused normal kernel, sharded multi-GPU execution, and
multi-GPU PSD projection are enabled by default.

#### Operator and diagnostics

| Option | Default | Description |
| --- | ---: | --- |
| `GPU_ADMM_PROFILE` | `false` | Print synchronized timing and memory diagnostics; disabled by default to avoid profiling synchronization overhead. |
| `GPU_ADMM_MATRIX_FREE` | `true` | Use the DQG matrix-free operator instead of explicit CSR matrices. Unsupported problems fall back to the explicit operator. |
| `GPU_ADMM_MATRIX_FREE_AT_CSR` | `false` | Use matrix-free `Au` with explicit CSR `A^T u`. |
| `GPU_ADMM_MATRIX_FREE_CUDA_AU` | `true` | Use the CUDA `Au` kernel. |
| `GPU_ADMM_MATRIX_FREE_CUDA_ATU` | `true` | Use the CUDA `A^T u` kernel. |
| `GPU_ADMM_CUDA_VERBOSE` | `false` | Show verbose PyTorch CUDA-extension build output. For debugging. |

#### Conjugate gradient and sharding

| Option | Default | Description |
| --- | ---: | --- |
| `CG_CONVERGENCE` | `1e-9` | Absolute floor for the dynamic inner-CG tolerance. |
| `CG_MAXITER` | `10000` | Maximum inner-CG iterations. |
| `GPU_ADMM_CG_DYNAMIC_FACTOR` | `0.01` | Dynamic tolerance factor applied to the smaller ADMM residual. |
| `GPU_ADMM_CG_FUSED_NORMAL` | `true` | Evaluate `A A^T` through the fused CUDA normal-operator entry point. |
| `GPU_ADMM_SHARD_PRIMAL` | `true` | Build the distributed normal-operator shard plan with compact caches and NCCL reduction. |
| `GPU_ADMM_SHARD_PRIMAL_STORE` | `true` | Store persistent primal vectors as disjoint device shards and keep full masters on the host. Requires `GPU_ADMM_SHARD_PRIMAL`. |

Primal sharding requires the matrix-free CUDA fused-normal path, multi-GPU PSD
block assignment and at least two PSD devices. The solver falls back to the single-GPU normal operator when those 
prerequisites are not met.

#### PSD projection

| Option | Default | Description |
| --- | ---: | --- |
| `GPU_ADMM_PSD_MULTI_GPU` | `true` | Distribute independent cone-block eigensolves across GPUs. One visible device falls back to the single-GPU path. |
| `GPU_ADMM_PSD_DEVICES` | `""` | Comma-separated logical CUDA device IDs. Empty means all visible devices. |
| `GPU_ADMM_PSD_BASE_MEMORY_FRACTION` | `0.92` | Fraction of base-GPU memory available to the PSD scheduler. |
| `GPU_ADMM_PSD_WORKSPACE_SCALE` | `5.0` | Conservative eigensolver-workspace multiplier used by the scheduler. |
| `GPU_ADMM_PSD_AVOID_BASE_LARGE_BLOCKS` | `true` | Keep the largest blocks off the base GPU when enough non-base GPUs are available. |

#### Orbital optimization

| Option | Default | Description |
| --- | ---: | --- |
| `ORBOPT_FOCAS_DF_CUDA` | `true` | Attempt the CUDA/cuBLAS density-fitted FOCAS transform. Incompatible cases automatically fall back to the CPU implementation. |
| `ORBOPT_FOCAS_COMPACT_ROTATION` | `true` | Use the exact low-rank representation of `exp(K)-I`, a rank-aware crossover selects the dense path when cheaper. |
| `ORBOPT_EXACT_DIAGONAL_HESSIAN` | `false` | Use exact diagonal orbital-Hessian expressions. Default to faster diagonal. |
| `ORBOPT_FOCAS_DF_CUDA_NUM_GPUS` | `0` | Maximum visible GPUs for CUDA FOCAS. Non-positive uses all visible devices. |

CUDA orbital optimization is opportunistic. Setting it as the default does not make CUDA a
requirement for orbital optimization. Hilbert falls back to the blocked CPU
transform if the helper cannot be loaded or the problem is incompatible.

#### Convergence and acceleration

| Option | Default | Description |
| --- | ---: | --- |
| `GPU_ADMM_RELAXATION` | `1.0` | ADMM over-relaxation factor; `1.0` disables over-relaxation. |
| `GPU_ADMM_GAP_RELATIVE` | `false` | Add a relative duality-gap convergence target. |
| `GPU_ADMM_GAP_RELATIVE_TOL` | `0.0` | Relative gap tolerance when enabled. |
| `GPU_ADMM_STAGNATION_WINDOW` | `100` | Energy-stagnation window in ADMM iterations. |
| `GPU_ADMM_STAGNATION_ENERGY_TOL` | `0.0` | Energy span threshold; non-positive values use `E_CONVERGENCE`. |
