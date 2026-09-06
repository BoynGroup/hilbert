/*
 *  @BEGIN LICENSE
 *
 *  Hilbert: a space for quantum chemistry plugins to Psi4
 *
 *  Copyright (c) 2020 by its authors (LICENSE).
 *
 *  The copyrights for code used from other parties are included in
 *  the corresponding files.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 *  @END LICENSE
 */

#include <psi4/libmints/wavefunction.h>
#include <psi4/liboptions/liboptions.h>
#include <psi4/libpsi4util/PsiOutStream.h>
#include <psi4/libpsi4util/process.h>
#include <psi4/libpsio/psio.hpp>
#include <psi4/psi4-dec.h>

#include <doci/doci_solver.h>
#include <jellium/jellium_scf_solver.h>
#include <misc/backtransform_tpdm.h>
#include <p2rdm/p2rdm_solver.h>
#include <polaritonic_scf/rcis.h>
#include <polaritonic_scf/rhf.h>
#include <polaritonic_scf/rks.h>
#include <polaritonic_scf/rohf.h>
#include <polaritonic_scf/rtddft.h>
#include <polaritonic_scf/uccsd.h>
#include <polaritonic_scf/uhf.h>
#include <polaritonic_scf/uks.h>
#include <polaritonic_scf/utddft.h>
#include <pp2rdm/pp2rdm_solver.h>
#include <v2rdm_casscf/v2rdm_solver.h>
#include <v2rdm_doci/v2rdm_doci_solver.h>

#ifdef WITH_TA
#include "cc_cavity/include/cc_cavity.h"
#include "python_api/python_helpers.h"
#include <tiledarray.h>

#include "cc_cavity/include/ccsd/ccsd.h"
#include "cc_cavity/include/qed_ccsd_21/qed_ccsd_21.h"
#include "cc_cavity/include/qed_ccsd_22/qed_ccsd_22.h"

#include "cc_cavity/include/ccsd/eom_ee_ccsd.h"
#include "cc_cavity/include/ccsd/eom_ee_rdm.h"

#include "cc_cavity/include/qed_ccsd_21/eom_ee_qed_ccsd_21.h"
#include "cc_cavity/include/qed_ccsd_21/eom_ee_qed_rdm_21.h"

#include "cc_cavity/include/ccsd/eom_ea_ccsd.h"
#include "cc_cavity/include/ccsd/eom_ea_rdm.h"

#include "cc_cavity/include/qed_ccsd_21/eom_ea_qed_ccsd_21.h"
#include "cc_cavity/include/qed_ccsd_21/eom_ea_qed_rdm_21.h"

#endif

using namespace psi;

namespace hilbert {

extern "C" PSI_API int read_options(std::string name, Options &options) {
  if (name == "HILBERT" || options.read_globals()) {

    /*- SUBSECTION General -*/

    /*- qc solver. used internally !expert -*/
    options.add_str(
        "HILBERT_METHOD", "",
        "DOCI P2RDM PP2RDM V2RDM_DOCI V2RDM_CASSCF JELLIUM_SCF POLARITONIC_RHF "
        "POLARITONIC_UHF POLARITONIC_ROHF POLARITONIC_UKS POLARITONIC_RKS "
        "POLARITONIC_RCIS POLARITONIC_UCCSD POLARITONIC_RTDDFT "
        "POLARITONIC_UTDDFT POLARITONIC_RPA MCPDFT CC_CAVITY");

    /*- Do DIIS? -*/
    options.add_bool("DIIS", true);

    /*- convergence in the energy -*/
    options.add_double("E_CONVERGENCE", 1e-6);

    /*- convergence in the CI coefficients -*/
    options.add_double("R_CONVERGENCE", 1e-5);

    /*- maximum number of macroiterations -*/
    options.add_int("MAXITER", 50);

    /*- Do write the 2-RDM to disk? All nonzero elements of the 2-RDM will be
     * written.  -*/
    options.add_bool("TPDM_WRITE_FULL", false);

    /*- Do print 2-RDM and 1-RDM to the output file? All nonzero elements of the
     * RDMs will be written.  -*/
    options.add_bool("PRINT_RDMS", false);

    /*- Do print t-and z-amplitudes to the output file? -*/
    options.add_bool("PRINT_PCCD_AMPLITUDES", false);

    /*- Auxiliary basis set for SCF density fitting computations.
    :ref:`Defaults <apdx:basisFamily>` to a JKFIT basis. -*/
    options.add_str("DF_BASIS_SCF", "");

    /*- What algorithm to use for the SCF computation. See Table :ref:`SCF
    Convergence & Algorithm <table:conv_scf>` for default algorithm for
    different calculation types. -*/
    options.add_str("SCF_TYPE", "DF", "DF CD DISK_DF MEM_DF");

    /*- SUBSECTION DOCI -*/

    /*- maximum size of Davidson subspace (will be multiplied by number of
     * desired roots) -*/
    options.add_double("DAVIDSON_MAXDIM", 20);

    /*- Tolerance for Cholesky decomposition of the ERI tensor -*/
    options.add_double("CHOLESKY_TOLERANCE", 1e-4);

    /*- Do localize orbitals prior to v2RDM-DOCI? -*/
    options.add_bool("LOCALIZE_ORBITALS", false);

    /*- Do localize virtual orbitals prior to v2RDM-DOCI? -*/
    options.add_bool("LOCALIZE_VIRTUAL_ORBITALS", false);

    /*- Do add random noise to initial orbitals prior to DOCI -*/
    options.add_bool("NOISY_ORBITALS", false);

    /*- Do optimize orbitals? -*/
    options.add_bool("OPTIMIZE_ORBITALS", true);

    /*- SUBSECTION ORBITAL OPTIMIZATION -*/

    options.add_bool("SAVE_SCF", false);
    /*- Write a MOLDEN file for the orbitals at each CASSCF iteration. -*/

    options.add_bool("MOLDEN_WRITE", false);

    /*- Filename for the final v2RDM-CASSCF MOLDEN file. If empty, use the
    Psi4 writer prefix plus .molden. -*/
    options.add_str("MOLDEN_FILE", "");

    /*- Do write a MOLDEN file for guess orbitals?  If so, the filename will
    end in .guess.molden, and the prefix is determined by
    |globals__writer_file_label| (if set), or else by the name of the output
    file plus the name of the current molecule. -*/
    options.add_bool("GUESS_ORBITALS_WRITE", false);

    /*- flag to optimize orbitals using a one-step type approach -*/
    options.add_bool("ORBOPT_ONE_STEP", true);

    /*- algorithm for orbital optimization. only valid for v2rdm-doci -*/
    options.add_str(
        "ORBOPT_ALGORITHM", "HAGER_ZHANG",
        "STEEPEST_DESCENT HESTENES_STIEFEL DAI_YUAN HAGER_ZHANG KOU_DAI");

    /*- frequency of orbital optimization.  optimization occurs every
    orbopt_frequency iterations. only valid for v2rdm-doci -*/
    options.add_int("ORBOPT_FREQUENCY", 500);

    /*- convergence in gradient norm -*/
    options.add_double("ORBOPT_GRADIENT_CONVERGENCE", 1.0e-4);

    /*- convergence in energy for rotations -*/
    options.add_double("ORBOPT_ENERGY_CONVERGENCE", 1.0e-8);

    /*- do rotate active-active orbital pairs. No methods in Hilbert use this
     * flag currently !expert -*/
    options.add_bool("ORBOPT_ACTIVE_ACTIVE_ROTATIONS", false);

    /*- Use exact expressions for diagonal orbital-Hessian elements. -*/
    options.add_bool("ORBOPT_EXACT_DIAGONAL_HESSIAN", false);

    /*- number of DIIS vectors to keep in orbital optimization -*/
    options.add_int("ORBOPT_NUM_DIIS_VECTORS", 0);

    /*- maximum number of iterations for orbital optimization -*/
    options.add_int("ORBOPT_MAXITER", 10);

    /*- Use a smaller orbital-optimization microiteration cap for early
    CASSCF orbital-rotation steps, then optionally restore the full cap for
    final/refinement steps. -*/
    options.add_bool("ORBOPT_ADAPTIVE_MAXITER", false);

    /*- Early orbital-optimization microiteration cap used when
    ORBOPT_ADAPTIVE_MAXITER is true. -*/
    options.add_int("ORBOPT_ADAPTIVE_START_MAXITER", 5);

    /*- Refinement orbital-optimization microiteration cap used when
    ORBOPT_ADAPTIVE_MAXITER is true. A value <= 0 means use
    ORBOPT_MAXITER. -*/
    options.add_int("ORBOPT_ADAPTIVE_FINAL_MAXITER", 0);

    /*- Switch to the refinement orbital-optimization microiteration cap
    when the previous orbital-gradient norm is at or below this threshold.
    A value <= 0 disables gradient-based switching. -*/
    options.add_double("ORBOPT_ADAPTIVE_SWITCH_GRADIENT", 2.0e-2);

    /*- Keep the last accepted orbital step size and use it as the
    first trial step in later matching CASSCF orbital-optimization calls. -*/
    options.add_bool("ORBOPT_FOCAS_STEP_MEMORY", true);

    /*- Use the exact compact representation U=I+V*A*V^T when the external
    orbital space is larger than the nonfrozen occupied-plus-active space.
    Reduces both the exponential and DF transform scaling without a rank
    truncation. A rank-aware crossover selects the dense path when cheaper. -*/
    options.add_bool("ORBOPT_FOCAS_COMPACT_ROTATION", true);

    /*- Factor used to grow the next orbital trial step after an
    accepted step. The default is 2.0; smaller values can reduce
    rejected DF integral transforms. -*/
    options.add_double("ORBOPT_FOCAS_STEP_INCREASE_FACTOR", 2.0);

    /*- Use the C1 blocked density-fitted integral transform when
    compatible. -*/
    options.add_bool("ORBOPT_FOCAS_DF_C1_BLOCKED", true);

    /*- Number of auxiliary functions per block in the C1 blocked
    density-fitted integral transform. A value <= 0 gives the
    automatic default. -*/
    options.add_int("ORBOPT_FOCAS_DF_C1_BLOCK_Q", 0);

    /*- Fraction of the Psi4 memory available to use as total
    per-call scratch for automatic C1 blocked density-fitted
    transforms. Only used when ORBOPT_FOCAS_DF_C1_BLOCK_Q <= 0. -*/
    options.add_double("ORBOPT_FOCAS_DF_C1_BLOCK_MEMORY_FRACTION", 0.05);

    /*- For automatic C1 blocked density-fitted transforms, scale
    the scratch budget from the smaller of the Psi4 memory setting and
    currently available host memory. -*/
    options.add_bool("ORBOPT_FOCAS_DF_C1_BLOCK_USE_AVAILABLE_MEMORY", true);

    /*- Maximum automatic auxiliary block size for the CPU C1 blocked
    density-fitted integral transform. Explicit
    ORBOPT_FOCAS_DF_C1_BLOCK_Q values may exceed this cap. The CUDA path
    chooses its own GPU-memory-aware block size when this option is
    automatic. -*/
    options.add_int("ORBOPT_FOCAS_DF_C1_BLOCK_Q_MAX", 32);

    /*- Use the CUDA/cuBLAS implementation of the density-fitted FOCAS
    integral transform. The CUDA path is attempted by default and falls back
    to the CPU implementation when the helper is unavailable or the system is
    incompatible. The helper is JIT-compiled with nvcc on first use. -*/
    options.add_bool("ORBOPT_FOCAS_DF_CUDA", true);

    /*- Maximum number of visible CUDA GPUs to use for the optional
    density-fitted FOCAS transform. A value <= 0 uses all visible CUDA devices.
    -*/
    options.add_int("ORBOPT_FOCAS_DF_CUDA_NUM_GPUS", 0);

    /*-Alias for ORBOPT_FOCAS_DF_CUDA -*/
    options.add_bool("ORBOPT_FOCAS_DF_C1_CUDA", true);

    /*-Alias for ORBOPT_FOCAS_DF_CUDA_NUM_GPUS -*/
    options.add_int("ORBOPT_FOCAS_DF_C1_CUDA_NUM_GPUS", 0);

    /*- Backend for the initial, direct block-streamed DF AO-to-MO transform.
    AUTO attempts the JIT CUDA helper and safely falls back to the direct CPU
    implementation. Neither direct backend writes full QSO or half-transformed
    tensors to scratch. -*/
    options.add_str("DF_INTEGRAL_TRANSFORM_BACKEND", "AUTO", "AUTO CPU CUDA");

    /*- Host auxiliary-function block size for the initial direct DF transform.
    A value <= 0 chooses a memory-aware block size. -*/
    options.add_int("DF_INTEGRAL_TRANSFORM_BLOCK_Q", 0);

    /*- Fraction of configured Psi4 memory available for host scratch in the
    initial direct DF transform, after reserving the final QMO tensor. -*/
    options.add_double("DF_INTEGRAL_TRANSFORM_MEMORY_FRACTION", 0.05);

    /*- Limit automatic host blocks for the initial direct DF transform. An
    explicit DF_INTEGRAL_TRANSFORM_BLOCK_Q may exceed this cap when it fits the
    checked memory budget. -*/
    options.add_int("DF_INTEGRAL_TRANSFORM_BLOCK_Q_MAX", 32);

    /*- Bound initial direct-transform host scratch using currently available
    system memory in addition to the configured Psi4 memory. -*/
    options.add_bool("DF_INTEGRAL_TRANSFORM_USE_AVAILABLE_MEMORY", true);

    /*- Maximum visible GPUs for the initial JIT CUDA DF transform. A value <=
    0 uses all visible GPUs. Device block sizes are selected from free memory
    independently of the host block size. -*/
    options.add_int("DF_INTEGRAL_TRANSFORM_CUDA_NUM_GPUS", 0);

    /*- maximum number of cycles for CASSCF -*/
    options.add_int("SCF_MAXITER", 75);

    /*- Do write a ORBOPT output file?  If so, the filename will end in
    .molden, and the prefix is determined by |globals__writer_file_label|
    (if set), or else by the name of the output file plus the name of
    the current molecule. -*/
    options.add_bool("ORBOPT_WRITE", false);

    /*- SUBSECTION pp2RDM -*/

    /*- do check analytic gradient for accuracy? -*/
    options.add_bool("CHECK_GRADIENT", false);

    /*- do check analytic hessian for accuracy? -*/
    options.add_bool("CHECK_HESSIAN", false);

    /*- File containing previous primal/dual solutions and integrals. -*/
    options.add_str("RESTART_FROM_CHECKPOINT_FILE", "");

    /*- algorithm type -*/
    options.add_str("P2RDM_ALGORITHM", "PROJECTION",
                    "PROJECTION LBFGS NEWTON_RAPHSON");

    /*- Which parametric 2-RDM method is called? Set by driver. !expert -*/
    options.add_str("P2RDM_TYPE", "K", "K CEPA(0) CEPA(1) CID ACPF AQCC CCD");

    /*- Do print 1- and 2-electron to the output file? Only J-, K-, and L-type
     * integrals will be printed. -*/
    options.add_bool("PRINT_INTEGRALS", false);

    /*- SUBSECTION v2RDM-DOCI -*/

    /* Do v2RDM-DOCI gradient? !expert */
    options.add_str("DERTYPE", "NONE", "NONE FIRST");

    /*- Do semicanonicalize orbitals? -*/
    options.add_bool("SEMICANONICALIZE_ORBITALS", false);

    /*- Type of guess -*/
    options.add_str("TPDM_GUESS", "RANDOM", "RANDOM HF");

    /*- Do save progress in a checkpoint file? -*/
    options.add_bool("WRITE_CHECKPOINT_FILE", false);

    /*- When WRITE_CHECKPOINT_FILE is true, also save the MO-basis 3-index
    integrals (Qmo) to a companion .qmo file alongside the checkpoint.
    Enabling this allows a restart with RESTART_FROM_CHECKPOINT_FILE to skip
    the AO->MO integral transformation entirely. Only written at the
    final/post-orbopt checkpoint, not at every EACH_STEP write. Note that this
    may be a very large file! File size: nQ * nret*(nret+1)/2 * 8 bytes, where
    nret excludes frozen virtual orbitals. -*/
    options.add_bool("CHECKPOINT_WRITE_QMO", false);

    /*- Filename for durable v2RDM checkpoint writes. If empty, use the
    Psi4 writer prefix plus .v2rdm.chk. -*/
    options.add_str("CHECKPOINT_FILE", "");

    /*- When WRITE_CHECKPOINT_FILE is true, write only the final solution or
    update the checkpoint after each CASSCF/SDP macro step. -*/
    options.add_str("CHECKPOINT_WRITE_MODE", "FINAL", "FINAL EACH_STEP");

    /*- Frequency of checkpoint file generation.  The checkpoint file is
    updated every CHECKPOINT_FREQUENCY iterations.  The default frequency
    will be ORBOPT_FREQUENCY. -*/
    options.add_int("CHECKPOINT_FREQUENCY", 500);

    /*- Frequency with which the penalty parameter, mu, is considered for
    hybrid adaptive residual-balancing updates. Set to 0 to disable mu updates.
    -*/
    options.add_int("MU_UPDATE_FREQUENCY", 1000);

    /*- The type of 2-positivity computation -*/
    options.add_str("POSITIVITY", "DQG",
                    "DQG D DQ DG DQGT1 DQGT2 DQGT1T2 3POS");

    /*- Do enforce generalized pauli constraints -*/
    options.add_str("GPC_CONSTRAINTS", "NONE", "NONE 1RDM 2RDM");

    /*- Do constrain D3 to D2 mapping? -*/
    options.add_bool("CONSTRAIN_D3", false);

    /*- Do constrain E3 to D3 mapping? -*/
    options.add_bool("CONSTRAIN_E3", false);

    /*- Do constrain F3 to D3 mapping? -*/
    options.add_bool("CONSTRAIN_F3", false);

    /*- Do constrain Q3 to D3 mapping? -*/
    options.add_bool("CONSTRAIN_Q3", false);

    /*- Absolute convergence floor for the GPU-ADMM conjugate-gradient
    solver. Dynamic CG tolerances are never allowed below this value. -*/
    options.add_double("CG_CONVERGENCE", 1e-9);

    /*- maximum number of conjugate gradient iterations -*/
    options.add_int("CG_MAXITER", 10000);

    /*- SUBSECTION v2RDM-CASSCF -*/

    /*- SDP solver -*/
    options.add_str("SDP_SOLVER", "BPSDP", "BPSDP RRSDP CVXPY GPU_ADMM");

    /*- Replace fixed ORBOPT_FREQUENCY chunks with step-wise ladder GPU-ADMM
    CASSCF path. A scalar accuracy target starts at 5e-2 and halves continuously
    to the configured production residual and gap tolerances. The raw-residual
    target is capped at 1e-2. Intermediate promotion requires two consecutive
    certified macrocycles whose total CASSCF energy changes are below
    max(CASSCF_ENERGY_CONVERGENCE, 0.1 times the current gap). Two consecutive
    user-level energy passes freeze the orbitals and request one final solve at
    the configured production residual and gap targets. The final polished
    energy controls adaptive convergence. Explicitly setting ORBOPT_FREQUENCY
    always selects the legacy fixed-frequency path and overrides this option.
    -*/
    options.add_bool("CASSCF_ADAPTIVE_SDP", false);

    /*- Required total-energy change for adaptive CASSCF convergence. Two
    consecutive certified macrocycles must pass this threshold before the
    final user-accuracy SDP solve. -*/
    options.add_double("CASSCF_ENERGY_CONVERGENCE", 1.0e-5);

    /*- Validate GPU-ADMM sparse A against callback matvecs. For
     * debugging/development only! -*/
    options.add_bool("GPU_ADMM_VALIDATE_A", false);

    /*- Print synchronized GPU-ADMM timing breakdowns. -*/
    options.add_bool("GPU_ADMM_PROFILE", false);

    /*- Use the matrix-free DQG GPU-ADMM operator instead of storing explicit
    CSR(A,A^T). Unsupported systems automatically use the explicit operator. -*/
    options.add_bool("GPU_ADMM_MATRIX_FREE", true);

    /*- When GPU_ADMM_MATRIX_FREE is true, also build explicit CSR(A,A^T) and
    compare random matrix-vector products against the matrix-free operator.
    For debug/testing only! -*/
    options.add_bool("GPU_ADMM_MATRIX_FREE_VALIDATE", false);

    /*- Hybrid compatibility mode for GPU_ADMM_MATRIX_FREE: use matrix-free Au
    but explicit CSR(A^T) for ATu. This stores roughly half of explicit
    CSR(A,A^T) on the GPU. For testing/debug only. -*/
    options.add_bool("GPU_ADMM_MATRIX_FREE_AT_CSR", false);

    /*- Use a custom CUDA kernel for the DQG matrix-free Au operation.
    Requires CUDA and PyTorch's JIT extension support. -*/
    options.add_bool("GPU_ADMM_MATRIX_FREE_CUDA_AU", true);

    /*- Use a custom CUDA kernel for the DQG matrix-free ATu operation.
    This removes the explicit CSR(A^T) GPU memory requirement when
    GPU_ADMM_MATRIX_FREE_AT_CSR is false. -*/
    options.add_bool("GPU_ADMM_MATRIX_FREE_CUDA_ATU", true);

    /*- Print verbose PyTorch CUDA-extension build output when JIT-compiling
    the GPU-ADMM matrix-free kernels. -*/
    options.add_bool("GPU_ADMM_CUDA_VERBOSE", false);

    /*- Run the expensive Au/ATu range, distributed-normal, primal-store,
    and PSD-store validation suite for the custom CUDA operator. For
    testing/debug only. -*/
    options.add_bool("GPU_ADMM_VALIDATE_AURANGE", false);

    /*- Dynamic CG convergence factor for GPU_ADMM. When dynamic CG is
    enabled, the inner CG tolerance is max(CG_CONVERGENCE,
    GPU_ADMM_CG_DYNAMIC_FACTOR * min(primal_error, dual_error)), with a
    first-iteration tolerance of GPU_ADMM_CG_DYNAMIC_FACTOR.
    Larger values reduce CG work but need to be validated for energy and
    convergence stability. Testing/development only! Do not change! -*/
    options.add_double("GPU_ADMM_CG_DYNAMIC_FACTOR", 0.01);

    /*- For matrix-free CUDA DQG/C1 CG, evaluate the normal operator A A^T
    through a fused-normal CUDA extension entry point. Requires
    GPU_ADMM_MATRIX_FREE_CUDA_AU and
    GPU_ADMM_MATRIX_FREE_CUDA_ATU. -*/
    options.add_bool("GPU_ADMM_CG_FUSED_NORMAL", true);

    /*- Build the distributed CUDA normal-operator shard plan from the
    multi-GPU PSD block assignment. This enables compact per-GPU caches and
    NCCL reduction of the shared D2/D1 region. Requires matrix-free CUDA
    fused-normal CG and at least two GPU_ADMM_PSD_DEVICES. -*/
    options.add_bool("GPU_ADMM_SHARD_PRIMAL", true);

    /*- Store the persistent primal vectors as disjoint per-GPU block
    shards and keep their full masters on the host. Requires
    GPU_ADMM_SHARD_PRIMAL. The solver falls back to full vectors when the
    sharded execution prerequisites are not met. -*/
    options.add_bool("GPU_ADMM_SHARD_PRIMAL_STORE", true);

    /*- Distribute GPU-ADMM PSD projections over multiple visible CUDA GPUs.
    This parallelizes independent cone blocks; a single large block still
    uses one GPU eigensolve. A single visible device uses the normal
    single-GPU path. -*/
    options.add_bool("GPU_ADMM_PSD_MULTI_GPU", true);

    /*- Comma-separated CUDA device ids for GPU_ADMM_PSD_MULTI_GPU. An empty
    value uses all visible CUDA devices. Example: "0,1,2,3". -*/
    options.add_str("GPU_ADMM_PSD_DEVICES", "");

    /*- Fraction of the base CUDA GPU memory budget that may be considered
    usable when assigning PSD projection blocks in multi-GPU mode. The
    scheduler subtracts an estimate of resident ADMM/CG vectors before
    assigning PSD work to the base GPU. -*/
    options.add_double("GPU_ADMM_PSD_BASE_MEMORY_FRACTION", 0.92);

    /*- Conservative multiplier for estimating temporary eigensolver
    workspace for each PSD block on the base GPU. Larger values keep more
    memory headroom when assigning PSD work to cuda:0. -*/
    options.add_double("GPU_ADMM_PSD_WORKSPACE_SCALE", 5.0);

    /*- In multi-GPU PSD mode, this avoids assigning the largest PSD blocks to
    the base GPU when there are enough non-base GPUs to hold those largest
    blocks. Improves memory headroom for large matrix-free runs
    as cuda:0 also owns ADMM/CG vectors. -*/
    options.add_bool("GPU_ADMM_PSD_AVOID_BASE_LARGE_BLOCKS", true);

    /*- GPU_ADMM convergence acceleration: ADMM over-relaxation factor alpha.
    Relaxes the projection coupling to A^T y -> alpha*A^T y + (1-alpha)(c - z),
    leaving the fixed point unchanged. Disabled with 1.0. Values in (1.0, 2.0)
    (ideally around ~1.6) reduce iteration count. DEVELOPMENT ONLY. DO NOT USE
    FOR PRODUCTION SIMULATIONS. -*/
    options.add_double("GPU_ADMM_RELAXATION", 1.0);

    /*- GPU_ADMM: use a relative duality-gap target. When true the effective
    gap tolerance is max(E_CONVERGENCE, GPU_ADMM_GAP_RELATIVE_TOL*|E|). For
    debugging/testing. -*/
    options.add_bool("GPU_ADMM_GAP_RELATIVE", false);

    /*- GPU_ADMM relative duality-gap tolerance (used when
    GPU_ADMM_GAP_RELATIVE is true). -*/
    options.add_double("GPU_ADMM_GAP_RELATIVE_TOL", 0.0);

    /*- GPU_ADMM always-on energy-stagnation stop. window (in ADMM iterations)
    over which the primal (variational) energy must stay flat. -*/
    options.add_int("GPU_ADMM_STAGNATION_WINDOW", 100);

    /*- GPU_ADMM energy-stagnation tolerance: the solver stops when feasibility
    is met and the primal energy spans less than this over the window. A
    non-positive value defaults to E_CONVERGENCE. -*/
    options.add_double("GPU_ADMM_STAGNATION_ENERGY_TOL", 0.0);

    /*- CVXPY underlying solver -*/
    options.add_str("CVXPY_SOLVER", "");

    /*- do use hubbard model? -*/
    options.add_bool("HUBBARD_HAMILTONIAN", false);

    /*- hubbard hopping integral -*/
    options.add_double("HUBBARD_T", 1.0);

    /*- hubbard on-site repulsion -*/
    options.add_double("HUBBARD_U", 1.0);

    /*- number of sites in hubbard model -*/
    options.add_int("N_HUBBARD_SITES", 4);

    /*- total number of spins in hubbard model -*/
    options.add_int("N_HUBBARD_SPINS", 4);

    /*- multiplicity in hubbard model -*/
    options.add_int("HUBBARD_MULTIPLICITY", 1);

    /*- fractional charge -*/
    options.add_double("FRACTIONAL_CHARGE", 0.0);

    /*- do extended koopmans theorem computation? -*/
    options.add_bool("EXTENDED_KOOPMANS", false);

    /*- Do v2RDM-CASSCF gradient? !expert -*/
    options.add_str("DERTYPE", "NONE", "NONE FIRST");

    /* Do write fcidump files? -*/
    options.add_bool("FCIDUMP", false);

    /*- Rotate guess orbitals -*/
    options.add("MCSCF_ROTATE", new ArrayType());

    /*- Do compute natural orbitals and transform 1- and 2-RDM to the natural
    orbital basis? The OPDM and Ca/Cb matrices pushed onto the wavefunction will
    correspond to the natural orbital basis -*/
    options.add_bool("NAT_ORBS", false);

    /*- Do write the 1-RDM to disk? All nonzero elements of the 1-RDM will be
     * written.  -*/
    options.add_bool("OPDM_WRITE_FULL", false);

    /*- Do write the spin-free 2-RDM to disk? All nonzero elements of the 2-RDM
     * will be written.  -*/
    options.add_bool("TPDM_WRITE_SPIN_FREE", false);

    /*- Do write the 2-RDM to disk? Only the nonzero elements of the active
     * 2-RDM will be written. -*/
    options.add_bool("TPDM_WRITE", false);

    /*- Do write the 3-RDM to disk? -*/
    options.add_bool("3PDM_WRITE", false);

    /*- A parameter introduced by Mazziotti [PRL 106, 083001 (2011)] to
    "increase the sensitivity of y on the deviation of x from primal
    feasibility."  Should lie on the interval [1.0, 1.6]. -*/
    options.add_double("TAU_PARAMETER", 1.0);

    /*- Do constrain D4 to D3 mapping? -*/
    options.add_bool("CONSTRAIN_D4", false);

    /*- Do constrain spin squared? -*/
    options.add_bool("CONSTRAIN_SPIN", true);

    /*- Do constrain sz? -*/
    options.add_bool("CONSTRAIN_SZ", true);

    /*- SUBSECTION JELLIUM -*/

    /*- An array containing the number of doubly-occupied orbitals per irrep
    (in Cotton order) -*/
    options.add("DOCC", new ArrayType());

    /*- The length of the box in nm. No default. If not specified, the
    box length is chosen to satisfy <rho> = 1e-/a0^3 -*/
    options.add_double("JELLIUM_BOX_LENGTH", 1.0);

    /*- The number of grid points for the Gauss-Legendre quadrature -*/
    options.add_int("N_GRID_POINTS", 10);

    /*- The number of electrons -*/
    options.add_int("N_ELECTRONS", 2);

    /*- The number of basis functions -*/
    options.add_int("N_BASIS_FUNCTIONS", 26);

    /*- The length of the box in nm -*/
    options.add_double("LENGTH", 1.0);
    // options.add_double("LENGTH", 0.166245);

    ///*- The density of the box in e/nm^3 -*/
    // options.add_double("DENSITY", 92);

    /*- The number of electronic states to computed, per irreducible
    representation -*/
    options.add("ROOTS_PER_IRREP", new ArrayType());

    /*- Do smart guess in Davidson? Requires exact hamiltonian elements
    and could get expensive, default = false -*/
    options.add_bool("JELLIUM_CIS_SMART_GUESS", false);

    /*- SUBSECTION POLARITONIC SCF -*/

    /*- do compute static polarizability / hyperpolarizability QED-HF and
     * QED-DFT -*/
    options.add_bool("COMPUTE_STATIC_RESPONSE", false);

    /*- functional for cavity QED-DFT -*/
    options.add_str("QED_DFT_FUNCTIONAL", "B3LYP");

    /*- number of photon number states -*/
    options.add_int("N_PHOTON_STATES", 2);

    /*- do use coherent-state basis? !expert -*/
    options.add_bool("USE_COHERENT_STATE_BASIS", true);

    /*- do use quadrupole integrals for squared dipole terms? -*/
    options.add_bool("USE_QUADRUPOLE_INTEGRALS", true);

    /*- cavity excitation energy for the modes along the x, y and z axis (a.u.)
     * -*/
    options.add("CAVITY_FREQUENCY", new ArrayType());

    /*- cavity coupling strength (a.u.) -*/
    options.add("CAVITY_COUPLING_STRENGTH", new ArrayType());

    /*- do include u0 in polaritioinic ccsd? -*/
    options.add_bool("QED_CC_INCLUDE_U0", true);

    /*- do include u1 in polaritioinic ccsd? -*/
    options.add_bool("QED_CC_INCLUDE_U1", true);

    /*- do include u2 in polaritioinic ccsd? -*/
    options.add_bool("QED_CC_INCLUDE_U2", true);

    /*- do use TDA in TDDFT? -*/
    options.add_bool("TDSCF_TDA", false);

    /*- do relax orbitals in QED-SCF [unlike QED-TDDFT described in J. Chem.
     * Phys. 155, 064107 (2021)?] -*/
    options.add_bool("QED_USE_RELAXED_ORBITALS", true);

    /*- change cavity mode polarization by redefining x, y, and z -*/
    options.add_str("ROTATE_POLARIZATION_AXIS", "XYZ",
                    "XYZ YZX ZXY XZY YXZ ZYX");

    /*- residual norm -*/
    options.add_double("RESIDUAL_NORM", 1.0e-5);

    /*- initial size of Davidson subspace (will be multiplied by number of
     * desired roots) -*/
    options.add_int("INDIM", 5);

    /*- maximum size of Davidson subspace (will be multiplied by number of
     * desired roots) -*/
    options.add_int("MAXDIM", 20);

    /*- number of roots -*/
    options.add_int("NUMBER_ROOTS", 5);

    /*- SUBSECTION MCPDFT -*/

    /*- MCPDFT type -*/
    options.add_str("MCPDFT_METHOD", "MCPDFT", "MCPDFT");
    /*- MCPDFT functional -*/
    options.add_str("MCPDFT_FUNCTIONAL", "PBE");
    /*- JK object type can be DF or PK -*/
    options.add_str("MCPDFT_TYPE", "DF", "DF PK");
    /*- reference type -*/
    options.add_str("MCPDFT_REFERENCE", "V2RDM");
    /*- hybrid MCPDFT lambda parameter -*/
    options.add_double("MCPDFT_LAMBDA", 0.0);
    /*- custom functional parameters for exchange -*/
    options.add_str("MCPDFT_CUSTOM_PARAMS_X", "");
    /*- custom functional parameters for correlation -*/
    options.add_str("MCPDFT_CUSTOM_PARAMS_C", "");

    /*- SUBSECTION CC_Cavity -*/

    /*- Maximum number of vectors in DIIS -*/
    options.add_int("DIIS_MAX_VECS", 8);

    /*- tile size for tiled array -*/
    options.add_int("TILE_SIZE", -1); // -1 means to use all data in one tile

    /*- number of threads to use for MADNESS -*/
    options.add_int("MAD_NUM_THREADS", 1);

    /*- override inclusion of u0, u1, u2, u3, u4 -*/
    /*- CCSD-xy means include singles and doubles with x'th order excitations
     * coupled to y photon states -*/
    options.add_str("QED_CC_TYPE", "CCSD-21", "CCSD-00 CCSD-21 CCSD-22");

    /*- explicitly set u0=0 -*/
    options.add_bool("ZERO_U0", false);

    /*- Perform EOM for cc_cavity -*/
    options.add_bool("PERFORM_EOM", false);

    /*- Generate lambda amplitudes for cc_cavity -*/
    options.add_bool("PERFORM_LAMBDA", false);

    /*- Perform EOM with full hamiltonian for qed-ccsd (not implemented) -*/
    options.add_bool("BUILD_HAMILTONIAN", false);

    /*- Convergence criteria for davidson solver -*/
    options.add_double("EOM_R_CONV", 1e-6);
    options.add_double("EOM_E_CONV", 1e-8);
    options.add_int("EOM_MAXITER", 250);

    /*- Use diagonals from singles block in EOM-CC -*/
    options.add_bool("EOM_SS_GUESS", true);

    /*- value of shift to apply to davidson solver -*/
    options.add_double("EOM_SHIFT", 0.0l); // 0.0 means no shift

    /*- keep degenerate states in EOM-CC -*/
    options.add_bool("NO_DEGENERACY", false);

    /*- Save and load eigenvectors -*/
    options.add_bool("SAVE_EVECS", false); // save eigenvectors to file?
    options.add_int("LOAD_ID", -1); // id of file to load eigenvectors from

    /*- EOM-CC TYPE -*/
    options.add_str("EOM_TYPE", "EE", "EE EA");

    /*- do print excited state transition dipoles (computed either way) -*/
    options.add_bool("GET_PROPERTIES", true);

    /*- do print dominant transitions in each state -*/
    options.add_bool("PRINT_TRANSITIONS", true);

    /*- number of dominant transitions to print in each state -*/
    options.add_int("NUM_PRINT_TRANSITIONS", 8);

    /*- Select left and right states to visualize density or transition density
     * -*/
    options.add_array("RDM_STATES");

    /*- boolean for whether to compute the 2-RDM -*/
    options.add_bool("COMPUTE_2RDM", false);
  }

  return true;
}

extern "C" PSI_API SharedWavefunction hilbert(SharedWavefunction ref_wfn,
                                              Options &options) {

  if (options.get_str("HILBERT_METHOD") == "DOCI") {

    std::shared_ptr<DOCISolver> doci(new DOCISolver(ref_wfn, options));
    double energy = doci->compute_energy();
    return (std::shared_ptr<Wavefunction>)doci;

  } else if (options.get_str("HILBERT_METHOD") == "JELLIUM_SCF") {

    std::shared_ptr<Jellium_SCFSolver> jellium(new Jellium_SCFSolver(options));
    double energy = jellium->compute_energy();
    return ref_wfn;

  } else if (options.get_str("HILBERT_METHOD") == "PP2RDM") {

    std::shared_ptr<pp2RDMSolver> pp2rdm(new pp2RDMSolver(ref_wfn, options));
    double energy = pp2rdm->compute_energy();
    return (std::shared_ptr<Wavefunction>)pp2rdm;

  } else if (options.get_str("HILBERT_METHOD") == "P2RDM") {

    std::shared_ptr<p2RDMSolver> p2rdm(new p2RDMSolver(ref_wfn, options));
    double energy = p2rdm->compute_energy();
    return (std::shared_ptr<Wavefunction>)p2rdm;

  } else if (options.get_str("HILBERT_METHOD") == "V2RDM_DOCI") {

    std::shared_ptr<v2RDM_DOCISolver> v2rdm_doci(
        new v2RDM_DOCISolver(ref_wfn, options));
    double energy = v2rdm_doci->compute_energy();
    return (std::shared_ptr<Wavefunction>)v2rdm_doci;

  } else if (options.get_str("HILBERT_METHOD") == "V2RDM_CASSCF") {

    std::shared_ptr<v2RDMSolver> v2rdm(new v2RDMSolver(ref_wfn, options));
    double energy = v2rdm->compute_energy();

    if (options.get_str("DERTYPE") == "FIRST") {

      // backtransform the tpdm
      std::vector<std::shared_ptr<MOSpace>> spaces;
      spaces.push_back(MOSpace::all);
      std::shared_ptr<TPDMBackTransform> transform = std::shared_ptr<
          TPDMBackTransform>(new TPDMBackTransform(
          ref_wfn, spaces,
          IntegralTransform::TransformationType::Unrestricted, // Transformation
                                                               // type
          IntegralTransform::OutputType::DPDOnly,              // Output buffer
          IntegralTransform::MOOrdering::QTOrder,              // MO ordering
          IntegralTransform::FrozenOrbitals::None)); // Frozen orbitals?
      transform->backtransform_density();
      transform.reset();
    }

    return (std::shared_ptr<Wavefunction>)v2rdm;

  } else if (options.get_str("HILBERT_METHOD") == "POLARITONIC_RHF") {

    std::shared_ptr<PolaritonicRHF> rhf(new PolaritonicRHF(ref_wfn, options));
    double energy = rhf->compute_energy();

    return (std::shared_ptr<Wavefunction>)rhf;

  } else if (options.get_str("HILBERT_METHOD") == "POLARITONIC_RTDDFT") {

    std::shared_ptr<PolaritonicRKS> rks(new PolaritonicRKS(ref_wfn, options));
    double energy = rks->compute_energy();

    std::shared_ptr<PolaritonicRTDDFT> rtddft(new PolaritonicRTDDFT(
        (std::shared_ptr<Wavefunction>)rks, options, ref_wfn));
    double dum = rtddft->compute_energy();

    return (std::shared_ptr<Wavefunction>)rks;

  } else if (options.get_str("HILBERT_METHOD") == "POLARITONIC_UTDDFT") {

    std::shared_ptr<PolaritonicUKS> uks(new PolaritonicUKS(ref_wfn, options));
    double energy = uks->compute_energy();

    std::shared_ptr<PolaritonicUTDDFT> utddft(new PolaritonicUTDDFT(
        (std::shared_ptr<Wavefunction>)uks, options, ref_wfn));
    double dum = utddft->compute_energy();

    return (std::shared_ptr<Wavefunction>)uks;

  } else if (options.get_str("HILBERT_METHOD") == "POLARITONIC_RCIS") {

    std::shared_ptr<PolaritonicRHF> rhf(new PolaritonicRHF(ref_wfn, options));
    double energy = rhf->compute_energy();

    std::shared_ptr<PolaritonicRCIS> rcis(
        new PolaritonicRCIS((std::shared_ptr<Wavefunction>)rhf, options));
    double dum = rcis->compute_energy();

    return (std::shared_ptr<Wavefunction>)rhf;

  } else if (options.get_str("HILBERT_METHOD") == "POLARITONIC_UHF") {

    std::shared_ptr<PolaritonicUHF> uhf(new PolaritonicUHF(ref_wfn, options));
    double energy = uhf->compute_energy();

    return (std::shared_ptr<Wavefunction>)uhf;

  } else if (options.get_str("HILBERT_METHOD") == "POLARITONIC_ROHF") {

    std::shared_ptr<PolaritonicROHF> rohf(
        new PolaritonicROHF(ref_wfn, options));
    double energy = rohf->compute_energy();

    return (std::shared_ptr<Wavefunction>)rohf;

  } else if (options.get_str("HILBERT_METHOD") == "POLARITONIC_RKS") {

    std::shared_ptr<PolaritonicRKS> rks(new PolaritonicRKS(ref_wfn, options));
    double energy = rks->compute_energy();

    return (std::shared_ptr<Wavefunction>)rks;

  } else if (options.get_str("HILBERT_METHOD") == "POLARITONIC_UKS") {

    std::shared_ptr<PolaritonicUKS> uks(new PolaritonicUKS(ref_wfn, options));
    double energy = uks->compute_energy();

    if (options.get_bool("COMPUTE_STATIC_RESPONSE")) {
      std::shared_ptr<PolaritonicUTDDFT> utddft(new PolaritonicUTDDFT(
          (std::shared_ptr<Wavefunction>)uks, options, ref_wfn));
      utddft->compute_static_responses();
    }

    return (std::shared_ptr<Wavefunction>)uks;

  } else if (options.get_str("HILBERT_METHOD") == "POLARITONIC_UCCSD") {

    if (options.get_str("REFERENCE") == "UHF") {

      std::shared_ptr<PolaritonicUHF> uhf(new PolaritonicUHF(ref_wfn, options));
      double energy = uhf->compute_energy();

      std::shared_ptr<PolaritonicUCCSD> uccsd(
          new PolaritonicUCCSD((std::shared_ptr<Wavefunction>)uhf, options));
      energy = uccsd->compute_energy();
      return (std::shared_ptr<Wavefunction>)uccsd;

    } else if (options.get_str("REFERENCE") == "ROHF") {

      std::shared_ptr<PolaritonicROHF> rohf(
          new PolaritonicROHF(ref_wfn, options));
      double energy = rohf->compute_energy();

      std::shared_ptr<PolaritonicUCCSD> uccsd(
          new PolaritonicUCCSD((std::shared_ptr<Wavefunction>)rohf, options));
      energy = uccsd->compute_energy();
      return (std::shared_ptr<Wavefunction>)uccsd;

    } else if (options.get_str("REFERENCE") == "RHF") {

      std::shared_ptr<PolaritonicRHF> rhf(new PolaritonicRHF(ref_wfn, options));
      double energy = rhf->compute_energy();

      std::shared_ptr<PolaritonicUCCSD> uccsd(
          new PolaritonicUCCSD((std::shared_ptr<Wavefunction>)rhf, options));
      energy = uccsd->compute_energy();
      return (std::shared_ptr<Wavefunction>)uccsd;

    } else {

      throw PsiException("unknown REFERENCE for polaritonic UHF", __FILE__,
                         __LINE__);
    }

  } else if (options.get_str("HILBERT_METHOD") == "CC_CAVITY") {
#ifndef WITH_TA
    throw PsiException(
        "CC_CAVITY requires the WITH_TA flag to be set at compile time",
        __FILE__, __LINE__);
#else

    // get the reference wavefunction
    std::shared_ptr<Wavefunction> qed_ref_wfn;
    if (options.get_str("REFERENCE") == "UHF") {

      std::shared_ptr<PolaritonicUHF> uhf(new PolaritonicUHF(ref_wfn, options));
      double energy = uhf->compute_energy();

      qed_ref_wfn = (std::shared_ptr<Wavefunction>)uhf;

    } else if (options.get_str("REFERENCE") == "ROHF") {

      std::shared_ptr<PolaritonicROHF> rohf(
          new PolaritonicROHF(ref_wfn, options));
      double energy = rohf->compute_energy();

      qed_ref_wfn = (std::shared_ptr<Wavefunction>)rohf;

    } else if (options.get_str("REFERENCE") == "RHF") {

      std::shared_ptr<PolaritonicRHF> rhf(new PolaritonicRHF(ref_wfn, options));
      double energy = rhf->compute_energy();

      qed_ref_wfn = (std::shared_ptr<Wavefunction>)rhf;

    } else {
      throw PsiException("unknown REFERENCE for polaritonic UHF", __FILE__,
                         __LINE__);
    }

    // set the number of threads for MADNESS
    int mad_num_threads = options.get_int("MAD_NUM_THREADS");
    psi::Process::environment.globals["MAD_NUM_THREADS"] = mad_num_threads;
    string mad_num_env = "MAD_NUM_THREADS=" + std::to_string(mad_num_threads);
    putenv(const_cast<char *>(mad_num_env.c_str()));

    // initialize tiledarray with MPIComm from mpi4py if not already initialized
    if (!CavityHelper::initialized_)
      CavityHelper::ta_initialize();

    // create the CC_CAVITY object
    std::shared_ptr<CC_Cavity> qedcc;

    // get the options
    map<std::string, bool> includes_;
    std::string qed_type = options.get_str("QED_CC_TYPE");

    // select the appropriate derived CC_CAVITY object

    if (qed_type == "CCSD-00") {
      includes_["t0_1"] = false;
      includes_["t0_2"] = false;
      includes_["t1_1"] = false;
      includes_["t1_2"] = false;
      includes_["t2_1"] = false;
      includes_["t2_2"] = false;
      qedcc =
          std::shared_ptr<CC_Cavity>(new CCSD(qed_ref_wfn, options, includes_));
    } else if (qed_type == "CCSD-21") {
      includes_["t0_1"] = true;
      includes_["t0_2"] = false;
      includes_["t1_1"] = true;
      includes_["t1_2"] = false;
      includes_["t2_1"] = true;
      includes_["t2_2"] = false;
      qedcc = std::shared_ptr<CC_Cavity>(
          new QED_CCSD_21(qed_ref_wfn, options, includes_));
    } else if (qed_type == "CCSD-22") {
      includes_["t0_1"] = true;
      includes_["t0_2"] = true;
      includes_["t1_1"] = true;
      includes_["t1_2"] = true;
      includes_["t2_1"] = true;
      includes_["t2_2"] = true;
      qedcc = std::shared_ptr<CC_Cavity>(
          new QED_CCSD_22(qed_ref_wfn, options, includes_));
    } else
      throw PsiException("QED_CC_TYPE not recognized. Please choose CCSD-00, "
                         "CCSD-21, or CCSD-22.",
                         __FILE__, __LINE__);

    // compute the energy
    double energy = qedcc->compute_energy();

    // perform QED-EOM-CC if requested
    bool do_eom = options.get_bool("PERFORM_EOM");

    // return the wavefunction if not performing EOM
    if (!do_eom)
      return (std::shared_ptr<Wavefunction>)qedcc;

    std::shared_ptr<EOM_Driver> eom_driver;
    if (options.get_str("EOM_TYPE") ==
        "EE") { // use EOM for excitation energies
      if (qed_type == "CCSD-00") {
        eom_driver =
            std::shared_ptr<EOM_Driver>(new EOM_EE_CCSD(qedcc, options));
      } else if (qed_type == "CCSD-21") {
        eom_driver =
            std::shared_ptr<EOM_Driver>(new EOM_EE_QED_CCSD_21(qedcc, options));
      } else {
        throw PsiException("EOM-EE not implemented for " + qed_type, __FILE__,
                           __LINE__);
      }
    } else if (options.get_str("EOM_TYPE") ==
               "EA") { // use EOM for electron attachment
      if (qed_type == "CCSD-00") {
        eom_driver =
            std::shared_ptr<EOM_Driver>(new EOM_EA_CCSD(qedcc, options));
      } else if (qed_type == "CCSD-21") {
        eom_driver =
            std::shared_ptr<EOM_Driver>(new EOM_EA_QED_CCSD_21(qedcc, options));
      } else {
        throw PsiException("EOM-EA not implemented for " + qed_type, __FILE__,
                           __LINE__);
      }
    } else {
      throw PsiException("EOM_TYPE not recognized. Please choose EE or EA.",
                         __FILE__, __LINE__);
    }

    // compute the excited state energies for the given EOM type
    eom_driver->compute_eom_energy();

    bool build_rdms = options.get_bool("GET_PROPERTIES");
    if (!build_rdms)
      return (std::shared_ptr<Wavefunction>)qedcc;

    // compute the RDMs and oscillator strengths for the given EOM type
    std::shared_ptr<EOM_RDM> rdm;
    Printf("Computing 1-RDMs and oscillator strengths...");

    if (options.get_str("EOM_TYPE") == "EE") {
      if (qed_type == "CCSD-00") {
        rdm = std::shared_ptr<EOM_RDM>(new EOM_EE_RDM(eom_driver, options));
      } else if (qed_type == "CCSD-21") {
        rdm = std::shared_ptr<EOM_RDM>(
            new EOM_EE_QED_RDM_21(eom_driver, options));
      } else {
        throw PsiException("RDM construction not implemented for " + qed_type,
                           __FILE__, __LINE__);
      }
    } else if (options.get_str("EOM_TYPE") == "EA") {
      if (qed_type == "CCSD-00") {
        rdm = std::shared_ptr<EOM_RDM>(new EOM_EA_RDM(eom_driver, options));
      } else if (qed_type == "CCSD-21") {
        rdm = std::shared_ptr<EOM_RDM>(
            new EOM_EA_QED_RDM_21(eom_driver, options));
      } else {
        throw PsiException("RDM construction for EOM_EA not implemented for " +
                               qed_type,
                           __FILE__, __LINE__);
      }
    }

    // build the 1-RDMs
    rdm->compute_eom_1rdm();

    // compute and print the oscillator strengths
    rdm->compute_oscillators();
    rdm->print_oscillators();

    // return the wavefunction
    return (std::shared_ptr<Wavefunction>)qedcc;
#endif
  } else {
    throw PsiException("unknown HILBERT_METHOD", __FILE__, __LINE__);
  }

  return ref_wfn;
}

} // namespace hilbert
