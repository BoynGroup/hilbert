/*
 *  @BEGIN LICENSE
 *
 *  Hilbert: a space for quantum chemistry plugins to Psi4
 *
 *  Copyright (c) 2026 by its authors (LICENSE).
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

#ifndef GPU_ADMM_SOLVER_H
#define GPU_ADMM_SOLVER_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <misc/omp.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdp_solver.h>
#include <string>
#include <vector>

namespace libsdp {

typedef std::function<bool(std::vector<int> &, std::vector<int> &,
                           std::vector<double> &, void *)>
    GPUSparseAFunction;
typedef std::function<bool(std::vector<int> &, std::vector<double> &, void *)>
    GPUMatrixFreeMetadataFunction;

class GPUADMMSolver : public SDPSolver {
  using Clock = std::chrono::steady_clock;

public:
  GPUADMMSolver(long int n_primal, long int n_dual, SDPOptions options)
      : SDPSolver(n_primal, n_dual, options), is_initialized_(false),
        cached_n_primal_(0), cached_n_dual_(0),
        cached_from_direct_builder_(false), build_sparse_A_(nullptr),
        build_matrix_free_metadata_(nullptr), cached_csr_valid_(false),
        cached_a_csr_valid_(false), cached_at_csr_valid_(false),
        matrix_free_(false), matrix_free_validate_(false),
        matrix_free_at_csr_(false), matrix_free_cuda_au_(false),
        matrix_free_cuda_atu_(false), cuda_verbose_(false),
        validate_au_range_(false), cg_dynamic_factor_(0.01),
        cg_fused_normal_(false), shard_primal_(false),
        shard_primal_store_(false), psd_projection_multi_gpu_(true),
        psd_projection_devices_(""), psd_projection_base_memory_fraction_(0.92),
        psd_projection_workspace_scale_(5.0),
        psd_projection_avoid_base_large_blocks_(true),
        validate_sparse_A_(false), profile_timing_(false) {}

  GPUADMMSolver(long int n_primal, long int n_dual, SDPOptions options,
                GPUSparseAFunction build_sparse_A,
                GPUMatrixFreeMetadataFunction build_matrix_free_metadata,
                bool validate_sparse_A = false, bool profile_timing = false,
                bool matrix_free = true, bool matrix_free_validate = false,
                bool matrix_free_at_csr = false,
                bool matrix_free_cuda_au = true,
                bool matrix_free_cuda_atu = true, bool cuda_verbose = false,
                bool validate_au_range = false, double cg_dynamic_factor = 0.01,
                bool cg_fused_normal = true, bool shard_primal = true,
                bool shard_primal_store = true,
                bool psd_projection_multi_gpu = true,
                const std::string &psd_projection_devices = "",
                double psd_projection_base_memory_fraction = 0.92,
                double psd_projection_workspace_scale = 5.0,
                bool psd_projection_avoid_base_large_blocks = true,
                const std::map<std::string, double> &accel_options =
                    std::map<std::string, double>())
      : SDPSolver(n_primal, n_dual, options), is_initialized_(false),
        cached_n_primal_(0), cached_n_dual_(0),
        cached_from_direct_builder_(false), build_sparse_A_(build_sparse_A),
        build_matrix_free_metadata_(build_matrix_free_metadata),
        cached_csr_valid_(false), cached_a_csr_valid_(false),
        cached_at_csr_valid_(false), matrix_free_(matrix_free),
        matrix_free_validate_(matrix_free_validate),
        matrix_free_at_csr_(matrix_free_at_csr),
        matrix_free_cuda_au_(matrix_free_cuda_au),
        matrix_free_cuda_atu_(matrix_free_cuda_atu),
        cuda_verbose_(cuda_verbose), validate_au_range_(validate_au_range),
        cg_dynamic_factor_(cg_dynamic_factor),
        cg_fused_normal_(cg_fused_normal), shard_primal_(shard_primal),
        shard_primal_store_(shard_primal_store),
        psd_projection_multi_gpu_(psd_projection_multi_gpu),
        psd_projection_devices_(psd_projection_devices),
        psd_projection_base_memory_fraction_(
            psd_projection_base_memory_fraction),
        psd_projection_workspace_scale_(psd_projection_workspace_scale),
        psd_projection_avoid_base_large_blocks_(
            psd_projection_avoid_base_large_blocks),
        validate_sparse_A_(validate_sparse_A), profile_timing_(profile_timing),
        accel_options_(accel_options) {}

  ~GPUADMMSolver() {}

  /// Configure convergence handling for the next CASSCF SDP solve. Negative
  /// convergence values restore the solver-wide defaults.
  void configure_casscf_solve(double error_convergence,
                              double objective_convergence,
                              bool hamiltonian_changed,
                              bool entry_diagnostics = true) {
    runtime_error_convergence_ = error_convergence;
    runtime_objective_convergence_ = objective_convergence;
    // Energy stagnation is not a convergence certificate for adaptive
    // CASSCF. Intermediate and final user-accuracy solves must reach the
    // requested gap.
    runtime_options_["require_gap_convergence"] = 1.0;
    runtime_options_["gap_relative"] = 0.0;
    runtime_options_["hamiltonian_changed"] =
        hamiltonian_changed ? 1.0 : 0.0;
    runtime_options_["entry_diagnostics"] = entry_diagnostics ? 1.0 : 0.0;
  }

  void clear_casscf_solve_configuration() {
    runtime_error_convergence_ = -1.0;
    runtime_objective_convergence_ = -1.0;
    runtime_options_.clear();
  }

  void solve(double *x, double *b, double *c, std::vector<int> primal_block_dim,
             int maxiter, SDPCallbackFunction evaluate_Au,
             SDPCallbackFunction evaluate_ATu,
             SDPProgressMonitorFunction progress_monitor, int print_level,
             void *data) override {

    if (print_level > 0) {
      std::cout << "  ==> Initializing GPU-ADMM Solver Interface <=="
                << std::endl;
    }

    // Acquire GIL before calling Python
    pybind11::gil_scoped_acquire acquire;

    try {
      pybind11::object hilbert = pybind11::module_::import("hilbert");
      pybind11::object gpu_admm_solve = hilbert.attr("gpu_admm_solve");

      // Construct the sparse A matrix representation using multi-pass peeling
      // and divide-and-conquer fallback
      std::vector<int> rows;
      std::vector<int> cols;
      std::vector<double> vals;
      long int extraction_atu_calls = 0;
      long int extraction_au_calls = 0;
      const long int n_primal_local = n_primal_;
      const long int n_dual_local = n_dual_;
      auto extraction_start = Clock::now();
      pybind11::object py_matrix_free_meta = pybind11::none();
      bool use_matrix_free_path = false;

      if (matrix_free_) {
        if (!build_matrix_free_metadata_) {
          throw std::runtime_error(
              "GPU_ADMM_MATRIX_FREE was requested, but no matrix-free "
              "metadata callback is available.");
        }
        std::vector<int> matrix_free_int_meta;
        std::vector<double> matrix_free_double_meta;
        if (!build_matrix_free_metadata_(matrix_free_int_meta,
                                         matrix_free_double_meta, data)) {
          if (print_level > 0) {
            std::cout << "  ==> [GPU-ADMM] GPU_ADMM_MATRIX_FREE requested, "
                         "but this constraint layout is not supported by the "
                         "DQG/C1 matrix-free operator. Falling back to the "
                         "direct explicit CSR path."
                      << std::endl;
          }
        } else if (matrix_free_int_meta.size() < 8 ||
                   matrix_free_double_meta.size() < 2) {
          throw std::runtime_error(
              "GPU-ADMM matrix-free metadata callback returned an incomplete "
              "metadata payload.");
        } else {
          constexpr int kDqgSymMatrixFreeMagic = -20260623;
          const bool is_symmetry_payload =
              matrix_free_int_meta[0] == kDqgSymMatrixFreeMagic;
          const bool matrix_free_cuda_requested =
              matrix_free_cuda_au_ || matrix_free_cuda_atu_;
          const bool matrix_free_cuda_supported =
              is_symmetry_payload ? (matrix_free_int_meta[4] != 0 &&
                                     matrix_free_int_meta[5] != 0)
                                  : (matrix_free_int_meta[4] != 0 &&
                                     matrix_free_int_meta[5] != 0);
          if (matrix_free_cuda_requested && !matrix_free_cuda_supported) {
            if (print_level > 0) {
              std::cout
                  << "  ==> [GPU-ADMM] GPU_ADMM_MATRIX_FREE_CUDA_AU/ATU "
                     "requested, but the custom CUDA matrix-free kernels "
                     "currently require the constrain_sz=True, "
                     "constrain_spin=True DQG layout. Falling back to the "
                     "direct explicit CSR path."
                  << std::endl;
            }
          } else if (is_symmetry_payload &&
                     !(matrix_free_cuda_au_ && matrix_free_cuda_atu_)) {
            if (print_level > 0) {
              std::cout
                  << "  ==> [GPU-ADMM] Symmetry-aware matrix-free DQG metadata "
                     "is available, but it requires both custom CUDA Au and "
                     "ATu "
                     "kernels. Falling back to the direct explicit CSR path."
                  << std::endl;
            }
          } else {

            pybind11::dict meta;
            if (is_symmetry_payload) {
              pybind11::list int_payload;
              for (int value : matrix_free_int_meta) {
                int_payload.append(value);
              }
              meta["operator"] = "dqg_sym";
              meta["int_meta"] = int_payload;
            } else {
              meta["operator"] = "dqg_c1";
              meta["amo"] = matrix_free_int_meta[0];
              meta["amopi"] = matrix_free_int_meta[1];
              meta["gab"] = matrix_free_int_meta[2];
              meta["gaa"] = matrix_free_int_meta[3];
              meta["constrain_sz"] = matrix_free_int_meta[4] != 0;
              meta["constrain_spin"] = matrix_free_int_meta[5] != 0;
              meta["constrain_q2"] = matrix_free_int_meta[6] != 0;
              meta["constrain_g2"] = matrix_free_int_meta[7] != 0;
            }
            meta["nalpha_active"] = matrix_free_double_meta[0];
            meta["nbeta_active"] = matrix_free_double_meta[1];
            meta["n_primal"] = n_primal_;
            meta["n_dual"] = n_dual_;
            py_matrix_free_meta = meta;
            use_matrix_free_path = true;

            if (print_level > 0) {
              std::cout << "  ==> [GPU-ADMM] Matrix-free "
                        << (is_symmetry_payload ? "DQG/symmetry" : "DQG/C1")
                        << " metadata ready: ";
              if (is_symmetry_payload) {
                std::cout << "nirrep=" << matrix_free_int_meta[2]
                          << ", amo=" << matrix_free_int_meta[3];
              } else {
                std::cout << "amo=" << matrix_free_int_meta[0]
                          << ", gab=" << matrix_free_int_meta[2]
                          << ", gaa=" << matrix_free_int_meta[3];
              }
              std::cout
                  << ", spin_layout="
                  << (std::abs(matrix_free_double_meta[0] -
                               matrix_free_double_meta[1]) < 1.0e-12
                          ? "singlet"
                          : "open-shell")
                  << (matrix_free_validate_
                          ? ", explicit CSR will also be built for "
                            "validation."
                      : matrix_free_at_csr_
                          ? ", explicit A^T CSR will also be built for the "
                            "hybrid matvec path."
                          : ", skipping explicit A/CSR build.")
                  << std::endl;
            }
          }
        }
      }

      const bool need_explicit_A =
          !use_matrix_free_path || matrix_free_validate_ || matrix_free_at_csr_;
      const bool need_a_csr =
          need_explicit_A && (!use_matrix_free_path || matrix_free_validate_);
      const bool need_at_csr = need_explicit_A;
      size_t sparse_nnz_report = 0;

      bool cache_valid = false;
      if (need_explicit_A && is_initialized_ && cached_n_primal_ == n_primal_ &&
          cached_n_dual_ == n_dual_) {
        if (!need_a_csr && need_at_csr && cached_at_csr_valid_) {
          cache_valid = true;
          sparse_nnz_report = cached_at_vals_.size();
          if (print_level > 0) {
            std::cout << "  ==> [GPU-ADMM] Found cached direct CSR(A^T). "
                         "Skipping sparse A materialization."
                      << std::endl;
          }
        } else if (!validate_sparse_A_ && cached_from_direct_builder_ &&
                   !cached_vals_.empty()) {
          cache_valid = true;
          if (print_level > 0) {
            std::cout << "  ==> [GPU-ADMM] Found cached direct A matrix. "
                         "Skipping validation."
                      << std::endl;
          }
          rows = cached_rows_;
          cols = cached_cols_;
          vals = cached_vals_;
          sparse_nnz_report = vals.size();
        } else {
          if (print_level > 0) {
            std::cout
                << "  ==> [Debug] Found cached A matrix. Verifying validity..."
                << std::endl;
          }
          // Run randomized check
          std::vector<double> u_rand(n_dual_);
          unsigned long temp_rand = 123456789;
          for (long int i = 0; i < n_dual_; ++i) {
            temp_rand = temp_rand * 1103515245 + 12345;
            u_rand[i] = 0.5 + 1.0 * ((double)(temp_rand % 10000) / 10000.0);
          }

          std::vector<double> ATu_rand_ref(n_primal_, 0.0);
          evaluate_ATu(ATu_rand_ref.data(), u_rand.data(), data);
          extraction_atu_calls++;

          cache_valid = true;
          std::vector<double> ATu_cached_ref;
          if (cached_decoded_.size() != static_cast<size_t>(n_primal_)) {
            ATu_cached_ref.assign(n_primal_, 0.0);
            for (size_t k = 0; k < cached_rows_.size(); ++k) {
              ATu_cached_ref[cached_cols_[k]] +=
                  cached_vals_[k] * u_rand[cached_rows_[k]];
            }
          }
#pragma omp parallel
          {
            bool local_valid = true;
#pragma omp for schedule(static)
            for (long int j = 0; j < n_primal_local; ++j) {
              if (!local_valid)
                continue;
              double val_cached = 0.0;
              if (cached_decoded_.size() == static_cast<size_t>(n_primal_)) {
                for (const auto &item : cached_decoded_[j]) {
                  val_cached += item.second * u_rand[item.first];
                }
              } else {
                val_cached = ATu_cached_ref[j];
              }
              if (std::abs(ATu_rand_ref[j] - val_cached) > 1e-8) {
                local_valid = false;
              }
            }
            if (!local_valid) {
#pragma omp critical
              cache_valid = false;
            }
          }

          if (cache_valid) {
            if (print_level > 0) {
              std::cout << "  ==> [Debug] Cached A matrix is VALID. Skipping "
                           "extraction."
                        << std::endl;
            }
            rows = cached_rows_;
            cols = cached_cols_;
            vals = cached_vals_;
          } else {
            if (print_level > 0) {
              std::cout << "  ==> [Debug] Cached A matrix is INVALID. "
                           "Re-extracting..."
                        << std::endl;
            }
          }
        }
      }

      if (need_explicit_A && !cache_valid && build_sparse_A_) {
        auto direct_start = Clock::now();
        if (print_level > 0) {
          std::cout << "  ==> [GPU-ADMM] Trying direct sparse A builder..."
                    << std::endl;
        }
        if (build_sparse_A_(rows, cols, vals, data)) {
          cache_valid = true;
          is_initialized_ = true;
          cached_n_primal_ = n_primal_;
          cached_n_dual_ = n_dual_;
          cached_rows_ = rows;
          cached_cols_ = cols;
          cached_vals_ = vals;
          sparse_nnz_report = vals.size();
          cached_decoded_.clear();
          cached_from_direct_builder_ = true;
          cached_csr_valid_ = false;
          cached_a_csr_valid_ = false;
          cached_at_csr_valid_ = false;
          if (print_level > 0) {
            std::cout << "  ==> [GPU-ADMM] Direct sparse A builder succeeded "
                      << "in " << elapsed_seconds(direct_start) << " s."
                      << std::endl;
          }
        } else if (print_level > 0) {
          std::cout << "  ==> [GPU-ADMM] Direct sparse A builder unavailable "
                       "for this problem; falling back to callback extraction."
                    << std::endl;
        }
      }

      if (need_explicit_A && !cache_valid) {
        auto peel_start = Clock::now();
        std::vector<std::vector<std::pair<int, double>>> decoded(n_primal_);
        std::vector<long int> primes = {311, 313, 317, 331, 337};
        std::vector<double> u1(n_dual_, 0.0);
        std::vector<double> u2(n_dual_, 0.0);
        std::vector<double> A1(n_primal_, 0.0);
        std::vector<double> A2(n_primal_, 0.0);

        if (print_level > 0) {
          std::cout << "  ==> [GPU-ADMM] Extracting sparse A with hashed ATu "
                       "peeling..."
                    << std::endl;
        }

        for (long int p : primes) {
          for (long int b = 0; b < p; ++b) {
            std::fill(u1.begin(), u1.end(), 0.0);
            std::fill(u2.begin(), u2.end(), 0.0);
            bool has_elements = false;
            for (long int i = b; i < n_dual_; i += p) {
              u1[i] = 1.0;
              u2[i] = (double)(i + 1);
              has_elements = true;
            }
            if (!has_elements)
              continue;

            evaluate_ATu(A1.data(), u1.data(), data);
            evaluate_ATu(A2.data(), u2.data(), data);
            extraction_atu_calls += 2;

#pragma omp parallel for schedule(static)
            for (long int j = 0; j < n_primal_local; ++j) {
              double a1_rem = A1[j];
              double a2_rem = A2[j];

              // Subtract contributions of already decoded constraints for this
              // column (safe since different threads modify different
              // decoded[j] vectors)
              for (const auto &item : decoded[j]) {
                long int i_dec = item.first;
                double v_dec = item.second;
                if (i_dec % p == b) {
                  a1_rem -= v_dec;
                  a2_rem -= v_dec * (i_dec + 1);
                }
              }

              if (std::abs(a1_rem) > 1e-15) {
                double ratio = a2_rem / a1_rem;
                long int I = std::round(ratio);
                if (std::abs(ratio - I) < 1e-11 && I >= 1 && I <= n_dual_ &&
                    ((I - 1) % p == b)) {
                  long int decoded_i = I - 1;
                  double decoded_val = a1_rem;

                  append_decoded_entry(decoded[j], static_cast<int>(decoded_i),
                                       decoded_val);
                }
              }
            }
          }
        }

        // Validation check using a random probe vector to ensure 100%
        // completeness
        std::vector<double> u_rand(n_dual_);
        unsigned long temp_rand = 1337;
        for (long int i = 0; i < n_dual_; ++i) {
          temp_rand = temp_rand * 1103515245 + 12345;
          u_rand[i] = 0.5 + 1.0 * ((double)(temp_rand % 10000) / 10000.0);
        }

        std::vector<double> ATu_rand_ref(n_primal_, 0.0);
        evaluate_ATu(ATu_rand_ref.data(), u_rand.data(), data);
        extraction_atu_calls++;

        std::vector<int> unresolved_cols =
            unresolved_columns_from_ref(decoded, u_rand, ATu_rand_ref, 1e-6);

        if (!unresolved_cols.empty()) {
          if (print_level > 0) {
            std::cout << "  ==> [Debug] Peeling left " << unresolved_cols.size()
                      << " columns unresolved after "
                      << elapsed_seconds(peel_start)
                      << " s. Using batched hashed Au fallback." << std::endl;
          }
          // Clear partial decodings. The fallback recovers complete columns for
          // these indices, avoiding stale false-positive hash decodes.
          for (int j : unresolved_cols) {
            decoded[j].clear();
          }

          const size_t max_batched_unresolved = 5000;
          if (unresolved_cols.size() <= max_batched_unresolved) {
            batched_Au_extract_unresolved(decoded, unresolved_cols, evaluate_Au,
                                          data, u_rand, ATu_rand_ref,
                                          extraction_au_calls, print_level);
          } else if (print_level > 0) {
            std::cout << "  ==> [Debug] Skipping batched Au fallback for "
                      << unresolved_cols.size()
                      << " unresolved columns; going straight to parallel "
                         "one-column fallback."
                      << std::endl;
          }

          unresolved_cols =
              unresolved_columns_from_ref(decoded, u_rand, ATu_rand_ref, 1e-6);

          if (!unresolved_cols.empty()) {
            if (print_level > 0) {
              std::cout << "  ==> [Debug] Batched Au fallback left "
                        << unresolved_cols.size()
                        << " columns unresolved. Using final parallel "
                           "one-column fallback."
                        << std::endl;
            }

            parallel_column_fallback(decoded, unresolved_cols, evaluate_Au,
                                     data, extraction_au_calls, print_level);
          }

          if (print_level > 0) {
            std::vector<int> final_unresolved = unresolved_columns_from_ref(
                decoded, u_rand, ATu_rand_ref, 1e-6);
            std::cout << "  ==> [Debug] Fallback extraction complete; "
                      << final_unresolved.size()
                      << " columns unresolved by random check." << std::endl;
          }
        }

        // Populate rows, cols, vals from decoded entries
        size_t nnz_estimate = decoded_entry_count(decoded);
        rows.reserve(nnz_estimate);
        cols.reserve(nnz_estimate);
        vals.reserve(nnz_estimate);
        for (long int j = 0; j < n_primal_; ++j) {
          for (const auto &item : decoded[j]) {
            rows.push_back(item.first);
            cols.push_back(static_cast<int>(j));
            vals.push_back(item.second);
          }
        }

        // Cache the extracted sparse matrix A
        is_initialized_ = true;
        cached_n_primal_ = n_primal_;
        cached_n_dual_ = n_dual_;
        cached_rows_ = rows;
        cached_cols_ = cols;
        cached_vals_ = vals;
        cached_decoded_ = decoded;
        sparse_nnz_report = vals.size();
        cached_from_direct_builder_ = false;
        cached_csr_valid_ = false;
        cached_a_csr_valid_ = false;
        cached_at_csr_valid_ = false;
      }

      if (need_explicit_A && ((need_a_csr && !cached_a_csr_valid_) ||
                              (need_at_csr && !cached_at_csr_valid_))) {
        if (need_a_csr) {
          build_cached_csr_from_coo(rows, cols, vals, print_level);
        } else {
          build_cached_at_csr_from_coo(rows, cols, vals, print_level);
          rows.clear();
          cols.clear();
          vals.clear();
          cached_rows_.clear();
          cached_cols_.clear();
          cached_vals_.clear();
          cached_decoded_.clear();
        }
      } else if (need_explicit_A && print_level > 0) {
        if (need_a_csr) {
          std::cout << "  ==> [GPU-ADMM] Reusing cached direct CSR(A,A^T), "
                    << "estimated GPU CSR int32(A+A^T)="
                    << estimate_csr_storage_gib(cached_a_vals_.size(),
                                                sizeof(int))
                    << " GiB" << std::endl;
        } else {
          std::cout << "  ==> [GPU-ADMM] Reusing cached direct CSR(A^T), "
                    << "estimated GPU CSR int32(A^T)="
                    << estimate_single_csr_storage_gib(cached_at_vals_.size(),
                                                       n_primal_, sizeof(int))
                    << " GiB" << std::endl;
        }
      }

      if (!need_explicit_A) {
        is_initialized_ = true;
        cached_n_primal_ = n_primal_;
        cached_n_dual_ = n_dual_;
        cached_rows_.clear();
        cached_cols_.clear();
        cached_vals_.clear();
        cached_decoded_.clear();
        cached_a_crow_.clear();
        cached_a_col_.clear();
        cached_a_vals_.clear();
        cached_at_crow_.clear();
        cached_at_col_.clear();
        cached_at_vals_.clear();
        cached_from_direct_builder_ = false;
        cached_csr_valid_ = false;
        cached_a_csr_valid_ = false;
        cached_at_csr_valid_ = false;
      }

      if (need_explicit_A && print_level > 0) {
        if (sparse_nnz_report == 0 && cached_at_csr_valid_) {
          sparse_nnz_report = cached_at_vals_.size();
        }
        if (sparse_nnz_report == 0 && !rows.empty()) {
          sparse_nnz_report = rows.size();
        }
        const double avg_row_nnz =
            n_dual_ > 0 ? static_cast<double>(sparse_nnz_report) / n_dual_
                        : 0.0;
        std::cout << "  ==> [GPU-ADMM] Sparse A ready: nnz="
                  << sparse_nnz_report << ", avg nnz/constraint=" << avg_row_nnz
                  << ", extraction/check callbacks ATu=" << extraction_atu_calls
                  << ", Au=" << extraction_au_calls
                  << ", elapsed=" << elapsed_seconds(extraction_start) << " s";
        if (need_a_csr) {
          std::cout << ", estimated final GPU CSR int32(A+A^T)="
                    << estimate_csr_storage_gib(rows.size(), sizeof(int))
                    << " GiB" << std::endl;
        } else {
          std::cout << ", estimated final GPU CSR int32(A^T)="
                    << estimate_single_csr_storage_gib(cached_at_vals_.size(),
                                                       n_primal_, sizeof(int))
                    << " GiB" << std::endl;
        }
      }

      // Verify extracted matrix A against evaluate_Au and evaluate_ATu
      if (need_explicit_A && validate_sparse_A_) {
        std::vector<double> test_x(n_primal_);
        std::vector<double> test_y(n_dual_);
        unsigned long r_state = 42;
        for (long int j = 0; j < n_primal_; ++j) {
          r_state = r_state * 1103515245 + 12345;
          test_x[j] = -1.0 + 2.0 * ((double)(r_state % 10000) / 10000.0);
        }
        for (long int i = 0; i < n_dual_; ++i) {
          r_state = r_state * 1103515245 + 12345;
          test_y[i] = -1.0 + 2.0 * ((double)(r_state % 10000) / 10000.0);
        }

        // 1. Check if C++ callbacks are transpose adjoints
        std::vector<double> Ax(n_dual_, 0.0);
        std::vector<double> ATy(n_primal_, 0.0);
        evaluate_Au(Ax.data(), test_x.data(), data);
        evaluate_ATu(ATy.data(), test_y.data(), data);

        double dot1 = 0.0;
        for (long int i = 0; i < n_dual_; ++i)
          dot1 += Ax[i] * test_y[i];

        double dot2 = 0.0;
        for (long int j = 0; j < n_primal_; ++j)
          dot2 += test_x[j] * ATy[j];

        std::cout << "  ==> [Debug C++] Callback Adjoint Check: dot1 = " << dot1
                  << ", dot2 = " << dot2 << ", diff = " << std::abs(dot1 - dot2)
                  << std::endl;

        // 2. Check if extracted matrix A matches evaluate_Au
        std::vector<double> Ax_extracted(n_dual_, 0.0);
        for (size_t k = 0; k < rows.size(); ++k) {
          Ax_extracted[rows[k]] += vals[k] * test_x[cols[k]];
        }
        double diff_Ax = 0.0;
        for (long int i = 0; i < n_dual_; ++i) {
          diff_Ax += std::abs(Ax[i] - Ax_extracted[i]);
        }
        std::cout << "  ==> [Debug C++] Extracted A vs evaluate_Au: sum diff = "
                  << diff_Ax << std::endl;

        // 3. Check if extracted matrix A matches evaluate_ATu
        std::vector<double> ATy_extracted(n_primal_, 0.0);
        for (size_t k = 0; k < rows.size(); ++k) {
          ATy_extracted[cols[k]] += vals[k] * test_y[rows[k]];
        }
        double diff_ATy = 0.0;
        int mismatch_count = 0;
        for (long int j = 0; j < n_primal_; ++j) {
          double diff = std::abs(ATy[j] - ATy_extracted[j]);
          diff_ATy += diff;
          if (diff > 1e-6) {
            mismatch_count++;
            if (mismatch_count <= 10) {
              std::cout << "  ==> [Debug C++] Column " << j
                        << " mismatch: true=" << ATy[j]
                        << ", extracted=" << ATy_extracted[j]
                        << ", diff=" << diff << std::endl;
            }
          }
        }
        std::cout
            << "  ==> [Debug C++] Extracted A^T vs evaluate_ATu: sum diff = "
            << diff_ATy << ", mismatch count = " << mismatch_count << std::endl;
      }

      // CSR handoff is the production path. Keep the legacy COO arguments empty
      // so Python cannot accidentally allocate the expensive COO/coalesce path.
      std::vector<int> empty_int;
      std::vector<double> empty_double;
      pybind11::array_t<int> py_rows(empty_int.size(), empty_int.data());
      pybind11::array_t<int> py_cols(empty_int.size(), empty_int.data());
      pybind11::array_t<double> py_vals(empty_double.size(),
                                        empty_double.data());

      pybind11::array_t<int> py_a_crow(cached_a_crow_.size(),
                                       cached_a_crow_.data());
      pybind11::array_t<int> py_a_col(cached_a_col_.size(),
                                      cached_a_col_.data());
      pybind11::array_t<double> py_a_vals(cached_a_vals_.size(),
                                          cached_a_vals_.data());
      pybind11::array_t<int> py_at_crow(cached_at_crow_.size(),
                                        cached_at_crow_.data());
      pybind11::array_t<int> py_at_col(cached_at_col_.size(),
                                       cached_at_col_.data());
      pybind11::array_t<double> py_at_vals(cached_at_vals_.size(),
                                           cached_at_vals_.data());

      std::function<void(int, int, int, double, double, double, double, double)>
          py_progress_monitor =
              [progress_monitor,
               data](int pr_lvl, int o_it, int i_it, double p_obj, double d_obj,
                     double mu_val, double p_err, double d_err) {
                if (progress_monitor) {
                  progress_monitor(pr_lvl, o_it, i_it, p_obj, d_obj, mu_val,
                                   p_err, d_err, data);
                }
              };

      if (print_level > 1) {
        std::cout << "  ==> [Debug] n_primal_ = " << n_primal_
                  << ", n_dual_ = " << n_dual_ << std::endl;
        std::cout << "  ==> [Debug] pointers: c=" << c << ", b=" << b
                  << ", x=" << x << ", y_=" << y_ << ", z_=" << z_ << std::endl;
      }

      // Prepare views/copies of inputs as NumPy arrays
      if (print_level > 1) {
        std::cout << "  ==> [Debug] Creating py_c" << std::endl;
      }
      pybind11::array_t<double> py_c(n_primal_, c);
      if (print_level > 1) {
        std::cout << "  ==> [Debug] Creating py_b" << std::endl;
      }
      pybind11::array_t<double> py_b(n_dual_, b);
      if (print_level > 1) {
        std::cout << "  ==> [Debug] Creating py_x" << std::endl;
      }
      pybind11::array_t<double> py_x(n_primal_, x);
      if (print_level > 1) {
        std::cout << "  ==> [Debug] Creating py_y" << std::endl;
      }
      pybind11::array_t<double> py_y(n_dual_, y_);
      if (print_level > 1) {
        std::cout << "  ==> [Debug] Creating py_z" << std::endl;
      }
      pybind11::array_t<double> py_z(n_primal_, z_);
      if (print_level > 1) {
        std::cout << "  ==> [Debug] All NumPy arrays created successfully"
                  << std::endl;
      }

      // Forward the convergence-acceleration options as a flat dict of numbers.
      pybind11::dict py_accel_options;
      for (const auto &kv : accel_options_) {
        py_accel_options[pybind11::str(kv.first)] = kv.second;
      }
      for (const auto &kv : runtime_options_) {
        py_accel_options[pybind11::str(kv.first)] = kv.second;
      }

      const double solve_error_convergence =
          runtime_error_convergence_ > 0.0
              ? runtime_error_convergence_
              : options_.sdp_error_convergence;
      const double solve_objective_convergence =
          runtime_objective_convergence_ > 0.0
              ? runtime_objective_convergence_
              : options_.sdp_objective_convergence;

      // Call PyTorch-based ADMM solver in python
      pybind11::tuple result = gpu_admm_solve(
          py_c, py_b, primal_block_dim, py_rows, py_cols, py_vals,
          py_progress_monitor, py_x, py_y, py_z, mu_, maxiter,
          solve_error_convergence, solve_objective_convergence,
          options_.cg_maxiter, options_.cg_convergence,
          options_.dynamic_cg_convergence, options_.mu_update_frequency,
          print_level, oiter_, iiter_total_, profile_timing_, py_a_crow,
          py_a_col, py_a_vals, py_at_crow, py_at_col, py_at_vals,
          py_matrix_free_meta, use_matrix_free_path,
          use_matrix_free_path && matrix_free_validate_,
          use_matrix_free_path && matrix_free_at_csr_,
          use_matrix_free_path && matrix_free_cuda_au_,
          use_matrix_free_path && matrix_free_cuda_atu_, cuda_verbose_,
          validate_au_range_, cg_dynamic_factor_, cg_fused_normal_,
          shard_primal_, shard_primal_store_, psd_projection_multi_gpu_,
          psd_projection_devices_, psd_projection_base_memory_fraction_,
          psd_projection_workspace_scale_,
          psd_projection_avoid_base_large_blocks_, py_accel_options);

      // Extract results from python
      pybind11::array_t<double> py_x_opt =
          result[0].cast<pybind11::array_t<double>>();
      pybind11::array_t<double> py_y_opt =
          result[1].cast<pybind11::array_t<double>>();
      pybind11::array_t<double> py_z_opt =
          result[2].cast<pybind11::array_t<double>>();
      double mu_opt = result[3].cast<double>();
      bool converged = result[4].cast<bool>();
      int oiter_opt = result[5].cast<int>();
      int iiter_opt = result[6].cast<int>();

      // Update C++ pointers with optimal values. The Python GPU-ADMM path
      // usually writes directly into these host buffers to avoid duplicating
      // huge final x/y/z arrays, so skip the copy when the returned NumPy array
      // aliases the destination.
      if (py_x_opt.data() != x) {
        std::memcpy(x, py_x_opt.data(), n_primal_ * sizeof(double));
      }
      if (py_y_opt.data() != y_) {
        std::memcpy(y_, py_y_opt.data(), n_dual_ * sizeof(double));
      }
      if (py_z_opt.data() != z_) {
        std::memcpy(z_, py_z_opt.data(), n_primal_ * sizeof(double));
      }

      mu_ = mu_opt;
      is_converged_ = converged;
      oiter_ = oiter_opt;
      iiter_total_ = iiter_opt;

      // Recalculate final error metrics to make sure C++ members are synced
      // Primal error: ||Ax - b||
      evaluate_Au(Au_, x, data);
      double primal_err_sq = 0.0;
      for (long int i = 0; i < n_dual_; ++i) {
        double diff = Au_[i] - b[i];
        primal_err_sq += diff * diff;
      }
      primal_error_ = std::sqrt(primal_err_sq);

      // Dual error: ||A^T y + z - c||
      evaluate_ATu(ATu_, y_, data);
      double dual_err_sq = 0.0;
      for (long int i = 0; i < n_primal_; ++i) {
        double diff = ATu_[i] + z_[i] - c[i];
        dual_err_sq += diff * diff;
      }
      dual_error_ = std::sqrt(dual_err_sq);

    } catch (const std::exception &ex) {
      std::cerr << "Error in GPUADMMSolver execution: " << ex.what()
                << std::endl;
      is_converged_ = false;
      throw;
    }
  }

private:
  static double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  static bool append_decoded_entry(std::vector<std::pair<int, double>> &entries,
                                   int row, double value) {
    for (const auto &entry : entries) {
      if (entry.first == row) {
        return false;
      }
    }
    entries.push_back({row, value});
    return true;
  }

  static size_t decoded_entry_count(
      const std::vector<std::vector<std::pair<int, double>>> &decoded) {
    size_t count = 0;
    for (const auto &column : decoded) {
      count += column.size();
    }
    return count;
  }

  static void build_csr_from_coo(int n_rows, const std::vector<int> &coo_rows,
                                 const std::vector<int> &coo_cols,
                                 const std::vector<double> &coo_vals,
                                 std::vector<int> &crow, std::vector<int> &col,
                                 std::vector<double> &val) {
    if (coo_rows.size() != coo_cols.size() ||
        coo_rows.size() != coo_vals.size()) {
      throw std::runtime_error("GPU-ADMM CSR build received inconsistent COO "
                               "array lengths.");
    }
    if (n_rows < 0 ||
        static_cast<size_t>(n_rows) + 1 >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        coo_rows.size() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("GPU-ADMM direct CSR handoff requires int32 "
                               "row pointers; problem is too large for this "
                               "explicit-A path.");
    }

    crow.assign(static_cast<size_t>(n_rows) + 1, 0);
    col.resize(coo_rows.size());
    val.resize(coo_rows.size());

    for (size_t k = 0; k < coo_rows.size(); ++k) {
      int row = coo_rows[k];
      if (row < 0 || row >= n_rows) {
        throw std::runtime_error("GPU-ADMM CSR build encountered an out-of-"
                                 "range COO row.");
      }
      crow[static_cast<size_t>(row) + 1]++;
    }

    for (int row = 0; row < n_rows; ++row) {
      crow[static_cast<size_t>(row) + 1] += crow[static_cast<size_t>(row)];
    }

    std::vector<int> next = crow;
    for (size_t k = 0; k < coo_rows.size(); ++k) {
      int row = coo_rows[k];
      int dest = next[static_cast<size_t>(row)]++;
      col[static_cast<size_t>(dest)] = coo_cols[k];
      val[static_cast<size_t>(dest)] = coo_vals[k];
    }
  }

  void build_cached_csr_from_coo(const std::vector<int> &rows,
                                 const std::vector<int> &cols,
                                 const std::vector<double> &vals,
                                 int print_level) {
    if (n_dual_ > std::numeric_limits<int>::max() ||
        n_primal_ > std::numeric_limits<int>::max()) {
      throw std::runtime_error("GPU-ADMM direct CSR handoff requires int32 "
                               "matrix dimensions; problem is too large for "
                               "this explicit-A path.");
    }
    auto csr_start = Clock::now();
    build_csr_from_coo(static_cast<int>(n_dual_), rows, cols, vals,
                       cached_a_crow_, cached_a_col_, cached_a_vals_);
    build_csr_from_coo(static_cast<int>(n_primal_), cols, rows, vals,
                       cached_at_crow_, cached_at_col_, cached_at_vals_);
    cached_csr_valid_ = true;
    cached_a_csr_valid_ = true;
    cached_at_csr_valid_ = true;
    if (print_level > 0) {
      std::cout << "  ==> [GPU-ADMM] Built cached direct CSR(A,A^T) in "
                << elapsed_seconds(csr_start)
                << " s, estimated GPU CSR int32(A+A^T)="
                << estimate_csr_storage_gib(rows.size(), sizeof(int)) << " GiB"
                << std::endl;
    }
  }

  void build_cached_at_csr_from_coo(const std::vector<int> &rows,
                                    const std::vector<int> &cols,
                                    const std::vector<double> &vals,
                                    int print_level) {
    if (n_primal_ > std::numeric_limits<int>::max()) {
      throw std::runtime_error("GPU-ADMM direct CSR(A^T) handoff requires "
                               "int32 matrix dimensions; problem is too "
                               "large for this explicit-A^T path.");
    }
    auto csr_start = Clock::now();
    cached_a_crow_.clear();
    cached_a_col_.clear();
    cached_a_vals_.clear();
    build_csr_from_coo(static_cast<int>(n_primal_), cols, rows, vals,
                       cached_at_crow_, cached_at_col_, cached_at_vals_);
    cached_csr_valid_ = true;
    cached_a_csr_valid_ = false;
    cached_at_csr_valid_ = true;
    if (print_level > 0) {
      std::cout << "  ==> [GPU-ADMM] Built cached direct CSR(A^T) in "
                << elapsed_seconds(csr_start)
                << " s, estimated GPU CSR int32(A^T)="
                << estimate_single_csr_storage_gib(rows.size(), n_primal_,
                                                   sizeof(int))
                << " GiB" << std::endl;
    }
  }

  std::vector<int> unresolved_columns_from_ref(
      const std::vector<std::vector<std::pair<int, double>>> &decoded,
      const std::vector<double> &u_rand,
      const std::vector<double> &ATu_rand_ref, double tolerance) const {
    std::vector<int> unresolved_cols;
    const long int n_primal_local = n_primal_;
#pragma omp parallel
    {
      std::vector<int> local_unresolved;
#pragma omp for schedule(static)
      for (long int j = 0; j < n_primal_local; ++j) {
        double val_decoded = 0.0;
        for (const auto &item : decoded[j]) {
          val_decoded += item.second * u_rand[item.first];
        }
        if (std::abs(ATu_rand_ref[j] - val_decoded) > tolerance) {
          local_unresolved.push_back(static_cast<int>(j));
        }
      }
#pragma omp critical
      {
        unresolved_cols.insert(unresolved_cols.end(), local_unresolved.begin(),
                               local_unresolved.end());
      }
    }
    return unresolved_cols;
  }

  void batched_Au_extract_unresolved(
      std::vector<std::vector<std::pair<int, double>>> &decoded,
      std::vector<int> &unresolved_cols, SDPCallbackFunction evaluate_Au,
      void *data, const std::vector<double> &u_rand,
      const std::vector<double> &ATu_rand_ref, long int &extraction_au_calls,
      int print_level) const {

    const std::vector<long int> primes = {257, 263, 269, 271,
                                          277, 281, 283, 293};
    std::vector<double> x1(n_primal_, 0.0);
    std::vector<double> x2(n_primal_, 0.0);
    std::vector<double> Ax1(n_dual_, 0.0);
    std::vector<double> Ax2(n_dual_, 0.0);
    std::vector<int> touched_cols;
    const long int n_dual_local = n_dual_;
    std::vector<std::vector<std::pair<int, std::pair<int, double>>>>
        thread_additions;
#ifdef _OPENMP
    thread_additions.resize(omp_get_max_threads());
#else
    thread_additions.resize(1);
#endif
    auto fallback_start = Clock::now();

    for (long int p : primes) {
      if (unresolved_cols.empty()) {
        break;
      }
      const size_t unresolved_before = unresolved_cols.size();

      std::vector<char> is_unresolved(n_primal_, 0);
      for (int j : unresolved_cols) {
        is_unresolved[j] = 1;
      }

      for (long int b = 0; b < p; ++b) {
        touched_cols.clear();
        for (int j : unresolved_cols) {
          if (j % p == b) {
            x1[j] = 1.0;
            x2[j] = static_cast<double>(j) + 1.0;
            touched_cols.push_back(j);
          }
        }
        if (touched_cols.empty()) {
          continue;
        }

        evaluate_Au(Ax1.data(), x1.data(), data);
        evaluate_Au(Ax2.data(), x2.data(), data);
        extraction_au_calls += 2;

        for (int j : touched_cols) {
          for (const auto &item : decoded[j]) {
            Ax1[item.first] -= item.second;
            Ax2[item.first] -= item.second * (static_cast<double>(j) + 1.0);
          }
        }

        for (auto &local_additions : thread_additions) {
          local_additions.clear();
        }

#pragma omp parallel
        {
#ifdef _OPENMP
          int tid = omp_get_thread_num();
#else
          int tid = 0;
#endif
          auto &local_additions = thread_additions[tid];
#pragma omp for schedule(static)
          for (long int i = 0; i < n_dual_local; ++i) {
            double a1 = Ax1[i];
            if (std::abs(a1) <= 1e-15) {
              continue;
            }
            double ratio = Ax2[i] / a1;
            long int J = std::lround(ratio);
            if (std::abs(ratio - static_cast<double>(J)) < 1e-11 && J >= 1 &&
                J <= n_primal_) {
              int decoded_j = static_cast<int>(J - 1);
              if (is_unresolved[decoded_j] && decoded_j % p == b) {
                local_additions.push_back(
                    {decoded_j, {static_cast<int>(i), a1}});
              }
            }
          }
        }

        for (const auto &local_additions : thread_additions) {
          for (const auto &entry : local_additions) {
            append_decoded_entry(decoded[entry.first], entry.second.first,
                                 entry.second.second);
          }
        }

        for (int j : touched_cols) {
          x1[j] = 0.0;
          x2[j] = 0.0;
        }
      }

      unresolved_cols =
          unresolved_columns_from_ref(decoded, u_rand, ATu_rand_ref, 1e-6);
      if (print_level > 0) {
        std::cout << "  ==> [Debug] Batched Au hash p=" << p
                  << " complete; unresolved columns=" << unresolved_cols.size()
                  << ", elapsed=" << elapsed_seconds(fallback_start) << " s"
                  << std::endl;
      }
      if (unresolved_cols.size() >= unresolved_before) {
        if (print_level > 0) {
          std::cout
              << "  ==> [Debug] Batched Au hash made no progress; aborting "
                 "batched fallback."
              << std::endl;
        }
        break;
      }
    }
  }

  void parallel_column_fallback(
      std::vector<std::vector<std::pair<int, double>>> &decoded,
      const std::vector<int> &unresolved_cols, SDPCallbackFunction evaluate_Au,
      void *data, long int &extraction_au_calls, int print_level) const {

    int cols_done = 0;
    int print_interval =
        std::max(1, static_cast<int>(unresolved_cols.size()) / 10);
    int last_printed = 0;

    if (print_level > 0) {
#ifdef _OPENMP
      std::cout << "  ==> [Debug] Parallel column fallback using up to "
                << omp_get_max_threads() << " OpenMP threads." << std::endl;
#else
      std::cout << "  ==> [Debug] Parallel column fallback running without "
                   "OpenMP."
                << std::endl;
#endif
    }

#pragma omp parallel
    {
      std::vector<double> x_probe(n_primal_, 0.0);
      std::vector<double> Ax_result(n_dual_, 0.0);

#pragma omp for schedule(dynamic)
      for (size_t k = 0; k < unresolved_cols.size(); ++k) {
        int j = unresolved_cols[k];
        decoded[j].clear();
        x_probe[j] = 1.0;
        evaluate_Au(Ax_result.data(), x_probe.data(), data);
        x_probe[j] = 0.0;

        for (long int i = 0; i < n_dual_; ++i) {
          if (std::abs(Ax_result[i]) > 1e-15) {
            decoded[j].push_back({static_cast<int>(i), Ax_result[i]});
          }
          Ax_result[i] = 0.0;
        }

#pragma omp atomic
        extraction_au_calls++;

#pragma omp atomic
        cols_done++;

        if (print_level > 0) {
          int current_done;
#pragma omp atomic read
          current_done = cols_done;
          if (current_done % print_interval == 0 ||
              current_done == static_cast<int>(unresolved_cols.size())) {
#pragma omp critical(progress_print)
            {
              if (current_done > last_printed) {
                std::cout << "  ==> [Debug] Column fallback progress: "
                          << current_done << "/" << unresolved_cols.size()
                          << std::endl;
                last_printed = current_done;
              }
            }
          }
        }
      }
    }
  }

  double estimate_csr_storage_gib(size_t nnz, size_t index_bytes) const {
    long double bytes = 2.0L * static_cast<long double>(nnz) *
                            (static_cast<long double>(sizeof(double)) +
                             static_cast<long double>(index_bytes)) +
                        static_cast<long double>(n_primal_ + n_dual_ + 2) *
                            static_cast<long double>(index_bytes);
    return static_cast<double>(bytes / (1024.0L * 1024.0L * 1024.0L));
  }

  double estimate_single_csr_storage_gib(size_t nnz, long int n_rows,
                                         size_t index_bytes) const {
    long double bytes = static_cast<long double>(nnz) *
                            (static_cast<long double>(sizeof(double)) +
                             static_cast<long double>(index_bytes)) +
                        static_cast<long double>(n_rows + 1) *
                            static_cast<long double>(index_bytes);
    return static_cast<double>(bytes / (1024.0L * 1024.0L * 1024.0L));
  }

  bool is_initialized_;
  long int cached_n_primal_;
  long int cached_n_dual_;
  bool cached_from_direct_builder_;
  std::vector<int> cached_rows_;
  std::vector<int> cached_cols_;
  std::vector<double> cached_vals_;
  std::vector<std::vector<std::pair<int, double>>> cached_decoded_;
  GPUSparseAFunction build_sparse_A_;
  GPUMatrixFreeMetadataFunction build_matrix_free_metadata_;
  bool cached_csr_valid_;
  bool cached_a_csr_valid_;
  bool cached_at_csr_valid_;
  std::vector<int> cached_a_crow_;
  std::vector<int> cached_a_col_;
  std::vector<double> cached_a_vals_;
  std::vector<int> cached_at_crow_;
  std::vector<int> cached_at_col_;
  std::vector<double> cached_at_vals_;
  bool matrix_free_;
  bool matrix_free_validate_;
  bool matrix_free_at_csr_;
  bool matrix_free_cuda_au_;
  bool matrix_free_cuda_atu_;
  bool cuda_verbose_;
  bool validate_au_range_;
  double cg_dynamic_factor_;
  bool cg_fused_normal_;
  bool shard_primal_;
  bool shard_primal_store_;
  bool psd_projection_multi_gpu_;
  std::string psd_projection_devices_;
  double psd_projection_base_memory_fraction_;
  double psd_projection_workspace_scale_;
  bool psd_projection_avoid_base_large_blocks_;
  bool validate_sparse_A_;
  bool profile_timing_;
  // Convergence options forwarded verbatim to the Python solver
  // (over-relaxation, relative gap, energy-stagnation stop).
  std::map<std::string, double> accel_options_;
  double runtime_error_convergence_ = -1.0;
  double runtime_objective_convergence_ = -1.0;
  std::map<std::string, double> runtime_options_;
};

} // namespace libsdp

#endif // GPU_ADMM_SOLVER_H
