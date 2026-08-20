#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace {

__host__ __device__ __forceinline__ long long packed_pair_base(long long j) {
  return j * (j + 1) / 2;
}

__device__ __forceinline__ void packed_pair(long long packed, int &i, int &j) {
  double root = sqrt(static_cast<double>(8 * packed + 1));
  j = static_cast<int>((root - 1.0) * 0.5);
  while (packed_pair_base(j) > packed) {
    --j;
  }
  while (packed_pair_base(static_cast<long long>(j) + 1) <= packed) {
    ++j;
  }
  i = static_cast<int>(packed - packed_pair_base(j));
}

__global__ void unpack_packed_symmetric_kernel(const double *__restrict__ packed,
                                               double *__restrict__ right,
                                               int nmo, int bq,
                                               long long ngem) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(bq) * ngem;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  const int nrow = nmo * bq;

  for (; idx < total; idx += stride) {
    const int q = static_cast<int>(idx / ngem);
    const long long p = idx - static_cast<long long>(q) * ngem;
    int i = 0;
    int j = 0;
    packed_pair(p, i, j);
    const double value = packed[idx];
    const int row_offset = q * nmo;
    right[(row_offset + i) + static_cast<long long>(j) * nrow] = value;
    right[(row_offset + j) + static_cast<long long>(i) * nrow] = value;
  }
}

__global__ void make_left_mat_kernel(const double *__restrict__ tmp,
                                     double *__restrict__ left, int nmo,
                                     int bq) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(bq) * nmo * nmo;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  const int nrow_tmp = nmo * bq;

  for (; idx < total; idx += stride) {
    const int q = static_cast<int>(idx / (static_cast<long long>(nmo) * nmo));
    const long long rem = idx - static_cast<long long>(q) * nmo * nmo;
    const int i = static_cast<int>(rem % nmo);
    const int j = static_cast<int>(rem / nmo);
    const int row_offset = q * nmo;
    left[i + static_cast<long long>(row_offset + j) * nmo] =
        tmp[(row_offset + i) + static_cast<long long>(j) * nrow_tmp];
  }
}

__global__ void scatter_packed_symmetric_kernel(
    const double *__restrict__ result, double *__restrict__ packed, int nmo,
    int bq, long long ngem) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(bq) * ngem;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

  for (; idx < total; idx += stride) {
    const int q = static_cast<int>(idx / ngem);
    const long long p = idx - static_cast<long long>(q) * ngem;
    int i = 0;
    int j = 0;
    packed_pair(p, i, j);
    const int row_offset = q * nmo;
    packed[idx] = result[i + static_cast<long long>(row_offset + j) * nmo];
  }
}

__global__ void build_pair_column_matrix_kernel(
    const double *__restrict__ int2, double *__restrict__ x, int nmo,
    int inner_begin, int ninner, long long q_begin, int q_count,
    long long ngem, int ldx) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total =
      static_cast<long long>(ldx) * static_cast<long long>(nmo);
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

  for (; idx < total; idx += stride) {
    const int p = static_cast<int>(idx / ldx);
    const int row = static_cast<int>(idx - static_cast<long long>(p) * ldx);
    const int q_local = row / ninner;
    const int inner_local = row - q_local * ninner;
    const int inner = inner_begin + inner_local;
    const int hi = p >= inner ? p : inner;
    const int lo = p >= inner ? inner : p;
    const long long pair = packed_pair_base(hi) + lo;
    x[row + static_cast<long long>(p) * ldx] =
        int2[(q_begin + q_local) * ngem + pair];
  }
}

// Variant of build_pair_column_matrix_kernel where the inner (summed) orbital
// index is taken from an explicit df-order list rather than a contiguous range.
// Used by the symmetry-general path, where doc/active orbitals are scattered in
// df (packing) order. x[(q_local*ninner + inner_local), p] = (p, inner | Q).
__global__ void build_pair_column_matrix_list_kernel(
    const double *__restrict__ int2, double *__restrict__ x, int nmo,
    const int *__restrict__ inner_list, int ninner, int q_count,
    long long ngem, int ldx) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total =
      static_cast<long long>(ldx) * static_cast<long long>(nmo);
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

  for (; idx < total; idx += stride) {
    const int p = static_cast<int>(idx / ldx);
    const int row = static_cast<int>(idx - static_cast<long long>(p) * ldx);
    const int q_local = row / ninner;
    const int inner_local = row - q_local * ninner;
    const int inner = inner_list[inner_local];
    const int hi = p >= inner ? p : inner;
    const int lo = p >= inner ? inner : p;
    const long long pair = packed_pair_base(hi) + lo;
    x[row + static_cast<long long>(p) * ldx] =
        int2[static_cast<long long>(q_local) * ngem + pair];
  }
}

__global__ void apply_active_density_kernel(const double *__restrict__ x,
                                            const double *__restrict__ den1,
                                            double *__restrict__ y, int nmo,
                                            int nact, int q_count, int ldx) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total =
      static_cast<long long>(ldx) * static_cast<long long>(nmo);
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

  for (; idx < total; idx += stride) {
    const int p = static_cast<int>(idx / ldx);
    const int row = static_cast<int>(idx - static_cast<long long>(p) * ldx);
    const int q_local = row / nact;
    const int t = row - q_local * nact;
    double value = 0.0;
    const long long q_offset = static_cast<long long>(q_local) * nact;
    const long long p_offset = static_cast<long long>(p) * ldx;
    for (int u = 0; u < nact; ++u) {
      const int hi = t >= u ? t : u;
      const int lo = t >= u ? u : t;
      const long long den_pair = packed_pair_base(hi) + lo;
      value += den1[den_pair] * x[(q_offset + u) + p_offset];
    }
    y[row + p_offset] = value;
  }
}

__global__ void build_active_pair_matrix_kernel(
    const double *__restrict__ int2, double *__restrict__ active, int ndoc,
    int nact, int q_count, long long ngem, int lda,
    const int *__restrict__ act_df) {
  const long long ngem_act = static_cast<long long>(nact) * (nact + 1) / 2;
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(q_count) * ngem_act;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

  for (; idx < total; idx += stride) {
    const int q_local = static_cast<int>(idx % q_count);
    const long long active_pair = idx / q_count;
    int lo = 0;
    int hi = 0;
    packed_pair(active_pair, lo, hi);
    // map the two local active indices to absolute (df-order) orbital indices:
    // contiguous [ndoc,ndoc+nact) for C1 (act_df==nullptr), else the scattered
    // df-order list for the symmetry-general path.
    const int a_lo = act_df ? act_df[lo] : ndoc + lo;
    const int a_hi = act_df ? act_df[hi] : ndoc + hi;
    const int abs_hi = a_hi >= a_lo ? a_hi : a_lo;
    const int abs_lo = a_hi >= a_lo ? a_lo : a_hi;
    const long long pair = packed_pair_base(abs_hi) + abs_lo;
    active[q_local + active_pair * static_cast<long long>(lda)] =
        int2[static_cast<long long>(q_local) * ngem + pair];
  }
}

__global__ void build_general_active_matrix_kernel(
    const double *__restrict__ int2, double *__restrict__ b, int nmo, int ndoc,
    int active_u, int q_count, long long ngem, int ldb,
    const int *__restrict__ act_df) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(q_count) * nmo;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  const int abs_u = act_df ? act_df[active_u] : ndoc + active_u;

  for (; idx < total; idx += stride) {
    const int q_local = static_cast<int>(idx % q_count);
    const int p = static_cast<int>(idx / q_count);
    const int hi = p >= abs_u ? p : abs_u;
    const int lo = p >= abs_u ? abs_u : p;
    const long long pair = packed_pair_base(hi) + lo;
    b[q_local + static_cast<long long>(p) * ldb] =
        int2[static_cast<long long>(q_local) * ngem + pair];
  }
}

__global__ void scatter_q_result_kernel(const double *__restrict__ result,
                                        double *__restrict__ q, int nmo,
                                        int nact, int active_u) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(nact) * nmo;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

  for (; idx < total; idx += stride) {
    const int t = static_cast<int>(idx % nact);
    const int p = static_cast<int>(idx / nact);
    const int hi = t >= active_u ? t : active_u;
    const int lo = t >= active_u ? active_u : t;
    const long long tu_pair = packed_pair_base(hi) + lo;
    q[t + static_cast<long long>(p) * nact] +=
        result[p + tu_pair * static_cast<long long>(nmo)];
  }
}

int cuda_blocks(long long n, int threads) {
  long long blocks = (n + threads - 1) / threads;
  blocks = std::max<long long>(1, std::min<long long>(blocks, 65535));
  return static_cast<int>(blocks);
}

double bytes_to_gib(double bytes) {
  return bytes / 1024.0 / 1024.0 / 1024.0;
}

int choose_block_q(int device, int nmo, long long q_count, long long ngem,
                   int requested_block_q, int verbose) {
  if (q_count <= 0) {
    return 1;
  }
  if (requested_block_q > 0) {
    return static_cast<int>(
        std::max<long long>(1, std::min<long long>(requested_block_q, q_count)));
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    const int fallback = static_cast<int>(std::min<long long>(32, q_count));
    if (verbose) {
      std::fprintf(stderr,
                   "Hilbert FOCAS CUDA: cuda:%d could not query free memory; "
                   "using fallback block_q=%d\n",
                   device, fallback);
    }
    return fallback;
  }

  constexpr int max_auto_block_q = 1024;
  const double nmo2 = static_cast<double>(nmo) * static_cast<double>(nmo);
  const double u_bytes = nmo2 * sizeof(double);
  const double bytes_per_q =
      (4.0 * nmo2 + static_cast<double>(ngem)) * sizeof(double);

  const double free_d = static_cast<double>(free_bytes);
  const double reserve =
      std::max(512.0 * 1024.0 * 1024.0, 0.10 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  const double variable_budget = std::max(0.0, budget - u_bytes);

  int block_q = static_cast<int>(variable_budget / bytes_per_q);
  block_q = std::max(1, block_q);
  block_q = std::min(block_q, max_auto_block_q);
  block_q = static_cast<int>(std::min<long long>(block_q, q_count));

  if (verbose) {
    const double estimated_bytes = u_bytes + bytes_per_q * block_q;
    std::fprintf(stderr,
                 "Hilbert FOCAS CUDA: cuda:%d auto block_q=%d "
                 "(free=%.3f GiB, total=%.3f GiB, estimated workspace=%.3f GiB)\n",
                 device, block_q, bytes_to_gib(free_d),
                 bytes_to_gib(static_cast<double>(total_bytes)),
                 bytes_to_gib(estimated_bytes));
  }
  return block_q;
}

int choose_gradient_q_chunk(int device, int nmo, int ninner, long long q_count,
                            long long ngem, int x_matrices, int requested_block_q,
                            int verbose, const char *label) {
  if (q_count <= 0 || ninner <= 0) {
    return 1;
  }
  if (requested_block_q > 0) {
    return static_cast<int>(
        std::max<long long>(1, std::min<long long>(requested_block_q, q_count)));
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    const int fallback = static_cast<int>(std::min<long long>(32, q_count));
    if (verbose) {
      std::fprintf(stderr,
                   "Hilbert FOCAS CUDA: cuda:%d could not query free memory; "
                   "using %s fallback q_chunk=%d\n",
                   device, label, fallback);
    }
    return fallback;
  }

  constexpr int max_auto_q_chunk = 4096;
  const double free_d = static_cast<double>(free_bytes);
  const double reserve =
      std::max(512.0 * 1024.0 * 1024.0, 0.10 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  const double c_bytes =
      static_cast<double>(nmo) * static_cast<double>(nmo) * sizeof(double);
  const double per_q_bytes =
      (static_cast<double>(x_matrices) * static_cast<double>(ninner) *
           static_cast<double>(nmo) +
       static_cast<double>(ngem)) *
      sizeof(double);
  int q_chunk = static_cast<int>((budget - c_bytes) / per_q_bytes);
  q_chunk = std::max(1, q_chunk);
  q_chunk = std::min(q_chunk, max_auto_q_chunk);
  q_chunk = static_cast<int>(std::min<long long>(q_chunk, q_count));

  if (verbose) {
    const double estimated_bytes = c_bytes + per_q_bytes * q_chunk;
    std::fprintf(stderr,
                 "Hilbert FOCAS CUDA: cuda:%d %s auto q_chunk=%d "
                 "(free=%.3f GiB, total=%.3f GiB, estimated workspace=%.3f GiB)\n",
                 device, label, q_chunk, bytes_to_gib(free_d),
                 bytes_to_gib(static_cast<double>(total_bytes)),
                 bytes_to_gib(estimated_bytes));
  }
  return q_chunk;
}

int choose_q_q_chunk(int device, int nmo, int nact, long long q_count,
                     long long ngem, int requested_block_q, int verbose) {
  if (q_count <= 0 || nact <= 0) {
    return 1;
  }
  if (requested_block_q > 0) {
    return static_cast<int>(
        std::max<long long>(1, std::min<long long>(requested_block_q, q_count)));
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    const int fallback = static_cast<int>(std::min<long long>(32, q_count));
    if (verbose) {
      std::fprintf(stderr,
                   "Hilbert FOCAS CUDA: cuda:%d could not query free memory; "
                   "using Q contraction fallback q_chunk=%d\n",
                   device, fallback);
    }
    return fallback;
  }

  constexpr int max_auto_q_chunk = 4096;
  const double free_d = static_cast<double>(free_bytes);
  const double reserve =
      std::max(512.0 * 1024.0 * 1024.0, 0.10 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  const double ngem_act =
      static_cast<double>(nact) * static_cast<double>(nact + 1) * 0.5;
  const double fixed_bytes =
      (ngem_act * ngem_act + static_cast<double>(nmo) * ngem_act +
       static_cast<double>(nact) * static_cast<double>(nmo)) *
      sizeof(double);
  const double per_q_bytes =
      (static_cast<double>(ngem) + 2.0 * ngem_act + static_cast<double>(nmo)) *
      sizeof(double);
  int q_chunk = static_cast<int>((budget - fixed_bytes) / per_q_bytes);
  q_chunk = std::max(1, q_chunk);
  q_chunk = std::min(q_chunk, max_auto_q_chunk);
  q_chunk = static_cast<int>(std::min<long long>(q_chunk, q_count));

  if (verbose) {
    const double estimated_bytes = fixed_bytes + per_q_bytes * q_chunk;
    std::fprintf(stderr,
                 "Hilbert FOCAS CUDA: cuda:%d Q contraction auto q_chunk=%d "
                 "(free=%.3f GiB, total=%.3f GiB, estimated workspace=%.3f GiB)\n",
                 device, q_chunk, bytes_to_gib(free_d),
                 bytes_to_gib(static_cast<double>(total_bytes)),
                 bytes_to_gib(estimated_bytes));
  }
  return q_chunk;
}

int choose_coulomb_q_chunk(int device, long long q_count, long long ngem,
                           int requested_block_q, int verbose,
                           const char *label) {
  if (q_count <= 0) {
    return 1;
  }
  if (requested_block_q > 0) {
    return static_cast<int>(
        std::max<long long>(1, std::min<long long>(requested_block_q, q_count)));
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    const int fallback = static_cast<int>(std::min<long long>(32, q_count));
    if (verbose) {
      std::fprintf(stderr,
                   "Hilbert FOCAS CUDA: cuda:%d could not query free memory; "
                   "using %s fallback q_chunk=%d\n",
                   device, label, fallback);
    }
    return fallback;
  }

  constexpr int max_auto_q_chunk = 4096;
  const double free_d = static_cast<double>(free_bytes);
  const double reserve =
      std::max(512.0 * 1024.0 * 1024.0, 0.10 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  const double fixed_bytes = static_cast<double>(ngem) * sizeof(double);
  const double per_q_bytes = (static_cast<double>(ngem) + 1.0) * sizeof(double);
  int q_chunk = static_cast<int>((budget - fixed_bytes) / per_q_bytes);
  q_chunk = std::max(1, q_chunk);
  q_chunk = std::min(q_chunk, max_auto_q_chunk);
  q_chunk = static_cast<int>(std::min<long long>(q_chunk, q_count));

  if (verbose) {
    const double estimated_bytes = fixed_bytes + per_q_bytes * q_chunk;
    std::fprintf(stderr,
                 "Hilbert FOCAS CUDA: cuda:%d %s auto q_chunk=%d "
                 "(free=%.3f GiB, total=%.3f GiB, estimated workspace=%.3f GiB)\n",
                 device, label, q_chunk, bytes_to_gib(free_d),
                 bytes_to_gib(static_cast<double>(total_bytes)),
                 bytes_to_gib(estimated_bytes));
  }
  return q_chunk;
}

int process_range_on_device(int device, int nmo, long long q_begin,
                            long long q_end, long long ngem, double *int2,
                            const double *u_host, int requested_block_q,
                            int verbose) {
  if (q_begin >= q_end) {
    return 0;
  }

  cudaError_t cerr = cudaSetDevice(device);
  if (cerr != cudaSuccess) {
    return 10;
  }

  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    return 11;
  }

  const int block_q =
      choose_block_q(device, nmo, q_end - q_begin, ngem, requested_block_q,
                     verbose);
  const int nrow = nmo * block_q;
  const std::size_t u_bytes = static_cast<std::size_t>(nmo) * nmo * sizeof(double);
  const std::size_t packed_bytes =
      static_cast<std::size_t>(block_q) * ngem * sizeof(double);
  const std::size_t rectangular_bytes =
      static_cast<std::size_t>(nrow) * nmo * sizeof(double);
  const std::size_t left_bytes = rectangular_bytes;

  double *d_u = nullptr;
  double *d_packed = nullptr;
  double *d_right = nullptr;
  double *d_tmp = nullptr;
  double *d_left = nullptr;
  double *d_result = nullptr;

  auto cleanup = [&]() {
    if (d_result) cudaFree(d_result);
    if (d_left) cudaFree(d_left);
    if (d_tmp) cudaFree(d_tmp);
    if (d_right) cudaFree(d_right);
    if (d_packed) cudaFree(d_packed);
    if (d_u) cudaFree(d_u);
    cublasDestroy(handle);
  };

  if (cudaMalloc(&d_u, u_bytes) != cudaSuccess ||
      cudaMalloc(&d_packed, packed_bytes) != cudaSuccess ||
      cudaMalloc(&d_right, rectangular_bytes) != cudaSuccess ||
      cudaMalloc(&d_tmp, rectangular_bytes) != cudaSuccess ||
      cudaMalloc(&d_left, left_bytes) != cudaSuccess ||
      cudaMalloc(&d_result, left_bytes) != cudaSuccess) {
    cleanup();
    return 12;
  }

  if (cudaMemcpy(d_u, u_host, u_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cleanup();
    return 13;
  }

  const double alpha = 1.0;
  const double beta = 0.0;
  constexpr int threads = 256;

  for (long long q = q_begin; q < q_end; q += block_q) {
    const int bq = static_cast<int>(std::min<long long>(block_q, q_end - q));
    const int nrow_bq = nmo * bq;
    const std::size_t packed_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    double *host_block = int2 + q * ngem;

    if (cudaMemcpy(d_packed, host_block, packed_bq_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return 14;
    }

    const long long packed_total = static_cast<long long>(bq) * ngem;
    unpack_packed_symmetric_kernel<<<cuda_blocks(packed_total, threads),
                                     threads>>>(d_packed, d_right, nmo, bq,
                                                ngem);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 15;
    }

    if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nrow_bq, nmo, nmo,
                    &alpha, d_right, nrow_bq, d_u, nmo, &beta, d_tmp,
                    nrow_bq) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 16;
    }

    const long long dense_total = static_cast<long long>(bq) * nmo * nmo;
    make_left_mat_kernel<<<cuda_blocks(dense_total, threads), threads>>>(
        d_tmp, d_left, nmo, bq);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 17;
    }

    if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nmo, nrow_bq, nmo,
                    &alpha, d_u, nmo, d_left, nmo, &beta, d_result,
                    nmo) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 18;
    }

    scatter_packed_symmetric_kernel<<<cuda_blocks(packed_total, threads),
                                      threads>>>(d_result, d_packed, nmo, bq,
                                                 ngem);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 19;
    }

    if (cudaMemcpy(host_block, d_packed, packed_bq_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      cleanup();
      return 20;
    }
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
    cleanup();
    return 21;
  }

  cleanup();
  return 0;
}

int compute_fi_exchange_on_device(int device, int nmo, int ndoc,
                                  long long nQ, long long q_begin,
                                  long long q_end, long long ngem,
                                  const double *int2, const int *doc_df,
                                  int requested_q_chunk,
                                  int verbose, std::vector<double> &host_c) {
  if (q_begin >= q_end || ndoc <= 0) {
    std::fill(host_c.begin(), host_c.end(), 0.0);
    return 0;
  }

  cudaError_t cerr = cudaSetDevice(device);
  if (cerr != cudaSuccess) {
    return 30;
  }

  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    return 31;
  }

  const int q_chunk = choose_gradient_q_chunk(
      device, nmo, ndoc, q_end - q_begin, ngem, 1, requested_q_chunk, verbose,
      "Fi exchange");
  const int max_ldx_ll = static_cast<int>(
      std::min<long long>(static_cast<long long>(q_chunk) * ndoc,
                          std::numeric_limits<int>::max()));
  if (max_ldx_ll != static_cast<long long>(q_chunk) * ndoc) {
    cublasDestroy(handle);
    return 32;
  }
  const int max_ldx = q_chunk * ndoc;
  const std::size_t x_bytes =
      static_cast<std::size_t>(max_ldx) * nmo * sizeof(double);
  const std::size_t int2_bytes =
      static_cast<std::size_t>(q_chunk) * ngem * sizeof(double);
  const std::size_t c_bytes =
      static_cast<std::size_t>(nmo) * nmo * sizeof(double);

  double *d_int2 = nullptr;
  double *d_x = nullptr;
  double *d_c = nullptr;
  int *d_inner = nullptr;
  auto cleanup = [&]() {
    if (d_inner) cudaFree(d_inner);
    if (d_c) cudaFree(d_c);
    if (d_x) cudaFree(d_x);
    if (d_int2) cudaFree(d_int2);
    cublasDestroy(handle);
  };

  if (cudaMalloc(&d_int2, int2_bytes) != cudaSuccess ||
      cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
      cudaMalloc(&d_c, c_bytes) != cudaSuccess) {
    cleanup();
    return 33;
  }
  if (cudaMemset(d_c, 0, c_bytes) != cudaSuccess) {
    cleanup();
    return 34;
  }
  // Optional df-order doc-orbital list (symmetry-general path). When null, the
  // doc orbitals are the contiguous range [0, ndoc) (C1 path).
  if (doc_df != nullptr) {
    if (cudaMalloc(&d_inner, static_cast<std::size_t>(ndoc) * sizeof(int)) !=
            cudaSuccess ||
        cudaMemcpy(d_inner, doc_df, static_cast<std::size_t>(ndoc) * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return 34;
    }
  }

  constexpr int threads = 256;
  const double alpha = 1.0;
  const double beta = 1.0;
  for (long long q = q_begin; q < q_end; q += q_chunk) {
    const int bq = static_cast<int>(std::min<long long>(q_chunk, q_end - q));
    const int ldx = bq * ndoc;
    const std::size_t int2_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    if (cudaMemcpy(d_int2, int2 + q * ngem, int2_bq_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return 35;
    }
    const long long total = static_cast<long long>(ldx) * nmo;
    if (d_inner != nullptr) {
      build_pair_column_matrix_list_kernel<<<cuda_blocks(total, threads),
                                             threads>>>(d_int2, d_x, nmo,
                                                        d_inner, ndoc, bq, ngem,
                                                        ldx);
    } else {
      build_pair_column_matrix_kernel<<<cuda_blocks(total, threads), threads>>>(
          d_int2, d_x, nmo, 0, ndoc, 0, bq, ngem, ldx);
    }
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 36;
    }
    if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nmo, nmo, ldx,
                    &alpha, d_x, ldx, d_x, ldx, &beta, d_c,
                    nmo) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 37;
    }
  }

  if (cudaMemcpy(host_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    cleanup();
    return 38;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cleanup();
    return 39;
  }
  cleanup();
  return 0;
}

int compute_fa_exchange_on_device(int device, int nmo, int ndoc, int nact,
                                  long long nQ, long long q_begin,
                                  long long q_end, long long ngem,
                                  const double *int2, const double *den1,
                                  const int *act_df,
                                  int requested_q_chunk, int verbose,
                                  std::vector<double> &host_c) {
  if (q_begin >= q_end || nact <= 0) {
    std::fill(host_c.begin(), host_c.end(), 0.0);
    return 0;
  }

  cudaError_t cerr = cudaSetDevice(device);
  if (cerr != cudaSuccess) {
    return 40;
  }

  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    return 41;
  }

  const int q_chunk = choose_gradient_q_chunk(
      device, nmo, nact, q_end - q_begin, ngem, 2, requested_q_chunk, verbose,
      "Fa exchange");
  const long long max_ldx_ll = static_cast<long long>(q_chunk) * nact;
  if (max_ldx_ll > std::numeric_limits<int>::max()) {
    cublasDestroy(handle);
    return 42;
  }
  const int max_ldx = static_cast<int>(max_ldx_ll);
  const std::size_t x_bytes =
      static_cast<std::size_t>(max_ldx) * nmo * sizeof(double);
  const std::size_t int2_bytes =
      static_cast<std::size_t>(q_chunk) * ngem * sizeof(double);
  const std::size_t c_bytes =
      static_cast<std::size_t>(nmo) * nmo * sizeof(double);
  const std::size_t den_bytes =
      static_cast<std::size_t>(nact) * (nact + 1) / 2 * sizeof(double);

  double *d_int2 = nullptr;
  double *d_x = nullptr;
  double *d_y = nullptr;
  double *d_c = nullptr;
  double *d_den1 = nullptr;
  int *d_inner = nullptr;
  auto cleanup = [&]() {
    if (d_inner) cudaFree(d_inner);
    if (d_den1) cudaFree(d_den1);
    if (d_c) cudaFree(d_c);
    if (d_y) cudaFree(d_y);
    if (d_x) cudaFree(d_x);
    if (d_int2) cudaFree(d_int2);
    cublasDestroy(handle);
  };

  if (cudaMalloc(&d_int2, int2_bytes) != cudaSuccess ||
      cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
      cudaMalloc(&d_y, x_bytes) != cudaSuccess ||
      cudaMalloc(&d_c, c_bytes) != cudaSuccess ||
      cudaMalloc(&d_den1, den_bytes) != cudaSuccess) {
    cleanup();
    return 43;
  }
  if (cudaMemcpy(d_den1, den1, den_bytes, cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemset(d_c, 0, c_bytes) != cudaSuccess) {
    cleanup();
    return 44;
  }
  // optional df-order active-orbital list (symmetry-general path); null => the
  // active orbitals are the contiguous block [ndoc, ndoc+nact) (C1)
  if (act_df != nullptr) {
    if (cudaMalloc(&d_inner, static_cast<std::size_t>(nact) * sizeof(int)) !=
            cudaSuccess ||
        cudaMemcpy(d_inner, act_df, static_cast<std::size_t>(nact) * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return 44;
    }
  }

  constexpr int threads = 256;
  const double alpha = 1.0;
  const double beta = 1.0;
  for (long long q = q_begin; q < q_end; q += q_chunk) {
    const int bq = static_cast<int>(std::min<long long>(q_chunk, q_end - q));
    const int ldx = bq * nact;
    const std::size_t int2_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    if (cudaMemcpy(d_int2, int2 + q * ngem, int2_bq_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return 45;
    }
    const long long total = static_cast<long long>(ldx) * nmo;
    if (d_inner != nullptr) {
      build_pair_column_matrix_list_kernel<<<cuda_blocks(total, threads),
                                             threads>>>(d_int2, d_x, nmo,
                                                        d_inner, nact, bq, ngem,
                                                        ldx);
    } else {
      build_pair_column_matrix_kernel<<<cuda_blocks(total, threads), threads>>>(
          d_int2, d_x, nmo, ndoc, nact, 0, bq, ngem, ldx);
    }
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 46;
    }
    apply_active_density_kernel<<<cuda_blocks(total, threads), threads>>>(
        d_x, d_den1, d_y, nmo, nact, bq, ldx);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 47;
    }
    if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nmo, nmo, ldx,
                    &alpha, d_x, ldx, d_y, ldx, &beta, d_c,
                    nmo) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 48;
    }
  }

  if (cudaMemcpy(host_c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    cleanup();
    return 49;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cleanup();
    return 50;
  }
  cleanup();
  return 0;
}

void build_scaled_c1_d2(int nact, const double *den2,
                        std::vector<double> &scaled_d2) {
  const long long ngem_act = static_cast<long long>(nact) * (nact + 1) / 2;
  std::fill(scaled_d2.begin(), scaled_d2.end(), 0.0);
  for (int t = 0; t < nact; ++t) {
    for (int u = 0; u <= t; ++u) {
      const long long tu = packed_pair_base(t) + u;
      for (int v = 0; v < nact; ++v) {
        for (int w = 0; w <= v; ++w) {
          const long long vw = packed_pair_base(v) + w;
          const long long hi = tu >= vw ? tu : vw;
          const long long lo = tu >= vw ? vw : tu;
          const long long den_index = packed_pair_base(hi) + lo;
          const double scale = v == w ? 1.0 : 2.0;
          scaled_d2[vw + tu * ngem_act] = scale * den2[den_index];
        }
      }
    }
  }
}

int compute_q_on_device(int device, int nmo, int ndoc, int nact, long long nQ,
                        long long q_begin, long long q_end, long long ngem,
                        const double *int2,
                        const std::vector<double> &scaled_d2,
                        const int *act_df,
                        int requested_q_chunk, int verbose,
                        std::vector<double> &host_q) {
  (void)nQ;
  if (q_begin >= q_end || nact <= 0) {
    std::fill(host_q.begin(), host_q.end(), 0.0);
    return 0;
  }

  cudaError_t cerr = cudaSetDevice(device);
  if (cerr != cudaSuccess) {
    return 80;
  }

  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    return 81;
  }

  const long long ngem_act_ll = static_cast<long long>(nact) * (nact + 1) / 2;
  if (ngem_act_ll > std::numeric_limits<int>::max()) {
    cublasDestroy(handle);
    return 82;
  }
  const int ngem_act = static_cast<int>(ngem_act_ll);
  const int q_chunk = choose_q_q_chunk(device, nmo, nact, q_end - q_begin,
                                       ngem, requested_q_chunk, verbose);
  const std::size_t int2_bytes =
      static_cast<std::size_t>(q_chunk) * ngem * sizeof(double);
  const std::size_t active_bytes =
      static_cast<std::size_t>(q_chunk) * ngem_act * sizeof(double);
  const std::size_t b_bytes =
      static_cast<std::size_t>(q_chunk) * nmo * sizeof(double);
  const std::size_t d2_bytes =
      static_cast<std::size_t>(ngem_act) * ngem_act * sizeof(double);
  const std::size_t result_bytes =
      static_cast<std::size_t>(nmo) * ngem_act * sizeof(double);
  const std::size_t q_bytes =
      static_cast<std::size_t>(nact) * nmo * sizeof(double);

  double *d_int2 = nullptr;
  double *d_active = nullptr;
  double *d_d2 = nullptr;
  double *d_qint = nullptr;
  double *d_b = nullptr;
  double *d_result = nullptr;
  double *d_q = nullptr;
  int *d_act = nullptr;
  auto cleanup = [&]() {
    if (d_act) cudaFree(d_act);
    if (d_q) cudaFree(d_q);
    if (d_result) cudaFree(d_result);
    if (d_b) cudaFree(d_b);
    if (d_qint) cudaFree(d_qint);
    if (d_d2) cudaFree(d_d2);
    if (d_active) cudaFree(d_active);
    if (d_int2) cudaFree(d_int2);
    cublasDestroy(handle);
  };

  if (cudaMalloc(&d_int2, int2_bytes) != cudaSuccess ||
      cudaMalloc(&d_active, active_bytes) != cudaSuccess ||
      cudaMalloc(&d_d2, d2_bytes) != cudaSuccess ||
      cudaMalloc(&d_qint, active_bytes) != cudaSuccess ||
      cudaMalloc(&d_b, b_bytes) != cudaSuccess ||
      cudaMalloc(&d_result, result_bytes) != cudaSuccess ||
      cudaMalloc(&d_q, q_bytes) != cudaSuccess) {
    cleanup();
    return 83;
  }
  if (cudaMemcpy(d_d2, scaled_d2.data(), d2_bytes, cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemset(d_q, 0, q_bytes) != cudaSuccess) {
    cleanup();
    return 84;
  }
  // optional df-order active-orbital list (symmetry-general path); null => the
  // active orbitals are the contiguous block [ndoc, ndoc+nact) (C1)
  if (act_df != nullptr) {
    if (cudaMalloc(&d_act, static_cast<std::size_t>(nact) * sizeof(int)) !=
            cudaSuccess ||
        cudaMemcpy(d_act, act_df, static_cast<std::size_t>(nact) * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return 84;
    }
  }

  constexpr int threads = 256;
  const double alpha = 1.0;
  const double beta0 = 0.0;
  for (long long q = q_begin; q < q_end; q += q_chunk) {
    const int bq = static_cast<int>(std::min<long long>(q_chunk, q_end - q));
    const std::size_t int2_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    if (cudaMemcpy(d_int2, int2 + q * ngem, int2_bq_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return 85;
    }

    const long long active_total = static_cast<long long>(bq) * ngem_act;
    build_active_pair_matrix_kernel<<<cuda_blocks(active_total, threads),
                                      threads>>>(d_int2, d_active, ndoc, nact,
                                                 bq, ngem, bq, d_act);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 86;
    }
    if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, bq, ngem_act, ngem_act,
                    &alpha, d_active, bq, d_d2, ngem_act, &beta0, d_qint,
                    bq) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 87;
    }

    const long long b_total = static_cast<long long>(bq) * nmo;
    for (int u = 0; u < nact; ++u) {
      build_general_active_matrix_kernel<<<cuda_blocks(b_total, threads),
                                           threads>>>(d_int2, d_b, nmo, ndoc, u,
                                                      bq, ngem, bq, d_act);
      if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return 88;
      }
      if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nmo, ngem_act, bq,
                      &alpha, d_b, bq, d_qint, bq, &beta0, d_result,
                      nmo) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return 89;
      }
      scatter_q_result_kernel<<<cuda_blocks(static_cast<long long>(nact) * nmo,
                                             threads),
                                threads>>>(d_result, d_q, nmo, nact, u);
      if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return 90;
      }
    }
  }

  if (cudaMemcpy(host_q.data(), d_q, q_bytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    cleanup();
    return 91;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cleanup();
    return 92;
  }
  cleanup();
  return 0;
}

void build_fi_coulomb_vector(int ndoc, long long nQ, long long ngem,
                             const double *int2,
                             std::vector<double> &qvec) {
  std::fill(qvec.begin(), qvec.end(), 0.0);
  for (long long q = 0; q < nQ; ++q) {
    const double *row = int2 + q * ngem;
    double value = 0.0;
    for (int i = 0; i < ndoc; ++i) {
      const long long pair = packed_pair_base(i) + i;
      value += 2.0 * row[pair];
    }
    qvec[static_cast<std::size_t>(q)] = value;
  }
}

void build_fa_coulomb_vector(int ndoc, int nact, long long nQ, long long ngem,
                             const double *int2, const double *den1,
                             std::vector<double> &qvec) {
  std::fill(qvec.begin(), qvec.end(), 0.0);
  for (long long q = 0; q < nQ; ++q) {
    const double *row = int2 + q * ngem;
    double value = 0.0;
    for (int t = 0; t < nact; ++t) {
      const int abs_t = ndoc + t;
      for (int u = 0; u < t; ++u) {
        const int abs_u = ndoc + u;
        const long long den_pair = packed_pair_base(t) + u;
        const long long int_pair = packed_pair_base(abs_t) + abs_u;
        value += 2.0 * den1[den_pair] * row[int_pair];
      }
      const long long den_pair = packed_pair_base(t) + t;
      const long long int_pair = packed_pair_base(abs_t) + abs_t;
      value += den1[den_pair] * row[int_pair];
    }
    qvec[static_cast<std::size_t>(q)] = value;
  }
}

int compute_coulomb_on_device(int device, long long q_begin, long long q_end,
                              long long ngem, const double *int2,
                              const double *qvec, int requested_q_chunk,
                              int verbose, const char *label,
                              std::vector<double> &host_pairs) {
  if (q_begin >= q_end) {
    std::fill(host_pairs.begin(), host_pairs.end(), 0.0);
    return 0;
  }

  cudaError_t cerr = cudaSetDevice(device);
  if (cerr != cudaSuccess) {
    return 110;
  }

  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    return 111;
  }

  const int q_chunk = choose_coulomb_q_chunk(
      device, q_end - q_begin, ngem, requested_q_chunk, verbose, label);
  const std::size_t int2_bytes =
      static_cast<std::size_t>(q_chunk) * ngem * sizeof(double);
  const std::size_t qvec_bytes =
      static_cast<std::size_t>(q_chunk) * sizeof(double);
  const std::size_t pair_bytes = static_cast<std::size_t>(ngem) * sizeof(double);

  double *d_int2 = nullptr;
  double *d_qvec = nullptr;
  double *d_pairs = nullptr;
  auto cleanup = [&]() {
    if (d_pairs) cudaFree(d_pairs);
    if (d_qvec) cudaFree(d_qvec);
    if (d_int2) cudaFree(d_int2);
    cublasDestroy(handle);
  };

  if (cudaMalloc(&d_int2, int2_bytes) != cudaSuccess ||
      cudaMalloc(&d_qvec, qvec_bytes) != cudaSuccess ||
      cudaMalloc(&d_pairs, pair_bytes) != cudaSuccess) {
    cleanup();
    return 112;
  }
  if (cudaMemset(d_pairs, 0, pair_bytes) != cudaSuccess) {
    cleanup();
    return 113;
  }

  const double alpha = 1.0;
  const double beta = 1.0;
  for (long long q = q_begin; q < q_end; q += q_chunk) {
    const int bq = static_cast<int>(std::min<long long>(q_chunk, q_end - q));
    const std::size_t int2_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    const std::size_t qvec_bq_bytes =
        static_cast<std::size_t>(bq) * sizeof(double);
    if (cudaMemcpy(d_int2, int2 + q * ngem, int2_bq_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_qvec, qvec + q, qvec_bq_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return 114;
    }
    if (cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(ngem), bq, &alpha,
                    d_int2, static_cast<int>(ngem), d_qvec, 1, &beta,
                    d_pairs, 1) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 115;
    }
  }

  if (cudaMemcpy(host_pairs.data(), d_pairs, pair_bytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    cleanup();
    return 116;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cleanup();
    return 117;
  }
  cleanup();
  return 0;
}

int device_count_from_request(int max_devices);

template <typename Worker>
int run_pair_workers(long long nQ, long long pair_count, int max_devices,
                     int verbose, const char *label, Worker worker,
                     std::vector<double> &pair_total) {
  const int devices = device_count_from_request(max_devices);
  if (devices <= 0) {
    return 118;
  }
  if (verbose) {
    std::fprintf(stderr,
                 "Hilbert FOCAS CUDA: using %d CUDA device(s) for C1 DF %s%s\n",
                 devices, label, max_devices <= 0 ? " (all visible)" : "");
  }

  std::vector<std::thread> workers;
  std::vector<int> statuses(devices, 0);
  std::vector<std::vector<double>> partials(
      devices, std::vector<double>(static_cast<std::size_t>(pair_count), 0.0));

  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share = (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] = worker(dev, q_begin, q_end, partials[dev]);
    });
  }

  for (auto &thread : workers) {
    thread.join();
  }
  for (int status : statuses) {
    if (status != 0) {
      return status;
    }
  }

  std::fill(pair_total.begin(), pair_total.end(), 0.0);
  for (const auto &partial : partials) {
    for (std::size_t i = 0; i < pair_total.size(); ++i) {
      pair_total[i] += partial[i];
    }
  }
  return 0;
}

void scatter_c1_coulomb(int nmo, int ndoc, int nact,
                        const std::vector<double> &pair_values,
                        const double *int1, double *fock_occ,
                        double *fock_ext) {
  const int nocc = ndoc + nact;
  for (int q = 0; q < nocc; ++q) {
    for (int p = 0; p < nmo; ++p) {
      const int hi = p >= q ? p : q;
      const int lo = p >= q ? q : p;
      const long long pair = packed_pair_base(hi) + lo;
      double value = pair_values[static_cast<std::size_t>(pair)];
      if (int1 != nullptr) {
        value += int1[pair];
      }
      fock_occ[p + static_cast<long long>(q) * nmo] = value;
    }
  }
  for (int p = nocc; p < nmo; ++p) {
    const long long pair = packed_pair_base(p) + p;
    double value = pair_values[static_cast<std::size_t>(pair)];
    if (int1 != nullptr) {
      value += int1[pair];
    }
    fock_ext[p - nocc] = value;
  }
}

void scatter_c1_exchange(int nmo, int ndoc, int nact, double scale,
                         const std::vector<double> &c, double *fock_occ,
                         double *fock_ext) {
  const int nocc = ndoc + nact;
  for (int q = 0; q < nocc; ++q) {
    for (int p = 0; p < nmo; ++p) {
      fock_occ[p + static_cast<long long>(q) * nmo] +=
          scale * c[p + static_cast<long long>(q) * nmo];
    }
  }
  for (int p = nocc; p < nmo; ++p) {
    fock_ext[p - nocc] += scale * c[p + static_cast<long long>(p) * nmo];
  }
}

int device_count_from_request(int max_devices) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return 0;
  }
  if (max_devices > 0) {
    device_count = std::min(device_count, max_devices);
  }
  return std::max(1, device_count);
}

template <typename Worker>
int run_exchange_workers(int nmo, long long nQ, int max_devices, int verbose,
                         const char *label, Worker worker,
                         std::vector<double> &c_total) {
  const int devices = device_count_from_request(max_devices);
  if (devices <= 0) {
    return 50;
  }
  if (verbose) {
    std::fprintf(stderr,
                 "Hilbert FOCAS CUDA: using %d CUDA device(s) for C1 DF %s%s\n",
                 devices, label, max_devices <= 0 ? " (all visible)" : "");
  }

  std::vector<std::thread> workers;
  std::vector<int> statuses(devices, 0);
  std::vector<std::vector<double>> partials(
      devices, std::vector<double>(static_cast<std::size_t>(nmo) * nmo, 0.0));

  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share = (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] = worker(dev, q_begin, q_end, partials[dev]);
    });
  }

  for (auto &thread : workers) {
    thread.join();
  }
  for (int status : statuses) {
    if (status != 0) {
      return status;
    }
  }

  std::fill(c_total.begin(), c_total.end(), 0.0);
  for (const auto &partial : partials) {
    for (std::size_t i = 0; i < c_total.size(); ++i) {
      c_total[i] += partial[i];
    }
  }
  return 0;
}

}  // namespace

extern "C" int hilbert_focas_df_c1_cuda_transform(
    int nmo, long long nQ, double *int2, const double *u, int block_q,
    int max_devices, int verbose) {
  (void)verbose;
  if (nmo <= 0 || nQ <= 0 || int2 == nullptr || u == nullptr) {
    return 2;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return 1;
  }
  if (max_devices > 0) {
    device_count = std::min(device_count, max_devices);
  }
  if (verbose) {
    std::fprintf(stderr,
                 "Hilbert FOCAS CUDA: using %d CUDA device(s) for C1 DF "
                 "transform%s\n",
                 device_count, max_devices <= 0 ? " (all visible)" : "");
  }

  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  const int devices = std::max(1, device_count);
  std::vector<std::thread> workers;
  std::vector<int> statuses(devices, 0);

  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share = (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] = process_range_on_device(dev, nmo, q_begin, q_end, ngem,
                                               int2, u, block_q, verbose);
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  for (int status : statuses) {
    if (status != 0) {
      return status;
    }
  }
  return 0;
}

extern "C" int hilbert_focas_df_c1_cuda_fi_exchange(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    double *fock_occ, double *fock_ext, int q_chunk, int max_devices,
    int verbose) {
  if (nmo <= 0 || ndoc < 0 || nact < 0 || nQ <= 0 || int2 == nullptr ||
      fock_occ == nullptr || fock_ext == nullptr || ndoc + nact > nmo) {
    return 60;
  }
  if (ndoc == 0) {
    return 0;
  }
  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  std::vector<double> c_total(static_cast<std::size_t>(nmo) * nmo, 0.0);
  const int status = run_exchange_workers(
      nmo, nQ, max_devices, verbose, "Fi exchange",
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_fi_exchange_on_device(dev, nmo, ndoc, nQ, q_begin,
                                             q_end, ngem, int2, nullptr, q_chunk,
                                             verbose, partial);
      },
      c_total);
  if (status != 0) {
    return status;
  }
  scatter_c1_exchange(nmo, ndoc, nact, -1.0, c_total, fock_occ, fock_ext);
  return 0;
}

// Symmetry-general Fi exchange. Returns the dense nmo x nmo exchange matrix
// C(p_df, q_df) = sum_{i in doc} (p_df i | q_df i) in df (packing) order. The
// caller (Fortran) scatters this into the per-irrep Fock blocks with the
// appropriate -1 sign, using the orbital symmetry maps it already owns. doc_df
// holds the df-order indices of the doubly-occupied orbitals.
extern "C" int hilbert_focas_df_sym_cuda_fi_exchange(
    int nmo, int ndoc, long long nQ, const double *int2, const int *doc_df,
    double *c_out, int q_chunk, int max_devices, int verbose) {
  if (nmo <= 0 || ndoc < 0 || nQ <= 0 || int2 == nullptr || c_out == nullptr) {
    return 60;
  }
  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  const std::size_t c_size = static_cast<std::size_t>(nmo) * nmo;
  std::fill(c_out, c_out + c_size, 0.0);
  if (ndoc == 0) {
    return 0;
  }
  if (doc_df == nullptr) {
    return 61;
  }
  std::vector<double> c_total(c_size, 0.0);
  const int status = run_exchange_workers(
      nmo, nQ, max_devices, verbose, "Fi exchange (sym)",
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_fi_exchange_on_device(dev, nmo, ndoc, nQ, q_begin,
                                             q_end, ngem, int2, doc_df, q_chunk,
                                             verbose, partial);
      },
      c_total);
  if (status != 0) {
    return status;
  }
  std::copy(c_total.begin(), c_total.end(), c_out);
  return 0;
}

extern "C" int hilbert_focas_df_c1_cuda_fa_exchange(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    const double *den1, double *fock_occ, double *fock_ext, int q_chunk,
    int max_devices, int verbose) {
  if (nmo <= 0 || ndoc < 0 || nact < 0 || nQ <= 0 || int2 == nullptr ||
      den1 == nullptr || fock_occ == nullptr || fock_ext == nullptr ||
      ndoc + nact > nmo) {
    return 70;
  }
  if (nact == 0) {
    return 0;
  }
  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  std::vector<double> c_total(static_cast<std::size_t>(nmo) * nmo, 0.0);
  const int status = run_exchange_workers(
      nmo, nQ, max_devices, verbose, "Fa exchange",
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_fa_exchange_on_device(dev, nmo, ndoc, nact, nQ,
                                             q_begin, q_end, ngem, int2, den1,
                                             nullptr, q_chunk, verbose, partial);
      },
      c_total);
  if (status != 0) {
    return status;
  }
  scatter_c1_exchange(nmo, ndoc, nact, -0.5, c_total, fock_occ, fock_ext);
  return 0;
}

// Symmetry-general Fa exchange. den1 is the symmetry-blocked, packed active
// 1-RDM (local active-pair packing); act_df is the df-order active list. Returns
// the dense nmo x nmo matrix C(p_df,q_df) = sum_{tu} (p_df t | q_df u) D1(t,u);
// the caller scatters it into the per-irrep Fa blocks with the -0.5 factor.
extern "C" int hilbert_focas_df_sym_cuda_fa_exchange(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    const double *den1, const int *act_df, double *c_out, int q_chunk,
    int max_devices, int verbose) {
  if (nmo <= 0 || ndoc < 0 || nact < 0 || nQ <= 0 || int2 == nullptr ||
      den1 == nullptr || c_out == nullptr) {
    return 70;
  }
  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  const std::size_t c_size = static_cast<std::size_t>(nmo) * nmo;
  std::fill(c_out, c_out + c_size, 0.0);
  if (nact == 0) {
    return 0;
  }
  if (act_df == nullptr) {
    return 71;
  }
  std::vector<double> c_total(c_size, 0.0);
  const int status = run_exchange_workers(
      nmo, nQ, max_devices, verbose, "Fa exchange (sym)",
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_fa_exchange_on_device(dev, nmo, ndoc, nact, nQ,
                                             q_begin, q_end, ngem, int2, den1,
                                             act_df, q_chunk, verbose, partial);
      },
      c_total);
  if (status != 0) {
    return status;
  }
  std::copy(c_total.begin(), c_total.end(), c_out);
  return 0;
}

extern "C" int hilbert_focas_df_c1_cuda_fi_coulomb(
    int nmo, int ndoc, int nact, long long nQ, const double *int1,
    const double *int2, double *fock_occ, double *fock_ext, int q_chunk,
    int max_devices, int verbose) {
  if (nmo <= 0 || ndoc < 0 || nact < 0 || nQ <= 0 || int1 == nullptr ||
      int2 == nullptr || fock_occ == nullptr || fock_ext == nullptr ||
      ndoc + nact > nmo) {
    return 120;
  }
  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  std::vector<double> qvec(static_cast<std::size_t>(nQ), 0.0);
  build_fi_coulomb_vector(ndoc, nQ, ngem, int2, qvec);
  std::vector<double> pair_total(static_cast<std::size_t>(ngem), 0.0);
  const int status = run_pair_workers(
      nQ, ngem, max_devices, verbose, "Fi Coulomb",
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_coulomb_on_device(dev, q_begin, q_end, ngem, int2,
                                         qvec.data(), q_chunk, verbose,
                                         "Fi Coulomb", partial);
      },
      pair_total);
  if (status != 0) {
    return status;
  }
  scatter_c1_coulomb(nmo, ndoc, nact, pair_total, int1, fock_occ, fock_ext);
  return 0;
}

extern "C" int hilbert_focas_df_c1_cuda_fa_coulomb(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    const double *den1, double *fock_occ, double *fock_ext, int q_chunk,
    int max_devices, int verbose) {
  if (nmo <= 0 || ndoc < 0 || nact < 0 || nQ <= 0 || int2 == nullptr ||
      den1 == nullptr || fock_occ == nullptr || fock_ext == nullptr ||
      ndoc + nact > nmo) {
    return 130;
  }
  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  std::vector<double> qvec(static_cast<std::size_t>(nQ), 0.0);
  build_fa_coulomb_vector(ndoc, nact, nQ, ngem, int2, den1, qvec);
  std::vector<double> pair_total(static_cast<std::size_t>(ngem), 0.0);
  const int status = run_pair_workers(
      nQ, ngem, max_devices, verbose, "Fa Coulomb",
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_coulomb_on_device(dev, q_begin, q_end, ngem, int2,
                                         qvec.data(), q_chunk, verbose,
                                         "Fa Coulomb", partial);
      },
      pair_total);
  if (status != 0) {
    return status;
  }
  scatter_c1_coulomb(nmo, ndoc, nact, pair_total, nullptr, fock_occ, fock_ext);
  return 0;
}

extern "C" int hilbert_focas_df_c1_cuda_q(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    const double *den2, double *q_out, int q_chunk, int max_devices,
    int verbose) {
  if (nmo <= 0 || ndoc < 0 || nact < 0 || nQ <= 0 || int2 == nullptr ||
      den2 == nullptr || q_out == nullptr || ndoc + nact > nmo) {
    return 100;
  }
  if (nact == 0) {
    return 0;
  }

  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  const long long ngem_act = static_cast<long long>(nact) * (nact + 1) / 2;
  const int devices = device_count_from_request(max_devices);
  if (devices <= 0) {
    return 101;
  }
  if (verbose) {
    std::fprintf(stderr,
                 "Hilbert FOCAS CUDA: using %d CUDA device(s) for C1 DF Q "
                 "contraction%s\n",
                 devices, max_devices <= 0 ? " (all visible)" : "");
  }

  std::vector<double> scaled_d2(static_cast<std::size_t>(ngem_act) * ngem_act,
                                0.0);
  build_scaled_c1_d2(nact, den2, scaled_d2);

  std::vector<std::thread> workers;
  std::vector<int> statuses(devices, 0);
  std::vector<std::vector<double>> partials(
      devices, std::vector<double>(static_cast<std::size_t>(nact) * nmo, 0.0));

  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share = (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] = compute_q_on_device(dev, nmo, ndoc, nact, nQ, q_begin,
                                           q_end, ngem, int2, scaled_d2, nullptr,
                                           q_chunk, verbose, partials[dev]);
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }
  for (int status : statuses) {
    if (status != 0) {
      return status;
    }
  }

  const std::size_t q_size = static_cast<std::size_t>(nact) * nmo;
  std::fill(q_out, q_out + q_size, 0.0);
  for (const auto &partial : partials) {
    for (std::size_t i = 0; i < q_size; ++i) {
      q_out[i] += partial[i];
    }
  }
  return 0;
}

// Symmetry-general Q contraction. The scaled, symmetry-blocked 2-RDM
// (scaled_d2, ngem_act x ngem_act in local active-pair packing) and the
// df-order active-orbital list (act_df) are assembled on the host by Fortran;
// the device runs the same two dense DGEMMs as the C1 path. q_out is returned
// as [nact x nmo] with the orbital (column) index in df order; the caller
// remaps it to class order.
extern "C" int hilbert_focas_df_sym_cuda_q(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    const double *scaled_d2_in, const int *act_df, double *q_out, int q_chunk,
    int max_devices, int verbose) {
  if (nmo <= 0 || ndoc < 0 || nact < 0 || nQ <= 0 || int2 == nullptr ||
      scaled_d2_in == nullptr || act_df == nullptr || q_out == nullptr ||
      ndoc + nact > nmo) {
    return 100;
  }
  if (nact == 0) {
    return 0;
  }

  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  const long long ngem_act = static_cast<long long>(nact) * (nact + 1) / 2;
  const int devices = device_count_from_request(max_devices);
  if (devices <= 0) {
    return 101;
  }
  if (verbose) {
    std::fprintf(stderr,
                 "Hilbert FOCAS CUDA: using %d CUDA device(s) for sym DF Q "
                 "contraction%s\n",
                 devices, max_devices <= 0 ? " (all visible)" : "");
  }

  std::vector<double> scaled_d2(
      scaled_d2_in,
      scaled_d2_in + static_cast<std::size_t>(ngem_act) * ngem_act);

  std::vector<std::thread> workers;
  std::vector<int> statuses(devices, 0);
  std::vector<std::vector<double>> partials(
      devices, std::vector<double>(static_cast<std::size_t>(nact) * nmo, 0.0));

  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share = (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] = compute_q_on_device(dev, nmo, ndoc, nact, nQ, q_begin,
                                           q_end, ngem, int2, scaled_d2, act_df,
                                           q_chunk, verbose, partials[dev]);
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }
  for (int status : statuses) {
    if (status != 0) {
      return status;
    }
  }

  const std::size_t q_size = static_cast<std::size_t>(nact) * nmo;
  std::fill(q_out, q_out + q_size, 0.0);
  for (const auto &partial : partials) {
    for (std::size_t i = 0; i < q_size; ++i) {
      q_out[i] += partial[i];
    }
  }
  return 0;
}
