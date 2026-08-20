#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace {

__device__ __forceinline__ int64_t ab_index(int i, int j, int n) {
  return static_cast<int64_t>(j) * n + i;
}

__device__ __forceinline__ int aa_sign(int i, int j) {
  return (i < j) ? 1 : -1;
}

__device__ __forceinline__ int64_t aa_index(int i, int j, int n) {
  if (i > j) {
    int tmp = i;
    i = j;
    j = tmp;
  }
  return static_cast<int64_t>(j) * (j - 1) / 2 + i;
}

__device__ __forceinline__ void aa_pair(int64_t idx, int &i, int &j) {
  double root = sqrt(static_cast<double>(1 + 8 * idx));
  j = static_cast<int>((1.0 + root) * 0.5);
  while (static_cast<int64_t>(j) * (j - 1) / 2 > idx) {
    --j;
  }
  while (static_cast<int64_t>(j + 1) * j / 2 <= idx) {
    ++j;
  }
  i = static_cast<int>(idx - static_cast<int64_t>(j) * (j - 1) / 2);
}

__device__ __forceinline__ double mat_get(const double *x, int64_t offset,
                                          int64_t dim, int64_t row,
                                          int64_t col) {
  return x[offset + row * dim + col];
}

__global__ void cg_update_y_r_kernel(
    double *__restrict__ y, double *__restrict__ r,
    const double *__restrict__ p, const double *__restrict__ Ap,
    const double *__restrict__ alpha_ptr, double *__restrict__ rr_out,
    int64_t n) {
  extern __shared__ double shared[];
  const double alpha = alpha_ptr[0];
  const int tid = threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;

  double sum = 0.0;
  for (; idx < n; idx += stride) {
    const double p_i = p[idx];
    const double r_new = r[idx] - alpha * Ap[idx];
    y[idx] += alpha * p_i;
    r[idx] = r_new;
    sum += r_new * r_new;
  }

  shared[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(rr_out, shared[0]);
  }
}

__global__ void cg_update_p_kernel(double *__restrict__ p,
                                   const double *__restrict__ r,
                                   const double *__restrict__ beta_ptr,
                                   int64_t n) {
  const double beta = beta_ptr[0];
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; idx < n; idx += stride) {
    p[idx] = r[idx] + beta * p[idx];
  }
}

__global__ void dqg_c1_au_kernel(
    const double *__restrict__ x, double *__restrict__ out, int n, double na,
    double nb, int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
    int64_t d200off, int64_t d1aoff, int64_t d1boff, int64_t q1aoff,
    int64_t q1boff, int64_t q2aboff, int64_t q2aaoff, int64_t q2bboff,
    int64_t g2aboff, int64_t g2baoff, int64_t g2aaoff, int64_t n_dual) {

  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n_dual) {
    return;
  }

  const int64_t gab = static_cast<int64_t>(n) * n;
  const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
  const int64_t n2 = static_cast<int64_t>(n) * n;
  const int64_t gab2 = gab * gab;
  const int64_t gaa2 = gaa * gaa;
  const bool spin_singlet = fabs(na - nb) < 1.0e-12;

  int64_t pos = 0;

  if (row < 3) {
    double value = 0.0;
    if (row == 0) {
      for (int64_t ij = 0; ij < gab; ++ij) {
        value += mat_get(x, d2aboff, gab, ij, ij);
      }
    } else if (row == 1) {
      for (int64_t ij = 0; ij < gaa; ++ij) {
        value += 2.0 * mat_get(x, d2aaoff, gaa, ij, ij);
      }
    } else {
      for (int64_t ij = 0; ij < gaa; ++ij) {
        value += 2.0 * mat_get(x, d2bboff, gaa, ij, ij);
      }
    }
    out[row] = value;
    return;
  }
  pos += 3;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    out[row] = mat_get(x, d2aaoff, gaa, ij, kl) -
               mat_get(x, d2aaoff, gaa, kl, ij);
    return;
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    out[row] = mat_get(x, d2bboff, gaa, ij, kl) -
               mat_get(x, d2bboff, gaa, kl, ij);
    return;
  }
  pos += gaa2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    out[row] = mat_get(x, d2aboff, gab, ij, kl) -
               mat_get(x, d2aboff, gab, kl, ij);
    return;
  }
  pos += gab2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    out[row] = mat_get(x, d1aoff, n, j, i) + mat_get(x, q1aoff, n, i, j);
    return;
  }
  pos += n2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    out[row] = mat_get(x, d1boff, n, j, i) + mat_get(x, q1boff, n, i, j);
    return;
  }
  pos += n2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    double value = nb * mat_get(x, d1aoff, n, i, j);
    for (int k = 0; k < n; ++k) {
      value -= mat_get(x, d2aboff, gab, ab_index(i, k, n),
                       ab_index(j, k, n));
    }
    out[row] = value;
    return;
  }
  pos += n2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    double value = na * mat_get(x, d1boff, n, i, j);
    for (int k = 0; k < n; ++k) {
      value -= mat_get(x, d2aboff, gab, ab_index(k, i, n),
                       ab_index(k, j, n));
    }
    out[row] = value;
    return;
  }
  pos += n2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    double value = (na - 1.0) * mat_get(x, d1aoff, n, i, j);
    for (int k = 0; k < n; ++k) {
      if (i == k || j == k) {
        continue;
      }
      int sik = aa_sign(i, k);
      int sjk = aa_sign(j, k);
      value -= static_cast<double>(sik * sjk) *
               mat_get(x, d2aaoff, gaa, aa_index(i, k, n),
                       aa_index(j, k, n));
    }
    out[row] = value;
    return;
  }
  pos += n2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    double value = (nb - 1.0) * mat_get(x, d1boff, n, i, j);
    for (int k = 0; k < n; ++k) {
      if (i == k || j == k) {
        continue;
      }
      int sik = aa_sign(i, k);
      int sjk = aa_sign(j, k);
      value -= static_cast<double>(sik * sjk) *
               mat_get(x, d2bboff, gaa, aa_index(i, k, n),
                       aa_index(j, k, n));
    }
    out[row] = value;
    return;
  }
  pos += n2;

  if (row == pos) {
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        value += mat_get(x, d2aboff, gab, ab_index(i, j, n),
                         ab_index(j, i, n));
      }
    }
    out[row] = value;
    return;
  }
  pos += 1;

  if (spin_singlet) {
  if (row < pos + n2) {
    int64_t local = row - pos;
    out[row] = x[d1aoff + local] - x[d1boff + local];
    return;
  }
  pos += n2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    out[row] = x[d2aaoff + local] - x[d2bboff + local];
    return;
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    int i, j, k, l;
    aa_pair(ij, i, j);
    aa_pair(kl, k, l);
    int64_t f_ij = ab_index(i, j, n);
    int64_t r_ij = ab_index(j, i, n);
    int64_t f_kl = ab_index(k, l, n);
    int64_t r_kl = ab_index(l, k, n);
    out[row] = mat_get(x, d2aaoff, gaa, ij, kl) -
               0.5 * mat_get(x, d2aboff, gab, f_ij, f_kl) +
               0.5 * mat_get(x, d2aboff, gab, r_ij, f_kl) +
               0.5 * mat_get(x, d2aboff, gab, f_ij, r_kl) -
               0.5 * mat_get(x, d2aboff, gab, r_ij, r_kl);
    return;
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    int i, j, k, l;
    aa_pair(ij, i, j);
    aa_pair(kl, k, l);
    int64_t f_ij = ab_index(i, j, n);
    int64_t r_ij = ab_index(j, i, n);
    int64_t f_kl = ab_index(k, l, n);
    int64_t r_kl = ab_index(l, k, n);
    out[row] = mat_get(x, d2bboff, gaa, ij, kl) -
               0.5 * mat_get(x, d2aboff, gab, f_ij, f_kl) +
               0.5 * mat_get(x, d2aboff, gab, r_ij, f_kl) +
               0.5 * mat_get(x, d2aboff, gab, f_ij, r_kl) -
               0.5 * mat_get(x, d2aboff, gab, r_ij, r_kl);
    return;
  }
  pos += gaa2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    int i = static_cast<int>(ij % n);
    int j = static_cast<int>(ij / n);
    int k = static_cast<int>(kl % n);
    int l = static_cast<int>(kl / n);
    int64_t ji = ab_index(j, i, n);
    int64_t lk = ab_index(l, k, n);
    double dij = (i == j) ? sqrt(2.0) : 1.0;
    double dkl = (k == l) ? sqrt(2.0) : 1.0;
    double v = -0.5 / (dij * dkl);
    out[row] = mat_get(x, d200off, gab, ij, kl) +
               v * (mat_get(x, d2aboff, gab, ij, kl) +
                    mat_get(x, d2aboff, gab, ji, kl) +
                    mat_get(x, d2aboff, gab, ij, lk) +
                    mat_get(x, d2aboff, gab, ji, lk));
    return;
  }
  pos += gab2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    int i = static_cast<int>(ij % n);
    int j = static_cast<int>(ij / n);
    int k = static_cast<int>(kl % n);
    int l = static_cast<int>(kl / n);
    out[row] = mat_get(x, d2aboff, gab, ij, kl) -
               mat_get(x, d2aboff, gab, ab_index(j, i, n),
                       ab_index(l, k, n));
    return;
  }
  pos += gab2;

  } else {
    if (row < pos + 4 * gab2) {
      int64_t local = row - pos;
      int64_t dim = 2 * gab;
      int64_t drow = local / dim;
      int64_t dcol = local % dim;
      int64_t ij = (drow < gab) ? drow : drow - gab;
      int64_t kl = (dcol < gab) ? dcol : dcol - gab;
      int i = static_cast<int>(ij % n);
      int j = static_cast<int>(ij / n);
      int k = static_cast<int>(kl % n);
      int l = static_cast<int>(kl / n);
      int64_t ji = ab_index(j, i, n);
      int64_t lk = ab_index(l, k, n);
      double dij = (i == j) ? sqrt(2.0) : 1.0;
      double dkl = (k == l) ? sqrt(2.0) : 1.0;
      double value = mat_get(x, d200off, dim, drow, dcol);
      if (drow < gab && dcol < gab) {
        double v = -0.5 / (dij * dkl);
        value += v * (mat_get(x, d2aboff, gab, ij, kl) +
                      mat_get(x, d2aboff, gab, ji, kl) +
                      mat_get(x, d2aboff, gab, ij, lk) +
                      mat_get(x, d2aboff, gab, ji, lk));
      } else if (drow < gab) {
        double v = 0.5 / dij;
        value += -v * mat_get(x, d2aboff, gab, ij, kl);
        value +=  v * mat_get(x, d2aboff, gab, ij, lk);
        value += -v * mat_get(x, d2aboff, gab, ji, kl);
        value +=  v * mat_get(x, d2aboff, gab, ji, lk);
      } else if (dcol < gab) {
        double v = 0.5 / dkl;
        value += -v * mat_get(x, d2aboff, gab, ij, kl);
        value += -v * mat_get(x, d2aboff, gab, ij, lk);
        value +=  v * mat_get(x, d2aboff, gab, ji, kl);
        value +=  v * mat_get(x, d2aboff, gab, ji, lk);
      } else {
        value += -0.5 * mat_get(x, d2aboff, gab, ij, kl);
        value +=  0.5 * mat_get(x, d2aboff, gab, ji, kl);
        value +=  0.5 * mat_get(x, d2aboff, gab, ij, lk);
        value += -0.5 * mat_get(x, d2aboff, gab, ji, lk);
      }
      out[row] = value;
      return;
    }
    pos += 4 * gab2;
  }

  if (row < pos + gab) {
    int64_t kl = row - pos;
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
      int64_t ii = ab_index(i, i, n);
      value += mat_get(x, g2baoff, gab, kl, ii);
    }
    out[row] = value;
    return;
  }
  pos += gab;

  if (row < pos + gab) {
    int64_t kl = row - pos;
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
      int64_t ii = ab_index(i, i, n);
      value += mat_get(x, g2baoff, gab, ii, kl);
    }
    out[row] = value;
    return;
  }
  pos += gab;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    int i = static_cast<int>(ij % n);
    int j = static_cast<int>(ij / n);
    int k = static_cast<int>(kl % n);
    int l = static_cast<int>(kl / n);
    double value = mat_get(x, d2aboff, gab, ij, kl) -
                   mat_get(x, q2aboff, gab, ij, kl);
    if (j == l) {
      value -= mat_get(x, d1aoff, n, k, i);
    }
    if (i == k) {
      value -= mat_get(x, d1boff, n, l, j);
    }
    out[row] = value;
    return;
  }
  pos += gab2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    int i, j, k, l;
    aa_pair(ij, i, j);
    aa_pair(kl, k, l);
    double value = mat_get(x, d2aaoff, gaa, ij, kl) -
                   mat_get(x, q2aaoff, gaa, ij, kl);
    if (j == l) value -= mat_get(x, d1aoff, n, k, i);
    if (j == k) value += mat_get(x, d1aoff, n, l, i);
    if (i == l) value += mat_get(x, d1aoff, n, k, j);
    if (i == k) value -= mat_get(x, d1aoff, n, l, j);
    out[row] = value;
    return;
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    int i, j, k, l;
    aa_pair(ij, i, j);
    aa_pair(kl, k, l);
    double value = mat_get(x, d2bboff, gaa, ij, kl) -
                   mat_get(x, q2bboff, gaa, ij, kl);
    if (j == l) value -= mat_get(x, d1boff, n, k, i);
    if (j == k) value += mat_get(x, d1boff, n, l, i);
    if (i == l) value += mat_get(x, d1boff, n, k, j);
    if (i == k) value -= mat_get(x, d1boff, n, l, j);
    out[row] = value;
    return;
  }
  pos += gaa2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    int i = static_cast<int>(ij % n);
    int j = static_cast<int>(ij / n);
    int k = static_cast<int>(kl % n);
    int l = static_cast<int>(kl / n);
    double value = -mat_get(x, g2aboff, gab, ij, kl);
    if (j == l) {
      value += mat_get(x, d1aoff, n, i, k);
    }
    value -= mat_get(x, d2aboff, gab, ab_index(i, l, n),
                     ab_index(k, j, n));
    out[row] = value;
    return;
  }
  pos += gab2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    int i = static_cast<int>(ij % n);
    int j = static_cast<int>(ij / n);
    int k = static_cast<int>(kl % n);
    int l = static_cast<int>(kl / n);
    double value = -mat_get(x, g2baoff, gab, ij, kl);
    if (j == l) {
      value += mat_get(x, d1boff, n, i, k);
    }
    // Mirrors d2ab4.permute(1, 2, 3, 0): T[i,j,k,l] <- D2ab[l,i,j,k].
    value -= mat_get(x, d2aboff, gab, ab_index(l, i, n),
                     ab_index(j, k, n));
    out[row] = value;
    return;
  }
  pos += gab2;

  int64_t local = row - pos;
  int64_t dim = 2 * gab;
  int64_t a = local / dim;
  int64_t b = local % dim;

  if (a < gab && b < gab) {
    int i = static_cast<int>(a % n);
    int j = static_cast<int>(a / n);
    int k = static_cast<int>(b % n);
    int l = static_cast<int>(b / n);
    double value = -mat_get(x, g2aaoff, dim, a, b);
    if (j == l) {
      value += mat_get(x, d1aoff, n, i, k);
    }
    if (i != l && k != j) {
      value -= static_cast<double>(aa_sign(i, l) * aa_sign(k, j)) *
               mat_get(x, d2aaoff, gaa, aa_index(i, l, n),
                       aa_index(k, j, n));
    }
    out[row] = value;
    return;
  }

  if (a >= gab && b >= gab) {
    int64_t aa = a - gab;
    int64_t bb = b - gab;
    int i = static_cast<int>(aa % n);
    int j = static_cast<int>(aa / n);
    int k = static_cast<int>(bb % n);
    int l = static_cast<int>(bb / n);
    double value = -mat_get(x, g2aaoff, dim, a, b);
    if (j == l) {
      value += mat_get(x, d1boff, n, i, k);
    }
    if (i != l && k != j) {
      value -= static_cast<double>(aa_sign(i, l) * aa_sign(k, j)) *
               mat_get(x, d2bboff, gaa, aa_index(i, l, n),
                       aa_index(k, j, n));
    }
    out[row] = value;
    return;
  }

  if (a < gab) {
    int64_t bb = b - gab;
    int i = static_cast<int>(a % n);
    int j = static_cast<int>(a / n);
    int k = static_cast<int>(bb % n);
    int l = static_cast<int>(bb / n);
    // Mirrors d2ab4.permute(0, 2, 3, 1): T[i,j,k,l] <- D2ab[i,l,j,k].
    out[row] = -mat_get(x, g2aaoff, dim, a, b) +
               mat_get(x, d2aboff, gab, ab_index(i, l, n),
                       ab_index(j, k, n));
    return;
  }

  int64_t aa = a - gab;
  int i = static_cast<int>(aa % n);
  int j = static_cast<int>(aa / n);
  int k = static_cast<int>(b % n);
  int l = static_cast<int>(b / n);
  // Mirrors d2ab4.permute(1, 3, 2, 0): T[i,j,k,l] <- D2ab[l,i,k,j].
  out[row] = -mat_get(x, g2aaoff, dim, a, b) +
             mat_get(x, d2aboff, gab, ab_index(l, i, n),
                     ab_index(k, j, n));
}

__device__ __forceinline__ void atomic_emit(double *out, int64_t col,
                                            double value) {
  if (value != 0.0) {
    atomicAdd(out + col, value);
  }
}

__global__ void dqg_c1_atu_kernel(
    const double *__restrict__ y, double *__restrict__ out, int n, double na,
    double nb, int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
    int64_t d200off, int64_t d1aoff, int64_t d1boff, int64_t q1aoff,
    int64_t q1boff, int64_t q2aboff, int64_t q2aaoff, int64_t q2bboff,
    int64_t g2aboff, int64_t g2baoff, int64_t g2aaoff, int64_t row_start,
    int64_t row_count) {

  int64_t row_local =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row_local >= row_count) {
    return;
  }
  int64_t row = row_start + row_local;

  const double yr = y[row_local];
  if (yr == 0.0) {
    return;
  }

  const int64_t gab = static_cast<int64_t>(n) * n;
  const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
  const int64_t n2 = static_cast<int64_t>(n) * n;
  const int64_t gab2 = gab * gab;
  const int64_t gaa2 = gaa * gaa;
  const bool spin_singlet = fabs(na - nb) < 1.0e-12;

  int64_t pos = 0;

  if (row < 3) {
    if (row == 0) {
      for (int64_t ij = 0; ij < gab; ++ij) {
        atomic_emit(out, d2aboff + ij * gab + ij, yr);
      }
    } else if (row == 1) {
      for (int64_t ij = 0; ij < gaa; ++ij) {
        atomic_emit(out, d2aaoff + ij * gaa + ij, 2.0 * yr);
      }
    } else {
      for (int64_t ij = 0; ij < gaa; ++ij) {
        atomic_emit(out, d2bboff + ij * gaa + ij, 2.0 * yr);
      }
    }
    return;
  }
  pos += 3;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    atomic_emit(out, d2aaoff + ij * gaa + kl, yr);
    atomic_emit(out, d2aaoff + kl * gaa + ij, -yr);
    return;
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    atomic_emit(out, d2bboff + ij * gaa + kl, yr);
    atomic_emit(out, d2bboff + kl * gaa + ij, -yr);
    return;
  }
  pos += gaa2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    atomic_emit(out, d2aboff + ij * gab + kl, yr);
    atomic_emit(out, d2aboff + kl * gab + ij, -yr);
    return;
  }
  pos += gab2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    atomic_emit(out, d1aoff + j * n + i, yr);
    atomic_emit(out, q1aoff + i * n + j, yr);
    return;
  }
  pos += n2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    atomic_emit(out, d1boff + j * n + i, yr);
    atomic_emit(out, q1boff + i * n + j, yr);
    return;
  }
  pos += n2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    atomic_emit(out, d1aoff + i * n + j, nb * yr);
    for (int k = 0; k < n; ++k) {
      atomic_emit(out,
                  d2aboff + ab_index(i, k, n) * gab + ab_index(j, k, n),
                  -yr);
    }
    return;
  }
  pos += n2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    atomic_emit(out, d1boff + i * n + j, na * yr);
    for (int k = 0; k < n; ++k) {
      atomic_emit(out,
                  d2aboff + ab_index(k, i, n) * gab + ab_index(k, j, n),
                  -yr);
    }
    return;
  }
  pos += n2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    atomic_emit(out, d1aoff + i * n + j, (na - 1.0) * yr);
    for (int k = 0; k < n; ++k) {
      if (i == k || j == k) {
        continue;
      }
      int sik = aa_sign(i, k);
      int sjk = aa_sign(j, k);
      atomic_emit(out,
                  d2aaoff + aa_index(i, k, n) * gaa + aa_index(j, k, n),
                  -static_cast<double>(sik * sjk) * yr);
    }
    return;
  }
  pos += n2;

  if (row < pos + n2) {
    int64_t local = row - pos;
    int i = static_cast<int>(local / n);
    int j = static_cast<int>(local % n);
    atomic_emit(out, d1boff + i * n + j, (nb - 1.0) * yr);
    for (int k = 0; k < n; ++k) {
      if (i == k || j == k) {
        continue;
      }
      int sik = aa_sign(i, k);
      int sjk = aa_sign(j, k);
      atomic_emit(out,
                  d2bboff + aa_index(i, k, n) * gaa + aa_index(j, k, n),
                  -static_cast<double>(sik * sjk) * yr);
    }
    return;
  }
  pos += n2;

  if (row == pos) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        atomic_emit(out,
                    d2aboff + ab_index(i, j, n) * gab + ab_index(j, i, n),
                    yr);
      }
    }
    return;
  }
  pos += 1;

  if (spin_singlet) {
  if (row < pos + n2) {
    int64_t local = row - pos;
    atomic_emit(out, d1aoff + local, yr);
    atomic_emit(out, d1boff + local, -yr);
    return;
  }
  pos += n2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    atomic_emit(out, d2aaoff + local, yr);
    atomic_emit(out, d2bboff + local, -yr);
    return;
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    int i, j, k, l;
    aa_pair(ij, i, j);
    aa_pair(kl, k, l);
    int64_t f_ij = ab_index(i, j, n);
    int64_t r_ij = ab_index(j, i, n);
    int64_t f_kl = ab_index(k, l, n);
    int64_t r_kl = ab_index(l, k, n);
    atomic_emit(out, d2aaoff + ij * gaa + kl, yr);
    atomic_emit(out, d2aboff + f_ij * gab + f_kl, -0.5 * yr);
    atomic_emit(out, d2aboff + r_ij * gab + f_kl, 0.5 * yr);
    atomic_emit(out, d2aboff + f_ij * gab + r_kl, 0.5 * yr);
    atomic_emit(out, d2aboff + r_ij * gab + r_kl, -0.5 * yr);
    return;
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    int i, j, k, l;
    aa_pair(ij, i, j);
    aa_pair(kl, k, l);
    int64_t f_ij = ab_index(i, j, n);
    int64_t r_ij = ab_index(j, i, n);
    int64_t f_kl = ab_index(k, l, n);
    int64_t r_kl = ab_index(l, k, n);
    atomic_emit(out, d2bboff + ij * gaa + kl, yr);
    atomic_emit(out, d2aboff + f_ij * gab + f_kl, -0.5 * yr);
    atomic_emit(out, d2aboff + r_ij * gab + f_kl, 0.5 * yr);
    atomic_emit(out, d2aboff + f_ij * gab + r_kl, 0.5 * yr);
    atomic_emit(out, d2aboff + r_ij * gab + r_kl, -0.5 * yr);
    return;
  }
  pos += gaa2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    int i = static_cast<int>(ij % n);
    int j = static_cast<int>(ij / n);
    int k = static_cast<int>(kl % n);
    int l = static_cast<int>(kl / n);
    int64_t ji = ab_index(j, i, n);
    int64_t lk = ab_index(l, k, n);
    double dij = (i == j) ? sqrt(2.0) : 1.0;
    double dkl = (k == l) ? sqrt(2.0) : 1.0;
    double v = -0.5 / (dij * dkl);
    atomic_emit(out, d200off + ij * gab + kl, yr);
    atomic_emit(out, d2aboff + ij * gab + kl, v * yr);
    atomic_emit(out, d2aboff + ji * gab + kl, v * yr);
    atomic_emit(out, d2aboff + ij * gab + lk, v * yr);
    atomic_emit(out, d2aboff + ji * gab + lk, v * yr);
    return;
  }
  pos += gab2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    int i = static_cast<int>(ij % n);
    int j = static_cast<int>(ij / n);
    int k = static_cast<int>(kl % n);
    int l = static_cast<int>(kl / n);
    atomic_emit(out, d2aboff + ij * gab + kl, yr);
    atomic_emit(out, d2aboff + ab_index(j, i, n) * gab + ab_index(l, k, n),
                -yr);
    return;
  }
  pos += gab2;

  } else {
    if (row < pos + 4 * gab2) {
      int64_t local = row - pos;
      int64_t dim = 2 * gab;
      int64_t drow = local / dim;
      int64_t dcol = local % dim;
      int64_t ij = (drow < gab) ? drow : drow - gab;
      int64_t kl = (dcol < gab) ? dcol : dcol - gab;
      int i = static_cast<int>(ij % n);
      int j = static_cast<int>(ij / n);
      int k = static_cast<int>(kl % n);
      int l = static_cast<int>(kl / n);
      int64_t ji = ab_index(j, i, n);
      int64_t lk = ab_index(l, k, n);
      double dij = (i == j) ? sqrt(2.0) : 1.0;
      double dkl = (k == l) ? sqrt(2.0) : 1.0;

      atomic_emit(out, d200off + drow * dim + dcol, yr);
      if (drow < gab && dcol < gab) {
        double v = -0.5 / (dij * dkl);
        atomic_emit(out, d2aboff + ij * gab + kl, v * yr);
        atomic_emit(out, d2aboff + ji * gab + kl, v * yr);
        atomic_emit(out, d2aboff + ij * gab + lk, v * yr);
        atomic_emit(out, d2aboff + ji * gab + lk, v * yr);
      } else if (drow < gab) {
        double v = 0.5 / dij;
        atomic_emit(out, d2aboff + ij * gab + kl, -v * yr);
        atomic_emit(out, d2aboff + ij * gab + lk,  v * yr);
        atomic_emit(out, d2aboff + ji * gab + kl, -v * yr);
        atomic_emit(out, d2aboff + ji * gab + lk,  v * yr);
      } else if (dcol < gab) {
        double v = 0.5 / dkl;
        atomic_emit(out, d2aboff + ij * gab + kl, -v * yr);
        atomic_emit(out, d2aboff + ij * gab + lk, -v * yr);
        atomic_emit(out, d2aboff + ji * gab + kl,  v * yr);
        atomic_emit(out, d2aboff + ji * gab + lk,  v * yr);
      } else {
        atomic_emit(out, d2aboff + ij * gab + kl, -0.5 * yr);
        atomic_emit(out, d2aboff + ji * gab + kl,  0.5 * yr);
        atomic_emit(out, d2aboff + ij * gab + lk,  0.5 * yr);
        atomic_emit(out, d2aboff + ji * gab + lk, -0.5 * yr);
      }
      return;
    }
    pos += 4 * gab2;
  }

  if (row < pos + gab) {
    int64_t kl = row - pos;
    for (int i = 0; i < n; ++i) {
      int64_t ii = ab_index(i, i, n);
      atomic_emit(out, g2baoff + kl * gab + ii, yr);
    }
    return;
  }
  pos += gab;

  if (row < pos + gab) {
    int64_t kl = row - pos;
    for (int i = 0; i < n; ++i) {
      int64_t ii = ab_index(i, i, n);
      atomic_emit(out, g2baoff + ii * gab + kl, yr);
    }
    return;
  }
  pos += gab;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    int i = static_cast<int>(ij % n);
    int j = static_cast<int>(ij / n);
    int k = static_cast<int>(kl % n);
    int l = static_cast<int>(kl / n);
    atomic_emit(out, d2aboff + ij * gab + kl, yr);
    atomic_emit(out, q2aboff + ij * gab + kl, -yr);
    if (j == l) {
      atomic_emit(out, d1aoff + k * n + i, -yr);
    }
    if (i == k) {
      atomic_emit(out, d1boff + l * n + j, -yr);
    }
    return;
  }
  pos += gab2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    int i, j, k, l;
    aa_pair(ij, i, j);
    aa_pair(kl, k, l);
    atomic_emit(out, d2aaoff + ij * gaa + kl, yr);
    atomic_emit(out, q2aaoff + ij * gaa + kl, -yr);
    if (j == l) atomic_emit(out, d1aoff + k * n + i, -yr);
    if (j == k) atomic_emit(out, d1aoff + l * n + i, yr);
    if (i == l) atomic_emit(out, d1aoff + k * n + j, yr);
    if (i == k) atomic_emit(out, d1aoff + l * n + j, -yr);
    return;
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    int i, j, k, l;
    aa_pair(ij, i, j);
    aa_pair(kl, k, l);
    atomic_emit(out, d2bboff + ij * gaa + kl, yr);
    atomic_emit(out, q2bboff + ij * gaa + kl, -yr);
    if (j == l) atomic_emit(out, d1boff + k * n + i, -yr);
    if (j == k) atomic_emit(out, d1boff + l * n + i, yr);
    if (i == l) atomic_emit(out, d1boff + k * n + j, yr);
    if (i == k) atomic_emit(out, d1boff + l * n + j, -yr);
    return;
  }
  pos += gaa2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    int i = static_cast<int>(ij % n);
    int j = static_cast<int>(ij / n);
    int k = static_cast<int>(kl % n);
    int l = static_cast<int>(kl / n);
    atomic_emit(out, g2aboff + ij * gab + kl, -yr);
    if (j == l) {
      atomic_emit(out, d1aoff + i * n + k, yr);
    }
    atomic_emit(out, d2aboff + ab_index(i, l, n) * gab + ab_index(k, j, n),
                -yr);
    return;
  }
  pos += gab2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    int i = static_cast<int>(ij % n);
    int j = static_cast<int>(ij / n);
    int k = static_cast<int>(kl % n);
    int l = static_cast<int>(kl / n);
    atomic_emit(out, g2baoff + ij * gab + kl, -yr);
    if (j == l) {
      atomic_emit(out, d1boff + i * n + k, yr);
    }
    atomic_emit(out, d2aboff + ab_index(l, i, n) * gab + ab_index(j, k, n),
                -yr);
    return;
  }
  pos += gab2;

  int64_t local = row - pos;
  int64_t dim = 2 * gab;
  int64_t a = local / dim;
  int64_t b = local % dim;

  if (a < gab && b < gab) {
    int i = static_cast<int>(a % n);
    int j = static_cast<int>(a / n);
    int k = static_cast<int>(b % n);
    int l = static_cast<int>(b / n);
    atomic_emit(out, g2aaoff + a * dim + b, -yr);
    if (j == l) {
      atomic_emit(out, d1aoff + i * n + k, yr);
    }
    if (i != l && k != j) {
      int sil = aa_sign(i, l);
      int skj = aa_sign(k, j);
      atomic_emit(out,
                  d2aaoff + aa_index(i, l, n) * gaa + aa_index(k, j, n),
                  -static_cast<double>(sil * skj) * yr);
    }
    return;
  }

  if (a >= gab && b >= gab) {
    int64_t aa = a - gab;
    int64_t bb = b - gab;
    int i = static_cast<int>(aa % n);
    int j = static_cast<int>(aa / n);
    int k = static_cast<int>(bb % n);
    int l = static_cast<int>(bb / n);
    atomic_emit(out, g2aaoff + a * dim + b, -yr);
    if (j == l) {
      atomic_emit(out, d1boff + i * n + k, yr);
    }
    if (i != l && k != j) {
      int sil = aa_sign(i, l);
      int skj = aa_sign(k, j);
      atomic_emit(out,
                  d2bboff + aa_index(i, l, n) * gaa + aa_index(k, j, n),
                  -static_cast<double>(sil * skj) * yr);
    }
    return;
  }

  if (a < gab) {
    int64_t bb = b - gab;
    int i = static_cast<int>(a % n);
    int j = static_cast<int>(a / n);
    int k = static_cast<int>(bb % n);
    int l = static_cast<int>(bb / n);
    atomic_emit(out, g2aaoff + a * dim + b, -yr);
    atomic_emit(out, d2aboff + ab_index(i, l, n) * gab + ab_index(j, k, n),
                yr);
    return;
  }

  int64_t aa = a - gab;
  int i = static_cast<int>(aa % n);
  int j = static_cast<int>(aa / n);
  int k = static_cast<int>(b % n);
  int l = static_cast<int>(b / n);
  atomic_emit(out, g2aaoff + a * dim + b, -yr);
  atomic_emit(out, d2aboff + ab_index(l, i, n) * gab + ab_index(k, j, n),
              yr);
}

} // namespace

void dqg_c1_au_out(torch::Tensor x, torch::Tensor out, int64_t n, double na,
                   double nb, int64_t d2aboff, int64_t d2aaoff,
                   int64_t d2bboff, int64_t d200off, int64_t d1aoff,
                   int64_t d1boff, int64_t q1aoff, int64_t q1boff,
                   int64_t q2aboff, int64_t q2aaoff, int64_t q2bboff,
                   int64_t g2aboff, int64_t g2baoff, int64_t g2aaoff,
                   int64_t n_dual) {
  TORCH_CHECK(x.is_cuda(), "dqg_c1_au expects a CUDA tensor.");
  TORCH_CHECK(out.is_cuda(), "dqg_c1_au_out expects a CUDA output tensor.");
  TORCH_CHECK(x.device() == out.device(),
              "dqg_c1_au_out input and output must be on the same CUDA device.");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Double,
              "dqg_c1_au currently expects float64 input.");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Double,
              "dqg_c1_au_out currently expects float64 output.");
  TORCH_CHECK(x.is_contiguous(), "dqg_c1_au expects contiguous input.");
  TORCH_CHECK(out.is_contiguous(), "dqg_c1_au_out expects contiguous output.");
  TORCH_CHECK(out.numel() == n_dual, "dqg_c1_au_out output has wrong size.");
  TORCH_CHECK(n > 0 && n <= 4096, "dqg_c1_au received invalid n.");

  constexpr int threads = 256;
  int64_t blocks = (n_dual + threads - 1) / threads;
  dqg_c1_au_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<double>(), out.data_ptr<double>(), static_cast<int>(n), na, nb,
      d2aboff, d2aaoff, d2bboff, d200off, d1aoff, d1boff, q1aoff, q1boff,
      q2aboff, q2aaoff, q2bboff, g2aboff, g2baoff, g2aaoff, n_dual);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor dqg_c1_au(torch::Tensor x, int64_t n, double na, double nb,
                        int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
                        int64_t d200off, int64_t d1aoff, int64_t d1boff,
                        int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
                        int64_t q2aaoff, int64_t q2bboff, int64_t g2aboff,
                        int64_t g2baoff, int64_t g2aaoff, int64_t n_dual) {
  auto out = torch::empty({n_dual}, x.options());
  dqg_c1_au_out(x, out, n, na, nb, d2aboff, d2aaoff, d2bboff, d200off,
                d1aoff, d1boff, q1aoff, q1boff, q2aboff, q2aaoff, q2bboff,
                g2aboff, g2baoff, g2aaoff, n_dual);
  return out;
}

void dqg_c1_atu_range_out(torch::Tensor y, torch::Tensor out,
                          int64_t row_start, int64_t row_stop, int64_t n,
                          double na, double nb, int64_t d2aboff,
                          int64_t d2aaoff, int64_t d2bboff,
                          int64_t d200off, int64_t d1aoff, int64_t d1boff,
                          int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
                          int64_t q2aaoff, int64_t q2bboff,
                          int64_t g2aboff, int64_t g2baoff,
                          int64_t g2aaoff, int64_t n_primal);

void dqg_c1_atu_range_add_out(torch::Tensor y, torch::Tensor out,
                              int64_t row_start, int64_t row_stop, int64_t n,
                              double na, double nb, int64_t d2aboff,
                              int64_t d2aaoff, int64_t d2bboff,
                              int64_t d200off, int64_t d1aoff, int64_t d1boff,
                              int64_t q1aoff, int64_t q1boff,
                              int64_t q2aboff, int64_t q2aaoff,
                              int64_t q2bboff, int64_t g2aboff,
                              int64_t g2baoff, int64_t g2aaoff,
                              int64_t n_primal);

void dqg_c1_atu_out(torch::Tensor y, torch::Tensor out, int64_t n, double na,
                    double nb, int64_t d2aboff, int64_t d2aaoff,
                    int64_t d2bboff, int64_t d200off, int64_t d1aoff,
                    int64_t d1boff, int64_t q1aoff, int64_t q1boff,
                    int64_t q2aboff, int64_t q2aaoff, int64_t q2bboff,
                    int64_t g2aboff, int64_t g2baoff, int64_t g2aaoff,
                    int64_t n_primal) {
  dqg_c1_atu_range_out(y, out, 0, y.numel(), n, na, nb, d2aboff, d2aaoff,
                       d2bboff, d200off, d1aoff, d1boff, q1aoff, q1boff,
                       q2aboff, q2aaoff, q2bboff, g2aboff, g2baoff,
                       g2aaoff, n_primal);
}

void dqg_c1_atu_range_out(torch::Tensor y, torch::Tensor out,
                          int64_t row_start, int64_t row_stop, int64_t n,
                          double na, double nb, int64_t d2aboff,
                          int64_t d2aaoff, int64_t d2bboff,
                          int64_t d200off, int64_t d1aoff, int64_t d1boff,
                          int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
                          int64_t q2aaoff, int64_t q2bboff,
                          int64_t g2aboff, int64_t g2baoff,
                          int64_t g2aaoff, int64_t n_primal) {
  out.zero_();
  dqg_c1_atu_range_add_out(y, out, row_start, row_stop, n, na, nb, d2aboff,
                           d2aaoff, d2bboff, d200off, d1aoff, d1boff,
                           q1aoff, q1boff, q2aboff, q2aaoff, q2bboff,
                           g2aboff, g2baoff, g2aaoff, n_primal);
}

void dqg_c1_atu_range_add_out(torch::Tensor y, torch::Tensor out,
                              int64_t row_start, int64_t row_stop, int64_t n,
                              double na, double nb, int64_t d2aboff,
                              int64_t d2aaoff, int64_t d2bboff,
                              int64_t d200off, int64_t d1aoff,
                              int64_t d1boff, int64_t q1aoff,
                              int64_t q1boff, int64_t q2aboff,
                              int64_t q2aaoff, int64_t q2bboff,
                              int64_t g2aboff, int64_t g2baoff,
                              int64_t g2aaoff, int64_t n_primal) {
  TORCH_CHECK(y.is_cuda(), "dqg_c1_atu expects a CUDA tensor.");
  TORCH_CHECK(out.is_cuda(), "dqg_c1_atu_out expects a CUDA output tensor.");
  TORCH_CHECK(y.device() == out.device(),
              "dqg_c1_atu_out input and output must be on the same CUDA device.");
  TORCH_CHECK(y.scalar_type() == at::ScalarType::Double,
              "dqg_c1_atu currently expects float64 input.");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Double,
              "dqg_c1_atu_out currently expects float64 output.");
  TORCH_CHECK(y.is_contiguous(), "dqg_c1_atu expects contiguous input.");
  TORCH_CHECK(out.is_contiguous(), "dqg_c1_atu_out expects contiguous output.");
  TORCH_CHECK(out.numel() == n_primal, "dqg_c1_atu_out output has wrong size.");
  TORCH_CHECK(n > 0 && n <= 4096, "dqg_c1_atu received invalid n.");
  TORCH_CHECK(row_start >= 0 && row_stop >= row_start,
              "dqg_c1_atu_range_out received an invalid row range.");
  TORCH_CHECK(y.numel() == row_stop - row_start,
              "dqg_c1_atu_range_out y chunk size does not match row range.");

  constexpr int threads = 256;
  int64_t row_count = y.numel();
  int64_t blocks = (row_count + threads - 1) / threads;
  dqg_c1_atu_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<double>(), out.data_ptr<double>(), static_cast<int>(n), na, nb,
      d2aboff, d2aaoff, d2bboff, d200off, d1aoff, d1boff, q1aoff, q1boff,
      q2aboff, q2aaoff, q2bboff, g2aboff, g2baoff, g2aaoff, row_start,
      row_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor dqg_c1_atu(torch::Tensor y, int64_t n, double na, double nb,
                         int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
                         int64_t d200off, int64_t d1aoff, int64_t d1boff,
                         int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
                         int64_t q2aaoff, int64_t q2bboff, int64_t g2aboff,
                         int64_t g2baoff, int64_t g2aaoff,
                         int64_t n_primal) {
  auto out = torch::empty({n_primal}, y.options());
  dqg_c1_atu_out(y, out, n, na, nb, d2aboff, d2aaoff, d2bboff, d200off,
                 d1aoff, d1boff, q1aoff, q1boff, q2aboff, q2aaoff, q2bboff,
                 g2aboff, g2baoff, g2aaoff, n_primal);
  return out;
}

torch::Tensor cg_update_y_r(torch::Tensor y, torch::Tensor r, torch::Tensor p,
                            torch::Tensor Ap, torch::Tensor alpha) {
  TORCH_CHECK(y.is_cuda() && r.is_cuda() && p.is_cuda() && Ap.is_cuda() &&
                  alpha.is_cuda(),
              "cg_update_y_r expects CUDA tensors.");
  TORCH_CHECK(y.device() == r.device() && y.device() == p.device() &&
                  y.device() == Ap.device() && y.device() == alpha.device(),
              "cg_update_y_r tensors must be on the same CUDA device.");
  TORCH_CHECK(y.scalar_type() == at::ScalarType::Double &&
                  r.scalar_type() == at::ScalarType::Double &&
                  p.scalar_type() == at::ScalarType::Double &&
                  Ap.scalar_type() == at::ScalarType::Double &&
                  alpha.scalar_type() == at::ScalarType::Double,
              "cg_update_y_r expects float64 tensors.");
  TORCH_CHECK(y.is_contiguous() && r.is_contiguous() && p.is_contiguous() &&
                  Ap.is_contiguous() && alpha.is_contiguous(),
              "cg_update_y_r expects contiguous tensors.");
  TORCH_CHECK(y.numel() == r.numel() && y.numel() == p.numel() &&
                  y.numel() == Ap.numel(),
              "cg_update_y_r vector sizes do not match.");
  TORCH_CHECK(alpha.numel() == 1, "cg_update_y_r alpha must be scalar.");

  auto rr = torch::zeros({1}, y.options());
  constexpr int threads = 256;
  const int64_t n = y.numel();
  const int64_t raw_blocks = (n + threads - 1) / threads;
  const int blocks = static_cast<int>(std::min<int64_t>(raw_blocks, 65535));
  cg_update_y_r_kernel<<<blocks, threads, threads * sizeof(double),
                         at::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<double>(), r.data_ptr<double>(), p.data_ptr<double>(),
      Ap.data_ptr<double>(), alpha.data_ptr<double>(), rr.data_ptr<double>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rr;
}

void cg_update_p(torch::Tensor p, torch::Tensor r, torch::Tensor beta) {
  TORCH_CHECK(p.is_cuda() && r.is_cuda() && beta.is_cuda(),
              "cg_update_p expects CUDA tensors.");
  TORCH_CHECK(p.device() == r.device() && p.device() == beta.device(),
              "cg_update_p tensors must be on the same CUDA device.");
  TORCH_CHECK(p.scalar_type() == at::ScalarType::Double &&
                  r.scalar_type() == at::ScalarType::Double &&
                  beta.scalar_type() == at::ScalarType::Double,
              "cg_update_p expects float64 tensors.");
  TORCH_CHECK(p.is_contiguous() && r.is_contiguous() && beta.is_contiguous(),
              "cg_update_p expects contiguous tensors.");
  TORCH_CHECK(p.numel() == r.numel(), "cg_update_p vector sizes do not match.");
  TORCH_CHECK(beta.numel() == 1, "cg_update_p beta must be scalar.");

  constexpr int threads = 256;
  const int64_t n = p.numel();
  const int64_t raw_blocks = (n + threads - 1) / threads;
  const int blocks = static_cast<int>(std::min<int64_t>(raw_blocks, 65535));
  cg_update_p_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      p.data_ptr<double>(), r.data_ptr<double>(), beta.data_ptr<double>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::string dqg_c1_au_version() {
  return "dqg_c1_au_atu_20260612_fused_cg_v1";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dqg_c1_au", &dqg_c1_au, "DQG/C1 matrix-free Au CUDA kernel");
  m.def("dqg_c1_au_out", &dqg_c1_au_out,
        "DQG/C1 matrix-free Au CUDA kernel with caller-provided output");
  m.def("dqg_c1_atu", &dqg_c1_atu, "DQG/C1 matrix-free ATu CUDA kernel");
  m.def("dqg_c1_atu_out", &dqg_c1_atu_out,
        "DQG/C1 matrix-free ATu CUDA kernel with caller-provided output");
  m.def("dqg_c1_atu_range_add_out", &dqg_c1_atu_range_add_out,
        "DQG/C1 matrix-free ATu CUDA kernel over a global dual-row range, accumulating into output");
  m.def("cg_update_y_r", &cg_update_y_r,
        "Fused CUDA update y += alpha p, r -= alpha Ap, and ||r||^2");
  m.def("cg_update_p", &cg_update_p,
        "Fused CUDA update p = r + beta p");
  m.def("dqg_c1_au_version", &dqg_c1_au_version,
        "DQG/C1 matrix-free Au/ATu CUDA kernel version");
}
