#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace {

enum BlockOffset {
  D2AB = 0,
  D2AA = 1,
  D2BB = 2,
  D200 = 3,
  D1A = 4,
  D1B = 5,
  Q1A = 6,
  Q1B = 7,
  Q2AB = 8,
  Q2AA = 9,
  Q2BB = 10,
  G2AB = 11,
  G2BA = 12,
  G2AA = 13,
};

enum SpanType {
  TRACE_D2AB = 0,
  TRACE_D2AA = 1,
  TRACE_D2BB = 2,
  TRACE_D2_TOTAL = 3,
  HERM_D2AA = 4,
  HERM_D2BB = 5,
  HERM_D2AB = 6,
  D1A_Q1A = 7,
  D1B_Q1B = 8,
  CONTRACT_AB_D1A = 9,
  CONTRACT_AB_D1B = 10,
  CONTRACT_AA_D1A = 11,
  CONTRACT_BB_D1B = 12,
  CONTRACT_MIX_A = 13,
  CONTRACT_MIX_B = 14,
  SPIN_TRACE = 15,
  SPIN_D1_EQ = 16,
  SPIN_D2AA_EQ = 17,
  SPIN_D2AA_FROM_AB = 18,
  SPIN_D2BB_FROM_AB = 19,
  SPIN_D200_SINGLET = 20,
  SPIN_D2AB_SYM = 21,
  SPIN_D200_NONSINGLET = 22,
  SPIN_G2BA_COL_TRACE = 23,
  SPIN_G2BA_ROW_TRACE = 24,
  Q2AB_SPAN = 25,
  Q2AA_SPAN = 26,
  Q2BB_SPAN = 27,
  G2AB_SPAN = 28,
  G2BA_SPAN = 29,
  G2AA_SPAN = 30,
};

__device__ __forceinline__ int64_t block_off(const int64_t *offsets,
                                             int block, int h, int nirrep) {
  return offsets[static_cast<int64_t>(block) * nirrep + h];
}

__device__ __forceinline__ int sym_pair(const int32_t *table, int a, int b) {
  return static_cast<int>(table[a * 8 + b]);
}

__device__ __forceinline__ int64_t ibas(const int32_t *map, int h, int amo,
                                        int i, int j) {
  return static_cast<int64_t>(map[(static_cast<int64_t>(h) * amo + i) * amo + j]);
}

__device__ __forceinline__ int pair_i(const int64_t *offsets,
                                      const int32_t *items, int h,
                                      int64_t idx) {
  return static_cast<int>(items[offsets[h] + idx]);
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

__device__ __forceinline__ void atomic_emit(double *out, int64_t col,
                                            double value) {
  if (value != 0.0) {
    atomicAdd(out + col, value);
  }
}

__device__ __forceinline__ int aa_sign(int i, int j) {
  return (i < j) ? 1 : -1;
}

__global__ void dqg_sym_au_kernel(
    const double *__restrict__ x, double *__restrict__ out, int nirrep, int amo,
    double na, double nb, const int32_t *__restrict__ amopi,
    const int32_t *__restrict__ pitzer, const int32_t *__restrict__ symmetry,
    const int32_t *__restrict__ sym_table, const int32_t *__restrict__ gems_ab,
    const int32_t *__restrict__ gems_aa, const int64_t *__restrict__ offsets,
    const int64_t *__restrict__ ab_offsets, const int64_t *__restrict__ aa_offsets,
    const int32_t *__restrict__ ab_i, const int32_t *__restrict__ ab_j,
    const int32_t *__restrict__ aa_i, const int32_t *__restrict__ aa_j,
    const int32_t *__restrict__ ibas_ab, const int32_t *__restrict__ ibas_aa,
    const int32_t *__restrict__ row_type, const int32_t *__restrict__ row_h,
    const int32_t *__restrict__ row_local, int64_t n_dual) {

  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n_dual) {
    return;
  }

  const int type = static_cast<int>(row_type[row]);
  const int h = static_cast<int>(row_h[row]);
  const int local = row_local[row];

  if (type == TRACE_D2AB) {
    double value = 0.0;
    for (int i = 0; i < amo; ++i) {
      for (int j = 0; j < amo; ++j) {
        const int hp = sym_pair(sym_table, symmetry[i], symmetry[j]);
        const int64_t gab = gems_ab[hp];
        if (gab == 0) continue;
        const int64_t ij = ibas(ibas_ab, hp, amo, i, j);
        value += mat_get(x, block_off(offsets, D2AB, hp, nirrep), gab, ij, ij);
      }
    }
    out[row] = value;
    return;
  }
  if (type == TRACE_D2AA || type == TRACE_D2BB) {
    double value = 0.0;
    const int block = (type == TRACE_D2AA) ? D2AA : D2BB;
    for (int i = 0; i < amo; ++i) {
      for (int j = 0; j < amo; ++j) {
        if (i == j) continue;
        const int hp = sym_pair(sym_table, symmetry[i], symmetry[j]);
        const int64_t gaa = gems_aa[hp];
        if (gaa == 0) continue;
        const int64_t ij = ibas(ibas_aa, hp, amo, i, j);
        value += mat_get(x, block_off(offsets, block, hp, nirrep), gaa, ij, ij);
      }
    }
    out[row] = value;
    return;
  }
  if (type == TRACE_D2_TOTAL) {
    double value = 0.0;
    for (int i = 0; i < amo; ++i) {
      for (int j = 0; j < amo; ++j) {
        const int hp = sym_pair(sym_table, symmetry[i], symmetry[j]);
        const int64_t gab = gems_ab[hp];
        if (gab != 0) {
          const int64_t ij = ibas(ibas_ab, hp, amo, i, j);
          value += 2.0 * mat_get(x, block_off(offsets, D2AB, hp, nirrep), gab, ij, ij);
        }
        if (i == j) continue;
        const int64_t gaa = gems_aa[hp];
        if (gaa == 0) continue;
        const int64_t ij = ibas(ibas_aa, hp, amo, i, j);
        value += mat_get(x, block_off(offsets, D2AA, hp, nirrep), gaa, ij, ij);
        value += mat_get(x, block_off(offsets, D2BB, hp, nirrep), gaa, ij, ij);
      }
    }
    out[row] = value;
    return;
  }

  if (type == HERM_D2AA || type == HERM_D2BB || type == HERM_D2AB) {
    const bool ab = type == HERM_D2AB;
    const int block = (type == HERM_D2AA) ? D2AA : (type == HERM_D2BB ? D2BB : D2AB);
    const int dim = ab ? gems_ab[h] : gems_aa[h];
    const int ij = local / dim;
    const int kl = local % dim;
    out[row] = mat_get(x, block_off(offsets, block, h, nirrep), dim, ij, kl) -
               mat_get(x, block_off(offsets, block, h, nirrep), dim, kl, ij);
    return;
  }

  if (type == D1A_Q1A || type == D1B_Q1B) {
    const int dim = amopi[h];
    const int i = local / dim;
    const int j = local % dim;
    const int dblock = (type == D1A_Q1A) ? D1A : D1B;
    const int qblock = (type == D1A_Q1A) ? Q1A : Q1B;
    out[row] = mat_get(x, block_off(offsets, dblock, h, nirrep), dim, j, i) +
               mat_get(x, block_off(offsets, qblock, h, nirrep), dim, i, j);
    return;
  }

  if (type == CONTRACT_AB_D1A || type == CONTRACT_AB_D1B ||
      type == CONTRACT_AA_D1A || type == CONTRACT_BB_D1B ||
      type == CONTRACT_MIX_A || type == CONTRACT_MIX_B) {
    const int dim = amopi[h];
    const int i_local = local / dim;
    const int j_local = local % dim;
    const int ii = i_local + pitzer[h];
    const int jj = j_local + pitzer[h];
    double value = 0.0;
    if (type == CONTRACT_AB_D1A) {
      value = nb * mat_get(x, block_off(offsets, D1A, h, nirrep), dim, i_local, j_local);
      for (int k = 0; k < amo; ++k) {
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gab = gems_ab[hp];
        const int64_t ik = ibas(ibas_ab, hp, amo, ii, k);
        const int64_t jk = ibas(ibas_ab, hp, amo, jj, k);
        value -= mat_get(x, block_off(offsets, D2AB, hp, nirrep), gab, ik, jk);
      }
    } else if (type == CONTRACT_AB_D1B) {
      value = na * mat_get(x, block_off(offsets, D1B, h, nirrep), dim, i_local, j_local);
      for (int k = 0; k < amo; ++k) {
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gab = gems_ab[hp];
        const int64_t ik = ibas(ibas_ab, hp, amo, k, ii);
        const int64_t jk = ibas(ibas_ab, hp, amo, k, jj);
        value -= mat_get(x, block_off(offsets, D2AB, hp, nirrep), gab, ik, jk);
      }
    } else if (type == CONTRACT_AA_D1A) {
      value = (na - 1.0) * mat_get(x, block_off(offsets, D1A, h, nirrep), dim, i_local, j_local);
      for (int k = 0; k < amo; ++k) {
        if (ii == k || jj == k) continue;
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gaa = gems_aa[hp];
        const int64_t ik = ibas(ibas_aa, hp, amo, ii, k);
        const int64_t jk = ibas(ibas_aa, hp, amo, jj, k);
        value -= static_cast<double>(aa_sign(ii, k) * aa_sign(jj, k)) *
                 mat_get(x, block_off(offsets, D2AA, hp, nirrep), gaa, ik, jk);
      }
    } else if (type == CONTRACT_BB_D1B) {
      value = (nb - 1.0) * mat_get(x, block_off(offsets, D1B, h, nirrep), dim, i_local, j_local);
      for (int k = 0; k < amo; ++k) {
        if (ii == k || jj == k) continue;
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gaa = gems_aa[hp];
        const int64_t ik = ibas(ibas_aa, hp, amo, ii, k);
        const int64_t jk = ibas(ibas_aa, hp, amo, jj, k);
        value -= static_cast<double>(aa_sign(ii, k) * aa_sign(jj, k)) *
                 mat_get(x, block_off(offsets, D2BB, hp, nirrep), gaa, ik, jk);
      }
    } else if (type == CONTRACT_MIX_A) {
      const double n = na + nb;
      value = (n - 1.0) * mat_get(x, block_off(offsets, D1A, h, nirrep), dim, i_local, j_local);
      for (int k = 0; k < amo; ++k) {
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gab = gems_ab[hp];
        value -= mat_get(x, block_off(offsets, D2AB, hp, nirrep), gab,
                         ibas(ibas_ab, hp, amo, ii, k),
                         ibas(ibas_ab, hp, amo, jj, k));
      }
      for (int k = 0; k < amo; ++k) {
        if (ii == k || jj == k) continue;
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gaa = gems_aa[hp];
        value -= static_cast<double>(aa_sign(ii, k) * aa_sign(jj, k)) *
                 mat_get(x, block_off(offsets, D2AA, hp, nirrep), gaa,
                         ibas(ibas_aa, hp, amo, ii, k),
                         ibas(ibas_aa, hp, amo, jj, k));
      }
    } else {
      const double n = na + nb;
      value = (n - 1.0) * mat_get(x, block_off(offsets, D1B, h, nirrep), dim, i_local, j_local);
      for (int k = 0; k < amo; ++k) {
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gab = gems_ab[hp];
        value -= mat_get(x, block_off(offsets, D2AB, hp, nirrep), gab,
                         ibas(ibas_ab, hp, amo, k, ii),
                         ibas(ibas_ab, hp, amo, k, jj));
      }
      for (int k = 0; k < amo; ++k) {
        if (ii == k || jj == k) continue;
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gaa = gems_aa[hp];
        value -= static_cast<double>(aa_sign(ii, k) * aa_sign(jj, k)) *
                 mat_get(x, block_off(offsets, D2BB, hp, nirrep), gaa,
                         ibas(ibas_aa, hp, amo, ii, k),
                         ibas(ibas_aa, hp, amo, jj, k));
      }
    }
    out[row] = value;
    return;
  }

  if (type == SPIN_TRACE) {
    double value = 0.0;
    for (int i = 0; i < amo; ++i) {
      for (int j = 0; j < amo; ++j) {
        const int hp = sym_pair(sym_table, symmetry[i], symmetry[j]);
        const int64_t gab = gems_ab[hp];
        if (gab == 0) continue;
        value += mat_get(x, block_off(offsets, D2AB, hp, nirrep), gab,
                         ibas(ibas_ab, hp, amo, i, j),
                         ibas(ibas_ab, hp, amo, j, i));
      }
    }
    out[row] = value;
    return;
  }

  if (type == SPIN_D1_EQ) {
    out[row] = x[block_off(offsets, D1A, h, nirrep) + local] -
               x[block_off(offsets, D1B, h, nirrep) + local];
    return;
  }
  if (type == SPIN_D2AA_EQ) {
    out[row] = x[block_off(offsets, D2AA, h, nirrep) + local] -
               x[block_off(offsets, D2BB, h, nirrep) + local];
    return;
  }
  if (type == SPIN_D2AA_FROM_AB || type == SPIN_D2BB_FROM_AB) {
    const int gaa = gems_aa[h];
    const int gab = gems_ab[h];
    const int ij = local / gaa;
    const int kl = local % gaa;
    const int i = pair_i(aa_offsets, aa_i, h, ij);
    const int j = pair_i(aa_offsets, aa_j, h, ij);
    const int k = pair_i(aa_offsets, aa_i, h, kl);
    const int l = pair_i(aa_offsets, aa_j, h, kl);
    const int64_t ijb = ibas(ibas_ab, h, amo, i, j);
    const int64_t jib = ibas(ibas_ab, h, amo, j, i);
    const int64_t klb = ibas(ibas_ab, h, amo, k, l);
    const int64_t lkb = ibas(ibas_ab, h, amo, l, k);
    const int dblock = (type == SPIN_D2AA_FROM_AB) ? D2AA : D2BB;
    out[row] = mat_get(x, block_off(offsets, dblock, h, nirrep), gaa, ij, kl) -
               0.5 * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ijb, klb) +
               0.5 * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, jib, klb) +
               0.5 * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ijb, lkb) -
               0.5 * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, jib, lkb);
    return;
  }
  if (type == SPIN_D200_SINGLET || type == SPIN_D2AB_SYM ||
      type == SPIN_D200_NONSINGLET) {
    const int gab = gems_ab[h];
    if (type == SPIN_D200_NONSINGLET) {
      const int dim = 2 * gab;
      const int drow = local / dim;
      const int dcol = local % dim;
      const int ij = (drow < gab) ? drow : drow - gab;
      const int kl = (dcol < gab) ? dcol : dcol - gab;
      const int i = pair_i(ab_offsets, ab_i, h, ij);
      const int j = pair_i(ab_offsets, ab_j, h, ij);
      const int k = pair_i(ab_offsets, ab_i, h, kl);
      const int l = pair_i(ab_offsets, ab_j, h, kl);
      const int64_t ji = ibas(ibas_ab, h, amo, j, i);
      const int64_t lk = ibas(ibas_ab, h, amo, l, k);
      const double dij = (i == j) ? sqrt(2.0) : 1.0;
      const double dkl = (k == l) ? sqrt(2.0) : 1.0;
      double value = mat_get(x, block_off(offsets, D200, h, nirrep), dim, drow, dcol);
      if (drow < gab && dcol < gab) {
        const double v = -0.5 / (dij * dkl);
        value += v * (mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, kl) +
                      mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, kl) +
                      mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, lk) +
                      mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, lk));
      } else if (drow < gab) {
        const double v = 0.5 / dij;
        value += -v * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, kl);
        value +=  v * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, lk);
        value += -v * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, kl);
        value +=  v * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, lk);
      } else if (dcol < gab) {
        const double v = 0.5 / dkl;
        value += -v * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, kl);
        value += -v * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, lk);
        value +=  v * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, kl);
        value +=  v * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, lk);
      } else {
        value += -0.5 * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, kl);
        value +=  0.5 * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, kl);
        value +=  0.5 * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, lk);
        value += -0.5 * mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, lk);
      }
      out[row] = value;
      return;
    }
    const int ij = local / gab;
    const int kl = local % gab;
    const int i = pair_i(ab_offsets, ab_i, h, ij);
    const int j = pair_i(ab_offsets, ab_j, h, ij);
    const int k = pair_i(ab_offsets, ab_i, h, kl);
    const int l = pair_i(ab_offsets, ab_j, h, kl);
    const int64_t ji = ibas(ibas_ab, h, amo, j, i);
    const int64_t lk = ibas(ibas_ab, h, amo, l, k);
    if (type == SPIN_D2AB_SYM) {
      out[row] = mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, kl) -
                 mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, lk);
      return;
    }
    const double dij = (i == j) ? sqrt(2.0) : 1.0;
    const double dkl = (k == l) ? sqrt(2.0) : 1.0;
    const double v = -0.5 / (dij * dkl);
    out[row] = mat_get(x, block_off(offsets, D200, h, nirrep), gab, ij, kl) +
               v * (mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, kl) +
                    mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, kl) +
                    mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ij, lk) +
                    mat_get(x, block_off(offsets, D2AB, h, nirrep), gab, ji, lk));
    return;
  }

  if (type == SPIN_G2BA_COL_TRACE || type == SPIN_G2BA_ROW_TRACE) {
    const int64_t gab0 = gems_ab[0];
    double value = 0.0;
    for (int i = 0; i < amo; ++i) {
      const int64_t ii = ibas(ibas_ab, 0, amo, i, i);
      if (type == SPIN_G2BA_COL_TRACE) {
        value += mat_get(x, block_off(offsets, G2BA, 0, nirrep), gab0, local, ii);
      } else {
        value += mat_get(x, block_off(offsets, G2BA, 0, nirrep), gab0, ii, local);
      }
    }
    out[row] = value;
    return;
  }

  if (type == Q2AB_SPAN || type == Q2AA_SPAN || type == Q2BB_SPAN) {
    const bool ab = type == Q2AB_SPAN;
    const bool beta = type == Q2BB_SPAN;
    const int dim = ab ? gems_ab[h] : gems_aa[h];
    const int ij = local / dim;
    const int kl = local % dim;
    const int i = pair_i(ab ? ab_offsets : aa_offsets, ab ? ab_i : aa_i, h, ij);
    const int j = pair_i(ab ? ab_offsets : aa_offsets, ab ? ab_j : aa_j, h, ij);
    const int k = pair_i(ab ? ab_offsets : aa_offsets, ab ? ab_i : aa_i, h, kl);
    const int l = pair_i(ab ? ab_offsets : aa_offsets, ab ? ab_j : aa_j, h, kl);
    int d2block = ab ? D2AB : (beta ? D2BB : D2AA);
    int q2block = ab ? Q2AB : (beta ? Q2BB : Q2AA);
    int d1block = beta ? D1B : D1A;
    double value = mat_get(x, block_off(offsets, d2block, h, nirrep), dim, ij, kl) -
                   mat_get(x, block_off(offsets, q2block, h, nirrep), dim, ij, kl);
    if (ab) {
      if (j == l) {
        const int hp = symmetry[i];
        value -= mat_get(x, block_off(offsets, D1A, hp, nirrep), amopi[hp],
                         k - pitzer[hp], i - pitzer[hp]);
      }
      if (i == k) {
        const int hp = symmetry[j];
        value -= mat_get(x, block_off(offsets, D1B, hp, nirrep), amopi[hp],
                         l - pitzer[hp], j - pitzer[hp]);
      }
    } else {
      if (j == l) {
        const int hp = symmetry[i];
        value -= mat_get(x, block_off(offsets, d1block, hp, nirrep), amopi[hp],
                         k - pitzer[hp], i - pitzer[hp]);
      }
      if (j == k) {
        const int hp = symmetry[i];
        value += mat_get(x, block_off(offsets, d1block, hp, nirrep), amopi[hp],
                         l - pitzer[hp], i - pitzer[hp]);
      }
      if (i == l) {
        const int hp = symmetry[j];
        value += mat_get(x, block_off(offsets, d1block, hp, nirrep), amopi[hp],
                         k - pitzer[hp], j - pitzer[hp]);
      }
      if (i == k) {
        const int hp = symmetry[j];
        value -= mat_get(x, block_off(offsets, d1block, hp, nirrep), amopi[hp],
                         l - pitzer[hp], j - pitzer[hp]);
      }
    }
    out[row] = value;
    return;
  }

  if (type == G2AB_SPAN || type == G2BA_SPAN || type == G2AA_SPAN) {
    const int gab = gems_ab[h];
    if (type == G2AB_SPAN || type == G2BA_SPAN) {
      const int ij = local / gab;
      const int kl = local % gab;
      const int i = pair_i(ab_offsets, ab_i, h, ij);
      const int j = pair_i(ab_offsets, ab_j, h, ij);
      const int k = pair_i(ab_offsets, ab_i, h, kl);
      const int l = pair_i(ab_offsets, ab_j, h, kl);
      const int hp = sym_pair(sym_table, symmetry[i], symmetry[l]);
      double value = -mat_get(x, block_off(offsets, type == G2AB_SPAN ? G2AB : G2BA, h, nirrep),
                              gab, ij, kl);
      if (j == l) {
        const int h3 = symmetry[i];
        value += mat_get(x, block_off(offsets, type == G2AB_SPAN ? D1A : D1B, h3, nirrep),
                         amopi[h3], i - pitzer[h3], k - pitzer[h3]);
      }
      if (type == G2AB_SPAN) {
        value -= mat_get(x, block_off(offsets, D2AB, hp, nirrep), gems_ab[hp],
                         ibas(ibas_ab, hp, amo, i, l),
                         ibas(ibas_ab, hp, amo, k, j));
      } else {
        value -= mat_get(x, block_off(offsets, D2AB, hp, nirrep), gems_ab[hp],
                         ibas(ibas_ab, hp, amo, l, i),
                         ibas(ibas_ab, hp, amo, j, k));
      }
      out[row] = value;
      return;
    }

    const int dim = 2 * gab;
    const int a = local / dim;
    const int b = local % dim;
    const bool bottom = a >= gab;
    const bool right = b >= gab;
    const int ij = bottom ? a - gab : a;
    const int kl = right ? b - gab : b;
    const int i = pair_i(ab_offsets, ab_i, h, ij);
    const int j = pair_i(ab_offsets, ab_j, h, ij);
    const int k = pair_i(ab_offsets, ab_i, h, kl);
    const int l = pair_i(ab_offsets, ab_j, h, kl);
    double value = -mat_get(x, block_off(offsets, G2AA, h, nirrep), dim, a, b);
    const int hp = sym_pair(sym_table, symmetry[i], symmetry[l]);
    if (!bottom && !right) {
      if (j == l) {
        const int h3 = symmetry[i];
        value += mat_get(x, block_off(offsets, D1A, h3, nirrep), amopi[h3],
                         i - pitzer[h3], k - pitzer[h3]);
      }
      if (i != l && k != j) {
        value -= static_cast<double>(aa_sign(i, l) * aa_sign(k, j)) *
                 mat_get(x, block_off(offsets, D2AA, hp, nirrep), gems_aa[hp],
                         ibas(ibas_aa, hp, amo, i, l),
                         ibas(ibas_aa, hp, amo, k, j));
      }
    } else if (bottom && right) {
      if (j == l) {
        const int h3 = symmetry[i];
        value += mat_get(x, block_off(offsets, D1B, h3, nirrep), amopi[h3],
                         i - pitzer[h3], k - pitzer[h3]);
      }
      if (i != l && k != j) {
        value -= static_cast<double>(aa_sign(i, l) * aa_sign(k, j)) *
                 mat_get(x, block_off(offsets, D2BB, hp, nirrep), gems_aa[hp],
                         ibas(ibas_aa, hp, amo, i, l),
                         ibas(ibas_aa, hp, amo, k, j));
      }
    } else if (!bottom && right) {
      value += mat_get(x, block_off(offsets, D2AB, hp, nirrep), gems_ab[hp],
                       ibas(ibas_ab, hp, amo, i, l),
                       ibas(ibas_ab, hp, amo, j, k));
    } else {
      value += mat_get(x, block_off(offsets, D2AB, hp, nirrep), gems_ab[hp],
                       ibas(ibas_ab, hp, amo, l, i),
                       ibas(ibas_ab, hp, amo, k, j));
    }
    out[row] = value;
  }
}

__global__ void dqg_sym_atu_kernel(
    const double *__restrict__ y, double *__restrict__ out, int nirrep, int amo,
    double na, double nb, const int32_t *__restrict__ amopi,
    const int32_t *__restrict__ pitzer, const int32_t *__restrict__ symmetry,
    const int32_t *__restrict__ sym_table, const int32_t *__restrict__ gems_ab,
    const int32_t *__restrict__ gems_aa, const int64_t *__restrict__ offsets,
    const int64_t *__restrict__ ab_offsets, const int64_t *__restrict__ aa_offsets,
    const int32_t *__restrict__ ab_i, const int32_t *__restrict__ ab_j,
    const int32_t *__restrict__ aa_i, const int32_t *__restrict__ aa_j,
    const int32_t *__restrict__ ibas_ab, const int32_t *__restrict__ ibas_aa,
    const int32_t *__restrict__ row_type, const int32_t *__restrict__ row_h,
    const int32_t *__restrict__ row_local_meta, int64_t row_start,
    int64_t row_count) {

  const int64_t row_local = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row_local >= row_count) {
    return;
  }
  const int64_t row = row_start + row_local;
  const double yr = y[row_local];
  if (yr == 0.0) {
    return;
  }

  const int type = static_cast<int>(row_type[row]);
  const int h = static_cast<int>(row_h[row]);
  const int64_t local = row_local_meta[row];

  if (type == TRACE_D2AB) {
    for (int i = 0; i < amo; ++i) {
      for (int j = 0; j < amo; ++j) {
        const int hp = sym_pair(sym_table, symmetry[i], symmetry[j]);
        const int64_t gab = gems_ab[hp];
        if (gab == 0) continue;
        const int64_t ij = ibas(ibas_ab, hp, amo, i, j);
        atomic_emit(out, block_off(offsets, D2AB, hp, nirrep) + ij * gab + ij, yr);
      }
    }
    return;
  }
  if (type == TRACE_D2AA || type == TRACE_D2BB) {
    const int block = (type == TRACE_D2AA) ? D2AA : D2BB;
    for (int i = 0; i < amo; ++i) {
      for (int j = 0; j < amo; ++j) {
        if (i == j) continue;
        const int hp = sym_pair(sym_table, symmetry[i], symmetry[j]);
        const int64_t gaa = gems_aa[hp];
        if (gaa == 0) continue;
        const int64_t ij = ibas(ibas_aa, hp, amo, i, j);
        atomic_emit(out, block_off(offsets, block, hp, nirrep) + ij * gaa + ij, yr);
      }
    }
    return;
  }
  if (type == TRACE_D2_TOTAL) {
    for (int i = 0; i < amo; ++i) {
      for (int j = 0; j < amo; ++j) {
        const int hp = sym_pair(sym_table, symmetry[i], symmetry[j]);
        const int64_t gab = gems_ab[hp];
        if (gab != 0) {
          const int64_t ij = ibas(ibas_ab, hp, amo, i, j);
          atomic_emit(out, block_off(offsets, D2AB, hp, nirrep) + ij * gab + ij, 2.0 * yr);
        }
        if (i == j) continue;
        const int64_t gaa = gems_aa[hp];
        if (gaa == 0) continue;
        const int64_t ij = ibas(ibas_aa, hp, amo, i, j);
        atomic_emit(out, block_off(offsets, D2AA, hp, nirrep) + ij * gaa + ij, yr);
        atomic_emit(out, block_off(offsets, D2BB, hp, nirrep) + ij * gaa + ij, yr);
      }
    }
    return;
  }

  if (type == HERM_D2AA || type == HERM_D2BB || type == HERM_D2AB) {
    const bool ab = type == HERM_D2AB;
    const int block = (type == HERM_D2AA) ? D2AA : (type == HERM_D2BB ? D2BB : D2AB);
    const int64_t dim = ab ? gems_ab[h] : gems_aa[h];
    const int64_t ij = local / dim;
    const int64_t kl = local % dim;
    const int64_t off = block_off(offsets, block, h, nirrep);
    atomic_emit(out, off + ij * dim + kl, yr);
    atomic_emit(out, off + kl * dim + ij, -yr);
    return;
  }

  if (type == D1A_Q1A || type == D1B_Q1B) {
    const int64_t dim = amopi[h];
    const int64_t i = local / dim;
    const int64_t j = local % dim;
    const int dblock = (type == D1A_Q1A) ? D1A : D1B;
    const int qblock = (type == D1A_Q1A) ? Q1A : Q1B;
    atomic_emit(out, block_off(offsets, dblock, h, nirrep) + j * dim + i, yr);
    atomic_emit(out, block_off(offsets, qblock, h, nirrep) + i * dim + j, yr);
    return;
  }

  if (type == CONTRACT_AB_D1A || type == CONTRACT_AB_D1B ||
      type == CONTRACT_AA_D1A || type == CONTRACT_BB_D1B ||
      type == CONTRACT_MIX_A || type == CONTRACT_MIX_B) {
    const int64_t dim = amopi[h];
    const int i_local = static_cast<int>(local / dim);
    const int j_local = static_cast<int>(local % dim);
    const int ii = i_local + pitzer[h];
    const int jj = j_local + pitzer[h];
    if (type == CONTRACT_AB_D1A || type == CONTRACT_MIX_A) {
      atomic_emit(out, block_off(offsets, D1A, h, nirrep) + i_local * dim + j_local,
                  (type == CONTRACT_AB_D1A ? nb : na + nb - 1.0) * yr);
      for (int k = 0; k < amo; ++k) {
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gab = gems_ab[hp];
        atomic_emit(out, block_off(offsets, D2AB, hp, nirrep) +
                         ibas(ibas_ab, hp, amo, ii, k) * gab +
                         ibas(ibas_ab, hp, amo, jj, k), -yr);
      }
    }
    if (type == CONTRACT_AB_D1B || type == CONTRACT_MIX_B) {
      atomic_emit(out, block_off(offsets, D1B, h, nirrep) + i_local * dim + j_local,
                  (type == CONTRACT_AB_D1B ? na : na + nb - 1.0) * yr);
      for (int k = 0; k < amo; ++k) {
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gab = gems_ab[hp];
        atomic_emit(out, block_off(offsets, D2AB, hp, nirrep) +
                         ibas(ibas_ab, hp, amo, k, ii) * gab +
                         ibas(ibas_ab, hp, amo, k, jj), -yr);
      }
    }
    if (type == CONTRACT_AA_D1A || type == CONTRACT_MIX_A) {
      if (type == CONTRACT_AA_D1A) {
        atomic_emit(out, block_off(offsets, D1A, h, nirrep) + i_local * dim + j_local,
                    (na - 1.0) * yr);
      }
      for (int k = 0; k < amo; ++k) {
        if (ii == k || jj == k) continue;
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gaa = gems_aa[hp];
        atomic_emit(out, block_off(offsets, D2AA, hp, nirrep) +
                         ibas(ibas_aa, hp, amo, ii, k) * gaa +
                         ibas(ibas_aa, hp, amo, jj, k),
                    -static_cast<double>(aa_sign(ii, k) * aa_sign(jj, k)) * yr);
      }
    }
    if (type == CONTRACT_BB_D1B || type == CONTRACT_MIX_B) {
      if (type == CONTRACT_BB_D1B) {
        atomic_emit(out, block_off(offsets, D1B, h, nirrep) + i_local * dim + j_local,
                    (nb - 1.0) * yr);
      }
      for (int k = 0; k < amo; ++k) {
        if (ii == k || jj == k) continue;
        const int hp = sym_pair(sym_table, symmetry[ii], symmetry[k]);
        const int64_t gaa = gems_aa[hp];
        atomic_emit(out, block_off(offsets, D2BB, hp, nirrep) +
                         ibas(ibas_aa, hp, amo, ii, k) * gaa +
                         ibas(ibas_aa, hp, amo, jj, k),
                    -static_cast<double>(aa_sign(ii, k) * aa_sign(jj, k)) * yr);
      }
    }
    return;
  }

  if (type == SPIN_TRACE) {
    for (int i = 0; i < amo; ++i) {
      for (int j = 0; j < amo; ++j) {
        const int hp = sym_pair(sym_table, symmetry[i], symmetry[j]);
        const int64_t gab = gems_ab[hp];
        if (gab == 0) continue;
        atomic_emit(out, block_off(offsets, D2AB, hp, nirrep) +
                         ibas(ibas_ab, hp, amo, i, j) * gab +
                         ibas(ibas_ab, hp, amo, j, i), yr);
      }
    }
    return;
  }
  if (type == SPIN_D1_EQ) {
    atomic_emit(out, block_off(offsets, D1A, h, nirrep) + local, yr);
    atomic_emit(out, block_off(offsets, D1B, h, nirrep) + local, -yr);
    return;
  }
  if (type == SPIN_D2AA_EQ) {
    atomic_emit(out, block_off(offsets, D2AA, h, nirrep) + local, yr);
    atomic_emit(out, block_off(offsets, D2BB, h, nirrep) + local, -yr);
    return;
  }
  if (type == SPIN_D2AA_FROM_AB || type == SPIN_D2BB_FROM_AB) {
    const int64_t gaa = gems_aa[h];
    const int64_t gab = gems_ab[h];
    const int64_t ij = local / gaa;
    const int64_t kl = local % gaa;
    const int i = pair_i(aa_offsets, aa_i, h, ij);
    const int j = pair_i(aa_offsets, aa_j, h, ij);
    const int k = pair_i(aa_offsets, aa_i, h, kl);
    const int l = pair_i(aa_offsets, aa_j, h, kl);
    const int64_t ijb = ibas(ibas_ab, h, amo, i, j);
    const int64_t jib = ibas(ibas_ab, h, amo, j, i);
    const int64_t klb = ibas(ibas_ab, h, amo, k, l);
    const int64_t lkb = ibas(ibas_ab, h, amo, l, k);
    atomic_emit(out, block_off(offsets, type == SPIN_D2AA_FROM_AB ? D2AA : D2BB, h, nirrep) +
                     ij * gaa + kl, yr);
    atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ijb * gab + klb, -0.5 * yr);
    atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + jib * gab + klb, 0.5 * yr);
    atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ijb * gab + lkb, 0.5 * yr);
    atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + jib * gab + lkb, -0.5 * yr);
    return;
  }
  if (type == SPIN_D200_SINGLET || type == SPIN_D2AB_SYM ||
      type == SPIN_D200_NONSINGLET) {
    const int64_t gab = gems_ab[h];
    if (type == SPIN_D200_NONSINGLET) {
      const int64_t dim = 2 * gab;
      const int64_t drow = local / dim;
      const int64_t dcol = local % dim;
      const int64_t ij = (drow < gab) ? drow : drow - gab;
      const int64_t kl = (dcol < gab) ? dcol : dcol - gab;
      const int i = pair_i(ab_offsets, ab_i, h, ij);
      const int j = pair_i(ab_offsets, ab_j, h, ij);
      const int k = pair_i(ab_offsets, ab_i, h, kl);
      const int l = pair_i(ab_offsets, ab_j, h, kl);
      const int64_t ji = ibas(ibas_ab, h, amo, j, i);
      const int64_t lk = ibas(ibas_ab, h, amo, l, k);
      const double dij = (i == j) ? sqrt(2.0) : 1.0;
      const double dkl = (k == l) ? sqrt(2.0) : 1.0;
      atomic_emit(out, block_off(offsets, D200, h, nirrep) + drow * dim + dcol, yr);
      if (drow < gab && dcol < gab) {
        const double v = -0.5 / (dij * dkl) * yr;
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + kl, v);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + kl, v);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + lk, v);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + lk, v);
      } else if (drow < gab) {
        const double v = 0.5 / dij * yr;
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + kl, -v);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + lk, v);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + kl, -v);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + lk, v);
      } else if (dcol < gab) {
        const double v = 0.5 / dkl * yr;
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + kl, -v);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + lk, -v);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + kl, v);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + lk, v);
      } else {
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + kl, -0.5 * yr);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + kl, 0.5 * yr);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + lk, 0.5 * yr);
        atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + lk, -0.5 * yr);
      }
      return;
    }
    const int64_t ij = local / gab;
    const int64_t kl = local % gab;
    const int i = pair_i(ab_offsets, ab_i, h, ij);
    const int j = pair_i(ab_offsets, ab_j, h, ij);
    const int k = pair_i(ab_offsets, ab_i, h, kl);
    const int l = pair_i(ab_offsets, ab_j, h, kl);
    const int64_t ji = ibas(ibas_ab, h, amo, j, i);
    const int64_t lk = ibas(ibas_ab, h, amo, l, k);
    if (type == SPIN_D2AB_SYM) {
      atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + kl, yr);
      atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + lk, -yr);
      return;
    }
    const double dij = (i == j) ? sqrt(2.0) : 1.0;
    const double dkl = (k == l) ? sqrt(2.0) : 1.0;
    const double v = -0.5 / (dij * dkl) * yr;
    atomic_emit(out, block_off(offsets, D200, h, nirrep) + ij * gab + kl, yr);
    atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + kl, v);
    atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + kl, v);
    atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ij * gab + lk, v);
    atomic_emit(out, block_off(offsets, D2AB, h, nirrep) + ji * gab + lk, v);
    return;
  }

  if (type == SPIN_G2BA_COL_TRACE || type == SPIN_G2BA_ROW_TRACE) {
    const int64_t gab0 = gems_ab[0];
    for (int i = 0; i < amo; ++i) {
      const int64_t ii = ibas(ibas_ab, 0, amo, i, i);
      if (type == SPIN_G2BA_COL_TRACE) {
        atomic_emit(out, block_off(offsets, G2BA, 0, nirrep) + local * gab0 + ii, yr);
      } else {
        atomic_emit(out, block_off(offsets, G2BA, 0, nirrep) + ii * gab0 + local, yr);
      }
    }
    return;
  }

  if (type == Q2AB_SPAN || type == Q2AA_SPAN || type == Q2BB_SPAN) {
    const bool ab = type == Q2AB_SPAN;
    const bool beta = type == Q2BB_SPAN;
    const int64_t dim = ab ? gems_ab[h] : gems_aa[h];
    const int64_t ij = local / dim;
    const int64_t kl = local % dim;
    const int i = pair_i(ab ? ab_offsets : aa_offsets, ab ? ab_i : aa_i, h, ij);
    const int j = pair_i(ab ? ab_offsets : aa_offsets, ab ? ab_j : aa_j, h, ij);
    const int k = pair_i(ab ? ab_offsets : aa_offsets, ab ? ab_i : aa_i, h, kl);
    const int l = pair_i(ab ? ab_offsets : aa_offsets, ab ? ab_j : aa_j, h, kl);
    const int d2block = ab ? D2AB : (beta ? D2BB : D2AA);
    const int q2block = ab ? Q2AB : (beta ? Q2BB : Q2AA);
    const int d1block = beta ? D1B : D1A;
    atomic_emit(out, block_off(offsets, d2block, h, nirrep) + ij * dim + kl, yr);
    atomic_emit(out, block_off(offsets, q2block, h, nirrep) + ij * dim + kl, -yr);
    if (ab) {
      if (j == l) {
        const int hp = symmetry[i];
        atomic_emit(out, block_off(offsets, D1A, hp, nirrep) +
                         (k - pitzer[hp]) * amopi[hp] + (i - pitzer[hp]), -yr);
      }
      if (i == k) {
        const int hp = symmetry[j];
        atomic_emit(out, block_off(offsets, D1B, hp, nirrep) +
                         (l - pitzer[hp]) * amopi[hp] + (j - pitzer[hp]), -yr);
      }
    } else {
      if (j == l) {
        const int hp = symmetry[i];
        atomic_emit(out, block_off(offsets, d1block, hp, nirrep) +
                         (k - pitzer[hp]) * amopi[hp] + (i - pitzer[hp]), -yr);
      }
      if (j == k) {
        const int hp = symmetry[i];
        atomic_emit(out, block_off(offsets, d1block, hp, nirrep) +
                         (l - pitzer[hp]) * amopi[hp] + (i - pitzer[hp]), yr);
      }
      if (i == l) {
        const int hp = symmetry[j];
        atomic_emit(out, block_off(offsets, d1block, hp, nirrep) +
                         (k - pitzer[hp]) * amopi[hp] + (j - pitzer[hp]), yr);
      }
      if (i == k) {
        const int hp = symmetry[j];
        atomic_emit(out, block_off(offsets, d1block, hp, nirrep) +
                         (l - pitzer[hp]) * amopi[hp] + (j - pitzer[hp]), -yr);
      }
    }
    return;
  }

  if (type == G2AB_SPAN || type == G2BA_SPAN || type == G2AA_SPAN) {
    const int64_t gab = gems_ab[h];
    if (type == G2AB_SPAN || type == G2BA_SPAN) {
      const int64_t ij = local / gab;
      const int64_t kl = local % gab;
      const int i = pair_i(ab_offsets, ab_i, h, ij);
      const int j = pair_i(ab_offsets, ab_j, h, ij);
      const int k = pair_i(ab_offsets, ab_i, h, kl);
      const int l = pair_i(ab_offsets, ab_j, h, kl);
      const int hp = sym_pair(sym_table, symmetry[i], symmetry[l]);
      atomic_emit(out, block_off(offsets, type == G2AB_SPAN ? G2AB : G2BA, h, nirrep) +
                       ij * gab + kl, -yr);
      if (j == l) {
        const int h3 = symmetry[i];
        atomic_emit(out, block_off(offsets, type == G2AB_SPAN ? D1A : D1B, h3, nirrep) +
                         (i - pitzer[h3]) * amopi[h3] + (k - pitzer[h3]), yr);
      }
      if (type == G2AB_SPAN) {
        atomic_emit(out, block_off(offsets, D2AB, hp, nirrep) +
                         ibas(ibas_ab, hp, amo, i, l) * gems_ab[hp] +
                         ibas(ibas_ab, hp, amo, k, j), -yr);
      } else {
        atomic_emit(out, block_off(offsets, D2AB, hp, nirrep) +
                         ibas(ibas_ab, hp, amo, l, i) * gems_ab[hp] +
                         ibas(ibas_ab, hp, amo, j, k), -yr);
      }
      return;
    }

    const int64_t dim = 2 * gab;
    const int64_t a = local / dim;
    const int64_t b = local % dim;
    const bool bottom = a >= gab;
    const bool right = b >= gab;
    const int64_t ij = bottom ? a - gab : a;
    const int64_t kl = right ? b - gab : b;
    const int i = pair_i(ab_offsets, ab_i, h, ij);
    const int j = pair_i(ab_offsets, ab_j, h, ij);
    const int k = pair_i(ab_offsets, ab_i, h, kl);
    const int l = pair_i(ab_offsets, ab_j, h, kl);
    const int hp = sym_pair(sym_table, symmetry[i], symmetry[l]);
    atomic_emit(out, block_off(offsets, G2AA, h, nirrep) + a * dim + b, -yr);
    if (!bottom && !right) {
      if (j == l) {
        const int h3 = symmetry[i];
        atomic_emit(out, block_off(offsets, D1A, h3, nirrep) +
                         (i - pitzer[h3]) * amopi[h3] + (k - pitzer[h3]), yr);
      }
      if (i != l && k != j) {
        atomic_emit(out, block_off(offsets, D2AA, hp, nirrep) +
                         ibas(ibas_aa, hp, amo, i, l) * gems_aa[hp] +
                         ibas(ibas_aa, hp, amo, k, j),
                    -static_cast<double>(aa_sign(i, l) * aa_sign(k, j)) * yr);
      }
    } else if (bottom && right) {
      if (j == l) {
        const int h3 = symmetry[i];
        atomic_emit(out, block_off(offsets, D1B, h3, nirrep) +
                         (i - pitzer[h3]) * amopi[h3] + (k - pitzer[h3]), yr);
      }
      if (i != l && k != j) {
        atomic_emit(out, block_off(offsets, D2BB, hp, nirrep) +
                         ibas(ibas_aa, hp, amo, i, l) * gems_aa[hp] +
                         ibas(ibas_aa, hp, amo, k, j),
                    -static_cast<double>(aa_sign(i, l) * aa_sign(k, j)) * yr);
      }
    } else if (!bottom && right) {
      atomic_emit(out, block_off(offsets, D2AB, hp, nirrep) +
                       ibas(ibas_ab, hp, amo, i, l) * gems_ab[hp] +
                       ibas(ibas_ab, hp, amo, j, k), yr);
    } else {
      atomic_emit(out, block_off(offsets, D2AB, hp, nirrep) +
                       ibas(ibas_ab, hp, amo, l, i) * gems_ab[hp] +
                       ibas(ibas_ab, hp, amo, k, j), yr);
    }
  }
}

void check_double_cuda(torch::Tensor tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Double, name,
              " must be float64.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_long_cuda(torch::Tensor tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Long, name,
              " must be int64.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_int_cuda(torch::Tensor tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Int, name,
              " must be int32.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_sym_tensors(
    torch::Tensor amopi, torch::Tensor pitzer, torch::Tensor symmetry,
    torch::Tensor sym_table, torch::Tensor gems_ab, torch::Tensor gems_aa,
    torch::Tensor offsets, torch::Tensor ab_offsets, torch::Tensor aa_offsets,
    torch::Tensor ab_i, torch::Tensor ab_j, torch::Tensor aa_i,
    torch::Tensor aa_j, torch::Tensor ibas_ab, torch::Tensor ibas_aa,
    torch::Tensor row_type, torch::Tensor row_h, torch::Tensor row_local,
    int64_t nirrep, int64_t amo) {
  check_int_cuda(amopi, "amopi");
  check_int_cuda(pitzer, "pitzer");
  check_int_cuda(symmetry, "symmetry");
  check_int_cuda(sym_table, "sym_table");
  check_int_cuda(gems_ab, "gems_ab");
  check_int_cuda(gems_aa, "gems_aa");
  check_long_cuda(offsets, "offsets");
  check_long_cuda(ab_offsets, "ab_offsets");
  check_long_cuda(aa_offsets, "aa_offsets");
  check_int_cuda(ab_i, "ab_i");
  check_int_cuda(ab_j, "ab_j");
  check_int_cuda(aa_i, "aa_i");
  check_int_cuda(aa_j, "aa_j");
  check_int_cuda(ibas_ab, "ibas_ab");
  check_int_cuda(ibas_aa, "ibas_aa");
  check_int_cuda(row_type, "row_type");
  check_int_cuda(row_h, "row_h");
  check_int_cuda(row_local, "row_local");
  TORCH_CHECK(amopi.numel() == nirrep && pitzer.numel() == nirrep,
              "amopi/pitzer sizes do not match nirrep.");
  TORCH_CHECK(symmetry.numel() == amo, "symmetry size does not match amo.");
  TORCH_CHECK(sym_table.numel() == 64, "symmetry product table must have 64 entries.");
  TORCH_CHECK(gems_ab.numel() == nirrep && gems_aa.numel() == nirrep,
              "gem sizes do not match nirrep.");
  TORCH_CHECK(offsets.numel() == 14 * nirrep,
              "offset tensor size does not match block count.");
  TORCH_CHECK(ab_offsets.numel() == nirrep + 1 &&
                  aa_offsets.numel() == nirrep + 1,
              "pair offset sizes do not match nirrep.");
  TORCH_CHECK(ibas_ab.numel() == nirrep * amo * amo &&
                  ibas_aa.numel() == nirrep * amo * amo,
              "inverse pair-map sizes do not match nirrep/amo.");
  TORCH_CHECK(row_type.numel() == row_h.numel() &&
                  row_type.numel() == row_local.numel(),
              "row metadata tensors must have matching sizes.");
}

} // namespace

void dqg_sym_au_out(
    torch::Tensor x, torch::Tensor out, int64_t nirrep, int64_t amo, double na,
    double nb, torch::Tensor amopi, torch::Tensor pitzer,
    torch::Tensor symmetry, torch::Tensor sym_table, torch::Tensor gems_ab,
    torch::Tensor gems_aa, torch::Tensor offsets, torch::Tensor ab_offsets,
    torch::Tensor aa_offsets, torch::Tensor ab_i, torch::Tensor ab_j,
    torch::Tensor aa_i, torch::Tensor aa_j, torch::Tensor ibas_ab,
    torch::Tensor ibas_aa, torch::Tensor row_type, torch::Tensor row_h,
    torch::Tensor row_local, int64_t n_dual) {
  check_double_cuda(x, "x");
  check_double_cuda(out, "out");
  TORCH_CHECK(x.device() == out.device(), "x and out must be on the same CUDA device.");
  TORCH_CHECK(out.numel() == n_dual, "dqg_sym_au_out output has wrong size.");
  TORCH_CHECK(nirrep > 0 && nirrep <= 8 && amo > 0,
              "dqg_sym_au_out received invalid nirrep/amo.");
  check_sym_tensors(amopi, pitzer, symmetry, sym_table, gems_ab, gems_aa,
                    offsets, ab_offsets, aa_offsets, ab_i, ab_j, aa_i, aa_j,
                    ibas_ab, ibas_aa, row_type, row_h, row_local, nirrep, amo);
  TORCH_CHECK(row_type.numel() == n_dual,
              "dqg_sym_au_out row metadata size does not match n_dual.");

  constexpr int threads = 256;
  const int64_t blocks = (n_dual + threads - 1) / threads;
  dqg_sym_au_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<double>(), out.data_ptr<double>(), static_cast<int>(nirrep),
      static_cast<int>(amo), na, nb, amopi.data_ptr<int32_t>(),
      pitzer.data_ptr<int32_t>(), symmetry.data_ptr<int32_t>(),
      sym_table.data_ptr<int32_t>(), gems_ab.data_ptr<int32_t>(),
      gems_aa.data_ptr<int32_t>(), offsets.data_ptr<int64_t>(),
      ab_offsets.data_ptr<int64_t>(), aa_offsets.data_ptr<int64_t>(),
      ab_i.data_ptr<int32_t>(), ab_j.data_ptr<int32_t>(),
      aa_i.data_ptr<int32_t>(), aa_j.data_ptr<int32_t>(),
      ibas_ab.data_ptr<int32_t>(), ibas_aa.data_ptr<int32_t>(),
      row_type.data_ptr<int32_t>(), row_h.data_ptr<int32_t>(),
      row_local.data_ptr<int32_t>(), n_dual);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor dqg_sym_au(
    torch::Tensor x, int64_t nirrep, int64_t amo, double na, double nb,
    torch::Tensor amopi, torch::Tensor pitzer, torch::Tensor symmetry,
    torch::Tensor sym_table, torch::Tensor gems_ab, torch::Tensor gems_aa,
    torch::Tensor offsets, torch::Tensor ab_offsets, torch::Tensor aa_offsets,
    torch::Tensor ab_i, torch::Tensor ab_j, torch::Tensor aa_i,
    torch::Tensor aa_j, torch::Tensor ibas_ab, torch::Tensor ibas_aa,
    torch::Tensor row_type, torch::Tensor row_h, torch::Tensor row_local,
    int64_t n_dual) {
  auto out = torch::empty({n_dual}, x.options());
  dqg_sym_au_out(x, out, nirrep, amo, na, nb, amopi, pitzer, symmetry,
                 sym_table, gems_ab, gems_aa, offsets, ab_offsets, aa_offsets,
                 ab_i, ab_j, aa_i, aa_j, ibas_ab, ibas_aa, row_type, row_h,
                 row_local, n_dual);
  return out;
}

void dqg_sym_atu_range_add_out(
    torch::Tensor y, torch::Tensor out, int64_t row_start, int64_t row_stop,
    int64_t nirrep, int64_t amo, double na, double nb, torch::Tensor amopi,
    torch::Tensor pitzer, torch::Tensor symmetry, torch::Tensor sym_table,
    torch::Tensor gems_ab, torch::Tensor gems_aa, torch::Tensor offsets,
    torch::Tensor ab_offsets, torch::Tensor aa_offsets, torch::Tensor ab_i,
    torch::Tensor ab_j, torch::Tensor aa_i, torch::Tensor aa_j,
    torch::Tensor ibas_ab, torch::Tensor ibas_aa, torch::Tensor row_type,
    torch::Tensor row_h, torch::Tensor row_local, int64_t n_primal) {
  check_double_cuda(y, "y");
  check_double_cuda(out, "out");
  TORCH_CHECK(y.device() == out.device(), "y and out must be on the same CUDA device.");
  TORCH_CHECK(out.numel() == n_primal, "dqg_sym_atu_range_add_out output has wrong size.");
  TORCH_CHECK(row_start >= 0 && row_stop >= row_start,
              "dqg_sym_atu_range_add_out received an invalid row range.");
  TORCH_CHECK(y.numel() == row_stop - row_start,
              "dqg_sym_atu_range_add_out y chunk size does not match row range.");
  check_sym_tensors(amopi, pitzer, symmetry, sym_table, gems_ab, gems_aa,
                    offsets, ab_offsets, aa_offsets, ab_i, ab_j, aa_i, aa_j,
                    ibas_ab, ibas_aa, row_type, row_h, row_local, nirrep, amo);
  TORCH_CHECK(row_stop <= row_type.numel(),
              "dqg_sym_atu_range_add_out row range exceeds row metadata.");

  constexpr int threads = 256;
  const int64_t row_count = y.numel();
  if (row_count == 0) {
    return;
  }
  const int64_t blocks = (row_count + threads - 1) / threads;
  dqg_sym_atu_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                       at::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<double>(), out.data_ptr<double>(), static_cast<int>(nirrep),
      static_cast<int>(amo), na, nb, amopi.data_ptr<int32_t>(),
      pitzer.data_ptr<int32_t>(), symmetry.data_ptr<int32_t>(),
      sym_table.data_ptr<int32_t>(), gems_ab.data_ptr<int32_t>(),
      gems_aa.data_ptr<int32_t>(), offsets.data_ptr<int64_t>(),
      ab_offsets.data_ptr<int64_t>(), aa_offsets.data_ptr<int64_t>(),
      ab_i.data_ptr<int32_t>(), ab_j.data_ptr<int32_t>(),
      aa_i.data_ptr<int32_t>(), aa_j.data_ptr<int32_t>(),
      ibas_ab.data_ptr<int32_t>(), ibas_aa.data_ptr<int32_t>(),
      row_type.data_ptr<int32_t>(), row_h.data_ptr<int32_t>(),
      row_local.data_ptr<int32_t>(), row_start, row_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dqg_sym_atu_range_out(
    torch::Tensor y, torch::Tensor out, int64_t row_start, int64_t row_stop,
    int64_t nirrep, int64_t amo, double na, double nb, torch::Tensor amopi,
    torch::Tensor pitzer, torch::Tensor symmetry, torch::Tensor sym_table,
    torch::Tensor gems_ab, torch::Tensor gems_aa, torch::Tensor offsets,
    torch::Tensor ab_offsets, torch::Tensor aa_offsets, torch::Tensor ab_i,
    torch::Tensor ab_j, torch::Tensor aa_i, torch::Tensor aa_j,
    torch::Tensor ibas_ab, torch::Tensor ibas_aa, torch::Tensor row_type,
    torch::Tensor row_h, torch::Tensor row_local, int64_t n_primal) {
  out.zero_();
  dqg_sym_atu_range_add_out(y, out, row_start, row_stop, nirrep, amo, na, nb,
                            amopi, pitzer, symmetry, sym_table, gems_ab,
                            gems_aa, offsets, ab_offsets, aa_offsets, ab_i,
                            ab_j, aa_i, aa_j, ibas_ab, ibas_aa, row_type,
                            row_h, row_local, n_primal);
}

void dqg_sym_atu_out(
    torch::Tensor y, torch::Tensor out, int64_t nirrep, int64_t amo, double na,
    double nb, torch::Tensor amopi, torch::Tensor pitzer,
    torch::Tensor symmetry, torch::Tensor sym_table, torch::Tensor gems_ab,
    torch::Tensor gems_aa, torch::Tensor offsets, torch::Tensor ab_offsets,
    torch::Tensor aa_offsets, torch::Tensor ab_i, torch::Tensor ab_j,
    torch::Tensor aa_i, torch::Tensor aa_j, torch::Tensor ibas_ab,
    torch::Tensor ibas_aa, torch::Tensor row_type, torch::Tensor row_h,
    torch::Tensor row_local, int64_t n_primal) {
  dqg_sym_atu_range_out(y, out, 0, y.numel(), nirrep, amo, na, nb, amopi,
                        pitzer, symmetry, sym_table, gems_ab, gems_aa, offsets,
                        ab_offsets, aa_offsets, ab_i, ab_j, aa_i, aa_j,
                        ibas_ab, ibas_aa, row_type, row_h, row_local,
                        n_primal);
}

torch::Tensor dqg_sym_atu(
    torch::Tensor y, int64_t nirrep, int64_t amo, double na, double nb,
    torch::Tensor amopi, torch::Tensor pitzer, torch::Tensor symmetry,
    torch::Tensor sym_table, torch::Tensor gems_ab, torch::Tensor gems_aa,
    torch::Tensor offsets, torch::Tensor ab_offsets, torch::Tensor aa_offsets,
    torch::Tensor ab_i, torch::Tensor ab_j, torch::Tensor aa_i,
    torch::Tensor aa_j, torch::Tensor ibas_ab, torch::Tensor ibas_aa,
    torch::Tensor row_type, torch::Tensor row_h, torch::Tensor row_local,
    int64_t n_primal) {
  auto out = torch::empty({n_primal}, y.options());
  dqg_sym_atu_out(y, out, nirrep, amo, na, nb, amopi, pitzer, symmetry,
                  sym_table, gems_ab, gems_aa, offsets, ab_offsets,
                  aa_offsets, ab_i, ab_j, aa_i, aa_j, ibas_ab, ibas_aa,
                  row_type, row_h, row_local, n_primal);
  return out;
}

torch::Tensor cg_update_y_r(torch::Tensor y, torch::Tensor r, torch::Tensor p,
                            torch::Tensor Ap, torch::Tensor alpha) {
  check_double_cuda(y, "y");
  check_double_cuda(r, "r");
  check_double_cuda(p, "p");
  check_double_cuda(Ap, "Ap");
  check_double_cuda(alpha, "alpha");
  TORCH_CHECK(y.device() == r.device() && y.device() == p.device() &&
                  y.device() == Ap.device() && y.device() == alpha.device(),
              "cg_update_y_r tensors must be on the same CUDA device.");
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
  check_double_cuda(p, "p");
  check_double_cuda(r, "r");
  check_double_cuda(beta, "beta");
  TORCH_CHECK(p.device() == r.device() && p.device() == beta.device(),
              "cg_update_p tensors must be on the same CUDA device.");
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

std::string dqg_sym_version() {
  return "dqg_sym_au_atu_20260623_v7_introwmeta_cg";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dqg_sym_au", &dqg_sym_au,
        "Symmetry-aware DQG matrix-free Au CUDA kernel");
  m.def("dqg_sym_au_out", &dqg_sym_au_out,
        "Symmetry-aware DQG matrix-free Au CUDA kernel with caller-provided output");
  m.def("dqg_sym_atu", &dqg_sym_atu,
        "Symmetry-aware DQG matrix-free ATu CUDA kernel");
  m.def("dqg_sym_atu_out", &dqg_sym_atu_out,
        "Symmetry-aware DQG matrix-free ATu CUDA kernel with caller-provided output");
  m.def("dqg_sym_atu_range_out", &dqg_sym_atu_range_out,
        "Symmetry-aware DQG matrix-free ATu CUDA kernel over a dual-row range");
  m.def("dqg_sym_atu_range_add_out", &dqg_sym_atu_range_add_out,
        "Symmetry-aware DQG matrix-free ATu CUDA kernel over a dual-row range, accumulating into output");
  m.def("cg_update_y_r", &cg_update_y_r,
        "Fused CUDA update y += alpha p, r -= alpha Ap, and ||r||^2");
  m.def("cg_update_p", &cg_update_p,
        "Fused CUDA update p = r + beta p");
  m.def("dqg_sym_version", &dqg_sym_version,
        "Symmetry-aware DQG matrix-free CUDA kernel version");
}
