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

struct DualOffsets {
  int64_t herm_d2aa;
  int64_t herm_d2bb;
  int64_t herm_d2ab;
  int64_t d1a_q1a;
  int64_t d1b_q1b;
  int64_t contract_d2ab_d1a;
  int64_t contract_d2ab_d1b;
  int64_t contract_d2aa_d1a;
  int64_t contract_d2bb_d1b;
  int64_t spin_trace;
  int64_t spin_d1a_eq_d1b;
  int64_t spin_d2aa_eq_d2bb;
  int64_t spin_d2aa_from_d2ab;
  int64_t spin_d2bb_from_d2ab;
  int64_t spin_d200;
  int64_t spin_d2ab_sym;
  int64_t spin_d200_nonsinglet;
  int64_t spin_g2ba_col_trace;
  int64_t spin_g2ba_row_trace;
  int64_t q2ab;
  int64_t q2aa;
  int64_t q2bb;
  int64_t g2ab;
  int64_t g2ba;
  int64_t g2aa;
};

__device__ __forceinline__ DualOffsets make_dual_offsets(int n, double na,
                                                         double nb) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
  const int64_t n2 = static_cast<int64_t>(n) * n;
  const int64_t gab2 = gab * gab;
  const int64_t gaa2 = gaa * gaa;
  const bool spin_singlet = fabs(na - nb) < 1.0e-12;

  DualOffsets offsets;
  int64_t pos = 3;
  offsets.herm_d2aa = pos;
  pos += gaa2;
  offsets.herm_d2bb = pos;
  pos += gaa2;
  offsets.herm_d2ab = pos;
  pos += gab2;
  offsets.d1a_q1a = pos;
  pos += n2;
  offsets.d1b_q1b = pos;
  pos += n2;
  offsets.contract_d2ab_d1a = pos;
  pos += n2;
  offsets.contract_d2ab_d1b = pos;
  pos += n2;
  offsets.contract_d2aa_d1a = pos;
  pos += n2;
  offsets.contract_d2bb_d1b = pos;
  pos += n2;
  offsets.spin_trace = pos;
  pos += 1;

  offsets.spin_d1a_eq_d1b = -1;
  offsets.spin_d2aa_eq_d2bb = -1;
  offsets.spin_d2aa_from_d2ab = -1;
  offsets.spin_d2bb_from_d2ab = -1;
  offsets.spin_d200 = -1;
  offsets.spin_d2ab_sym = -1;
  offsets.spin_d200_nonsinglet = -1;
  if (spin_singlet) {
    offsets.spin_d1a_eq_d1b = pos;
    pos += n2;
    offsets.spin_d2aa_eq_d2bb = pos;
    pos += gaa2;
    offsets.spin_d2aa_from_d2ab = pos;
    pos += gaa2;
    offsets.spin_d2bb_from_d2ab = pos;
    pos += gaa2;
    offsets.spin_d200 = pos;
    pos += gab2;
    offsets.spin_d2ab_sym = pos;
    pos += gab2;
  } else {
    offsets.spin_d200_nonsinglet = pos;
    pos += 4 * gab2;
  }

  offsets.spin_g2ba_col_trace = pos;
  pos += gab;
  offsets.spin_g2ba_row_trace = pos;
  pos += gab;
  offsets.q2ab = pos;
  pos += gab2;
  offsets.q2aa = pos;
  pos += gaa2;
  offsets.q2bb = pos;
  pos += gaa2;
  offsets.g2ab = pos;
  pos += gab2;
  offsets.g2ba = pos;
  pos += gab2;
  offsets.g2aa = pos;
  return offsets;
}

__device__ __forceinline__ double y_mat(const double *y, int64_t offset,
                                        int64_t dim, int64_t row, int64_t col) {
  return y[offset + row * dim + col];
}

__device__ __forceinline__ double
q2_same_spin_d1_value(const double *__restrict__ y, int64_t q2off, int n, int p,
                      int q) {
  const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
  double value = 0.0;
  for (int t = 0; t < n; ++t) {
    if (q < t && p < t) {
      value -= y_mat(y, q2off, gaa, aa_index(q, t, n), aa_index(p, t, n));
    }
    if (q < t && t < p) {
      value += y_mat(y, q2off, gaa, aa_index(q, t, n), aa_index(t, p, n));
    }
    if (t < q && p < t) {
      value += y_mat(y, q2off, gaa, aa_index(t, q, n), aa_index(p, t, n));
    }
    if (t < q && t < p) {
      value -= y_mat(y, q2off, gaa, aa_index(t, q, n), aa_index(t, p, n));
    }
  }
  return value;
}

__device__ __forceinline__ double
d1a_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                    int p, int q) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const int64_t dim_g2aa = 2 * gab;
  const bool spin_singlet = fabs(na - nb) < 1.0e-12;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);

  double value = y[offsets.d1a_q1a + static_cast<int64_t>(q) * n + p];
  value += nb * y[offsets.contract_d2ab_d1a + static_cast<int64_t>(p) * n + q];
  value += (na - 1.0) *
           y[offsets.contract_d2aa_d1a + static_cast<int64_t>(p) * n + q];
  if (spin_singlet) {
    value += y[offsets.spin_d1a_eq_d1b + static_cast<int64_t>(p) * n + q];
  }

  for (int t = 0; t < n; ++t) {
    value -= y_mat(y, offsets.q2ab, gab, ab_index(q, t, n), ab_index(p, t, n));
    value += y_mat(y, offsets.g2ab, gab, ab_index(p, t, n), ab_index(q, t, n));
    value +=
        y_mat(y, offsets.g2aa, dim_g2aa, ab_index(p, t, n), ab_index(q, t, n));
  }
  value += q2_same_spin_d1_value(y, offsets.q2aa, n, p, q);
  return value;
}

__device__ __forceinline__ double
d1b_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                    int p, int q) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const int64_t dim_g2aa = 2 * gab;
  const bool spin_singlet = fabs(na - nb) < 1.0e-12;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);

  double value = y[offsets.d1b_q1b + static_cast<int64_t>(q) * n + p];
  value += na * y[offsets.contract_d2ab_d1b + static_cast<int64_t>(p) * n + q];
  value += (nb - 1.0) *
           y[offsets.contract_d2bb_d1b + static_cast<int64_t>(p) * n + q];
  if (spin_singlet) {
    value -= y[offsets.spin_d1a_eq_d1b + static_cast<int64_t>(p) * n + q];
  }

  for (int t = 0; t < n; ++t) {
    value -= y_mat(y, offsets.q2ab, gab, ab_index(t, q, n), ab_index(t, p, n));
    value += y_mat(y, offsets.g2ba, gab, ab_index(p, t, n), ab_index(q, t, n));
    value += y_mat(y, offsets.g2aa, dim_g2aa, gab + ab_index(p, t, n),
                   gab + ab_index(q, t, n));
  }
  value += q2_same_spin_d1_value(y, offsets.q2bb, n, p, q);
  return value;
}

__device__ __forceinline__ int64_t transpose_ab(int64_t idx, int n) {
  const int i = static_cast<int>(idx % n);
  const int j = static_cast<int>(idx / n);
  return ab_index(j, i, n);
}

__device__ __forceinline__ double ab_diag_scale(int64_t idx, int n) {
  return (idx % n == idx / n) ? sqrt(2.0) : 1.0;
}

__device__ __forceinline__ bool ab_to_aa_orient(int64_t idx, int n, int64_t &aa,
                                                double &orient) {
  const int i = static_cast<int>(idx % n);
  const int j = static_cast<int>(idx / n);
  if (i == j) {
    return false;
  }
  aa = aa_index(i, j, n);
  orient = (i < j) ? 1.0 : -1.0;
  return true;
}

__device__ __forceinline__ double
d2ab_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                     int64_t a, int64_t b) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const int64_t dim_g2aa = 2 * gab;
  const bool spin_singlet = fabs(na - nb) < 1.0e-12;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);

  const int p = static_cast<int>(a % n);
  const int q = static_cast<int>(a / n);
  const int r = static_cast<int>(b % n);
  const int s = static_cast<int>(b / n);
  const int64_t at = transpose_ab(a, n);
  const int64_t bt = transpose_ab(b, n);

  double value = (a == b) ? y[0] : 0.0;
  value += y_mat(y, offsets.herm_d2ab, gab, a, b);
  value -= y_mat(y, offsets.herm_d2ab, gab, b, a);
  if (q == s) {
    value -= y[offsets.contract_d2ab_d1a + static_cast<int64_t>(p) * n + r];
  }
  if (p == r) {
    value -= y[offsets.contract_d2ab_d1b + static_cast<int64_t>(q) * n + s];
  }
  if (b == at) {
    value += y[offsets.spin_trace];
  }

  if (spin_singlet) {
    int64_t aa_a = 0;
    int64_t aa_b = 0;
    double orient_a = 0.0;
    double orient_b = 0.0;
    if (ab_to_aa_orient(a, n, aa_a, orient_a) &&
        ab_to_aa_orient(b, n, aa_b, orient_b)) {
      const double coeff = -0.5 * orient_a * orient_b;
      value += coeff * y_mat(y, offsets.spin_d2aa_from_d2ab,
                             static_cast<int64_t>(n) * (n - 1) / 2, aa_a, aa_b);
      value += coeff * y_mat(y, offsets.spin_d2bb_from_d2ab,
                             static_cast<int64_t>(n) * (n - 1) / 2, aa_a, aa_b);
    }

    const double dscale = -0.5 / (ab_diag_scale(a, n) * ab_diag_scale(b, n));
    value += dscale * (y_mat(y, offsets.spin_d200, gab, a, b) +
                       y_mat(y, offsets.spin_d200, gab, at, b) +
                       y_mat(y, offsets.spin_d200, gab, a, bt) +
                       y_mat(y, offsets.spin_d200, gab, at, bt));
    value += y_mat(y, offsets.spin_d2ab_sym, gab, a, b);
    value -= y_mat(y, offsets.spin_d2ab_sym, gab, at, bt);
  } else {
    const int64_t dim_d200 = 2 * gab;
    const double dij = ab_diag_scale(a, n);
    const double dkl = ab_diag_scale(b, n);
    const double tl = -0.5 / (dij * dkl);
    value += tl * (y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, a, b) +
                   y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, at, b) +
                   y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, a, bt) +
                   y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, at, bt));

    const double tr = 0.5 / dij;
    value += -tr * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, a, gab + b);
    value += tr * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, a, gab + bt);
    value +=
        -tr * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, at, gab + b);
    value +=
        tr * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, at, gab + bt);

    const double bl = 0.5 / dkl;
    value += -bl * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, gab + a, b);
    value +=
        -bl * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, gab + a, bt);
    value += bl * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, gab + at, b);
    value +=
        bl * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, gab + at, bt);

    value += -0.5 *
             y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, gab + a, gab + b);
    value += 0.5 * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, gab + at,
                         gab + b);
    value += 0.5 * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, gab + a,
                         gab + bt);
    value += -0.5 * y_mat(y, offsets.spin_d200_nonsinglet, dim_d200, gab + at,
                          gab + bt);
  }

  value += y_mat(y, offsets.q2ab, gab, a, b);
  value -= y_mat(y, offsets.g2ab, gab, ab_index(p, s, n), ab_index(r, q, n));
  value -= y_mat(y, offsets.g2ba, gab, ab_index(q, r, n), ab_index(s, p, n));
  value += y_mat(y, offsets.g2aa, dim_g2aa, ab_index(p, r, n),
                 gab + ab_index(s, q, n));
  value += y_mat(y, offsets.g2aa, dim_g2aa, gab + ab_index(q, s, n),
                 ab_index(r, p, n));
  return value;
}

__device__ __forceinline__ bool aa_contains_orbital(int64_t pair_idx, int t,
                                                    int &other) {
  int i = 0;
  int j = 0;
  aa_pair(pair_idx, i, j);
  if (i == t) {
    other = j;
    return true;
  }
  if (j == t) {
    other = i;
    return true;
  }
  return false;
}

__device__ __forceinline__ double
same_spin_contract_from_dual_value(const double *__restrict__ y, int64_t offset,
                                   int n, int ai, int aj, int bi, int bj) {
  double value = 0.0;
  if (ai == bi) {
    const int t = ai;
    const int p = aj;
    const int q = bj;
    value -= static_cast<double>(aa_sign(p, t) * aa_sign(q, t)) *
             y[offset + static_cast<int64_t>(p) * n + q];
  }
  if (ai == bj) {
    const int t = ai;
    const int p = aj;
    const int q = bi;
    value -= static_cast<double>(aa_sign(p, t) * aa_sign(q, t)) *
             y[offset + static_cast<int64_t>(p) * n + q];
  }
  if (aj == bi) {
    const int t = aj;
    const int p = ai;
    const int q = bj;
    value -= static_cast<double>(aa_sign(p, t) * aa_sign(q, t)) *
             y[offset + static_cast<int64_t>(p) * n + q];
  }
  if (aj == bj) {
    const int t = aj;
    const int p = ai;
    const int q = bi;
    value -= static_cast<double>(aa_sign(p, t) * aa_sign(q, t)) *
             y[offset + static_cast<int64_t>(p) * n + q];
  }
  return value;
}

__device__ __forceinline__ double
d2aa_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                     int64_t a, int64_t b) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
  const int64_t dim_g2aa = 2 * gab;
  const bool spin_singlet = fabs(na - nb) < 1.0e-12;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);

  int ai = 0;
  int aj = 0;
  int bi = 0;
  int bj = 0;
  aa_pair(a, ai, aj);
  aa_pair(b, bi, bj);

  double value = (a == b) ? 2.0 * y[1] : 0.0;
  value += y_mat(y, offsets.herm_d2aa, gaa, a, b);
  value -= y_mat(y, offsets.herm_d2aa, gaa, b, a);
  value += same_spin_contract_from_dual_value(y, offsets.contract_d2aa_d1a, n,
                                              ai, aj, bi, bj);

  if (spin_singlet) {
    value += y_mat(y, offsets.spin_d2aa_eq_d2bb, gaa, a, b);
    value += y_mat(y, offsets.spin_d2aa_from_d2ab, gaa, a, b);
  }

  value += y_mat(y, offsets.q2aa, gaa, a, b);

  const int a_i[2] = {ai, aj};
  const int a_l[2] = {aj, ai};
  const int b_k[2] = {bi, bj};
  const int b_j[2] = {bj, bi};
  for (int ao = 0; ao < 2; ++ao) {
    for (int bo = 0; bo < 2; ++bo) {
      const int i = a_i[ao];
      const int l = a_l[ao];
      const int k = b_k[bo];
      const int j = b_j[bo];
      const double coeff = -static_cast<double>(aa_sign(i, l) * aa_sign(k, j));
      value += coeff * y_mat(y, offsets.g2aa, dim_g2aa, ab_index(i, j, n),
                             ab_index(k, l, n));
    }
  }

  return value;
}

__device__ __forceinline__ double
d2bb_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                     int64_t a, int64_t b) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
  const int64_t dim_g2aa = 2 * gab;
  const bool spin_singlet = fabs(na - nb) < 1.0e-12;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);

  int ai = 0;
  int aj = 0;
  int bi = 0;
  int bj = 0;
  aa_pair(a, ai, aj);
  aa_pair(b, bi, bj);

  double value = (a == b) ? 2.0 * y[2] : 0.0;
  value += y_mat(y, offsets.herm_d2bb, gaa, a, b);
  value -= y_mat(y, offsets.herm_d2bb, gaa, b, a);
  value += same_spin_contract_from_dual_value(y, offsets.contract_d2bb_d1b, n,
                                              ai, aj, bi, bj);

  if (spin_singlet) {
    value -= y_mat(y, offsets.spin_d2aa_eq_d2bb, gaa, a, b);
    value += y_mat(y, offsets.spin_d2bb_from_d2ab, gaa, a, b);
  }

  value += y_mat(y, offsets.q2bb, gaa, a, b);

  const int a_i[2] = {ai, aj};
  const int a_l[2] = {aj, ai};
  const int b_k[2] = {bi, bj};
  const int b_j[2] = {bj, bi};
  for (int ao = 0; ao < 2; ++ao) {
    for (int bo = 0; bo < 2; ++bo) {
      const int i = a_i[ao];
      const int l = a_l[ao];
      const int k = b_k[bo];
      const int j = b_j[bo];
      const double coeff = -static_cast<double>(aa_sign(i, l) * aa_sign(k, j));
      value += coeff * y_mat(y, offsets.g2aa, dim_g2aa, gab + ab_index(i, j, n),
                             gab + ab_index(k, l, n));
    }
  }

  return value;
}

__device__ __forceinline__ double
d200_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                     int64_t a, int64_t b) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const bool spin_singlet = fabs(na - nb) < 1.0e-12;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  if (spin_singlet) {
    return y_mat(y, offsets.spin_d200, gab, a, b);
  }
  return y_mat(y, offsets.spin_d200_nonsinglet, 2 * gab, a, b);
}

__device__ __forceinline__ double
q1a_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                    int p, int q) {
  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  return y[offsets.d1a_q1a + static_cast<int64_t>(p) * n + q];
}

__device__ __forceinline__ double
q1b_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                    int p, int q) {
  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  return y[offsets.d1b_q1b + static_cast<int64_t>(p) * n + q];
}

__device__ __forceinline__ double
q2ab_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                     int64_t a, int64_t b) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  return -y_mat(y, offsets.q2ab, gab, a, b);
}

__device__ __forceinline__ double
q2aa_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                     int64_t a, int64_t b) {
  const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  return -y_mat(y, offsets.q2aa, gaa, a, b);
}

__device__ __forceinline__ double
q2bb_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                     int64_t a, int64_t b) {
  const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  return -y_mat(y, offsets.q2bb, gaa, a, b);
}

__device__ __forceinline__ double
g2ab_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                     int64_t a, int64_t b) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  return -y_mat(y, offsets.g2ab, gab, a, b);
}

__device__ __forceinline__ double
g2ba_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                     int64_t a, int64_t b) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  double value = -y_mat(y, offsets.g2ba, gab, a, b);
  const int b_i = static_cast<int>(b % n);
  const int b_j = static_cast<int>(b / n);
  if (b_i == b_j) {
    value += y[offsets.spin_g2ba_col_trace + a];
  }
  const int a_i = static_cast<int>(a % n);
  const int a_j = static_cast<int>(a / n);
  if (a_i == a_j) {
    value += y[offsets.spin_g2ba_row_trace + b];
  }
  return value;
}

__device__ __forceinline__ double
g2aa_from_dual_value(const double *__restrict__ y, int n, double na, double nb,
                     int64_t a, int64_t b) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  return -y_mat(y, offsets.g2aa, 2 * gab, a, b);
}

struct NormalCacheOffsets {
  int64_t d1a;
  int64_t d1b;
  int64_t d2ab;
  int64_t d2aa;
  int64_t d2bb;
  int64_t size;
};

__host__ __device__ __forceinline__ NormalCacheOffsets
make_normal_cache_offsets(int n) {
  const int64_t n2 = static_cast<int64_t>(n) * n;
  const int64_t gab = n2;
  const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
  const int64_t gab2 = gab * gab;
  const int64_t gaa2 = gaa * gaa;

  NormalCacheOffsets offsets;
  int64_t pos = 0;
  offsets.d1a = pos;
  pos += n2;
  offsets.d1b = pos;
  pos += n2;
  offsets.d2ab = pos;
  pos += gab2;
  offsets.d2aa = pos;
  pos += gaa2;
  offsets.d2bb = pos;
  pos += gaa2;
  offsets.size = pos;
  return offsets;
}

__device__ __forceinline__ double
d1a_normal_value(const double *__restrict__ y,
                 const double *__restrict__ normal_cache, int n, double na,
                 double nb, int p, int q) {
  if (normal_cache != nullptr) {
    const NormalCacheOffsets offsets = make_normal_cache_offsets(n);
    return normal_cache[offsets.d1a + static_cast<int64_t>(p) * n + q];
  }
  return d1a_from_dual_value(y, n, na, nb, p, q);
}

__device__ __forceinline__ double
d1b_normal_value(const double *__restrict__ y,
                 const double *__restrict__ normal_cache, int n, double na,
                 double nb, int p, int q) {
  if (normal_cache != nullptr) {
    const NormalCacheOffsets offsets = make_normal_cache_offsets(n);
    return normal_cache[offsets.d1b + static_cast<int64_t>(p) * n + q];
  }
  return d1b_from_dual_value(y, n, na, nb, p, q);
}

__device__ __forceinline__ double
d2ab_normal_value(const double *__restrict__ y,
                  const double *__restrict__ normal_cache, int n, double na,
                  double nb, int64_t a, int64_t b) {
  if (normal_cache != nullptr) {
    const int64_t gab = static_cast<int64_t>(n) * n;
    const NormalCacheOffsets offsets = make_normal_cache_offsets(n);
    return normal_cache[offsets.d2ab + a * gab + b];
  }
  return d2ab_from_dual_value(y, n, na, nb, a, b);
}

__device__ __forceinline__ double
d2aa_normal_value(const double *__restrict__ y,
                  const double *__restrict__ normal_cache, int n, double na,
                  double nb, int64_t a, int64_t b) {
  if (normal_cache != nullptr) {
    const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
    const NormalCacheOffsets offsets = make_normal_cache_offsets(n);
    return normal_cache[offsets.d2aa + a * gaa + b];
  }
  return d2aa_from_dual_value(y, n, na, nb, a, b);
}

__device__ __forceinline__ double
d2bb_normal_value(const double *__restrict__ y,
                  const double *__restrict__ normal_cache, int n, double na,
                  double nb, int64_t a, int64_t b) {
  if (normal_cache != nullptr) {
    const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
    const NormalCacheOffsets offsets = make_normal_cache_offsets(n);
    return normal_cache[offsets.d2bb + a * gaa + b];
  }
  return d2bb_from_dual_value(y, n, na, nb, a, b);
}

__global__ void
dqg_c1_normal_build_cache_kernel(const double *__restrict__ y,
                                 double *__restrict__ normal_cache, int n,
                                 double na, double nb, int64_t cache_size) {
  const int64_t n2 = static_cast<int64_t>(n) * n;
  const int64_t gab = n2;
  const int64_t gaa = static_cast<int64_t>(n) * (n - 1) / 2;
  const int64_t gab2 = gab * gab;
  const int64_t gaa2 = gaa * gaa;
  const NormalCacheOffsets offsets = make_normal_cache_offsets(n);

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < cache_size; idx += stride) {
    if (idx < offsets.d1b) {
      const int64_t local = idx - offsets.d1a;
      const int p = static_cast<int>(local / n);
      const int q = static_cast<int>(local % n);
      normal_cache[idx] = d1a_from_dual_value(y, n, na, nb, p, q);
    } else if (idx < offsets.d2ab) {
      const int64_t local = idx - offsets.d1b;
      const int p = static_cast<int>(local / n);
      const int q = static_cast<int>(local % n);
      normal_cache[idx] = d1b_from_dual_value(y, n, na, nb, p, q);
    } else if (idx < offsets.d2aa) {
      const int64_t local = idx - offsets.d2ab;
      const int64_t a = local / gab;
      const int64_t b = local % gab;
      normal_cache[idx] = d2ab_from_dual_value(y, n, na, nb, a, b);
    } else if (idx < offsets.d2bb) {
      const int64_t local = idx - offsets.d2aa;
      const int64_t a = local / gaa;
      const int64_t b = local % gaa;
      normal_cache[idx] = d2aa_from_dual_value(y, n, na, nb, a, b);
    } else {
      const int64_t local = idx - offsets.d2bb;
      const int64_t a = local / gaa;
      const int64_t b = local % gaa;
      normal_cache[idx] = d2bb_from_dual_value(y, n, na, nb, a, b);
    }
  }

  (void)n2;
  (void)gab2;
  (void)gaa2;
}

__global__ void dqg_c1_normal_q1_kernel(const double *__restrict__ y,
                                        double *__restrict__ out, int n,
                                        double na, double nb) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n2 = static_cast<int64_t>(n) * n;
  if (idx >= 2 * n2) {
    return;
  }

  const DualOffsets offsets = make_dual_offsets(n, na, nb);

  if (idx < n2) {
    const int64_t local = idx;
    const int i = static_cast<int>(local / n);
    const int j = static_cast<int>(local % n);
    const int64_t row = offsets.d1a_q1a + local;
    out[row] =
        d1a_from_dual_value(y, n, na, nb, j, i) + y[offsets.d1a_q1a + local];
    return;
  }

  const int64_t local = idx - n2;
  const int i = static_cast<int>(local / n);
  const int j = static_cast<int>(local % n);
  const int64_t row = offsets.d1b_q1b + local;
  out[row] =
      d1b_from_dual_value(y, n, na, nb, j, i) + y[offsets.d1b_q1b + local];
}

__global__ void dqg_c1_normal_q2ab_kernel(const double *__restrict__ y,
                                          double *__restrict__ out, int n,
                                          double na, double nb) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const int64_t gab2 = gab * gab;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= gab2) {
    return;
  }

  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  const int64_t ij = idx / gab;
  const int64_t kl = idx % gab;
  const int i = static_cast<int>(ij % n);
  const int j = static_cast<int>(ij / n);
  const int k = static_cast<int>(kl % n);
  const int l = static_cast<int>(kl / n);

  double value = d2ab_from_dual_value(y, n, na, nb, ij, kl);
  value += y[offsets.q2ab + idx];
  if (j == l) {
    value -= d1a_from_dual_value(y, n, na, nb, k, i);
  }
  if (i == k) {
    value -= d1b_from_dual_value(y, n, na, nb, l, j);
  }
  out[offsets.q2ab + idx] = value;
}

__global__ void dqg_c1_normal_g2ab_ba_kernel(const double *__restrict__ y,
                                             double *__restrict__ out, int n,
                                             double na, double nb) {
  const int64_t gab = static_cast<int64_t>(n) * n;
  const int64_t gab2 = gab * gab;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= 2 * gab2) {
    return;
  }

  const DualOffsets offsets = make_dual_offsets(n, na, nb);
  if (idx < gab2) {
    const int64_t ij = idx / gab;
    const int64_t kl = idx % gab;
    const int i = static_cast<int>(ij % n);
    const int j = static_cast<int>(ij / n);
    const int k = static_cast<int>(kl % n);
    const int l = static_cast<int>(kl / n);

    double value = y[offsets.g2ab + idx];
    if (j == l) {
      value += d1a_from_dual_value(y, n, na, nb, i, k);
    }
    value -= d2ab_from_dual_value(y, n, na, nb, ab_index(i, l, n),
                                  ab_index(k, j, n));
    out[offsets.g2ab + idx] = value;
    return;
  }

  const int64_t local = idx - gab2;
  const int64_t ij = local / gab;
  const int64_t kl = local % gab;
  const int i = static_cast<int>(ij % n);
  const int j = static_cast<int>(ij / n);
  const int k = static_cast<int>(kl % n);
  const int l = static_cast<int>(kl / n);

  double value = y[offsets.g2ba + local];
  if (j == l) {
    value += d1b_from_dual_value(y, n, na, nb, i, k);
  }
  value -=
      d2ab_from_dual_value(y, n, na, nb, ab_index(l, i, n), ab_index(j, k, n));
  out[offsets.g2ba + local] = value;
}

__device__ __forceinline__ double
dqg_c1_normal_direct_row_value(const double *__restrict__ y,
                               const double *__restrict__ normal_cache, int n,
                               double na, double nb, int64_t row) {
#define d1a_from_dual_value(Y, N, NA, NB, P, Q)                                \
  d1a_normal_value((Y), normal_cache, (N), (NA), (NB), (P), (Q))
#define d1b_from_dual_value(Y, N, NA, NB, P, Q)                                \
  d1b_normal_value((Y), normal_cache, (N), (NA), (NB), (P), (Q))
#define d2ab_from_dual_value(Y, N, NA, NB, A, B)                               \
  d2ab_normal_value((Y), normal_cache, (N), (NA), (NB), (A), (B))
#define d2aa_from_dual_value(Y, N, NA, NB, A, B)                               \
  d2aa_normal_value((Y), normal_cache, (N), (NA), (NB), (A), (B))
#define d2bb_from_dual_value(Y, N, NA, NB, A, B)                               \
  d2bb_normal_value((Y), normal_cache, (N), (NA), (NB), (A), (B))
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
        value += d2ab_from_dual_value(y, n, na, nb, ij, ij);
      }
    } else if (row == 1) {
      for (int64_t ij = 0; ij < gaa; ++ij) {
        value += 2.0 * d2aa_from_dual_value(y, n, na, nb, ij, ij);
      }
    } else {
      for (int64_t ij = 0; ij < gaa; ++ij) {
        value += 2.0 * d2bb_from_dual_value(y, n, na, nb, ij, ij);
      }
    }
    return value;
  }
  pos += 3;

  if (row < pos + gaa2) {
    const int64_t local = row - pos;
    const int64_t ij = local / gaa;
    const int64_t kl = local % gaa;
    return d2aa_from_dual_value(y, n, na, nb, ij, kl) -
           d2aa_from_dual_value(y, n, na, nb, kl, ij);
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    const int64_t local = row - pos;
    const int64_t ij = local / gaa;
    const int64_t kl = local % gaa;
    return d2bb_from_dual_value(y, n, na, nb, ij, kl) -
           d2bb_from_dual_value(y, n, na, nb, kl, ij);
  }
  pos += gaa2;

  if (row < pos + gab2) {
    const int64_t local = row - pos;
    const int64_t ij = local / gab;
    const int64_t kl = local % gab;
    return d2ab_from_dual_value(y, n, na, nb, ij, kl) -
           d2ab_from_dual_value(y, n, na, nb, kl, ij);
  }
  pos += gab2;

  if (row < pos + n2) {
    const int64_t local = row - pos;
    const int i = static_cast<int>(local / n);
    const int j = static_cast<int>(local % n);
    return d1a_from_dual_value(y, n, na, nb, j, i) +
           q1a_from_dual_value(y, n, na, nb, i, j);
  }
  pos += n2;

  if (row < pos + n2) {
    const int64_t local = row - pos;
    const int i = static_cast<int>(local / n);
    const int j = static_cast<int>(local % n);
    return d1b_from_dual_value(y, n, na, nb, j, i) +
           q1b_from_dual_value(y, n, na, nb, i, j);
  }
  pos += n2;

  if (row < pos + n2) {
    const int64_t local = row - pos;
    const int i = static_cast<int>(local / n);
    const int j = static_cast<int>(local % n);
    double value = nb * d1a_from_dual_value(y, n, na, nb, i, j);
    for (int k = 0; k < n; ++k) {
      value -= d2ab_from_dual_value(y, n, na, nb, ab_index(i, k, n),
                                    ab_index(j, k, n));
    }
    return value;
  }
  pos += n2;

  if (row < pos + n2) {
    const int64_t local = row - pos;
    const int i = static_cast<int>(local / n);
    const int j = static_cast<int>(local % n);
    double value = na * d1b_from_dual_value(y, n, na, nb, i, j);
    for (int k = 0; k < n; ++k) {
      value -= d2ab_from_dual_value(y, n, na, nb, ab_index(k, i, n),
                                    ab_index(k, j, n));
    }
    return value;
  }
  pos += n2;

  if (row < pos + n2) {
    const int64_t local = row - pos;
    const int i = static_cast<int>(local / n);
    const int j = static_cast<int>(local % n);
    double value = (na - 1.0) * d1a_from_dual_value(y, n, na, nb, i, j);
    for (int k = 0; k < n; ++k) {
      if (i == k || j == k) {
        continue;
      }
      value -= static_cast<double>(aa_sign(i, k) * aa_sign(j, k)) *
               d2aa_from_dual_value(y, n, na, nb, aa_index(i, k, n),
                                    aa_index(j, k, n));
    }
    return value;
  }
  pos += n2;

  if (row < pos + n2) {
    const int64_t local = row - pos;
    const int i = static_cast<int>(local / n);
    const int j = static_cast<int>(local % n);
    double value = (nb - 1.0) * d1b_from_dual_value(y, n, na, nb, i, j);
    for (int k = 0; k < n; ++k) {
      if (i == k || j == k) {
        continue;
      }
      value -= static_cast<double>(aa_sign(i, k) * aa_sign(j, k)) *
               d2bb_from_dual_value(y, n, na, nb, aa_index(i, k, n),
                                    aa_index(j, k, n));
    }
    return value;
  }
  pos += n2;

  if (row == pos) {
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        value += d2ab_from_dual_value(y, n, na, nb, ab_index(i, j, n),
                                      ab_index(j, i, n));
      }
    }
    return value;
  }
  pos += 1;

  if (spin_singlet) {
    return 0.0;
  }

  if (row < pos + 4 * gab2) {
    const int64_t local = row - pos;
    const int64_t dim = 2 * gab;
    const int64_t drow = local / dim;
    const int64_t dcol = local % dim;
    const int64_t ij = (drow < gab) ? drow : drow - gab;
    const int64_t kl = (dcol < gab) ? dcol : dcol - gab;
    const int i = static_cast<int>(ij % n);
    const int j = static_cast<int>(ij / n);
    const int k = static_cast<int>(kl % n);
    const int l = static_cast<int>(kl / n);
    const int64_t ji = ab_index(j, i, n);
    const int64_t lk = ab_index(l, k, n);
    const double dij = ab_diag_scale(ij, n);
    const double dkl = ab_diag_scale(kl, n);
    double value = d200_from_dual_value(y, n, na, nb, drow, dcol);
    if (drow < gab && dcol < gab) {
      const double v = -0.5 / (dij * dkl);
      value += v * (d2ab_from_dual_value(y, n, na, nb, ij, kl) +
                    d2ab_from_dual_value(y, n, na, nb, ji, kl) +
                    d2ab_from_dual_value(y, n, na, nb, ij, lk) +
                    d2ab_from_dual_value(y, n, na, nb, ji, lk));
    } else if (drow < gab) {
      const double v = 0.5 / dij;
      value += -v * d2ab_from_dual_value(y, n, na, nb, ij, kl);
      value += v * d2ab_from_dual_value(y, n, na, nb, ij, lk);
      value += -v * d2ab_from_dual_value(y, n, na, nb, ji, kl);
      value += v * d2ab_from_dual_value(y, n, na, nb, ji, lk);
    } else if (dcol < gab) {
      const double v = 0.5 / dkl;
      value += -v * d2ab_from_dual_value(y, n, na, nb, ij, kl);
      value += -v * d2ab_from_dual_value(y, n, na, nb, ij, lk);
      value += v * d2ab_from_dual_value(y, n, na, nb, ji, kl);
      value += v * d2ab_from_dual_value(y, n, na, nb, ji, lk);
    } else {
      value += -0.5 * d2ab_from_dual_value(y, n, na, nb, ij, kl);
      value += 0.5 * d2ab_from_dual_value(y, n, na, nb, ji, kl);
      value += 0.5 * d2ab_from_dual_value(y, n, na, nb, ij, lk);
      value += -0.5 * d2ab_from_dual_value(y, n, na, nb, ji, lk);
    }
    return value;
  }
  pos += 4 * gab2;

  if (row < pos + gab) {
    const int64_t kl = row - pos;
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
      value += g2ba_from_dual_value(y, n, na, nb, kl, ab_index(i, i, n));
    }
    return value;
  }
  pos += gab;

  if (row < pos + gab) {
    const int64_t kl = row - pos;
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
      value += g2ba_from_dual_value(y, n, na, nb, ab_index(i, i, n), kl);
    }
    return value;
  }
  pos += gab;

  if (row < pos + gab2) {
    const int64_t local = row - pos;
    const int64_t ij = local / gab;
    const int64_t kl = local % gab;
    const int i = static_cast<int>(ij % n);
    const int j = static_cast<int>(ij / n);
    const int k = static_cast<int>(kl % n);
    const int l = static_cast<int>(kl / n);
    double value = d2ab_from_dual_value(y, n, na, nb, ij, kl) -
                   q2ab_from_dual_value(y, n, na, nb, ij, kl);
    if (j == l) {
      value -= d1a_from_dual_value(y, n, na, nb, k, i);
    }
    if (i == k) {
      value -= d1b_from_dual_value(y, n, na, nb, l, j);
    }
    return value;
  }
  pos += gab2;

  if (row < pos + gaa2) {
    const int64_t local = row - pos;
    const int64_t ij = local / gaa;
    const int64_t kl = local % gaa;
    int i = 0;
    int j = 0;
    int k = 0;
    int l = 0;
    aa_pair(ij, i, j);
    aa_pair(kl, k, l);
    double value = d2aa_from_dual_value(y, n, na, nb, ij, kl) -
                   q2aa_from_dual_value(y, n, na, nb, ij, kl);
    if (j == l)
      value -= d1a_from_dual_value(y, n, na, nb, k, i);
    if (j == k)
      value += d1a_from_dual_value(y, n, na, nb, l, i);
    if (i == l)
      value += d1a_from_dual_value(y, n, na, nb, k, j);
    if (i == k)
      value -= d1a_from_dual_value(y, n, na, nb, l, j);
    return value;
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    const int64_t local = row - pos;
    const int64_t ij = local / gaa;
    const int64_t kl = local % gaa;
    int i = 0;
    int j = 0;
    int k = 0;
    int l = 0;
    aa_pair(ij, i, j);
    aa_pair(kl, k, l);
    double value = d2bb_from_dual_value(y, n, na, nb, ij, kl) -
                   q2bb_from_dual_value(y, n, na, nb, ij, kl);
    if (j == l)
      value -= d1b_from_dual_value(y, n, na, nb, k, i);
    if (j == k)
      value += d1b_from_dual_value(y, n, na, nb, l, i);
    if (i == l)
      value += d1b_from_dual_value(y, n, na, nb, k, j);
    if (i == k)
      value -= d1b_from_dual_value(y, n, na, nb, l, j);
    return value;
  }
  pos += gaa2;

  if (row < pos + gab2) {
    const int64_t local = row - pos;
    const int64_t ij = local / gab;
    const int64_t kl = local % gab;
    const int i = static_cast<int>(ij % n);
    const int j = static_cast<int>(ij / n);
    const int k = static_cast<int>(kl % n);
    const int l = static_cast<int>(kl / n);
    double value = -g2ab_from_dual_value(y, n, na, nb, ij, kl);
    if (j == l) {
      value += d1a_from_dual_value(y, n, na, nb, i, k);
    }
    value -= d2ab_from_dual_value(y, n, na, nb, ab_index(i, l, n),
                                  ab_index(k, j, n));
    return value;
  }
  pos += gab2;

  if (row < pos + gab2) {
    const int64_t local = row - pos;
    const int64_t ij = local / gab;
    const int64_t kl = local % gab;
    const int i = static_cast<int>(ij % n);
    const int j = static_cast<int>(ij / n);
    const int k = static_cast<int>(kl % n);
    const int l = static_cast<int>(kl / n);
    double value = -g2ba_from_dual_value(y, n, na, nb, ij, kl);
    if (j == l) {
      value += d1b_from_dual_value(y, n, na, nb, i, k);
    }
    value -= d2ab_from_dual_value(y, n, na, nb, ab_index(l, i, n),
                                  ab_index(j, k, n));
    return value;
  }
  pos += gab2;

  const int64_t local = row - pos;
  const int64_t dim = 2 * gab;
  const int64_t a = local / dim;
  const int64_t b = local % dim;

  if (a < gab && b < gab) {
    const int i = static_cast<int>(a % n);
    const int j = static_cast<int>(a / n);
    const int k = static_cast<int>(b % n);
    const int l = static_cast<int>(b / n);
    double value = -g2aa_from_dual_value(y, n, na, nb, a, b);
    if (j == l) {
      value += d1a_from_dual_value(y, n, na, nb, i, k);
    }
    if (i != l && k != j) {
      value -= static_cast<double>(aa_sign(i, l) * aa_sign(k, j)) *
               d2aa_from_dual_value(y, n, na, nb, aa_index(i, l, n),
                                    aa_index(k, j, n));
    }
    return value;
  }

  if (a >= gab && b >= gab) {
    const int64_t aa = a - gab;
    const int64_t bb = b - gab;
    const int i = static_cast<int>(aa % n);
    const int j = static_cast<int>(aa / n);
    const int k = static_cast<int>(bb % n);
    const int l = static_cast<int>(bb / n);
    double value = -g2aa_from_dual_value(y, n, na, nb, a, b);
    if (j == l) {
      value += d1b_from_dual_value(y, n, na, nb, i, k);
    }
    if (i != l && k != j) {
      value -= static_cast<double>(aa_sign(i, l) * aa_sign(k, j)) *
               d2bb_from_dual_value(y, n, na, nb, aa_index(i, l, n),
                                    aa_index(k, j, n));
    }
    return value;
  }

  if (a < gab) {
    const int64_t bb = b - gab;
    const int i = static_cast<int>(a % n);
    const int j = static_cast<int>(a / n);
    const int k = static_cast<int>(bb % n);
    const int l = static_cast<int>(bb / n);
    return -g2aa_from_dual_value(y, n, na, nb, a, b) +
           d2ab_from_dual_value(y, n, na, nb, ab_index(i, l, n),
                                ab_index(j, k, n));
  }

  const int64_t aa = a - gab;
  const int i = static_cast<int>(aa % n);
  const int j = static_cast<int>(aa / n);
  const int k = static_cast<int>(b % n);
  const int l = static_cast<int>(b / n);
  return -g2aa_from_dual_value(y, n, na, nb, a, b) +
         d2ab_from_dual_value(y, n, na, nb, ab_index(l, i, n),
                              ab_index(k, j, n));
#undef d1a_from_dual_value
#undef d1b_from_dual_value
#undef d2ab_from_dual_value
#undef d2aa_from_dual_value
#undef d2bb_from_dual_value
}

__global__ void dqg_c1_normal_direct_range_kernel(
    const double *__restrict__ y, const double *__restrict__ normal_cache,
    double *__restrict__ out, int n, double na, double nb, int64_t row_start,
    int64_t row_count) {
  int64_t local = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= row_count) {
    return;
  }
  const int64_t row = row_start + local;
  out[local] = dqg_c1_normal_direct_row_value(y, normal_cache, n, na, nb, row);
}

__global__ void cg_update_y_r_kernel(double *__restrict__ y,
                                     double *__restrict__ r,
                                     const double *__restrict__ p,
                                     const double *__restrict__ Ap,
                                     const double *__restrict__ alpha_ptr,
                                     double *__restrict__ rr_out, int64_t n) {
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

__global__ void
dqg_c1_au_kernel(const double *__restrict__ x, double *__restrict__ out, int n,
                 double na, double nb, int64_t d2aboff, int64_t d2aaoff,
                 int64_t d2bboff, int64_t d200off, int64_t d1aoff,
                 int64_t d1boff, int64_t q1aoff, int64_t q1boff,
                 int64_t q2aboff, int64_t q2aaoff, int64_t q2bboff,
                 int64_t g2aboff, int64_t g2baoff, int64_t g2aaoff,
                 int64_t n_dual, int64_t row_start, int64_t row_count) {

  // One thread per output dual row. The full-Au launcher passes
  // row_start=0, row_count=n_dual (so row==local and out is unbiased). The
  // range launcher (dqg_c1_au_range_out) passes a sub-range and a
  // row_count-sized output buffer. Bias out by -row_start so the existing
  // out[row] writes land at out_real[row - row_start]. Only in-range indices
  // are dereferenced.
  int64_t range_local =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (range_local >= row_count) {
    return;
  }
  int64_t row = row_start + range_local;
  out = out - row_start;

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
    out[row] =
        mat_get(x, d2aaoff, gaa, ij, kl) - mat_get(x, d2aaoff, gaa, kl, ij);
    return;
  }
  pos += gaa2;

  if (row < pos + gaa2) {
    int64_t local = row - pos;
    int64_t ij = local / gaa;
    int64_t kl = local % gaa;
    out[row] =
        mat_get(x, d2bboff, gaa, ij, kl) - mat_get(x, d2bboff, gaa, kl, ij);
    return;
  }
  pos += gaa2;

  if (row < pos + gab2) {
    int64_t local = row - pos;
    int64_t ij = local / gab;
    int64_t kl = local % gab;
    out[row] =
        mat_get(x, d2aboff, gab, ij, kl) - mat_get(x, d2aboff, gab, kl, ij);
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
      value -= mat_get(x, d2aboff, gab, ab_index(i, k, n), ab_index(j, k, n));
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
      value -= mat_get(x, d2aboff, gab, ab_index(k, i, n), ab_index(k, j, n));
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
               mat_get(x, d2aaoff, gaa, aa_index(i, k, n), aa_index(j, k, n));
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
               mat_get(x, d2bboff, gaa, aa_index(i, k, n), aa_index(j, k, n));
    }
    out[row] = value;
    return;
  }
  pos += n2;

  if (row == pos) {
    double value = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        value += mat_get(x, d2aboff, gab, ab_index(i, j, n), ab_index(j, i, n));
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
                 mat_get(x, d2aboff, gab, ab_index(j, i, n), ab_index(l, k, n));
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
        value += v * mat_get(x, d2aboff, gab, ij, lk);
        value += -v * mat_get(x, d2aboff, gab, ji, kl);
        value += v * mat_get(x, d2aboff, gab, ji, lk);
      } else if (dcol < gab) {
        double v = 0.5 / dkl;
        value += -v * mat_get(x, d2aboff, gab, ij, kl);
        value += -v * mat_get(x, d2aboff, gab, ij, lk);
        value += v * mat_get(x, d2aboff, gab, ji, kl);
        value += v * mat_get(x, d2aboff, gab, ji, lk);
      } else {
        value += -0.5 * mat_get(x, d2aboff, gab, ij, kl);
        value += 0.5 * mat_get(x, d2aboff, gab, ji, kl);
        value += 0.5 * mat_get(x, d2aboff, gab, ij, lk);
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
    double value =
        mat_get(x, d2aboff, gab, ij, kl) - mat_get(x, q2aboff, gab, ij, kl);
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
    double value =
        mat_get(x, d2aaoff, gaa, ij, kl) - mat_get(x, q2aaoff, gaa, ij, kl);
    if (j == l)
      value -= mat_get(x, d1aoff, n, k, i);
    if (j == k)
      value += mat_get(x, d1aoff, n, l, i);
    if (i == l)
      value += mat_get(x, d1aoff, n, k, j);
    if (i == k)
      value -= mat_get(x, d1aoff, n, l, j);
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
    double value =
        mat_get(x, d2bboff, gaa, ij, kl) - mat_get(x, q2bboff, gaa, ij, kl);
    if (j == l)
      value -= mat_get(x, d1boff, n, k, i);
    if (j == k)
      value += mat_get(x, d1boff, n, l, i);
    if (i == l)
      value += mat_get(x, d1boff, n, k, j);
    if (i == k)
      value -= mat_get(x, d1boff, n, l, j);
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
    value -= mat_get(x, d2aboff, gab, ab_index(i, l, n), ab_index(k, j, n));
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
    value -= mat_get(x, d2aboff, gab, ab_index(l, i, n), ab_index(j, k, n));
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
               mat_get(x, d2aaoff, gaa, aa_index(i, l, n), aa_index(k, j, n));
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
               mat_get(x, d2bboff, gaa, aa_index(i, l, n), aa_index(k, j, n));
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
               mat_get(x, d2aboff, gab, ab_index(i, l, n), ab_index(j, k, n));
    return;
  }

  int64_t aa = a - gab;
  int i = static_cast<int>(aa % n);
  int j = static_cast<int>(aa / n);
  int k = static_cast<int>(b % n);
  int l = static_cast<int>(b / n);
  // Mirrors d2ab4.permute(1, 3, 2, 0): T[i,j,k,l] <- D2ab[l,i,k,j].
  out[row] = -mat_get(x, g2aaoff, dim, a, b) +
             mat_get(x, d2aboff, gab, ab_index(l, i, n), ab_index(k, j, n));
}

__device__ __forceinline__ void atomic_emit(double *out, int64_t col,
                                            double value) {
  if (value != 0.0) {
    atomicAdd(out + col, value);
  }
}

__global__ void
dqg_c1_atu_kernel(const double *__restrict__ y, double *__restrict__ out, int n,
                  double na, double nb, int64_t d2aboff, int64_t d2aaoff,
                  int64_t d2bboff, int64_t d200off, int64_t d1aoff,
                  int64_t d1boff, int64_t q1aoff, int64_t q1boff,
                  int64_t q2aboff, int64_t q2aaoff, int64_t q2bboff,
                  int64_t g2aboff, int64_t g2baoff, int64_t g2aaoff,
                  int64_t row_start, int64_t row_count) {

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
      atomic_emit(out, d2aboff + ab_index(i, k, n) * gab + ab_index(j, k, n),
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
      atomic_emit(out, d2aboff + ab_index(k, i, n) * gab + ab_index(k, j, n),
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
      atomic_emit(out, d2aaoff + aa_index(i, k, n) * gaa + aa_index(j, k, n),
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
      atomic_emit(out, d2bboff + aa_index(i, k, n) * gaa + aa_index(j, k, n),
                  -static_cast<double>(sik * sjk) * yr);
    }
    return;
  }
  pos += n2;

  if (row == pos) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        atomic_emit(out, d2aboff + ab_index(i, j, n) * gab + ab_index(j, i, n),
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
        atomic_emit(out, d2aboff + ij * gab + lk, v * yr);
        atomic_emit(out, d2aboff + ji * gab + kl, -v * yr);
        atomic_emit(out, d2aboff + ji * gab + lk, v * yr);
      } else if (dcol < gab) {
        double v = 0.5 / dkl;
        atomic_emit(out, d2aboff + ij * gab + kl, -v * yr);
        atomic_emit(out, d2aboff + ij * gab + lk, -v * yr);
        atomic_emit(out, d2aboff + ji * gab + kl, v * yr);
        atomic_emit(out, d2aboff + ji * gab + lk, v * yr);
      } else {
        atomic_emit(out, d2aboff + ij * gab + kl, -0.5 * yr);
        atomic_emit(out, d2aboff + ji * gab + kl, 0.5 * yr);
        atomic_emit(out, d2aboff + ij * gab + lk, 0.5 * yr);
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
    if (j == l)
      atomic_emit(out, d1aoff + k * n + i, -yr);
    if (j == k)
      atomic_emit(out, d1aoff + l * n + i, yr);
    if (i == l)
      atomic_emit(out, d1aoff + k * n + j, yr);
    if (i == k)
      atomic_emit(out, d1aoff + l * n + j, -yr);
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
    if (j == l)
      atomic_emit(out, d1boff + k * n + i, -yr);
    if (j == k)
      atomic_emit(out, d1boff + l * n + i, yr);
    if (i == l)
      atomic_emit(out, d1boff + k * n + j, yr);
    if (i == k)
      atomic_emit(out, d1boff + l * n + j, -yr);
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
      atomic_emit(out, d2aaoff + aa_index(i, l, n) * gaa + aa_index(k, j, n),
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
      atomic_emit(out, d2bboff + aa_index(i, l, n) * gaa + aa_index(k, j, n),
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
    atomic_emit(out, d2aboff + ab_index(i, l, n) * gab + ab_index(j, k, n), yr);
    return;
  }

  int64_t aa = a - gab;
  int i = static_cast<int>(aa % n);
  int j = static_cast<int>(aa / n);
  int k = static_cast<int>(b % n);
  int l = static_cast<int>(b / n);
  atomic_emit(out, g2aaoff + a * dim + b, -yr);
  atomic_emit(out, d2aboff + ab_index(l, i, n) * gab + ab_index(k, j, n), yr);
}

} // namespace

void dqg_c1_au_out(torch::Tensor x, torch::Tensor out, int64_t n, double na,
                   double nb, int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
                   int64_t d200off, int64_t d1aoff, int64_t d1boff,
                   int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
                   int64_t q2aaoff, int64_t q2bboff, int64_t g2aboff,
                   int64_t g2baoff, int64_t g2aaoff, int64_t n_dual) {
  TORCH_CHECK(x.is_cuda(), "dqg_c1_au expects a CUDA tensor.");
  TORCH_CHECK(out.is_cuda(), "dqg_c1_au_out expects a CUDA output tensor.");
  TORCH_CHECK(
      x.device() == out.device(),
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
      q2aboff, q2aaoff, q2bboff, g2aboff, g2baoff, g2aaoff, n_dual,
      /*row_start=*/0, /*row_count=*/n_dual);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor dqg_c1_au(torch::Tensor x, int64_t n, double na, double nb,
                        int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
                        int64_t d200off, int64_t d1aoff, int64_t d1boff,
                        int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
                        int64_t q2aaoff, int64_t q2bboff, int64_t g2aboff,
                        int64_t g2baoff, int64_t g2aaoff, int64_t n_dual) {
  auto out = torch::empty({n_dual}, x.options());
  dqg_c1_au_out(x, out, n, na, nb, d2aboff, d2aaoff, d2bboff, d200off, d1aoff,
                d1boff, q1aoff, q1boff, q2aboff, q2aaoff, q2bboff, g2aboff,
                g2baoff, g2aaoff, n_dual);
  return out;
}

// Row-range variant of Au. Compute the dual output rows [row_start, row_stop)
// of A*x into a row_count-sized out, reading the primal x at the block
// offsets passed in. For the distributed/sharded operator the caller passes a
// compact primal buffer (owned large blocks + shared, reduced D2/D1) together
// with remapped offsets, plus the device's owned dual-row range so each
// device computes only its owned rows from data it actually holds. With the
// standard global offsets and row_start=0, row_stop=n_dual this is identical to
// dqg_c1_au_out (the validation path).
void dqg_c1_au_range_out(torch::Tensor x, torch::Tensor out, int64_t row_start,
                         int64_t row_stop, int64_t n, double na, double nb,
                         int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
                         int64_t d200off, int64_t d1aoff, int64_t d1boff,
                         int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
                         int64_t q2aaoff, int64_t q2bboff, int64_t g2aboff,
                         int64_t g2baoff, int64_t g2aaoff, int64_t n_dual) {
  TORCH_CHECK(x.is_cuda(), "dqg_c1_au_range_out expects a CUDA tensor.");
  TORCH_CHECK(out.is_cuda(),
              "dqg_c1_au_range_out expects a CUDA output tensor.");
  TORCH_CHECK(
      x.device() == out.device(),
      "dqg_c1_au_range_out input and output must be on the same device.");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Double,
              "dqg_c1_au_range_out currently expects float64 input.");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Double,
              "dqg_c1_au_range_out currently expects float64 output.");
  TORCH_CHECK(x.is_contiguous(),
              "dqg_c1_au_range_out expects contiguous input.");
  TORCH_CHECK(out.is_contiguous(),
              "dqg_c1_au_range_out expects contiguous output.");
  TORCH_CHECK(n > 0 && n <= 4096, "dqg_c1_au_range_out received invalid n.");
  TORCH_CHECK(row_start >= 0 && row_stop >= row_start && row_stop <= n_dual,
              "dqg_c1_au_range_out received an invalid row range.");
  TORCH_CHECK(out.numel() == row_stop - row_start,
              "dqg_c1_au_range_out output size does not match the row range.");
  // note that x may be a compact primal buffer with remapped offsets, so we
  // deliberately do not check x.numel() against any global n_primal.

  const int64_t row_count = row_stop - row_start;
  if (row_count == 0) {
    return;
  }
  constexpr int threads = 256;
  const int64_t blocks = (row_count + threads - 1) / threads;
  dqg_c1_au_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<double>(), out.data_ptr<double>(), static_cast<int>(n), na, nb,
      d2aboff, d2aaoff, d2bboff, d200off, d1aoff, d1boff, q1aoff, q1boff,
      q2aboff, q2aaoff, q2bboff, g2aboff, g2baoff, g2aaoff, n_dual, row_start,
      row_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dqg_c1_atu_range_out(torch::Tensor y, torch::Tensor out, int64_t row_start,
                          int64_t row_stop, int64_t n, double na, double nb,
                          int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
                          int64_t d200off, int64_t d1aoff, int64_t d1boff,
                          int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
                          int64_t q2aaoff, int64_t q2bboff, int64_t g2aboff,
                          int64_t g2baoff, int64_t g2aaoff, int64_t n_primal);

void dqg_c1_atu_range_add_out(torch::Tensor y, torch::Tensor out,
                              int64_t row_start, int64_t row_stop, int64_t n,
                              double na, double nb, int64_t d2aboff,
                              int64_t d2aaoff, int64_t d2bboff, int64_t d200off,
                              int64_t d1aoff, int64_t d1boff, int64_t q1aoff,
                              int64_t q1boff, int64_t q2aboff, int64_t q2aaoff,
                              int64_t q2bboff, int64_t g2aboff, int64_t g2baoff,
                              int64_t g2aaoff, int64_t n_primal);

void dqg_c1_atu_out(torch::Tensor y, torch::Tensor out, int64_t n, double na,
                    double nb, int64_t d2aboff, int64_t d2aaoff,
                    int64_t d2bboff, int64_t d200off, int64_t d1aoff,
                    int64_t d1boff, int64_t q1aoff, int64_t q1boff,
                    int64_t q2aboff, int64_t q2aaoff, int64_t q2bboff,
                    int64_t g2aboff, int64_t g2baoff, int64_t g2aaoff,
                    int64_t n_primal) {
  dqg_c1_atu_range_out(y, out, 0, y.numel(), n, na, nb, d2aboff, d2aaoff,
                       d2bboff, d200off, d1aoff, d1boff, q1aoff, q1boff,
                       q2aboff, q2aaoff, q2bboff, g2aboff, g2baoff, g2aaoff,
                       n_primal);
}

void dqg_c1_atu_range_out(torch::Tensor y, torch::Tensor out, int64_t row_start,
                          int64_t row_stop, int64_t n, double na, double nb,
                          int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
                          int64_t d200off, int64_t d1aoff, int64_t d1boff,
                          int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
                          int64_t q2aaoff, int64_t q2bboff, int64_t g2aboff,
                          int64_t g2baoff, int64_t g2aaoff, int64_t n_primal) {
  out.zero_();
  dqg_c1_atu_range_add_out(y, out, row_start, row_stop, n, na, nb, d2aboff,
                           d2aaoff, d2bboff, d200off, d1aoff, d1boff, q1aoff,
                           q1boff, q2aboff, q2aaoff, q2bboff, g2aboff, g2baoff,
                           g2aaoff, n_primal);
}

void dqg_c1_atu_range_add_out(torch::Tensor y, torch::Tensor out,
                              int64_t row_start, int64_t row_stop, int64_t n,
                              double na, double nb, int64_t d2aboff,
                              int64_t d2aaoff, int64_t d2bboff, int64_t d200off,
                              int64_t d1aoff, int64_t d1boff, int64_t q1aoff,
                              int64_t q1boff, int64_t q2aboff, int64_t q2aaoff,
                              int64_t q2bboff, int64_t g2aboff, int64_t g2baoff,
                              int64_t g2aaoff, int64_t n_primal) {
  TORCH_CHECK(y.is_cuda(), "dqg_c1_atu expects a CUDA tensor.");
  TORCH_CHECK(out.is_cuda(), "dqg_c1_atu_out expects a CUDA output tensor.");
  TORCH_CHECK(
      y.device() == out.device(),
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
                         int64_t g2baoff, int64_t g2aaoff, int64_t n_primal) {
  auto out = torch::empty({n_primal}, y.options());
  dqg_c1_atu_out(y, out, n, na, nb, d2aboff, d2aaoff, d2bboff, d200off, d1aoff,
                 d1boff, q1aoff, q1boff, q2aboff, q2aaoff, q2bboff, g2aboff,
                 g2baoff, g2aaoff, n_primal);
  return out;
}

int64_t dqg_c1_normal_direct_cache_size(int64_t n) {
  TORCH_CHECK(n > 0 && n <= 4096,
              "dqg_c1_normal_direct_cache_size received invalid n.");
  return make_normal_cache_offsets(static_cast<int>(n)).size;
}

void dqg_c1_normal_direct_build_cache_out(torch::Tensor y, torch::Tensor cache,
                                          int64_t n, double na, double nb,
                                          int64_t n_dual) {
  TORCH_CHECK(y.is_cuda() && cache.is_cuda(),
              "dqg_c1_normal_direct_build_cache_out expects CUDA tensors.");
  TORCH_CHECK(y.device() == cache.device(),
              "dqg_c1_normal_direct_build_cache_out tensors must be on the "
              "same CUDA device.");
  TORCH_CHECK(y.scalar_type() == at::ScalarType::Double &&
                  cache.scalar_type() == at::ScalarType::Double,
              "dqg_c1_normal_direct_build_cache_out currently expects float64 "
              "tensors.");
  TORCH_CHECK(
      y.is_contiguous() && cache.is_contiguous(),
      "dqg_c1_normal_direct_build_cache_out expects contiguous tensors.");
  TORCH_CHECK(
      y.numel() == n_dual,
      "dqg_c1_normal_direct_build_cache_out input has wrong dual size.");
  TORCH_CHECK(n > 0 && n <= 4096,
              "dqg_c1_normal_direct_build_cache_out received invalid n.");
  TORCH_CHECK(fabs(na - nb) >= 1.0e-12,
              "dqg_c1_normal_direct_build_cache_out currently supports the "
              "open-shell DQG/C1 layout only.");
  const int64_t cache_size =
      make_normal_cache_offsets(static_cast<int>(n)).size;
  TORCH_CHECK(cache.numel() == cache_size,
              "dqg_c1_normal_direct_build_cache_out cache has wrong size.");

  constexpr int threads = 256;
  const int64_t raw_blocks = (cache_size + threads - 1) / threads;
  const int blocks = static_cast<int>(std::min<int64_t>(raw_blocks, 65535));
  dqg_c1_normal_build_cache_kernel<<<blocks, threads, 0,
                                     at::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<double>(), cache.data_ptr<double>(), static_cast<int>(n), na,
      nb, cache_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dqg_c1_normal_direct_cached_range_out(
    torch::Tensor y, torch::Tensor cache, torch::Tensor out, int64_t row_start,
    int64_t row_stop, int64_t n, double na, double nb, int64_t d2aboff,
    int64_t d2aaoff, int64_t d2bboff, int64_t d200off, int64_t d1aoff,
    int64_t d1boff, int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
    int64_t q2aaoff, int64_t q2bboff, int64_t g2aboff, int64_t g2baoff,
    int64_t g2aaoff, int64_t n_primal, int64_t n_dual) {
  (void)d2aboff;
  (void)d2aaoff;
  (void)d2bboff;
  (void)d200off;
  (void)d1aoff;
  (void)d1boff;
  (void)q1aoff;
  (void)q1boff;
  (void)q2aboff;
  (void)q2aaoff;
  (void)q2bboff;
  (void)g2aboff;
  (void)g2baoff;
  (void)g2aaoff;
  (void)n_primal;

  TORCH_CHECK(y.is_cuda() && cache.is_cuda() && out.is_cuda(),
              "dqg_c1_normal_direct_cached_range_out expects CUDA tensors.");
  TORCH_CHECK(y.device() == cache.device() && y.device() == out.device(),
              "dqg_c1_normal_direct_cached_range_out tensors must be on the "
              "same CUDA device.");
  TORCH_CHECK(y.scalar_type() == at::ScalarType::Double &&
                  cache.scalar_type() == at::ScalarType::Double &&
                  out.scalar_type() == at::ScalarType::Double,
              "dqg_c1_normal_direct_cached_range_out currently expects float64 "
              "tensors.");
  TORCH_CHECK(
      y.is_contiguous() && cache.is_contiguous() && out.is_contiguous(),
      "dqg_c1_normal_direct_cached_range_out expects contiguous tensors.");
  TORCH_CHECK(
      y.numel() == n_dual,
      "dqg_c1_normal_direct_cached_range_out input has wrong dual size.");
  TORCH_CHECK(
      row_start >= 0 && row_stop >= row_start && row_stop <= n_dual,
      "dqg_c1_normal_direct_cached_range_out received an invalid row range.");
  TORCH_CHECK(out.numel() == row_stop - row_start,
              "dqg_c1_normal_direct_cached_range_out output size does not "
              "match row range.");
  TORCH_CHECK(n > 0 && n <= 4096,
              "dqg_c1_normal_direct_cached_range_out received invalid n.");
  TORCH_CHECK(fabs(na - nb) >= 1.0e-12,
              "dqg_c1_normal_direct_cached_range_out currently supports the "
              "open-shell DQG/C1 layout only.");
  const int64_t cache_size =
      make_normal_cache_offsets(static_cast<int>(n)).size;
  TORCH_CHECK(cache.numel() == cache_size,
              "dqg_c1_normal_direct_cached_range_out cache has wrong size.");

  constexpr int threads = 256;
  const int64_t row_count = row_stop - row_start;
  if (row_count == 0) {
    return;
  }
  const int64_t blocks = (row_count + threads - 1) / threads;
  dqg_c1_normal_direct_range_kernel<<<static_cast<unsigned int>(blocks),
                                      threads, 0,
                                      at::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<double>(), cache.data_ptr<double>(), out.data_ptr<double>(),
      static_cast<int>(n), na, nb, row_start, row_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dqg_c1_normal_direct_cached_out(
    torch::Tensor y, torch::Tensor cache, torch::Tensor out, int64_t n,
    double na, double nb, int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
    int64_t d200off, int64_t d1aoff, int64_t d1boff, int64_t q1aoff,
    int64_t q1boff, int64_t q2aboff, int64_t q2aaoff, int64_t q2bboff,
    int64_t g2aboff, int64_t g2baoff, int64_t g2aaoff, int64_t n_primal,
    int64_t n_dual) {
  dqg_c1_normal_direct_build_cache_out(y, cache, n, na, nb, n_dual);
  dqg_c1_normal_direct_cached_range_out(
      y, cache, out, 0, n_dual, n, na, nb, d2aboff, d2aaoff, d2bboff, d200off,
      d1aoff, d1boff, q1aoff, q1boff, q2aboff, q2aaoff, q2bboff, g2aboff,
      g2baoff, g2aaoff, n_primal, n_dual);
}

void dqg_c1_normal_direct_range_out(
    torch::Tensor y, torch::Tensor out, int64_t row_start, int64_t row_stop,
    int64_t n, double na, double nb, int64_t d2aboff, int64_t d2aaoff,
    int64_t d2bboff, int64_t d200off, int64_t d1aoff, int64_t d1boff,
    int64_t q1aoff, int64_t q1boff, int64_t q2aboff, int64_t q2aaoff,
    int64_t q2bboff, int64_t g2aboff, int64_t g2baoff, int64_t g2aaoff,
    int64_t n_primal, int64_t n_dual) {
  (void)d2aboff;
  (void)d2aaoff;
  (void)d2bboff;
  (void)d200off;
  (void)d1aoff;
  (void)d1boff;
  (void)q1aoff;
  (void)q1boff;
  (void)q2aboff;
  (void)q2aaoff;
  (void)q2bboff;
  (void)g2aboff;
  (void)g2baoff;
  (void)g2aaoff;
  (void)n_primal;

  TORCH_CHECK(y.is_cuda() && out.is_cuda(),
              "dqg_c1_normal_direct_range_out expects CUDA tensors.");
  TORCH_CHECK(y.device() == out.device(),
              "dqg_c1_normal_direct_range_out tensors must be on the same CUDA "
              "device.");
  TORCH_CHECK(
      y.scalar_type() == at::ScalarType::Double &&
          out.scalar_type() == at::ScalarType::Double,
      "dqg_c1_normal_direct_range_out currently expects float64 tensors.");
  TORCH_CHECK(y.is_contiguous() && out.is_contiguous(),
              "dqg_c1_normal_direct_range_out expects contiguous tensors.");
  TORCH_CHECK(y.numel() == n_dual,
              "dqg_c1_normal_direct_range_out input has wrong dual size.");
  TORCH_CHECK(row_start >= 0 && row_stop >= row_start && row_stop <= n_dual,
              "dqg_c1_normal_direct_range_out received an invalid row range.");
  TORCH_CHECK(
      out.numel() == row_stop - row_start,
      "dqg_c1_normal_direct_range_out output size does not match row range.");
  TORCH_CHECK(n > 0 && n <= 4096,
              "dqg_c1_normal_direct_range_out received invalid n.");
  TORCH_CHECK(fabs(na - nb) >= 1.0e-12,
              "dqg_c1_normal_direct_range_out currently supports the "
              "open-shell DQG/C1 layout only.");

  constexpr int threads = 256;
  const int64_t row_count = row_stop - row_start;
  if (row_count == 0) {
    return;
  }
  const int64_t blocks = (row_count + threads - 1) / threads;
  dqg_c1_normal_direct_range_kernel<<<static_cast<unsigned int>(blocks),
                                      threads, 0,
                                      at::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<double>(), nullptr, out.data_ptr<double>(),
      static_cast<int>(n), na, nb, row_start, row_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dqg_c1_normal_direct_out(torch::Tensor y, torch::Tensor out, int64_t n,
                              double na, double nb, int64_t d2aboff,
                              int64_t d2aaoff, int64_t d2bboff, int64_t d200off,
                              int64_t d1aoff, int64_t d1boff, int64_t q1aoff,
                              int64_t q1boff, int64_t q2aboff, int64_t q2aaoff,
                              int64_t q2bboff, int64_t g2aboff, int64_t g2baoff,
                              int64_t g2aaoff, int64_t n_primal,
                              int64_t n_dual) {
  dqg_c1_normal_direct_range_out(y, out, 0, n_dual, n, na, nb, d2aboff, d2aaoff,
                                 d2bboff, d200off, d1aoff, d1boff, q1aoff,
                                 q1boff, q2aboff, q2aaoff, q2bboff, g2aboff,
                                 g2baoff, g2aaoff, n_primal, n_dual);
}

void dqg_c1_normal_out(torch::Tensor y, torch::Tensor scratch,
                       torch::Tensor out, int64_t n, double na, double nb,
                       int64_t d2aboff, int64_t d2aaoff, int64_t d2bboff,
                       int64_t d200off, int64_t d1aoff, int64_t d1boff,
                       int64_t q1aoff, int64_t q1boff, int64_t q2aboff,
                       int64_t q2aaoff, int64_t q2bboff, int64_t g2aboff,
                       int64_t g2baoff, int64_t g2aaoff, int64_t n_primal,
                       int64_t n_dual) {
  TORCH_CHECK(y.is_cuda() && scratch.is_cuda() && out.is_cuda(),
              "dqg_c1_normal_out expects CUDA tensors.");
  TORCH_CHECK(y.device() == scratch.device() && y.device() == out.device(),
              "dqg_c1_normal_out tensors must be on the same CUDA device.");
  TORCH_CHECK(y.scalar_type() == at::ScalarType::Double &&
                  scratch.scalar_type() == at::ScalarType::Double &&
                  out.scalar_type() == at::ScalarType::Double,
              "dqg_c1_normal_out currently expects float64 tensors.");
  TORCH_CHECK(y.is_contiguous() && scratch.is_contiguous() &&
                  out.is_contiguous(),
              "dqg_c1_normal_out expects contiguous tensors.");
  TORCH_CHECK(y.numel() == n_dual,
              "dqg_c1_normal_out input has wrong dual size.");
  TORCH_CHECK(scratch.numel() == n_primal,
              "dqg_c1_normal_out scratch has wrong primal size.");
  TORCH_CHECK(out.numel() == n_dual,
              "dqg_c1_normal_out output has wrong dual size.");

  dqg_c1_atu_out(y, scratch, n, na, nb, d2aboff, d2aaoff, d2bboff, d200off,
                 d1aoff, d1boff, q1aoff, q1boff, q2aboff, q2aaoff, q2bboff,
                 g2aboff, g2baoff, g2aaoff, n_primal);
  dqg_c1_au_out(scratch, out, n, na, nb, d2aboff, d2aaoff, d2bboff, d200off,
                d1aoff, d1boff, q1aoff, q1boff, q2aboff, q2aaoff, q2bboff,
                g2aboff, g2baoff, g2aaoff, n_dual);

  constexpr int threads = 256;
  const int64_t q1_rows = 2 * n * n;
  int64_t blocks = (q1_rows + threads - 1) / threads;
  dqg_c1_normal_q1_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<double>(), out.data_ptr<double>(), static_cast<int>(n), na,
      nb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // The q2ab/g2ab/g2ba direct-row kernels are kept here for isolated
  // validation. Since this entry point still computes the full scratch-backed
  // ATu/Au normal product above, launching the large q2/g2 overwrites adds work
  // without removing the corresponding scratch-backed work.
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
  return "dqg_c1_au_atu_20260730_cg_normal_sharded_v1_aurange";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dqg_c1_au", &dqg_c1_au, "DQG/C1 matrix-free Au CUDA kernel");
  m.def("dqg_c1_au_out", &dqg_c1_au_out,
        "DQG/C1 matrix-free Au CUDA kernel with caller-provided output");
  m.def("dqg_c1_au_range_out", &dqg_c1_au_range_out,
        "DQG/C1 matrix-free Au over a dual-row range, reading primal at the "
        "given (optionally remapped/compact) block offsets");
  m.def("dqg_c1_atu", &dqg_c1_atu, "DQG/C1 matrix-free ATu CUDA kernel");
  m.def("dqg_c1_atu_out", &dqg_c1_atu_out,
        "DQG/C1 matrix-free ATu CUDA kernel with caller-provided output");
  m.def("dqg_c1_atu_range_add_out", &dqg_c1_atu_range_add_out,
        "DQG/C1 matrix-free ATu CUDA kernel over a global dual-row range, "
        "accumulating into output");
  m.def("dqg_c1_normal_out", &dqg_c1_normal_out,
        "DQG/C1 matrix-free normal operator Au(ATu(y)) with caller-provided "
        "scratch/output");
  m.def(
      "dqg_c1_normal_direct_out", &dqg_c1_normal_direct_out,
      "DQG/C1 matrix-free direct normal operator with caller-provided output");
  m.def(
      "dqg_c1_normal_direct_range_out", &dqg_c1_normal_direct_range_out,
      "DQG/C1 matrix-free direct normal operator over a global dual-row range");
  m.def("dqg_c1_normal_direct_cache_size", &dqg_c1_normal_direct_cache_size,
        "Compact cache size for the DQG/C1 direct normal operator");
  m.def("dqg_c1_normal_direct_build_cache_out",
        &dqg_c1_normal_direct_build_cache_out,
        "Build compact d1/d2 cache for the DQG/C1 direct normal operator");
  m.def("dqg_c1_normal_direct_cached_out", &dqg_c1_normal_direct_cached_out,
        "DQG/C1 direct normal operator using a compact d1/d2 cache");
  m.def("dqg_c1_normal_direct_cached_range_out",
        &dqg_c1_normal_direct_cached_range_out,
        "DQG/C1 direct normal operator over a row range using a compact d1/d2 "
        "cache");
  m.def("cg_update_y_r", &cg_update_y_r,
        "Fused CUDA update y += alpha p, r -= alpha Ap, and ||r||^2");
  m.def("cg_update_p", &cg_update_p, "Fused CUDA update p = r + beta p");
  m.def("dqg_c1_au_version", &dqg_c1_au_version,
        "DQG/C1 matrix-free Au/ATu CUDA kernel version");
}
