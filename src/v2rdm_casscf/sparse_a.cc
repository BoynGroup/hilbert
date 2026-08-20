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

#include "v2rdm_solver.h"

#include <cmath>
#include <iostream>

namespace hilbert {

namespace {
constexpr int kDqgSymMatrixFreeMagic = -20260623;
constexpr int kDqgSymMatrixFreeVersion = 1;

void append_int_array(std::vector<int> &out, const int *data, int count) {
  for (int i = 0; i < count; ++i) {
    out.push_back(data[i]);
  }
}
} // namespace

bool v2RDMSolver::build_dqg_matrix_free_meta(std::vector<int> &int_meta,
                                             std::vector<double> &double_meta) {
  if (!constrain_q2_ || !constrain_g2_ || constrain_t1_ || constrain_t2_ ||
      constrain_e3_ || constrain_f3_ || constrain_q3_ || constrain_d3_ ||
      constrain_d4_ || constrain_gpc_) {
    return false;
  }

  int_meta.clear();

  double_meta.clear();
  double_meta.push_back(nalpha_ - nrstc_ - nfrzc_);
  double_meta.push_back(nbeta_ - nrstc_ - nfrzc_);

  if (nirrep_ == 1) {
    const int h = 0;
    int_meta.push_back(amo_);
    int_meta.push_back(amopi_[h]);
    int_meta.push_back(gems_ab[h]);
    int_meta.push_back(gems_aa[h]);
    int_meta.push_back(constrain_sz_ ? 1 : 0);
    int_meta.push_back(constrain_spin_ ? 1 : 0);
    int_meta.push_back(constrain_q2_ ? 1 : 0);
    int_meta.push_back(constrain_g2_ ? 1 : 0);
    return true;
  }

  // The first symmetry-aware CUDA operator covers the same custom-kernel
  // layout as the existing C1 path.
  if (!constrain_sz_ || !constrain_spin_) {
    return false;
  }

  int_meta.push_back(kDqgSymMatrixFreeMagic);
  int_meta.push_back(kDqgSymMatrixFreeVersion);
  int_meta.push_back(nirrep_);
  int_meta.push_back(amo_);
  int_meta.push_back(constrain_sz_ ? 1 : 0);
  int_meta.push_back(constrain_spin_ ? 1 : 0);
  int_meta.push_back(constrain_q2_ ? 1 : 0);
  int_meta.push_back(constrain_g2_ ? 1 : 0);
  int_meta.push_back(std::abs(double_meta[0] - double_meta[1]) < 1.0e-12 ? 1
                                                                          : 0);

  append_int_array(int_meta, amopi_, nirrep_);
  append_int_array(int_meta, pitzer_offset, nirrep_);
  append_int_array(int_meta, symmetry, amo_);
  append_int_array(int_meta, table, 64);
  append_int_array(int_meta, gems_ab, nirrep_);
  append_int_array(int_meta, gems_aa, nirrep_);

  int *offset_blocks[] = {d2aboff, d2aaoff, d2bboff, d200off, d1aoff,
                          d1boff, q1aoff,  q1boff,  q2aboff, q2aaoff,
                          q2bboff, g2aboff, g2baoff, g2aaoff};
  for (int block = 0; block < 14; ++block) {
    append_int_array(int_meta, offset_blocks[block], nirrep_);
  }

  std::vector<int> ab_pair_offsets(nirrep_ + 1, 0);
  std::vector<int> aa_pair_offsets(nirrep_ + 1, 0);
  for (int h = 0; h < nirrep_; ++h) {
    ab_pair_offsets[h + 1] = ab_pair_offsets[h] + gems_ab[h];
    aa_pair_offsets[h + 1] = aa_pair_offsets[h] + gems_aa[h];
  }
  append_int_array(int_meta, ab_pair_offsets.data(), nirrep_ + 1);
  append_int_array(int_meta, aa_pair_offsets.data(), nirrep_ + 1);

  for (int h = 0; h < nirrep_; ++h) {
    for (int ij = 0; ij < gems_ab[h]; ++ij) {
      int_meta.push_back(bas_ab_sym[h][ij][0]);
      int_meta.push_back(bas_ab_sym[h][ij][1]);
    }
  }
  for (int h = 0; h < nirrep_; ++h) {
    for (int ij = 0; ij < gems_aa[h]; ++ij) {
      int_meta.push_back(bas_aa_sym[h][ij][0]);
      int_meta.push_back(bas_aa_sym[h][ij][1]);
    }
  }

  for (int h = 0; h < nirrep_; ++h) {
    for (int i = 0; i < amo_; ++i) {
      for (int j = 0; j < amo_; ++j) {
        int_meta.push_back(ibas_ab_sym[h][i][j]);
      }
    }
  }
  for (int h = 0; h < nirrep_; ++h) {
    for (int i = 0; i < amo_; ++i) {
      for (int j = 0; j < amo_; ++j) {
        int_meta.push_back(ibas_aa_sym[h][i][j]);
      }
    }
  }

  return true;
}

bool v2RDMSolver::build_sparse_A_dqg(std::vector<int> &rows,
                                     std::vector<int> &cols,
                                     std::vector<double> &vals) {

  if (nirrep_ != 1 || !constrain_q2_ || !constrain_g2_ || constrain_t1_ ||
      constrain_t2_ || constrain_e3_ || constrain_f3_ || constrain_q3_ ||
      constrain_d3_ || constrain_d4_ || constrain_gpc_) {
    return false;
  }

  rows.clear();
  cols.clear();
  vals.clear();

  auto emit = [&](long int row, long int col, double val) {
    if (std::abs(val) <= 1e-15) {
      return;
    }
    rows.push_back(static_cast<int>(row));
    cols.push_back(static_cast<int>(col));
    vals.push_back(val);
  };

  const int h = 0;
  const int gab = gems_ab[h];
  const int gaa = gems_aa[h];
  const int amo = amo_;
  const int amopi = amopi_[h];
  long int row = 0;

  // D2 traces.
  if (constrain_sz_) {
    for (int i = 0; i < amo; i++) {
      for (int j = 0; j < amo; j++) {
        int ij = ibas_ab_sym[h][i][j];
        emit(row, d2aboff[h] + ij * gab + ij, 1.0);
      }
    }
    row++;
    for (int i = 0; i < amo; i++) {
      for (int j = 0; j < amo; j++) {
        if (i == j)
          continue;
        int ij = ibas_aa_sym[h][i][j];
        emit(row, d2aaoff[h] + ij * gaa + ij, 1.0);
      }
    }
    row++;
    for (int i = 0; i < amo; i++) {
      for (int j = 0; j < amo; j++) {
        if (i == j)
          continue;
        int ij = ibas_aa_sym[h][i][j];
        emit(row, d2bboff[h] + ij * gaa + ij, 1.0);
      }
    }
    row++;
  } else {
    for (int i = 0; i < amo; i++) {
      for (int j = 0; j < amo; j++) {
        int ij = ibas_ab_sym[h][i][j];
        emit(row, d2aboff[h] + ij * gab + ij, 2.0);
        if (i == j)
          continue;
        ij = ibas_aa_sym[h][i][j];
        emit(row, d2aaoff[h] + ij * gaa + ij, 1.0);
        emit(row, d2bboff[h] + ij * gaa + ij, 1.0);
      }
    }
    row++;
  }

  // Hermiticity rows.
  for (int ij = 0; ij < gaa; ij++) {
    for (int kl = 0; kl < gaa; kl++) {
      emit(row, d2aaoff[h] + ij * gaa + kl, 1.0);
      emit(row, d2aaoff[h] + kl * gaa + ij, -1.0);
      row++;
    }
  }
  for (int ij = 0; ij < gaa; ij++) {
    for (int kl = 0; kl < gaa; kl++) {
      emit(row, d2bboff[h] + ij * gaa + kl, 1.0);
      emit(row, d2bboff[h] + kl * gaa + ij, -1.0);
      row++;
    }
  }
  for (int ij = 0; ij < gab; ij++) {
    for (int kl = 0; kl < gab; kl++) {
      emit(row, d2aboff[h] + ij * gab + kl, 1.0);
      emit(row, d2aboff[h] + kl * gab + ij, -1.0);
      row++;
    }
  }

  // D1/Q1 definitions.
  for (int i = 0; i < amopi; i++) {
    for (int j = 0; j < amopi; j++) {
      long int r = row + i * amopi + j;
      emit(r, d1aoff[h] + j * amopi + i, 1.0);
      emit(r, q1aoff[h] + i * amopi + j, 1.0);
    }
  }
  row += amopi * amopi;
  for (int i = 0; i < amopi; i++) {
    for (int j = 0; j < amopi; j++) {
      long int r = row + i * amopi + j;
      emit(r, d1boff[h] + j * amopi + i, 1.0);
      emit(r, q1boff[h] + i * amopi + j, 1.0);
    }
  }
  row += amopi * amopi;

  const double na = nalpha_ - nrstc_ - nfrzc_;
  const double nb = nbeta_ - nrstc_ - nfrzc_;
  const double n = na + nb;

  if (!constrain_sz_) {
    for (int i = 0; i < amopi; i++) {
      for (int j = 0; j < amopi; j++) {
        long int r = row + i * amopi + j;
        emit(r, d1aoff[h] + i * amopi + j, n - 1.0);
        for (int k = 0; k < amo; k++) {
          int ik = ibas_ab_sym[h][i][k];
          int jk = ibas_ab_sym[h][j][k];
          emit(r, d2aboff[h] + ik * gab + jk, -1.0);
        }
        for (int k = 0; k < amo; k++) {
          if (i == k || j == k)
            continue;
          int ik = ibas_aa_sym[h][i][k];
          int jk = ibas_aa_sym[h][j][k];
          int sik = (i < k) ? 1 : -1;
          int sjk = (j < k) ? 1 : -1;
          emit(r, d2aaoff[h] + ik * gaa + jk, -sik * sjk);
        }
      }
    }
    row += amopi * amopi;

    for (int i = 0; i < amopi; i++) {
      for (int j = 0; j < amopi; j++) {
        long int r = row + i * amopi + j;
        emit(r, d1boff[h] + i * amopi + j, n - 1.0);
        for (int k = 0; k < amo; k++) {
          int ik = ibas_ab_sym[h][k][i];
          int jk = ibas_ab_sym[h][k][j];
          emit(r, d2aboff[h] + ik * gab + jk, -1.0);
        }
        for (int k = 0; k < amo; k++) {
          if (i == k || j == k)
            continue;
          int ik = ibas_aa_sym[h][i][k];
          int jk = ibas_aa_sym[h][j][k];
          int sik = (i < k) ? 1 : -1;
          int sjk = (j < k) ? 1 : -1;
          emit(r, d2bboff[h] + ik * gaa + jk, -sik * sjk);
        }
      }
    }
    row += amopi * amopi;
  } else {
    for (int i = 0; i < amopi; i++) {
      for (int j = 0; j < amopi; j++) {
        long int r = row + i * amopi + j;
        emit(r, d1aoff[h] + i * amopi + j, nb);
        for (int k = 0; k < amo; k++) {
          int ik = ibas_ab_sym[h][i][k];
          int jk = ibas_ab_sym[h][j][k];
          emit(r, d2aboff[h] + ik * gab + jk, -1.0);
        }
      }
    }
    row += amopi * amopi;
    for (int i = 0; i < amopi; i++) {
      for (int j = 0; j < amopi; j++) {
        long int r = row + i * amopi + j;
        emit(r, d1boff[h] + i * amopi + j, na);
        for (int k = 0; k < amo; k++) {
          int ik = ibas_ab_sym[h][k][i];
          int jk = ibas_ab_sym[h][k][j];
          emit(r, d2aboff[h] + ik * gab + jk, -1.0);
        }
      }
    }
    row += amopi * amopi;
    for (int i = 0; i < amopi; i++) {
      for (int j = 0; j < amopi; j++) {
        long int r = row + i * amopi + j;
        emit(r, d1aoff[h] + i * amopi + j, na - 1.0);
        for (int k = 0; k < amo; k++) {
          if (i == k || j == k)
            continue;
          int ik = ibas_aa_sym[h][i][k];
          int jk = ibas_aa_sym[h][j][k];
          int sik = (i < k) ? 1 : -1;
          int sjk = (j < k) ? 1 : -1;
          emit(r, d2aaoff[h] + ik * gaa + jk, -sik * sjk);
        }
      }
    }
    row += amopi * amopi;
    for (int i = 0; i < amopi; i++) {
      for (int j = 0; j < amopi; j++) {
        long int r = row + i * amopi + j;
        emit(r, d1boff[h] + i * amopi + j, nb - 1.0);
        for (int k = 0; k < amo; k++) {
          if (i == k || j == k)
            continue;
          int ik = ibas_aa_sym[h][i][k];
          int jk = ibas_aa_sym[h][j][k];
          int sik = (i < k) ? 1 : -1;
          int sjk = (j < k) ? 1 : -1;
          emit(r, d2bboff[h] + ik * gaa + jk, -sik * sjk);
        }
      }
    }
    row += amopi * amopi;
  }

  if (constrain_spin_) {
    for (int i = 0; i < amo; i++) {
      for (int j = 0; j < amo; j++) {
        int ij = ibas_ab_sym[h][i][j];
        int ji = ibas_ab_sym[h][j][i];
        emit(row, d2aboff[h] + ij * gab + ji, 1.0);
      }
    }
    row++;

    if (nalpha_ == nbeta_) {
      for (int p = 0; p < amopi * amopi; p++) {
        emit(row + p, d1aoff[h] + p, 1.0);
        emit(row + p, d1boff[h] + p, -1.0);
      }
      row += amopi * amopi;

      for (int p = 0; p < gaa * gaa; p++) {
        emit(row + p, d2aaoff[h] + p, 1.0);
        emit(row + p, d2bboff[h] + p, -1.0);
      }
      row += gaa * gaa;

      for (int ij = 0; ij < gaa; ij++) {
        int i = bas_aa_sym[h][ij][0];
        int j = bas_aa_sym[h][ij][1];
        int ijb = ibas_ab_sym[h][i][j];
        int jib = ibas_ab_sym[h][j][i];
        for (int kl = 0; kl < gaa; kl++) {
          int k = bas_aa_sym[h][kl][0];
          int l = bas_aa_sym[h][kl][1];
          int klb = ibas_ab_sym[h][k][l];
          int lkb = ibas_ab_sym[h][l][k];
          long int r = row + ij * gaa + kl;
          emit(r, d2aaoff[h] + ij * gaa + kl, 1.0);
          emit(r, d2aboff[h] + ijb * gab + klb, -0.5);
          emit(r, d2aboff[h] + jib * gab + klb, 0.5);
          emit(r, d2aboff[h] + ijb * gab + lkb, 0.5);
          emit(r, d2aboff[h] + jib * gab + lkb, -0.5);
        }
      }
      row += gaa * gaa;

      for (int ij = 0; ij < gaa; ij++) {
        int i = bas_aa_sym[h][ij][0];
        int j = bas_aa_sym[h][ij][1];
        int ijb = ibas_ab_sym[h][i][j];
        int jib = ibas_ab_sym[h][j][i];
        for (int kl = 0; kl < gaa; kl++) {
          int k = bas_aa_sym[h][kl][0];
          int l = bas_aa_sym[h][kl][1];
          int klb = ibas_ab_sym[h][k][l];
          int lkb = ibas_ab_sym[h][l][k];
          long int r = row + ij * gaa + kl;
          emit(r, d2bboff[h] + ij * gaa + kl, 1.0);
          emit(r, d2aboff[h] + ijb * gab + klb, -0.5);
          emit(r, d2aboff[h] + jib * gab + klb, 0.5);
          emit(r, d2aboff[h] + ijb * gab + lkb, 0.5);
          emit(r, d2aboff[h] + jib * gab + lkb, -0.5);
        }
      }
      row += gaa * gaa;

      for (int ij = 0; ij < gab; ij++) {
        int i = bas_ab_sym[h][ij][0];
        int j = bas_ab_sym[h][ij][1];
        int ji = ibas_ab_sym[h][j][i];
        double dij = (i == j) ? std::sqrt(2.0) : 1.0;
        for (int kl = 0; kl < gab; kl++) {
          int k = bas_ab_sym[h][kl][0];
          int l = bas_ab_sym[h][kl][1];
          int lk = ibas_ab_sym[h][l][k];
          double dkl = (k == l) ? std::sqrt(2.0) : 1.0;
          double v = -0.5 / (dij * dkl);
          long int r = row + ij * gab + kl;
          emit(r, d200off[h] + ij * gab + kl, 1.0);
          emit(r, d2aboff[h] + ij * gab + kl, v);
          emit(r, d2aboff[h] + ji * gab + kl, v);
          emit(r, d2aboff[h] + ij * gab + lk, v);
          emit(r, d2aboff[h] + ji * gab + lk, v);
        }
      }
      row += gab * gab;

      for (int ij = 0; ij < gab; ij++) {
        int i = bas_ab_sym[h][ij][0];
        int j = bas_ab_sym[h][ij][1];
        int ji = ibas_ab_sym[h][j][i];
        for (int kl = 0; kl < gab; kl++) {
          int k = bas_ab_sym[h][kl][0];
          int l = bas_ab_sym[h][kl][1];
          int lk = ibas_ab_sym[h][l][k];
          long int r = row + ij * gab + kl;
          emit(r, d2aboff[h] + ij * gab + kl, 1.0);
          emit(r, d2aboff[h] + ji * gab + lk, -1.0);
        }
      }
      row += gab * gab;
    } else {
      for (int ij = 0; ij < gab; ij++) {
        int i = bas_ab_sym[h][ij][0];
        int j = bas_ab_sym[h][ij][1];
        int ji = ibas_ab_sym[h][j][i];
        double dij = (i == j) ? std::sqrt(2.0) : 1.0;
        for (int kl = 0; kl < gab; kl++) {
          int k = bas_ab_sym[h][kl][0];
          int l = bas_ab_sym[h][kl][1];
          int lk = ibas_ab_sym[h][l][k];
          double dkl = (k == l) ? std::sqrt(2.0) : 1.0;
          double v = -0.5 / (dij * dkl);
          long int r = row + ij * 2 * gab + kl;
          emit(r, d200off[h] + ij * 2 * gab + kl, 1.0);
          emit(r, d2aboff[h] + ij * gab + kl, v);
          emit(r, d2aboff[h] + ji * gab + kl, v);
          emit(r, d2aboff[h] + ij * gab + lk, v);
          emit(r, d2aboff[h] + ji * gab + lk, v);
        }
      }

      for (int ij = 0; ij < gab; ij++) {
        int i = bas_ab_sym[h][ij][0];
        int j = bas_ab_sym[h][ij][1];
        int ji = ibas_ab_sym[h][j][i];
        double dij = (i == j) ? std::sqrt(2.0) : 1.0;
        for (int kl = 0; kl < gab; kl++) {
          int k = bas_ab_sym[h][kl][0];
          int l = bas_ab_sym[h][kl][1];
          int lk = ibas_ab_sym[h][l][k];
          long int r = row + ij * 2 * gab + (kl + gab);
          emit(r, d200off[h] + ij * 2 * gab + (kl + gab), 1.0);
          emit(r, d2aboff[h] + ij * gab + kl, -0.5 / dij);
          emit(r, d2aboff[h] + ij * gab + lk, 0.5 / dij);
          emit(r, d2aboff[h] + ji * gab + kl, -0.5 / dij);
          emit(r, d2aboff[h] + ji * gab + lk, 0.5 / dij);
        }
      }

      for (int ij = 0; ij < gab; ij++) {
        int i = bas_ab_sym[h][ij][0];
        int j = bas_ab_sym[h][ij][1];
        int ji = ibas_ab_sym[h][j][i];
        for (int kl = 0; kl < gab; kl++) {
          int k = bas_ab_sym[h][kl][0];
          int l = bas_ab_sym[h][kl][1];
          int lk = ibas_ab_sym[h][l][k];
          double dkl = (k == l) ? std::sqrt(2.0) : 1.0;
          long int r = row + (ij + gab) * 2 * gab + kl;
          emit(r, d200off[h] + (ij + gab) * 2 * gab + kl, 1.0);
          emit(r, d2aboff[h] + ij * gab + kl, -0.5 / dkl);
          emit(r, d2aboff[h] + ij * gab + lk, -0.5 / dkl);
          emit(r, d2aboff[h] + ji * gab + kl, 0.5 / dkl);
          emit(r, d2aboff[h] + ji * gab + lk, 0.5 / dkl);
        }
      }

      for (int ij = 0; ij < gab; ij++) {
        int i = bas_ab_sym[h][ij][0];
        int j = bas_ab_sym[h][ij][1];
        int ji = ibas_ab_sym[h][j][i];
        for (int kl = 0; kl < gab; kl++) {
          int k = bas_ab_sym[h][kl][0];
          int l = bas_ab_sym[h][kl][1];
          int lk = ibas_ab_sym[h][l][k];
          long int r = row + (ij + gab) * 2 * gab + (kl + gab);
          emit(r, d200off[h] + (ij + gab) * 2 * gab + (kl + gab), 1.0);
          emit(r, d2aboff[h] + ij * gab + kl, -0.5);
          emit(r, d2aboff[h] + ji * gab + kl, 0.5);
          emit(r, d2aboff[h] + ij * gab + lk, 0.5);
          emit(r, d2aboff[h] + ji * gab + lk, -0.5);
        }
      }
      row += 4 * gab * gab;
    }

    if (constrain_g2_) {
      for (int kl = 0; kl < gab; kl++) {
        long int r = row + kl;
        for (int i = 0; i < amo; i++) {
          int ii = ibas_ab_sym[h][i][i];
          emit(r, g2baoff[h] + kl * gab + ii, 1.0);
        }
      }
      row += gab;
      for (int kl = 0; kl < gab; kl++) {
        long int r = row + kl;
        for (int i = 0; i < amo; i++) {
          int ii = ibas_ab_sym[h][i][i];
          emit(r, g2baoff[h] + ii * gab + kl, 1.0);
        }
      }
      row += gab;
    }
  }

  if (constrain_q2_) {
    for (int ij = 0; ij < gab; ij++) {
      int i = bas_ab_sym[h][ij][0];
      int j = bas_ab_sym[h][ij][1];
      for (int kl = 0; kl < gab; kl++) {
        int k = bas_ab_sym[h][kl][0];
        int l = bas_ab_sym[h][kl][1];
        long int r = row + ij * gab + kl;
        emit(r, d2aboff[h] + ij * gab + kl, 1.0);
        emit(r, q2aboff[h] + ij * gab + kl, -1.0);
        if (j == l)
          emit(r, d1aoff[h] + k * amopi + i, -1.0);
        if (i == k)
          emit(r, d1boff[h] + l * amopi + j, -1.0);
      }
    }
    row += gab * gab;

    for (int ij = 0; ij < gaa; ij++) {
      int i = bas_aa_sym[h][ij][0];
      int j = bas_aa_sym[h][ij][1];
      for (int kl = 0; kl < gaa; kl++) {
        int k = bas_aa_sym[h][kl][0];
        int l = bas_aa_sym[h][kl][1];
        long int r = row + ij * gaa + kl;
        emit(r, d2aaoff[h] + ij * gaa + kl, 1.0);
        emit(r, q2aaoff[h] + ij * gaa + kl, -1.0);
        if (j == l)
          emit(r, d1aoff[h] + k * amopi + i, -1.0);
        if (j == k)
          emit(r, d1aoff[h] + l * amopi + i, 1.0);
        if (i == l)
          emit(r, d1aoff[h] + k * amopi + j, 1.0);
        if (i == k)
          emit(r, d1aoff[h] + l * amopi + j, -1.0);
      }
    }
    row += gaa * gaa;

    for (int ij = 0; ij < gaa; ij++) {
      int i = bas_aa_sym[h][ij][0];
      int j = bas_aa_sym[h][ij][1];
      for (int kl = 0; kl < gaa; kl++) {
        int k = bas_aa_sym[h][kl][0];
        int l = bas_aa_sym[h][kl][1];
        long int r = row + ij * gaa + kl;
        emit(r, d2bboff[h] + ij * gaa + kl, 1.0);
        emit(r, q2bboff[h] + ij * gaa + kl, -1.0);
        if (j == l)
          emit(r, d1boff[h] + k * amopi + i, -1.0);
        if (j == k)
          emit(r, d1boff[h] + l * amopi + i, 1.0);
        if (i == l)
          emit(r, d1boff[h] + k * amopi + j, 1.0);
        if (i == k)
          emit(r, d1boff[h] + l * amopi + j, -1.0);
      }
    }
    row += gaa * gaa;
  }

  if (constrain_g2_) {
    for (int ij = 0; ij < gab; ij++) {
      int i = bas_ab_sym[h][ij][0];
      int j = bas_ab_sym[h][ij][1];
      for (int kl = 0; kl < gab; kl++) {
        int k = bas_ab_sym[h][kl][0];
        int l = bas_ab_sym[h][kl][1];
        long int r = row + ij * gab + kl;
        emit(r, g2aboff[h] + ij * gab + kl, -1.0);
        if (j == l)
          emit(r, d1aoff[h] + i * amopi + k, 1.0);
        int ild = ibas_ab_sym[h][i][l];
        int kjd = ibas_ab_sym[h][k][j];
        emit(r, d2aboff[h] + ild * gab + kjd, -1.0);
      }
    }
    row += gab * gab;

    for (int ij = 0; ij < gab; ij++) {
      int i = bas_ab_sym[h][ij][0];
      int j = bas_ab_sym[h][ij][1];
      for (int kl = 0; kl < gab; kl++) {
        int k = bas_ab_sym[h][kl][0];
        int l = bas_ab_sym[h][kl][1];
        long int r = row + ij * gab + kl;
        emit(r, g2baoff[h] + ij * gab + kl, -1.0);
        if (j == l)
          emit(r, d1boff[h] + i * amopi + k, 1.0);
        int lid = ibas_ab_sym[h][l][i];
        int jkd = ibas_ab_sym[h][j][k];
        emit(r, d2aboff[h] + lid * gab + jkd, -1.0);
      }
    }
    row += gab * gab;

    for (int ij = 0; ij < gab; ij++) {
      int i = bas_ab_sym[h][ij][0];
      int j = bas_ab_sym[h][ij][1];
      for (int kl = 0; kl < gab; kl++) {
        int k = bas_ab_sym[h][kl][0];
        int l = bas_ab_sym[h][kl][1];
        long int r = row + ij * 2 * gab + kl;
        emit(r, g2aaoff[h] + ij * 2 * gab + kl, -1.0);
        if (j == l)
          emit(r, d1aoff[h] + i * amopi + k, 1.0);
        if (i != l && k != j) {
          int sil = (i < l) ? 1 : -1;
          int skj = (k < j) ? 1 : -1;
          int ild = ibas_aa_sym[h][i][l];
          int kjd = ibas_aa_sym[h][k][j];
          emit(r, d2aaoff[h] + ild * gaa + kjd, -sil * skj);
        }
      }
    }

    for (int ij = 0; ij < gab; ij++) {
      int i = bas_ab_sym[h][ij][0];
      int j = bas_ab_sym[h][ij][1];
      for (int kl = 0; kl < gab; kl++) {
        int k = bas_ab_sym[h][kl][0];
        int l = bas_ab_sym[h][kl][1];
        long int r = row + (gab + ij) * 2 * gab + (gab + kl);
        emit(r, g2aaoff[h] + (gab + ij) * 2 * gab + (gab + kl), -1.0);
        if (j == l)
          emit(r, d1boff[h] + i * amopi + k, 1.0);
        if (i != l && k != j) {
          int sil = (i < l) ? 1 : -1;
          int skj = (k < j) ? 1 : -1;
          int ild = ibas_aa_sym[h][i][l];
          int kjd = ibas_aa_sym[h][k][j];
          emit(r, d2bboff[h] + ild * gaa + kjd, -sil * skj);
        }
      }
    }

    for (int ij = 0; ij < gab; ij++) {
      int i = bas_ab_sym[h][ij][0];
      int j = bas_ab_sym[h][ij][1];
      for (int kl = 0; kl < gab; kl++) {
        int k = bas_ab_sym[h][kl][0];
        int l = bas_ab_sym[h][kl][1];
        long int r = row + ij * 2 * gab + (gab + kl);
        emit(r, g2aaoff[h] + ij * 2 * gab + (gab + kl), -1.0);
        int ild = ibas_ab_sym[h][i][l];
        int jkd = ibas_ab_sym[h][j][k];
        emit(r, d2aboff[h] + ild * gab + jkd, 1.0);
      }
    }

    for (int ij = 0; ij < gab; ij++) {
      int i = bas_ab_sym[h][ij][0];
      int j = bas_ab_sym[h][ij][1];
      for (int kl = 0; kl < gab; kl++) {
        int k = bas_ab_sym[h][kl][0];
        int l = bas_ab_sym[h][kl][1];
        long int r = row + (gab + ij) * 2 * gab + kl;
        emit(r, g2aaoff[h] + (gab + ij) * 2 * gab + kl, -1.0);
        int lid = ibas_ab_sym[h][l][i];
        int kjd = ibas_ab_sym[h][k][j];
        emit(r, d2aboff[h] + lid * gab + kjd, 1.0);
      }
    }
    row += 4 * gab * gab;
  }

  if (row != n_dual_) {
    std::cout << "  ==> [GPU-ADMM] Direct DQG sparse A builder row mismatch: "
              << row << " emitted vs " << n_dual_ << " expected."
              << std::endl;
    rows.clear();
    cols.clear();
    vals.clear();
    return false;
  }

  return true;
}

} // namespace hilbert
