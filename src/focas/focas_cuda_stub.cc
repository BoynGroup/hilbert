extern "C" int hilbert_focas_df_c1_cuda_transform(int, long long, double *,
                                                   const double *, int, int,
                                                   int) {
  return 1;
}

extern "C" int hilbert_focas_df_c1_cuda_fi_exchange(
    int, int, int, long long, const double *, double *, double *, int, int,
    int) {
  return 1;
}

extern "C" int hilbert_focas_df_sym_cuda_fi_exchange(
    int, int, long long, const double *, const int *, double *, int, int,
    int) {
  return 1;
}

extern "C" int hilbert_focas_df_c1_cuda_fa_exchange(
    int, int, int, long long, const double *, const double *, double *,
    double *, int, int, int) {
  return 1;
}

extern "C" int hilbert_focas_df_sym_cuda_fa_exchange(
    int, int, int, long long, const double *, const double *, const int *,
    double *, int, int, int) {
  return 1;
}

extern "C" int hilbert_focas_df_c1_cuda_fi_coulomb(
    int, int, int, long long, const double *, const double *, double *,
    double *, int, int, int) {
  return 1;
}

extern "C" int hilbert_focas_df_c1_cuda_fa_coulomb(
    int, int, int, long long, const double *, const double *, double *,
    double *, int, int, int) {
  return 1;
}

extern "C" int hilbert_focas_df_c1_cuda_q(
    int, int, int, long long, const double *, const double *, double *, int,
    int, int) {
  return 1;
}

extern "C" int hilbert_focas_df_sym_cuda_q(
    int, int, int, long long, const double *, const double *, const int *,
    double *, int, int, int) {
  return 1;
}
