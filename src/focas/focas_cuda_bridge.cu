#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kResidentSessionFatal = 290;
constexpr std::size_t kPinnedResidentUploadThreshold =
    static_cast<std::size_t>(8) << 30;

// Streamed (non-resident) Q tiles above this size go through the pooled pinned
// double-buffered staging path.  Below it the driver-managed pageable copy is
// competitive and avoids an extra host memcpy.
constexpr std::size_t kPinnedStreamTileThreshold =
    static_cast<std::size_t>(32) * 1024 * 1024;

// Absolute floor on the device memory held back from DF residency.
constexpr std::size_t kMinWorkingReserve = static_cast<std::size_t>(4) << 30;

// Q rows assumed in flight when sizing the working reserve.  This is the chunk
// depth the transform/gradient kernels are expected to use; it only sets how
// much memory is held back, never how much is actually processed at once.
constexpr double kReserveBlockQ = 32.0;

// ---------------------------------------------------------------------------
// Diagnostics.  This path is meant to stay on the GPU, so every host fallback
// and every residency decision that costs PCIe bandwidth is announced.  Notices
// are de-duplicated because the gradient operators run hundreds of times per
// optimization; without that a real warning would be buried.
// ---------------------------------------------------------------------------
std::mutex focas_notice_mutex;
std::vector<std::string> focas_notice_seen;

void focas_notice(const char *fmt, ...) {
  char message[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(message, sizeof(message), fmt, args);
  va_end(args);
  {
    std::lock_guard<std::mutex> lock(focas_notice_mutex);
    for (const auto &seen : focas_notice_seen) {
      if (seen == message) {
        return;
      }
    }
    focas_notice_seen.emplace_back(message);
  }
  std::fprintf(stderr, "  ==> [FOCAS-GPU] %s\n", message);
  std::fflush(stderr);
}

double to_gib(std::size_t bytes) {
  return static_cast<double>(bytes) / static_cast<double>(1ull << 30);
}

// A FOCAS orbital-optimization session spans the initial gradient, all trial
// rotations, and the final gradient.  When a device has enough free memory we
// keep its Q slice of the packed DF tensor resident for that whole interval.
// Resident slices become authoritative after the first successful transform.
// They are committed to the host once, when the optimization session ends.
struct FocasSessionDevice {
  int device = -1;
  long long q_begin = 0;
  long long q_end = 0;
  // Residency is partial: rows [q_begin, q_res_end) live on the device for the
  // whole session and are authoritative once a transform has run; rows
  // [q_res_end, q_end) stay on the host and are streamed per tile.  When
  // q_res_end == q_begin nothing is resident, when it equals q_end the slice is
  // fully resident (the historical all-or-nothing behaviour).
  long long q_res_end = 0;
  double *d_int2 = nullptr;
  std::size_t int2_bytes = 0;
  unsigned char *workspace = nullptr;
  std::size_t workspace_bytes = 0;

  bool has_resident_rows() const {
    return d_int2 != nullptr && q_res_end > q_begin;
  }
  // A tile is resident only when it lies wholly inside the prefix; callers clamp
  // tiles to the boundary so this never splits one.
  bool tile_resident(long long q, long long count) const {
    return has_resident_rows() && q >= q_begin && q + count <= q_res_end;
  }
};

struct FocasSession {
  bool active = false;
  bool dirty = false;
  int nmo = 0;
  long long nQ = 0;
  long long ngem = 0;
  const double *host_int2 = nullptr;
  std::vector<FocasSessionDevice> devices;
};

FocasSession focas_session;
std::mutex focas_session_mutex;

void clear_focas_session() {
  for (auto &entry : focas_session.devices) {
    if (entry.d_int2 != nullptr) {
      cudaSetDevice(entry.device);
      cudaFree(entry.d_int2);
      entry.d_int2 = nullptr;
    }
    if (entry.workspace != nullptr) {
      cudaSetDevice(entry.device);
      cudaFree(entry.workspace);
      entry.workspace = nullptr;
      entry.workspace_bytes = 0;
    }
  }
  focas_session = FocasSession{};
}

bool ensure_session_workspace(FocasSessionDevice *entry,
                              std::size_t required_bytes) {
  if (entry == nullptr)
    return false;
  if (entry->workspace != nullptr && entry->workspace_bytes >= required_bytes) {
    return true;
  }
  if (entry->workspace != nullptr) {
    cudaFree(entry->workspace);
    entry->workspace = nullptr;
    entry->workspace_bytes = 0;
  }
  if (required_bytes == 0)
    return true;
  if (cudaMalloc(&entry->workspace, required_bytes) != cudaSuccess) {
    entry->workspace = nullptr;
    return false;
  }
  entry->workspace_bytes = required_bytes;
  return true;
}

template <typename Value>
Value *take_workspace(unsigned char *&cursor, std::size_t count) {
  Value *result = reinterpret_cast<Value *>(cursor);
  cursor += count * sizeof(Value);
  return result;
}

FocasSessionDevice *session_device_slice(int device, const double *host_int2,
                                         long long ngem, long long q_begin,
                                         long long q_end) {
  if (!focas_session.active || focas_session.host_int2 != host_int2 ||
      focas_session.ngem != ngem) {
    return nullptr;
  }
  for (auto &entry : focas_session.devices) {
    if (entry.device == device && entry.q_begin == q_begin &&
        entry.q_end == q_end && entry.d_int2 != nullptr) {
      return &entry;
    }
  }
  return nullptr;
}

// True when the slice keeps only part of its Q range on the device, so the
// caller must also provide staging space for the streamed remainder.
bool slice_is_partial(const FocasSessionDevice *resident, long long q_end) {
  return resident != nullptr && resident->q_res_end < q_end;
}

// True only when the slice keeps its whole Q range on the device.  The chunk
// choosers size their per-Q working set from this: a fully resident slice needs
// no staging, while a partially resident one still pays ngem doubles per Q row
// for the streamed tiles and must be sized like the non-resident case.
bool slice_is_fully_resident(const FocasSessionDevice *resident,
                             long long q_end) {
  return resident != nullptr && resident->q_res_end >= q_end;
}

// Scratch a partially resident slice needs to land one streamed tile.
std::size_t streaming_stage_bytes(const FocasSessionDevice *resident,
                                  long long q_end, int q_chunk,
                                  long long ngem) {
  if (!slice_is_partial(resident, q_end))
    return 0;
  return static_cast<std::size_t>(q_chunk) * static_cast<std::size_t>(ngem) *
         sizeof(double);
}

// Device memory held back from DF residency.  A fraction of the card says
// nothing about the calculation being run, so size the reserve from the problem:
// the transform and gradient kernels allocate up to four rectangular nmo*nmo
// blocks per Q row in flight plus the nmo*nmo rotation matrix and accumulators.
// A small fractional margin is kept on top for the co-resident GPU_ADMM solver,
// whose caching allocator can grow its pool between macrocycles.  Because
// residency is now partial, being generous here costs a little streaming rather
// than the whole tensor, so the reserve is allowed to be safe.
std::size_t focas_working_reserve_bytes(int nmo, std::size_t free_bytes) {
  const double nmo2 = static_cast<double>(nmo) * static_cast<double>(nmo);
  const double per_q_bytes = 4.0 * nmo2 * sizeof(double);
  const double fixed_bytes = 2.0 * nmo2 * sizeof(double);
  const double working = fixed_bytes + kReserveBlockQ * per_q_bytes;
  const double co_tenant_margin = 0.10 * static_cast<double>(free_bytes);
  const double reserve =
      std::max({static_cast<double>(kMinWorkingReserve), 1.5 * working,
                co_tenant_margin});
  return static_cast<std::size_t>(reserve);
}

// Shrink a tile so it never straddles the residency boundary.  Every tile is
// then either wholly resident or wholly streamed, which keeps the transform's
// write-back unambiguous: a straddling tile would otherwise be written to the
// host while its leading rows stayed authoritative (and now stale) on device.
int clamp_tile_to_residency(const FocasSessionDevice *resident, long long q,
                            int bq) {
  if (resident == nullptr || !resident->has_resident_rows())
    return bq;
  const long long res_end = resident->q_res_end;
  if (q < res_end && q + bq > res_end) {
    return static_cast<int>(res_end - q);
  }
  return bq;
}

bool session_has_resident_slices(const double *host_int2, long long ngem) {
  if (!focas_session.active || focas_session.host_int2 != host_int2 ||
      focas_session.ngem != ngem) {
    return false;
  }
  return std::any_of(focas_session.devices.begin(), focas_session.devices.end(),
                     [](const FocasSessionDevice &entry) {
                       return entry.has_resident_rows();
                     });
}

bool session_is_fully_resident(const double *host_int2, long long ngem,
                               long long nQ) {
  if (!focas_session.active || focas_session.host_int2 != host_int2 ||
      focas_session.ngem != ngem || focas_session.nQ != nQ) {
    return false;
  }
  long long next_q = 0;
  for (const auto &entry : focas_session.devices) {
    // Fully resident requires the device copy to cover the slice completely;
    // a partially resident slice leaves the host authoritative for its tail.
    if (entry.d_int2 == nullptr || entry.q_begin != next_q ||
        entry.q_end < entry.q_begin || entry.q_res_end != entry.q_end) {
      return false;
    }
    next_q = entry.q_end;
  }
  return next_q == nQ;
}

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

__global__ void build_fi_coulomb_vector_kernel(const double *__restrict__ int2,
                                               double *__restrict__ qvec,
                                               int ndoc, int q_count,
                                               long long ngem) {
  long long q = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  for (; q < q_count; q += stride) {
    const double *row = int2 + q * ngem;
    double value = 0.0;
    for (int i = 0; i < ndoc; ++i) {
      value += 2.0 * row[packed_pair_base(i) + i];
    }
    qvec[q] = value;
  }
}

__global__ void build_fa_coulomb_vector_kernel(const double *__restrict__ int2,
                                               const double *__restrict__ den1,
                                               double *__restrict__ qvec,
                                               int ndoc, int nact, int q_count,
                                               long long ngem) {
  long long q = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  for (; q < q_count; q += stride) {
    const double *row = int2 + q * ngem;
    double value = 0.0;
    for (int t = 0; t < nact; ++t) {
      const int abs_t = ndoc + t;
      for (int u = 0; u < t; ++u) {
        const int abs_u = ndoc + u;
        value += 2.0 * den1[packed_pair_base(t) + u] *
                 row[packed_pair_base(abs_t) + abs_u];
      }
      value +=
          den1[packed_pair_base(t) + t] * row[packed_pair_base(abs_t) + abs_t];
    }
    qvec[q] = value;
  }
}

__global__ void
unpack_packed_symmetric_kernel(const double *__restrict__ packed,
                               double *__restrict__ right, int nmo, int bq,
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

__global__ void unpack_packed_q_major_kernel(const double *__restrict__ packed,
                                             double *__restrict__ dense,
                                             int nmo, int bq, long long ngem) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(bq) * ngem;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  const long long matrix_size = static_cast<long long>(nmo) * nmo;

  for (; idx < total; idx += stride) {
    const int q = static_cast<int>(idx / ngem);
    const long long p = idx - static_cast<long long>(q) * ngem;
    int i = 0;
    int j = 0;
    packed_pair(p, i, j);
    const double value = packed[idx];
    const long long q_offset = static_cast<long long>(q) * matrix_size;
    dense[q_offset + i + static_cast<long long>(j) * nmo] = value;
    dense[q_offset + j + static_cast<long long>(i) * nmo] = value;
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

__global__ void make_rectangular_left_mat_kernel(const double *__restrict__ tmp,
                                                 double *__restrict__ left,
                                                 int nao, int nmo, int bq) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long per_q = static_cast<long long>(nao) * nmo;
  const long long total = static_cast<long long>(bq) * per_q;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  const int nrow_tmp = nao * bq;

  for (; idx < total; idx += stride) {
    const int q = static_cast<int>(idx / per_q);
    const long long rem = idx - static_cast<long long>(q) * per_q;
    const int mu = static_cast<int>(rem % nao);
    const int i = static_cast<int>(rem / nao);
    left[mu + static_cast<long long>(q * nmo + i) * nao] =
        tmp[(q * nao + mu) + static_cast<long long>(i) * nrow_tmp];
  }
}

__global__ void
scatter_packed_symmetric_kernel(const double *__restrict__ result,
                                double *__restrict__ packed, int nmo, int bq,
                                long long ngem) {
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

__global__ void
scatter_low_rank_update_kernel(const double *__restrict__ update,
                               double *__restrict__ packed, int nmo, int bq,
                               long long ngem) {
  long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const long long total = static_cast<long long>(bq) * ngem;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  const long long matrix_size = static_cast<long long>(nmo) * nmo;

  for (; idx < total; idx += stride) {
    const int q = static_cast<int>(idx / ngem);
    const long long p = idx - static_cast<long long>(q) * ngem;
    int i = 0;
    int j = 0;
    packed_pair(p, i, j);
    const long long q_offset = static_cast<long long>(q) * matrix_size;
    packed[idx] += update[q_offset + i + static_cast<long long>(j) * nmo] +
                   update[q_offset + j + static_cast<long long>(i) * nmo];
  }
}

__global__ void build_pair_column_matrix_kernel(const double *__restrict__ int2,
                                                double *__restrict__ x, int nmo,
                                                int inner_begin, int ninner,
                                                long long q_begin, int q_count,
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
    const int *__restrict__ inner_list, int ninner, int q_count, long long ngem,
    int ldx) {
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

__global__ void
build_active_pair_matrix_kernel(const double *__restrict__ int2,
                                double *__restrict__ active, int ndoc, int nact,
                                int q_count, long long ngem, int lda,
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

__global__ void
build_general_active_matrix_kernel(const double *__restrict__ int2,
                                   double *__restrict__ b, int nmo, int ndoc,
                                   int active_u, int q_count, long long ngem,
                                   int ldb, const int *__restrict__ act_df) {
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

int bounded_auto_chunk(double estimate, int automatic_cap,
                       long long item_count) {
  // Clamp in floating point before converting to int.
  const double bounded =
      std::min(static_cast<double>(automatic_cap), std::max(1.0, estimate));
  return static_cast<int>(
      std::min<long long>(static_cast<long long>(bounded), item_count));
}

// A single very large pageable upload makes the CUDA runtime manage an equally
// large internal staging operation.  For resident DF slices of at least 8 GiB,
// keep locked memory bounded and explicitly stage through two 64-MiB buffers.
// Alternating streams overlap the CPU memcpy for one chunk with DMA for the
// other.  Smaller uploads and every D2H transfer use direct cudaMemcpy because
// benchmarks showed that the driver-managed pageable path is faster than the
// extra explicit host copy.  Allocation failure remains harmless because
// copy_h2d falls back to direct cudaMemcpy.
class PinnedTransferStaging {
public:
  PinnedTransferStaging(int device, std::size_t maximum_copy_bytes)
      : device_(device) {
    cudaSetDevice(device_);
    initialize(maximum_copy_bytes);
  }

  ~PinnedTransferStaging() { release(); }

  PinnedTransferStaging(const PinnedTransferStaging &) = delete;
  PinnedTransferStaging &operator=(const PinnedTransferStaging &) = delete;

  bool enabled() const { return enabled_; }
  std::size_t stage_bytes() const { return enabled_ ? stage_bytes_ : 0; }

  cudaError_t copy_h2d(void *device_destination, const void *host_source,
                       std::size_t bytes) {
    if (bytes == 0)
      return cudaSuccess;
    if (!enabled_) {
      return cudaMemcpy(device_destination, host_source, bytes,
                        cudaMemcpyHostToDevice);
    }

    const auto *source = static_cast<const unsigned char *>(host_source);
    auto *destination = static_cast<unsigned char *>(device_destination);
    auto drain = [&](cudaError_t first_status) {
      for (auto &slot : slots_) {
        const cudaError_t status = finish_h2d(slot);
        if (first_status == cudaSuccess && status != cudaSuccess) {
          first_status = status;
        }
      }
      return first_status;
    };
    std::size_t offset = 0;
    std::size_t chunk_index = 0;
    while (offset < bytes) {
      Slot &slot = slots_[chunk_index % 2];
      cudaError_t status = finish_h2d(slot);
      if (status != cudaSuccess)
        return drain(status);
      const std::size_t count = std::min(stage_bytes_, bytes - offset);
      std::memcpy(slot.host, source + offset, count);
      status = cudaMemcpyAsync(destination + offset, slot.host, count,
                               cudaMemcpyHostToDevice, slot.stream);
      if (status != cudaSuccess)
        return drain(status);
      slot.pending = true;
      offset += count;
      ++chunk_index;
    }
    return drain(cudaSuccess);
  }

private:
  struct Slot {
    unsigned char *host = nullptr;
    cudaStream_t stream = nullptr;
    bool pending = false;
  };

  static constexpr std::size_t kStageBytes =
      static_cast<std::size_t>(64) * 1024 * 1024;

  void initialize(std::size_t maximum_copy_bytes) {
    if (maximum_copy_bytes == 0)
      return;
    stage_bytes_ = std::min(kStageBytes, maximum_copy_bytes);
    for (auto &slot : slots_) {
      if (cudaHostAlloc(reinterpret_cast<void **>(&slot.host), stage_bytes_,
                        cudaHostAllocPortable) != cudaSuccess ||
          cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking) !=
              cudaSuccess) {
        release();
        // Clear the sticky allocation error before the pageable fallback.
        cudaGetLastError();
        return;
      }
    }
    enabled_ = true;
  }

  cudaError_t finish_h2d(Slot &slot) {
    if (!slot.pending)
      return cudaSuccess;
    const cudaError_t status = cudaStreamSynchronize(slot.stream);
    slot.pending = false;
    return status;
  }

  void release() {
    cudaSetDevice(device_);
    for (auto &slot : slots_) {
      if (slot.stream != nullptr) {
        cudaStreamSynchronize(slot.stream);
        cudaStreamDestroy(slot.stream);
        slot.stream = nullptr;
      }
      if (slot.host != nullptr) {
        cudaFreeHost(slot.host);
        slot.host = nullptr;
      }
      slot.pending = false;
    }
    enabled_ = false;
    stage_bytes_ = 0;
  }

  Slot slots_[2];
  int device_ = 0;
  bool enabled_ = false;
  std::size_t stage_bytes_ = 0;
};

struct PinnedStagingPoolEntry {
  std::mutex mutex;
  std::unique_ptr<PinnedTransferStaging> staging;
};

std::mutex pinned_staging_pool_mutex;
std::vector<std::unique_ptr<PinnedStagingPoolEntry>> pinned_staging_pool;

PinnedStagingPoolEntry *pinned_staging_pool_entry(int device) {
  std::lock_guard<std::mutex> lock(pinned_staging_pool_mutex);
  if (device < 0)
    return nullptr;
  if (pinned_staging_pool.size() <= static_cast<std::size_t>(device)) {
    pinned_staging_pool.resize(static_cast<std::size_t>(device) + 1);
  }
  auto &entry = pinned_staging_pool[static_cast<std::size_t>(device)];
  if (!entry)
    entry = std::make_unique<PinnedStagingPoolEntry>();
  return entry.get();
}

// A top-level CUDA operation has at most one worker per device.  The lease
// additionally makes that invariant explicit and keeps a future concurrent
// caller from reusing a staging buffer until its outstanding DMA is complete.
class PinnedStagingLease {
public:
  PinnedStagingLease(int device, std::size_t maximum_copy_bytes)
      : entry_(pinned_staging_pool_entry(device)), lock_(entry_->mutex),
        active_(maximum_copy_bytes > 0) {
    const std::size_t needed = std::min(
        static_cast<std::size_t>(64) * 1024 * 1024, maximum_copy_bytes);
    if (!entry_->staging || entry_->staging->stage_bytes() < needed) {
      entry_->staging.reset();
      entry_->staging =
          std::make_unique<PinnedTransferStaging>(device, maximum_copy_bytes);
    }
  }

  PinnedTransferStaging &staging() { return *entry_->staging; }
  bool enabled() const {
    return active_ && entry_->staging && entry_->staging->enabled();
  }
  std::size_t stage_bytes() const {
    return enabled() ? entry_->staging->stage_bytes() : 0;
  }

private:
  PinnedStagingPoolEntry *entry_;
  std::unique_lock<std::mutex> lock_;
  bool active_ = false;
};

// Upload one streamed Q tile.  Tiles large enough to amortize the extra host
// copy go through the pooled pinned double-buffered path, which overlaps the
// host memcpy of one 64 MiB chunk with the DMA of the previous one; smaller
// tiles keep the driver-managed pageable copy.
cudaError_t upload_stream_tile(int device, double *device_block,
                               const double *host_block, std::size_t bytes) {
  if (bytes >= kPinnedStreamTileThreshold) {
    PinnedStagingLease lease(device, bytes);
    if (lease.enabled()) {
      // The staging copies run on cudaStreamNonBlocking streams, which by
      // definition are NOT ordered against the default stream that the
      // contraction kernels and cuBLAS calls use.  copy_h2d waits for its own
      // streams, but nothing waits for the previous tile's kernels, which are
      // still reading this very buffer -- a write-after-read race that silently
      // corrupts results once the per-tile GPU work outlives the next upload.
      // The synchronous cudaMemcpy fallback below does not need this because it
      // orders against the default stream itself.
      const cudaError_t pending = cudaStreamSynchronize(0);
      if (pending != cudaSuccess) {
        return pending;
      }
      return lease.staging().copy_h2d(device_block, host_block, bytes);
    }
  }
  return cudaMemcpy(device_block, host_block, bytes, cudaMemcpyHostToDevice);
}

int copy_resident_slices_to_host_locked(double *host_int2) {
  if (!focas_session.active || host_int2 == nullptr ||
      focas_session.host_int2 != host_int2) {
    return 233;
  }
  for (auto &entry : focas_session.devices) {
    // Only the resident prefix can be newer than the host: streamed tiles are
    // written back as they are produced, so int2_bytes covers exactly the rows
    // [q_begin, q_res_end) that live on the device.
    if (!entry.has_resident_rows())
      continue;
    if (cudaSetDevice(entry.device) != cudaSuccess) {
      return 233;
    }
    if (cudaMemcpy(host_int2 + entry.q_begin * focas_session.ngem, entry.d_int2,
                   entry.int2_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
      return 233;
    }
  }
  focas_session.dirty = false;
  return 0;
}

int recover_session_for_host_fallback(int operation_status,
                                      const double *host_int2) {
  std::lock_guard<std::mutex> lock(focas_session_mutex);
  if (!focas_session.active || focas_session.host_int2 != host_int2) {
    return operation_status;
  }

  const bool was_dirty = focas_session.dirty;
  int sync_status = 0;
  if (was_dirty) {
    sync_status =
        copy_resident_slices_to_host_locked(const_cast<double *>(host_int2));
  }
  clear_focas_session();
  return sync_status == 0 ? operation_status : kResidentSessionFatal;
}

int handle_transform_failure(int operation_status, const double *host_int2,
                             long long ngem, bool tensor_mutated) {
  std::lock_guard<std::mutex> lock(focas_session_mutex);
  const bool matching_session = focas_session.active &&
                                focas_session.host_int2 == host_int2 &&
                                focas_session.ngem == ngem;
  const bool was_dirty = matching_session && focas_session.dirty;

  // Once any worker has written a transformed tile, multiple devices or tiles
  // may be at different points in the update.  A CPU retry would then rotate
  // some tiles twice.  Before the first write it remains safe to discard the
  // session and use the established host fallback.  If an earlier successful
  // transform made the session dirty, commit that still-consistent state first.
  if (!tensor_mutated) {
    int sync_status = 0;
    if (was_dirty) {
      sync_status =
          copy_resident_slices_to_host_locked(const_cast<double *>(host_int2));
    }
    if (matching_session)
      clear_focas_session();
    return sync_status == 0 ? operation_status : kResidentSessionFatal;
  }

  if (matching_session)
    clear_focas_session();
  return kResidentSessionFatal;
}

void mark_resident_session_dirty(const double *host_int2, long long ngem) {
  std::lock_guard<std::mutex> lock(focas_session_mutex);
  if (session_has_resident_slices(host_int2, ngem)) {
    focas_session.dirty = true;
  }
}

int choose_block_q(int device, int nmo, long long q_count, long long ngem,
                   int requested_block_q, bool int2_resident,
                   std::size_t reclaimable_bytes) {
  if (q_count <= 0) {
    return 1;
  }
  if (requested_block_q > 0) {
    return static_cast<int>(std::max<long long>(
        1, std::min<long long>(requested_block_q, q_count)));
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    const int fallback = static_cast<int>(std::min<long long>(32, q_count));
    return fallback;
  }

  constexpr int max_auto_block_q = 1024;
  const double nmo2 = static_cast<double>(nmo) * static_cast<double>(nmo);
  const double u_bytes = nmo2 * sizeof(double);
  const double bytes_per_q =
      (4.0 * nmo2 + (int2_resident ? 0.0 : static_cast<double>(ngem))) *
      sizeof(double);

  const double free_d = static_cast<double>(free_bytes) + reclaimable_bytes;
  const double reserve = std::max(512.0 * 1024.0 * 1024.0, 0.10 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  const double variable_budget = std::max(0.0, budget - u_bytes);

  const int block_q = bounded_auto_chunk(variable_budget / bytes_per_q,
                                         max_auto_block_q, q_count);
  return block_q;
}

int choose_low_rank_block_q(int device, int nmo, int rank, long long q_count,
                            long long ngem, int requested_block_q,
                            bool int2_resident, std::size_t reclaimable_bytes) {
  if (q_count <= 0)
    return 1;
  if (requested_block_q > 0) {
    return static_cast<int>(std::max<long long>(
        1, std::min<long long>(requested_block_q, q_count)));
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    return static_cast<int>(std::min<long long>(32, q_count));
  }

  constexpr int max_auto_block_q = 1024;
  const double n = static_cast<double>(nmo);
  const double r = static_cast<double>(rank);
  const double fixed_bytes = (n * r + r * r) * sizeof(double);
  // Per Q: one dense symmetric matrix, Y and R (n*r), and Z plus a
  // temporary (r*r).  A nonresident operation also owns its packed tile.
  const double bytes_per_q =
      (n * n + 2.0 * n * r + 2.0 * r * r +
       (int2_resident ? 0.0 : static_cast<double>(ngem))) *
      sizeof(double);
  const double free_d = static_cast<double>(free_bytes) + reclaimable_bytes;
  const double reserve = std::max(512.0 * 1024.0 * 1024.0, 0.10 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  const int block_q =
      bounded_auto_chunk(std::max(0.0, budget - fixed_bytes) / bytes_per_q,
                         max_auto_block_q, q_count);
  return block_q;
}

int choose_ao_to_mo_block_q(int device, int nao, int nmo, long long q_count,
                            long long ao_pair, long long mo_pair,
                            int requested_block_q) {
  if (q_count <= 0)
    return 1;
  if (requested_block_q > 0) {
    return static_cast<int>(std::max<long long>(
        1, std::min<long long>(requested_block_q, q_count)));
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    return static_cast<int>(std::min<long long>(8, q_count));
  }

  const double nao2 = static_cast<double>(nao) * nao;
  const double naomo = static_cast<double>(nao) * nmo;
  const double nmo2 = static_cast<double>(nmo) * nmo;
  const double fixed_bytes = naomo * sizeof(double);
  const double bytes_per_q = (std::max<double>(ao_pair, mo_pair) +
                              std::max(nao2, naomo) + std::max(naomo, nmo2)) *
                             sizeof(double);
  const double free_d = static_cast<double>(free_bytes);
  const double reserve = std::max(1024.0 * 1024.0 * 1024.0, 0.15 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  const int block_q = bounded_auto_chunk(
      std::max(0.0, budget - fixed_bytes) / bytes_per_q, 1024, q_count);
  return block_q;
}

int choose_gradient_q_chunk(int device, int nmo, int ninner, long long q_count,
                            long long ngem, int x_matrices,
                            int requested_block_q, bool int2_resident,
                            std::size_t reclaimable_bytes) {
  if (q_count <= 0 || ninner <= 0) {
    return 1;
  }
  if (requested_block_q > 0) {
    return static_cast<int>(std::max<long long>(
        1, std::min<long long>(requested_block_q, q_count)));
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    const int fallback = static_cast<int>(std::min<long long>(32, q_count));
    return fallback;
  }

  constexpr int max_auto_q_chunk = 4096;
  const double free_d = static_cast<double>(free_bytes) + reclaimable_bytes;
  const double reserve = std::max(512.0 * 1024.0 * 1024.0, 0.10 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  const double c_bytes =
      static_cast<double>(nmo) * static_cast<double>(nmo) * sizeof(double);
  const double per_q_bytes =
      (static_cast<double>(x_matrices) * static_cast<double>(ninner) *
           static_cast<double>(nmo) +
       (int2_resident ? 0.0 : static_cast<double>(ngem))) *
      sizeof(double);
  const int q_chunk = bounded_auto_chunk((budget - c_bytes) / per_q_bytes,
                                         max_auto_q_chunk, q_count);
  return q_chunk;
}

int choose_q_q_chunk(int device, int nmo, int nact, long long q_count,
                     long long ngem, int requested_block_q, bool int2_resident,
                     std::size_t reclaimable_bytes) {
  if (q_count <= 0 || nact <= 0) {
    return 1;
  }
  if (requested_block_q > 0) {
    return static_cast<int>(std::max<long long>(
        1, std::min<long long>(requested_block_q, q_count)));
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    const int fallback = static_cast<int>(std::min<long long>(32, q_count));
    return fallback;
  }

  constexpr int max_auto_q_chunk = 4096;
  const double free_d = static_cast<double>(free_bytes) + reclaimable_bytes;
  const double reserve = std::max(512.0 * 1024.0 * 1024.0, 0.10 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  const double ngem_act =
      static_cast<double>(nact) * static_cast<double>(nact + 1) * 0.5;
  const double fixed_bytes =
      (ngem_act * ngem_act + static_cast<double>(nmo) * ngem_act +
       static_cast<double>(nact) * static_cast<double>(nmo)) *
      sizeof(double);
  const double per_q_bytes =
      ((int2_resident ? 0.0 : static_cast<double>(ngem)) + 2.0 * ngem_act +
       static_cast<double>(nmo)) *
      sizeof(double);
  const int q_chunk = bounded_auto_chunk((budget - fixed_bytes) / per_q_bytes,
                                         max_auto_q_chunk, q_count);
  return q_chunk;
}

int choose_coulomb_q_chunk(int device, long long q_count, long long ngem,
                           int requested_block_q, bool int2_resident,
                           std::size_t reclaimable_bytes) {
  if (q_count <= 0) {
    return 1;
  }
  if (requested_block_q > 0) {
    return static_cast<int>(std::max<long long>(
        1, std::min<long long>(requested_block_q, q_count)));
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    const int fallback = static_cast<int>(std::min<long long>(32, q_count));
    return fallback;
  }

  constexpr int max_auto_q_chunk = 4096;
  const double free_d = static_cast<double>(free_bytes) + reclaimable_bytes;
  const double reserve = std::max(512.0 * 1024.0 * 1024.0, 0.10 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  const double fixed_bytes = static_cast<double>(ngem) * sizeof(double);
  const double per_q_bytes =
      ((int2_resident ? 0.0 : static_cast<double>(ngem)) + 1.0) *
      sizeof(double);
  const int q_chunk = bounded_auto_chunk((budget - fixed_bytes) / per_q_bytes,
                                         max_auto_q_chunk, q_count);
  return q_chunk;
}

int process_range_on_device(int device, int nmo, long long q_begin,
                            long long q_end, long long ngem, double *int2,
                            const double *u_host, int requested_block_q,
                            std::atomic<bool> *tensor_mutated) {
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

  FocasSessionDevice *resident =
      session_device_slice(device, int2, ngem, q_begin, q_end);

  const int block_q = choose_block_q(
      device, nmo, q_end - q_begin, ngem, requested_block_q,
      slice_is_fully_resident(resident, q_end),
      resident == nullptr ? 0 : resident->workspace_bytes);
  const int nrow = nmo * block_q;
  const std::size_t u_bytes =
      static_cast<std::size_t>(nmo) * nmo * sizeof(double);
  const std::size_t packed_bytes =
      static_cast<std::size_t>(block_q) * ngem * sizeof(double);
  const std::size_t rectangular_bytes =
      static_cast<std::size_t>(nrow) * nmo * sizeof(double);
  const std::size_t left_bytes = rectangular_bytes;
  // A partially resident slice still needs somewhere to land its streamed tiles.
  const std::size_t stage_bytes =
      streaming_stage_bytes(resident, q_end, block_q, ngem);
  const std::size_t operation_workspace_bytes =
      u_bytes + (resident == nullptr ? packed_bytes : 0) +
      4 * rectangular_bytes;

  double *d_u = nullptr;
  double *d_packed = nullptr;
  double *d_stage = nullptr;
  double *d_right = nullptr;
  double *d_tmp = nullptr;
  double *d_left = nullptr;
  double *d_result = nullptr;
  bool pooled_workspace = false;

  auto cleanup = [&]() {
    if (d_result && !pooled_workspace)
      cudaFree(d_result);
    if (d_left && !pooled_workspace)
      cudaFree(d_left);
    if (d_tmp && !pooled_workspace)
      cudaFree(d_tmp);
    if (d_right && !pooled_workspace)
      cudaFree(d_right);
    if (d_packed && resident == nullptr)
      cudaFree(d_packed);
    if (d_u && !pooled_workspace)
      cudaFree(d_u);
    cublasDestroy(handle);
  };
  if (resident != nullptr) {
    d_packed = resident->d_int2;
    pooled_workspace = ensure_session_workspace(
        resident, u_bytes + 4 * rectangular_bytes + stage_bytes);
    if (pooled_workspace) {
      unsigned char *cursor = resident->workspace;
      d_u = take_workspace<double>(cursor, u_bytes / sizeof(double));
      d_right =
          take_workspace<double>(cursor, rectangular_bytes / sizeof(double));
      d_tmp =
          take_workspace<double>(cursor, rectangular_bytes / sizeof(double));
      d_left = take_workspace<double>(cursor, left_bytes / sizeof(double));
      d_result = take_workspace<double>(cursor, left_bytes / sizeof(double));
      if (stage_bytes > 0) {
        d_stage = take_workspace<double>(cursor, stage_bytes / sizeof(double));
      }
    }
  }
  if ((resident != nullptr && !pooled_workspace) ||
      (resident == nullptr &&
       (cudaMalloc(&d_u, u_bytes) != cudaSuccess ||
        cudaMalloc(&d_packed, packed_bytes) != cudaSuccess ||
        cudaMalloc(&d_right, rectangular_bytes) != cudaSuccess ||
        cudaMalloc(&d_tmp, rectangular_bytes) != cudaSuccess ||
        cudaMalloc(&d_left, left_bytes) != cudaSuccess ||
        cudaMalloc(&d_result, left_bytes) != cudaSuccess))) {
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

  for (long long q = q_begin; q < q_end;) {
    int bq = static_cast<int>(std::min<long long>(block_q, q_end - q));
    bq = clamp_tile_to_residency(resident, q, bq);
    const bool tile_resident =
        resident != nullptr && resident->tile_resident(q, bq);
    const int nrow_bq = nmo * bq;
    const std::size_t packed_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    double *host_block = int2 + q * ngem;
    // Resident tiles are transformed in place and stay authoritative on the
    // device; streamed tiles land in the staging buffer and are written back.
    double *device_block =
        tile_resident ? resident->d_int2 + (q - resident->q_begin) * ngem
                      : (resident == nullptr ? d_packed : d_stage);

    if (!tile_resident) {
      if (upload_stream_tile(device, device_block, host_block,
                             packed_bq_bytes) != cudaSuccess) {
        cleanup();
        return 14;
      }
    }
    const long long packed_total = static_cast<long long>(bq) * ngem;
    unpack_packed_symmetric_kernel<<<cuda_blocks(packed_total, threads),
                                     threads>>>(device_block, d_right, nmo, bq,
                                                ngem);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 15;
    }

    if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nrow_bq, nmo, nmo, &alpha,
                    d_right, nrow_bq, d_u, nmo, &beta, d_tmp,
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

    if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nmo, nrow_bq, nmo, &alpha,
                    d_u, nmo, d_left, nmo, &beta, d_result,
                    nmo) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 18;
    }

    // A resident tile is rotated in place, so the authoritative copy is mutated
    // by the scatter itself.  A streamed tile is scattered into staging memory
    // and does not touch the authoritative host copy until the write-back
    // below, so the flag is deferred until then: before it is set, a failure can
    // still be recovered by discarding the session and retrying on the host.
    if (tile_resident && tensor_mutated != nullptr) {
      tensor_mutated->store(true, std::memory_order_relaxed);
    }
    scatter_packed_symmetric_kernel<<<cuda_blocks(packed_total, threads),
                                      threads>>>(d_result, device_block, nmo,
                                                 bq, ngem);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 19;
    }

    if (!tile_resident) {
      if (tensor_mutated != nullptr) {
        tensor_mutated->store(true, std::memory_order_relaxed);
      }
      if (cudaMemcpy(host_block, device_block, packed_bq_bytes,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        cleanup();
        return 20;
      }
    }
    q += bq;
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
    cleanup();
    return 21;
  }
  cleanup();
  return 0;
}

int process_low_rank_range_on_device(int device, int nmo, int rank,
                                     long long q_begin, long long q_end,
                                     long long ngem, double *int2,
                                     const double *v_host, const double *a_host,
                                     int requested_block_q,
                                     std::atomic<bool> *tensor_mutated) {
  if (q_begin >= q_end)
    return 0;
  if (cudaSetDevice(device) != cudaSuccess)
    return 30;

  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    return 31;

  FocasSessionDevice *resident =
      session_device_slice(device, int2, ngem, q_begin, q_end);
  const int block_q = choose_low_rank_block_q(
      device, nmo, rank, q_end - q_begin, ngem, requested_block_q,
      slice_is_fully_resident(resident, q_end),
      resident == nullptr ? 0 : resident->workspace_bytes);

  const std::size_t n = static_cast<std::size_t>(nmo);
  const std::size_t r = static_cast<std::size_t>(rank);
  const std::size_t b = static_cast<std::size_t>(block_q);
  const std::size_t v_elements = n * r;
  const std::size_t a_elements = r * r;
  const std::size_t packed_elements = b * static_cast<std::size_t>(ngem);
  const std::size_t dense_elements = b * n * n;
  const std::size_t rectangular_elements = b * n * r;
  const std::size_t small_elements = b * r * r;
  const std::size_t pooled_elements = v_elements + a_elements + dense_elements +
                                      2 * rectangular_elements +
                                      2 * small_elements;
  const std::size_t operation_workspace_bytes =
      (pooled_elements + (resident == nullptr ? packed_elements : 0)) *
      sizeof(double);
  const std::size_t stage_bytes =
      streaming_stage_bytes(resident, q_end, block_q, ngem);

  double *d_v = nullptr;
  double *d_stage = nullptr;
  double *d_a = nullptr;
  double *d_packed = nullptr;
  double *d_dense = nullptr;
  double *d_y = nullptr;
  double *d_r = nullptr;
  double *d_z = nullptr;
  double *d_small_tmp = nullptr;
  bool pooled_workspace = false;

  auto cleanup = [&]() {
    if (d_small_tmp && !pooled_workspace)
      cudaFree(d_small_tmp);
    if (d_z && !pooled_workspace)
      cudaFree(d_z);
    if (d_r && !pooled_workspace)
      cudaFree(d_r);
    if (d_y && !pooled_workspace)
      cudaFree(d_y);
    if (d_dense && !pooled_workspace)
      cudaFree(d_dense);
    if (d_packed && resident == nullptr)
      cudaFree(d_packed);
    if (d_a && !pooled_workspace)
      cudaFree(d_a);
    if (d_v && !pooled_workspace)
      cudaFree(d_v);
    cublasDestroy(handle);
  };
  if (resident != nullptr) {
    d_packed = resident->d_int2;
    pooled_workspace = ensure_session_workspace(
        resident, pooled_elements * sizeof(double) + stage_bytes);
    if (pooled_workspace) {
      unsigned char *cursor = resident->workspace;
      d_v = take_workspace<double>(cursor, v_elements);
      d_a = take_workspace<double>(cursor, a_elements);
      d_dense = take_workspace<double>(cursor, dense_elements);
      d_y = take_workspace<double>(cursor, rectangular_elements);
      d_r = take_workspace<double>(cursor, rectangular_elements);
      d_z = take_workspace<double>(cursor, small_elements);
      d_small_tmp = take_workspace<double>(cursor, small_elements);
      if (stage_bytes > 0) {
        d_stage = take_workspace<double>(cursor, stage_bytes / sizeof(double));
      }
    }
  }
  if ((resident != nullptr && !pooled_workspace) ||
      (resident == nullptr &&
       (cudaMalloc(&d_v, v_elements * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_a, a_elements * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_packed, packed_elements * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc(&d_dense, dense_elements * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_y, rectangular_elements * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc(&d_r, rectangular_elements * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc(&d_z, small_elements * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_small_tmp, small_elements * sizeof(double)) !=
            cudaSuccess))) {
    cleanup();
    return 32;
  }
  if (cudaMemcpy(d_v, v_host, v_elements * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_a, a_host, a_elements * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cleanup();
    return 33;
  }

  const double one = 1.0;
  const double zero = 0.0;
  const double half = 0.5;
  constexpr int threads = 256;
  const long long dense_stride = static_cast<long long>(n) * n;
  const long long rectangular_stride = static_cast<long long>(n) * r;
  const long long small_stride = static_cast<long long>(r) * r;

  for (long long q = q_begin; q < q_end;) {
    int bq = static_cast<int>(std::min<long long>(block_q, q_end - q));
    bq = clamp_tile_to_residency(resident, q, bq);
    const bool tile_resident =
        resident != nullptr && resident->tile_resident(q, bq);
    const std::size_t packed_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    double *host_block = int2 + q * ngem;
    double *device_block =
        tile_resident ? d_packed + (q - resident->q_begin) * ngem
                      : (resident == nullptr ? d_packed : d_stage);

    if (!tile_resident) {
      if (upload_stream_tile(device, device_block, host_block,
                             packed_bq_bytes) != cudaSuccess) {
        cleanup();
        return 34;
      }
    }
    const long long packed_total = static_cast<long long>(bq) * ngem;
    unpack_packed_q_major_kernel<<<cuda_blocks(packed_total, threads),
                                   threads>>>(device_block, d_dense, nmo, bq,
                                              ngem);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 35;
    }

    // Y_q = B_q V
    if (cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nmo, rank,
                                  nmo, &one, d_dense, nmo, dense_stride, d_v,
                                  nmo, 0, &zero, d_y, nmo, rectangular_stride,
                                  bq) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 36;
    }
    // Z_q = V^T Y_q
    if (cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, rank, rank,
                                  nmo, &one, d_v, nmo, 0, d_y, nmo,
                                  rectangular_stride, &zero, d_z, rank,
                                  small_stride, bq) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 37;
    }
    // R_q = Y_q A
    if (cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nmo, rank,
                                  rank, &one, d_y, nmo, rectangular_stride, d_a,
                                  rank, 0, &zero, d_r, nmo, rectangular_stride,
                                  bq) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 38;
    }
    // C_q = A^T Z_q A, using the two rank-by-rank buffers.
    if (cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, rank, rank,
                                  rank, &one, d_a, rank, 0, d_z, rank,
                                  small_stride, &zero, d_small_tmp, rank,
                                  small_stride, bq) != CUBLAS_STATUS_SUCCESS ||
        cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank,
                                  rank, &one, d_small_tmp, rank, small_stride,
                                  d_a, rank, 0, &zero, d_z, rank, small_stride,
                                  bq) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 39;
    }
    // R_q = Y_q A + 1/2 V C_q.  Then B'_q-B_q = R_q V^T + V R_q^T.
    if (cublasDgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, nmo, rank, rank, &half, d_v, nmo,
            0, d_z, rank, small_stride, &one, d_r, nmo, rectangular_stride,
            bq) != CUBLAS_STATUS_SUCCESS ||
        cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, nmo, nmo,
                                  rank, &one, d_r, nmo, rectangular_stride, d_v,
                                  nmo, 0, &zero, d_dense, nmo, dense_stride,
                                  bq) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 40;
    }

    if (tile_resident && tensor_mutated != nullptr) {
      tensor_mutated->store(true, std::memory_order_relaxed);
    }
    scatter_low_rank_update_kernel<<<cuda_blocks(packed_total, threads),
                                     threads>>>(d_dense, device_block, nmo, bq,
                                                ngem);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 41;
    }

    if (!tile_resident) {
      if (tensor_mutated != nullptr) {
        tensor_mutated->store(true, std::memory_order_relaxed);
      }
      if (cudaMemcpy(host_block, device_block, packed_bq_bytes,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        cleanup();
        return 42;
      }
    }
    q += bq;
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
    cleanup();
    return 43;
  }
  cleanup();
  return 0;
}

int transform_ao_to_mo_on_device(int device, int nao, int nmo,
                                 long long q_begin, long long q_end,
                                 long long ao_pair, long long mo_pair,
                                 const double *qao_host, double *qmo_host,
                                 const double *c_host, int requested_block_q) {
  if (q_begin >= q_end)
    return 0;
  if (cudaSetDevice(device) != cudaSuccess)
    return 210;

  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    return 211;
  const int block_q = choose_ao_to_mo_block_q(
      device, nao, nmo, q_end - q_begin, ao_pair, mo_pair, requested_block_q);
  if (static_cast<long long>(nao) * block_q > std::numeric_limits<int>::max() ||
      static_cast<long long>(nmo) * block_q > std::numeric_limits<int>::max()) {
    cublasDestroy(handle);
    return 212;
  }

  const std::size_t c_bytes =
      static_cast<std::size_t>(nao) * nmo * sizeof(double);
  const std::size_t packed_bytes =
      static_cast<std::size_t>(block_q) *
      static_cast<std::size_t>(std::max(ao_pair, mo_pair)) * sizeof(double);
  const std::size_t right_elements =
      static_cast<std::size_t>(block_q) *
      std::max(static_cast<std::size_t>(nao) * nao,
               static_cast<std::size_t>(nao) * nmo);
  const std::size_t tmp_elements =
      static_cast<std::size_t>(block_q) *
      std::max(static_cast<std::size_t>(nao) * nmo,
               static_cast<std::size_t>(nmo) * nmo);
  const std::size_t right_bytes = right_elements * sizeof(double);
  const std::size_t tmp_bytes = tmp_elements * sizeof(double);

  double *d_c = nullptr;
  double *d_packed = nullptr;
  double *d_right = nullptr;
  double *d_tmp = nullptr;
  auto cleanup = [&]() {
    if (d_tmp)
      cudaFree(d_tmp);
    if (d_right)
      cudaFree(d_right);
    if (d_packed)
      cudaFree(d_packed);
    if (d_c)
      cudaFree(d_c);
    cublasDestroy(handle);
  };
  if (cudaMalloc(&d_c, c_bytes) != cudaSuccess ||
      cudaMalloc(&d_packed, packed_bytes) != cudaSuccess ||
      cudaMalloc(&d_right, right_bytes) != cudaSuccess ||
      cudaMalloc(&d_tmp, tmp_bytes) != cudaSuccess) {
    cleanup();
    return 213;
  }
  if (cudaMemcpy(d_c, c_host, c_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cleanup();
    return 214;
  }

  constexpr int threads = 256;
  const double alpha = 1.0;
  const double beta = 0.0;
  for (long long q = q_begin; q < q_end; q += block_q) {
    const int bq = static_cast<int>(std::min<long long>(block_q, q_end - q));
    const int ao_rows = nao * bq;
    const int mo_columns = nmo * bq;
    const std::size_t input_bytes =
        static_cast<std::size_t>(bq) * ao_pair * sizeof(double);
    const std::size_t output_bytes =
        static_cast<std::size_t>(bq) * mo_pair * sizeof(double);
    if (cudaMemcpy(d_packed, qao_host + q * ao_pair, input_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return 215;
    }
    const long long packed_total = static_cast<long long>(bq) * ao_pair;
    unpack_packed_symmetric_kernel<<<cuda_blocks(packed_total, threads),
                                     threads>>>(d_packed, d_right, nao, bq,
                                                ao_pair);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 216;
    }

    // c_host is row-major (AO x MO), hence column-major (MO x AO) to
    // cuBLAS. The transpose supplies C for B*C in the first contraction.
    if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, ao_rows, nmo, nao, &alpha,
                    d_right, ao_rows, d_c, nmo, &beta, d_tmp,
                    ao_rows) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 217;
    }

    const long long half_total = static_cast<long long>(bq) * nao * nmo;
    make_rectangular_left_mat_kernel<<<cuda_blocks(half_total, threads),
                                       threads>>>(d_tmp, d_right, nao, nmo, bq);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 218;
    }

    if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nmo, mo_columns, nao,
                    &alpha, d_c, nmo, d_right, nao, &beta, d_tmp,
                    nmo) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 219;
    }

    const long long output_total = static_cast<long long>(bq) * mo_pair;
    scatter_packed_symmetric_kernel<<<cuda_blocks(output_total, threads),
                                      threads>>>(d_tmp, d_packed, nmo, bq,
                                                 mo_pair);
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 220;
    }
    if (cudaMemcpy(qmo_host + q * mo_pair, d_packed, output_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      cleanup();
      return 221;
    }
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
    cleanup();
    return 222;
  }
  cleanup();
  return 0;
}

int compute_fi_exchange_on_device(int device, int nmo, int ndoc, long long nQ,
                                  long long q_begin, long long q_end,
                                  long long ngem, const double *int2,
                                  const int *doc_df, int requested_q_chunk,
                                  std::vector<double> &host_c) {
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

  FocasSessionDevice *resident =
      session_device_slice(device, int2, ngem, q_begin, q_end);

  const int q_chunk = choose_gradient_q_chunk(
      device, nmo, ndoc, q_end - q_begin, ngem, 1, requested_q_chunk,
      slice_is_fully_resident(resident, q_end),
      resident == nullptr ? 0 : resident->workspace_bytes);
  const int max_ldx_ll = static_cast<int>(std::min<long long>(
      static_cast<long long>(q_chunk) * ndoc, std::numeric_limits<int>::max()));
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
  const std::size_t inner_bytes =
      doc_df == nullptr ? 0 : static_cast<std::size_t>(ndoc) * sizeof(int);

  const std::size_t stage_bytes =
      streaming_stage_bytes(resident, q_end, q_chunk, ngem);

  double *d_int2 = nullptr;
  double *d_stage = nullptr;
  double *d_x = nullptr;
  double *d_c = nullptr;
  int *d_inner = nullptr;
  bool pooled_workspace = false;
  auto cleanup = [&]() {
    if (d_inner && !pooled_workspace)
      cudaFree(d_inner);
    if (d_c && !pooled_workspace)
      cudaFree(d_c);
    if (d_x && !pooled_workspace)
      cudaFree(d_x);
    if (d_int2 && resident == nullptr)
      cudaFree(d_int2);
    cublasDestroy(handle);
  };
  if (resident != nullptr) {
    d_int2 = resident->d_int2;
    pooled_workspace = ensure_session_workspace(
        resident, x_bytes + c_bytes + inner_bytes + stage_bytes);
    if (pooled_workspace) {
      unsigned char *cursor = resident->workspace;
      d_x = take_workspace<double>(cursor, x_bytes / sizeof(double));
      d_c = take_workspace<double>(cursor, c_bytes / sizeof(double));
      if (inner_bytes > 0) {
        d_inner = take_workspace<int>(cursor, inner_bytes / sizeof(int));
      }
      if (stage_bytes > 0) {
        d_stage = take_workspace<double>(cursor, stage_bytes / sizeof(double));
      }
    }
  }
  if ((resident != nullptr && !pooled_workspace) ||
      (resident == nullptr && (cudaMalloc(&d_int2, int2_bytes) != cudaSuccess ||
                               cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
                               cudaMalloc(&d_c, c_bytes) != cudaSuccess))) {
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
    if (!pooled_workspace && cudaMalloc(&d_inner, inner_bytes) != cudaSuccess) {
      cleanup();
      return 34;
    }
    if (cudaMemcpy(d_inner, doc_df, inner_bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      cleanup();
      return 34;
    }
  }

  constexpr int threads = 256;
  const double alpha = 1.0;
  const double beta = 1.0;
  for (long long q = q_begin; q < q_end;) {
    int bq = static_cast<int>(std::min<long long>(q_chunk, q_end - q));
    bq = clamp_tile_to_residency(resident, q, bq);
    const bool tile_resident =
        resident != nullptr && resident->tile_resident(q, bq);
    const int ldx = bq * ndoc;
    const std::size_t int2_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    double *const stage_block = resident == nullptr ? d_int2 : d_stage;
    const double *device_block =
        tile_resident ? d_int2 + (q - resident->q_begin) * ngem : stage_block;
    if (!tile_resident) {
      if (upload_stream_tile(device, stage_block, int2 + q * ngem,
                             int2_bq_bytes) != cudaSuccess) {
        cleanup();
        return 35;
      }
    }
    const long long total = static_cast<long long>(ldx) * nmo;
    if (d_inner != nullptr) {
      build_pair_column_matrix_list_kernel<<<cuda_blocks(total, threads),
                                             threads>>>(
          device_block, d_x, nmo, d_inner, ndoc, bq, ngem, ldx);
    } else {
      build_pair_column_matrix_kernel<<<cuda_blocks(total, threads), threads>>>(
          device_block, d_x, nmo, 0, ndoc, 0, bq, ngem, ldx);
    }
    if (cudaGetLastError() != cudaSuccess) {
      cleanup();
      return 36;
    }
    if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nmo, nmo, ldx, &alpha,
                    d_x, ldx, d_x, ldx, &beta, d_c,
                    nmo) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 37;
    }
    q += bq;
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
                                  const int *act_df, int requested_q_chunk,
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

  FocasSessionDevice *resident =
      session_device_slice(device, int2, ngem, q_begin, q_end);

  const int q_chunk = choose_gradient_q_chunk(
      device, nmo, nact, q_end - q_begin, ngem, 2, requested_q_chunk,
      slice_is_fully_resident(resident, q_end),
      resident == nullptr ? 0 : resident->workspace_bytes);
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
  const std::size_t inner_bytes =
      act_df == nullptr ? 0 : static_cast<std::size_t>(nact) * sizeof(int);
  const std::size_t stage_bytes =
      streaming_stage_bytes(resident, q_end, q_chunk, ngem);

  double *d_int2 = nullptr;
  double *d_stage = nullptr;
  double *d_x = nullptr;
  double *d_y = nullptr;
  double *d_c = nullptr;
  double *d_den1 = nullptr;
  int *d_inner = nullptr;
  bool pooled_workspace = false;
  auto cleanup = [&]() {
    if (d_inner && !pooled_workspace)
      cudaFree(d_inner);
    if (d_den1 && !pooled_workspace)
      cudaFree(d_den1);
    if (d_c && !pooled_workspace)
      cudaFree(d_c);
    if (d_y && !pooled_workspace)
      cudaFree(d_y);
    if (d_x && !pooled_workspace)
      cudaFree(d_x);
    if (d_int2 && resident == nullptr)
      cudaFree(d_int2);
    cublasDestroy(handle);
  };
  if (resident != nullptr) {
    d_int2 = resident->d_int2;
    pooled_workspace = ensure_session_workspace(
        resident,
        2 * x_bytes + c_bytes + den_bytes + inner_bytes + stage_bytes);
    if (pooled_workspace) {
      unsigned char *cursor = resident->workspace;
      d_x = take_workspace<double>(cursor, x_bytes / sizeof(double));
      d_y = take_workspace<double>(cursor, x_bytes / sizeof(double));
      d_c = take_workspace<double>(cursor, c_bytes / sizeof(double));
      d_den1 = take_workspace<double>(cursor, den_bytes / sizeof(double));
      if (inner_bytes > 0) {
        d_inner = take_workspace<int>(cursor, inner_bytes / sizeof(int));
      }
      if (stage_bytes > 0) {
        d_stage = take_workspace<double>(cursor, stage_bytes / sizeof(double));
      }
    }
  }
  if ((resident != nullptr && !pooled_workspace) ||
      (resident == nullptr &&
       (cudaMalloc(&d_int2, int2_bytes) != cudaSuccess ||
        cudaMalloc(&d_x, x_bytes) != cudaSuccess ||
        cudaMalloc(&d_y, x_bytes) != cudaSuccess ||
        cudaMalloc(&d_c, c_bytes) != cudaSuccess ||
        cudaMalloc(&d_den1, den_bytes) != cudaSuccess))) {
    cleanup();
    return 43;
  }
  if (cudaMemcpy(d_den1, den1, den_bytes, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cleanup();
    return 44;
  }
  if (cudaMemset(d_c, 0, c_bytes) != cudaSuccess) {
    cleanup();
    return 44;
  }
  // optional df-order active-orbital list (symmetry-general path); null => the
  // active orbitals are the contiguous block [ndoc, ndoc+nact) (C1)
  if (act_df != nullptr) {
    if (!pooled_workspace && cudaMalloc(&d_inner, inner_bytes) != cudaSuccess) {
      cleanup();
      return 44;
    }
    if (cudaMemcpy(d_inner, act_df, inner_bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      cleanup();
      return 44;
    }
  }

  constexpr int threads = 256;
  const double alpha = 1.0;
  const double beta = 1.0;
  for (long long q = q_begin; q < q_end;) {
    int bq = static_cast<int>(std::min<long long>(q_chunk, q_end - q));
    bq = clamp_tile_to_residency(resident, q, bq);
    const bool tile_resident =
        resident != nullptr && resident->tile_resident(q, bq);
    const int ldx = bq * nact;
    const std::size_t int2_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    double *const stage_block = resident == nullptr ? d_int2 : d_stage;
    const double *device_block =
        tile_resident ? d_int2 + (q - resident->q_begin) * ngem : stage_block;
    if (!tile_resident) {
      if (upload_stream_tile(device, stage_block, int2 + q * ngem,
                             int2_bq_bytes) != cudaSuccess) {
        cleanup();
        return 45;
      }
    }
    const long long total = static_cast<long long>(ldx) * nmo;
    if (d_inner != nullptr) {
      build_pair_column_matrix_list_kernel<<<cuda_blocks(total, threads),
                                             threads>>>(
          device_block, d_x, nmo, d_inner, nact, bq, ngem, ldx);
    } else {
      build_pair_column_matrix_kernel<<<cuda_blocks(total, threads), threads>>>(
          device_block, d_x, nmo, ndoc, nact, 0, bq, ngem, ldx);
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
    if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nmo, nmo, ldx, &alpha,
                    d_x, ldx, d_y, ldx, &beta, d_c,
                    nmo) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 48;
    }
    q += bq;
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
                        const std::vector<double> &scaled_d2, const int *act_df,
                        int requested_q_chunk, std::vector<double> &host_q) {
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

  FocasSessionDevice *resident =
      session_device_slice(device, int2, ngem, q_begin, q_end);

  const long long ngem_act_ll = static_cast<long long>(nact) * (nact + 1) / 2;
  if (ngem_act_ll > std::numeric_limits<int>::max()) {
    cublasDestroy(handle);
    return 82;
  }
  const int ngem_act = static_cast<int>(ngem_act_ll);
  const int q_chunk = choose_q_q_chunk(
      device, nmo, nact, q_end - q_begin, ngem, requested_q_chunk,
      slice_is_fully_resident(resident, q_end),
      resident == nullptr ? 0 : resident->workspace_bytes);
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
  const std::size_t act_index_bytes =
      act_df == nullptr ? 0 : static_cast<std::size_t>(nact) * sizeof(int);
  const std::size_t stage_bytes =
      streaming_stage_bytes(resident, q_end, q_chunk, ngem);

  double *d_int2 = nullptr;
  double *d_stage = nullptr;
  double *d_active = nullptr;
  double *d_d2 = nullptr;
  double *d_qint = nullptr;
  double *d_b = nullptr;
  double *d_result = nullptr;
  double *d_q = nullptr;
  int *d_act = nullptr;
  bool pooled_workspace = false;
  auto cleanup = [&]() {
    if (d_act && !pooled_workspace)
      cudaFree(d_act);
    if (d_q && !pooled_workspace)
      cudaFree(d_q);
    if (d_result && !pooled_workspace)
      cudaFree(d_result);
    if (d_b && !pooled_workspace)
      cudaFree(d_b);
    if (d_qint && !pooled_workspace)
      cudaFree(d_qint);
    if (d_d2 && !pooled_workspace)
      cudaFree(d_d2);
    if (d_active && !pooled_workspace)
      cudaFree(d_active);
    if (d_int2 && resident == nullptr)
      cudaFree(d_int2);
    cublasDestroy(handle);
  };
  if (resident != nullptr) {
    d_int2 = resident->d_int2;
    pooled_workspace = ensure_session_workspace(
        resident, 2 * active_bytes + d2_bytes + b_bytes + result_bytes +
                      q_bytes + act_index_bytes + stage_bytes);
    if (pooled_workspace) {
      unsigned char *cursor = resident->workspace;
      d_active = take_workspace<double>(cursor, active_bytes / sizeof(double));
      d_d2 = take_workspace<double>(cursor, d2_bytes / sizeof(double));
      d_qint = take_workspace<double>(cursor, active_bytes / sizeof(double));
      d_b = take_workspace<double>(cursor, b_bytes / sizeof(double));
      d_result = take_workspace<double>(cursor, result_bytes / sizeof(double));
      d_q = take_workspace<double>(cursor, q_bytes / sizeof(double));
      if (act_index_bytes > 0) {
        d_act = take_workspace<int>(cursor, act_index_bytes / sizeof(int));
      }
      if (stage_bytes > 0) {
        d_stage = take_workspace<double>(cursor, stage_bytes / sizeof(double));
      }
    }
  }
  if ((resident != nullptr && !pooled_workspace) ||
      (resident == nullptr &&
       (cudaMalloc(&d_int2, int2_bytes) != cudaSuccess ||
        cudaMalloc(&d_active, active_bytes) != cudaSuccess ||
        cudaMalloc(&d_d2, d2_bytes) != cudaSuccess ||
        cudaMalloc(&d_qint, active_bytes) != cudaSuccess ||
        cudaMalloc(&d_b, b_bytes) != cudaSuccess ||
        cudaMalloc(&d_result, result_bytes) != cudaSuccess ||
        cudaMalloc(&d_q, q_bytes) != cudaSuccess))) {
    cleanup();
    return 83;
  }
  if (cudaMemcpy(d_d2, scaled_d2.data(), d2_bytes, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cleanup();
    return 84;
  }
  if (cudaMemset(d_q, 0, q_bytes) != cudaSuccess) {
    cleanup();
    return 84;
  }
  // optional df-order active-orbital list (symmetry-general path); null => the
  // active orbitals are the contiguous block [ndoc, ndoc+nact) (C1)
  if (act_df != nullptr) {
    if (!pooled_workspace &&
        cudaMalloc(&d_act, act_index_bytes) != cudaSuccess) {
      cleanup();
      return 84;
    }
    if (cudaMemcpy(d_act, act_df, act_index_bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      cleanup();
      return 84;
    }
  }

  constexpr int threads = 256;
  const double alpha = 1.0;
  const double beta0 = 0.0;
  for (long long q = q_begin; q < q_end;) {
    int bq = static_cast<int>(std::min<long long>(q_chunk, q_end - q));
    bq = clamp_tile_to_residency(resident, q, bq);
    const bool tile_resident =
        resident != nullptr && resident->tile_resident(q, bq);
    const std::size_t int2_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    double *const stage_block = resident == nullptr ? d_int2 : d_stage;
    const double *device_block =
        tile_resident ? d_int2 + (q - resident->q_begin) * ngem : stage_block;
    if (!tile_resident) {
      if (upload_stream_tile(device, stage_block, int2 + q * ngem,
                             int2_bq_bytes) != cudaSuccess) {
        cleanup();
        return 85;
      }
    }
    const long long active_total = static_cast<long long>(bq) * ngem_act;
    build_active_pair_matrix_kernel<<<cuda_blocks(active_total, threads),
                                      threads>>>(device_block, d_active, ndoc,
                                                 nact, bq, ngem, bq, d_act);
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
                                           threads>>>(
          device_block, d_b, nmo, ndoc, u, bq, ngem, bq, d_act);
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
      scatter_q_result_kernel<<<
          cuda_blocks(static_cast<long long>(nact) * nmo, threads), threads>>>(
          d_result, d_q, nmo, nact, u);
      if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return 90;
      }
    }
    q += bq;
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
                             const double *int2, std::vector<double> &qvec) {
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
                              const double *qvec, const double *den1, int ndoc,
                              int nact, int requested_q_chunk,
                              const char *label,
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

  FocasSessionDevice *resident =
      session_device_slice(device, int2, ngem, q_begin, q_end);

  const int q_chunk = choose_coulomb_q_chunk(
      device, q_end - q_begin, ngem, requested_q_chunk,
      slice_is_fully_resident(resident, q_end),
      resident == nullptr ? 0 : resident->workspace_bytes);
  const std::size_t int2_bytes =
      static_cast<std::size_t>(q_chunk) * ngem * sizeof(double);
  const std::size_t qvec_bytes =
      static_cast<std::size_t>(q_chunk) * sizeof(double);
  const std::size_t pair_bytes =
      static_cast<std::size_t>(ngem) * sizeof(double);
  const bool inactive_coulomb = label[1] == 'i';
  const std::size_t den1_bytes =
      resident != nullptr && !inactive_coulomb
          ? static_cast<std::size_t>(nact) * (nact + 1) / 2 * sizeof(double)
          : 0;

  const std::size_t stage_bytes =
      streaming_stage_bytes(resident, q_end, q_chunk, ngem);

  double *d_int2 = nullptr;
  double *d_stage = nullptr;
  double *d_qvec = nullptr;
  double *d_pairs = nullptr;
  double *d_den1 = nullptr;
  bool pooled_workspace = false;
  auto cleanup = [&]() {
    if (d_den1 && !pooled_workspace)
      cudaFree(d_den1);
    if (d_pairs && !pooled_workspace)
      cudaFree(d_pairs);
    if (d_qvec && !pooled_workspace)
      cudaFree(d_qvec);
    if (d_int2 && resident == nullptr)
      cudaFree(d_int2);
    cublasDestroy(handle);
  };
  if (resident != nullptr) {
    d_int2 = resident->d_int2;
    pooled_workspace = ensure_session_workspace(
        resident, qvec_bytes + pair_bytes + den1_bytes + stage_bytes);
    if (pooled_workspace) {
      unsigned char *cursor = resident->workspace;
      d_qvec = take_workspace<double>(cursor, qvec_bytes / sizeof(double));
      d_pairs = take_workspace<double>(cursor, pair_bytes / sizeof(double));
      if (den1_bytes > 0) {
        d_den1 = take_workspace<double>(cursor, den1_bytes / sizeof(double));
      }
      if (stage_bytes > 0) {
        d_stage = take_workspace<double>(cursor, stage_bytes / sizeof(double));
      }
    }
  }
  if ((resident != nullptr && !pooled_workspace) ||
      (resident == nullptr &&
       (cudaMalloc(&d_int2, int2_bytes) != cudaSuccess ||
        cudaMalloc(&d_qvec, qvec_bytes) != cudaSuccess ||
        cudaMalloc(&d_pairs, pair_bytes) != cudaSuccess)) ||
      (resident != nullptr && den1_bytes > 0 && d_den1 == nullptr)) {
    cleanup();
    return 112;
  }
  if (cudaMemset(d_pairs, 0, pair_bytes) != cudaSuccess) {
    cleanup();
    return 113;
  }

  if (den1_bytes > 0) {
    if (den1 == nullptr || cudaMemcpy(d_den1, den1, den1_bytes,
                                      cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return 113;
    }
  }

  const double alpha = 1.0;
  const double beta = 1.0;
  for (long long q = q_begin; q < q_end;) {
    int bq = static_cast<int>(std::min<long long>(q_chunk, q_end - q));
    bq = clamp_tile_to_residency(resident, q, bq);
    const bool tile_resident =
        resident != nullptr && resident->tile_resident(q, bq);
    const std::size_t int2_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    const std::size_t qvec_bq_bytes =
        static_cast<std::size_t>(bq) * sizeof(double);
    double *const stage_block = resident == nullptr ? d_int2 : d_stage;
    const double *device_block =
        tile_resident ? d_int2 + (q - resident->q_begin) * ngem : stage_block;
    // Resident rows may be newer on the device than on the host, so their
    // Coulomb vector is rebuilt from device data.  Streamed rows are current on
    // the host, so the precomputed host vector is uploaded with the tile.
    if (!tile_resident) {
      if (upload_stream_tile(device, stage_block, int2 + q * ngem,
                             int2_bq_bytes) != cudaSuccess ||
          cudaMemcpy(d_qvec, qvec + q, qvec_bq_bytes, cudaMemcpyHostToDevice) !=
              cudaSuccess) {
        cleanup();
        return 114;
      }
    }
    if (tile_resident) {
      constexpr int threads = 256;
      if (inactive_coulomb) {
        build_fi_coulomb_vector_kernel<<<cuda_blocks(bq, threads), threads>>>(
            device_block, d_qvec, ndoc, bq, ngem);
      } else {
        build_fa_coulomb_vector_kernel<<<cuda_blocks(bq, threads), threads>>>(
            device_block, d_den1, d_qvec, ndoc, nact, bq, ngem);
      }
      if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return 114;
      }
    }
    if (cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(ngem), bq, &alpha,
                    device_block, static_cast<int>(ngem), d_qvec, 1, &beta,
                    d_pairs, 1) != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return 115;
    }
    q += bq;
  }
  if (cudaMemcpy(host_pairs.data(), d_pairs, pair_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
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
                     Worker worker, std::vector<double> &pair_total) {
  const int devices = device_count_from_request(max_devices);
  if (devices <= 0) {
    return 118;
  }

  std::vector<std::thread> workers;
  std::vector<int> statuses(devices, 0);
  std::vector<std::vector<double>> partials(
      devices, std::vector<double>(static_cast<std::size_t>(pair_count), 0.0));

  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share =
        (remaining + remaining_devices - 1) / remaining_devices;
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
int run_exchange_workers(int nmo, long long nQ, int max_devices, Worker worker,
                         std::vector<double> &c_total) {
  const int devices = device_count_from_request(max_devices);
  if (devices <= 0) {
    return 50;
  }

  std::vector<std::thread> workers;
  std::vector<int> statuses(devices, 0);
  std::vector<std::vector<double>> partials(
      devices, std::vector<double>(static_cast<std::size_t>(nmo) * nmo, 0.0));

  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share =
        (remaining + remaining_devices - 1) / remaining_devices;
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

// ---------------------------------------------------------------------------
// Fused C1 gradient.
//
// The five C1 gradient operators -- Fi Coulomb, Fa Coulomb, Fi exchange, Fa
// exchange and the Q contraction -- all read the same DF tensor.  Run
// separately they walk it five times, so a partially resident tensor pays five
// PCIe passes per gradient evaluation on top of the transform's two.  This
// routine walks it once: every Q tile is made available on the device a single
// time and immediately consumed by all five contractions, cutting a gradient
// from five passes to one.
//
// It also removes the host-side Coulomb vector entirely.  Because the tile is
// always on the device here (resident or staged), both Coulomb vectors are
// built on the GPU, so the host never has to sweep the packed tensor.
// ---------------------------------------------------------------------------
int choose_fused_gradient_q_chunk(int device, int nmo, int ndoc, int nact,
                                  long long q_count, long long ngem,
                                  int ngem_act, int requested_block_q,
                                  bool int2_resident,
                                  std::size_t reclaimable_bytes) {
  if (q_count <= 0) {
    return 1;
  }
  if (requested_block_q > 0) {
    return static_cast<int>(std::max<long long>(
        1, std::min<long long>(requested_block_q, q_count)));
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      free_bytes == 0) {
    return static_cast<int>(std::min<long long>(32, q_count));
  }

  constexpr int max_auto_q_chunk = 4096;
  const double n = static_cast<double>(nmo);
  const double free_d = static_cast<double>(free_bytes) + reclaimable_bytes;
  const double reserve = std::max(512.0 * 1024.0 * 1024.0, 0.10 * free_d);
  const double budget = std::max(0.0, free_d - reserve) * 0.75;
  // Fixed: two nmo*nmo exchange accumulators, two packed Coulomb accumulators,
  // the scaled 2-RDM, the Q result and output, and the packed active 1-RDM.
  const double fixed_bytes =
      (2.0 * n * n + 2.0 * static_cast<double>(ngem) +
       static_cast<double>(ngem_act) * static_cast<double>(ngem_act) +
       n * static_cast<double>(ngem_act) + static_cast<double>(nact) * n +
       static_cast<double>(nact) * (nact + 1) / 2) *
      sizeof(double);
  // Per Q row: the doc and active pair-column matrices, the active-density
  // product, the Q active/intermediate pair blocks, one general-active column,
  // the two Coulomb vector entries, and the streamed tile when not resident.
  const double per_q_bytes =
      (static_cast<double>(ndoc) * n + 2.0 * static_cast<double>(nact) * n +
       2.0 * static_cast<double>(ngem_act) + n + 2.0 +
       (int2_resident ? 0.0 : static_cast<double>(ngem))) *
      sizeof(double);
  return bounded_auto_chunk((budget - fixed_bytes) / per_q_bytes,
                            max_auto_q_chunk, q_count);
}

int compute_gradient_all_on_device(
    int device, int nmo, int ndoc, int nact, long long q_begin, long long q_end,
    long long ngem, const double *int2, const double *den1_act,
    const std::vector<double> &scaled_d2, int requested_q_chunk,
    std::vector<double> &host_c_fi, std::vector<double> &host_c_fa,
    std::vector<double> &host_pairs_fi, std::vector<double> &host_pairs_fa,
    std::vector<double> &host_q) {
  if (q_begin >= q_end) {
    std::fill(host_c_fi.begin(), host_c_fi.end(), 0.0);
    std::fill(host_c_fa.begin(), host_c_fa.end(), 0.0);
    std::fill(host_pairs_fi.begin(), host_pairs_fi.end(), 0.0);
    std::fill(host_pairs_fa.begin(), host_pairs_fa.end(), 0.0);
    std::fill(host_q.begin(), host_q.end(), 0.0);
    return 0;
  }
  if (cudaSetDevice(device) != cudaSuccess) {
    return 300;
  }
  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    return 301;
  }

  FocasSessionDevice *resident =
      session_device_slice(device, int2, ngem, q_begin, q_end);

  const long long ngem_act_ll = static_cast<long long>(nact) * (nact + 1) / 2;
  if (ngem_act_ll > std::numeric_limits<int>::max()) {
    cublasDestroy(handle);
    return 302;
  }
  const int ngem_act = static_cast<int>(ngem_act_ll);
  const int q_chunk = choose_fused_gradient_q_chunk(
      device, nmo, ndoc, nact, q_end - q_begin, ngem, ngem_act,
      requested_q_chunk, slice_is_fully_resident(resident, q_end),
      resident == nullptr ? 0 : resident->workspace_bytes);

  const long long ldx_doc_ll = static_cast<long long>(q_chunk) * ndoc;
  const long long ldx_act_ll = static_cast<long long>(q_chunk) * nact;
  if (ldx_doc_ll > std::numeric_limits<int>::max() ||
      ldx_act_ll > std::numeric_limits<int>::max()) {
    cublasDestroy(handle);
    return 303;
  }

  const std::size_t x_doc_bytes =
      static_cast<std::size_t>(ldx_doc_ll) * nmo * sizeof(double);
  const std::size_t x_act_bytes =
      static_cast<std::size_t>(ldx_act_ll) * nmo * sizeof(double);
  const std::size_t c_bytes =
      static_cast<std::size_t>(nmo) * nmo * sizeof(double);
  const std::size_t pair_bytes =
      static_cast<std::size_t>(ngem) * sizeof(double);
  const std::size_t qvec_bytes =
      static_cast<std::size_t>(q_chunk) * sizeof(double);
  const std::size_t den1_bytes =
      static_cast<std::size_t>(ngem_act) * sizeof(double);
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
  const std::size_t int2_bytes =
      static_cast<std::size_t>(q_chunk) * ngem * sizeof(double);
  const std::size_t stage_bytes =
      streaming_stage_bytes(resident, q_end, q_chunk, ngem);
  const std::size_t pooled_bytes =
      x_doc_bytes + 2 * x_act_bytes + 2 * c_bytes + 2 * pair_bytes +
      2 * qvec_bytes + den1_bytes + 2 * active_bytes + b_bytes + d2_bytes +
      result_bytes + q_bytes + stage_bytes;

  double *d_int2 = nullptr;
  double *d_stage = nullptr;
  double *d_x_doc = nullptr;
  double *d_x_act = nullptr;
  double *d_y_act = nullptr;
  double *d_c_fi = nullptr;
  double *d_c_fa = nullptr;
  double *d_pairs_fi = nullptr;
  double *d_pairs_fa = nullptr;
  double *d_qvec_fi = nullptr;
  double *d_qvec_fa = nullptr;
  double *d_den1 = nullptr;
  double *d_active = nullptr;
  double *d_qint = nullptr;
  double *d_b = nullptr;
  double *d_d2 = nullptr;
  double *d_result = nullptr;
  double *d_q = nullptr;
  bool pooled_workspace = false;

  auto cleanup = [&]() {
    if (!pooled_workspace) {
      cudaFree(d_q);
      cudaFree(d_result);
      cudaFree(d_d2);
      cudaFree(d_b);
      cudaFree(d_qint);
      cudaFree(d_active);
      cudaFree(d_den1);
      cudaFree(d_qvec_fa);
      cudaFree(d_qvec_fi);
      cudaFree(d_pairs_fa);
      cudaFree(d_pairs_fi);
      cudaFree(d_c_fa);
      cudaFree(d_c_fi);
      cudaFree(d_y_act);
      cudaFree(d_x_act);
      cudaFree(d_x_doc);
    }
    if (d_int2 != nullptr && resident == nullptr) {
      cudaFree(d_int2);
    }
    cublasDestroy(handle);
  };

  if (resident != nullptr) {
    d_int2 = resident->d_int2;
    pooled_workspace = ensure_session_workspace(resident, pooled_bytes);
    if (pooled_workspace) {
      unsigned char *cursor = resident->workspace;
      d_x_doc = take_workspace<double>(cursor, x_doc_bytes / sizeof(double));
      d_x_act = take_workspace<double>(cursor, x_act_bytes / sizeof(double));
      d_y_act = take_workspace<double>(cursor, x_act_bytes / sizeof(double));
      d_c_fi = take_workspace<double>(cursor, c_bytes / sizeof(double));
      d_c_fa = take_workspace<double>(cursor, c_bytes / sizeof(double));
      d_pairs_fi = take_workspace<double>(cursor, pair_bytes / sizeof(double));
      d_pairs_fa = take_workspace<double>(cursor, pair_bytes / sizeof(double));
      d_qvec_fi = take_workspace<double>(cursor, qvec_bytes / sizeof(double));
      d_qvec_fa = take_workspace<double>(cursor, qvec_bytes / sizeof(double));
      d_den1 = take_workspace<double>(cursor, den1_bytes / sizeof(double));
      d_active = take_workspace<double>(cursor, active_bytes / sizeof(double));
      d_qint = take_workspace<double>(cursor, active_bytes / sizeof(double));
      d_b = take_workspace<double>(cursor, b_bytes / sizeof(double));
      d_d2 = take_workspace<double>(cursor, d2_bytes / sizeof(double));
      d_result = take_workspace<double>(cursor, result_bytes / sizeof(double));
      d_q = take_workspace<double>(cursor, q_bytes / sizeof(double));
      if (stage_bytes > 0) {
        d_stage = take_workspace<double>(cursor, stage_bytes / sizeof(double));
      }
    }
  }
  if ((resident != nullptr && !pooled_workspace) ||
      (resident == nullptr &&
       (cudaMalloc(&d_int2, int2_bytes) != cudaSuccess ||
        cudaMalloc(&d_x_doc, x_doc_bytes) != cudaSuccess ||
        cudaMalloc(&d_x_act, x_act_bytes) != cudaSuccess ||
        cudaMalloc(&d_y_act, x_act_bytes) != cudaSuccess ||
        cudaMalloc(&d_c_fi, c_bytes) != cudaSuccess ||
        cudaMalloc(&d_c_fa, c_bytes) != cudaSuccess ||
        cudaMalloc(&d_pairs_fi, pair_bytes) != cudaSuccess ||
        cudaMalloc(&d_pairs_fa, pair_bytes) != cudaSuccess ||
        cudaMalloc(&d_qvec_fi, qvec_bytes) != cudaSuccess ||
        cudaMalloc(&d_qvec_fa, qvec_bytes) != cudaSuccess ||
        cudaMalloc(&d_den1, den1_bytes) != cudaSuccess ||
        cudaMalloc(&d_active, active_bytes) != cudaSuccess ||
        cudaMalloc(&d_qint, active_bytes) != cudaSuccess ||
        cudaMalloc(&d_b, b_bytes) != cudaSuccess ||
        cudaMalloc(&d_d2, d2_bytes) != cudaSuccess ||
        cudaMalloc(&d_result, result_bytes) != cudaSuccess ||
        cudaMalloc(&d_q, q_bytes) != cudaSuccess))) {
    cleanup();
    return 304;
  }

  if (cudaMemset(d_c_fi, 0, c_bytes) != cudaSuccess ||
      cudaMemset(d_c_fa, 0, c_bytes) != cudaSuccess ||
      cudaMemset(d_pairs_fi, 0, pair_bytes) != cudaSuccess ||
      cudaMemset(d_pairs_fa, 0, pair_bytes) != cudaSuccess ||
      cudaMemset(d_q, 0, q_bytes) != cudaSuccess) {
    cleanup();
    return 305;
  }
  if (cudaMemcpy(d_den1, den1_act, den1_bytes, cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemcpy(d_d2, scaled_d2.data(), d2_bytes, cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    cleanup();
    return 306;
  }

  constexpr int threads = 256;
  const double one = 1.0;
  const double zero = 0.0;

  for (long long q = q_begin; q < q_end;) {
    int bq = static_cast<int>(std::min<long long>(q_chunk, q_end - q));
    bq = clamp_tile_to_residency(resident, q, bq);
    const bool tile_resident =
        resident != nullptr && resident->tile_resident(q, bq);
    const std::size_t int2_bq_bytes =
        static_cast<std::size_t>(bq) * ngem * sizeof(double);
    double *const stage_block = resident == nullptr ? d_int2 : d_stage;
    const double *device_block =
        tile_resident ? d_int2 + (q - resident->q_begin) * ngem : stage_block;

    // The single upload of this tile; everything below reuses it.
    if (!tile_resident) {
      if (upload_stream_tile(device, stage_block, int2 + q * ngem,
                             int2_bq_bytes) != cudaSuccess) {
        cleanup();
        return 307;
      }
    }

    // --- Fi exchange -------------------------------------------------------
    if (ndoc > 0) {
      const int ldx = bq * ndoc;
      const long long total = static_cast<long long>(ldx) * nmo;
      build_pair_column_matrix_kernel<<<cuda_blocks(total, threads), threads>>>(
          device_block, d_x_doc, nmo, 0, ndoc, 0, bq, ngem, ldx);
      if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return 308;
      }
      if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nmo, nmo, ldx, &one,
                      d_x_doc, ldx, d_x_doc, ldx, &one, d_c_fi,
                      nmo) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return 309;
      }
    }

    // --- Fa exchange -------------------------------------------------------
    if (nact > 0) {
      const int ldx = bq * nact;
      const long long total = static_cast<long long>(ldx) * nmo;
      build_pair_column_matrix_kernel<<<cuda_blocks(total, threads), threads>>>(
          device_block, d_x_act, nmo, ndoc, nact, 0, bq, ngem, ldx);
      if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return 310;
      }
      apply_active_density_kernel<<<cuda_blocks(total, threads), threads>>>(
          d_x_act, d_den1, d_y_act, nmo, nact, bq, ldx);
      if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return 311;
      }
      if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nmo, nmo, ldx, &one,
                      d_x_act, ldx, d_y_act, ldx, &one, d_c_fa,
                      nmo) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return 312;
      }
    }

    // --- Fi and Fa Coulomb -------------------------------------------------
    if (ndoc > 0) {
      build_fi_coulomb_vector_kernel<<<cuda_blocks(bq, threads), threads>>>(
          device_block, d_qvec_fi, ndoc, bq, ngem);
      if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return 313;
      }
      if (cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(ngem), bq, &one,
                      device_block, static_cast<int>(ngem), d_qvec_fi, 1, &one,
                      d_pairs_fi, 1) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return 314;
      }
    }
    if (nact > 0) {
      build_fa_coulomb_vector_kernel<<<cuda_blocks(bq, threads), threads>>>(
          device_block, d_den1, d_qvec_fa, ndoc, nact, bq, ngem);
      if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return 315;
      }
      if (cublasDgemv(handle, CUBLAS_OP_N, static_cast<int>(ngem), bq, &one,
                      device_block, static_cast<int>(ngem), d_qvec_fa, 1, &one,
                      d_pairs_fa, 1) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return 316;
      }
    }

    // --- Q contraction -----------------------------------------------------
    if (nact > 0) {
      const long long active_total = static_cast<long long>(bq) * ngem_act;
      build_active_pair_matrix_kernel<<<cuda_blocks(active_total, threads),
                                        threads>>>(device_block, d_active, ndoc,
                                                   nact, bq, ngem, bq, nullptr);
      if (cudaGetLastError() != cudaSuccess) {
        cleanup();
        return 317;
      }
      if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, bq, ngem_act, ngem_act,
                      &one, d_active, bq, d_d2, ngem_act, &zero, d_qint,
                      bq) != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return 318;
      }
      const long long b_total = static_cast<long long>(bq) * nmo;
      for (int u = 0; u < nact; ++u) {
        build_general_active_matrix_kernel<<<cuda_blocks(b_total, threads),
                                             threads>>>(
            device_block, d_b, nmo, ndoc, u, bq, ngem, bq, nullptr);
        if (cudaGetLastError() != cudaSuccess) {
          cleanup();
          return 319;
        }
        if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nmo, ngem_act, bq,
                        &one, d_b, bq, d_qint, bq, &zero, d_result,
                        nmo) != CUBLAS_STATUS_SUCCESS) {
          cleanup();
          return 320;
        }
        scatter_q_result_kernel<<<
            cuda_blocks(static_cast<long long>(nact) * nmo, threads),
            threads>>>(d_result, d_q, nmo, nact, u);
        if (cudaGetLastError() != cudaSuccess) {
          cleanup();
          return 321;
        }
      }
    }
    q += bq;
  }

  if (cudaMemcpy(host_c_fi.data(), d_c_fi, c_bytes, cudaMemcpyDeviceToHost) !=
          cudaSuccess ||
      cudaMemcpy(host_c_fa.data(), d_c_fa, c_bytes, cudaMemcpyDeviceToHost) !=
          cudaSuccess ||
      cudaMemcpy(host_pairs_fi.data(), d_pairs_fi, pair_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(host_pairs_fa.data(), d_pairs_fa, pair_bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(host_q.data(), d_q, q_bytes, cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
    cleanup();
    return 322;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cleanup();
    return 323;
  }
  cleanup();
  return 0;
}

} // namespace

extern "C" int hilbert_focas_df_cuda_session_begin(int nmo, long long nQ,
                                                   const double *int2,
                                                   int max_devices) {
  if (nmo <= 0 || nQ <= 0 || int2 == nullptr)
    return 230;
  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  if (ngem <= 0 ||
      static_cast<unsigned long long>(ngem) >
          std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    return 231;
  }
  const int available_devices = device_count_from_request(max_devices);
  if (available_devices <= 0)
    return 232;
  const int devices =
      static_cast<int>(std::min<long long>(available_devices, nQ));

  std::lock_guard<std::mutex> lock(focas_session_mutex);
  clear_focas_session();
  focas_session.active = true;
  focas_session.nmo = nmo;
  focas_session.nQ = nQ;
  focas_session.ngem = ngem;
  focas_session.host_int2 = int2;
  focas_session.devices.resize(devices);

  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share =
        (remaining + remaining_devices - 1) / remaining_devices;
    FocasSessionDevice &entry = focas_session.devices[dev];
    entry.device = dev;
    entry.q_begin = q_cursor;
    entry.q_end = std::min(nQ, q_cursor + share);
    entry.q_res_end = entry.q_begin; // nothing resident until admitted below
    q_cursor = entry.q_end;

    if (cudaSetDevice(dev) != cudaSuccess)
      continue;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
      continue;
    const auto q_count =
        static_cast<unsigned long long>(entry.q_end - entry.q_begin);
    const auto row_bytes =
        static_cast<unsigned long long>(ngem) * sizeof(double);
    if (q_count > std::numeric_limits<std::size_t>::max() / row_bytes)
      continue;
    const std::size_t slice_bytes =
        static_cast<std::size_t>(q_count * row_bytes);

    // Hold back a working reserve, then keep as much of the slice on the device
    // as the remainder allows.  Residency is a prefix, not all-or-nothing: the
    // rows that do not fit are streamed per tile instead of forcing the whole
    // slice back onto the host.  That removes the cliff where one extra Q row
    // turned a fully resident tensor into a fully streamed one.
    const std::size_t reserve = focas_working_reserve_bytes(nmo, free_bytes);
    const std::size_t budget = free_bytes > reserve ? free_bytes - reserve : 0;
    long long rows = static_cast<long long>(budget / row_bytes);
    rows = std::min<long long>(rows, static_cast<long long>(q_count));

    // Back off on allocation failure: free memory can be fragmented enough that
    // a block the size reported by cudaMemGetInfo is not actually obtainable.
    while (rows > 0) {
      const std::size_t bytes =
          static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_bytes);
      if (cudaMalloc(&entry.d_int2, bytes) == cudaSuccess) {
        entry.int2_bytes = bytes;
        break;
      }
      entry.d_int2 = nullptr;
      cudaGetLastError(); // clear the sticky OOM before retrying
      rows /= 2;
    }

    if (rows > 0) {
      const double *host_slice = int2 + entry.q_begin * ngem;
      cudaError_t upload = cudaSuccess;
      if (entry.int2_bytes >= kPinnedResidentUploadThreshold) {
        PinnedStagingLease staging_lease(dev, entry.int2_bytes);
        upload = staging_lease.staging().copy_h2d(entry.d_int2, host_slice,
                                                  entry.int2_bytes);
      } else {
        upload = cudaMemcpy(entry.d_int2, host_slice, entry.int2_bytes,
                            cudaMemcpyHostToDevice);
      }
      if (upload == cudaSuccess) {
        entry.q_res_end = entry.q_begin + rows;
      } else {
        cudaFree(entry.d_int2);
        entry.d_int2 = nullptr;
        entry.int2_bytes = 0;
        rows = 0;
      }
    }

    const double resident_fraction =
        q_count == 0 ? 0.0
                     : static_cast<double>(rows) / static_cast<double>(q_count);
    if (rows == 0) {
      focas_notice("device %d: DF tensor slice %.2f GiB does not fit "
                   "(free %.2f GiB, reserve %.2f GiB); every Q tile will be "
                   "streamed over PCIe",
                   dev, to_gib(slice_bytes), to_gib(free_bytes),
                   to_gib(reserve));
    } else if (rows < static_cast<long long>(q_count)) {
      focas_notice("device %d: DF tensor slice %.2f GiB partially resident "
                   "(%.2f GiB, %.0f%% of Q rows; free %.2f GiB, reserve "
                   "%.2f GiB); the remainder is streamed per tile",
                   dev, to_gib(slice_bytes), to_gib(entry.int2_bytes),
                   100.0 * resident_fraction, to_gib(free_bytes),
                   to_gib(reserve));
    }
  }
  return 0;
}

extern "C" int hilbert_focas_df_cuda_session_end(double *int2, int commit) {
  std::lock_guard<std::mutex> lock(focas_session_mutex);
  int status = 0;
  const bool was_dirty = focas_session.active && focas_session.dirty;
  if (was_dirty && commit != 0) {
    status = copy_resident_slices_to_host_locked(int2);
  } else if (was_dirty) {
    // Refuse to silently discard transformed resident data.  Gradient-only
    // sessions remain clean and may still end without a commit.
    status = 234;
  }
  clear_focas_session();
  return status;
}

extern "C" int hilbert_focas_df_ao_to_mo_cuda_transform(
    int nao, int nmo, long long nQ, const double *qao, double *qmo,
    const double *c_pitzer, int block_q, int max_devices) {
  if (nao <= 0 || nmo <= 0 || nmo > nao || nQ <= 0 || qao == nullptr ||
      qmo == nullptr || c_pitzer == nullptr) {
    return 200;
  }
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return 201;
  }
  if (max_devices > 0)
    device_count = std::min(device_count, max_devices);
  device_count = static_cast<int>(std::min<long long>(device_count, nQ));
  const long long ao_pair = static_cast<long long>(nao) * (nao + 1) / 2;
  const long long mo_pair = static_cast<long long>(nmo) * (nmo + 1) / 2;

  std::vector<std::thread> workers;
  std::vector<int> statuses(device_count, 0);
  long long q_cursor = 0;
  for (int dev = 0; dev < device_count; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = device_count - dev;
    const long long share =
        (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] =
          transform_ao_to_mo_on_device(dev, nao, nmo, q_begin, q_end, ao_pair,
                                       mo_pair, qao, qmo, c_pitzer, block_q);
    });
  }
  for (auto &worker : workers)
    worker.join();
  for (int status : statuses) {
    if (status != 0)
      return status;
  }
  return 0;
}

extern "C" int hilbert_focas_df_c1_cuda_transform(int nmo, long long nQ,
                                                  double *int2, const double *u,
                                                  int block_q,
                                                  int max_devices) {
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

  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  const int devices = std::max(1, device_count);
  std::vector<std::thread> workers;
  std::vector<int> statuses(devices, 0);
  std::atomic<bool> tensor_mutated{false};

  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share =
        (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] = process_range_on_device(
          dev, nmo, q_begin, q_end, ngem, int2, u, block_q, &tensor_mutated);
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  for (int status : statuses) {
    if (status != 0) {
      return handle_transform_failure(
          status, int2, ngem, tensor_mutated.load(std::memory_order_relaxed));
    }
  }
  mark_resident_session_dirty(int2, ngem);
  return 0;
}

extern "C" int hilbert_focas_df_c1_cuda_transform_low_rank(
    int nmo, int rank, long long nQ, double *int2, const double *v,
    const double *a, int block_q, int max_devices) {
  if (nmo <= 0 || rank <= 0 || rank >= nmo || nQ <= 0 || int2 == nullptr ||
      v == nullptr || a == nullptr) {
    return 2;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return 1;
  }
  if (max_devices > 0)
    device_count = std::min(device_count, max_devices);
  const int devices = std::max(1, device_count);

  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  std::vector<std::thread> workers;
  std::vector<int> statuses(devices, 0);
  std::atomic<bool> tensor_mutated{false};
  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share =
        (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] = process_low_rank_range_on_device(
          dev, nmo, rank, q_begin, q_end, ngem, int2, v, a, block_q,
          &tensor_mutated);
    });
  }
  for (auto &worker : workers)
    worker.join();
  for (int status : statuses) {
    if (status != 0) {
      return handle_transform_failure(
          status, int2, ngem, tensor_mutated.load(std::memory_order_relaxed));
    }
  }
  mark_resident_session_dirty(int2, ngem);
  return 0;
}

extern "C" int hilbert_focas_df_c1_cuda_fi_exchange(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    double *fock_occ, double *fock_ext, int q_chunk, int max_devices) {
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
      nmo, nQ, max_devices,
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_fi_exchange_on_device(dev, nmo, ndoc, nQ, q_begin, q_end,
                                             ngem, int2, nullptr, q_chunk,
                                             partial);
      },
      c_total);
  if (status != 0) {
    return recover_session_for_host_fallback(status, int2);
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
    double *c_out, int q_chunk, int max_devices) {
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
      nmo, nQ, max_devices,
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_fi_exchange_on_device(dev, nmo, ndoc, nQ, q_begin, q_end,
                                             ngem, int2, doc_df, q_chunk,
                                             partial);
      },
      c_total);
  if (status != 0) {
    return recover_session_for_host_fallback(status, int2);
  }
  std::copy(c_total.begin(), c_total.end(), c_out);
  return 0;
}

extern "C" int
hilbert_focas_df_c1_cuda_fa_exchange(int nmo, int ndoc, int nact, long long nQ,
                                     const double *int2, const double *den1,
                                     double *fock_occ, double *fock_ext,
                                     int q_chunk, int max_devices) {
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
      nmo, nQ, max_devices,
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_fa_exchange_on_device(dev, nmo, ndoc, nact, nQ, q_begin,
                                             q_end, ngem, int2, den1, nullptr,
                                             q_chunk, partial);
      },
      c_total);
  if (status != 0) {
    return recover_session_for_host_fallback(status, int2);
  }
  scatter_c1_exchange(nmo, ndoc, nact, -0.5, c_total, fock_occ, fock_ext);
  return 0;
}

// Symmetry-general Fa exchange. den1 is the symmetry-blocked, packed active
// 1-RDM (local active-pair packing); act_df is the df-order active list.
// Returns the dense nmo x nmo matrix C(p_df,q_df) = sum_{tu} (p_df t | q_df u)
// D1(t,u); the caller scatters it into the per-irrep Fa blocks with the -0.5
// factor.
extern "C" int
hilbert_focas_df_sym_cuda_fa_exchange(int nmo, int ndoc, int nact, long long nQ,
                                      const double *int2, const double *den1,
                                      const int *act_df, double *c_out,
                                      int q_chunk, int max_devices) {
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
      nmo, nQ, max_devices,
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_fa_exchange_on_device(dev, nmo, ndoc, nact, nQ, q_begin,
                                             q_end, ngem, int2, den1, act_df,
                                             q_chunk, partial);
      },
      c_total);
  if (status != 0) {
    return recover_session_for_host_fallback(status, int2);
  }
  std::copy(c_total.begin(), c_total.end(), c_out);
  return 0;
}

extern "C" int
hilbert_focas_df_c1_cuda_fi_coulomb(int nmo, int ndoc, int nact, long long nQ,
                                    const double *int1, const double *int2,
                                    double *fock_occ, double *fock_ext,
                                    int q_chunk, int max_devices) {
  if (nmo <= 0 || ndoc < 0 || nact < 0 || nQ <= 0 || int1 == nullptr ||
      int2 == nullptr || fock_occ == nullptr || fock_ext == nullptr ||
      ndoc + nact > nmo) {
    return 120;
  }
  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  std::vector<double> qvec(static_cast<std::size_t>(nQ), 0.0);
  if (!session_is_fully_resident(int2, ngem, nQ)) {
    build_fi_coulomb_vector(ndoc, nQ, ngem, int2, qvec);
  }
  std::vector<double> pair_total(static_cast<std::size_t>(ngem), 0.0);
  const int status = run_pair_workers(
      nQ, ngem, max_devices,
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_coulomb_on_device(dev, q_begin, q_end, ngem, int2,
                                         qvec.data(), nullptr, ndoc, nact,
                                         q_chunk, "Fi Coulomb", partial);
      },
      pair_total);
  if (status != 0) {
    return recover_session_for_host_fallback(status, int2);
  }
  scatter_c1_coulomb(nmo, ndoc, nact, pair_total, int1, fock_occ, fock_ext);
  return 0;
}

extern "C" int
hilbert_focas_df_c1_cuda_fa_coulomb(int nmo, int ndoc, int nact, long long nQ,
                                    const double *int2, const double *den1,
                                    double *fock_occ, double *fock_ext,
                                    int q_chunk, int max_devices) {
  if (nmo <= 0 || ndoc < 0 || nact < 0 || nQ <= 0 || int2 == nullptr ||
      den1 == nullptr || fock_occ == nullptr || fock_ext == nullptr ||
      ndoc + nact > nmo) {
    return 130;
  }
  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  std::vector<double> qvec(static_cast<std::size_t>(nQ), 0.0);
  if (!session_is_fully_resident(int2, ngem, nQ)) {
    build_fa_coulomb_vector(ndoc, nact, nQ, ngem, int2, den1, qvec);
  }
  std::vector<double> pair_total(static_cast<std::size_t>(ngem), 0.0);
  const int status = run_pair_workers(
      nQ, ngem, max_devices,
      [&](int dev, long long q_begin, long long q_end,
          std::vector<double> &partial) {
        return compute_coulomb_on_device(dev, q_begin, q_end, ngem, int2,
                                         qvec.data(), den1, ndoc, nact, q_chunk,
                                         "Fa Coulomb", partial);
      },
      pair_total);
  if (status != 0) {
    return recover_session_for_host_fallback(status, int2);
  }
  scatter_c1_coulomb(nmo, ndoc, nact, pair_total, nullptr, fock_occ, fock_ext);
  return 0;
}

extern "C" int hilbert_focas_df_c1_cuda_q(int nmo, int ndoc, int nact,
                                          long long nQ, const double *int2,
                                          const double *den2, double *q_out,
                                          int q_chunk, int max_devices) {
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
    const long long share =
        (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] =
          compute_q_on_device(dev, nmo, ndoc, nact, nQ, q_begin, q_end, ngem,
                              int2, scaled_d2, nullptr, q_chunk, partials[dev]);
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }
  for (int status : statuses) {
    if (status != 0) {
      return recover_session_for_host_fallback(status, int2);
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
extern "C" int hilbert_focas_df_sym_cuda_q(int nmo, int ndoc, int nact,
                                           long long nQ, const double *int2,
                                           const double *scaled_d2_in,
                                           const int *act_df, double *q_out,
                                           int q_chunk, int max_devices) {
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
    const long long share =
        (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] =
          compute_q_on_device(dev, nmo, ndoc, nact, nQ, q_begin, q_end, ngem,
                              int2, scaled_d2, act_df, q_chunk, partials[dev]);
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }
  for (int status : statuses) {
    if (status != 0) {
      return recover_session_for_host_fallback(status, int2);
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

// Fused C1 gradient: computes the inactive and active Fock matrices and the
// auxiliary Q matrix in a single sweep of the DF tensor, replacing the five
// separate operator calls.  Semantics are identical to running
// hilbert_focas_df_c1_cuda_{fi_coulomb,fi_exchange,fa_coulomb,fa_exchange,q}
// in that order -- the Coulomb scatters assign and the exchange scatters
// accumulate, so the order below is load-bearing.
extern "C" int hilbert_focas_df_c1_cuda_gradient_all(
    int nmo, int ndoc, int nact, long long nQ, const double *int1,
    const double *int2, const double *den1, const double *den2,
    double *fock_i_occ, double *fock_i_ext, double *fock_a_occ,
    double *fock_a_ext, double *q_out, int q_chunk, int max_devices) {
  if (nmo <= 0 || ndoc < 0 || nact < 0 || nQ <= 0 || int1 == nullptr ||
      int2 == nullptr || den1 == nullptr || den2 == nullptr ||
      fock_i_occ == nullptr || fock_i_ext == nullptr ||
      fock_a_occ == nullptr || fock_a_ext == nullptr || q_out == nullptr ||
      ndoc + nact > nmo) {
    return 340;
  }
  const int devices = device_count_from_request(max_devices);
  if (devices <= 0) {
    return 341;
  }

  const long long ngem = static_cast<long long>(nmo) * (nmo + 1) / 2;
  const long long ngem_act = static_cast<long long>(nact) * (nact + 1) / 2;
  const std::size_t c_size = static_cast<std::size_t>(nmo) * nmo;
  const std::size_t pair_size = static_cast<std::size_t>(ngem);
  const std::size_t q_size = static_cast<std::size_t>(nact) * nmo;

  std::vector<double> scaled_d2(
      static_cast<std::size_t>(ngem_act) * ngem_act, 0.0);
  if (nact > 0) {
    build_scaled_c1_d2(nact, den2, scaled_d2);
  }

  std::vector<std::thread> workers;
  std::vector<int> statuses(devices, 0);
  std::vector<std::vector<double>> part_c_fi(devices,
                                             std::vector<double>(c_size, 0.0));
  std::vector<std::vector<double>> part_c_fa(devices,
                                             std::vector<double>(c_size, 0.0));
  std::vector<std::vector<double>> part_p_fi(
      devices, std::vector<double>(pair_size, 0.0));
  std::vector<std::vector<double>> part_p_fa(
      devices, std::vector<double>(pair_size, 0.0));
  std::vector<std::vector<double>> part_q(devices,
                                          std::vector<double>(q_size, 0.0));

  long long q_cursor = 0;
  for (int dev = 0; dev < devices; ++dev) {
    const long long remaining = nQ - q_cursor;
    const int remaining_devices = devices - dev;
    const long long share =
        (remaining + remaining_devices - 1) / remaining_devices;
    const long long q_begin = q_cursor;
    const long long q_end = std::min(nQ, q_begin + share);
    q_cursor = q_end;
    workers.emplace_back([&, dev, q_begin, q_end]() {
      statuses[dev] = compute_gradient_all_on_device(
          dev, nmo, ndoc, nact, q_begin, q_end, ngem, int2, den1, scaled_d2,
          q_chunk, part_c_fi[dev], part_c_fa[dev], part_p_fi[dev],
          part_p_fa[dev], part_q[dev]);
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  for (int status : statuses) {
    if (status != 0) {
      return recover_session_for_host_fallback(status, int2);
    }
  }

  std::vector<double> c_fi(c_size, 0.0), c_fa(c_size, 0.0);
  std::vector<double> p_fi(pair_size, 0.0), p_fa(pair_size, 0.0);
  std::vector<double> q_total(q_size, 0.0);
  for (int dev = 0; dev < devices; ++dev) {
    for (std::size_t i = 0; i < c_size; ++i) {
      c_fi[i] += part_c_fi[dev][i];
      c_fa[i] += part_c_fa[dev][i];
    }
    for (std::size_t i = 0; i < pair_size; ++i) {
      p_fi[i] += part_p_fi[dev][i];
      p_fa[i] += part_p_fa[dev][i];
    }
    for (std::size_t i = 0; i < q_size; ++i) {
      q_total[i] += part_q[dev][i];
    }
  }

  // Coulomb assigns, exchange accumulates: keep this order.
  scatter_c1_coulomb(nmo, ndoc, nact, p_fi, int1, fock_i_occ, fock_i_ext);
  if (ndoc > 0) {
    scatter_c1_exchange(nmo, ndoc, nact, -1.0, c_fi, fock_i_occ, fock_i_ext);
  }
  scatter_c1_coulomb(nmo, ndoc, nact, p_fa, nullptr, fock_a_occ, fock_a_ext);
  if (nact > 0) {
    scatter_c1_exchange(nmo, ndoc, nact, -0.5, c_fa, fock_a_occ, fock_a_ext);
  }
  for (std::size_t i = 0; i < q_size; ++i) {
    q_out[i] = q_total[i];
  }
  return 0;
}
