#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

using TransformFn = int (*)(int, long long, double *, const double *, int, int);
using LowRankTransformFn = int (*)(int, int, long long, double *,
                                   const double *, const double *, int, int);
using SessionBeginFn = int (*)(int, long long, const double *, int);
using SessionEndFn = int (*)(double *, int);
using AoToMoFn = int (*)(int, int, long long, const double *, double *,
                         const double *, int, int);
using FiExchangeFn = int (*)(int, int, int, long long, const double *, double *,
                             double *, int, int);
using SymFiExchangeFn = int (*)(int, int, long long, const double *,
                                const int *, double *, int, int);
using SymFaExchangeFn = int (*)(int, int, int, long long, const double *,
                                const double *, const int *, double *, int,
                                int);
using FaExchangeFn = int (*)(int, int, int, long long, const double *,
                             const double *, double *, double *, int, int);
using FiCoulombFn = int (*)(int, int, int, long long, const double *,
                            const double *, double *, double *, int, int);
using FaCoulombFn = int (*)(int, int, int, long long, const double *,
                            const double *, double *, double *, int, int);
using QFn = int (*)(int, int, int, long long, const double *, const double *,
                    double *, int, int);
using SymQFn = int (*)(int, int, int, long long, const double *,
                       const double *, const int *, double *, int, int);
// Fused C1 gradient: Fi/Fa Coulomb, Fi/Fa exchange and Q in one sweep.
using GradientAllFn = int (*)(int, int, int, long long, const double *,
                              const double *, const double *, const double *,
                              double *, double *, double *, double *, double *,
                              int, int);

std::mutex load_mutex;
void *jit_handle = nullptr;
TransformFn jit_transform = nullptr;
LowRankTransformFn jit_low_rank_transform = nullptr;
SessionBeginFn jit_session_begin = nullptr;
SessionEndFn jit_session_end = nullptr;
AoToMoFn jit_ao_to_mo = nullptr;
FiExchangeFn jit_fi_exchange = nullptr;
SymFiExchangeFn jit_sym_fi_exchange = nullptr;
SymFaExchangeFn jit_sym_fa_exchange = nullptr;
FaExchangeFn jit_fa_exchange = nullptr;
FiCoulombFn jit_fi_coulomb = nullptr;
FaCoulombFn jit_fa_coulomb = nullptr;
QFn jit_q = nullptr;
SymQFn jit_sym_q = nullptr;
// Optional: absent when hilbert.so is newer than the .cu it JIT-compiles.
// A null pointer degrades to the five-operator path instead of disabling the
// whole CUDA FOCAS layer.
GradientAllFn jit_gradient_all = nullptr;
int jit_load_status = 0;

bool file_exists(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool dir_exists(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string dirname(const std::string &path) {
  const std::size_t pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return ".";
  }
  if (pos == 0) {
    return "/";
  }
  return path.substr(0, pos);
}

std::string join_path(const std::string &a, const std::string &b) {
  if (a.empty()) {
    return b;
  }
  if (a.back() == '/') {
    return a + b;
  }
  return a + "/" + b;
}

bool mkdir_p(const std::string &path) {
  if (path.empty() || dir_exists(path)) {
    return true;
  }
  std::string current;
  std::size_t pos = 0;
  if (path[0] == '/') {
    current = "/";
    pos = 1;
  }
  while (pos <= path.size()) {
    std::size_t next = path.find('/', pos);
    std::string part = path.substr(pos, next == std::string::npos ? next : next - pos);
    if (!part.empty()) {
      current = join_path(current, part);
      if (!dir_exists(current) && mkdir(current.c_str(), 0755) != 0 &&
          !dir_exists(current)) {
        return false;
      }
    }
    if (next == std::string::npos) {
      break;
    }
    pos = next + 1;
  }
  return true;
}

std::string read_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::vector<std::string> split_paths(const char *value) {
  std::vector<std::string> out;
  if (value == nullptr) {
    return out;
  }
  std::string text(value);
  std::size_t start = 0;
  while (start <= text.size()) {
    std::size_t end = text.find(':', start);
    std::string item = text.substr(start, end == std::string::npos ? end : end - start);
    if (!item.empty()) {
      out.push_back(item);
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return out;
}

std::vector<std::string> split_list(const std::string &text) {
  std::vector<std::string> out;
  std::string token;
  for (char ch : text) {
    if (ch == ';' || ch == ',' || ch == ' ' || ch == '\t' || ch == '\n') {
      if (!token.empty()) {
        out.push_back(token);
        token.clear();
      }
    } else {
      token.push_back(ch);
    }
  }
  if (!token.empty()) {
    out.push_back(token);
  }
  return out;
}

std::string shell_quote(const std::string &text) {
  std::string out = "'";
  for (char ch : text) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out.push_back(ch);
    }
  }
  out += "'";
  return out;
}

std::string env_or_empty(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

std::string find_source() {
  const std::string source_name = "focas_cuda_bridge.cu";
  std::vector<std::string> candidates;

  const std::string explicit_source =
      env_or_empty("HILBERT_FOCAS_CUDA_SOURCE");
  if (!explicit_source.empty()) {
    candidates.push_back(explicit_source);
  }

  candidates.push_back(join_path(dirname(__FILE__), source_name));

  char cwd_buffer[4096];
  if (getcwd(cwd_buffer, sizeof(cwd_buffer)) != nullptr) {
    candidates.push_back(join_path(join_path(cwd_buffer, "src/focas"), source_name));
  }

  for (const std::string &entry : split_paths(std::getenv("PYTHONPATH"))) {
    candidates.push_back(join_path(join_path(entry, "src/focas"), source_name));
    candidates.push_back(join_path(entry, source_name));
  }

  for (const std::string &path : candidates) {
    if (file_exists(path)) {
      return path;
    }
  }
  return std::string();
}

std::string find_nvcc() {
  for (const char *name : {"CUDACXX", "NVCC"}) {
    std::string value = env_or_empty(name);
    if (!value.empty()) {
      return value;
    }
  }
  for (const char *name : {"CUDA_HOME", "CUDA_PATH"}) {
    std::string value = env_or_empty(name);
    if (!value.empty()) {
      return join_path(join_path(value, "bin"), "nvcc");
    }
  }
  return "nvcc";
}

std::string cache_dir() {
  std::string value = env_or_empty("TORCH_EXTENSIONS_DIR");
  if (!value.empty()) {
    return join_path(value, "hilbert_focas_cuda");
  }
  value = env_or_empty("HOME");
  if (!value.empty()) {
    return join_path(value, ".cache/torch_extensions/hilbert_focas_cuda");
  }
  value = env_or_empty("TMPDIR");
  if (value.empty()) {
    value = "/tmp";
  }
  std::string user = env_or_empty("USER");
  if (user.empty()) {
    user = "unknown";
  }
  return join_path(value, "hilbert_focas_cuda_" + user);
}

std::string cuda_arch_flags() {
  std::string arch_list = env_or_empty("TORCH_CUDA_ARCH_LIST");
  std::ostringstream flags;
  for (std::string token : split_list(arch_list)) {
    bool ptx = false;
    const std::string suffix = "+PTX";
    if (token.size() >= suffix.size() &&
        token.substr(token.size() - suffix.size()) == suffix) {
      ptx = true;
      token = token.substr(0, token.size() - suffix.size());
    }
    std::string digits;
    for (char ch : token) {
      if (ch >= '0' && ch <= '9') {
        digits.push_back(ch);
      }
    }
    if (digits.size() == 1) {
      digits.push_back('0');
    }
    if (digits.empty()) {
      continue;
    }
    flags << " -gencode=arch=compute_" << digits << ",code=sm_" << digits;
    if (ptx) {
      flags << " -gencode=arch=compute_" << digits << ",code=compute_" << digits;
    }
  }
  return flags.str();
}

std::string fnv1a_hex(const std::string &text) {
  std::uint64_t hash = 1469598103934665603ull;
  for (unsigned char ch : text) {
    hash ^= static_cast<std::uint64_t>(ch);
    hash *= 1099511628211ull;
  }
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

int ensure_loaded() {
  std::lock_guard<std::mutex> lock(load_mutex);
  if (jit_transform != nullptr && jit_low_rank_transform != nullptr &&
      jit_session_begin != nullptr &&
      jit_session_end != nullptr && jit_ao_to_mo != nullptr &&
      jit_fi_exchange != nullptr &&
      jit_sym_fi_exchange != nullptr && jit_fa_exchange != nullptr &&
      jit_fi_coulomb != nullptr && jit_fa_coulomb != nullptr &&
      jit_q != nullptr && jit_sym_q != nullptr &&
      jit_sym_fa_exchange != nullptr) {
    return 0;
  }
  if (jit_load_status != 0) {
    return jit_load_status;
  }

  const std::string source = find_source();
  if (source.empty()) {
    jit_load_status = 101;
    return jit_load_status;
  }

  const std::string source_text = read_file(source);
  const std::string arch_flags = cuda_arch_flags();
  const std::string key = source_text + "\n" + arch_flags +
                          "\nhilbert_focas_cuda_jit_v14";
  const std::string dir = cache_dir();
  if (!mkdir_p(dir)) {
    jit_load_status = 102;
    return jit_load_status;
  }

  const std::string hash = fnv1a_hex(key);
  const std::string library = join_path(dir, "libhilbert_focas_cuda_" + hash + ".so");
  const std::string tmp_library =
      library + ".tmp." + std::to_string(static_cast<long long>(getpid()));

  if (!file_exists(library)) {
    const std::string nvcc = find_nvcc();
    std::ostringstream cmd;
    cmd << shell_quote(nvcc)
        << " -O3 -std=c++17 --expt-relaxed-constexpr -Xcompiler -fPIC"
        << " -Xcompiler -pthread -shared" << arch_flags;
    cmd << " " << shell_quote(source) << " -o " << shell_quote(tmp_library)
        << " -lcublas -lcudart -lpthread -ldl";
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
      jit_load_status = 103;
      return jit_load_status;
    }
    if (rename(tmp_library.c_str(), library.c_str()) != 0 && !file_exists(library)) {
      jit_load_status = 104;
      return jit_load_status;
    }
  }

  jit_handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (jit_handle == nullptr) {
    jit_load_status = 105;
    return jit_load_status;
  }

  void *symbol_transform =
      dlsym(jit_handle, "hilbert_focas_df_c1_cuda_transform");
  void *symbol_low_rank_transform =
      dlsym(jit_handle, "hilbert_focas_df_c1_cuda_transform_low_rank");
  void *symbol_session_begin =
      dlsym(jit_handle, "hilbert_focas_df_cuda_session_begin");
  void *symbol_session_end =
      dlsym(jit_handle, "hilbert_focas_df_cuda_session_end");
  void *symbol_ao_to_mo =
      dlsym(jit_handle, "hilbert_focas_df_ao_to_mo_cuda_transform");
  void *symbol_fi =
      dlsym(jit_handle, "hilbert_focas_df_c1_cuda_fi_exchange");
  void *symbol_sym_fi =
      dlsym(jit_handle, "hilbert_focas_df_sym_cuda_fi_exchange");
  void *symbol_fa =
      dlsym(jit_handle, "hilbert_focas_df_c1_cuda_fa_exchange");
  void *symbol_fi_coulomb =
      dlsym(jit_handle, "hilbert_focas_df_c1_cuda_fi_coulomb");
  void *symbol_fa_coulomb =
      dlsym(jit_handle, "hilbert_focas_df_c1_cuda_fa_coulomb");
  void *symbol_q = dlsym(jit_handle, "hilbert_focas_df_c1_cuda_q");
  void *symbol_sym_q = dlsym(jit_handle, "hilbert_focas_df_sym_cuda_q");
  void *symbol_sym_fa =
      dlsym(jit_handle, "hilbert_focas_df_sym_cuda_fa_exchange");
  void *symbol_gradient_all =
      dlsym(jit_handle, "hilbert_focas_df_c1_cuda_gradient_all");
  if (symbol_transform == nullptr || symbol_low_rank_transform == nullptr ||
      symbol_session_begin == nullptr ||
      symbol_session_end == nullptr || symbol_ao_to_mo == nullptr ||
      symbol_fi == nullptr ||
      symbol_sym_fi == nullptr || symbol_fa == nullptr ||
      symbol_fi_coulomb == nullptr || symbol_fa_coulomb == nullptr ||
      symbol_q == nullptr || symbol_sym_q == nullptr ||
      symbol_sym_fa == nullptr) {
    jit_load_status = 106;
    return jit_load_status;
  }
  jit_transform = reinterpret_cast<TransformFn>(symbol_transform);
  jit_low_rank_transform =
      reinterpret_cast<LowRankTransformFn>(symbol_low_rank_transform);
  jit_session_begin = reinterpret_cast<SessionBeginFn>(symbol_session_begin);
  jit_session_end = reinterpret_cast<SessionEndFn>(symbol_session_end);
  jit_ao_to_mo = reinterpret_cast<AoToMoFn>(symbol_ao_to_mo);
  jit_fi_exchange = reinterpret_cast<FiExchangeFn>(symbol_fi);
  jit_sym_fi_exchange = reinterpret_cast<SymFiExchangeFn>(symbol_sym_fi);
  jit_fa_exchange = reinterpret_cast<FaExchangeFn>(symbol_fa);
  jit_fi_coulomb = reinterpret_cast<FiCoulombFn>(symbol_fi_coulomb);
  jit_fa_coulomb = reinterpret_cast<FaCoulombFn>(symbol_fa_coulomb);
  jit_q = reinterpret_cast<QFn>(symbol_q);
  jit_sym_q = reinterpret_cast<SymQFn>(symbol_sym_q);
  jit_sym_fa_exchange = reinterpret_cast<SymFaExchangeFn>(symbol_sym_fa);
  jit_gradient_all = reinterpret_cast<GradientAllFn>(symbol_gradient_all);
  return 0;
}

template <typename Function>
int dispatch(Function function) {
  const int load_status = ensure_loaded();
  return load_status == 0 ? function() : load_status;
}

}  // namespace

extern "C" int hilbert_focas_df_cuda_session_begin(
    int nmo, long long nQ, const double *int2, int max_devices) {
  return dispatch(
      [&]() { return jit_session_begin(nmo, nQ, int2, max_devices); });
}

extern "C" int hilbert_focas_df_cuda_session_end(double *int2, int commit) {
  return dispatch([&]() { return jit_session_end(int2, commit); });
}

extern "C" int hilbert_focas_df_ao_to_mo_cuda_transform(
    int nao, int nmo, long long nQ, const double *qao, double *qmo,
    const double *c_pitzer, int block_q, int max_devices) {
  return dispatch([&]() {
    return jit_ao_to_mo(nao, nmo, nQ, qao, qmo, c_pitzer, block_q,
                        max_devices);
  });
}

extern "C" int hilbert_focas_df_c1_cuda_transform(
    int nmo, long long nQ, double *int2, const double *u, int block_q,
    int max_devices) {
  return dispatch([&]() {
    return jit_transform(nmo, nQ, int2, u, block_q, max_devices);
  });
}

extern "C" int hilbert_focas_df_c1_cuda_transform_low_rank(
    int nmo, int rank, long long nQ, double *int2, const double *v,
    const double *a, int block_q, int max_devices) {
  return dispatch([&]() {
    return jit_low_rank_transform(nmo, rank, nQ, int2, v, a, block_q,
                                  max_devices);
  });
}

extern "C" int hilbert_focas_df_c1_cuda_fi_exchange(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    double *fock_occ, double *fock_ext, int q_chunk, int max_devices) {
  return dispatch([&]() {
    return jit_fi_exchange(nmo, ndoc, nact, nQ, int2, fock_occ, fock_ext,
                           q_chunk, max_devices);
  });
}

extern "C" int hilbert_focas_df_sym_cuda_fi_exchange(
    int nmo, int ndoc, long long nQ, const double *int2, const int *doc_df,
    double *c_out, int q_chunk, int max_devices) {
  return dispatch([&]() {
    return jit_sym_fi_exchange(nmo, ndoc, nQ, int2, doc_df, c_out, q_chunk,
                               max_devices);
  });
}

extern "C" int hilbert_focas_df_c1_cuda_fa_exchange(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    const double *den1, double *fock_occ, double *fock_ext, int q_chunk,
    int max_devices) {
  return dispatch([&]() {
    return jit_fa_exchange(nmo, ndoc, nact, nQ, int2, den1, fock_occ,
                           fock_ext, q_chunk, max_devices);
  });
}

extern "C" int hilbert_focas_df_sym_cuda_fa_exchange(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    const double *den1, const int *act_df, double *c_out, int q_chunk,
    int max_devices) {
  return dispatch([&]() {
    return jit_sym_fa_exchange(nmo, ndoc, nact, nQ, int2, den1, act_df,
                               c_out, q_chunk, max_devices);
  });
}

extern "C" int hilbert_focas_df_c1_cuda_fi_coulomb(
    int nmo, int ndoc, int nact, long long nQ, const double *int1,
    const double *int2, double *fock_occ, double *fock_ext, int q_chunk,
    int max_devices) {
  return dispatch([&]() {
    return jit_fi_coulomb(nmo, ndoc, nact, nQ, int1, int2, fock_occ,
                          fock_ext, q_chunk, max_devices);
  });
}

extern "C" int hilbert_focas_df_c1_cuda_fa_coulomb(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    const double *den1, double *fock_occ, double *fock_ext, int q_chunk,
    int max_devices) {
  return dispatch([&]() {
    return jit_fa_coulomb(nmo, ndoc, nact, nQ, int2, den1, fock_occ,
                          fock_ext, q_chunk, max_devices);
  });
}

extern "C" int hilbert_focas_df_c1_cuda_q(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    const double *den2, double *q_out, int q_chunk, int max_devices) {
  return dispatch([&]() {
    return jit_q(nmo, ndoc, nact, nQ, int2, den2, q_out, q_chunk,
                 max_devices);
  });
}

extern "C" int hilbert_focas_df_c1_cuda_gradient_all(
    int nmo, int ndoc, int nact, long long nQ, const double *int1,
    const double *int2, const double *den1, const double *den2,
    double *fock_i_occ, double *fock_i_ext, double *fock_a_occ,
    double *fock_a_ext, double *q_out, int q_chunk, int max_devices) {
  return dispatch([&]() {
    // Older JIT sources do not export the fused entry point; report a distinct
    // status so the caller falls back to the five separate operators.
    if (jit_gradient_all == nullptr) {
      return 342;
    }
    return jit_gradient_all(nmo, ndoc, nact, nQ, int1, int2, den1, den2,
                            fock_i_occ, fock_i_ext, fock_a_occ, fock_a_ext,
                            q_out, q_chunk, max_devices);
  });
}

extern "C" int hilbert_focas_df_sym_cuda_q(
    int nmo, int ndoc, int nact, long long nQ, const double *int2,
    const double *scaled_d2, const int *act_df, double *q_out, int q_chunk,
    int max_devices) {
  return dispatch([&]() {
    return jit_sym_q(nmo, ndoc, nact, nQ, int2, scaled_d2, act_df, q_out,
                     q_chunk, max_devices);
  });
}
