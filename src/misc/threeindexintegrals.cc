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

#include <psi4/libpsi4util/process.h>
#include <psi4/libmints/wavefunction.h>
#include <psi4/libmints/basisset.h>
#include <psi4/libpsio/psio.hpp>
#include <psi4/libmints/integral.h>
#include <psi4/libmints/twobody.h>
#include <psi4/libmints/matrix.h>
#include <psi4/libmints/vector.h>
#include <psi4/psifiles.h>
#include <psi4/libtrans/integraltransform.h>
#include <psi4/psi4-dec.h>

#include "blas.h"
#include "hilbert_psifiles.h"
#include "threeindexintegrals.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

using namespace psi;
using namespace fnocc;

namespace hilbert{

extern "C" int hilbert_focas_df_ao_to_mo_cuda_transform(
    int nao, int nmo, long long nQ, const double *qao, double *qmo,
    const double *c_pitzer, int block_q, int max_devices);

namespace {

std::size_t checked_product(std::size_t a, std::size_t b,
                            const char *label) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw PsiException(std::string("size overflow while allocating ") + label,
                           __FILE__, __LINE__);
    }
    return a * b;
}

std::size_t available_host_bytes() {
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo) {
        std::string key;
        unsigned long long value_kib = 0;
        std::string unit;
        while (meminfo >> key >> value_kib >> unit) {
            if (key == "MemAvailable:") {
                return static_cast<std::size_t>(value_kib) * 1024U;
            }
        }
    }
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        return checked_product(static_cast<std::size_t>(pages),
                               static_cast<std::size_t>(page_size),
                               "available-memory estimate");
    }
#endif
    return 0;
}

std::string uppercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return std::toupper(ch); });
    return value;
}

void determine_auxiliary_size(const std::shared_ptr<Wavefunction> &ref,
                              const std::shared_ptr<PSIO> &psio,
                              long int &nQ) {
    const std::string scf_type = ref->options().get_str("SCF_TYPE");
    if (scf_type == "DF" || scf_type == "DISK_DF" || scf_type == "MEM_DF") {
        nQ = ref->get_basisset("DF_BASIS_SCF")->nbf();
        Process::environment.globals["NAUX (SCF)"] = nQ;
    } else if (scf_type == "CD") {
        psio->open(PSIF_DFSCF_BJ, PSIO_OPEN_OLD);
        psio->read_entry(PSIF_DFSCF_BJ, "length", reinterpret_cast<char *>(&nQ),
                         sizeof(long int));
        psio->close(PSIF_DFSCF_BJ, 1);
    } else {
        throw PsiException("direct three-index transformation requires DF or CD SCF",
                           __FILE__, __LINE__);
    }
}

std::vector<double> retained_pitzer_coefficients(
    const std::shared_ptr<Wavefunction> &ref, int retained_nmo) {
    const int nmo = ref->nmo();
    const int nso = ref->nso();
    const int nirrep = ref->nirrep();
    std::vector<int> selected(nmo, 0);
    std::vector<int> energy_irrep(nmo, -1);
    std::vector<int> energy_local(nmo, -1);

    for (int energy = 0; energy < nmo; ++energy) {
        double lowest = std::numeric_limits<double>::max();
        int lowest_global = -1;
        int lowest_local = -1;
        int lowest_irrep = -1;
        int global = 0;
        for (int h = 0; h < nirrep; ++h) {
            for (int local = 0; local < ref->nmopi()[h]; ++local) {
                if (!selected[global + local] &&
                    ref->epsilon_a()->pointer(h)[local] < lowest) {
                    lowest = ref->epsilon_a()->pointer(h)[local];
                    lowest_global = global + local;
                    lowest_local = local;
                    lowest_irrep = h;
                }
            }
            global += ref->nmopi()[h];
        }
        if (lowest_global < 0) {
            throw PsiException("failed to construct energy-to-Pitzer orbital map",
                               __FILE__, __LINE__);
        }
        selected[lowest_global] = 1;
        energy_irrep[energy] = lowest_irrep;
        energy_local[energy] = lowest_local;
    }

    SharedMatrix ca_energy(new Matrix(ref->Ca_subset("AO", "ALL")));
    std::vector<double> c_pitzer(
        checked_product(static_cast<std::size_t>(nso),
                        static_cast<std::size_t>(retained_nmo),
                        "retained AO-MO coefficients"),
        0.0);

    int kept = 0;
    for (int energy = 0; energy < nmo; ++energy) {
        const int h = energy_irrep[energy];
        const int local = energy_local[energy];
        const int kept_in_irrep = ref->nmopi()[h] - ref->frzvpi()[h];
        if (local >= kept_in_irrep) continue;
        int pitzer = local;
        for (int prior = 0; prior < h; ++prior) {
            pitzer += ref->nmopi()[prior] - ref->frzvpi()[prior];
        }
        for (int mu = 0; mu < nso; ++mu) {
            c_pitzer[static_cast<std::size_t>(mu) * retained_nmo + pitzer] =
                ca_energy->pointer()[mu][energy];
        }
        ++kept;
    }
    if (kept != retained_nmo) {
        throw PsiException("retained orbital count is inconsistent with frozen virtuals",
                           __FILE__, __LINE__);
    }
    return c_pitzer;
}

} // namespace

void ThreeIndexIntegrals(std::shared_ptr<Wavefunction> ref, long int &nQ, long int memory) {

    int nmo    = ref->nmo();
    int nso    = ref->nso();
    int nirrep = ref->nirrep();


    std::shared_ptr<BasisSet> basisset = ref->basisset();

    // get ntri from sieve
    IntegralFactory factory(basisset, basisset, basisset, basisset);
    auto eri_computer = std::shared_ptr<TwoBodyAOInt>(factory.eri());
    const std::vector<std::pair<int, int> >& function_pairs = eri_computer->function_pairs();
    long int ntri = function_pairs.size();

    // read integrals that were written to disk in the scf
    std::shared_ptr<PSIO> psio(new PSIO());

    if ( (ref->options().get_str("SCF_TYPE") == "DF" || ref->options().get_str("SCF_TYPE") == "DISK_DF" || ref->options().get_str("SCF_TYPE") == "MEM_DF") ) {
        std::shared_ptr<BasisSet> primary = ref->basisset(); 
        std::shared_ptr<BasisSet> auxiliary = ref->get_basisset("DF_BASIS_SCF");

        nQ = auxiliary->nbf();
        Process::environment.globals["NAUX (SCF)"] = nQ;
    }else if ( ref->options().get_str("SCF_TYPE") == "CD" ) {
        psio->open(PSIF_DFSCF_BJ,PSIO_OPEN_OLD);
        psio->read_entry(PSIF_DFSCF_BJ, "length", (char*)&nQ, sizeof(long int));
        psio->close(PSIF_DFSCF_BJ,1);
    }

    // 100 mb extra to account for all mapping arrays already 
    // allocated. this should be WAY more than necessary.
    long int extra = 100 * 1024 * 1024;  
    long int ndoubles = (memory-extra) / 8;

    // orbitals will end up in energy order.  
    // we will want them in pitzer.  for sorting: 
    long int * reorder  = (long int*)malloc(nmo*sizeof(long int));
    long int * sym      = (long int*)malloc(nmo*sizeof(long int));
    bool * skip    = (bool*)malloc(nmo*sizeof(bool));

    for (long int i = 0; i < nmo; i++) {
        skip[i] = false;
    }
    for (long int i = 0; i < nmo; i++) {
        double min   = 1.e99;
        long int count    = 0;
        long int minj     = -999;
        long int minh     = -999;
        long int mincount = -999;
        for (int h = 0; h < nirrep; h++) {
            for (long int j = 0; j < ref->nmopi()[h]; j++) {
                if ( skip[count+j] ) continue;
                if ( ref->epsilon_a()->pointer(h)[j] < min ) {
                    min      = ref->epsilon_a()->pointer(h)[j];
                    mincount = count;
                    minj     = j;
                    minh     = h;
                }
            }
            count += ref->nmopi()[h];
        }
        skip[mincount + minj]     = true;
        reorder[i]                = minj;
        sym[i]                    = minh;
    }

    // how many rows of (Q|mn) can we read in at once?
    if ( ndoubles < nso*nso ) {
        throw PsiException("holy moses, we can't fit nso^2 doubles in memory.  increase memory!",__FILE__,__LINE__);
    }

    long int nrows = 1;
    long int rowsize = nQ;
    while ( rowsize*nso*nso*2 > ndoubles ) {
        nrows++;
        rowsize = nQ / nrows;
        if (nrows * rowsize < nQ) rowsize++;
        if (rowsize == 1) break;
    }
    long int lastrowsize = nQ - (nrows - 1L) * rowsize;
    long int * rowdims = new long int [nrows];
    for (int i = 0; i < nrows-1; i++) rowdims[i] = rowsize;
    rowdims[nrows-1] = lastrowsize;

    double * tmp1 = (double*)malloc(rowdims[0]*nso*nso*sizeof(double));
    double * tmp2 = (double*)malloc(rowdims[0]*nso*nso*sizeof(double));

    long int nn1mo = nmo*(nmo+1)/2;

    psio->open(PSIF_DCC_QSO,PSIO_OPEN_NEW);
    psio->open(PSIF_DCC_QMO,PSIO_OPEN_NEW);
    psio_address addr  = PSIO_ZERO;
    psio_address addr2 = PSIO_ZERO;
    for (long int row = 0; row < nrows; row++) {
        psio->write(PSIF_DCC_QSO, "(Q|mn) Integrals", (char*) tmp1, sizeof(double) * rowdims[row] * nso * nso,addr,&addr);
        psio->write(PSIF_DCC_QMO, "(Q|mn) Integrals", (char*) tmp1, sizeof(double) * rowdims[row] * nn1mo,addr2,&addr2);
    }
    addr = PSIO_ZERO;
    for (long int row = 0; row < nrows; row++) {
        psio->write(PSIF_DCC_QSO, "(Q|mn) Half-Transformed Integrals", (char*) tmp1, sizeof(double) * rowdims[row] * nso * nso,addr,&addr);
    }
    psio->close(PSIF_DCC_QSO,1);
    psio->close(PSIF_DCC_QMO,1);

    // read integrals from SCF and unpack them
    addr  = PSIO_ZERO;
    addr2 = PSIO_ZERO;
    psio->open(PSIF_DFSCF_BJ,PSIO_OPEN_OLD);
    psio->open(PSIF_DCC_QSO,PSIO_OPEN_OLD);

    memset((void*)tmp1,'\0',nso*nso*rowdims[0]*sizeof(double));
    for (long int row = 0; row < nrows; row++) {

        // read
        psio->read(PSIF_DFSCF_BJ, "(Q|mn) Integrals", (char*) tmp2, sizeof(double) * ntri * rowdims[row],addr,&addr);

        // unpack
        #pragma omp parallel for schedule (static)
        for (long int Q = 0; Q < rowdims[row]; Q++) {
            for (long int mn = 0; mn < ntri; mn++) {

                long int m = function_pairs[mn].first;
                long int n = function_pairs[mn].second;

                tmp1[Q*nso*nso+m*nso+n] = tmp2[Q*ntri+mn];
                tmp1[Q*nso*nso+n*nso+m] = tmp2[Q*ntri+mn];
            }
        }

        // write
        psio->write(PSIF_DCC_QSO, "(Q|mn) Integrals", (char*) tmp1, sizeof(double) * nso*nso * rowdims[row],addr2,&addr2);
    }
    psio->close(PSIF_DFSCF_BJ,1);

    // AO->MO transformation matrix:
    SharedMatrix myCa (new Matrix(ref->Ca_subset("AO","ALL")));

    // transform first index:
    addr  = PSIO_ZERO;
    addr2 = PSIO_ZERO;
    for (long int row = 0; row < nrows; row++) {
        // read
        psio->read(PSIF_DCC_QSO, "(Q|mn) Integrals", (char*) tmp1, sizeof(double) * nso*nso * rowdims[row],addr,&addr);

        // transform first index:
        F_DGEMM('n','n',nmo,nso*rowdims[row],nso,1.0,&(myCa->pointer()[0][0]),nmo,tmp1,nso,0.0,tmp2,nmo);

        // sort
        #pragma omp parallel for schedule (static)
        for (long int Q = 0; Q < rowdims[row]; Q++) {
            for (long int i = 0; i < nmo; i++) {
                for (long int m = 0; m < nso; m++) {
                    tmp1[Q*nso*nmo+i*nso+m] = tmp2[Q*nso*nmo+m*nmo+i];
                }
            }
        }

        // write
        psio->write(PSIF_DCC_QSO, "(Q|mn) Half-Transformed Integrals", (char*) tmp1, sizeof(double) * nso*nmo * rowdims[row],addr2,&addr2);
    }
    // transform second index:
    addr  = PSIO_ZERO;
    addr2 = PSIO_ZERO;
    psio->open(PSIF_DCC_QMO,PSIO_OPEN_OLD);
    for (long int row = 0; row < nrows; row++) {
        // read
        psio->read(PSIF_DCC_QSO, "(Q|mn) Half-Transformed Integrals", (char*) tmp1, sizeof(double) * nso*nmo * rowdims[row],addr,&addr);

        // transform second index:
        F_DGEMM('n','n',nmo,nmo*rowdims[row],nso,1.0,&(myCa->pointer()[0][0]),nmo,tmp1,nso,0.0,tmp2,nmo);

        // sort orbitals into pitzer order
        #pragma omp parallel for schedule (static)
        for (long int Q = 0; Q < rowdims[row]; Q++) {
            for (long int m = 0; m < nmo; m++) {
                int hm = sym[m];
                long int offm = 0;
                for (int h = 0; h < hm; h++) {
                    offm += ref->nmopi()[h] - ref->frzvpi()[h];
                }
                if ( reorder[m] >= ref->nmopi()[hm] - ref->frzvpi()[hm] ) continue;
                long int mm = reorder[m] + offm;
                for (long int n = 0; n < nmo; n++) {
                    int hn = sym[n];
                    long int offn = 0;
                    for (int h = 0; h < hn; h++) {
                        offn += ref->nmopi()[h] - ref->frzvpi()[h];
                    }
                    if ( reorder[n] >= ref->nmopi()[hn] - ref->frzvpi()[hn] ) continue;
                    long int nn = reorder[n] + offn;
                    tmp1[Q*nn1mo+INDEX(mm,nn)] = tmp2[Q*nmo*nmo+m*nmo+n];
                }
            }
        }

        // write
        psio->write(PSIF_DCC_QMO, "(Q|mn) Integrals", (char*) tmp1, sizeof(double) * nn1mo * rowdims[row],addr2,&addr2);
    }
    psio->close(PSIF_DCC_QMO,1);
    psio->close(PSIF_DCC_QSO,1);

    delete[] rowdims;

    //F_DGEMM('t','t',nso*nQ,nso,nso,1.0,tmp1,nso,&(myCa->pointer()[0][0]),nso,0.0,tmp2,nso*nQ);
    //F_DGEMM('t','t',nso*nQ,nso,nso,1.0,tmp2,nso,&(myCa->pointer()[0][0]),nso,0.0,tmp1,nso*nQ);

    free(reorder);
    free(skip);
    free(sym);
    free(tmp2);
    free(tmp1);

    //Qmo_ = (double*)malloc(nn1mo*nQ*sizeof(double));
    //memset((void*)Qmo_,'\0',nn1mo*nQ*sizeof(double));
    //psio->open(PSIF_DCC_QMO,PSIO_OPEN_OLD);
    //psio->read_entry(PSIF_DCC_QMO,"(Q|mn) Integrals",(char*)Qmo_,sizeof(double)*nQ * nn1mo);
    //psio->close(PSIF_DCC_QMO,1);

}

void ThreeIndexIntegralsDirect(std::shared_ptr<Wavefunction> ref, long int &nQ,
                               long int memory, int retained_nmo, double *&qmo,
                               const DirectThreeIndexOptions &options) {
    const int nmo = ref->nmo();
    const int nso = ref->nso();
    if (nso <= 0 || nmo <= 0 || retained_nmo <= 0 || retained_nmo > nmo ||
        nmo > nso) {
        throw PsiException("invalid dimensions for direct three-index transformation",
                           __FILE__, __LINE__);
    }

    const std::string requested_backend = uppercase(options.backend);
    if (requested_backend != "AUTO" && requested_backend != "CPU" &&
        requested_backend != "CUDA") {
        throw PsiException("DF_INTEGRAL_TRANSFORM_BACKEND must be AUTO, CPU, or CUDA",
                           __FILE__, __LINE__);
    }

    std::shared_ptr<BasisSet> basisset = ref->basisset();
    IntegralFactory factory(basisset, basisset, basisset, basisset);
    auto eri_computer = std::shared_ptr<TwoBodyAOInt>(factory.eri());
    const std::vector<std::pair<int, int>> &function_pairs =
        eri_computer->function_pairs();
    const std::size_t ntri = function_pairs.size();
    const std::size_t ao_pair =
        checked_product(static_cast<std::size_t>(nso), nso + 1U,
                        "packed AO pairs") /
        2U;
    const std::size_t mo_pair =
        checked_product(static_cast<std::size_t>(retained_nmo),
                        retained_nmo + 1U, "packed MO pairs") /
        2U;

    std::shared_ptr<PSIO> psio(new PSIO());
    determine_auxiliary_size(ref, psio, nQ);
    if (nQ <= 0) {
        throw PsiException("SCF three-index file contains no auxiliary functions",
                           __FILE__, __LINE__);
    }

    const std::size_t final_elements =
        checked_product(static_cast<std::size_t>(nQ), mo_pair,
                        "final MO three-index tensor");
    const std::size_t final_bytes =
        checked_product(final_elements, sizeof(double),
                        "final MO three-index tensor");
    if (memory > 0 && final_bytes > static_cast<std::size_t>(memory)) {
        throw PsiException(
            "the retained MO three-index tensor exceeds the configured Psi4 memory",
            __FILE__, __LINE__);
    }
    qmo = static_cast<double *>(std::malloc(final_bytes));
    if (qmo == nullptr) {
        throw PsiException("could not allocate the final MO three-index tensor",
                           __FILE__, __LINE__);
    }

    bool scf_file_open = false;
    try {
        std::vector<double> c_pitzer =
            retained_pitzer_coefficients(ref, retained_nmo);

        const std::size_t dense_ao =
            checked_product(static_cast<std::size_t>(nso), nso,
                            "dense AO block");
        const std::size_t ao_mo =
            checked_product(static_cast<std::size_t>(nso), retained_nmo,
                            "half-transformed block");
        const std::size_t dense_mo =
            checked_product(static_cast<std::size_t>(retained_nmo), retained_nmo,
                            "dense MO block");
        const std::size_t cpu_doubles_per_q =
            ntri + std::max(dense_ao, ao_mo) + std::max(ao_mo, dense_mo);
        const std::size_t cuda_host_doubles_per_q = ntri + ao_pair;
        const std::size_t scratch_doubles_per_q =
            std::max(cpu_doubles_per_q, cuda_host_doubles_per_q);
        const std::size_t scratch_bytes_per_q =
            checked_product(scratch_doubles_per_q, sizeof(double),
                            "per-block transformation scratch");

        double fraction = options.memory_fraction;
        if (fraction <= 0.0) fraction = 0.05;
        fraction = std::min(1.0, fraction);
        const std::size_t configured_bytes =
            memory > 0 ? static_cast<std::size_t>(memory)
                       : std::numeric_limits<std::size_t>::max();
        const std::size_t reserve_bytes = 100U * 1024U * 1024U;
        const std::size_t configured_remaining =
            configured_bytes > final_bytes + reserve_bytes
                ? configured_bytes - final_bytes - reserve_bytes
                : 0;
        std::size_t scratch_budget =
            memory > 0
                ? static_cast<std::size_t>(fraction * static_cast<double>(memory))
                : scratch_bytes_per_q;
        scratch_budget = std::min(scratch_budget, configured_remaining);

        if (options.use_available_memory) {
            const std::size_t available = available_host_bytes();
            if (available > 0) {
                const std::size_t live_reserve =
                    std::max<std::size_t>(512U * 1024U * 1024U, available / 10U);
                const std::size_t live_budget =
                    available > final_bytes + live_reserve
                        ? static_cast<std::size_t>(
                              0.8 * (available - final_bytes - live_reserve))
                        : 0;
                scratch_budget = std::min(scratch_budget, live_budget);
            }
        }
        if (scratch_budget < scratch_bytes_per_q) {
            throw PsiException(
                "insufficient host memory for one direct three-index transform block after allocating Qmo",
                __FILE__, __LINE__);
        }

        std::size_t automatic_block = scratch_budget / scratch_bytes_per_q;
        automatic_block = std::max<std::size_t>(1, automatic_block);
        if (options.block_q_max > 0) {
            automatic_block = std::min<std::size_t>(automatic_block,
                                                    options.block_q_max);
        }
        int block_q = options.block_q > 0 ? options.block_q
                                         : static_cast<int>(automatic_block);
        block_q = static_cast<int>(std::min<long int>(block_q, nQ));
        if (block_q <= 0 ||
            checked_product(static_cast<std::size_t>(block_q),
                            scratch_bytes_per_q, "requested transform block") >
                scratch_budget) {
            throw PsiException(
                "DF_INTEGRAL_TRANSFORM_BLOCK_Q exceeds the safe host-memory budget",
                __FILE__, __LINE__);
        }

        bool use_cuda = requested_backend != "CPU";
        std::vector<double> scf_block(
            checked_product(static_cast<std::size_t>(block_q), ntri,
                            "screened SCF block"));
        std::vector<double> packed_ao;
        std::vector<double> dense_work;
        std::vector<double> transformed;

        auto allocate_cpu_scratch = [&]() {
            if (!dense_work.empty()) return;
            packed_ao.clear();
            packed_ao.shrink_to_fit();
            dense_work.resize(checked_product(
                static_cast<std::size_t>(block_q),
                std::max(dense_ao, ao_mo), "CPU dense work block"));
            transformed.resize(checked_product(
                static_cast<std::size_t>(block_q),
                std::max(ao_mo, dense_mo), "CPU transformed block"));
        };
        if (use_cuda) {
            packed_ao.resize(checked_product(
                static_cast<std::size_t>(block_q), ao_pair,
                "packed AO CUDA staging block"));
        } else {
            allocate_cpu_scratch();
        }

        psio_address address = PSIO_ZERO;
        psio->open(PSIF_DFSCF_BJ, PSIO_OPEN_OLD);
        scf_file_open = true;
        for (long int q_begin = 0; q_begin < nQ; q_begin += block_q) {
            const int bq = static_cast<int>(
                std::min<long int>(block_q, nQ - q_begin));
            const std::size_t scf_count =
                checked_product(static_cast<std::size_t>(bq), ntri,
                                "SCF read block");

            psio->read(PSIF_DFSCF_BJ, "(Q|mn) Integrals",
                       reinterpret_cast<char *>(scf_block.data()),
                       checked_product(scf_count, sizeof(double),
                                       "SCF read bytes"),
                       address, &address);
            bool block_done = false;
            if (use_cuda) {
                std::fill(packed_ao.begin(),
                          packed_ao.begin() +
                              checked_product(static_cast<std::size_t>(bq),
                                              ao_pair, "packed AO block"),
                          0.0);
#pragma omp parallel for schedule(static)
                for (int q = 0; q < bq; ++q) {
                    for (std::size_t mn = 0; mn < ntri; ++mn) {
                        const int mu = function_pairs[mn].first;
                        const int nu = function_pairs[mn].second;
                        const std::size_t pair =
                            mu >= nu ? static_cast<std::size_t>(mu) * (mu + 1) / 2 + nu
                                     : static_cast<std::size_t>(nu) * (nu + 1) / 2 + mu;
                        packed_ao[static_cast<std::size_t>(q) * ao_pair + pair] =
                            scf_block[static_cast<std::size_t>(q) * ntri + mn];
                    }
                }
                const int status = hilbert_focas_df_ao_to_mo_cuda_transform(
                    nso, retained_nmo, bq, packed_ao.data(),
                    qmo + static_cast<std::size_t>(q_begin) * mo_pair,
                    c_pitzer.data(), 0, options.cuda_num_gpus);
                if (status == 0) {
                    block_done = true;
                } else {
                    use_cuda = false;
                    outfile->Printf(
                        "        CUDA initial DF transform unavailable (status %d); "
                        "continuing with direct CPU blocks.\n",
                        status);
                    allocate_cpu_scratch();
                }
            }

            if (!block_done) {
                const std::size_t dense_count =
                    checked_product(static_cast<std::size_t>(bq), dense_ao,
                                    "dense AO block");
                std::fill(dense_work.begin(), dense_work.begin() + dense_count,
                          0.0);
#pragma omp parallel for schedule(static)
                for (int q = 0; q < bq; ++q) {
                    double *dense =
                        dense_work.data() + static_cast<std::size_t>(q) * dense_ao;
                    for (std::size_t mn = 0; mn < ntri; ++mn) {
                        const int mu = function_pairs[mn].first;
                        const int nu = function_pairs[mn].second;
                        const double value =
                            scf_block[static_cast<std::size_t>(q) * ntri + mn];
                        dense[static_cast<std::size_t>(mu) * nso + nu] = value;
                        dense[static_cast<std::size_t>(nu) * nso + mu] = value;
                    }
                }
                F_DGEMM('n', 'n', retained_nmo, nso * bq, nso, 1.0,
                        c_pitzer.data(), retained_nmo, dense_work.data(), nso,
                        0.0, transformed.data(), retained_nmo);
#pragma omp parallel for schedule(static)
                for (int q = 0; q < bq; ++q) {
                    for (int i = 0; i < retained_nmo; ++i) {
                        for (int mu = 0; mu < nso; ++mu) {
                            dense_work[static_cast<std::size_t>(q) * ao_mo +
                                       static_cast<std::size_t>(i) * nso + mu] =
                                transformed[static_cast<std::size_t>(q) * ao_mo +
                                            static_cast<std::size_t>(mu) *
                                                retained_nmo +
                                            i];
                        }
                    }
                }
                F_DGEMM('n', 'n', retained_nmo, retained_nmo * bq, nso, 1.0,
                        c_pitzer.data(), retained_nmo, dense_work.data(), nso,
                        0.0, transformed.data(), retained_nmo);
#pragma omp parallel for schedule(static)
                for (int q = 0; q < bq; ++q) {
                    double *out = qmo +
                        (static_cast<std::size_t>(q_begin) + q) * mo_pair;
                    const double *dense = transformed.data() +
                        static_cast<std::size_t>(q) * dense_mo;
                    for (int hi = 0; hi < retained_nmo; ++hi) {
                        for (int lo = 0; lo <= hi; ++lo) {
                            out[static_cast<std::size_t>(hi) * (hi + 1) / 2 + lo] =
                                dense[static_cast<std::size_t>(hi) * retained_nmo + lo];
                        }
                    }
                }
            }
        }
        psio->close(PSIF_DFSCF_BJ, 1);
        scf_file_open = false;

    } catch (...) {
        if (scf_file_open) {
            psio->close(PSIF_DFSCF_BJ, 1);
        }
        std::free(qmo);
        qmo = nullptr;
        throw;
    }
}


}
