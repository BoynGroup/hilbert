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

// psi4 stuff
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libscf_solver/hf.h"

// real_space_density
#include "real_space_density.h"

#include <cmath>
#include <vector>

// blas
#include <misc/blas.h>

// openmp
#include <misc/omp.h>

// tpdm and opdm structs live here
#include <v2rdm_casscf/v2rdm_solver.h>

using namespace psi;
using namespace fnocc;

namespace hilbert{ 

// the RealSpaceDensity class derives from the Wavefunction class and inherits its members
RealSpaceDensity::RealSpaceDensity(std::shared_ptr<Wavefunction> reference_wavefunction,Options & options_):
    Wavefunction(options_){

    reference_wavefunction_ = reference_wavefunction;
    common_init();
}

RealSpaceDensity::~RealSpaceDensity() {
}

// initialize members of the RealSpaceDensity class
void RealSpaceDensity::common_init() {

    reference_energy_ = Process::environment.globals["V2RDM TOTAL ENERGY"];
    
    shallow_copy(reference_wavefunction_);

    // number of alpha electrons
    nalpha_   = reference_wavefunction_->nalpha();

    // number of beta electrons
    nbeta_    = reference_wavefunction_->nbeta();

    // number of alpha electrons per irrep
    nalphapi_ = reference_wavefunction_->nalphapi();

    // number of beta electrons per irrep
    nbetapi_  = reference_wavefunction_->nbetapi();

    // number of frozen core orbitals per irrep
    frzcpi_   = reference_wavefunction_->frzcpi();

    // number of frozen virtual orbitals per irrep
    frzvpi_   = reference_wavefunction_->frzvpi();

    // number of molecular orbials per irrep
    nmopi_    = reference_wavefunction_->nmopi();

    // number of symmetry orbials per irrep
    nsopi_    = reference_wavefunction_->nsopi();

    // number of irreducible representations
    nirrep_   = reference_wavefunction_->nirrep();

    // total number of symmetry orbitals
    nso_      = reference_wavefunction_->nso();

    // total number of molecular orbitals
    nmo_      = reference_wavefunction_->nmo();

    // grab the molecule from the reference wave function
    molecule_ = reference_wavefunction_->molecule();

    // SO/MO transformation matrices
    Ca_ = std::shared_ptr<Matrix>(reference_wavefunction_->Ca());
    Cb_ = std::shared_ptr<Matrix>(reference_wavefunction_->Cb());

    // overlap integrals
    S_  = std::shared_ptr<Matrix>(reference_wavefunction_->S());

    // SO-basis Fock matrices
    Fa_ = std::shared_ptr<Matrix>(reference_wavefunction_->Fa());
    Fb_ = std::shared_ptr<Matrix>(reference_wavefunction_->Fb());

    // SO-basis density matrices
    Da_ = std::shared_ptr<Matrix>(reference_wavefunction_->Da());
    Db_ = std::shared_ptr<Matrix>(reference_wavefunction_->Db());

    // orbital energies
    epsilon_a_ = std::make_shared<Vector>(nmopi_);
    epsilon_a_->copy(*reference_wavefunction_->epsilon_a().get());
    epsilon_b_ = std::make_shared<Vector>(nmopi_);
    epsilon_b_->copy(*reference_wavefunction_->epsilon_b().get());

    // set the wavefunction name
    name_ = "WHOKNOWS";

    // restricted orbitals, unrestricted rdms
// TODO get these from reference wave function
    same_a_b_orbs_ = false;
    same_a_b_dens_ = false;

    // symmetry of orbitals:
    symmetry_ = (int*)malloc(nmo_*sizeof(int));
    memset((void*)symmetry_,'\0',nmo_*sizeof(int));
    int count = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int i = 0; i < nsopi_[h]; i++) {
            symmetry_[count++] = h;
        }
    }

    // pitzer offset
    pitzer_offset_ = (int*)malloc(nirrep_*sizeof(int));
    count = 0;
    for (int h = 0; h < nirrep_; h++) {
        pitzer_offset_[h] = count;
        count += nsopi_[h];
    }

    // evaluate basis function values on a grid:

    scf::HF* scfwfn = (scf::HF*)reference_wavefunction_.get();
    std::shared_ptr<SuperFunctional> functional = scfwfn->functional();
    potential_ = scfwfn->V_potential();

    // phi matrix (real-space <- AO basis mapping)
    // since grid is stored in blocks, we need to build a full phi matrix
    // from the individual blocks:

    std::shared_ptr<BasisSet> primary = reference_wavefunction_->get_basisset("ORBITAL");
    int nao = primary->nbf();
    std::shared_ptr<Matrix> Da_ao (new Matrix(nao,nao));
    std::shared_ptr<Matrix> Db_ao (new Matrix(nao,nao));

    points_func_ = potential_->properties()[0];
    points_func_->set_pointers(Da_ao,Db_ao);

    // determine number of grid points
    int nblocks = potential_->nblocks();
    phi_points_    = 0;
    max_functions_ = 0;
    max_points_    = 0;
    for (int myblock = 0; myblock < nblocks; myblock++) {
        std::shared_ptr<BlockOPoints> block = potential_->get_block(myblock);
        points_func_->compute_points(block);
        int npoints = block->npoints();
        phi_points_ += npoints;
        const std::vector<int>& function_map = block->functions_local_to_global();
        int nlocal = function_map.size();
        if ( nlocal > max_functions_ ) max_functions_ = nlocal;
        if ( npoints > max_points_ )   max_points_    = npoints;
    }

    // estimate memory requirements

    //outfile->Printf("\n"); 
    //outfile->Printf("    ==> Memory requirements <==\n");
    //outfile->Printf("\n");
    
    // memory is from process::environment
    memory_ = Process::environment.get_memory();

    // assume 500 MB is already allocated by psi4
    long int n_mem         = 500 * 1024 * 1024 / 8;

    // default basic vectors and matrices initialized using common_init()
    n_mem +=     nso_ * nso_;                             // S_ matrix
    n_mem += 2 * nso_ * nso_;                             // Ca_ and Cb_ matrices
    n_mem += 2 * nso_ * nso_;                             // Fa_ and Fb_ matrices
    n_mem += 2 * nmo_;                                    // epsilon_a_ and epsilon_b_ vectors

    // phiAO, phiAO_x, phiAO_y, phiAO_z
    n_mem += phi_points_ * nso_;
    n_mem += 3 * phi_points_ * nso_;

    // required memory for grids x, y, and z and weights w vectors
    n_mem += 4 * phi_points_;

    // memory for AO->MO transformation
    n_mem += nirrep_ * phi_points_;    // phi_points_list vector
    n_mem += phi_points_ * nso_;       // super_phi_ matrix
    n_mem += nso_ * nso_;              // AO2SO matrix in TransformPhiMatrixAOMO()
    n_mem += phi_points_ * nso_;       // temporary matrix called three times in TransformPhiMatrixAOMO()
    n_mem += 3 * phi_points_ * nso_;   // for super_phi_x_, _y_ and _z_ gradient matrices

    // memory for rho and rho'
    n_mem += 3 * phi_points_;          // rho_a_, rho_b_ and rho_ vectors
    n_mem += 3 * phi_points_;          // rho_a_x_, rho_a_y_ and rho_a_z_ gradient vectors
    n_mem += 3 * phi_points_;          // rho_b_x_, rho_b_y_ and rho_b_z_ gradient vectors

    // memory needed for storing pi vector
    n_mem += phi_points_;              // pi_ vector
    n_mem += 3 * phi_points_;          // pi_x_, pi_y_ and pi_z_ gradient vectors
 
    if ( n_mem * 8.0 > (double)memory_ ) {
        throw PsiException("not enough memory. see plugin mcpdft for a low-memory algorithm",__FILE__,__LINE__);
    }

    grid_x_      = std::shared_ptr<Vector>(new Vector("GRID X",phi_points_));
    grid_y_      = std::shared_ptr<Vector>(new Vector("GRID Y",phi_points_));
    grid_z_      = std::shared_ptr<Vector>(new Vector("GRID Z",phi_points_));
    grid_w_      = std::shared_ptr<Vector>(new Vector("GRID W",phi_points_));

    outfile->Printf("\n"); 
    outfile->Printf("    ==> Build Phi and Phi' matrices ..."); fflush(stdout);

    std::shared_ptr<Matrix> super_phi_ao = std::shared_ptr<Matrix>(new Matrix("SUPER PHI (AO)",phi_points_,nso_));

    std::shared_ptr<Matrix> super_phi_x_ao (new Matrix("SUPER PHI X (AO)",phi_points_,nso_));
    std::shared_ptr<Matrix> super_phi_y_ao (new Matrix("SUPER PHI Y (AO)",phi_points_,nso_));
    std::shared_ptr<Matrix> super_phi_z_ao (new Matrix("SUPER PHI Z (AO)",phi_points_,nso_));

    // build phi matrix and derivative phi matrices (ao basis)
    BuildPhiMatrixAO("PHI",super_phi_ao);
    BuildPhiMatrixAO("PHI_X",super_phi_x_ao);
    BuildPhiMatrixAO("PHI_Y",super_phi_y_ao);
    BuildPhiMatrixAO("PHI_Z",super_phi_z_ao);

    // transform the orbital label in phi matrix and derivative phi matrix to the MO basis

    Dimension phi_points_list = Dimension(nirrep_,"phi_points");
    for (int h = 0; h < nirrep_; h++) {
        phi_points_list[h] = phi_points_;
    }

    super_phi_ = std::shared_ptr<Matrix>(new Matrix("SUPER PHI",phi_points_list,nsopi_));

    TransformPhiMatrixAOMO(super_phi_ao,super_phi_);

    super_phi_x_ = std::shared_ptr<Matrix>(new Matrix("SUPER PHI X",phi_points_list,nsopi_));
    super_phi_y_ = std::shared_ptr<Matrix>(new Matrix("SUPER PHI Y",phi_points_list,nsopi_));
    super_phi_z_ = std::shared_ptr<Matrix>(new Matrix("SUPER PHI Z",phi_points_list,nsopi_));

    TransformPhiMatrixAOMO(super_phi_x_ao,super_phi_x_);
    TransformPhiMatrixAOMO(super_phi_y_ao,super_phi_y_);
    TransformPhiMatrixAOMO(super_phi_z_ao,super_phi_z_);

    outfile->Printf(" Done. <==\n"); 
    outfile->Printf("\n");

}// end of common_init()

void RealSpaceDensity::GetGridInfo() {

    int nblocks = potential_->nblocks();

    phi_points_ = 0;
    for (int myblock = 0; myblock < nblocks; myblock++) {
        std::shared_ptr<BlockOPoints> block = potential_->get_block(myblock);
        points_func_->compute_points(block);
        int npoints = block->npoints();
        const std::vector<int>& function_map = block->functions_local_to_global();
        int nlocal = function_map.size();

        double ** phi = points_func_->basis_value("PHI")->pointer();

        double * x = block->x();
        double * y = block->y();
        double * z = block->z();
        double * w = block->w();

        for (int p = 0; p < npoints; p++) {

            grid_x_->pointer()[phi_points_ + p] = x[p];
            grid_y_->pointer()[phi_points_ + p] = y[p];
            grid_z_->pointer()[phi_points_ + p] = z[p];
            grid_w_->pointer()[phi_points_ + p] = w[p];

        }
        phi_points_ += npoints;
    }
}//end of GetGridInfo()

void RealSpaceDensity::BuildPhiMatrixAO(std::string phi_type, std::shared_ptr<Matrix> myphi) {

    int nblocks = potential_->nblocks();

    phi_points_ = 0;
    for (int myblock = 0; myblock < nblocks; myblock++) {
        std::shared_ptr<BlockOPoints> block = potential_->get_block(myblock);
        points_func_->compute_points(block);
        int npoints = block->npoints();
        const std::vector<int>& function_map = block->functions_local_to_global();
        int nlocal = function_map.size();

        double ** phi = points_func_->basis_value(phi_type)->pointer();

        double * x = block->x();
        double * y = block->y();
        double * z = block->z();
        double * w = block->w();

        for (int p = 0; p < npoints; p++) {

            grid_x_->pointer()[phi_points_ + p] = x[p];
            grid_y_->pointer()[phi_points_ + p] = y[p];
            grid_z_->pointer()[phi_points_ + p] = z[p];
            grid_w_->pointer()[phi_points_ + p] = w[p];

            for (int nu = 0; nu < nlocal; nu++) {
                int nug = function_map[nu];
                myphi->pointer()[phi_points_ + p][nug] = phi[p][nu];
            }
        }
        phi_points_ += npoints;
    }
}

void RealSpaceDensity::TransformPhiMatrixAOMO(std::shared_ptr<Matrix> phi_in, std::shared_ptr<Matrix> phi_out) {
        
    std::shared_ptr<Matrix> phi_temp (new Matrix(phi_out));

    // grab AO->SO transformation matrix
    std::shared_ptr<Matrix> ao2so = reference_wavefunction_->aotoso();

    for (int h = 0; h < nirrep_; h++) {
        if (nsopi_[h] != 0 )
           F_DGEMM('n','n',nsopi_[h],phi_points_,nso_,1.0,ao2so->pointer(h)[0],nsopi_[h],phi_in->pointer()[0],nso_,0.0,phi_temp->pointer(h)[0],nsopi_[h]);
    }

    for (int h = 0; h < nirrep_; h++) {
        if (nsopi_[h] != 0 )
           F_DGEMM('n','n',nsopi_[h],phi_points_,nsopi_[h],1.0,Ca_->pointer(h)[0],nsopi_[h],phi_temp->pointer(h)[0],nsopi_[h],0.0,phi_out->pointer(h)[0],nsopi_[h]);
    }
}

void RealSpaceDensity::SetD1(std::vector<opdm> my_opdm, std::shared_ptr<Matrix> D1) {

    for (size_t n = 0; n < my_opdm.size(); n++) {

        int i = my_opdm[n].i;
        int j = my_opdm[n].j;

        int hi = symmetry_[i];
        int hj = symmetry_[j];

        if ( hi != hj ) {
            throw PsiException("error: something is wrong with the symmetry of the OPDM",__FILE__,__LINE__);
        }

        int ii = i - pitzer_offset_[hi];
        int jj = j - pitzer_offset_[hi];

        D1->pointer(hi)[ii][jj] = my_opdm[n].value;
    }
}

void RealSpaceDensity::SetOPDM(std::vector<opdm> opdm_a, std::vector<opdm> opdm_b) {

    opdm_a_ = opdm_a;
    opdm_b_ = opdm_b;

    SetD1(opdm_a_, Da_);
    SetD1(opdm_b_, Db_);
}

void RealSpaceDensity::SetTPDM(std::vector<tpdm> tpdm_ab) {
    tpdm_ab_ = tpdm_ab;
}

void RealSpaceDensity::ReadOPDM() {
    
    // read opdm from disk 

    std::shared_ptr<PSIO> psio (new PSIO());

    if ( !psio->exists(PSIF_V2RDM_D1A) ) throw PsiException("No D1a on disk",__FILE__,__LINE__);
    if ( !psio->exists(PSIF_V2RDM_D1B) ) throw PsiException("No D1b on disk",__FILE__,__LINE__);

    // D1a

    psio->open(PSIF_V2RDM_D1A,PSIO_OPEN_OLD);

    long int na;
    psio->read_entry(PSIF_V2RDM_D1A,"length",(char*)&na,sizeof(long int));

    opdm_a_.resize(na);
    psio->read_entry(PSIF_V2RDM_D1A,"D1a",(char*)&opdm_a_[0],na * sizeof(opdm));
    psio->close(PSIF_V2RDM_D1A,1);

    SetD1(opdm_a_, Da_);

    // D1b

    psio->open(PSIF_V2RDM_D1B,PSIO_OPEN_OLD);

    long int nb;
    psio->read_entry(PSIF_V2RDM_D1B,"length",(char*)&nb,sizeof(long int));

    opdm_b_.resize(nb);
    psio->read_entry(PSIF_V2RDM_D1B,"D1b",(char*)&opdm_b_[0],nb * sizeof(opdm));
    psio->close(PSIF_V2RDM_D1B,1);

    SetD1(opdm_b_, Db_);

}

void RealSpaceDensity::BuildExchangeCorrelationHole(size_t p) {

    outfile->Printf("\n");
    outfile->Printf("    ==> Build Exchange-Correlation Hole ...");

    std::shared_ptr<PSIO> psio (new PSIO());

    if ( !psio->exists(PSIF_V2RDM_D2AB) ) throw PsiException("No D2ab on disk",__FILE__,__LINE__);
    if ( !psio->exists(PSIF_V2RDM_D2AA) ) throw PsiException("No D2ab on disk",__FILE__,__LINE__);
    if ( !psio->exists(PSIF_V2RDM_D2BB) ) throw PsiException("No D2ab on disk",__FILE__,__LINE__);

    psio_address addr_ab = PSIO_ZERO;

    // ab
    psio->open(PSIF_V2RDM_D2AB,PSIO_OPEN_OLD);
    long int nab;
    psio->read_entry(PSIF_V2RDM_D2AB,"length",(char*)&nab,sizeof(long int));
    tpdm * D2ab = (tpdm *)malloc(nab * sizeof(tpdm));
    memset((void*)D2ab,'\0',nab * sizeof(tpdm));
    psio->read_entry(PSIF_V2RDM_D2AB,"D2ab",(char*)D2ab,nab * sizeof(tpdm));
    psio->close(PSIF_V2RDM_D2AB,1);

    // aa
    psio->open(PSIF_V2RDM_D2AA,PSIO_OPEN_OLD);
    long int naa;
    psio->read_entry(PSIF_V2RDM_D2AA,"length",(char*)&naa,sizeof(long int));
    tpdm * D2aa = (tpdm *)malloc(naa * sizeof(tpdm));
    memset((void*)D2aa,'\0',naa * sizeof(tpdm));
    psio->read_entry(PSIF_V2RDM_D2AA,"D2aa",(char*)D2aa,naa * sizeof(tpdm));
    psio->close(PSIF_V2RDM_D2AA,1);

    // bb
    psio->open(PSIF_V2RDM_D2BB,PSIO_OPEN_OLD);
    long int nbb;
    psio->read_entry(PSIF_V2RDM_D2BB,"length",(char*)&nbb,sizeof(long int));
    tpdm * D2bb = (tpdm *)malloc(nbb * sizeof(tpdm));
    memset((void*)D2bb,'\0',nbb * sizeof(tpdm));
    psio->read_entry(PSIF_V2RDM_D2BB,"D2bb",(char*)D2bb,nbb * sizeof(tpdm));
    psio->close(PSIF_V2RDM_D2BB,1);

    double integral = 0.0;
    double * rho_p = rho_->pointer();
    double * w_p   = grid_w_->pointer();

    xc_hole_.reset();
    xc_hole_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));
    double * xc_hole_p = xc_hole_->pointer();
    for (int q = 0; q < phi_points_; q++) {

        // pi(r,r') = D(mu,nu; lambda,sigma) * phi(r,mu) * phi(r',nu) * phi(r,lambda) * phi(r',sigma)
        double pi = 0.0;
        for (int n = 0; n < nab; n++) {

            int i = D2ab[n].i;
            int j = D2ab[n].j;
            int k = D2ab[n].k;
            int l = D2ab[n].l;

            int hi = symmetry_[i];
            int hj = symmetry_[j];
            int hk = symmetry_[k];
            int hl = symmetry_[l];

            int ii = i - pitzer_offset_[hi];
            int jj = j - pitzer_offset_[hj];
            int kk = k - pitzer_offset_[hk];
            int ll = l - pitzer_offset_[hl];

            pi += super_phi_->pointer(hi)[p][ii] * 
                  super_phi_->pointer(hj)[q][jj] * 
                  super_phi_->pointer(hk)[p][kk] * 
                  super_phi_->pointer(hl)[q][ll] * D2ab[n].value * 0.5;

            pi += super_phi_->pointer(hi)[q][ii] * 
                  super_phi_->pointer(hj)[p][jj] * 
                  super_phi_->pointer(hk)[q][kk] * 
                  super_phi_->pointer(hl)[p][ll] * D2ab[n].value * 0.5;

        }
        for (int n = 0; n < naa; n++) {

            int i = D2aa[n].i;
            int j = D2aa[n].j;
            int k = D2aa[n].k;
            int l = D2aa[n].l;

            int hi = symmetry_[i];
            int hj = symmetry_[j];
            int hk = symmetry_[k];
            int hl = symmetry_[l];

            int ii = i - pitzer_offset_[hi];
            int jj = j - pitzer_offset_[hj];
            int kk = k - pitzer_offset_[hk];
            int ll = l - pitzer_offset_[hl];

            pi += super_phi_->pointer(hi)[p][ii] * 
                  super_phi_->pointer(hj)[q][jj] * 
                  super_phi_->pointer(hk)[p][kk] * 
                  super_phi_->pointer(hl)[q][ll] * D2aa[n].value * 0.5;

        }
        for (int n = 0; n < nbb; n++) {

            int i = D2bb[n].i;
            int j = D2bb[n].j;
            int k = D2bb[n].k;
            int l = D2bb[n].l;

            int hi = symmetry_[i];
            int hj = symmetry_[j];
            int hk = symmetry_[k];
            int hl = symmetry_[l];

            int ii = i - pitzer_offset_[hi];
            int jj = j - pitzer_offset_[hj];
            int kk = k - pitzer_offset_[hk];
            int ll = l - pitzer_offset_[hl];

            pi += super_phi_->pointer(hi)[p][ii] * 
                  super_phi_->pointer(hj)[q][jj] * 
                  super_phi_->pointer(hk)[p][kk] * 
                  super_phi_->pointer(hl)[q][ll] * D2bb[n].value * 0.5;

        }
        double nxc = (2.0 * pi - rho_p[p] * rho_p[q]) / rho_p[p];
        //if ( fabs(grid_x_->pointer()[q] < 1e-6 ) ) {
        //printf("%20.12lf %20.12lf %20.12lf %20.12lf\n",grid_x_->pointer()[q],grid_y_->pointer()[q],grid_z_->pointer()[q],nxc);
        //}

        xc_hole_p[q] = nxc;

        integral += nxc * w_p[q];

    }

    outfile->Printf(" Done. <==\n");

    //printf("integral over xc hole (should be -1): %20.12lf\n",integral);
    //exit(0);

    free(D2ab);
    free(D2aa);
    free(D2bb);
}

// build pi. note that there is a low-memory version of this
// function in edeprince3/real_space_density.git, should we ever need it
void RealSpaceDensity::BuildPiFast(const std::vector<tpdm> & D2ab) {

    outfile->Printf("\n");
    outfile->Printf("    ==> Build Pi ...");

    pi_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));

    double * pi_p  = pi_->pointer();

    // By default only the on-top pair density VALUE pi(r) is built. The
    // "translated" MC-PDFT functionals (run_mcpdft in pymodule.py) build the
    // translated density gradients from grad(rho) and zeta, NOT from grad(pi),
    // so grad(pi) is dead work (~90% of this routine: 12 of every 16 phi
    // products per TPDM element). The gradient is only needed by FULLY-
    // translated (ft) functionals; enable it with set_pi_gradient(true).

    // The dense active-active block (~nact^4 elements, the bulk of the work for
    // a large active space) is evaluated as a BLAS-3 GEMM:
    //     pi_aa(p) = sum_{(ij),(kl)} B[p,(ij)] D2act[(ij),(kl)] B[p,(kl)]
    // with B[p,(ij)] = phi_i(p) phi_j(p) over active orbitals. The remaining
    // (core-core, core-active) elements are kept in a scalar list. This only
    // applies to the value-only path; the gradient (ft) path stays scalar, and
    // we fall back to fully-scalar if the active-orbital list is unknown.
    const int  nact     = (int)active_full_indices_.size();
    const bool use_gemm = ( !do_pi_gradient_ ) && ( nact > 0 );

    std::vector<int> actmap;                  // full pitzer index -> compact active (-1 if not active)
    if ( use_gemm ) {
        actmap.assign(nmo_, -1);
        for (int c = 0; c < nact; c++) actmap[ active_full_indices_[c] ] = c;
    }

    const int pairlen = nact * nact;
    std::shared_ptr<Matrix> D2act;
    double ** D2actp = nullptr;
    if ( use_gemm ) {
        D2act = std::make_shared<Matrix>("D2 active", pairlen, pairlen);
        D2actp = D2act->pointer();
    }

    // Resolve symmetry/pitzer arithmetic and the value ONCE per element (these
    // are invariant across grid points), screen out negligible elements
    // (WriteTPDM stores the symmetry-allowed block densely incl. near-zeros),
    // and route active-active elements to the dense D2act matrix, the rest to
    // the scalar list.
    const size_t nn = D2ab.size();
    std::vector<int>    li, lj, lk, ll;   // local (within-irrep) orbital indices
    std::vector<int>    hi, hj, hk, hl;   // irreps
    std::vector<double> val;
    li.reserve(nn); lj.reserve(nn); lk.reserve(nn); ll.reserve(nn);
    hi.reserve(nn); hj.reserve(nn); hk.reserve(nn); hl.reserve(nn);
    val.reserve(nn);
    size_t n_aa = 0;
    for (size_t n = 0; n < nn; n++) {
        double v = D2ab[n].value;
        if ( std::fabs(v) < rdm_screening_tol_ ) continue;
        int i = D2ab[n].i;
        int j = D2ab[n].j;
        int k = D2ab[n].k;
        int l = D2ab[n].l;
        if ( use_gemm ) {
            int ti = actmap[i], tj = actmap[j], tk = actmap[k], tl = actmap[l];
            if ( ti >= 0 && tj >= 0 && tk >= 0 && tl >= 0 ) {
                // pi = sum phi_i phi_j D(ij;kl) phi_k phi_l  -> bra=(ij), ket=(kl)
                D2actp[ti*nact + tj][tk*nact + tl] += v;
                n_aa++;
                continue;
            }
        }
        int Hi = symmetry_[i], Hj = symmetry_[j], Hk = symmetry_[k], Hl = symmetry_[l];
        hi.push_back(Hi); hj.push_back(Hj); hk.push_back(Hk); hl.push_back(Hl);
        li.push_back(i - pitzer_offset_[Hi]);
        lj.push_back(j - pitzer_offset_[Hj]);
        lk.push_back(k - pitzer_offset_[Hk]);
        ll.push_back(l - pitzer_offset_[Hl]);
        val.push_back(v);
    }
    const size_t nkeep = val.size();

    // cache the per-irrep block base pointers so the inner loop avoids the
    // virtual Matrix::pointer(h) dispatch on every term
    std::vector<double**> phi(nirrep_);
    for (int h = 0; h < nirrep_; h++) {
        phi[h]  = super_phi_->pointer(h);
    }

    const int *  hip = hi.data(); const int * hjp = hj.data();
    const int *  hkp = hk.data(); const int * hlp = hl.data();
    const int *  lip = li.data(); const int * ljp = lj.data();
    const int *  lkp = lk.data(); const int * llp = ll.data();
    const double * vp = val.data();
    double *** phib  = (double***)phi.data();

    // --- diagnostics: where does Build Pi spend its time? ---
    outfile->Printf("\n");
    outfile->Printf("        [pi] grid points (phi_points_) = %ld\n", (long int)phi_points_);
    outfile->Printf("        [pi] |D2ab| (raw)              = %zu\n", nn);
    outfile->Printf("        [pi] active orbitals (nact)    = %d  (GEMM %s)\n",
                    nact, use_gemm ? "ENGAGED" : "OFF -> all scalar");
    outfile->Printf("        [pi] active-active -> GEMM     = %zu\n", n_aa);
    outfile->Printf("        [pi] remainder    -> scalar    = %zu\n", nkeep);
    outfile->Flush();

    if ( !do_pi_gradient_ ) {

        // ---- scalar remainder: core-core, core-active (or everything if no GEMM) ----
        double t_scalar = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < phi_points_; p++) {

            double dum = 0.0;

            // pi(r) = D(mu,nu;lambda,sigma) phi(r,mu) phi(r,nu) phi(r,lambda) phi(r,sigma)
            for (size_t n = 0; n < nkeep; n++) {
                double phi_i = phib[hip[n]][p][lip[n]];
                double phi_j = phib[hjp[n]][p][ljp[n]];
                double phi_k = phib[hkp[n]][p][lkp[n]];
                double phi_l = phib[hlp[n]][p][llp[n]];
                dum += phi_i * phi_j * phi_k * phi_l * vp[n];
            }

            pi_p[p] = dum;
        }
        outfile->Printf("        [pi] scalar remainder time     = %8.1f s\n", omp_get_wtime() - t_scalar);
        outfile->Flush();

        // ---- dense active-active block via BLAS-3 GEMM ----
        double t_gemm = omp_get_wtime();
        if ( use_gemm && pairlen > 0 ) {

            // (irrep, local index) of each active orbital for gathering phi(active)
            std::vector<int> a_h(nact), a_l(nact);
            for (int c = 0; c < nact; c++) {
                int full = active_full_indices_[c];
                int h = symmetry_[full];
                a_h[c] = h;
                a_l[c] = full - pitzer_offset_[h];
            }

            const int BLOCK = 2048;
            auto PhiAct = std::make_shared<Matrix>("phi active", BLOCK, nact);
            auto Bmat   = std::make_shared<Matrix>("B pair products", BLOCK, pairlen);
            auto Tmat   = std::make_shared<Matrix>("T = B D2act^T", BLOCK, pairlen);
            double ** PAp = PhiAct->pointer();
            double ** Bp  = Bmat->pointer();

            for (int pstart = 0; pstart < phi_points_; pstart += BLOCK) {
                int np = ( phi_points_ - pstart < BLOCK ) ? (int)(phi_points_ - pstart) : BLOCK;

                // gather active phi values: PhiAct[pp][c] = phi_{active c}(pstart+pp)
                #pragma omp parallel for schedule(static)
                for (int pp = 0; pp < np; pp++) {
                    int p = pstart + pp;
                    for (int c = 0; c < nact; c++) PAp[pp][c] = phib[a_h[c]][p][a_l[c]];
                }

                // orbital-pair products: B[pp][t*nact+u] = phi_t * phi_u
                #pragma omp parallel for schedule(static)
                for (int pp = 0; pp < np; pp++) {
                    double * pa = PAp[pp];
                    double * b  = Bp[pp];
                    for (int t = 0; t < nact; t++) {
                        double pt = pa[t];
                        int off = t * nact;
                        for (int u = 0; u < nact; u++) b[off + u] = pt * pa[u];
                    }
                }

                // T = B * D2act^T  (D2act is symmetric in bra<->ket; transpose is exact)
                Tmat->gemm(false, true, 1.0, Bmat, D2act, 0.0);
                double ** Tp = Tmat->pointer();

                // pi_aa(p) += sum_{(t,u)} B[pp][(t,u)] * T[pp][(t,u)]
                #pragma omp parallel for schedule(static)
                for (int pp = 0; pp < np; pp++) {
                    double s = 0.0;
                    double * b  = Bp[pp];
                    double * tt = Tp[pp];
                    for (int q = 0; q < pairlen; q++) s += b[q] * tt[q];
                    pi_p[pstart + pp] += s;
                }
            }
        }
        outfile->Printf("        [pi] active-active GEMM time   = %8.1f s\n", omp_get_wtime() - t_gemm);
        outfile->Flush();

    } else {

        // value + gradient -- needed only for fully-translated functionals
        pi_x_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));
        pi_y_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));
        pi_z_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));

        double * pi_xp = pi_x_->pointer();
        double * pi_yp = pi_y_->pointer();
        double * pi_zp = pi_z_->pointer();

        std::vector<double**> phix(nirrep_), phiy(nirrep_), phiz(nirrep_);
        for (int h = 0; h < nirrep_; h++) {
            phix[h] = super_phi_x_->pointer(h);
            phiy[h] = super_phi_y_->pointer(h);
            phiz[h] = super_phi_z_->pointer(h);
        }
        double *** phixb = (double***)phix.data();
        double *** phiyb = (double***)phiy.data();
        double *** phizb = (double***)phiz.data();

        #pragma omp parallel for schedule(static)
        for (int p = 0; p < phi_points_; p++) {

            double dum = 0.0, dum_x = 0.0, dum_y = 0.0, dum_z = 0.0;

            for (size_t n = 0; n < nkeep; n++) {
                double phi_i = phib[hip[n]][p][lip[n]];
                double phi_j = phib[hjp[n]][p][ljp[n]];
                double phi_k = phib[hkp[n]][p][lkp[n]];
                double phi_l = phib[hlp[n]][p][llp[n]];
                double v     = vp[n];

                dum += phi_i * phi_j * phi_k * phi_l * v;

                dum_x += ( phixb[hip[n]][p][lip[n]] * phi_j * phi_k * phi_l
                         + phi_i * phixb[hjp[n]][p][ljp[n]] * phi_k * phi_l
                         + phi_i * phi_j * phixb[hkp[n]][p][lkp[n]] * phi_l
                         + phi_i * phi_j * phi_k * phixb[hlp[n]][p][llp[n]] ) * v;

                dum_y += ( phiyb[hip[n]][p][lip[n]] * phi_j * phi_k * phi_l
                         + phi_i * phiyb[hjp[n]][p][ljp[n]] * phi_k * phi_l
                         + phi_i * phi_j * phiyb[hkp[n]][p][lkp[n]] * phi_l
                         + phi_i * phi_j * phi_k * phiyb[hlp[n]][p][llp[n]] ) * v;

                dum_z += ( phizb[hip[n]][p][lip[n]] * phi_j * phi_k * phi_l
                         + phi_i * phizb[hjp[n]][p][ljp[n]] * phi_k * phi_l
                         + phi_i * phi_j * phizb[hkp[n]][p][lkp[n]] * phi_l
                         + phi_i * phi_j * phi_k * phizb[hlp[n]][p][llp[n]] ) * v;
            }

            pi_p[p]  = dum;
            pi_xp[p] = dum_x;
            pi_yp[p] = dum_y;
            pi_zp[p] = dum_z;
        }
    }
    outfile->Printf(" Done. <==\n\n");
}

void RealSpaceDensity::BuildRhoFast(){

    outfile->Printf("\n");
    outfile->Printf("    ==> Build Rho ...\n");

    rho_a_   = (std::shared_ptr<Vector>)(new Vector(phi_points_));
    rho_b_   = (std::shared_ptr<Vector>)(new Vector(phi_points_));
    rho_   = (std::shared_ptr<Vector>)(new Vector(phi_points_));
    tau_a_   = (std::shared_ptr<Vector>)(new Vector(phi_points_));
    tau_b_   = (std::shared_ptr<Vector>)(new Vector(phi_points_));

    double * rho_ap = rho_a_->pointer();
    double * rho_bp = rho_b_->pointer();
    double * rho_p = rho_->pointer();
    double * tau_ap = tau_a_->pointer();
    double * tau_bp = tau_b_->pointer();
    double * grid_wp = grid_w_->pointer();

    // Resolve symmetry / pitzer-offset arithmetic once per non-zero OPDM
    // element rather than at every grid point (it is invariant across p), and
    // screen out negligible elements (WriteOPDM stores the active block densely).
    const size_t na_raw = opdm_a_.size();
    const size_t nb_raw = opdm_b_.size();
    std::vector<int> a_hi, a_hj, a_li, a_lj;
    std::vector<int> b_hi, b_hj, b_li, b_lj;
    std::vector<double> a_val, b_val;
    a_hi.reserve(na_raw); a_hj.reserve(na_raw); a_li.reserve(na_raw); a_lj.reserve(na_raw); a_val.reserve(na_raw);
    b_hi.reserve(nb_raw); b_hj.reserve(nb_raw); b_li.reserve(nb_raw); b_lj.reserve(nb_raw); b_val.reserve(nb_raw);
    for (size_t n = 0; n < na_raw; n++) {
        if ( std::fabs(opdm_a_[n].value) < rdm_screening_tol_ ) continue;
        int i = opdm_a_[n].i, j = opdm_a_[n].j;
        int Hi = symmetry_[i], Hj = symmetry_[j];
        a_hi.push_back(Hi); a_hj.push_back(Hj);
        a_li.push_back(i - pitzer_offset_[Hi]);
        a_lj.push_back(j - pitzer_offset_[Hj]);
        a_val.push_back(opdm_a_[n].value);
    }
    for (size_t n = 0; n < nb_raw; n++) {
        if ( std::fabs(opdm_b_[n].value) < rdm_screening_tol_ ) continue;
        int i = opdm_b_[n].i, j = opdm_b_[n].j;
        int Hi = symmetry_[i], Hj = symmetry_[j];
        b_hi.push_back(Hi); b_hj.push_back(Hj);
        b_li.push_back(i - pitzer_offset_[Hi]);
        b_lj.push_back(j - pitzer_offset_[Hj]);
        b_val.push_back(opdm_b_[n].value);
    }
    const size_t na = a_val.size();
    const size_t nb = b_val.size();

    // cache per-irrep block base pointers (avoid virtual pointer() in the loop)
    std::vector<double**> phi(nirrep_), phix(nirrep_), phiy(nirrep_), phiz(nirrep_);
    for (int h = 0; h < nirrep_; h++) {
        phi[h]  = super_phi_->pointer(h);
        phix[h] = super_phi_x_->pointer(h);
        phiy[h] = super_phi_y_->pointer(h);
        phiz[h] = super_phi_z_->pointer(h);
    }
    double *** phib  = (double***)phi.data();
    double *** phixb = (double***)phix.data();
    double *** phiyb = (double***)phiy.data();
    double *** phizb = (double***)phiz.data();

    double temp_tot = 0.0;
    double temp_a = 0.0;
    double temp_b = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:temp_tot,temp_a,temp_b)
    for (int p = 0; p < phi_points_; p++) {

        // rho_a(r)
        double duma = 0.0;
        for (size_t n = 0; n < na; n++) {
            duma += phib[a_hi[n]][p][a_li[n]] *
                    phib[a_hj[n]][p][a_lj[n]] * a_val[n];
        }
        rho_ap[p] = duma;

        // rho_b(r)
        double dumb = 0.0;
        for (size_t n = 0; n < nb; n++) {
            dumb += phib[b_hi[n]][p][b_li[n]] *
                    phib[b_hj[n]][p][b_lj[n]] * b_val[n];
        }
        rho_bp[p] = dumb;

        rho_p[p] = duma + dumb;

        temp_tot += rho_p[p]  * grid_wp[p];
        temp_a   += duma * grid_wp[p];
        temp_b   += dumb * grid_wp[p];
    }

    outfile->Printf("\n");
    outfile->Printf("      Integrated total density = %20.12lf\n",temp_tot);
    outfile->Printf("      Integrated alpha density = %20.12lf\n",temp_a);
    outfile->Printf("      Integrated beta density  = %20.12lf\n",temp_b);
    outfile->Printf("\n");

    rho_a_x_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));
    rho_b_x_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));

    rho_a_y_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));
    rho_b_y_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));

    rho_a_z_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));
    rho_b_z_ = (std::shared_ptr<Vector>)(new Vector(phi_points_));

    double * rho_a_xp = rho_a_x_->pointer();
    double * rho_b_xp = rho_b_x_->pointer();

    double * rho_a_yp = rho_a_y_->pointer();
    double * rho_b_yp = rho_b_y_->pointer();

    double * rho_a_zp = rho_a_z_->pointer();
    double * rho_b_zp = rho_b_z_->pointer();

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < phi_points_; p++) {

        // rho'_a(r)

        double duma_x = 0.0;
        double duma_y = 0.0;
        double duma_z = 0.0;
        double dumta = 0.0;

        for (size_t n = 0; n < na; n++) {

            int hi = a_hi[n], hj = a_hj[n], ii = a_li[n], jj = a_lj[n];
            double v = a_val[n];

            double pi_  = phib[hi][p][ii],  pj_  = phib[hj][p][jj];
            double pix = phixb[hi][p][ii], pjx = phixb[hj][p][jj];
            double piy = phiyb[hi][p][ii], pjy = phiyb[hj][p][jj];
            double piz = phizb[hi][p][ii], pjz = phizb[hj][p][jj];

            duma_x += ( pix * pj_ + pi_ * pjx ) * v;
            duma_y += ( piy * pj_ + pi_ * pjy ) * v;
            duma_z += ( piz * pj_ + pi_ * pjz ) * v;
            dumta  += ( pix * pjx + piy * pjy + piz * pjz ) * v;
        }

        // rho'_b(r)

        double dumb_x = 0.0;
        double dumb_y = 0.0;
        double dumb_z = 0.0;
        double dumtb = 0.0;

        for (size_t n = 0; n < nb; n++) {

            int hi = b_hi[n], hj = b_hj[n], ii = b_li[n], jj = b_lj[n];
            double v = b_val[n];

            double pi_  = phib[hi][p][ii],  pj_  = phib[hj][p][jj];
            double pix = phixb[hi][p][ii], pjx = phixb[hj][p][jj];
            double piy = phiyb[hi][p][ii], pjy = phiyb[hj][p][jj];
            double piz = phizb[hi][p][ii], pjz = phizb[hj][p][jj];

            dumb_x += ( pix * pj_ + pi_ * pjx ) * v;
            dumb_y += ( piy * pj_ + pi_ * pjy ) * v;
            dumb_z += ( piz * pj_ + pi_ * pjz ) * v;
            dumtb  += ( pix * pjx + piy * pjy + piz * pjz ) * v;
        }

        rho_a_xp[p] = duma_x;
        rho_b_xp[p] = dumb_x;

        rho_a_yp[p] = duma_y;
        rho_b_yp[p] = dumb_y;

        rho_a_zp[p] = duma_z;
        rho_b_zp[p] = dumb_z;

        tau_ap[p] = 0.5 * dumta;
        tau_bp[p] = 0.5 * dumtb;
    }

    outfile->Printf("    ... Done. <==\n");
    outfile->Printf("\n");
}

void RealSpaceDensity::ReadTPDM() {

    std::shared_ptr<PSIO> psio (new PSIO());

    if ( !psio->exists(PSIF_V2RDM_D2AB) ) throw PsiException("No D2ab on disk",__FILE__,__LINE__);
    
    psio_address addr_ab = PSIO_ZERO;

    // ab
    psio->open(PSIF_V2RDM_D2AB,PSIO_OPEN_OLD);

    long int nab;
    psio->read_entry(PSIF_V2RDM_D2AB,"length",(char*)&nab,sizeof(long int));

    tpdm_ab_.resize(nab);
    memset((void*)&tpdm_ab_[0],'\0',nab * sizeof(tpdm));

    psio->read_entry(PSIF_V2RDM_D2AB,"D2ab",(char*)&tpdm_ab_[0],nab * sizeof(tpdm));

    // also grab the list of active orbitals (full-space pitzer indices) that
    // WriteTPDM stored. this lets BuildPiFast peel off the dense active-active
    // block and evaluate it as a BLAS-3 GEMM (see BuildPiFast).
    active_full_indices_.clear();
    if ( psio->tocscan(PSIF_V2RDM_D2AB,"ACTIVE ORBITALS") != nullptr ) {
        int nact = 0;
        psio->read_entry(PSIF_V2RDM_D2AB,"NUMBER ACTIVE ORBITALS",(char*)&nact,sizeof(int));
        if ( nact > 0 ) {
            active_full_indices_.resize(nact);
            psio->read_entry(PSIF_V2RDM_D2AB,"ACTIVE ORBITALS",(char*)&active_full_indices_[0],nact * sizeof(int));
        }
    }

    psio->close(PSIF_V2RDM_D2AB,1);
}

std::shared_ptr<Vector> RealSpaceDensity::xc_hole(double x, double y, double z) {

    // find appropriate grid point
    double * x_p = grid_x_->pointer();
    double * y_p = grid_y_->pointer();
    double * z_p = grid_z_->pointer();

    int p = 0;
    double min = 9e9;
    for (int myp = 0; myp < phi_points_; myp++) {
        double dx = x_p[myp] - x;
        double dy = y_p[myp] - y;
        double dz = z_p[myp] - z;
        double r2 = dx*dx + dy*dy + dz*dz;
        if ( r2 < min ) {
            min = r2;
            p = myp;
        }
    }
    BuildExchangeCorrelationHole(p);
    return xc_hole_;
}


} //end namespaces
