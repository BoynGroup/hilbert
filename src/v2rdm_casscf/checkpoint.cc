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

#include <psi4/psi4-dec.h>
#include <psi4/liboptions/liboptions.h>
#include <psi4/libqt/qt.h>
#include <psi4/libtrans/integraltransform.h>
#include <psi4/libtrans/mospace.h>
#include <psi4/libmints/wavefunction.h>
#include <psi4/libmints/vector.h>
#include <psi4/libmints/matrix.h>
#include <psi4/libpsi4util/PsiOutStream.h>

#include"v2rdm_solver.h"

#include <misc/omp.h>

#include<bpsdp_solver.h>

using namespace psi;

namespace hilbert{ 


void v2RDMSolver::WriteCheckpointFile() {

    // Update Ca_/Cb_
    //UpdateTransformationMatrix();

    std::shared_ptr<PSIO> psio ( new PSIO() );

    psio->open(PSIF_V2RDM_CHECKPOINT,PSIO_OPEN_NEW);

    // mu
    double mu = sdp_->get_mu();
    psio->write_entry(PSIF_V2RDM_CHECKPOINT,"MU",(char*)(&mu),sizeof(double));

    // x
    psio->write_entry(PSIF_V2RDM_CHECKPOINT,"PRIMAL",(char*)x->pointer(),n_primal_*sizeof(double));

    // y
    //psio->write_entry(PSIF_V2RDM_CHECKPOINT,"DUAL 1",(char*)sdp_->get_y()->pointer(),n_dual_*sizeof(double));
    psio->write_entry(PSIF_V2RDM_CHECKPOINT,"DUAL 1",(char*)sdp_->get_y(),n_dual_*sizeof(double));

    // z
    //psio->write_entry(PSIF_V2RDM_CHECKPOINT,"DUAL 2",(char*)sdp_->get_z()->pointer(),n_primal_*sizeof(double));
    psio->write_entry(PSIF_V2RDM_CHECKPOINT,"DUAL 2",(char*)sdp_->get_z(),n_primal_*sizeof(double));

    // mo/mo' transformation matrix
    psio_address addr = PSIO_ZERO;
    for (int h = 0; h < nirrep_; h++) {
        if ( nmopi_[h] == 0 ) continue;
        double ** np = newMO_->pointer(h);
        psio->write(PSIF_V2RDM_CHECKPOINT,"MO TO MO' TRANSFORMATION MATRIX",(char*)&(np[0][0]),nmopi_[h]*nmopi_[h]*sizeof(double),addr,&addr);
    }
    // so/mo transformation matrix
    addr = PSIO_ZERO;
    for (int h = 0; h < nirrep_; h++) {
        if ( nsopi_[h] == 0 || nmopi_[h] == 0 ) continue;
        double ** cp = Ca_->pointer(h);
        psio->write(PSIF_V2RDM_CHECKPOINT,"SO TO MO TRANSFORMATION MATRIX",(char*)&(cp[0][0]),nsopi_[h]*nmopi_[h]*sizeof(double),addr,&addr);
    }

    psio->close(PSIF_V2RDM_CHECKPOINT,1);
}

void v2RDMSolver::ReadFromCheckpointFile() {

    std::shared_ptr<PSIO> psio ( new PSIO() );

    if ( !psio->exists(PSIF_V2RDM_CHECKPOINT) ) {
        return;
    }

    outfile->Printf("\n");
    outfile->Printf("    ==> Restarting from checkpoint file <==\n");

    psio->open(PSIF_V2RDM_CHECKPOINT,PSIO_OPEN_OLD);

    // mu
    double mu = 0.1;
    psio->read_entry(PSIF_V2RDM_CHECKPOINT,"MU",(char*)(&mu),sizeof(double));
    sdp_->set_mu(mu);

    // x
    psio->read_entry(PSIF_V2RDM_CHECKPOINT,"PRIMAL",(char*)x->pointer(),n_primal_*sizeof(double));

    // y
    std::shared_ptr<Vector> y (new Vector(n_dual_));
    psio->read_entry(PSIF_V2RDM_CHECKPOINT,"DUAL 1",(char*)y->pointer(),n_dual_*sizeof(double));
    sdp_->set_y(y->pointer());
    y.reset();

    // z
    std::shared_ptr<Vector> z (new Vector(n_primal_));
    psio->read_entry(PSIF_V2RDM_CHECKPOINT,"DUAL 2",(char*)z->pointer(),n_primal_*sizeof(double));
    sdp_->set_z(z->pointer());
    z.reset();

    psio->close(PSIF_V2RDM_CHECKPOINT,1);
}

void v2RDMSolver::ReadOrbitalsFromCheckpointFile() {

    std::shared_ptr<PSIO> psio ( new PSIO() );

    if ( !psio->exists(PSIF_V2RDM_CHECKPOINT) ) {
        return;
    }

    psio->open(PSIF_V2RDM_CHECKPOINT,PSIO_OPEN_OLD);

    // mo/mo' transformation matrix
    psio_address addr = PSIO_ZERO;
    for (int h = 0; h < nirrep_; h++) {
        if ( nmopi_[h] == 0 ) continue;
        double ** np = newMO_->pointer(h);
        psio->read(PSIF_V2RDM_CHECKPOINT,"MO TO MO' TRANSFORMATION MATRIX",(char*)&(np[0][0]),nmopi_[h]*nmopi_[h]*sizeof(double),addr,&addr);
    }

    SharedMatrix tempCa ( new Matrix(Ca_));
    // so/mo transformation matrix ( to check phase )
    addr = PSIO_ZERO;
    for (int h = 0; h < nirrep_; h++) {
        if ( nsopi_[h] == 0 || nmopi_[h] == 0 ) continue;
        double ** tp = tempCa->pointer(h);
        psio->read(PSIF_V2RDM_CHECKPOINT,"SO TO MO TRANSFORMATION MATRIX",(char*)&(tp[0][0]),nsopi_[h]*nmopi_[h]*sizeof(double),addr,&addr);
    }

    // transform mo index to mo' basis
    SharedMatrix temp (new Matrix(Ca_));
    for (int h = 0; h < nirrep_; h++) {

        if ( nmopi_[h] == 0 ) continue;

        double ** tp = temp->pointer(h);
        double ** cp = Ca_->pointer(h);
        double ** np = newMO_->pointer(h);

        //F_DGEMM('n','t',nmo_,nsopi_[h],nao,1.0,temp,nmo_,&(AO2SO_->pointer(h)[0][0]),nsopi_[h],0.0,&(tp[0][0]),nmo_);

        // check phase:
        for (int i = 0; i < nmopi_[h]; i++) {
            double max  = -9e9;
            int mumax   = 0;
            for (int mu = 0; mu < nsopi_[h]; mu++) {
                if ( fabs(cp[mu][i]) > max ) {
                    mumax = mu;
                    max = fabs(cp[mu][i]);
                }
            }
            if ( cp[mumax][i] * tempCa->pointer(h)[mumax][i] < 0.0 ) {
                for (int j = 0; j < nmopi_[h]; j++) {
                    np[j][i] *= -1;
                }
            }
        }

        for (int mu = 0; mu < nsopi_[h]; mu++) {
            for (int i = 0; i < nmopi_[h]; i++) {
                double dum = 0.0;
                for (int j = 0; j < nmopi_[h]; j++) {
                    dum += cp[mu][j] * np[i][j];
                }
                tp[mu][i] = dum;
            }
        }
    }
    Ca_->copy(temp);
    Cb_->copy(temp);

    psio->close(PSIF_V2RDM_CHECKPOINT,1);
}

}
