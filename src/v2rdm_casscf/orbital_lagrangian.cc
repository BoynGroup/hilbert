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

#include "v2rdm_solver.h"

#include <misc/omp.h>

using namespace psi;

namespace hilbert{ 

void v2RDMSolver::OrbitalLagrangian() {

    // unpack lagrangian, which was returned from orbital optimizer as
    // 
    // d-d / d-a / d-e / a-d / a-a / a-e
    // 
    // and the e-a, e-d, e-e blocks are zero

    Lagrangian_->zero();

    // d-d
    offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        double ** L_p = Lagrangian_->pointer(h);
        for (int i = 0; i < rstcpi_[h] + frzcpi_[h]; i++) {
            for (int j = 0; j < rstcpi_[h] + frzcpi_[h]; j++) {
                L_p[i][j] = X_[offset + i*(rstcpi_[h] + frzcpi_[h]) + j];
            }
        }
        offset += (rstcpi_[h] + frzcpi_[h])*(rstcpi_[h] + frzcpi_[h]);
    }

    // d-a
    for (int h = 0; h < nirrep_; h++) {
        double ** L_p = Lagrangian_->pointer(h);
        for (int i = 0; i < rstcpi_[h] + frzcpi_[h]; i++) {
            for (int j = 0; j < amopi_[h]; j++) {
                L_p[i][j + rstcpi_[h] + frzcpi_[h]] = X_[offset + i*amopi_[h] + j];
            }
        }
        offset += (rstcpi_[h] + frzcpi_[h])*amopi_[h];
    }

    // d-e
    for (int h = 0; h < nirrep_; h++) {
        double ** L_p = Lagrangian_->pointer(h);
        for (int i = 0; i < rstcpi_[h] + frzcpi_[h]; i++) {
            for (int j = 0; j < rstvpi_[h] + frzvpi_[h]; j++) {
                L_p[i][j + rstcpi_[h] + frzcpi_[h] + amopi_[h]] = X_[offset + i*(rstvpi_[h] + frzvpi_[h]) + j];
            }
        }
        offset += (rstcpi_[h] + frzcpi_[h])*(rstvpi_[h] + frzvpi_[h]);
    }

    // a-d
    for (int h = 0; h < nirrep_; h++) {
        double ** L_p = Lagrangian_->pointer(h);
        for (int i = 0; i < amopi_[h]; i++) {
            for (int j = 0; j < rstcpi_[h] + frzcpi_[h]; j++) {
                L_p[i+rstcpi_[h] + frzcpi_[h]][j] = X_[offset + i*(rstcpi_[h] + frzcpi_[h]) + j];
            }
        }
        offset += amopi_[h]*(rstcpi_[h] + frzcpi_[h]);
    }

    // a-a
    for (int h = 0; h < nirrep_; h++) {
        double ** L_p = Lagrangian_->pointer(h);
        for (int i = 0; i < amopi_[h]; i++) {
            for (int j = 0; j < amopi_[h]; j++) {
                L_p[i+rstcpi_[h] + frzcpi_[h]][j+rstcpi_[h] + frzcpi_[h]] = X_[offset + i*amopi_[h] + j];
            }
        }
        offset += amopi_[h]*amopi_[h];
    }

    // a-e
    for (int h = 0; h < nirrep_; h++) {
        double ** L_p = Lagrangian_->pointer(h);
        for (int i = 0; i < amopi_[h]; i++) {
            for (int j = 0; j < rstvpi_[h] + frzvpi_[h]; j++) {
                L_p[i+rstcpi_[h] + frzcpi_[h]][j+rstcpi_[h] + frzcpi_[h] + amopi_[h]] = X_[offset + i*(rstvpi_[h] + frzvpi_[h]) + j];
            }
        }
        offset += amopi_[h]*(rstvpi_[h] + frzvpi_[h]);
    }

    // transform orbital lagrangian to SO basis

    SharedMatrix Xmo(new Matrix(Lagrangian_));

    int symm = Lagrangian_->symmetry();

    double * temp = (double*)malloc(Ca_->max_ncol() * Ca_->max_nrow() * sizeof(double));

    Lagrangian_->zero();
    for (int h = 0; h < nirrep_; h++) {
        int nmol = Ca_->colspi()[h];
        int nmor = Ca_->colspi()[h^symm];
        int nsol = Ca_->rowspi()[h];
        int nsor = Ca_->rowspi()[h^symm];
        if (!nmol || !nmor || !nsol || !nsor) continue;
        double** Clp = Ca_->pointer(h);
        double** Crp = Ca_->pointer(h^symm);
        double** Xmop = Xmo->pointer(h^symm);
        double** Xsop = Lagrangian_->pointer(h^symm);
        C_DGEMM('N','T',nmol,nsor,nmor,1.0,Xmop[0],nmor,Crp[0],nmor,0.0,temp,nsor);
        C_DGEMM('N','N',nsol,nsor,nmol,1.0,Clp[0],nmol,temp,nsor,0.0,Xsop[0],nsor);
    }

    free(temp);

    //Lagrangian_->print();

}

}
