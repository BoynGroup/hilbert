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

#include"v2rdm_doci_solver.h"

#include <misc/omp.h>

using namespace psi;

namespace hilbert{

// G2 portion of A.x (with symmetry)
void v2RDM_DOCISolver::G2_constraints_Au(double* A,double* u){

    int aux = 0;
    // Eq (35) ... note AED symmetried
    for (int i = 0; i < amo_; i++) {
        for (int j = i + 1; j < amo_; j++) {
            //if ( i == j ) continue;

            double dum;

            dum    = -u[g2s2off_ + aux + 0];     // - G2s2(ij,ij)
            dum   +=  u[d1off_ + i];             // + D1(i)
            dum   -=  0.5 * u[d2s2off_ + i*amo_+ j];   // - D2s2(ij,ij)
            dum   -=  0.5 * u[d2s2off_ + j*amo_+ i];   // - D2s2(ij,ij)

            A[offset + 0] = dum;

            dum    = -u[g2s2off_ + aux + 1];     // - G2s2(ij,ij)
            dum   +=  0.5 * u[d2s0off_ + i*amo_+ j];   // + D2s0(ij,ij)
            dum   +=  0.5 * u[d2s0off_ + j*amo_+ i];   // + D2s0(ij,ij)

            A[offset + 1] = dum;

            dum    = -u[g2s2off_ + aux + 2];     // - G2s2(ij,ij)
            dum   +=  0.5 * u[d2s0off_ + i*amo_+ j];   // + D2s0(ij,ij)
            dum   +=  0.5 * u[d2s0off_ + j*amo_+ i];   // + D2s0(ij,ij)

            A[offset + 2] = dum;

            dum    = -u[g2s2off_ + aux + 3];     // - G2s2(ij,ij)
            dum   +=  u[d1off_ + j];             // + D1(i)
            dum   -=  0.5 * u[d2s2off_ + i*amo_+ j];   // - D2s2(ij,ij)
            dum   -=  0.5 * u[d2s2off_ + j*amo_+ i];   // - D2s2(ij,ij)

            A[offset + 3] = dum;

            offset += 2*2;
            aux += 2*2;
        }
    }

    // G2s0 (note AED symmetrized):
    for (int i = 0; i < amo_; i++) {
        for (int j = 0; j < amo_; j++) {
            double dum = -u[g2s0off_ + i*amo_+j];
            if ( i != j ) {
                dum += 0.5 * u[d2s2off_ + i*amo_+j];
                dum += 0.5 * u[d2s2off_ + j*amo_+i];
            }else {
                dum += u[d1off_ + i];
            }

            A[offset + i*amo_+j] = dum;

        }
    }
    offset += amo_*amo_;

}

// G2 portion of A^T.y (with symmetry)
void v2RDM_DOCISolver::G2_constraints_ATu(double* A,double* u){

    // Eq (35) ... note AED symmetrized
    int aux = 0;
    for (int i = 0; i < amo_; i++) {
        for (int j = i + 1; j < amo_; j++) {
            //if ( i == j ) continue;

            double dum = u[offset + 0];

            A[g2s2off_ + aux + 0]   -= dum;
            A[d1off_ + i]           += dum;
            A[d2s2off_ + i*amo_+j]  -= 0.5 * dum;
            A[d2s2off_ + j*amo_+i]  -= 0.5 * dum;

            dum = u[offset + 1];

            A[g2s2off_ + aux + 1]   -= dum;
            A[d2s0off_ + i*amo_+ j] += 0.5 * dum;
            A[d2s0off_ + j*amo_+ i] += 0.5 * dum;

            dum = u[offset + 2];

            A[g2s2off_ + aux + 2]   -= dum;
            A[d2s0off_ + i*amo_+ j] += 0.5 * dum;
            A[d2s0off_ + j*amo_+ i] += 0.5 * dum;

            dum = u[offset + 3];
            A[g2s2off_ + aux + 3]   -= dum;
            A[d1off_ + j]           += dum;
            A[d2s2off_ + j*amo_+i]  -= 0.5 * dum;
            A[d2s2off_ + i*amo_+j]  -= 0.5 * dum;

            aux    += 2*2;
            offset += 2*2;
        }
    }

    // G2s0 ... note AED symmetrized:
    for (int i = 0; i < amo_; i++) {
        for (int j = 0; j < amo_; j++) {
            double dum = u[offset + i*amo_+j];
            A[g2s0off_ + i*amo_+j] -= dum;
            if ( i != j ) {
                A[d2s2off_ + i*amo_+j] += 0.5 * dum;
                A[d2s2off_ + j*amo_+i] += 0.5 * dum;
            }else {
                A[d1off_ + i] += dum;
            }

        }
    }
    offset += amo_*amo_;

}

} // end namespaces
