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

#ifndef CVXPY_SOLVER_H
#define CVXPY_SOLVER_H

#include <sdp_solver.h>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace libsdp {

class CVXPYSolver : public SDPSolver {
public:
    CVXPYSolver(long int n_primal, long int n_dual, SDPOptions options, const std::string& cvxpy_solver_name)
        : SDPSolver(n_primal, n_dual, options), cvxpy_solver_name_(cvxpy_solver_name) {}

    ~CVXPYSolver() {}

    void solve(double * x,
               double * b,
               double * c,
               std::vector<int> primal_block_dim,
               int maxiter,
               SDPCallbackFunction evaluate_Au,
               SDPCallbackFunction evaluate_ATu,
               SDPProgressMonitorFunction progress_monitor,
               int print_level,
               void * data) override {
        
        if (print_level > 0) {
            std::cout << "  ==> Solving SDP with CVXPY (Backend: " 
                      << (cvxpy_solver_name_.empty() ? "Default" : cvxpy_solver_name_) 
                      << ") <==" << std::endl;
        }

        // Construct the explicit A matrix of size n_dual_ x n_primal_
        // A^T e_i is the i-th column of A^T, i.e., the i-th row of A.
        std::vector<double> A(n_dual_ * n_primal_, 0.0);
        std::vector<double> e(n_dual_, 0.0);
        for (long int i = 0; i < n_dual_; ++i) {
            e[i] = 1.0;
            evaluate_ATu(&A[i * n_primal_], e.data(), data);
            e[i] = 0.0;
        }

        // Acquire GIL before calling python
        pybind11::gil_scoped_acquire acquire;

        try {
            pybind11::object hilbert = pybind11::module_::import("hilbert");
            pybind11::object cvxpy_solve = hilbert.attr("cvxpy_solve");

            // Convert c, A, b to numpy arrays
            pybind11::array_t<double> py_c(n_primal_, c);
            pybind11::array_t<double> py_A({n_dual_, n_primal_}, A.data());
            pybind11::array_t<double> py_b(n_dual_, b);

            pybind11::tuple result = cvxpy_solve(py_c, py_A, py_b, primal_block_dim, cvxpy_solver_name_, print_level > 0);

            pybind11::array_t<double> py_x = result[0].cast<pybind11::array_t<double>>();
            pybind11::array_t<double> py_y = result[1].cast<pybind11::array_t<double>>();

            // Copy values back to C++ pointers
            std::memcpy(x, py_x.data(), n_primal_ * sizeof(double));
            std::memcpy(y_, py_y.data(), n_dual_ * sizeof(double));

            // Compute dual slack z_: z = c - A^T y
            evaluate_ATu(z_, y_, data); // Computes A^T y and stores it in z_
            for (long int i = 0; i < n_primal_; ++i) {
                z_[i] = c[i] - z_[i];
            }

            // Calculate objective values and error metrics for convergence monitoring
            double objective_primal = 0.0;
            for (long int i = 0; i < n_primal_; ++i) {
                objective_primal += c[i] * x[i];
            }
            double objective_dual = 0.0;
            for (long int i = 0; i < n_dual_; ++i) {
                objective_dual += b[i] * y_[i];
            }

            // Primal constraint residual norm: ||Ax - b||
            evaluate_Au(Au_, x, data);
            double primal_err_sq = 0.0;
            for (long int i = 0; i < n_dual_; ++i) {
                double diff = Au_[i] - b[i];
                primal_err_sq += diff * diff;
            }
            primal_error_ = std::sqrt(primal_err_sq);

            // Dual constraint residual norm: ||A^T y + z - c||
            evaluate_ATu(ATu_, y_, data);
            double dual_err_sq = 0.0;
            for (long int i = 0; i < n_primal_; ++i) {
                double diff = ATu_[i] + z_[i] - c[i];
                dual_err_sq += diff * diff;
            }
            dual_error_ = std::sqrt(dual_err_sq);

            is_converged_ = true;
            oiter_ = 1;
            iiter_total_ = 1;

            // Notify progress monitor so that upper-level solvers update convergence flags
            if (progress_monitor) {
                progress_monitor(print_level, oiter_, iiter_total_, objective_primal, objective_dual, mu_, primal_error_, dual_error_, data);
            }

        } catch (const std::exception& ex) {
            std::cerr << "Error in CVXPY Solver: " << ex.what() << std::endl;
            is_converged_ = false;
            throw;
        }
    }

private:
    std::string cvxpy_solver_name_;
};

}

#endif // CVXPY_SOLVER_H
