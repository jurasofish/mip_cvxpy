"""

Based on
    https://github.com/cvxgrp/cvxpy/
    blob/5d5c7d606e39b3ea4b54391f772c7e3dc38ede20/cvxpy/reductions/
    solvers/conic_solvers/cbc_conif.py

Copyright 2016 Sascha-Dominic Schnug
Copyright 2021 Michael Jurasovic

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy.settings as s
from cvxpy.reductions.solvers.conic_solvers.cbc_conif import (
    CBC,
    dims_to_solver_dict,
)
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solution import Solution, failure_solution
import numpy as np


class PYTHON_MIP(CBC):  # uppercase consistent with cvxopt
    """ An interface to the python-mip solver
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    def name(self):
        """The name of the solver.
        """
        return "PYTHON_MIP"

    def import_solver(self):
        """Imports the solver.
        """
        import mip
        _ = mip  # For flake8

    def accepts(self, problem):
        """Can python-mip solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in self.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(PYTHON_MIP, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['primal']}
            return Solution(status, opt_val, primal_vars, None, {})
        else:
            return failure_solution(status)

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        # Import basic modelling tools of cylp
        import mip

        c = data[s.C]
        b = data[s.B]
        A = data[s.A]
        dims = dims_to_solver_dict(data[s.DIMS])

        n = c.shape[0]

        # Problem
        model = mip.Model()

        # Variables
        x = []
        bool_idxs = set(data[s.BOOL_IDX])
        int_idxs = set(data[s.INT_IDX])
        for i in range(n):
            if i in bool_idxs:
                x.append(model.add_var(var_type=mip.BINARY))
            elif i in int_idxs:
                x.append(model.add_var(var_type=mip.INTEGER))
            else:
                x.append(model.add_var())

        # Constraints
        # eq
        def add_eq_constraints(_model):
            coeffs = A[0:dims[s.EQ_DIM], :]
            vals = b[0:dims[s.EQ_DIM]]
            for i in range(coeffs.shape[0]):
                coeff_list = np.squeeze(np.array(coeffs[i].todense())).tolist()
                expr = mip.LinExpr(variables=x, coeffs=coeff_list)
                _model += expr == vals[i]
        add_eq_constraints(model)

        # leq
        def add_leq_constraints(_model):
            leq_start = dims[s.EQ_DIM]
            leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
            coeffs = A[leq_start:leq_end, :].tocsr()  # CSR format faster as we're going row by row
            vals = b[leq_start:leq_end]
            indices, indptr, data = coeffs.indices, coeffs.indptr, coeffs.data
            for i in range(coeffs.shape[0]):
                col_idxs = indices[indptr[i]:indptr[i+1]]
                row_vals = data[indptr[i]:indptr[i+1]]
                vars = [x[j] for j in col_idxs]
                expr = mip.LinExpr(variables=vars, coeffs=row_vals.tolist())
                _model += expr <= vals[i]
        add_leq_constraints(model)

        # Objective
        model.objective = mip.minimize(mip.LinExpr(variables=x, coeffs=c.tolist()))

        model.verbose = verbose
        status = model.optimize()

        status_map = {
            mip.OptimizationStatus.OPTIMAL: s.OPTIMAL,
            mip.OptimizationStatus.INFEASIBLE: s.INFEASIBLE,
            mip.OptimizationStatus.INT_INFEASIBLE: s.INFEASIBLE,
            mip.OptimizationStatus.NO_SOLUTION_FOUND: s.INFEASIBLE,
            mip.OptimizationStatus.ERROR: s.SOLVER_ERROR,
            mip.OptimizationStatus.UNBOUNDED: s.UNBOUNDED,
            mip.OptimizationStatus.CUTOFF: s.INFEASIBLE,
            mip.OptimizationStatus.FEASIBLE: s.OPTIMAL_INACCURATE,
            mip.OptimizationStatus.LOADED: s.SOLVER_ERROR,  # No match really
        }

        solution = {
            "status": status_map[status],
            "primal": [var.x for var in x],
            "value": model.objective_value,
        }

        return solution
