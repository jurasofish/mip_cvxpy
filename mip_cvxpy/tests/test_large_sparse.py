import numpy as np
import cvxpy as cp
from mip_cvxpy import PYTHON_MIP
from cvxpy.tests.base_test import BaseTest


class TestLargeSparse(BaseTest):
    def large_sparse(self, solver, n=int(1e5)):

        vars = cp.Variable(n, integer=True)

        objective = cp.Maximize(cp.sum(vars))
        constraints = [
            vars[0] == 1,
            vars <= np.array(list(range(10, 10 + n))),
        ]
        problem = cp.Problem(objective, constraints)

        solver_name = solver if isinstance(solver, str) else solver.name()
        print("solving with", solver_name)

        optimal_value = problem.solve(solver=solver)
        print(problem.status)
        print(optimal_value)
        return optimal_value

    def test_large_sparse(self):
        solutions = []
        for solver in (cp.CBC, PYTHON_MIP()):
            solutions.append(self.large_sparse(solver))
        # Check all solvers returned same solution
        assert np.allclose(solutions, solutions[0])
