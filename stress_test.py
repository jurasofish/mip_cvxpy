import numpy as np
import cvxpy as cp
from mip_conif import PYTHON_MIP
import mip  # So that subsequent imports are quick.


def run_sample_optimization(solver):

    n = int(1e6)
    vars = cp.Variable(n, integer=True)

    objective = cp.Maximize(cp.sum(vars))
    constraints = [
        vars[0] == 1,
        vars <= np.linspace(10, n + 10, num=n),
    ]
    problem = cp.Problem(objective, constraints)

    solver_name = solver if isinstance(solver, str) else solver.name()
    print("solving with", solver_name)

    optimal_value = problem.solve(solver=solver)
    print(problem.status)
    # print(vars.value)


def main():
    solver = PYTHON_MIP()
    run_sample_optimization(solver)


if __name__ == "__main__":
    main()
