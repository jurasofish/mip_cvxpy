import numpy as np
import cvxpy as cp
from mip_cvxpy import PYTHON_MIP


def run_sample_optimization(solver):
    demand = np.array([[100, 500, 30], [20, 200, 50], [150, 15, 35], [10, 5, 25]])
    product_supply = np.array([550, 200, 170, 40])
    allocation = cp.Variable(demand.shape, integer=True)
    a = cp.Variable(shape=1)  # To help with debugging
    objective = cp.Maximize(cp.sum(allocation / demand) + cp.sum(a))
    constraints = [
        cp.sum(allocation, axis=1) <= product_supply,
        allocation <= demand,
        allocation >= 0,
        cp.sum(a) == 0,
    ]
    problem = cp.Problem(objective, constraints)

    solver_name = solver if isinstance(solver, str) else solver.name()
    print("solving with", solver_name)

    optimal_value = problem.solve(solver=solver)

    print("product supply:", product_supply)
    print("demand:\n", demand)
    print("allocation:\n", allocation.value)
    print("calculated score:", optimal_value)
    return product_supply, demand, allocation.value, optimal_value


def main():
    for solver in (cp.GLPK_MI, PYTHON_MIP()):
        run_sample_optimization(solver)


if __name__ == "__main__":
    main()
