import numpy as np
import cvxpy as cp
import mip


def run_sample_optimization_cvxopt():
    demand = np.array([[100, 500, 30], [20, 200, 50], [150, 15, 35], [10, 5, 25]])
    product_supply = np.array([550, 200, 170, 40])
    allocation = cp.Variable(demand.shape, integer=True)
    objective = cp.Maximize(cp.sum(allocation/demand))
    constraints =[cp.sum(allocation, axis=1) <= product_supply,
                allocation <= demand,
                allocation >= 0]
    problem = cp.Problem(objective, constraints)

    optimal_value = problem.solve(solver=cp.GLPK_MI)

    print('CVXOPT')
    print('product supply:', product_supply)
    print('demand:\n', demand)
    print('allocation:\n', allocation.value)
    print('calculated score:', optimal_value)
    return product_supply, demand, allocation.value, optimal_value


run_sample_optimization_cvxopt()
