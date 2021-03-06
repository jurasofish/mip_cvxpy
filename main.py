import numpy as np
import cvxpy as cp
import mip
from mip_conif import PythonMIP


def run_sample_optimization_cvxopt():
    demand = np.array([[100, 500, 30], [20, 200, 50], [150, 15, 35], [10, 5, 25]])
    product_supply = np.array([550, 200, 170, 40])
    allocation = cp.Variable(demand.shape, integer=True)
    a = cp.Variable(shape=1)
    objective = cp.Maximize(cp.sum(allocation/demand) + cp.sum(a))
    constraints =[
        cp.sum(allocation, axis=1) <= product_supply,
        allocation <= demand,
        allocation >= 0,
        cp.sum(a) == 2,
    ]
    problem = cp.Problem(objective, constraints)

    optimal_value = problem.solve(solver=cp.GLPK_MI)

    print('CVXOPT')
    print('product supply:', product_supply)
    print('demand:\n', demand)
    print('allocation:\n', allocation.value)
    print('calculated score:', optimal_value)
    return product_supply, demand, allocation.value, optimal_value


def run_sample_optimization_mip():
    demand = np.array([[100, 500, 30], [20, 200, 50], [150, 15, 35], [10, 5, 25]])
    product_supply = np.array([550, 200, 170, 40])
    allocation = cp.Variable(demand.shape, integer=True)
    a = cp.Variable(shape=1)
    objective = cp.Maximize(cp.sum(allocation/demand) + cp.sum(a))
    constraints =[
        cp.sum(allocation, axis=1) <= product_supply,
        allocation <= demand,
        allocation >= 0,
        cp.sum(a) == 2,
    ]
    problem = cp.Problem(objective, constraints)

    optimal_value = problem.solve(solver=PythonMIP())

    print('CVXOPT')
    print('product supply:', product_supply)
    print('demand:\n', demand)
    print('allocation:\n', allocation.value)
    print('calculated score:', optimal_value)
    return product_supply, demand, allocation.value, optimal_value


run_sample_optimization_cvxopt()
run_sample_optimization_mip()
