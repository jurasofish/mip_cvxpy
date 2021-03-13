import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
import numpy as np
import time
from mip_cvxpy import PYTHON_MIP


def benchmark(func, iters=1, name=None):
    vals = []
    for _ in range(iters):
        start = time.perf_counter()
        func()
        vals.append(time.perf_counter() - start)
    name = func.__name__ if name is None else name
    print(
        "{:s}: avg={:.3e} s , std={:.3e} s ({:d} iterations)".format(
            name, np.mean(vals), np.std(vals), iters
        )
    )


class TestBenchmarks(BaseTest):
    def test_small_lp(self):
        m = 200
        n = 200
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        c = np.random.rand(n)

        def small_lp(solver_name):
            if solver_name == "PYTHON_MIP":
                solver = PYTHON_MIP()
            else:
                solver = solver_name
            x = cp.Variable(n)
            cost = cp.matmul(c, x)
            constraints = [A @ x <= b]
            problem = cp.Problem(cp.Minimize(cost), constraints)
            optimal_value = problem.solve(solver=solver)

        f_cylp = lambda: small_lp(cp.CBC)
        f_mip = lambda: small_lp("PYTHON_MIP")

        benchmark(f_cylp, iters=10, name="small_lp_CBC_first")
        benchmark(f_cylp, iters=10, name="small_lp_CBC_second")
        benchmark(f_mip, iters=10, name="small_lp_PYTHON_MIP_first")
        benchmark(f_mip, iters=10, name="small_lp_PYTHON_MIP_second")
