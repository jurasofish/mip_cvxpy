# mip-cvxpy

  [![PyPI](https://img.shields.io/pypi/v/mip-cvxpy.svg)](https://pypi.org/project/mip-cvxpy/) 
| [![Test](https://github.com/jurasofish/mip_cvxpy/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/jurasofish/mip_cvxpy/actions/workflows/test.yml)

This package allows you to solve [CVXPY](https://github.com/cvxgrp/cvxpy)
problems using the [python-mip](https://github.com/coin-or/python-mip)
package as a backend solver. It works for mixed integer linear problems.

This allows you to use CBC from CVXPY without needing to manually install
CBC. By default, CVXOPT calls [CyLP](https://github.com/coin-or/CyLP)
to use CBC and requires CBC to be manually installed. python-mip, on
the other hand, comes with CBC bundled through pypi.

This package is based heavily off the [CyLP/CBC interface](
https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/reductions/solvers/conic_solvers/cbc_conif.py)
and is slower: on smaller problems mip_cvxpy interface takes perhaps 1.3x as long
as CyLP, and on larger problems perhaps 5x as long (see the benchmark in the
test suite). CyLP has a significant advantage in natively supporting sparse 
matrices and vectorisation.


## Installation

Install from pypi

```pip install mip_cvxpy```

## Usage

Use as a [custom solver](https://www.cvxpy.org/tutorial/advanced/index.html#custom-solvers)

```python
import numpy as np
import cvxpy as cp
from mip_cvxpy import PYTHON_MIP

n = int(1e3)
vars = cp.Variable(n, integer=True)
objective = cp.Maximize(cp.sum(vars))
constraints = [
    vars[0] == 1,
    vars <= np.linspace(10, n + 10, num=n),
]
problem = cp.Problem(objective, constraints)

optimal_value = problem.solve(solver=PYTHON_MIP())
print(problem.status)
```

#### Additional solver options

You can pass additional solver options like

```python
optimal_value = problem.solve(solver=solver, max_seconds=10, other_option=7)
```

This is equivalent to 

```python
import mip
m = mip.Model()
m.max_seconds=10
m.other_option=7
...
```