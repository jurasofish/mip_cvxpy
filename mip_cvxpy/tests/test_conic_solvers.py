"""
Copyright 2019, the CVXPY developers.

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

import math
import unittest

import numpy as np
import scipy.linalg as la
import pytest

import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS, INSTALLED_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
    StandardTestECPs,
    StandardTestLPs,
    StandardTestMixedCPs,
    StandardTestSDPs,
    StandardTestSOCPs,
    StandardTestPCPs,
)


@unittest.skipUnless("CBC" in INSTALLED_SOLVERS, "CBC is not installed.")
class TestCBC(BaseTest):
    def setUp(self):
        self.a = cp.Variable(name="a")
        self.b = cp.Variable(name="b")
        self.c = cp.Variable(name="c")

        self.x = cp.Variable(2, name="x")
        self.y = cp.Variable(3, name="y")
        self.z = cp.Variable(2, name="z")

        self.A = cp.Variable((2, 2), name="A")
        self.B = cp.Variable((2, 2), name="B")
        self.C = cp.Variable((3, 2), name="C")

    def test_options(self):
        """Test that all the cvx.CBC solver options work."""
        prob = cp.Problem(
            cp.Minimize(cp.norm(self.x, 1)), [self.x == cp.Variable(2, boolean=True)]
        )
        if cp.CBC in INSTALLED_SOLVERS:
            for i in range(2):
                # Some cut-generators seem to be buggy for now -> set to false
                # prob.solve(solver=cvx.CBC, verbose=True, GomoryCuts=True, MIRCuts=True,
                #            MIRCuts2=True, TwoMIRCuts=True, ResidualCapacityCuts=True,
                #            KnapsackCuts=True, FlowCoverCuts=True, CliqueCuts=True,
                #            LiftProjectCuts=True, AllDifferentCuts=False, OddHoleCuts=True,
                #            RedSplitCuts=False, LandPCuts=False, PreProcessCuts=False,
                #            ProbingCuts=True, SimpleRoundingCuts=True)
                prob.solve(solver=cp.CBC, verbose=True, maximumSeconds=100)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])
        else:
            with self.assertRaises(Exception) as cm:
                prob.solve(solver=cp.CBC)
                self.assertEqual(
                    str(cm.exception), "The solver %s is not installed." % cp.CBC
                )

    def test_cbc_lp_0(self):
        StandardTestLPs.test_lp_0(solver="CBC", duals=False)

    def test_cbc_lp_1(self):
        StandardTestLPs.test_lp_1(solver="CBC", duals=False)

    def test_cbc_lp_2(self):
        StandardTestLPs.test_lp_2(solver="CBC", duals=False)

    def test_cbc_lp_3(self):
        StandardTestLPs.test_lp_3(solver="CBC")

    def test_cbc_lp_4(self):
        StandardTestLPs.test_lp_4(solver="CBC")

    def test_cbc_lp_5(self):
        StandardTestLPs.test_lp_5(solver="CBC")

    def test_cbc_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver="CBC")

    def test_cbc_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver="CBC")

    def test_cbc_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver="CBC")

    def test_cbc_mi_lp_3(self):
        StandardTestLPs.test_mi_lp_3(solver="CBC")
