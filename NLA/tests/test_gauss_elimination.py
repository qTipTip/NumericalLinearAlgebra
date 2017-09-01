from unittest import TestCase
import numpy as np

from NLA.matrices import rforwardsolve, rbackwardsolve, cforwardsolve, cbackwardsolve, L1U


class TestTriangularSolves(TestCase):

    def test_forwardsolve_lower_triangular_rows(self):

        A = np.array([
            [1.0, 0, 0],
            [1.0, 1.0, 0],
            [0, 1.0, 1.0]
        ])

        b = np.array([1.0, 2.0, 3.0])
        d = 1

        expected_x = np.array([1, 1, 2])
        computed_x = rforwardsolve(A, b, d)

        np.testing.assert_almost_equal(expected_x, computed_x)

    def test_forwardsolve_lower_triangular_columns(self):

        A = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1]
        ])

        b = np.array([1, 2, 3])
        d = 1

        expected_x = np.array([1, 1, 2])
        computed_x = cforwardsolve(A, b, d)

        np.testing.assert_almost_equal(expected_x, computed_x)

    def test_backwardsolve_upper_triangular_rows(self):
        A = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1]
        ])

        b = np.array([3, 2, 1])
        d = 1

        expected_x = np.array([2, 1, 1])
        computed_x = rbackwardsolve(A, b, d)

        np.testing.assert_almost_equal(expected_x, computed_x)


    def test_backwardsolve_upper_triangular_columns(self):
        A = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1]
        ])

        b = np.array([3, 2, 1])
        d = 1

        expected_x = np.array([2, 1, 1])
        computed_x = cbackwardsolve(A, b, d)

        np.testing.assert_almost_equal(expected_x, computed_x)

    def test_LU1(self):

        A = np.array([
            [2.0, -1],
            [-1, 2]
        ])
        d = 1

        expected_L = np.array([
            [1, 0],
            [-1/2, 1]
        ])

        expected_U = np.array([
            [2, -1],
            [0, 3.0/2]
        ])

        computed_L, computed_U = L1U(A, d)

        np.testing.assert_almost_equal(expected_L, computed_L)
        np.testing.assert_almost_equal(expected_U, computed_U)

    def test_rfoward_1d(self):
        A = np.array([[2.0]])
        b = np.array([-1.0])
        d = 0
        expected_x = [-1/2]
        computed_x = rforwardsolve(A, b, 0)

        np.testing.assert_almost_equal(expected_x, computed_x)
