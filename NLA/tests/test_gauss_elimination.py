from unittest import TestCase
import numpy as np

from NLA.matrices import rforwardsolve, rbackwardsolve, cforwardsolve, cbackwardsolve


class TestTriangularSolves(TestCase):

    def test_forwardsolve_lower_triangular_rows(self):

        A = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1]
        ])

        b = np.array([1, 2, 3])
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