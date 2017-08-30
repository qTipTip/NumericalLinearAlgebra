from unittest import TestCase
import numpy as np

from NLA.matrices import forwardsolve_lower_triangular


class TestTriangularSolves(TestCase):
    def test_forwardsolve_lower_triangular(self):

        A = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1]
        ])

        b = np.array([1, 2, 3])
        d = 1

        expected_x = np.array([1, 1, 2])
        computed_x = forwardsolve_lower_triangular(A, b, d)

        np.testing.assert_almost_equal(expected_x, computed_x)