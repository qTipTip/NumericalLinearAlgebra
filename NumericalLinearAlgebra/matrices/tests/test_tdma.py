from unittest import TestCase
import numpy as np

from NumericalLinearAlgebra.matrices.matrices import tdma_diagonals, tdma_lu


class TestTdma_diagonals(TestCase):
    def test_tdma_diagonals(self):
        """ Given a matrix A, check whether diagonals are fetched properly"""
        A = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [4, 5, 6]
        ])

        expected_sub = np.array([2, 5])
        expected_sup = np.array([2, 4])
        expected_main = np.array([1, 3, 6])

        sub, main, sup = tdma_diagonals(A)

        np.testing.assert_almost_equal(sub, expected_sub)
        np.testing.assert_almost_equal(main, expected_main)
        np.testing.assert_almost_equal(sup, expected_sup)

    def test_tdma_diagonals_typeerror(self):
        """ Check that the function tdma_diagonals raises a TypeError if not an ndarray"""
        A = [
            [1, 2, 3],
            [2, 3, 4],
            [4, 5, 6]
            ]

        self.assertRaises(TypeError, tdma_diagonals, A)

    def test_tdma_lu(self):
        """ Checks that factorization is computed correctly, small known solution"""
        A = np.array([
            [4, 1, 0],
            [1, 4, 1],
            [0, 1, 4]
        ])

        expected_l = np.array([0.25, 1/3.75])
        expected_u = np.array([4, 15.0/4, 4 - 1/3.75])

        a, d, c = tdma_diagonals(A)
        computed_l, computed_u = tdma_lu(a, d, c)

        np.testing.assert_almost_equal(expected_l, computed_l)
        np.testing.assert_almost_equal(expected_u, computed_u)

    def test_tdma_lu_mult(self):
        """ Checks that the factorization is computed properly and that multiplying L, U yields A."""
        n = 100

        a = np.full(n-1, 1, dtype=np.float64)
        c = np.full(n-1, 1, dtype=np.float64)
        d = np.full(n, 4, dtype=np.float64)

        A = np.diag(d, 0) + np.diag(a, -1) + np.diag(c, 1)
        l, u = tdma_lu(a, d, c)

        L = np.eye(n) + np.diag(l, -1)
        U = np.diag(u) + np.diag(c, 1)

        np.testing.assert_almost_equal(A, L.dot(U))