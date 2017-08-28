from unittest import TestCase

import numpy as np

from NLA.matrices import tdma_diagonals, tdma_lu, tdma_solve
from NLA.splines import spline_interpolation, find_subintervals


class TestSpline(TestCase):

    def test_spline_int(self):

        a = 0
        b = 4
        y = [0, 1/6, 2/3, 1/6, 0]
        mu0 = 0
        munp1 = 0

        expected_x = np.array([0, 1, 2, 3, 4])
        expected_c = np.array([
            [0, 0, 0, 1/6],
            [1/6, 1/2, 1/2, -1/2],
            [2/3, 0, -1, 1/2],
            [1/6, -1/2, 1/2, -1/6]
        ])

        computed_x, computed_c = spline_interpolation(a, b, y, mu0, munp1)
        np.testing.assert_almost_equal(expected_x, computed_x)
        np.testing.assert_almost_equal(expected_c, computed_c)

    def test_find_subintervals(self):

        x = np.linspace(0, 4, 5)
        t = np.linspace(0, 4, 10)

        expected_intervals = [0, 2, 4, 6, 9]
        computed_intervals = find_subintervals(t, x)

        np.testing.assert_almost_equal(expected_intervals, computed_intervals)