import numpy as np

from NLA.matrices import tdma_solve


def spline_interpolation(a, b, y, mu0, munp1):
    """
    Computes the spline coefficients of the spline solving the D2 spline interpolation problem over the interval [a, b]
    :param a: left endpoint
    :param b: right endpoint
    :param y: y values
    :param mu0: left moment
    :param munp1: right moment
    :return: set of coefficients and corresponding knot values x.
    """

    n = len(y) - 1
    h = (b - a) / n

    x = np.linspace(a, b, n + 1)

    if isinstance(y, list):
        y = np.array(y, dtype=np.float64)

    # right hand side of linear system
    b = (6 / h ** 2) * (y[2:n + 1] - 2 * y[1:n] + y[0:n - 1])
    b[0] -= mu0
    b[n - 2] -= munp1

    # diagonals
    a = np.full(n - 2, 1, dtype=np.float64)
    d = np.full(n - 1, 4, dtype=np.float64)
    c = np.full(n - 2, 1, dtype=np.float64)

    # compute moments
    mu = np.zeros(n + 1)
    mu[0] = mu0
    mu[1:n] = tdma_solve(a, d, c, b)
    mu[n] = munp1

    # compute spline coefficients
    c = np.zeros(shape=(n, 4))
    c[:, 0] = y[:n]
    c[:, 1] = (y[1:n + 1] - y[:n]) / h - h * mu[:n] / 3 - h * mu[1:n + 1] / 6
    c[:, 2] = mu[:n] / 2
    c[:, 3] = (mu[1:n + 1] - mu[:n]) / (6 * h)

    return x, c
