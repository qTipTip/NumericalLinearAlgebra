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

def find_subintervals(t, x):
    """
    Given a set of real numbers 't', and a real value x, we compute the integer i such that
    t[i] <= x < t[i+1]. If x is a list of numbers, we compute a corresponding vector of indices
    :param knots: set of real values [t1, t2, ..., tk]
    :param x: a real value or a vector of real values
    :return: corresponding index, or vector of indices
    """
    x = np.array(x)
    t = np.array(t)

    k = len(t)
    m = len(x)

    if k < 2:
        # if the knot set is trivial, we just return zeros
        return np.zeros(m)

    else:
        j = np.concatenate((t, x)).argsort() # indices that would sort the array
        i = np.nonzero(j >= k) # indices of indices that are larger than the number of knots

        arr = np.arange(0, m)
        arr = (i - arr - 1)[0]

        return arr

def spline_evaluation(t, c, x):
    """
    Given a set of knots t, the corresponding spline coefficients c, and a set of evaluation points x, return a set
    of spline evaluations.
    :param t: knot values
    :param c: spline coefficients
    :param x: set of points of evaluation, the last point cannot be equal to the endpoint of the interpolation interval.
    :return: set of spline evaluations
    """

    m = len(x)
    idx = find_subintervals(t, x)
    s_values = np.zeros(m)
    for i in range(m):
        k = idx[i]
        x_val = x[i] - t[k]  # the shifted polynomials
        coefficients = c[k, :]
        s_values[i] = np.array([1, x_val, x_val**2, x_val**3]).dot(coefficients)

    return s_values