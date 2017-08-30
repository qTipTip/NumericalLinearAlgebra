import numpy as np

def tdma_diagonals(matrix):
    """
    Returns the sub-, main- and super-diagonals of a matrix.
    :param matrix: tridiagonal matrix
    :return: sub, main, sup - diagonals.
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("Matrix is of type {} while required type is np.ndarray".format(type(matrix)))

    a = matrix.diagonal(-1).copy().astype(np.float64)
    c = matrix.diagonal(1).copy().astype(np.float64)
    d = matrix.diagonal().copy().astype(np.float64)

    return a, d, c


def tdma_lu(a, d, c):
    """
    Returns the LU factorisation of a tridiagonal matrix with diagonals sub, main and sup.
    :param sub: sub-diagonal
    :param main: main diagonal
    :param sup: sup-diagonal
    :return: the sub-diagonal l of L, main-diagonal u of U.
    """

    u = d
    l = a
    n = len(d)

    for k in range(n-1):
        l[k] = a[k] / u[k]
        u[k+1] = d[k+1] - l[k] * c[k]

    return l, u

def tdma_solve(a, d, c, b):
    """
    Solves a tri-diagonal linear system with diagonals a, d, c and right hand side b.
    :param a: sub-diagonal
    :param d: main diagonal
    :param c: super-diagonal
    :param b: right hand side
    :return: the solution x to the system Ax = b.
    """

    x = b
    n = len(d)


    # factorize
    l, u = tdma_lu(a, d, c)

    # forward sweep
    for k in range(1, n):
        x[k] = b[k] - l[k-1] * x[k-1]

    # backward sweep
    x[n-1] = x[n-1] / u[n-1]
    for k in range(n-2, -1, -1):
        x[k] = (x[k] - c[k] * x[k+1]) / u[k]

    return x

def forwardsolve_lower_triangular(A, b, d):
    """
    Solves a matrix equation Ax = b where A is a d-banded matrix.
    :param A: Lower triangular d-banded matrix
    :param b: right hand side
    :param d: band width
    :return: solution x to Ax = b
    """

    n = len(b)
    x = b

    for k in range(1, n):
        lk = max(0, k-d)
        x[k] = (b[k] - A[k, lk : k]).dot(x[lk : (k)]) / A[k, k]

    return x

