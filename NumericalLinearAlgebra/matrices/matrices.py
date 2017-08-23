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
