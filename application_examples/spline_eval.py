import NLA
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.arctan(10*x) + np.pi / 2


a = -1
b = 1
n = 14
x_vals = np.linspace(a, b, n)
y_vals = np.array([f(x) for x in x_vals])

x, c = NLA.spline_interpolation(a, b, y_vals, 0, 0)

N = 100
evaluation_points = np.linspace(a, b, N, endpoint=False) # note we cannot include the endpoint.
spline_points = NLA.spline_evaluation(x, c, evaluation_points)

plt.plot(evaluation_points, spline_points, label="$s(x)$")
plt.scatter(x_vals, y_vals, label="$f(x)$")
plt.legend()
plt.title("Spline interpolation of a function")
plt.show()
