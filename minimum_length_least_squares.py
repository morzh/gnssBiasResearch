import cvxpy as cp
import numpy as np

n = 10
np.random.seed(1)
A = np.random.randn(n, n)
x_star = np.random.randn(n)
b = A @ x_star
epsilon = 1e6

x = cp.Variable(n)
mse = cp.sum_squares(A @ x - b)
problem = cp.Problem(cp.Minimize(mse))
# problem = cp.Problem(cp.Minimize(cp.length(x)), [mse <= epsilon])
print("Is problem DQCP?: ", problem.is_dqcp())
print("Is problem DCP?: ", problem.is_dcp())

problem.solve(qcp=True)
print("Found a solution, with length: ", problem.value)