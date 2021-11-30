import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

nx = 1000
x = np.linspace(0, nx-1, nx)
y = np.zeros(nx)
y[:nx//2] = 10
y[nx//2:3*nx//4] = -5

n = np.random.normal(0, 1, nx)
y = y + n

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.title('Model and data')
plt.show()


y__ = cp.Variable((nx, ))
lambda_l1 = 7
lambda_l2 = 7
lambda_tv = 15

# functions_to_optimize = [cp.sum_squares(y__ - y), lambda_l1 * cp.norm1(y__)]
# functions_to_optimize = [cp.sum_squares(y__ - y), lambda_l2 * cp.sum_squares(y__)]
functions_to_optimize = [cp.sum_squares(y__ - y), lambda_tv * cp.tv(y__)]


problem = cp.Problem(cp.Minimize(sum(functions_to_optimize)))
result = problem.solve()


plt.figure(figsize=(10, 5))
plt.title('Model and data')
plt.plot(x, y, color='grey', linewidth=3)
plt.plot(x, y__.value, color='r', linewidth=4)
plt.tight_layout()
plt.show()
