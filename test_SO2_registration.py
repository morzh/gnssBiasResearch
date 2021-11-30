import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from cvxpy_atoms.atom_exp_so2 import *

random_points_num = 100
points1 = np.random.rand(2, random_points_num)
theta = np.random.rand(1)[0]
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
points2 = np.matmul(R, points1)
# print(R)
# print('----------------------------')

'''
plt.scatter(points1[0], points1[1], c='g')
plt.scatter(points2[0], points2[1], c='b')
plt.show()
'''
# thetas = np.linspace(-2.5, +2.5, 500)
# vals = []
x = cp.Variable()
# for shift in thetas:
A = np.array([[cp.my_cos(x), -np.sin(x)], [np.sin(x), cp.my_cos(x)]])
func = cp.sum_squares(A @ points1 - points2)
cp.transforms.weighted_sum()
# vals.append(np.float64(func.value))
objective = cp.Minimize(func)
# print('------------------------------')
# print('theta is:', theta)
print('objective function value', func.value)
problem = cp.Problem(objective)
print("Is problem DQCP?: ", problem.is_dqcp())

result = problem.solve(verbose=True, max_iters=100)

# plt.plot(thetas, vals)
# plt.show()