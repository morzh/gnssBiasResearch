import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from cvxpy_atoms.atom_exp_so2 import *

random_points_num = 100
points1 = np.random.rand(2, random_points_num)
theta = np.random.rand(1)[0]
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
A = np.random.rand(2, 2)
points2 = np.matmul(R, points1)
epsilon = 1e-1
'''
plt.scatter(points1[0], points1[1], c='g')
plt.scatter(points2[0], points2[1], c='b')
plt.show()
'''
x = cp.Variable()
# func = cp.sum_squares(A @ points1 - points2)
func = cp.sum_squares(atom_exp_so2(x) @ points1 - points2)
# func = cp.sum(exp_so2(x))
objective = cp.Minimize(func)
problem = cp.Problem(objective, [func <= epsilon])
print("Is problem DQCP?: ", problem.is_dqcp())
