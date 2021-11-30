import numpy as np
import cvxpy as cvx
import dccp
from cvxpy_atoms import *
from cvxpy_atoms.atom_exp_so2 import *



random_points_num = 100
points1 = np.random.rand(2, random_points_num)
theta = np.random.rand(1)[0]
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
points2 = np.matmul(R, points1)


theta_hat = cvx.Variable(1)
# theta_hat = 0.0
# A = cvx.Variable((2, 2))
# A = np.array([[cvx.my_cos(theta_hat), -cvx.my_sin(theta_hat)], [cvx.my_sin(theta_hat), cvx.my_cos(theta_hat)]])
func = cvx.sum_squares(so2(theta_hat) @ points1 - points2)
myprob = cvx.Problem(cvx.Minimize(func))

print("problem is DCP:", myprob.is_dcp())   # false
print("problem is DCCP:", dccp.is_dccp(myprob))  # true
result = myprob.solve(method='dccp')
# result = myprob.solve()
