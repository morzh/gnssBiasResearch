import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import os
# from cvxpy_atoms.total_variation_1d import *
from cvxpy.reductions.cvx_attr2constr import *
from cvxpy.reductions.dcp2cone.dcp2cone import *
from cvxpy.reductions.cvx_attr2constr import *
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.promote import promote
from cvxpy_atoms.total_variation_1d import *


path2save = '/home/morzh/work/DSOPP/test/test_data/lasso_tv_problem'
dimensions = 3
np.random.seed(19)
data = np.empty((0, 3))

elements_number = 5
# y = np.linspace(-dimensions*elements_number-1, -1, dimensions*elements_number)
y = np.linspace(1, dimensions*elements_number, dimensions*elements_number)
# y = np.linspace(int(-0.5*dimensions*elements_number), int(0.5*dimensions*elements_number), dimensions*elements_number)
y = -y.reshape(dimensions, elements_number)
# y[y >= 0] += 0.2
# y[y < 0] -= 0.3
# y[1] = 1.5
lambdas = np.random.random(2)
size = dimensions * elements_number

beta = cp.Variable((dimensions, elements_number))
functions = [cp.sum_squares(beta - y), lambdas[0] * cp.sum(cp.norm1(beta, axis=0)), lambdas[1] * tv_norm_cols(beta)]
problem = cp.Problem(cp.Minimize(sum(functions)))
problem_data, _, _ = problem.get_problem_data(cp.ECOS)
np.save('/home/morzh/temp/ECOS/problem_data_negative_G', problem_data['G'].toarray())
np.save('/home/morzh/temp/ECOS/problem_data_negative_c', problem_data['c'])
np.save('/home/morzh/temp/ECOS/problem_data_negative_h', problem_data['h'])
solution = problem.solve(solver=cp.ECOS)
G_rows = 3 * size + 3 + (elements_number - 1) * (dimensions + 1) + (elements_number-1) * 2 - 1
nnz = 5 * size + 2 + 9 * (elements_number - 1)
print(G_rows)
