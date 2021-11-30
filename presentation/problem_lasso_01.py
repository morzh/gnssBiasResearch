import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import sophuspy as sp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cvxpy as cp
from cvxpy_atoms.total_variation_1d import *


number_elements = 2


rho = 0.75
lambda_l1 = 0.2
lambda_tv = 0.9

x = cp.Variable(number_elements,)
b = np.random.random(number_elements)

# functions = [rho * cp.sum_squares(x - b), lambda_l1 * cp.norm1(x), lambda_tv * cp.tv(x)]
# functions = [rho * cp.sum_squares(x - b), lambda_l1 * cp.norm1(x[0])]
# functions = [rho * cp.sum_squares(x - b)]
# functions = [cp.sum_squares(x)]
functions = [lambda_l1 * cp.norm1(x - b)]

problem = cp.Problem(cp.Minimize(sum(functions)))
result = problem.solve(warm_start=False, solver=cp.ECOS)

data, chain, inverse_data = problem.get_problem_data(cp.ECOS)

print(b)
print(x.value)
