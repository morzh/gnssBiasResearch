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


def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K] = 1
    np.random.shuffle(arr)
    return arr

path2save = '/home/morzh/work/DSOPP/test/test_data/lasso_tv_problem_2d'
number_data_samples = 300
dimensions = 2
np.random.seed(19)

for idx in range(number_data_samples):
    # elements_number = np.random.randint(dimensions, 5000)
    elements_number = 4
    random_spikes_binary_array = rand_bin_array(int(0.3*elements_number), elements_number).reshape(1, -1)
    spikes_multiplier = np.random.randint(1, 6, elements_number).reshape(1, -1)
    random_spikes_array = random_spikes_binary_array * spikes_multiplier
    random_spikes_array = np.repeat(random_spikes_array, dimensions, axis=0)
    y = np.random.random((dimensions, elements_number)) * random_spikes_array
    weights = 0.06 + np.random.random((dimensions, elements_number))

    negative_ys = y[y.any() <= 0]
    if negative_ys.size > 0:
        print('found negative number')

    parameters = np.random.random(3)
    beta = cp.Variable((dimensions, elements_number))
    functions = [0.5 * parameters[2] * cp.sum_squares(beta - y),
                 parameters[0] * cp.sum(cp.norm1(cp.multiply(weights, beta), axis=0)),
                 parameters[1] * tv_norm_cols(beta)]
                 # parameters[1] * tv_norm_cols(cp.multiply(weights, beta))]

    problem = cp.Problem(cp.Minimize(sum(functions)))
    original_data, original_chain, original_inverse_data = problem.get_problem_data(cp.ECOS)
    solution = problem.solve(solver=cp.ECOS)
    # print(beta.value)

    np.save(os.path.join(path2save, 'input_data.' + str(idx)), y)
    np.save(os.path.join(path2save, 'data_weights.' + str(idx)), weights)
    np.save(os.path.join(path2save, 'solution.' + str(idx)), beta.value)
    np.save(os.path.join(path2save, 'parameters.' + str(idx)), parameters)
    # print('loop end')
