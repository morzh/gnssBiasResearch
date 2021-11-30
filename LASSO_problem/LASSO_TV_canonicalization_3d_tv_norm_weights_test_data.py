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

path2save = '/home/morzh/work/DSOPP/test/test_data/lasso_tv_problem'
number_data_samples = 100
dimensions = 3
np.random.seed(19)

for idx in range(number_data_samples):
    elements_number = np.random.randint(3, 3000)
    # elements_number = 3
    # y_linspace = np.linspace(1, dimensions * elements_number, dimensions * elements_number).reshape(3, -1)
    # y = y_linspace + 0.2

    random_spikes_binary_array = rand_bin_array(int(0.3*elements_number), elements_number).reshape(1, -1)
    spikes_multiplier = np.random.randint(1, 6, elements_number).reshape(1, -1)
    random_spikes_array = random_spikes_binary_array * spikes_multiplier
    random_spikes_array = np.repeat(random_spikes_array, 3, axis=0)
    y = np.random.random((dimensions, elements_number)) * random_spikes_array
    weights = 0.06 + np.random.random((dimensions, elements_number))
    '''
    weights_l1 = np.ones((dimensions, elements_number))
    weights_l1[0] *= 0.25
    weights_l1[1] *= 1.25
    weights_l1[2] *= 2.25

    weights_tv = np.ones((dimensions, elements_number))
    weights_tv[0] *= 4.25
    weights_tv[1] *= 5.25
    weights_tv[2] *= 6.25
    '''
    negative_ys = y[y.any() <= 0]
    if negative_ys.size > 0:
        print('found negative number')

    parameters = np.random.random(3)

    beta = cp.Variable((dimensions, elements_number))
    functions = [0.5 * parameters[2] * cp.sum_squares(beta - y),
                 parameters[0] * cp.sum(cp.norm1(cp.multiply(weights, beta), axis=0)),
                 parameters[1] * tv_norm_cols(cp.multiply(weights, beta))]
    problem = cp.Problem(cp.Minimize(sum(functions)))
    original_data, original_chain, original_inverse_data = problem.get_problem_data(cp.ECOS)
    solution = problem.solve(solver=cp.ECOS)

    np.save(os.path.join(path2save, 'input_data.' + str(idx)), y)
    np.save(os.path.join(path2save, 'data_weights.' + str(idx)), weights)
    np.save(os.path.join(path2save, 'solution.' + str(idx)), beta.value)
    np.save(os.path.join(path2save, 'parameters.' + str(idx)), parameters)

    '''
    np.save(os.path.join(path2save, 'internal_data_c.'+str(idx)), original_data['c'])
    np.save(os.path.join(path2save, 'internal_data_h.'+str(idx)), original_data['h'])
    np.save(os.path.join(path2save, 'internal_data_Gpr.'+str(idx)), original_data['G'].data)
    np.save(os.path.join(path2save, 'internal_data_Gjc.'+str(idx)), original_data['G'].indptr)
    np.save(os.path.join(path2save, 'internal_data_Gir.'+str(idx)), original_data['G'].indices)
    '''
    '''
    G = original_data['G']
    plt.spy(G, markersize=2)
    plt.tight_layout()
    plt.show()

    for idx in range(len(G.indices)):
        print(G.indices[idx], '', end='')
    '''
    # data_frame = np.array([y, lambdas, beta.value], dtype=object)
    # data_frame = np.expand_dims(data_frame, axis=0)
    # data = np.vstack((data, data_frame))
    # print('--------------------------------------------')

    '''
    G_as_dense = original_data['G'].toarray()


    for idx1 in range(G_as_dense.shape[0]):
        for idx2 in range(G_as_dense.shape[1]):
            print(str(int(G_as_dense[idx1, idx2])).zfill(2), '', end='')
        print('')
    
    print('--------------------------------------------')

    print(beta.value.reshape(1, -1, order='F'))
    '''
# np.save(os.path.join(path2save, 'lasso_tv_data'), data)
