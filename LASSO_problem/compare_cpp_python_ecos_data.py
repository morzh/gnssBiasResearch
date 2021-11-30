import numpy as np
import matplotlib.pyplot as plt
import os
from cvxpy.reductions.cvx_attr2constr import *

from scipy.sparse import csc_matrix


path = '/home/morzh/work/DSOPP/test/test_data/lasso_tv_problem'

number_data_samples = 100
dimensions = 3

for idx in range(number_data_samples):

    input_data = np.load(os.path.join(path, 'input_data.'+str(idx)) + '.npy')
    solution = np.load(os.path.join(path, 'solution.'+str(idx)) + '.npy')
    lambdas = np.load(os.path.join(path, 'lambdas.'+str(idx)) + '.npy')

    # dense matrices c and h
    internal_c = np.load(os.path.join(path, 'internal_data_c.'+str(idx)) + '.npy')
    internal_h = np.load(os.path.join(path, 'internal_data_h.'+str(idx)) + '.npy')

    internal_check_c = np.load(os.path.join(path, 'internal_data_check_c.'+str(idx)) + '.npy')
    internal_check_h = np.load(os.path.join(path, 'internal_data_check_h.'+str(idx)) + '.npy')

    # sparse Matrix G
    internal_Gpr = np.load(os.path.join(path, 'internal_data_Gpr.'+str(idx)) + '.npy')
    internal_Gjc = np.load(os.path.join(path, 'internal_data_Gjc.'+str(idx)) + '.npy')
    internal_Gir = np.load(os.path.join(path, 'internal_data_Gir.'+str(idx)) + '.npy')

    internal_check_Gpr = np.load(os.path.join(path, 'internal_data_check_Gpr.'+str(idx)) + '.npy')
    internal_check_Gjc = np.load(os.path.join(path, 'internal_data_check_Gjc.'+str(idx)) + '.npy')
    internal_check_Gir = np.load(os.path.join(path, 'internal_data_check_Gir.'+str(idx)) + '.npy')

    check_h = internal_h == internal_check_h
    check_c = internal_c == internal_check_c
    check_Gpr = internal_Gpr == internal_check_Gpr
    check_Gjc = internal_Gjc == internal_check_Gjc
    check_Gir = internal_Gir == internal_check_Gir

    result_h = np.all(check_h == check_h[0])
    result_c = np.all(check_c == check_c[0])
    result_Gpr = np.all(check_Gpr == check_Gpr[0])
    result_Gjc = np.all(check_Gjc == check_Gjc[0])
    # result_Gir = np.all(check_Gir == check_Gir[0])

    result = result_h and result_c and result_Gjc and result_Gpr# and result_Gir
    # internal_G = csc_matrix((internal_Gpr, internal_Gir, internal_Gjc), shape=(internal_h.size, internal_c.size))
    # internal_check_G = csc_matrix((internal_check_Gpr, internal_check_Gir,internal_check_Gjc), shape=(internal_check_h.size, internal_check_c.size))

    if not result:
        print('iteration', idx, result)

print('all ok')


