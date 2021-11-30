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

def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


odometry_frames = np.load('/home/morzh/work/DSOPP/test/test_data/trajectories_with_bias/frames_translates.0.npy')
gps_frames = np.load('/home/morzh/work/DSOPP/test/test_data/trajectories_with_bias/gps_translates.0.npy')
lambdas = np.load('/home/morzh/work/DSOPP/test/test_data/trajectories_with_bias/lambdas.0.npy')

odometry_frames = odometry_frames.reshape(3, -1, order='F')
gps_frames = gps_frames.reshape(3, -1, order='F')

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(odometry_frames[0], odometry_frames[1], odometry_frames[2], linewidth=5)
ax.scatter(gps_frames[0], gps_frames[1], gps_frames[2])
plt.tight_layout()
plt.show()


# ##======================================================================== ## #
number_elements = odometry_frames.shape[1]
dimensions_number = 3
group_num_parameters = sp.SIM3().getNumParameters()
z = np.zeros((dimensions_number, number_elements))
w = np.zeros((dimensions_number, number_elements))
init_guess = np.zeros((dimensions_number * number_elements + group_num_parameters,))
init_guess[3] = 1.0

rho = 1.0
lambda_l1 = 0.2
lambda_tv = 0.9

# lambda_l1 = lambdas[0]
# lambda_tv = lambdas[1]

weights = 0.25 * np.ones(odometry_frames.shape)

def func_optimize(params, gps_frames, odometry_frames, z, w, rho, weights):

    sim3_grp = sp.SIM3()
    sim3_grp.setParameters(params[:7])
    bias = params[7:].reshape(dimensions_number, -1)
    sR = sim3_grp.matrix()[0:3, 0:3]
    t = sim3_grp.translation()

    e = weights * (gps_frames - (sR @ odometry_frames + t.reshape(3, 1) + bias))
    e2 = bias - z + w
    return np.sum(e.flatten() ** 2) + rho * np.sum(e2.flatten() ** 2)
    # return 0.5*np.sum(e.flatten()**2) + 0.5*rho * np.sum(np.linalg.norm(e2, axis=0)**2)


z_reg = cp.Variable((3, number_elements))

for idx in range(900):
    # ||R(\theta) x - y ||_2^2
    res_optimize = scipy.optimize.minimize(func_optimize, init_guess, args=(gps_frames, odometry_frames, z, w, rho, weights))
    b_reg = res_optimize.x_1[7:].reshape(3, -1)
    '''
    ceres_bias = np.load('/home/morzh/work/DSOPP_tests_data_temp/test_alignment_sim3_bias__bias_ceres.npy').reshape(3, -1, order='F')
    bias_base = np.linspace(0, ceres_bias.shape[1] - 1, ceres_bias.shape[1])
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(2, 2, 1)
    plt.stem(bias_base, ceres_bias[0])
    plt.stem(bias_base, b_reg[0], linefmt='r-',  markerfmt='r')
    ax = fig.add_subplot(2, 2, 2)
    plt.stem(bias_base, ceres_bias[1])
    plt.stem(bias_base, b_reg[1], linefmt='r-',  markerfmt='r')
    ax = fig.add_subplot(2, 2, 3)
    plt.stem(bias_base, ceres_bias[2])
    plt.stem(bias_base, b_reg[2], linefmt='r-',  markerfmt='r')
    plt.tight_layout()
    plt.show()
    '''
    # weights__ = np.vstack((weights, weights))
    functions_to_optimize = [rho * cp.sum_squares(z_reg - b_reg - w),# should be 0.5 here
                             lambda_l1 * cp.sum(cp.norm1(cp.multiply(weights, z_reg), axis=0)),
                             lambda_tv * tv_norm_cols(cp.multiply(weights, z_reg))
                             ]
                             # lambda_l1 * cp.sum(cp.norm1(cp.multiply(weights, z_reg), axis=0))]
    problem = cp.Problem(cp.Minimize(sum(functions_to_optimize)))
    result = problem.solve(warm_start=False, solver=cp.ECOS)

    data, chain, inverse_data = problem.get_problem_data(cp.ECOS)

    ecos_z = np.load('/home/morzh/work/DSOPP_tests_data_temp/test_alignment_sim3_bias__admm_z_ecos.npy').reshape(3, -1, order='F')
    admm_z_base = np.linspace(0, ecos_z.shape[1] - 1, ecos_z.shape[1])
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(2, 2, 1)
    plt.stem(admm_z_base, ecos_z[0])
    plt.stem(admm_z_base, z_reg.value[0], linefmt='r-',  markerfmt='r')
    ax = fig.add_subplot(2, 2, 2)
    plt.stem(admm_z_base, ecos_z[1])
    plt.stem(admm_z_base, z_reg.value[1], linefmt='r-',  markerfmt='r')
    ax = fig.add_subplot(2, 2, 3)
    plt.stem(admm_z_base, ecos_z[2])
    plt.stem(admm_z_base, z_reg.value[2], linefmt='r-',  markerfmt='r')
    plt.tight_layout()
    plt.show()

    

    ecos_z_c = np.load('/home/morzh/work/DSOPP_tests_data_temp/test_alignment_sim3_bias__admm_z_ecos_c.npy')
    ecos_z_h = np.load('/home/morzh/work/DSOPP_tests_data_temp/test_alignment_sim3_bias__admm_z_ecos_h.npy')
    admm_z_c_base = np.linspace(0, ecos_z_c.size - 1, ecos_z_c.size)
    admm_z_h_base = np.linspace(0, ecos_z_h.size - 1, ecos_z_h.size)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 2, 1)
    plt.stem(admm_z_c_base, ecos_z_c)
    plt.stem(admm_z_c_base, data['c'], linefmt='r-',  markerfmt='r')
    ax = fig.add_subplot(1, 2, 2)
    plt.stem(admm_z_h_base, ecos_z_h)
    plt.stem(admm_z_h_base, data['h'], linefmt='r-',  markerfmt='r')
    plt.tight_layout()
    plt.show()

    ecos_z_Gpr = np.load('/home/morzh/work/DSOPP_tests_data_temp/alignment_sim3_bias_Gpr.0.npy')
    ecos_z_Gjc = np.load('/home/morzh/work/DSOPP_tests_data_temp/alignment_sim3_bias_Gjc.0.npy')
    ecos_z_Gir = np.load('/home/morzh/work/DSOPP_tests_data_temp/alignment_sim3_bias_Gir.0.npy')
    ecos_z_x = np.load('/home/morzh/work/DSOPP_tests_data_temp/test_alignment_sim3_bias__admm_z_ecos_x.npy')
    admm_z_Gpr_base = np.linspace(0, ecos_z_Gpr.size - 1, ecos_z_Gpr.size)
    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.stem(admm_z_Gpr_base, ecos_z_Gpr)
    # plt.stem(admm_z_Gpr_base, data['G'].data)
    # plt.tight_layout()
    # plt.show()

    Gpr_compare = ecos_z_Gpr == data['G'].data
    Gjc_compare = ecos_z_Gjc == data['G'].indptr
    Gir_compare = ecos_z_Gir == data['G'].indices

    Gpr_bool = Gpr_compare.any() == Gpr_compare[0]
    Gjc_bool = Gjc_compare.any() == Gjc_compare[0]
    Gir_bool = Gir_compare.any() == Gir_compare[0]

    solution_difference = problem.solution.attr['solver_specific_stats']['x'] - ecos_z_x

    z_reg_value_1d = z_reg.value.reshape(1, -1, order='F')
    solver_solution_x = problem.solution.attr['solver_specific_stats']['x'][1:]
    subarray = rolling_window(solver_solution_x, z_reg_value_1d.size) == z_reg_value_1d


    w += b_reg - z_reg.value
    init_guess = res_optimize.x_1
    z = z_reg.value

    ## -------------------------- INFO AND GRAPHS BLOCK -------------------------- ##
    print('------------------------------------------------------')
    print('iteration', str(idx))
    print('SIM3 params are::', res_optimize.x_1[:group_num_parameters])
    print('bias mean:', np.mean(b_reg, axis=1), 'bias std:', np.std(b_reg, axis=1))

    if not idx % 10:
        # ------------------------ FIRST PLOT -------------------------------#
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.plot(z_reg.value[0], z_reg.value[1], z_reg.value[2])
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.plot(b_reg[0], b_reg[1], b_reg[2])
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.plot(w[0], w[1], w[2])
        plt.tight_layout()
        plt.show()
        # ------------------------ SECOND PLOT -------------------------------#
        grp_params_optimized = res_optimize.x_1[:group_num_parameters]
        bias_optimized = res_optimize.x_1[group_num_parameters:].reshape(dimensions_number, -1)

        check_grp = sp.SIM3()
        check_grp.setParameters(grp_params_optimized)
        sR = check_grp.rotationMatrix()
        t = check_grp.translation()
        y_test = sR @ odometry_frames + t.reshape(3, 1) + bias_optimized
        y_test_2 = sR @ odometry_frames + t.reshape(3, 1)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot(y_test[0], y_test[1], y_test[2], linewidth=5)
        ax.plot(gps_frames[0], gps_frames[1], gps_frames[2])
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot(y_test_2[0], y_test_2[1], y_test_2[2], linewidth=5)
        ax.plot(gps_frames[0], gps_frames[1], gps_frames[2])
        plt.tight_layout()
        plt.show()

