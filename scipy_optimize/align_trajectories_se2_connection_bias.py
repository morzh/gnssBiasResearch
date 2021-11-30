import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import sophuspy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cvxpy as cp
from cvxpy_atoms.total_variation_1d import *

vertical_line_length = 50
horizontal_line_length = 80

vertical_line_major_axis = np.linspace(0, vertical_line_length - 1, vertical_line_length)
vertical_tilted_line_minor_axis = np.linspace(0, 7, vertical_line_length)
vertical_line_minor_axis = np.zeros(vertical_line_length)
vertical_line = np.vstack((vertical_line_minor_axis, vertical_line_major_axis))
vertical_tilted_line = np.vstack((vertical_tilted_line_minor_axis, vertical_line_major_axis))

horizontal_line_major_axis = np.linspace(0, horizontal_line_length - 1, horizontal_line_length)
horizontal_line_minor_axis = np.zeros(horizontal_line_length)
horizontal_line = np.vstack((horizontal_line_major_axis, horizontal_line_minor_axis))

shift = horizontal_line[0, -1]
vertical_line_2 = np.vstack((-vertical_line_minor_axis, vertical_line_major_axis)) + np.array([[shift], [0]])

chunk_1 = np.hstack((np.flip(vertical_line, axis=1), horizontal_line))
chunk_tilted_1 = np.hstack((np.flip(vertical_tilted_line, axis=1), horizontal_line))
chunk_2 = np.hstack((horizontal_line, vertical_line_2))
chunk_tilted_2 = np.hstack((horizontal_line, vertical_line_2))

chunks_matching_indices_row_1 = np.linspace(vertical_line_length, vertical_line_length + horizontal_line_length - 1, horizontal_line_length)
chunks_matching_indices_row_2 = horizontal_line_major_axis
chunks_matching_indices = np.vstack((chunks_matching_indices_row_1, chunks_matching_indices_row_2))
chunks_matching_indices = chunks_matching_indices.astype(np.int32)

gps_horizontal_scale = 1.2
gps_vertical_scale = 0.9
gps_chunk_1 = np.vstack((chunk_1[0] * gps_horizontal_scale, chunk_1[1] * gps_vertical_scale))
gps_chunk_bias_1 = np.vstack((chunk_1[0] * gps_horizontal_scale, chunk_1[1] * gps_vertical_scale))
gps_chunk_bias_1[:, int(0.33333 * gps_chunk_bias_1.shape[1]):int(0.66666 * gps_chunk_bias_1.shape[1])] += 6 * np.array([[-1], [-1]])

gps_chunk_2 = np.vstack((chunk_2[0] * gps_horizontal_scale, chunk_2[1] * gps_vertical_scale))
gps_chunk_bias_2 = np.vstack((chunk_2[0] * gps_horizontal_scale, chunk_2[1] * gps_vertical_scale))
gps_chunk_bias_2[:, int(0.33333 * gps_chunk_bias_2.shape[1]):int(0.66666 * gps_chunk_bias_2.shape[1])] += 9 * np.array([[1.2], [-1]])

odometry_horizontal_scale = 0.8
odometry_vertical_scale = 1.1
odometry_chunk_1 = np.vstack((chunk_tilted_1[0] * odometry_horizontal_scale, chunk_tilted_1[1] * odometry_vertical_scale))
odometry_chunk_2 = np.vstack((chunk_tilted_2[0] * odometry_horizontal_scale, chunk_tilted_2[1] * odometry_vertical_scale))
odometry_chunk_1[0] += 12
odometry_chunk_2[0] += 12
odometry_chunk_1[1] -= 5
odometry_chunk_2[1] -= 5
r = 0.25
R = Rotation.from_euler('z', r, degrees=False)
R = R.as_matrix()[0:2, 0:2]
odometry_chunk_1 = R @ odometry_chunk_1
odometry_chunk_2 = R @ odometry_chunk_2
points_cost_weight = 1.0

plt.title('GPS-SLAM odometry poses')
plt.axis('equal')
plt.plot(gps_chunk_bias_1[0], gps_chunk_bias_1[1], linewidth=4)
plt.plot(gps_chunk_bias_2[0], gps_chunk_bias_2[1])
plt.plot(odometry_chunk_1[0], odometry_chunk_1[1], linewidth=4)
plt.plot(odometry_chunk_2[0], odometry_chunk_2[1])
plt.show()

se2_num_parameters = 3


def func_optimize(parameters, odometry_1, gps_1, odometry_2, gps_2, z, w, rho, indices_match):
    se2_algebra_1 = parameters[0:se2_num_parameters]
    se2_algebra_2 = parameters[se2_num_parameters: 2*se2_num_parameters]
    bias_1 = parameters[2 * se2_num_parameters: 2 * se2_num_parameters + odometry_chunk_1.size].reshape((2, -1), order='F')
    bias_2 = parameters[2 * se2_num_parameters + odometry_chunk_1.size:].reshape((2, -1), order='F')

    angle_1 = se2_algebra_1[0]
    angle_2 = se2_algebra_2[0]

    translate_1 = se2_algebra_1[1:3]
    translate_2 = se2_algebra_2[1:3]

    R_1 = Rotation.from_euler('z', angle_1, degrees=False)
    R_1 = R_1.as_matrix()[0:2, 0:2]
    R_2 = Rotation.from_euler('z', angle_2, degrees=False)
    R_2 = R_2.as_matrix()[0:2, 0:2]

    chunk_1_cost = (gps_1 - (R_1 @ odometry_1 + translate_1.reshape(2, 1) + bias_1))
    chunk_2_cost = (gps_2 - (R_2 @ odometry_2 + translate_2.reshape(2, 1) + bias_2))
    points_match_cost = points_cost_weight * ((R_1 @ odometry_1[:, indices_match[0]] + translate_1.reshape(2, 1)) -
                                              (R_2 @ odometry_2[:, indices_match[1]] + translate_2.reshape(2, 1)))
    bias = np.hstack((bias_1, bias_2))
    admm_cost = bias - z + w

    return np.sum(chunk_1_cost.flatten() ** 2) + np.sum(chunk_2_cost.flatten() ** 2) + rho * np.sum(admm_cost.flatten() ** 2) + \
           np.sum(points_match_cost.flatten() ** 2)


lambda_l1 = 0.3
lambda_tv = 0.9
rho = 1.0

dimensions = 2
number_bias_vectors = odometry_chunk_1.shape[1] + odometry_chunk_2.shape[1]

weights = np.ones((dimensions, number_bias_vectors))
bias = np.zeros((dimensions, number_bias_vectors))
z_subproblem_1 = np.zeros((dimensions, number_bias_vectors))
w = np.zeros((dimensions, number_bias_vectors))

init_guess = np.zeros((2 * se2_num_parameters + bias.size,))

z_subproblem_2 = cp.Variable(bias.shape)

for idx in range(90):
    print('iteration:', idx)
    problem_constants = (odometry_chunk_1, gps_chunk_bias_1, odometry_chunk_2, gps_chunk_bias_2, z_subproblem_1, w, rho, chunks_matching_indices)
    subproblem_1 = scipy.optimize.minimize(func_optimize, init_guess, args=problem_constants)
    bias_subproblem_2 = subproblem_1.x_1[2 * se2_num_parameters:].reshape(2, -1, order='F')
    functions_to_optimize = [cp.sum_squares(bias_subproblem_2 + w - z_subproblem_2), lambda_tv * tv_norm_cols(z_subproblem_2),
                             lambda_l1 * cp.sum(cp.norm1(cp.multiply(weights, z_subproblem_2), axis=0))]
    subproblem_2 = cp.Problem(cp.Minimize(sum(functions_to_optimize)))
    subproblem_2.solve(warm_start=True)

    w += bias_subproblem_2 - z_subproblem_2.value
    init_guess = subproblem_1.x_1
    z_subproblem_1 = z_subproblem_2.value

    if idx % 10 == 0 and idx > 0:
        """visualize results"""
        optimum = subproblem_1.x_1
        bias_1 = optimum[2 * se2_num_parameters: 2 * se2_num_parameters + odometry_chunk_1.size].reshape((2, -1), order='F')
        bias_2 = optimum[2 * se2_num_parameters + odometry_chunk_1.size:].reshape((2, -1), order='F')
        se2_algebra_1 = optimum[0:se2_num_parameters]
        se2_algebra_2 = optimum[se2_num_parameters:2 * se2_num_parameters]

        angle_1 = se2_algebra_1[0]
        angle_2 = se2_algebra_2[0]

        translate_1 = se2_algebra_1[1:3]
        translate_2 = se2_algebra_2[1:3]

        R_1 = Rotation.from_euler('z', angle_1, degrees=False)
        R_1 = R_1.as_matrix()[0:2, 0:2]
        R_2 = Rotation.from_euler('z', angle_2, degrees=False)
        R_2 = R_2.as_matrix()[0:2, 0:2]

        odometry_chunk_1_transformed = R_1 @ odometry_chunk_1 + translate_1.reshape(2, 1)# + bias_1
        odometry_chunk_2_transformed = R_2 @ odometry_chunk_2 + translate_2.reshape(2, 1)# + bias_2

        plt.title('GPS-SLAM optimized odometry poses, iteration: ' + str(idx))
        plt.axis('equal')
        plt.plot(gps_chunk_1[0], gps_chunk_1[1], linewidth=4)
        plt.plot(gps_chunk_2[0], gps_chunk_2[1])
        plt.plot(odometry_chunk_1_transformed[0], odometry_chunk_1_transformed[1], linewidth=1)
        plt.plot(odometry_chunk_2_transformed[0], odometry_chunk_2_transformed[1], linewidth=1)
        plt.show()
