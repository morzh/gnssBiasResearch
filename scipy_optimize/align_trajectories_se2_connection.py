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
vertical_line_minor_axis = np.zeros(vertical_line_length)
vertical_line = np.vstack((vertical_line_minor_axis, vertical_line_major_axis))

horizontal_line_major_axis = np.linspace(0, horizontal_line_length - 1, horizontal_line_length)
horizontal_line_minor_axis = np.zeros(horizontal_line_length)
horizontal_line = np.vstack((horizontal_line_major_axis, horizontal_line_minor_axis))

shift = horizontal_line[0, -1]
vertical_line_2 = vertical_line + np.array([[shift], [0]])

chunk_1 = np.hstack((vertical_line, horizontal_line))
chunk_2 = np.hstack((horizontal_line, vertical_line_2))
chunks_matching_indices_row_1 = np.linspace(vertical_line_length, vertical_line_length + horizontal_line_length - 1, horizontal_line_length)
chunks_matching_indices_row_2 = horizontal_line_major_axis
chunks_matching_indices = np.vstack((chunks_matching_indices_row_1, chunks_matching_indices_row_2))
chunks_matching_indices = chunks_matching_indices.astype(np.int32)

gps_horizontal_scale = 1.3
gps_vertical_scale = 0.85
gps_chunk_1 = np.vstack((chunk_1[0] * gps_horizontal_scale, chunk_1[1] * gps_vertical_scale))
gps_chunk_2 = np.vstack((chunk_2[0] * gps_horizontal_scale, chunk_2[1] * gps_vertical_scale))

odometry_horizontal_scale = 0.9
odometry_vertical_scale = 1.15
odometry_chunk_1 = np.vstack((chunk_1[0] * odometry_horizontal_scale, chunk_1[1] * odometry_vertical_scale))
odometry_chunk_2 = np.vstack((chunk_2[0] * odometry_horizontal_scale, chunk_2[1] * odometry_vertical_scale))
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

'''
plt.plot(chunk_1[0], chunk_1[1])
plt.plot(chunk_2[0], chunk_2[1])
plt.show()
plt.title('gps poses')
plt.plot(gps_chunk_1[0], gps_chunk_1[1])
plt.scatter(gps_chunk_1[0], gps_chunk_1[1], s=15)
plt.plot(gps_chunk_2[0], gps_chunk_2[1])
plt.scatter(gps_chunk_2[0], gps_chunk_2[1], s=5)
plt.show()
plt.title('odometry poses')
plt.plot(odometry_chunk_1[0], odometry_chunk_1[1])
plt.scatter(odometry_chunk_1[0], odometry_chunk_1[1], s=15)
plt.plot(odometry_chunk_2[0], odometry_chunk_2[1])
plt.scatter(odometry_chunk_2[0], odometry_chunk_2[1], s=5)
plt.show()
'''

plt.title('GPS-SLAM odometry poses')
plt.plot(gps_chunk_1[0], gps_chunk_1[1], linewidth=4)
plt.plot(gps_chunk_2[0], gps_chunk_2[1])
plt.plot(odometry_chunk_1[0], odometry_chunk_1[1], linewidth=4)
plt.plot(odometry_chunk_2[0], odometry_chunk_2[1])
plt.show()


def func_optimize(parameters, odometry_1, gps_1, odometry_2, gps_2, indices_match):
    se2_algebra_1 = parameters[0:3]
    se2_algebra_2 = parameters[3:6]

    angle_1 = se2_algebra_1[0]
    angle_2 = se2_algebra_2[0]

    translate_1 = se2_algebra_1[1:3]
    translate_2 = se2_algebra_2[1:3]

    R_1 = Rotation.from_euler('z', angle_1, degrees=False)
    R_1 = R_1.as_matrix()[0:2, 0:2]
    R_2 = Rotation.from_euler('z', angle_2, degrees=False)
    R_2 = R_2.as_matrix()[0:2, 0:2]

    chunk_1_cost = (gps_1 - R_1 @ odometry_1 - translate_1.reshape(2, 1))
    chunk_2_cost = (gps_2 - R_2 @ odometry_2 - translate_2.reshape(2, 1))
    points_match_cost = points_cost_weight * ((R_1 @ odometry_1[:, indices_match[0]] + translate_1.reshape(2, 1)) -
                                              (R_2 @ odometry_2[:, indices_match[1]] + translate_2.reshape(2, 1)))

    return np.sum(chunk_1_cost.flatten()**2) + np.sum(chunk_2_cost.flatten()**2) + np.sum(points_match_cost.flatten() ** 2)


se2_num_parameters = 3
init_guess = np.zeros((2 * se2_num_parameters,))
problem_constants = (odometry_chunk_1, gps_chunk_1, odometry_chunk_2, gps_chunk_2, chunks_matching_indices)
problem = scipy.optimize.minimize(func_optimize, init_guess, args=problem_constants)


"""visualize results"""
optimum = problem.x_1
se2_algebra_1 = optimum[0:3]
se2_algebra_2 = optimum[3:6]

angle_1 = se2_algebra_1[0]
angle_2 = se2_algebra_2[0]

translate_1 = se2_algebra_1[1:3]
translate_2 = se2_algebra_2[1:3]

R_1 = Rotation.from_euler('z', angle_1, degrees=False)
R_1 = R_1.as_matrix()[0:2, 0:2]
R_2 = Rotation.from_euler('z', angle_2, degrees=False)
R_2 = R_2.as_matrix()[0:2, 0:2]

odometry_chunk_1_transformed = R_1 @ odometry_chunk_1 + translate_1.reshape(2, 1)
odometry_chunk_2_transformed = R_2 @ odometry_chunk_2 + translate_2.reshape(2, 1)

plt.title('GPS-SLAM odometry poses')
plt.plot(gps_chunk_1[0], gps_chunk_1[1], linewidth=4)
plt.plot(gps_chunk_2[0], gps_chunk_2[1])
plt.plot(odometry_chunk_1_transformed[0], odometry_chunk_1_transformed[1], linewidth=1)
plt.plot(odometry_chunk_2_transformed[0], odometry_chunk_2_transformed[1], linewidth=1)
plt.show()