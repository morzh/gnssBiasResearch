import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import interpolate
from scipy.interpolate import splprep, splev, splrep
import scipy.interpolate as si
from cvxpy_atoms.total_variation_1d import *
import open3d as o3d
# import scipy.spatial.procrustes


def get_biases(num_samples, biases_values, biases_durations_normalized):
    biases_timeline = np.array([0])
    biases_durations = num_samples*biases_durations_normalized
    x = np.empty((0, 1))
    for idx in range(biases_durations.shape[0]):
        biases_timeline = np.vstack((biases_timeline, biases_timeline[-1]+biases_durations[idx]))

    for idx in range(biases_timeline.shape[0]-1):
        x = np.vstack((x, biases_values[idx]*np.ones((round(float(biases_timeline[idx+1])) - round(float(biases_timeline[idx])), 1))))

    if x.shape[0] < num_samples:
        x = np.vstack((x, x[-1]*np.ones(num_samples-x.shape[0])))
    return x.reshape(num_samples,)


show_vector_plot = False
random_points_num = 120
trajectory_points_num = 1000
points = np.random.rand(random_points_num, 2)
convex_hull = ConvexHull(points)
convex_vertices_indices = convex_hull.vertices
convex_hull_points = points[convex_vertices_indices]
tck, u = interpolate.splprep(convex_hull_points.t, s=0, per=True)
xi, yi = interpolate.splev(np.linspace(0, 1, trajectory_points_num+1), tck)

src_points = np.array([xi, yi])
src_points = src_points[:, 1:]
spline_points_derivatives = np.roll(src_points, 1, 1) - src_points
spline_points_ortho = np.roll(spline_points_derivatives, 1, 0)
spline_points_ortho[0, :] *= -1.0

if show_vector_plot:
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(src_points[0, :], src_points[1, :], s=1)
    q = ax.quiver(src_points[0, :], src_points[1, :], spline_points_derivatives[0, :], spline_points_derivatives[1, :])
    r = ax.quiver(src_points[0, :], src_points[1, :], spline_points_ortho[0, :], spline_points_ortho[1, :])
    # ax.quiverkey(q, X=1, Y=1, U=1, label='Quiver key, length = 1', labelpos='E')
    plt.show()

spline_points_ortho /= np.linalg.norm(spline_points_ortho, axis=0)

noise_amplitude_1 = 0.04
noise_amplitude_2 = 0.035
biases_number_jumps = np.random.randint(5, 15)
biases_values = np.random.randn(biases_number_jumps)
biases_durations = np.abs(np.random.randn(biases_number_jumps))
biases_durations_normalized = biases_durations / np.sum(biases_durations)
np.random.seed(np.random.randint(0,100))
noise_1 = noise_amplitude_1 * np.random.normal(0, 1, size=trajectory_points_num)
np.random.seed(np.random.randint(0,100))
noise_2 = noise_amplitude_2 * np.random.normal(0, 1, size=trajectory_points_num)

show_trajectories = True
from scipy.spatial.transform import Rotation
from random import randint
from random import random

bias = get_biases(trajectory_points_num, biases_values, biases_durations_normalized)
src_points = src_points + 0.1 * noise_2 * spline_points_ortho
src_points_biased = src_points + 0.1 * bias * spline_points_ortho + 0.1 * noise_1 * spline_points_ortho

scale = 1#0.5*random() + 0.2
angle = randint(0, 60)
translate = 2*np.random.randn(0, 0)

R = Rotation.from_euler('z', angle, degrees=True)
R = scale*R.as_dcm()
R = R[0:2, 0:2]

src_points = np.matmul(R, src_points)

if show_trajectories:
    plt.figure(figsize=(20, 10))
    plt.plot(src_points[0, :], src_points[1, :])
    plt.plot(src_points_biased[0, :], src_points_biased[1, :])
    plt.show()



pc_points1 = o3d.geometry.PointCloud()
pc_points2 = o3d.geometry.PointCloud()
pc_points1.points = o3d.utility.Vector3dVector(np.vstack((src_points, np.zeros((1, src_points.shape[1])))).T)
pc_points2.points = o3d.utility.Vector3dVector(np.vstack((src_points_biased, np.zeros((1, src_points_biased.shape[1])))).T)
pc_points1.paint_uniform_color([1, 0, 0])
pc_points2.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([pc_points1, pc_points2])

import copy
from probreg import cpd
tf_param, _, _ = cpd.registration_cpd(pc_points1, pc_points2)
result = copy.deepcopy(pc_points2)
result.points = tf_param.transform(result.points)
result.paint_uniform_color([0, 0, 1])
o3d.visualization.draw_geometries([pc_points1, pc_points2, result])

lamda_1 = np.random.uniform(0.1, 2)
lamda_2 = np.random.uniform(0.1, 1)
lamda_3 = np.random.uniform(0.1, 1)

x = cp.Variable((trajectory_points_num, 2))
b = cp.Variable((trajectory_points_num, 2))

# funcs = [cp.sum_squares(x - src_points.T), lamda_1*cp.sum_squares(x + b - src_points_biased.T), lamda_2*cp.norm(x, 2), lamda_3*cp.norm(b, 2)]
funcs = [cp.sum_squares(x - src_points.T), lamda_1 * cp.sum_squares(x + b - src_points_biased.t), lamda_2 * tv_norm_rows(b)]# + lamda_3*tv1d(b)]

opts = {'maxiters': 900}
objective = cp.Minimize(sum(funcs))
prob = cp.Problem(objective)
result = prob.solve(verbose=True, max_iters=900)

bias_norms_linspace = np.linspace(0, trajectory_points_num-1, trajectory_points_num)
bias_norms = np.linalg.norm(b.value, axis=1)

fig=plt.figure(figsize=(20, 10))
ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.45, 0.2, 0.2])
textbox_objective_function = 'objective: $argmin_{x,b}  ||x - p1||_2^2 + \lambda_1  || \left [x+b - p2 \\right ] ||_2^2 + \lambda_2 || \  |b| \  ||_{TV}$'
textbox_lambda_values = '\n'.join((r'$\lambda_1=%.2f$' % (lamda_1,), '$\lambda_2=%.2f$' % (lamda_2,)))#, r'$\lambda_3=%.2f$' % (lamda_3,)))
textbox_bias_norms = 'bias norms graph:'
props = dict(boxstyle='round', facecolor='white', alpha=0.75)
ax1.text(0.65, 0.9, textbox_objective_function, fontsize=12, verticalalignment='top', bbox=props)
ax1.text(0.65, 0.75, textbox_lambda_values, fontsize=12, verticalalignment='bottom', bbox=props)
ax1.text(0.65, 0.7, textbox_bias_norms, fontsize=12, verticalalignment='bottom', bbox=props)
ax1.plot(src_points[0, :], src_points[1, :],  linewidth=5, color='black')
ax1.plot(src_points_biased[0, :], src_points_biased[1, :], linewidth=5, color='blue')
ax1.plot(x.value[:, 0], x.value[:, 1], linewidth=1, color='red')
ax1.plot(x.value[:, 0]+b.value[:, 0], x.value[:, 1]+b.value[:, 1], linewidth=1, color='orange')
ax1.plot(b.value[:, 0]+0.5, b.value[:, 1]+0.5, linewidth=1, color='green')
ax1.scatter(0.5, 0.5, s=50, c='g')
ax1.legend(('noisy trajectory', 'noisy trajectory with bias', 'estimated trajectory', 'estimated trajectory + bias', 'bias vectors only (shifted, green dot is the bias origin)'), loc='upper left', shadow=True, framealpha=0.75)
ax2.plot(bias_norms_linspace, bias_norms)
textbox_bias = 'norms of bias vectors'
plt.tight_layout()
plt.show()
