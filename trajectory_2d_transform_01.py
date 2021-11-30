import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy import interpolate
from cvxpy_atoms.total_variation_1d import *
# from scipy.spatial import procrustes
from utils.procrustes import  *

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

scale = 1.0# 0.5*random() + 0.2
angle = randint(0, 60)
translate = 0*np.random.randn(2, 3)

R = Rotation.from_euler('z', angle, degrees=True)
R = scale*R.as_dcm()
R = R[0:2, 0:2]

src_points_init = np.copy(src_points)
src_points = np.matmul(R, src_points)

if show_trajectories:
    plt.figure(figsize=(20, 10))
    plt.plot(src_points[0, :], src_points[1, :])
    plt.plot(src_points_biased[0, :], src_points_biased[1, :])
    plt.show()


_, Z, tform = procrustes(src_points_biased.t, src_points.T, scaling=False)
src_points_transformed = Z.t

if show_trajectories:
    plt.figure(figsize=(20, 10))
    plt.plot(src_points_init[0, :], src_points_init[1, :], color='grey', linewidth=2)
    plt.plot(src_points_transformed[0, :], src_points_transformed[1, :])
    plt.plot(src_points_biased[0, :], src_points_biased[1, :])
    plt.show()
