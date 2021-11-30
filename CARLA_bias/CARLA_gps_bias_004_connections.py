from HDMapperPP.hdmapper_utils import *
from KF.generate_trajectory import *
import os
import pandas as pd
from CARLA_bias.my_procrustes_3d import *
from scipy.spatial.transform import Rotation as Rotation
import sophuspy as sp


path_DSO = '/media/morzh/ext4_volume/work/DSO_SLAM'
path_CARLA = '/media/morzh/ext4_volume/progz/CARLA_0.9.9.4/PythonAPI/examples/track01'

slam_poses_graph_filename = 'cloud.rdr'
gps_poses_filename = 'gps.csv'
coords_filename = 'coords.txt'
gt_filename = 'gt.tum'

gps_data = pd.read_csv(os.path.join(path_CARLA, gps_poses_filename))
coords = pd.read_csv(os.path.join(path_CARLA, coords_filename), delimiter=' ')
ground_truth = np.loadtxt(os.path.join(path_CARLA, gt_filename))

odometry_track = OdometryTrack()
gps_track = GPSTrack()

odometry_track = TrackLoader.parse_odometry(os.path.join(path_DSO, slam_poses_graph_filename), odometry_track)
gps_track = TrackLoader.parse_gps(os.path.join(path_CARLA, gps_poses_filename), gps_track)
gps_track.convertLatLonToMeters()

kMaxDiffTime = 0.0005
associated_track = AssociatedTrack()
associated_track.associateOdometryGPSTracks(gps_track, odometry_track, kMaxDiffTime)
# associated_track.timestampsStatistics()

ododmetry_translates = associated_track.getOdometryTranslates()
gps_translates = associated_track.getGPSMetersTranslates()

pc_gps, pc_slam, shift_gps, shift_slam, scale_gps, scale_slam = showPointClouds(gps_translates, ododmetry_translates, clamp=False, show_info=True)
aligned_pc_1, aligned_pc_2 = my_procrustes_3d(pc_gps, pc_slam, show_plots=False)

fig = plt.figure(figsize=(20, 10))
ax = fig.gca(projection='3d')
ax.plot(aligned_pc_1[0], aligned_pc_1[1], aligned_pc_1[2])
ax.plot(aligned_pc_2[0], aligned_pc_2[1], aligned_pc_2[2])
plt.show()


# aligned_pc_1 += 0.005*np.random.randn(aligned_pc_1.shape[0], aligned_pc_1.shape[1]) + np.array([0.1, 0.2, 0.3]).reshape(3, 1)
# aligned_pc_2 += 0.005*np.random.randn(aligned_pc_2.shape[0], aligned_pc_2.shape[1])
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(aligned_pc_1[0], aligned_pc_1[1], aligned_pc_1[2])
ax.plot(aligned_pc_2[0], aligned_pc_2[1], aligned_pc_2[2])
plt.show()
'''

from scipy.optimize import minimize
import cvxpy as cp
from cvxpy_atoms.total_variation_1d import *


rho = 0.5
lmbda = 0.3 #0.6
lmbda_2 = 0.1 #0.15
dims_num = 3
N = associated_track.framesNumber()
grp_params_num = sp.SIM3().getNumParameters()
z = np.zeros((dims_num, N))
w = np.zeros((dims_num, N))
init_guess = np.zeros((dims_num * N + grp_params_num,))
init_guess[3] = 1.0
# init_guess = sim_3_init.getParameters()
z_reg = cp.Variable((dims_num, N))


def func_optimize(params, pc_1, pc_2, z, w, rho):
    sim3_grp = sp.SIM3()
    sim3_grp.setParameters(params[:7])
    bias = params[7:].reshape(dims_num, -1)

    sR = sim3_grp.rotationMatrix()
    t = sim3_grp.translation()
    e = pc_2 - (sR @ pc_1 + t.reshape(3, 1) + bias)
    e2 = bias - z + w
    return 0.5*np.sum(e.flatten()**2) + 0.5*rho * np.sum(e2.flatten()**2)


for idx in range(150):
    least_squares = minimize(func_optimize, init_guess, args=(aligned_pc_1, aligned_pc_2, z, w, rho), options={'maxiter': 8})

    b_reg = least_squares.x_1[grp_params_num:].reshape(dims_num, -1)
    funcs = [0.5*rho * cp.sum_squares(b_reg - z_reg + w), lmbda * tv_norm_cols(z_reg), lmbda_2 * cp.sum(cp.norm1(z_reg, axis=0))]
    prob = cp.Problem(cp.Minimize(sum(funcs)))
    result = prob.solve(warm_start=True)

    w += b_reg - z_reg.value
    init_guess = least_squares.x_1
    z = z_reg.value

    ## -------------------------- INFO AND GRAPHS BLOCK -------------------------- ##
    print('------------------------------------------------------')
    print('iteration', str(idx))
    print('SIM3 params are::', least_squares.x_1[:grp_params_num])
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
        grp_params_optimized = least_squares.x_1[:grp_params_num]
        bias_optimized = least_squares.x_1[grp_params_num:].reshape(dims_num, -1)

        check_grp = sp.SIM3()
        check_grp.setParameters(grp_params_optimized)
        sR = check_grp.rotationMatrix()
        t = check_grp.translation()
        y_test = sR @ aligned_pc_1 + t.reshape(3, 1) + bias_optimized
        y_test_2 = sR @ aligned_pc_1 + t.reshape(3, 1)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot(y_test[0], y_test[1], y_test[2], linewidth=5)
        ax.plot(aligned_pc_2[0], aligned_pc_2[1], aligned_pc_2[2])
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot(y_test_2[0], y_test_2[1], y_test_2[2], linewidth=5)
        ax.plot(aligned_pc_2[0], aligned_pc_2[1], aligned_pc_2[2])
        plt.tight_layout()
        plt.show()
