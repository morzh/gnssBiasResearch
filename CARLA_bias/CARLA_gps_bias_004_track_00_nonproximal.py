from HDMapperPP.hdmapper_utils import *
from KF.generate_trajectory import *
import os
import pandas as pd
from CARLA_bias.my_procrustes_3d import *
from scipy.spatial.transform import Rotation as Rotation
import sophuspy as sp


path = '/media/morzh/ext4_volume/work/hdmapper_data'

path_DSO = os.path.join(path, 'track01')
path_CARLA = os.path.join(path, 'track01')


slam_poses_graph_filename = 'cloud.rdr'
gps_poses_filename = 'gps.csv'
# coords_filename = 'coords.txt'
# gt_filename = 'gt.tum'

gps_data = pd.read_csv(os.path.join(path_CARLA, gps_poses_filename))
# coords = pd.read_csv(os.path.join(path_CARLA, coords_filename), delimiter=' ')
# ground_truth = np.loadtxt(os.path.join(path_CARLA, gt_filename))

odometry_track = TrackLoader.parse_odometry(os.path.join(path_DSO, slam_poses_graph_filename))
gps_track = TrackLoader.parse_gps(os.path.join(path_CARLA, gps_poses_filename))
gps_track.convertLatLonToMeters()

kMaxDiffTime = 0.02
associated_track = AssociatedTrack()
associated_track.associateOdometryGPSTracks(gps_track, odometry_track, kMaxDiffTime)
# associated_track.timestampsStatistics()

ododmetry_translates = associated_track.getOdometryTranslates()
gps_translates = associated_track.getGPSMetersTranslates()

# viz_frames_connections(odometry_track)

pc_gps, pc_slam, shift_gps, shift_slam, scale_gps, scale_slam = showPointClouds(gps_translates, ododmetry_translates, clamp=False, show_info=True)
aligned_pc_1, aligned_pc_2, _, _, _, _ = my_procrustes_3d(pc_gps, pc_slam)

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


rho = 1.0
lmbda = 0.015
lmbda_2 = 0.0025
dims_num = 3
N = associated_track.framesNumber()
grp_params_num = sp.SIM3().getNumParameters()
z = np.zeros((dims_num, N))
w = np.zeros((dims_num, N))
init_guess = np.zeros((dims_num * N + grp_params_num,))
init_guess[3] = 1.0
# init_guess = sim_3_init.getParameters()
z_reg = cp.Variable((dims_num, N))

def clip(beta, alpha):
    clipped = np.minimum(beta, alpha)
    clipped = np.maximum(clipped, -alpha)
    return clipped

def proxL1Norm(betaHat, alpha, penalizeAll=True):
    out = betaHat - clip(betaHat, alpha)
    if not penalizeAll:
        out[0] = betaHat[0]
    return out


def func_optimize(params, pc_1, pc_2):
    sim3_grp = sp.SIM3()
    sim3_grp.setParameters(params[:7])
    bias = params[7:].reshape(dims_num, -1)

    sR = sim3_grp.rotationMatrix()
    t = sim3_grp.translation()
    e = pc_2 - (sR @ pc_1 + t.reshape(3, 1) + bias)
    return np.sum(e.flatten()**2)

show_admm_plot = False

for idx in range(901):
    least_squares = minimize(func_optimize, init_guess, args=(aligned_pc_1, aligned_pc_2), options={'maxiter': 8})

    b_reg = least_squares.x_1[grp_params_num:].reshape(dims_num, -1)
    grad = least_squares.jac[grp_params_num:].reshape(dims_num, -1)

    step = 0.99
    b_reg = proxL1Norm(b_reg - step*grad, step*lmbda_2)

    init_guess = least_squares.x_1.copy()
    init_guess[grp_params_num:] = b_reg.flatten()

    ## -------------------------- INFO AND GRAPHS BLOCK -------------------------- ##
    print('------------------------------------------------------')
    print('iteration', str(idx))
    print('SIM3 params are::', least_squares.x_1[:grp_params_num])
    print('bias mean:', np.mean(b_reg, axis=1), 'bias std:', np.std(b_reg, axis=1))

    if not idx % 100:
        if show_admm_plot:
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

