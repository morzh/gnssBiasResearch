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
gps_track.convertToENU()

kMaxDiffTime = 0.0005
associated_track = AssociatedTrack()
associated_track.odometry_connections = odometry_track.active_constraints
associated_track.associateOdometryGPSTracks(gps_track, odometry_track, kMaxDiffTime)

odometry_translates = associated_track.getOdometryTranslates()
gps_enu = associated_track.getENUTranslates()
# active_constraints_SE3 = odometry_track.getActiveConstraintsSE3()
active_constraints_SIM3 = odometry_track.getActiveConstraintsSIM3()


_, _, s, R, t, R_normalizer = my_procrustes_3d(odometry_translates, gps_enu)
odometry_trandform_init = s * R @ odometry_translates + t

associated_track.updateOdometryOptimizedTransform(s, R, t)
associated_track.updateENUTranslates(np.linalg.inv(R_normalizer))

ododmetry_translates = associated_track.getOdometryOptimizedTranslates()
gps_translates = associated_track.getENUTranslates()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(gps_translates[0], gps_translates[1], gps_translates[2])
ax.plot(odometry_trandform_init[0], odometry_trandform_init[1], odometry_trandform_init[2], linewidth=5)
ax.plot(ododmetry_translates[0], ododmetry_translates[1], ododmetry_translates[2])
plt.show()



from scipy.optimize import minimize
import cvxpy as cp
from cvxpy_atoms.total_variation_1d import *


rho = 0.5
lmbda = 0.3 #0.6
lmbda_2 = 0.1 #0.15
dims_num = 3
N = associated_track.framesNumber()
grp_params_num = sp.SIM3().getNumParameters()
frames_number = associated_track.framesNumber()
z = np.zeros((dims_num, frames_number))
w = np.zeros((dims_num, frames_number))
init_guess = np.zeros((dims_num * frames_number + grp_params_num,))
init_guess[3] = 1.0
# init_guess = sim_3_init.getParameters()
z_reg = cp.Variable((dims_num, frames_number))

def optimize_(params, associated_track, gps_ENU, active_constraints_SIM3, z, w, rho):

    # ------- minimising ||t_SLAM - t_GPS|| --------- #
    bias = params[:3*frames_number].reshape(dims_num, -1)
    frames_sim3_params = params[3*frames_number:].reshape(sp.SIM3().getDoF(), -1)
    e_gnss_residuals = 0.0
    for idx in range(frames_number):
        frame_translate = frames_sim3_params[4:, idx]
        residuals = (gps_ENU[:, idx] - (frame_translate + bias[:, idx]))
        e_gnss_residuals += np.sum(residuals.flatten())
    e_gnss_residuals += np.sum(rho*(bias - z + w).flatten())

    # -------- minimizing frames connections -------- #
    e_connections_residuals = 0.0
    # e_connections = np.zeros((associated_track.framesNumber()*sp.SIM3().getDoF()))
    for ref_id in range(frames_number):
        if not associated_track.odometryFrameExists(ref_id):
            continue
        for target_id in associated_track.odometry_connections[ref_id]:
            t_target_reference = (active_constraints_SIM3[ref_id])[target_id-associated_track.odometry_connections[ref_id][0]]
            t_local_reference = associated_track.frames[ref_id].odometry_frame.optimized_t_w_c
            t_local_target = associated_track.frames[target_id].odometry_frame.optimized_t_w_c
            e_connections = (t_target_reference * t_local_reference.inverse() * t_local_target).log()
            e_connections_residuals += np.sum(e_connections.flatten())


    return e_gnss_residuals + e_connections_residuals


init_guess = np.zeros((associated_track.framesNumber() * (sp.SIM3().getDoF() + 3),))

for idx in range(1):
    least_squares = minimize(optimize_, init_guess, args=(associated_track, gps_enu, active_constraints_SIM3, z, w, rho), options={'maxiter': 8})

    b_reg = least_squares.x_1[:3 * frames_number].reshape(dims_num, -1)
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



