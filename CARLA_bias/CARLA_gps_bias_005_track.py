from HDMapperPP.hdmapper_utils import *
from KF.generate_trajectory import *
import os
import pandas as pd
from CARLA_bias.my_procrustes_3d import *
from scipy.spatial.transform import Rotation as Rotation
import sophuspy as sp


path_CARLA = path_DSO = '/media/morzh/ext4_volume/work/hdmapper_data/track04'

slam_poses_graph_filename = 'cloud.rdr'
gps_poses_filename = 'gps.csv'


odometry_track = OdometryTrack()
gps_track = GPSTrack()
gps_data = pd.read_csv(os.path.join(path_CARLA, gps_poses_filename))

odometry_track = TrackLoader.parse_odometry(os.path.join(path_DSO, slam_poses_graph_filename), odometry_track)
gps_track = TrackLoader.parse_gps(os.path.join(path_CARLA, gps_poses_filename), gps_track)
gps_track.convertToENU()

# viz_frames_connections(odometry_track)

kMaxDiffTime = 0.05
associated_track = AssociatedTrack()
associated_track.odometry_connections = odometry_track.active_constraints
associated_track.associateOdometryGPSTracks(gps_track, odometry_track, kMaxDiffTime)

odometry_translates = associated_track.getOdometryTranslates()
gps_enu = associated_track.getENUTranslates()
# active_constraints_SE3 = odometry_track.getActiveConstraintsSE3()
active_constraints_SIM3 = odometry_track.getActiveConstraintsSIM3()


_, _, s, R, t, R_normalizer = my_procrustes_3d(odometry_translates, gps_enu)
odometry_trandform_init = s*R @ odometry_translates + t

associated_track.updateOdometryOptimizedTransform(s, R, t)
associated_track.updateENUTranslates(R_normalizer)

ododmetry_translates = associated_track.getOdometryOptimizedTranslates()
gps_translates = associated_track.getENUTranslates()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(gps_translates[0], gps_translates[1], gps_translates[2])
ax.plot(odometry_trandform_init[0], odometry_trandform_init[1], odometry_trandform_init[2], linewidth=5)
ax.plot(ododmetry_translates[0], ododmetry_translates[1], ododmetry_translates[2])
plt.show()


associated_track.visualizeOdometryFrameConnections()

from scipy.optimize import minimize
import cvxpy as cp
from cvxpy_atoms.total_variation_1d import *


rho = 0.5
lmbda = 0.3 #0.6
lmbda_2 = 0.1 #0.15
dims_num = 3
N = associated_track.framesNumber()
grp_params_num = sp.SIM3().getNumParameters()
asscociative_frames_number = associated_track.framesNumber()
odometry_frames_number = odometry_track.framesNumber()
z = np.zeros((dims_num, asscociative_frames_number))
w = np.zeros((dims_num, asscociative_frames_number))
init_guess = np.zeros((dims_num * asscociative_frames_number + grp_params_num,))
init_guess[3] = 1.0
# init_guess = sim_3_init.getParameters()
z_reg = cp.Variable((dims_num, asscociative_frames_number))


import time

number_of_objective_function_calls = 0

def objective_function(params, associated_track, gps_ENU, active_constraints_SIM3, z, w, rho):
    start_time = time.time()
    global number_of_objective_function_calls
    number_of_objective_function_calls += 1
    number_of_residuals = 0
    # ------- minimising ||t_SLAM - t_GPS|| --------- #
    bias = params[:3 * asscociative_frames_number].reshape(dims_num, -1)
    frames_sim3_params = params[3 * asscociative_frames_number:].reshape(sp.SIM3().getDoF(), -1)
    e_gnss_residuals = 0.0
    for idx in range(asscociative_frames_number):
        frame_translate = frames_sim3_params[4:, idx]
        residuals = (gps_ENU[:, idx] - (frame_translate + bias[:, idx]))
        e_gnss_residuals += np.sum(residuals.flatten())
        number_of_residuals += 1
    e_gnss_residuals += np.sum(rho*(bias - z + w).flatten())

    # -------- minimizing frames connections -------- #
    e_connections_residuals = 0.0
    for source_frame_id in range(odometry_frames_number):
        source_frame = associated_track.findOdometryFrameWithId(source_frame_id)
        if source_frame is None:
            continue
        for target_id in associated_track.odometry_connections[source_frame.id]:
            target_frame = associated_track.findOdometryFrameWithId(target_id)
            if target_frame is None:
                continue

            t_target_reference = active_constraints_SIM3[source_frame_id][target_frame.id]
            t_local_reference = source_frame.optimized_t_w_c
            t_local_target = target_frame.optimized_t_w_c
            e_connections = (t_target_reference * t_local_reference.inverse() * t_local_target).log()
            e_connections_residuals += np.sum(e_connections.flatten())
            number_of_residuals += 1

    print(number_of_residuals)
    if not number_of_objective_function_calls % 1e3:
        print(number_of_objective_function_calls)
    end_time = time.time()
    print('objective exec time is:', end_time-start_time)
    return e_gnss_residuals + e_connections_residuals


init_guess = np.zeros((associated_track.framesNumber() * (sp.SIM3().getDoF() + 3),))
print('parameters number', init_guess.shape)

for idx in range(1):
    least_squares = minimize(objective_function, init_guess, args=(associated_track, gps_enu, active_constraints_SIM3, z, w, rho), options={'maxiter': 1})
    print('least squares ended')
    b_reg = least_squares.x_1[:3 * asscociative_frames_number].reshape(dims_num, -1)
    funcs = [0.5*rho * cp.sum_squares(b_reg - z_reg + w), lmbda * tv_norm_cols(z_reg), lmbda_2 * cp.sum(cp.norm1(z_reg, axis=0))]
    prob = cp.Problem(cp.Minimize(sum(funcs)))
    result = prob.solve(warm_start=True)

    w += b_reg - z_reg.value
    init_guess = least_squares.x_1
    z = z_reg.value

    ## -------------------------- INFO AND GRAPHS BLOCK -------------------------- ##
    print('------------------------------------------------------')
    print('iteration', str(idx))



