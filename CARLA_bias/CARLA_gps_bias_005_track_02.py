from HDMapperPP.hdmapper_utils import *
from KF.generate_trajectory import *
import os
import pandas as pd
from CARLA_bias.my_procrustes_3d import *
from scipy.spatial.transform import Rotation as Rotation
import sophuspy as sp


path_CARLA = path_DSO = '/home/morzh/work/hdmapper_data/track04'

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

from scipy.optimize import least_squares
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
conncetions_dict = associated_track.getConnectionsIDsDict(odometry_frames_number)

def objective_function(params, associated_track, gps_ENU, active_constraints_SIM3, conncetions_dict, z, w, rho):
    start_time = time.time()
    global number_of_objective_function_calls
    global odometry_frames_number
    number_of_objective_function_calls += 1
    # ------- minimising ||t_SLAM - t_GPS|| --------- #
    bias = params[:3 * asscociative_frames_number].reshape(dims_num, -1)
    frames_sim3_params = params[3 * asscociative_frames_number:].reshape(sp.SIM3().getDoF(), -1)
    frames_translates = frames_sim3_params[4:, :]
    # e_gnss_residuals = np.zeros((asscociative_frames_number*3,))
    e_gnss_residuals = gps_enu - frames_translates - bias
    # for idx in range(asscociative_frames_number):
    #     frame_translate = frames_sim3_params[4:, idx]
    #     residuals = (gps_ENU[:, idx] - (frame_translate + bias[:, idx]))
    #     e_gnss_residuals[3*idx:3*(idx+1)] = residuals

    check_point_time = time.time()
    # -------- minimizing frames connections -------- #
    connections_ids_dict = associated_track.getConnectionsIDsDict(odometry_frames_number)
    connections_residuals_number = associated_track.getNumberOfConnections(odometry_frames_number)
    e_connections_residuals = np.zeros((7*connections_residuals_number, ))
    idx_connection = 0
    for source_frame_id in connections_ids_dict.keys():
        t_local_reference = associated_track.frames[source_frame_id].odometry_frame.optimized_t_w_c
        if len(connections_ids_dict[source_frame_id]) == 0:
            continue
        for target_frame_id in connections_ids_dict[source_frame_id]:
            t_target_reference = active_constraints_SIM3[source_frame_id][target_frame_id]
            # t_target_reference = active_constraints_SIM3[source_frame_id][target_frame_id]
            t_local_target = associated_track.frames[target_frame_id].odometry_frame.optimized_t_w_c
            e_connections = (t_target_reference * t_local_reference.inverse() * t_local_target).log()
            e_connections_residuals[7*idx_connection:7*(idx_connection+1)] = e_connections
            idx_connection += 1

    end_time = time.time()
    if not number_of_objective_function_calls % 1e3:
        print(number_of_objective_function_calls)
    print('objective exec time is:', check_point_time-start_time, 'and', end_time-check_point_time)
    return np.concatenate((e_gnss_residuals.flatten(), e_connections_residuals))


init_guess = np.zeros((associated_track.framesNumber() * (sp.SIM3().getDoF() + 3),))
print('parameters number', init_guess.shape)

time_start = time.time()
for idx in range(1):
    ls_obj = least_squares(objective_function, init_guess, args=(associated_track, gps_enu, active_constraints_SIM3, conncetions_dict, z, w, rho), tr_solver='lsmr', tr_options={'maxiter': 1})
    print('least squares ended')
    b_reg = ls_obj.x[:3 * asscociative_frames_number].reshape(dims_num, -1)
    funcs = [0.5*rho * cp.sum_squares(b_reg - z_reg + w), lmbda * tv_norm_cols(z_reg), lmbda_2 * cp.sum(cp.norm1(z_reg, axis=0))]
    prob = cp.Problem(cp.Minimize(sum(funcs)))
    result = prob.solve(warm_start=True)

    w += b_reg - z_reg.value
    init_guess = ls_obj.x
    z = z_reg.value

    ## -------------------------- INFO AND GRAPHS BLOCK -------------------------- ##
    print('------------------------------------------------------')
    print('iteration', str(idx))

time_end = time.time()

print('exec time overall is', time_end - time_start, 'seconds')

