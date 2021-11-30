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
gps_track.convertToENU()

kMaxDiffTime = 0.01
associated_track = AssociatedTrack()
associated_track.associateOdometryGPSTracks(gps_track, odometry_track, kMaxDiffTime)
# associated_track.timestampsStatistics()

ododmetry_translates = associated_track.getOdometryTranslates()
gps_translates = associated_track.getENUTranslates()
bias_smoothed_1 = get_smoothed_piecewise_bias(number_iterations=gps_translates.shape[1], jumps_number=(3, 9), mult=5, smooth_factor=(25, 15), savgol_params=(21, 7))
bias_smoothed_2 = get_smoothed_piecewise_bias(number_iterations=gps_translates.shape[1], jumps_number=(3, 9), mult=4, smooth_factor=(25, 15), savgol_params=(21, 7))
bias_smoothed_3 = get_smoothed_piecewise_bias(number_iterations=gps_translates.shape[1], jumps_number=(3, 9), mult=2, smooth_factor=(25, 15), savgol_params=(21, 9))
gps_translates += np.vstack((bias_smoothed_1, bias_smoothed_2, bias_smoothed_3))

pc_gps, pc_slam, shift_gps, shift_slam, scale_gps, scale_slam = showPointClouds(gps_translates, ododmetry_translates, clamp=False, show_info=True)
aligned_pc_1, aligned_pc_2, s, R, t, R_normalizer = my_procrustes_3d(pc_gps, pc_slam)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(aligned_pc_1[0], aligned_pc_1[1], aligned_pc_1[2])
ax.plot(aligned_pc_2[0], aligned_pc_2[1], aligned_pc_2[2])
plt.show()



