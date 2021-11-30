from HDMapperPP.hdmapper_utils import *
from KF.generate_trajectory import *
import os
import pandas as pd
from scipy.spatial.transform import Rotation as Rotation

path_DSO = '/media/morzh/ext4_volume/work/DSO_SLAM'
path_CARLA = '/media/morzh/ext4_volume/progz/CARLA_0.9.9.4/PythonAPI/examples/track01'

slam_poses_graph_filename = 'cloud.rdr'
gps_poses_filename = 'gps.csv'
coords_filename = 'coords.txt'
gt_filename = 'gt.tum'

gps_data = pd.read_csv(os.path.join(path_CARLA, gps_poses_filename))
coords = pd.read_csv(os.path.join(path_CARLA, coords_filename), delimiter=' ')
ground_truth = np.loadtxt(os.path.join(path_CARLA, gt_filename))
# track_odometry = OdometryTrack()
# rdr_loader = TrackLoader(track)
track_odometry = TrackLoader.parse_odometry(os.path.join(path_DSO, slam_poses_graph_filename))

keyframes_number = len(track_odometry.frames)
gps_poses_number = len(gps_data)

bias_smoothed = get_smoothed_piecewise_bias(number_iterations=gps_poses_number, jumps_number=(6, 20), mult=0.2, smooth_factor=(25, 15), savgol_params=(35, 3))

gps_poses = get_gps_translations(gps_data)
gps_poses_meters = convert_lat_lon_to_km(gps_poses)
init_point_gps = gps_poses_meters[:, 0]
gps_poses_meters -= init_point_gps.reshape(3, 1)
slam_translates = track_odometry.getFramesTranslates()
init_point_slam = slam_translates[:, 0]
pc_gps, pc_slam, shift_gps, shift_slam, scale_gps, scale_slam = showPointClouds(gps_poses_meters, slam_translates, clamp=False, show_info=True)

R1 = Rotation.from_euler('y', -90, degrees=True)
R2 = Rotation.from_euler('x', 90, degrees=True)
R1 = R1.as_matrix()
R2 = R2.as_matrix()

slam_translates = np.matmul(R2, np.matmul(R1, slam_translates))
init_point_slam = slam_translates[:, 0]
slam_translates -= init_point_slam.reshape(3, 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(gps_poses_meters[0], gps_poses_meters[1], gps_poses_meters[2])
# ax.plot(pc_slam[0], pc_slam[1], pc_slam[2])
ax.plot(slam_translates[0], slam_translates[1], -slam_translates[2])
plt.show()