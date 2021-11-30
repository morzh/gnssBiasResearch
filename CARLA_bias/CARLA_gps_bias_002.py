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

odometry_track = OdometryTrack()
gps_track = GPSTrack()

odometry_track = TrackLoader.parse_odometry(os.path.join(path_DSO, slam_poses_graph_filename), odometry_track)
gps_track = TrackLoader.parse_gps(os.path.join(path_CARLA, gps_poses_filename), gps_track)
gps_track.convertLatLonToMeters()

kMaxDiffTime = 0.01
associated_track = AssociatedTrack()
associated_track.associateOdometryGPSTracks(gps_track, odometry_track, kMaxDiffTime)
associated_track.timestampsStatistics()

'''
def gnss_position_cost(t_local_keyframe, t_local_camera, local_coordinates, local_coordinates_deviations):
    return (t_local_camera - local_coordinates) / local_coordinates_deviations

def relative_transformation_cost(t_local_target_, t_local_reference, covariance_matrix_pose, t_target_reference):
    residuals = (t_target_reference_.cast<T>() * t_local_reference.inverse() * t_local_target).log();
    # residuals.applyOnTheLeft(l_t_inverse_covariance_matrix_);
    return residuals
'''