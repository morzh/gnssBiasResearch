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

kMaxDiffTime = 0.01
associated_track = AssociatedTrack()
associated_track.associateOdometryGPSTracks(gps_track, odometry_track, kMaxDiffTime)
# associated_track.timestampsStatistics()

ododmetry_translates = associated_track.getOdometryTranslates()
gps_translates = associated_track.getGPSMetersTranslates()

pc_gps, pc_slam, shift_gps, shift_slam, scale_gps, scale_slam = showPointClouds(gps_translates, ododmetry_translates, clamp=False, show_info=True)
aligned_pc_1, aligned_pc_2, s, R, t, R_normalizer = my_procrustes_3d(pc_gps, pc_slam)

fig = plt.figure()
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

def func_optimize(sim3_params, pc_1, pc_2):
    sim3_grp = sp.SIM3()
    sim3_grp.setParameters(sim3_params)
    print(sim3_grp.matrix())
    sR = sim3_grp.rotationMatrix()
    t = sim3_grp.translation()
    e = pc_2 - (sR @ pc_1 + t.reshape(3, 1))
    return np.sum(e.flatten()**2)


sim_3_init = sp.SIM3(np.eye(4))
init_guess = sim_3_init.getParameters()

min_arg = minimize(func_optimize, init_guess, args=(aligned_pc_1, aligned_pc_2))

print(min_arg)

