import os
import pandas as pd
from scipy.interpolate import splprep, splev
import fdasrsf
from HDMapperPP.hdmapper_utils import *
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes


path_data = '/home/morzh/work/hdmapper_data/track02'
gps_filename = 'track02.csv'
rdr_filename = 'track02.rdr'

gps_data = pd.read_csv(os.path.join(path_data, gps_filename))
track = Track()
rdr_loader = TrackLoader(track)
rdr_loader.parse_odometry(os.path.join(path_data, rdr_filename))

gps_poses = get_gps_translations(gps_data)
gps_accuracy = get_gps_accuracy(gps_data)
slam_poses = rdr_loader.getFramesTranslates()
gps_poses, duplicates_indices = remove_duplicate_columns(gps_poses)
gps_accuracy = np.delete(gps_accuracy, duplicates_indices, axis=1)
# np.save('/home/morzh/work/gps_poses', gps_poses)
# np.save('/home/morzh/work/slam_poses', slam_poses)

# gps_poses[2] = 0.0
# showPointClouds(gps_poses, slam_poses, clamp=False)

number_gps_poses = gps_poses.shape[1]
number_slam_poses = slam_poses.shape[1]
gps_poses_meters = convert_lat_lon_to_km(gps_poses)
interpolated_gps_poses = linear_interpolate_poses(gps_poses_meters, number_slam_poses)
gps_poses_meters = interpolated_gps_poses
gps_poses_meters_raw = convert_lat_lon_to_km(gps_poses)
'''
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gps_poses_meters[0], gps_poses_meters[1], gps_poses_meters[2])
plt.show()
'''
pc_gps, pc_slam, shift_gps, shift_slam, scale_gps, scale_slam = showPointClouds(gps_poses_meters, slam_poses, clamp=False, show_info=True)
gps_poses_meters_raw = (gps_poses_meters_raw - shift_gps) / scale_gps
save_houdini_accuracy_csv(gps_poses_meters_raw, gps_accuracy, '/home/morzh/work/hdmapper_data/Hou/track02_trajectory_gps_raw.csv')
save_houdini_csv(pc_gps, '/home/morzh/work/hdmapper_data/Hou/track02_trajectory_gps.csv')
save_houdini_csv(pc_slam, '/home/morzh/work/hdmapper_data/Hou/track02_trajectory_slam.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pc_gps[0], pc_gps[1], pc_gps[2])
ax.plot(pc_slam[0], pc_slam[1], pc_slam[2])
ax.plot(gps_poses_meters_raw[0], gps_poses_meters_raw[1], gps_poses_meters_raw[2])
plt.show()

'''
# mtx1, mtx2, _ = procrustes(slam_poses, gps_poses_meters)
mtx1, mtx2, _ = orthogonal_procrustes(pc_1.T, pc_2.T)
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(mtx1[0], mtx1[1], mtx1[2])
ax.plot(mtx2[0], mtx2[1], mtx2[2])
plt.show()
'''




# curves = np.dstack((slam_poses, interpolated_gps_poses))
curves = np.dstack((pc_gps, pc_slam))
fda_curve = fdasrsf.curve_stats.fdacurve(curves, mode='O', N=slam_poses.shape[1], scale=True)
fda_curve.karcher_mean()

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(fda_curve.beta_mean[0], fda_curve.beta_mean[1], fda_curve.beta_mean[2])
plt.show()

fda_curve.srvf_align()
print(fda_curve.betan.shape)


# np.save('/home/morzh/work/curves__', fda_curve.betan)
# np.save('/home/morzh/work/curves_mean', fda_curve.beta_mean)


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
# ax.plot(fda_curve.beta_mean[0], fda_curve.beta_mean[1], fda_curve.beta_mean[2])
# ax.plot(fda_curve.betan[0, :, 0], fda_curve.betan[1, :, 0], fda_curve.betan[2, :, 0])
# for idx in range(fda_curve.betan.shape[-1]):
#     ax.plot(fda_curve.betan[0, :, idx], fda_curve.betan[1, :, idx], fda_curve.betan[2, :, idx])
ax.plot(fda_curve.betan[0, :, 0], fda_curve.betan[1, :, 0], fda_curve.betan[2, :, 0])
ax.scatter(fda_curve.betan[0, :, 1], fda_curve.betan[1, :, 1], fda_curve.betan[2, :, 1])
plt.show()

save_houdini_csv(fda_curve.betan[:, :, 0], '/home/morzh/work/curve_1.csv')
save_houdini_csv(fda_curve.betan[:, :, 1], '/home/morzh/work/curve_2.csv')

