import os
import pandas as pd
from scipy.interpolate import splprep, splev
import fdasrsf
from HDMapperPP.hdmapper_utils import *


path_data = '/home/morzh/work/hdmapper_data'
gps_filename = 'track01.csv'
rdr_filename = 'track1.rdr'

gps_data = pd.read_csv(os.path.join(path_data, gps_filename))
track = Track()
rdr_loader = TrackLoader(track)
rdr_loader.parse_odometry(os.path.join(path_data, rdr_filename))

gps_poses = get_gps_translations(gps_data)
slam_poses = rdr_loader.getFramesTranslates()
gps_poses = remove_duplicate_columns(gps_poses)

showPointClouds(gps_poses, slam_poses, clamp=False)

number_gps_poses = gps_poses.shape[1]
number_slam_poses = slam_poses.shape[1]


tck, u = splprep(gps_poses, k=2)
u_new = np.linspace(0, 1, number_slam_poses)
x_new, y_new, z_new = splev(u_new, tck, der=0)
gps_poses = np.vstack((x_new, y_new, z_new))

showPointClouds(gps_poses, slam_poses, clamp=False)

curves = np.dstack((slam_poses, gps_poses))
fda_curve = fdasrsf.curve_stats.fdacurve(curves, N=slam_poses.shape[1])
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
for idx in range(fda_curve.betan.shape[-1]):
    ax.plot(fda_curve.betan[0, :, idx], fda_curve.betan[1, :, idx], fda_curve.betan[2, :, idx])
plt.show()

