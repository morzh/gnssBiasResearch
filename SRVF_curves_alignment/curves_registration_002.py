import os
import pandas as pd
from KF.generate_trajectory import *
from HDMapperPP.hdmapper_utils import *
import fdasrsf

N_iter = 600

[points_1, points_2] = generate_trajectories(num_samples=N_iter, trajectory_noise=(0.125, 0.1), noise_mult=(0.007, 0.009), scale=5.0, random_points_num=150)

points_1 = np.array(points_1)
points_2_src = np.array(points_2)

path_data = '/home/morzh/work/hdmapper_data'
gps_filename = 'track1.csv'
rdr_filename = 'track1.rdr'

gps_data = pd.read_csv(os.path.join(path_data, gps_filename))
track = Track()
rdr_loader = TrackLoader(track)
rdr_loader.parse_odometry(os.path.join(path_data, rdr_filename))

gps_poses = get_gps_translations(gps_data)
slam_poses = rdr_loader.getFramesTranslates()


curves = np.dstack((points_1, points_2))

fda_curve = fdasrsf.curve_stats.fdacurve(curves, N=N_iter)
fda_curve.karcher_mean()

print(fda_curve.beta_mean)

plt.plot(fda_curve.beta_mean[0], fda_curve.beta_mean[1])
plt.plot(points_1[0], points_1[1])
plt.plot(points_2[0], points_2[1])
plt.show()

print('aligning curve ....')
fda_curve.srvf_align()
print(type(fda_curve.betan))
print(fda_curve.betan.shape)


plt.figure(figsize=(20, 10))
plt.plot(fda_curve.beta_mean[0], fda_curve.beta_mean[1])
plt.plot(points_1[0], points_1[1])
plt.plot(points_2_src[0], points_2_src[1])
for idx in range(fda_curve.betan.shape[-1]):
    plt.plot(fda_curve.betan[0, :, idx], fda_curve.betan[1, :, idx])
plt.show()
