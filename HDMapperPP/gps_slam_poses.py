import  numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from HDMapperPP.hdmapper_utils import *


gps_poses = np.load('/home/morzh/work/gps_poses.npy')
slam_poses = np.load('/home/morzh/work/slam_poses.npy')

save_houdini_csv(gps_poses, '/home/morzh/work/gps_poses.csv')
save_houdini_csv(slam_poses, '/home/morzh/work/slam_poses.csv')


number_slam_poses = slam_poses.shape[1]
gps_poses = gps_poses[:, 20:200]
slam_poses = slam_poses[:, 70:1100]

# showPointClouds(gps_poses, slam_poses, clamp=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(gps_poses[0], gps_poses[1], gps_poses[2])
# ax.plot(slam_poses[0], slam_poses[1], slam_poses[2])
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(slam_poses[0], slam_poses[1], slam_poses[2])
# ax.plot(slam_poses[0], slam_poses[1], slam_poses[2])
plt.show()

tck, u = splprep(gps_poses, k=1)
u_new = np.linspace(0, 1, number_slam_poses)
x_new, y_new, z_new = splev(u_new, tck, der=0)
gps_poses = np.vstack((x_new, y_new, z_new))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(gps_poses[0], gps_poses[1], gps_poses[2])
# ax.plot(slam_poses[0], slam_poses[1], slam_poses[2])
plt.show()