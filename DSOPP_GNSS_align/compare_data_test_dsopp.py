from HDMapperPP.hdmapper_utils import *
from KF.generate_trajectory import *
import os
import pandas as pd
from CARLA_bias.my_procrustes_3d import *
from scipy.spatial.transform import Rotation as Rotation

path_track30seconds = '/home/morzh/work/DSOPP/test/test_data/track30seconds'

# slam_poses_graph_filename = 'cloud.rdr'
gps_poses_filename = 'gps_local.csv'
# coords_filename = 'coords.txt'
gt_filename = 'gt.tum'

gps_data = pd.read_csv(os.path.join(path_track30seconds, gps_poses_filename))
# coords = pd.read_csv(os.path.join(path_CARLA, coords_filename), delimiter=' ')
gt = np.loadtxt(os.path.join(path_track30seconds, gt_filename))
'''
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot(gps_data['x'], gps_data['y'], gps_data['z'])
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot(gt[:, 1], gt[:, 2], gt[:, 3])
plt.tight_layout()
plt.show()
'''
from CARLA_bias.my_procrustes_3d import *

gps_translates = np.hstack((np.array(gps_data['x']).reshape(-1, 1), np.array(gps_data['y']).reshape(-1, 1), np.array(gps_data['z']).reshape(-1, 1)))
gt_translates = np.hstack((gt[:, 1].reshape(-1, 1), gt[:, 2].reshape(-1, 1), gt[:, 3].reshape(-1, 1)))
gps_translates = gps_translates[1:, :]

pc_1, pc_2, _, _, _, _ = my_procrustes_3d(gt_translates.T, gps_translates.T)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot(pc_1[0], pc_1[1], pc_1[2])
ax.plot(pc_2[0], pc_2[1], pc_2[2])
plt.tight_layout()
plt.show()