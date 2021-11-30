import os
import pandas as pd
from scipy.interpolate import splprep, splev
import fdasrsf
from HDMapperPP.hdmapper_utils import *

path_data = '/home/morzh/work/hdmapper_data'
gps_filename = 'track1.csv'
rdr_filename = 'track1.rdr'

gps_data = pd.read_csv(os.path.join(path_data, gps_filename))
gps_poses = get_gps_translations(gps_data)
gps_poses = remove_duplicate_columns(gps_poses)

number_slam_poses = 1806

tck, u = splprep(gps_poses, k=1)
u_new = np.linspace(0, 1, number_slam_poses)
x_new, y_new, z_new = splev(u_new, tck)
gps_poses = np.vstack((x_new, y_new, z_new))

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(gps_poses[0], gps_poses[1], gps_poses[2])
ax.scatter(gps_poses[0], gps_poses[1], gps_poses[2])
plt.show()