import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = '/home/morzh/work/DSOPP_tests_data_temp/proto_track/statistics_018'
kSampleNumber = 1
kFileBase = 'alignment_track'
bias_suffix = 'align_bias'
kFileExtension = 'tum'
lambdas = [1e-8]

ground_truth_filename = "/media/morzh/ext4_volume/progz/CARLA_0.9.9.4/PythonAPI/examples/track01/gt.tum"
gps_local_filename = '/media/morzh/ext4_volume/progz/CARLA_0.9.9.4/PythonAPI/examples/track01/gps_local.csv'

gt_tum = np.loadtxt(ground_truth_filename)
gps_local = pd.read_csv(gps_local_filename)
print(gt_tum.shape)

positions = gt_tum[:, 1:4]

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
ax.plot(gps_local['x'], gps_local['y'], gps_local['z'])
plt.tight_layout()
plt.show()

