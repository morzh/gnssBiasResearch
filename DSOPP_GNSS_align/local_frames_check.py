import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


gnss_data_proxy = pd.read_csv('/home/morzh/work/DSOPP_data_output/gnss_data_proxy.csv')
local_frames = pd.read_csv('/home/morzh/work/DSOPP_data_output/after_fusion.csv')

fig = plt.figure(figsize=(20, 10))
ax = fig.gca(projection='3d')
for idx in range(10):
    local_frames_subtrack = pd.read_csv('/home/morzh/work/DSOPP_data_output/track_' + str(idx) + '.csv')
    ax.plot(local_frames_subtrack.x_1, local_frames_subtrack.y_1, local_frames_subtrack.z, linewidth=2)
ax.plot(local_frames.x_1, local_frames.y_1, local_frames.z, linewidth=4)
ax.plot(gnss_data_proxy.x_1, gnss_data_proxy.y_1, gnss_data_proxy.z, linewidth=2)
plt.tight_layout()
plt.show()