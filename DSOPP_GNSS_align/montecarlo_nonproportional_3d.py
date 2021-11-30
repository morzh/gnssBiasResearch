import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


cov = 25 * np.eye(3)
cov[2, 2] = 1.0
sqrt_cov = np.linalg.cholesky(cov)
print(sqrt_cov)
local_frames = pd.read_csv("/home/morzh/work/DSOPP_data_output/local_coordinates_after_fusion.csv")

fig = plt.figure(figsize=(20, 10))
ax = fig.gca(projection='3d')
ax.plot(local_frames.local_tx, local_frames.local_ty, local_frames.local_tz, linewidth=6)
ax.plot(local_frames.frame_tx, local_frames.frame_ty, local_frames.frame_tz, linewidth=6)
plt.show()
