import numpy as np
import matplotlib.pyplot as plt


curves_array = np.load('/home/morzh/work/curves__.npy')
curves_mean = np.load('/home/morzh/work/curves_mean.npy')

print(curves_array.shape)



fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
idx = 0
ax.plot(curves_mean[0], curves_mean[1], curves_mean[2])
ax.plot(curves_array[0, :, idx], curves_array[1, :, idx], curves_array[2, :, idx])
plt.show()