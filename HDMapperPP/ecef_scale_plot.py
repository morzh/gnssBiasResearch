import numpy as np
import matplotlib.pyplot as plt

filepath = '/home/morzh/work/DSOPP_tests_data_temp/proto_track/scales.npy'
scales = np.load(filepath)

xs = np.linspace(0, scales.size-1, scales.size)

plt.plot(xs, scales)
plt.show()
